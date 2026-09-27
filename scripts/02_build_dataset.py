#!/usr/bin/env python3
"""Сборка обучающего датасета уровня «изображение» из меток уровня «исследование».

Проблема, которую решает скрипт
-------------------------------
Организатор дал 7 бинарных меток на *исследование × область* (разметка.xlsx),
а выходная таблица по ТЗ 2.5 — одна строка на *изображение*. Значит метку надо
спустить на изображение, а для этого изображение надо отнести к области
(позвоночник / левое бедро / правое бедро). Пер-изображенческой разметки нет.

Ключевое наблюдение, которое делает это возможным без ручной разметки:
после дедупликации по пикселям в каждом исследовании ровно столько уникальных
изображений, сколько у него размеченных областей (1, 2 или 3). То есть
соответствие «изображение ↔ область» — это паросочетание, а не свободная задача.

Скрипт:
  1. дедуплицирует изображения внутри исследования (49.5% файлов — точные копии);
  2. считает геометрические признаки (зеркальная симметрия, латеральность);
  3. назначает области с учётом ожидаемого состава из разметки;
  4. помечает флагами `region_confident` (геометрия однозначна),
     `region_matches_markup` (состав совпал с разметкой) и `labeled`
     (для области есть метки организатора) — см. reports/day2_findings.md.

Область определяется геометрией, а не порядком файлов: разделяющий признак —
латеральность |lat_score| (зазор 31.4 против 45.3), сторона — её знак.

Запуск:
    python scripts/02_build_dataset.py \
        --inventory reports/inventory_train.csv \
        --markup data/train/разметка.xlsx \
        --root data/train/Исследования \
        --out reports/dataset_images.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dxa_qc.dicom_io import read_dicom  # noqa: E402

MARKUP_COLS = [
    "n", "study",
    "spine_positioning", "spine_axis_tilt", "spine_artifact",
    "right_femur_rotation", "right_femur_roi",
    "left_femur_rotation", "left_femur_roi",
    "total_spine", "total_femur_right", "total_femur_left",
    "comment",
]

SYM_THRESHOLD = 0.45  # зеркальная корреляция: используется только как перекрёстная проверка
LAT_SPLIT = 35.0      # абсолютная граница |lat_score|, когда сравнивать не с чем
LAT_MARGIN = 20.0     # минимальный отрыв позвоночника от ближайшего бедра по |lat_score|


def load_markup(path: Path) -> pd.DataFrame:
    """Читает разметку.xlsx: две строки шапки, дальше 100 исследований."""
    raw = pd.read_excel(path, header=None, sheet_name=0)
    raw = raw.iloc[:, : len(MARKUP_COLS)]
    raw.columns = MARKUP_COLS
    df = raw.iloc[2:].reset_index(drop=True)
    df = df[df["study"].notna()].copy()
    df["study"] = df["study"].astype(str).str.strip()
    for c in MARKUP_COLS[2:-1]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["comment"] = df["comment"].astype(str).str.strip().replace({"nan": None})
    return df.drop(columns=["n"])


def image_features(path: str) -> dict:
    """Признаки для чернового роутера: симметрия и латеральность."""
    img = read_dicom(path, with_pixels=True)
    if not img.ok or img.pixels is None:
        return {"sym": np.nan, "lat_score": np.nan}
    a = img.pixels
    sym = float(np.corrcoef(a.ravel(), np.fliplr(a).ravel())[0, 1])
    m = a > 0.35
    h = m.shape[0]
    up = np.nonzero(m[: h // 3])[1]
    lo = np.nonzero(m[2 * h // 3 :])[1]
    # у бедра таз (верх кадра) смещён в сторону, диафиз (низ) — к центру
    lat = float(up.mean() - lo.mean()) if len(up) and len(lo) else np.nan
    return {"sym": sym, "lat_score": lat}


def assign_regions(g: pd.DataFrame, expect: dict[str, bool]) -> pd.DataFrame:
    """Назначение области каждому изображению исследования.

    Область определяется геометрией (латеральность, симметрия), а состав
    разметки `expect` используется как перекрёстная проверка и как ответ
    на вопрос «есть ли в этом исследовании позвоночник вообще».

    Возвращает те же строки с колонками region, region_confident
    (геометрия однозначна), region_matches_markup (состав совпал с разметкой)
    и region_unlabeled (для этой области меток организатора нет).
    """
    g = g.copy()
    slots = ["spine"] if expect["spine"] else []
    if expect["femur_right"]:
        slots.append("femur_right")
    if expect["femur_left"]:
        slots.append("femur_left")

    g["region"] = None
    g["region_confident"] = False

    # --- шаг 1. позвоночник ------------------------------------------------- #
    # Разделяющий признак — не симметрия, а латеральность |lat_score|.
    # Симметрия обманывает: у позвоночника sym падает до 0.08 (сколиоз,
    # узкий кроп), у бедра доходит до 0.37, и в трёх исследованиях правило
    # «самый симметричный — позвоночник» меняет местами позвоночник и бедро
    # (проверено глазами на контрольных листах).
    # Латеральность разделяет классы с большим зазором: у позвоночника
    # |lat| не больше 27.5, у бедра не меньше 47.8 — запас почти вдвое.
    g["abs_lat"] = g["lat_score"].abs()
    top = g["abs_lat"].idxmin()
    if expect["spine"]:
        has_spine = True
    else:
        has_spine = bool(g.loc[top, "abs_lat"] < LAT_SPLIT)

    if has_spine:
        g.loc[top, "region"] = "spine"

    # --- шаг 2. сторона бедра по знаку латеральности ------------------------ #
    # Направление откалибровано по файлам «Для теста.zip» с явными именами:
    # ППОБ (правое) -> lat_score = +75, ЛПОБ (левое) -> -57.
    # На обучающем наборе femur_left максимум -10.4, femur_right минимум +47.8:
    # разделение по знаку без единой ошибки.
    fem = g["region"].isna()
    g.loc[fem & (g["lat_score"] >= 0), "region"] = "femur_right"
    g.loc[fem & (g["lat_score"] < 0), "region"] = "femur_left"
    g.loc[fem & g["lat_score"].isna(), "region"] = "femur_?"

    # --- шаг 3. перекрёстная проверка с разметкой --------------------------- #
    # Несовпадение состава само по себе НЕ означает ошибку геометрии:
    # в трёх исследованиях с эндопротезом организатор не разметил
    # протезированное бедро, а изображение в наборе есть. Такие изображения
    # остаются в таблице с labeled=False и в обучение не идут.
    got = list(g["region"].dropna())
    expected = {k for k, v in expect.items() if v}
    unique_sides = len(got) == len(set(got))

    # Геометрический зазор. Внутри исследования важна не абсолютная величина
    # |lat|, а отрыв позвоночника от ближайшего бедра: у «косых» укладок
    # позвоночник доходит до |lat| = 57, но ближайшее бедро всё равно дальше.
    # Если бедёр в исследовании нет, сравнивать не с чем — тогда работает
    # абсолютный порог, подстрахованный симметрией.
    fem_lat = g.loc[g["region"].astype(str).str.startswith("femur"), "abs_lat"]
    if has_spine and len(fem_lat):
        lat_ok = bool(fem_lat.min() - g.loc[top, "abs_lat"] > LAT_MARGIN)
    elif has_spine:
        lat_ok = bool(
            g.loc[top, "abs_lat"] < LAT_SPLIT or g.loc[top, "sym"] > SYM_THRESHOLD
        )
    else:
        lat_ok = bool(len(fem_lat) == 0 or fem_lat.min() > LAT_SPLIT)

    g["region_confident"] = unique_sides and "femur_?" not in got and lat_ok
    g["region_matches_markup"] = unique_sides and expected == set(got)
    g["region_unlabeled"] = ~g["region"].isin(expected)
    return g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", required=True, type=Path)
    ap.add_argument("--markup", required=True, type=Path)
    ap.add_argument("--root", required=True, type=Path,
                    help="каталог Исследования: rel_path из инвентаря строится от него")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    inv = pd.read_csv(args.inventory)
    inv["path"] = inv["rel_path"].map(lambda r: str(args.root / r))
    mk = load_markup(args.markup)

    missing = set(inv["study"]) ^ set(mk["study"])
    print(f"исследований в инвентаре: {inv['study'].nunique()}, в разметке: {len(mk)}, "
          f"расхождений по ключу: {len(missing)}")

    # 1. дедупликация внутри исследования
    inv = inv.sort_values(["study", "rel_path"])
    inv["dup_rank"] = inv.groupby(["study", "md5"]).cumcount()
    inv["n_copies"] = inv.groupby(["study", "md5"])["md5"].transform("size")
    uniq = inv[inv["dup_rank"] == 0].copy()
    print(f"после дедупликации: {len(uniq)} изображений из {len(inv)} файлов")

    # 2. признаки
    feats = uniq["path"].map(image_features).apply(pd.Series)
    uniq = pd.concat([uniq.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)

    # 3. назначение областей
    mk_idx = mk.set_index("study")
    parts = []
    for study, g in uniq.groupby("study"):
        if study in mk_idx.index:
            r = mk_idx.loc[study]
            expect = {
                "spine": bool(pd.notna(r["total_spine"])),
                "femur_right": bool(pd.notna(r["total_femur_right"])),
                "femur_left": bool(pd.notna(r["total_femur_left"])),
            }
        else:
            expect = {"spine": True, "femur_right": False, "femur_left": False}
        parts.append(assign_regions(g, expect))
    ds = pd.concat(parts, ignore_index=True)

    # 4. спуск меток на изображение
    ds = ds.merge(mk, on="study", how="left")

    def lab(row, col):
        return row[col] if pd.notna(row.get(col)) else np.nan

    recs = []
    for _, r in ds.iterrows():
        reg = r["region"]
        if reg == "spine":
            viol = {
                "SPINE_POSITIONING": lab(r, "spine_positioning"),
                "SPINE_AXIS_TILT": lab(r, "spine_axis_tilt"),
                "SPINE_ARTIFACT": lab(r, "spine_artifact"),
            }
            qc = r["total_spine"]
        elif reg == "femur_right":
            viol = {
                "FEMUR_POSITIONING_ROTATION": lab(r, "right_femur_rotation"),
                "FEMUR_ROI_COVERAGE": lab(r, "right_femur_roi"),
            }
            qc = r["total_femur_right"]
        elif reg == "femur_left":
            viol = {
                "FEMUR_POSITIONING_ROTATION": lab(r, "left_femur_rotation"),
                "FEMUR_ROI_COVERAGE": lab(r, "left_femur_roi"),
            }
            qc = r["total_femur_left"]
        else:
            viol, qc = {}, np.nan
        recs.append({
            "study": r["study"],
            "rel_path": r["rel_path"],
            "md5": r["md5"],
            "n_copies": r["n_copies"],
            "region": reg,
            "region_confident": r["region_confident"],
            "region_matches_markup": r["region_matches_markup"],
            "labeled": not bool(r["region_unlabeled"]),
            "sym": r["sym"],
            "lat_score": r["lat_score"],
            "rows": r["rows"],
            "cols": r["cols"],
            "mm_per_px": r["mm_per_px"],
            "mm_per_px_source": r["mm_per_px_source"],
            "quality_class": qc,
            "violations": "|".join(k for k, v in viol.items() if v == 1),
            **{f"y_{k}": v for k, v in viol.items()},
            "comment": r["comment"],
        })
    out = pd.DataFrame(recs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print("\nраспределение областей:")
    print(out["region"].value_counts().to_string())
    print(f"\nненадёжное назначение области: {(~out['region_confident']).sum()} изображений "
          f"в {out.loc[~out['region_confident'], 'study'].nunique()} исследованиях")
    n_unlab = int((~out["labeled"]).sum())
    print(f"изображений без меток организатора: {n_unlab} "
          f"в {out.loc[~out['labeled'], 'study'].nunique()} исследованиях "
          "(эндопротез / неразмеченная область — в обучение не идут)")
    mism = out.loc[~out["region_matches_markup"], "study"].nunique()
    print(f"исследований, где состав не совпал с разметкой: {mism}")
    print("\nquality_class по областям:")
    print(pd.crosstab(out["region"], out["quality_class"], dropna=False).to_string())
    print(f"\nзаписано: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
