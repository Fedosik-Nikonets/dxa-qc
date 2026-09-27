#!/usr/bin/env python3
"""Метрики контрольной точки 21 сентября: все правила на пяти фолдах.

Сводит в одну таблицу то, что до сих пор считалось по частям (скрипты 07,
08, 09), и отвечает на три вопроса, которых в пулированных цифрах README нет.

1. **Разброс по фолдам.** Каждое правило и каждый агрегат `quality_class`
   считаются отдельно на каждом фолде `reports/split.csv` (единица —
   исследование). На 6–10 позитивах в фолде оказывается 1–3 позитива, и
   разброс F1 между фолдами показывает, насколько вообще можно доверять
   одной цифре.
2. **Оптимизм порогов.** Пороги геометрических правил (укладка 0.04, наклон
   4.5°, охват 133 мм) выбирались глазами на всех данных. Здесь тот же выбор
   повторяется честно: на каждом внешнем фолде порог подбирается по F1 на
   остальных четырёх и применяется к отложенному. Разница с фиксированным
   порогом — цена выбора порога на тех же данных. Детектор артефактов уже
   out-of-fold (скрипт 08, порог из обучающих фолдов) и берётся как есть.
3. **Разрезы ТЗ 8.4.** Отдельно по областям, по типам нарушений, в целом;
   macro-F1 по типам нарушений; уровень исследования («есть ли в
   исследовании нарушение») и уровень строки выходной таблицы (с
   дубликатами, как её увидит организатор).

Решения правил берутся из `spine_rules.csv` / `femur_rules.csv` и сверяются
с диагностикой сквозного прогона: для геометрии обязаны совпасть построчно.
Артефакты в сквозном прогоне считаются моделью, обученной на всех данных, —
их расхождение с OOF печатается как мера того, насколько прогон на обучающем
наборе завышает качество, и в метрики не идёт.

Изображения не читаются, время работы — секунды (бутстреп 10 000 повторов).

Запуск:
    python scripts/15_checkpoint_metrics.py                      # с ротацией, reports/checkpoint_day9
    python scripts/15_checkpoint_metrics.py --no-rotation \
        --diagnostics <прогон до 23 сентября> --out reports/checkpoint_day7
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dxa_qc.femur import FIELD_MIN_MM  # noqa: E402
from dxa_qc.metrics import auc_ci, binary_metrics, f1_ci  # noqa: E402
from dxa_qc.spine import AXIS_TILT_DEG, COVERAGE_MIN  # noqa: E402

N_BOOT = 10000
FOLDS = range(5)

# Порядок строк в таблицах. Ключ -> (название, область, код типа нарушения).
RULES = {
    "pos": ("Позвоночник: укладка", "spine", "SPINE_POSITIONING"),
    "tilt": ("Позвоночник: наклон оси", "spine", "SPINE_AXIS_TILT"),
    "art": ("Позвоночник: артефакты", "spine", "SPINE_ARTIFACT"),
    "roi": ("Бедро: охват ROI", "femur", "FEMUR_ROI_COVERAGE"),
    "rot": ("Бедро: ротация", "femur", "FEMUR_POSITIONING_ROTATION"),
}
# Геометрические правила: для них честный порог подбирается заново.
GEOMETRIC = ("pos", "tilt", "roi")


def load(reports: Path, rotation: bool = True) -> pd.DataFrame:
    """Одна строка на размеченное уникальное изображение: метки, фолд, оценки, решения.

    `rotation=False` воспроизводит контрольную точку 21 сентября, когда
    правила ротации в конвейере не было (`p_rot = 0`). С 23 сентября
    правило есть: решение конвейера берётся из `rotation_rule/decisions.csv`
    (границы подогнаны на всех 150 изображениях — это цифра на обучении),
    а честное — из вложенной проверки `rotation_human_profile/nested_oof.csv`.
    """
    ds = pd.read_csv(reports / "dataset_images.csv")
    ds = ds[ds.labeled.astype(bool)].copy()
    split = pd.read_csv(reports / "split.csv")[["study", "fold"]]
    ds = ds.merge(split, on="study", how="left", validate="many_to_one")
    if ds.fold.isna().any():
        raise ValueError("изображения без фолда в split.csv")
    ds["area"] = np.where(ds.region.eq("spine"), "spine", "femur")

    sp = pd.read_csv(reports / "spine_rules.csv")[["rel_path", "coverage", "angle", "pred_pos", "pred_tilt"]]
    art = pd.read_csv(reports / "artifact_oof.csv")[["rel_path", "fold", "prob_oof", "pred_oof"]]
    fe = pd.read_csv(reports / "femur_rules.csv")[["rel_path", "field_h_mm", "pred_roi"]]
    df = (ds.merge(sp, on="rel_path", how="left", validate="one_to_one")
            .merge(art.rename(columns={"fold": "fold_art"}), on="rel_path", how="left", validate="one_to_one")
            .merge(fe, on="rel_path", how="left", validate="one_to_one"))

    spine, femur = df.area.eq("spine"), df.area.eq("femur")
    need = {"spine": ["coverage", "angle", "prob_oof", "pred_oof"], "femur": ["field_h_mm"]}
    for area, cols in need.items():
        mask = spine if area == "spine" else femur
        if df.loc[mask, cols].isna().any().any():
            raise ValueError(f"не у всех изображений области {area} есть решения правил")
    if not (df.loc[spine, "fold_art"] == df.loc[spine, "fold"]).all():
        raise ValueError("фолды artifact_oof.csv не совпадают со split.csv")

    # Метки: NaN для чужой области — у изображения бедра нет метки укладки позвоночника.
    df["y_pos"] = df.y_SPINE_POSITIONING
    df["y_tilt"] = df.y_SPINE_AXIS_TILT
    df["y_art"] = df.y_SPINE_ARTIFACT
    df["y_roi"] = df.y_FEMUR_ROI_COVERAGE
    df["y_rot"] = df.y_FEMUR_POSITIONING_ROTATION
    df["y_quality"] = df.quality_class.astype(int)

    # Оценки ориентированы так, что «больше — вероятнее нарушение».
    df["s_pos"] = -df.coverage
    df["s_tilt"] = df.angle.abs()
    df["s_art"] = df.prob_oof
    df["s_roi"] = -df.field_h_mm
    df["s_rot"] = np.nan

    # Решения конвейера, пересчитанные из порогов модулей — так видно, что
    # таблицы правил не устарели относительно кода.
    df["p_pos"] = (df.coverage < COVERAGE_MIN).astype(int)
    df["p_tilt"] = (df.angle.abs() > AXIS_TILT_DEG).astype(int)
    df["p_art"] = df.pred_oof.fillna(0).astype(int)
    df["p_roi"] = (df.field_h_mm < FIELD_MIN_MM).astype(int)
    df["p_rot"] = 0
    df["nr_rot"] = 0
    if rotation:
        dec = pd.read_csv(reports / "rotation_rule/decisions.csv")[["rel_path", "pred"]]
        nst = pd.read_csv(reports / "rotation_human_profile/nested_oof.csv")[["rel_path", "pred"]]
        df = (df.merge(dec.rename(columns={"pred": "rot_pipeline"}), on="rel_path", how="left",
                       validate="one_to_one")
                .merge(nst.rename(columns={"pred": "rot_nested"}), on="rel_path", how="left",
                       validate="one_to_one"))
        femur = df.area.eq("femur")
        if df.loc[femur, ["rot_pipeline", "rot_nested"]].isna().any().any():
            raise ValueError("не у всех изображений бедра есть решение ротации")
        # -1 — отказ правила: нарушения не ставится, как в конвейере.
        df["p_rot"] = df.rot_pipeline.eq(1).astype(int)
        df["nr_rot"] = df.rot_nested.eq(1).astype(int)
    for k, col in (("pos", "pred_pos"), ("tilt", "pred_tilt"), ("roi", "pred_roi")):
        mask = df[col].notna()
        bad = int((df.loc[mask, f"p_{k}"] != df.loc[mask, col].astype(int)).sum())
        if bad:
            raise ValueError(f"{k}: порог в модуле и решения в таблице правил разошлись на {bad} изображениях")
    for k in RULES:
        other = df[f"y_{k}"].isna()
        df.loc[other, f"p_{k}"] = 0
    return df


def check_pipeline(df: pd.DataFrame, diag_path: Path) -> dict:
    """Сверка с диагностикой сквозного прогона по типам нарушений."""
    diag = pd.read_csv(diag_path)
    diag = diag.drop_duplicates("md5").set_index("md5")
    out = {"matched_images": 0, "geometry_mismatches": 0, "artifact_in_sample_vs_oof": 0}
    for r in df.itertuples():
        if r.md5 not in diag.index:
            continue
        out["matched_images"] += 1
        got = set(str(diag.at[r.md5, "violation_type"]).split(";")) - {"", "nan"}
        for k in GEOMETRIC + ("rot",):
            if k == "rot" and r.area != "femur":
                continue
            if (RULES[k][2] in got) != bool(getattr(r, f"p_{k}")):
                out["geometry_mismatches"] += 1
        if r.area == "spine" and (RULES["art"][2] in got) != bool(r.p_art):
            out["artifact_in_sample_vs_oof"] += 1
    return out


def fit_threshold(score: np.ndarray, y: np.ndarray) -> float:
    """Порог по максимуму F1 на обучающих фолдах; нарушение, если score > t.

    Кандидаты — середины промежутков между соседними значениями, как ставились
    пороги вручную («порог в середине зазора»). При равенстве F1 берётся
    меньший порог: пропуск нарушения на скрининге дороже лишнего просмотра.
    """
    u = np.unique(score[np.isfinite(score)])
    cands = np.concatenate([[u[0] - 1.0], (u[:-1] + u[1:]) / 2, [u[-1] + 1.0]])
    best_f1, best_t = -1.0, float(cands[-1])
    for t in cands:
        p = score > t
        tp = int((p & (y == 1)).sum())
        fp = int((p & (y == 0)).sum())
        fn = int((~p & (y == 1)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if f1 > best_f1 + 1e-12:
            best_f1, best_t = f1, float(t)
    return best_t


def nested_predictions(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Для геометрических правил — порог из четырёх фолдов, решение на пятом."""
    df = df.copy()
    thresholds = []
    for k in GEOMETRIC:
        col = f"n_{k}"
        df[col] = 0
        lab = df[f"y_{k}"].notna()
        for f in FOLDS:
            train = lab & df.fold.ne(f)
            test = lab & df.fold.eq(f)
            t = fit_threshold(df.loc[train, f"s_{k}"].to_numpy(float), df.loc[train, f"y_{k}"].to_numpy(int))
            df.loc[test, col] = (df.loc[test, f"s_{k}"] > t).astype(int)
            # Обратно в единицы правила: для укладки и охвата оценка — со знаком минус.
            human = -t if k in ("pos", "roi") else t
            thresholds.append({"rule": k, "fold": f, "threshold": round(human, 4)})
    # Ротация: честное решение — из вложенной проверки напарника (`nr_rot`),
    # подбирать здесь нечего. Без него — то же, что у конвейера.
    df["n_art"] = df.p_art
    df["n_rot"] = df.nr_rot if "nr_rot" in df else df.p_rot
    return df, thresholds


def aggregate(df: pd.DataFrame, prefix: str) -> pd.Series:
    """`quality_class` изображения: нарушение, если сработало хоть одно правило."""
    return df[[f"{prefix}_{k}" for k in RULES]].max(axis=1).astype(int)


def row(name: str, y, p, score=None, groups=None, ci: bool = True, seed: int = 0) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=int)
    m = binary_metrics(y, p, None if score is None else np.asarray(score, float))
    out = {"срез": name, **{k: v for k, v in m.as_dict().items()}}
    if ci and m.n_pos and m.n_pos < m.n:
        out["f1_lo"], out["f1_hi"] = f1_ci(p, y, groups=groups, n_bootstrap=N_BOOT, seed=seed)
        if score is not None:
            out["auc_lo"], out["auc_hi"] = auc_ci(np.asarray(score, float), y, groups=groups,
                                                  n_bootstrap=N_BOOT, seed=seed)
    return out


def pooled_table(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    rows = []
    df = df.assign(q=aggregate(df, prefix))
    for k, (name, area, _) in RULES.items():
        lab = df[f"y_{k}"].notna()
        s = df.loc[lab, f"s_{k}"] if df.loc[lab, f"s_{k}"].notna().any() else None
        rows.append(row(name, df.loc[lab, f"y_{k}"], df.loc[lab, f"{prefix}_{k}"], s, df.loc[lab, "study"]))
    for area, name in (("spine", "Позвоночник, quality_class"), ("femur", "Бедро, quality_class")):
        m = df.area.eq(area)
        rows.append(row(name, df.loc[m, "y_quality"], df.loc[m, "q"], groups=df.loc[m, "study"]))
    rows.append(row("Все изображения, quality_class", df.y_quality, df.q, groups=df.study))

    # Уровень исследования: нарушение, если оно есть хоть на одном изображении.
    st = df.groupby("study").agg(y=("y_quality", "max"), p=("q", "max")).reset_index()
    rows.append(row("По исследованиям, quality_class", st.y, st.p, groups=st.study))

    # Уровень строки выходной таблицы: дубликаты весят столько, сколько их в наборе.
    rep = df.loc[df.index.repeat(df.n_copies)]
    rows.append(row("По строкам таблицы (с дубликатами)", rep.y_quality, rep.q, groups=rep.study))
    return pd.DataFrame(rows)


def macro_f1(pooled: pd.DataFrame) -> dict:
    """macro-F1 по типам нарушений (ТЗ 8.4): среднее F1 по типам в срезе."""
    f1 = dict(zip(pooled["срез"], pooled["f1"]))
    names = {k: v[0] for k, v in RULES.items()}
    spine = [f1[names[k]] for k in ("pos", "tilt", "art")]
    femur = [f1[names[k]] if f1[names[k]] == f1[names[k]] else 0.0 for k in ("roi", "rot")]
    return {"spine": float(np.mean(spine)), "femur": float(np.mean(femur)),
            "all_types": float(np.mean(spine + femur))}


def per_fold_table(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    rows = []
    df = df.assign(q=aggregate(df, prefix))
    slices = [(k, RULES[k][0]) for k in RULES if k != "rot" or df.p_rot.any()]
    for f in FOLDS:
        d = df[df.fold.eq(f)]
        for k, name in slices:
            lab = d[f"y_{k}"].notna()
            s = d.loc[lab, f"s_{k}"]
            r = row(name, d.loc[lab, f"y_{k}"], d.loc[lab, f"{prefix}_{k}"], s, ci=False)
            rows.append({"fold": f, **r})
        for area, name in (("spine", "Позвоночник, quality_class"), ("femur", "Бедро, quality_class")):
            m = d.area.eq(area)
            rows.append({"fold": f, **row(name, d.loc[m, "y_quality"], d.loc[m, "q"], ci=False)})
        rows.append({"fold": f, **row("Все изображения, quality_class", d.y_quality, d.q, ci=False)})
    return pd.DataFrame(rows)


def fold_summary(per_fold: pd.DataFrame) -> pd.DataFrame:
    """Среднее и разброс F1 по фолдам; фолды без позитивов не усредняются."""
    out = []
    for name, g in per_fold.groupby("срез", sort=False):
        g = g.sort_values("fold")
        ok = g[g.n_pos > 0]
        out.append({
            "срез": name,
            "позитивов по фолдам": "/".join(str(int(v)) for v in g.n_pos),
            # В фолде без позитивов F1 не определён: там либо 0/0, либо 0 из-за
            # одних ложных срабатываний. Ни то, ни другое не оценка правила.
            "F1 по фолдам": "/".join("—" if (n == 0 or v != v) else f"{v:.2f}"
                                     for n, v in zip(g.n_pos, g.f1)),
            "F1 среднее": ok.f1.mean(),
            "F1 ст.откл.": ok.f1.std(ddof=1),
            "F1 мин": ok.f1.min(),
            "F1 макс": ok.f1.max(),
            "фолдов без позитивов": int((g.n_pos == 0).sum()),
        })
    return pd.DataFrame(out)


def fmt_ci(r: pd.Series, lo: str, hi: str) -> str:
    return f"[{r[lo]:.2f}; {r[hi]:.2f}]" if lo in r and r[lo] == r[lo] else ""


def print_pooled(title: str, t: pd.DataFrame) -> None:
    print(f"\n{title}")
    for _, r in t.iterrows():
        auc = f" AUC={r.roc_auc:.3f}{fmt_ci(r, 'auc_lo', 'auc_hi')}" if r.roc_auc == r.roc_auc else ""
        print(f"  {r['срез']:36s} n={int(r.n):3d} pos={int(r.n_pos):2d} "
              f"TP={int(r.tp):2d} FP={int(r.fp):2d} FN={int(r.fn):2d} TN={int(r.tn):3d} "
              f"чувств={r.sensitivity:.2f} специф={r.specificity:.3f} "
              f"F1={r.f1:.2f}{fmt_ci(r, 'f1_lo', 'f1_hi')}{auc}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reports", type=Path, default=ROOT / "reports")
    ap.add_argument("--diagnostics", type=Path, default=ROOT / "reports/batch_diagnostics_train.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "reports/checkpoint_day9")
    ap.add_argument("--no-rotation", action="store_true",
                    help="без правила ротации — как на контрольной точке 21 сентября "
                         "(вместе с --diagnostics старого прогона и --out reports/checkpoint_day7)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = load(args.reports, rotation=not args.no_rotation)
    print(f"изображений: {len(df)}, исследований: {df.study.nunique()}, "
          f"строк таблицы: {int(df.n_copies.sum())}")
    check = check_pipeline(df, args.diagnostics)
    print(f"сверка со сквозным прогоном: {check}")
    if check["geometry_mismatches"]:
        raise SystemExit("геометрия разошлась со сквозным прогоном — метрики не о том конвейере")

    df, thresholds = nested_predictions(df)

    fixed = pooled_table(df, "p")
    nested = pooled_table(df, "n")
    print_pooled("ПОРОГИ КОНВЕЙЕРА, объединение пяти фолдов (артефакты OOF)", fixed)
    print_pooled("ЧЕСТНЫЕ ПОРОГИ: подбор на четырёх фолдах, решение на пятом", nested)

    per_fold = per_fold_table(df, "p")
    summary = fold_summary(per_fold)
    print("\nРАЗБРОС ПО ФОЛДАМ (пороги конвейера)")
    for _, r in summary.iterrows():
        print(f"  {r['срез']:36s} позитивы {r['позитивов по фолдам']:14s} F1 {r['F1 по фолдам']:28s} "
              f"среднее {r['F1 среднее']:.2f} ± {r['F1 ст.откл.']:.2f}")

    th = pd.DataFrame(thresholds)
    print("\nПОРОГИ, ПОДОБРАННЫЕ НА ЧЕТЫРЁХ ФОЛДАХ")
    for k, g in th.groupby("rule", sort=False):
        fixed_t = {"pos": COVERAGE_MIN, "tilt": AXIS_TILT_DEG, "roi": FIELD_MIN_MM}[k]
        print(f"  {RULES[k][0]:28s} конвейер {fixed_t:<7} по фолдам: "
              + ", ".join(f"{v:g}" for v in g.threshold))

    mf_fixed, mf_nested = macro_f1(fixed), macro_f1(nested)
    print(f"\nmacro-F1 по типам нарушений: конвейер {mf_fixed}, честные пороги {mf_nested}")

    fixed.to_csv(args.out / "pooled_pipeline_thresholds.csv", index=False)
    nested.to_csv(args.out / "pooled_nested_thresholds.csv", index=False)
    per_fold.to_csv(args.out / "per_fold.csv", index=False)
    summary.to_csv(args.out / "fold_summary.csv", index=False)
    th.to_csv(args.out / "nested_thresholds.csv", index=False)
    json.dump({"pipeline_check": check, "macro_f1_pipeline": mf_fixed, "macro_f1_nested": mf_nested,
               "n_images": int(len(df)), "n_studies": int(df.study.nunique()),
               "n_rows": int(df.n_copies.sum()), "n_bootstrap": N_BOOT},
              open(args.out / "summary.json", "w"), ensure_ascii=False, indent=2)
    print(f"\nзаписано: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
