#!/usr/bin/env python3
"""Проверка правил позвоночника на обучающем наборе.

Считает по каждому размеченному изображению позвоночника два геометрических
правила из `dxa_qc.spine` и сводит метрики модулем `dxa_qc.metrics`:
чувствительность, специфичность, сбалансированная точность, F1, macro-F1,
ROC-AUC, PR-AUC со стратифицированным бутстреп-ДИ 95%.

Детектор артефактов сюда входит **только out-of-fold** — предсказаниями из
`reports/artifact_oof.csv`, посчитанными скриптом 08. Взять его решение
на обучающих данных значило бы показать метрику модели на том, на чём она
обучалась, и агрегат `quality_class` стал бы недостоверным. Отдельно печатает ту же метрику по
подвыборке без находок вне перечня ТЗ 2.3 — как договорились в таксономии
17 сентября: основная метрика считается по полной выборке (на инференсе
ничего не выкинешь), а подвыборка нужна только для разбора источника ошибок.

Дополнительно сверяет решение правил с колонкой `violation_type` в
диагностике сквозного прогона: конвейер и этот скрипт обязаны совпадать
построчно, иначе где-то разъехались параметры.

Запуск:
    python scripts/07_spine_rules.py --dataset reports/dataset_images.csv \
        --root "Data/НД_для_обучения/Исследования" \
        --out reports/spine_rules.csv \
        --diagnostics reports/batch_diagnostics_train.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dxa_qc.dicom_io import read_dicom  # noqa: E402
from dxa_qc.metrics import auc_ci, binary_metrics  # noqa: E402
from dxa_qc.spine import (  # noqa: E402
    AXIS_TILT_DEG,
    COVERAGE_MIN,
    SPINE_AXIS_TILT,
    SPINE_POSITIONING,
    evaluate,
)

N_BOOT = 10000


def report(name: str, pred, y, score=None, groups=None, seed: int = 0) -> dict:
    """Печатает строку метрик; при наличии score добавляет AUC с ДИ."""
    m = binary_metrics(y, pred, score)
    line = "  " + m.line(name)
    if score is not None:
        lo, hi = auc_ci(score, y, groups=groups, n_bootstrap=N_BOOT, seed=seed)
        line += f" ДИ95%[{lo:.3f}, {hi:.3f}]"
    print(line)
    return {"правило": name, **m.as_dict()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--diagnostics", type=Path, default=None,
                    help="csv сквозного прогона для сверки построчно")
    ap.add_argument("--artifact-oof", type=Path, default=Path("reports/artifact_oof.csv"),
                    help="out-of-fold предсказания детектора артефактов (скрипт 08)")
    ap.add_argument("--out-of-scope", type=Path, default=Path("reports/out_of_scope_spine.csv"))
    args = ap.parse_args()

    ds = pd.read_csv(args.dataset)
    spine = ds[(ds["region"] == "spine") & ds["labeled"]]
    print(f"изображений позвоночника: {len(spine)}  "
          f"позитивов: укладка {int(spine['y_SPINE_POSITIONING'].sum())}, "
          f"ось {int(spine['y_SPINE_AXIS_TILT'].sum())}, "
          f"артефакты {int(spine['y_SPINE_ARTIFACT'].sum())}")
    print(f"пороги: наклон оси > {AXIS_TILT_DEG}°, латеральная плотность < {COVERAGE_MIN}\n")

    rows = []
    for i, x in enumerate(spine.itertuples(), 1):
        img = read_dicom(str(args.root / x.rel_path))
        f = evaluate(img.pixels if img.ok else None, img.mm_per_px)
        rows.append({
            "study": x.study, "rel_path": x.rel_path,
            "y_pos": int(x.y_SPINE_POSITIONING), "y_tilt": int(x.y_SPINE_AXIS_TILT),
            "y_art": int(x.y_SPINE_ARTIFACT), "y_quality": int(x.quality_class),
            "pred_pos": int(SPINE_POSITIONING in f.violations),
            "pred_tilt": int(SPINE_AXIS_TILT in f.violations),
            "coverage": f.coverage, "angle": f.axis_angle_deg,
            "res_rms": f.res_rms, "res_amp": f.res_amp,
            "slope_spread": f.slope_spread,
            "violations": ";".join(f.violations), "ok": int(f.ok),
            "comment": x.comment,
        })
        if i % 25 == 0:
            print(f"  {i}/{len(spine)}", file=sys.stderr)

    res = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)

    # Артефакты — только out-of-fold, иначе метрика смотрит на обучающие данные.
    if args.artifact_oof and args.artifact_oof.exists():
        oof = pd.read_csv(args.artifact_oof)[["rel_path", "pred_oof", "prob_oof"]]
        res = res.merge(oof, on="rel_path", how="left", validate="one_to_one")
        if res[["pred_oof", "prob_oof"]].isna().any().any():
            raise ValueError("incomplete artifact OOF predictions; cannot score missing rows as negative")
        res["pred_art"] = res.pred_oof.astype(int)
    else:
        res["pred_art"] = 0
        res["prob_oof"] = float("nan")
        print("предупреждение: нет out-of-fold артефактов, агрегат без них")

    print("ПОЛНАЯ ВЫБОРКА (основная метрика)")
    g = res.study.to_numpy()
    report("укладка", res.pred_pos, res.y_pos, -res.coverage.fillna(1e9), g)
    report("наклон оси", res.pred_tilt, res.y_tilt, res.angle.abs(), g)
    report("артефакты (OOF)", res.pred_art, res.y_art, res.prob_oof, g)
    pred_any = (res.pred_pos | res.pred_tilt | res.pred_art).astype(int)
    report("quality_class", pred_any, res.y_quality, groups=g)

    # Диагностика отвечает на один вопрос: сколько наших ошибок создают
    # находки вне перечня 2.3 (в основном сколиоз). Поэтому выбрасываются
    # только НЕГАТИВЫ с такой находкой — случаи, где правило сработало, а
    # эксперт нарушения не увидел. Позитивы остаются все: убрать их значит
    # считать метрику на другой выборке, а не искать источник ошибок.
    #
    # Проверено 19 сентября: при выбрасывании исследований целиком из 99
    # изображений остаётся 79, теряется 3 позитива артефактов из 17, и
    # диагностика становится ХУЖЕ основной метрики (укладка 0.80 -> 0.75,
    # артефакты 0.71 -> 0.67). Такая цифра не диагностирует ничего.
    #
    # Список исследований берётся из out_of_scope_spine.csv, а не из
    # регулярки по комментарию: он явный и проверяемый. На цифры выбор
    # источника не влияет, оба дают одно и то же.
    oos = res.study.isin(set(pd.read_csv(args.out_of_scope).study))
    print("\nДИАГНОСТИКА: без негативов с находками вне перечня 2.3")
    k = ~(oos & (res.y_pos == 0))
    report("укладка", res.pred_pos[k], res.y_pos[k], -res.coverage[k].fillna(1e9), res.study[k])
    k = ~(oos & (res.y_tilt == 0))
    report("наклон оси", res.pred_tilt[k], res.y_tilt[k], res.angle[k].abs(), res.study[k])
    k = ~(oos & (res.y_art == 0))
    report("артефакты (OOF)", res.pred_art[k], res.y_art[k], res.prob_oof[k], res.study[k])
    k = ~(oos & (res.y_quality == 0))
    report("quality_class", pred_any[k], res.y_quality[k], groups=res.study[k])

    if args.diagnostics and args.diagnostics.exists():
        diag = pd.read_csv(args.diagnostics)
        diag = diag[diag["region_key"] == "spine"]
        key = diag.set_index("path_to_study")["violation_type"].fillna("")
        mismatched = 0
        for r in res.itertuples():
            want = key.get(r.rel_path)
            if want is None:
                continue
            want = ";".join(v for v in str(want).split(";") if v and v != "SPINE_ARTIFACT")
            if want != r.violations:
                mismatched += 1
                print(f"  РАСХОЖДЕНИЕ {r.rel_path}: конвейер '{want}' против правил '{r.violations}'")
        print(f"\nсверка со сквозным прогоном: расхождений {mismatched} "
              f"из {int(key.index.isin(res.rel_path).sum())} сопоставленных строк")

    print(f"\nзаписано: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
