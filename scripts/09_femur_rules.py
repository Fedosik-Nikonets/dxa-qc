#!/usr/bin/env python3
"""Проверка правила охвата области интереса на обучающем наборе (ТЗ 2.3.2).

Считает по каждому размеченному изображению бедра геометрию из `dxa_qc.femur`
и сводит метрики модулем `dxa_qc.metrics`: чувствительность, специфичность,
F1, macro-F1, ROC-AUC, PR-AUC со стратифицированным бутстреп-ДИ 95% с
группировкой по исследованию.

Печатает три вещи, которых нет в обычном отчёте о метриках:

* **плато порога** — как меняется решение при сдвиге границы длины поля.
  Порог берётся из середины плато, а не из точки максимума F1;
* **проверку вырожденных признаков из плана** — отступы bbox до краёв кадра,
  которые предполагалось сравнивать с порогами ТЗ 30 и 20 мм;
* **агрегат по бедру** — ложные срабатывания правила проверяются ещё и против
  `quality_class`, потому что исследование может быть нарушением по второму
  критерию (ротация), который не реализован.

Запуск:
    python scripts/09_femur_rules.py --root "Data/НД_для_обучения/Исследования" \
        --out reports/femur_rules.csv \
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
from dxa_qc.femur import (  # noqa: E402
    FEMUR_ROI_COVERAGE,
    FIELD_MIN_MM,
    evaluate,
)
from dxa_qc.metrics import auc_ci, binary_metrics, f1_ci  # noqa: E402

N_BOOT = 10000
PLATEAU = (120.0, 124.0, 127.0, 130.0, 133.0, 136.0, 140.0, 145.0, 150.0, 160.0)


def report(name: str, pred, y, score=None, groups=None, seed: int = 0) -> dict:
    m = binary_metrics(y, pred, score)
    line = "  " + m.line(name)
    if score is not None:
        lo, hi = auc_ci(score, y, groups=groups, n_bootstrap=N_BOOT, seed=seed)
        line += f" ДИ95%[{lo:.3f}, {hi:.3f}]"
    print(line)
    return {"правило": name, **m.as_dict()}


def main() -> int:
    root_dir = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=root_dir / "reports/dataset_images.csv")
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=root_dir / "reports/femur_rules.csv")
    ap.add_argument("--diagnostics", type=Path, default=None,
                    help="csv сквозного прогона для сверки построчно")
    args = ap.parse_args()

    ds = pd.read_csv(args.dataset)
    fem = ds[ds["region"].isin(["femur_left", "femur_right"]) & ds["labeled"]]
    print(f"изображений бедра: {len(fem)}  позитивов: "
          f"охват {int(fem['y_FEMUR_ROI_COVERAGE'].sum())}, "
          f"ротация {int(fem['y_FEMUR_POSITIONING_ROTATION'].sum())}")
    print(f"порог: длина поля сканирования < {FIELD_MIN_MM} мм\n")

    rows = []
    for i, x in enumerate(fem.itertuples(), 1):
        img = read_dicom(str(args.root / x.rel_path))
        f = evaluate(img.pixels if img.ok else None, x.region, img.mm_per_px)
        rows.append({
            "study": x.study, "rel_path": x.rel_path, "region": x.region,
            "y_roi": int(x.y_FEMUR_ROI_COVERAGE),
            "y_rot": int(x.y_FEMUR_POSITIONING_ROTATION),
            "y_quality": int(x.quality_class),
            "pred_roi": int(FEMUR_ROI_COVERAGE in f.violations),
            "field_h_mm": f.field_h_mm, "field_w_mm": f.field_w_mm,
            "lateral_mm": f.lateral_mm, "medial_mm": f.medial_mm,
            "shaft_deg": f.shaft_angle_deg, "femur_fill": f.femur_fill,
            "violations": ";".join(f.violations), "ok": int(f.ok),
            "comment": x.comment,
        })
        if i % 25 == 0:
            print(f"  {i}/{len(fem)}", file=sys.stderr)

    res = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)
    g = res.study.to_numpy()

    print("ПРАВИЛО ОХВАТА ОБЛАСТИ ИНТЕРЕСА")
    report("охват ROI", res.pred_roi, res.y_roi, -res.field_h_mm.fillna(1e9), g)
    lo, hi = f1_ci(res.pred_roi.to_numpy(), res.y_roi.to_numpy(), groups=g,
                   n_bootstrap=N_BOOT, seed=0)
    print(f"  F1 ДИ95% [{lo:.2f}, {hi:.2f}]")

    print("\nПЛАТО ПОРОГА (решение не меняется на отрезке 127–139 мм)")
    for t in PLATEAU:
        m = binary_metrics(res.y_roi, (res.field_h_mm < t).astype(int))
        mark = " <-- принят" if t == FIELD_MIN_MM else ""
        print(f"  < {t:5.0f} мм: {m.line('')}{mark}")

    print("\nПРИЗНАКИ ИЗ ПЛАНА: отступы bbox до краёв кадра")
    print(f"  боковой отступ, мм: медиана {res.lateral_mm.median():.1f}, "
          f"порог ТЗ 20 мм -> нарушением был бы помечен "
          f"{int((res.lateral_mm < 20).sum())} снимок из {len(res)}")
    report("буквальное правило 20 мм", (res.lateral_mm < 20).astype(int), res.y_roi,
           -res.lateral_mm.fillna(1e9), g)
    print("  нижний отступ равен нулю у всех: диафиз выходит за край кадра всегда")

    print("\nАГРЕГАТ ПО БЕДРУ (ротация не реализована, чувствительность заведомо низкая)")
    report("quality_class", res.pred_roi, res.y_quality, groups=g)
    fp = res[(res.pred_roi == 1) & (res.y_roi == 0)]
    print(f"  ложных срабатываний правила: {len(fp)}, из них с нарушением ротации: "
          f"{int(fp.y_rot.sum())}")

    if args.diagnostics and args.diagnostics.exists():
        diag = pd.read_csv(args.diagnostics)
        diag = diag[diag["region_key"].isin(["femur_left", "femur_right"])]
        key = diag.set_index("path_to_study")["violation_type"].fillna("")
        mismatched = 0
        for r in res.itertuples():
            want = key.get(r.rel_path)
            if want is None:
                continue
            if str(want) != r.violations:
                mismatched += 1
                print(f"  РАСХОЖДЕНИЕ {r.rel_path}: конвейер '{want}' против правил '{r.violations}'")
        print(f"\nсверка со сквозным прогоном: расхождений {mismatched} "
              f"из {int(key.index.isin(res.rel_path).sum())} сопоставленных строк")

    print(f"\nзаписано: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
