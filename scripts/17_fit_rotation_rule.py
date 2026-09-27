#!/usr/bin/env python3
"""Заморозка правила ротации бедра для конвейера (23 сентября).

Вложенная проверка напарника (`scripts/14_rotation_human_profile.py`)
оценила **процедуру**: на каждом внешнем фолде признак выбирается из
четырёх по F1 при специфичности ≥ 0.75 на внутренних фолдах, границы двух
хвостов подгоняются на обучающих фолдах. Итог процедуры — F1 0.48.

Здесь та же процедура применяется один раз ко всем 150 изображениям —
так получают модель для развёртывания после оценки. Никаких новых
степеней свободы: те же четыре признака, то же правило выбора с теми же
разрывами ничьих (`12_refine_rotation.nested_select`), та же подгонка
интервала (`rotation.fit_normal_interval`).

Выход:
* `models/rotation_profile_v1.json` — признак, границы, происхождение;
* `reports/rotation_rule/decisions.csv` — решение на каждом из 150
  изображений; по нему сквозной прогон сверяется построчно;
* `reports/rotation_rule/selection.csv` — метрики пяти фолдов по каждому
  признаку на всех данных (то, по чему выбран признак).

Цифры в `decisions.csv` — **на обучении**, их нельзя показывать как
качество. Качество — вложенная оценка, она же переносится в json.

Запуск (изображения не читаются, секунды):
    python scripts/17_fit_rotation_rule.py
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dxa_qc.metrics import binary_metrics  # noqa: E402
from dxa_qc.rotation import fit_normal_interval  # noqa: E402
from dxa_qc.rotation_rule import FEATURES, RotationModel  # noqa: E402

_spec = importlib.util.spec_from_file_location("refinement", ROOT / "scripts/12_refine_rotation.py")
cv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cv)

VARIANTS = {name: [name] for name in FEATURES}   # как в скрипте 14
MIN_SPECIFICITY = 0.75                            # как в nested_select
FEATURES_CSV = ROOT / "reports/rotation_human_profile/features.csv"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(features: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Внутренний шаг `nested_select`, применённый ко всем пяти фолдам."""
    rows = []
    for variant in VARIANTS:
        m = cv.metric(cv.cross_validate(features, variant, VARIANTS))
        rows.append({"variant": variant, **m})
    table = pd.DataFrame(rows)
    ok = table[table.specificity >= MIN_SPECIFICITY]
    if ok.empty:
        raise SystemExit("ни один признак не проходит порог специфичности — правило не строится")
    best = sorted(ok.to_dict("records"), key=lambda m: (-m["f1"], m["fp"], m["variant"]))[0]
    return best["variant"], table


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--features", type=Path, default=FEATURES_CSV)
    ap.add_argument("--model", type=Path, default=ROOT / "models/rotation_profile_v1.json")
    ap.add_argument("--out", type=Path, default=ROOT / "reports/rotation_rule")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(args.features)
    features["image_id"] = features.image_id.fillna("")
    if len(features) != 150 or features.y.sum() != 36:
        raise SystemExit("ожидались 150 изображений бедра и 36 позитивов ротации")

    variant, table = select(features)
    ref = pd.read_csv(ROOT / "reports/rotation_human_profile/candidate_metrics.csv")
    for col in ("tp", "fp", "fn", "tn"):
        if not (ref.set_index("variant")[col] == table.set_index("variant")[col]).all():
            raise SystemExit("метрики выбора не совпали с candidate_metrics.csv скрипта 14")

    score = features[variant].to_numpy(float)
    y = features.y.to_numpy(int)
    low, high = fit_normal_interval(score, y, MIN_SPECIFICITY)

    nested = json.loads((ROOT / "reports/rotation_human_profile/summary.json").read_text())
    nm = nested["nested_metrics"]
    model = RotationModel(feature=variant, low=low, high=high, meta={
        "fitted_on": "150 бедренных изображений обучающего набора, все пять фолдов",
        "selection_rule": f"max F1 при специфичности >= {MIN_SPECIFICITY} на 5 фолдах; "
                          "ничьи — меньше FP, затем имя",
        "selection_cv": {r.variant: {"f1": round(r.f1, 4), "specificity": round(r.specificity, 4)}
                         for r in table.itertuples()},
        "nested_estimate": {"f1": round(nm["f1"], 4), "f1_ci95": [round(v, 4) for v in nm["f1_ci95"]],
                            "specificity": round(nm["specificity"], 4),
                            "tp": nm["tp"], "fp": nm["fp"], "fn": nm["fn"], "tn": nm["tn"],
                            "source": "reports/rotation_human_profile/summary.json"},
        "inputs_sha256": {"reports/rotation_human_profile/features.csv": sha(args.features),
                          "reports/split.csv": sha(ROOT / "reports/split.csv")},
    })
    model.save(args.model)

    dec = features[["study", "rel_path", "region", "fold", "y"]].copy()
    dec["feature"] = variant
    dec["value"] = score
    dec["pred"] = [model.decide(v) for v in score]
    dec.to_csv(args.out / "decisions.csv", index=False)
    table.to_csv(args.out / "selection.csv", index=False)

    ins = binary_metrics(y, dec.pred.clip(lower=0).to_numpy())
    print("выбор признака (5 фолдов, все данные):")
    for r in table.itertuples():
        mark = " <-" if r.variant == variant else ""
        print(f"  {r.variant:18s} F1={r.f1:.3f} специф={r.specificity:.3f}{mark}")
    print(f"признак: {variant}, нормальный интервал [{low:.4g}; {high:.4g}]")
    print(f"на обучении (не качество!): TP={ins.tp} FP={ins.fp} FN={ins.fn} TN={ins.tn} F1={ins.f1:.3f}; "
          f"отказов: {int((dec.pred < 0).sum())}")
    print(f"качество процедуры (вложенно): F1={nm['f1']:.3f} {nm['f1_ci95']}")
    print(f"записано: {args.model}, {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
