#!/usr/bin/env python3
"""Заморозить локализатор и правило v2 только после прохождения метрик go."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dxa_qc.rotation import fit_normal_interval
from dxa_qc.trochanter import FEATURE_VERSION, fit


def import_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


localizer = import_script("trochanter_localizer", ROOT / "scripts/20_trochanter_localizer.py")
cv = import_script("rotation_refinement", ROOT / "scripts/12_refine_rotation.py")
VARIANTS = {name: [name] for name in
            ("trochanter_bump_mm", "trochanter_intensity", "trochanter_contrast")}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT / "Data/НД_для_обучения/Исследования")
    ap.add_argument("--report", type=Path, default=ROOT / "reports/rotation_localizer")
    ap.add_argument("--localizer", type=Path, default=ROOT / "models/trochanter_localizer_v1.npz")
    ap.add_argument("--rule", type=Path, default=ROOT / "models/rotation_profile_v2.json")
    args = ap.parse_args()

    summary = json.loads((args.report / "summary.json").read_text(encoding="utf-8"))
    nm, qm = summary["nested_metrics"], summary["quality_femur"]
    failed = []
    if nm["f1"] < .53 or nm["specificity"] < .75:
        failed.append("вложенный F1/специфичность ротации")
    if qm["f1"] < .527:
        failed.append("вложенный F1 quality_class бедра")
    if failed:
        print("ВЕРДИКТ: no-go — " + ", ".join(failed) + "; файлы models/ не изменены")
        return 0

    # Этот путь выполняется только после прохождения двух первых ворот.
    data, ann, images, _ = localizer.load(args.root)
    model = fit(localizer.train_points(ann, images, set(int(v) for v in data.fold.unique())))
    features = pd.read_csv(args.report / "oof_features.csv")
    choices = []
    for variant in VARIANTS:
        m = cv.metric(cv.cross_validate(features, variant, VARIANTS))
        choices.append({"variant": variant, **m})
    qualifying = [r for r in choices if r["specificity"] >= .75]
    if not qualifying:
        raise SystemExit("ни один признак не прошёл порог специфичности")
    best = sorted(qualifying, key=lambda r: (-r["f1"], r["fp"], r["variant"]))[0]
    low, high = fit_normal_interval(features[best["variant"]], features.y, .75)
    model.save(args.localizer)
    payload = {
        "feature_version": f"{FEATURE_VERSION}_rotation_profile_v2",
        "feature": best["variant"], "low": low, "high": high,
        "localizer": args.localizer.name,
        "fitted_on": "150 кадров бедра; локализатор — 58 точек врача",
        "selection_rule": "max F1 при специфичности >= 0.75; OOF-признаки локализатора",
        "selection_cv": {r["variant"]: {"f1": r["f1"], "specificity": r["specificity"]}
                         for r in choices},
        "nested_estimate": nm,
        "inputs_sha256": {
            "reports/rotation_localizer/oof_features.csv": sha(args.report / "oof_features.csv"),
            "reports/split.csv": sha(ROOT / "reports/split.csv"),
            "annotations/femur_landmarks_v1.csv": sha(ROOT / "annotations/femur_landmarks_v1.csv"),
        },
    }
    args.rule.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                    allow_nan=False)+"\n", encoding="utf-8")
    print(f"кандидаты записаны: {args.localizer}, {args.rule}; до переключения default нужны пункты 3–5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
