#!/usr/bin/env python3
"""Двойной кросс-фиттинг локализатора и вложенная проверка ротации v2."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dxa_qc.metrics import binary_metrics, bootstrap_ci, f1_ci
from dxa_qc.trochanter import fit, rotation_features


def import_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


localizer = import_script("trochanter_localizer", ROOT / "scripts/20_trochanter_localizer.py")
cv = import_script("rotation_refinement", ROOT / "scripts/12_refine_rotation.py")
checkpoint = import_script("checkpoint", ROOT / "scripts/15_checkpoint_metrics.py")

VARIANTS = {name: [name] for name in
            ("trochanter_bump_mm", "trochanter_intensity", "trochanter_contrast")}
N_BOOT = 10000
TRANSFER_REF = "origin/codex/day21-rotation-transfer"
TRANSFER_PATH = "reports/rotation_local_transfer/nested_oof.csv"
TRANSFER_SNAPSHOT = ROOT / "reports/rotation_localizer/transfer_nested_oof.csv"


def feature_frame(frame, model, ann, images, tracks):
    rows = []
    for row in frame.sort_values("rel_path").itertuples(index=False):
        rec = {"study": row.study, "rel_path": row.rel_path, "image_id": "",
               "region": row.region, "fold": int(row.fold),
               "y": int(row.y_FEMUR_POSITIONING_ROTATION), "reason": ""}
        if row.rel_path in ann.index:
            rec["image_id"] = ann.loc[row.rel_path].image_id
        img, track = images[row.rel_path], tracks[row.rel_path]
        try:
            rec.update(rotation_features(img.pixels, row.region, img.mm_per_px, track, model,
                                         mm_per_px_source=img.mm_per_px_source,
                                         scanner_model=img.model))
        except ValueError as exc:
            rec["reason"] = str(exc)
        rows.append(rec)
    out = pd.DataFrame(rows)
    for feature in VARIANTS:
        if feature not in out:
            out[feature] = np.nan
    return out


def nested(data, ann, images, tracks):
    folds = sorted(int(v) for v in data.fold.unique())
    all_folds = set(folds)
    predictions, selections, inner_metrics, audit = [], [], [], []
    for outer in folds:
        inner_rows = []
        for inner in folds:
            if inner == outer:
                continue
            model = fit(localizer.train_points(ann, images, all_folds - {outer, inner}))
            part = feature_frame(data[data.fold == inner], model, ann, images, tracks)
            part["outer_fold"] = outer
            part["feature_role"] = "inner_oof"
            inner_rows.append(part)
        train = pd.concat(inner_rows, ignore_index=True)
        outer_model = fit(localizer.train_points(ann, images, all_folds - {outer}))
        val = feature_frame(data[data.fold == outer], outer_model, ann, images, tracks)
        val["outer_fold"] = outer
        val["feature_role"] = "outer_test"
        audit.extend(inner_rows + [val])

        candidates = []
        for variant in VARIANTS:
            score = cv.metric(cv.cross_validate(train, variant, VARIANTS))
            score.update(outer_fold=outer, variant=variant)
            inner_metrics.append(score)
            if score["specificity"] >= .75:
                candidates.append(score)
        if candidates:
            best = sorted(candidates, key=lambda m: (-m["f1"], m["fp"], m["variant"]))[0]
            out = cv.fit_predict(train, val, best["variant"], VARIANTS)
            selections.append({"fold": outer, "selected": best["variant"],
                               "inner_f1": best["f1"],
                               "inner_specificity": best["specificity"]})
        else:
            out = val[["study", "rel_path", "image_id", "region", "fold", "y"]].copy()
            out["variant"], out["pred"], out["risk"] = "no_qualifying_variant", -1, np.nan
            selections.append({"fold": outer, "selected": "no_qualifying_variant"})
        predictions.append(out)
        print(f"outer fold {outer}: {selections[-1]['selected']}", flush=True)
    return (pd.concat(predictions).sort_values("rel_path").reset_index(drop=True),
            pd.DataFrame(selections), pd.DataFrame(inner_metrics),
            pd.concat(audit, ignore_index=True))


def transfer_snapshot(out: Path) -> pd.DataFrame:
    """Вложенные решения ResNet18 из неслитой ветки. Если ветки нет (архив без
    `.git`, клон без удалённых веток), берётся закоммиченный снимок того же
    файла — `TRANSFER_SNAPSHOT`, иначе скрипт нельзя повторить вне репозитория."""
    try:
        raw = subprocess.check_output(["git", "show", f"{TRANSFER_REF}:{TRANSFER_PATH}"], cwd=ROOT,
                                      stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        raw = TRANSFER_SNAPSHOT.read_bytes()
    (out / "transfer_nested_oof.csv").write_bytes(raw)
    return pd.read_csv(io.BytesIO(raw))


def metrics_with_ci(frame: pd.DataFrame) -> dict:
    pred = frame.pred.clip(lower=0).to_numpy(int)
    m = cv.metric(frame)
    m["f1_ci95"] = list(f1_ci(pred, frame.y, groups=frame.study,
                               n_bootstrap=N_BOOT, seed=0))
    return m


def paired_delta(reference: pd.DataFrame, candidate: pd.DataFrame) -> list[float]:
    m = reference[["rel_path", "study", "y", "pred"]].merge(
        candidate[["rel_path", "y", "pred"]], on="rel_path", suffixes=("_a", "_b"),
        validate="one_to_one")
    if len(m) != 150 or not (m.y_a == m.y_b).all():
        raise ValueError("варианты ротации не совпадают по кадрам или меткам")
    values = m.pred_a.clip(lower=0).to_numpy(int)*2 + m.pred_b.clip(lower=0).to_numpy(int)
    stat = lambda y, v: (binary_metrics(y, (v % 2).astype(int)).f1
                         - binary_metrics(y, (v // 2).astype(int)).f1)
    point = stat(m.y_a.to_numpy(int), values)
    lo, hi = bootstrap_ci(m.y_a, values, stat, groups=m.study,
                          n_bootstrap=N_BOOT, seed=0)
    return [float(point), float(lo), float(hi)]


def comparison(v2: pd.DataFrame, transfer: pd.DataFrame):
    v1 = pd.read_csv(ROOT / "reports/rotation_human_profile/nested_oof.csv")
    rows, frames = [], {"v1": v1, "v2": v2, "transfer": transfer}
    for name, frame in frames.items():
        m = metrics_with_ci(frame)
        rows.append({"variant": name, **m,
                     "delta_f1_vs_v1": 0.0 if name == "v1" else paired_delta(v1, frame)[0]})
    return pd.DataFrame(rows), frames


def quality_comparison(frames: dict[str, pd.DataFrame]):
    base, _ = checkpoint.nested_predictions(checkpoint.load(ROOT / "reports", rotation=False))
    femur = base[base.area == "femur"][["rel_path", "study", "y_quality", "n_roi"]]
    rows = []
    for name, rot in frames.items():
        d = femur.merge(rot[["rel_path", "pred"]], on="rel_path", validate="one_to_one")
        pred = np.maximum(d.n_roi.to_numpy(int), d.pred.eq(1).to_numpy(int))
        m = binary_metrics(d.y_quality, pred).as_dict()
        m.pop("roc_auc")
        m.pop("pr_auc")
        m["f1_ci95"] = list(f1_ci(pred, d.y_quality, groups=d.study,
                                   n_bootstrap=N_BOOT, seed=0))
        rows.append({"variant": name, **m})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT / "Data/НД_для_обучения/Исследования")
    ap.add_argument("--out", type=Path, default=ROOT / "reports/rotation_localizer")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data, ann, images, tracks = localizer.load(args.root)
    v2, selections, inner, audit = nested(data, ann, images, tracks)
    transfer = transfer_snapshot(args.out)
    rotation, frames = comparison(v2, transfer)
    quality = quality_comparison(frames)
    v2.to_csv(args.out / "nested_oof.csv", index=False)
    selections.to_csv(args.out / "inner_selection.csv", index=False)
    inner.to_csv(args.out / "inner_metrics.csv", index=False)
    audit.to_csv(args.out / "double_crossfit_features.csv", index=False)
    rotation.to_csv(args.out / "rotation_comparison.csv", index=False)
    quality.to_csv(args.out / "quality_comparison.csv", index=False)
    v2[(v2.pred < 0) | (v2.pred != v2.y)].to_csv(args.out / "errors.csv", index=False)

    nested_metrics = metrics_with_ci(v2)
    summary = {
        "nested_metrics": nested_metrics,
        "delta_f1_vs_v1_ci95": paired_delta(frames["v1"], v2),
        "quality_femur": quality.set_index("variant").loc["v2"].to_dict(),
        "selections": selections.to_dict("records"),
        "variants": VARIANTS,
        "double_crossfit": True,
        "n_bootstrap": N_BOOT,
        "deployed": False,
        "input_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                         for p in ("reports/split.csv", "reports/dataset_images.csv",
                                   "annotations/femur_landmarks_v1.csv")},
        "limitations": ("Повторно использованный разведочный набор, независимого теста нет. "
                        "Точки врача используются для обучения локализатора, но не на инференсе. "
                        "ДИ бутстрепит фиксированные OOF-решения и не включает переобучение."),
    }
    # pandas/numpy типы приводим через JSON encoder round-trip.
    text = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False,
                      default=lambda x: x.item() if hasattr(x, "item") else list(x))
    (args.out / "summary.json").write_text(text+"\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
