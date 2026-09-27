#!/usr/bin/env python3
"""Evaluate landmark-assisted and automatic rotation hypotheses, fixed folds.

Manual anchors are an exploratory measurement on the selected 80 images;
they are never used by automatic inference. Thresholds use training studies
only. This script writes research reports, not production model weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dxa_qc.artifact import fit_logistic, standardize
from dxa_qc.dicom_io import read_dicom
from dxa_qc.femur import femur_mask, field_scale, shaft_angle
from dxa_qc.metrics import best_threshold, binary_metrics, f1_ci, auc_ci
from dxa_qc.rotation import (
    automatic_anchor, fit_normal_interval, interval_predict, medial_contour, prominence,
)
from dxa_qc.segmentation import segment_bone


def extract(root: Path) -> pd.DataFrame:
    ds = pd.read_csv(ROOT / "reports/dataset_images.csv")
    ds = ds[ds.region.isin(["femur_left", "femur_right"]) & ds.labeled].copy()
    split = pd.read_csv(ROOT / "reports/split.csv")
    ds = ds.merge(split[["study", "fold"]], on="study", validate="many_to_one")
    manifest = pd.read_csv(ROOT / "annotations/femur_manifest_v1.csv")
    points = pd.read_csv(ROOT / "annotations/femur_points_v1.csv")
    ann = manifest.merge(points, on="image_id", validate="one_to_one").set_index("rel_path")
    rows = []
    for row in ds.sort_values("rel_path").itertuples(index=False):
        rec = {"study": row.study, "rel_path": row.rel_path, "region": row.region,
               "fold": int(row.fold), "y_rot": int(row.y_FEMUR_POSITIONING_ROTATION),
               "selected": row.rel_path in ann.index, "image_id": "",
               "manual_reason": "not_annotated", "auto_reason": "",
               "manual_prominence_mm": np.nan, "auto_prominence_mm": np.nan,
               "manual_minus5_mm": np.nan, "manual_plus5_mm": np.nan,
               "auto_anchor_error_mm": np.nan}
        annotation = ann.loc[row.rel_path] if rec["selected"] else None
        if annotation is not None:
            if annotation.study != row.study or annotation.fold != row.fold:
                raise ValueError("annotation manifest does not match fixed split")
            rec.update(image_id=annotation.image_id, lt_state=annotation.lt_state,
                       manual_reason="not_visible" if annotation.lt_state == "not_visible" else "")
        img = read_dicom(root / row.rel_path)
        if not img.ok or img.pixels is None:
            raise ValueError(f"Cannot decode {row.rel_path}: {img.error}")
        mm, _ = field_scale(img.mm_per_px)
        mask = femur_mask(img.pixels, segment_bone(img.pixels))
        if mask is None:
            rec["auto_reason"] = "missing femur mask"
            rows.append(rec)
            continue
        rec["mm_per_px"] = mm
        rec["shaft_deg"] = shaft_angle(mask)
        rr, cc = np.nonzero(mask)
        rec["fill"] = float(mask.sum() / ((rr.max()-rr.min()+1) * (cc.max()-cc.min()+1)))
        try:
            contour = medial_contour(mask, row.region, mm)
        except ValueError as exc:
            rec["auto_reason"] = str(exc)
            rows.append(rec)
            continue
        try:
            ay = automatic_anchor(contour)
            auto = prominence(contour, ay)
            rec.update(auto_prominence_mm=auto["prominence_mm"], auto_anchor_y=ay)
            if annotation is not None and np.isfinite(annotation.lt_y):
                rec["auto_anchor_error_mm"] = abs(ay - annotation.lt_y) * mm
        except ValueError as exc:
            rec["auto_reason"] = str(exc)
        if annotation is not None and np.isfinite(annotation.lt_y):
            try:
                manual = prominence(contour, annotation.lt_y)
                rec.update(manual_prominence_mm=manual["prominence_mm"],
                           manual_anchor_y=float(annotation.lt_y),
                           manual_window_overlaps_fit=manual["window_overlaps_shaft_fit"])
                rec["manual_minus5_mm"] = prominence(contour, annotation.lt_y-5)["prominence_mm"]
                rec["manual_plus5_mm"] = prominence(contour, annotation.lt_y+5)["prominence_mm"]
            except ValueError as exc:
                rec["manual_reason"] = str(exc)
        rows.append(rec)
    result = pd.DataFrame(rows)
    # Reuse the exact rectified-mask fill from the day5 comparator. Original
    # and rectified bbox fill differ; substituting one changes the baseline.
    reference = pd.read_csv(ROOT / "reports/femur_rotation.csv")
    reference = reference[["rel_path", "study", "fold", "y_rot", "fill"]].rename(
        columns={"fill": "reference_rectified_fill"})
    result = result.merge(reference, on=["rel_path", "study", "fold", "y_rot"],
                          how="left", validate="one_to_one")
    if result.reference_rectified_fill.isna().any():
        raise ValueError("Day5 comparator does not match dataset and split")
    return result


def evaluate_oof(frame: pd.DataFrame, variant: str, columns: list[str], interval: bool) -> pd.DataFrame:
    x = frame[columns].to_numpy(float)
    y, folds = frame.y_rot.to_numpy(int), frame.fold.to_numpy(int)
    valid = np.isfinite(x).all(axis=1)
    result = frame[["study", "rel_path", "region", "fold", "y_rot", "image_id", "selected"]].copy()
    result["variant"] = variant
    result["pred"] = -1
    result["score"] = np.nan
    result["threshold_low"] = np.nan
    result["threshold_high"] = np.nan
    for fold in sorted(np.unique(folds)):
        train, val = (folds != fold) & valid, (folds == fold) & valid
        assert not set(frame.study[train]) & set(frame.study[val])
        if len(np.unique(y[train])) != 2:
            raise ValueError(f"Training fold {fold} has only one class")
        if interval:
            low, high = fit_normal_interval(x[train, 0], y[train])
            pred, score = interval_predict(x[val, 0], low, high)
            # Normalize the boundary distance using training data only.
            score /= max(float(np.std(x[train, 0])), 1e-6)
            result.loc[val, ["threshold_low", "threshold_high"]] = [low, high]
        else:
            mu, sd = standardize(x[train])
            w, b = fit_logistic((x[train]-mu)/sd, y[train], 1.)
            ptrain = 1 / (1 + np.exp(-np.clip((x[train]-mu)/sd @ w + b, -60, 60)))
            t = best_threshold(ptrain, y[train])
            score = 1 / (1 + np.exp(-np.clip((x[val]-mu)/sd @ w + b, -60, 60)))
            pred = (score >= t).astype(int)
            result.loc[val, "threshold_high"] = t
        result.loc[val, "pred"] = pred
        result.loc[val, "score"] = score
    return result


def summarize(oof: pd.DataFrame, cohort: str, n_boot: int) -> dict:
    valid = oof.pred >= 0
    scored = oof[valid]
    m = binary_metrics(scored.y_rot, scored.pred, scored.score).as_dict()
    ci = f1_ci(scored.pred, scored.y_rot, groups=scored.study, n_bootstrap=n_boot)
    auc = auc_ci(scored.score, scored.y_rot, groups=scored.study, n_bootstrap=n_boot)
    return {"variant": oof.variant.iloc[0], "cohort": cohort,
            "n_total": len(oof), "n_abstain": int((~valid).sum()),
            "abstain_positives": int(oof.loc[~valid, "y_rot"].sum()),
            **m, "f1_ci_low": ci[0], "f1_ci_high": ci[1],
            "auc_ci_low": auc[0], "auc_ci_high": auc[1]}


def render_review(features: pd.DataFrame, root: Path, output: Path):
    """Scientific overlay: contour, shaft fit and annotation search window."""
    selected = features[features.image_id.isin(["F001", "F015", "F028", "F038", "F043", "F063", "F072", "F080"])]
    page = Image.new("RGB", (4 * 360, 2 * 440), (24, 24, 24))
    for i, row in enumerate(selected.itertuples()):
        img = read_dicom(root / row.rel_path)
        mask = femur_mask(img.pixels, segment_bone(img.pixels))
        c = medial_contour(mask, row.region, row.mm_per_px)
        tile = Image.new("RGB", (360, 440), (24, 24, 24))
        pixels = img.pixels[:, ::-1] if row.region == "femur_right" else img.pixels
        tile.paste(Image.fromarray(np.uint8(np.clip(pixels, 0, 1) * 255)).convert("RGB"), (35, 35))
        draw = ImageDraw.Draw(tile)
        draw.text((8, 8), f"{row.image_id} y={row.y_rot} {row.lt_state}", fill="white")
        for y, x in enumerate(c.medial):
            if np.isfinite(x):
                draw.point((35 + x, 35 + y), fill="cyan")
        draw.line((35+c.intercept, 35, 35+c.slope*(len(c.medial)-1)+c.intercept,
                   35+len(c.medial)-1), fill="magenta", width=1)
        if np.isfinite(row.manual_anchor_y):
            for y in [row.manual_anchor_y-6/c.mm, row.manual_anchor_y+6/c.mm]:
                draw.line((35, 35+y, 315, 35+y), fill="yellow")
        draw.text((8, 390), f"prominence={row.manual_prominence_mm:.1f} mm", fill="white")
        draw.text((8, 406), "Cyan=contour  Magenta=shaft  Yellow=window", fill="white")
        draw.text((8, 422), "Right femurs reflected; research only", fill="white")
        page.paste(tile, ((i % 4) * 360, (i // 4) * 440))
    page.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "Data/НД_для_обучения/Исследования")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/rotation_day6")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    features = extract(args.root)
    features["manual_disjoint_mm"] = features.manual_prominence_mm.where(
        features.manual_window_overlaps_fit == False)
    features.to_csv(args.out / "features.csv", index=False)
    # 80 selected images contain every positive, so their prevalence differs
    # from all 150 images. Compare variants on exactly the same 72 anchors too.
    assisted = features[features.manual_prominence_mm.notna()].reset_index(drop=True)
    variants = []
    for frame, cohort in [(features, "all150"), (assisted, "paired_annotated")]:
        for name, cols, interval in [
            ("shaft", ["shaft_deg"], False),
            ("shaft_fill_reference", ["shaft_deg", "reference_rectified_fill"], False),
            ("automatic_contour", ["auto_prominence_mm"], True),
        ]:
            variants.append((cohort, evaluate_oof(frame, name, cols, interval)))
    variants.append(("selected80", evaluate_oof(features[features.selected].reset_index(drop=True),
                                               "manual_contour", ["manual_prominence_mm"], True)))
    variants.append(("selected80", evaluate_oof(features[features.selected].reset_index(drop=True),
                                               "manual_disjoint", ["manual_disjoint_mm"], True)))
    metrics, predictions = [], []
    for cohort, oof in variants:
        metrics.append(summarize(oof, cohort, args.bootstrap))
        oof["cohort"] = cohort
        predictions.append(oof)
        m = metrics[-1]
        print(f"{cohort:18s} {m['variant']:22s} n={m['n']} abstain={m['n_abstain']} "
              f"F1={m['f1']:.3f} spec={m['specificity']:.3f} AUC={m['roc_auc']:.3f}")
    pred = pd.concat(predictions, ignore_index=True)
    pred.to_csv(args.out / "oof.csv", index=False)
    pred[(pred.pred >= 0) & (pred.pred != pred.y_rot)].to_csv(args.out / "errors.csv", index=False)
    pd.DataFrame(metrics).to_csv(args.out / "metrics.csv", index=False)
    subgroups = []
    manual_oof = pred[pred.variant == "manual_contour"].merge(
        features[["rel_path", "lt_state"]], on="rel_path", validate="one_to_one")
    for name, group in manual_oof.groupby("lt_state"):
        if (group.pred >= 0).any():
            subgroups.append(summarize(group, f"lt_{name}", args.bootstrap))
    for name, group in manual_oof.groupby("region"):
        subgroups.append(summarize(group, name, args.bootstrap))
    pd.DataFrame(subgroups).to_csv(args.out / "subgroups.csv", index=False)
    # Perturb only validation anchor location; reuse original training bounds.
    manual = pred[(pred.variant == "manual_contour")].merge(
        features[["rel_path", "manual_minus5_mm", "manual_plus5_mm"]],
        on="rel_path", validate="one_to_one")
    valid = manual.pred >= 0
    for delta in ["minus5", "plus5"]:
        s = manual[f"manual_{delta}_mm"]
        manual[f"pred_{delta}"] = np.where(s.notna(),
            (s < manual.threshold_low) | (s > manual.threshold_high), -1).astype(int)
    manual.to_csv(args.out / "anchor_sensitivity.csv", index=False)
    finite_error = features.auto_anchor_error_mm.dropna()
    summary = {
        "fixed_split_sha256": hashlib.sha256((ROOT / "reports/split.csv").read_bytes()).hexdigest(),
        "annotation_sha256": hashlib.sha256((ROOT / "annotations/femur_points_v1.csv").read_bytes()).hexdigest(),
        "baseline_features_sha256": hashlib.sha256((ROOT / "reports/femur_rotation.csv").read_bytes()).hexdigest(),
        "n_images": len(features), "n_studies": int(features.study.nunique()),
        "n_manual_anchors": len(assisted), "annotation_status": "draft_not_expert_validated",
        "auto_anchor_comparisons": len(finite_error),
        "auto_anchor_median_error_mm": float(finite_error.median()),
        "auto_anchor_p90_error_mm": float(finite_error.quantile(.9)),
        "manual_predictions_changed_minus5px": int((manual.pred[valid] != manual.pred_minus5[valid]).sum()),
        "manual_predictions_changed_plus5px": int((manual.pred[valid] != manual.pred_plus5[valid]).sum()),
        "manual_windows_overlapping_shaft_fit": int((features.manual_window_overlaps_fit == True).sum()),
        "threshold_fit": "outer training studies only; max F1 subject to training specificity >= 0.75",
        "deployed": False,
        "limitation": "Exploratory reused folds; selected80 is enriched; manual anchors unavailable at inference.",
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    render_review(features, args.root, args.out / "contour_review.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
