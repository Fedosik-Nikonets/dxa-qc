#!/usr/bin/env python3
"""Compare medial tracking hypotheses, including nested study-fold selection.

Landmarks are used only to report localization error. They are never passed
to the image-only tracker or to rotation classifiers. All variants explored
in this refinement are retained, including unsuccessful ones.
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
from dxa_qc.medial_track import track_medial_edge
from dxa_qc.metrics import binary_metrics, f1_ci
from dxa_qc.rotation import fit_normal_interval, interval_predict


VARIANTS = {"global_bump": ["global_bump_mm"], "global_absolute": ["global_absolute_mm"]}
for mode in ("restricted", "shaft_bounded"):
    for feature in ("bump", "absolute", "last_bump"):
        VARIANTS[f"{mode}_{feature}"] = [f"{mode}_{feature}_mm"]
    VARIANTS[f"{mode}_angle_bump"] = ["shaft_angle_deg", f"{mode}_bump_mm"]
    VARIANTS[f"{mode}_angle_absolute"] = ["shaft_angle_deg", f"{mode}_absolute_mm"]
    VARIANTS[f"{mode}_angle_both"] = ["shaft_angle_deg", f"{mode}_bump_mm", f"{mode}_absolute_mm"]


def extract(root: Path):
    data = pd.read_csv(ROOT / "reports/dataset_images.csv")
    data = data[data.labeled & data.region.isin(["femur_left", "femur_right"])].copy()
    data = data.merge(pd.read_csv(ROOT / "reports/split.csv")[["study", "fold"]],
                      on="study", validate="many_to_one")
    manifest = pd.read_csv(ROOT / "annotations/femur_manifest_v1.csv")
    annotations = manifest.merge(pd.read_csv(ROOT / "annotations/femur_points_v1.csv"),
                                  on="image_id", validate="one_to_one").set_index("rel_path")
    rows, tracks = [], {}
    for row in data.sort_values("rel_path").itertuples(index=False):
        rec = {"study": row.study, "rel_path": row.rel_path, "region": row.region,
               "fold": int(row.fold), "y": int(row.y_FEMUR_POSITIONING_ROTATION),
               "image_id": "", "manual_y": np.nan, "manual_state": "unannotated", "reason": ""}
        if row.rel_path in annotations.index:
            ann = annotations.loc[row.rel_path]
            if ann.study != row.study or int(ann.fold) != int(row.fold):
                raise ValueError("annotation does not match split")
            rec.update(image_id=ann.image_id, manual_y=ann.lt_y, manual_state=ann.lt_state)
        img = read_dicom(root / row.rel_path)
        if not img.ok:
            raise ValueError(f"DICOM read failed: {row.rel_path}")
        try:
            track = track_medial_edge(img.pixels, row.region, img.mm_per_px)
            tracks[row.rel_path] = track
            rec.update(shaft_angle_deg=abs(float(np.degrees(np.arctan(track.slope)))),
                       fit_rms_mm=track.fit_rms_mm, fit_start=track.fit_start,
                       mm_per_px=track.mm, shaft_width_px=track.width_px)
            for mode in ("global", "restricted", "shaft_bounded"):
                try:
                    candidate = track.candidate(mode)
                    rec.update({f"{mode}_{k}": v for k, v in candidate.items()})
                    rec[f"{mode}_error_mm"] = abs(candidate["anchor_y"]-rec["manual_y"])*track.mm
                except ValueError as exc:
                    rec[f"{mode}_reason"] = str(exc)
        except ValueError as exc:
            rec["reason"] = str(exc)
        rows.append(rec)
    features = pd.DataFrame(rows)
    for cols in VARIANTS.values():
        for column in cols:
            if column not in features:
                features[column] = np.nan
    return features, tracks


def fit_predict(train: pd.DataFrame, val: pd.DataFrame, variant: str,
                variants: dict | None = None) -> pd.DataFrame:
    if set(train.study) & set(val.study):
        raise ValueError("study leakage between training and validation")
    cols = (VARIANTS if variants is None else variants)[variant]
    tx, vx = train[cols].to_numpy(float), val[cols].to_numpy(float)
    good_train, good_val = np.isfinite(tx).all(axis=1), np.isfinite(vx).all(axis=1)
    tx, y = tx[good_train], train.y.to_numpy(int)[good_train]
    out = val[["study", "rel_path", "image_id", "region", "fold", "y"]].copy()
    out["variant"], out["pred"], out["risk"] = variant, -1, np.nan
    if len(np.unique(y)) != 2:
        raise ValueError("calibration requires both classes")
    if len(cols) == 1:
        train_score, val_score = tx[:, 0], vx[good_val, 0]
    else:
        mu, sd = standardize(tx)
        w, b = fit_logistic((tx-mu)/sd, y, 1.)
        train_score = 1/(1+np.exp(-np.clip((tx-mu)/sd @ w+b, -60, 60)))
        val_score = 1/(1+np.exp(-np.clip((vx[good_val]-mu)/sd @ w+b, -60, 60)))
    low, high = fit_normal_interval(train_score, y)
    pred, risk = interval_predict(val_score, low, high)
    out.loc[good_val, "pred"] = pred
    out.loc[good_val, "risk"] = risk/max(float(np.std(train_score)), 1e-6)
    out["low"], out["high"] = low, high
    out["train_n"] = len(y)
    out["train_studies_sha256"] = hashlib.sha256(
        "\n".join(sorted(train.loc[good_train, "study"].unique())).encode()).hexdigest()
    return out


def cross_validate(frame: pd.DataFrame, variant: str, variants: dict | None = None) -> pd.DataFrame:
    rows = []
    for fold in sorted(frame.fold.unique()):
        rows.append(fit_predict(frame[frame.fold != fold], frame[frame.fold == fold], variant, variants))
    return pd.concat(rows).sort_index()


def metric(oof: pd.DataFrame) -> dict:
    # Full-cohort detection performance: abstentions do not detect positives.
    # Coverage is always reported alongside this operational fallback.
    m = binary_metrics(oof.y, oof.pred.clip(lower=0)).as_dict()
    m.update(n_abstain=int((oof.pred < 0).sum()),
             abstain_positives=int(oof.loc[oof.pred < 0, "y"].sum()))
    # A mixed selected-model risk is not a calibrated shared probability.
    m.pop("roc_auc")
    m.pop("pr_auc")
    return m


def nested_select(frame: pd.DataFrame, variants: dict | None = None):
    variants = VARIANTS if variants is None else variants
    outer_predictions, choices, inner_metrics = [], [], []
    for fold in sorted(frame.fold.unique()):
        train, val = frame[frame.fold != fold], frame[frame.fold == fold]
        candidates = []
        for variant in variants:
            score = metric(cross_validate(train, variant, variants))
            score.update(outer_fold=int(fold), variant=variant)
            inner_metrics.append(score)
            if score["specificity"] >= .75:
                candidates.append(score)
        if candidates:
            best = sorted(candidates, key=lambda m: (-m["f1"], m["fp"], m["variant"]))[0]
            out = fit_predict(train, val, best["variant"], variants)
            choices.append({"fold": int(fold), "selected": best["variant"],
                            "inner_f1": best["f1"], "inner_specificity": best["specificity"]})
        else:
            out = val[["study", "rel_path", "image_id", "region", "fold", "y"]].copy()
            out["variant"], out["pred"], out["risk"] = "no_qualifying_variant", -1, np.nan
            choices.append({"fold": int(fold), "selected": "no_qualifying_variant"})
        outer_predictions.append(out)
        print(f"nested fold {fold}: {choices[-1]['selected']}", flush=True)
    return pd.concat(outer_predictions).sort_index(), choices, pd.DataFrame(inner_metrics)


def review(features, tracks, root, output):
    ids = ["F004", "F015", "F024", "F043", "F063", "F070", "F075", "F080"]
    page = Image.new("RGB", (4*360, 2*440), (24, 24, 24))
    for i, key in enumerate(ids):
        row = features[features.image_id == key].iloc[0]
        img = read_dicom(root / row.rel_path)
        track = tracks[row.rel_path]
        a = img.pixels if row.region == "femur_left" else img.pixels[:, ::-1]
        tile = Image.new("RGB", (360, 440), (24, 24, 24))
        tile.paste(Image.fromarray(np.uint8(np.clip(a, 0, 1)*255)).convert("RGB"), (35, 30))
        draw = ImageDraw.Draw(tile)
        draw.text((8, 7), f"{key} class={row.y} LT={row.manual_state}", fill="white")
        for y, x in zip(track.ys, track.xs):
            draw.point((35+x, 30+y), fill="cyan")
        # Only draw the fitted line on its own distal interval.
        draw.line((35+track.slope*track.fit_start+track.intercept, 30+track.fit_start,
                   35+track.slope*track.ys[-1]+track.intercept, 30+track.ys[-1]), fill="yellow", width=2)
        if np.isfinite(row.manual_y):
            draw.line((35, 30+row.manual_y, 315, 30+row.manual_y), fill="green")
        if np.isfinite(row.restricted_anchor_y):
            draw.line((35, 30+row.restricted_anchor_y, 315, 30+row.restricted_anchor_y), fill="magenta")
        draw.text((8, 385), f"candidate y={row.restricted_anchor_y:.0f} error={row.restricted_error_mm:.1f}mm", fill="white")
        draw.text((8, 402), "Cyan=edge Yellow=fit Green=draft Pink=auto", fill="white")
        draw.text((8, 418), "Canonical reflection; research only", fill="white")
        page.paste(tile, ((i%4)*360, (i//4)*440))
    page.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "Data/НД_для_обучения/Исследования")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/rotation_refinement")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    features, tracks = extract(args.root)
    features.to_csv(args.out / "features.csv", index=False)
    metrics, predictions = [], []
    for variant in VARIANTS:
        oof = cross_validate(features, variant)
        predictions.append(oof)
        metrics.append({"variant": variant, **metric(oof)})
        print(f"{variant}: F1={metrics[-1]['f1']:.3f}, specificity={metrics[-1]['specificity']:.3f}", flush=True)
    pd.DataFrame(metrics).to_csv(args.out / "candidate_metrics.csv", index=False)
    pd.concat(predictions).to_csv(args.out / "candidate_oof.csv", index=False)
    nested, choices, inner = nested_select(features)
    nested.to_csv(args.out / "nested_oof.csv", index=False)
    inner.to_csv(args.out / "inner_metrics.csv", index=False)
    nested[(nested.pred < 0) | (nested.pred != nested.y)].to_csv(args.out / "errors.csv", index=False)
    m = metric(nested)
    m["f1_ci95"] = f1_ci(nested.pred.clip(lower=0), nested.y, groups=nested.study,
                            n_bootstrap=2000, seed=0)
    localization = []
    previous = pd.read_csv(ROOT / "reports/rotation_day6/features.csv")
    paired = features.merge(previous[["rel_path", "auto_anchor_error_mm"]],
                            on="rel_path", validate="one_to_one")
    for mode in ("global", "restricted", "shaft_bounded"):
        for group, subset in [("all_annotated", paired),
                              ("same_previous_images", paired[paired.auto_anchor_error_mm.notna()]),
                              ("visible_only", paired[paired.manual_state == "visible"]),
                              ("uncertain_only", paired[paired.manual_state == "uncertain"])]:
            error = subset[f"{mode}_error_mm"].dropna()
            localization.append({"mode": mode, "group": group, "n": len(error),
                                 "median_mm": float(error.median()), "p90_mm": float(error.quantile(.9))})
    pd.DataFrame(localization).to_csv(args.out / "localization.csv", index=False)
    summary = {"images": len(features), "tracked_images": len(tracks),
               "restricted_candidates": int(features.restricted_anchor_y.notna().sum()),
               "variants_explored": len(VARIANTS), "nested_metrics": m,
               "selections": choices, "deployed": False,
               "split_sha256": hashlib.sha256((ROOT / "reports/split.csv").read_bytes()).hexdigest(),
               "points_sha256": hashlib.sha256((ROOT / "annotations/femur_points_v1.csv").read_bytes()).hexdigest(),
               "limitations": "Exploratory reused dataset; nested CV isolates label-based variant choice, not prior human feature development. Landmarks remain draft. No new held-out dataset."}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    review(features, tracks, args.root, args.out / "tracking_review.png")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
