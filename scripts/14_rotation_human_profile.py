#!/usr/bin/env python3
"""Day 6: human landmark audit and image-only cortex intensity experiment.

Four fixed hypotheses: contour bump, intensity at the automatic anchor,
median intensity along the search window, inside/outside cortex contrast.
Each receives a two-tail interval fitted only on training studies. Variant
selection uses inner folds; the outer fold is never used for selection.
Human points are for localization/intensity diagnostics only, never input
features for the 150-image automatic experiment.
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
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dxa_qc.dicom_io import read_dicom
from dxa_qc.medial_track import track_medial_edge
from dxa_qc.metrics import auc_ci, f1_ci, roc_auc
from dxa_qc.rotation_profile import canonical_image, patch_intensity, profile_features

spec = importlib.util.spec_from_file_location("refinement", ROOT / "scripts/12_refine_rotation.py")
cv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cv)
VARIANTS = {name: [name] for name in
            ("bump_mm", "anchor_intensity", "profile_intensity", "profile_contrast")}


def load_annotations():
    lm = pd.read_csv(ROOT / "annotations/femur_landmarks_v1.csv")
    if not (lm.annotation_source.eq("human") & lm.human_reviewed.eq(True)
            & lm.eligible_for_training.eq(True)).all():
        raise ValueError("requires the reviewed human annotation export")
    if lm.duplicated(["image_id", "landmark"]).any():
        raise ValueError("duplicate landmark")
    return lm[lm.landmark == "lesser_trochanter"].set_index("rel_path", verify_integrity=True)


def extract(root: Path):
    data = pd.read_csv(ROOT / "reports/dataset_images.csv")
    data = data[data.labeled & data.region.isin(["femur_left", "femur_right"])].copy()
    data = data.merge(pd.read_csv(ROOT / "reports/split.csv")[["study", "fold"]],
                      on="study", validate="many_to_one")
    annotations = load_annotations()
    rows, tracks = [], {}
    for row in data.sort_values("rel_path").itertuples(index=False):
        rec = {"study": row.study, "rel_path": row.rel_path, "region": row.region,
               "fold": int(row.fold), "y": int(row.y_FEMUR_POSITIONING_ROTATION),
               "image_id": "", "human_state": "unannotated", "reason": "",
               "human_x": np.nan, "human_y": np.nan, "human_intensity": np.nan}
        img = read_dicom(root / row.rel_path)
        if not img.ok:
            raise ValueError(f"cannot read {row.rel_path}: {img.error}")
        a = canonical_image(img.pixels, row.region)
        if row.rel_path in annotations.index:
            ann = annotations.loc[row.rel_path]
            if ann.study != row.study or int(ann.fold) != row.fold or ann.region != row.region:
                raise ValueError("annotation and manifest/split disagree")
            x = ann.x if row.region == "femur_left" else a.shape[1]-1-ann.x
            rec.update(image_id=ann.image_id, human_x=x, human_y=ann.y,
                       human_state=ann.visibility,
                       human_intensity=patch_intensity(a, x, ann.y))
        try:
            track = track_medial_edge(img.pixels, row.region, img.mm_per_px,
                                      mm_per_px_source=img.mm_per_px_source, model=img.model)
            tracks[row.rel_path] = track
            rec.update(mm_per_px=track.mm, mm_per_px_source=img.mm_per_px_source,
                       fit_start=track.fit_start, fit_rms_mm=track.fit_rms_mm)
            rec.update(profile_features(img.pixels, row.region, track))
            rec["error_y_mm"] = abs(rec["anchor_y"]-rec["human_y"])*track.mm
            rec["error_xy_mm"] = np.hypot(rec["anchor_x"]-rec["human_x"],
                                          rec["anchor_y"]-rec["human_y"])*track.mm
        except ValueError as exc:
            rec["reason"] = str(exc)
        rows.append(rec)
    return pd.DataFrame(rows), tracks


def localization(frame):
    rows = []
    for state in ("all_locatable", "visible", "uncertain"):
        subset = frame if state == "all_locatable" else frame[frame.human_state == state]
        for feature in ("error_y_mm", "error_xy_mm"):
            v = subset[feature].dropna()
            rows.append({"group": state, "metric": feature, "n": len(v),
                         "median_mm": float(v.median()), "p90_mm": float(v.quantile(.9)),
                         "within_5mm": float((v <= 5).mean())})
    return pd.DataFrame(rows)


def intensity_diagnostic(frame):
    # Direction specified by the handoff: uncertain is brighter than visible.
    subset = frame[frame.human_state.isin(["visible", "uncertain"])]
    rows = []
    for feature in ("human_intensity", "anchor_intensity", "profile_intensity", "profile_contrast"):
        d = subset.dropna(subset=[feature])
        y = d.human_state.eq("uncertain").astype(int)
        lo, hi = auc_ci(d[feature], y, groups=d.study, n_bootstrap=2000, seed=0)
        rows.append({"feature": feature, "n": len(d), "auc_uncertain_brighter": roc_auc(d[feature], y),
                     "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def review(frame, tracks, root, output):
    # Same fixed cases as the previous tracker review, chosen before new scores.
    keys = ["F004", "F015", "F024", "F043", "F063", "F070", "F075", "F080"]
    page = Image.new("RGB", (4*360, 2*450), (24, 24, 24))
    for i, key in enumerate(keys):
        row = frame[frame.image_id == key].iloc[0]
        a = canonical_image(read_dicom(root / row.rel_path).pixels, row.region)
        track = tracks[row.rel_path]
        tile = Image.new("RGB", (360, 450), (24, 24, 24))
        tile.paste(Image.fromarray(np.uint8(a*255)).convert("RGB"), (35, 30))
        draw = ImageDraw.Draw(tile)
        draw.text((8, 7), f"{key} rotation={row.y} LT={row.human_state}", fill="white")
        for y, x in zip(track.ys, track.xs):
            draw.point((35+x, 30+y), fill="cyan")
        if np.isfinite(row.human_y):
            x, y = 35+row.human_x, 30+row.human_y
            draw.ellipse((x-4, y-4, x+4, y+4), outline="lime", width=2)
        if np.isfinite(row.anchor_y):
            x, y = 35+row.anchor_x, 30+row.anchor_y
            draw.ellipse((x-4, y-4, x+4, y+4), outline="magenta", width=2)
            for yy in (row.window_start, row.window_end):
                draw.line((35, 30+yy, 315, 30+yy), fill="yellow")
        draw.text((8, 395), f"XY error={row.error_xy_mm:.1f}mm", fill="white")
        draw.text((8, 412), "Green=human Pink=auto Cyan=edge", fill="white")
        draw.text((8, 429), "Yellow=window; canonical reflection", fill="white")
        page.paste(tile, ((i%4)*360, (i//4)*450))
    page.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "Data/НД_для_обучения/Исследования")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/rotation_human_profile")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    features, tracks = extract(args.root)
    features.to_csv(args.out / "features.csv", index=False)
    localization(features).to_csv(args.out / "localization.csv", index=False)
    intensity_diagnostic(features).to_csv(args.out / "intensity_diagnostic.csv", index=False)
    review(features, tracks, args.root, args.out / "human_tracking_review.png")
    metrics, predictions = [], []
    for variant in VARIANTS:
        oof = cv.cross_validate(features, variant, VARIANTS)
        predictions.append(oof)
        metrics.append({"variant": variant, **cv.metric(oof)})
        print(f"{variant}: {metrics[-1]}", flush=True)
    pd.DataFrame(metrics).to_csv(args.out / "candidate_metrics.csv", index=False)
    pd.concat(predictions).to_csv(args.out / "candidate_oof.csv", index=False)
    nested, choices, inner = cv.nested_select(features, VARIANTS)
    nested.to_csv(args.out / "nested_oof.csv", index=False)
    inner.to_csv(args.out / "inner_metrics.csv", index=False)
    nested[(nested.pred < 0) | (nested.pred != nested.y)].to_csv(args.out / "errors.csv", index=False)
    m = cv.metric(nested)
    m["f1_ci95"] = f1_ci(nested.pred.clip(lower=0), nested.y, groups=nested.study,
                           n_bootstrap=2000, seed=0)
    summary = {"images": len(features), "studies": int(features.study.nunique()),
               "positive_images": int(features.y.sum()), "tracked_images": len(tracks),
               "measured_images": int(features.profile_intensity.notna().sum()),
               "human_images": int(features.image_id.ne("").sum()),
               "variants": VARIANTS, "nested_metrics": m, "selections": choices,
               "deployed": False, "input_sha256": {
                   str(path): hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                   for path in ("reports/split.csv", "reports/dataset_images.csv",
                                "annotations/femur_landmarks_v1.csv",
                                "annotations/femur_points_human.csv")},
               "limitations": "Exploratory reused dataset, not an untouched test. Human coordinates only for diagnostics. Human visibility not an inference input. CI bootstraps fixed OOF predictions by study and excludes model-refitting uncertainty."}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
