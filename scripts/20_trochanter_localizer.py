#!/usr/bin/env python3
"""OOF-локализация малого вертела по 58 точкам врача и контактный лист."""
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
from dxa_qc.dicom_io import read_dicom
from dxa_qc.medial_track import track_medial_edge
from dxa_qc.rotation_profile import canonical_image
from dxa_qc.trochanter import TrainingPoint, fit, rotation_features


def load(root: Path):
    data = pd.read_csv(ROOT / "reports/dataset_images.csv")
    data = data[data.labeled & data.region.isin(["femur_left", "femur_right"])].copy()
    data = data.merge(pd.read_csv(ROOT / "reports/split.csv")[["study", "fold"]],
                      on="study", validate="many_to_one").sort_values("rel_path").reset_index(drop=True)
    ann = pd.read_csv(ROOT / "annotations/femur_landmarks_v1.csv")
    ann = ann[(ann.landmark == "lesser_trochanter") & ann.visibility.isin(["visible", "uncertain"])]
    if len(ann) != 58 or not (ann.annotation_source.eq("human") & ann.human_reviewed.eq(True)
                              & ann.eligible_for_training.eq(True)).all():
        raise ValueError("ожидались 58 пригодных точек врача")
    ann = ann.set_index("rel_path", verify_integrity=True)
    images, tracks = {}, {}
    for row in data.itertuples(index=False):
        img = read_dicom(root / row.rel_path)
        if not img.ok:
            raise ValueError(f"не читается {row.rel_path}: {img.error}")
        track = track_medial_edge(img.pixels, row.region, img.mm_per_px,
                                  mm_per_px_source=img.mm_per_px_source, model=img.model)
        images[row.rel_path], tracks[row.rel_path] = img, track
    return data, ann, images, tracks


def train_points(ann: pd.DataFrame, images: dict, folds: set[int]):
    points = []
    for path, row in ann[ann.fold.isin(folds)].iterrows():
        img = images[path]
        points.append(TrainingPoint(img.pixels, row.region, row.x, row.y, img.mm_per_px,
                                    img.mm_per_px_source, img.model))
    return points


def oof_features(data, ann, images, tracks):
    rows, templates = [], {}
    all_folds = set(int(v) for v in data.fold.unique())
    for fold in sorted(all_folds):
        model = fit(train_points(ann, images, all_folds - {fold}))
        templates[f"fold_{fold}"] = model.template
        for row in data[data.fold == fold].itertuples(index=False):
            rec = {"study": row.study, "rel_path": row.rel_path, "region": row.region,
                   "fold": int(row.fold), "y": int(row.y_FEMUR_POSITIONING_ROTATION),
                   "image_id": "", "human_state": "unannotated", "human_x": np.nan,
                   "human_y": np.nan, "reason": ""}
            img, track = images[row.rel_path], tracks[row.rel_path]
            if row.rel_path in ann.index:
                a = ann.loc[row.rel_path]
                x = a.x if row.region == "femur_left" else img.cols - 1 - a.x
                rec.update(image_id=a.image_id, human_state=a.visibility,
                           human_x=float(x), human_y=float(a.y))
            try:
                features = rotation_features(img.pixels, row.region, img.mm_per_px, track, model,
                                             mm_per_px_source=img.mm_per_px_source,
                                             scanner_model=img.model)
                rec.update(features)
                rec["error_xy_mm"] = float(np.hypot(features["canonical_anchor_x"]-rec["human_x"],
                                                     features["anchor_y"]-rec["human_y"])*track.mm)
            except ValueError as exc:
                rec["reason"] = str(exc)
            rows.append(rec)
    return pd.DataFrame(rows).sort_values("rel_path").reset_index(drop=True), templates


def localization_table(features: pd.DataFrame):
    old = pd.read_csv(ROOT / "reports/rotation_human_profile/features.csv")
    values = [("v1_anchor", old.error_xy_mm.dropna()),
              ("v2_localizer", features.error_xy_mm.dropna())]
    return pd.DataFrame([{"variant": name, "n": len(v), "median_mm": float(v.median()),
                          "p90_mm": float(v.quantile(.9)), "within_5mm": float((v <= 5).mean())}
                         for name, v in values])


def by_state(features: pd.DataFrame):
    rows = []
    for state in ("visible", "uncertain"):
        v = features.loc[features.human_state == state, "error_xy_mm"].dropna()
        rows.append({"variant": "v2_localizer", "state": state, "n": len(v),
                     "median_mm": float(v.median()), "p90_mm": float(v.quantile(.9)),
                     "within_5mm": float((v <= 5).mean())})
    return pd.DataFrame(rows)


def contact_sheet(features, images, tracks, output: Path):
    worst = features.dropna(subset=["error_xy_mm"]).nlargest(10, "error_xy_mm")
    unannotated = features[features.human_state == "unannotated"].sort_values("rel_path")
    random = unannotated.iloc[np.random.default_rng(0).choice(len(unannotated), 10, replace=False)]
    chosen = pd.concat([worst, random]).reset_index(drop=True)
    page = Image.new("RGB", (5*340, 4*440), (24, 24, 24))
    for i, row in chosen.iterrows():
        img, track = images[row.rel_path], tracks[row.rel_path]
        a = canonical_image(img.pixels, row.region)
        tile = Image.new("RGB", (340, 440), (24, 24, 24))
        view = Image.fromarray(np.uint8(a*255)).convert("RGB")
        tile.paste(view, ((340-view.width)//2, 30))
        ox = (340-view.width)//2
        draw = ImageDraw.Draw(tile)
        label = f"{row.image_id or 'unannotated'} fold={row.fold} y={row.y}"
        draw.text((7, 7), label, fill="white")
        for y, x in zip(track.ys, track.xs):
            draw.point((ox+float(x), 30+float(y)), fill="cyan")
        if np.isfinite(row.human_y):
            x, y = ox+row.human_x, 30+row.human_y
            draw.ellipse((x-5, y-5, x+5, y+5), outline="lime", width=2)
        if np.isfinite(row.anchor_y):
            x, y = ox+row.canonical_anchor_x, 30+row.anchor_y
            draw.ellipse((x-5, y-5, x+5, y+5), outline="magenta", width=2)
        for y in (row.window_start, row.window_end):
            if np.isfinite(y):
                draw.line((ox, 30+y, ox+view.width, 30+y), fill="yellow")
        err = f"error={row.error_xy_mm:.1f} mm" if np.isfinite(row.error_xy_mm) else "no doctor point"
        draw.text((7, 397), err, fill="white")
        draw.text((7, 414), "green=doctor magenta=v2 cyan=edge", fill="white")
        page.paste(tile, ((i % 5)*340, (i // 5)*440))
    page.save(output)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT / "Data/НД_для_обучения/Исследования")
    ap.add_argument("--out", type=Path, default=ROOT / "reports/rotation_localizer")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data, ann, images, tracks = load(args.root)
    features, templates = oof_features(data, ann, images, tracks)
    features.to_csv(args.out / "oof_features.csv", index=False)
    localization_table(features).to_csv(args.out / "localization.csv", index=False)
    by_state(features).to_csv(args.out / "localization_by_state.csv", index=False)
    np.savez_compressed(args.out / "oof_templates.npz", **templates)
    contact_sheet(features, images, tracks, args.out / "localization.png")
    summary = {"images": len(features), "points_by_doctor": len(ann),
               "localization": localization_table(features).to_dict("records"),
               "input_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                                for p in ("reports/split.csv", "reports/dataset_images.csv",
                                          "annotations/femur_landmarks_v1.csv")}}
    (args.out / "localizer_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
