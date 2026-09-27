#!/usr/bin/env python3
"""Select 80 femur images and export auditable visual landmark annotations.

Coordinates refer to the original decoded DICOM: x right, y down, origin (0,0).
No reflection, crop, or resize is applied to annotations. Unseen landmarks stay
null. The visual draft is not an expert reference and is excluded from training
unless explicitly approved by a human reviewer.
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
from dxa_qc.dicom_io import read_dicom

POINTS = ("greater_trochanter", "lesser_trochanter", "femoral_neck", "ischium")
SHORT = ("gt", "lt", "neck", "isch")
COLORS = ((255, 110, 60), (0, 220, 255), (255, 225, 0), (255, 90, 220))


def select_images(data: pd.DataFrame, split: pd.DataFrame, count: int = 80) -> pd.DataFrame:
    if split.study.duplicated().any():
        raise ValueError("duplicate study in fixed split")
    eligible = data[data.region.isin(["femur_left", "femur_right"]) & data.labeled].copy()
    eligible = eligible.merge(split[["study", "fold"]], on="study", how="left", validate="many_to_one")
    if eligible.fold.isna().any() or eligible.rel_path.duplicated().any():
        raise ValueError("missing fold or duplicate image")
    eligible = eligible.sort_values("rel_path").reset_index(drop=True)
    mandatory = (eligible.y_FEMUR_POSITIONING_ROTATION == 1) | (eligible.y_FEMUR_ROI_COVERAGE == 1)
    chosen = list(eligible.index[mandatory])
    if not len(chosen) <= count <= len(eligible):
        raise ValueError("sample size cannot retain all violation-positive images")
    # Fill the remaining places evenly across the existing fold/side buckets.
    ranked = sorted(eligible.index[~mandatory], key=lambda i: hashlib.sha256(
        ("femur-v1:" + eligible.at[i, "rel_path"]).encode()).hexdigest())
    while len(chosen) < count:
        counts = eligible.loc[chosen].groupby(["fold", "region"]).size().to_dict()
        best = min(ranked, key=lambda i: counts.get((eligible.at[i, "fold"], eligible.at[i, "region"]), 0))
        chosen.append(best)
        ranked.remove(best)
    out = eligible.loc[chosen].sort_values("rel_path").reset_index(drop=True)
    out.insert(0, "image_id", [f"F{i+1:03d}" for i in range(len(out))])
    out["selection_reason"] = np.where(
        (out.y_FEMUR_POSITIONING_ROTATION == 1) | (out.y_FEMUR_ROI_COVERAGE == 1),
        "all_violation_positives", "fold_side_balanced_control",
    )
    return out[["image_id", "study", "rel_path", "md5", "region", "rows", "cols", "fold",
                "y_FEMUR_POSITIONING_ROTATION", "y_FEMUR_ROI_COVERAGE", "selection_reason"]]


def export_points(manifest: pd.DataFrame, points: pd.DataFrame,
                  allow_partial: bool = False, source: str = "ai_visual_draft") -> pd.DataFrame:
    if points.image_id.duplicated().any():
        raise ValueError("annotations must cover each selected image exactly once")
    extra = set(points.image_id) - set(manifest.image_id)
    if extra:
        raise ValueError(f"annotations refer to images outside the manifest: {sorted(extra)[:3]}")
    missing = set(manifest.image_id) - set(points.image_id)
    if missing:
        # Проверочная выборка размечается первой (16 кадров из 80), и
        # требовать полного покрытия на этом шаге значит не дать её обсчитать.
        if not allow_partial:
            raise ValueError("annotations must cover each selected image exactly once")
        manifest = manifest[manifest.image_id.isin(points.image_id)]
    rows = []
    for row in manifest.merge(points, on="image_id", validate="one_to_one").itertuples(index=False):
        for name, prefix in zip(POINTS, SHORT):
            x, y, state = (getattr(row, prefix + suffix) for suffix in ("_x", "_y", "_state"))
            if state not in {"visible", "uncertain", "not_visible"}:
                raise ValueError(f"invalid visibility for {row.image_id}/{name}")
            if state == "not_visible":
                if pd.notna(x) or pd.notna(y):
                    raise ValueError("not_visible coordinates must be empty")
            elif not (np.isfinite(x) and np.isfinite(y) and 0 <= x < row.cols and 0 <= y < row.rows):
                raise ValueError(f"point outside image: {row.image_id}/{name}")
            rows.append({"image_id": row.image_id, "study": row.study, "rel_path": row.rel_path,
                         "pixel_md5": row.md5, "region": row.region, "fold": row.fold,
                         "landmark": name, "x": x, "y": y, "visibility": state,
                         "annotation_source": source,
                         "human_reviewed": source == "human",
                         "eligible_for_training": source == "human"})
    return pd.DataFrame(rows)


def render(manifest, dicom_root, output, annotations=None):
    # Подпись под каждым кадром обязана называть источник точек. Раньше
    # там стояло «AI DRAFT» жёстко, и контактные листы с человеческой
    # разметкой выглядели бы черновиком автоматики — ровно наоборот тому,
    # что говорят флаги в таблице.
    if annotations is None or not len(annotations):
        stamp = "NO POINTS"
    else:
        sources = sorted(set(annotations.annotation_source))
        stamp = "/".join(s.upper().replace("_", " ") for s in sources)
    output.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(manifest), 8):
        page = Image.new("RGB", (4 * 360, 2 * 420), (24, 24, 24))
        for offset, row in enumerate(manifest.iloc[start:start+8].itertuples(index=False)):
            decoded = read_dicom(dicom_root / row.rel_path)
            if not decoded.ok or decoded.pixels is None:
                raise ValueError(f"cannot decode {row.image_id}: {decoded.error}")
            a = decoded.pixels
            tile = Image.new("RGB", (360, 420), (24, 24, 24))
            tile.paste(Image.fromarray(np.uint8(np.clip(a, 0, 1) * 255)).convert("RGB"), (40, 35))
            draw = ImageDraw.Draw(tile)
            draw.text((8, 7), f"{row.image_id} {row.region} {a.shape[1]}x{a.shape[0]}", fill="white")
            for x in range(0, a.shape[1], 20):
                draw.line((40+x, 35, 40+x, 35+a.shape[0]), fill=(65, 65, 30), width=1)
                if x % 40 == 0:
                    draw.text((36+x, 23), str(x), fill=(255, 220, 100))
            for y in range(0, a.shape[0], 20):
                draw.line((40, 35+y, 40+a.shape[1], 35+y), fill=(65, 65, 30), width=1)
                if y % 40 == 0:
                    draw.text((8, 31+y), str(y), fill=(255, 220, 100))
            if annotations is not None:
                subset = annotations[annotations.image_id == row.image_id].set_index("landmark")
                # При частичной разметке неразмеченные кадры просто рисуются
                # без точек, а не роняют сборку контактных листов.
                for number, (name, color) in enumerate(zip(POINTS, COLORS), 1) if len(subset) else []:
                    point = subset.loc[name]
                    if point.visibility == "not_visible":
                        continue
                    x, y = 40 + point.x, 35 + point.y
                    draw.ellipse((x-4, y-4, x+4, y+4), outline=color, width=2)
                    draw.text((x+5, y-7), str(number), fill=color)
            draw.text((8, 394), f"1 GT  2 LT  3 NECK  4 ISCH | {stamp}", fill=(220, 220, 220))
            page.paste(tile, ((offset % 4) * 360, (offset // 4) * 420))
        page.save(output / f"page_{start//8+1:02d}.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT / "Data/НД_для_обучения/Исследования")
    ap.add_argument("--points", type=Path, default=ROOT / "annotations/femur_points_v1.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "annotations")
    ap.add_argument("--allow-partial", action="store_true",
                    help="принять разметку части кадров (проверочная выборка)")
    ap.add_argument("--source", choices=("ai_visual_draft", "human"),
                    default="ai_visual_draft",
                    help="кто ставил точки; human помечает разметку пригодной для обучения")
    args = ap.parse_args()
    manifest = select_images(pd.read_csv(ROOT / "reports/dataset_images.csv"), pd.read_csv(ROOT / "reports/split.csv"))
    args.out.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.out / "femur_manifest_v1.csv", index=False)
    annotations = None
    if args.points.exists():
        annotations = export_points(manifest, pd.read_csv(args.points),
                                    args.allow_partial, args.source)
        annotations.to_csv(args.out / "femur_landmarks_v1.csv", index=False)
    render(manifest, args.root, ROOT / "reports/femur_landmarks_review", annotations)
    annotated = 0 if annotations is None else int(annotations.image_id.nunique())
    summary = {"images": len(manifest), "annotated_images": annotated,
               "studies": int(manifest.study.nunique()),
               "regions": manifest.region.value_counts().to_dict(),
               "rotation_positives": int(manifest.y_FEMUR_POSITIONING_ROTATION.sum()),
               "roi_positives": int(manifest.y_FEMUR_ROI_COVERAGE.sum()),
               "fixed_split_sha256": hashlib.sha256((ROOT / "reports/split.csv").read_bytes()).hexdigest(),
               "annotation_source": args.source,
               "human_reviewed_images": annotated if args.source == "human" else 0}
    if annotations is not None:
        summary["point_visibility"] = annotations.visibility.value_counts().to_dict()
    (args.out / "femur_summary_v1.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
