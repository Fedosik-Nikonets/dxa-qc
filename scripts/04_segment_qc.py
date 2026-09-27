#!/usr/bin/env python3
"""Прогон сегментации по всему набору и контрольные листы для проверки глазами.

Маска кости — ядро геометрического слоя (plan_v2.md, решение Р1), поэтому она
проверяется не выборочно, а на всех уникальных изображениях. Скрипт делает три
вещи:

1. считает маску для каждого изображения и сохраняет диагностику в CSV;
2. отмечает подозрительные случаи по формальным правилам (пустая маска,
   слишком большая или слишком маленькая доля кости, рыхлый контур,
   касание всех четырёх краёв кадра) — их надо смотреть в первую очередь;
3. рисует контрольные листы: контур маски поверх снимка, подозрительные —
   на первом листе.

Запуск:
    python scripts/04_segment_qc.py --dataset reports/dataset_images.csv \
        --root data/train/Исследования \
        --out-csv reports/segmentation_qc.csv --out-dir reports/qc_sheets
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dxa_qc.dicom_io import read_dicom  # noqa: E402
from dxa_qc.segmentation import overlay, principal_axis, segment_bone  # noqa: E402

# Границы «нормальности» получены из распределения по обучающему набору:
# всё, что вне, идёт на ручной просмотр, а не отбраковывается автоматически.
MIN_BONE_FRACTION = 0.03   # доля кадра
MAX_BONE_FRACTION = 0.55
MIN_FILL_RATIO = 0.20      # площадь маски к площади bbox


def suspicion(row: dict) -> str:
    """Причины, по которым маску стоит посмотреть глазами. Пусто — вопросов нет."""
    flags: list[str] = []
    if not row["seg_ok"]:
        flags.append("сегментация не удалась")
        return ";".join(flags)
    if row["bone_frame_fraction"] < MIN_BONE_FRACTION:
        flags.append("маска слишком мала")
    if row["bone_frame_fraction"] > MAX_BONE_FRACTION:
        flags.append("маска слишком велика")
    if row["fill_ratio"] < MIN_FILL_RATIO:
        flags.append("рыхлый контур")
    if all(row[k] for k in ("touch_top", "touch_bottom", "touch_left", "touch_right")):
        flags.append("касание всех четырёх краёв")
    if row["n_kept"] >= 4:
        flags.append("много компонент")
    return ";".join(flags)


def sheets(df: pd.DataFrame, out_dir: Path, cols: int = 5, rows: int = 4) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("sheet_*.png"):
        f.unlink()

    per = cols * rows
    n_sheets = (len(df) + per - 1) // per
    for s in range(n_sheets):
        chunk = df.iloc[s * per : (s + 1) * per]
        fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.4 * rows))
        for ax, (_, r) in zip(np.ravel(axes), chunk.iterrows()):
            img = read_dicom(r["path"])
            if img.ok and img.pixels is not None:
                m = segment_bone(img.pixels)
                ax.imshow(overlay(img.pixels, m.mask))
            flag = r["suspicion"] or ""
            ax.set_title(
                f"{r['region']} · {r['study'][-6:]}\n"
                f"доля={r['bone_frame_fraction']:.2f} fill={r['fill_ratio']:.2f} "
                f"угол={r['axis_angle_deg']:.1f}\n{flag}",
                fontsize=6.5,
                color="crimson" if flag else "black",
            )
            ax.axis("off")
        for ax in np.ravel(axes)[len(chunk) :]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"sheet_{s:02d}.png", dpi=85)
        plt.close(fig)
    return n_sheets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--root", required=True, type=Path,
                    help="каталог Исследования: rel_path из датасета строится от него")
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    ds = pd.read_csv(args.dataset)
    ds["path"] = ds["rel_path"].map(lambda r: str(args.root / r))
    recs: list[dict] = []
    for _, r in ds.iterrows():
        img = read_dicom(r["path"])
        rec = {
            "study": r["study"],
            "rel_path": r["rel_path"],
            "path": r["path"],          # только для отрисовки, в CSV не пишется
            "region": r["region"],
            "quality_class": r["quality_class"],
            "violations": r["violations"],
        }
        if not img.ok or img.pixels is None:
            rec.update({"seg_ok": False, "seg_reason": f"чтение: {img.error}"})
            recs.append(rec)
            continue
        m = segment_bone(img.pixels)
        st = m.stats()
        h, w = img.pixels.shape
        angle, (cy, cx) = principal_axis(m.mask)
        rec.update(st)
        rec.update({
            "rows": h,
            "cols": w,
            "bone_frame_fraction": round(m.area_px / (h * w), 4),
            "axis_angle_deg": round(float(angle), 2) if np.isfinite(angle) else np.nan,
            "centroid_row": round(float(cy), 1) if np.isfinite(cy) else np.nan,
            "centroid_col": round(float(cx), 1) if np.isfinite(cx) else np.nan,
            "mm_per_px": r["mm_per_px"],
            # отступы bbox до краёв кадра в миллиметрах — заготовка правила
            # охвата ROI (3 см сверху/снизу, 2 см сбоку, ТЗ 2.3.2)
            "margin_top_mm": round(st["bbox_r0"] * r["mm_per_px"], 1),
            "margin_bottom_mm": round((h - st["bbox_r1"]) * r["mm_per_px"], 1),
            "margin_left_mm": round(st["bbox_c0"] * r["mm_per_px"], 1),
            "margin_right_mm": round((w - st["bbox_c1"]) * r["mm_per_px"], 1),
        })
        recs.append(rec)

    qc = pd.DataFrame(recs)
    for c in ("bone_frame_fraction", "fill_ratio", "n_kept"):
        if c not in qc:
            qc[c] = np.nan
    qc["suspicion"] = qc.apply(lambda r: suspicion(r.to_dict()), axis=1)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    # Абсолютные пути машины в коммит не попадают: ключ строки — rel_path.
    qc.drop(columns=["path"]).to_csv(args.out_csv, index=False)

    n_bad = int((qc["suspicion"] != "").sum())
    print(f"изображений: {len(qc)}  сегментация не удалась: {int((~qc['seg_ok']).sum())}")
    print(f"подозрительных: {n_bad} ({n_bad / len(qc):.1%})")
    if n_bad:
        print("\nпричины:")
        reasons = qc.loc[qc["suspicion"] != "", "suspicion"].str.split(";").explode()
        print(reasons.value_counts().to_string())

    print("\nдоля кости в кадре по областям:")
    print(qc.groupby("region")["bone_frame_fraction"].describe()
          [["count", "min", "25%", "50%", "75%", "max"]].round(3).to_string())
    print("\nугол главной оси по областям (градусы):")
    print(qc.groupby("region")["axis_angle_deg"].describe()
          [["count", "min", "25%", "50%", "75%", "max"]].round(1).to_string())

    spine = qc[qc["region"] == "spine"]
    if len(spine) and "SPINE_AXIS_TILT" in str(spine["violations"].astype(str).tolist()):
        tilt = spine["violations"].astype(str).str.contains("SPINE_AXIS_TILT")
        print("\nугол оси позвоночника: разметка против геометрии (предварительно)")
        print(spine.groupby(tilt)["axis_angle_deg"].apply(
            lambda s: f"n={len(s)} |угол| медиана={s.abs().median():.1f} max={s.abs().max():.1f}"
        ).to_string())

    if args.out_dir:
        order = qc.sort_values(
            ["suspicion", "region", "study"], ascending=[False, True, True]
        )
        n = sheets(order, args.out_dir)
        print(f"\nконтрольные листы: {n} шт. в {args.out_dir}")
    print(f"записано: {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
