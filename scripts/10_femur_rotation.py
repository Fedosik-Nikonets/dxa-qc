#!/usr/bin/env python3
"""Ротация бедра (ТЗ 2.3.2): проверка гипотез и обоснование отказа.

План отводил это правило напарнику и опирался на ручную разметку четырёх
точек. Разметки к 19 сентября нет, поэтому проверено, можно ли обойтись без
неё. Ответ — нельзя, и здесь считаются цифры, на которых этот ответ стоит.

Сравниваются на **одних и тех же фолдах** зафиксированного сплита, с порогом
из обучающих фолдов и метрикой out-of-fold:

1. угол оси диафиза к вертикали — один признак с физическим смыслом;
2. набор геометрии выпрямленного бедра — 10 признаков;
3. пиксели выпрямленного и отзеркаленного бедра 16x24 — 384 признака.
   Это та же проверка, что была сделана для артефактов 18 сентября: если
   модель на сырых пикселях не бьёт один признак, обучать CNN тем более
   не на чем;
4. угол + заполненность bbox маской — лучшая найденная пара.

Почему даже лучший вариант не ставится в конвейер
--------------------------------------------------
Специфичность лучшей пары 0.69 на 114 негативах — это 35 ложных
срабатываний против 23 пойманных. 17 сентября по такому же соотношению был
принят отказ ставить флаг находок вне перечня 2.3 в `quality_class`
(14 ложных против 1 совпадения). Решение здесь то же и по той же причине.

Отдельно про заполненность: доля позитивов растёт по её квартилям
монотонно (0.11 / 0.14 / 0.27 / 0.45), но это, скорее всего, не ротация, а
плотность кости — у разреженной кости мягкий порог даёт рыхлую маску.
Строить правило качества укладки на признаке, который меряет остеопороз
пациента, нельзя.

Что нужно, чтобы правило появилось
----------------------------------
Разметка четырёх точек на ~80 изображениях: верхушка большого вертела,
малый вертел, середина шейки, седалищная кость. Прямое измерение выступа
малого вертела провалилось именно из-за привязки: точка «расширение
диафиза» уезжает на шейку, и окно измерения вместо вертела захватывает
шейку (медиана выступа 40 мм там, где у вертела 10–15 мм). Ручная точка
снимает ровно эту проблему.

Запуск:
    python scripts/10_femur_rotation.py --root "Data/НД_для_обучения/Исследования"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dxa_qc.artifact import fit_logistic, standardize  # noqa: E402
from dxa_qc.dicom_io import read_dicom  # noqa: E402
from dxa_qc.femur import femur_mask, shaft_angle  # noqa: E402
from dxa_qc.metrics import best_threshold, binary_metrics, roc_auc  # noqa: E402
from dxa_qc.segmentation import _to_u8, segment_bone  # noqa: E402

PATCH = (16, 24)   # ширина x высота уменьшенного кропа бедра


def rectified(image: np.ndarray, mask: np.ndarray, region: str) -> tuple[np.ndarray, np.ndarray]:
    """Кроп бедра, повёрнутый до вертикального диафиза; правое зеркалится к левому."""
    ang = shaft_angle(mask)
    if ang != ang:
        ang = 0.0
    rows = np.flatnonzero(mask.any(axis=1))
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    lo = int(r1 - (r1 - r0) * 0.35)
    rr = [r for r in range(lo, r1) if mask[r].any()]
    cc = [float(np.flatnonzero(mask[r]).mean()) for r in rr]
    k = float(np.polyfit(np.array(rr, float), np.array(cc, float), 1)[0])
    deg = float(np.degrees(np.arctan(k)))
    h, w = mask.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), -deg, 1.0)
    rm = cv2.warpAffine(mask.astype(np.uint8), m, (w, h), flags=cv2.INTER_NEAREST).astype(bool)
    ri = cv2.warpAffine(_to_u8(image), m, (w, h), flags=cv2.INTER_LINEAR)
    if region == "femur_right":
        rm, ri = rm[:, ::-1], ri[:, ::-1]
    rws, cls = np.flatnonzero(rm.any(axis=1)), np.flatnonzero(rm.any(axis=0))
    sl = (slice(rws.min(), rws.max() + 1), slice(cls.min(), cls.max() + 1))
    return ri[sl], rm[sl]


def geometry(crop_mask: np.ndarray, mm: float, deg: float) -> dict:
    """Геометрия выпрямленного бедра: диафиз, вертельная область, выступ."""
    n = crop_mask.shape[0]
    left = np.full(n, np.nan)
    wid = np.full(n, np.nan)
    for i in range(n):
        idx = np.flatnonzero(crop_mask[i])
        if idx.size:
            left[i] = idx.min()
            wid[i] = idx.max() - idx.min()
    nb = max(8, int(round(25 / mm)))
    w0 = float(np.nanmedian(wid[n - nb:]))
    i = n - 1
    while i > 0 and (not np.isfinite(wid[i]) or wid[i] <= 1.3 * w0):
        i -= 1
    g = {"shaft_deg": abs(deg), "shaft_w": w0 * mm, "base_frac": i / n,
         "h_mm": n * mm, "wmax_mm": float(np.nanmax(wid)) * mm,
         "fill": float(crop_mask.mean()),
         "aspect": float(crop_mask.shape[1] / crop_mask.shape[0])}
    nfit = max(8, int(round(30 / mm)))
    fi = np.arange(min(i + 2, n - 1), min(i + 2 + nfit, n))
    fi = fi[np.isfinite(left[fi])]
    for win_mm in (12, 20, 30):
        g[f"lt{win_mm}"] = float("nan")
    if fi.size >= 8:
        kk, c0 = np.polyfit(fi.astype(float), left[fi], 1)
        for win_mm in (12, 20, 30):
            nw = max(4, int(round(win_mm / mm)))
            win = np.arange(max(i - nw, 0), min(i + 2, n))
            win = win[np.isfinite(left[win])]
            if win.size:
                g[f"lt{win_mm}"] = float(
                    np.max(np.polyval((kk, c0), win.astype(float)) - left[win])) * mm
    return g


def cross_validate(x: np.ndarray, y: np.ndarray, folds: np.ndarray, l2: float):
    """Порог берётся из обучающих фолдов, предсказание — out-of-fold."""
    oof = np.full(len(y), np.nan)
    pred = np.zeros(len(y), dtype=int)
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        if not tr.any() or not va.any() or y[tr].sum() == 0:
            continue
        mu, sd = standardize(x[tr])
        w, b = fit_logistic((x[tr] - mu) / sd, y[tr], l2)
        p_tr = 1 / (1 + np.exp(-np.clip(((x[tr] - mu) / sd) @ w + b, -60, 60)))
        t = best_threshold(p_tr, y[tr])
        p_va = 1 / (1 + np.exp(-np.clip(((x[va] - mu) / sd) @ w + b, -60, 60)))
        oof[va], pred[va] = p_va, (p_va >= t).astype(int)
    return oof, pred


def main() -> int:
    root_dir = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=root_dir / "reports/dataset_images.csv")
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--split", type=Path, default=root_dir / "reports/split.csv")
    ap.add_argument("--out", type=Path, default=root_dir / "reports/femur_rotation.csv")
    args = ap.parse_args()

    ds = pd.read_csv(args.dataset)
    fem = ds[ds["region"].isin(["femur_left", "femur_right"]) & ds["labeled"]]
    split = pd.read_csv(args.split)[["study", "fold"]].drop_duplicates()
    fem = fem.merge(split, on="study", how="left")

    geo, patches, y, folds, keep = [], [], [], [], []
    for i, x in enumerate(fem.itertuples(), 1):
        img = read_dicom(str(args.root / x.rel_path))
        if not img.ok or img.pixels is None:
            continue
        bone = segment_bone(img.pixels)
        fm = femur_mask(img.pixels, bone)
        if fm is None or not fm.any():
            continue
        crop, crop_mask = rectified(img.pixels, fm, x.region)
        if min(crop_mask.shape) < 10:
            continue
        mm = float(img.mm_per_px or 0.6)
        geo.append(geometry(crop_mask, mm, shaft_angle(fm)))
        p = cv2.resize(crop.astype(np.float32), PATCH, interpolation=cv2.INTER_AREA)
        patches.append(((p - p.mean()) / (p.std() + 1e-6)).ravel())
        y.append(int(x.y_FEMUR_POSITIONING_ROTATION))
        folds.append(int(x.fold))
        keep.append({"study": x.study, "rel_path": x.rel_path, "region": x.region,
                     "y_rot": int(x.y_FEMUR_POSITIONING_ROTATION), "fold": int(x.fold)})
        if i % 25 == 0:
            print(f"  {i}/{len(fem)}", file=sys.stderr)

    g = pd.DataFrame(geo)
    y = np.array(y)
    folds = np.array(folds)
    patches = np.array(patches)
    print(f"объектов: {len(y)}, позитивов ротации: {int(y.sum())}\n")

    print("ОДИНОЧНЫЕ ПРИЗНАКИ, ROC-AUC на всей выборке (не out-of-fold)")
    for c in g.columns:
        s = np.nan_to_num(g[c].to_numpy(float), nan=float(np.nanmedian(g[c])))
        print(f"  {c:12s} {roc_auc(s, y):.3f}")

    x_all = np.nan_to_num(g.to_numpy(float), nan=0.0)
    variants = [
        ("угол оси диафиза", x_all[:, [list(g.columns).index("shaft_deg")]], 1.0),
        ("угол + заполненность", x_all[:, [list(g.columns).index("shaft_deg"),
                                           list(g.columns).index("fill")]], 1.0),
        (f"геометрия ({x_all.shape[1]})", x_all, 10.0),
        (f"пиксели {PATCH[0]}x{PATCH[1]} ({patches.shape[1]})", patches, 50.0),
    ]
    print("\nOUT-OF-FOLD ПО ЗАФИКСИРОВАННОМУ СПЛИТУ (фолды по исследованиям)")
    rows = []
    for name, x, l2 in variants:
        oof, pred = cross_validate(x, y, folds, l2)
        m = binary_metrics(y, pred, oof)
        print(f"  {m.line(name)}")
        rows.append({"вариант": name, **m.as_dict()})

    print("\nЗАПОЛНЕННОСТЬ ПРОТИВ ДОЛИ ПОЗИТИВОВ ПО КВАРТИЛЯМ (проверка на подмену)")
    q = pd.qcut(g["fill"], 4, labels=False)
    tab = pd.DataFrame({"q": q, "y": y}).groupby("q").y.agg(["mean", "sum", "count"])
    for i, r in tab.iterrows():
        print(f"  квартиль {i}: доля позитивов {r['mean']:.2f} ({int(r['sum'])} из {int(r['count'])})")
    print("  монотонный рост -> признак меряет рыхлость маски, а не укладку")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out = pd.concat([pd.DataFrame(keep), g], axis=1)
    out.to_csv(args.out, index=False)
    pd.DataFrame(rows).to_csv(args.out.with_name("femur_rotation_cv.csv"), index=False)
    print(f"\nзаписано: {args.out} и {args.out.with_name('femur_rotation_cv.csv')}")
    print("ВЫВОД: правило ротации в конвейер не ставится, нужна разметка четырёх точек")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
