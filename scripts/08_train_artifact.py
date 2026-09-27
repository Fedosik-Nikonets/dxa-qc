#!/usr/bin/env python3
"""Обучение и валидация детектора артефактов позвоночника (ТЗ 2.3.1, рис. 3).

Задача напарника на 17 сентября. В плане стоял efficientnet-b0; здесь
обучается логистическая регрессия на одном физическом признаке, а CNN
заменён прямым сравнением вариантов на тех же фолдах — см. раздел
«сравнение вариантов модели» в выводе и `reports/artifact_cv.json`.

Что считается и почему именно так
---------------------------------
*Два протокола валидации.* Зафиксированный сплит (`reports/split.csv`)
даёт воспроизводимое число, но в одном из его пяти фолдов **ноль
позитивов**: сплит стратифицировался не по артефактам. Поэтому основная
цифра — многократная стратифицированная 5-фолдовая перекрёстная проверка
(20 повторов, позитивы раскладываются по фолдам равномерно), а прогон по
зафиксированному сплиту печатается рядом как воспроизводимая ссылка.

*Порог выбирается внутри обучающих фолдов*, а не по OOF-предсказаниям:
иначе метрика заглядывает в валидацию и завышается. Итоговый порог модели —
медиана порогов по фолдам.

Запуск:
    python scripts/08_train_artifact.py --dataset reports/dataset_images.csv \
        --root "Data/НД_для_обучения/Исследования" --split reports/split.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dxa_qc.artifact import (  # noqa: E402
    FEATURE_VERSION,
    ArtifactModel,
    features as artifact_features,
    fit_logistic,
    standardize,
)
from dxa_qc.dicom_io import read_dicom  # noqa: E402
from dxa_qc.metrics import auc_ci, best_threshold, binary_metrics, f1_ci, roc_auc  # noqa: E402
from dxa_qc.segmentation import segment_bone  # noqa: E402

L2 = 10.0
N_REPEATS = 20


def collect(dataset: Path, root: Path) -> pd.DataFrame:
    ds = pd.read_csv(dataset)
    spine = ds[(ds["region"] == "spine") & ds["labeled"]].sort_values("rel_path")
    rows = []
    for i, x in enumerate(spine.itertuples(), 1):
        img = read_dicom(str(root / x.rel_path))
        bone = segment_bone(img.pixels) if img.ok and img.pixels is not None else None
        f = artifact_features(img.pixels if img.ok else None,
                              bone.mask if bone is not None else None, img.mm_per_px)
        rows.append({"study": x.study, "rel_path": x.rel_path,
                     "y": int(x.y_SPINE_ARTIFACT), "tophat": f.tophat_p999,
                     "ridge": f.ridge_p9999, "ok": int(f.ok),
                     "hog": hog_vector(img.pixels) if img.ok else None})
        if i % 25 == 0:
            print(f"  признаки {i}/{len(spine)}", file=sys.stderr)
    return pd.DataFrame(rows)


def hog_vector(image: np.ndarray, cells: int = 4, bins: int = 9, size: int = 64) -> np.ndarray:
    """HOG для сравнения вариантов: 4x4 клетки, 9 ориентаций, 144 признака."""
    a = np.asarray(image, dtype=np.float32)
    a = (a - a.min()) / max(float(np.ptp(a)), 1.0) * 255.0
    im = cv2.resize(a, (size, size))
    gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = np.hypot(gx, gy), np.degrees(np.arctan2(gy, gx)) % 180
    step, out = size // cells, []
    for i in range(cells):
        for j in range(cells):
            m = mag[i * step:(i + 1) * step, j * step:(j + 1) * step].ravel()
            t = ang[i * step:(i + 1) * step, j * step:(j + 1) * step].ravel()
            h, _ = np.histogram(t, bins=bins, range=(0, 180), weights=m)
            out.append(h / (np.linalg.norm(h) + 1e-6))
    return np.concatenate(out)


def cv_once(x: np.ndarray, y: np.ndarray, folds: np.ndarray,
            l2: float = L2) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Один прогон перекрёстной проверки. Порог берётся из обучающих фолдов."""
    oof = np.full(len(y), np.nan)
    pred = np.zeros(len(y), dtype=int)
    thresholds: list[float] = []
    for k in np.unique(folds):
        tr, va = folds != k, folds == k
        if not tr.any() or not va.any():
            continue
        mu, sd = standardize(x[tr])
        w, b = fit_logistic((x[tr] - mu) / sd, y[tr], l2)
        p_tr = 1.0 / (1.0 + np.exp(-np.clip(((x[tr] - mu) / sd) @ w + b, -60, 60)))
        t = best_threshold(p_tr, y[tr])
        thresholds.append(t)
        p_va = 1.0 / (1.0 + np.exp(-np.clip(((x[va] - mu) / sd) @ w + b, -60, 60)))
        oof[va] = p_va
        pred[va] = (p_va >= t).astype(int)
    return oof, pred, thresholds


def repeated_cv(x: np.ndarray, y: np.ndarray, repeats: int = N_REPEATS,
                l2: float = L2) -> dict:
    """Многократная стратифицированная проверка: позитивы поровну по фолдам."""
    aucs, f1s, thr = [], [], []
    for seed in range(repeats):
        rng = np.random.default_rng(seed)
        folds = np.empty(len(y), dtype=int)
        for cls in (0, 1):
            idx = np.flatnonzero(y == cls)
            rng.shuffle(idx)
            folds[idx] = np.arange(len(idx)) % 5
        oof, pred, t = cv_once(x, y, folds, l2)
        aucs.append(roc_auc(oof, y))
        f1s.append(binary_metrics(y, pred).f1)
        thr += t
    return {"roc_auc_mean": float(np.mean(aucs)), "roc_auc_std": float(np.std(aucs)),
            "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
            "threshold_median": float(np.median(thr)), "repeats": repeats}


def main() -> int:
    ap = argparse.ArgumentParser()
    root_dir = Path(__file__).resolve().parents[1]
    ap.add_argument("--dataset", type=Path, default=root_dir / "reports/dataset_images.csv")
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--split", type=Path, default=root_dir / "reports/split.csv")
    ap.add_argument("--features-out", type=Path, default=root_dir / "reports/artifact_features.csv")
    ap.add_argument("--oof-out", type=Path, default=root_dir / "reports/artifact_oof.csv")
    ap.add_argument("--report-out", type=Path, default=root_dir / "reports/artifact_cv.json")
    ap.add_argument("--model-out", type=Path, default=root_dir / "models/artifact_tophat_v1.npz")
    ap.add_argument("--bootstrap", type=int, default=10000)
    args = ap.parse_args()

    df = collect(args.dataset, args.root)
    hog = np.stack(df.pop("hog").to_numpy())
    df.to_csv(args.features_out, index=False)
    y = df.y.to_numpy()
    studies = df.study.to_numpy()
    print(f"\nизображений {len(df)}, позитивов {int(y.sum())}, "
          f"признаки вычислены у {int(df.ok.sum())}")

    split = pd.read_csv(args.split)[["study", "fold"]]
    folds = df[["study"]].merge(split, on="study", how="left").fold.to_numpy()
    per_fold = pd.DataFrame({"fold": folds, "y": y}).groupby("fold").y.agg(["size", "sum"])
    print("\nзафиксированный сплит (исследований / позитивов по фолдам):")
    print(per_fold.to_string())
    empty = per_fold[per_fold["sum"] == 0]
    if len(empty):
        print(f"ВНИМАНИЕ: в фолдах {list(empty.index.astype(int))} ноль позитивов — "
              f"сплит стратифицирован не по артефактам, метрики по фолдам неустойчивы")

    variants = {
        "top-hat (1 признак)": df[["tophat"]].to_numpy(float),
        "гребень (1 признак)": df[["ridge"]].to_numpy(float),
        "top-hat + гребень": df[["tophat", "ridge"]].to_numpy(float),
        "HOG 144 признака": hog,
    }
    print("\nСРАВНЕНИЕ ВАРИАНТОВ МОДЕЛИ (многократная проверка, "
          f"{N_REPEATS} повторов 5 фолдов)")
    comparison = {}
    for name, x in variants.items():
        r = repeated_cv(x, y)
        comparison[name] = r
        print(f"  {name:22s} ROC-AUC {r['roc_auc_mean']:.3f}±{r['roc_auc_std']:.3f}  "
              f"F1 {r['f1_mean']:.2f}±{r['f1_std']:.2f}")

    x = variants["top-hat (1 признак)"]
    main_cv = comparison["top-hat (1 признак)"]

    print("\nЗАФИКСИРОВАННЫЙ СПЛИТ (воспроизводимая ссылка)")
    oof, pred, thr_fixed = cv_once(x, y, folds)
    m_fixed = binary_metrics(y, pred, oof)
    print("  " + m_fixed.line("out-of-fold"))
    lo, hi = auc_ci(oof, y, groups=studies, n_bootstrap=args.bootstrap)
    f_lo, f_hi = f1_ci(pred, y, groups=studies, n_bootstrap=args.bootstrap)
    print(f"  ROC-AUC ДИ95% [{lo:.3f}, {hi:.3f}]   F1 ДИ95% [{f_lo:.2f}, {f_hi:.2f}]")

    # Итоговая модель: обучение на всех данных, порог — медиана по фолдам.
    mu, sd = standardize(x)
    w, b = fit_logistic((x - mu) / sd, y, L2)
    model = ArtifactModel(mean=mu, scale=sd, coef=w, intercept=b,
                          threshold=main_cv["threshold_median"])
    model.save(args.model_out)
    reloaded = ArtifactModel.load(args.model_out)
    diff = float(np.max(np.abs(reloaded.predict_proba(x) - model.predict_proba(x))))
    if diff > 1e-12:
        raise AssertionError(f"веса не переносятся: расхождение {diff}")
    print(f"\nмодель: {args.model_out} (порог {model.threshold:.3f}, "
          f"перенос весов совпадает до {diff:.1e})")

    out = df[["study", "rel_path", "y", "tophat", "ridge"]].copy()
    out["fold"] = folds
    out["prob_oof"] = oof
    out["pred_oof"] = pred
    out.to_csv(args.oof_out, index=False)

    report = {
        "задача": "детектор артефактов позвоночника, ТЗ 2.3.1 рис. 3",
        "модель": "логистическая регрессия на одном признаке top-hat, L2=10, веса классов сбалансированы",
        "версия_признаков": FEATURE_VERSION,
        "изображений": int(len(df)), "позитивов": int(y.sum()),
        "основная_проверка": main_cv,
        "сравнение_вариантов": comparison,
        "зафиксированный_сплит": {
            **m_fixed.as_dict(),
            "roc_auc_ci95": [lo, hi], "f1_ci95": [f_lo, f_hi],
            "пороги_по_фолдам": thr_fixed,
            "позитивов_по_фолдам": per_fold["sum"].astype(int).to_dict(),
        },
        "порог_модели": model.threshold,
    }
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
    print(f"отчёт: {args.report_out}\nout-of-fold: {args.oof_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
