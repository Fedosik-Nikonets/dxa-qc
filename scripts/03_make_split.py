#!/usr/bin/env python3
"""Фиксированное разбиение выборки. Единица разбиения — ИССЛЕДОВАНИЕ.

Почему исследование, а не изображение: у одного пациента позвоночник и оба
бедра лежат в одном исследовании и сильно скоррелированы (укладка делается
за один сеанс). Разбиение по изображениям даёт утечку и завышенные метрики —
прямой минус по критерию 8.1.

Почему 5-fold, а не один сплит: размеченных исследований всего 100, при
80/20 в валидации окажется ~20 исследований и единицы позитивов на редкие
классы (FEMUR_ROI_COVERAGE: 7 позитивов на весь набор). Доверительный
интервал на такой валидации будет шире полезного диапазона. Пятифолдовая
кросс-валидация даёт оценку по всем 100 исследованиям; fold 0 дополнительно
помечен как `holdout` для быстрых прогонов в течение спринта.

ВАЖНО: файл разбиения генерируется один раз и коммитится. Повторный запуск
с тем же seed обязан давать тот же результат — это проверяет tests/.

Запуск:
    python scripts/03_make_split.py --dataset reports/dataset_images.csv \
        --out reports/split.csv --seed 42
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

N_FOLDS = 5


def stratum_key(g: pd.DataFrame) -> str:
    """Ключ стратификации на уровне исследования.

    Сочетает состав (есть ли бедро) и профиль нарушений — так в каждом фолде
    сохраняется и доля исследований «только позвоночник», и доля позитивов.
    """
    has_femur = int(any(r.startswith("femur") and r != "femur_?" for r in g["region"]))
    spine_bad = int((g.loc[g["region"] == "spine", "quality_class"] == 1).any())
    femur_bad = int((g.loc[g["region"].str.startswith("femur"), "quality_class"] == 1).any())
    return f"f{has_femur}_s{spine_bad}_h{femur_bad}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ds = pd.read_csv(args.dataset)
    strata = ds.groupby("study").apply(stratum_key, include_groups=False).rename("stratum")
    studies = strata.reset_index()

    rng = np.random.RandomState(args.seed)
    studies["fold"] = -1
    for stratum, g in studies.groupby("stratum"):
        idx = np.array(g.index.to_numpy(), copy=True)
        rng.shuffle(idx)
        studies.loc[idx, "fold"] = np.arange(len(idx)) % N_FOLDS

    studies["split_holdout"] = np.where(studies["fold"] == 0, "val", "train")
    studies = studies.sort_values("study").reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    studies.to_csv(args.out, index=False)

    digest = hashlib.md5(
        studies[["study", "fold"]].to_csv(index=False).encode()
    ).hexdigest()

    print(f"исследований: {len(studies)}  seed={args.seed}  md5(split)={digest}")
    print("\nразмер фолдов:")
    print(studies["fold"].value_counts().sort_index().to_string())
    print("\nстраты по фолдам:")
    print(pd.crosstab(studies["stratum"], studies["fold"]).to_string())

    merged = ds.merge(studies[["study", "fold"]], on="study")
    print("\nпозитивы (quality_class=1) по фолдам и областям:")
    pos = merged[merged["quality_class"] == 1]
    print(pd.crosstab(pos["region"], pos["fold"]).to_string())
    print(f"\nзаписано: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
