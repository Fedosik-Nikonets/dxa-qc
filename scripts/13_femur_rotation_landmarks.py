#!/usr/bin/env python3
"""Проверяет гипотезу ротации по ручной разметке четырёх точек.

Вопрос, ради которого размечались 80 кадров (`reports/day5_rotation.md`):
разделяет ли выступ малого вертела кадры с нарушением ротации и без него.
День 5 отвечал на него геометрией по маске и получил AUC 0.65 — почти
монету. Здесь тот же вопрос задан разметке, поставленной человеком.

На этой подвыборке связь есть, но она не монотонная:
нарушение соответствует обоим краям — и чётко видимому
вертелу, и не видимому вовсе, — а норме соответствует середина, «еле
виден». ROC-AUC исходной порядковой шкалы может скрывать такую связь.
Два порога или нелинейно преобразованный score способны её выявить.

Скрипту нужны только две таблицы, изображения не читаются: всё, что он
считает, выводится из координат и состояний видимости.

Запуск:
    PYTHONPATH=src python scripts/13_femur_rotation_landmarks.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dxa_qc.metrics import binary_metrics, f1_ci, roc_auc  # noqa: E402

SEEN = {"not_visible": 0, "uncertain": 1, "visible": 2}
NAMES = {0: "не видно", 1: "не уверен", 2: "видно"}

# Негативов в выборке 44 из 114 размеченных: позитивы ротации вошли все,
# негативы — подвыборка. Перенос специфичности ниже — только сценарная
# оценка при предположении репрезентативности, не измеренные FP на 150.
NEGATIVES_TOTAL = 114


def widen(landmarks: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Одна строка на кадр: координаты, состояния, метка, фолд."""
    w = landmarks.pivot(index="image_id", columns="landmark",
                        values=["x", "y", "visibility"])
    d = pd.DataFrame(index=w.index)
    for full, short in (("greater_trochanter", "gt"), ("lesser_trochanter", "lt"),
                        ("femoral_neck", "neck"), ("ischium", "isch")):
        d[short + "_x"] = w[("x", full)]
        d[short + "_y"] = w[("y", full)]
        d[short + "_v"] = w[("visibility", full)]
    d = d.join(manifest.set_index("image_id")[
        ["study", "region", "fold", "y_FEMUR_POSITIONING_ROTATION"]])
    if d[["study", "fold"]].isna().any().any():
        raise ValueError("разметка ссылается на кадры вне манифеста")
    d["y"] = d.y_FEMUR_POSITIONING_ROTATION.astype(int)
    d["seen"] = d.lt_v.map(SEEN)
    return d


def geometry(d: pd.DataFrame) -> pd.DataFrame:
    """Положение малого вертела относительно оси «большой вертел — шейка».

    Ось берётся от большого вертела к шейке: она сама несёт сторону тела,
    поэтому левые и правые кадры не нужно зеркалить руками. `along` — на
    сколько вертел медиальнее шейки, `perp` — на сколько ниже оси; обе
    величины в долях длины оси, то есть без масштаба в миллиметрах.
    """
    gt = d[["gt_x", "gt_y"]].to_numpy(float)
    nk = d[["neck_x", "neck_y"]].to_numpy(float)
    lt = d[["lt_x", "lt_y"]].to_numpy(float)
    axis = nk - gt
    scale = np.linalg.norm(axis, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        u = axis / scale[:, None]
        perp = np.stack([-u[:, 1], u[:, 0]], axis=1)
        v = lt - nk
        d = d.copy()
        d["lt_along"] = (v * u).sum(1) / scale
        d["lt_perp"] = np.abs((v * perp).sum(1)) / scale
        d["lt_dist"] = np.linalg.norm(v, axis=1) / scale
    return d


def rule_table(d: pd.DataFrame) -> pd.DataFrame:
    """Три кандидата в правило — по одному на каждую форму связи."""
    y = d.y.to_numpy(int)
    out = []
    for name, pred in (
        ("видим ИЛИ не видим вовсе", (d.seen != 1).astype(int).to_numpy()),
        ("чётко видим", (d.seen == 2).astype(int).to_numpy()),
        ("не видим вовсе", (d.seen == 0).astype(int).to_numpy()),
    ):
        m = binary_metrics(y, pred)
        lo, hi = f1_ci(pred, y, groups=d.study)
        fp_full = round(NEGATIVES_TOTAL * (1 - m.specificity))
        prec = m.tp / (m.tp + fp_full) if m.tp + fp_full else float("nan")
        f1 = (2 * prec * m.sensitivity / (prec + m.sensitivity)
              if prec and m.sensitivity else float("nan"))
        out.append({"правило": name, "TP": m.tp, "FP": m.fp, "FN": m.fn, "TN": m.tn,
                    "чувств.": round(m.sensitivity, 2),
                    "специф.": round(m.specificity, 3),
                    "F1": round(m.f1, 2), "F1 ДИ": f"[{lo:.2f}, {hi:.2f}]",
                    "оценка FP на 114 негативах": fp_full,
                    "оценка точности": round(prec, 2), "оценка F1": round(f1, 2)})
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--landmarks", type=Path,
                    default=ROOT / "annotations/femur_landmarks_v1.csv")
    ap.add_argument("--manifest", type=Path,
                    default=ROOT / "annotations/femur_manifest_v1.csv")
    args = ap.parse_args()

    lm = pd.read_csv(args.landmarks)
    if not lm.human_reviewed.all():
        raise SystemExit("это не человеческая разметка: пересоберите с --source human")
    d = geometry(widen(lm, pd.read_csv(args.manifest)))
    print(f"кадров {len(d)}, позитивов ротации {int(d.y.sum())}, "
          f"негативов {int((1 - d.y).sum())} из {NEGATIVES_TOTAL} размеченных\n")

    print("СОСТОЯНИЕ МАЛОГО ВЕРТЕЛА ПРОТИВ МЕТКИ")
    t = pd.crosstab(d.seen.map(NAMES), d.y)
    t.columns = ["норма", "нарушение"]
    t["доля нарушений"] = (t["нарушение"] / t.sum(axis=1)).round(3)
    print(t.loc[[n for n in ("видно", "не уверен", "не видно") if n in t.index]].to_string())
    print("\nСвязь не монотонная: нарушению отвечают оба края, норме — середина.")

    print("\nМОНОТОННЫЕ ПРИЗНАКИ (ровно то, что мерил день 5)")
    for f in ("seen", "lt_along", "lt_perp", "lt_dist"):
        m = d[f].notna()
        s, yy = d.loc[m, f].to_numpy(float), d.loc[m, "y"].to_numpy(int)
        print(f"   {f:<10} кадров {m.sum():>3}  ROC-AUC {roc_auc(s, yy):.3f}")
    print("   U-образная связь наблюдается в таблице видимости выше.")
    print("   AUC около 0.5 сама по себе не доказывает форму связи геометрии.")

    print("\nКАНДИДАТЫ В ПРАВИЛО")
    print(rule_table(d).to_string(index=False))
    print("   Перенос на 114 негативов — оценка при сохранении специфичности;")
    print("   это не проверка на 150 кадрах и не верхний предел качества модели.")

    print("\nУСТОЙЧИВОСТЬ ПО ФОЛДАМ (доля нарушений / число кадров)")
    piv = d.assign(lab=d.seen.map(NAMES)).pivot_table(
        index="fold", columns="lab", values="y", aggfunc=["mean", "size"])
    print(piv.round(2).to_string())

    print("\nКОНФАУНДЫ")
    print(d.assign(lab=d.seen.map(NAMES)).groupby("lab").agg(
        кадров=("y", "size"), доля_нарушений=("y", "mean"),
        доля_правых=("region", lambda s: (s == "femur_right").mean())).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
