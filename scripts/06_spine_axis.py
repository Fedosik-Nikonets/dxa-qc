#!/usr/bin/env python3
"""Разведка под контрольную точку 18 сентября: правило наклона оси позвоночника.

Задача на 18-е — довести AUC по правилу ТЗ 2.3.1 («наклон оси более 5°») до
0.9. День 2 дал 0.88 на модуле угла главной оси всей маски и объяснил
недобор тем, что «в маску входят рёбра и подвздошные кости, они тянут ось».

**Это объяснение неверно.** Маска позвоночника — узкая вертикальная полоса:
bbox по ширине 101 пиксель из 300, поперечник строки 64 пикселя, одна
компонента. Ни рёбер, ни крыльев таза в ней нет, выделять из неё столб
нечего (проверено прослеживанием столба по строкам: в него попадает 96%
площади маски, AUC не растёт, а падает).

Настоящая причина — в семантике метки. Скрипт считает пять оценок угла и
четыре признака кривизны, сводит AUC и разбирает ложные срабатывания по
комментариям эксперта.

Запуск:
    python scripts/06_spine_axis.py --dataset reports/dataset_images.csv \
        --root "Data/НД_для_обучения/Исследования" --out reports/spine_axis.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dxa_qc.dicom_io import read_dicom  # noqa: E402
from dxa_qc.segmentation import principal_axis, segment_bone  # noqa: E402

THRESHOLDS = (3.5, 4.0, 4.5, 5.0, 5.5)
OUT_OF_SCOPE = "сколиоз|перелом|внимани|люмбализ|эндопротез"


def centerline(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Построчные центроиды маски: (номера строк, координаты центров)."""
    w = mask.sum(axis=1)
    ok = w > 0
    cen = (mask[ok] * np.arange(mask.shape[1])).sum(axis=1) / w[ok]
    return np.arange(mask.shape[0])[ok].astype(float), cen


def angle_features(mask: np.ndarray) -> dict:
    """Пять способов оценить наклон оси. Знак: плюс — верх отклонён вправо."""
    rr = np.flatnonzero(mask.any(axis=1))
    half = (int(rr[0]) + int(rr[-1])) // 2
    top, bot = mask.copy(), mask.copy()
    top[half:, :] = False
    bot[:half, :] = False

    r, c = centerline(mask)
    slope = float(np.polyfit(r, c, 1)[0])
    q = max(int(len(r) * 0.25), 1)
    chord = (c[:q].mean() - c[-q:].mean()) / (r[:q].mean() - r[-q:].mean())

    return {
        "ang_full": principal_axis(mask)[0],          # главная ось всей маски
        "ang_top": principal_axis(top)[0],            # только верхняя половина
        "ang_bot": principal_axis(bot)[0],            # только нижняя половина
        "ang_cen": float(np.degrees(np.arctan(slope))),   # линия центроидов
        "ang_chord": float(np.degrees(np.arctan(chord))), # хорда четвертей
    }


def curvature_features(mask: np.ndarray) -> dict:
    """Признаки кривизны: чем сколиотический позвоночник отличается от косо
    уложенного прямого.

    Косая укладка — прямая линия под углом. Сколиоз — изогнутая линия, у
    которой локальные наклоны разъезжаются. Оба дают большой общий угол,
    но эксперт первое считает нарушением техники, а второе — анатомией.
    """
    r, c = centerline(mask)
    k = max(5, len(r) // 15) | 1
    sm = np.convolve(c, np.ones(k) / k, mode="valid")
    rs = r[k // 2: k // 2 + len(sm)]
    lin = np.polyval(np.polyfit(rs, sm, 1), rs)
    cub = np.polyval(np.polyfit(rs, sm, 3), rs)
    res = sm - lin

    n, th = len(rs), len(rs) // 3
    slopes = [float(np.polyfit(rs[i * th:(i + 1) * th], sm[i * th:(i + 1) * th], 1)[0])
              for i in range(3)] if th >= 3 else [0.0]

    return {
        "res_rms": float(np.sqrt((res ** 2).mean())),
        "res_amp": float(res.max() - res.min()),
        "cub_gain": float(np.sqrt((res ** 2).mean())
                          - np.sqrt(((sm - cub) ** 2).mean())),
        "slope_spread": float(np.degrees(np.arctan(max(slopes)))
                              - np.degrees(np.arctan(min(slopes)))),
    }


def auc(score, y) -> float:
    """ROC-AUC через статистику Манна — Уитни, с учётом совпадений."""
    s = np.asarray(score, dtype=float)
    y = np.asarray(y)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if not npos or not nneg:
        return float("nan")
    ranks = pd.Series(s).rank().to_numpy()
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def metrics_at(score, y, t: float) -> dict:
    s, y = np.asarray(score, float), np.asarray(y)
    pred = (s >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return {"порог": t, "чувств.": tp / (tp + fn) if tp + fn else float("nan"),
            "специф.": tn / (tn + fp) if tn + fp else float("nan"),
            "F1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
            "TP": tp, "FP": fp, "FN": fn}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    ds = pd.read_csv(args.dataset)
    spine = ds[(ds["region"] == "spine") & ds["labeled"]]
    print(f"позвоночников: {len(spine)}, позитивов по наклону оси: "
          f"{int(spine['y_SPINE_AXIS_TILT'].sum())}")

    rows = []
    for i, x in enumerate(spine.itertuples(), 1):
        img = read_dicom(str(args.root / x.rel_path))
        rec = {"study": x.study, "rel_path": x.rel_path,
               "y": int(x.y_SPINE_AXIS_TILT), "comment": x.comment}
        if img.ok and img.pixels is not None:
            mask = segment_bone(img.pixels).mask
            if mask.any():
                rec.update(angle_features(mask))
                rec.update(curvature_features(mask))
        rows.append(rec)
        if i % 25 == 0:
            print(f"  {i}/{len(spine)}")

    res = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)

    y = res["y"].to_numpy()
    angles = ["ang_full", "ang_cen", "ang_chord", "ang_top", "ang_bot"]
    print("\nAUC, признак — модуль угла:")
    for a in angles:
        print(f"  |{a}|".ljust(14) + f" {auc(res[a].abs(), y):.3f}")
    print("\nAUC, признаки кривизны (наклон оси):")
    for c in ["res_rms", "res_amp", "cub_gain", "slope_spread"]:
        print(f"  {c}".ljust(14) + f" {auc(res[c], y):.3f}")

    # Порог из ТЗ буквальный (5°), но величина угла зависит от способа оценки,
    # поэтому таблица порогов печатается для обоих кандидатов: у хорды AUC
    # выше, а модуль угла систематически меньше, и на фиксированном пороге
    # она проигрывает главной оси по чувствительности.
    best = max(angles, key=lambda a: auc(res[a].abs(), y))
    for a in dict.fromkeys(["ang_full", best]):
        print(f"\nметрики по порогам, |{a}| (AUC {auc(res[a].abs(), y):.3f}):")
        print(pd.DataFrame([metrics_at(res[a].abs(), y, t) for t in THRESHOLDS])
              .to_string(index=False))
    score = res[best].abs()

    # --- откуда берутся ложные срабатывания ------------------------------- #
    oos = res["comment"].fillna("").str.lower().str.contains(OUT_OF_SCOPE)
    fp45 = res[(y == 0) & (score >= 4.5)]
    print(f"\nложных срабатываний при 4.5°: {len(fp45)}, "
          f"из них с находкой вне перечня 2.3 в комментарии: "
          f"{int(fp45['comment'].fillna('').str.lower().str.contains(OUT_OF_SCOPE).sum())}")
    keep = ~(oos & (res["y"] == 0))
    print(f"AUC |{best}| без негативов с находками вне 2.3: "
          f"{auc(score[keep], y[keep]):.3f} (негативов {int((y[keep] == 0).sum())})")

    scol = res["comment"].fillna("").str.lower().str.contains("сколиоз").to_numpy().astype(int)
    print(f"\nсколиоз в комментариях: {int(scol.sum())} из {len(res)}")
    print("AUC «сколиоз против остальных» по признакам кривизны:")
    for c in ["res_amp", "slope_spread", "cub_gain", "res_rms"]:
        print(f"  {c}".ljust(14) + f" {auc(res[c], scol):.3f}")
    print(f"\nзаписано: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
