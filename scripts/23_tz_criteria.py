#!/usr/bin/env python3
"""Три критерия ТЗ, которые не стали правилами, — проверка на данных, 24 сентября.

Отвечает на вопрос «почему не проверяется», цифрами, а не словами:

1. **Половина тела Th12 (ТЗ 2.3.1).** Из шести позитивов укладки правило по
   гребням не ловит два — у них нарушена верхняя половина критерия. Здесь
   для них считаются длина скана и расстояние от верха кадра до гребней
   подвздошных костей (первая строка нижних двух третей, где плотность сбоку
   от позвоночника выше порога) и ранг позитивов среди негативов.
2. **Большой вертел, шейка, седалищная кость (ТЗ 2.3.2, рис. 4).**
   Видимость каждого ориентира оценил врач на 80 кадрах
   (`annotations/femur_points_human.csv`). Таблица сопряжённости видимости с
   меткой эксперта организатора «позиционирование/ротация» и перекрытие с
   решениями текущего правила ротации.
3. **2 см сбоку (ТЗ 2.3.2, рис. 6).** Буквальное правило по отступу кости
   со стороны вертела до края поля сканирования (`femur_rules.csv`).

Выход — `reports/tz_criteria/`: `summary.json`, `femur_landmarks.csv`,
`spine_th12.csv`. Время — около 20 с (читаются 99 снимков позвоночника).

Запуск:
    python scripts/23_tz_criteria.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dxa_qc.dicom_io import read_dicom  # noqa: E402
from dxa_qc.segmentation import segment_bone  # noqa: E402
from dxa_qc.spine import DEFAULT_MM_PER_PX, LATERAL_PAD_MM, SMOOTH_MM  # noqa: E402

REPORTS = ROOT / "reports"
OUT = REPORTS / "tz_criteria"
DATA = ROOT / "Data" / "НД_для_обучения" / "Исследования"
LANDMARKS = {"gt_state": "большой вертел", "neck_state": "шейка", "isch_state": "седалищная кость",
             "lt_state": "малый вертел"}
CREST_THRESHOLDS = (0.05, 0.08, 0.12)


def auc(score, y) -> float:
    s, y = np.asarray(score, float), np.asarray(y, int)
    p, n = s[y == 1], s[y == 0]
    return float(((p[:, None] > n[None]).sum() + 0.5 * (p[:, None] == n[None]).sum()) / len(p) / len(n))


def femur_landmarks() -> tuple[pd.DataFrame, dict]:
    h = pd.read_csv(ROOT / "annotations/femur_points_human.csv")
    m = pd.read_csv(ROOT / "annotations/femur_manifest_v1.csv")[["image_id", "rel_path", "y_FEMUR_POSITIONING_ROTATION"]]
    dec = pd.read_csv(REPORTS / "rotation_rule/decisions.csv")[["rel_path", "pred"]]
    fr = pd.read_csv(REPORTS / "femur_rules.csv")[["rel_path", "lateral_mm"]]
    d = (h.merge(m, on="image_id", validate="one_to_one").merge(dec, on="rel_path", validate="one_to_one")
          .merge(fr, on="rel_path", validate="one_to_one")
          .rename(columns={"y_FEMUR_POSITIONING_ROTATION": "y_rot", "pred": "pred_rot"}))
    out = {}
    for col, name in LANDMARKS.items():
        t = pd.crosstab(d[col], d.y_rot.astype(int))
        out[name] = {state: {"норма": int(t.loc[state].get(0, 0)), "нарушение": int(t.loc[state].get(1, 0))}
                     for state in t.index}
    gt = d[d.gt_state == "not_visible"]
    out["большой вертел не виден: нарушений, уже пойманных правилом ротации"] = int(((gt.y_rot == 1) & (gt.pred_rot == 1)).sum())
    out["боковой отступ, медиана мм: вертел не виден / виден"] = [
        float(d.loc[d.gt_state == "not_visible", "lateral_mm"].median()),
        float(d.loc[d.gt_state == "visible", "lateral_mm"].median())]
    return d, out


def lateral_rule() -> dict:
    f = pd.read_csv(REPORTS / "femur_rules.csv")
    v, y = f.lateral_mm, f.y_roi.astype(int)
    flag = v < 20
    return {"кадров": int(len(f)), "отступ < 20 мм": int(flag.sum()),
            "медиана отступа у нормы, мм": float(v[y == 0].median()),
            "TP/FP/FN": [int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum()), int((~flag & (y == 1)).sum())],
            "ROC-AUC (меньше отступ — нарушение)": round(auc(-v, y), 3)}


def lateral_profile(a: np.ndarray, mask: np.ndarray, mm: float) -> np.ndarray:
    """Плотность сбоку от позвоночника по строкам — как в `spine.lateral_coverage`."""
    h, w = a.shape
    mid = mask[h // 3: 2 * h // 3]
    cols = np.flatnonzero(mid.any(axis=0))
    c0, c1 = int(cols.min()), int(cols.max()) + 1
    pad = max(int(round(LATERAL_PAD_MM / mm)), 2)
    lat = np.zeros(w, bool)
    lat[:max(c0 - pad, 0)] = True
    lat[min(c1 + pad, w):] = True
    base, bone = float(np.percentile(a, 1)), float(a[mask].mean())
    prof = (a[:, lat].mean(axis=1) - base) / max(bone - base, 1e-6)
    k = max(3, int(round(SMOOTH_MM / mm)) | 1)
    return np.convolve(prof, np.ones(k) / k, mode="same")


def spine_th12() -> tuple[pd.DataFrame, dict]:
    sp = pd.read_csv(REPORTS / "spine_rules.csv")
    rows = []
    for r in sp.itertuples():
        img = read_dicom(DATA / r.rel_path)
        a = img.pixels.astype(float)
        mm = img.mm_per_px or DEFAULT_MM_PER_PX
        prof = lateral_profile(a, segment_bone(a).mask, mm)
        h = a.shape[0]
        row = {"rel_path": r.rel_path, "y_pos": int(r.y_pos), "pred_pos": int(r.pred_pos),
               "coverage": r.coverage, "frame_mm": h * mm}
        for t in CREST_THRESHOLDS:
            idx = np.flatnonzero(prof[h // 3:] > t)
            row[f"crest_mm_{t}"] = (h // 3 + idx[0]) * mm if idx.size else np.nan
        rows.append(row)
    d = pd.DataFrame(rows)
    d["group"] = np.select([(d.y_pos == 1) & (d.pred_pos == 1), (d.y_pos == 1) & (d.pred_pos == 0)],
                           ["TP", "FN_th12"], "negative")
    neg, fn = d[d.group == "negative"], d[d.group == "FN_th12"]
    out = {"позитивов укладки": int(d.y_pos.sum()), "из них не ловит правило по гребням": int(len(fn))}
    for c in ["frame_mm"] + [f"crest_mm_{t}" for t in CREST_THRESHOLDS]:
        v = neg[c].dropna()
        out[c] = {"позитивы Th12": [round(float(x), 1) for x in fn[c]],
                  "негативы: медиана": round(float(v.median()), 1),
                  "ранг позитивов среди негативов": [int((v < x).sum()) for x in fn[c]],
                  "негативов": int(len(v))}
    return d, out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fl, fl_sum = femur_landmarks()
    sp, sp_sum = spine_th12()
    summary = {"Th12 (ТЗ 2.3.1)": sp_sum,
               "ориентиры бедра по оценке врача, 80 кадров (ТЗ 2.3.2, рис. 4)": fl_sum,
               "2 см сбоку, 150 кадров (ТЗ 2.3.2, рис. 6)": lateral_rule()}
    fl.to_csv(OUT / "femur_landmarks.csv", index=False)
    sp.round(4).to_csv(OUT / "spine_th12.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
