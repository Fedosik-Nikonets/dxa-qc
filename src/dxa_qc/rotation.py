"""Research measurements of the medial femur contour, not a deployed rule.

Both sides are reflected to left-femur coordinates (medial is left).
Prominence is the perpendicular distance from the distal medial shaft
line to the contour near the lesser trochanter. A supplied anchor belongs
to the annotation-assisted experiment only; automatic inference never
loads the annotation table. Missing landmarks are not zero prominence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Contour:
    medial: np.ndarray
    lateral: np.ndarray
    slope: float
    intercept: float
    width: float
    fit_start: int
    mm: float


def medial_contour(mask: np.ndarray, region: str, mm: float) -> Contour:
    a = np.asarray(mask)
    if a.ndim != 2 or min(a.shape, default=0) < 30 or not a.any():
        raise ValueError("empty or too small femur mask")
    if region not in {"femur_left", "femur_right"}:
        raise ValueError("unknown femur side")
    if not np.isfinite(mm) or mm <= 0:
        raise ValueError("invalid pixel scale")
    a = a.astype(bool)
    if region == "femur_right":
        a = a[:, ::-1]
    ys = np.arange(a.shape[0])
    medial = np.full(len(ys), np.nan)
    lateral = medial.copy()
    for y in ys:
        xs = np.flatnonzero(a[y])
        if xs.size:
            medial[y], lateral[y] = xs[0], xs[-1]
    # Exclude the last two rows, which can contain padding/border artefacts.
    start = int(0.75 * len(ys))
    fit = ys[(ys >= start) & (ys < len(ys) - 2) & np.isfinite(medial)]
    if len(fit) < 12:
        raise ValueError("insufficient distal shaft contour")
    slope, intercept = np.polyfit(fit, medial[fit], 1)
    width = float(np.median(lateral[fit] - medial[fit]))
    if width < 4:
        raise ValueError("shaft too narrow")
    return Contour(medial, lateral, float(slope), float(intercept), width, start, mm)


def prominence(contour: Contour, anchor_y: float, radius_mm: float = 6.0) -> dict:
    if not np.isfinite(anchor_y) or not 0 <= anchor_y < len(contour.medial):
        raise ValueError("missing or out-of-frame lesser trochanter")
    if not np.isfinite(radius_mm) or radius_mm <= 0:
        raise ValueError("invalid search radius")
    ys = np.arange(len(contour.medial))
    win = (abs(ys - anchor_y) <= radius_mm / contour.mm) & np.isfinite(contour.medial)
    if win.sum() < 5:
        raise ValueError("insufficient contour near landmark")
    residual = (contour.slope * ys + contour.intercept - contour.medial)
    residual *= contour.mm / np.hypot(1, contour.slope)
    return {
        "prominence_mm": float(np.quantile(residual[win], 0.9)),
        "anchor_y": float(anchor_y),
        "shaft_width_mm": contour.width * contour.mm / np.hypot(1, contour.slope),
        "shaft_angle_deg": abs(float(np.degrees(np.arctan(contour.slope)))),
        "window_overlaps_shaft_fit": bool(anchor_y + radius_mm / contour.mm >= contour.fit_start),
    }


def automatic_anchor(contour: Contour) -> float:
    """Fixed geometric hypothesis: GT + two shaft widths, no learned points.

    The GT is approximated by the first contour row lateral to the shaft
    centre plus 0.2 shaft widths. This deliberately simple localizer is
    evaluated against annotations before any deployment decision.
    """
    ys = np.arange(len(contour.medial))
    shaft_centre = contour.slope * ys + contour.intercept + contour.width / 2
    candidates = ys[(contour.lateral > shaft_centre + 0.2 * contour.width)
                    & (ys < contour.fit_start)]
    if len(candidates) < 5:
        raise ValueError("cannot anchor greater trochanter")
    gt_y = float(candidates[0])
    anchor = gt_y + 2 * contour.width
    if anchor >= contour.fit_start:
        raise ValueError("automatic landmark overlaps distal shaft fit")
    return anchor


def fit_normal_interval(scores: np.ndarray, labels: np.ndarray,
                        min_specificity: float = 0.75) -> tuple[float, float]:
    """Fit two tails on training data only: score < low or score > high.

    Maximize F1 subject to the specificity gate from handoff_day5.
    Ties prefer fewer false positives, then a wider normal interval.
    NaN measurements are excluded from fitting, not assigned zero.
    """
    s, y = np.asarray(scores, float), np.asarray(labels)
    if s.ndim != 1 or y.shape != s.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("expected equally sized score and binary label vectors")
    if not 0 <= min_specificity <= 1:
        raise ValueError("specificity must be in [0, 1]")
    valid = np.isfinite(s)
    s, y = s[valid], y[valid]
    if len(np.unique(y)) != 2:
        raise ValueError("training data need both classes")
    unique = np.unique(s)
    cuts = np.r_[-np.inf, (unique[:-1] + unique[1:]) / 2, np.inf]
    best, interval = (-1., -np.inf, -np.inf), (-np.inf, np.inf)
    for low in cuts:
        for high in cuts[cuts >= low]:
            pred = (s < low) | (s > high)
            tp = int((pred & (y == 1)).sum())
            fp = int((pred & (y == 0)).sum())
            if fp / int((y == 0).sum()) > 1 - min_specificity + 1e-12:
                continue
            f1 = 2 * tp / (int((y == 1).sum()) + tp + fp)
            width = high - low if high > low else 0.
            rank = (f1, -fp, width)
            if rank > best:
                best, interval = rank, (float(low), float(high))
    return interval


def interval_predict(scores, low: float, high: float) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(scores, float)
    risk = np.maximum(low - s, s - high)
    pred = np.where(np.isfinite(s), (s < low) | (s > high), -1).astype(int)
    # No finite boundary means no abnormal tails were supported in training.
    if np.isneginf(low) and np.isposinf(high):
        risk = np.where(np.isfinite(s), 0., np.nan)
    return pred, risk
