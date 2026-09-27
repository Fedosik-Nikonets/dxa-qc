"""Image-only medial-edge tracking for rotation research.

Returns measurements and candidate locations, not clinical rotation decisions.
No labels, studies, landmarks or model weights are loaded here. Coordinates
are canonical: right femurs are reflected to left. The named search windows
are hypotheses to compare, not validated lesser-trochanter detectors.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .femur import femur_mask, field_scale
from .segmentation import segment_bone


@dataclass
class MedialTrack:
    ys: np.ndarray
    xs: np.ndarray
    residual_mm: np.ndarray
    local_bump_mm: np.ndarray
    slope: float
    intercept: float
    width_px: float
    fit_start: int
    fit_rms_mm: float
    mm: float
    rows: int

    def candidate(self, mode: str = "restricted") -> dict:
        if mode not in {"global", "restricted", "shaft_bounded"}:
            raise ValueError("unknown candidate window")
        if mode == "global":
            start, end = .45 * self.rows, .85 * self.rows
        else:
            # Last large deviation marks the putative neck-to-shaft transition.
            wide = self.ys[self.residual_mm > self.width_px * self.mm * .5]
            neck_end = float(wide[-1]) if len(wide) else .4 * self.rows
            start = neck_end + 5 / self.mm
            end = (int(.8 * self.rows) if mode == "restricted" else self.fit_start) - 6 / self.mm
        idx = np.flatnonzero((self.ys >= start) & (self.ys <= end))
        if len(idx) < 5:
            raise ValueError("insufficient separate search window below neck")
        j = idx[np.argmax(self.local_bump_mm[idx])]
        peaks = idx[(self.local_bump_mm[idx] >= self.local_bump_mm[np.maximum(idx-1, 0)])
                    & (self.local_bump_mm[idx] >= self.local_bump_mm[np.minimum(idx+1, len(self.ys)-1)])]
        last = peaks[-1] if len(peaks) else j
        return {"anchor_y": float(self.ys[j]), "anchor_x": float(self.xs[j]),
                "bump_mm": float(self.local_bump_mm[j]),
                "absolute_mm": float(self.residual_mm[j]),
                "last_bump_mm": float(self.local_bump_mm[last]),
                "last_y": float(self.ys[last]),
                "window_start": float(start), "window_end": float(end),
                "at_window_boundary": bool(j in (idx[0], idx[-1]))}


def _shaft(mask: np.ndarray, mm: float):
    h = mask.shape[0]
    # At most 20 mm and at most 20% of image height; reserve the last 3 rows.
    rows = np.arange(max(h-int(round(20/mm)), int(.8*h)), h-3)
    medial, lateral, valid_rows = [], [], []
    for y in rows:
        xs = np.flatnonzero(mask[y])
        if len(xs):
            valid_rows.append(y)
            medial.append(xs[0])
            lateral.append(xs[-1])
    rows, medial, lateral = map(np.asarray, (valid_rows, medial, lateral))
    if len(rows) < 12:
        raise ValueError("insufficient distal shaft")
    # Median of pairwise slopes: small contour defects do not swing the line.
    dy = rows[:, None] - rows[None, :]
    use = np.triu(np.ones(dy.shape, bool), k=5)
    slope = float(np.median((medial[:, None]-medial[None, :])[use] / dy[use]))
    intercept = float(np.median(medial-slope*rows))
    width = float(np.median(lateral-medial))
    rms = float(np.sqrt(np.mean((medial-slope*rows-intercept)**2)))
    if width < 5:
        raise ValueError("shaft too narrow")
    return slope, intercept, width, int(rows[0]), rms * mm


def track_medial_edge(image: np.ndarray, region: str, mm_per_px: float = .6,
                      mask: np.ndarray | None = None, *,
                      mm_per_px_source: str | None = None,
                      model: str | None = None) -> MedialTrack:
    a = np.asarray(image, dtype=float) if image is not None else np.empty((0, 0))
    if a.ndim != 2 or min(a.shape, default=0) < 60 or not np.isfinite(a).all():
        raise ValueError("expected finite 2D image, at least 60 pixels per dimension")
    if np.ptp(a) <= 0:
        raise ValueError("constant image has no edge")
    if region not in {"femur_left", "femur_right"}:
        raise ValueError("unknown femur side")
    mm, _ = field_scale(mm_per_px, source=mm_per_px_source, model=model)
    if mask is None:
        mask = femur_mask(a, segment_bone(a))
    if mask is None or np.shape(mask) != a.shape or not np.any(mask):
        raise ValueError("missing or incompatible femur mask")
    mask = np.asarray(mask, bool)
    if region == "femur_right":
        a, mask = a[:, ::-1], mask[:, ::-1]
    # Intensity normalization makes path costs insensitive to linear exposure.
    a = (a-a.min())/np.ptp(a)
    h, w = a.shape
    slope, intercept, width, fit_start, rms = _shaft(mask, mm)
    smooth = cv2.GaussianBlur(a.astype(np.float32), (0, 0), 1.)
    gradient = np.maximum(np.gradient(smooth, axis=1), 0)
    norm = np.maximum(np.quantile(gradient, .98, axis=1), 1e-4)
    gradient = np.minimum(gradient/norm[:, None], 2.)
    xs = np.arange(w)
    ys = np.arange(int(h*.3), h-3)
    cost = ((xs-(slope*ys[-1]+intercept))/2)**2
    paths = []
    steps = np.arange(-4, 5)
    for y in ys[::-1]:
        expected = slope*y+intercept
        band = (xs >= expected-width*1.2) & (xs <= expected+width*.4)
        transitions = []
        for step in steps:
            source = np.clip(xs+step, 0, w-1)
            shifted = cost[source]+.10*(step-slope)**2
            shifted[(xs+step < 0) | (xs+step >= w)] = 1e9
            transitions.append(shifted)
        transitions = np.asarray(transitions)
        choices = np.argmin(transitions, axis=0)
        cost = np.min(transitions, axis=0)-gradient[y]+.08*((xs-expected)/width)**2
        cost[~band] = 1e9
        paths.append(steps[choices])
    position = int(np.argmin(cost))
    if cost[position] >= 1e8:
        raise ValueError("no continuous edge path")
    edge = []
    for path in paths[::-1]:
        edge.append(position)
        position = int(np.clip(position+path[position], 0, w-1))
    edge = np.asarray(edge)
    residual = (slope*ys+intercept-edge)*mm/np.hypot(1, slope)
    narrow = cv2.GaussianBlur(residual.reshape(-1, 1), (1, 0), sigmaX=0, sigmaY=2/mm).ravel()
    broad = cv2.GaussianBlur(residual.reshape(-1, 1), (1, 0), sigmaX=0, sigmaY=10/mm).ravel()
    return MedialTrack(ys, edge, residual, narrow-broad, slope, intercept,
                       width, fit_start, rms, mm, h)
