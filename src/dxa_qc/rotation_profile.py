"""Image-only intensity measurements along the tracked medial cortex.

Research features, not a rotation rule. Right femurs use canonical reflected
coordinates, exactly as MedialTrack. No human coordinates enter this API.
"""
from __future__ import annotations

import numpy as np

from .medial_track import MedialTrack


def canonical_image(image: np.ndarray, region: str) -> np.ndarray:
    a = np.asarray(image, dtype=float)
    if a.ndim != 2 or not np.isfinite(a).all() or np.ptp(a) <= 0:
        raise ValueError("expected finite nonconstant image")
    if region not in {"femur_left", "femur_right"}:
        raise ValueError("unknown femur side")
    a = (a-a.min())/np.ptp(a)
    return a if region == "femur_left" else a[:, ::-1]


def patch_intensity(canonical: np.ndarray, x: float, y: float, radius: int = 2) -> float:
    """Median in a square; incomplete patches are missing, never clipped."""
    if not np.isfinite([x, y]).all():
        return float("nan")
    x, y = int(round(x)), int(round(y))
    h, w = canonical.shape
    if radius < 0 or x-radius < 0 or y-radius < 0 or x+radius >= w or y+radius >= h:
        return float("nan")
    return float(np.median(canonical[y-radius:y+radius+1, x-radius:x+radius+1]))


def profile_features(image: np.ndarray, region: str, track: MedialTrack) -> dict:
    a = canonical_image(image, region)
    if a.shape[0] != track.rows:
        raise ValueError("track does not match image height")
    candidate = track.candidate("restricted")
    use = (track.ys >= candidate["window_start"]) & (track.ys <= candidate["window_end"])
    ys, xs = track.ys[use], track.xs[use]
    radius = max(1, int(round(1.2/track.mm)))
    offset = max(1, int(round(3/track.mm)))
    edge = np.array([patch_intensity(a, x, y, radius) for x, y in zip(xs, ys)])
    inside = np.array([patch_intensity(a, x+offset, y, radius) for x, y in zip(xs, ys)])
    outside = np.array([patch_intensity(a, x-offset, y, radius) for x, y in zip(xs, ys)])
    good = np.isfinite(edge) & np.isfinite(inside) & np.isfinite(outside)
    if good.sum() < 5 or good.mean() < .8:
        raise ValueError("insufficient complete profile patches")
    return {
        "bump_mm": candidate["bump_mm"],
        "anchor_intensity": patch_intensity(a, candidate["anchor_x"], candidate["anchor_y"], radius),
        "profile_intensity": float(np.median(edge[good])),
        "profile_contrast": float(np.median(inside[good]-outside[good])),
        "profile_rows": int(good.sum()),
        **{k: candidate[k] for k in ("anchor_x", "anchor_y", "window_start", "window_end", "at_window_boundary")},
    }
