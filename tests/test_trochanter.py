"""Контракт шаблонного локализатора малого вертела."""
from __future__ import annotations

import numpy as np

from dxa_qc.medial_track import MedialTrack
from dxa_qc.trochanter import TrainingPoint, fit, locate


def _case():
    image = np.zeros((140, 120), np.float32)
    yy, xx = np.mgrid[:140, :120]
    image += np.exp(-((xx-46)**2+(yy-73)**2)/30)
    image += .35*np.exp(-((xx-52)**2+(yy-66)**2)/18)
    ys = np.arange(42, 137)
    xs = np.full(len(ys), 48.0)
    bump = 2*np.exp(-((ys-73)/5)**2)
    track = MedialTrack(ys, xs, bump, bump, 0.0, 48.0, 25.0, 115,
                        0.1, 0.6, image.shape[0])
    model = fit([TrainingPoint(image, "femur_left", 46, 73, 0.6)])
    return image, track, model


def test_localizer_is_deterministic_and_inside_window():
    image, track, model = _case()
    a = locate(image, "femur_left", 0.6, track, model)
    b = locate(image, "femur_left", 0.6, track, model)
    assert a == b and a.ok
    assert a.window_start <= a.y <= a.window_end
    assert abs(a.canonical_x-48) <= model.search_across_mm/track.mm


def test_left_right_mirror_returns_source_coordinates():
    image, track, model = _case()
    left = locate(image, "femur_left", 0.6, track, model)
    right = locate(image[:, ::-1], "femur_right", 0.6, track, model)
    assert right.ok
    assert right.y == left.y
    assert right.canonical_x == left.canonical_x
    assert right.x == image.shape[1]-1-left.x


def test_empty_image_abstains_without_exception():
    _, track, model = _case()
    result = locate(np.zeros((140, 120)), "femur_left", 0.6, track, model)
    assert not result.ok
    assert result.reason
