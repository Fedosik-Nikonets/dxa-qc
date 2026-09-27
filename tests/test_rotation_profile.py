"""Physical/intensity invariance and study isolation for the new experiment."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dxa_qc.medial_track import track_medial_edge
from dxa_qc.rotation_profile import canonical_image, patch_intensity, profile_features


def phantom():
    mask = np.zeros((300, 260), bool)
    for y in range(50, 300):
        bump = int(round(12*np.exp(-((y-155)/9)**2)))
        mask[y, 130-bump:181] = True
    return np.where(mask, .9, .05), mask


def test_intensity_features_match_mirror_and_linear_exposure():
    image, mask = phantom()
    reference = None
    for img, side, m in ((image, "femur_left", mask),
                         (image[:, ::-1], "femur_right", mask[:, ::-1]),
                         (image*17+50, "femur_left", mask)):
        track = track_medial_edge(img, side, mask=m)
        features = profile_features(img, side, track)
        if reference is None:
            reference = features
        else:
            for key in features:
                assert features[key] == pytest.approx(reference[key])
    assert reference["profile_contrast"] > .5


def test_trusted_pixel_spacing_is_not_clamped_by_tracker():
    image, mask = phantom()
    trusted = track_medial_edge(image, "femur_left", .8, mask,
                                mm_per_px_source="pixel_spacing", model="other")
    estimated = track_medial_edge(image, "femur_left", .8, mask,
                                  mm_per_px_source="exposed_area", model="Lunar Prodigy")
    assert trusted.mm == .8
    assert estimated.mm == .6


def test_missing_or_border_patch_is_not_background_intensity():
    image, _ = phantom()
    a = canonical_image(image, "femur_left")
    assert np.isnan(patch_intensity(a, np.nan, 150))
    assert np.isnan(patch_intensity(a, 0, 150))
    assert patch_intensity(a, 150, 150) == 1.


def test_human_columns_and_outer_labels_do_not_change_automatic_predictions():
    path = Path(__file__).resolve().parents[1] / "scripts/14_rotation_human_profile.py"
    spec = importlib.util.spec_from_file_location("human_profile", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    frame = pd.DataFrame({
        "study": [f"S{i}" for i in range(30)], "rel_path": [f"P{i}" for i in range(30)],
        "image_id": "", "region": "femur_left", "fold": np.repeat(np.arange(5), 6),
        "y": [1, 0, 0, 0, 0, 1]*5, "human_y": 100., "human_state": "visible",
    })
    for i, key in enumerate(mod.VARIANTS):
        frame[key] = [0., 2., 3., 4., 5., 7.]*5 if i == 0 else float(i)
    before, choices_before, _ = mod.cv.nested_select(frame, mod.VARIANTS)
    frame.loc[frame.fold == 0, "y"] = 1-frame.loc[frame.fold == 0, "y"]
    frame["human_y"], frame["human_state"] = -1000., "not_visible"
    after, choices_after, _ = mod.cv.nested_select(frame, mod.VARIANTS)
    assert choices_before[0] == choices_after[0]
    cols = ["variant", "pred", "risk", "low", "high", "train_studies_sha256"]
    pd.testing.assert_frame_equal(before.loc[before.fold == 0, cols],
                                  after.loc[after.fold == 0, cols])
