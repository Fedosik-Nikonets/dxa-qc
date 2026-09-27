"""Image geometry and isolation of model choice from outer-fold labels."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dxa_qc.medial_track import _shaft, track_medial_edge


def phantom():
    mask = np.zeros((300, 260), bool)
    for y in range(50, 300):
        bump = int(round(12*np.exp(-((y-155)/9)**2)))
        mask[y, 130-bump:181] = True
    image = np.where(mask, .9, .05)
    return image, mask


def test_edge_follows_known_medial_bump_and_mirror():
    image, mask = phantom()
    left = track_medial_edge(image, "femur_left", mask=mask)
    right = track_medial_edge(image[:, ::-1], "femur_right", mask=mask[:, ::-1])
    np.testing.assert_array_equal(left.xs, right.xs)
    np.testing.assert_allclose(left.local_bump_mm, right.local_bump_mm)
    at_peak = np.flatnonzero(left.ys == 155)[0]
    assert abs(left.xs[at_peak]-118) <= 2
    candidate = left.candidate("restricted")
    assert abs(candidate["anchor_y"]-155) <= 4
    assert candidate["bump_mm"] > 2


def test_exposure_change_does_not_change_the_path():
    image, mask = phantom()
    a = track_medial_edge(image, "femur_left", mask=mask)
    b = track_medial_edge(image*17+50, "femur_left", mask=mask)
    np.testing.assert_array_equal(a.xs, b.xs)


def test_short_distal_fit_ignores_sparse_medial_outliers():
    _, mask = phantom()
    mask[271:273, 85:130] = True
    slope, intercept, width, start, rms = _shaft(mask, .6)
    assert slope == pytest.approx(0)
    assert intercept == pytest.approx(130)
    assert width == pytest.approx(50)
    assert start >= 240
    assert rms > 0   # Residual is still exposed; robustness does not hide defects.


def test_search_windows_cannot_overlap_the_distal_fit():
    image, mask = phantom()
    track = track_medial_edge(image, "femur_left", mask=mask)
    for mode in ("restricted", "shaft_bounded"):
        candidate = track.candidate(mode)
        assert candidate["window_start"] <= candidate["anchor_y"] <= candidate["window_end"]
        assert candidate["window_end"]+6/track.mm <= track.fit_start
    # A late neck transition leaves no measurable window; do not widen it
    # into the fitted shaft as a hidden fallback.
    track.residual_mm[:] = track.width_px*track.mm
    with pytest.raises(ValueError, match="separate search window"):
        track.candidate("restricted")


@pytest.mark.parametrize("image", [None, np.zeros((1, 1)), np.zeros((100, 100)), np.full((100, 100), np.nan)])
def test_unusable_images_fail_explicitly(image):
    with pytest.raises(ValueError):
        track_medial_edge(image, "femur_left")


def test_unknown_side_and_incompatible_mask_fail():
    image, mask = phantom()
    with pytest.raises(ValueError, match="side"):
        track_medial_edge(image, "spine", mask=mask)
    with pytest.raises(ValueError, match="mask"):
        track_medial_edge(image, "femur_left", mask=mask[:100])


def load_experiment():
    path = Path(__file__).resolve().parents[1] / "scripts/12_refine_rotation.py"
    spec = importlib.util.spec_from_file_location("refine_rotation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_outer_labels_do_not_change_selected_variant_or_predictions():
    module = load_experiment()
    module.VARIANTS = {"signal": ["signal"], "noise": ["noise"]}
    frame = pd.DataFrame({
        "study": [f"S{i}" for i in range(30)], "rel_path": [f"P{i}" for i in range(30)],
        "image_id": "", "region": "femur_left", "fold": np.repeat(np.arange(5), 6),
        "y": [1, 0, 0, 0, 0, 1]*5, "signal": [0., 2., 3., 4., 5., 7.]*5,
        "noise": [1.]*30,
    })
    before, choices_before, _ = module.nested_select(frame)
    frame.loc[frame.fold == 0, "y"] = 1-frame.loc[frame.fold == 0, "y"]
    after, choices_after, _ = module.nested_select(frame)
    assert choices_before[0] == choices_after[0]
    cols = ["variant", "pred", "risk", "low", "high", "train_studies_sha256"]
    pd.testing.assert_frame_equal(before.loc[before.fold == 0, cols],
                                  after.loc[after.fold == 0, cols])


def test_shared_study_across_folds_is_rejected():
    module = load_experiment()
    frame = pd.DataFrame({"study": ["same"], "global_bump_mm": [1.], "y": [0]})
    with pytest.raises(ValueError, match="leakage"):
        module.fit_predict(frame, frame, "global_bump")
