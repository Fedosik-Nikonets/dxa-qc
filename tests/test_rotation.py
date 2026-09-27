"""Physical meaning, missingness and fold isolation of rotation research."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dxa_qc.rotation import (
    fit_normal_interval, interval_predict, medial_contour, prominence,
)


def mask_with_bump(shift=0):
    mask = np.zeros((300, 260), bool)
    for y in range(50, 300):
        bump = 12 if 145 <= y <= 165 else 0
        mask[y, 130+shift-bump:180+shift] = True
    return mask


def test_prominence_is_distance_in_mm_and_reflection_invariant():
    mask = mask_with_bump()
    left = medial_contour(mask, "femur_left", .6)
    right = medial_contour(mask[:, ::-1], "femur_right", .6)
    assert prominence(left, 155)["prominence_mm"] == pytest.approx(7.2)
    assert prominence(right, 155)["prominence_mm"] == pytest.approx(7.2)
    shifted = medial_contour(mask_with_bump(10), "femur_left", .6)
    assert prominence(shifted, 155)["prominence_mm"] == pytest.approx(7.2)


def test_blank_or_missing_landmark_is_not_a_zero_measurement():
    with pytest.raises(ValueError, match="empty"):
        medial_contour(np.zeros((300, 280)), "femur_left", .6)
    c = medial_contour(mask_with_bump(), "femur_left", .6)
    with pytest.raises(ValueError, match="missing"):
        prominence(c, np.nan)
    pred, score = interval_predict([np.nan, 2, 10], 1, 6)
    assert pred.tolist() == [-1, 0, 1]
    assert np.isnan(score[0])


def test_two_tails_recover_small_and_large_prominence():
    scores = np.array([0, 1, 3, 4, 5, 7, 8.])
    labels = np.array([1, 1, 0, 0, 0, 1, 1])
    low, high = fit_normal_interval(scores, labels)
    pred, _ = interval_predict(scores, low, high)
    assert np.array_equal(pred, labels)


def test_constant_scores_do_not_turn_every_case_into_a_violation():
    low, high = fit_normal_interval(np.ones(8), np.array([0]*4+[1]*4))
    pred, score = interval_predict(np.ones(8), low, high)
    assert not pred.any()
    assert np.isfinite(score).all()


def test_invalid_scale_side_and_single_class_rejected():
    with pytest.raises(ValueError, match="scale"):
        medial_contour(mask_with_bump(), "femur_left", 0)
    with pytest.raises(ValueError, match="side"):
        medial_contour(mask_with_bump(), "spine", .6)
    with pytest.raises(ValueError, match="both classes"):
        fit_normal_interval(np.arange(5), np.zeros(5))


def test_changing_validation_labels_cannot_change_its_thresholds():
    path = Path(__file__).resolve().parents[1] / "scripts/11_rotation_landmarks.py"
    spec = importlib.util.spec_from_file_location("rotation_experiment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frame = pd.DataFrame({
        "study": [f"S{i}" for i in range(15)], "rel_path": [f"P{i}" for i in range(15)],
        "region": "femur_left", "fold": np.repeat(np.arange(5), 3),
        "y_rot": [1, 0, 1]*5, "image_id": "", "selected": True,
        "feature": [0., 3., 7.]*5,
    })
    first = module.evaluate_oof(frame, "test", ["feature"], True)
    frame.loc[frame.fold == 0, "y_rot"] = [0, 1, 0]
    second = module.evaluate_oof(frame, "test", ["feature"], True)
    columns = ["pred", "score", "threshold_low", "threshold_high"]
    pd.testing.assert_frame_equal(first.loc[first.fold == 0, columns],
                                  second.loc[second.fold == 0, columns])
