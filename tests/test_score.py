"""Балл нарушения quality_score: согласованность с решением и крайние случаи."""
import math

import pytest

from dxa_qc import score
from dxa_qc.femur import FIELD_MIN_MM
from dxa_qc.spine import AXIS_TILT_DEG, COVERAGE_MIN

PARAMS = {"version": score.PARAMS_VERSION,
          "slopes": {"pos": 16.0, "tilt": 0.7, "art": 2.5, "roi": 0.2, "rot": 1.7}}


def spine_diag(angle=1.0, coverage=0.3, prob=0.1):
    return {"spine_angle_deg": angle, "spine_coverage": coverage, "artifact_prob": prob}


def femur_diag(field=150.0, value=0.2, low=-0.2, high=0.66):
    return {"femur_field_h_mm": field, "rotation_value": value,
            "rotation_low": low, "rotation_high": high}


def test_margins_sign_matches_rules():
    m = score.margins("spine", spine_diag(angle=AXIS_TILT_DEG + 1, coverage=COVERAGE_MIN - 0.01, prob=0.9),
                      artifact_threshold=0.5)
    assert m["tilt"] == pytest.approx(1.0) and m["pos"] > 0 and m["art"] > 0
    m = score.margins("femur_left", femur_diag(field=FIELD_MIN_MM + 10, value=1.0))
    assert m["roi"] == pytest.approx(-10.0) and m["rot"] == pytest.approx(1.0 - 0.66)
    m = score.margins("femur_right", femur_diag(value=-0.5))
    assert m["rot"] == pytest.approx(0.3)


def test_score_above_half_iff_violation():
    clean = score.quality_score("spine", spine_diag(), 0, params=PARAMS, artifact_threshold=0.5)
    bad = score.quality_score("spine", spine_diag(angle=8.0), 1, params=PARAMS, artifact_threshold=0.5)
    assert 0 <= clean < 0.5 < bad <= 1
    # Правило чуть не дотянуло до порога, а другие сработали бы вместе — всё равно ниже 0.5.
    near = score.quality_score("femur_left", femur_diag(field=FIELD_MIN_MM + 0.1, value=0.65), 0, params=PARAMS)
    assert near < 0.5
    # Решение «есть нарушение» при маленьких маржах — не ниже 0.5001.
    edge = score.quality_score("femur_left", femur_diag(field=FIELD_MIN_MM - 0.01), 1, params=PARAMS)
    assert edge >= 0.5001


def test_score_grows_with_margin():
    s = [score.quality_score("spine", spine_diag(angle=a), 1, params=PARAMS, artifact_threshold=0.5)
         for a in (5.0, 7.0, 10.0)]
    assert s[0] < s[1] < s[2]


def test_missing_measurements_are_skipped():
    s = score.quality_score("spine", {"spine_angle_deg": "", "spine_coverage": "", "artifact_prob": ""}, 0,
                            params=PARAMS, artifact_threshold=0.5)
    assert s == pytest.approx(0.0)
    s = score.quality_score("femur_left", {"femur_field_h_mm": 150.0}, 0, params=PARAMS)
    assert 0 <= s < 0.5


def test_no_score_without_decision_or_params(tmp_path):
    assert score.quality_score("spine", spine_diag(), None, params=PARAMS) is None
    assert score.quality_score("unknown", spine_diag(), 0, params=PARAMS) is None
    assert score.load_params(tmp_path / "нет.json") is None
    bad = tmp_path / "old.json"
    bad.write_text('{"version": "other", "slopes": {}}', encoding="utf-8")
    assert score.load_params(bad) is None


def test_study_score_follows_study_class():
    assert score.study_score([0.2, 0.3], 0) < 0.5
    assert score.study_score([0.2, 0.8], 1) >= 0.8
    assert score.study_score([None, None], 0) is None


def test_default_params_are_shipped():
    p = score.load_params()
    assert p is not None and set(p["slopes"]) == {"pos", "tilt", "art", "roi", "rot"}
    assert all(v > 0 and math.isfinite(v) for v in p["slopes"].values())
