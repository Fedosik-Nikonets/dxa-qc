"""Правило ротации бедра в конвейере: два хвоста, отказ, заморозка границ."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dxa_qc import rotation_rule as rr
from dxa_qc.batch import predict
from dxa_qc.dicom_io import DicomImage

ROOT = Path(__file__).resolve().parents[1]


def _model(low=-0.2, high=0.7):
    return rr.RotationModel(feature="bump_mm", low=low, high=high)


def test_two_tails_and_normal_interval():
    m = _model()
    assert m.decide(0.0) == 0
    assert m.decide(-0.2) == 0 and m.decide(0.7) == 0     # границы — ещё норма
    assert m.decide(-0.5) == 1                             # вертел спрятан
    assert m.decide(1.5) == 1                              # вертел выступает
    assert m.decide(float("nan")) == -1                    # не измерено — отказ


def test_open_interval_survives_json(tmp_path):
    m = rr.RotationModel(feature="profile_contrast", low=-np.inf, high=0.9)
    back = rr.RotationModel.load(m.save(tmp_path / "m.json"))
    assert back.low == -np.inf and back.high == 0.9
    assert json.loads((tmp_path / "m.json").read_text())["low"] is None


def test_stale_feature_version_is_refused(tmp_path):
    p = _model().save(tmp_path / "m.json")
    d = json.loads(p.read_text())
    d["feature_version"] = "что-то старое"
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        rr.RotationModel.load(p)


def test_unknown_feature_is_refused():
    with pytest.raises(ValueError):
        rr.RotationModel(feature="shaft_deg", low=0.0, high=1.0)


def test_missing_model_file_means_no_check(tmp_path):
    assert rr.load_default(tmp_path / "нет.json") is None


def test_stale_default_model_is_skipped_without_exception(tmp_path, monkeypatch):
    p = _model().save(tmp_path / "m.json")
    d = json.loads(p.read_text())
    d["feature_version"] = "чужая версия"
    p.write_text(json.dumps(d))
    rr._load_cached.cache_clear()
    assert rr.load_default(p) is None
    monkeypatch.setattr(rr, "DEFAULT_MODEL_PATH", p)
    result = rr.evaluate(np.ones((80, 80)), "femur_left", 0.6)
    assert not result.ok and result.violations == []
    assert "не найдены" in result.reason


def test_unmeasurable_image_abstains_without_violation():
    res = rr.evaluate(np.zeros((80, 80)), "femur_left", 0.6, model=_model())
    assert not res.ok and res.violations == []
    assert res.reason.startswith("отказ")


def test_committed_model_matches_its_procedure():
    """Замороженная модель — ровно то, что пишет скрипт 17, и несёт честную оценку."""
    m = rr.load_default()
    assert m is not None and m.feature in rr.FEATURES
    assert np.isfinite(m.low) and np.isfinite(m.high)
    assert m.meta["nested_estimate"]["f1"] == pytest.approx(0.4762, abs=1e-4)
    chosen = max(m.meta["selection_cv"].items(), key=lambda kv: kv[1]["f1"])[0]
    assert chosen == m.feature


def test_committed_decisions_follow_committed_model():
    m = rr.load_default()
    dec = pd.read_csv(ROOT / "reports/rotation_rule/decisions.csv")
    assert len(dec) == 150 and (dec.feature == m.feature).all()
    assert [m.decide(v) for v in dec.value] == dec.pred.tolist()


def test_femur_prediction_reports_rotation_columns():
    a = np.full((240, 160), 18.0)
    a[:80, 128:152] = 220
    a[160:, 72:88] = 220
    img = DicomImage(path="x", ok=True, pixels=a, rows=240, cols=160, mm_per_px=0.6)
    q, v, diag = predict("femur_left", img)
    assert {"rotation_feature", "rotation_value", "rotation_reason"} <= diag.keys()
    assert q == (1 if v else 0)
