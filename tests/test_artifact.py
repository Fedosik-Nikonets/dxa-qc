"""Детектор артефактов: признак, переносимость весов, поведение без весов.

Признак проверяется на фантоме: тонкая яркая линия вне позвоночника — это и
есть провод или клипса, ради которых правило существует. Модель проверяется
на переносимость: веса из файла обязаны давать те же вероятности до
последнего знака, иначе в контейнере окажется другая модель, чем на обучении.
"""
from __future__ import annotations

import numpy as np
import pytest

from dxa_qc.artifact import (
    FEATURE_VERSION,
    SPINE_ARTIFACT,
    ArtifactModel,
    evaluate,
    features,
    fit_logistic,
    load_default,
    standardize,
)
from dxa_qc.segmentation import segment_bone


def phantom(wire: bool = False, rows: int = 300, cols: int = 300, peak: int = 220):
    """Столб позвоночника; при wire — тонкая яркая дуга в стороне от него."""
    a = np.full((rows, cols), int(peak * 0.10), dtype=np.float32)
    a[:, cols // 2 - 20: cols // 2 + 20] = peak
    a[int(rows * 0.78):, : cols // 2 - 60] = int(peak * 0.35)
    a[int(rows * 0.78):, cols // 2 + 60:] = int(peak * 0.35)
    if wire:
        for x in range(20, 90):
            y = int(30 + 0.35 * (x - 20))
            a[y:y + 2, x:x + 2] = peak * 1.1
    return a.astype(np.int32)


def _feat(img):
    return features(img, segment_bone(img).mask, 0.6)


# --- признак ---------------------------------------------------------------- #

def test_wire_raises_the_feature_several_times_over():
    clean, wired = _feat(phantom(False)), _feat(phantom(True))
    assert clean.ok and wired.ok
    assert wired.tophat_p999 > 3 * clean.tophat_p999


def test_feature_ignores_the_spine_itself():
    """Позвоночник вырезается с запасом: сама кость не должна давать отклик."""
    f = _feat(phantom(False))
    assert f.tophat_p999 < 0.2


def test_feature_does_not_depend_on_overall_brightness():
    base = phantom(True)
    dim = (base * 0.4).astype(np.int32)
    assert _feat(dim).tophat_p999 == pytest.approx(_feat(base).tophat_p999, rel=0.3)


def test_ridge_feature_is_computed_but_not_in_the_vector():
    """Гребень уходит в диагностику и в мультилейбл-голову, в модель — нет."""
    f = _feat(phantom(True))
    assert f.ridge_p9999 == f.ridge_p9999
    assert f.vector().shape == (1,)


@pytest.mark.parametrize("img,mask", [
    (None, None),
    (np.zeros((0, 0), dtype=np.int32), None),
    (np.zeros((40, 40), dtype=np.int32), np.zeros((40, 40), dtype=bool)),
    (np.zeros((40, 40), dtype=np.int32), np.ones((10, 10), dtype=bool)),
])
def test_degenerate_input_gives_reason_not_crash(img, mask):
    f = features(img, mask, 0.6)
    assert not f.ok and f.reason


def test_mask_covering_everything_leaves_no_area():
    img = phantom(True)
    f = features(img, np.ones(img.shape, dtype=bool), 0.6)
    assert not f.ok and "площад" in f.reason


# --- модель ----------------------------------------------------------------- #

def fitted_model(threshold: float = 0.5) -> ArtifactModel:
    x = np.array([[0.05], [0.07], [0.09], [0.4], [0.5], [0.6]])
    y = np.array([0, 0, 0, 1, 1, 1])
    mu, sd = standardize(x)
    w, b = fit_logistic((x - mu) / sd, y)
    return ArtifactModel(mean=mu, scale=sd, coef=w, intercept=b, threshold=threshold)


def test_fit_separates_a_separable_toy_task():
    m = fitted_model()
    assert m.predict_proba(np.array([[0.05]]))[0] < 0.5 < m.predict_proba(np.array([[0.6]]))[0]


def test_fit_is_deterministic():
    a, b = fitted_model(), fitted_model()
    assert a.coef.tolist() == pytest.approx(b.coef.tolist())
    assert a.intercept == pytest.approx(b.intercept)


def test_weights_survive_a_round_trip_exactly(tmp_path):
    m = fitted_model(0.37)
    m.save(tmp_path / "m.npz")
    back = ArtifactModel.load(tmp_path / "m.npz")
    x = np.array([[0.05], [0.3], [0.9]])
    assert back.predict_proba(x).tolist() == pytest.approx(m.predict_proba(x).tolist(), abs=0.0)
    assert back.threshold == pytest.approx(0.37)


def test_stale_weights_are_rejected_loudly(tmp_path):
    """Признаки поменялись, веса остались — загрузка обязана упасть."""
    m = fitted_model()
    p = tmp_path / "old.npz"
    np.savez(p, mean=m.mean, scale=m.scale, coef=m.coef,
             intercept=np.array([m.intercept]), threshold=np.array([0.5]),
             feature_version=np.array("что-то другое"))
    with pytest.raises(ValueError, match="переобучите"):
        ArtifactModel.load(p)


def test_wrong_feature_count_is_an_error():
    with pytest.raises(ValueError):
        fitted_model().predict_proba(np.array([[0.1, 0.2]]))


def test_committed_weights_match_the_current_feature_version():
    """Веса в репозитории и код признаков не должны разъезжаться."""
    model = load_default()
    if model is None:
        pytest.skip("веса не обучены в этом окружении")
    assert model.feature_version == FEATURE_VERSION
    assert 0.0 < model.threshold < 1.0


# --- сквозное поведение ------------------------------------------------------ #

def test_missing_weights_do_not_break_the_pipeline(tmp_path):
    """Без файла модели артефакты не проверяются, но прогон продолжается."""
    img = phantom(True)
    res = evaluate(img, segment_bone(img).mask, 0.6,
                   model=None) if load_default() is None else None
    if res is None:
        from dxa_qc import artifact as mod
        saved = mod._CACHE.copy()
        mod._CACHE.clear()
        mod._CACHE[str(mod.DEFAULT_MODEL_PATH)] = None
        try:
            res = evaluate(img, segment_bone(img).mask, 0.6)
        finally:
            mod._CACHE.clear()
            mod._CACHE.update(saved)
    assert res.violations == [] and not res.ok and res.reason


def test_wire_is_flagged_and_clean_frame_is_not():
    model = load_default()
    if model is None:
        pytest.skip("веса не обучены в этом окружении")
    wired = evaluate(phantom(True), segment_bone(phantom(True)).mask, 0.6, model)
    clean = evaluate(phantom(False), segment_bone(phantom(False)).mask, 0.6, model)
    assert SPINE_ARTIFACT in wired.violations
    assert clean.violations == []
    assert wired.probability > clean.probability


def test_diagnostics_are_csv_ready():
    res = evaluate(phantom(True), segment_bone(phantom(True)).mask, 0.6)
    for k, v in res.diagnostics().items():
        assert isinstance(v, (str, float, int)), k
        assert v == v, k
