"""Мультилейбл-голова (scripts/16_multilabel_head.py): утечки и парный бутстреп."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("mlh", ROOT / "scripts/16_multilabel_head.py")
mlh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mlh)


def _data(n=60, seed=0):
    rng = np.random.default_rng(seed)
    G = rng.normal(size=(n, 4))
    G[rng.random(size=G.shape) < 0.05] = np.nan
    I = rng.normal(size=(n, 40))
    y = (np.nan_to_num(G[:, 0]) + 0.5 * rng.normal(size=n) > 0.7).astype(int)
    return G, I, y


def test_head_score_does_not_depend_on_other_test_rows():
    """Медианы, масштабы и PCA подгоняются только на обучении: оценка строки
    не должна меняться от того, какие ещё строки попали в тест."""
    G, I, y = _data()
    m = mlh.Head("geom+img", 0.3).fit(G[:40], I[:40], y[:40])
    full = m.score(G[40:], I[40:])
    one_by_one = np.array([m.score(G[i:i + 1], I[i:i + 1])[0] for i in range(40, 60)])
    assert np.allclose(full, one_by_one)


def test_head_is_deterministic():
    G, I, y = _data()
    a = mlh.Head("geom+img", 1.0).fit(G, I, y).score(G, I)
    b = mlh.Head("geom+img", 1.0).fit(G, I, y).score(G, I)
    assert np.array_equal(a, b)


def test_nested_head_never_uses_outer_fold_for_selection():
    """Если перемешать метки внешнего фолда, выбор C и порога на нём не меняется."""
    G, I, y = _data(n=100, seed=1)
    folds = np.arange(100) % 5
    _, _, sel_a = mlh.nested_head(G, I, y, folds, "geom")
    y2 = y.copy()
    y2[folds == 0] = 1 - y2[folds == 0]
    _, _, sel_b = mlh.nested_head(G, I, y2, folds, "geom")
    assert sel_a[0] == sel_b[0]


def test_paired_delta_of_identical_systems_is_exactly_zero():
    y = np.array([1, 0, 1, 0, 1, 0, 0, 1])
    p = np.array([1, 0, 0, 1, 1, 0, 0, 1])
    g = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    assert mlh.paired_delta(y, p, p, g, n_boot=200) == (0.0, 0.0, 0.0)


def test_paired_delta_sign():
    y = np.array([1, 1, 0, 0, 1, 0])
    perfect, empty = y.copy(), np.zeros(6, dtype=int)
    d, lo, hi = mlh.paired_delta(y, perfect, empty, np.arange(6), n_boot=200)
    assert d == 1.0 and lo > 0


def test_descriptor_mirrors_right_femur():
    a = np.zeros((48, 48), dtype=np.float32)
    a[:, :12] = 1.0  # яркая полоса слева
    left = mlh.image_descriptor(a, "femur_left")
    right = mlh.image_descriptor(a[:, ::-1], "femur_right")
    assert np.allclose(left, right)
    assert left.shape == (mlh.IMG_SIDE ** 2 + mlh.N_HIST,)
