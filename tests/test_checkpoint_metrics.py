"""Скрипт 15: честный порог не должен видеть метки отложенного фолда."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load():
    path = ROOT / "scripts" / "15_checkpoint_metrics.py"
    spec = importlib.util.spec_from_file_location("checkpoint_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_threshold_is_the_middle_of_the_gap():
    m = _load()
    score = np.array([0.0, 1.0, 2.0, 10.0, 12.0])
    y = np.array([0, 0, 0, 1, 1])
    assert m.fit_threshold(score, y) == 6.0


def test_threshold_tie_prefers_the_more_sensitive_rule():
    m = _load()
    score = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([1, 0, 0, 1])
    # t=-1: всё нарушение, TP=2 FP=2, F1=2/3; t=2.5: TP=1 FN=1, F1=2/3.
    # Равенство — берётся меньший порог: пропуск дороже лишнего просмотра.
    assert m.fit_threshold(score, y) == -1.0


def _frame(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 60
    f = pd.DataFrame({
        "fold": np.repeat(np.arange(5), n // 5),
        "study": [f"S{i // 2}" for i in range(n)],
    })
    for k in ("pos", "tilt", "roi"):
        f[f"y_{k}"] = rng.integers(0, 2, n).astype(float)
        f[f"s_{k}"] = f[f"y_{k}"] * 2 + rng.normal(0, 1, n)
    f["p_art"] = 0
    f["p_rot"] = 0
    return f


def test_outer_fold_labels_do_not_change_its_predictions():
    m = _load()
    base = _frame(0)
    changed = base.copy()
    held = changed.fold.eq(2)
    for k in ("pos", "tilt", "roi"):
        changed.loc[held, f"y_{k}"] = 1 - changed.loc[held, f"y_{k}"]
    a, _ = m.nested_predictions(base)
    b, _ = m.nested_predictions(changed)
    for k in ("pos", "tilt", "roi"):
        assert a.loc[held, f"n_{k}"].tolist() == b.loc[held, f"n_{k}"].tolist()
