"""Метрики ТЗ 8.4: сверка с ручным расчётом, совпадения, вырожденные случаи.

Модуль считает цифры, которые пойдут в отчёт и на слайд, поэтому проверяется
не «не падает», а совпадение с посчитанным на бумаге.
"""
from __future__ import annotations

import numpy as np
import pytest

from dxa_qc.metrics import (
    auc_ci,
    best_threshold,
    binary_metrics,
    bootstrap_ci,
    f1_ci,
    pr_auc,
    roc_auc,
)


# --- ROC-AUC и PR-AUC ------------------------------------------------------- #

def test_auc_matches_hand_computed_example():
    """y=[0,0,1,1], score=[0.1,0.4,0.35,0.8]: 3 правильные пары из 4."""
    y, s = [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]
    assert roc_auc(s, y) == pytest.approx(0.75)
    assert pr_auc(s, y) == pytest.approx(0.8333333, abs=1e-6)


def test_perfect_and_inverted_separation():
    y = [0, 0, 1, 1]
    assert roc_auc([0.1, 0.2, 0.8, 0.9], y) == pytest.approx(1.0)
    assert roc_auc([0.9, 0.8, 0.2, 0.1], y) == pytest.approx(0.0)


def test_all_scores_equal_gives_half():
    """Совпадения получают средний ранг: полностью неинформативный признак — 0.5."""
    assert roc_auc([0.5] * 6, [0, 0, 0, 1, 1, 1]) == pytest.approx(0.5)


def test_partial_ties_are_counted_as_half_pairs():
    # позитив совпал с одним негативом и выше другого: 1.5 пары из 2
    assert roc_auc([0.1, 0.4, 0.4], [0, 0, 1]) == pytest.approx(0.75)


def test_average_precision_equal_scores_is_prevalence_and_order_independent():
    assert pr_auc([0.5] * 4, [1, 1, 0, 0]) == pytest.approx(0.5)
    assert pr_auc([0.5] * 4, [0, 0, 1, 1]) == pytest.approx(0.5)


def test_average_precision_partial_ties_use_a_single_threshold():
    # At 0.8: recall 1/2, precision 1/2. At 0.2: recall 1, precision 2/3.
    assert pr_auc([0.8, 0.8, 0.2], [1, 0, 1]) == pytest.approx(7 / 12)
    assert pr_auc([0.8, 0.8, 0.2], [0, 1, 1]) == pytest.approx(7 / 12)


def test_auc_is_nan_without_one_of_the_classes():
    assert roc_auc([0.1, 0.2], [0, 0]) != roc_auc([0.1, 0.2], [0, 0])
    assert pr_auc([0.1, 0.2], [0, 0]) != pr_auc([0.1, 0.2], [0, 0])


def test_nan_scores_are_dropped_not_crash():
    assert roc_auc([0.1, float("nan"), 0.9], [0, 1, 1]) == pytest.approx(1.0)


# --- матрица ошибок --------------------------------------------------------- #

def test_binary_metrics_arithmetic():
    y = [1, 1, 1, 0, 0, 0, 0, 0]
    p = [1, 1, 0, 1, 0, 0, 0, 0]
    m = binary_metrics(y, p)
    assert (m.tp, m.fp, m.fn, m.tn) == (2, 1, 1, 4)
    assert m.sensitivity == pytest.approx(2 / 3)
    assert m.specificity == pytest.approx(0.8)
    assert m.precision == pytest.approx(2 / 3)
    assert m.f1 == pytest.approx(2 / 3)
    assert m.balanced_accuracy == pytest.approx((2 / 3 + 0.8) / 2)


def test_macro_f1_differs_from_f1_under_imbalance():
    """Смысл macro-F1: он учитывает и класс «норма», где решение сильнее."""
    y = [1] * 5 + [0] * 95
    p = [1] * 3 + [0] * 2 + [0] * 95
    m = binary_metrics(y, p)
    assert m.f1 == pytest.approx(0.75)
    assert m.macro_f1 > m.f1
    assert m.macro_f1 == pytest.approx((0.75 + 2 * 95 / (2 * 95 + 2)) / 2)


def test_length_mismatch_is_an_error():
    with pytest.raises(ValueError):
        binary_metrics([1, 0], [1, 0, 1])


def test_metrics_line_is_printable():
    assert "F1=" in binary_metrics([1, 0], [1, 0], [0.9, 0.1]).line("правило")


# --- выбор порога ----------------------------------------------------------- #

def test_best_threshold_maximizes_f1():
    y = [0, 0, 0, 1, 1]
    s = [0.1, 0.2, 0.3, 0.7, 0.9]
    t = best_threshold(s, y)
    assert binary_metrics(y, (np.asarray(s) >= t).astype(int)).f1 == pytest.approx(1.0)


def test_best_threshold_prefers_the_lower_of_equal_options():
    """На скрининге лишний показ врачу дешевле пропуска."""
    y = [0, 1]
    s = [0.2, 0.8]
    assert best_threshold(s, y) <= 0.8


# --- бутстреп --------------------------------------------------------------- #

def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(20), np.zeros(80)].astype(int)
    s = np.r_[rng.normal(1.0, 1.0, 20), rng.normal(0.0, 1.0, 80)]
    lo, hi = auc_ci(s, y, n_bootstrap=500)
    assert lo < roc_auc(s, y) < hi
    assert 0.0 <= lo < hi <= 1.0


def test_grouped_bootstrap_is_wider_than_ungrouped_on_duplicates():
    """Копии одного снимка — не независимые наблюдения.

    Если ресэмплить изображения, дубликаты выглядят как новые данные и
    интервал получается неправдоподобно узким.
    """
    rng = np.random.default_rng(1)
    y = np.r_[np.ones(10), np.zeros(40)].astype(int)
    s = np.r_[rng.normal(1.0, 1.0, 10), rng.normal(0.0, 1.0, 40)]
    y5, s5 = np.repeat(y, 5), np.repeat(s, 5)
    groups = np.repeat(np.arange(len(y)), 5)
    flat = auc_ci(s5, y5, n_bootstrap=400, seed=3)
    grouped = auc_ci(s5, y5, groups=groups, n_bootstrap=400, seed=3)
    assert (grouped[1] - grouped[0]) > (flat[1] - flat[0])


def test_stratified_resampling_keeps_positives_in_every_sample():
    """На шести позитивах нестратифицированный бутстреп давал бы NaN."""
    y = np.r_[np.ones(6), np.zeros(93)].astype(int)
    s = np.r_[np.linspace(0.6, 1.0, 6), np.linspace(0.0, 0.5, 93)]
    lo, hi = auc_ci(s, y, n_bootstrap=300)
    assert lo == lo and hi == hi


def test_f1_ci_returns_a_valid_interval():
    y = np.r_[np.ones(8), np.zeros(30)].astype(int)
    p = np.r_[np.ones(6), np.zeros(2), np.zeros(28), np.ones(2)].astype(int)
    lo, hi = f1_ci(p, y, n_bootstrap=300)
    assert 0.0 <= lo <= binary_metrics(y, p).f1 <= hi <= 1.0


def test_bootstrap_without_positives_is_nan_not_crash():
    lo, hi = bootstrap_ci([0, 0, 0], [0.1, 0.2, 0.3], lambda y, s: roc_auc(s, y),
                          n_bootstrap=10)
    assert lo != lo and hi != hi
