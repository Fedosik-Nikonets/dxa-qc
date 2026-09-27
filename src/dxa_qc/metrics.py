"""Метрики качества классификации и доверительные интервалы.

Зачем отдельный модуль. Метрики по ТЗ 8.4 считаются в четырёх местах —
правила позвоночника, детектор артефактов, будущие правила бедра, разбор
ошибок 26 сентября — и считать их в каждом скрипте заново значит рано или
поздно получить две разные цифры для одного и того же. Здесь один расчёт,
покрытый тестами, включая сверку с ручными примерами.

Что внутри и почему именно это
------------------------------

*Полный набор ТЗ 8.4*: чувствительность, специфичность, сбалансированная
точность, точность, F1, macro-F1, ROC-AUC, PR-AUC. F1 — приоритетная
метрика по ТЗ, поэтому у неё же считается и подбор порога.

*macro-F1 считается по двум классам*, а не только по классу «нарушение».
При дисбалансе 17 к 82 F1 по одному классу и macro-F1 расходятся вдвое, и
показывать надо обе: первая говорит, как найдены нарушения, вторая — как
решение ведёт себя на обоих классах сразу.

*PR-AUC важнее ROC-AUC при таком дисбалансе.* ROC-AUC на 6–17 позитивах
выглядит прилично почти всегда: у него в знаменателе 80–90 негативов.
PR-AUC падает сразу, как только ложных срабатываний становится сравнимо с
позитивами. Обе цифры печатаются рядом.

*Бутстреп стратифицированный и с группировкой.* Позитивы и негативы
ресэмплятся раздельно, иначе на 6 позитивах часть выборок остаётся без
единого позитива и метрика не определена. Группировка по исследованию
(`groups`) нужна там, где на одно исследование приходится несколько
изображений: ресэмплить изображения независимо значило бы считать копии
одного и того же снимка независимыми наблюдениями и получить неправдоподобно
узкий интервал.

*ROC-AUC через ранги Манна — Уитни*, с корректной обработкой совпадений:
на признаках вроде «доля насыщенных пикселей» совпадений много, и наивный
расчёт по парам даёт смещение.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np


@dataclass
class BinaryMetrics:
    """Метрики бинарной классификации при фиксированном решении."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    sensitivity: float = float("nan")      # recall по классу «нарушение»
    specificity: float = float("nan")
    balanced_accuracy: float = float("nan")
    precision: float = float("nan")
    f1: float = float("nan")
    macro_f1: float = float("nan")
    roc_auc: float = float("nan")
    pr_auc: float = float("nan")
    n: int = 0
    n_pos: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def line(self, name: str = "") -> str:
        """Одна строка для печати в консоль и в отчёт."""
        head = f"{name:24s}" if name else ""
        s = (f"{head}TP={self.tp:3d} FP={self.fp:3d} FN={self.fn:3d} TN={self.tn:3d}  "
             f"чувств={self.sensitivity:.2f} специф={self.specificity:.3f} "
             f"F1={self.f1:.2f} macroF1={self.macro_f1:.2f}")
        if self.roc_auc == self.roc_auc:
            s += f" ROC-AUC={self.roc_auc:.3f} PR-AUC={self.pr_auc:.3f}"
        return s


def _as_int_array(v: Iterable) -> np.ndarray:
    return np.asarray(list(v) if not isinstance(v, np.ndarray) else v).astype(int).ravel()


def roc_auc(score: Sequence[float], y_true: Sequence[int]) -> float:
    """ROC-AUC через статистику Манна — Уитни, совпадения получают средний ранг."""
    s = np.asarray(score, dtype=float).ravel()
    y = _as_int_array(y_true)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if not n_pos or not n_neg:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0  # средний ранг на группу совпадений
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def pr_auc(score: Sequence[float], y_true: Sequence[int]) -> float:
    """Площадь под кривой точность-полнота (average precision)."""
    s = np.asarray(score, dtype=float).ravel()
    y = _as_int_array(y_true)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    if not (y == 1).any():
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    # One threshold includes the entire tied-score group. Counting individual
    # rows inside it made AP depend on row order, especially in bootstrap draws.
    ends = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tp = np.cumsum(y == 1)[ends]
    fp = np.cumsum(y == 0)[ends]
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int((y == 1).sum()), 1)
    # Ступенчатая сумма: прибавляем точность в точках, где растёт полнота.
    d_recall = np.diff(np.concatenate([[0.0], recall]))
    return float((precision * d_recall).sum())


def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int],
                   score: Sequence[float] | None = None) -> BinaryMetrics:
    """Полный набор метрик ТЗ 8.4. `score` нужен только для AUC."""
    y = _as_int_array(y_true)
    p = _as_int_array(y_pred)
    if y.shape != p.shape:
        raise ValueError(f"разная длина: меток {y.shape}, предсказаний {p.shape}")
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum())

    def _div(a: float, b: float) -> float:
        return a / b if b else float("nan")

    sens = _div(tp, tp + fn)
    spec = _div(tn, tn + fp)
    prec = _div(tp, tp + fp)
    f1 = _div(2 * tp, 2 * tp + fp + fn)
    # F1 по классу «норма»: тот же расчёт с обменом ролей.
    f1_neg = _div(2 * tn, 2 * tn + fn + fp)
    both = [v for v in (f1, f1_neg) if v == v]
    m = BinaryMetrics(
        tp=tp, fp=fp, fn=fn, tn=tn,
        sensitivity=sens, specificity=spec,
        balanced_accuracy=(sens + spec) / 2 if sens == sens and spec == spec else float("nan"),
        precision=prec, f1=f1,
        macro_f1=float(np.mean(both)) if both else float("nan"),
        n=int(y.size), n_pos=int((y == 1).sum()),
    )
    if score is not None:
        m.roc_auc = roc_auc(score, y)
        m.pr_auc = pr_auc(score, y)
    return m


def best_threshold(score: Sequence[float], y_true: Sequence[int],
                   grid: Sequence[float] | None = None) -> float:
    """Порог, максимизирующий F1 (приоритетная метрика ТЗ 8.4).

    При равенстве F1 берётся меньший порог: на скрининге качества дешевле
    лишний раз показать снимок врачу, чем пропустить нарушение.
    """
    s = np.asarray(score, dtype=float).ravel()
    y = _as_int_array(y_true)
    if grid is None:
        uniq = np.unique(s[np.isfinite(s)])
        grid = uniq if uniq.size <= 200 else np.quantile(uniq, np.linspace(0, 1, 200))
    best_f1, best_t = -1.0, float(grid[0]) if len(grid) else 0.5
    for t in grid:
        pred = (s >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def bootstrap_ci(y_true: Sequence[int], values: Sequence[float],
                 statistic: Callable[[np.ndarray, np.ndarray], float],
                 groups: Sequence[Any] | None = None,
                 n_bootstrap: int = 10000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Стратифицированный бутстреп-ДИ. `statistic(y, values) -> float`.

    Позитивы и негативы ресэмплятся раздельно, поэтому в каждой выборке
    сохраняется исходное число позитивов и метрика всегда определена.
    Если задан `groups`, единицей ресэмплинга становится группа
    (исследование), а не изображение.
    """
    y = _as_int_array(y_true)
    v = np.asarray(values, dtype=float).ravel()
    rng = np.random.default_rng(seed)

    if groups is None:
        units: list[np.ndarray] = [np.array([i]) for i in range(len(y))]
        unit_pos = y.astype(bool)
    else:
        g = np.asarray(list(groups))
        uniq = np.unique(g)
        units = [np.flatnonzero(g == u) for u in uniq]
        unit_pos = np.array([bool(y[idx].max()) for idx in units])

    pos_units = [units[i] for i in np.flatnonzero(unit_pos)]
    neg_units = [units[i] for i in np.flatnonzero(~unit_pos)]
    if not pos_units or not neg_units:
        return (float("nan"), float("nan"))

    out: list[float] = []
    for _ in range(n_bootstrap):
        pick = [pos_units[i] for i in rng.integers(0, len(pos_units), len(pos_units))]
        pick += [neg_units[i] for i in rng.integers(0, len(neg_units), len(neg_units))]
        idx = np.concatenate(pick)
        val = statistic(y[idx], v[idx])
        if val == val:
            out.append(float(val))
    if not out:
        return (float("nan"), float("nan"))
    arr = np.asarray(out)
    return (float(np.percentile(arr, 100 * alpha / 2)),
            float(np.percentile(arr, 100 * (1 - alpha / 2))))


def auc_ci(score: Sequence[float], y_true: Sequence[int],
           groups: Sequence[Any] | None = None, **kw) -> tuple[float, float]:
    """ДИ для ROC-AUC — самый частый случай, вынесен отдельно."""
    return bootstrap_ci(y_true, score, lambda y, s: roc_auc(s, y), groups=groups, **kw)


def f1_ci(y_pred: Sequence[int], y_true: Sequence[int],
          groups: Sequence[Any] | None = None, **kw) -> tuple[float, float]:
    """ДИ для F1 при уже принятом решении."""
    return bootstrap_ci(y_true, np.asarray(y_pred, dtype=float),
                        lambda y, p: binary_metrics(y, p.astype(int)).f1,
                        groups=groups, **kw)
