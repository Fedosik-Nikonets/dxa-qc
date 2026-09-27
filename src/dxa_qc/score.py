"""Балл нарушения `quality_score` — непрерывная оценка к `quality_class`.

Зачем. ТЗ 8.4 называет F1 и ROC-AUC приоритетными метриками, а ROC-AUC по
бинарному `quality_class` вырождается в сбалансированную точность: у
решения нет «уверенности», только 0 и 1. Балл даёт организатору ранжирование
без изменения решений.

Как считается. У каждого правила есть маржа `m` в его собственных единицах:
положительная, когда правило сработало, и тем больше, чем дальше измерение
за порогом.

* укладка позвоночника: `COVERAGE_MIN - spine_coverage`;
* наклон оси: `|spine_angle_deg| - AXIS_TILT_DEG`;
* артефакты: `logit(artifact_prob) - logit(порог модели)`;
* охват области интереса бедра: `FIELD_MIN_MM - femur_field_h_mm`;
* ротация: `max(low - value, value - high)` — насколько признак вне нормы.

Маржа переводится в вероятность нарушения своего типа `p = sigmoid(a·m)`;
крутизна `a` одна на правило, подогнана по разметке без свободного члена
(`scripts/26_quality_score.py` → `models/quality_score_v1.json`), поэтому
порог правила переходит ровно в 0.5. Балл снимка — вероятность хотя бы
одного нарушения среди правил его области, `1 - Π(1 - p)`, зажатая так,
что **балл больше 0.5 тогда и только тогда, когда `quality_class = 1`**.
Таблица не может противоречить сама себе.

Если измерение не вычислилось (NaN), правило в балл не входит. Если файла
параметров нет, балл не выдаётся (пустая ячейка), а решения не меняются.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from .femur import FIELD_MIN_MM
from .spine import AXIS_TILT_DEG, COVERAGE_MIN

DEFAULT_PARAMS_PATH = Path(__file__).resolve().parents[2] / "models" / "quality_score_v1.json"
PARAMS_VERSION = "quality_score_v1"

# Правила по областям; ключи совпадают с ключами крутизны в json.
RULES = {
    "spine": ("pos", "tilt", "art"),
    "femur": ("roi", "rot"),
}
# Зазор вокруг 0.5 при зажатии: балл снимка с нарушением не ниже 0.5001,
# без нарушения — не выше 0.4999 (после округления до четырёх знаков).
EDGE = 1e-4


def _num(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def area_of(region_key: str) -> str | None:
    if region_key == "spine":
        return "spine"
    if region_key in ("femur_left", "femur_right"):
        return "femur"
    return None


def margins(region_key: str, diag: Mapping[str, Any],
            artifact_threshold: float | None = None) -> dict[str, float]:
    """Маржи правил области по диагностике кадра; NaN — измерения нет."""
    area = area_of(region_key)
    out: dict[str, float] = {}
    if area == "spine":
        out["pos"] = COVERAGE_MIN - _num(diag.get("spine_coverage"))
        out["tilt"] = abs(_num(diag.get("spine_angle_deg"))) - AXIS_TILT_DEG
        prob = _num(diag.get("artifact_prob"))
        if artifact_threshold is None or prob != prob:
            out["art"] = float("nan")
        else:
            out["art"] = _logit(prob) - _logit(artifact_threshold)
    elif area == "femur":
        out["roi"] = FIELD_MIN_MM - _num(diag.get("femur_field_h_mm"))
        v = _num(diag.get("rotation_value"))
        lo = _num(diag.get("rotation_low"))
        hi = _num(diag.get("rotation_high"))
        if v != v or lo != lo or hi != hi:
            out["rot"] = float("nan")
        else:
            out["rot"] = max(lo - v, v - hi)
    return out


@lru_cache(maxsize=4)
def _load(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if d.get("version") != PARAMS_VERSION:
        return None
    return d


def load_params(path: str | Path | None = None) -> dict | None:
    """Параметры балла или None, если файла нет или он другой версии."""
    return _load(str(path or DEFAULT_PARAMS_PATH))


def combine(probs: list[float], quality_class: int) -> float:
    """noisy-OR вероятностей правил, зажатый по решению."""
    keep = 1.0
    for p in probs:
        keep *= 1.0 - p
    s = 1.0 - keep
    if quality_class == 1:
        s = max(s, 0.5 + EDGE)
    else:
        s = min(s, 0.5 - EDGE)
    return round(s, 4)


def quality_score(region_key: str, diag: Mapping[str, Any], quality_class: int | None,
                  params: dict | None = None,
                  artifact_threshold: float | None = None) -> float | None:
    """Балл снимка в [0, 1]; None — для отказа обработки или без параметров."""
    if quality_class is None or area_of(region_key) is None:
        return None
    if params is None:
        params = load_params()
    if params is None:
        return None
    slopes = params.get("slopes", {})
    probs = []
    for k, m in margins(region_key, diag, artifact_threshold).items():
        a = slopes.get(k)
        if a is None or m != m:
            continue
        z = max(min(float(a) * m, 60.0), -60.0)
        probs.append(1.0 / (1.0 + math.exp(-z)))
    return combine(probs, int(quality_class))


def study_score(scores: list[float | None], study_class: int) -> float | None:
    """Балл исследования: вероятность нарушения хотя бы на одном снимке."""
    got = [s for s in scores if s is not None]
    if not got:
        return None
    return combine(got, int(study_class))
