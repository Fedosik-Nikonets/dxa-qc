"""Правило ротации бедра (ТЗ 2.3.2, рис. 4–5) — с 23 сентября в конвейере.

Что считается
-------------
Медиальный край бедра прослеживается от диафиза вверх (`medial_track`),
в окне ниже шейки ищется выступ малого вертела, и по нему считаются
признаки `rotation_profile.profile_features`. Решение — **два хвоста** по
одному признаку: нарушение, если значение ниже `low` или выше `high`.
Два хвоста, а не порог, потому что связь U-образная: при переротации
малый вертел прячется за диафиз, при недоротации выступает сильнее нормы
(день 6, по человеческим точкам — `reports/day6_rotation_landmarks.md`).

Какой признак и какие границы — не выбирается здесь. Их выбирает и
замораживает `scripts/17_fit_rotation_rule.py` тем же правилом, что и во
вложенной проверке напарника (максимум F1 при специфичности не ниже 0.75
на пяти фолдах), и пишет в `models/rotation_profile_v1.json`.

Качество, которое можно обещать
-------------------------------
Вложенная проверка всей процедуры (выбор признака и границ внутри
обучающих фолдов): F1 **0.48** [0.34; 0.61], специфичность 0.75,
20 TP / 28 FP / 16 FN — `reports/rotation_human_profile/summary.json`.
Это ниже планки handoff 0.49 на своей метке, но включено сознательно:
без правила `quality_class` бедра (F1 0.22) хуже тривиальной базы
«всё — нарушение» (0.43), с ним — 0.53. Обоснование —
`reports/day8_multilabel.md`.

Отказ
-----
Если край не прослеживается или профиль измерен меньше чем на 80% окна,
правило **не ставит нарушение** и пишет причину в диагностику. Так же
вложенная проверка считала отказ (1 из 150). Без файла модели правило не
применяется вовсе, как детектор артефактов без весов.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .medial_track import track_medial_edge
from .rotation_profile import profile_features

FEMUR_POSITIONING_ROTATION = "FEMUR_POSITIONING_ROTATION"

# Меняется при любом изменении `medial_track` или `rotation_profile`,
# которое сдвигает значения признаков: старые границы к ним неприменимы.
FEATURE_VERSION = "medial_track_restricted_profile_v1"
FEATURES = ("bump_mm", "anchor_intensity", "profile_intensity", "profile_contrast")
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "rotation_profile_v1.json"


@dataclass(frozen=True)
class RotationModel:
    """Замороженное правило: признак и нормальный интервал [low, high]."""

    feature: str
    low: float
    high: float
    feature_version: str = FEATURE_VERSION
    meta: dict = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self):
        if self.feature not in FEATURES:
            raise ValueError(f"неизвестный признак ротации: {self.feature}")
        if not self.low <= self.high:
            raise ValueError("нижняя граница выше верхней")

    def decide(self, value: float) -> int:
        """1 — нарушение, 0 — норма, -1 — отказ (значение не измерено)."""
        if value is None or not np.isfinite(value):
            return -1
        return int(value < self.low or value > self.high)

    def to_json(self) -> dict:
        def num(x):  # JSON не знает бесконечностей, а интервал может быть открытым
            return None if not np.isfinite(x) else float(x)
        return {"feature_version": self.feature_version, "feature": self.feature,
                "low": num(self.low), "high": num(self.high), **self.meta}

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "RotationModel":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        version = d.get("feature_version")
        if version != FEATURE_VERSION:
            raise ValueError(f"границы посчитаны на признаках '{version}', а код считает "
                             f"'{FEATURE_VERSION}' — перезапустите scripts/17_fit_rotation_rule.py")
        low = -np.inf if d["low"] is None else float(d["low"])
        high = np.inf if d["high"] is None else float(d["high"])
        meta = {k: v for k, v in d.items() if k not in ("feature_version", "feature", "low", "high")}
        return cls(feature=d["feature"], low=low, high=high, feature_version=version, meta=meta)


@lru_cache(maxsize=4)
def _load_cached(path: str) -> RotationModel | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return RotationModel.load(p)
    except Exception:
        # Неисправная или несовместимая модель не должна ронять /health и
        # пакетную обработку: правило ротации в этом случае пропускается.
        return None


def load_default(path: str | Path | None = None) -> RotationModel | None:
    return _load_cached(str(path or DEFAULT_MODEL_PATH))


@dataclass
class RotationFindings:
    violations: list[str] = field(default_factory=list)
    feature: str = ""
    value: float = float("nan")
    low: float = float("nan")
    high: float = float("nan")
    measures: dict = field(default_factory=dict)
    ok: bool = False
    reason: str | None = None

    def diagnostics(self) -> dict[str, Any]:
        def r(x, n=4):
            return "" if x is None or x != x or not np.isfinite(x) else round(float(x), n)
        return {
            "rotation_feature": self.feature,
            "rotation_value": r(self.value),
            "rotation_low": r(self.low),
            "rotation_high": r(self.high),
            "rotation_anchor_y": r(self.measures.get("anchor_y"), 1),
            "rotation_reason": self.reason or "",
        }


def measure(image: np.ndarray, region: str, mm_per_px: float | None = None, *,
            mm_per_px_source: str | None = None, scanner_model: str | None = None) -> dict:
    """Признаки профиля медиального края. ValueError — измерить нельзя.

    Маска бедра считается внутри трекера по тому же пути, что и в
    `scripts/14_rotation_human_profile.py`, где признаки для границ и
    посчитаны. Проверено на всех 150 изображениях: расхождение не больше
    6e-15 (округление при записи CSV), решения совпадают все, ближайшее
    значение отстоит от границы на 1e-3.
    """
    track = track_medial_edge(image, region, mm_per_px,
                              mm_per_px_source=mm_per_px_source, model=scanner_model)
    out = profile_features(image, region, track)
    out["fit_rms_mm"] = track.fit_rms_mm
    return out


def evaluate(image: np.ndarray, region: str, mm_per_px: float | None = None, *,
             mm_per_px_source: str | None = None, scanner_model: str | None = None,
             model: RotationModel | None = None) -> RotationFindings:
    """Проверяет ротацию. Исключений не выбрасывает: отказ — это `reason`."""
    res = RotationFindings()
    if model is None:
        model = load_default()
    if model is None:
        res.reason = "границы правила ротации не найдены (models/rotation_profile_v1.json)"
        return res
    res.feature, res.low, res.high = model.feature, model.low, model.high
    try:
        res.measures = measure(image, region, mm_per_px, mm_per_px_source=mm_per_px_source,
                               scanner_model=scanner_model)
    except ValueError as exc:
        res.reason = f"отказ: {exc}"
        return res
    res.value = float(res.measures.get(model.feature, float("nan")))
    decision = model.decide(res.value)
    if decision < 0:
        res.reason = "отказ: признак не измерен"
        return res
    res.ok = True
    if decision:
        res.violations.append(FEMUR_POSITIONING_ROTATION)
    return res
