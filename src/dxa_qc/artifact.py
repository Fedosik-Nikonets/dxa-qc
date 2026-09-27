"""Детектор инородных тел и артефактов на снимке позвоночника (ТЗ 2.3.1, рис. 3).

Почему не CNN
-------------
В плане на 17 сентября стоял efficientnet-b0 с 5 фолдами, focal loss и
Grad-CAM. На 17 позитивах это не обучается: проверено прямым сравнением на
тех же фолдах — 144 признака HOG плюс логистическая регрессия дают
OOF ROC-AUC 0.59-0.62 и F1 0.17-0.22, то есть немногим лучше монетки, и
причина не в HOG, а в числе параметров на один позитив. CNN с миллионами
параметров положение не улучшит, а на техэкспертизе добавит torch в образ,
недетерминированность и веса, которые нечем проверить.

Вместо этого — один физический признак и логистическая регрессия на нём.

Что видно на снимке
-------------------
Металл, застёжки, провода и наложения дают **мелкие очень яркие структуры
вне позвоночника**: тонкие дуги в верхних углах кадра, клипсы, отрезки
проводов. Мягкие ткани и рёбра такого контраста на малом масштабе не дают —
они широкие и размытые.

Отсюда признак: **99.9-й перцентиль отклика top-hat вне маски
позвоночника**, нормированный на собственный контраст кадра.
Top-hat (яркое минус его морфологическое открытие) отвечает ровно на
«объект ярче окружения и мельче ядра»; ядро 5.4 мм — примерно толщина
провода или клипсы. Перцентиль 99.9, а не максимум: максимум ловит
единичный горячий пиксель матрицы.

Калибровка ядра — плато, а не удачное число: 4.2 / 5.4 / 6.6 мм дают
OOF F1 0.70 / 0.72 / 0.67, выбрана середина. Перцентиль при этом важен:
99-й и 99.5-й проваливаются до 0.39-0.53, потому что артефакт занимает
доли процента кадра.

Второй признак — отклик на светлый гребень (наименьшее собственное число
гессиана, сигма 2 пикселя) — в модель **не** входит: сам по себе он даёт
ROC-AUC выше (0.879 против 0.841), но порог по нему неустойчив между
фолдами и F1 падает с 0.72 до 0.59. Он считается и уходит в диагностику:
это готовый второй вход мультилейбл-головы 22-23 сентября.

Модель
------
Логистическая регрессия на одном стандартизованном признаке, веса классов
сбалансированы, L2 = 10. Обучение — в этом же модуле на numpy, без sklearn:
на инференсе остаются четыре числа (среднее, масштаб, коэффициент, сдвиг)
плюс порог, и контейнеру не нужен ни sklearn, ни torch. Веса лежат в
`models/artifact_tophat_v1.npz` и проверяются по `FEATURE_VERSION`: если
признаки изменились, а веса остались старыми, загрузка падает, а не считает
молча ерунду.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Идентификатор нарушения из taxonomy.yaml, раздел labels.spine.
SPINE_ARTIFACT = "SPINE_ARTIFACT"

FEATURE_VERSION = "spine_tophat_5.4mm_p999_v1"
FEATURE_NAMES = ("tophat_p999",)

TOPHAT_MM = 5.4        # диаметр ядра top-hat: провод, клипса, застёжка
SPINE_PAD_MM = 8.0     # насколько расширяется маска позвоночника перед вырезанием
PERCENTILE = 99.9      # перцентиль отклика вне позвоночника
RIDGE_SIGMA_PX = 2.0   # масштаб гребневого отклика (только диагностика)
DEFAULT_MM_PER_PX = 0.6

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "artifact_tophat_v1.npz"


@dataclass
class ArtifactFeatures:
    """Признаки одного кадра. `ok=False` — признаки не вычислимы."""

    tophat_p999: float = float("nan")
    ridge_p9999: float = float("nan")   # в модель не входит, только диагностика
    ok: bool = False
    reason: str | None = None

    def vector(self) -> np.ndarray:
        return np.array([self.tophat_p999], dtype=float)


@dataclass
class ArtifactFindings:
    violations: list[str] = field(default_factory=list)
    probability: float = float("nan")
    features: ArtifactFeatures = field(default_factory=ArtifactFeatures)
    ok: bool = False
    reason: str | None = None

    def diagnostics(self) -> dict[str, Any]:
        def r(x, n=4):
            return "" if x != x else round(float(x), n)
        return {
            "artifact_prob": r(self.probability),
            "artifact_tophat": r(self.features.tophat_p999),
            "artifact_ridge": r(self.features.ridge_p9999),
            "artifact_reason": self.reason or "",
        }


def _odd(n: int) -> int:
    return int(n) | 1


def ridge_response(norm: np.ndarray, sigma: float = RIDGE_SIGMA_PX) -> np.ndarray:
    """Отклик на светлую тонкую линию: минус наименьшее собственное число гессиана."""
    g = cv2.GaussianBlur(norm, (0, 0), sigma)
    gxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
    gyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
    gxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)
    disc = np.sqrt(np.maximum((gxx - gyy) ** 2 + 4 * gxy ** 2, 0.0))
    return -0.5 * (gxx + gyy - disc) * sigma ** 2


def _response(a: np.ndarray, mask: np.ndarray, mm_per_px: float | None
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Нормированный кадр, область вне позвоночника и отклик top-hat.

    Одна функция и для признака, и для overlay-серии: карта, которую видит
    врач, — ровно та, по которой принято решение.
    """
    mm = float(mm_per_px) if mm_per_px and mm_per_px == mm_per_px else DEFAULT_MM_PER_PX
    base = float(np.percentile(a, 1))
    bone = float(a[mask].mean())
    norm = (a - base) / max(bone - base, 1e-6)

    pad = max(_odd(round(SPINE_PAD_MM / mm)), 3)
    outside = ~cv2.dilate(mask.astype(np.uint8), np.ones((pad, pad), np.uint8)).astype(bool)

    ks = max(_odd(round(TOPHAT_MM / mm)), 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    tophat = cv2.morphologyEx(norm, cv2.MORPH_TOPHAT, kernel)
    return norm, outside, tophat


def tophat_map(image: np.ndarray, bone_mask: np.ndarray,
               mm_per_px: float | None = None) -> np.ndarray | None:
    """Карта отклика top-hat вне позвоночника (внутри — нули); None — не вычислима."""
    a = np.asarray(image, dtype=np.float32) if image is not None else None
    mask = np.asarray(bone_mask, dtype=bool) if bone_mask is not None else None
    if a is None or a.ndim != 2 or mask is None or mask.shape != a.shape or not mask.any():
        return None
    _, outside, tophat = _response(a, mask, mm_per_px)
    return np.where(outside, tophat, 0.0).astype(np.float32)


def features(image: np.ndarray, bone_mask: np.ndarray,
             mm_per_px: float | None = None) -> ArtifactFeatures:
    """Признаки артефакта вне маски позвоночника.

    Нормировка — на собственный контраст кадра: 1.0 означает «ярче фона
    ровно настолько же, насколько кость». Поэтому признак не зависит от
    настроек аппарата и разрядности.
    """
    res = ArtifactFeatures()
    a = np.asarray(image, dtype=np.float32) if image is not None else None
    if a is None or a.ndim != 2 or a.size == 0:
        res.reason = "нет двумерного изображения"
        return res
    mask = np.asarray(bone_mask, dtype=bool) if bone_mask is not None else None
    if mask is None or mask.shape != a.shape or not mask.any():
        res.reason = "маска позвоночника пуста"
        return res

    norm, outside, tophat = _response(a, mask, mm_per_px)
    if outside.sum() < 100:
        res.reason = "вне позвоночника не осталось площади"
        return res

    res.tophat_p999 = float(np.percentile(tophat[outside], PERCENTILE))
    res.ridge_p9999 = float(np.percentile(ridge_response(norm)[outside], 99.99))
    res.ok = True
    return res


@dataclass(frozen=True)
class ArtifactModel:
    """Переносимые веса: стандартизация, коэффициенты, порог решения."""

    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    threshold: float
    feature_version: str = FEATURE_VERSION

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        if x.shape[1] != self.coef.size:
            raise ValueError(f"ожидалось {self.coef.size} признаков, получено {x.shape[1]}")
        z = ((x - self.mean) / self.scale) @ self.coef + self.intercept
        return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))

    def decide(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x) >= self.threshold).astype(int)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean, scale=self.scale, coef=self.coef,
                 intercept=np.array([self.intercept]),
                 threshold=np.array([self.threshold]),
                 feature_version=np.array(self.feature_version))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ArtifactModel":
        with np.load(path, allow_pickle=False) as d:
            version = str(d["feature_version"].item())
            if version != FEATURE_VERSION:
                raise ValueError(
                    f"веса посчитаны на признаках '{version}', а код считает "
                    f"'{FEATURE_VERSION}' — переобучите модель")
            return cls(mean=d["mean"].astype(float), scale=d["scale"].astype(float),
                       coef=d["coef"].astype(float), intercept=float(d["intercept"][0]),
                       threshold=float(d["threshold"][0]), feature_version=version)


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 10.0,
                 iters: int = 3000, lr: float = 0.15) -> tuple[np.ndarray, float]:
    """Логистическая регрессия со сбалансированными весами классов.

    Полный градиентный спуск, фиксированное число шагов: при 99 объектах
    это доли секунды, зато результат детерминирован побитово и не зависит
    от версии sklearn. Веса классов обязательны — позитивов 17 из 99.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n_pos, n_neg = max(y.sum(), 1.0), max((1 - y).sum(), 1.0)
    sw = np.where(y == 1, len(y) / (2 * n_pos), len(y) / (2 * n_neg))
    w = np.zeros(x.shape[1])
    b = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(x @ w + b, -60, 60)))
        w -= lr * ((sw * (p - y)) @ x / len(y) + l2 * w / len(y))
        b -= lr * float((sw * (p - y)).mean())
    return w, float(b)


def standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    return x.mean(axis=0), x.std(axis=0) + 1e-9


_CACHE: dict[str, ArtifactModel | None] = {}


def load_default(path: str | Path | None = None) -> ArtifactModel | None:
    """Веса по умолчанию; None, если файла нет.

    Отсутствие весов — не ошибка: конвейер обязан работать и без них
    (тогда артефакты просто не проверяются), иначе любой прогон на машине
    без обученной модели падал бы целиком.
    """
    key = str(path or DEFAULT_MODEL_PATH)
    if key not in _CACHE:
        p = Path(key)
        try:
            _CACHE[key] = ArtifactModel.load(p) if p.exists() else None
        except Exception:  # noqa: BLE001 — битые веса не должны ронять прогон
            _CACHE[key] = None
    return _CACHE[key]


def evaluate(image: np.ndarray, bone_mask: np.ndarray,
             mm_per_px: float | None = None,
             model: ArtifactModel | None = None) -> ArtifactFindings:
    """Проверяет кадр на инородные тела. Без весов возвращает пустой вердикт."""
    res = ArtifactFindings()
    f = features(image, bone_mask, mm_per_px)
    res.features = f
    if not f.ok:
        res.reason = f.reason
        return res
    if model is None:
        model = load_default()
    if model is None:
        res.reason = "веса детектора артефактов не найдены"
        return res
    res.ok = True
    res.probability = float(model.predict_proba(f.vector())[0])
    if res.probability >= model.threshold:
        res.violations.append(SPINE_ARTIFACT)
    return res
