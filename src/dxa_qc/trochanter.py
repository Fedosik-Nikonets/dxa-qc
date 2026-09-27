"""Шаблонный локализатор малого вертела без зависимостей времени инференса.

Правое бедро зеркалится в каноническую ориентацию. Шаблон строится как
среднее локально нормированных патчей вокруг точек врача. Поиск выполняется
нормированной корреляцией в физическом окне вдоль уже найденного медиального
края. Сигнатуры и поведение ``medial_track`` этот модуль не меняет.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .femur import field_scale
from .medial_track import MedialTrack
from .rotation_profile import canonical_image, patch_intensity


FEATURE_VERSION = "trochanter_template_v1"
TARGET_MM_PER_PX = 0.6
PATCH_MM = 18.0
SEARCH_DOWN_MM = 60.0
SEARCH_ACROSS_MM = 15.0


@dataclass(frozen=True)
class TrainingPoint:
    image: np.ndarray
    region: str
    x: float                         # координата врача в исходном снимке
    y: float
    mm_per_px: float | None = None
    mm_per_px_source: str | None = None
    scanner_model: str | None = None


@dataclass(frozen=True)
class TrochanterModel:
    template: np.ndarray
    target_mm_per_px: float = TARGET_MM_PER_PX
    patch_mm: float = PATCH_MM
    search_down_mm: float = SEARCH_DOWN_MM
    search_across_mm: float = SEARCH_ACROSS_MM
    feature_version: str = FEATURE_VERSION

    def __post_init__(self):
        t = np.asarray(self.template)
        if t.ndim != 2 or min(t.shape, default=0) < 5 or t.shape[0] % 2 == 0 or t.shape[1] % 2 == 0:
            raise ValueError("шаблон должен быть нечётным двумерным патчем")
        if not np.isfinite(t).all() or np.std(t) < 1e-6:
            raise ValueError("пустой или постоянный шаблон")
        if self.feature_version != FEATURE_VERSION:
            raise ValueError("несовместимая версия локализатора")
        if self.target_mm_per_px <= 0 or self.patch_mm <= 0:
            raise ValueError("неверный физический размер шаблона")

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            template=np.asarray(self.template, np.float32),
            target_mm_per_px=float(self.target_mm_per_px),
            patch_mm=float(self.patch_mm),
            search_down_mm=float(self.search_down_mm),
            search_across_mm=float(self.search_across_mm),
            feature_version=self.feature_version,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TrochanterModel":
        with np.load(path, allow_pickle=False) as data:
            version = str(data["feature_version"].item())
            if version != FEATURE_VERSION:
                raise ValueError(f"локализатор '{version}', код ожидает '{FEATURE_VERSION}'")
            return cls(
                template=data["template"].astype(np.float32),
                target_mm_per_px=float(data["target_mm_per_px"]),
                patch_mm=float(data["patch_mm"]),
                search_down_mm=float(data["search_down_mm"]),
                search_across_mm=float(data["search_across_mm"]),
                feature_version=version,
            )


@dataclass(frozen=True)
class Localization:
    x: float = float("nan")          # координаты исходного снимка
    y: float = float("nan")
    confidence: float = float("nan")
    canonical_x: float = float("nan")
    window_start: float = float("nan")
    window_end: float = float("nan")
    ok: bool = False
    reason: str = ""


def _odd_size(mm: float, spacing: float) -> int:
    n = max(5, int(round(mm / spacing)))
    return n if n % 2 else n + 1


def _resample(image: np.ndarray, region: str, mm: float, target: float) -> tuple[np.ndarray, float]:
    a = canonical_image(image, region).astype(np.float32)
    scale = float(mm) / float(target)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("неверный масштаб")
    if abs(scale - 1.0) > 1e-9:
        a = cv2.resize(a, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    return a, scale


def _normalise_patch(patch: np.ndarray) -> np.ndarray | None:
    p = np.asarray(patch, np.float32)
    sd = float(p.std())
    if not np.isfinite(p).all() or sd < 1e-5:
        return None
    return (p - float(p.mean())) / sd


def fit(points: Iterable[TrainingPoint], *, target_mm_per_px: float = TARGET_MM_PER_PX,
        patch_mm: float = PATCH_MM) -> TrochanterModel:
    size = _odd_size(patch_mm, target_mm_per_px)
    half = size // 2
    patches = []
    for point in points:
        mm, _ = field_scale(point.mm_per_px, source=point.mm_per_px_source,
                            model=point.scanner_model)
        a, scale = _resample(point.image, point.region, mm, target_mm_per_px)
        x = point.x if point.region == "femur_left" else point.image.shape[1] - 1 - point.x
        x, y = int(round(x * scale)), int(round(point.y * scale))
        if x-half < 0 or y-half < 0 or x+half >= a.shape[1] or y+half >= a.shape[0]:
            continue
        patch = _normalise_patch(a[y-half:y+half+1, x-half:x+half+1])
        if patch is not None:
            patches.append(patch)
    if not patches:
        raise ValueError("нет пригодных точек врача для шаблона")
    template = np.mean(np.stack(patches), axis=0)
    template = _normalise_patch(template)
    if template is None:
        raise ValueError("усреднённый шаблон вырожден")
    return TrochanterModel(template=template, target_mm_per_px=target_mm_per_px,
                           patch_mm=patch_mm)


def _search_window(track: MedialTrack, model: TrochanterModel) -> tuple[float, float]:
    # ``restricted`` начинается на 5 мм ниже найденного перехода шейка–диафиз.
    try:
        start = float(track.candidate("restricted")["window_start"]) - 5.0 / track.mm
    except ValueError:
        start = float(max(track.ys.min(), 0.35 * track.rows))
    start = max(float(track.ys.min()), start)
    end = min(float(track.ys.max()), start + model.search_down_mm / track.mm)
    if end <= start:
        raise ValueError("пустое окно поиска малого вертела")
    return start, end


def locate(image: np.ndarray, region: str, mm_per_px: float | None, track: MedialTrack,
           model: TrochanterModel, *, mm_per_px_source: str | None = None,
           scanner_model: str | None = None) -> Localization:
    """Возвращает точку в координатах исходного снимка; отказ не бросает."""
    try:
        mm, _ = field_scale(mm_per_px, source=mm_per_px_source, model=scanner_model)
        a, scale = _resample(image, region, mm, model.target_mm_per_px)
        template = np.asarray(model.template, np.float32)
        th, tw = template.shape
        if a.shape[0] < th or a.shape[1] < tw:
            raise ValueError("снимок меньше шаблона")
        response = cv2.matchTemplate(a, template, cv2.TM_CCOEFF_NORMED)
        half_y, half_x = th // 2, tw // 2
        start, end = _search_window(track, model)
        cy = (np.arange(response.shape[0]) + half_y) / scale
        cx = (np.arange(response.shape[1]) + half_x) / scale
        allowed_y = (cy >= start) & (cy <= end)
        allowed = np.zeros(response.shape, bool)
        if allowed_y.any():
            y_native = cy[allowed_y]
            edge_x = np.interp(y_native, track.ys, track.xs)
            across = model.search_across_mm / track.mm
            allowed[allowed_y] = np.abs(cx[None, :] - edge_x[:, None]) <= across
        finite = allowed & np.isfinite(response)
        if not finite.any():
            raise ValueError("нет допустимых позиций шаблона")
        score = np.where(finite, response, -np.inf)
        top, left = np.unravel_index(int(np.argmax(score)), score.shape)
        x_can = (left + half_x) / scale
        y = (top + half_y) / scale
        x = x_can if region == "femur_left" else image.shape[1] - 1 - x_can
        return Localization(float(x), float(y), float(score[top, left]), float(x_can),
                            float(start), float(end), True, "")
    except (ValueError, cv2.error) as exc:
        return Localization(reason=str(exc))


def rotation_features(image: np.ndarray, region: str, mm_per_px: float | None,
                      track: MedialTrack, model: TrochanterModel, *,
                      mm_per_px_source: str | None = None,
                      scanner_model: str | None = None) -> dict:
    """Выступ и яркостные признаки в автоматически найденной точке."""
    loc = locate(image, region, mm_per_px, track, model,
                 mm_per_px_source=mm_per_px_source, scanner_model=scanner_model)
    if not loc.ok:
        raise ValueError(loc.reason)
    a = canonical_image(image, region)
    j = int(np.argmin(np.abs(track.ys - loc.y)))
    radius = max(1, int(round(1.2 / track.mm)))
    offset = max(1, int(round(3.0 / track.mm)))
    edge = patch_intensity(a, loc.canonical_x, loc.y, radius)
    inside = patch_intensity(a, loc.canonical_x + offset, loc.y, radius)
    outside = patch_intensity(a, loc.canonical_x - offset, loc.y, radius)
    values = (edge, inside, outside)
    if not np.isfinite(values).all():
        raise ValueError("патчи найденной точки выходят за снимок")
    return {
        "trochanter_bump_mm": float(track.local_bump_mm[j]),
        "trochanter_intensity": float(edge),
        "trochanter_contrast": float(inside - outside),
        "trochanter_score": float(loc.confidence),
        "anchor_x": float(loc.x),
        "anchor_y": float(loc.y),
        "canonical_anchor_x": float(loc.canonical_x),
        "window_start": float(loc.window_start),
        "window_end": float(loc.window_end),
        "localizer": "trochanter_v1",
    }
