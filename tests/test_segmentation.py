"""Тесты сегментации кости.

Проверяется не «красота» маски, а те свойства, на которых стоят правила ТЗ:
маска попадает на плотную структуру, угол главной оси совпадает с реальным
наклоном, вырожденный вход не роняет процесс.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dxa_qc.segmentation import (  # noqa: E402
    _otsu3,
    overlay,
    principal_axis,
    segment_bone,
)


def _phantom(rows=120, cols=90, bone=0.9, soft=0.30, air=0.05, tilt_deg=0.0,
             blur=True):
    """Три уровня яркости: воздух, мягкие ткани, кость — как на денситометрии.

    Границы размываются: на реальном снимке переход кость/ткань занимает
    несколько пикселей, и алгоритм на это рассчитывает.
    """
    img = np.full((rows, cols), air, dtype=np.float32)
    img[rows // 10 : -rows // 10, cols // 6 : -cols // 6] = soft  # тело
    rr, cc = np.mgrid[0:rows, 0:cols]
    r0, c0 = rows / 2, cols / 2
    shift = np.tan(np.radians(tilt_deg)) * (rr - r0)
    band = np.abs(cc - c0 - shift) < cols * 0.10
    band &= (rr > rows * 0.15) & (rr < rows * 0.85)
    img[band] = bone
    if blur:
        import cv2

        img = cv2.GaussianBlur(img, (5, 5), 1.2)
    return img


def test_bone_mask_hits_the_dense_structure():
    img = _phantom()
    res = segment_bone(img)
    assert res.ok, res.reason
    truth = img >= 0.85
    inter = (res.mask & truth).sum()
    assert inter / truth.sum() > 0.9                       # кость покрыта
    assert res.area_px < truth.sum() * 1.6                 # без расползания на ткани


def test_three_class_otsu_separates_air_tissue_bone():
    img = _phantom()
    u8 = (img * 255).astype(np.uint8)
    t_air, t_bone = _otsu3(u8.ravel())
    # порог обязан лечь между уровнями классов, а не внутри класса
    assert 0.05 * 255 <= t_air < 0.30 * 255     # между воздухом и мягкими тканями
    assert 0.30 * 255 <= t_bone < 0.90 * 255    # между мягкими тканями и костью


@pytest.mark.parametrize("tilt", [-12.0, -5.0, 0.0, 5.0, 12.0])
def test_principal_axis_recovers_tilt(tilt):
    """Основная метрика правила «наклон оси более 5°» — угол главной оси."""
    res = segment_bone(_phantom(tilt_deg=tilt))
    assert res.ok, res.reason
    angle, _ = principal_axis(res.mask)
    assert abs(angle - tilt) < 2.0, f"ожидали {tilt}, получили {angle}"


def test_bbox_and_margins_are_consistent():
    img = _phantom()
    res = segment_bone(img)
    r0, c0, r1, c1 = res.bbox
    rr, cc = np.nonzero(res.mask)
    assert (r0, c0, r1, c1) == (rr.min(), cc.min(), rr.max() + 1, cc.max() + 1)
    assert 0.0 < res.fill_ratio <= 1.0
    assert not any(res.touches_border.values())  # фантом не касается краёв


def test_saturated_implant_does_not_empty_the_mask():
    """Эндопротез насыщает шкалу: в наборе два таких исследования.

    Перцентильный пол порога уезжает в максимум, и наивная реализация
    возвращает пустую маску. Регрессия на этот случай.
    """
    img = _phantom()
    img[40:110, 30:60] = 1.0  # металл, больше 30% площади тела
    res = segment_bone(img)
    assert res.ok, res.reason
    assert res.area_px > 0
    assert res.mask[70, 45]


def test_mask_survives_inverted_contrast():
    """Полярность уже снята загрузчиком, но маска не должна зависеть от сдвига."""
    img = _phantom()
    a = segment_bone(img)
    b = segment_bone(img * 0.5 + 0.25)  # сжатый динамический диапазон
    assert a.ok and b.ok
    iou = (a.mask & b.mask).sum() / max((a.mask | b.mask).sum(), 1)
    assert iou > 0.95


# --------------------------------------------------------------------------- #
# вырожденные входы: ok=False и reason, без исключений
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "arr",
    [
        np.zeros((64, 64), dtype=np.float32),
        np.ones((64, 64), dtype=np.float32),   # однородный кадр — контраста нет
        np.zeros((0, 0), dtype=np.float32),
        np.zeros((3, 3), dtype=np.float32),
        np.zeros((8, 8, 3), dtype=np.float32),
    ],
)
def test_degenerate_inputs_return_reason(arr):
    res = segment_bone(arr)
    assert res.ok is False
    assert res.reason


def test_stats_are_serializable():
    res = segment_bone(_phantom())
    st = res.stats()
    assert st["seg_ok"] is True
    assert st["bone_area_px"] > 0
    assert set(st) >= {"bbox_r0", "bbox_c1", "fill_ratio", "n_kept", "largest_share"}


def test_overlay_returns_rgb_of_same_size():
    img = _phantom()
    res = segment_bone(img)
    rgb = overlay(img, res.mask)
    assert rgb.shape == img.shape + (3,)
    assert rgb.dtype == np.uint8
    # контур нарисован
    assert (rgb[..., 1] > rgb[..., 0]).any()


def test_overlay_handles_empty_mask():
    img = _phantom()
    rgb = overlay(img, np.zeros_like(img, dtype=bool))
    assert rgb.shape == img.shape + (3,)
