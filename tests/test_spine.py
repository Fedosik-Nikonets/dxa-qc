"""Геометрические правила позвоночника: пороги, краевые случаи, нормировка.

Правила из ТЗ 2.3.1 считаются из маски кости, поэтому проверять их можно на
синтетике: столб с заданным наклоном и «таз» по бокам внизу воспроизводят
ровно то, на что смотрит правило. Калибровка порогов сделана на реальных
данных (reports/day4_findings.md), здесь проверяется, что реализация делает
то, что заявлено, и не падает на вырожденном входе.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from dxa_qc.segmentation import segment_bone
from dxa_qc.spine import (
    AXIS_TILT_DEG,
    COVERAGE_MIN,
    SPINE_AXIS_TILT,
    SPINE_POSITIONING,
    curvature,
    evaluate,
    lateral_coverage,
)


def phantom(angle_deg: float = 0.0, pelvis: bool = True,
            rows: int = 300, cols: int = 300, peak: int = 220,
            bar_half: int = 20) -> np.ndarray:
    """Столб под заданным углом; при pelvis — плотные структуры по бокам внизу."""
    a = np.full((rows, cols), int(peak * 0.08), dtype=np.float32)
    a[:, cols // 2 - bar_half: cols // 2 + bar_half] = peak
    if angle_deg:
        # знак: положительный угол наклоняет верх столба вправо — как в principal_axis
        m = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle_deg, 1.0)
        a = cv2.warpAffine(a, m, (cols, rows), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
    if pelvis:
        lo = int(rows * 0.78)
        a[lo:, : cols // 2 - 60] = int(peak * 0.35)
        a[lo:, cols // 2 + 60:] = int(peak * 0.35)
    return a.astype(np.int32)


def curved_phantom(amp: float = 14.0, rows: int = 300, cols: int = 300,
                   peak: int = 220) -> np.ndarray:
    """Столб-дуга: центральная линия — половина синусоиды (модель сколиоза)."""
    a = np.full((rows, cols), int(peak * 0.08), dtype=np.int32)
    for r in range(rows):
        c = int(cols / 2 + amp * np.sin(np.pi * r / rows))
        a[r, c - 20: c + 20] = peak
    return a


# --- правило наклона оси ---------------------------------------------------- #

@pytest.mark.parametrize("angle", [0.0, 2.0, -3.0])
def test_small_tilt_is_not_a_violation(angle):
    f = evaluate(phantom(angle))
    assert f.ok
    assert SPINE_AXIS_TILT not in f.violations
    assert abs(f.axis_angle_deg - angle) < 1.0


@pytest.mark.parametrize("angle", [8.0, -8.0, 15.0])
def test_large_tilt_is_a_violation(angle):
    f = evaluate(phantom(angle))
    assert SPINE_AXIS_TILT in f.violations
    assert f.quality_class == 1
    assert np.sign(f.axis_angle_deg) == np.sign(angle)


def test_threshold_is_the_calibrated_one_not_the_literal_five():
    """Порог — 4.5°, а не буквальные 5° из ТЗ: см. docstring модуля."""
    assert AXIS_TILT_DEG == pytest.approx(4.5)
    assert SPINE_AXIS_TILT in evaluate(phantom(4.8)).violations
    assert SPINE_AXIS_TILT not in evaluate(phantom(4.0)).violations


# --- правило укладки -------------------------------------------------------- #

def test_frame_without_iliac_crests_is_a_positioning_violation():
    f = evaluate(phantom(0.0, pelvis=False))
    assert SPINE_POSITIONING in f.violations
    assert f.coverage < COVERAGE_MIN


def test_frame_with_iliac_crests_is_clean():
    f = evaluate(phantom(0.0, pelvis=True))
    assert f.violations == []
    assert f.coverage > COVERAGE_MIN


def test_coverage_does_not_depend_on_overall_brightness():
    """Нормировка на собственный контраст кадра: настройки аппарата не влияют."""
    base = phantom(0.0, pelvis=True)
    dim = (base * 0.4).astype(np.int32)
    bright = np.clip(base.astype(np.int64) * 2, 0, 4095).astype(np.int32)
    c0 = lateral_coverage(base, segment_bone(base).mask)
    assert lateral_coverage(dim, segment_bone(dim).mask) == pytest.approx(c0, rel=0.25)
    assert lateral_coverage(bright, segment_bone(bright).mask) == pytest.approx(c0, rel=0.25)


def test_coverage_is_nan_when_there_is_no_lateral_area():
    """Маска во весь кадр: судить не по чему, правило молчит."""
    full = np.full((200, 200), 200, dtype=np.int32)
    full[::2, ::2] = 150            # немного контраста, иначе кадр однороден
    cov = lateral_coverage(full, np.ones((200, 200), dtype=bool))
    assert np.isnan(cov)


def test_nan_coverage_never_becomes_a_violation():
    f = evaluate(np.zeros((40, 40), dtype=np.int32))
    assert f.violations == []


# --- признаки кривизны ------------------------------------------------------ #

def test_curved_column_has_larger_residual_than_tilted_straight_one():
    """Сколиоз против косой укладки: у дуги остаток от прямой больше."""
    arc = curvature(segment_bone(curved_phantom()).mask)
    straight = curvature(segment_bone(phantom(8.0, pelvis=False)).mask)
    assert arc["res_rms"] > straight["res_rms"]
    assert arc["res_amp"] > straight["res_amp"]


def test_curvature_on_short_mask_is_nan_not_crash():
    small = np.zeros((10, 10), dtype=bool)
    small[2:5, 4:6] = True
    assert all(v != v for v in curvature(small).values())


# --- краевые случаи --------------------------------------------------------- #

@pytest.mark.parametrize("img", [
    np.zeros((0, 0), dtype=np.int32),
    np.zeros((1, 1), dtype=np.int32),
    np.zeros((300, 300), dtype=np.int32),
    np.full((300, 300), 255, dtype=np.int32),
])
def test_degenerate_input_gives_no_violations_and_a_reason(img):
    f = evaluate(img)
    assert f.violations == []
    assert f.quality_class == 0
    if not f.ok:
        assert f.reason


def test_none_image_is_safe():
    f = evaluate(None)
    assert not f.ok and f.violations == [] and f.reason


def test_diagnostics_have_no_nan_and_are_csv_ready():
    """NaN в csv превращается в строку 'nan'; наружу уходит пустая ячейка."""
    for img in (phantom(0.0), np.zeros((40, 40), dtype=np.int32)):
        for k, v in evaluate(img).diagnostics().items():
            assert isinstance(v, (str, float, int)), k
            assert v == v, k


def test_violation_order_is_stable():
    """Укладка идёт перед наклоном оси: строки таблицы сравнимы построчно."""
    # 8°, а не больше: при сильном наклоне столб сам заходит в боковую
    # область внизу кадра, и правило укладки перестаёт срабатывать —
    # на синтетике это артефакт фантома, а не поведение на снимке.
    f = evaluate(phantom(8.0, pelvis=False))
    assert f.violations == [SPINE_POSITIONING, SPINE_AXIS_TILT]


def test_precomputed_mask_gives_the_same_answer():
    img = phantom(9.0, pelvis=False)
    bone = segment_bone(img)
    a, b = evaluate(img), evaluate(img, bone=bone)
    assert a.violations == b.violations
    assert a.coverage == pytest.approx(b.coverage)
    assert a.axis_angle_deg == pytest.approx(b.axis_angle_deg)
