"""Правило охвата области интереса и геометрия бедра (ТЗ 2.3.2).

Правило считается из длины поля сканирования и маски кости, поэтому
проверяется на синтетике: «бедро» — наклонная трубка, выходящая за нижний
край кадра, рядом «таз», по бокам нулевые столбцы (в реальных кадрах бедра
поле сканирования уже кадра, и больше половины пикселей — ровные нули).
Порог 133 мм откалиброван на реальных данных (reports/day5_findings.md);
здесь проверяется, что реализация делает заявленное и не падает на
вырожденном входе.
"""
from __future__ import annotations

import numpy as np
import pytest

from dxa_qc.femur import (
    DEFAULT_MM_PER_PX,
    FEMUR_ROI_COVERAGE,
    FIELD_MIN_MM,
    evaluate,
    femur_mask,
    field_scale,
    scan_field,
    shaft_angle,
)
from dxa_qc.segmentation import segment_bone

MM = 0.6


def phantom(rows: int = 300, cols: int = 280, tilt_deg: float = 0.0,
            pad: int = 40, peak: int = 220, side: str = "femur_left") -> np.ndarray:
    """Кадр бедра: таз сверху-медиально, диафиз до нижнего края, нули по бокам.

    Диафиз рисуется сразу под нужным углом, а не поворотом кадра: поворот
    заносит по углам чёрные клинья и меняет то, что меряет правило. Между
    тазом и вертельной областью оставлен зазор — суставная щель, из-за
    которой маска бедра не смыкается с тазом.
    """
    a = np.zeros((rows, cols), dtype=np.float32)
    inner = a[:, pad: cols - pad]
    inner[:] = int(peak * 0.15)                      # мягкие ткани
    w = inner.shape[1]
    top = rows // 3
    if side == "femur_left":                         # таз слева, бедро справа
        inner[:top, : w // 2 - 6] = int(peak * 0.75)
        c = int(w * 0.68)
        troch = slice(max(c - 40, w // 2 + 6), c + 40)
    else:                                            # зеркально
        inner[:top, w // 2 + 6:] = int(peak * 0.75)
        c = int(w * 0.32)
        troch = slice(max(c - 40, 0), min(c + 40, w // 2 - 6))
    inner[top: top + 30, troch] = int(peak * 0.9)    # вертельная область
    shift = np.tan(np.radians(tilt_deg))
    for r in range(top, rows):                       # диафиз до нижнего края
        cr = int(round(c + shift * (r - top)))
        inner[r, max(cr - 22, 0): min(cr + 22, w)] = peak
    a[:, pad: cols - pad] = inner
    return a.astype(np.int32)


# --- правило охвата --------------------------------------------------------- #

def test_long_field_is_not_a_violation():
    f = evaluate(phantom(rows=300), "femur_left", MM)
    assert f.ok
    assert f.violations == []
    assert f.quality_class == 0
    assert f.field_h_mm == pytest.approx(180.0)


def test_short_field_is_a_violation():
    f = evaluate(phantom(rows=200), "femur_left", MM)
    assert f.violations == [FEMUR_ROI_COVERAGE]
    assert f.quality_class == 1


def test_threshold_is_where_it_is_declared():
    just_under = int(round((FIELD_MIN_MM - 1.0) / MM))
    just_over = int(round((FIELD_MIN_MM + 1.0) / MM))
    assert evaluate(phantom(rows=just_under), "femur_left", MM).violations
    assert not evaluate(phantom(rows=just_over), "femur_left", MM).violations


def test_scale_changes_the_decision_not_the_pixel_count():
    """Один и тот же кадр при другом размере пикселя решается иначе."""
    rows = 230
    assert evaluate(phantom(rows=rows), "femur_left", 0.5).violations
    assert not evaluate(phantom(rows=rows), "femur_left", 0.7).violations


def test_missing_scale_falls_back_to_scanner_default():
    f = evaluate(phantom(rows=200), "femur_left", None)
    assert f.field_h_mm == pytest.approx(200 * DEFAULT_MM_PER_PX)
    assert "паспорт" in (f.reason or "")


# --- масштаб: тег даёт оценку сверху, берётся минимум ------------------------

def test_inflated_tag_scale_is_clamped_to_the_passport():
    """Кадр — вырезка из поля сканирования, поэтому тег завышает масштаб."""
    mm, note = field_scale(0.6513, "exposed_area", "GE Lunar Prodigy Advance")
    assert mm == pytest.approx(DEFAULT_MM_PER_PX)
    assert note is None                      # 8.5% — в пределах ожидаемого


def test_smaller_tag_scale_is_trusted():
    """Пиксель меньше паспортного — это другой аппарат, а не обрезка."""
    mm, _ = field_scale(0.45)
    assert mm == pytest.approx(0.45)


def test_large_disagreement_is_reported():
    mm, note = field_scale(0.9)
    assert mm == pytest.approx(DEFAULT_MM_PER_PX)
    assert note and "50%" in note


@pytest.mark.parametrize("bad", [None, float("nan"), 0.0, -1.0])
def test_broken_scale_falls_back(bad):
    mm, note = field_scale(bad)
    assert mm == pytest.approx(DEFAULT_MM_PER_PX)
    assert note


def test_inflated_tag_does_not_hide_a_short_field():
    """209 строк при завышенном теге читались бы как 136 мм вместо 125."""
    a = phantom(rows=209)
    assert evaluate(a, "femur_left", 0.6513).violations == [FEMUR_ROI_COVERAGE]
    assert evaluate(a, "femur_left", 0.6).violations == [FEMUR_ROI_COVERAGE]


# --- достоверный тег зажимать нельзя ---------------------------------------- #

@pytest.mark.parametrize("source", ["pixel_spacing", "imager_pixel_spacing"])
def test_trusted_tag_is_not_clamped(source):
    """PixelSpacing — не оценка сверху, а измерение; паспорт ему не указ."""
    mm, note = field_scale(0.8, source, "Lunar Prodigy Advance")
    assert mm == pytest.approx(0.8)
    assert note and "расхождение" in note        # сказать сказали, но не испортили


def test_trusted_tag_close_to_passport_is_silent():
    mm, note = field_scale(0.61, "pixel_spacing", "Lunar Prodigy Advance")
    assert mm == pytest.approx(0.61)
    assert note is None


def test_exposed_area_is_still_clamped():
    """Оценка из Exposed Area остаётся оценкой сверху."""
    mm, _ = field_scale(0.6513, "exposed_area", "Lunar Prodigy Advance")
    assert mm == pytest.approx(DEFAULT_MM_PER_PX)


def test_trusted_tag_keeps_a_long_field_out_of_violations():
    """Сквозная проверка: аппарат с крупным пикселем не ловит ложное нарушение.

    240 строк при настоящих 0.8 мм/пиксель — это 192 мм, норма. При старом
    зажиме до паспортных 0.6 вышло бы 144 мм: тоже норма, но уже впритык.
    Берём 180 строк: 144 мм по тегу против 108 мм по зажатому паспорту —
    разница между нормой и нарушением.
    """
    a = phantom(rows=180)
    trusted = evaluate(a, "femur_left", 0.8, mm_per_px_source="pixel_spacing",
                       model="Lunar Prodigy Advance")
    assert trusted.field_h_mm == pytest.approx(144.0)
    assert trusted.violations == []
    clamped = evaluate(a, "femur_left", 0.8, mm_per_px_source="exposed_area",
                       model="Lunar Prodigy Advance")
    assert clamped.field_h_mm == pytest.approx(108.0)
    assert clamped.violations == [FEMUR_ROI_COVERAGE]


# --- паспортный масштаб относится к конкретному аппарату --------------------- #

def test_foreign_scanner_without_tag_is_reported():
    _, note = field_scale(None, "scanner_default", "Hologic Horizon A")
    assert note and "Hologic Horizon A" in note


def test_unknown_model_without_tag_is_reported():
    _, note = field_scale(None, "scanner_default", None)
    assert note and "неизвестна" in note


def test_calibrated_scanner_without_tag_is_silent_about_the_model():
    _, note = field_scale(None, "scanner_default", "GE Lunar Prodigy Advance")
    assert note == "масштаб не определён, взят паспортный"


def test_applied_scale_reaches_diagnostics():
    d = evaluate(phantom(rows=240), "femur_left", 0.8,
                 mm_per_px_source="pixel_spacing", model="Lunar Prodigy Advance").diagnostics()
    assert d["femur_mm_per_px"] == pytest.approx(0.8)
    assert d["femur_mm_per_px_source"] == "pixel_spacing"


# --- стороны кадра ---------------------------------------------------------- #

@pytest.mark.parametrize("side", ["femur_left", "femur_right"])
def test_lateral_and_medial_are_swapped_by_region(side):
    f = evaluate(phantom(side=side), side, MM)
    assert f.ok
    assert f.lateral_mm == f.lateral_mm and f.medial_mm == f.medial_mm
    mirrored = evaluate(phantom(side="femur_right" if side == "femur_left" else "femur_left"),
                        "femur_right" if side == "femur_left" else "femur_left", MM)
    assert f.lateral_mm == pytest.approx(mirrored.lateral_mm, abs=2.0)
    assert f.medial_mm == pytest.approx(mirrored.medial_mm, abs=2.0)


def test_scan_field_ignores_zero_padding():
    a = phantom(pad=40)
    c0, c1 = scan_field(a)
    assert c0 == 40 and c1 == a.shape[1] - 40


def test_scan_field_is_the_span_of_columns_with_data():
    a = np.full((10, 20), 5, dtype=np.int32)
    a[:, 3:17] = 90
    assert scan_field(a) == (3, 17)


def test_scan_field_on_uniform_frame_is_the_whole_width():
    a = np.full((10, 20), 5, dtype=np.int32)
    assert scan_field(a) == (0, 20)


# --- ось диафиза ------------------------------------------------------------ #

@pytest.mark.parametrize("tilt", [0.0, 6.0, -6.0])
def test_shaft_angle_follows_the_tilt(tilt):
    a = phantom(tilt_deg=tilt)
    bone = segment_bone(a)
    m = femur_mask(a, bone)
    assert m is not None
    assert shaft_angle(m) == pytest.approx(abs(tilt), abs=2.5)


def test_femur_mask_takes_the_component_touching_the_bottom():
    a = phantom()
    m = femur_mask(a, segment_bone(a))
    assert m is not None
    assert m[-1].any()          # диафиз выходит за нижний край
    assert not m[0].any()       # таз в маску бедра не попал


# --- краевые случаи --------------------------------------------------------- #

def test_none_image_does_not_raise():
    f = evaluate(None, "femur_left", MM)
    assert not f.ok and f.violations == [] and f.reason


def test_empty_frame_still_reports_field_length():
    f = evaluate(np.zeros((150, 280), dtype=np.int32), "femur_left", MM)
    assert not f.ok
    assert f.field_h_mm == pytest.approx(90.0)
    assert f.violations == [FEMUR_ROI_COVERAGE]   # длина поля известна и без маски


def test_tiny_frame_does_not_raise():
    f = evaluate(np.zeros((4, 4), dtype=np.int32), "femur_left", MM)
    assert not f.ok and f.reason


def test_diagnostics_are_serialisable():
    d = evaluate(phantom(), "femur_left", MM).diagnostics()
    assert set(d) == {"femur_field_h_mm", "femur_field_w_mm", "femur_lateral_mm",
                      "femur_medial_mm", "femur_shaft_deg", "femur_fill",
                      "femur_mm_per_px", "femur_mm_per_px_source", "femur_reason"}
    assert all(isinstance(v, (int, float, str)) for v in d.values())
