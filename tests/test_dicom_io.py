"""Тесты загрузчика на синтетических DICOM.

Каждый тест соответствует краевому случаю, который может прийти в закрытом
тестовом наборе. Обучающий набор ни один из них не содержит (см.
reports/day1_findings.md, раздел 2), поэтому проверка возможна только синтетикой.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dxa_qc.dicom_io import (  # noqa: E402
    _as_floats,
    _first,
    _spacing,
    is_dicom,
    iter_dicom_files,
    read_dicom,
)
import synth_dicom as sd  # noqa: E402


class MultiValueLike:
    """Заглушка pydicom.multival.MultiValue.

    Важное свойство оригинала: это последовательность, но НЕ list и не tuple.
    Из-за проверки isinstance(v, (list, tuple)) PixelSpacing терялся, и
    масштаб молча откатывался на константу сканера. Тесты ниже — регрессия.
    """

    def __init__(self, values):
        self._v = list(values)

    def __iter__(self):
        return iter(self._v)

    def __len__(self):
        return len(self._v)

    def __getitem__(self, i):
        return self._v[i]


class DSFloatLike(float):
    """Заглушка pydicom.valuerep.DSfloat: float с исходной строкой."""

    def __new__(cls, value):
        obj = super().__new__(cls, float(value))
        obj.original_string = str(value)
        return obj


# --------------------------------------------------------------------------- #
# базовый профиль набора
# --------------------------------------------------------------------------- #
def test_baseline_implicit_le_8bit(tmp_path):
    p = sd.build_dicom(tmp_path / "a.dcm", sd.phantom())
    img = read_dicom(p)
    assert img.ok and img.error is None
    assert img.pixels.dtype == np.float32
    assert img.pixels.shape == (64, 48)
    assert 0.0 <= img.pixels.min() and img.pixels.max() <= 1.0
    assert img.photometric == "MONOCHROME2"
    assert img.modality == "CR"
    assert img.transfer_syntax == sd.TS_IMPLICIT_LE
    # кость (центр кадра) ярче фона
    assert img.pixels[32, 24] > img.pixels[2, 2]


@pytest.mark.parametrize(
    "ts", [sd.TS_IMPLICIT_LE, sd.TS_EXPLICIT_LE, sd.TS_EXPLICIT_BE]
)
def test_all_uncompressed_transfer_syntaxes(tmp_path, ts):
    """Explicit VR и big-endian: старые аппараты GE отдают и то, и другое."""
    p = sd.build_dicom(tmp_path / f"{ts}.dcm", sd.phantom(), transfer_syntax=ts)
    img = read_dicom(p)
    assert img.ok, img.error
    assert img.rows == 64 and img.cols == 48
    assert img.pixels[32, 24] > img.pixels[2, 2]


# --------------------------------------------------------------------------- #
# полярность
# --------------------------------------------------------------------------- #
def test_monochrome1_inverted(tmp_path):
    """MONOCHROME1: 0 — белый. После загрузки кость обязана остаться ярче фона."""
    ph = sd.phantom()
    inv = 255 - ph  # кость становится тёмной в native-значениях
    p1 = sd.build_dicom(tmp_path / "m1.dcm", inv, photometric="MONOCHROME1")
    p2 = sd.build_dicom(tmp_path / "m2.dcm", ph, photometric="MONOCHROME2")
    a, b = read_dicom(p1), read_dicom(p2)
    assert a.ok and b.ok
    assert a.pixels[32, 24] > a.pixels[2, 2]
    # обе полярности приводятся к одному изображению
    assert np.allclose(a.pixels, b.pixels, atol=1e-5)


# --------------------------------------------------------------------------- #
# битность и знак
# --------------------------------------------------------------------------- #
def test_16bit_unsigned(tmp_path):
    ph = sd.phantom(peak=4000, dtype_max=4095)
    p = sd.build_dicom(tmp_path / "x.dcm", ph, bits_allocated=16, bits_stored=12)
    img = read_dicom(p)
    assert img.ok, img.error
    assert img.bits_allocated == 16 and img.bits_stored == 12
    assert img.pixels.max() <= 1.0
    assert img.pixels[32, 24] > img.pixels[2, 2]


def test_16bit_signed(tmp_path):
    ph = sd.phantom(peak=3000, dtype_max=4095).astype(np.int32) - 1000
    p = sd.build_dicom(tmp_path / "x.dcm", ph, bits_allocated=16, bits_stored=16,
                       pixel_repr=1)
    img = read_dicom(p)
    assert img.ok, img.error
    assert img.pixel_repr == 1
    assert img.pixels[32, 24] > img.pixels[2, 2]


# --------------------------------------------------------------------------- #
# LUT и масштабирование значений
# --------------------------------------------------------------------------- #
def test_window_does_not_break_polarity(tmp_path):
    p = sd.build_dicom(tmp_path / "w.dcm", sd.phantom(), window=(120.0, 200.0))
    img = read_dicom(p)
    assert img.ok, img.error
    assert img.pixels[32, 24] > img.pixels[2, 2]


def test_rescale_slope_intercept(tmp_path):
    """RescaleSlope/Intercept не должны менять относительный порядок яркостей."""
    p = sd.build_dicom(tmp_path / "r.dcm", sd.phantom(), rescale=(2.0, -100.0))
    ref = sd.build_dicom(tmp_path / "ref.dcm", sd.phantom())
    a, b = read_dicom(p), read_dicom(ref)
    assert a.ok and b.ok
    assert np.allclose(a.pixels, b.pixels, atol=1e-5)


# --------------------------------------------------------------------------- #
# multi-frame
# --------------------------------------------------------------------------- #
def test_multiframe_selects_requested_frame(tmp_path):
    f0 = sd.phantom(peak=200)
    f1 = np.full_like(f0, 20)
    f1[40:60, 10:20] = 250
    stack = np.stack([f0, f1])
    p = sd.build_dicom(tmp_path / "mf.dcm", stack, n_frames=2)

    a = read_dicom(p, frame=0)
    b = read_dicom(p, frame=1)
    assert a.ok and b.ok
    assert a.n_frames == 2 and b.frame_index == 1
    assert not np.allclose(a.pixels, b.pixels)
    assert b.pixels[50, 15] > b.pixels[32, 24]


def test_multiframe_out_of_range_frame_is_clamped(tmp_path):
    stack = np.stack([sd.phantom(), sd.phantom(peak=100)])
    p = sd.build_dicom(tmp_path / "mf.dcm", stack, n_frames=2)
    img = read_dicom(p, frame=99)
    assert img.ok
    assert img.frame_index == 1
    assert any("вне диапазона" in w for w in img.warnings)


# --------------------------------------------------------------------------- #
# масштаб
# --------------------------------------------------------------------------- #
def test_pixel_spacing_wins_over_default(tmp_path):
    p = sd.build_dicom(tmp_path / "s.dcm", sd.phantom(), pixel_spacing=(0.25, 0.25))
    img = read_dicom(p)
    assert img.mm_per_px == pytest.approx(0.25)
    assert img.mm_per_px_source == "pixel_spacing"


def test_no_spacing_falls_back_to_scanner_default(tmp_path):
    p = sd.build_dicom(tmp_path / "s.dcm", sd.phantom())
    img = read_dicom(p)
    assert img.mm_per_px == pytest.approx(0.6)
    assert img.mm_per_px_source == "scanner_default"


@pytest.mark.parametrize(
    "raw",
    [
        MultiValueLike([DSFloatLike("0.25"), DSFloatLike("0.25")]),  # как отдаёт pydicom
        [0.25, 0.25],
        (0.25, 0.25),
        "0.25\\0.25",                                                # сырое значение DS
        b"0.25\\0.25",
        0.25,
        DSFloatLike("0.25"),
    ],
)
def test_pixel_spacing_is_read_from_any_value_shape(raw):
    """PixelSpacing приходит в разных обёртках; масштаб обязан браться из него.

    Регрессия: pydicom отдаёт MultiValue, который не является list или tuple,
    и прежняя проверка типа молча его отбрасывала.
    """
    mm, src = _spacing(raw, None, None, 300, 300)
    assert src == "pixel_spacing"
    assert mm == pytest.approx(0.25)


def test_imager_pixel_spacing_is_the_fallback():
    mm, src = _spacing(None, MultiValueLike([DSFloatLike("0.3")]), None, 300, 300)
    assert src == "imager_pixel_spacing"
    assert mm == pytest.approx(0.3)


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, []),
        ("", []),
        ("нет", []),
        (MultiValueLike([0.0, -1.0]), []),   # нулевой и отрицательный шаг — мусор
        (float("nan"), []),
        (MultiValueLike([DSFloatLike("0.6"), "мусор"]), [0.6]),
    ],
)
def test_as_floats_drops_garbage(raw, expected):
    assert _as_floats(raw) == expected


def test_first_handles_multivalue():
    """WindowCenter='40\\400' у pydicom тоже MultiValue."""
    assert _first(MultiValueLike([DSFloatLike("40"), DSFloatLike("400")])) == 40.0
    assert _first(MultiValueLike([])) is None
    assert _first(0.6) == 0.6
    assert _first("MONOCHROME2") == "MONOCHROME2"


def test_implausible_exposed_area_is_rejected(tmp_path):
    """Exposed Area описывает поле сканирования, а не кроп: 2 мм/пиксель невозможны."""
    p = sd.build_dicom(tmp_path / "s.dcm", sd.phantom(rows=64, cols=48),
                       exposed_area=(180, 192))
    img = read_dicom(p)
    assert img.mm_per_px_source == "scanner_default"


def test_plausible_exposed_area_is_used(tmp_path):
    ph = sd.phantom(rows=300, cols=300)
    p = sd.build_dicom(tmp_path / "s.dcm", ph, exposed_area=(180, 180))
    img = read_dicom(p)
    assert img.mm_per_px_source == "exposed_area"
    assert img.mm_per_px == pytest.approx(0.6, abs=0.01)


# --------------------------------------------------------------------------- #
# отказы: всё через error, ни одного исключения наружу
# --------------------------------------------------------------------------- #
def test_compressed_transfer_syntax_without_pydicom(tmp_path):
    """Сжатый поток без декодера — это Failure с внятным текстом, а не падение."""
    p = sd.build_dicom(tmp_path / "j.dcm", sd.phantom(),
                       transfer_syntax=sd.TS_JPEG_BASELINE, encapsulated=True)
    img = read_dicom(p)
    assert isinstance(img.ok, bool)
    if not img.ok:
        assert img.error and "pydicom" in img.error


def test_truncated_pixel_data(tmp_path):
    p = sd.build_dicom(tmp_path / "t.dcm", sd.phantom(), truncate_pixels=1000)
    img = read_dicom(p)
    assert not img.ok
    assert img.error


def test_not_a_dicom_file(tmp_path):
    p = tmp_path / "junk.dcm"
    p.write_bytes(b"this is not a dicom file" * 10)
    assert not is_dicom(p)
    img = read_dicom(p)
    assert not img.ok and img.error


def test_empty_file(tmp_path):
    p = tmp_path / "empty.dcm"
    p.write_bytes(b"")
    img = read_dicom(p)
    assert not img.ok and img.error


def test_one_by_one_image(tmp_path):
    """Вырожденный кадр не должен ронять нормировку (деление на ноль)."""
    p = sd.build_dicom(tmp_path / "px.dcm", np.array([[7]], dtype=np.int32))
    img = read_dicom(p)
    assert img.ok, img.error
    assert img.pixels.shape == (1, 1)
    assert np.isfinite(img.pixels).all()


def test_color_image_rejected(tmp_path):
    p = sd.build_dicom(tmp_path / "rgb.dcm", sd.phantom(), samples_per_pixel=3)
    img = read_dicom(p)
    assert not img.ok
    assert "SamplesPerPixel" in (img.error or "")


def test_missing_preamble_is_still_readable(tmp_path):
    """Некоторые выгрузки отдают «голый» dataset без 128-байтовой преамбулы."""
    p = sd.build_dicom(tmp_path / "raw.dcm", sd.phantom(), preamble=False)
    img = read_dicom(p)
    assert img.ok, img.error
    assert img.pixels.shape == (64, 48)


def test_sequence_is_skipped_without_losing_tags(tmp_path):
    p = sd.build_dicom(tmp_path / "sq.dcm", sd.phantom(), extra_sequence=True)
    img = read_dicom(p)
    assert img.ok, img.error
    assert img.rows == 64 and img.cols == 48
    assert img.study_instance_uid == "1.2.3.4.5"


# --------------------------------------------------------------------------- #
# обход каталога
# --------------------------------------------------------------------------- #
def test_iter_ignores_extension_and_hidden(tmp_path):
    sd.build_dicom(tmp_path / "sub" / "CR000000", sd.phantom())  # без .dcm
    (tmp_path / "sub" / ".DS_Store").write_bytes(b"x")
    files = list(iter_dicom_files(tmp_path))
    assert len(files) == 1
    assert read_dicom(files[0]).ok

def test_pydicom_warnings_go_into_the_row_not_into_stderr(tmp_path):
    """Предупреждения pydicom собираются в warnings изображения.

    На наборе организатора их около двухсот на прогон («Invalid value for
    VR UI»: компоненты UID начинаются с нуля). В stderr им не место —
    они топят всё остальное; в диагностической таблице рядом с файлом —
    место.
    """
    pytest.importorskip("pydicom")
    import warnings as W

    path = tmp_path / "bad_uid.dcm"
    sd.build_dicom(path, sd.phantom(), sop_uid="1.2.643.5.1.13.12.2.77.08110703091310")
    with W.catch_warnings(record=True) as leaked:
        W.simplefilter("always")
        img = read_dicom(path)
    assert img.ok
    assert img.reader == "pydicom"
    assert not [w for w in leaked if "Invalid value for VR" in str(w.message)]
    assert any("VR" in w for w in img.warnings)


def test_conforming_file_collects_no_read_warnings(tmp_path):
    pytest.importorskip("pydicom")
    path = tmp_path / "ok.dcm"
    sd.build_dicom(path, sd.phantom(), sop_uid="1.2.3.4")
    img = read_dicom(path)
    assert img.ok
    assert not [w for w in img.warnings if "VR" in w]


def test_error_text_is_stable_between_runs(tmp_path):
    """Адреса объектов и временные пути не попадают в таблицу (ТЗ 2.7)."""
    from dxa_qc.dicom_io import stable_error
    p = tmp_path / "sub" / "x.dcm"
    a = stable_error(f"pillow: cannot identify <_io.BytesIO object at 0x7f950b780a40>\n  in {p}", p)
    b = stable_error(f"pillow: cannot identify <_io.BytesIO object at 0x7f9506332020>\n  in {p}", p)
    assert a == b == "pillow: cannot identify <_io.BytesIO object> in x.dcm"
