"""Overlay-серия: картинка на каждый файл, детерминизм, сбой не трогает таблицу."""
from __future__ import annotations

import zipfile

import cv2
import numpy as np

from dxa_qc import batch, overlay
from dxa_qc.batch import run
from test_batch import femur_pixels, make_study, read_rows, spine_pixels


def _png(data: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def test_one_png_per_file_and_copies_share_bytes(tmp_path):
    make_study(tmp_path / "in")                 # 4 файла, один — копия позвоночника
    z = tmp_path / "ov.zip"
    assert run(tmp_path / "in", tmp_path / "r.csv", overlays=z, quiet=True) == 0
    names = zipfile.ZipFile(z).namelist()
    rows = read_rows(tmp_path / "r.csv")
    assert sorted(names) == sorted([r["path_to_study"] + ".png" for r in rows]
                                   + [r["path_to_study"] + ".overlay.dcm" for r in rows]
                                   + ["study_A/" + batch.SR_NAME])          # отчёт SR на исследование
    zf = zipfile.ZipFile(z)
    spine = [n for n in names if n.endswith(("CR000000.dcm.png", "CR000003.dcm.png"))]
    assert zf.read(spine[0]) == zf.read(spine[1])
    img = _png(zf.read(next(n for n in names if n.endswith(".png"))))
    assert img is not None and img.ndim == 3


def test_overlay_archive_is_byte_identical_between_runs(tmp_path):
    make_study(tmp_path / "in")
    a, b = tmp_path / "a.zip", tmp_path / "b.zip"
    run(tmp_path / "in", tmp_path / "a.csv", overlays=a, quiet=True)
    run(tmp_path / "in", tmp_path / "b.csv", overlays=b, quiet=True)
    assert a.read_bytes() == b.read_bytes()


def test_overlay_does_not_change_the_table(tmp_path):
    make_study(tmp_path / "in")
    run(tmp_path / "in", tmp_path / "plain.csv", quiet=True)
    run(tmp_path / "in", tmp_path / "with.csv", overlays=tmp_path / "o.zip", quiet=True)
    strip = lambda rows: [{k: v for k, v in r.items() if k != "time_of_processing"} for r in rows]
    assert strip(read_rows(tmp_path / "plain.csv")) == strip(read_rows(tmp_path / "with.csv"))


def test_render_failure_is_isolated(tmp_path):
    make_study(tmp_path / "in")
    original = overlay.render

    def broken(*a, **k):
        raise RuntimeError("сломалось")

    overlay.render = broken
    try:
        z = tmp_path / "o.zip"
        assert run(tmp_path / "in", tmp_path / "r.csv", overlays=z, quiet=True) == 0
    finally:
        overlay.render = original
    zf = zipfile.ZipFile(z)
    # Отрисовка упала — картинок нет, но текстовый отчёт SR от неё не зависит.
    assert zf.namelist() == ["study_A/" + batch.SR_NAME, "overlay_errors.txt"]
    assert "сломалось" in zf.read("overlay_errors.txt").decode("utf-8")
    assert all(r["processing_status"] == "Success" for r in read_rows(tmp_path / "r.csv"))


def test_violation_is_drawn_red():
    """Сработавшее правило — красным в шапке; без нарушений красного нет."""
    px = spine_pixels().astype(float)
    bad = overlay.render("spine", px, ["SPINE_AXIS_TILT"], {"spine_coverage": 0.3})
    good = overlay.render("spine", px, [], {"spine_coverage": 0.3})
    red = lambda im: int(((im[..., 2] > 200) & (im[..., 1] < 80) & (im[..., 0] < 80)).sum())
    assert red(bad) > 0 and red(good) == 0


def test_femur_render_survives_untrackable_image():
    img = overlay.render("femur_left", femur_pixels().astype(float), [], {})
    assert img.shape[1] == femur_pixels().shape[1] * overlay.SCALE


def test_batch_uses_overlay_module_attribute():
    """Подмена в тесте выше работает, только если batch зовёт overlay.render по имени модуля."""
    assert batch.overlay_series is overlay


def test_femur_lateral_margin_is_reference_only():
    """Строка про 2 см сбоку добавляется, только если отступ измерен, и не меняет вердикт."""
    px = femur_pixels().astype(float)
    with_margin = overlay.render("femur_left", px, [], {"femur_lateral_mm": "8.4"})
    without = overlay.render("femur_left", px, [], {})
    assert with_margin.shape[0] == without.shape[0] + overlay.LINE_H
    assert np.array_equal(with_margin[-px.shape[0] * overlay.SCALE:], without[-px.shape[0] * overlay.SCALE:])
