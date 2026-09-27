"""Сквозной прогон: состав таблицы, ветка Failure, дедупликация, детерминизм.

Главное требование ТЗ 2.7 к этому уровню: ни один входной файл не может
уронить обработку остальных. Поэтому большинство тестов здесь — про мусор
на входе, а не про счастливый путь.
"""
from __future__ import annotations

import csv
import shutil

import numpy as np
import pytest

from dxa_qc.batch import main, run
from dxa_qc.report import ALL_COLUMNS, COLUMNS, STATUS_FAILURE, STATUS_SUCCESS
from synth_dicom import build_dicom


def spine_pixels(rows=180, cols=160, peak=220, pelvis=True):
    """Фантом позвоночника: яркий столб по центру.

    `pelvis` добавляет по бокам внизу плотные структуры — на настоящем
    снимке это крылья подвздошных костей и окружающие ткани. Без них
    правило укладки (ТЗ 2.3.1) обязано сработать: именно их отсутствие
    означает, что сканирование остановилось выше гребней.
    """
    a = np.full((rows, cols), int(peak * 0.08), dtype=np.int32)
    a[:, cols // 2 - 12: cols // 2 + 12] = peak
    if pelvis:
        lo = int(rows * 0.78)
        a[lo:, : cols // 2 - 30] = int(peak * 0.35)
        a[lo:, cols // 2 + 30:] = int(peak * 0.35)
    return a


def femur_pixels(rows=240, cols=160, side=+1, offset=60, peak=220):
    """Фантом бедра: таз сверху сбоку, диафиз по центру снизу.

    240 строк при масштабе по умолчанию 0.6 мм/пиксель дают поле 144 мм —
    длиннее порога охвата области интереса (ТЗ 2.3.2, 133 мм). Короткий
    кадр проверяется отдельным тестом: правило обязано на нём срабатывать.
    """
    a = np.full((rows, cols), int(peak * 0.08), dtype=np.int32)
    top_c = cols // 2 + side * offset
    a[: rows // 3, top_c - 12: top_c + 12] = peak
    a[2 * rows // 3:, cols // 2 - 8: cols // 2 + 8] = peak
    return a


def make_study(root, name="study_A", with_duplicate=True):
    """Типичное исследование: позвоночник, два бедра и копия позвоночника."""
    d = root / name / "series_001" / "DXA"
    d.mkdir(parents=True, exist_ok=True)
    build_dicom(d / "CR000000.dcm", spine_pixels(), study_uid="1.2.3", sop_uid="1.2.3.1")
    build_dicom(d / "CR000001.dcm", femur_pixels(side=+1), study_uid="1.2.3", sop_uid="1.2.3.2")
    build_dicom(d / "CR000002.dcm", femur_pixels(side=-1), study_uid="1.2.3", sop_uid="1.2.3.3")
    if with_duplicate:
        shutil.copy(d / "CR000000.dcm", d / "CR000003.dcm")
    return d


def read_rows(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_happy_path_schema_and_regions(tmp_path):
    make_study(tmp_path / "in")
    out = tmp_path / "result.csv"
    assert run(tmp_path / "in", out, region_lang="code", quiet=True) == 0

    rows = read_rows(out)
    assert list(rows[0].keys()) == ALL_COLUMNS
    assert len(rows) == 4                       # строка на каждый ФАЙЛ, включая копию
    assert {r["processing_status"] for r in rows} == {STATUS_SUCCESS}
    assert {r["study"] for r in rows} == {"study_A"}
    assert [r["anatomical_region"] for r in rows] == [
        "LUMBAR_SPINE", "RIGHT_PROXIMAL_FEMUR", "LEFT_PROXIMAL_FEMUR", "LUMBAR_SPINE",
    ]
    # Фантом ровный и с «тазом» по бокам внизу: геометрических нарушений нет.
    assert all(r["quality_class"] == "0" for r in rows)
    assert all(r["violation_type"] == "" for r in rows)
    assert all(float(r["time_of_processing"]) > 0 for r in rows)
    assert all(r["study_uid"] == "1.2.3" for r in rows)
    assert rows[0]["image_uid"] != rows[1]["image_uid"]


def test_short_femur_frame_is_flagged_as_roi_coverage(tmp_path):
    """Короткое поле сканирования бедра — нарушение охвата области интереса."""
    d = make_study(tmp_path / "in")
    build_dicom(d / "CR000001.dcm", femur_pixels(rows=180, side=+1),
                study_uid="1.2.3", sop_uid="1.2.3.2")
    out = tmp_path / "r.csv"
    assert run(tmp_path / "in", out, region_lang="code", quiet=True) == 0
    rows = read_rows(out)
    short = [r for r in rows if r["anatomical_region"] == "RIGHT_PROXIMAL_FEMUR"]
    assert short and all(r["violation_type"] == "FEMUR_ROI_COVERAGE" for r in short)
    assert all(r["quality_class"] == "1" for r in short)
    spine = [r for r in rows if r["anatomical_region"] == "LUMBAR_SPINE"]
    assert all(r["violation_type"] == "" for r in spine)


def test_duplicate_inherits_region_of_its_twin(tmp_path):
    make_study(tmp_path / "in")
    out = tmp_path / "r.csv"
    run(tmp_path / "in", out, region_lang="code", quiet=True)
    rows = read_rows(out)
    assert rows[0]["anatomical_region"] == rows[3]["anatomical_region"]


def test_garbage_file_becomes_failure_row(tmp_path):
    d = make_study(tmp_path / "in")
    (d / "CR000009.dcm").write_bytes(b"not a dicom at all" * 10)
    out = tmp_path / "r.csv"
    assert run(tmp_path / "in", out, quiet=True) == 0

    rows = read_rows(out)
    bad = [r for r in rows if r["path_to_study"].endswith("CR000009.dcm")]
    assert len(bad) == 1
    assert bad[0]["processing_status"] == STATUS_FAILURE
    assert bad[0]["quality_class"] == ""        # не 0: снимок не прочитан
    assert bad[0]["violation_type"]
    # остальные файлы обработаны как ни в чём не бывало
    assert sum(r["processing_status"] == STATUS_SUCCESS for r in rows) == 4


def test_truncated_pixel_data_becomes_failure_row(tmp_path):
    d = make_study(tmp_path / "in")
    build_dicom(d / "CR000010.dcm", spine_pixels(), truncate_pixels=5000,
                study_uid="1.2.3", sop_uid="1.2.3.9")
    out = tmp_path / "r.csv"
    run(tmp_path / "in", out, quiet=True)
    rows = read_rows(out)
    bad = [r for r in rows if r["path_to_study"].endswith("CR000010.dcm")][0]
    assert bad["processing_status"] == STATUS_FAILURE


def test_degenerate_image_does_not_crash(tmp_path):
    """Изображение 1x1: область не определима -> Failure, но не исключение."""
    d = (tmp_path / "in" / "study_B" / "s")
    d.mkdir(parents=True)
    build_dicom(d / "tiny.dcm", np.array([[7]], dtype=np.int32))
    out = tmp_path / "r.csv"
    assert run(tmp_path / "in", out, quiet=True) == 0
    rows = read_rows(out)
    assert len(rows) == 1
    assert rows[0]["processing_status"] == STATUS_FAILURE
    assert "область" in rows[0]["violation_type"]


def test_empty_input_gives_valid_empty_table(tmp_path):
    (tmp_path / "in").mkdir()
    out = tmp_path / "r.csv"
    assert run(tmp_path / "in", out, quiet=True) == 0
    with open(out, encoding="utf-8") as fh:
        assert fh.read().strip() == ",".join(ALL_COLUMNS)


def test_missing_input_returns_error_code(tmp_path):
    assert run(tmp_path / "нет-такого", tmp_path / "r.csv", quiet=True) == 2


def test_study_key_is_directory_not_tag(tmp_path):
    make_study(tmp_path / "in", name="2.25.999")
    out, diag = tmp_path / "r.csv", tmp_path / "d.csv"
    run(tmp_path / "in", out, diagnostics=diag, quiet=True)
    rows = read_rows(diag)
    assert {r["study"] for r in rows} == {"2.25.999"}
    # при этом тег из файла честно попадает в study_uid
    assert {r["study_uid"] for r in rows} == {"1.2.3"}


def test_results_do_not_depend_on_run(tmp_path):
    """Два прогона совпадают во всём, кроме колонки времени."""
    make_study(tmp_path / "in")
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    run(tmp_path / "in", a, quiet=True)
    run(tmp_path / "in", b, quiet=True)
    stable = [c for c in ALL_COLUMNS if c != "time_of_processing"]
    ra = [[r[c] for c in stable] for r in read_rows(a)]
    rb = [[r[c] for c in stable] for r in read_rows(b)]
    assert ra == rb


def test_cli_writes_xlsx(tmp_path):
    pytest.importorskip("openpyxl")
    make_study(tmp_path / "in")
    out = tmp_path / "result.xlsx"
    code = main(["--input", str(tmp_path / "in"), "--output", str(out), "--quiet"])
    assert code == 0 and out.exists() and out.stat().st_size > 0


def test_region_name_is_russian_by_default(tmp_path):
    """По умолчанию колонка заполняется словами ТЗ 2.3, а не кодами."""
    make_study(tmp_path / "in", with_duplicate=False)
    out = tmp_path / "r.csv"
    run(tmp_path / "in", out, quiet=True)
    assert read_rows(out)[0]["anatomical_region"].startswith("Поясничный")


def test_region_lang_code(tmp_path):
    make_study(tmp_path / "in", with_duplicate=False)
    out = tmp_path / "r.csv"
    run(tmp_path / "in", out, region_lang="code", quiet=True)
    assert read_rows(out)[0]["anatomical_region"] == "LUMBAR_SPINE"


def test_spine_without_pelvis_is_flagged_as_positioning(tmp_path):
    """Кадр, обрезанный выше гребней подвздошных костей, — нарушение укладки.

    Это единственное правило ТЗ 2.3.1, которое видно на синтетике: наклон
    оси требует наклонённого столба, а артефакты — классификатора.
    """
    d = tmp_path / "in" / "study_B" / "s" / "DXA"
    d.mkdir(parents=True)
    build_dicom(d / "CR000000.dcm", spine_pixels(pelvis=False),
                study_uid="1.2.4", sop_uid="1.2.4.1")
    out = tmp_path / "r.csv"
    assert run(tmp_path / "in", out, region_lang="code", quiet=True) == 0

    row = read_rows(out)[0]
    assert row["processing_status"] == STATUS_SUCCESS
    assert row["quality_class"] == "1"
    assert row["violation_type"] == "SPINE_POSITIONING"
