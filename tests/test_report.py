"""Выходная таблица: схема ТЗ 2.5 и побайтовая воспроизводимость записи."""
from __future__ import annotations

import csv

import pytest

from dxa_qc.report import (
    ALL_COLUMNS,
    COLUMNS,
    STATUS_FAILURE,
    STATUS_SUCCESS,
    ReportRow,
    write_report,
)

SAMPLE = [
    ReportRow(path_to_study="s1/a.dcm", study_uid="1.2.3", image_uid="1.2.3.4",
              anatomical_region="Поясничный отдел позвоночника", quality_class=0,
              violation_type="", processing_status=STATUS_SUCCESS,
              time_of_processing=0.0123, study="s1"),
    ReportRow(path_to_study="s1/b.dcm", study_uid="1.2.3", image_uid="1.2.3.5",
              anatomical_region="Проксимальный отдел правого бедра", quality_class=1,
              violation_type="FEMUR_POSITIONING_ROTATION",
              processing_status=STATUS_SUCCESS, time_of_processing=0.5, study="s1"),
    ReportRow(path_to_study="s2/c.dcm", anatomical_region="Не определена",
              violation_type="файл не является DICOM",
              processing_status=STATUS_FAILURE, time_of_processing=0.001, study="s2"),
]


def test_columns_are_exactly_tz_25():
    assert COLUMNS == [
        "path_to_study", "study_uid", "image_uid", "anatomical_region",
        "quality_class", "violation_type", "processing_status", "time_of_processing",
    ]


def test_study_column_goes_after_the_tz_ones():
    """Дополнительные колонки не ломают позиционный разбор первых восьми."""
    assert ALL_COLUMNS[:len(COLUMNS)] == COLUMNS
    assert ALL_COLUMNS[len(COLUMNS):] == ["quality_score", "study", "projection"]


def test_csv_schema_and_failure_row(tmp_path):
    out = write_report(SAMPLE, tmp_path / "r.csv")
    with open(out, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0].keys()) == ALL_COLUMNS
    assert len(rows) == 3
    assert rows[0]["study"] == "s1" and rows[2]["study"] == "s2"
    assert rows[0]["quality_class"] == "0"
    # у Failure quality_class пуст: ноль означал бы «снимок качественный»
    assert rows[2]["processing_status"] == STATUS_FAILURE
    assert rows[2]["quality_class"] == ""
    assert rows[2]["violation_type"]


def test_time_is_rounded_not_truncated(tmp_path):
    row = ReportRow(time_of_processing=0.12345678)
    out = write_report([row], tmp_path / "r.csv")
    with open(out, encoding="utf-8") as fh:
        assert "0.1235" in fh.read()


@pytest.mark.parametrize("name", ["r.csv", "r.xlsx"])
def test_write_is_byte_identical_between_runs(tmp_path, name):
    """Два прогона на одних данных дают одинаковый файл (ТЗ 2.7)."""
    if name.endswith(".xlsx"):
        pytest.importorskip("openpyxl")
    a = (tmp_path / "a" / name)
    b = (tmp_path / "b" / name)
    write_report(SAMPLE, a)
    write_report(SAMPLE, b)
    assert a.read_bytes() == b.read_bytes()


def test_xlsx_modified_time_is_fixed(tmp_path):
    """openpyxl пишет в docProps/core.xml время сохранения. Тест выше его не
    ловит, если оба файла записаны в пределах одной секунды, — этот ловит."""
    pytest.importorskip("openpyxl")
    import zipfile
    out = write_report(SAMPLE, tmp_path / "r.xlsx")
    with zipfile.ZipFile(out) as z:
        core = z.read("docProps/core.xml").decode()
    assert "2026-01-01T00:00:00Z</dcterms:modified>" in core
    assert "2026-01-01T00:00:00Z</dcterms:created>" in core


def test_xlsx_roundtrip_values(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    out = write_report(SAMPLE, tmp_path / "r.xlsx")
    ws = openpyxl.load_workbook(out).active
    got = [[c.value for c in row] for row in ws.iter_rows()]
    assert got[0] == ALL_COLUMNS
    assert got[1][4] == 0 and isinstance(got[1][4], int)
    assert got[3][4] is None          # Failure -> пустая ячейка, не 0
    assert got[2][5] == "FEMUR_POSITIONING_ROTATION"
    assert got[0][8] == "quality_score"      # девятая колонка — балл нарушения
    assert got[1][9] == "s1"                 # десятая — идентификатор исследования


def test_unknown_extension_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        write_report(SAMPLE, tmp_path / "r.parquet")
