"""Отчёт DICOM SR (ТЗ 2.6): структура, воспроизводимость, место в конвейере."""
from __future__ import annotations

import struct
import zipfile

import pytest

from dxa_qc import batch, dicom_sc, dicom_sr
from dxa_qc.batch import run
from test_batch import make_study, read_rows
from test_dicom_sc import _elements

SC = "1.2.840.10008.5.1.4.1.1.1"


def _finding(q=1, violations=("SPINE_AXIS_TILT",), copies=1, **kw):
    refs = [(SC, f"1.2.3.{i + 1}", "1.2.3.9", "1.2.3") for i in range(copies)]
    args = dict(path="S/DXA/CR000000.dcm", region_key="spine", region_name="Поясничный отдел позвоночника",
                quality_class=q, violations=list(violations),
                diag={"spine_angle_deg": 6.7, "spine_coverage": 0.3, "artifact_prob": 0.1},
                score=0.83 if q else 0.21, refs=refs)
    args.update(kw)
    return dicom_sr.ImageFinding(**args)


def _sr(findings=None, **kw):
    args = dict(study_uid="1.2.3", series_uid=dicom_sc.derive_uid("sr-series"),
                sop_uid=dicom_sc.derive_uid("sr"), study_score=0.83, artifact_threshold=0.52,
                context={"study_date": "Anonymized", "patient_sex": "M"})
    args.update(kw)
    return dicom_sr.basic_text_sr(findings or [_finding()], **args)


def test_header_structure_and_type1_attributes():
    data = _sr()
    assert data[128:132] == b"DICM"
    els = _elements(data)
    body = [t for t in els if t[0] != 0x0002]
    assert body == sorted(body)
    assert all(len(raw) % 2 == 0 for _, raw in els.values())
    txt = lambda t: els[t][1].rstrip(b"\x00 ").decode()  # noqa: E731
    assert txt((0x0008, 0x0016)) == dicom_sr.SR_SOP_CLASS and txt((0x0008, 0x0060)) == "SR"
    assert txt((0x0040, 0xA491)) == "COMPLETE" and txt((0x0040, 0xA493)) == "UNVERIFIED"
    assert txt((0x0040, 0xA040)) == "CONTAINER"
    # Дата исследования обезличена: StudyDate пуст, ContentDate (тип 1) — фиксированный.
    assert els[(0x0008, 0x0020)][1] == b"" and txt((0x0008, 0x0023)) == dicom_sr.FIXED_DATE
    meta_len = struct.unpack("<I", els[(0x0002, 0x0000)][1])[0]
    assert data.index(struct.pack("<HH", 0x0008, 0x0005), 132) == 132 + 12 + meta_len


def test_text_is_in_russian_and_mentions_measurements():
    text = _sr().decode("utf-8", errors="ignore")
    assert "наклон оси позвоночника 6.7° (порог 4.5°)" in text
    assert "Есть нарушения качества на 1 из 1 снимка" in text
    assert dicom_sr.VIOLATION_TEXT["SPINE_AXIS_TILT"] in text


def test_copies_are_described_once_and_all_referenced():
    data = _sr([_finding(copies=2)])
    text = data.decode("utf-8", errors="ignore")
    assert text.count(dicom_sr.CODES["image"][1]) == 1          # один контейнер «Снимок»
    assert b"1.2.3.1\x00" in data and b"1.2.3.2\x00" in data     # обе копии в ссылках
    assert "1 из 1 снимка (2 файла)" in text


def test_deterministic_and_no_findings_rejected():
    assert _sr() == _sr()
    with pytest.raises(ValueError):
        dicom_sr.basic_text_sr([], study_uid="1", series_uid="2", sop_uid="3")


def test_plural_forms():
    got = [dicom_sr.plural(n, "снимок", "снимка", "снимков") for n in (1, 2, 5, 11, 21, 22, 112)]
    assert got == ["снимок", "снимка", "снимков", "снимков", "снимок", "снимка", "снимков"]


def test_strict_pydicom_reads_tree(tmp_path):
    """Строгое чтение pydicom — на пинах (Mac, образ); без pydicom тест пропускается."""
    pydicom = pytest.importorskip("pydicom")
    from pydicom import config
    (tmp_path / "r.sr.dcm").write_bytes(_sr([_finding(copies=2), _finding(q=0, violations=())]))
    old = config.settings.reading_validation_mode
    config.settings.reading_validation_mode = config.RAISE
    try:
        ds = pydicom.dcmread(tmp_path / "r.sr.dcm")
        pydicom.dataset.validate_file_meta(ds.file_meta, enforce_standard=True)
        for _ in ds.iterall():
            pass
        assert ds.SOPClassUID == dicom_sr.SR_SOP_CLASS and ds.Modality == "SR"
        root = ds.ContentSequence
        assert root[0].ConceptNameCodeSequence[0].CodeMeaning == "Заключение по исследованию"
        images = [it for it in root if it.ValueType == "CONTAINER"]
        assert len(images) == 2
        refs = [it for it in images[0].ContentSequence if it.ValueType == "IMAGE"]
        assert len(refs) == 2
        ev = ds.CurrentRequestedProcedureEvidenceSequence[0]
        assert ev.StudyInstanceUID == "1.2.3"
        assert len(ev.ReferencedSeriesSequence[0].ReferencedSOPSequence) == 2
    finally:
        config.settings.reading_validation_mode = old


def test_batch_writes_one_report_per_study_in_the_same_study(tmp_path):
    make_study(tmp_path / "in")
    make_study(tmp_path / "in", name="study_B", with_duplicate=False)
    z = tmp_path / "ov.zip"
    assert run(tmp_path / "in", tmp_path / "r.csv", overlays=z, quiet=True, overlay_format="dicom") == 0
    zf = zipfile.ZipFile(z)
    sr = sorted(n for n in zf.namelist() if n.endswith(".sr.dcm"))
    assert sr == ["study_A/" + batch.SR_NAME, "study_B/" + batch.SR_NAME]
    els = _elements(zf.read(sr[0]))
    assert els[(0x0020, 0x000D)][1].rstrip(b"\x00") == b"1.2.3"          # исследование исходника
    text = zf.read(sr[0]).decode("utf-8", errors="ignore")
    assert "(4 файла)" in text                                           # копия позвоночника учтена
    rows = [r for r in read_rows(tmp_path / "r.csv") if r["study"] == "study_A"]
    bad = sum(1 for r in rows if r["quality_class"] == "1")
    assert ("Есть нарушения" in text) == (bad > 0)
