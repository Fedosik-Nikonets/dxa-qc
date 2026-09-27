"""Overlay как DICOM-серия (Secondary Capture) и колонка «проекция»."""
from __future__ import annotations

import csv
import re
import struct
import zipfile

import numpy as np
import pytest

from dxa_qc import _minidicom, dicom_sc, routing
from dxa_qc.batch import run
from dxa_qc.report import ALL_COLUMNS
from synth_dicom import build_dicom
from test_batch import femur_pixels, make_study, read_rows, spine_pixels

CYR = "ISO_IR 144".encode("ascii")


def _sample(**kw) -> bytes:
    img = (np.arange(6 * 5 * 3) * 7 % 256).astype(np.uint8).reshape(6, 5, 3)
    args = dict(study_uid="1.2.3", series_uid=dicom_sc.derive_uid("s"),
                sop_uid=dicom_sc.derive_uid("i"), instance_number=2,
                context={"patient_name": "Anonymized", "study_date": "Anonymized",
                         "patient_birth_date": "19700101", "patient_sex": "M"},
                source_sop_class="1.2.840.10008.5.1.4.1.1.1", source_sop_uid="1.2.3.4",
                comments="quality_class=1; violation_type=SPINE_AXIS_TILT")
    args.update(kw)
    return dicom_sc.secondary_capture(img, **args), img


def _elements(buf: bytes):
    """Элементы верхнего уровня — независимый от ридера разбор Explicit VR LE."""
    pos, out = 132, {}
    while pos < len(buf):
        g, e = struct.unpack_from("<HH", buf, pos)
        vr = buf[pos + 4:pos + 6].decode("ascii")
        if vr in dicom_sc.LONG_VRS:
            ln = struct.unpack_from("<I", buf, pos + 8)[0]
            start = pos + 12
        else:
            ln = struct.unpack_from("<H", buf, pos + 6)[0]
            start = pos + 8
        out[(g, e)] = (vr, buf[start:start + ln])
        pos = start + ln
    return out


def test_roundtrip_through_builtin_reader(tmp_path):
    data, img = _sample()
    (tmp_path / "sc.dcm").write_bytes(data)
    r = _minidicom.read_file(str(tmp_path / "sc.dcm"))
    v = r["values"]
    assert r["TransferSyntaxUID"] == dicom_sc.EXPLICIT_VR_LE
    assert v[(0x0008, 0x0016)] == dicom_sc.SC_SOP_CLASS
    assert (v[(0x0028, 0x0010)], v[(0x0028, 0x0011)], v[(0x0028, 0x0002)]) == (6, 5, 3)
    assert v[(0x0028, 0x0004)] == "RGB" and v[(0x0020, 0x000D)] == "1.2.3"
    px = np.frombuffer(r["pixel_raw"][:6 * 5 * 3], np.uint8).reshape(6, 5, 3)
    assert np.array_equal(px, img[:, :, ::-1])          # BGR -> RGB


def test_structure_even_lengths_sorted_tags_and_group_length():
    data, _ = _sample()
    assert data[128:132] == b"DICM"
    els = _elements(data)
    tags = list(els)
    meta = [t for t in tags if t[0] == 0x0002]
    body = [t for t in tags if t[0] != 0x0002]
    assert meta == sorted(meta) and body == sorted(body)
    assert all(len(raw) % 2 == 0 for _, raw in els.values())
    meta_len = struct.unpack("<I", els[(0x0002, 0x0000)][1])[0]
    first_body = data.index(struct.pack("<HH", 0x0008, 0x0005), 132)
    assert first_body == 132 + 12 + meta_len


def test_invalid_copied_values_become_empty_not_copied():
    data, _ = _sample()
    els = _elements(data)
    assert els[(0x0008, 0x0020)][1] == b""                  # StudyDate «Anonymized»
    assert els[(0x0010, 0x0030)][1] == b"19700101"
    assert els[(0x0010, 0x0010)][1].rstrip() == b"Anonymized"
    assert b"SPINE_AXIS_TILT" in els[(0x0020, 0x4000)][1]
    assert els[(0x0028, 0x0301)][1].rstrip() == b"YES"


def test_clean_rules():
    assert dicom_sc.clean("DA", "Anonymized") == "" and dicom_sc.clean("DA", "20250403") == "20250403"
    assert dicom_sc.clean("TM", "090444") == "090444" and dicom_sc.clean("TM", "9:04") == ""
    assert dicom_sc.clean("CS", "Anonymized") == "" and dicom_sc.clean("CS", "F") == "F"
    assert dicom_sc.clean("SH", "x" * 40) == "x" * 16
    assert dicom_sc.clean("LO", "a\\b") == "a/b"
    assert len(dicom_sc.clean("PN", "Ж" * 60).encode("utf-8")) <= 64


def test_uids_are_valid_and_deterministic():
    u = dicom_sc.derive_uid("a", "b")
    assert re.fullmatch(r"2\.25\.(0|[1-9]\d*)", u) and len(u) <= 64
    assert u == dicom_sc.derive_uid("a", "b") != dicom_sc.derive_uid("a", "c")
    assert _sample()[0] == _sample()[0]


def test_rejects_non_rgb_input():
    with pytest.raises(ValueError):
        dicom_sc.secondary_capture(np.zeros((4, 4), np.uint8), study_uid="1", series_uid="2",
                                   sop_uid="3", instance_number=1)


def test_strict_pydicom_reads_it(tmp_path):
    """Строгое чтение pydicom — на пинах (Mac, образ); без pydicom тест пропускается."""
    pydicom = pytest.importorskip("pydicom")
    from pydicom import config
    data, img = _sample()
    (tmp_path / "sc.dcm").write_bytes(data)
    old = config.settings.reading_validation_mode
    config.settings.reading_validation_mode = config.RAISE
    try:
        ds = pydicom.dcmread(tmp_path / "sc.dcm")
        pydicom.dataset.validate_file_meta(ds.file_meta, enforce_standard=True)
        assert ds.SOPClassUID == dicom_sc.SC_SOP_CLASS
        assert ds.SourceImageSequence[0].ReferencedSOPInstanceUID == "1.2.3.4"
        assert np.array_equal(ds.pixel_array, img[:, :, ::-1])
        assert list(ds.ImageType) == ["DERIVED", "SECONDARY"] and ds.InstanceNumber == 2
    finally:
        config.settings.reading_validation_mode = old


# --------------------------------------------------------------------------- #
# в конвейере
# --------------------------------------------------------------------------- #
def _dicoms(z):
    zf = zipfile.ZipFile(z)
    return {n: _elements(zf.read(n)) for n in zf.namelist() if n.endswith(".overlay.dcm")}


def test_batch_series_lands_in_the_source_study(tmp_path):
    make_study(tmp_path / "in")
    z = tmp_path / "ov.zip"
    assert run(tmp_path / "in", tmp_path / "r.csv", overlays=z, quiet=True) == 0
    rows = {r["path_to_study"]: r for r in read_rows(tmp_path / "r.csv")}
    ds = _dicoms(z)
    assert len(ds) == 4
    txt = lambda e, t: e[t][1].rstrip(b"\x00 ").decode()  # noqa: E731
    assert {txt(e, (0x0020, 0x000D)) for e in ds.values()} == {"1.2.3"}      # то же исследование
    assert len({txt(e, (0x0020, 0x000E)) for e in ds.values()}) == 1        # одна серия
    assert sorted(int(txt(e, (0x0020, 0x0013))) for e in ds.values()) == [1, 2, 3, 4]
    assert len({txt(e, (0x0008, 0x0018)) for e in ds.values()}) == 4        # копия — свой UID
    for name, e in ds.items():
        row = rows[name[:-len(".overlay.dcm")]]
        assert row["image_uid"].encode() in e[(0x0008, 0x2112)][1]          # ссылка на исходник
        assert f"quality_class={row['quality_class']}" in txt(e, (0x0020, 0x4000))


def test_overlay_format_switch(tmp_path):
    make_study(tmp_path / "in")
    for fmt, suffix in (("png", ".png"), ("dicom", ".overlay.dcm")):
        z = tmp_path / f"{fmt}.zip"
        assert run(tmp_path / "in", tmp_path / f"{fmt}.csv", overlays=z, quiet=True,
                   overlay_format=fmt) == 0
        names = zipfile.ZipFile(z).namelist()
        # С DICOM-серией идёт и отчёт SR (один на исследование), с PNG — нет.
        sr = [n for n in names if n.endswith(".sr.dcm")]
        assert len(sr) == (1 if fmt == "dicom" else 0)
        rest = [n for n in names if n not in sr]
        assert len(rest) == 4 and all(n.endswith(suffix) for n in rest)


def test_patient_context_is_recoded_and_kept_out_of_diagnostics(tmp_path):
    d = tmp_path / "in" / "S" / "DXA"
    name = "Иванов^Иван".encode("iso8859_5")
    build_dicom(d / "CR000000.dcm", spine_pixels(), study_uid="1.2.9", sop_uid="1.2.9.1",
                extra_elements={(0x0008, 0x0005): (b"CS", CYR),
                                (0x0010, 0x0010): (b"PN", name),
                                (0x0010, 0x0040): (b"CS", b"F")})
    z = tmp_path / "ov.zip"
    assert run(tmp_path / "in", tmp_path / "r.csv", tmp_path / "diag.csv", overlays=z,
               quiet=True, overlay_format="dicom") == 0
    (e,) = _dicoms(z).values()
    assert e[(0x0010, 0x0010)][1].rstrip().decode("utf-8") == "Иванов^Иван"
    assert e[(0x0008, 0x0005)][1].rstrip() == b"ISO_IR 192"
    diag = (tmp_path / "diag.csv").read_text(encoding="utf-8")
    assert "Иванов" not in diag and "patient" not in diag


def test_projection_column_from_tag_or_protocol(tmp_path):
    d = tmp_path / "in" / "S" / "DXA"
    build_dicom(d / "CR000000.dcm", spine_pixels(), study_uid="1.2.8", sop_uid="1.2.8.1",
                extra_elements={(0x0018, 0x5101): (b"CS", b"PA")})
    build_dicom(d / "CR000001.dcm", femur_pixels(side=+1), study_uid="1.2.8", sop_uid="1.2.8.2")
    assert ALL_COLUMNS[-1] == "projection"
    assert run(tmp_path / "in", tmp_path / "r.csv", tmp_path / "d.csv", quiet=True) == 0
    assert [r["projection"] for r in read_rows(tmp_path / "r.csv")] == ["прямая (PA)", "прямая (AP)"]
    with open(tmp_path / "d.csv", encoding="utf-8", newline="") as fh:
        assert [r["projection_source"] for r in csv.DictReader(fh)] == ["tag", "protocol"]
    assert run(tmp_path / "in", tmp_path / "c.csv", region_lang="code", quiet=True) == 0
    assert [r["projection"] for r in read_rows(tmp_path / "c.csv")] == ["PA", "AP"]


def test_projection_rules():
    assert routing.projection(routing.SPINE, None) == ("AP", "protocol")
    assert routing.projection(routing.FEMUR_LEFT, "ll") == ("LAT", "tag")
    assert routing.projection(routing.SPINE, "OBLIQUE") == ("OTHER", "tag")
    assert routing.projection(routing.UNKNOWN, "AP") == ("", "")
    assert routing.projection_name("", "ru") == "" and routing.projection_name("LAT", "ru") == "боковая (LAT)"


def test_failure_row_has_empty_projection(tmp_path):
    (tmp_path / "in" / "S").mkdir(parents=True)
    (tmp_path / "in" / "S" / "junk.dcm").write_bytes(b"not a dicom at all" * 20)
    run(tmp_path / "in", tmp_path / "r.csv", quiet=True)
    (row,) = read_rows(tmp_path / "r.csv")
    assert row["processing_status"] == "Failure" and row["projection"] == ""
