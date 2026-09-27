"""Вход из zip-архива: кодировка имён, безопасность, уровень каталога исследований."""
from __future__ import annotations

import io
import zipfile

import pytest

from dxa_qc import archive
from dxa_qc.batch import run
from test_batch import make_study, read_rows


def _zip_dir(src, dest, prefix=""):
    with zipfile.ZipFile(dest, "w") as zf:
        for p in sorted(src.rglob("*")):
            if p.is_file():
                zf.write(p, prefix + p.relative_to(src).as_posix())
    return dest


class _LegacyInfo(zipfile.ZipInfo):
    """Пишет имя байтами `raw` без флага UTF-8 — как архиваторы Windows."""

    raw = b""

    def _encodeFilenameFlags(self):
        return self.raw, self.flag_bits & ~0x800


def _legacy_zip(dest, names: dict[bytes, bytes]):
    with zipfile.ZipFile(dest, "w") as zf:
        for raw, data in names.items():
            info = _LegacyInfo(raw.decode("cp437"))
            info.raw = raw
            zf.writestr(info, data)
    return dest


def _cp866_zip(dest, names):
    """Архив как у организатора: имена в cp866, флага UTF-8 нет."""
    return _legacy_zip(dest, {n.encode("cp866"): d for n, d in names.items()})


def test_windows_cyrillic_names_are_decoded(tmp_path):
    z = _cp866_zip(tmp_path / "a.zip", {"Исследования/п1/CR000000_ПОП.dcm": b"x"})
    info = zipfile.ZipFile(z).infolist()[0]
    assert info.flag_bits & 0x800 == 0
    assert archive.decode_name(info) == "Исследования/п1/CR000000_ПОП.dcm"


def test_utf8_names_without_flag_are_decoded(tmp_path):
    z = _legacy_zip(tmp_path / "a.zip", {"Для теста/x.dcm".encode("utf-8"): b"x"})
    assert zipfile.ZipFile(z).infolist()[0].flag_bits & 0x800 == 0
    assert archive.decode_name(zipfile.ZipFile(z).infolist()[0]) == "Для теста/x.dcm"


def test_ascii_and_flagged_names_untouched(tmp_path):
    z = tmp_path / "a.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("plain/x.dcm", b"x")
        zf.writestr("Юникод/y.dcm", b"y")        # zipfile сам ставит флаг UTF-8
    names = [archive.decode_name(i) for i in zipfile.ZipFile(z).infolist()]
    assert names == ["plain/x.dcm", "Юникод/y.dcm"]


@pytest.mark.parametrize("bad", ["../evil.dcm", "a/../../evil.dcm", "/etc/evil", "C:/evil.dcm"])
def test_path_traversal_is_rejected(tmp_path, bad):
    z = tmp_path / "a.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr(zipfile.ZipInfo(bad), b"x")
    with pytest.raises(archive.ArchiveError):
        archive.extract(z, tmp_path / "out")
    assert not (tmp_path / "evil.dcm").exists()


def test_symlink_member_is_rejected(tmp_path):
    z = tmp_path / "a.zip"
    info = zipfile.ZipInfo("link")
    info.create_system = 3
    info.external_attr = (0o120777 << 16)
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr(info, "/etc/passwd")
    with pytest.raises(archive.ArchiveError):
        archive.extract(z, tmp_path / "out")


def test_size_limit(tmp_path):
    z = tmp_path / "a.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("big.bin", b"\x00" * 10_000)
    old = archive.MAX_UNZIP_BYTES
    archive.MAX_UNZIP_BYTES = 1000
    try:
        with pytest.raises(archive.ArchiveError):
            archive.extract(z, tmp_path / "out")
    finally:
        archive.MAX_UNZIP_BYTES = old


def test_not_a_zip(tmp_path):
    p = tmp_path / "x.zip"
    p.write_bytes(b"not a zip")
    with pytest.raises(archive.ArchiveError):
        archive.extract(p, tmp_path / "out")


def test_organizer_style_archive_end_to_end(tmp_path):
    """cp866-имена и обёртка: исследования и пути — по-русски и правильные."""
    src = tmp_path / "in"
    for i in range(15):
        make_study(src, name=f"с{i:02d}")
    members = {}
    for p in sorted(src.rglob("*")):
        if p.is_file():
            members[("Исследования/" + p.relative_to(src).as_posix()).encode("cp866")] = p.read_bytes()
    members["разметка.xlsx".encode("cp866")] = b"xlsx"
    z = _legacy_zip(tmp_path / "nd.zip", members)
    out = tmp_path / "r.csv"
    assert run(z, out, quiet=True) == 0
    rows = [r for r in read_rows(out) if r["path_to_study"] != "разметка.xlsx"]
    assert {r["study"] for r in rows} == {f"с{i:02d}" for i in range(15)}
    assert all(r["processing_status"] == "Success" for r in rows)


def test_wrappers_are_stripped_but_single_study_is_not(tmp_path):
    many = tmp_path / "many" / "НД" / "Исследования"
    for i in range(20):                                   # 20 исследований по 4 файла
        make_study(many, name=f"s{i:02d}")
    (tmp_path / "many" / "НД" / "разметка.xlsx").write_bytes(b"xlsx")
    assert archive.studies_root(tmp_path / "many") == many

    one = tmp_path / "one"
    make_study(one, name="study_A")
    assert archive.studies_root(one) == one               # study_A — исследование, не обёртка


def test_zip_input_equals_directory_input(tmp_path):
    src = tmp_path / "in"
    for name in ("A", "B"):
        make_study(src, name=name)
    z = _zip_dir(src, tmp_path / "in.zip")
    assert run(src, tmp_path / "dir.csv", quiet=True) == 0
    assert run(z, tmp_path / "zip.csv", quiet=True) == 0
    strip = lambda rows: [{k: v for k, v in r.items() if k != "time_of_processing"} for r in rows]
    assert strip(read_rows(tmp_path / "dir.csv")) == strip(read_rows(tmp_path / "zip.csv"))


def test_wrapped_zip_keeps_studies_apart(tmp_path):
    """Обёртка «НД/Исследования» в архиве не склеивает исследования в одно."""
    src = tmp_path / "in"
    for i in range(15):
        make_study(src, name=f"s{i:02d}")
    z = _zip_dir(src, tmp_path / "wrapped.zip", prefix="НД/Исследования/")
    out = tmp_path / "r.csv"
    assert run(z, out, quiet=True) == 0
    rows = read_rows(out)
    assert {r["study"] for r in rows} == {f"s{i:02d}" for i in range(15)}
    assert all(r["path_to_study"].startswith("НД/Исследования/") for r in rows)


def test_bad_archive_gives_error_code_not_crash(tmp_path):
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr(zipfile.ZipInfo("../evil.dcm"), b"x")
    assert run(z, tmp_path / "r.csv", quiet=True) == 2


def test_copy_upload_streams(tmp_path):
    data = b"abc" * 1000
    out = archive.copy_upload(io.BytesIO(data), tmp_path / "u.bin")
    assert out.read_bytes() == data
