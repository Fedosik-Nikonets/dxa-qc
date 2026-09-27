"""HTTP API (ТЗ 3.2). Логика — без FastAPI, эндпоинты — через TestClient."""
from __future__ import annotations

import csv
import io
import zipfile

import pytest

from dxa_qc import __version__, service
from test_batch import make_study, read_rows


def _zip_bytes(src) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for p in sorted(src.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(src).as_posix())
    return buf.getvalue()


def _study_zip(tmp_path, names=("A", "B")):
    src = tmp_path / "in"
    for n in names:
        make_study(src, name=n)
    z = tmp_path / "upload.zip"
    z.write_bytes(_zip_bytes(src))
    return z


# ---------------------------------------------------------------- логика

def test_health_reports_models():
    h = service.health_info()
    assert h["status"] == "ok" and h["version"] == __version__
    assert h["models"]["artifact"]["loaded"] and h["models"]["rotation"]["loaded"]
    assert len(h["models"]["rotation"]["sha256"]) == 64


def test_run_batch_table_matches_cli(tmp_path):
    z = _study_zip(tmp_path)
    (tmp_path / "w").mkdir()
    path, media = service.run_batch(z, tmp_path / "w", fmt="csv")
    assert media.startswith("text/csv")
    service.batch.run(z, tmp_path / "cli.csv", quiet=True)
    strip = lambda rows: [{k: v for k, v in r.items() if k != "time_of_processing"} for r in rows]
    assert strip(read_rows(path)) == strip(read_rows(tmp_path / "cli.csv"))


def test_run_batch_with_overlays_bundles_table_and_series(tmp_path):
    z = _study_zip(tmp_path)
    (tmp_path / "w").mkdir()
    path, media = service.run_batch(z, tmp_path / "w", fmt="xlsx", overlays=True)
    assert media == "application/zip"
    names = zipfile.ZipFile(path).namelist()
    assert names == ["overlays.zip", "result.xlsx"]
    inner = zipfile.ZipFile(io.BytesIO(zipfile.ZipFile(path).read("overlays.zip"))).namelist()
    assert len(inner) == 18                                   # 8 PNG, 8 DICOM, 2 отчёта SR
    assert sum(n.endswith(".sr.dcm") for n in inner) == 2
    assert sum(n.endswith(".png") for n in inner) == 8
    assert sum(n.endswith(".overlay.dcm") for n in inner) == 8


@pytest.mark.parametrize("kw", [{"fmt": "pdf"}, {"region_lang": "en"}])
def test_run_batch_rejects_bad_params(tmp_path, kw):
    z = _study_zip(tmp_path)
    with pytest.raises(service.BadRequest):
        service.run_batch(z, tmp_path, **kw)


def test_run_batch_rejects_non_zip(tmp_path):
    p = tmp_path / "x.zip"
    p.write_bytes(b"nope")
    with pytest.raises(service.BadRequest):
        service.run_batch(p, tmp_path)


def test_run_batch_rejects_unsafe_archive(tmp_path):
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr(zipfile.ZipInfo("../evil.dcm"), b"x")
    with pytest.raises(service.BadRequest):
        service.run_batch(z, tmp_path)


def test_run_predict_single_study_files(tmp_path):
    d = make_study(tmp_path / "in", name="X")
    uploads = [(p.name, open(p, "rb")) for p in sorted(d.iterdir())]
    try:
        (tmp_path / "w").mkdir()
        res = service.run_predict(uploads, tmp_path / "w", study="мой")
    finally:
        for _, f in uploads:
            f.close()
    assert res["study"] == "мой" and len(res["rows"]) == 4
    assert {r["study"] for r in res["rows"]} == {"мой"}
    assert all(r["processing_status"] == "Success" for r in res["rows"])
    assert isinstance(res["rows"][0]["quality_class"], int)


def test_run_predict_zip_and_hostile_names(tmp_path):
    z = _study_zip(tmp_path, names=("A",))
    (tmp_path / "w").mkdir()
    with open(z, "rb") as f:
        res = service.run_predict([("../../x.zip", f)], tmp_path / "w", study="../s")
    assert res["study"] == "s" and len(res["rows"]) == 4


def test_run_predict_garbage_file_is_failure_row(tmp_path):
    (tmp_path / "w").mkdir()
    res = service.run_predict([("junk.dcm", io.BytesIO(b"junk"))], tmp_path / "w")
    assert res["rows"][0]["processing_status"] == "Failure"
    assert res["rows"][0]["quality_class"] is None


# ---------------------------------------------------------------- HTTP

def _client():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    return TestClient(service.create_app())


def test_http_health():
    r = _client().get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_http_batch_csv(tmp_path):
    c = _client()
    z = _study_zip(tmp_path)
    r = c.post("/batch?format=csv", files={"file": ("in.zip", z.read_bytes(), "application/zip")})
    assert r.status_code == 200
    rows = list(csv.DictReader(io.StringIO(r.content.decode("utf-8"))))
    assert len(rows) == 8 and {row["study"] for row in rows} == {"A", "B"}


def test_http_batch_xlsx_with_overlays(tmp_path):
    c = _client()
    z = _study_zip(tmp_path)
    r = c.post("/batch?overlays=true", files={"file": ("in.zip", z.read_bytes(), "application/zip")})
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    assert zipfile.ZipFile(io.BytesIO(r.content)).namelist() == ["overlays.zip", "result.xlsx"]


def test_http_batch_bad_input_is_400(tmp_path):
    c = _client()
    r = c.post("/batch", files={"file": ("x.zip", b"not a zip", "application/zip")})
    assert r.status_code == 400
    r = c.post("/batch?format=pdf", files={"file": ("x.zip", b"x", "application/zip")})
    assert r.status_code == 422


def test_openapi_describes_batch_files():
    spec = _client().get("/openapi.json").json()
    resp = spec["paths"]["/batch"]["post"]["responses"]
    assert set(resp["200"]["content"]) == {*service.FORMATS.values(), "application/zip"}
    assert all(c["schema"]["format"] == "binary" for c in resp["200"]["content"].values())
    assert "400" in resp and "422" in resp


def test_http_predict(tmp_path):
    c = _client()
    d = make_study(tmp_path / "in", name="X")
    files = [("files", (p.name, p.read_bytes(), "application/dicom")) for p in sorted(d.iterdir())]
    r = c.post("/predict?study=S1", files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["study"] == "S1" and len(body["rows"]) == 4
