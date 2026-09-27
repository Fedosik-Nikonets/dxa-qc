"""Веб-интерфейс (ТЗ 2.6): задания без FastAPI, маршруты — через TestClient на пинах."""
from __future__ import annotations

import io
import os
import time
import zipfile

import pytest

from dxa_qc import webui
from test_batch import make_study, read_rows
from test_service import _study_zip


def _job(tmp_path, **kw):
    z = _study_zip(tmp_path)
    with open(z, "rb") as fh:
        return webui.create_job([("upload.zip", fh)], tmp_path / "jobs", **kw)


def test_zip_job_matches_cli_and_keeps_only_outputs(tmp_path):
    job = _job(tmp_path)
    d = tmp_path / "jobs" / job["job"]
    assert sorted(p.name for p in d.iterdir()) == ["overlays.zip", "result.csv", "result.xlsx"]
    webui.batch.run(tmp_path / "upload.zip", tmp_path / "cli.csv", quiet=True)
    strip = lambda rows: [{k: str(v) for k, v in r.items() if k not in ("time_of_processing", "quality_class")}  # noqa: E731
                          for r in rows]
    assert strip(job["rows"]) == strip(read_rows(tmp_path / "cli.csv"))
    s = job["summary"]
    assert (s["files"], s["studies"], s["success"], s["failures"]) == (8, 2, 8, 0)
    assert {st["study"] for st in s["study_list"]} == {"A", "B"}
    assert job["files"] == ["result.xlsx", "result.csv", "overlays.zip"]
    assert set(job["violations"]) >= {"SPINE_AXIS_TILT", "FEMUR_ROI_COVERAGE"}


def test_dicom_files_of_one_study(tmp_path):
    src = make_study(tmp_path / "in")
    files = sorted(src.glob("*.dcm"))
    handles = [open(p, "rb") for p in files]
    try:
        job = webui.create_job([(p.name, h) for p, h in zip(files, handles)], tmp_path / "jobs")
    finally:
        for h in handles:
            h.close()
    assert len(job["rows"]) == 4 and {r["study"] for r in job["rows"]} == {"study"}


def test_overlay_entry_png_dcm_and_no_escape(tmp_path):
    job = _job(tmp_path)
    root, path = tmp_path / "jobs", job["rows"][0]["path_to_study"]
    png, media = webui.overlay_entry(job["job"], path, "png", root)
    assert png[:4] == b"\x89PNG" and media == "image/png"
    dcm, media = webui.overlay_entry(job["job"], path, "dcm", root)
    assert dcm[128:132] == b"DICM" and media == "application/dicom"
    for bad in ("../../etc/passwd", "/etc/passwd", path + "/../x"):
        with pytest.raises(KeyError):
            webui.overlay_entry(job["job"], bad, "png", root)
    with pytest.raises(KeyError):
        webui.overlay_entry(job["job"], path, "exe", root)


def test_job_file_whitelist_and_job_id_check(tmp_path):
    job = _job(tmp_path)
    root = tmp_path / "jobs"
    p, media = webui.job_file(job["job"], "result.xlsx", root)
    assert p.exists() and "spreadsheet" in media
    for bad_job in ("..", "../jobs", "x" * 32, job["job"].upper()):
        with pytest.raises(KeyError):
            webui.job_file(bad_job, "result.xlsx", root)
    with pytest.raises(KeyError):
        webui.job_file(job["job"], "../result.csv", root)


def test_without_overlays(tmp_path):
    job = _job(tmp_path, overlays=False)
    assert "overlays.zip" not in job["files"] and job["overlays"] is False
    with pytest.raises(KeyError):
        webui.overlay_entry(job["job"], job["rows"][0]["path_to_study"], "png", tmp_path / "jobs")


def test_old_jobs_are_pruned(tmp_path):
    root = tmp_path / "jobs"
    src = make_study(tmp_path / "one", with_duplicate=False)
    ids = []
    for i in range(webui.KEEP_JOBS + 2):
        with open(src / "CR000000.dcm", "rb") as fh:
            ids.append(webui.create_job([("a.dcm", fh)], root, overlays=False)["job"])
        t = time.time() - 100 + i
        os.utime(root / ids[-1], (t, t))
    left = {p.name for p in root.iterdir()}
    assert left == set(ids[-webui.KEEP_JOBS:])


def test_bad_inputs_are_job_errors(tmp_path):
    with pytest.raises(webui.JobError):
        webui.create_job([], tmp_path / "jobs")
    with pytest.raises(webui.JobError):
        webui.create_job([("a.dcm", io.BytesIO(b"x"))], tmp_path / "jobs", region_lang="xx")
    evil = io.BytesIO()
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../evil.dcm", b"x")
    evil.seek(0)
    with pytest.raises(webui.JobError):
        webui.create_job([("evil.zip", evil)], tmp_path / "jobs")
    assert not any((tmp_path / "jobs").iterdir())          # неудачное задание не остаётся


def test_index_is_self_contained():
    html = webui.INDEX_HTML.read_text(encoding="utf-8")
    assert "<title>" in html and "/ui/jobs" in html
    for external in ("http://", "https://", "//cdn", "fonts.googleapis"):
        assert external not in html                       # контейнер работает без сети
    assert "innerHTML" not in html                        # данные — только через textContent


# ---------------------------------------------------------------- HTTP

def _client():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from dxa_qc import service
    return TestClient(service.create_app())


def test_http_index_and_full_flow(tmp_path):
    c = _client()
    old = os.environ.get("DXA_QC_JOBS")
    os.environ["DXA_QC_JOBS"] = str(tmp_path / "jobs")
    try:
        _flow(c, tmp_path)
    finally:
        if old is None:
            os.environ.pop("DXA_QC_JOBS", None)
        else:
            os.environ["DXA_QC_JOBS"] = old


def _flow(c, tmp_path):
    r = c.get("/")
    assert r.status_code == 200 and "DXA QC" in r.text
    z = _study_zip(tmp_path)
    r = c.post("/ui/jobs", files={"files": ("in.zip", z.read_bytes(), "application/zip")})
    assert r.status_code == 200, r.text
    job = r.json()
    assert len(job["rows"]) == 8
    x = c.get(f"/ui/jobs/{job['job']}/files/result.xlsx")
    assert x.status_code == 200 and x.content[:2] == b"PK"
    path = job["rows"][0]["path_to_study"]
    png = c.get(f"/ui/jobs/{job['job']}/overlay", params={"path": path})
    assert png.status_code == 200 and png.headers["content-type"] == "image/png"
    dcm = c.get(f"/ui/jobs/{job['job']}/overlay", params={"path": path, "kind": "dcm"})
    assert dcm.status_code == 200 and "attachment" in dcm.headers["content-disposition"]
    assert c.get(f"/ui/jobs/{job['job']}/files/secret.txt").status_code == 404
    assert c.get(f"/ui/jobs/{job['job']}/overlay", params={"path": "../x"}).status_code == 404
    assert c.post("/ui/jobs", files={"files": ("x.zip", b"PK\x03\x04broken", "application/zip")}).status_code in (200, 400)
