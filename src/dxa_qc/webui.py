"""Веб-интерфейс (ТЗ 2.6): загрузка, пакетная обработка и просмотр результата.

Страница `web/index.html` — один файл без внешних ресурсов (контейнер
работает с `--network none`, CDN недоступны). Она обращается к трём
эндпоинтам, которые добавляет `service.create_app`:

* `POST /ui/jobs` — zip-архив с исследованиями или DICOM-файлы одного
  исследования → задание: таблица в JSON, сводка, ссылки на файлы;
* `GET /ui/jobs/{job}/files/{name}` — `result.xlsx`, `result.csv`, `overlays.zip`;
* `GET /ui/jobs/{job}/overlay?path=...&kind=png|dcm` — overlay одного файла.

Задание — каталог во временной папке: таблица и архив overlay. Входные
файлы удаляются сразу после прогона, хранятся последние `KEEP_JOBS`
заданий. Конвейер тот же `batch.run`, что у CLI и `/batch`, поэтому таблица
в интерфейсе совпадает с таблицей командной строки.

Логика — без FastAPI: тестируется там, где его нет.
"""
from __future__ import annotations

import csv
import os
import re
import shutil
import tempfile
import time
import uuid
import zipfile
from collections import Counter
from pathlib import Path

from . import __version__
from . import archive as archive_io
from . import batch
from .report import ALL_COLUMNS, write_xlsx

KEEP_JOBS = 5
JOB_ID = re.compile(r"[0-9a-f]{32}")
FILES = {
    "result.xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "result.csv": "text/csv; charset=utf-8",
    "overlays.zip": "application/zip",
}
OVERLAY_KINDS = {"png": (".png", "image/png"), "dcm": (".overlay.dcm", "application/dicom")}
INDEX_HTML = Path(__file__).resolve().parent / "web" / "index.html"

# Подписи нарушений — из taxonomy.yaml (PyYAML в образе нет, поэтому здесь).
VIOLATIONS = {
    "SPINE_POSITIONING": ("Укладка", "Некорректная укладка: в кадр не попали верхние края "
                          "подвздошных костей и/или половина тела Th12"),
    "SPINE_AXIS_TILT": ("Наклон оси", "Наклон оси позвоночника более 5°"),
    "SPINE_ARTIFACT": ("Артефакт / наложение", "Инородные тела, выраженные артефакты или наложения"),
    "FEMUR_POSITIONING_ROTATION": ("Ротация бедра", "Некорректное позиционирование или ротация "
                                   "(пере-/недоротация по малому вертелу)"),
    "FEMUR_ROI_COVERAGE": ("Охват области интереса", "Недостаточный охват области интереса "
                           "(менее 3 см сверху/снизу, 2 см сбоку)"),
    "OUT_OF_SCOPE_FINDING": ("Находка вне перечня", "Находка вне перечня ТЗ 2.3, требует внимания врача"),
}


class JobError(ValueError):
    """Ошибка во входных данных задания — ответ 400."""


def jobs_root() -> Path:
    return Path(os.environ.get("DXA_QC_JOBS", Path(tempfile.gettempdir()) / "dxa-qc-jobs"))


def _safe_name(name: str | None, default: str) -> str:
    base = Path((name or "").replace("\\", "/")).name
    base = "".join(ch for ch in base if ch.isprintable() and ch not in '<>:"|?*')
    return base if base not in ("", ".", "..") else default


def _stage(uploads: list[tuple[str, object]], work: Path) -> Path:
    """Загрузки на диск: один zip — как есть (его разберёт `batch.run`),
    иначе DICOM-файлы одного исследования — в каталог `study`."""
    if not uploads:
        raise JobError("не передано ни одного файла")
    staged = work / "in"
    staged.mkdir()
    if len(uploads) == 1:
        dest = staged / _safe_name(uploads[0][0], "upload")
        archive_io.copy_upload(uploads[0][1], dest)
        if zipfile.is_zipfile(dest) and not batch.is_dicom(dest):
            return dest
        study = staged / "study"
        study.mkdir()
        shutil.move(str(dest), study / dest.name)
        return staged
    study = staged / "study"
    study.mkdir()
    seen: set[str] = set()
    for i, (name, stream) in enumerate(uploads):
        n = _safe_name(name, f"file_{i:03d}")
        if n in seen:
            n = f"{i:03d}_{n}"
        seen.add(n)
        archive_io.copy_upload(stream, study / n)
    return staged


def _records(csv_path: Path) -> list[dict]:
    with open(csv_path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["quality_class"] = int(r["quality_class"]) if r["quality_class"] != "" else None
        r["quality_score"] = float(r["quality_score"]) if r.get("quality_score", "") != "" else None
        r["time_of_processing"] = float(r["time_of_processing"])
    return rows


def _summary(rows: list[dict], seconds: float) -> dict:
    ok = [r for r in rows if r["processing_status"] == "Success"]
    studies: dict[str, dict] = {}
    for r in rows:
        s = studies.setdefault(r["study"], {"study": r["study"], "files": 0, "quality_class": 0,
                                            "failures": 0, "violations": set()})
        s["files"] += 1
        if r["processing_status"] != "Success":
            s["failures"] += 1
        elif r["quality_class"]:
            s["quality_class"] = 1
            s["violations"].update(v for v in r["violation_type"].split(";") if v)
    counts = Counter(v for r in ok for v in r["violation_type"].split(";") if v)
    return {
        "files": len(rows),
        "studies": len(studies),
        "success": len(ok),
        "failures": len(rows) - len(ok),
        "with_violations": sum(1 for r in ok if r["quality_class"] == 1),
        "without_violations": sum(1 for r in ok if r["quality_class"] == 0),
        "studies_with_violations": sum(1 for s in studies.values() if s["quality_class"]),
        "violation_counts": dict(sorted(counts.items())),
        "seconds": round(seconds, 2),
        "seconds_per_study": round(seconds / max(len(studies), 1), 3),
        "study_list": [{**s, "violations": sorted(s["violations"])} for s in studies.values()],
    }


def _prune(root: Path, keep: int = KEEP_JOBS) -> None:
    jobs = sorted((p for p in root.iterdir() if p.is_dir() and JOB_ID.fullmatch(p.name)),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    for old in jobs[keep:]:
        shutil.rmtree(old, ignore_errors=True)


def create_job(uploads: list[tuple[str, object]], root: Path | None = None, *,
               overlays: bool = True, region_lang: str = "ru", lock=None) -> dict:
    """Загрузки → готовое задание: JSON для страницы и файлы на диске."""
    if region_lang not in ("ru", "code"):
        raise JobError("region_lang должен быть ru или code")
    root = Path(root or jobs_root())
    root.mkdir(parents=True, exist_ok=True)
    job = uuid.uuid4().hex
    work = root / job
    work.mkdir()
    try:
        src = _stage(uploads, work)
        table = work / "result.csv"
        ov = work / "overlays.zip" if overlays else None
        t0 = time.perf_counter()
        if lock is not None:
            with lock:
                code = batch.run(src, table, region_lang=region_lang, quiet=True, overlays=ov)
        else:
            code = batch.run(src, table, region_lang=region_lang, quiet=True, overlays=ov)
        seconds = time.perf_counter() - t0
        if code != 0 or not table.exists():
            raise JobError("архив не удалось обработать (небезопасные пути или превышен размер)")
        rows = _records(table)
        write_xlsx(rows, work / "result.xlsx", columns=ALL_COLUMNS)
    except Exception:
        shutil.rmtree(work, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(work / "in", ignore_errors=True)    # входные файлы не храним
    _prune(root)
    return {
        "job": job,
        "version": __version__,
        "overlays": bool(overlays),
        "rows": rows,
        "summary": _summary(rows, seconds),
        "violations": {k: {"short": a, "text": b} for k, (a, b) in VIOLATIONS.items()},
        "files": [n for n in FILES if (work / n).exists()],
    }


def _job_dir(root: Path | None, job: str) -> Path:
    if not JOB_ID.fullmatch(job or ""):
        raise KeyError("неизвестное задание")
    d = Path(root or jobs_root()) / job
    if not d.is_dir():
        raise KeyError("задание не найдено (хранятся последние пять)")
    return d


def job_file(job: str, name: str, root: Path | None = None) -> tuple[Path, str]:
    """Файл задания по имени из белого списка; KeyError — 404."""
    if name not in FILES:
        raise KeyError("нет такого файла")
    p = _job_dir(root, job) / name
    if not p.exists():
        raise KeyError("файл не создан в этом задании")
    return p, FILES[name]


def overlay_entry(job: str, path: str, kind: str = "png",
                  root: Path | None = None) -> tuple[bytes, str]:
    """Overlay одного файла из архива задания. Имя ищется среди записей
    архива, а не на диске, — выйти за пределы архива нельзя."""
    if kind not in OVERLAY_KINDS:
        raise KeyError("kind должен быть png или dcm")
    suffix, media = OVERLAY_KINDS[kind]
    z = _job_dir(root, job) / "overlays.zip"
    if not z.exists():
        raise KeyError("overlay в этом задании не строились")
    with zipfile.ZipFile(z) as zf:
        try:
            return zf.read(path + suffix), media
        except KeyError:
            raise KeyError("для этого файла overlay нет") from None
