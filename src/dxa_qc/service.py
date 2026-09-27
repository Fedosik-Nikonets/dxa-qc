"""HTTP API для пакетной обработки (ТЗ 3.2: «API для пакетной обработки
тестового набора»).

    python -m dxa_qc.service --host 0.0.0.0 --port 8000

Эндпоинты
---------
* `GET  /health` — жив ли сервис, версия, какие файлы моделей загружены.
* `POST /batch` — zip-архив с исследованиями (поле `file`) → таблица ТЗ 2.5.
  Параметры запроса: `format=xlsx|csv`, `overlays=true|false`,
  `overlay_format=both|dicom|png`, `region_lang=ru|code`. Без оверлеев
  ответ — сам файл таблицы; с оверлеями — zip с `result.<format>` и
  `overlays.zip` (ТЗ 2.7: «общая таблица и zip-архив с дополнительными
  сериями»): на каждый файл PNG и/или DICOM Secondary Capture в том же
  исследовании.
* `POST /predict` — файлы одного исследования (поле `files`, один или
  несколько DICOM либо zip) → JSON со строками таблицы. Для демонстрации
  и интеграции; для закрытого набора — `/batch` или CLI.

Устройство
----------
Сервис — тонкая оболочка над `batch.run`: тот же конвейер, те же строки,
та же обработка ошибок, что и в командной строке. Загрузка пишется на
диск потоком, а не читается в память; всё временное удаляется после
ответа. Прогоны сериализуются одной блокировкой: конвейер загружает CPU
целиком, а два параллельных прогона только делили бы его и удваивали
память. Внешних обращений нет ни одного (ТЗ 3.2): модели — файлы в
образе, контейнер запускается с `--network none`.

Логика вынесена в функции без зависимости от FastAPI (`health_info`,
`run_batch`, `run_predict`) — они тестируются там, где FastAPI нет.

Здесь сознательно нет `from __future__ import annotations`: FastAPI читает
аннотации параметров эндпоинтов во время выполнения, а `UploadFile`
импортируется внутри `create_app`. Со строковыми аннотациями FastAPI ищет
`UploadFile` в глобалах модуля, не находит и падает при создании
приложения (так и было, поймали HTTP-тесты на пинах).
"""

import argparse
import csv
import hashlib
import re
import shutil
import tempfile
import threading
import zipfile
from pathlib import Path

from . import __version__
from . import archive as archive_io
from . import score as score_rules
from . import artifact as artifact_rules
from . import batch
from . import rotation_rule
from .overlay import ZIP_DATE

FORMATS = {"xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
           "csv": "text/csv; charset=utf-8"}
_RUN_LOCK = threading.Lock()


_FILE = {"schema": {"type": "string", "format": "binary"}}
# Описание ответа /batch для Swagger: без него /docs показывает у файла
# ответа пример "string" в application/json.
BATCH_RESPONSES = {
    200: {"description": "Таблица ТЗ 2.5 в формате `format` (xlsx или csv). При `overlays=true` — "
                         "zip с `result.<format>` и `overlays.zip` (overlay-серия).",
          "content": {FORMATS["xlsx"]: _FILE, FORMATS["csv"]: _FILE, "application/zip": _FILE}},
    400: {"description": "Вход не принят: не zip-архив, небезопасные пути в архиве или превышен "
                         "предел распаковки. Причина — в поле `detail`."},
}


class BadRequest(ValueError):
    """Ошибка во входных данных запроса — ответ 400, а не 500."""


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def health_info() -> dict:
    art, rot = artifact_rules.DEFAULT_MODEL_PATH, rotation_rule.DEFAULT_MODEL_PATH
    qs = score_rules.DEFAULT_PARAMS_PATH
    return {
        "status": "ok",
        "version": __version__,
        "models": {
            "artifact": {"file": art.name, "loaded": artifact_rules.load_default() is not None,
                         "sha256": _sha256(art)},
            "rotation": {"file": rot.name, "loaded": rotation_rule.load_default() is not None,
                         "sha256": _sha256(rot)},
            "quality_score": {"file": qs.name, "loaded": score_rules.load_params() is not None,
                              "sha256": _sha256(qs)},
        },
    }


def _safe_name(name: str | None, default: str) -> str:
    """Имя загруженного файла без каталогов и управляющих символов."""
    base = Path((name or "").replace("\\", "/")).name
    base = re.sub(r"[\x00-\x1f]", "", base).strip()
    return base if base not in ("", ".", "..") else default


def run_batch(archive_path: Path, workdir: Path, fmt: str = "xlsx", overlays: bool = False,
              region_lang: str = "ru", overlay_format: str = "both") -> tuple[Path, str]:
    """Архив → файл ответа и его media type. Файл лежит в `workdir`."""
    if fmt not in FORMATS:
        raise BadRequest(f"format должен быть одним из {sorted(FORMATS)}")
    if region_lang not in ("ru", "code"):
        raise BadRequest("region_lang должен быть ru или code")
    if overlay_format not in batch.OVERLAY_FORMATS:
        raise BadRequest(f"overlay_format должен быть одним из {list(batch.OVERLAY_FORMATS)}")
    if not zipfile.is_zipfile(archive_path):
        raise BadRequest("ожидается zip-архив с исследованиями")
    table = workdir / f"result.{fmt}"
    ov = workdir / "overlays.zip" if overlays else None
    with _RUN_LOCK:
        code = batch.run(archive_path, table, region_lang=region_lang, quiet=True, overlays=ov,
                         overlay_format=overlay_format)
    if code != 0 or not table.exists():
        raise BadRequest("архив не удалось обработать (небезопасные пути или превышен размер)")
    if not overlays:
        return table, FORMATS[fmt]
    bundle = workdir / "result_bundle.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted((table, ov)):
            info = zipfile.ZipInfo(p.name, date_time=ZIP_DATE)
            info.compress_type = zipfile.ZIP_STORED if p.suffix == ".zip" else zipfile.ZIP_DEFLATED
            info.create_system, info.external_attr = 3, 0o644 << 16
            # Потоком, а не read_bytes(): архив overlay на большом наборе — сотни МБ.
            with open(p, "rb") as src, zf.open(info, "w") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
    return bundle, "application/zip"


def run_predict(uploads: list[tuple[str, object]], workdir: Path, study: str = "study",
                region_lang: str = "ru") -> dict:
    """Файлы одного исследования → строки таблицы в JSON-виде.

    `uploads` — пары (имя, поток). Один zip разворачивается как есть,
    иначе файлы кладутся в каталог исследования `study`.
    """
    if not uploads:
        raise BadRequest("не передано ни одного файла")
    study = _safe_name(study, "study")
    src = workdir / "in"
    first = workdir / _safe_name(uploads[0][0], "upload")
    if len(uploads) == 1:
        archive_io.copy_upload(uploads[0][1], first)
        if zipfile.is_zipfile(first) and not batch.is_dicom(first):
            try:
                archive_io.extract(first, src / study)
            except archive_io.ArchiveError as exc:
                raise BadRequest(str(exc)) from exc
        else:
            (src / study).mkdir(parents=True)
            shutil.move(str(first), src / study / first.name)
    else:
        (src / study).mkdir(parents=True)
        seen: set[str] = set()
        for i, (name, stream) in enumerate(uploads):
            n = _safe_name(name, f"file_{i:03d}")
            if n in seen:
                n = f"{i:03d}_{n}"
            seen.add(n)
            archive_io.copy_upload(stream, src / study / n)
    out = workdir / "result.csv"
    with _RUN_LOCK:
        batch.run(src, out, region_lang=region_lang, quiet=True)
    with open(out, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:   # в JSON числа — числами, пустое — null
        r["quality_class"] = int(r["quality_class"]) if r["quality_class"] != "" else None
        r["quality_score"] = float(r["quality_score"]) if r.get("quality_score", "") != "" else None
        r["time_of_processing"] = float(r["time_of_processing"])
    study_class = max((r["quality_class"] or 0 for r in rows), default=0)
    return {"study": study, "rows": rows, "quality_class": study_class,
            "quality_score": score_rules.study_score([r["quality_score"] for r in rows], study_class)}


def create_app():
    """FastAPI-приложение. Импорт FastAPI — только здесь."""
    from urllib.parse import quote

    from fastapi import FastAPI, File, HTTPException, Query, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse, Response
    from starlette.background import BackgroundTask

    from . import webui

    app = FastAPI(title="DXA QC", version=__version__,
                  description="Оценка качества денситометрических исследований (ТЗ 2.3–2.7)")

    @app.get("/health")
    def health() -> dict:
        return health_info()

    # --- веб-интерфейс (ТЗ 2.6), страница — один файл без внешних ресурсов ---
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index():
        return HTMLResponse(webui.INDEX_HTML.read_text(encoding="utf-8"))

    @app.post("/ui/jobs", tags=["веб-интерфейс"])
    def ui_create_job(files: list[UploadFile] = File(..., description="zip-архив или DICOM-файлы одного исследования"),
                      overlays: bool = Query(True),
                      region_lang: str = Query("ru", pattern="^(ru|code)$")) -> dict:
        try:
            return webui.create_job([(f.filename, f.file) for f in files], overlays=overlays,
                                    region_lang=region_lang, lock=_RUN_LOCK)
        except (webui.JobError, archive_io.ArchiveError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/ui/jobs/{job}/files/{name}", tags=["веб-интерфейс"])
    def ui_job_file(job: str, name: str):
        try:
            path, media = webui.job_file(job, name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc
        return FileResponse(path, media_type=media, filename=name)

    @app.get("/ui/jobs/{job}/overlay", tags=["веб-интерфейс"])
    def ui_overlay(job: str, path: str = Query(..., description="path_to_study из таблицы"),
                   kind: str = Query("png", pattern="^(png|dcm)$")):
        try:
            data, media = webui.overlay_entry(job, path, kind)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc.args[0])) from exc
        headers = {}
        if kind == "dcm":
            name = Path(path).name + ".overlay.dcm"
            headers["Content-Disposition"] = f"attachment; filename*=UTF-8''{quote(name)}"
        return Response(data, media_type=media, headers=headers)

    @app.post("/batch", response_class=FileResponse, responses=BATCH_RESPONSES,
              summary="Пакетная обработка zip-архива: таблица и overlay-серия")
    def batch_endpoint(file: UploadFile = File(..., description="zip-архив с исследованиями"),
                       format: str = Query("xlsx", pattern="^(xlsx|csv)$"),
                       overlays: bool = Query(False),
                       overlay_format: str = Query("both", pattern="^(both|dicom|png)$",
                                                   description="both — PNG и DICOM, dicom, png"),
                       region_lang: str = Query("ru", pattern="^(ru|code)$")):
        tmp = Path(tempfile.mkdtemp(prefix="dxa-qc-api-"))
        try:
            src = archive_io.copy_upload(file.file, tmp / "upload.zip")
            path, media = run_batch(src, tmp, format, overlays, region_lang, overlay_format)
        except BadRequest as exc:
            shutil.rmtree(tmp, ignore_errors=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return FileResponse(path, media_type=media, filename=path.name,
                            background=BackgroundTask(shutil.rmtree, tmp, True))

    @app.post("/predict")
    def predict_endpoint(files: list[UploadFile] = File(..., description="DICOM-файлы одного исследования или zip"),
                         study: str = Query("study"),
                         region_lang: str = Query("ru", pattern="^(ru|code)$")) -> dict:
        with tempfile.TemporaryDirectory(prefix="dxa-qc-api-") as tmp:
            try:
                return run_predict([(f.filename, f.file) for f in files], Path(tmp), study, region_lang)
            except BadRequest as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dxa_qc.service", description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)
    import uvicorn
    uvicorn.run(create_app(), host=args.host, port=args.port, workers=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
