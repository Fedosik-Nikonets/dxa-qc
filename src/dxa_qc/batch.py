"""Сквозной прогон: каталог с исследованиями -> таблица по ТЗ 2.5.

    python -m dxa_qc.batch --input Исследования --output result.xlsx [--overlays overlays.zip]

Предсказания с 18 сентября настоящие для позвоночника: два геометрических
правила ТЗ 2.3.1 из трёх (наклон оси, укладка) считаются из маски кости
в `predict`, а третье — артефакты — детектором на признаке top-hat.
С 19 сентября у бедра считается охват области интереса (ТЗ 2.3.2), с 23-го —
ротация по профилю медиального края (`rotation_rule`). Всё остальное
настоящее с 17-го: обход каталога, чтение DICOM со всеми краевыми случаями,
определение анатомической области, обработка ошибок, формат и порядок
колонок, детерминированная запись.

Что здесь важно и не очевидно
-----------------------------

*Ключ исследования — имя каталога, а не StudyInstanceUID.* В обучающем наборе
идентификатор из `разметка.xlsx` совпадает с именем каталога верхнего уровня
100 раз из 100, а StudyInstanceUID (0020,000D) внутри файлов принадлежит другой
серии идентификаторов и с разметкой не пересекается вообще. Тег при этом
пишется в колонку `study_uid`, как требует ТЗ 2.5. Вопрос задан организатору,
п. 3.

*Строка на каждый файл, дедупликация — только внутри конвейера.* Половина
файлов набора — точные копии. Роутер и предсказание считаются один раз на
уникальный кадр внутри исследования, а копии получают тот же результат:
это экономит время обработки, но не меняет состав таблицы. С появлением
сегментации экономия стала существенной: маска кости дороже чтения файла.

*Ни одного исключения наружу.* Любая ошибка на любом файле даёт строку
с `processing_status=Failure` и текстом в `violation_type`. Обработка
следующих файлов продолжается (ТЗ 2.7).

*Имя анатомической области — по-русски, как в ТЗ 2.3.* В каком виде эту
колонку ждёт организатор, неизвестно: единого словаря нет ни в ТЗ, ни в
разметке. Русское название совпадает с текстом ТЗ, и если колонку смотрит
человек, оно читается без словаря. Ключ `--region-lang code` даёт короткие
коды `LUMBAR_SPINE` / `LEFT_PROXIMAL_FEMUR` / `RIGHT_PROXIMAL_FEMUR`.

*Время обработки* в строке — это время на сам файл (чтение) плюс его доля
от работы уровня исследования (роутер и предсказание, которые считаются на
уникальных кадрах). Сумма по колонке близка к общему времени прогона, а не
занижена, и копии не выглядят бесплатными.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import time
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import archive as archive_io
from .dicom_io import DicomImage, is_dicom, iter_dicom_files, read_dicom, stable_error
from .report import (
    STATUS_FAILURE,
    STATUS_SUCCESS,
    ReportRow,
    write_diagnostics,
    write_report,
)
from .routing import (
    FEMUR_LEFT,
    FEMUR_RIGHT,
    RegionFeatures,
    SPINE,
    UNKNOWN,
    image_features,
    projection,
    projection_name,
    region_name,
    route_study,
)
from .segmentation import segment_bone
from . import overlay as overlay_series
from . import dicom_sc
from . import dicom_sr
from . import artifact as artifact_rules
from . import femur as femur_rules
from . import rotation_rule as rotation_rules
from . import spine as spine_rules
from . import score as score_rules


@dataclass
class _FileState:
    """Промежуточное состояние одного файла между проходами."""

    path: Path
    rel: str
    study: str
    seconds: float = 0.0
    img: DicomImage | None = None
    error: str | None = None
    md5: str | None = None
    feats: RegionFeatures = field(default_factory=RegionFeatures)
    region: str = UNKNOWN
    region_confident: bool = False
    region_reason: str | None = None


def study_key(path: Path, root: Path) -> str:
    """Идентификатор исследования: первый компонент пути относительно корня."""
    try:
        rel = path.relative_to(root)
    except ValueError:
        return path.parent.name or path.stem
    return rel.parts[0] if len(rel.parts) > 1 else rel.stem


def pixel_md5(pixels: np.ndarray | None) -> str | None:
    """Хэш пикселей для дедупликации внутри исследования."""
    if pixels is None:
        return None
    try:
        return hashlib.md5(np.ascontiguousarray(pixels).tobytes()).hexdigest()
    except Exception:  # noqa: BLE001
        return None


def predict(region: str, img: DicomImage) -> tuple[int, str, dict]:
    """Предсказание по одному кадру: `quality_class`, `violation_type`, диагностика.

    Контракт для всех, кто сюда встраивается: `violation_type` — перечень
    идентификаторов нарушений из taxonomy.yaml через ';', пустая строка при
    их отсутствии; `quality_class` — 1, если перечень непуст. Третьим
    значением идут служебные колонки, они попадают только в диагностический
    csv, а в таблицу ТЗ 2.5 — нет.

    Позвоночник заполнен целиком: два геометрических правила (наклон оси,
    укладка) и детектор артефактов на признаке top-hat. Бедро с 23 сентября
    тоже целиком: охват области интереса по длине поля сканирования и
    ротация — два хвоста по профилю медиального края (`rotation_rule`).
    Отказ правила ротации нарушения не ставит и виден в диагностике.

    Веса детектора артефактов и границы правила ротации не обязательны:
    если файла нет, проверка пропускается, а прогон идёт дальше. Иначе
    любая машина без файлов модели не смогла бы получить таблицу вообще.

    Маска кости считается здесь один раз и передаётся правилам: это самая
    дорогая операция конвейера, а нужна она и правилам, и будущим
    классификаторам.
    """
    if region == SPINE:
        bone = segment_bone(img.pixels)
        geom = spine_rules.evaluate(img.pixels, img.mm_per_px, bone=bone)
        art = artifact_rules.evaluate(img.pixels, bone.mask if bone.ok else None,
                                      img.mm_per_px)
        violations = geom.violations + art.violations
        diag = {**geom.diagnostics(), **art.diagnostics()}
        return _with_score(region, violations, diag)
    if region in (FEMUR_LEFT, FEMUR_RIGHT):
        bone = segment_bone(img.pixels)
        geom = femur_rules.evaluate(img.pixels, region, img.mm_per_px, bone=bone,
                                    mm_per_px_source=img.mm_per_px_source,
                                    model=img.model)
        rot = rotation_rules.evaluate(img.pixels, region, img.mm_per_px,
                                      mm_per_px_source=img.mm_per_px_source,
                                      scanner_model=img.model)
        # Порядок как в taxonomy.yaml: сначала ротация, затем охват.
        violations = rot.violations + geom.violations
        diag = {**geom.diagnostics(), **rot.diagnostics()}
        return _with_score(region, violations, diag)
    return 0, "", {}


def _with_score(region: str, violations: list[str], diag: dict) -> tuple[int, str, dict]:
    """Решение кадра и балл нарушения `quality_score` (модуль score)."""
    quality = 1 if violations else 0
    model = artifact_rules.load_default()
    diag["quality_score"] = score_rules.quality_score(
        region, diag, quality, artifact_threshold=model.threshold if model else None)
    return quality, ";".join(violations), diag


def _read_one(state: _FileState) -> None:
    """Первый проход по файлу: чтение и признаки. Исключений не выбрасывает."""
    t0 = time.perf_counter()
    try:
        if not is_dicom(state.path):
            state.error = "файл не является DICOM (нет префикса DICM и разбор не удался)"
        else:
            img = read_dicom(state.path, with_pixels=True)
            state.img = img
            if not img.ok:
                state.error = img.error or "не удалось прочитать файл"
            elif img.pixels is None:
                state.error = "в файле нет пиксельных данных"
            else:
                state.md5 = pixel_md5(img.pixels)
    except Exception as exc:  # noqa: BLE001 — наружу не выпускаем ничего
        state.error = f"{type(exc).__name__}: {exc}"
    state.seconds += time.perf_counter() - t0


def _route_studies(states: list[_FileState]) -> dict[str, float]:
    """Второй проход: область для каждого уникального кадра исследования."""
    by_study: dict[str, list[_FileState]] = defaultdict(list)
    for s in states:
        by_study[s.study].append(s)

    route_seconds: dict[str, float] = {}
    for study, group in by_study.items():
        t0 = time.perf_counter()
        # Представитель на каждый уникальный кадр; остальные наследуют решение.
        reps: dict[str, _FileState] = {}
        order: list[_FileState] = []
        for s in group:
            if s.error is not None or s.md5 is None:
                continue
            if s.md5 not in reps:
                reps[s.md5] = s
                order.append(s)

        for s in order:
            s.feats = image_features(s.img.pixels if s.img else None)

        decisions = route_study([s.feats for s in order])
        for s, d in zip(order, decisions):
            s.region, s.region_confident, s.region_reason = d.region, d.confident, d.reason

        for s in group:
            if s.error is not None or s.md5 is None:
                continue
            rep = reps[s.md5]
            if rep is not s:
                s.feats = rep.feats
                s.region = rep.region
                s.region_confident = rep.region_confident
                s.region_reason = rep.region_reason

        route_seconds[study] = time.perf_counter() - t0
    return route_seconds


MAX_FRAMES_PER_STUDY = 3   # ТЗ 2.4: не более трёх изображений в исследовании


def _warn_crowded_studies(states: list[_FileState], log) -> None:
    """Предупреждение, если «исследование» подозрительно велико.

    Роутер считает, что позвоночник в исследовании один. Больше трёх
    уникальных кадров — признак того, что уровень каталога указан неверно
    и несколько исследований слились в одно.
    """
    frames: dict[str, set] = defaultdict(set)
    for s in states:
        if s.md5:
            frames[s.study].add(s.md5)
    crowded = {k: len(v) for k, v in frames.items() if len(v) > MAX_FRAMES_PER_STUDY}
    for study, n in sorted(crowded.items())[:5]:
        log(f"  внимание: исследование {study!r} — {n} уникальных кадров, больше "
            f"{MAX_FRAMES_PER_STUDY} (ТЗ 2.4); проверьте, что --input указывает на каталог "
            "с папками исследований")


def _predict_frames(states: list[_FileState]) -> tuple[dict, dict[str, float]]:
    """Третий проход: предсказание — один раз на уникальный кадр исследования.

    Половина файлов набора — точные копии, а сегментация кости дороже всего
    остального вместе взятого. Копии наследуют результат представителя;
    состав таблицы от этого не меняется, меняется только время.
    """
    cache: dict[tuple[str, str], tuple[int | None, str, dict, str | None]] = {}
    seconds: dict[str, float] = defaultdict(float)
    for s in states:
        if s.error is not None or s.md5 is None or s.region == UNKNOWN:
            continue
        key = (s.study, s.md5)
        if key in cache:
            continue
        t0 = time.perf_counter()
        try:
            quality, violation, diag = predict(s.region, s.img)
            err = None
        except Exception as exc:  # noqa: BLE001 — наружу не выпускаем ничего
            quality, violation, diag = None, "", {}
            err = f"ошибка предсказания: {type(exc).__name__}: {exc}"
        seconds[s.study] += time.perf_counter() - t0
        cache[key] = (quality, violation, diag, err)
    return cache, seconds


def _build_rows(states: list[_FileState], route_seconds: dict[str, float],
                predict_seconds: dict[str, float], cache: dict,
                region_lang: str) -> list[ReportRow]:
    """Четвёртый проход: сборка строк выходной таблицы."""
    per_study_files: dict[str, int] = defaultdict(int)
    for s in states:
        per_study_files[s.study] += 1

    rows: list[ReportRow] = []
    for s in states:
        img = s.img
        error = s.error
        quality: int | None = None
        violation = ""
        diag: dict = {}

        if error is None:
            if s.region == UNKNOWN:
                error = ("не удалось определить анатомическую область: "
                         + (s.region_reason or "признаки не вычислимы"))
            else:
                quality, violation, diag, error = cache[(s.study, s.md5)]

        shared = route_seconds.get(s.study, 0.0) + predict_seconds.get(s.study, 0.0)
        seconds = s.seconds + shared / max(per_study_files[s.study], 1)

        failed = error is not None
        proj, proj_source = projection(s.region, img.view_position if img else None)
        if failed:
            proj, proj_source = "", ""
        row_diag = {
            "study": s.study,
            "md5": s.md5 or "",
            "region_key": s.region,
            "region_confident": int(s.region_confident),
            "region_reason": s.region_reason or "",
            "lat_score": s.feats.lat_score,
            "sym": s.feats.sym,
            "reader": (img.reader if img else "") or "",
            "rows": (img.rows if img else "") or "",
            "cols": (img.cols if img else "") or "",
            "mm_per_px": (img.mm_per_px if img else "") or "",
            "projection_source": proj_source,
            "warnings": "; ".join(img.warnings) if img and img.warnings else "",
        }
        row_diag.update(diag)
        rows.append(ReportRow(
            path_to_study=s.rel,
            study_uid=(img.study_instance_uid if img else None) or "",
            image_uid=(img.sop_instance_uid if img else None) or "",
            anatomical_region=region_name(s.region, region_lang),
            quality_class=None if failed else quality,
            quality_score=None if failed else diag.get("quality_score"),
            # Текст ошибки — без адресов в памяти и временных путей (ТЗ 2.7).
            violation_type=stable_error(error, s.path) if failed else violation,
            processing_status=STATUS_FAILURE if failed else STATUS_SUCCESS,
            time_of_processing=seconds,
            study=s.study,
            projection=projection_name(proj, region_lang),
            diagnostics=row_diag,
        ))
    return rows


OVERLAY_FORMATS = ("both", "dicom", "png")


def _overlay_dicom(s: _FileState, png: bytes, quality, violation: str,
                   instance_number: int) -> bytes:
    """DICOM Secondary Capture по отрисованной картинке — в исследование исходника."""
    img = s.img
    study_uid = img.study_instance_uid or dicom_sc.derive_uid("dxa-qc-study", s.study)
    bgr = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    verdict = f"quality_class={quality}; violation_type={violation or 'none'}"
    return dicom_sc.secondary_capture(
        bgr,
        study_uid=study_uid,
        series_uid=dicom_sc.derive_uid("dxa-qc-overlay-series", study_uid),
        sop_uid=dicom_sc.derive_uid("dxa-qc-overlay", img.sop_instance_uid or "", s.rel),
        instance_number=instance_number,
        context=img.context,
        source_sop_class=img.sop_class_uid,
        source_sop_uid=img.sop_instance_uid,
        comments=verdict,
        description="DXA quality control overlay, burned-in annotation. " + verdict,
    )


SR_NAME = "DXA_QC_report.sr.dcm"


def _sr_groups(states: list[_FileState], cache: dict) -> dict[str, tuple[list, int]]:
    """Файлы по исследованиям для отчёта SR: имя в архиве → (обработанные, число сбоев).

    Исследование — каталог верхнего уровня; если в нём несколько
    StudyInstanceUID, отчётов столько же (SR лежит в одном исследовании).
    """
    by: dict[tuple[str, str], list] = defaultdict(list)
    failed: dict[str, int] = defaultdict(int)
    for s in states:
        ok = (s.error is None and s.img is not None and s.region != UNKNOWN
              and s.md5 is not None and cache[(s.study, s.md5)][3] is None)
        if not ok:
            failed[s.study] += 1
            continue
        uid = s.img.study_instance_uid or dicom_sc.derive_uid("dxa-qc-study", s.study)
        by[(s.study, uid)].append(s)
    out: dict[str, tuple[list, int]] = {}
    per_study = defaultdict(int)
    for (study, uid) in sorted(by):
        per_study[study] += 1
        n = per_study[study]
        fname = SR_NAME if n == 1 else SR_NAME.replace(".sr.dcm", f"_{n}.sr.dcm")
        name = f"{study}/{fname}" if study else fname
        out[name] = (by[(study, uid)], failed[study] if n == 1 else 0)
    return out


def _study_sr(group: list[_FileState], failed: int, cache: dict) -> bytes:
    """Отчёт DICOM SR по файлам одного исследования (`dicom_sr`)."""
    by_frame: dict[str, dicom_sr.ImageFinding] = {}
    for s in sorted(group, key=lambda x: x.rel):
        img = s.img
        ref = (img.sop_class_uid or "", img.sop_instance_uid or "",
               img.series_instance_uid or "", img.study_instance_uid or "")
        if s.md5 in by_frame:                   # точная копия уже описанного кадра
            by_frame[s.md5].refs.append(ref)
            continue
        quality, violation, diag, _ = cache[(s.study, s.md5)]
        by_frame[s.md5] = dicom_sr.ImageFinding(
            path=s.rel, region_key=s.region, region_name=region_name(s.region, "ru"),
            quality_class=int(quality), violations=[v for v in violation.split(";") if v],
            diag=diag, score=diag.get("quality_score"), refs=[ref])
    findings = list(by_frame.values())
    first = group[0].img
    study_uid = first.study_instance_uid or dicom_sc.derive_uid("dxa-qc-study", group[0].study)
    study_class = max(f.quality_class for f in findings)
    model = artifact_rules.load_default()
    return dicom_sr.basic_text_sr(
        findings, study_uid=study_uid,
        series_uid=dicom_sc.derive_uid("dxa-qc-sr-series", study_uid),
        sop_uid=dicom_sc.derive_uid("dxa-qc-sr", study_uid, group[0].study),
        context=first.context,
        study_score=score_rules.study_score([f.score for f in findings], study_class),
        artifact_threshold=model.threshold if model else None, failed=failed)


def _write_overlays(states: list[_FileState], cache: dict, path: Path,
                    fmt: str = "both") -> tuple[int, int]:
    """Пятый проход (по запросу): overlay-серия в zip, на каждый файл.

    `fmt`: `png` — картинка `<path_to_study>.png`; `dicom` — DICOM Secondary
    Capture `<path_to_study>.overlay.dcm` в том же исследовании, что и
    исходник (`dicom_sc`), и отчёт DICOM SR `<исследование>/DXA_QC_report.sr.dcm`
    на каждое исследование (`dicom_sr`, ТЗ 2.6); `both` — всё вместе. Сбой отрисовки не трогает таблицу:
    файл пропускается, причина — в `overlay_errors.txt` в конце архива. Время
    отрисовки в `time_of_processing` не входит: это дополнительная серия, а
    не анализ.

    Память не растёт с размером набора: записи идут в архив потоком, по
    порядку имён (архив побайтово воспроизводим), картинка рисуется один раз
    на уникальный кадр и держится, только пока её ждёт копия. Несжатый DICOM
    — около 1.2 МБ на снимок; собирать архив в памяти, как до 24 сентября,
    значило бы гигабайты на большом наборе.
    """
    if fmt not in OVERLAY_FORMATS:
        raise ValueError(f"формат overlay должен быть одним из {OVERLAY_FORMATS}")
    # Номер экземпляра в серии: по порядку путей внутри StudyInstanceUID.
    numbers: dict[str, int] = {}
    counter: dict[str, int] = defaultdict(int)
    for s in sorted(states, key=lambda x: x.rel):
        if s.error is None and s.img is not None:
            uid = s.img.study_instance_uid or s.study
            counter[uid] += 1
            numbers[s.rel] = counter[uid]

    plan: list[tuple[str, str, _FileState, tuple[str, str]]] = []
    uses: dict[tuple[str, str], int] = defaultdict(int)
    for s in states:
        if s.error is not None or s.md5 is None or s.region == UNKNOWN:
            continue
        key = (s.study, s.md5)
        if cache[key][3] is not None:           # предсказание упало — рисовать нечего
            continue
        for kind, suffix in (("png", ".png"), ("dicom", ".overlay.dcm")):
            if fmt in ("both", kind):
                plan.append((f"{s.rel}{suffix}", kind, s, key))
                uses[key] += 1
    # Отчёт DICOM SR — один на исследование (ТЗ 2.6), вместе с DICOM-серией.
    sr_groups = _sr_groups(states, cache) if fmt in ("both", "dicom") else {}
    for name in sr_groups:
        plan.append((name, "sr", None, None))
    plan.sort(key=lambda x: x[0])

    rendered: dict[tuple[str, str], bytes | str] = {}
    errors: dict[str, str] = {}
    written = 0
    with overlay_series.ZipWriter(path) as zw:
        for name, kind, s, key in plan:
            if kind == "sr":
                try:
                    zw.write(name, _study_sr(*sr_groups[name], cache))
                    written += 1
                except Exception as exc:  # noqa: BLE001 — отчёт не должен ронять прогон
                    errors[f"{name} (SR)"] = f"{type(exc).__name__}: {exc}"
                continue
            quality, violation, diag, _ = cache[key]
            if key not in rendered:
                try:
                    img = overlay_series.render(
                        s.region, s.img.pixels, violation.split(";"), diag,
                        mm_per_px=s.img.mm_per_px, mm_per_px_source=s.img.mm_per_px_source,
                        scanner_model=s.img.model)
                    rendered[key] = overlay_series.encode_png(img)
                except Exception as exc:  # noqa: BLE001 — оверлей не должен ронять прогон
                    rendered[key] = f"{type(exc).__name__}: {exc}"
            got = rendered[key]
            uses[key] -= 1
            if uses[key] == 0:
                del rendered[key]
            if not isinstance(got, bytes):
                errors.setdefault(s.rel, got)
                continue
            if kind == "png":
                data = got
            else:
                try:
                    data = _overlay_dicom(s, got, quality, violation, numbers.get(s.rel, 1))
                except Exception as exc:  # noqa: BLE001
                    errors[f"{s.rel} (DICOM)"] = f"{type(exc).__name__}: {exc}"
                    continue
            zw.write(name, data)
            written += 1
        if errors:
            lines = sorted(f"{rel}\t{msg}" for rel, msg in errors.items())
            zw.write("overlay_errors.txt", ("\n".join(lines) + "\n").encode("utf-8"))
    return written, len(errors)


def run(input_path: Path, output: Path, diagnostics: Path | None = None,
        region_lang: str = "ru", limit: int | None = None,
        quiet: bool = False, overlays: Path | None = None,
        overlay_format: str = "both") -> int:
    """Полный прогон. Возвращает код возврата процесса."""
    def log(*a):
        if not quiet:
            print(*a, file=sys.stderr, flush=True)

    t_start = time.perf_counter()
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"нет такого пути: {input_path}", file=sys.stderr)
        return 2

    # Архив (ТЗ 2.7) распаковывается во временный каталог; пути в таблице —
    # пути внутри архива.
    with tempfile.TemporaryDirectory(prefix="dxa-qc-") as tmp:
        # DICOM с сигнатурой zip в пикселях не должен сойти за архив.
        if input_path.is_file() and zipfile.is_zipfile(input_path) and not is_dicom(input_path):
            try:
                n = archive_io.extract(input_path, tmp)
            except archive_io.ArchiveError as exc:
                print(f"архив не распакован: {exc}", file=sys.stderr)
                return 2
            log(f"архив: {n} файлов распаковано")
            input_path = Path(tmp)
        return _run_tree(input_path, output, diagnostics, region_lang, limit, overlays,
                         log, quiet, t_start, overlay_format)


def _run_tree(input_path: Path, output: Path, diagnostics: Path | None, region_lang: str,
              limit: int | None, overlays: Path | None, log, quiet: bool,
              t_start: float, overlay_format: str = "both") -> int:
    if input_path.is_file():
        files = [input_path]
        root = sroot = input_path.parent
    else:
        root = input_path
        files = list(iter_dicom_files(root))
        sroot = archive_io.studies_root(root, files)
        if sroot != root:
            log(f"каталог исследований: {sroot.relative_to(root).as_posix()}")
    if limit:
        files = files[:limit]

    # Путь в таблице — от того, что указал пользователь; исследование — от
    # найденного каталога исследований (обёртки вроде «НД/Исследования» сняты).
    # Файлы вне каталога исследований (например, `разметка.xlsx` рядом с ним)
    # получают ключ от корня: строка Failure на них остаётся, а ключ — стабильный.
    states = [_FileState(path=p, rel=str(p.relative_to(root).as_posix()),
                         study=study_key(p, sroot if sroot in p.parents else root))
              for p in files]
    log(f"файлов: {len(states)}, исследований: {len({s.study for s in states})}")

    for i, s in enumerate(states, 1):
        _read_one(s)
        if not quiet and i % 100 == 0:
            log(f"  прочитано {i}/{len(states)}")

    route_seconds = _route_studies(states)
    _warn_crowded_studies(states, log)
    cache, predict_seconds = _predict_frames(states)
    rows = _build_rows(states, route_seconds, predict_seconds, cache, region_lang)

    write_report(rows, output)
    if diagnostics:
        write_diagnostics(rows, diagnostics)
    if overlays:
        n_files, n_err = _write_overlays(states, cache, overlays, overlay_format)
        log(f"overlay-серия ({overlay_format}): {n_files} файлов, ошибок отрисовки: {n_err} -> {overlays}")

    n_fail = sum(1 for r in rows if r.processing_status == STATUS_FAILURE)
    n_uniq = len({s.md5 for s in states if s.md5})
    elapsed = time.perf_counter() - t_start
    log(f"строк: {len(rows)}  уникальных кадров: {n_uniq}  Failure: {n_fail}")
    if n_fail and not quiet:
        for r in rows:
            if r.processing_status == STATUS_FAILURE:
                log(f"  {r.path_to_study} -> {r.violation_type}")
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        counts[r.anatomical_region] += 1
    log("области: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    viol: dict[str, int] = defaultdict(int)
    for r in rows:
        if r.processing_status == STATUS_SUCCESS and r.violation_type:
            for v in r.violation_type.split(";"):
                viol[v] += 1
    n_bad = sum(1 for r in rows if r.quality_class == 1)
    log(f"с нарушениями: {n_bad} строк"
        + ("  (" + ", ".join(f"{k}={v}" for k, v in sorted(viol.items())) + ")" if viol else ""))
    # Сводка по масштабу. На закрытом наборе это первое, на что надо смотреть:
    # порог правила охвата калиброван в миллиметрах, а миллиметры берутся
    # отсюда. Смена аппарата или появление PixelSpacing меняют картину молча,
    # если о них не сказать вслух.
    src: dict[str, int] = defaultdict(int)
    dev: dict[str, int] = defaultdict(int)
    for st in states:
        if st.img is None:
            continue
        src[st.img.mm_per_px_source or "нет"] += 1
        dev[" ".join(x for x in (st.img.manufacturer, st.img.model) if x) or "неизвестен"] += 1
    log("масштаб: " + ", ".join(f"{k}={v}" for k, v in sorted(src.items())))
    log("аппараты: " + ", ".join(f"{k}={v}" for k, v in sorted(dev.items())))
    if not any(k in src for k in ("pixel_spacing", "imager_pixel_spacing")):
        log("  внимание: достоверных тегов масштаба нет, миллиметры взяты из "
            "Exposed Area и паспорта аппарата")
    log(f"время: {elapsed:.2f} с всего, "
        f"{elapsed / max(len({s.study for s in states}), 1):.3f} с на исследование")
    log(f"записано: {output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m dxa_qc.batch",
        description="Пакетная оценка качества денситометрических исследований.")
    ap.add_argument("--input", required=True, type=Path,
                    help="каталог с исследованиями (или один DICOM-файл)")
    ap.add_argument("--output", required=True, type=Path,
                    help="выходная таблица .xlsx или .csv")
    ap.add_argument("--diagnostics", type=Path, default=None,
                    help="дополнительный csv со служебными колонками для разбора ошибок")
    ap.add_argument("--region-lang", choices=("ru", "code"), default="ru",
                    help="имя анатомической области в таблице: по-русски (как в ТЗ 2.3) "
                         "или коротким кодом")
    ap.add_argument("--overlays", type=Path, default=None,
                    help="zip-архив с overlay-серией (ТЗ 2.6, 2.7) на каждый файл")
    ap.add_argument("--overlay-format", choices=OVERLAY_FORMATS, default="both",
                    help="both — PNG и DICOM Secondary Capture, dicom — только DICOM-серия, "
                         "png — только картинки")
    ap.add_argument("--limit", type=int, default=None, help="обработать только первые N файлов")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    return run(args.input, args.output, args.diagnostics, args.region_lang,
               args.limit, args.quiet, args.overlays, args.overlay_format)


if __name__ == "__main__":
    raise SystemExit(main())
