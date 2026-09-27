"""Выходная таблица по ТЗ 2.5 и её детерминированная запись.

Структура колонок взята из ТЗ дословно (раздел 2.5, «Рекомендуемая
структура»), порядок сохранён:

    path_to_study       String   путь к исследованию или файлу
    study_uid           String   идентификатор исследования из DICOM-тегов
    image_uid           String   идентификатор изображения из DICOM-тегов
    anatomical_region   String   определённая анатомическая область
    quality_class       Integer  0 — качественное, 1 — есть нарушение
    violation_type      String   тип или перечень нарушений
    processing_status   String   Success / Failure
    time_of_processing  Float    время обработки в секундах

К ним добавлены три колонки, которых в ТЗ нет: `quality_score` — балл
нарушения для ROC-AUC (ТЗ 8.4), `study` — идентификатор исследования из
имени каталога, и `projection` — проекция снимка (ТЗ 2.2). Все стоят после
восьми колонок ТЗ, подробности — у EXTRA_COLUMNS.

Две договорённости, которых в ТЗ нет и которые нужно помнить.

1. Одна строка — один *файл*, а не одно уникальное изображение. В обучающем
   наборе 49.5% файлов — точные копии, но ТЗ 2.5 требует строку на изображение
   в исследовании, а какие из них организатор считает отдельными, мы не знаем.
   Дедупликация применяется только внутри конвейера (роутер и предсказание
   считаются один раз на уникальный кадр), а в таблицу идут все файлы.
   Вопрос задан организатору, п. 6.

2. У строки со `processing_status=Failure` `quality_class` пустой, а не 0
   и не -1. Ноль означал бы «исследование качественное», то есть утверждение
   о снимке, который мы не смогли прочитать. Текст ошибки пишется
   в `violation_type` — отдельной колонки под него ТЗ не предусматривает.

Запись детерминирована побайтово: xlsx — это zip, а zip по умолчанию несёт
время создания каждой записи и метаданные документа. Оба источника
недетерминизма здесь подавлены, поэтому два прогона на одних данных дают
одинаковый файл (ТЗ 2.7, воспроизводимость).
"""
from __future__ import annotations

import csv
import io
import re
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

# Восемь колонок ТЗ 2.5 — ровно в его порядке и с его именами.
COLUMNS = [
    "path_to_study",
    "study_uid",
    "image_uid",
    "anatomical_region",
    "quality_class",
    "violation_type",
    "processing_status",
    "time_of_processing",
]

# Девятая колонка — балл нарушения `quality_score` в [0, 1] (модуль score):
# ТЗ 8.4 называет ROC-AUC приоритетной метрикой, а по бинарному
# quality_class она вырождается. Балл больше 0.5 тогда и только тогда, когда
# quality_class = 1; у Failure пуст, как и quality_class.
#
# Десятая колонка, которой в ТЗ нет. Идентификатор исследования — имя
# каталога верхнего уровня: именно по нему сделана `разметка.xlsx`, а
# StudyInstanceUID из тегов принадлежит другой серии идентификаторов и с
# разметкой не пересекается. Колонка стоит ПОСЛЕ восьми колонок ТЗ, чтобы
# строгий позиционный разбор первых восьми читался как раньше. Структура
# в ТЗ названа рекомендуемой, так что добавление колонки допустимо, а
# сопоставление с эталоном по ключу перестаёт быть догадкой.
#
# Одиннадцатая — проекция (ТЗ 2.2 требует определять область *и проекцию*, а в
# структуре 2.5 колонки под неё нет). Тоже после колонок ТЗ.
EXTRA_COLUMNS = ["quality_score", "study", "projection"]
ALL_COLUMNS = COLUMNS + EXTRA_COLUMNS

STATUS_SUCCESS = "Success"
STATUS_FAILURE = "Failure"

# Фиксированная метка времени для всех записей архива и свойств документа.
FIXED_DATETIME = datetime(2026, 1, 1, 0, 0, 0)
FIXED_ZIP_DATE = (2026, 1, 1, 0, 0, 0)

# Знаков после запятой во времени обработки. Меньше — теряется смысл на быстрых
# файлах, больше — в таблице шум от джиттера планировщика.
TIME_PRECISION = 4


@dataclass
class ReportRow:
    """Одна строка выходной таблицы."""

    path_to_study: str = ""
    study_uid: str = ""
    image_uid: str = ""
    anatomical_region: str = ""
    quality_class: int | None = None
    violation_type: str = ""
    processing_status: str = STATUS_FAILURE
    time_of_processing: float = 0.0
    quality_score: float | None = None
    study: str = ""
    projection: str = ""

    # Служебные поля: в выходную таблицу не попадают, идут в диагностику.
    diagnostics: dict[str, Any] = field(default_factory=dict, repr=False)

    def as_record(self) -> dict[str, Any]:
        d = {c: getattr(self, c) for c in ALL_COLUMNS}
        d["time_of_processing"] = round(float(self.time_of_processing), TIME_PRECISION)
        d["quality_class"] = None if self.quality_class is None else int(self.quality_class)
        for k, v in d.items():
            if v is None and k not in ("quality_class", "quality_score"):
                d[k] = ""
        return d

    def as_diagnostics(self) -> dict[str, Any]:
        d = self.as_record()
        d.update(asdict(self)["diagnostics"])
        return d


def _cell(value: Any) -> Any:
    """Пустое значение в ячейке — пусто, а не строка 'None' и не 'nan'."""
    if value is None:
        return None
    if isinstance(value, float) and value != value:  # NaN
        return None
    if isinstance(value, bool):
        return int(value)
    return value


def write_csv(records: Iterable[dict[str, Any]], path: str | Path,
              columns: list[str] | None = None) -> Path:
    """CSV с фиксированным переводом строки: побайтово одинаков между прогонами."""
    records = list(records)
    columns = columns or ALL_COLUMNS
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n",
                           extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow({c: ("" if _cell(r.get(c)) is None else r.get(c)) for c in columns})
    return path


def write_xlsx(records: Iterable[dict[str, Any]], path: str | Path,
               columns: list[str] | None = None, sheet_name: str = "result") -> Path:
    """XLSX без времени создания в архиве и в свойствах документа."""
    try:
        from openpyxl import Workbook
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "для записи .xlsx нужен openpyxl (см. requirements.txt); "
            "для csv используйте --output с расширением .csv"
        ) from exc

    records = list(records)
    columns = columns or ALL_COLUMNS
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    wb = Workbook()
    ws = wb.active
    ws.title = sheet_name
    ws.append(columns)
    for r in records:
        ws.append([_cell(r.get(c)) for c in columns])

    # Свойства документа — второй источник недетерминизма после времён zip.
    # `modified` здесь выставлять бесполезно: openpyxl в save_workbook
    # перезаписывает его текущим временем. Его чинит _repack_deterministic.
    props = wb.properties
    props.created = FIXED_DATETIME
    props.modified = FIXED_DATETIME
    props.creator = "dxa_qc"
    props.lastModifiedBy = "dxa_qc"

    buf = io.BytesIO()
    wb.save(buf)
    path.write_bytes(_repack_deterministic(buf.getvalue()))
    return path


_CORE_MODIFIED = re.compile(rb"(<dcterms:modified[^>]*>)[^<]*(</dcterms:modified>)")


def _repack_deterministic(data: bytes) -> bytes:
    """Пересобирает zip с фиксированными датами, порядком записей и
    временем изменения документа в docProps/core.xml."""
    fixed = FIXED_DATETIME.strftime("%Y-%m-%dT%H:%M:%SZ").encode()
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as src:
        names = sorted(src.namelist())
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as dst:
            for name in names:
                info = zipfile.ZipInfo(name, date_time=FIXED_ZIP_DATE)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                info.create_system = 0
                body = src.read(name)
                if name == "docProps/core.xml":
                    body = _CORE_MODIFIED.sub(rb"\g<1>" + fixed + rb"\g<2>", body)
                dst.writestr(info, body)
    return out.getvalue()


def write_report(rows: Iterable[ReportRow], path: str | Path) -> Path:
    """Пишет таблицу ТЗ 2.5; формат выбирается по расширению `path`."""
    records = [r.as_record() for r in rows]
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return write_csv(records, path)
    if suffix in (".xlsx", ".xlsm"):
        return write_xlsx(records, path)
    raise ValueError(f"неподдерживаемое расширение выходного файла: {path.suffix!r} "
                     "(ожидается .xlsx или .csv)")


def write_diagnostics(rows: Iterable[ReportRow], path: str | Path) -> Path:
    """Расширенная таблица для разбора ошибок: колонки ТЗ плюс служебные."""
    rows = list(rows)
    records = [r.as_diagnostics() for r in rows]
    extra: list[str] = []
    for rec in records:
        for k in rec:
            if k not in ALL_COLUMNS and k not in extra:
                extra.append(k)
    return write_csv(records, path, columns=ALL_COLUMNS + sorted(extra))
