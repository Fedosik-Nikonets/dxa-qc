"""Чтение DICOM-исследований денситометрии.

Контракт модуля
---------------
* Ни одна функция не бросает исключений наружу. Любая проблема возвращается
  в поле `error` объекта :class:`DicomImage` (требование ТЗ 2.7: необработанных
  исключений быть не должно, каждая строка отчёта — Success или Failure).
* Пиксели всегда приводятся к float32 в диапазоне [0, 1], где 1 — кость
  (максимальное ослабление), независимо от PhotometricInterpretation.
* Масштаб (мм/пиксель) всегда определён: из тегов, из Exposed Area или
  из паспортного значения сканера. Источник фиксируется в `mm_per_px_source`,
  чтобы это можно было честно показать на питче.

Два пути чтения
---------------
1. **pydicom** — основной. Покрывает сжатые TransferSyntax (JPEG, JPEG-LS,
   JPEG2000, RLE) и VOI LUT Sequence.
2. **Встроенный ридер** (`_minidicom`) — резервный, без внешних зависимостей.
   Покрывает все несжатые TransferSyntax, 8/16 бит, знаковые и беззнаковые,
   MONOCHROME1/2, multi-frame, Rescale и Window. Используется, когда pydicom
   недоступен, а также когда pydicom упал на конкретном файле.

Обучающий набор однороден (Implicit VR LE, 8 бит, MONOCHROME2, несжатый —
см. reports/day1_findings.md), поэтому все краевые случаи покрыты синтетикой
в tests/, а не реальными файлами.
"""
from __future__ import annotations

import re
import warnings as _warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import _minidicom
from ._minidicom import DicomParseError

try:  # pragma: no cover — зависит от окружения
    import pydicom
    from pydicom.pixels import apply_voi_lut  # pydicom >= 3

    HAVE_PYDICOM = True
except Exception:  # pragma: no cover
    try:
        import pydicom
        from pydicom.pixel_data_handlers.util import apply_voi_lut  # pydicom 2.x

        HAVE_PYDICOM = True
    except Exception:
        pydicom = None
        apply_voi_lut = None
        HAVE_PYDICOM = False

# Паспортный размер пикселя GE Lunar Prodigy Advance в режимах AP Spine / Femur.
# PixelSpacing (0028,0030) и ImagerPixelSpacing (0018,1164) отсутствуют во всех
# 499 файлах обучающего набора — см. reports/day1_findings.md, раздел 6.
FALLBACK_MM_PER_PX = 0.6

# Диапазон, в котором оценка из Exposed Area считается правдоподобной.
EA_MM_RANGE = (0.55, 0.70)

TAG_EXPOSED_AREA = (0x0040, 0x0303)

T = {
    "sop_instance_uid": (0x0008, 0x0018),
    "sop_class_uid": (0x0008, 0x0016),
    "specific_charset": (0x0008, 0x0005),
    "study_date": (0x0008, 0x0020),
    "study_time": (0x0008, 0x0030),
    "accession_number": (0x0008, 0x0050),
    "referring_physician": (0x0008, 0x0090),
    "modality": (0x0008, 0x0060),
    "manufacturer": (0x0008, 0x0070),
    "model": (0x0008, 0x1090),
    "series_description": (0x0008, 0x103E),
    "patient_name": (0x0010, 0x0010),
    "patient_id": (0x0010, 0x0020),
    "patient_birth_date": (0x0010, 0x0030),
    "patient_sex": (0x0010, 0x0040),
    "body_part": (0x0018, 0x0015),
    "imager_pixel_spacing": (0x0018, 0x1164),
    "view_position": (0x0018, 0x5101),
    "study_instance_uid": (0x0020, 0x000D),
    "study_id": (0x0020, 0x0010),
    "series_instance_uid": (0x0020, 0x000E),
    "instance_number": (0x0020, 0x0013),
    "laterality": (0x0020, 0x0060),
    "samples_per_pixel": (0x0028, 0x0002),
    "photometric": (0x0028, 0x0004),
    "n_frames": (0x0028, 0x0008),
    "rows": (0x0028, 0x0010),
    "cols": (0x0028, 0x0011),
    "pixel_spacing": (0x0028, 0x0030),
    "bits_allocated": (0x0028, 0x0100),
    "bits_stored": (0x0028, 0x0101),
    "high_bit": (0x0028, 0x0102),
    "pixel_repr": (0x0028, 0x0103),
    "window_center": (0x0028, 0x1050),
    "window_width": (0x0028, 0x1051),
    "rescale_intercept": (0x0028, 0x1052),
    "rescale_slope": (0x0028, 0x1053),
}


# Атрибуты, которые копируются в производную серию: ключ -> keyword pydicom.
CONTEXT_KEYWORDS = {
    "patient_name": "PatientName",
    "patient_id": "PatientID",
    "patient_birth_date": "PatientBirthDate",
    "patient_sex": "PatientSex",
    "study_date": "StudyDate",
    "study_time": "StudyTime",
    "study_id": "StudyID",
    "accession_number": "AccessionNumber",
    "referring_physician": "ReferringPhysicianName",
}

# Specific Character Set -> кодек Python. Встроенный ридер декодирует текст
# как latin-1 (биекция байт), поэтому строку можно точно перекодировать.
_CHARSETS = {
    "ISO_IR 6": "ascii", "ISO_IR 100": "latin-1", "ISO_IR 101": "iso8859_2",
    "ISO_IR 144": "iso8859_5", "ISO_IR 192": "utf-8", "ISO 2022 IR 6": "ascii",
    "ISO 2022 IR 100": "latin-1", "ISO 2022 IR 144": "iso8859_5", "GB18030": "gb18030",
}


def _charset_codec(value: Any) -> str:
    """Кодек по (0008,0005); при нескольких значениях берётся последнее."""
    parts = [p.strip() for p in str(value or "").split("\\") if p.strip()]
    return _CHARSETS.get(parts[-1], "latin-1") if parts else "latin-1"


@dataclass
class DicomImage:
    """Результат чтения одного DICOM-файла."""

    path: str
    ok: bool = False
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    reader: str | None = None

    pixels: np.ndarray | None = field(default=None, repr=False)
    rows: int | None = None
    cols: int | None = None
    n_frames: int = 1
    frame_index: int = 0

    sop_instance_uid: str | None = None
    sop_class_uid: str | None = None
    study_instance_uid: str | None = None
    series_instance_uid: str | None = None
    instance_number: int | None = None
    modality: str | None = None
    manufacturer: str | None = None
    model: str | None = None
    series_description: str | None = None
    body_part: str | None = None
    laterality: str | None = None
    view_position: str | None = None

    photometric: str | None = None
    bits_allocated: int | None = None
    bits_stored: int | None = None
    pixel_repr: int | None = None
    samples_per_pixel: int | None = None
    transfer_syntax: str | None = None

    mm_per_px: float | None = None
    mm_per_px_source: str | None = None
    exposed_area: tuple[int, int] | None = None

    # Атрибуты пациента и исследования (модули Patient и General Study) —
    # только чтобы производная overlay-серия легла в то же исследование в
    # PACS. В решение и в диагностику не идут (ТЗ 2.4: решение не зависит
    # от персональных данных).
    context: dict[str, str] = field(default_factory=dict, repr=False)

    def meta_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("pixels", None)
        d.pop("context", None)
        d["warnings"] = "; ".join(self.warnings) or None
        return d


# --------------------------------------------------------------------------- #
# общая постобработка пикселей
# --------------------------------------------------------------------------- #
def _is_sequence(v: Any) -> bool:
    """Последовательность значений, но не строка и не массив байт.

    Проверять через isinstance(v, (list, tuple)) нельзя: pydicom отдаёт
    многозначные теги как MultiValue, который не наследуется ни от list,
    ни от tuple. Из-за этого PixelSpacing молча терялся и масштаб
    откатывался на константу сканера — ровно этот баг поймал
    test_pixel_spacing_wins_over_default на ветке pydicom.
    """
    return hasattr(v, "__iter__") and not isinstance(v, (str, bytes, bytearray))


def _first(v: Any) -> Any:
    """Многозначные теги (WindowCenter='40\\400') приводим к первому значению."""
    if _is_sequence(v):
        vals = list(v)
        return vals[0] if vals else None
    return v


def _as_floats(raw: Any) -> list[float]:
    """Значения тега как список положительных float, что бы туда ни пришло.

    Принимает число, DSfloat, MultiValue, список, кортеж и строку вида
    '0.25\\0.25' — последнее встречается, когда тег читается сырым.
    """
    if raw is None:
        return []
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("latin-1", "replace")
    if isinstance(raw, str):
        items: list[Any] = [p for p in raw.replace("\\", " ").split() if p]
    elif _is_sequence(raw):
        items = list(raw)
    else:
        items = [raw]

    out: list[float] = []
    for v in items:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f > 0 and np.isfinite(f):
            out.append(f)
    return out


def _apply_window(a: np.ndarray, center: float | None, width: float | None) -> np.ndarray:
    """VOI LUT линейным окном по DICOM PS3.3 C.11.2.1.2."""
    if center is None or width is None or width <= 1:
        return a
    lo = center - 0.5 - (width - 1) / 2.0
    hi = center - 0.5 + (width - 1) / 2.0
    if hi <= lo:
        return a
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def _finalize(
    a: np.ndarray,
    photometric: str | None,
    center: float | None = None,
    width: float | None = None,
    windowed: bool = False,
) -> np.ndarray:
    """float32 [0, 1], 1 = кость, независимо от полярности и битности.

    Порядок соответствует конвейеру DICOM: Modality LUT (применена до вызова)
    → VOI LUT → Presentation LUT (инверсия для MONOCHROME1) → нормировка.
    """
    a = np.asarray(a, dtype=np.float32)
    if not np.isfinite(a).all():
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    if not windowed:
        a = _apply_window(a, center, width)
    if photometric == "MONOCHROME1":
        a = a.max() - a
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-6:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - lo) / (hi - lo)).astype(np.float32)


def _spacing(
    pixel_spacing: Any,
    imager_spacing: Any,
    ea: tuple[int, int] | None,
    rows: int | None,
    cols: int | None,
) -> tuple[float, str]:
    """Масштаб мм/пиксель и источник оценки.

    Приоритет: PixelSpacing → ImagerPixelSpacing → Exposed Area (только если
    результат правдоподобен) → паспортная константа сканера.

    (0040,0303) Exposed Area описывает поле сканирования, а не сохранённый кроп,
    поэтому используется лишь как подтверждение: в 46% файлов набора она даёт
    1.7–2.5 мм/пиксель, что для кропа ТБС физически невозможно.
    """
    for raw, src in ((pixel_spacing, "pixel_spacing"), (imager_spacing, "imager_pixel_spacing")):
        nums = _as_floats(raw)
        if nums:
            return float(sum(nums) / len(nums)), src

    if ea and rows and cols and len(ea) >= 2 and ea[0] > 0 and ea[1] > 0:
        sc, sr = ea[0] / cols, ea[1] / rows
        if all(EA_MM_RANGE[0] <= x <= EA_MM_RANGE[1] for x in (sc, sr)):
            return float((sc + sr) / 2), "exposed_area"

    return FALLBACK_MM_PER_PX, "scanner_default"


# --------------------------------------------------------------------------- #
# путь через pydicom
# --------------------------------------------------------------------------- #
# Сколько разных предупреждений чтения удерживается на файл: дальше это
# перестаёт быть информацией и становится шумом.
MAX_READ_WARNINGS = 3


def _read_pydicom(path: Path, with_pixels: bool, frame: int) -> DicomImage:
    """Чтение через pydicom. Предупреждения библиотеки идут в строку, не в stderr.

    На обучающем наборе pydicom выдаёт около двухсот предупреждений
    «Invalid value for VR UI»: в UID организатора компоненты начинаются с
    нуля (`...8252.08110703...`), а стандарт это запрещает. Файлы при этом
    читаются, и встроенный ридер их принимает молча.

    Двести строк в stderr на штатном прогоне — это не диагностика, а помеха:
    в ней тонет всё остальное, и сервис выглядит сломанным. Но и терять
    сигнал нельзя, поэтому предупреждения не подавляются, а собираются и
    кладутся в `warnings` самого изображения — то есть в диагностическую
    таблицу, рядом с файлом, к которому относятся.

    Перехват обязан накрывать **всю** работу с набором данных, а не один
    `dcmread`. pydicom разбирает значение элемента лениво, при первом
    обращении к нему, и предупреждение о неверном VR выходит не из чтения
    файла, а из строки `ds.SOPInstanceUID`. Перехват вокруг одного
    `dcmread` пропускал их все — это поймал тест
    `test_pydicom_warnings_go_into_the_row_not_into_stderr`.
    """
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        img = _read_pydicom_inner(path, with_pixels, frame)
    _attach_read_warnings(img, caught)
    return img


def _attach_read_warnings(img: DicomImage, caught: list) -> None:
    """Сворачивает перехваченные предупреждения и кладёт их в изображение."""
    seen: dict[str, int] = {}
    for w in caught:
        # Текст включает само значение тега, поэтому ключом берётся начало
        # сообщения: иначе на каждый UID выйдет отдельная строка.
        key = str(w.message).split(":")[0].strip()[:60]
        seen[key] = seen.get(key, 0) + 1
    for key, count in list(seen.items())[:MAX_READ_WARNINGS]:
        img.warnings.append(f"{key} (x{count})" if count > 1 else key)
    if len(seen) > MAX_READ_WARNINGS:
        img.warnings.append(f"и ещё {len(seen) - MAX_READ_WARNINGS} видов предупреждений")


def _read_pydicom_inner(path: Path, with_pixels: bool, frame: int) -> DicomImage:
    img = DicomImage(path=str(path), reader="pydicom")
    ds = pydicom.dcmread(str(path), force=True, stop_before_pixels=not with_pixels)

    def s(name: str) -> str | None:
        v = getattr(ds, name, None)
        return (str(v).strip() or None) if v is not None else None

    def i(name: str) -> int | None:
        v = getattr(ds, name, None)
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    img.transfer_syntax = str(getattr(ds.file_meta, "TransferSyntaxUID", "")) or None
    img.sop_instance_uid = s("SOPInstanceUID")
    img.sop_class_uid = s("SOPClassUID")
    img.study_instance_uid = s("StudyInstanceUID")
    img.series_instance_uid = s("SeriesInstanceUID")
    img.modality = s("Modality")
    img.manufacturer = s("Manufacturer")
    img.model = s("ManufacturerModelName")
    img.series_description = s("SeriesDescription")
    img.body_part = s("BodyPartExamined")
    img.laterality = s("Laterality")
    img.view_position = s("ViewPosition")
    img.context = {k: v for k, kw in CONTEXT_KEYWORDS.items() if (v := s(kw)) is not None}
    img.photometric = s("PhotometricInterpretation")
    img.bits_allocated = i("BitsAllocated")
    img.bits_stored = i("BitsStored")
    img.pixel_repr = i("PixelRepresentation")
    img.samples_per_pixel = i("SamplesPerPixel")
    img.rows = i("Rows")
    img.cols = i("Columns")
    img.instance_number = i("InstanceNumber")
    img.n_frames = i("NumberOfFrames") or 1

    ea = None
    if TAG_EXPOSED_AREA in ds:
        try:
            v = ds[TAG_EXPOSED_AREA].value
            ea = (int(v[0]), int(v[1]))
        except Exception:
            ea = None
    img.exposed_area = ea

    img.mm_per_px, img.mm_per_px_source = _spacing(
        getattr(ds, "PixelSpacing", None),
        getattr(ds, "ImagerPixelSpacing", None),
        ea,
        img.rows,
        img.cols,
    )

    if with_pixels:
        if img.samples_per_pixel and img.samples_per_pixel > 1:
            raise DicomParseError(
                f"цветное изображение (SamplesPerPixel={img.samples_per_pixel}) "
                "не поддерживается: ожидается монохромная денситометрия"
            )
        arr = ds.pixel_array
        if arr.ndim == 3 and img.n_frames > 1:
            img.n_frames = arr.shape[0]
            idx = min(max(frame, 0), arr.shape[0] - 1)
            if idx != frame:
                img.warnings.append(f"кадр {frame} вне диапазона, взят {idx}")
            img.frame_index = idx
            arr = arr[idx]
        elif arr.ndim == 3:
            arr = arr[0]
        if arr.ndim != 2:
            raise DicomParseError(f"неожиданная форма PixelData: {arr.shape}")

        windowed = False
        try:
            voi = apply_voi_lut(arr, ds)
            if voi is not None and np.asarray(voi).shape == arr.shape:
                arr = np.asarray(voi)
                windowed = True
        except Exception as exc:  # noqa: BLE001
            img.warnings.append(f"VOI LUT не применён: {type(exc).__name__}")

        slope = _first(getattr(ds, "RescaleSlope", None))
        inter = _first(getattr(ds, "RescaleIntercept", None))
        a = np.asarray(arr, dtype=np.float32)
        if not windowed and slope is not None:
            a = a * float(slope) + float(inter or 0.0)

        img.pixels = _finalize(
            a,
            img.photometric,
            _first(getattr(ds, "WindowCenter", None)),
            _first(getattr(ds, "WindowWidth", None)),
            windowed=windowed,
        )
        img.rows, img.cols = img.pixels.shape
    img.ok = True
    return img


# --------------------------------------------------------------------------- #
# встроенный резервный ридер
# --------------------------------------------------------------------------- #
def _read_mini(path: Path, with_pixels: bool, frame: int) -> DicomImage:
    img = DicomImage(path=str(path), reader="minidicom")
    parsed = _minidicom.read_file(str(path), with_pixels=with_pixels)
    v = parsed["values"]
    little = parsed["little"]
    img.transfer_syntax = parsed["TransferSyntaxUID"]

    def txt(key: str) -> str | None:
        raw = v.get(T[key])
        if raw is None:
            return None
        return str(raw).strip() or None

    def num(key: str) -> Any:
        return _first(v.get(T[key]))

    for key in (
        "sop_instance_uid", "sop_class_uid", "study_instance_uid",
        "series_instance_uid", "modality", "manufacturer", "model",
        "series_description", "body_part", "laterality", "photometric",
        "view_position",
    ):
        setattr(img, key, txt(key))

    codec = _charset_codec(v.get(T["specific_charset"]))
    for key in CONTEXT_KEYWORDS:
        val = txt(key)
        if val is None:
            continue
        try:
            val = val.encode("latin-1").decode(codec)
        except (UnicodeError, LookupError):
            pass
        img.context[key] = val

    for key in ("rows", "cols", "bits_allocated", "bits_stored", "pixel_repr",
                "samples_per_pixel", "instance_number"):
        val = num(key)
        setattr(img, key, int(val) if isinstance(val, (int, float)) else None)

    nf = num("n_frames")
    img.n_frames = int(nf) if isinstance(nf, (int, float)) and nf >= 1 else 1

    ea_raw = v.get(TAG_EXPOSED_AREA)
    ea = None
    if _is_sequence(ea_raw):
        vals = list(ea_raw)
        if len(vals) >= 2:
            ea = (int(vals[0]), int(vals[1]))
    img.exposed_area = ea

    img.mm_per_px, img.mm_per_px_source = _spacing(
        v.get(T["pixel_spacing"]), v.get(T["imager_pixel_spacing"]),
        ea, img.rows, img.cols,
    )

    if not with_pixels:
        img.ok = True
        return img

    raw = parsed["pixel_raw"]
    if raw is None:
        raise DicomParseError("PixelData отсутствует в файле")
    if not img.rows or not img.cols:
        raise DicomParseError("нет Rows/Columns — размер кадра неизвестен")
    if (img.samples_per_pixel or 1) > 1:
        raise DicomParseError(
            f"цветное изображение (SamplesPerPixel={img.samples_per_pixel}) не поддерживается"
        )

    ba = img.bits_allocated or 8
    if ba not in (8, 16):
        raise DicomParseError(f"BitsAllocated={ba} не поддерживается встроенным ридером")
    signed = bool(img.pixel_repr)
    dtype = np.dtype(
        ("i" if signed else "u") + ("1" if ba == 8 else "2")
    ).newbyteorder("<" if little else ">")

    px_per_frame = img.rows * img.cols
    bytes_per_frame = px_per_frame * (ba // 8)
    n_avail = len(raw) // bytes_per_frame if bytes_per_frame else 0
    if n_avail < 1:
        raise DicomParseError(
            f"PixelData обрезан: {len(raw)} байт при ожидаемых {bytes_per_frame}"
        )
    if n_avail != img.n_frames:
        img.warnings.append(
            f"NumberOfFrames={img.n_frames}, фактически кадров {n_avail}"
        )
        img.n_frames = n_avail

    idx = min(max(frame, 0), img.n_frames - 1)
    if idx != frame:
        img.warnings.append(f"кадр {frame} вне диапазона, взят {idx}")
    img.frame_index = idx

    off = idx * bytes_per_frame
    arr = np.frombuffer(raw[off : off + bytes_per_frame], dtype=dtype)
    arr = arr.reshape(img.rows, img.cols)

    a = arr.astype(np.float32)
    bs = img.bits_stored or ba
    if not signed and bs < ba:
        a = np.asarray(arr & np.uint16((1 << bs) - 1), dtype=np.float32)

    slope = num("rescale_slope")
    inter = num("rescale_intercept")
    if isinstance(slope, (int, float)):
        a = a * float(slope) + float(inter if isinstance(inter, (int, float)) else 0.0)

    img.pixels = _finalize(a, img.photometric, num("window_center"), num("window_width"))
    img.ok = True
    return img


# --------------------------------------------------------------------------- #
# публичный интерфейс
# --------------------------------------------------------------------------- #
def read_dicom(
    path: str | Path, with_pixels: bool = True, frame: int = 0
) -> DicomImage:
    """Безопасно читает один DICOM.

    Сначала pydicom, при его отказе — встроенный ридер. Наружу исключения
    не выходят: результат всегда DicomImage, при неудаче ok=False и error.
    """
    p = Path(path)
    errors: list[str] = []

    if HAVE_PYDICOM:
        try:
            return _read_pydicom(p, with_pixels, frame)
        except Exception as exc:  # noqa: BLE001 — сознательно ловим всё
            errors.append(f"pydicom: {type(exc).__name__}: {exc}")

    try:
        img = _read_mini(p, with_pixels, frame)
        if errors:
            img.warnings.append(stable_error(errors[0], p))
        return img
    except Exception as exc:  # noqa: BLE001
        errors.append(f"minidicom: {type(exc).__name__}: {exc}")

    return DicomImage(path=str(p), ok=False, error=stable_error(" | ".join(errors), p))


_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")


def stable_error(text: str, path: Path) -> str:
    """Текст ошибки, одинаковый между прогонами и между способами запуска.

    Он попадает в `violation_type` таблицы, а таблица обязана быть
    воспроизводимой (ТЗ 2.7). Сообщения библиотек содержат то, что от
    прогона к прогону меняется: адреса объектов в памяти (pillow пишет
    `<_io.BytesIO object at 0x7f95...>`) и полный путь к файлу — а при
    входе из архива это путь во временном каталоге. Адреса убираются, путь
    заменяется именем файла, переводы строк — пробелами, чтобы ячейка
    таблицы была одной строкой. Поймано стресс-тестом: каталог и архив
    давали разный текст на одном и том же битом JPEG.
    """
    text = _ADDRESS.sub("", text)
    for prefix in (str(path), str(path.parent)):
        if prefix and prefix not in (".", "/"):
            text = text.replace(prefix, path.name if prefix == str(path) else "")
    return " ".join(text.split())


def iter_dicom_files(root: str | Path):
    """Рекурсивный обход каталога.

    Расширение файла не используется как фильтр: в закрытом наборе файлы
    могут быть без .dcm (ТЗ структуру каталога не фиксирует).
    """
    root = Path(root)
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.name.startswith("."):
            yield p


def is_dicom(path: str | Path) -> bool:
    """Быстрая проверка, что файл похож на DICOM, без полного разбора."""
    return _minidicom.is_dicom(str(path))
