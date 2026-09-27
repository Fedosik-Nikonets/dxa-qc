"""Минимальный DICOM-ридер без внешних зависимостей.

Зачем он нужен, если в requirements.txt есть pydicom
----------------------------------------------------
1. Отладочные и CI-окружения без сетевого доступа к PyPI (у нас именно такое).
2. Страховка на техэкспертизе: если импорт pydicom по любой причине упадёт,
   сервис продолжит обрабатывать несжатые исследования, а не вернёт Failure
   на весь пакет.

Что поддерживается
------------------
* Implicit VR Little Endian (1.2.840.10008.1.2)
* Explicit VR Little Endian (1.2.840.10008.1.2.1)
* Explicit VR Big Endian (1.2.840.10008.1.2.2, retired, встречается у старых GE)
* BitsAllocated 8 и 16, PixelRepresentation 0 и 1
* MONOCHROME1 и MONOCHROME2
* Multi-frame (0028,0008 NumberOfFrames)
* RescaleSlope / RescaleIntercept, WindowCenter / WindowWidth
* Пропуск вложенных последовательностей (SQ), в том числе с неопределённой длиной

Чего сознательно нет: сжатые TransferSyntax (JPEG / JPEG-LS / JPEG2000 / RLE).
Для них возвращается явная ошибка с просьбой использовать pydicom — обработка
такого файла без специализированного декодера невозможна.
"""
from __future__ import annotations

import struct
from typing import Any, Iterator

# --------------------------------------------------------------------------- #
# константы
# --------------------------------------------------------------------------- #
TS_IMPLICIT_LE = "1.2.840.10008.1.2"
TS_EXPLICIT_LE = "1.2.840.10008.1.2.1"
TS_EXPLICIT_BE = "1.2.840.10008.1.2.2"
TS_DEFLATED = "1.2.840.10008.1.2.1.99"

UNCOMPRESSED_TS = {TS_IMPLICIT_LE, TS_EXPLICIT_LE, TS_EXPLICIT_BE}

EXPLICIT_VRS = {
    b"AE", b"AS", b"AT", b"CS", b"DA", b"DS", b"DT", b"FL", b"FD", b"IS", b"LO",
    b"LT", b"OB", b"OD", b"OF", b"OL", b"OV", b"OW", b"PN", b"SH", b"SL", b"SQ",
    b"SS", b"ST", b"SV", b"TM", b"UC", b"UI", b"UL", b"UN", b"UR", b"US", b"UT",
    b"UV",
}
# VR с 4-байтовым полем длины (после двух зарезервированных байт)
LONG_VRS = {b"OB", b"OD", b"OF", b"OL", b"OV", b"OW", b"SQ", b"SV", b"UC",
            b"UN", b"UR", b"UT", b"UV"}

ITEM_TAG = (0xFFFE, 0xE000)
ITEM_DELIM_TAG = (0xFFFE, 0xE00D)
SEQ_DELIM_TAG = (0xFFFE, 0xE0DD)
PIXEL_DATA_TAG = (0x7FE0, 0x0010)
UNDEFINED_LENGTH = 0xFFFFFFFF

# Минимальный словарь VR для Implicit VR: без него нельзя понять,
# как интерпретировать байты значения.
TAG_VR: dict[tuple[int, int], bytes] = {
    (0x0008, 0x0018): b"UI",  # SOPInstanceUID
    (0x0008, 0x0016): b"UI",  # SOPClassUID
    (0x0008, 0x0005): b"CS",  # SpecificCharacterSet
    (0x0008, 0x0020): b"DA",  # StudyDate
    (0x0008, 0x0030): b"TM",  # StudyTime
    (0x0008, 0x0050): b"SH",  # AccessionNumber
    (0x0008, 0x0060): b"CS",  # Modality
    (0x0008, 0x0070): b"LO",  # Manufacturer
    (0x0008, 0x1090): b"LO",  # ManufacturerModelName
    (0x0008, 0x0090): b"PN",  # ReferringPhysicianName
    (0x0008, 0x103E): b"LO",  # SeriesDescription
    (0x0010, 0x0010): b"PN",  # PatientName
    (0x0010, 0x0020): b"LO",  # PatientID
    (0x0010, 0x0030): b"DA",  # PatientBirthDate
    (0x0010, 0x0040): b"CS",  # PatientSex
    (0x0018, 0x0015): b"CS",  # BodyPartExamined
    (0x0018, 0x1164): b"DS",  # ImagerPixelSpacing
    (0x0018, 0x5101): b"CS",  # ViewPosition
    (0x0020, 0x000D): b"UI",  # StudyInstanceUID
    (0x0020, 0x000E): b"UI",  # SeriesInstanceUID
    (0x0020, 0x0010): b"SH",  # StudyID
    (0x0020, 0x0013): b"IS",  # InstanceNumber
    (0x0020, 0x0060): b"CS",  # Laterality
    (0x0028, 0x0002): b"US",  # SamplesPerPixel
    (0x0028, 0x0004): b"CS",  # PhotometricInterpretation
    (0x0028, 0x0006): b"US",  # PlanarConfiguration
    (0x0028, 0x0008): b"IS",  # NumberOfFrames
    (0x0028, 0x0010): b"US",  # Rows
    (0x0028, 0x0011): b"US",  # Columns
    (0x0028, 0x0030): b"DS",  # PixelSpacing
    (0x0028, 0x0100): b"US",  # BitsAllocated
    (0x0028, 0x0101): b"US",  # BitsStored
    (0x0028, 0x0102): b"US",  # HighBit
    (0x0028, 0x0103): b"US",  # PixelRepresentation
    (0x0028, 0x1050): b"DS",  # WindowCenter
    (0x0028, 0x1051): b"DS",  # WindowWidth
    (0x0028, 0x1052): b"DS",  # RescaleIntercept
    (0x0028, 0x1053): b"DS",  # RescaleSlope
    (0x0040, 0x0303): b"US",  # ExposedArea
    PIXEL_DATA_TAG: b"OW",
}


class DicomParseError(ValueError):
    """Файл не разобран. Текст пригоден для колонки comment выходной таблицы."""


# --------------------------------------------------------------------------- #
# разбор потока элементов
# --------------------------------------------------------------------------- #
def _iter_elements(
    buf: bytes, pos: int, end: int, explicit: bool, little: bool
) -> Iterator[tuple[int, int, bytes, int, int]]:
    """Итерирует элементы верхнего уровня.

    Возвращает (group, element, vr, value_start, value_end).
    Последовательности пропускаются целиком, PixelData с неопределённой
    длиной (encapsulated) возвращается как есть — решение принимает вызывающий.
    """
    while pos < end:
        h = _read_header(buf, pos, end, explicit, little)
        if h is None:
            return
        grp, el, vr, ln, pos = h

        if (grp, el) in (ITEM_TAG, ITEM_DELIM_TAG, SEQ_DELIM_TAG):
            # на верхнем уровне быть не должно; если встретилось — пропускаем
            if ln != UNDEFINED_LENGTH:
                pos += ln
            continue

        if ln == UNDEFINED_LENGTH:
            if (grp, el) == PIXEL_DATA_TAG:
                # encapsulated pixel data: отдаём хвост, декодировать не умеем
                yield grp, el, b"OB_ENCAPSULATED", pos, end
                return
            # последовательность с неопределённой длиной
            pos = _skip_sequence(buf, pos, end, explicit, little)
            continue

        val_end = min(pos + ln, end)
        if vr == b"SQ":
            pos = val_end
            continue
        yield grp, el, vr, pos, val_end
        pos = val_end


def _read_header(
    buf: bytes, pos: int, end: int, explicit: bool, little: bool
) -> tuple[int, int, bytes, int, int] | None:
    """Читает заголовок одного элемента: (group, element, vr, length, new_pos)."""
    e = "<" if little else ">"
    if pos + 8 > end:
        return None
    grp, el = struct.unpack(e + "HH", buf[pos : pos + 4])
    pos += 4
    if (grp, el) in (ITEM_TAG, ITEM_DELIM_TAG, SEQ_DELIM_TAG):
        # элементы-разделители всегда имеют 4-байтовую длину без VR
        ln = struct.unpack(e + "I", buf[pos : pos + 4])[0]
        return grp, el, b"NONE", ln, pos + 4
    if explicit:
        vr = buf[pos : pos + 2]
        if vr in EXPLICIT_VRS:
            pos += 2
            if vr in LONG_VRS:
                pos += 2
                if pos + 4 > end:
                    return None
                ln = struct.unpack(e + "I", buf[pos : pos + 4])[0]
                return grp, el, vr, ln, pos + 4
            if pos + 2 > end:
                return None
            ln = struct.unpack(e + "H", buf[pos : pos + 2])[0]
            return grp, el, vr, ln, pos + 2
        vr = TAG_VR.get((grp, el), b"UN")
    else:
        vr = TAG_VR.get((grp, el), b"UN")
    if pos + 4 > end:
        return None
    ln = struct.unpack(e + "I", buf[pos : pos + 4])[0]
    return grp, el, vr, ln, pos + 4


def _skip_sequence(buf: bytes, pos: int, end: int, explicit: bool, little: bool) -> int:
    """Пропускает SQ с неопределённой длиной: items до Sequence Delimitation Item."""
    while pos < end:
        h = _read_header(buf, pos, end, explicit, little)
        if h is None:
            return end
        grp, el, _vr, ln, pos = h
        if (grp, el) == SEQ_DELIM_TAG:
            return pos
        if (grp, el) == ITEM_TAG:
            pos = (
                _skip_item(buf, pos, end, explicit, little)
                if ln == UNDEFINED_LENGTH
                else min(pos + ln, end)
            )
            continue
        # мусор внутри SQ — пропускаем по длине
        pos = end if ln == UNDEFINED_LENGTH else min(pos + ln, end)
    return end


def _skip_item(buf: bytes, pos: int, end: int, explicit: bool, little: bool) -> int:
    """Пропускает item с неопределённой длиной: элементы до Item Delimitation Item."""
    while pos < end:
        h = _read_header(buf, pos, end, explicit, little)
        if h is None:
            return end
        grp, el, vr, ln, pos = h
        if (grp, el) == ITEM_DELIM_TAG:
            return pos
        if (grp, el) == SEQ_DELIM_TAG:  # некорректная вложенность — выходим
            return pos
        if ln == UNDEFINED_LENGTH:
            pos = _skip_sequence(buf, pos, end, explicit, little)
            continue
        pos = min(pos + ln, end)
    return end


# --------------------------------------------------------------------------- #
# приведение значений
# --------------------------------------------------------------------------- #
def _dec(raw: bytes) -> str:
    return raw.decode("latin-1", "replace").strip("\x00 \t\r\n")


def _as_value(vr: bytes, raw: bytes, little: bool) -> Any:
    e = "<" if little else ">"
    if vr in (b"US", b"SS"):
        n = len(raw) // 2
        if n == 0:
            return None
        fmt = "H" if vr == b"US" else "h"
        vals = struct.unpack(e + fmt * n, raw[: n * 2])
        return vals[0] if n == 1 else list(vals)
    if vr in (b"UL", b"SL"):
        n = len(raw) // 4
        if n == 0:
            return None
        fmt = "I" if vr == b"UL" else "i"
        vals = struct.unpack(e + fmt * n, raw[: n * 4])
        return vals[0] if n == 1 else list(vals)
    if vr in (b"DS", b"IS"):
        parts = [p for p in _dec(raw).split("\\") if p.strip() != ""]
        out: list[float] = []
        for p in parts:
            try:
                out.append(float(p))
            except ValueError:
                continue
        if not out:
            return None
        return out[0] if len(out) == 1 else out
    if vr in (b"FL", b"FD"):
        size = 4 if vr == b"FL" else 8
        fmt = "f" if vr == b"FL" else "d"
        n = len(raw) // size
        if n == 0:
            return None
        vals = struct.unpack(e + fmt * n, raw[: n * size])
        return vals[0] if n == 1 else list(vals)
    return _dec(raw) or None


# --------------------------------------------------------------------------- #
# публичный интерфейс
# --------------------------------------------------------------------------- #
def read_file(path: str, with_pixels: bool = True) -> dict[str, Any]:
    """Разбирает DICOM-файл. Бросает DicomParseError при неподдерживаемом формате.

    Возвращает словарь: TransferSyntaxUID, values (tag -> значение),
    pixel_raw (bytes) и pixel_len, если запрошены пиксели.
    """
    with open(path, "rb") as f:
        buf = f.read()

    if len(buf) < 132:
        raise DicomParseError(f"файл слишком мал ({len(buf)} байт)")

    if buf[128:132] == b"DICM":
        meta: dict[tuple[int, int], Any] = {}
        pos = 132
        meta_end = len(buf)
        h = _read_header(buf, pos, len(buf), True, True)
        if h and (h[0], h[1]) == (0x0002, 0x0000):
            grp_len = struct.unpack("<I", buf[h[4] : h[4] + 4])[0]
            meta_end = h[4] + h[3] + grp_len
        for grp, el, vr, s, en in _iter_elements(buf, pos, min(meta_end, len(buf)), True, True):
            if grp != 0x0002:
                break
            meta[(grp, el)] = _as_value(vr, buf[s:en], True)
        pos = min(meta_end, len(buf))
        ts = str(meta.get((0x0002, 0x0010)) or "").strip()
    else:
        # Файл без преамбулы и DICM: часть архивов отдаёт «голый» dataset.
        ts = TS_IMPLICIT_LE
        pos = 0

    if not ts:
        ts = TS_IMPLICIT_LE
    if ts == TS_DEFLATED:
        raise DicomParseError("Deflated Explicit VR LE не поддерживается встроенным ридером")
    if ts not in UNCOMPRESSED_TS:
        raise DicomParseError(
            f"сжатый TransferSyntax {ts}: нужен pydicom с pylibjpeg"
        )

    explicit = ts != TS_IMPLICIT_LE
    little = ts != TS_EXPLICIT_BE

    values: dict[tuple[int, int], Any] = {}
    pixel_raw: bytes | None = None
    encapsulated = False
    for grp, el, vr, s, en in _iter_elements(buf, pos, len(buf), explicit, little):
        if (grp, el) == PIXEL_DATA_TAG:
            if vr == b"OB_ENCAPSULATED":
                encapsulated = True
                break
            if with_pixels:
                pixel_raw = buf[s:en]
            values[(grp, el)] = None
            break
        values[(grp, el)] = _as_value(vr, buf[s:en], little)

    if encapsulated:
        raise DicomParseError("encapsulated PixelData (сжатие): нужен pydicom")

    return {
        "TransferSyntaxUID": ts,
        "explicit": explicit,
        "little": little,
        "values": values,
        "pixel_raw": pixel_raw,
        "file_size": len(buf),
    }


def get(values: dict, tag: tuple[int, int], default: Any = None) -> Any:
    v = values.get(tag, default)
    return default if v is None else v


def is_dicom(path: str) -> bool:
    """Проверка магии DICM. Файл без преамбулы проверяется по первому тегу."""
    try:
        with open(path, "rb") as f:
            head = f.read(140)
    except OSError:
        return False
    if len(head) >= 132 and head[128:132] == b"DICM":
        return True
    # «голый» dataset: первый элемент обычно из группы 0008 с разумной длиной
    if len(head) >= 8:
        grp, el = struct.unpack("<HH", head[:4])
        if grp in (0x0002, 0x0008) and el < 0x2000:
            return True
    return False
