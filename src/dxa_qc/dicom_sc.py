"""Overlay как DICOM-серия: Secondary Capture Image Storage (ТЗ 2.6, 2.7).

ТЗ 2.6 просит «дополнительную серию с визуализацией обнаруженного
нарушения», ТЗ 2.7 — «zip-архив с дополнительными сериями». Картинка
overlay (`overlay.render`) упаковывается в DICOM-объект, который ложится в
**то же исследование**, что и исходный снимок, отдельной серией: его можно
загрузить в PACS или открыть в любом DICOM-просмотрщике рядом с оригиналом.

Что записывается
----------------
* SOP Class — Secondary Capture Image Storage (1.2.840.10008.5.1.4.1.1.7),
  Explicit VR Little Endian, без сжатия: читается любым просмотрщиком;
* RGB 8 бит, `BurnedInAnnotation=YES` (подписи вшиты в пиксели),
  `ImageType=DERIVED\\SECONDARY`, `ConversionType=WSD`;
* StudyInstanceUID и атрибуты пациента/исследования — из исходного файла;
  значения, не проходящие проверку своего VR (в обучающем наборе даты и
  время заменены словом «Anonymized»), пишутся пустыми, а не копируются;
* ссылка на исходный снимок — SourceImageSequence (0008,2112);
* вердикт — в ImageComments (0020,4000): `quality_class` и нарушения.

UID детерминированы: `2.25.<128 бит SHA-256>` от исходных UID и пути файла,
поэтому два прогона на одних данных дают побайтово одинаковые файлы (ТЗ 2.7,
воспроизводимость), а дата и время создания не пишутся вовсе.

Запись — без pydicom, как и встроенный ридер `_minidicom`: overlay не
должен зависеть от того, импортировался ли pydicom. Совместимость проверена
чтением pydicom 3.0.1 в строгом режиме (тест `tests/test_dicom_sc.py`
делает это, если pydicom установлен).
"""
from __future__ import annotations

import hashlib
import re
import struct
from typing import Any, Iterable

import numpy as np

from . import __version__

SC_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.7"
EXPLICIT_VR_LE = "1.2.840.10008.1.2.1"
SERIES_NUMBER = 9001
SERIES_DESCRIPTION = "DXA QC overlay"
MANUFACTURER = "dxa-qc"

LONG_VRS = {"OB", "OD", "OF", "OL", "OV", "OW", "SQ", "SV", "UC", "UN", "UR", "UT", "UV"}
# Максимальная длина значения по PS3.5 6.2 (в байтах, без учёта многозначности).
MAX_LEN = {"CS": 16, "SH": 16, "LO": 64, "PN": 64, "DA": 8, "TM": 14, "IS": 12,
           "UI": 64, "ST": 1024, "LT": 10240}


def derive_uid(*parts: Any) -> str:
    """UID в корне 2.25 (PS3.5 B.2) из хэша: одинаковые входы — одинаковый UID."""
    h = hashlib.sha256("\x1f".join(str(p) for p in parts).encode("utf-8")).digest()
    return "2.25." + str(int.from_bytes(h[:16], "big"))


IMPLEMENTATION_UID = derive_uid("dxa-qc", "implementation-class")


# --------------------------------------------------------------------------- #
# проверка и очистка скопированных значений
# --------------------------------------------------------------------------- #
_DA = re.compile(r"\d{8}")
_TM = re.compile(r"\d{2}(\d{2}(\d{2}(\.\d{1,6})?)?)?")


def clean(vr: str, value: Any) -> str:
    """Значение, пригодное для VR, или пустая строка (атрибут типа 2)."""
    if value is None:
        return ""
    v = str(value).strip()
    if vr == "DA":
        return v if _DA.fullmatch(v) else ""
    if vr == "TM":
        return v if _TM.fullmatch(v) else ""
    if vr == "CS" and v not in ("M", "F", "O"):   # CS здесь только PatientSex
        return ""
    v = "".join(ch for ch in v if ch.isprintable()).replace("\\", "/")
    limit = MAX_LEN.get(vr, 64)
    while len(v.encode("utf-8")) > limit:
        v = v[:-1]
    return v


# --------------------------------------------------------------------------- #
# кодирование элементов Explicit VR Little Endian
# --------------------------------------------------------------------------- #
def _pad(vr: str, raw: bytes) -> bytes:
    if len(raw) % 2:
        raw += b"\x00" if vr in ("UI", "OB", "UN") else b" "
    return raw


def _value(vr: str, value: Any) -> bytes:
    if vr == "US":
        return struct.pack("<" + "H" * len(value), *value) if isinstance(value, (list, tuple)) \
            else struct.pack("<H", int(value))
    if vr == "UL":
        return struct.pack("<I", int(value))
    if vr in ("OB", "UN"):
        return bytes(value)
    if isinstance(value, (list, tuple)):
        value = "\\".join(str(v) for v in value)
    return str(value).encode("utf-8")


def element(tag: tuple[int, int], vr: str, value: Any) -> bytes:
    raw = value if vr == "SQ" else _pad(vr, _value(vr, value))
    head = struct.pack("<HH", *tag) + vr.encode("ascii")
    if vr in LONG_VRS:
        return head + b"\x00\x00" + struct.pack("<I", len(raw)) + raw
    if len(raw) > 0xFFFF:
        raise ValueError(f"значение {tag} длиннее 65535 байт для VR {vr}")
    return head + struct.pack("<H", len(raw)) + raw


def sequence(tag: tuple[int, int], items: Iterable[bytes]) -> bytes:
    """SQ определённой длины: каждый item — уже закодированные элементы."""
    body = b"".join(struct.pack("<HHI", 0xFFFE, 0xE000, len(it)) + it for it in items)
    return element(tag, "SQ", body)


def _dataset(elements: list[tuple[tuple[int, int], bytes]]) -> bytes:
    tags = [t for t, _ in elements]
    if tags != sorted(tags) or len(set(tags)) != len(tags):
        raise ValueError("элементы набора данных не по возрастанию тегов")
    return b"".join(e for _, e in elements)


# --------------------------------------------------------------------------- #
# публичный интерфейс
# --------------------------------------------------------------------------- #
def secondary_capture(bgr: np.ndarray, *, study_uid: str, series_uid: str, sop_uid: str,
                      instance_number: int, context: dict[str, str] | None = None,
                      source_sop_class: str | None = None, source_sop_uid: str | None = None,
                      comments: str = "", description: str = "") -> bytes:
    """Файл DICOM (Part 10) с RGB-картинкой `bgr` (порядок каналов OpenCV)."""
    a = np.asarray(bgr)
    if a.ndim != 3 or a.shape[2] != 3 or a.dtype != np.uint8:
        raise ValueError(f"ожидается BGR uint8 H×W×3, получено {a.dtype} {a.shape}")
    rows, cols = a.shape[:2]
    if not (0 < rows <= 0xFFFF and 0 < cols <= 0xFFFF):
        raise ValueError(f"размер {rows}×{cols} не помещается в Rows/Columns")
    pixels = np.ascontiguousarray(a[:, :, ::-1]).tobytes()   # BGR -> RGB, interleaved
    ctx = context or {}

    ds: list[tuple[tuple[int, int], bytes]] = []

    def add(tag, vr, value):
        ds.append((tag, element(tag, vr, value)))

    add((0x0008, 0x0005), "CS", "ISO_IR 192")
    add((0x0008, 0x0008), "CS", ["DERIVED", "SECONDARY"])
    add((0x0008, 0x0016), "UI", SC_SOP_CLASS)
    add((0x0008, 0x0018), "UI", sop_uid)
    add((0x0008, 0x0020), "DA", clean("DA", ctx.get("study_date")))
    add((0x0008, 0x0030), "TM", clean("TM", ctx.get("study_time")))
    add((0x0008, 0x0050), "SH", clean("SH", ctx.get("accession_number")))
    add((0x0008, 0x0060), "CS", "OT")
    add((0x0008, 0x0064), "CS", "WSD")
    add((0x0008, 0x0070), "LO", MANUFACTURER)
    add((0x0008, 0x0090), "PN", clean("PN", ctx.get("referring_physician")))
    add((0x0008, 0x103E), "LO", clean("LO", SERIES_DESCRIPTION))
    if description:
        add((0x0008, 0x2111), "ST", clean("ST", description))
    if source_sop_class and source_sop_uid:
        item = (element((0x0008, 0x1150), "UI", clean("UI", source_sop_class))
                + element((0x0008, 0x1155), "UI", clean("UI", source_sop_uid)))
        ds.append(((0x0008, 0x2112), sequence((0x0008, 0x2112), [item])))
    add((0x0010, 0x0010), "PN", clean("PN", ctx.get("patient_name")))
    add((0x0010, 0x0020), "LO", clean("LO", ctx.get("patient_id")))
    add((0x0010, 0x0030), "DA", clean("DA", ctx.get("patient_birth_date")))
    add((0x0010, 0x0040), "CS", clean("CS", ctx.get("patient_sex")))
    add((0x0018, 0x1016), "LO", MANUFACTURER)
    add((0x0018, 0x1018), "LO", "overlay")
    add((0x0018, 0x1019), "LO", __version__)
    add((0x0020, 0x000D), "UI", study_uid)
    add((0x0020, 0x000E), "UI", series_uid)
    add((0x0020, 0x0010), "SH", clean("SH", ctx.get("study_id")))
    add((0x0020, 0x0011), "IS", str(SERIES_NUMBER))
    add((0x0020, 0x0013), "IS", str(int(instance_number)))
    add((0x0020, 0x0020), "CS", "")
    if comments:
        add((0x0020, 0x4000), "LT", clean("LT", comments))
    add((0x0028, 0x0002), "US", 3)
    add((0x0028, 0x0004), "CS", "RGB")
    add((0x0028, 0x0006), "US", 0)
    add((0x0028, 0x0010), "US", rows)
    add((0x0028, 0x0011), "US", cols)
    add((0x0028, 0x0100), "US", 8)
    add((0x0028, 0x0101), "US", 8)
    add((0x0028, 0x0102), "US", 7)
    add((0x0028, 0x0103), "US", 0)
    add((0x0028, 0x0301), "CS", "YES")
    add((0x0028, 0x2110), "CS", "00")
    add((0x7FE0, 0x0010), "OB", pixels)
    body = _dataset(ds)

    meta = (element((0x0002, 0x0001), "OB", b"\x00\x01")
            + element((0x0002, 0x0002), "UI", SC_SOP_CLASS)
            + element((0x0002, 0x0003), "UI", sop_uid)
            + element((0x0002, 0x0010), "UI", EXPLICIT_VR_LE)
            + element((0x0002, 0x0012), "UI", IMPLEMENTATION_UID)
            + element((0x0002, 0x0013), "SH", clean("SH", f"DXA_QC_{__version__}")))
    group_len = element((0x0002, 0x0000), "UL", len(meta))
    return b"\x00" * 128 + b"DICM" + group_len + meta + body
