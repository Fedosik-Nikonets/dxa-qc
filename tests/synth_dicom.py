"""Генератор синтетических DICOM-файлов для тестов загрузчика.

Обучающий набор однороден (Implicit VR LE, 8 бит, MONOCHROME2, несжатый),
поэтому все краевые случаи, которые может прислать закрытый тестовый набор,
проверяются на синтетике. Генератор намеренно написан на `struct`, без pydicom:
тест не должен зависеть от той же библиотеки, которую он проверяет, и обязан
работать в окружении без сетевого доступа.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

TS_IMPLICIT_LE = "1.2.840.10008.1.2"
TS_EXPLICIT_LE = "1.2.840.10008.1.2.1"
TS_EXPLICIT_BE = "1.2.840.10008.1.2.2"
TS_JPEG_BASELINE = "1.2.840.10008.1.2.4.50"

LONG_VRS = {b"OB", b"OW", b"OF", b"SQ", b"UT", b"UN", b"UC", b"UR"}


def _pad(v: bytes) -> bytes:
    return v + b"\x00" if len(v) % 2 else v


def _enc_text(s: str) -> bytes:
    return _pad(s.encode("latin-1", "replace"))


def _element(tag: tuple[int, int], vr: bytes, value: bytes, explicit: bool, little: bool) -> bytes:
    e = "<" if little else ">"
    value = _pad(value)
    head = struct.pack(e + "HH", *tag)
    if not explicit:
        return head + struct.pack(e + "I", len(value)) + value
    if vr in LONG_VRS:
        return head + vr + b"\x00\x00" + struct.pack(e + "I", len(value)) + value
    return head + vr + struct.pack(e + "H", len(value)) + value


def _us(vals, little: bool) -> bytes:
    e = "<" if little else ">"
    if isinstance(vals, int):
        vals = [vals]
    return struct.pack(e + "H" * len(vals), *vals)


def build_dicom(
    path: str | Path,
    pixels: np.ndarray,
    *,
    transfer_syntax: str = TS_IMPLICIT_LE,
    photometric: str = "MONOCHROME2",
    bits_allocated: int = 8,
    bits_stored: int | None = None,
    pixel_repr: int = 0,
    n_frames: int = 1,
    window: tuple[float, float] | None = None,
    rescale: tuple[float, float] | None = None,
    pixel_spacing: tuple[float, float] | None = None,
    exposed_area: tuple[int, int] | None = None,
    samples_per_pixel: int = 1,
    truncate_pixels: int = 0,
    encapsulated: bool = False,
    preamble: bool = True,
    extra_sequence: bool = False,
    study_uid: str = "1.2.3.4.5",
    sop_uid: str = "1.2.3.4.5.6",
    instance_number: int = 1,
    extra_elements: dict | None = None,
) -> Path:
    """Собирает DICOM-файл заданного профиля.

    `pixels` — массив (rows, cols) или (frames, rows, cols) в native-значениях
    до нормировки. Тип приводится к BitsAllocated/PixelRepresentation.
    `extra_elements` — {тег: (VR, сырые байты)} для тегов между (0008,0005) и
    (0020,000D): кодировка, атрибуты пациента, ViewPosition; пишутся по
    возрастанию тегов.
    """
    path = Path(path)
    explicit = transfer_syntax != TS_IMPLICIT_LE
    little = transfer_syntax != TS_EXPLICIT_BE
    bits_stored = bits_stored or bits_allocated

    arr = np.asarray(pixels)
    if arr.ndim == 2:
        arr = arr[None, ...]
    frames, rows, cols = arr.shape
    dtype = np.dtype(("i" if pixel_repr else "u") + ("1" if bits_allocated == 8 else "2"))
    dtype = dtype.newbyteorder("<" if little else ">")
    pixel_bytes = np.ascontiguousarray(arr.astype(dtype)).tobytes()
    if truncate_pixels:
        pixel_bytes = pixel_bytes[:-truncate_pixels]

    el = lambda tag, vr, val: _element(tag, vr, val, explicit, little)  # noqa: E731

    extra = sorted((extra_elements or {}).items())
    head = [(t, v) for t, v in extra if t < (0x0008, 0x0016)]
    mid = [(t, v) for t, v in extra if t >= (0x0008, 0x0016)]

    ds = b""
    for tag, (vr, raw) in head:
        ds += el(tag, vr, _pad(raw))
    ds += el((0x0008, 0x0016), b"UI", _enc_text("1.2.840.10008.5.1.4.1.1.1"))
    ds += el((0x0008, 0x0018), b"UI", _enc_text(sop_uid))
    ds += el((0x0008, 0x0060), b"CS", _enc_text("CR"))
    ds += el((0x0008, 0x0070), b"LO", _enc_text("SYNTH"))
    ds += el((0x0008, 0x1090), b"LO", _enc_text("Prodigy Advance"))

    if extra_sequence:
        # Последовательность с неопределённой длиной: загрузчик обязан её
        # пропустить, не потеряв последующие теги.
        e = "<" if little else ">"
        item = struct.pack(e + "HHI", 0xFFFE, 0xE000, 0xFFFFFFFF)
        item += el((0x0008, 0x0100), b"SH", _enc_text("CODE"))
        item += struct.pack(e + "HHI", 0xFFFE, 0xE00D, 0)
        seq = struct.pack(e + "HH", 0x0008, 0x1032)
        if explicit:
            seq += b"SQ" + b"\x00\x00" + struct.pack(e + "I", 0xFFFFFFFF)
        else:
            seq += struct.pack(e + "I", 0xFFFFFFFF)
        seq += item + struct.pack(e + "HHI", 0xFFFE, 0xE0DD, 0)
        ds += seq

    for tag, (vr, raw) in mid:
        ds += el(tag, vr, _pad(raw))
    ds += el((0x0020, 0x000D), b"UI", _enc_text(study_uid))
    ds += el((0x0020, 0x000E), b"UI", _enc_text(study_uid + ".1"))
    ds += el((0x0020, 0x0013), b"IS", _enc_text(str(instance_number)))

    ds += el((0x0028, 0x0002), b"US", _us(samples_per_pixel, little))
    ds += el((0x0028, 0x0004), b"CS", _enc_text(photometric))
    if n_frames > 1:
        ds += el((0x0028, 0x0008), b"IS", _enc_text(str(n_frames)))
    ds += el((0x0028, 0x0010), b"US", _us(rows, little))
    ds += el((0x0028, 0x0011), b"US", _us(cols, little))
    if pixel_spacing:
        ds += el((0x0028, 0x0030), b"DS",
                 _enc_text(f"{pixel_spacing[0]}\\{pixel_spacing[1]}"))
    ds += el((0x0028, 0x0100), b"US", _us(bits_allocated, little))
    ds += el((0x0028, 0x0101), b"US", _us(bits_stored, little))
    ds += el((0x0028, 0x0102), b"US", _us(bits_stored - 1, little))
    ds += el((0x0028, 0x0103), b"US", _us(pixel_repr, little))
    if window:
        ds += el((0x0028, 0x1050), b"DS", _enc_text(str(window[0])))
        ds += el((0x0028, 0x1051), b"DS", _enc_text(str(window[1])))
    if rescale:
        ds += el((0x0028, 0x1052), b"DS", _enc_text(str(rescale[1])))
        ds += el((0x0028, 0x1053), b"DS", _enc_text(str(rescale[0])))
    if exposed_area:
        ds += el((0x0040, 0x0303), b"US", _us(list(exposed_area), little))

    e = "<" if little else ">"
    if encapsulated:
        # Encapsulated PixelData: неопределённая длина + basic offset table.
        ds += struct.pack(e + "HH", 0x7FE0, 0x0010)
        if explicit:
            ds += b"OB" + b"\x00\x00" + struct.pack(e + "I", 0xFFFFFFFF)
        else:
            ds += struct.pack(e + "I", 0xFFFFFFFF)
        ds += struct.pack(e + "HHI", 0xFFFE, 0xE000, 0)
        ds += struct.pack(e + "HHI", 0xFFFE, 0xE000, len(_pad(pixel_bytes)))
        ds += _pad(pixel_bytes)
        ds += struct.pack(e + "HHI", 0xFFFE, 0xE0DD, 0)
    else:
        vr = b"OB" if bits_allocated == 8 else b"OW"
        ds += el((0x7FE0, 0x0010), vr, pixel_bytes)

    meta = b""
    meta += _element((0x0002, 0x0002), b"UI",
                     _enc_text("1.2.840.10008.5.1.4.1.1.1"), True, True)
    meta += _element((0x0002, 0x0003), b"UI", _enc_text(sop_uid), True, True)
    meta += _element((0x0002, 0x0010), b"UI", _enc_text(transfer_syntax), True, True)
    header = b""
    if preamble:
        header += b"\x00" * 128 + b"DICM"
        header += _element((0x0002, 0x0000), b"UL",
                           struct.pack("<I", len(meta)), True, True)
        header += meta

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + ds)
    return path


def phantom(rows: int = 64, cols: int = 48, peak: int = 200, dtype_max: int = 255) -> np.ndarray:
    """Простой фантом: светлый вертикальный «позвоночник» на тёмном фоне."""
    a = np.full((rows, cols), int(dtype_max * 0.08), dtype=np.int32)
    c0, c1 = cols // 2 - cols // 8, cols // 2 + cols // 8
    r0, r1 = rows // 8, rows - rows // 8
    a[r0:r1, c0:c1] = peak
    return a
