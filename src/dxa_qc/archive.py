"""Вход из zip-архива и поиск каталога с исследованиями (ТЗ 2.7).

Две вещи, на которых пакетная обработка может молча сломаться.

*Имена в архиве.* Оба архива организатора собраны на Windows: имена в
кодировке cp866 и без флага UTF-8, поэтому стандартный `zipfile` читает
«Исследования» как «êßß½Ññ«óá¡¿∩». Имена здесь перекодируются: сначала
пробуется UTF-8 (так пишут Linux-утилиты без флага), затем cp866.

*Уровень каталога.* Исследование — первый компонент пути внутри корня
исследований, а роутер считает, что в исследовании не больше одного
позвоночника. Если корень указан уровнем выше (архив
`НД_для_обучения.zip` содержит `Исследования/<исследования>` и
`разметка.xlsx`), все 499 файлов стали бы одним исследованием, и роутер
нашёл бы один позвоночник из 99. Поэтому обёртки снимаются: пока на
уровне ровно один подкаталог, нет файлов DICOM и файлов под ним больше,
чем бывает в одном исследовании, корень опускается на уровень ниже. Порог
выбран с запасом: самое большое исследование обучающего набора — 29
файлов, ТЗ 2.4 допускает до трёх серий. Единственное исследование в
архиве (как «Для теста.zip») так не разворачивается.

Распаковка безопасна: пути с `..`, абсолютные пути и символические ссылки
отвергаются, суммарный размер и число файлов ограничены — архив приходит
по API от кого угодно.
"""
from __future__ import annotations

import os
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath

from .dicom_io import is_dicom

MAX_FILES_PER_STUDY = 50
MAX_UNZIP_BYTES = int(os.environ.get("DXA_QC_MAX_UNZIP_BYTES", 20 * 1024 ** 3))
MAX_UNZIP_FILES = int(os.environ.get("DXA_QC_MAX_UNZIP_FILES", 200_000))


class ArchiveError(ValueError):
    """Архив нельзя распаковать безопасно или он не архив."""


def decode_name(info: zipfile.ZipInfo) -> str:
    """Имя члена архива с исправленной кодировкой."""
    name = info.filename
    if info.flag_bits & 0x800:          # флаг UTF-8 стоит — zipfile уже прав
        return name
    try:
        raw = name.encode("cp437")
    except UnicodeEncodeError:          # имя уже не из cp437 — оставляем
        return name
    if raw.isascii():
        return name
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("cp866")


def _safe_parts(name: str) -> tuple[str, ...]:
    p = PurePosixPath(name.replace("\\", "/"))
    parts = tuple(x for x in p.parts if x not in ("", "."))
    if p.is_absolute() or not parts or ".." in parts or ":" in parts[0]:
        raise ArchiveError(f"небезопасный путь в архиве: {name!r}")
    return parts


def extract(archive: str | Path, dest: str | Path) -> int:
    """Распаковывает архив в `dest`. Возвращает число файлов."""
    archive, dest = Path(archive), Path(dest)
    if not zipfile.is_zipfile(archive):
        raise ArchiveError("файл не является zip-архивом")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        if len(infos) > MAX_UNZIP_FILES:
            raise ArchiveError(f"в архиве {len(infos)} файлов, предел {MAX_UNZIP_FILES}")
        if sum(i.file_size for i in infos) > MAX_UNZIP_BYTES:
            raise ArchiveError(f"архив больше {MAX_UNZIP_BYTES} байт в распакованном виде")
        written = 0
        for info in infos:
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ArchiveError(f"символическая ссылка в архиве: {info.filename!r}")
            parts = _safe_parts(decode_name(info))
            if parts[0] == "__MACOSX":
                continue                # служебные копии архиватора macOS
            target = dest.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                # Заголовку о размере не верим: считаем, сколько реально пришло.
                while chunk := src.read(1 << 20):
                    written += len(chunk)
                    if written > MAX_UNZIP_BYTES:
                        raise ArchiveError("распакованный объём превысил предел")
                    out.write(chunk)
        return len(infos)


def studies_root(root: str | Path, files: list[Path] | None = None) -> Path:
    """Каталог, в котором лежат папки исследований (снимаются обёртки)."""
    root = Path(root)
    files = [Path(f) for f in files] if files is not None else [
        p for p in sorted(root.rglob("*")) if p.is_file() and not p.name.startswith(".")]
    cur = root
    while True:
        entries = [e for e in cur.iterdir() if not e.name.startswith(".") and e.name != "__MACOSX"]
        subdirs = [e for e in entries if e.is_dir()]
        direct = [e for e in entries if e.is_file()]
        if len(subdirs) != 1 or any(is_dicom(f) for f in direct):
            return cur
        below = sum(1 for f in files if subdirs[0] in f.parents)
        if below <= MAX_FILES_PER_STUDY:
            return cur
        cur = subdirs[0]


def copy_upload(src, dest: str | Path) -> Path:
    """Поток загрузки -> файл на диске (без чтения целиком в память)."""
    dest = Path(dest)
    with open(dest, "wb") as out:
        shutil.copyfileobj(src, out, 1 << 20)
    return dest
