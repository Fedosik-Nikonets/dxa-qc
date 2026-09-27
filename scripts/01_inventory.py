#!/usr/bin/env python3
"""Инвентаризация набора DICOM: метаданные, размеры, масштаб, дубликаты.

Запуск:
    python scripts/01_inventory.py --root data/train/Исследования \
        --out reports/inventory_train.csv

Скрипт ничего не предполагает о структуре каталогов: он обходит дерево,
отфильтровывает не-DICOM по магии DICM и пишет одну строку на файл.
Идентификатор исследования берётся из имени каталога верхнего уровня
относительно --root, потому что в этом наборе StudyInstanceUID из тегов
НЕ совпадает с идентификатором в разметке (см. day1_findings.md).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dxa_qc.dicom_io import HAVE_PYDICOM, is_dicom, iter_dicom_files, read_dicom  # noqa: E402


def study_key(path: Path, root: Path) -> str:
    """Первый компонент относительного пути = идентификатор исследования."""
    rel = path.relative_to(root)
    return rel.parts[0] if len(rel.parts) > 1 else rel.stem


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--no-pixels", action="store_true",
                    help="не читать PixelData (быстрее, но без хэшей и статистик)")
    args = ap.parse_args()

    root: Path = args.root
    if not root.exists():
        print(f"нет каталога: {root}", file=sys.stderr)
        return 2

    print(f"ридер: {'pydicom' if HAVE_PYDICOM else 'встроенный fallback'}")

    rows: list[dict] = []
    n_skipped = 0
    for p in iter_dicom_files(root):
        if not is_dicom(p):
            n_skipped += 1
            continue
        img = read_dicom(p, with_pixels=not args.no_pixels)
        rec = img.meta_dict()
        rec["study"] = study_key(p, root)
        rec["rel_path"] = str(p.relative_to(root))
        # В CSV не должно быть абсолютных путей конкретной машины: таблица
        # коммитится в репозиторий и должна читаться у обоих участников.
        rec["path"] = rec["rel_path"]
        if img.ok and img.pixels is not None:
            a = img.pixels
            rec["md5"] = hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()
            rec["mean"] = float(a.mean())
            rec["std"] = float(a.std())
            rec["frac_bright"] = float((a > 0.35).mean())
        rows.append(rec)

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    n_err = int((~df["ok"]).sum()) if "ok" in df else 0
    print(f"файлов: {len(df)}  не-DICOM пропущено: {n_skipped}  ошибок чтения: {n_err}")
    if "study" in df:
        print(f"исследований: {df['study'].nunique()}")
    if "md5" in df:
        print(f"уникальных изображений по пикселям: {df['md5'].nunique()} "
              f"(точных дублей: {len(df) - df['md5'].nunique()})")
    if n_err:
        print("\nошибки:")
        for _, r in df[~df["ok"]].head(20).iterrows():
            print(" ", r["rel_path"], "->", r["error"])
    print(f"\nзаписано: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
