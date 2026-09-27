#!/usr/bin/env python3
"""Стресс-тест сквозного прогона (план, 25 сентября; ТЗ 2.7).

Строит корпус «всего плохого, что может прийти в закрытом наборе», и
прогоняет его через `batch.run` — каталогом и zip-архивом. Требования:

1. ни одного исключения наружу, код возврата 0;
2. строка на каждый файл (кроме скрытых), статус только Success/Failure,
   у Failure — текст причины и пустой `quality_class`;
3. **преобразования без потери информации не меняют решение**: реальные
   кадры обучающего набора, перепакованные в MONOCHROME1, 16 бит и
   Explicit VR Big Endian, дают ту же область и те же нарушения, что и
   перепакованный без изменений кадр;
4. время на исследование — с запасом меньше 3 минут (ТЗ 2.7), в том числе
   на кадре 2048×2048.

Корпус пишется на диск, поэтому его же можно прогнать через контейнер:
    python scripts/18_stress_test.py --make-only /tmp/stress
    ./run.sh /tmp/stress/corpus /tmp/stress/out_docker

Запуск (секунды):
    python scripts/18_stress_test.py --out reports/stress
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from dxa_qc import batch  # noqa: E402
from dxa_qc.dicom_io import read_dicom  # noqa: E402
from synth_dicom import (TS_EXPLICIT_BE, TS_JPEG_BASELINE, build_dicom,  # noqa: E402
                         phantom)

DATA = ROOT / "Data/НД_для_обучения/Исследования"
# Кадры с каждым типом решения: наклон оси, артефакт, укладка, ротация,
# охват + ротация, норма. Те же, что на reports/overlay_examples.png.
REAL = {
    "tilt": "2.25.102755089973625799055786462646268820450/series_002_2_CR/A2504257770 DXA/CR DXA/CR000001.dcm",
    "artifact": "1.2.840.113619.2.110.512719.20250403092052/series_003_3_CR/DXA/CR DXA/CR000001.dcm",
    "positioning": "2.25.125521310320333648427417313254965624364/series_001_2_CR/A250444126 DXA/CR DXA/CR000000.dcm",
    "rotation": "1.2.840.113619.2.110.512719.20250407113211/series_001_2_CR/DXA/CR DXA/CR000001.dcm",
    "roi": "2.25.122446767306607067205618235949445789306/series_001_2_CR/A2504255080 DXA/CR DXA/CR000000.dcm",
    "normal": "1.2.840.113619.2.110.512719.20250403090444/series_004_3_CR/DXA/CR DXA/CR000001.dcm",
}
LOSSLESS = ("mono1", "bit16", "explicit_be")


def _u8(pixels: np.ndarray) -> np.ndarray:
    return np.clip(np.round(np.asarray(pixels, float) * 255), 0, 255).astype(np.uint8)


def make_corpus(dst: Path) -> dict:
    """Пишет корпус в dst/corpus. Возвращает описание ожиданий."""
    corpus = dst / "corpus"
    if corpus.exists():
        shutil.rmtree(corpus)
    expect: dict[str, dict] = {"lossless": {}, "garbage": [], "hidden": []}

    # --- реальные кадры и их перепаковки: одно исследование на вариант ---
    for key, rel in REAL.items():
        img = read_dicom(DATA / rel)
        if not img.ok:
            raise SystemExit(f"не читается {rel}: {img.error}")
        p8 = _u8(img.pixels)
        ea = img.exposed_area
        variants = {
            "rebuilt": dict(pixels=p8, exposed_area=ea),
            "mono1": dict(pixels=255 - p8, photometric="MONOCHROME1", exposed_area=ea),
            "bit16": dict(pixels=p8.astype(np.uint16) * 257, bits_allocated=16, exposed_area=ea),
            "explicit_be": dict(pixels=p8.astype(np.uint16) * 257, bits_allocated=16,
                                transfer_syntax=TS_EXPLICIT_BE, exposed_area=ea),
            "no_exposed_area": dict(pixels=p8),
            "pixel_spacing": dict(pixels=p8, pixel_spacing=(0.6, 0.6)),
        }
        for v, kw in variants.items():
            # Один кадр — одно исследование: роутер не путает кадры вариантов.
            path = corpus / f"real_{key}_{v}" / "series_001" / "CR000000.dcm"
            build_dicom(path, **kw, sop_uid=f"9.9.{len(expect['lossless'])}.{v}")
            expect["lossless"].setdefault(key, {})[v] = path.relative_to(corpus).as_posix()

    # --- мусор и краевые случаи: все в одном «исследовании» мусора ---
    g = corpus / "garbage" / "series_001"
    g.mkdir(parents=True)
    (g / "empty.dcm").write_bytes(b"")
    (g / "notes.dcm").write_text("это не DICOM, а текст с расширением .dcm", encoding="utf-8")
    (g / "noise.dcm").write_bytes(np.random.default_rng(0).integers(0, 256, 4096, np.uint8).tobytes())
    (g / "разметка.xlsx").write_bytes(b"PK\x03\x04 not really a workbook")
    build_dicom(g / "truncated.dcm", phantom(), truncate_pixels=500)
    build_dicom(g / "one_by_one.dcm", np.array([[128]]))
    build_dicom(g / "constant.dcm", np.full((200, 200), 77))
    build_dicom(g / "rgb.dcm", np.zeros((64, 64)), samples_per_pixel=3)
    build_dicom(g / "jpeg_garbage.dcm", phantom(), transfer_syntax=TS_JPEG_BASELINE, encapsulated=True)
    build_dicom(g / "no_preamble.dcm", phantom(200, 160), preamble=False)
    expect["garbage"] = sorted(p.relative_to(corpus).as_posix() for p in g.iterdir())
    (g / ".DS_Store").write_bytes(b"\x00\x00")
    expect["hidden"] = ["garbage/series_001/.DS_Store"]

    # --- непонятная область: пятно без анатомии ---
    rng = np.random.default_rng(1)
    blob = (rng.random((240, 240)) * 60).astype(int)
    yy, xx = np.mgrid[:240, :240]
    blob[((yy - 120) ** 2 + (xx - 120) ** 2) < 60 ** 2] = 230
    build_dicom(corpus / "blob" / "CR000000.dcm", blob)

    # --- большой кадр: время ---
    big = np.full((2048, 2048), 18, np.int32)
    big[:, 924:1124] = 220
    big[1400:, :700] = 80
    big[1400:, 1350:] = 80
    build_dicom(corpus / "large_2048" / "CR000000.dcm", big)

    # --- 29 копий одного кадра и кириллица с пробелами в путях ---
    one = corpus / "Исследование с пробелами" / "серия 1"
    src = corpus / expect["lossless"]["normal"]["rebuilt"]
    for i in range(29):
        one.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, one / f"копия {i:02d}.dcm")

    # --- пустое исследование ---
    (corpus / "empty_study" / "series_001").mkdir(parents=True)

    # --- пустой архив ---
    with zipfile.ZipFile(dst / "empty.zip", "w"):
        pass
    return expect


def _rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def check(dst: Path, expect: dict) -> dict:
    corpus = dst / "corpus"
    out: dict = {"checks": {}, "problems": []}
    bad = out["problems"]

    # 1–2. каталогом и архивом
    zpath = dst / "corpus.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(corpus.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(corpus).as_posix())
    files = sorted(p.relative_to(corpus).as_posix() for p in corpus.rglob("*")
                   if p.is_file() and not p.name.startswith("."))
    tables = {}
    for name, src in (("dir", corpus), ("zip", zpath)):
        t0 = time.perf_counter()
        try:
            code = batch.run(src, dst / f"{name}.csv", diagnostics=dst / f"{name}_diag.csv",
                             quiet=True, overlays=dst / f"{name}_overlays.zip")
        except Exception as exc:  # noqa: BLE001
            bad.append(f"{name}: исключение наружу: {type(exc).__name__}: {exc}")
            continue
        out["checks"][f"{name}_seconds"] = round(time.perf_counter() - t0, 2)
        if code != 0:
            bad.append(f"{name}: код возврата {code}")
        rows = _rows(dst / f"{name}.csv")
        tables[name] = rows
        paths = sorted(r["path_to_study"] for r in rows)
        if paths != files:
            bad.append(f"{name}: строки не совпадают с файлами "
                       f"(лишние {sorted(set(paths) - set(files))[:3]}, "
                       f"пропущены {sorted(set(files) - set(paths))[:3]})")
        for r in rows:
            st = r["processing_status"]
            if st not in ("Success", "Failure"):
                bad.append(f"{name}: статус {st!r} у {r['path_to_study']}")
            if st == "Failure" and (not r["violation_type"] or r["quality_class"] != ""):
                bad.append(f"{name}: Failure без причины или с классом: {r['path_to_study']}")
        out["checks"][f"{name}_rows"] = len(rows)
        out["checks"][f"{name}_failures"] = sum(r["processing_status"] == "Failure" for r in rows)
        with zipfile.ZipFile(dst / f"{name}_overlays.zip") as zf:
            out["checks"][f"{name}_overlay_png"] = sum(n.endswith(".png") for n in zf.namelist())
            out["checks"][f"{name}_overlay_dicom"] = sum(n.endswith(".overlay.dcm")
                                                         for n in zf.namelist())

    if "dir" in tables and "zip" in tables:
        strip = lambda rows: sorted((r["path_to_study"], r["anatomical_region"], r["quality_class"],  # noqa: E731
                                     r["violation_type"], r["processing_status"]) for r in rows)
        same = strip(tables["dir"]) == strip(tables["zip"])
        out["checks"]["dir_equals_zip"] = same
        if not same:
            bad.append("каталог и архив дали разные таблицы")

    # 3. перепаковки без потерь
    if "dir" in tables:
        by = {r["path_to_study"]: r for r in tables["dir"]}
        lossless = {}
        for key, vs in expect["lossless"].items():
            base = by[vs["rebuilt"]]
            rec = {"rebuilt": (base["anatomical_region"], base["violation_type"])}
            for v, rel in vs.items():
                r = by[rel]
                rec[v] = (r["anatomical_region"], r["violation_type"], r["processing_status"])
                if v in LOSSLESS and (r["anatomical_region"], r["violation_type"]) != \
                        (base["anatomical_region"], base["violation_type"]):
                    bad.append(f"перепаковка {v} изменила решение на кадре {key}: "
                               f"{rec['rebuilt']} -> {rec[v][:2]}")
            lossless[key] = rec
        out["lossless"] = lossless

        diag = {r["path_to_study"]: r for r in _rows(dst / "dir_diag.csv")}
        per_study: dict[str, float] = {}
        for r in tables["dir"]:
            per_study[r["study"]] = per_study.get(r["study"], 0.0) + float(r["time_of_processing"])
        out["checks"]["max_seconds_per_study"] = round(max(per_study.values()), 3)
        out["checks"]["large_2048_seconds"] = round(per_study.get("large_2048", float("nan")), 3)
        if max(per_study.values()) > 60:
            bad.append(f"исследование дольше минуты: {max(per_study, key=per_study.get)}")
        out["garbage"] = {rel: (by[rel]["processing_status"], by[rel]["violation_type"][:80])
                          for rel in expect["garbage"]}
        out["blob"] = (by["blob/CR000000.dcm"]["processing_status"],
                       by["blob/CR000000.dcm"]["violation_type"][:80])
        copies = [r for r in tables["dir"] if r["study"] == "Исследование с пробелами"]
        out["checks"]["copies_29_rows"] = len(copies)
        out["checks"]["copies_29_same_decision"] = len({(r["anatomical_region"], r["violation_type"])
                                                        for r in copies}) == 1
        out["checks"]["scale_sources"] = sorted({diag[p]["femur_mm_per_px_source"]
                                                 for p in diag if diag[p].get("femur_mm_per_px_source")})

    # пустой архив
    code = batch.run(dst / "empty.zip", dst / "empty.csv", quiet=True)
    out["checks"]["empty_zip_code"] = code
    out["checks"]["empty_zip_rows"] = len(_rows(dst / "empty.csv")) if (dst / "empty.csv").exists() else None
    if code != 0 or out["checks"]["empty_zip_rows"] != 0:
        bad.append("пустой архив: не пустая таблица или ненулевой код")

    out["passed"] = not bad
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "reports/stress")
    ap.add_argument("--make-only", type=Path, default=None,
                    help="только построить корпус в этом каталоге (для прогона через контейнер)")
    args = ap.parse_args()
    if args.make_only:
        args.make_only.mkdir(parents=True, exist_ok=True)
        make_corpus(args.make_only)
        print(f"корпус: {args.make_only / 'corpus'}")
        return 0

    import tempfile
    with tempfile.TemporaryDirectory(prefix="dxa-stress-") as tmp:
        tmp = Path(tmp)
        expect = make_corpus(tmp)
        res = check(tmp, expect)
        args.out.mkdir(parents=True, exist_ok=True)
        shutil.copy(tmp / "dir.csv", args.out / "table.csv")
    (args.out / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
