"""Демо-набор для README: шесть синтетических исследований по три снимка.

Для каждого снимка задано, какое решение должен принять сервис (норма или
конкретное нарушение). Фантом рисуется со своим сидом, набор прогоняется
через настоящий `dxa_qc.batch`, и если решение не совпало с задуманным —
сид этого снимка сдвигается и прогон повторяется. Решения сервиса при этом
не подгоняются: подбирается только картинка.

    python docs/assets/src/make_demo.py docs/assets/src/build

В каталоге появятся `demo/` (DICOM), `demo.zip`, `result.csv`,
`diagnostics.csv` и `overlays.zip` с распакованной копией в `ovx/`.
"""
from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(HERE), str(REPO / "tests")]
import phantoms as ph  # noqa: E402
import synth_dicom as sd  # noqa: E402

# (исследование, файл, фантом, параметры, ожидаемое violation_type)
PLAN = [
    ("study_01", "CR000000.dcm", "spine", dict(tilt_deg=0.6), ""),
    ("study_01", "CR000001.dcm", "femur", dict(side="right"), ""),
    ("study_01", "CR000002.dcm", "femur", dict(side="left"), ""),
    ("study_02", "CR000000.dcm", "spine", dict(tilt_deg=8.5), "SPINE_AXIS_TILT"),
    ("study_02", "CR000001.dcm", "femur", dict(side="right"), ""),
    ("study_02", "CR000002.dcm", "femur", dict(side="left"), ""),
    ("study_03", "CR000000.dcm", "spine", dict(tilt_deg=1.0, metal=True), "SPINE_ARTIFACT"),
    ("study_03", "CR000001.dcm", "femur", dict(side="right"), ""),
    ("study_03", "CR000002.dcm", "femur", dict(side="left", scan_rows=205), "FEMUR_ROI_COVERAGE"),
    ("study_04", "CR000000.dcm", "spine", dict(tilt_deg=-1.0), ""),
    ("study_04", "CR000001.dcm", "femur", dict(side="right", bump=0.38), "FEMUR_POSITIONING_ROTATION"),
    ("study_04", "CR000002.dcm", "femur", dict(side="left"), ""),
    ("study_05", "CR000000.dcm", "spine", dict(tilt_deg=1.5), ""),
    ("study_05", "CR000001.dcm", "femur", dict(side="right"), ""),
    ("study_05", "CR000002.dcm", "femur", dict(side="left"), ""),
    ("study_06", "CR000000.dcm", "spine", dict(tilt_deg=-7.0), "SPINE_AXIS_TILT"),
    ("study_06", "CR000001.dcm", "femur", dict(side="right", bump=0.38, scan_rows=210),
     "FEMUR_POSITIONING_ROTATION;FEMUR_ROI_COVERAGE"),
    ("study_06", "CR000002.dcm", "femur", dict(side="left"), ""),
]
# Сиды, на которых всё совпало при сборке картинок README (numpy 2.5, OpenCV 5.0).
SEEDS = [100, 141, 174, 211, 250, 285, 322, 361, 400, 433, 471, 508, 544, 586, 623, 655, 693, 733]


def build(demo: Path, seeds: list[int]) -> None:
    shutil.rmtree(demo, ignore_errors=True)
    for i, (study, name, kind, kw, _) in enumerate(PLAN):
        rng = np.random.default_rng(seeds[i])
        img = ph.spine(rng, **kw) if kind == "spine" else ph.femur(rng, **kw)
        uid = f"2.25.{31337000 + int(study[-2:])}"
        sd.build_dicom(demo / study / name, img, pixel_spacing=(ph.PX, ph.PX), study_uid=uid,
                       sop_uid=f"{uid}.{i + 1}", instance_number=i % 3 + 1)


def run(demo: Path, out: Path, overlays: bool = False) -> dict[str, str]:
    cmd = [sys.executable, "-m", "dxa_qc.batch", "--input", str(demo),
           "--output", str(out / "result.csv"), "--quiet"]
    if overlays:
        cmd += ["--diagnostics", str(out / "diagnostics.csv"), "--overlays", str(out / "overlays.zip")]
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    subprocess.run(cmd, env=env, check=True)
    with open(out / "result.csv", encoding="utf-8", newline="") as fh:
        return {r["path_to_study"]: r["violation_type"] for r in csv.DictReader(fh)}


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    demo, seeds = out / "demo", list(SEEDS)
    for attempt in range(40):
        build(demo, seeds)
        got = run(demo, out)
        bad = [i for i, (st, fn, _, _, want) in enumerate(PLAN) if got.get(f"{st}/{fn}") != want]
        if not bad:
            break
        print(f"попытка {attempt}: не совпало {len(bad)} снимков, сдвигаю их сиды")
        for i in bad:
            seeds[i] += 1
    else:
        sys.exit("не удалось подобрать сиды за 40 попыток")
    run(demo, out, overlays=True)
    shutil.rmtree(out / "ovx", ignore_errors=True)
    with zipfile.ZipFile(out / "overlays.zip") as zf:
        zf.extractall(out / "ovx")
    with zipfile.ZipFile(out / "demo.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(demo.rglob("*.dcm")):
            zf.write(p, p.relative_to(demo).as_posix())
    print("сиды:", seeds)
    print("готово:", out)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else HERE / "build"))
