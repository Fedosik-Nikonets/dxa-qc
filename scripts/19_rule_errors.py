#!/usr/bin/env python3
"""Разбор ошибок геометрических правил и ротации, 24 сентября.

Продолжение `reports/day11_errors_artifact.md` на остальные четыре типа
нарушений: укладка и наклон оси позвоночника, охват ROI и ротация бедра.
Для каждого правила берутся решения конвейера (пороги модулей, как в
сквозном прогоне) и честные (порог из четырёх фолдов, `15_checkpoint_metrics`),
и по каждой ошибке выписывается то, что может её объяснить: признак правила,
комментарий эксперта, остальные метки кадра, источник масштаба.

Выход (`reports/errors/`):
    rule_errors.csv       — одна строка на ошибку (FP/FN конвейера или честного порога)
    <rule>_fp_fn.png      — листы overlay: те же картинки, что отдаёт сервис,
                            с подписью «исход, метка, оценка, комментарий»

Изображения читаются встроенным ридером; решения на них пересчитываются
`batch.predict` и сверяются с `batch_diagnostics_train.csv` — расхождение
остановит скрипт. Время — около минуты.

Запуск:
    python scripts/19_rule_errors.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dxa_qc import batch, overlay  # noqa: E402
from dxa_qc.dicom_io import read_dicom  # noqa: E402

REPORTS = ROOT / "reports"
OUT = REPORTS / "errors"
DATA = ROOT / "Data" / "НД_для_обучения" / "Исследования"
TILE_W = 300
COLS = 5

_spec = importlib.util.spec_from_file_location("cm", ROOT / "scripts" / "15_checkpoint_metrics.py")
cm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cm)

# ключ правила -> (score для подписи, формат, колонка с решением по честному порогу)
RULES = {
    "tilt": ("angle", "tilt {:+.1f} deg"),
    "pos": ("coverage", "coverage {:.3f}"),
    "roi": ("field_h_mm", "field {:.0f} mm"),
    "rot": ("rotation_value", "bump {:.2f} mm"),
}


def outcome(y, p) -> np.ndarray:
    y, p = np.asarray(y), np.asarray(p)
    return np.select([(y == 1) & (p == 1), (y == 0) & (p == 1), (y == 1) & (p == 0)],
                     ["TP", "FP", "FN"], "TN")


def load() -> pd.DataFrame:
    df, _ = cm.nested_predictions(cm.load(REPORTS))
    ax = pd.read_csv(REPORTS / "spine_axis.csv")[["rel_path", "ang_top", "ang_bot", "res_rms", "slope_spread"]]
    dg = (pd.read_csv(REPORTS / "batch_diagnostics_train.csv").drop_duplicates("md5")
          [["md5", "violation_type", "rotation_value", "rotation_low", "rotation_high",
            "femur_shaft_deg", "femur_mm_per_px_source"]])
    df = df.merge(ax, on="rel_path", how="left").merge(dg, on="md5", how="left", validate="one_to_one")
    return df


def error_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, (score, _) in RULES.items():
        lab = df[f"y_{k}"].notna()
        d = df[lab].assign(rule=k, pipeline=outcome(df.loc[lab, f"y_{k}"], df.loc[lab, f"p_{k}"]),
                           nested=outcome(df.loc[lab, f"y_{k}"], df.loc[lab, f"n_{k}"]),
                           value=df.loc[lab, score])
        rows.append(d[d.pipeline.isin(["FP", "FN"]) | d.nested.isin(["FP", "FN"])])
    t = pd.concat(rows, ignore_index=True)
    t["out_of_scope"] = t.comment.fillna("").str.contains("сколиоз|перелом", case=False).astype(int)
    keep = ["rule", "pipeline", "nested", "value", "region", "study", "rel_path", "comment", "out_of_scope",
            "y_pos", "y_tilt", "y_art", "y_roi", "y_rot", "ang_top", "ang_bot", "res_rms", "slope_spread",
            "rotation_low", "rotation_high", "femur_shaft_deg", "femur_mm_per_px_source"]
    return t[keep].sort_values(["rule", "pipeline", "value"]).reset_index(drop=True)


def tile(r, diag_row) -> np.ndarray:
    img = read_dicom(DATA / r.rel_path)
    if not img.ok:
        raise RuntimeError(f"не прочитан {r.rel_path}: {img.error}")
    _, vt, diag = batch.predict(r.region, img)
    want = "" if not isinstance(diag_row, str) else diag_row
    if set(vt.split(";")) - {""} != set(want.split(";")) - {""}:
        raise RuntimeError(f"решения разошлись с диагностикой прогона: {r.rel_path}: {vt!r} vs {want!r}")
    bgr = overlay.render(r.region, img.pixels, vt.split(";"), {k: str(v) for k, v in diag.items()},
                         mm_per_px=img.mm_per_px, mm_per_px_source=img.mm_per_px_source,
                         scanner_model=img.model)
    h = int(bgr.shape[0] * TILE_W / bgr.shape[1])
    bgr = cv2.resize(bgr, (TILE_W, h), interpolation=cv2.INTER_AREA)
    fmt = RULES[r.rule][1]
    lines = [f"{r.pipeline}/{r.nested} (pipeline/nested)  {fmt.format(r.value)}",
             f"{r.study[-14:]}"]
    if isinstance(r.comment, str):
        lines.append(translit(r.comment)[:40])
    cap = np.full((16 * len(lines) + 6, TILE_W, 3), 255, np.uint8)
    color = (0, 0, 200) if "FP" in (r.pipeline, r.nested) else (200, 90, 0)
    for i, s in enumerate(lines):
        cv2.putText(cap, s, (4, 15 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color if i == 0 else (40, 40, 40),
                    1, cv2.LINE_AA)
    return np.vstack([cap, bgr])


_TR = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
               "a b v g d e e zh z i y k l m n o p r s t u f h ts ch sh sch ' y ' e yu ya".split()))


def translit(s: str) -> str:
    """OpenCV не рисует кириллицу — комментарий эксперта пишется латиницей."""
    return "".join(_TR.get(c.lower(), c) for c in s)


def sheet(tiles: list[np.ndarray]) -> np.ndarray:
    h = max(t.shape[0] for t in tiles)
    tiles = [np.vstack([t, np.full((h - t.shape[0], TILE_W, 3), 255, np.uint8)]) for t in tiles]
    while len(tiles) % COLS:
        tiles.append(np.full((h, TILE_W, 3), 255, np.uint8))
    rows = [np.hstack([np.pad(t, ((4, 4), (4, 4), (0, 0)), constant_values=255) for t in tiles[i:i + COLS]])
            for i in range(0, len(tiles), COLS)]
    return np.vstack(rows)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    df = load()
    t = error_table(df)
    t.drop(columns=[]).to_csv(OUT / "rule_errors.csv", index=False)
    diag = df.set_index("rel_path").violation_type
    for k in RULES:
        sub = t[t.rule.eq(k)]
        print(f"{k}: конвейер FP {sub.pipeline.eq('FP').sum()}, FN {sub.pipeline.eq('FN').sum()}; "
              f"честно FP {sub.nested.eq('FP').sum()}, FN {sub.nested.eq('FN').sum()}")
        tiles = [tile(r, diag.get(r.rel_path)) for r in sub.itertuples()]
        cv2.imwrite(str(OUT / f"{k}_fp_fn.png"), sheet(tiles))
    print(f"-> {OUT / 'rule_errors.csv'} ({len(t)} строк)")


if __name__ == "__main__":
    main()
