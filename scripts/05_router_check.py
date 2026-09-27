#!/usr/bin/env python3
"""Проверка роутера области на обучающем наборе.

Роутер `dxa_qc.routing` работает без разметки — это его назначение на
инференсе. Здесь он сверяется с областями из `reports/dataset_images.csv`,
которые были назначены с подсказкой из разметки и просмотрены глазами на всех
252 изображениях (день 2). Любое расхождение — ошибка роутера, а не эталона.

Запуск:
    python scripts/05_router_check.py --dataset reports/dataset_images.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dxa_qc.routing import RegionFeatures, route_study  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=Path)
    args = ap.parse_args()

    ds = pd.read_csv(args.dataset)
    need = {"study", "rel_path", "region", "sym", "lat_score"}
    missing = need - set(ds.columns)
    if missing:
        print(f"в таблице нет колонок: {sorted(missing)}", file=sys.stderr)
        return 2

    rows = []
    for study, g in ds.groupby("study", sort=True):
        g = g.sort_values("rel_path")
        feats = [RegionFeatures(sym=r.sym, lat_score=r.lat_score) for r in g.itertuples()]
        for r, d in zip(g.itertuples(), route_study(feats)):
            rows.append({
                "study": study, "rel_path": r.rel_path, "expected": r.region,
                "got": d.region, "confident": d.confident, "reason": d.reason or "",
                "abs_lat": abs(r.lat_score), "sym": r.sym,
            })

    res = pd.DataFrame(rows)
    bad = res[res["expected"] != res["got"]]
    unsure = res[res["confident"] == False]  # noqa: E712

    print(f"изображений: {len(res)}  исследований: {res['study'].nunique()}")
    print(f"расхождений с эталоном дня 2: {len(bad)}")
    print(f"неуверенных решений: {len(unsure)}")
    if len(bad):
        print("\nрасхождения:")
        print(bad[["rel_path", "expected", "got", "abs_lat", "sym"]].to_string(index=False))
    if len(unsure):
        print("\nнеуверенные (область может быть верной, но правило сработало впритык):")
        print(unsure[["rel_path", "got", "abs_lat", "sym", "reason"]].to_string(index=False))
    return 1 if len(bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
