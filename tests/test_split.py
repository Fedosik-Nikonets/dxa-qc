"""Тесты разбиения выборки.

Разбиение коммитится в репозиторий и после 16 сентября не меняется. Значит
оно обязано быть воспроизводимым побайтово и не должно смешивать изображения
одного исследования между фолдами — иначе утечка и завышенные метрики
(прямой минус по критерию 8.1).
"""
from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "03_make_split.py"

spec = importlib.util.spec_from_file_location("make_split", SCRIPT)
make_split = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_split)


def _fake_dataset(path: Path, n_studies: int = 60, seed: int = 0) -> Path:
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n_studies):
        study = f"2.25.{i:040d}"
        has_femur = rng.rand() < 0.75
        regions = ["spine"] + (["femur_left", "femur_right"] if has_femur else [])
        for reg in regions:
            rows.append({
                "study": study,
                "rel_path": f"{study}/{reg}.dcm",
                "path": f"/tmp/{study}/{reg}.dcm",
                "region": reg,
                "quality_class": float(rng.rand() < 0.3),
                "labeled": True,
            })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


def _run(dataset: Path, out: Path, seed: int = 42) -> pd.DataFrame:
    res = subprocess.run(
        [sys.executable, str(SCRIPT), "--dataset", str(dataset),
         "--out", str(out), "--seed", str(seed)],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr
    return pd.read_csv(out)


def test_split_is_reproducible(tmp_path):
    ds = _fake_dataset(tmp_path / "ds.csv")
    a = _run(ds, tmp_path / "a.csv")
    b = _run(ds, tmp_path / "b.csv")
    ha = hashlib.md5((tmp_path / "a.csv").read_bytes()).hexdigest()
    hb = hashlib.md5((tmp_path / "b.csv").read_bytes()).hexdigest()
    assert ha == hb, "повторный запуск с тем же seed обязан давать тот же файл"
    assert a.equals(b)


def test_different_seed_changes_split(tmp_path):
    ds = _fake_dataset(tmp_path / "ds.csv")
    a = _run(ds, tmp_path / "a.csv", seed=42)
    c = _run(ds, tmp_path / "c.csv", seed=7)
    assert not a["fold"].equals(c["fold"])


def test_every_study_in_exactly_one_fold(tmp_path):
    ds = _fake_dataset(tmp_path / "ds.csv")
    sp = _run(ds, tmp_path / "s.csv")
    assert sp["study"].is_unique
    assert set(sp["fold"]) <= set(range(make_split.N_FOLDS))
    src = pd.read_csv(ds)
    assert set(sp["study"]) == set(src["study"])


def test_folds_are_balanced(tmp_path):
    ds = _fake_dataset(tmp_path / "ds.csv", n_studies=100)
    sp = _run(ds, tmp_path / "s.csv")
    sizes = sp["fold"].value_counts()
    assert sizes.max() - sizes.min() <= make_split.N_FOLDS


def test_strata_are_spread_across_folds(tmp_path):
    """Стратификация: редкие страты не должны собираться в одном фолде."""
    ds = _fake_dataset(tmp_path / "ds.csv", n_studies=100)
    sp = _run(ds, tmp_path / "s.csv")
    tab = pd.crosstab(sp["stratum"], sp["fold"])
    for stratum, row in tab.iterrows():
        if row.sum() >= make_split.N_FOLDS:
            assert (row > 0).sum() >= make_split.N_FOLDS - 1, stratum


def test_holdout_is_fold_zero(tmp_path):
    ds = _fake_dataset(tmp_path / "ds.csv")
    sp = _run(ds, tmp_path / "s.csv")
    assert set(sp.loc[sp["fold"] == 0, "split_holdout"]) == {"val"}
    assert set(sp.loc[sp["fold"] != 0, "split_holdout"]) == {"train"}


def test_committed_split_matches_committed_dataset():
    """Зафиксированный сплит обязан соответствовать зафиксированному датасету."""
    ds_path = ROOT / "reports" / "dataset_images.csv"
    sp_path = ROOT / "reports" / "split.csv"
    if not (ds_path.exists() and sp_path.exists()):
        return  # в окружении без данных проверять нечего
    ds = pd.read_csv(ds_path)
    sp = pd.read_csv(sp_path)
    assert set(sp["study"]) == set(ds["study"])
    assert sp["study"].is_unique
    assert sp["fold"].between(0, make_split.N_FOLDS - 1).all()
