"""Стресс-корпус (scripts/18_stress_test.py): ни одного вылета, перепаковки не меняют решений."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_stress_corpus_passes(tmp_path):
    if not (ROOT / "Data/НД_для_обучения/Исследования").exists():
        pytest.skip("нет обучающего набора: реальные кадры для перепаковки не из чего взять")
    spec = importlib.util.spec_from_file_location("stress", ROOT / "scripts/18_stress_test.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    res = m.check(tmp_path, m.make_corpus(tmp_path))
    assert res["problems"] == []
    assert res["passed"]
