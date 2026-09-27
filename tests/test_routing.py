"""Роутер анатомической области: правило и его края.

Пороги правила (LAT_SPLIT, SYM_SPINE) подобраны по обучающему набору, там же
проверено, что расхождений с просмотренным глазами эталоном нет
(`scripts/05_router_check.py`). Здесь проверяется логика: что правило
двухветвевое, что сторона берётся из знака, что неуверенность помечается,
и что вырожденный вход не роняет конвейер.
"""
from __future__ import annotations

import numpy as np

from dxa_qc.routing import (
    FEMUR_LEFT,
    FEMUR_RIGHT,
    LAT_SPLIT,
    SPINE,
    SYM_SPINE,
    UNKNOWN,
    RegionFeatures,
    image_features,
    is_spine_like,
    region_name,
    route_study,
)


def spine_phantom(rows=180, cols=160):
    """Светлая вертикальная полоса по центру: верх и низ совпадают -> lat ~ 0."""
    a = np.full((rows, cols), 0.08)
    a[:, cols // 2 - 12: cols // 2 + 12] = 0.9
    return a


def femur_phantom(rows=180, cols=160, side=+1, offset=60):
    """Таз смещён в сторону (верх кадра), диафиз идёт к центру (низ).

    `offset` подобран так, чтобы |lat_score| попал в диапазон реальных бёдер
    обучающего набора (45-105), а не в серую зону у порога.
    """
    a = np.full((rows, cols), 0.08)
    top_c = cols // 2 + side * offset
    a[: rows // 3, top_c - 12: top_c + 12] = 0.9
    a[2 * rows // 3:, cols // 2 - 8: cols // 2 + 8] = 0.9
    return a


def test_features_spine_centered():
    f = image_features(spine_phantom())
    assert abs(f.lat_score) < 1.0
    assert f.sym > 0.9


def test_features_femur_lateral_sign():
    right = image_features(femur_phantom(side=+1))
    left = image_features(femur_phantom(side=-1))
    assert right.lat_score > LAT_SPLIT
    assert left.lat_score < -LAT_SPLIT


def test_features_degenerate_input_is_safe():
    for bad in (None, np.zeros((0, 0)), np.zeros((1, 1)), np.full((5, 5), np.nan)):
        f = image_features(bad)
        assert isinstance(f, RegionFeatures)
        assert not f.valid or np.isfinite(f.lat_score)


def test_spine_rule_has_two_branches():
    # центрирован, но несимметричен — всё равно позвоночник
    assert is_spine_like(RegionFeatures(sym=0.02, lat_score=LAT_SPLIT - 5))
    # нецентрирован, но симметричен — тоже позвоночник (косая укладка, сколиоз)
    assert is_spine_like(RegionFeatures(sym=SYM_SPINE + 0.1, lat_score=LAT_SPLIT + 15))
    # ни то, ни другое — бедро
    assert not is_spine_like(RegionFeatures(sym=SYM_SPINE - 0.1, lat_score=LAT_SPLIT + 15))


def test_route_full_study():
    feats = [
        image_features(spine_phantom()),
        image_features(femur_phantom(side=+1)),
        image_features(femur_phantom(side=-1)),
    ]
    got = [d.region for d in route_study(feats)]
    assert got == [SPINE, FEMUR_RIGHT, FEMUR_LEFT]


def test_route_two_femurs_without_spine():
    feats = [image_features(femur_phantom(side=+1)), image_features(femur_phantom(side=-1))]
    got = [d.region for d in route_study(feats)]
    assert got == [FEMUR_RIGHT, FEMUR_LEFT]


def test_route_single_spine():
    d = route_study([image_features(spine_phantom())])[0]
    assert d.region == SPINE and d.confident


def test_route_marks_duplicate_region_unconfident():
    # два одинаковых правых бедра — физически невозможно, роутер это помечает
    feats = [image_features(femur_phantom(side=+1)), image_features(femur_phantom(side=+1))]
    out = route_study(feats)
    assert all(d.region == FEMUR_RIGHT for d in out)
    assert all(not d.confident for d in out)
    assert all("более одного раза" in (d.reason or "") for d in out)


def test_route_invalid_features_stay_unknown():
    out = route_study([RegionFeatures(), image_features(spine_phantom())])
    assert out[0].region == UNKNOWN and out[0].reason
    assert out[1].region == SPINE


def test_route_empty_study():
    assert route_study([]) == []


def test_region_names():
    assert region_name(SPINE) == "LUMBAR_SPINE"
    assert region_name(SPINE, "ru").startswith("Поясничный")
    assert region_name(UNKNOWN) == "UNKNOWN"
