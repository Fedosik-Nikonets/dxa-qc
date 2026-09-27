"""Integrity checks for the auditable femur landmark annotation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_femur_manifest_has_balanced_fixed_sample():
    manifest = pd.read_csv(ROOT / "annotations/femur_manifest_v1.csv")
    split = pd.read_csv(ROOT / "reports/split.csv")

    assert len(manifest) == 80
    assert manifest.image_id.tolist() == [f"F{i:03d}" for i in range(1, 81)]
    assert manifest.region.value_counts().to_dict() == {"femur_left": 40, "femur_right": 40}
    assert manifest.groupby(["fold", "region"]).size().eq(8).all()
    assert manifest.study.isin(split.study).all()
    assert manifest.rel_path.is_unique
    assert manifest.y_FEMUR_POSITIONING_ROTATION.sum() == 36
    assert manifest.y_FEMUR_ROI_COVERAGE.sum() == 7


def test_femur_landmarks_are_validated():
    """Проверки формата и границ.

    До 20 сентября здесь стояло `not human_reviewed.any()`: в репозитории
    лежал черновик автоматики, и тест охранял его от попадания в обучение.
    Черновик заменён разметкой человека, поэтому охраняется теперь не
    происхождение само по себе, а согласованность трёх флагов: пометка
    «пригодно для обучения» не может появиться отдельно от пометки
    «просмотрено человеком» и от источника `human`.
    """
    annotations = pd.read_csv(ROOT / "annotations/femur_landmarks_v1.csv")
    manifest = pd.read_csv(ROOT / "annotations/femur_manifest_v1.csv").set_index("image_id")

    assert len(annotations) == 320
    assert annotations.groupby("image_id").size().eq(4).all()
    assert set(annotations.landmark) == {
        "greater_trochanter", "lesser_trochanter", "femoral_neck", "ischium"
    }
    by_human = annotations.annotation_source == "human"
    assert (annotations.human_reviewed == by_human).all()
    assert (annotations.eligible_for_training == by_human).all()
    assert set(annotations.visibility) <= {"visible", "uncertain", "not_visible"}

    visible = annotations[annotations.visibility != "not_visible"]
    assert visible.x.notna().all() and visible.y.notna().all()
    assert (visible.x >= 0).all() and (visible.y >= 0).all()
    for row in visible.itertuples(index=False):
        image = manifest.loc[row.image_id]
        assert row.x < image.cols
        assert row.y < image.rows

    hidden = annotations[annotations.visibility == "not_visible"]
    assert hidden.x.isna().all() and hidden.y.isna().all()


def test_committed_femur_landmarks_are_the_human_pass():
    """То, что лежит в репозитории сейчас, размечено человеком.

    Пересборка скрипта 11 без `--source human` молча вернула бы файл в
    состояние черновика, и разметка выпала бы из обучения, ничего не
    сломав. Пусть ломает здесь.
    """
    annotations = pd.read_csv(ROOT / "annotations/femur_landmarks_v1.csv")
    assert (annotations.annotation_source == "human").all()
    assert annotations.image_id.nunique() == 80


def test_femur_landmarks_are_anatomically_coherent():
    """Стороны тела не перепутаны.

    Главный риск ручной разметки — что человек не заметил зеркальности и
    ставил точки по одной схеме на левых и правых бёдрах. Тогда большой
    вертел систематически оказался бы медиальнее шейки, а седалищная кость
    латеральнее. Порог 0.9, а не 1.0: отдельные спорные кадры допустимы,
    систематическая путаница — нет.
    """
    a = pd.read_csv(ROOT / "annotations/femur_landmarks_v1.csv")
    w = a.pivot(index="image_id", columns="landmark", values=["x", "y"])
    region = a.drop_duplicates("image_id").set_index("image_id").region
    # Слева латеральная сторона справа на кадре, справа — наоборот.
    side = np.where(region.reindex(w.index) == "femur_left", 1.0, -1.0)

    gt_lateral = side * (w[("x", "greater_trochanter")] - w[("x", "femoral_neck")]) > 0
    isch_medial = side * (w[("x", "ischium")] - w[("x", "femoral_neck")]) < 0
    lt_below = w[("y", "lesser_trochanter")] > w[("y", "femoral_neck")]

    for name, ok, defined in (
        ("большой вертел латеральнее шейки", gt_lateral, w[("x", "greater_trochanter")].notna()),
        ("седалищная кость медиальнее шейки", isch_medial, w[("x", "ischium")].notna()),
        ("малый вертел ниже шейки", lt_below, w[("x", "lesser_trochanter")].notna()),
    ):
        n = int(defined.sum())
        assert n >= 40, f"{name}: слишком мало точек для проверки"
        assert (ok & defined).sum() / n >= 0.9, name
