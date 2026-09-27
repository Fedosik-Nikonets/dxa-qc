#!/usr/bin/env python3
"""Мультилейбл-голова 22 сентября: обучаемый слой поверх геометрии.

Вопрос дня (plan_v2, 22–23 сентября): даёт ли обучаемая голова на признаках
геометрии и изображения что-то сверх пороговых правил. Сравнение — с
**честными** цифрами контрольной точки (агрегат `quality_class` по всем
изображениям 0.50), а не с порогами, поставленными на тех же данных.

Устройство головы. На каждую метку ТЗ — своя L2-логистическая регрессия с
весами классов `balanced` (для логистической потери это то же самое, что
oversampling до баланса, поэтому отдельного oversampling нет). Входы — только
признаки своей области:

* `geom` — геометрия и признаки правил, посчитанные конвейером без обучения
  (позвоночник: покрытие, угол оси, остатки аппроксимации, top-hat/ridge
  артефактов; бедро: длина и ширина поля, отступы, угол диафиза, заполнение,
  профиль ширины, автоматический профиль медиального контура);
* `img` — эмбеддинг изображения: кадр в канонической ориентации 24×24 плюс
  гистограмма яркости, PCA до 8 компонент, **подогнанный внутри обучающих
  фолдов**. В окружении нет torch и весов CNN, поэтому это честная замена
  эмбеддинга сети, а не он сам;
* `geom+img` — конкатенация, как в плане. Это основной, заранее объявленный
  вариант; `geom` и `img` — абляции.

Вложенная проверка по `reports/split.csv` (единица — исследование): на
внешнем фолде C и порог выбираются по F1 на out-of-fold оценках внутренних
четырёх фолдов, модель переобучается на всех четырёх и применяется к
пятому. Внешний фолд ни в какой выбор не попадает.

`quality_class` собирается двумя способами: «OR» голов по меткам (как в
конвейере) и отдельная голова прямо на `quality_class` области со своим
порогом по F1. Рядом — честные правила, правила плюс вложенный предиктор
ротации напарника (`reports/rotation_human_profile/nested_oof.csv`) и
тривиальная база «всё — нарушение», без которой F1 при распространённости
~30% не с чем сравнивать.

Запуск (изображения читаются, ~1 мин):
    python scripts/16_multilabel_head.py --out reports/multilabel_head
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dxa_qc.dicom_io import read_dicom  # noqa: E402

_spec = importlib.util.spec_from_file_location("ckpt", ROOT / "scripts/15_checkpoint_metrics.py")
ckpt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ckpt)

FOLDS = tuple(range(5))
C_GRID = (0.03, 0.1, 0.3, 1.0)
N_PCA = 8
IMG_SIDE = 24
N_HIST = 16
N_BOOT = 10000

SPINE_GEOM = ["coverage", "abs_angle", "res_rms", "res_amp", "slope_spread", "tophat", "ridge"]
FEMUR_GEOM = ["field_h_mm", "field_w_mm", "lateral_mm", "medial_mm", "shaft_deg", "femur_fill",
              "shaft_w", "base_frac", "wmax_mm", "aspect", "lt12", "lt20", "lt30",
              "bump_mm", "anchor_intensity", "profile_intensity", "profile_contrast", "fit_rms_mm"]
GEOM = {"spine": SPINE_GEOM, "femur": FEMUR_GEOM}
LABELS = {"spine": ("pos", "tilt", "art"), "femur": ("roi", "rot")}
VARIANTS = ("geom+img", "geom", "img")
PRIMARY = "geom+img"


# ---------------------------------------------------------------- признаки

def image_descriptor(pixels: np.ndarray, region: str) -> np.ndarray:
    """Кадр в канонической ориентации 24×24 и гистограмма яркости.

    Правое бедро отражается, чтобы медиальный край был с одной стороны, как в
    `rotation_profile.canonical_image`. Яркость нормируется по 1–99
    перцентилям: окна аппарата различаются, а форма и текстура — нет.
    """
    a = np.asarray(pixels, dtype=np.float32)
    if region == "femur_right":
        a = a[:, ::-1]
    lo, hi = np.percentile(a, [1, 99])
    a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    small = cv2.resize(a, (IMG_SIDE, IMG_SIDE), interpolation=cv2.INTER_AREA)
    hist, _ = np.histogram(a, bins=N_HIST, range=(0, 1))
    return np.concatenate([small.ravel(), hist / a.size]).astype(np.float64)


def load(data_root: Path) -> tuple[pd.DataFrame, np.ndarray]:
    df = ckpt.load(ROOT / "reports")
    df, _ = ckpt.nested_predictions(df)

    sp = pd.read_csv(ROOT / "reports/spine_rules.csv")[["rel_path", "res_rms", "res_amp", "slope_spread"]]
    art = pd.read_csv(ROOT / "reports/artifact_features.csv")[["rel_path", "tophat", "ridge"]]
    fe = pd.read_csv(ROOT / "reports/femur_rules.csv")[
        ["rel_path", "field_w_mm", "lateral_mm", "medial_mm", "shaft_deg", "femur_fill"]]
    rot = pd.read_csv(ROOT / "reports/femur_rotation.csv")[
        ["rel_path", "shaft_w", "base_frac", "wmax_mm", "aspect", "lt12", "lt20", "lt30"]]
    # Только автоматические признаки профиля: человеческие точки на вход не идут.
    prof = pd.read_csv(ROOT / "reports/rotation_human_profile/features.csv")[
        ["rel_path", "bump_mm", "anchor_intensity", "profile_intensity", "profile_contrast", "fit_rms_mm"]]
    partner = pd.read_csv(ROOT / "reports/rotation_human_profile/nested_oof.csv")[["rel_path", "pred", "fold"]]
    partner = partner.rename(columns={"pred": "partner_rot", "fold": "partner_fold"})
    for t in (sp, art, fe, rot, prof, partner):
        df = df.merge(t, on="rel_path", how="left", validate="one_to_one")
    df["abs_angle"] = df.angle.abs()

    fem = df.area.eq("femur")
    if df.loc[fem, "partner_rot"].isna().any():
        raise ValueError("не у всех бедренных изображений есть вложенный прогноз ротации напарника")
    if not (df.loc[fem, "partner_fold"] == df.loc[fem, "fold"]).all():
        raise ValueError("фолды nested_oof напарника не совпадают со split.csv")
    # -1 — отказ напарника (профиль не измерен, одно изображение): нарушения не ставим.
    df["partner_rot"] = df.partner_rot.eq(1).astype(int)

    desc = []
    for r in df.itertuples():
        img = read_dicom(data_root / r.rel_path)
        if not img.ok:
            raise ValueError(f"не читается {r.rel_path}: {img.error}")
        desc.append(image_descriptor(img.pixels, r.region))
    return df.reset_index(drop=True), np.vstack(desc)


# ---------------------------------------------------------------- модель

class Head:
    """Импутация медианой → стандартизация геометрии, PCA изображения → логрегрессия.

    Всё, что подгоняется (медианы, масштабы, PCA), подгоняется на обучающей
    части и только применяется к тестовой.
    """

    def __init__(self, variant: str, C: float):
        self.variant, self.C = variant, C

    def _design(self, G: np.ndarray, I: np.ndarray, fit: bool) -> np.ndarray:
        parts = []
        if "geom" in self.variant:
            if fit:
                self.med = np.nanmedian(G, axis=0)
                self.sc = StandardScaler().fit(np.where(np.isnan(G), self.med, G))
            parts.append(self.sc.transform(np.where(np.isnan(G), self.med, G)))
        if "img" in self.variant:
            if fit:
                self.isc = StandardScaler().fit(I)
                self.pca = PCA(n_components=N_PCA, svd_solver="full").fit(self.isc.transform(I))
                self.psc = StandardScaler().fit(self.pca.transform(self.isc.transform(I)))
            parts.append(self.psc.transform(self.pca.transform(self.isc.transform(I))))
        return np.hstack(parts)

    def fit(self, G, I, y) -> "Head":
        self.lr = LogisticRegression(C=self.C, class_weight="balanced", max_iter=5000)
        self.lr.fit(self._design(G, I, fit=True), y)
        return self

    def score(self, G, I) -> np.ndarray:
        return self.lr.decision_function(self._design(G, I, fit=False))


def inner_oof(G, I, y, folds, variant, C) -> np.ndarray:
    s = np.full(len(y), np.nan)
    for g in np.unique(folds):
        tr, te = folds != g, folds == g
        if y[tr].min() == y[tr].max():
            raise ValueError("внутренний обучающий набор без одного из классов")
        s[te] = Head(variant, C).fit(G[tr], I[tr], y[tr]).score(G[te], I[te])
    return s


def f1_at(score, y, t) -> float:
    p = score > t
    tp = int((p & (y == 1)).sum())
    fp = int((p & (y == 0)).sum())
    fn = int((~p & (y == 1)).sum())
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def nested_head(G, I, y, folds, variant) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Внешний фолд: C и порог — по внутренним OOF, решение — на отложенном."""
    score = np.full(len(y), np.nan)
    pred = np.zeros(len(y), dtype=int)
    sel = []
    for f in FOLDS:
        tr, te = folds != f, folds == f
        best = None
        for C in C_GRID:  # при равенстве остаётся меньший C — сильнее регуляризация
            s_in = inner_oof(G[tr], I[tr], y[tr], folds[tr], variant, C)
            t = ckpt.fit_threshold(s_in, y[tr])
            f1 = f1_at(s_in, y[tr], t)
            if best is None or f1 > best[0] + 1e-12:
                best = (f1, C, t)
        f1_in, C, t = best
        m = Head(variant, C).fit(G[tr], I[tr], y[tr])
        score[te] = m.score(G[te], I[te])
        pred[te] = (score[te] > t).astype(int)
        sel.append({"fold": f, "C": C, "threshold": round(float(t), 4), "inner_f1": round(f1_in, 4)})
    return score, pred, sel


# ---------------------------------------------------------------- сравнение

def paired_delta(y, p_new, p_old, groups, n_boot=N_BOOT, seed=0) -> tuple[float, float, float]:
    """Разность F1 (новое − старое) и ДИ 95% бутстрепом по исследованиям.

    Парный: в каждом повторе обе системы оцениваются на одной и той же
    выборке исследований, поэтому общий шум выборки вычитается.
    """
    y, p_new, p_old, groups = map(np.asarray, (y, p_new, p_old, groups))
    uniq = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in uniq}

    def f1(yy, pp):
        tp = int(((pp == 1) & (yy == 1)).sum())
        fp = int(((pp == 1) & (yy == 0)).sum())
        fn = int(((pp == 0) & (yy == 1)).sum())
        return 2 * tp / (2 * tp + fp + fn) if tp else 0.0

    point = f1(y, p_new) - f1(y, p_old)
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for b in range(n_boot):
        take = np.concatenate([idx[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])
        d[b] = f1(y[take], p_new[take]) - f1(y[take], p_old[take])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, default=ROOT / "Data/НД_для_обучения/Исследования")
    ap.add_argument("--out", type=Path, default=ROOT / "reports/multilabel_head")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df, D = load(args.data)
    print(f"изображений: {len(df)}, исследований: {df.study.nunique()}, дескриптор: {D.shape[1]}")

    selections = []
    for area, labels in LABELS.items():
        m = df.area.eq(area).to_numpy()
        G = df.loc[m, GEOM[area]].to_numpy(float)
        I = D[m]
        folds = df.loc[m, "fold"].to_numpy(int)
        targets = {k: df.loc[m, f"y_{k}"].to_numpy(int) for k in labels}
        targets["quality"] = df.loc[m, "y_quality"].to_numpy(int)
        for v in VARIANTS:
            for k, y in targets.items():
                s, p, sel = nested_head(G, I, y, folds, v)
                col = f"{k}_{area}" if k == "quality" else k
                df.loc[m, f"hs_{v}_{col}"] = s
                df.loc[m, f"h_{v}_{col}"] = p
                selections += [{"variant": v, "target": col, **r} for r in sel]
                print(f"  {area:5s} {v:8s} {col:14s} OOF F1={ckpt.binary_metrics(y, p).f1:.2f}  C по фолдам "
                      + ",".join(str(r['C']) for r in sel))

    # Предсказания меток чужой области — нули, как в конвейере.
    for v in VARIANTS:
        for k in ("pos", "tilt", "art", "roi", "rot"):
            c = f"h_{v}_{k}"
            df[c] = df[c].fillna(0).astype(int)
        df[f"hq_{v}"] = np.where(df.area.eq("spine"), df[f"h_{v}_quality_spine"], df[f"h_{v}_quality_femur"])
        df[f"hq_{v}"] = df[f"hq_{v}"].fillna(0).astype(int)

    # Системы quality_class для сравнения.
    sys_q = {
        "Правила, честные пороги (контрольная точка)": ckpt.aggregate(df, "n"),
        "Правила + ротация напарника (вложенная)": (ckpt.aggregate(df, "n") | df.partner_rot).astype(int),
        "Всё — нарушение (тривиальная база)": pd.Series(1, index=df.index),
    }
    for v in VARIANTS:
        sys_q[f"Голова {v}: OR по меткам"] = df[[f"h_{v}_{k}" for k in ("pos", "tilt", "art", "roi", "rot")]].max(axis=1)
        sys_q[f"Голова {v}: прямо на quality_class"] = df[f"hq_{v}"]

    rows = []
    for name, q in sys_q.items():
        q = q.astype(int)
        for area, label in (("spine", "позвоночник"), ("femur", "бедро"), (None, "все изображения")):
            mm = df.area.eq(area) if area else pd.Series(True, index=df.index)
            r = ckpt.row(label, df.loc[mm, "y_quality"], q[mm], groups=df.loc[mm, "study"])
            rows.append({"система": name, **r})
        st = df.assign(q=q).groupby("study").agg(y=("y_quality", "max"), p=("q", "max")).reset_index()
        rows.append({"система": name, **ckpt.row("по исследованиям", st.y, st.p, groups=st.study)})
    quality = pd.DataFrame(rows)

    # Метки: правило (честный порог) против головы по вариантам.
    rows = []
    for k, (name, area, _) in ckpt.RULES.items():
        lab = df[f"y_{k}"].notna()
        y = df.loc[lab, f"y_{k}"].astype(int)
        g = df.loc[lab, "study"]
        base = df.loc[lab, "partner_rot"] if k == "rot" else df.loc[lab, f"n_{k}"]
        base_name = "ротация напарника (вложенная)" if k == "rot" else "правило, честный порог"
        rows.append({"метка": name, "система": base_name, **ckpt.row(name, y, base, groups=g)})
        for v in VARIANTS:
            rows.append({"метка": name, "система": f"голова {v}",
                         **ckpt.row(name, y, df.loc[lab, f"h_{v}_{k}"], df.loc[lab, f"hs_{v}_{k}"], g)})
        rows.append({"метка": name, "система": "всё — нарушение",
                     **ckpt.row(name, y, np.ones(len(y), int), groups=g, ci=False)})
    labels = pd.DataFrame(rows)

    # Парные разности с честными правилами на агрегате «все изображения».
    base = sys_q["Правила, честные пороги (контрольная точка)"].astype(int)
    deltas = []
    for name, q in sys_q.items():
        if name.startswith("Правила, честные"):
            continue
        d, lo, hi = paired_delta(df.y_quality, q.astype(int), base, df.study)
        deltas.append({"система": name, "ΔF1": d, "lo": lo, "hi": hi})
    deltas = pd.DataFrame(deltas)

    print("\nQUALITY_CLASS")
    for _, r in quality.iterrows():
        print(f"  {r['система']:46s} {r['срез']:16s} TP={int(r.tp):2d} FP={int(r.fp):2d} FN={int(r.fn):2d} "
              f"F1={r.f1:.2f} [{r.get('f1_lo', np.nan):.2f}; {r.get('f1_hi', np.nan):.2f}]")
    print("\nМЕТКИ")
    for _, r in labels.iterrows():
        auc = f" AUC={r.roc_auc:.3f}" if r.roc_auc == r.roc_auc else ""
        print(f"  {r['метка']:26s} {r['система']:32s} TP={int(r.tp):2d} FP={int(r.fp):2d} FN={int(r.fn):2d} "
              f"F1={r.f1:.2f}{auc}")
    print("\nΔF1 К ЧЕСТНЫМ ПРАВИЛАМ, все изображения (парный бутстреп по исследованиям)")
    for _, r in deltas.iterrows():
        print(f"  {r['система']:46s} {r['ΔF1']:+.3f} [{r.lo:+.3f}; {r.hi:+.3f}]")

    keep = ["study", "rel_path", "region", "fold", "y_quality"] + [f"y_{k}" for k in ckpt.RULES] \
        + [c for c in df.columns if c.startswith(("h_", "hs_", "hq_"))] + ["partner_rot"]
    df[keep].to_csv(args.out / "oof.csv", index=False)
    quality.to_csv(args.out / "quality.csv", index=False)
    labels.to_csv(args.out / "labels.csv", index=False)
    deltas.to_csv(args.out / "deltas.csv", index=False)
    pd.DataFrame(selections).to_csv(args.out / "selections.csv", index=False)
    json.dump({"n_images": int(len(df)), "n_studies": int(df.study.nunique()), "primary": PRIMARY,
               "c_grid": C_GRID, "n_pca": N_PCA, "img_side": IMG_SIDE, "n_hist": N_HIST,
               "n_bootstrap": N_BOOT, "geom_features": GEOM},
              open(args.out / "summary.json", "w"), ensure_ascii=False, indent=2)
    print(f"\nзаписано: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
