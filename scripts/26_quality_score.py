#!/usr/bin/env python3
"""Балл нарушения `quality_score`: подгонка крутизны и честная оценка ROC-AUC.

Балл (`src/dxa_qc/score.py`) — noisy-OR вероятностей правил области,
`p = sigmoid(a·m)`, где `m` — маржа правила (>0 — правило сработало), а `a` —
крутизна, одна на правило. Скрипт делает две вещи.

1. **Крутизна для сервиса.** Логистическая регрессия без свободного члена по
   метке своего типа, на всех размеченных изображениях, с маржами
   конвейера: пороги из модулей, артефакты — out-of-fold вероятности с
   порогом модели, ротация — значение признака и границы из
   `models/rotation_profile_v1.json`. Результат — `models/quality_score_v1.json`.
2. **Вложенная оценка.** На каждом внешнем фолде маржи считаются с порогами,
   подобранными без него (как в `scripts/15_checkpoint_metrics.py`),
   ротация — из вложенной проверки `rotation_human_profile/nested_oof.csv`,
   артефакты — out-of-fold; крутизна подгоняется на четырёх фолдах и
   применяется к пятому. ROC-AUC и PR-AUC `quality_class` по снимкам и по
   исследованиям, ДИ 95% — бутстреп по исследованиям.

Балл не меняет ни одного решения: больше 0.5 он тогда и только тогда, когда
`quality_class = 1` (проверяется здесь же).

Запуск: python scripts/26_quality_score.py   (секунды, изображения не читаются)
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dxa_qc import artifact, rotation_rule  # noqa: E402
from dxa_qc.femur import FIELD_MIN_MM  # noqa: E402
from dxa_qc.score import PARAMS_VERSION, RULES, combine  # noqa: E402
from dxa_qc.spine import AXIS_TILT_DEG, COVERAGE_MIN  # noqa: E402

_spec = importlib.util.spec_from_file_location("m15", ROOT / "scripts/15_checkpoint_metrics.py")
m15 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m15)

KEYS = ("pos", "tilt", "art", "roi", "rot")
SLOPE_GRID = np.exp(np.linspace(np.log(0.01), np.log(100.0), 801))
N_BOOT = 2000


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_slope(m: np.ndarray, y: np.ndarray) -> float:
    """Крутизна a по максимуму правдоподобия sigmoid(a·m), перебором по сетке."""
    ok = np.isfinite(m)
    m, y = m[ok], y[ok].astype(float)
    z = np.clip(np.outer(SLOPE_GRID, m), -60, 60)
    p = 1 / (1 + np.exp(-z))
    ll = (y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)).sum(axis=1)
    return float(SLOPE_GRID[int(np.argmax(ll))])


def auc(s, y) -> float:
    s = np.asarray(s, float)
    y = np.asarray(y, int)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum())
                 / (len(pos) * len(neg)))


def pr_auc(s, y) -> float:
    s = np.asarray(s, float)
    y = np.asarray(y, int)
    o = np.argsort(-s, kind="stable")
    yy = y[o]
    prec = np.cumsum(yy) / np.arange(1, len(yy) + 1)
    return float((prec * yy).sum() / max(yy.sum(), 1))


def image_scores(M: dict, slopes: dict, area: np.ndarray, q: np.ndarray) -> np.ndarray:
    out = np.zeros(len(q))
    for i in range(len(q)):
        probs = []
        for k in RULES[area[i]]:
            v = M[k][i]
            if np.isfinite(v):
                probs.append(1 / (1 + np.exp(-np.clip(slopes[k] * v, -60, 60))))
        out[i] = combine(probs, int(q[i]))
    return out


def study_level(df: pd.DataFrame, s: np.ndarray, q: np.ndarray) -> pd.DataFrame:
    d = df.assign(s=s, q=q)
    rows = []
    for st, g in d.groupby("study", sort=True):
        cls = int(g.q.max())
        rows.append({"study": st, "y": int(g.y_quality.max()), "q": cls,
                     "s": combine(list(g.s), cls)})
    return pd.DataFrame(rows)


def boot_ci(df: pd.DataFrame, s: np.ndarray, y: np.ndarray, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    studies = df.study.unique()
    idx = {st: np.flatnonzero(df.study.to_numpy() == st) for st in studies}
    vals = []
    for _ in range(N_BOOT):
        pick = rng.choice(studies, len(studies))
        ix = np.concatenate([idx[st] for st in pick])
        if y[ix].min() == y[ix].max():
            continue
        vals.append(auc(s[ix], y[ix]))
    return [round(float(v), 3) for v in np.percentile(vals, [2.5, 97.5])]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    reports = ROOT / "reports"
    out_dir = reports / "quality_score"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = m15.load(reports, rotation=True)
    df, th = m15.nested_predictions(df)
    th = pd.DataFrame(th)
    area = np.where(df.area.eq("spine"), "spine", "femur")
    lab = {k: df[f"y_{k}"].notna().to_numpy() for k in KEYS}
    y = {k: df[f"y_{k}"].fillna(0).to_numpy(int) for k in KEYS}

    # --- маржи конвейера
    art_model = artifact.load_default()
    rot_model = rotation_rule.load_default()
    if art_model is None or rot_model is None:
        raise SystemExit("нет весов артефактов или границ ротации в models/")
    dec = pd.read_csv(reports / "rotation_rule/decisions.csv")[["rel_path", "feature", "value"]]
    if set(dec.feature) != {rot_model.feature}:
        raise SystemExit("признак в rotation_rule/decisions.csv не совпадает с моделью")
    val = df[["rel_path"]].merge(dec, on="rel_path", how="left")["value"].to_numpy(float)
    P = {
        "pos": COVERAGE_MIN - df.coverage.to_numpy(float),
        "tilt": np.abs(df.angle.to_numpy(float)) - AXIS_TILT_DEG,
        "art": logit(df.prob_oof.to_numpy(float)) - logit(art_model.threshold),
        "roi": FIELD_MIN_MM - df.field_h_mm.to_numpy(float),
        "rot": np.maximum(rot_model.low - val, val - rot_model.high),
    }
    for k in KEYS:
        P[k] = np.where(lab[k], P[k], np.nan)
    slopes = {k: fit_slope(P[k][lab[k]], y[k][lab[k]]) for k in KEYS}

    # --- вложенные маржи: пороги без своего фолда
    nst = pd.read_csv(reports / "rotation_human_profile/nested_oof.csv")[["rel_path", "risk"]]
    risk = df[["rel_path"]].merge(nst, on="rel_path", how="left")["risk"].to_numpy(float)
    N = {k: np.full(len(df), np.nan) for k in KEYS}
    fold = df.fold.to_numpy()

    def t(rule, f):
        return float(th[(th.rule == rule) & (th.fold == f)].threshold.iloc[0])

    for f in sorted(np.unique(fold)):
        ix = fold == f
        N["pos"][ix] = t("pos", f) - df.coverage.to_numpy(float)[ix]
        N["tilt"][ix] = np.abs(df.angle.to_numpy(float)[ix]) - t("tilt", f)
        N["roi"][ix] = t("roi", f) - df.field_h_mm.to_numpy(float)[ix]
        # Порог артефактов фолда восстанавливается по его же out-of-fold
        # решениям: середина между последней вероятностью «нет» и первой «да».
        p = df.prob_oof.to_numpy(float)[ix]
        d = df.pred_oof.to_numpy()[ix] == 1
        lo = np.nanmax(p[~d]) if (~d).any() else 0.0
        hi = np.nanmin(p[d]) if d.any() else 1.0
        N["art"][ix] = logit(p) - logit((lo + hi) / 2)
    N["rot"] = risk
    for k in KEYS:
        N[k] = np.where(lab[k], N[k], np.nan)
        agree = np.mean((np.nan_to_num(N[k][lab[k]], nan=-1) > 0).astype(int)
                        == df.loc[lab[k], f"n_{k}"].to_numpy())
        if agree < 1.0:
            raise SystemExit(f"{k}: вложенная маржа расходится с вложенным решением ({agree:.3f})")

    q_nested = df[[f"n_{k}" for k in KEYS]].max(axis=1).to_numpy(int)
    s_nested = np.zeros(len(df))
    fold_slopes = {}
    for f in sorted(np.unique(fold)):
        tr, te = fold != f, fold == f
        sl = {k: fit_slope(N[k][tr & lab[k]], y[k][tr & lab[k]]) for k in KEYS}
        fold_slopes[int(f)] = {k: round(v, 4) for k, v in sl.items()}
        s_nested[te] = image_scores({k: N[k][te] for k in KEYS}, sl, area[te], q_nested[te])
    if not np.array_equal((s_nested > 0.5).astype(int), q_nested):
        raise SystemExit("балл и quality_class разошлись")

    yq = df.y_quality.to_numpy(int)
    st = study_level(df, s_nested, q_nested)
    res = {
        "снимки": {"n": int(len(yq)), "позитивов": int(yq.sum()),
                   "roc_auc": round(auc(s_nested, yq), 3), "roc_auc_ci95": boot_ci(df, s_nested, yq),
                   "pr_auc": round(pr_auc(s_nested, yq), 3),
                   "roc_auc_бинарного_класса": round(auc(q_nested, yq), 3)},
        "исследования": {"n": int(len(st)), "позитивов": int(st.y.sum()),
                         "roc_auc": round(auc(st.s, st.y), 3), "pr_auc": round(pr_auc(st.s, st.y), 3),
                         "roc_auc_бинарного_класса": round(auc(st.q, st.y), 3)},
    }
    for a_ in ("spine", "femur"):
        mm = area == a_
        res[f"снимки: {a_}"] = {"roc_auc": round(auc(s_nested[mm], yq[mm]), 3),
                                "roc_auc_бинарного_класса": round(auc(q_nested[mm], yq[mm]), 3)}

    params = {
        "version": PARAMS_VERSION,
        "slopes": {k: round(v, 4) for k, v in slopes.items()},
        "rules": {"pos": "COVERAGE_MIN - spine_coverage", "tilt": "|spine_angle_deg| - AXIS_TILT_DEG",
                  "art": "logit(artifact_prob) - logit(threshold)", "roi": "FIELD_MIN_MM - femur_field_h_mm",
                  "rot": "max(low - value, value - high)"},
        "combine": "1 - prod(1 - sigmoid(slope * margin)); >0.5 iff quality_class = 1",
        "fitted_on": f"{int(len(df))} размеченных изображений обучающего набора, все пять фолдов",
        "nested_estimate": res,
        "inputs_sha256": {p: sha256(ROOT / p) for p in (
            "reports/dataset_images.csv", "reports/spine_rules.csv", "reports/artifact_oof.csv",
            "reports/femur_rules.csv", "reports/rotation_rule/decisions.csv",
            "reports/rotation_human_profile/nested_oof.csv", "reports/split.csv")},
    }
    (ROOT / "models/quality_score_v1.json").write_text(
        json.dumps(params, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps({"вложенная_оценка": res, "крутизна_по_фолдам": fold_slopes,
                    "крутизна_сервиса": params["slopes"]}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    pd.DataFrame({"rel_path": df.rel_path, "study": df.study, "fold": fold, "area": area,
                  "y_quality": yq, "quality_class_nested": q_nested,
                  "quality_score_nested": np.round(s_nested, 4)}).to_csv(out_dir / "nested_oof.csv", index=False)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    print("крутизна сервиса:", params["slopes"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
