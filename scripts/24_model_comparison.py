#!/usr/bin/env python3
"""Сравнение вариантов модели — слайд ТЗ 4, 24 сентября.

ТЗ 4 требует показать «результаты экспериментов, сравнение вариантов модели
и анализ ошибок». Здесь в одну таблицу и один слайд сведено всё, что
пробовалось за спринт, на одной мере и одной проверке:

* **агрегат** — F1 `quality_class` по всем 249 размеченным изображениям,
  честно (пороги и выбор — только на обучающих фолдах `split.csv`), ДИ 95%
  — бутстреп по исследованиям. Источники: `reports/multilabel_head/quality.csv`
  (день 8), решения конвейера со вложенной ротацией (`15_checkpoint_metrics`),
  ротация v2 в точке локализатора малого вертела (`rotation_localizer/`,
  вложенно, отклонена по критерию, заданному до проверки),
  ResNet18-ротация из неслитой ветки (снимок в `transfer_snapshot/`);
* **по задачам** — варианты детектора артефактов (`artifact_cv.json`, 20
  повторов стратифицированной проверки) и подходы к ротации (OOF и вложенные);
* **локализация** — расстояние от точки малого вертела до точки врача на 58
  кадрах (ТЗ 8.4, «расстояние между ключевыми точками»): якорь v1, который
  работает в сервисе, и локализатор v2 (OOF).

Выход — `reports/model_comparison/`: `aggregate.csv`, `tasks.csv`,
`localization.csv`, `summary.json`, `slide.png` (1920×1080, для презентации). Время — секунды,
снимки не читаются.

Запуск:
    python scripts/24_model_comparison.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dxa_qc.metrics import binary_metrics, bootstrap_ci, f1_ci  # noqa: E402

REPORTS = ROOT / "reports"
OUT = REPORTS / "model_comparison"
SNAP = OUT / "transfer_snapshot"
V2 = REPORTS / "rotation_localizer"
N_BOOT = 10000

_spec = importlib.util.spec_from_file_location("cm", ROOT / "scripts" / "15_checkpoint_metrics.py")
cm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cm)

# Системы дня 8 (quality.csv, срез «все изображения») -> подпись на слайде.
HEADS = {
    "Всё — нарушение (тривиальная база)": ("Всё — нарушение (база)", "base"),
    "Голова img: OR по меткам": ("Классификатор по пикселям\nснимка", "alt"),
    "Правила, честные пороги (контрольная точка)": ("Правила без ротации", "alt"),
    "Голова geom+img: OR по меткам": ("Гибрид: геометрия\n+ пиксели снимка", "alt"),
    "Голова geom: OR по меткам": ("Обучаемая голова\nна геометрии", "alt"),
}
SERVICE = "Правила + ротация по профилю края\n(в сервисе)"
TRANSFER = "Правила + ResNet18 на области вертела\n(эксперимент, torch)"
ROT_V2 = "Правила + ротация в точке\nлокализатора v2 (отклонена)"


def service_frame() -> pd.DataFrame:
    """Решения сервиса на 249 изображениях: геометрия с честными порогами,
    артефакты out-of-fold, ротация — вложенная проверка (как в README).

    `q_v2` — то же, но ротация — вложенные решения правила v2 в точке
    локализатора малого вертела (`rotation_localizer/nested_oof.csv`,
    двойной кросс-фиттинг). Остальные правила те же, так что разница
    двух колонок — только ротация.
    """
    df = cm.load(REPORTS)
    v2 = pd.read_csv(V2 / "nested_oof.csv")[["rel_path", "pred"]]
    df = df.merge(v2.rename(columns={"pred": "rot_v2"}), on="rel_path", how="left", validate="one_to_one")
    femur = df.area.eq("femur")
    if df.loc[femur, "rot_v2"].isna().any() or df.loc[~femur, "rot_v2"].notna().any():
        raise ValueError("решения v2 не совпадают с набором изображений бедра")
    df, _ = cm.nested_predictions(df)
    df["q_service"] = cm.aggregate(df, "n")
    df["n_rot"] = df.rot_v2.eq(1).astype(int)          # -1 (отказ) — не нарушение, как в конвейере
    df["q_v2"] = cm.aggregate(df, "n")
    return df[["rel_path", "study", "area", "y_quality", "q_service", "q_v2"]]


def check_femur(m: pd.DataFrame) -> None:
    """Бедро отдельно должно совпасть с таблицей Ильи (`quality_comparison.csv`):
    так видно, что v2 и v1 взяты из тех же решений, что в его отчёте."""
    qc = pd.read_csv(V2 / "quality_comparison.csv").set_index("variant")
    f = m[m.area.eq("femur")]
    for col, key in (("q_service", "v1"), ("q_v2", "v2")):
        got = binary_metrics(f.y_quality.to_numpy(int), f[col].to_numpy(int)).f1
        if abs(got - qc.loc[key, "f1"]) > 1e-9:
            raise ValueError(f"F1 бедра {key}: {got:.4f} против {qc.loc[key, 'f1']:.4f} в quality_comparison.csv")


def paired_delta(y, a, b, groups) -> tuple[float, float, float]:
    """ΔF1 = F1(b) − F1(a) и ДИ парного бутстрепа по исследованиям."""
    code = np.asarray(a, int) * 2 + np.asarray(b, int)       # одна колонка на оба решения
    stat = lambda yy, v: (binary_metrics(yy, (v % 2).astype(int)).f1  # noqa: E731
                          - binary_metrics(yy, (v // 2).astype(int)).f1)
    lo, hi = bootstrap_ci(y, code, stat, groups=groups, n_bootstrap=N_BOOT)
    return float(stat(np.asarray(y), code.astype(float))), lo, hi


def aggregate_table() -> tuple[pd.DataFrame, dict]:
    q = pd.read_csv(REPORTS / "multilabel_head/quality.csv")
    q = q[q["срез"] == "все изображения"].set_index("система")
    rows = []
    for key, (label, kind) in HEADS.items():
        r = q.loc[key]
        rows.append({"система": label, "вид": kind, "f1": r.f1, "lo": r.f1_lo, "hi": r.f1_hi,
                     "sensitivity": r.sensitivity, "specificity": r.specificity,
                     "источник": "reports/multilabel_head/quality.csv"})

    svc = service_frame()
    tr = pd.read_csv(SNAP / "local_aggregate_oof.csv")[["rel_path", "y_quality", "after"]]
    m = svc.merge(tr, on="rel_path", suffixes=("", "_tr"), validate="one_to_one")
    if len(m) != 249 or not (m.y_quality == m.y_quality_tr).all():
        raise ValueError("снимок ветки ResNet18 не совпадает с набором по изображениям или меткам")
    check_femur(m)
    y, g = m.y_quality.to_numpy(int), m.study.to_numpy()
    extra = {}
    for label, kind, col, src in ((ROT_V2, "alt", "q_v2", "rotation_localizer/nested_oof.csv (вложенная)"),
                                  (SERVICE, "service", "q_service", "scripts/15 (вложенная ротация)"),
                                  (TRANSFER, "experiment", "after", "transfer_snapshot/local_aggregate_oof.csv")):
        p = m[col].to_numpy(int)
        bm = binary_metrics(y, p)
        lo, hi = f1_ci(p, y, groups=g, n_bootstrap=N_BOOT)
        rows.append({"система": label, "вид": kind, "f1": bm.f1, "lo": lo, "hi": hi,
                     "sensitivity": bm.sensitivity, "specificity": bm.specificity, "источник": src})
    if abs(rows[-2]["f1"] - q.loc["Правила + ротация напарника (вложенная)"].f1) > 1e-9:
        raise ValueError("F1 сервиса разошёлся с таблицей дня 8")
    d, dlo, dhi = paired_delta(y, m.q_service, m.after, g)
    extra["ResNet18 минус сервис, ΔF1 [ДИ 95%]"] = [round(d, 4), round(dlo, 4), round(dhi, 4)]
    d, dlo, dhi = paired_delta(y, m.q_service, m.q_v2, g)
    extra["v2 минус сервис, ΔF1 [ДИ 95%]"] = [round(d, 4), round(dlo, 4), round(dhi, 4)]
    rules = m.q_service.to_numpy(int)                   # для сравнения с базой
    base = np.ones_like(rules)
    d, dlo, dhi = paired_delta(y, base, rules, g)
    extra["сервис минус «всё — нарушение», ΔF1 [ДИ 95%]"] = [round(d, 4), round(dlo, 4), round(dhi, 4)]
    t = pd.DataFrame(rows).sort_values("f1").reset_index(drop=True)
    return t, extra


def task_table() -> pd.DataFrame:
    rows = []
    art = json.loads((REPORTS / "artifact_cv.json").read_text(encoding="utf-8"))["сравнение_вариантов"]
    names = {"top-hat (1 признак)": ("top-hat, 1 признак", "service"),
             "гребень (1 признак)": ("гребень (гессиан), 1 признак", "alt"),
             "top-hat + гребень": ("top-hat + гребень", "alt"),
             "HOG 144 признака": ("HOG, 144 признака", "alt")}
    for key, (label, kind) in names.items():
        v = art[key]
        rows.append({"задача": "артефакты позвоночника (17 позитивов)", "вариант": label, "вид": kind,
                     "f1": v["f1_mean"], "разброс": v["f1_std"], "roc_auc": v["roc_auc_mean"],
                     "проверка": "20 повторов CV, порог из обучающих фолдов"})
    rot = pd.read_csv(REPORTS / "femur_rotation_cv.csv").set_index("вариант")
    for key, label, kind in (("угол оси диафиза", "угол оси диафиза", "alt"),
                             ("угол + заполненность", "угол + заполненность маски", "rejected"),
                             ("пиксели 16x24 (384)", "пиксели бедра, 384 признака", "alt")):
        rows.append({"задача": "ротация бедра (36 позитивов)", "вариант": label, "вид": kind,
                     "f1": rot.loc[key, "f1"], "разброс": np.nan, "roc_auc": rot.loc[key, "roc_auc"],
                     "проверка": "OOF, порог из обучающих фолдов"})
    for path, label, kind in ((SNAP / "full_summary.json", "ResNet18, весь снимок", "alt"),
                              (REPORTS / "rotation_human_profile/summary.json", "профиль медиального края", "service"),
                              (V2 / "summary.json", "профиль в точке локализатора v2", "prereg"),
                              (SNAP / "local_summary.json", "ResNet18, область вертела", "experiment")):
        nm = json.loads(path.read_text(encoding="utf-8"))["nested_metrics"]
        rows.append({"задача": "ротация бедра (36 позитивов)", "вариант": label, "вид": kind,
                     "f1": nm["f1"], "разброс": np.nan, "roc_auc": np.nan,
                     "проверка": f"вложенная, ДИ F1 [{nm['f1_ci95'][0]:.2f}; {nm['f1_ci95'][1]:.2f}]"})
    return pd.DataFrame(rows)


def localization_table() -> pd.DataFrame:
    """Ошибка точки малого вертела до точки врача, мм (58 кадров, OOF для v2)."""
    loc = pd.read_csv(V2 / "localization.csv")
    names = {"v1_anchor": "якорь v1 (в сервисе)", "v2_localizer": "локализатор v2 (OOF)"}
    if set(loc.variant) != set(names):
        raise ValueError(f"неожиданные варианты в localization.csv: {sorted(loc.variant)}")
    loc["вариант"] = loc.variant.map(names)
    return loc[["вариант", "variant", "n", "median_mm", "p90_mm", "within_5mm"]]


# --------------------------------------------------------------------------- #
# слайд
# --------------------------------------------------------------------------- #
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
SURFACE, GRID = "#fcfcfb", "#e6e5e1"
ACCENT, NEUTRAL, BASEC = "#2a78d6", "#b3b1ab", "#d9d7d1"


def slide(agg: pd.DataFrame, tasks: pd.DataFrame, loc: pd.DataFrame, extra: dict, path: Path) -> None:
    """Точка с ДИ, а не столбец: ось начинается не с нуля, столбцы её бы исказили."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15})
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=SURFACE)
    fig.text(0.04, 0.925, "Сравнение вариантов модели: что пробовали и почему в сервисе правила",
             fontsize=30, weight="bold", color=INK)
    fig.text(0.04, 0.88, "Одни и те же 249 изображений и 5 фолдов по исследованиям; пороги и выбор — "
             "только на обучающих фолдах; ДИ 95% — бутстреп по исследованиям",
             fontsize=15, color=INK2)

    # --- слева: агрегат ---
    fig.text(0.04, 0.795, "F1 «есть нарушение», все изображения", fontsize=19, color=INK, weight="bold")
    ax = fig.add_axes([0.285, 0.215, 0.30, 0.545], facecolor=SURFACE)
    n = len(agg)
    for i, r in agg.iterrows():
        kind = r["вид"]
        ax.plot([r.lo, r.hi], [i, i], color=INK2 if kind != "base" else MUTED, lw=2, zorder=2,
                solid_capstyle="round")
        face = {"service": ACCENT, "experiment": SURFACE, "base": BASEC}.get(kind, NEUTRAL)
        edge = ACCENT if kind in ("service", "experiment") else (MUTED if kind == "base" else NEUTRAL)
        ax.scatter([r.f1], [i], s=260 if kind in ("service", "experiment") else 170, color=face,
                   edgecolors=edge, linewidths=3 if kind == "experiment" else 1.5, zorder=3)
        strong = kind in ("service", "experiment")
        ax.text(r.hi + 0.012, i, f"{r.f1:.2f}", va="center", ha="left", fontsize=18 if strong else 16,
                weight="bold" if strong else "normal", color=INK)
        ax.text(-0.03, i, r["система"], va="center", ha="right", fontsize=15, linespacing=1.15,
                weight="bold" if kind == "service" else "normal", color=INK,
                transform=ax.get_yaxis_transform())      # x — в долях оси, y — в данных
    base = agg.loc[agg["вид"] == "base", "f1"].iloc[0]
    ax.axvline(base, color=MUTED, lw=1.4, ls=(0, (4, 3)), zorder=1)
    ax.set_xlim(0.35, 0.78)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks([])
    ax.set_xticks([0.4, 0.5, 0.6, 0.7])
    ax.tick_params(axis="x", colors=INK2, labelsize=14, length=0, pad=8)
    ax.grid(axis="x", color=GRID, lw=1, zorder=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    d = extra["ResNet18 минус сервис, ΔF1 [ДИ 95%]"]
    b = extra["сервис минус «всё — нарушение», ΔF1 [ДИ 95%]"]
    v = extra["v2 минус сервис, ΔF1 [ДИ 95%]"]
    fig.text(0.04, 0.135,
             f"Сервис лучше тривиальной базы «всё — нарушение» (пунктир): ΔF1 {b[0]:+.2f} "
             f"[{b[1]:+.2f}; {b[2]:+.2f}].\n"
             f"ResNet18 против сервиса: ΔF1 {d[0]:+.2f} [{d[1]:+.2f}; {d[2]:+.2f}] — в пределах шума; "
             "нужен torch в образе, область вертела\nвыбрана после первого эксперимента.\n"
             f"Ротация v2 против сервиса: ΔF1 {v[0]:+.2f} [{v[1]:+.2f}; {v[2]:+.2f}] — "
             "не включена: порог задан до проверки.",
             fontsize=14, color=INK2, va="top", linespacing=1.45)

    # --- справа: по задачам ---
    x0, xv, xm = 0.655, 0.905, 0.912
    fig.text(x0, 0.795, "По задачам: F1", fontsize=19, color=INK, weight="bold")
    y = 0.74
    marks = {"service": ("← в сервисе", ACCENT), "experiment": ("← эксперимент", INK2),
             "rejected": ("← отклонён*", INK2), "prereg": ("← отклонён**", INK2)}
    for task, grp in tasks.groupby("задача", sort=False):
        fig.text(x0, y, task, fontsize=15, color=INK2)
        y -= 0.038
        for r in grp.itertuples():
            strong = r.вид == "service"
            w = "bold" if strong else "normal"
            fig.text(x0 + 0.01, y, r.вариант, fontsize=15, color=INK, weight=w)
            fig.text(xv, y, f"{r.f1:.2f}", fontsize=15, color=INK, ha="right", weight=w)
            if r.вид in marks:
                text, color = marks[r.вид]
                fig.text(xm, y, text, fontsize=13, color=color)
            y -= 0.034
        y -= 0.02
    lv = loc.set_index("variant")
    fig.text(x0, y + 0.005, "* заполненность маски мерит плотность кости, а не укладку\n"
             f"** точка вертела точнее: медиана ошибки {lv.at['v1_anchor', 'median_mm']:.1f} → "
             f"{lv.at['v2_localizer', 'median_mm']:.1f} мм,\n"
             "    но F1 ниже порога 0.53, заданного до проверки",
             fontsize=12.5, color=INK2, va="top", linespacing=1.35)
    y -= 0.075
    fig.text(x0, y - 0.045,
             "Вывод: при 6–36 позитивах на метку обучаемые\n"
             "модели правилам не выигрывают — цена выбора\n"
             "порога съедает их преимущество. В сервисе —\n"
             "объяснимые правила, каждое видно на overlay.",
             fontsize=15, color=INK, va="top", linespacing=1.4)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    agg, extra = aggregate_table()
    tasks = task_table()
    loc = localization_table()
    agg.round(4).to_csv(OUT / "aggregate.csv", index=False)
    tasks.round(4).to_csv(OUT / "tasks.csv", index=False)
    loc.round(4).to_csv(OUT / "localization.csv", index=False)
    summary = {"агрегат (F1 quality_class, 249 изображений)":
               {r["система"]: [round(r.f1, 3), round(r.lo, 3), round(r.hi, 3)] for _, r in agg.iterrows()},
               **extra,
               "локализация малого вертела, мм (58 точек врача)":
               {r.вариант: {"медиана": round(r.median_mm, 2), "p90": round(r.p90_mm, 2),
                            "доля ≤ 5 мм": round(r.within_5mm, 3)} for r in loc.itertuples()}}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    slide(agg, tasks, loc, extra, OUT / "slide.png")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(tasks[["задача", "вариант", "f1"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
