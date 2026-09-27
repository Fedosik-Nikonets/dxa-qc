"""Геометрические правила по поясничному отделу позвоночника (ТЗ 2.3.1).

Два из трёх нарушений позвоночника считаются из маски кости детерминированно,
без обучения и без разметки:

* **наклон оси более 5°** — угол главной оси маски к вертикали;
* **некорректная укладка** — в кадр не попали верхние края подвздошных костей.

Третье (инородные тела и артефакты) остаётся за классификатором напарника.

Порог наклона оси: 4.5°, а не 5°
--------------------------------
В ТЗ порог записан как «до 5°», но величина угла зависит от способа оценки:
на одной и той же маске хорда четвертей, линия построчных центроидов и
главная ось дают систематически разные модули (разведка 17 сентября,
`scripts/06_spine_axis.py`). Главная ось всей маски выбрана как самая
устойчивая оценка, а порог откалиброван по F1 на 10 позитивах: 4.5° даёт
чувствительность 0.70 и специфичность 0.88, буквальные 5° — 0.60 и 0.90.

Потолок правила — не в геометрии, а в семантике метки: из 20 позвоночников
с находкой вне перечня 2.3 (в основном сколиоз) нарушением оси эксперт
пометил ровно один. Отклонение оси при сколиозе он считает анатомией, а не
ошибкой укладки, и шесть из десяти ложных срабатываний правила — такие
случаи. Признаки кривизны (`res_rms`, `res_amp`, `slope_spread`) считаются
здесь же и отдаются наружу: они идут вторым входом мультилейбл-головы
22–23 сентября, чтобы отделить изогнутый позвоночник от косо уложенного
прямого. Подробности — `reports/day3_findings.md`.

Укладка: проверяется одна половина критерия из двух
---------------------------------------------------
Формулировка ТЗ 2.3.1 — конъюнкция: «в кадр не попали верхние края
подвздошных костей **и/или** половина тела Th12». Из маски проверяема только
первая половина, и проверяется она не касанием маски нижнего края кадра
(маска позвоночника касается его во всех 99 исследованиях, признак
вырожден), а наличием плотных структур **сбоку** от позвоночника в нижней
трети кадра. Если сканирование остановилось выше гребней подвздошных костей,
по бокам внизу остаётся только воздух и тонкие мягкие ткани.

Мера — максимум сглаженного профиля латеральной плотности по нижней трети
кадра, нормированный на яркость самой кости. На обучающем наборе:
4 позитива из 6 ниже 0.02, все 93 негатива выше 0.074 — зазор вчетверо,
решение не меняется ни на одном из 36 сочетаний параметров (отступ 4–12 мм,
нижняя доля 0.25–0.40, сглаживание 4–10 мм).

Оставшиеся два позитива — вторая половина критерия (не попала половина тела
Th12). Шесть признаков верхнего края кадра (перцентили латеральной яркости,
энергия высоких частот как индикатор рёбер) их не отделяют: один из двух
лежит в середине распределения негативов. На двух позитивах правило не
строится, это записано в «Ограничения».
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .segmentation import BoneMask, principal_axis, segment_bone

# Идентификаторы нарушений — из taxonomy.yaml, раздел labels.spine.
SPINE_AXIS_TILT = "SPINE_AXIS_TILT"
SPINE_POSITIONING = "SPINE_POSITIONING"

# Порог наклона оси в градусах. Калибровка — по F1, см. docstring модуля.
AXIS_TILT_DEG = 4.5

# Порог латеральной плотности для правила укладки. Зазор на обучающем
# наборе: 0.018 (худший пойманный позитив) против 0.075 (самый бедный
# негатив); порог стоит примерно в геометрической середине зазора.
COVERAGE_MIN = 0.04

LATERAL_PAD_MM = 6.0      # отступ от края колонки позвоночника вбок
LOWER_FRACTION = 1.0 / 3  # какая доля кадра снизу считается «нижней третью»
SMOOTH_MM = 6.0           # окно сглаживания профиля латеральной плотности
DEFAULT_MM_PER_PX = 0.6   # паспорт GE Lunar Prodigy Advance, если тега нет


@dataclass
class SpineFindings:
    """Результат геометрической проверки позвоночника."""

    violations: list[str] = field(default_factory=list)
    axis_angle_deg: float = float("nan")   # знаковый: плюс — верх вправо
    coverage: float = float("nan")         # латеральная плотность внизу кадра
    res_rms: float = float("nan")          # кривизна: СКО остатка от прямой
    res_amp: float = float("nan")          # кривизна: размах остатка
    slope_spread: float = float("nan")     # кривизна: разброс наклонов третей
    ok: bool = False
    reason: str | None = None

    @property
    def quality_class(self) -> int:
        return 1 if self.violations else 0

    def diagnostics(self) -> dict[str, Any]:
        """Служебные колонки для разбора ошибок (reports/batch_diagnostics)."""
        return {
            "spine_angle_deg": _round(self.axis_angle_deg),
            "spine_coverage": _round(self.coverage),
            "spine_res_rms": _round(self.res_rms),
            "spine_res_amp": _round(self.res_amp),
            "spine_slope_spread": _round(self.slope_spread),
            "spine_reason": self.reason or "",
        }


def _round(x: float, n: int = 4) -> float | str:
    return "" if x != x else round(float(x), n)


def centerline(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Построчные центроиды маски: (номера непустых строк, центры)."""
    w = mask.sum(axis=1)
    ok = w > 0
    if not ok.any():
        return np.empty(0), np.empty(0)
    cen = (mask[ok] * np.arange(mask.shape[1])).sum(axis=1) / w[ok]
    return np.arange(mask.shape[0])[ok].astype(float), cen


def axis_angle(mask: np.ndarray) -> float:
    """Угол главной оси маски к вертикали в градусах, со знаком."""
    return principal_axis(mask)[0]


def curvature(mask: np.ndarray) -> dict[str, float]:
    """Признаки кривизны центральной линии.

    Косая укладка — прямая под углом, сколиоз — дуга: у неё остаток от
    прямой велик, а локальные наклоны третей разъезжаются. Признаки не
    участвуют в правилах, но уходят в мультилейбл-голову и в диагностику.
    """
    nan = {"res_rms": float("nan"), "res_amp": float("nan"),
           "slope_spread": float("nan")}
    r, c = centerline(mask)
    if r.size < 12:
        return nan
    k = max(5, len(r) // 15) | 1
    if len(c) < k:
        return nan
    sm = np.convolve(c, np.ones(k) / k, mode="valid")
    rs = r[k // 2: k // 2 + len(sm)]
    if len(rs) < 6:
        return nan
    res = sm - np.polyval(np.polyfit(rs, sm, 1), rs)
    third = len(rs) // 3
    if third >= 3:
        slopes = [float(np.polyfit(rs[i * third:(i + 1) * third],
                                   sm[i * third:(i + 1) * third], 1)[0])
                  for i in range(3)]
        spread = float(np.degrees(np.arctan(max(slopes)))
                       - np.degrees(np.arctan(min(slopes))))
    else:
        spread = float("nan")
    return {"res_rms": float(np.sqrt((res ** 2).mean())),
            "res_amp": float(res.max() - res.min()),
            "slope_spread": spread}


def lateral_coverage(image: np.ndarray, mask: np.ndarray,
                     mm_per_px: float = DEFAULT_MM_PER_PX,
                     *, pad_mm: float = LATERAL_PAD_MM,
                     lower_fraction: float = LOWER_FRACTION,
                     smooth_mm: float = SMOOTH_MM) -> float:
    """Плотность структур сбоку от позвоночника в нижней трети кадра.

    Единица измерения — доля от контраста «кость минус фон»: 0 — сбоку
    пусто, 1 — сбоку так же плотно, как в самой кости. Нормировка на
    собственную яркость кадра делает меру независимой от настроек
    аппарата и от толщины пациента.

    NaN означает «судить не по чему»: колонка позвоночника занимает всю
    ширину кадра и боковой области не осталось. Правило на NaN не
    срабатывает — см. `evaluate`.
    """
    a = np.asarray(image, dtype=float)
    if a.ndim != 2 or a.size == 0 or not np.asarray(mask).any():
        return float("nan")
    h, w = a.shape
    mm = float(mm_per_px) if mm_per_px and mm_per_px == mm_per_px else DEFAULT_MM_PER_PX

    # Границы колонки позвоночника берутся по центральной трети кадра:
    # внизу к маске может примыкать таз, вверху — рёбра, и по всему кадру
    # колонка «расползлась» бы на них.
    mid = mask[h // 3: max(2 * h // 3, h // 3 + 1)]
    cols = np.flatnonzero(mid.any(axis=0))
    if cols.size == 0:
        cols = np.flatnonzero(np.asarray(mask).any(axis=0))
    c0, c1 = int(cols.min()), int(cols.max()) + 1

    pad = max(int(round(pad_mm / mm)), 2)
    lat = np.zeros(w, dtype=bool)
    lat[:max(c0 - pad, 0)] = True
    lat[min(c1 + pad, w):] = True
    if not lat.any():
        return float("nan")

    base = float(np.percentile(a, 1))          # фон кадра
    bone = float(a[np.asarray(mask, bool)].mean())  # яркость кости
    scale = max(bone - base, 1e-6)
    prof = (a[:, lat].mean(axis=1) - base) / scale

    k = max(3, int(round(smooth_mm / mm)) | 1)
    if len(prof) > k:
        prof = np.convolve(prof, np.ones(k) / k, mode="same")
    lo = int(h * (1.0 - lower_fraction))
    band = prof[lo:]
    return float(band.max()) if band.size else float("nan")


def evaluate(image: np.ndarray, mm_per_px: float | None = None,
             bone: BoneMask | None = None) -> SpineFindings:
    """Проверяет позвоночник по двум геометрическим правилам ТЗ 2.3.1.

    `bone` можно передать снаружи, если маска уже посчитана: сегментация
    самая дорогая часть, а в конвейере она нужна и роутеру, и
    классификаторам напарника.
    """
    res = SpineFindings()
    a = np.asarray(image) if image is not None else None
    if a is None or a.ndim != 2 or a.size == 0:
        res.reason = "нет двумерного изображения"
        return res

    if bone is None:
        bone = segment_bone(a)
    if not bone.ok or not bone.mask.any():
        res.reason = bone.reason or "маска кости пуста"
        return res

    res.ok = True
    res.axis_angle_deg = axis_angle(bone.mask)
    res.coverage = lateral_coverage(a, bone.mask, mm_per_px or DEFAULT_MM_PER_PX)
    for k, v in curvature(bone.mask).items():
        setattr(res, k, v)

    # Порядок нарушений в строке фиксирован — так таблица сравнима построчно.
    if res.coverage == res.coverage and res.coverage < COVERAGE_MIN:
        res.violations.append(SPINE_POSITIONING)
    if res.axis_angle_deg == res.axis_angle_deg and abs(res.axis_angle_deg) > AXIS_TILT_DEG:
        res.violations.append(SPINE_AXIS_TILT)
    return res
