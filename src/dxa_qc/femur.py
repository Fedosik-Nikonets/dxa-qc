"""Геометрические правила по проксимальному отделу бедра (ТЗ 2.3.2).

Здесь считается одно из двух нарушений бедра — охват области интереса.
Второе (позиционирование/ротация) из геометрии маски не строится
(разведка 19 сентября, `reports/day5_rotation.md`); с 23 сентября оно
считается по профилю медиального края в `rotation_rule.py`.

Признак из плана оказался вырожденным
-------------------------------------
План на 19–20 сентября требовал считать «отступы bbox кости до краёв кадра
в миллиметрах» и сравнивать их с порогами ТЗ 30 мм сверху/снизу и 20 мм
сбоку. Измерено на 150 размеченных изображениях бедра:

* нижний отступ равен нулю у **всех 150** — диафиз выходит за нижний край
  кадра всегда, и у нарушений, и у нормы;
* верхний отступ равен нулю у большинства — сверху кадр режет таз;
* боковой отступ до края области сканирования имеет медиану 8.4 мм при
  пороге ТЗ 20 мм. Буквальное правило пометило бы нарушением почти весь
  набор, а направление связи у него обратное (ROC-AUC 0.168, то есть у
  позитивов боковой отступ *больше*).

Вертикальные отступы измерять не по чему: кадр обрезает кость с обеих
сторон по построению. Это тот же случай, что и «касание маской нижнего края»
у позвоночника 18 сентября — признак из плана вырожден, и проверять его
нужно было раньше, чем закладывать в правило.

Что работает: длина поля сканирования
-------------------------------------
Величина, которая у бедра действительно меняется от исследования к
исследованию, — высота кадра: от 180 до 349 пикселей, то есть от 109 до
209 мм. Это и есть решение оператора о длине скана, а критерий ТЗ 2.3.2
«не менее 3 см сверху и снизу» — ровно требование к этой длине: проксимальный
отдел бедра занимает 9–10 см, плюс два отступа по 3 см дают минимум около
15 см.

Масштаб для этой длины берётся как минимум из тега Exposed Area и
паспортных 0.6 мм/пиксель: тег описывает поле сканирования, а кадр — вырезку
из него, поэтому оценка по тегу завышена тем сильнее, чем сильнее обрезан
кадр (до 9.7% на обучении, 8.5% на отложенном наборе). Подробности — в
`field_scale`.

Порог откалиброван по F1 на 7 позитивах и равен **133 мм**, а не 150 мм из
буквального чтения ТЗ: эксперт помечает нарушением только совсем короткие
поля. Решение не меняется на всём отрезке от 125.4 мм (худший пойманный позитив)
до 139.8 мм (самый короткий непойманный негатив), порог стоит почти точно
в середине этого плато.
ROC-AUC 0.939, ДИ 95% [0.836, 0.998]; при пороге 133 мм чувствительность
0.71, специфичность 0.986, F1 0.71.

Почему это не ловушка «ширина кадра» из дня 3
---------------------------------------------
17 сентября ширину кадра пришлось выбросить из роутера: она идеально
разделяла области, но была свойством протокола аппарата, а не анатомии.
Высота кадра — не константа протокола: она меняется в пределах одного
аппарата в два раза и задаётся оператором при укладке. Это измерение самого
действия, качество которого оценивается, а не побочный признак железа.

Ограничения (в README)
----------------------
1. Два позитива из семи правило не ловит: поля 141.0 и 156.6 мм. Первое
   совпадает со значением сразу у нескольких негативов — по длине скана эти
   случаи неразличимы в принципе. У обоих боковой отступ мал (10.9 и 7.3 мм)
   при медиане 8.4 мм у негативов, то есть и боковым правилом они не
   отделяются. На двух примерах правило не строится.
2. Оба ложных срабатывания — исследования, в которых эксперт отметил
   нарушение ротации. На уровне `quality_class` по бедру ложных
   срабатываний у правила нет вовсе.
3. Ротация из геометрии маски не строится: угол оси диафиза к вертикали
   даёт out-of-fold F1 0.42 при специфичности 0.62 (43 ложных срабатывания
   на 114 негативов). Правило ротации в конвейере — по профилю медиального
   края, см. `rotation_rule.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .segmentation import BoneMask, _fill_holes, _to_u8, segment_bone

# Идентификаторы нарушений — из taxonomy.yaml, раздел labels.femur.
FEMUR_ROI_COVERAGE = "FEMUR_ROI_COVERAGE"
FEMUR_POSITIONING_ROTATION = "FEMUR_POSITIONING_ROTATION"  # считается в rotation_rule, не здесь

# Минимальная длина поля сканирования. Калибровка — по F1, плато 127–139 мм.
FIELD_MIN_MM = 133.0

DEFAULT_MM_PER_PX = 0.6   # паспорт GE Lunar Prodigy Advance

# Насколько масштаб из тега может разойтись с паспортным, прежде чем это
# попадёт в диагностику как подозрительное. См. `field_scale`.
SCALE_WARN = 0.15

# Мягкий порог маски именно бедренной кости: перцентиль яркости внутри тела.
# Основная маска (`segment_bone`) настроена на плотную кортикальную кость и
# в вертельной области рассыпается; для оси диафиза нужен сплошной контур.
FEMUR_PERCENTILE = 45.0
SHAFT_FRACTION = 0.35     # какая доля маски снизу считается диафизом


@dataclass
class FemurFindings:
    """Результат геометрической проверки бедра."""

    violations: list[str] = field(default_factory=list)
    field_h_mm: float = float("nan")      # длина поля сканирования
    field_w_mm: float = float("nan")      # ширина непустой части кадра
    lateral_mm: float = float("nan")      # отступ от кости до края со стороны вертела
    medial_mm: float = float("nan")       # отступ со стороны таза
    shaft_angle_deg: float = float("nan")  # угол оси диафиза к вертикали
    femur_fill: float = float("nan")      # заполненность bbox маской бедра
    ok: bool = False
    reason: str | None = None
    mm_per_px: float = float("nan")       # применённый масштаб
    mm_per_px_source: str | None = None   # откуда он взят

    @property
    def quality_class(self) -> int:
        return 1 if self.violations else 0

    def diagnostics(self) -> dict[str, Any]:
        """Служебные колонки. В правиле участвует только `femur_field_h_mm`."""
        return {
            "femur_field_h_mm": _round(self.field_h_mm, 1),
            "femur_field_w_mm": _round(self.field_w_mm, 1),
            "femur_lateral_mm": _round(self.lateral_mm, 1),
            "femur_medial_mm": _round(self.medial_mm, 1),
            "femur_shaft_deg": _round(self.shaft_angle_deg, 2),
            "femur_fill": _round(self.femur_fill, 4),
            "femur_mm_per_px": _round(self.mm_per_px, 4),
            "femur_mm_per_px_source": self.mm_per_px_source or "",
            "femur_reason": self.reason or "",
        }


def _round(x: float, n: int = 4) -> float | str:
    return "" if x != x else round(float(x), n)


TRUSTED_SOURCES = ("pixel_spacing", "imager_pixel_spacing")
CALIBRATED_MODEL = "lunar prodigy"   # аппарат, на котором калиброван паспортный 0.6


def field_scale(mm_per_px: float | None, source: str | None = None,
                model: str | None = None) -> tuple[float, str | None]:
    """Размер пикселя для измерения длины поля.

    Зажимать масштаб паспортом можно **только** для оценки из Exposed Area:
    она заведомо оценка сверху. Для PixelSpacing и ImagerPixelSpacing зажим
    был бы порчей достоверных данных — аппарат с пикселем 0.8 мм получил бы
    0.6, длина поля вышла бы на четверть короче настоящей, и правило начало
    бы штамповать ложные срабатывания. В обучающем наборе этих тегов нет ни
    в одном из 499 файлов, поэтому на наших метриках разницы нет; но именно
    их наличие в закрытом наборе — самый вероятный сценарий, и до 20
    сентября код молча портил бы их. Позвоночник, для сравнения, тегу верил
    всегда (`spine.lateral_coverage`).

    Exposed Area (0040,0303) описывает поле сканирования, а сохранённое
    изображение — вырезанный из него фрагмент (разведка 15 сентября).
    Поэтому отношение «ширина поля / ширина кадра» есть **оценка сверху**:
    чем сильнее обрезан кадр, тем больше завышен масштаб. На обучающем
    наборе завышение доходит до 9.7%, а в отложенном наборе «Для теста» оба
    бедра получают 0.6513 мм/пиксель вместо паспортных 0.6 — это 8.5%.

    Для правила охвата это существенно: 8.5% на кадре 209 строк — разница
    между 125 и 136 мм, то есть между нарушением и нормой при пороге 133.
    Если бы весь скрытый набор имел такое завышение, чувствительность
    правила упала бы с 0.71 до 0.29, и молча.

    Поэтому берётся минимум: завышение тегом отсекается, а аппарат с
    действительно меньшим пикселем (тег ниже паспортного) масштаб
    по-прежнему задаёт сам. Расхождение больше SCALE_WARN попадает в
    диагностику. На обучающем наборе выбор источника не меняет ни одного
    решения из 150 — правило от этого не зависит, но и не ломается.
    """
    unknown = _foreign_scanner(model)
    valid = (mm_per_px is not None and mm_per_px == mm_per_px
             and float(mm_per_px) > 0)

    if valid and source in TRUSTED_SOURCES:
        # Достоверный тег. Паспорт здесь ни при чём, он лишь повод сказать,
        # что аппарат не тот, на котором мы калибровались.
        mm = float(mm_per_px)
        off = abs(mm - DEFAULT_MM_PER_PX) / DEFAULT_MM_PER_PX
        if off > SCALE_WARN:
            return mm, (f"масштаб из тега {source} {mm:.3f} мм/пиксель, "
                        f"паспортный {DEFAULT_MM_PER_PX} — расхождение {off:.0%}, "
                        "порог правила калиброван на паспортном")
        return mm, None

    if not valid:
        note = "масштаб не определён, взят паспортный"
        return DEFAULT_MM_PER_PX, f"{note}; {unknown}" if unknown else note

    # Остаётся Exposed Area: оценка сверху, её и зажимаем.
    mm = min(float(mm_per_px), DEFAULT_MM_PER_PX)
    off = abs(float(mm_per_px) - DEFAULT_MM_PER_PX) / DEFAULT_MM_PER_PX
    note = None
    if off > SCALE_WARN:
        note = (f"масштаб из тега {float(mm_per_px):.3f} расходится с "
                f"паспортным на {off:.0%}")
    if unknown:
        note = f"{note}; {unknown}" if note else unknown
    return mm, note


def _foreign_scanner(model: str | None) -> str | None:
    """Замечание, если паспортное значение относится к другому аппарату.

    Порог 133 мм и паспортные 0.6 мм/пиксель калиброваны на GE Lunar
    Prodigy Advance — единственном аппарате обучающего набора (499 файлов
    из 499). На другом аппарате без достоверного тега масштаб есть догадка
    о чужой машине, и молчать об этом нельзя.
    """
    if not model:
        return "модель аппарата неизвестна, паспортный масштаб не подтверждён"
    if CALIBRATED_MODEL in str(model).strip().lower():
        return None
    return (f"аппарат «{model}» не тот, на котором калиброван паспортный "
            f"масштаб ({DEFAULT_MM_PER_PX} мм/пиксель, GE Lunar Prodigy Advance)")


def scan_field(image: np.ndarray) -> tuple[int, int]:
    """Границы области сканирования по столбцам: [c0, c1).

    Кадр бедра всегда 280 столбцов, но поле сканирования уже: по бокам
    дописаны нули (у типичного снимка ровно нулевых пикселей больше
    половины кадра). Отступ «сбоку» имеет смысл мерить до границы данных,
    а не до края массива.
    """
    a = np.asarray(image, dtype=float)
    if a.ndim != 2 or a.size == 0:
        return 0, 0
    nz = np.flatnonzero(a.max(axis=0) > a.min())
    if nz.size == 0:
        return 0, a.shape[1]
    return int(nz.min()), int(nz.max()) + 1


def femur_mask(image: np.ndarray, bone: BoneMask,
               percentile: float = FEMUR_PERCENTILE) -> np.ndarray | None:
    """Сплошная маска бедренной кости: компонента, выходящая за нижний край.

    Берётся мягкий порог по яркости внутри тела — так в маску попадает и
    трабекулярная кость вертельной области, которую строгий порог теряет.
    Из компонент выбирается та, что касается нижней строки кадра: диафиз
    выходит за нижний край во всех 150 изображениях набора, а таз — нет.
    """
    if not bone.ok or not bone.body.any():
        return None
    u8 = _to_u8(image)
    thr = float(np.percentile(u8[bone.body], percentile))
    m = (bone.body & (u8 > thr)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None
    bottom = set(np.unique(labels[-1])) - {0}
    cand = [i for i in range(1, n) if i in bottom] or list(range(1, n))
    best = max(cand, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    return _fill_holes(labels == best)


def shaft_angle(mask: np.ndarray, fraction: float = SHAFT_FRACTION) -> float:
    """Угол оси диафиза к вертикали, в градусах, по модулю.

    Считается по построчным центрам нижней доли маски: там кость — почти
    прямая трубка, и прямая по центрам устойчивее главной оси всей маски,
    на которую влияет вертельная область.

    В правилах не участвует: out-of-fold F1 по ротации 0.42 при
    специфичности 0.62 (см. `reports/day5_rotation.md`). Выдаётся в
    диагностику; мультилейбл-голова на таких признаках проверена 22 сентября
    и не принята (`reports/day8_multilabel.md`).
    """
    rows = np.flatnonzero(np.asarray(mask).any(axis=1))
    if rows.size < 30:
        return float("nan")
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    lo = int(r1 - (r1 - r0) * fraction)
    rr, cc = [], []
    for r in range(lo, r1):
        idx = np.flatnonzero(mask[r])
        if idx.size:
            rr.append(float(r))
            cc.append(float(idx.mean()))
    if len(rr) < 15:
        return float("nan")
    k = float(np.polyfit(np.array(rr), np.array(cc), 1)[0])
    return abs(float(np.degrees(np.arctan(k))))


def evaluate(image: np.ndarray, region: str, mm_per_px: float | None = None,
             bone: BoneMask | None = None, mm_per_px_source: str | None = None,
             model: str | None = None) -> FemurFindings:
    """Проверяет бедро по правилу охвата области интереса (ТЗ 2.3.2).

    `region` нужен, чтобы различить латеральную и медиальную стороны кадра:
    у левого бедра таз слева и вертел справа, у правого — зеркально
    (проверено на 150 изображениях, расхождений нет).
    """
    res = FemurFindings()
    a = np.asarray(image) if image is not None else None
    if a is None or a.ndim != 2 or a.size == 0:
        res.reason = "нет двумерного изображения"
        return res

    mm, note = field_scale(mm_per_px, mm_per_px_source, model)
    res.mm_per_px = mm
    res.mm_per_px_source = mm_per_px_source or "scanner_default"
    res.reason = note
    h = a.shape[0]
    res.field_h_mm = h * mm

    if bone is None:
        bone = segment_bone(a)
    if not bone.ok or not bone.mask.any():
        # Длина поля известна и без маски — правило считается всё равно.
        seg = bone.reason or "маска кости пуста"
        res.reason = f"{res.reason}; {seg}" if res.reason else seg
    else:
        res.ok = True
        c0, c1 = scan_field(a)
        res.field_w_mm = (c1 - c0) * mm
        _, bc0, _, bc1 = bone.bbox
        left, right = (bc0 - c0) * mm, (c1 - bc1) * mm
        if region == "femur_left":       # таз слева, вертел справа
            res.lateral_mm, res.medial_mm = right, left
        else:
            res.lateral_mm, res.medial_mm = left, right
        fm = femur_mask(a, bone)
        if fm is not None and fm.any():
            res.shaft_angle_deg = shaft_angle(fm)
            rr, cc = np.nonzero(fm)
            box = (rr.max() - rr.min() + 1) * (cc.max() - cc.min() + 1)
            res.femur_fill = float(fm.sum() / box) if box else float("nan")

    if res.field_h_mm == res.field_h_mm and res.field_h_mm < FIELD_MIN_MM:
        res.violations.append(FEMUR_ROI_COVERAGE)
    return res
