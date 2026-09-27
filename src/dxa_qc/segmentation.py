"""Сегментация кости на денситометрическом снимке.

Зачем это ядро решения
----------------------
Три правила из пяти в ТЗ 2.3 считаются детерминированно из маски кости и
вообще не требуют обучения (см. plan_v2.md, решение Р1):

* наклон оси позвоночника > 5° — главная ось маски позвоночника;
* охват области интереса (3 см / 2 см) — отступы bbox маски до краёв кадра;
* укладка позвоночника — касание маской нижнего края и вид верхней границы.

Поэтому маска обязана быть устойчивой и объяснимой: никакого обучения,
только пороги и морфология, каждый шаг можно показать на слайде.

Конвейер
--------
1. **Три класса яркости.** Трёхклассовый Otsu по всему кадру даёт сразу два
   порога: воздух|мягкие ткани и мягкие ткани|кость. Двухклассовый Otsu здесь
   не годится — он отделяет пациента от воздуха, и маска расползается на весь
   силуэт (проверено).
2. **Маска тела.** Крупнейшая компонента выше первого порога, дырки залиты.
   Это же снимает прямоугольные «заплатки» сшивки по углам кадра.
3. **Маска кости через гистерезис.** Плотное ядро — выше второго порога;
   оно наращивается внутри мягкого порога, но не дальше GROW_RADIUS пикселей.
   Один жёсткий порог оставил бы только кортикальный слой, а костномозговой
   канал диафиза выпал бы; неограниченный гистерезис на однородном кадре
   проглотил бы весь силуэт.
4. **Морфология.** Открытие убирает шум и тонкие перемычки, закрытие
   заращивает трабекулярные провалы внутри кости.
5. **Отбор компонент.** Берётся крупнейшая компонента; дополнительно
   сохраняются все компоненты крупнее доли от неё — для бедра это
   отдельно лежащие вертлужная впадина и диафиз, их нельзя терять.

Зависимости: только numpy и opencv — модули есть и в контейнере сервиса,
и в окружении разработки. scipy/skimage сознательно не используются.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

# Доля от площади крупнейшей компоненты, начиная с которой компонента
# считается значимой частью кости, а не шумом.
KEEP_COMPONENT_RATIO = 0.10

# Нижняя граница порога кости в перцентилях яркости тела: защита от кадров,
# где Otsu «съезжает» на мягкие ткани.
BONE_PERCENTILE_FLOOR = 70.0

# Мягкий порог гистерезиса как доля пути от порога тела к порогу кости.
HYSTERESIS_LOW = 0.5

# Предел наращивания маски от плотного ядра, в пикселях. На реальных снимках
# рост сам останавливается на границе кости (32 и 64 дают долю кости 0.213 и
# 0.214), предел нужен как страховка от однородных кадров, где мягкий порог
# связен и рост уйти может далеко. Выбор 32 подтверждён качеством правила
# наклона оси: AUC 0.82 при пределе 8 и 0.88 при 32.
GROW_RADIUS = 32


@dataclass
class BoneMask:
    """Маска кости и диагностика, по которой её можно принять или отбраковать."""

    mask: np.ndarray = field(repr=False)          # bool (rows, cols) — итоговая кость
    body: np.ndarray = field(repr=False)          # bool — маска тела
    raw: np.ndarray = field(repr=False)           # bool — до отбора компонент
    body_threshold: float = 0.0
    bone_threshold: float = 0.0
    low_threshold: float = 0.0
    area_px: int = 0
    body_area_px: int = 0
    n_components: int = 0
    n_kept: int = 0
    largest_share: float = 0.0                    # доля крупнейшей компоненты
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)  # r0, c0, r1, c1 (полуоткрытый)
    fill_ratio: float = 0.0                       # площадь маски / площадь bbox
    touches_border: dict[str, bool] = field(default_factory=dict)
    ok: bool = False
    reason: str | None = None

    def stats(self) -> dict[str, Any]:
        r0, c0, r1, c1 = self.bbox
        return {
            "seg_ok": self.ok,
            "seg_reason": self.reason,
            "body_threshold": round(self.body_threshold, 4),
            "bone_threshold": round(self.bone_threshold, 4),
            "low_threshold": round(self.low_threshold, 4),
            "bone_area_px": self.area_px,
            "body_area_px": self.body_area_px,
            "bone_body_ratio": round(self.area_px / self.body_area_px, 4)
            if self.body_area_px
            else 0.0,
            "n_components": self.n_components,
            "n_kept": self.n_kept,
            "largest_share": round(self.largest_share, 4),
            "bbox_r0": r0, "bbox_c0": c0, "bbox_r1": r1, "bbox_c1": c1,
            "bbox_h": r1 - r0, "bbox_w": c1 - c0,
            "fill_ratio": round(self.fill_ratio, 4),
            "touch_top": self.touches_border.get("top", False),
            "touch_bottom": self.touches_border.get("bottom", False),
            "touch_left": self.touches_border.get("left", False),
            "touch_right": self.touches_border.get("right", False),
        }


def _to_u8(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img, dtype=np.float32)
    if a.size and a.max() > 1.5:  # уже в 0..255
        a = a / 255.0
    return np.clip(a * 255.0, 0, 255).astype(np.uint8)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Заливка внутренних дыр без scipy.

    Кадр расширяется рамкой заведомого фона: иначе объект, касающийся угла
    (а бедро касается края кадра почти всегда), станет точкой старта заливки
    и «дырой» окажется весь фон.
    """
    h, w = mask.shape
    padded = np.ones((h + 2, w + 2), np.uint8)       # 1 = фон
    padded[1:-1, 1:-1] = (~mask).astype(np.uint8)
    ff = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(padded, ff, (0, 0), 2)             # внешний фон -> 2
    holes = padded[1:-1, 1:-1] == 1                  # осталось 1 -> замкнутая дыра
    return mask | holes


def _hysteresis(low: np.ndarray, high: np.ndarray, max_grow: int = GROW_RADIUS) -> np.ndarray:
    """Наращивает строгую маску внутри мягкой, но не дальше max_grow пикселей.

    Обычный гистерезис (взять всю связную компоненту мягкой маски) на
    денситометрии рискует «проглотить» весь силуэт: мягкая маска связна и
    покрывает мягкие ткани целиком. Ограниченное наращивание закрывает
    костномозговой канал диафиза (10–15 пикселей) и межпозвонковые
    промежутки, но физически не может уйти от плотной кости дальше, чем на
    max_grow пикселей.
    """
    if not high.any():
        return np.zeros_like(low, dtype=bool)
    cur = high.astype(np.uint8)
    lo = low.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for _ in range(max(max_grow, 0)):
        nxt = cv2.dilate(cur, kernel) & lo
        if np.array_equal(nxt, cur):
            break
        cur = nxt
    return cur.astype(bool)


def _largest_component(mask: np.ndarray) -> tuple[np.ndarray, int, float, int]:
    """Крупнейшая компонента + значимые соседние (для бедра их две-три)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if n <= 1:
        return np.zeros_like(mask, dtype=bool), 0, 0.0, 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(areas)[::-1]
    biggest = areas[order[0]]
    keep = [int(order[0]) + 1]
    for idx in order[1:]:
        if areas[idx] >= biggest * KEEP_COMPONENT_RATIO:
            keep.append(int(idx) + 1)
    out = np.isin(labels, keep)
    share = float(biggest) / float(areas.sum()) if areas.sum() else 0.0
    return out, int(n - 1), share, len(keep)


def _otsu(values: np.ndarray) -> float:
    """Порог Otsu по одномерной выборке значений 0..255."""
    if values.size == 0:
        return 128.0
    hist = np.bincount(values.astype(np.uint8), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 128.0
    omega = np.cumsum(hist) / total
    mu = np.cumsum(hist * np.arange(256)) / total
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = np.where(denom > 0, (mu_t * omega - mu) ** 2 / denom, 0.0)
    return float(np.argmax(sigma_b))


def _otsu3(values: np.ndarray) -> tuple[float, float]:
    """Трёхклассовый Otsu: воздух / мягкие ткани / кость.

    Обычный двухклассовый Otsu на денситометрии отделяет пациента от воздуха,
    а не кость от мягких тканей: мягкие ткани занимают большую часть тела и
    тянут порог вниз (проверено — маска расползается на весь силуэт).
    Трёх классов достаточно и они прямо соответствуют физике снимка.

    Возвращает (порог воздух|ткань, порог ткань|кость) в шкале 0..255.
    """
    if values.size == 0:
        return 32.0, 128.0
    hist = np.bincount(values.astype(np.uint8), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 32.0, 128.0
    p = hist / total
    w = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))

    def _seg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Вклад класса [a, b] в межклассовую дисперсию: mu^2 / w."""
        wa = w[b] - np.where(a > 0, w[np.maximum(a - 1, 0)], 0.0)
        ma = mu[b] - np.where(a > 0, mu[np.maximum(a - 1, 0)], 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(wa > 0, ma * ma / wa, 0.0)

    t1 = np.arange(1, 254)[:, None]          # порог воздух|ткань
    t2 = np.arange(1, 255)[None, :]          # порог ткань|кость
    valid = t2 > t1
    zeros = np.zeros_like(t1 + t2, dtype=np.int64)
    score = (
        _seg(zeros, np.broadcast_to(t1, valid.shape))
        + _seg(np.broadcast_to(t1, valid.shape) + 1, np.broadcast_to(t2, valid.shape))
        + _seg(np.broadcast_to(t2, valid.shape) + 1, np.full(valid.shape, 255))
    )
    score = np.where(valid, score, -np.inf)
    i, j = np.unravel_index(int(np.argmax(score)), score.shape)
    return float(t1[i, 0]), float(t2[0, j])


def segment_bone(
    img: np.ndarray,
    *,
    open_radius: int = 1,
    close_radius: int = 3,
    grow_radius: int = GROW_RADIUS,
    min_body_fraction: float = 0.02,
) -> BoneMask:
    """Строит маску кости. Не бросает исключений: при отказе ok=False и reason."""
    empty = np.zeros(np.asarray(img).shape[:2], dtype=bool)
    res = BoneMask(mask=empty, body=empty.copy(), raw=empty.copy())

    a = np.asarray(img)
    if a.ndim != 2 or a.size == 0:
        res.reason = f"ожидался двумерный кадр, получено {a.shape}"
        return res
    if min(a.shape) < 8:
        res.reason = f"кадр слишком мал: {a.shape}"
        return res

    u8 = _to_u8(a)
    if int(u8.max()) - int(u8.min()) < 2:
        res.reason = "кадр однороден: контраста нет, сегментировать нечего"
        return res

    # --- 1. тело против воздуха -------------------------------------------- #
    t_body, t_bone_global = _otsu3(u8.ravel())
    body = u8 > t_body
    if body.mean() < min_body_fraction:
        res.reason = f"тело не найдено: доля пикселей выше порога {body.mean():.4f}"
        res.body_threshold = t_body / 255.0
        return res
    body_cc, _, _, _ = _largest_component(body)
    body = _fill_holes(body_cc) if body_cc.any() else body
    res.body = body
    res.body_threshold = t_body / 255.0
    res.body_area_px = int(body.sum())

    # --- 2. кость против мягких тканей ------------------------------------- #
    inside = u8[body]
    t_bone = max(t_bone_global, float(np.percentile(inside, BONE_PERCENTILE_FLOOR)))
    # Эндопротез насыщает шкалу: у двух исследований набора металл занимает
    # больше 30% площади тела, перцентильный пол уезжает в 255 и маска
    # обнуляется. Порог не имеет права быть выше максимума кадра.
    t_bone = min(t_bone, float(inside.max()) - 1.0)
    res.bone_threshold = t_bone / 255.0

    # Гистерезис. Один порог по кости оставляет только кортикальный слой:
    # костномозговой канал диафиза и межпозвонковые промежутки темнее и
    # выпадают, маска рассыпается на фрагменты. Поэтому «ядро» кости берётся
    # по строгому порогу, а достраивается оно связными областями мягкого
    # порога — так контур совпадает с границей кости, а не с её плотным центром.
    t_low = t_body + HYSTERESIS_LOW * (t_bone - t_body)
    res.low_threshold = t_low / 255.0
    high = body & (u8 > t_bone)
    low = body & (u8 > t_low)
    bone = _hysteresis(low, high, max_grow=grow_radius)
    res.raw = bone.copy()

    if not bone.any():
        res.reason = "маска кости пуста после порога"
        return res

    # --- 3. морфология ------------------------------------------------------ #
    m = bone.astype(np.uint8)
    if open_radius > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_radius + 1,) * 2)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    if close_radius > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_radius + 1,) * 2)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    bone = m.astype(bool)

    if not bone.any():
        res.reason = "маска кости пуста после морфологии"
        return res

    # --- 4. отбор компонент ------------------------------------------------- #
    kept, n_comp, share, n_kept = _largest_component(bone)
    kept = _fill_holes(kept)
    res.mask = kept
    res.n_components = n_comp
    res.n_kept = n_kept
    res.largest_share = share
    res.area_px = int(kept.sum())

    if res.area_px == 0:
        res.reason = "не осталось ни одной компоненты"
        return res

    rr, cc = np.nonzero(kept)
    r0, r1 = int(rr.min()), int(rr.max()) + 1
    c0, c1 = int(cc.min()), int(cc.max()) + 1
    res.bbox = (r0, c0, r1, c1)
    box_area = (r1 - r0) * (c1 - c0)
    res.fill_ratio = float(res.area_px / box_area) if box_area else 0.0
    h, w = kept.shape
    res.touches_border = {
        "top": r0 == 0,
        "bottom": r1 == h,
        "left": c0 == 0,
        "right": c1 == w,
    }
    res.ok = True
    return res


def principal_axis(mask: np.ndarray) -> tuple[float, tuple[float, float]]:
    """Угол главной оси маски к вертикали в градусах и центр масс.

    Знак: положительный — верх наклонён вправо. Используется для правила
    «наклон оси позвоночника более 5°» (ТЗ 2.3.1).
    """
    rr, cc = np.nonzero(mask)
    if rr.size < 2:
        return float("nan"), (float("nan"), float("nan"))
    r_mean, c_mean = float(rr.mean()), float(cc.mean())
    r = rr - r_mean
    c = cc - c_mean
    cov = np.array([[float((c * c).mean()), float((c * r).mean())],
                    [float((c * r).mean()), float((r * r).mean())]])
    vals, vecs = np.linalg.eigh(cov)
    vx, vy = vecs[:, int(np.argmax(vals))]  # главное направление (x=col, y=row)
    if vy < 0:  # ориентируем вектор сверху вниз
        vx, vy = -vx, -vy
    angle = float(np.degrees(np.arctan2(vx, vy)))
    return angle, (r_mean, c_mean)


def overlay(img: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha: float = 0.35,
            contour: bool = True) -> np.ndarray:
    """RGB-визуализация маски поверх снимка — для контрольных листов и ТЗ 2.6."""
    base = cv2.cvtColor(_to_u8(img), cv2.COLOR_GRAY2RGB)
    if mask is None or not mask.any():
        return base
    if contour:
        cnts, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(base, cnts, -1, color, 1)
        return base
    tint = base.copy()
    tint[mask] = color
    return cv2.addWeighted(tint, alpha, base, 1 - alpha, 0)
