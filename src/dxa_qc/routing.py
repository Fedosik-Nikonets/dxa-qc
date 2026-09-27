"""Определение анатомической области изображения на инференсе.

На обучении область определялась в `scripts/02_build_dataset.py`, где состав
областей был известен из `разметка.xlsx` и использовался как подсказка
(«в этом исследовании есть позвоночник»). На инференсе разметки нет, поэтому
здесь то же правило работает только по геометрии.

Разделяющий признак — латеральность `lat_score`: разность средней координаты
светлых пикселей в верхней и нижней третях кадра. У позвоночника кадр
центрирован и |lat| мал (медиана 10.7), у бедра таз смещён в сторону, а
диафиз идёт к центру, и |lat| велик (медиана 74). Но диапазоны
соприкасаются: позвоночник доходит до 56.7, бедро начинается с 45.3, так что
одной латеральности мало. Знак
`lat_score` даёт сторону: + правое бедро, - левое (калибровано по файлам
«Для теста.zip» с явными именами ППОБ/ЛПОБ).

Симметрия `sym` как основной признак не годится: у позвоночника при сколиозе
или узком кропе она падает до 0.018, у бедра доходит до 0.369. В трёх
исследованиях правило «самый симметричный снимок — позвоночник» перепутало
позвоночник и бедро (см. reports/day2_findings.md). Но как *вторая ветвь*
она закрывает именно те позвоночники, которые проваливают правило по |lat|:
у всех трёх таких кадров обучающего набора sym не ниже 0.481.

Модуль временный по назначению, но не по качеству: напарник делает
классификатор области на 3 класса, и когда он будет готов, `route_study`
станет резервным путём при низкой уверенности классификатора.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Порог яркости, выше которого пиксель считается «телом». Тот же, что в
# инвентаризации (`frac_bright`) и в сборке датасета.
BRIGHT = 0.35

# Граница по |lat_score| между позвоночником и бедром. Классы в абсолютных
# величинах СОПРИКАСАЮТСЯ: на обучающем наборе позвоночник доходит до 56.7,
# а бедро начинается с 45.3, поэтому одного этого признака мало.
LAT_SPLIT = 41.5

# Вторая ветвь правила: зеркальная симметрия. Позвоночники, у которых |lat|
# велик (косая укладка, сколиоз), всё равно симметричны — у них sym не ниже
# 0.481, тогда как у бедра sym не поднимается выше 0.369.
#
# Итоговое правило «позвоночник, если |lat| < 41.5 ИЛИ sym > 0.42» разделяет
# все 252 изображения обучающего набора без единой ошибки. Каждый порог стоит
# в середине реального зазора: по |lat| между 38.1 и 45.3, по sym между
# 0.369 и 0.481. Проверка: scripts/05_router_check.py.
SYM_SPINE = 0.42

# Запас до границы, ниже которого решение помечается как неуверенное.
MARGIN_LAT = 3.0
MARGIN_SYM = 0.05

# Минимальный отрыв позвоночника от ближайшего бедра внутри исследования.
# Внутри исследования работает не абсолютная величина, а зазор, и он
# надёжнее: у косых укладок позвоночник доходит до |lat| = 57, но ближайшее
# бедро всё равно дальше.
LAT_MARGIN = 20.0

SPINE = "spine"
FEMUR_LEFT = "femur_left"
FEMUR_RIGHT = "femur_right"
UNKNOWN = "unknown"

REGIONS = (SPINE, FEMUR_LEFT, FEMUR_RIGHT)

# Имена областей в выходной таблице. Коды — из taxonomy.yaml.
REGION_CODE = {
    SPINE: "LUMBAR_SPINE",
    FEMUR_LEFT: "LEFT_PROXIMAL_FEMUR",
    FEMUR_RIGHT: "RIGHT_PROXIMAL_FEMUR",
    UNKNOWN: "UNKNOWN",
}

REGION_RU = {
    SPINE: "Поясничный отдел позвоночника",
    FEMUR_LEFT: "Проксимальный отдел левого бедра",
    FEMUR_RIGHT: "Проксимальный отдел правого бедра",
    UNKNOWN: "Не определена",
}


@dataclass
class RegionFeatures:
    """Признаки одного изображения для роутера."""

    sym: float = float("nan")
    lat_score: float = float("nan")

    @property
    def valid(self) -> bool:
        return bool(np.isfinite(self.lat_score))


@dataclass
class RegionDecision:
    """Решение роутера по одному изображению."""

    region: str = UNKNOWN
    confident: bool = False
    reason: str | None = None


def image_features(pixels: np.ndarray | None) -> RegionFeatures:
    """Считает признаки области. Исключений не выбрасывает.

    `pixels` — нормированное изображение из `dicom_io.read_dicom` (float в
    [0, 1], MONOCHROME1 уже инвертирован).
    """
    if pixels is None:
        return RegionFeatures()
    try:
        a = np.asarray(pixels, dtype=np.float64)
        if a.ndim != 2 or a.size == 0 or a.shape[0] < 3 or a.shape[1] < 2:
            return RegionFeatures()
        if not np.isfinite(a).all():
            a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)

        flipped = np.fliplr(a)
        if a.std() == 0 or flipped.std() == 0:
            sym = float("nan")
        else:
            sym = float(np.corrcoef(a.ravel(), flipped.ravel())[0, 1])

        m = a > BRIGHT
        h = m.shape[0]
        up = np.nonzero(m[: h // 3])[1]
        lo = np.nonzero(m[2 * h // 3 :])[1]
        lat = float(up.mean() - lo.mean()) if len(up) and len(lo) else float("nan")
        return RegionFeatures(sym=sym, lat_score=lat)
    except Exception:  # noqa: BLE001 — роутер не имеет права уронить конвейер
        return RegionFeatures()


def route_study(feats: list[RegionFeatures]) -> list[RegionDecision]:
    """Назначает область каждому уникальному изображению исследования.

    На вход — признаки уникальных (дедуплицированных) изображений одного
    исследования. Порядок сохраняется.
    """
    n = len(feats)
    out = [RegionDecision() for _ in range(n)]
    if n == 0:
        return out

    idx_valid = [i for i, f in enumerate(feats) if f.valid]
    for i in range(n):
        if i not in idx_valid:
            out[i].reason = "пустая или вырожденная маска тела: lat_score не вычислим"

    if not idx_valid:
        return out

    abs_lat = {i: abs(feats[i].lat_score) for i in idx_valid}
    spine_like = {i: is_spine_like(feats[i]) for i in idx_valid}
    margin_ok = {i: spine_margin_ok(feats[i]) for i in idx_valid}

    # --- шаг 1. есть ли в исследовании позвоночник ------------------------- #
    # Позвоночник в исследовании максимум один: из всех «похожих на
    # позвоночник» кадров берём самый центрированный.
    cands = [i for i in idx_valid if spine_like[i]]
    top = min(cands, key=lambda i: abs_lat[i]) if cands else None

    if top is not None:
        others = [i for i in idx_valid if i != top]
        if others:
            gap = min(abs_lat[i] for i in others) - abs_lat[top]
            confident = gap > LAT_MARGIN and len(cands) == 1
            reason = None if confident else (
                f"на позвоночник похожи {len(cands)} кадра исследования"
                if len(cands) > 1 else
                f"отрыв от ближайшего бедра {gap:.1f} < {LAT_MARGIN}"
            )
        else:
            confident = margin_ok[top]
            reason = None if confident else "одиночный кадр у самой границы правила"
        out[top].region = SPINE
        out[top].confident = confident
        out[top].reason = reason

    # --- шаг 2. сторона бедра по знаку латеральности ----------------------- #
    for i in idx_valid:
        if i == top:
            continue
        out[i].region = FEMUR_RIGHT if feats[i].lat_score >= 0 else FEMUR_LEFT
        out[i].confident = margin_ok[i]
        if not out[i].confident:
            out[i].reason = "бедро у самой границы правила по |lat| и sym"

    # --- шаг 3. дубли областей внутри исследования ------------------------- #
    # Два «правых» бедра в одном исследовании — признак ошибки роутера.
    seen: dict[str, int] = {}
    for i in idx_valid:
        seen[out[i].region] = seen.get(out[i].region, 0) + 1
    for i in idx_valid:
        if seen.get(out[i].region, 0) > 1:
            out[i].confident = False
            out[i].reason = f"область {out[i].region} назначена более одного раза"

    return out


def is_spine_like(f: RegionFeatures) -> bool:
    """Правило «кадр похож на позвоночник»: центрирован ИЛИ симметричен."""
    if not f.valid:
        return False
    lat_ok = abs(f.lat_score) < LAT_SPLIT
    sym_ok = bool(np.isfinite(f.sym)) and f.sym > SYM_SPINE
    return lat_ok or sym_ok


def spine_margin_ok(f: RegionFeatures) -> bool:
    """Решение по этому кадру принято с запасом, а не впритык к порогам."""
    if not f.valid:
        return False
    lat = abs(f.lat_score)
    sym = f.sym if np.isfinite(f.sym) else -1.0
    if is_spine_like(f):
        return lat < LAT_SPLIT - MARGIN_LAT or sym > SYM_SPINE + MARGIN_SYM
    return lat > LAT_SPLIT + MARGIN_LAT and sym < SYM_SPINE - MARGIN_SYM


def region_name(region: str, lang: str = "code") -> str:
    """Имя области для выходной таблицы."""
    table = REGION_RU if lang == "ru" else REGION_CODE
    return table.get(region, table[UNKNOWN])


# --------------------------------------------------------------------------- #
# проекция (ТЗ 2.2: «определение анатомической области и проекции»)
# --------------------------------------------------------------------------- #
# Денситометрия поясничного отдела и проксимального отдела бедра по
# протоколу снимается в прямой проекции (GE Lunar: «AP Spine», «Femur»).
# Если в файле есть ViewPosition (0018,5101), берётся он; в обучающем наборе
# тега нет ни в одном из 499 файлов, и тогда проекция — прямая по протоколу,
# что честно помечается в диагностике (`projection_source=protocol`).
PROJECTION_AP, PROJECTION_PA, PROJECTION_LAT, PROJECTION_OTHER = "AP", "PA", "LAT", "OTHER"
LATERAL_VIEWS = {"LL", "RL", "LAT", "LATERAL", "LLD", "RLD"}
PROJECTION_RU = {
    PROJECTION_AP: "прямая (AP)",
    PROJECTION_PA: "прямая (PA)",
    PROJECTION_LAT: "боковая (LAT)",
    PROJECTION_OTHER: "не распознана",
}


def projection(region: str, view_position: str | None) -> tuple[str, str]:
    """Код проекции и его источник: `tag`, `protocol` или пусто для неизвестной области."""
    if region not in REGIONS:
        return "", ""
    v = (view_position or "").strip().upper()
    if not v:
        return PROJECTION_AP, "protocol"
    if v in (PROJECTION_AP, PROJECTION_PA):
        return v, "tag"
    if v in LATERAL_VIEWS:
        return PROJECTION_LAT, "tag"
    return PROJECTION_OTHER, "tag"


def projection_name(code: str, lang: str = "code") -> str:
    """Имя проекции для выходной таблицы; язык — как у области."""
    if not code:
        return ""
    return PROJECTION_RU.get(code, code) if lang == "ru" else code
