"""Текстовое описание нарушений в формате DICOM SR (ТЗ 2.6).

Один документ на исследование — Basic Text SR
(1.2.840.10008.5.1.4.1.1.88.11), в том же StudyInstanceUID, что и снимки,
отдельной серией. Документ читается любым DICOM-просмотрщиком с поддержкой
SR и ложится в PACS рядом с исходниками и overlay-серией.

Дерево содержимого
------------------
    CONTAINER «Контроль качества денситометрии (DXA QC)»
      TEXT      «Заключение по исследованию»
      TEXT      «Балл нарушения исследования»        (если есть)
      CONTAINER «Снимок» — на каждый уникальный кадр
        TEXT    «Анатомическая область»
        TEXT    «Заключение»                          нарушения словами ТЗ 2.3
        TEXT    «Измерения»                           число против порога
        TEXT    «Балл нарушения»                      (если есть)
        IMAGE   «Исходный снимок»                     ссылка на каждый файл кадра

Точные копии кадра (в обучающем наборе их половина) описываются один раз, а
ссылки IMAGE идут на все файлы с этим кадром.
      TEXT      «Сведения о программе»

Коды концептов — из частной схемы `99DXAQC` (префикс 99 — локальная схема
по PS3.16). Ссылки IMAGE перечислены в Current Requested Procedure Evidence
Sequence, как требует модуль SR Document General.

Запись — без pydicom, теми же функциями, что и overlay (`dicom_sc`); UID
детерминированы, поэтому документ побайтово воспроизводим. Дата и время
содержимого (тип 1) — дата и время исследования из исходника, а если они
обезличены (в обучающем наборе вместо них «Anonymized») — фиксированные
2026-01-01 00:00: время создания в файл не пишется ради воспроизводимости.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import __version__
from .dicom_sc import (EXPLICIT_VR_LE, IMPLEMENTATION_UID, MANUFACTURER, _dataset, clean,
                       element, sequence)
from .femur import FIELD_MIN_MM
from .spine import AXIS_TILT_DEG, COVERAGE_MIN

SR_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.88.11"   # Basic Text SR Storage
SERIES_NUMBER = 9002
SERIES_DESCRIPTION = "DXA QC report"
SCHEME = "99DXAQC"
FIXED_DATE, FIXED_TIME = "20260101", "000000"

CODES = {
    "root": ("DXAQC-000", "Контроль качества денситометрии (DXA QC)"),
    "study_conclusion": ("DXAQC-001", "Заключение по исследованию"),
    "study_score": ("DXAQC-002", "Балл нарушения исследования"),
    "software": ("DXAQC-003", "Сведения о программе"),
    "image": ("DXAQC-010", "Снимок"),
    "region": ("DXAQC-011", "Анатомическая область"),
    "conclusion": ("DXAQC-012", "Заключение"),
    "measurements": ("DXAQC-013", "Измерения"),
    "score": ("DXAQC-014", "Балл нарушения"),
    "source": ("DXAQC-015", "Исходный снимок"),
}

# Нарушения словами ТЗ 2.3.
VIOLATION_TEXT = {
    "SPINE_POSITIONING": "некорректная укладка: в кадре нет верхних краёв подвздошных костей",
    "SPINE_AXIS_TILT": "наклон оси позвоночника больше допустимого",
    "SPINE_ARTIFACT": "посторонние предметы, артефакты или наложения",
    "FEMUR_POSITIONING_ROTATION": "некорректное позиционирование бедра, ротация (оценка малого вертела)",
    "FEMUR_ROI_COVERAGE": "область интереса охвачена не полностью",
}

SOFTWARE_NOTE = ("DXA QC {version}: автоматическая проверка качества выполнения денситометрии "
                 "по критериям ТЗ 2.3. Результат — подсказка специалисту, а не диагноз; "
                 "документ не верифицирован врачом.")


@dataclass
class ImageFinding:
    """Что известно о снимке к моменту записи отчёта."""

    path: str
    region_key: str
    region_name: str
    quality_class: int
    violations: list[str]
    diag: dict[str, Any] = field(default_factory=dict)
    score: float | None = None
    # Ссылки на файлы кадра: (SOP Class UID, SOP Instance UID, Series UID, Study UID).
    refs: list[tuple[str, str, str, str]] = field(default_factory=list)

    def valid_refs(self) -> list[tuple[str, str, str, str]]:
        return sorted({r for r in self.refs if all(r)})


def _num(diag: dict, key: str) -> float:
    try:
        return float(diag.get(key))
    except (TypeError, ValueError):
        return float("nan")


def measurements(f: ImageFinding, artifact_threshold: float | None = None) -> str:
    """Измерения против порогов — те же числа, что на overlay и в diagnostics.csv."""
    d, out = f.diag, []
    if f.region_key == "spine":
        a = _num(d, "spine_angle_deg")
        if a == a:
            out.append(f"наклон оси позвоночника {abs(a):.1f}° (порог {AXIS_TILT_DEG:g}°)")
        c = _num(d, "spine_coverage")
        if c == c:
            out.append(f"плотность сбоку от позвоночника в нижней трети кадра {c:.3f} "
                       f"(гребни подвздошных костей в кадре, если не ниже {COVERAGE_MIN:g})")
        p = _num(d, "artifact_prob")
        if p == p:
            thr = f" при пороге {artifact_threshold:.2f}" if artifact_threshold is not None else ""
            out.append(f"вероятность артефакта или наложения {p:.2f}{thr}")
    elif f.region_key in ("femur_left", "femur_right"):
        h = _num(d, "femur_field_h_mm")
        if h == h:
            out.append(f"длина поля сканирования {h:.0f} мм (не менее {FIELD_MIN_MM:g} мм)")
        v, lo, hi = _num(d, "rotation_value"), _num(d, "rotation_low"), _num(d, "rotation_high")
        if v == v and lo == lo and hi == hi:
            out.append(f"выступ малого вертела {v:.2f} мм (норма от {lo:.2f} до {hi:.2f} мм)")
        elif d.get("rotation_reason"):
            out.append(f"ротация не измерена: {d.get('rotation_reason')}")
    return "; ".join(out) if out else "измерения недоступны"


def conclusion(f: ImageFinding) -> str:
    if not f.violations:
        return "нарушений качества не выявлено"
    return "выявлены нарушения: " + "; ".join(VIOLATION_TEXT.get(v, v) for v in f.violations)


def plural(n: int, one: str, few: str, many: str) -> str:
    """Форма слова после числа: 1 снимок, 2 снимка, 5 снимков."""
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many


def study_conclusion(findings: list[ImageFinding], failed: int = 0) -> str:
    bad = [f for f in findings if f.quality_class == 1]
    n = len(findings)
    files = sum(max(len(f.refs), 1) for f in findings)
    in_files = f" ({files} {plural(files, 'файл', 'файла', 'файлов')})" if files != n else ""
    tail = (f" Не обработано файлов: {failed}, причина — в таблице результатов." if failed else "")
    if not bad:
        return (f"Нарушений качества не выявлено: {n} {plural(n, 'снимок', 'снимка', 'снимков')}"
                f"{in_files}." + tail)
    kinds = []
    for f in bad:
        for v in f.violations:
            t = VIOLATION_TEXT.get(v, v)
            if t not in kinds:
                kinds.append(t)
    return (f"Есть нарушения качества на {len(bad)} из {n} {'снимка' if n == 1 else 'снимков'}"
            f"{in_files}: " + "; ".join(kinds)
            + ". Рекомендуется проверить укладку и при необходимости повторить исследование." + tail)


# --------------------------------------------------------------------------- #
# кодирование содержимого
# --------------------------------------------------------------------------- #
def _code(key: str) -> bytes:
    value, meaning = CODES[key]
    return (element((0x0008, 0x0100), "SH", value)
            + element((0x0008, 0x0102), "SH", SCHEME)
            + element((0x0008, 0x0104), "LO", clean("LO", meaning)))


def _item(value_type: str, key: str, *, text: str | None = None,
          children: list[bytes] | None = None, ref: tuple[str, str] | None = None) -> bytes:
    """Элемент дерева SR, отношение CONTAINS; теги — по возрастанию."""
    parts: list[tuple[tuple[int, int], bytes]] = []
    if ref is not None:
        refitem = (element((0x0008, 0x1150), "UI", clean("UI", ref[0]))
                   + element((0x0008, 0x1155), "UI", clean("UI", ref[1])))
        parts.append(((0x0008, 0x1199), sequence((0x0008, 0x1199), [refitem])))
    parts.append(((0x0040, 0xA010), element((0x0040, 0xA010), "CS", "CONTAINS")))
    parts.append(((0x0040, 0xA040), element((0x0040, 0xA040), "CS", value_type)))
    parts.append(((0x0040, 0xA043), sequence((0x0040, 0xA043), [_code(key)])))
    if value_type == "CONTAINER":
        parts.append(((0x0040, 0xA050), element((0x0040, 0xA050), "CS", "SEPARATE")))
    if text is not None:
        parts.append(((0x0040, 0xA160), element((0x0040, 0xA160), "UT", text)))
    if children:
        parts.append(((0x0040, 0xA730), sequence((0x0040, 0xA730), children)))
    return _dataset(parts)


def _fmt_score(s: float | None) -> str:
    return f"{s:.4f} (больше 0.5 — есть нарушение)"


def _evidence(findings: list[ImageFinding]) -> bytes | None:
    """Current Requested Procedure Evidence Sequence: все снимки, на которые ссылается отчёт."""
    studies: dict[str, dict[str, list[tuple[str, str]]]] = {}
    for f in findings:
        for sop_class, sop_uid, series_uid, study_uid in f.valid_refs():
            studies.setdefault(study_uid, {}).setdefault(series_uid, []).append((sop_class, sop_uid))
    if not studies:
        return None
    study_items = []
    for st_uid in sorted(studies):
        series_items = []
        for se_uid in sorted(studies[st_uid]):
            refs = [element((0x0008, 0x1150), "UI", clean("UI", c)) + element((0x0008, 0x1155), "UI", clean("UI", u))
                    for c, u in sorted(set(studies[st_uid][se_uid]))]
            series_items.append(sequence((0x0008, 0x1199), refs)
                                + element((0x0020, 0x000E), "UI", clean("UI", se_uid)))
        study_items.append(sequence((0x0008, 0x1115), series_items)
                           + element((0x0020, 0x000D), "UI", clean("UI", st_uid)))
    return sequence((0x0040, 0xA375), study_items)


def basic_text_sr(findings: list[ImageFinding], *, study_uid: str, series_uid: str, sop_uid: str,
                  context: dict[str, str] | None = None, study_score: float | None = None,
                  artifact_threshold: float | None = None, failed: int = 0) -> bytes:
    """Файл DICOM (Part 10) с отчётом по исследованию."""
    if not findings:
        raise ValueError("отчёт без снимков не пишется")
    ctx = context or {}
    content: list[bytes] = [_item("TEXT", "study_conclusion", text=study_conclusion(findings, failed))]
    if study_score is not None:
        content.append(_item("TEXT", "study_score", text=_fmt_score(study_score)))
    for f in sorted(findings, key=lambda x: x.path):
        kids = [_item("TEXT", "region", text=f.region_name or f.region_key),
                _item("TEXT", "conclusion", text=conclusion(f)),
                _item("TEXT", "measurements", text=measurements(f, artifact_threshold))]
        if f.score is not None:
            kids.append(_item("TEXT", "score", text=_fmt_score(f.score)))
        for sop_class, sop_uid, _, _ in f.valid_refs():
            kids.append(_item("IMAGE", "source", ref=(sop_class, sop_uid)))
        content.append(_item("CONTAINER", "image", text=None, children=kids))
    content.append(_item("TEXT", "software", text=SOFTWARE_NOTE.format(version=__version__)))

    date = clean("DA", ctx.get("study_date")) or FIXED_DATE
    time = clean("TM", ctx.get("study_time")) or FIXED_TIME

    ds: list[tuple[tuple[int, int], bytes]] = []

    def add(tag, vr, value):
        ds.append((tag, element(tag, vr, value)))

    add((0x0008, 0x0005), "CS", "ISO_IR 192")
    add((0x0008, 0x0016), "UI", SR_SOP_CLASS)
    add((0x0008, 0x0018), "UI", sop_uid)
    add((0x0008, 0x0020), "DA", clean("DA", ctx.get("study_date")))
    add((0x0008, 0x0023), "DA", date)
    add((0x0008, 0x0030), "TM", clean("TM", ctx.get("study_time")))
    add((0x0008, 0x0033), "TM", time)
    add((0x0008, 0x0050), "SH", clean("SH", ctx.get("accession_number")))
    add((0x0008, 0x0060), "CS", "SR")
    add((0x0008, 0x0070), "LO", MANUFACTURER)
    add((0x0008, 0x0090), "PN", clean("PN", ctx.get("referring_physician")))
    add((0x0008, 0x103E), "LO", clean("LO", SERIES_DESCRIPTION))
    ds.append(((0x0008, 0x1111), sequence((0x0008, 0x1111), [])))
    add((0x0010, 0x0010), "PN", clean("PN", ctx.get("patient_name")))
    add((0x0010, 0x0020), "LO", clean("LO", ctx.get("patient_id")))
    add((0x0010, 0x0030), "DA", clean("DA", ctx.get("patient_birth_date")))
    add((0x0010, 0x0040), "CS", clean("CS", ctx.get("patient_sex")))
    add((0x0018, 0x1020), "LO", __version__)
    add((0x0020, 0x000D), "UI", study_uid)
    add((0x0020, 0x000E), "UI", series_uid)
    add((0x0020, 0x0010), "SH", clean("SH", ctx.get("study_id")))
    add((0x0020, 0x0011), "IS", str(SERIES_NUMBER))
    add((0x0020, 0x0013), "IS", "1")
    add((0x0040, 0xA040), "CS", "CONTAINER")
    ds.append(((0x0040, 0xA043), sequence((0x0040, 0xA043), [_code("root")])))
    add((0x0040, 0xA050), "CS", "SEPARATE")
    ds.append(((0x0040, 0xA372), sequence((0x0040, 0xA372), [])))
    ev = _evidence(findings)
    if ev is not None:
        ds.append(((0x0040, 0xA375), ev))
    add((0x0040, 0xA491), "CS", "COMPLETE")
    add((0x0040, 0xA493), "CS", "UNVERIFIED")
    ds.append(((0x0040, 0xA730), sequence((0x0040, 0xA730), content)))
    body = _dataset(ds)

    meta = (element((0x0002, 0x0001), "OB", b"\x00\x01")
            + element((0x0002, 0x0002), "UI", SR_SOP_CLASS)
            + element((0x0002, 0x0003), "UI", sop_uid)
            + element((0x0002, 0x0010), "UI", EXPLICIT_VR_LE)
            + element((0x0002, 0x0012), "UI", IMPLEMENTATION_UID)
            + element((0x0002, 0x0013), "SH", clean("SH", f"DXA_QC_{__version__}")))
    group_len = element((0x0002, 0x0000), "UL", len(meta))
    return b"\x00" * 128 + b"DICM" + group_len + meta + body
