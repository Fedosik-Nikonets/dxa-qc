"""Overlay-серия (ТЗ 2.6, 2.7): визуализация найденного нарушения на снимке.

Одна PNG-картинка на каждый файл исследования, все вместе — в zip-архиве
рядом с таблицей (ТЗ 2.7: «zip-архив с дополнительными сериями»). Рисуется
ровно то, по чему принято решение, а не иллюстрация «по мотивам»:

* позвоночник — контур маски кости; главная ось маски и вертикаль (наклон
  оси, ТЗ 2.3.1); граница нижней полосы, где мерится плотность сбоку от
  позвоночника (укладка); тепловая карта отклика top-hat вне позвоночника —
  **только если** артефакт найден, иначе она шум;
* бедро — контур маски; рамка поля сканирования с его длиной (охват
  области интереса, ТЗ 2.3.2) и справочно — боковой отступ против 2 см
  из ТЗ (решения не меняет, README п. 6); прослеженный медиальный край и
  точка выступа в окне малого вертела (ротация).

Красное — сработавшее правило, жёлтое/голубое — измерение без нарушения.
Подписи латиницей: шрифт OpenCV кириллицу не рисует, а тянуть шрифт в
образ ради подписей не стоит. Тепловая карта — не Grad-CAM: CNN в решении
нет, и показывать карту чужой модели было бы неправдой.

Оверлей — только визуализация. Его сбой не влияет на таблицу: картинка
не пишется, причина попадает в `overlay_errors.txt` внутри архива.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from . import artifact as artifact_rules
from .femur import scan_field
from .medial_track import track_medial_edge
from .routing import FEMUR_LEFT, FEMUR_RIGHT, SPINE
from .segmentation import principal_axis, segment_bone
from .spine import LOWER_FRACTION

SCALE = 2
LINE_H = 18
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.42
ZIP_DATE = (2026, 1, 1, 0, 0, 0)

# BGR
RED = (40, 40, 230)
GREEN = (80, 200, 80)
YELLOW = (40, 220, 230)
CYAN = (230, 210, 60)
MAGENTA = (220, 80, 220)
GRAY = (150, 150, 150)
WHITE = (240, 240, 240)

TITLES = {SPINE: "LUMBAR SPINE", FEMUR_LEFT: "LEFT PROXIMAL FEMUR", FEMUR_RIGHT: "RIGHT PROXIMAL FEMUR"}


def _gray(pixels: np.ndarray) -> np.ndarray:
    a = np.asarray(pixels, dtype=np.float32)
    lo, hi = np.percentile(a, [0.5, 99.5])
    u8 = np.clip((a - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    big = cv2.resize(u8, (u8.shape[1] * SCALE, u8.shape[0] * SCALE), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)


def _pt(x: float, y: float) -> tuple[int, int]:
    return int(round((x + 0.5) * SCALE - 0.5)), int(round((y + 0.5) * SCALE - 0.5))


def _contour(canvas: np.ndarray, mask: np.ndarray, color) -> None:
    if mask is None or not mask.any():
        return
    big = cv2.resize(mask.astype(np.uint8), (mask.shape[1] * SCALE, mask.shape[0] * SCALE),
                     interpolation=cv2.INTER_NEAREST)
    cnts, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, cnts, -1, color, 1, lineType=cv2.LINE_AA)


def _num(diag: dict, key: str) -> float:
    try:
        return float(diag.get(key, ""))
    except (TypeError, ValueError):
        return float("nan")


def _spine(canvas, pixels, mm, violations, diag, lines) -> None:
    bone = segment_bone(pixels)
    if not bone.ok:
        lines.append((f"bone mask: {bone.reason or 'failed'}", GRAY))
        return
    _contour(canvas, bone.mask, GREEN)
    h, w = bone.mask.shape

    # Наклон оси: главная ось маски против вертикали через тот же центр.
    angle, (r0, c0) = principal_axis(bone.mask)
    if angle == angle:
        rows = np.flatnonzero(bone.mask.any(axis=1))
        half = (rows.max() - rows.min()) / 2
        d = np.array([np.sin(np.radians(angle)), np.cos(np.radians(angle))])
        tilt = "SPINE_AXIS_TILT" in violations
        cv2.line(canvas, _pt(c0, r0 - half), _pt(c0, r0 + half), GRAY, 1, cv2.LINE_AA)
        cv2.line(canvas, _pt(c0 - d[0] * half, r0 - d[1] * half),
                 _pt(c0 + d[0] * half, r0 + d[1] * half), RED if tilt else YELLOW, 2, cv2.LINE_AA)
        lines.append((f"axis tilt {abs(angle):.1f} deg" + (" > limit" if tilt else ""),
                      RED if tilt else YELLOW))

    # Укладка: нижняя полоса кадра, где мерится плотность сбоку.
    pos = "SPINE_POSITIONING" in violations
    y = int(h * (1.0 - LOWER_FRACTION))
    cv2.line(canvas, _pt(0, y), _pt(w - 1, y), RED if pos else GRAY, 1, cv2.LINE_AA)
    cov = _num(diag, "spine_coverage")
    lines.append((f"lateral density in lower band {cov:.3f}" + (" - iliac crests not in field" if pos else ""),
                  RED if pos else GRAY))

    # Артефакт: карта отклика, по которому принято решение.
    if "SPINE_ARTIFACT" in violations:
        heat = artifact_rules.tophat_map(pixels, bone.mask, mm)
        if heat is not None and heat.max() > 0:
            norm = np.clip(heat / heat.max(), 0, 1)
            big = cv2.resize(norm, (w * SCALE, h * SCALE), interpolation=cv2.INTER_LINEAR)
            color = cv2.applyColorMap((big * 255).astype(np.uint8), cv2.COLORMAP_JET)
            hot = big > 0.5   # только сильный отклик: слабый — края газа и мягких тканей
            canvas[hot] = (0.45 * canvas[hot] + 0.55 * color[hot]).astype(np.uint8)
            yy, xx = np.unravel_index(int(np.argmax(heat)), heat.shape)
            cv2.circle(canvas, _pt(xx, yy), 9 * SCALE // 2, RED, 2, cv2.LINE_AA)
        # Подпись нейтральная: на обучении 9 из 12 верных срабатываний — наложение
        # рёбер, а не инородное тело (reports/day11_errors_artifact.md).
        lines.append((f"artifact / overlap (ribs, foreign body), p={_num(diag, 'artifact_prob'):.2f}", RED))


def _femur(canvas, pixels, region, mm, source, model, violations, diag, lines) -> None:
    bone = segment_bone(pixels)
    if bone.ok:
        _contour(canvas, bone.mask, GREEN)
    h, w = np.asarray(pixels).shape

    # Охват: поле сканирования и его длина.
    roi = "FEMUR_ROI_COVERAGE" in violations
    c0, c1 = scan_field(pixels)
    cv2.rectangle(canvas, _pt(c0, 0), _pt(c1 - 1, h - 1), RED if roi else GRAY, 1)
    field = _num(diag, "femur_field_h_mm")
    lines.append((f"scan length {field:.0f} mm" + (" - too short for ROI margins" if roi else ""),
                  RED if roi else GRAY))
    # Боковой отступ ТЗ 2.3.2 (2 см) — только справка: буквальное правило
    # помечало бы 138 кадров из 150 против разметки эксперта (README, п. 6).
    lateral = _num(diag, "femur_lateral_mm")
    if lateral == lateral:
        lines.append((f"lateral margin {lateral:.0f} mm (TZ 20 mm, info only)", GRAY))

    # Ротация: медиальный край и выступ в окне малого вертела.
    rot = "FEMUR_POSITIONING_ROTATION" in violations
    try:
        track = track_medial_edge(pixels, region, mm, mm_per_px_source=source, model=model)
        cand = track.candidate("restricted")
    except ValueError as exc:
        lines.append((f"rotation: not measured ({exc})", GRAY))
        return

    def img_x(x):  # трек в канонических координатах: правое бедро отражено
        return x if region == FEMUR_LEFT else w - 1 - x

    pts = np.array([_pt(img_x(x), y) for x, y in zip(track.xs, track.ys)], np.int32)
    cv2.polylines(canvas, [pts], False, CYAN, 1, cv2.LINE_AA)
    for yy in (cand["window_start"], cand["window_end"]):
        x = img_x(np.interp(yy, track.ys, track.xs))
        cv2.line(canvas, _pt(x - 10, yy), _pt(x + 10, yy), YELLOW, 1, cv2.LINE_AA)
    cv2.circle(canvas, _pt(img_x(cand["anchor_x"]), cand["anchor_y"]), 6,
               RED if rot else MAGENTA, 2, cv2.LINE_AA)
    val, lo, hi = (_num(diag, k) for k in ("rotation_value", "rotation_low", "rotation_high"))
    feat = diag.get("rotation_feature", "") or "profile"
    lines.append((f"lesser trochanter {feat} {val:.3f}, normal [{lo:.3f}; {hi:.3f}]"
                  + (" - rotation" if rot else ""), RED if rot else MAGENTA))


def render(region: str, pixels: np.ndarray, violations: Iterable[str], diag: dict | None = None, *,
           mm_per_px: float | None = None, mm_per_px_source: str | None = None,
           scanner_model: str | None = None) -> np.ndarray:
    """BGR-картинка: снимок ×2 с разметкой и шапка с итогом по правилам."""
    violations = [v for v in violations if v]
    diag = diag or {}
    canvas = _gray(pixels)
    lines: list[tuple[str, tuple]] = []
    if region == SPINE:
        _spine(canvas, pixels, mm_per_px, violations, diag, lines)
    elif region in (FEMUR_LEFT, FEMUR_RIGHT):
        _femur(canvas, pixels, region, mm_per_px, mm_per_px_source, scanner_model,
               violations, diag, lines)
    verdict = ("VIOLATION: " + ", ".join(violations)) if violations else "OK: no violations found"
    head = [(TITLES.get(region, region.upper()), WHITE), (verdict, RED if violations else GREEN)] + lines
    strip = np.full((LINE_H * len(head) + 8, canvas.shape[1], 3), 24, np.uint8)
    for i, (text, color) in enumerate(head):
        cv2.putText(strip, text, (6, LINE_H * (i + 1)), FONT, FONT_SCALE, color, 1, cv2.LINE_AA)
    return np.vstack([strip, canvas])


def encode_png(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise ValueError("PNG не закодировался")
    return buf.tobytes()


class ZipWriter:
    """Детерминированный архив, записываемый потоком: запись за записью прямо
    в файл, без сборки архива в памяти. Дата и атрибуты записей фиксированы;
    порядок — тот, в котором пишет вызывающий (он обязан быть стабильным)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._zf = zipfile.ZipFile(self.path, "w", zipfile.ZIP_DEFLATED)

    def write(self, name: str, data: bytes) -> None:
        info = zipfile.ZipInfo(name, date_time=ZIP_DATE)
        info.compress_type = zipfile.ZIP_DEFLATED
        info.create_system = 3
        info.external_attr = 0o644 << 16
        self._zf.writestr(info, data)

    def close(self) -> None:
        self._zf.close()

    def __enter__(self) -> "ZipWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def write_zip(entries: dict[str, bytes], path: str | Path) -> Path:
    """Детерминированный архив из готовых записей: порядок имён, дата и атрибуты фиксированы."""
    with ZipWriter(path) as zw:
        for name in sorted(entries):
            zw.write(name, entries[name])
    return Path(path)
