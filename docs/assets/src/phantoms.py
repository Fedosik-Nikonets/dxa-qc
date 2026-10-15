"""Синтетические DXA-фантомы для иллюстраций README.

Это не данные организатора и не снимки пациентов: позвоночник и бедро
нарисованы примитивами OpenCV (тела позвонков, гребни подвздошных костей,
рёбра; головка, шейка, вертелы и диафиз бедра), размыты и зашумлены так,
чтобы настоящий конвейер `dxa_qc` прошёл по ним целиком — роутер, маска
кости, правила, overlay-серия. Файлы DICOM пишет `tests/synth_dicom.py`.
"""
from __future__ import annotations

import cv2
import numpy as np

PX = 0.6  # мм/пиксель, пишется в PixelSpacing


def _finish(canvas: np.ndarray, rng, blur=2.2, noise=6.0) -> np.ndarray:
    img = cv2.GaussianBlur(canvas.astype(np.float32), (0, 0), blur)
    img += rng.normal(0, noise, img.shape).astype(np.float32)
    # мелкозернистая «трабекулярная» текстура
    tex = cv2.GaussianBlur(rng.normal(0, 1, img.shape).astype(np.float32), (0, 0), 1.2)
    img += tex * 10 * (img > 90)
    return np.clip(img, 0, 255).astype(np.uint8)


def spine(rng, tilt_deg=0.0, crests=True, metal=False, rows=330, cols=270) -> np.ndarray:
    """Поясничный отдел: наклон `tilt_deg`, гребни в кадре, инородное тело `metal`."""
    big = np.zeros((rows * 2, cols * 2), np.float32)
    H, W = big.shape
    big[:] = 18
    # мягкие ткани
    cv2.ellipse(big, (W // 2, H // 2), (int(W * 0.42), int(H * 0.75)), 0, 0, 360, 48.0, -1)
    cx = W // 2
    top = int(H * 0.30)
    body_h, gap = 64, 14
    for i in range(6):
        y = top + i * (body_h + gap)
        w = 100 + i * 6
        tone = 178 + rng.integers(-8, 8)
        cv2.rectangle(big, (cx - w // 2, y), (cx + w // 2, y + body_h), float(tone), -1)
        cv2.ellipse(big, (cx - w // 2, y + body_h // 2), (12, body_h // 2), 0, 0, 360, float(tone), -1)
        cv2.ellipse(big, (cx + w // 2, y + body_h // 2), (12, body_h // 2), 0, 0, 360, float(tone), -1)
        # остистый отросток и корни дужек
        cv2.ellipse(big, (cx, y + body_h // 2), (9, 20), 0, 0, 360, float(tone + 25), -1)
        cv2.ellipse(big, (cx - w // 4, y + 14), (10, 8), 0, 0, 360, float(tone - 35), -1)
        cv2.ellipse(big, (cx + w // 4, y + 14), (10, 8), 0, 0, 360, float(tone - 35), -1)
        # поперечные отростки
        cv2.line(big, (cx - w // 2 - 4, y + 22), (cx - w // 2 - 34, y + 18), float(tone - 60), 9)
        cv2.line(big, (cx + w // 2 + 4, y + 22), (cx + w // 2 + 34, y + 18), float(tone - 60), 9)
    # рёбра сверху
    for k in range(2):
        y = top - 10 + k * 40
        cv2.ellipse(big, (cx - 120, y + 60), (130, 60), 0, 200, 300, 92, 7)
        cv2.ellipse(big, (cx + 120, y + 60), (130, 60), 0, 240, 340, 92, 7)
    if crests:
        yb = top + 6 * (body_h + gap) - 30
        cv2.ellipse(big, (cx - 210, yb + 120), (150, 140), 0, 190, 330, 175, 34)
        cv2.ellipse(big, (cx + 210, yb + 120), (150, 140), 0, 210, 350, 175, 34)
        cv2.ellipse(big, (cx - 190, yb + 150), (115, 110), 0, 0, 360, 120, -1)
        cv2.ellipse(big, (cx + 190, yb + 150), (115, 110), 0, 0, 360, 120, -1)
    M = cv2.getRotationMatrix2D((W / 2, H / 2), tilt_deg, 1.0)
    big = cv2.warpAffine(big, M, (W, H), borderValue=18)
    if metal:
        cv2.circle(big, (cx + 150, top + 120), 9, 255, -1)
        cv2.circle(big, (cx + 150, top + 120), 4, 255, -1)
        cv2.line(big, (cx + 140, top + 150), (cx + 175, top + 210), 250, 4)
    out = cv2.resize(big, (cols, rows), interpolation=cv2.INTER_AREA)
    return _finish(out, rng)


def femur(rng, side="right", scan_rows=270, cols=230, bump=0.0) -> np.ndarray:
    """Бедро: `bump` — размер малого вертела (0 — нет выступа), `scan_rows` —
    длина поля сканирования в пикселях (меньше 330 — поле обрезано снизу)."""
    S = 2
    H, W = 330 * S, cols * S
    big = np.full((H, W), 18, np.float32)
    cv2.ellipse(big, (W // 2, H // 2), (int(W * 0.48), int(H * 0.7)), 0, 0, 360, 46, -1)
    # таз и седалищная кость (медиально, слева)
    cv2.ellipse(big, (int(W * 0.10), int(H * 0.12)), (int(W * 0.30), int(H * 0.16)), 15, 0, 360, 150, -1)
    cv2.ellipse(big, (int(W * 0.12), int(H * 0.40)), (int(W * 0.10), int(H * 0.10)), 0, 0, 360, 135, -1)
    cv2.ellipse(big, (int(W * 0.12), int(H * 0.40)), (int(W * 0.05), int(H * 0.05)), 0, 0, 360, 60, -1)
    # головка
    hc = (int(W * 0.36), int(H * 0.20))
    cv2.circle(big, hc, int(W * 0.13), 210, -1)
    # шейка
    gt = (int(W * 0.66), int(H * 0.30))
    cv2.line(big, hc, (int(W * 0.58), int(H * 0.33)), 195, int(W * 0.15))
    # большой вертел
    cv2.ellipse(big, gt, (int(W * 0.13), int(H * 0.11)), -20, 0, 360, 200, -1)
    # межвертельная область и диафиз
    pts = np.array([
        [int(W * 0.50), int(H * 0.34)], [int(W * 0.76), int(H * 0.36)],
        [int(W * 0.64), int(H * 0.62)], [int(W * 0.60), int(H * 1.0)],
        [int(W * 0.40), int(H * 1.0)], [int(W * 0.42), int(H * 0.60)],
    ], np.int32)
    cv2.fillPoly(big, [pts], 200)
    # малый вертел — выступ на медиальном крае
    lt = (int(W * 0.415), int(H * 0.53))
    if bump > 0:
        cv2.ellipse(big, lt, (max(1, int(W * 0.035 * bump)), int(H * 0.035)), 30, 0, 360, 205, -1)
    # кортикальный слой диафиза
    cv2.line(big, (int(W * 0.42), int(H * 0.62)), (int(W * 0.40), H), 235, 10)
    cv2.line(big, (int(W * 0.64), int(H * 0.62)), (int(W * 0.60), H), 235, 10)
    if side == "left":
        big = big[:, ::-1].copy()
    out = cv2.resize(big, (cols, 330), interpolation=cv2.INTER_AREA)
    out = out[:scan_rows, :] if scan_rows < 330 else out
    return _finish(out, rng)
