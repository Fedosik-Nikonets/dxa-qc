#!/usr/bin/env python3
"""Собирает страницу для ручной разметки четырёх точек на бедре.

Почему страница, а не окно matplotlib: matplotlib нет в requirements.txt, а
opencv стоит headless — окна он не откроет. Страница не требует вообще
ничего: файл открывается двойным щелчком в браузере, работает без сети,
снимки зашиты в него самого и никуда не уходят. Это важно и потому, что
размечает человек вне команды: ему уходит один файл, а не репозиторий.

Порядок изображений не случаен. Первыми идут 16 кадров — по 8 позитивов и
негативов ротации, поровну левых и правых. Это проверочная выборка: если
выступ малого вертела на ней не разделяет классы, размечать остальные 64
незачем (обоснование — `reports/day5_rotation.md`). Внутри проверочной
группы и внутри остатка порядок перемешан с фиксированным зерном: иначе
позитивы и негативы шли бы подряд блоками по четыре, и разметчик мог бы
уловить этот ритм. Метка ротации в файл не попадает ни в каком виде.

Страница отдаёт `femur_points_human.csv` ровно в той схеме, которую ждёт
`scripts/11_femur_landmarks.py`: координаты в исходных пикселях DICOM, начало
в левом верхнем углу, x вправо, y вниз. Имя намеренно не совпадает с
`annotations/femur_points_v1.csv` — там лежит черновик автоматики, и затереть
его присланным файлом было бы легко:

    PYTHONPATH=src python scripts/11_femur_landmarks.py \
        --points annotations/femur_points_human.csv --allow-partial --source human

Запуск:
    python scripts/12_annotate_femur.py
    open annotations/annotator.html
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dxa_qc.dicom_io import read_dicom  # noqa: E402

PRIORITY_PER_CELL = 4     # позитивов/негативов ротации на каждую сторону
SHUFFLE_SEED = 20260920   # перемешивание внутри групп, чтобы метки не читались из порядка

POINTS = [
    ("gt", "Большой вертел", "#ff6e3c",
     "Самая верхняя точка большого вертела — костного бугра на наружной "
     "(латеральной) стороне, у основания шейки. Не головка бедра. Если "
     "вертел срезан краем кадра — клавиша n."),
    ("lt", "Малый вертел", "#00dcff",
     "Вершина малого вертела — небольшого выступа на внутренней (медиальной) "
     "стороне, ниже шейки. Ставьте на самую выступающую точку его контура. "
     "Виден плохо, но контур различим — поставьте и нажмите u («не уверен»). "
     "Не виден вовсе — клавиша n."),
    ("neck", "Середина шейки", "#ffe100",
     "Центр самого узкого места шейки — середина отрезка между её верхним и "
     "нижним контуром там, где шейка уже всего."),
    ("isch", "Седалищная кость", "#ff5adc",
     "Нижний край седалищной кости со стороны таза (медиальнее и выше "
     "головки). Точка обязана лежать на кости, а не в тёмном поле рядом. "
     "Кость не попала в кадр — клавиша n."),
]


def priority_order(manifest: pd.DataFrame, seed: int = SHUFFLE_SEED) -> pd.DataFrame:
    """16 кадров на проверку гипотезы вперёд, остальные следом.

    Внутри обеих групп порядок перемешивается: ячейки набираются по
    сторонам и меткам, и без перемешивания первые четыре кадра оказались бы
    позитивами ротации, следующие четыре — негативами и так далее. Разметчик
    метку не видит, но ритм «четыре одинаковых подряд» видно и без метки.
    """
    head = []
    for region in ("femur_left", "femur_right"):
        for y in (1, 0):
            cell = manifest[(manifest.region == region)
                            & (manifest.y_FEMUR_POSITIONING_ROTATION == y)]
            head += list(cell.image_id[:PRIORITY_PER_CELL])
    head_set = set(head)
    rng = np.random.default_rng(seed)
    rest = [i for i in manifest.image_id if i not in head_set]
    order = list(rng.permutation(head)) + list(rng.permutation(rest))
    rank = {img: n for n, img in enumerate(order)}
    manifest = manifest.copy()
    manifest["order"] = [rank[i] for i in manifest.image_id]
    manifest["priority"] = manifest.image_id.isin(head_set)
    return manifest.sort_values("order").reset_index(drop=True)


def encode(pixels: np.ndarray) -> str:
    a = np.asarray(pixels, dtype=float)
    lo, hi = float(a.min()), float(a.max())
    u8 = np.zeros_like(a, dtype=np.uint8) if hi <= lo else \
        np.clip((a - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", u8)
    if not ok:
        raise ValueError("не удалось закодировать кадр")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def build_page(images: list[dict]) -> str:
    """Подставляет данные в шаблон страницы."""
    html = TEMPLATE.replace("__DATA__", json.dumps(images, ensure_ascii=False))
    html = html.replace("__POINTS__", json.dumps(
        [{"key": k, "name": n, "color": c, "hint": h} for k, n, c, h in POINTS],
        ensure_ascii=False))
    return html


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "Data/НД_для_обучения/Исследования")
    ap.add_argument("--manifest", type=Path, default=ROOT / "annotations/femur_manifest_v1.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "annotations/annotator.html")
    ap.add_argument("--limit", type=int, default=None, help="взять только первые N кадров")
    args = ap.parse_args()

    manifest = priority_order(pd.read_csv(args.manifest))
    if args.limit:
        manifest = manifest.head(args.limit)

    images = []
    for x in manifest.itertuples(index=False):
        img = read_dicom(str(args.root / x.rel_path))
        if not img.ok or img.pixels is None:
            raise SystemExit(f"не читается {x.image_id}: {img.error}")
        images.append({
            "id": x.image_id, "region": x.region, "rows": int(img.pixels.shape[0]),
            "cols": int(img.pixels.shape[1]), "priority": bool(x.priority),
            "png": encode(img.pixels),
        })
        if len(images) % 20 == 0:
            print(f"  {len(images)}/{len(manifest)}", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_page(images), encoding="utf-8")
    size_mb = args.out.stat().st_size / 1e6
    print(f"записано: {args.out} ({size_mb:.1f} МБ, кадров {len(images)}, "
          f"из них проверочных {int(manifest.priority.sum())})")
    return 0


TEMPLATE = r"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Разметка бедра</title>
<style>
 :root { color-scheme: dark; --bg:#16181c; --panel:#1b1e24; --fg:#e8e8ea;
         --dim:#9aa0a8; --line:#2c3039; --ok:#57c07a; --warn:#ffb454; }
 * { box-sizing: border-box; }
 html,body { height:100%; }
 body { margin:0; background:var(--bg); color:var(--fg); display:flex;
        overflow:hidden;
        font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
 #stage { flex:1; display:flex; align-items:flex-start; justify-content:center;
          overflow:auto; padding:14px; }
 canvas#c { image-rendering:pixelated; cursor:crosshair; display:block;
            box-shadow:0 0 0 1px var(--line); }
 aside { width:352px; flex:none; border-left:1px solid var(--line); padding:16px;
         overflow-y:auto; background:var(--panel); }
 h1 { font-size:15px; margin:0 0 2px; }
 h2 { font-size:14px; margin:20px 0 8px; }
 .dim { color:var(--dim); font-size:12px; }
 .tag { display:inline-block; font-size:11px; padding:1px 7px; border-radius:10px;
        background:#3a4150; color:#cfe3ff; margin-left:6px; vertical-align:1px; }
 ol { padding-left:0; list-style:none; margin:12px 0; }
 li.pt { padding:8px 10px; border:1px solid var(--line); border-radius:7px;
         margin-bottom:6px; cursor:pointer; }
 li.pt.active { border-color:currentColor; background:#ffffff0f; }
 .name { font-weight:600; }
 .val { font-variant-numeric:tabular-nums; font-size:12px; color:var(--dim); }
 .hint { font-size:12px; color:var(--dim); margin-top:5px; display:none; }
 li.pt.active .hint { display:block; }
 button { background:#2b2f39; color:var(--fg); border:1px solid var(--line);
          border-radius:7px; padding:7px 10px; cursor:pointer; font-size:13px; }
 button:hover { background:#353a46; }
 button.wide { width:100%; margin-top:8px; }
 button.go { background:#2f6b46; border-color:#3d8a5b; }
 button.go:hover { background:#38804f; }
 .row { display:flex; gap:6px; margin-top:8px; }
 .row > button { flex:1; }
 kbd { background:#2b2f39; border:1px solid var(--line); border-radius:4px;
       padding:1px 5px; font-size:11px; font-family:inherit; }
 #bar { height:6px; background:#2b2f39; border-radius:3px; overflow:hidden; margin-top:10px; }
 #fill { height:100%; background:var(--ok); width:0; transition:width .15s; }
 .warn { color:var(--warn); font-size:12px; margin-top:8px; }
 label.ctl { display:block; font-size:12px; color:var(--dim); margin-top:8px; }
 input[type=range] { width:100%; accent-color:#5a8dd6; }
 #lupe { width:190px; height:190px; border:1px solid var(--line); border-radius:7px;
         image-rendering:pixelated; display:block; background:#000; }
 #grid { display:grid; grid-template-columns:repeat(10,1fr); gap:3px; margin-top:8px; }
 .cell { aspect-ratio:1/1; border-radius:3px; background:#2b2f39; cursor:pointer;
         border:1px solid transparent; }
 .cell.part { background:#8a6d2f; }
 .cell.done { background:var(--ok); }
 .cell.prio { border-color:#7fb3ff; }
 .cell.here { outline:2px solid #fff; outline-offset:1px; }
 #help { position:fixed; inset:0; background:#0b0d10ee; overflow:auto;
         padding:32px 20px; z-index:9; }
 #help .card { max-width:780px; margin:0 auto; background:var(--panel);
               border:1px solid var(--line); border-radius:12px; padding:26px 30px; }
 #help h2 { margin-top:22px; }
 #help table { border-collapse:collapse; width:100%; font-size:13px; margin-top:8px; }
 #help td, #help th { border:1px solid var(--line); padding:6px 9px; text-align:left;
                      vertical-align:top; }
 .lm { display:flex; gap:12px; margin:10px 0; align-items:flex-start; }
 .dot { width:14px; height:14px; border-radius:50%; flex:none; margin-top:4px; }
 .hide { display:none !important; }
 #pane { flex:1; overflow-y:auto; }
 #foot { border-top:1px solid var(--line); padding-top:10px; margin-top:10px; }
</style></head><body>

<div id="stage"><canvas id="c"></canvas></div>

<aside style="display:flex;flex-direction:column;overflow:hidden">
 <div id="pane">
  <h1 id="title">—</h1>
  <div class="dim" id="sub">—</div>
  <div id="bar"><div id="fill"></div></div>

  <ol id="list"></ol>

  <div class="row">
    <button id="prev">← назад</button>
    <button id="next">вперёд →</button>
  </div>
  <button class="wide go" id="jump">К следующему неразмеченному</button>

  <h2>Лупа</h2>
  <canvas id="lupe" width="190" height="190"></canvas>
  <div class="dim" id="coord" style="margin-top:4px">—</div>

  <h2>Изображение</h2>
  <div class="row">
    <button id="zout">− масштаб</button>
    <button id="zin">+ масштаб</button>
  </div>
  <label class="ctl">Яркость <span id="bval"></span>
    <input type="range" id="bright" min="-90" max="90" value="0"></label>
  <label class="ctl">Контраст <span id="cval"></span>
    <input type="range" id="contr" min="50" max="300" value="100"></label>
  <div class="row">
    <button id="inv">инверсия</button>
    <button id="reset">сброс</button>
  </div>

  <h2>Кадры</h2>
  <div class="dim">Голубая рамка — проверочные 16 кадров. Зелёный — готов,
    жёлтый — начат.</div>
  <div id="grid"></div>

  <h2>Управление</h2>
  <div class="dim">
    Клик — поставить точку и перейти к следующей<br>
    <kbd>shift</kbd>+клик или <kbd>u</kbd> — пометить «не уверен»<br>
    <kbd>n</kbd> — точки не видно · <kbd>z</kbd> — отменить<br>
    <kbd>←</kbd> <kbd>→</kbd> — кадры · <kbd>1</kbd>–<kbd>4</kbd> — выбрать точку<br>
    <kbd>+</kbd> <kbd>−</kbd> — масштаб · <kbd>h</kbd> — справка
  </div>
  <button class="wide" id="openhelp" style="margin-top:12px">Открыть справку</button>
  <div class="dim" style="margin-top:14px">
    Координаты пишутся в исходных пикселях снимка. Изображения зашиты в этот
    файл, никуда не отправляются и работают без интернета.
  </div>
 </div>

 <div id="foot">
  <button class="wide go" id="save" style="margin-top:0">Скачать CSV с разметкой</button>
  <div class="row">
    <button id="drop">Сохранить черновик</button>
    <button id="pick">Загрузить черновик</button>
  </div>
  <input type="file" id="file" accept=".json" class="hide">
  <div id="msg" class="warn"></div>
 </div>
</aside>

<div id="help" class="hide"><div class="card">
  <h1 style="font-size:20px">Разметка четырёх точек на снимках бедра</h1>
  <p class="dim">Перед вами кадры рентгеновской денситометрии проксимального
  отдела бедра. Нужно на каждом кадре отметить четыре анатомических ориентира.
  Никаких диагнозов и оценок ставить не нужно — только точки.</p>

  <h2>Зачем это</h2>
  <p>По взаимному расположению этих точек программа будет автоматически
  проверять, правильно ли уложена нога пациента при исследовании. Ваши точки —
  эталон, по которому проверяется автоматика, поэтому важна аккуратность, а не
  скорость. Если точка не видна — так и отметьте; пропуск честнее догадки.</p>

  <h2>Какие точки</h2>
  <p class="dim">Кадр — это обычная передне-задняя проекция проксимального
  отдела бедра, увеличенная и без подписей. Левое это бедро или правое,
  написано в заголовке справа от снимка.</p>
  <div class="lm"><div class="dot" style="background:#ff6e3c"></div><div>
    <b>1. Большой вертел</b> — самая верхняя точка его контура.
    <span class="dim">Именно верхушка, а не центр массива и не головка
    бедра. Срезан краем кадра — <kbd>n</kbd>.</span></div></div>
  <div class="lm"><div class="dot" style="background:#00dcff"></div><div>
    <b>2. Малый вертел</b> — самая выступающая медиально точка его контура.
    <span class="dim">Виден слабо — поставьте и пометьте <kbd>u</kbd>;
    контура нет вовсе — <kbd>n</kbd>. Оба ответа для нас содержательны, так
    что угадывать не нужно.</span></div></div>
  <div class="lm"><div class="dot" style="background:#ffe100"></div><div>
    <b>3. Середина шейки</b> — центр самого узкого её сечения.
    <span class="dim">Найдите место, где шейка уже всего, и поставьте точку
    посередине между верхним и нижним кортикальным контуром.</span></div></div>
  <div class="lm"><div class="dot" style="background:#ff5adc"></div><div>
    <b>4. Седалищная кость</b> — её нижний край.
    <span class="dim">Точка обязана лежать на кости, а не в тёмном поле рядом.
    Кость не попала в кадр — <kbd>n</kbd>.</span></div></div>
  <p class="dim">Если на конкретном кадре анатомия неоднозначна — ставьте
  «не уверен» или «не видно». Это нормальный ответ, и он нам полезнее
  догадки.</p>

  <h2>Как работать</h2>
  <table>
    <tr><th style="width:190px">Действие</th><th>Как</th></tr>
    <tr><td>Поставить точку</td><td>Щелчок по снимку. Программа сама
      переключится на следующую точку.</td></tr>
    <tr><td>Точку видно плохо</td><td><kbd>shift</kbd>+щелчок, либо поставьте
      обычным щелчком и нажмите <kbd>u</kbd>. Это не ошибка — такая пометка
      полезна.</td></tr>
    <tr><td>Точки не видно совсем</td><td><kbd>n</kbd>. Координаты не
      записываются, кадр всё равно считается размеченным.</td></tr>
    <tr><td>Исправить</td><td><kbd>z</kbd> убирает последнюю точку. Можно также
      выбрать точку цифрами <kbd>1</kbd>–<kbd>4</kbd> и щёлкнуть заново.</td></tr>
    <tr><td>Плохо видно кость</td><td>Ползунки «Яркость» и «Контраст» справа,
      кнопка «инверсия», кнопки масштаба. На картинку это не влияет —
      меняется только показ.</td></tr>
    <tr><td>Точное попадание</td><td>Смотрите в «Лупу» справа: она показывает
      место под курсором крупно.</td></tr>
    <tr><td>Перейти к кадру</td><td>Стрелки <kbd>←</kbd> <kbd>→</kbd>, кнопка
      «К следующему неразмеченному» или квадратик в сетке кадров.</td></tr>
  </table>

  <h2>Сколько размечать</h2>
  <p>Всего 80 кадров. <b>Первые 16 — самые важные</b> (в сетке кадров они
  обведены голубым). Если времени мало, сделайте хотя бы их и пришлите
  результат — этого достаточно для первой проверки. Остальные можно доразметить
  потом.</p>
  <p><b>Начните с пробы:</b> разметьте 2–3 кадра, нажмите «Скачать CSV» и
  пришлите файл сразу. Мы проверим, что он читается, и только тогда есть
  смысл тратить время на остальные.</p>

  <h2>Что прислать обратно</h2>
  <p>Нажмите <b>«Скачать CSV с разметкой»</b> — браузер сохранит файл
  <code>femur_points_human.csv</code>. Его и пришлите. В файл попадают только
  кадры, где заполнены все четыре точки.</p>
  <p class="dim">Работа сохраняется в браузере автоматически, но браузер может
  её потерять. Раз в полчаса нажимайте «Сохранить черновик» — получится файл
  <code>.json</code>, из которого всё восстанавливается кнопкой «Загрузить
  черновик». Не открывайте страницу в режиме инкогнито: там черновик не
  сохраняется.</p>

  <button class="wide go" id="closehelp" style="margin-top:20px">Понятно, начать</button>
</div></div>

<script>
const IMAGES = __DATA__, POINTS = __POINTS__;
const KEY = "femur-annot-v1", VIEWKEY = "femur-annot-view-v1";
const MAXROWS = Math.max.apply(null, IMAGES.map(i => i.rows));

let idx = 0, cur = 0, data = {}, storageOk = true, dirty = false;
let zoom = 3, bright = 0, contrast = 100, invert = false;
let hoverX = null, hoverY = null;

const el = id => document.getElementById(id);
const canvas = el("c"), ctx = canvas.getContext("2d");
const lupe = el("lupe"), lctx = lupe.getContext("2d");
const off = document.createElement("canvas");
const octx = off.getContext("2d", {willReadFrequently: true});
const native = {};            // id -> ImageData исходного кадра

/* ---------- хранение ---------- */
try {
  data = JSON.parse(localStorage.getItem(KEY) || "{}");
  const v = JSON.parse(localStorage.getItem(VIEWKEY) || "{}");
  if (typeof v.zoom === "number") zoom = v.zoom;
  if (typeof v.bright === "number") bright = v.bright;
  if (typeof v.contrast === "number") contrast = v.contrast;
  if (typeof v.invert === "boolean") invert = v.invert;
} catch (e) { storageOk = false; data = {}; }

function persist() {
  if (!storageOk) return;
  try {
    localStorage.setItem(KEY, JSON.stringify(data));
    localStorage.setItem(VIEWKEY, JSON.stringify({zoom, bright, contrast, invert}));
  } catch (e) { storageOk = false; note(); }
}

function blank() { return POINTS.map(() => ({x: null, y: null, state: null})); }
function rec() { const id = IMAGES[idx].id; return data[id] || (data[id] = blank()); }
function full(r) { return r && r.every(p => p.state !== null); }
function started(r) { return r && r.some(p => p.state !== null); }

/* ---------- отрисовка ---------- */
function lut() {
  const t = new Uint8ClampedArray(256), c = contrast / 100;
  for (let v = 0; v < 256; v++) {
    let u = (v - 128) * c + 128 + bright;
    u = u < 0 ? 0 : u > 255 ? 255 : u;
    t[v] = invert ? 255 - u : u;
  }
  return t;
}

function prepare(cb) {
  const im = IMAGES[idx];
  if (native[im.id]) return cb();
  const img = new Image();
  img.onload = () => {
    off.width = im.cols; off.height = im.rows;
    octx.clearRect(0, 0, im.cols, im.rows);
    octx.drawImage(img, 0, 0);
    native[im.id] = octx.getImageData(0, 0, im.cols, im.rows);
    cb();
  };
  img.onerror = () => { el("msg").textContent = "Кадр " + im.id + " не открылся."; };
  img.src = "data:image/png;base64," + im.png;
}

function toned() {
  const im = IMAGES[idx], src = native[im.id], t = lut();
  off.width = im.cols; off.height = im.rows;
  const out = octx.createImageData(im.cols, im.rows), d = out.data, s = src.data;
  for (let i = 0; i < s.length; i += 4) {
    const v = t[s[i]];
    d[i] = d[i + 1] = d[i + 2] = v; d[i + 3] = 255;
  }
  octx.putImageData(out, 0, 0);
  return off;
}

function draw() {
  const im = IMAGES[idx];
  if (!native[im.id]) return;
  const base = toned();
  canvas.width = im.cols * zoom; canvas.height = im.rows * zoom;
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(base, 0, 0, canvas.width, canvas.height);
  rec().forEach((p, i) => {
    if (p.x === null) return;
    const x = (p.x + 0.5) * zoom, y = (p.y + 0.5) * zoom;
    ctx.strokeStyle = POINTS[i].color;
    ctx.lineWidth = i === cur ? 2.5 : 1.5;
    ctx.setLineDash(p.state === "uncertain" ? [3, 3] : []);
    ctx.beginPath(); ctx.arc(x, y, 8, 0, 6.3); ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(x - 13, y); ctx.lineTo(x - 3, y);
    ctx.moveTo(x + 3, y); ctx.lineTo(x + 13, y);
    ctx.moveTo(x, y - 13); ctx.lineTo(x, y - 3);
    ctx.moveTo(x, y + 3); ctx.lineTo(x, y + 13);
    ctx.stroke();
    ctx.fillStyle = POINTS[i].color; ctx.font = "bold 13px sans-serif";
    ctx.fillText(String(i + 1), x + 11, y - 9);
  });
  render();
}

function drawLupe() {
  const im = IMAGES[idx];
  if (hoverX === null || !native[im.id]) {
    lctx.fillStyle = "#000"; lctx.fillRect(0, 0, lupe.width, lupe.height);
    el("coord").textContent = "наведите курсор на снимок";
    return;
  }
  const span = 20, k = lupe.width / span, base = toned();
  lctx.imageSmoothingEnabled = false;
  lctx.fillStyle = "#000"; lctx.fillRect(0, 0, lupe.width, lupe.height);
  lctx.drawImage(base, hoverX - span / 2, hoverY - span / 2, span, span,
                 0, 0, lupe.width, lupe.height);
  lctx.strokeStyle = POINTS[cur].color; lctx.lineWidth = 1;
  const c = lupe.width / 2;
  lctx.beginPath();
  lctx.moveTo(c, 0); lctx.lineTo(c, c - k * 0.7);
  lctx.moveTo(c, c + k * 0.7); lctx.lineTo(c, lupe.height);
  lctx.moveTo(0, c); lctx.lineTo(c - k * 0.7, c);
  lctx.moveTo(c + k * 0.7, c); lctx.lineTo(lupe.width, c);
  lctx.stroke();
  lctx.strokeRect(c - k / 2, c - k / 2, k, k);
  el("coord").textContent = "x " + hoverX + " · y " + hoverY;
}

function render() {
  const im = IMAGES[idx], r = rec();
  el("title").innerHTML = im.id + " · " + (im.region === "femur_left" ? "левое" : "правое")
    + " бедро" + (im.priority ? '<span class="tag">проверочный</span>' : "");
  const done = IMAGES.filter(i => full(data[i.id])).length;
  const pd = IMAGES.filter(i => i.priority && full(data[i.id])).length;
  const pn = IMAGES.filter(i => i.priority).length;
  el("sub").textContent = "кадр " + (idx + 1) + " из " + IMAGES.length
    + " · готово " + done + " · из проверочных " + pd + " из " + pn;
  el("fill").style.width = (100 * done / IMAGES.length) + "%";
  el("list").innerHTML = POINTS.map((p, i) => {
    const v = r[i];
    const txt = v.state === null ? "—"
      : v.state === "not_visible" ? "не видно"
      : v.x + ", " + v.y + (v.state === "uncertain" ? " · не уверен" : "");
    return '<li class="pt ' + (i === cur ? "active" : "") + '" style="color:' + p.color
      + '" data-i="' + i + '"><div><span class="name">' + (i + 1) + ". " + p.name
      + '</span></div><div class="val">' + txt + '</div><div class="hint">'
      + p.hint + "</div></li>";
  }).join("");
  [...el("list").children].forEach(li =>
    li.onclick = () => { cur = +li.dataset.i; draw(); });
  el("grid").innerHTML = IMAGES.map((im2, i) => {
    const r2 = data[im2.id];
    const cls = ["cell", full(r2) ? "done" : started(r2) ? "part" : "",
                 im2.priority ? "prio" : "", i === idx ? "here" : ""].join(" ");
    return '<div class="' + cls + '" data-i="' + i + '" title="' + im2.id + '"></div>';
  }).join("");
  [...el("grid").children].forEach(d =>
    d.onclick = () => { idx = +d.dataset.i; cur = 0; load(); });
  el("bval").textContent = bright;
  el("cval").textContent = (contrast / 100).toFixed(2);
  note();
}

function note() {
  const partial = IMAGES.filter(i => started(data[i.id]) && !full(data[i.id])).length;
  let s = "";
  if (!storageOk) s = "Браузер не сохраняет черновик. Чаще нажимайте «Сохранить черновик».";
  else if (partial) s = "Кадров начато, но не закончено: " + partial
    + ". В CSV они не попадут.";
  el("msg").textContent = s;
}

function load() { prepare(() => { draw(); drawLupe(); }); }

/* ---------- ввод ---------- */
function pixel(e) {
  const b = canvas.getBoundingClientRect(), im = IMAGES[idx];
  const sx = canvas.width / b.width, sy = canvas.height / b.height;
  const x = Math.floor((e.clientX - b.left) * sx / zoom);
  const y = Math.floor((e.clientY - b.top) * sy / zoom);
  if (x < 0 || y < 0 || x >= im.cols || y >= im.rows) return null;
  return {x, y};
}

canvas.onmousemove = e => {
  const p = pixel(e);
  hoverX = p ? p.x : null; hoverY = p ? p.y : null;
  drawLupe();
};
canvas.onmouseleave = () => { hoverX = hoverY = null; drawLupe(); };

canvas.onclick = e => {
  const p = pixel(e);
  if (!p) return;
  // shift — «не уверен». Отдельной клавиши для пустой точки нет сознательно:
  // uncertain без координат валидатор scripts/11 отвергает.
  rec()[cur] = {x: p.x, y: p.y, state: e.shiftKey ? "uncertain" : "visible"};
  cur = Math.min(cur + 1, POINTS.length - 1);
  dirty = true; persist(); draw();
};

function markNotVisible() {
  rec()[cur] = {x: null, y: null, state: "not_visible"};
  cur = Math.min(cur + 1, POINTS.length - 1);
  dirty = true; persist(); draw();
}

function markUncertain() {
  // Работает только по уже поставленной точке: пометка без координат невалидна.
  const r = rec();
  let i = r[cur].state ? cur : cur - 1;
  if (i < 0 || !r[i] || r[i].x === null) return;
  r[i].state = r[i].state === "uncertain" ? "visible" : "uncertain";
  dirty = true; persist(); draw();
}

function undo() {
  const r = rec();
  let i = r[cur].state !== null ? cur : cur - 1;
  if (i < 0) return;
  r[i] = {x: null, y: null, state: null};
  cur = i;
  dirty = true; persist(); draw();
}

function go(step) {
  idx = Math.min(Math.max(idx + step, 0), IMAGES.length - 1);
  cur = 0; load();
}

function nextUnfinished() {
  const order = IMAGES.map((im, i) => i)
    .filter(i => !full(data[IMAGES[i].id]))
    .sort((a, b) => (IMAGES[b].priority - IMAGES[a].priority) || (a - b));
  if (!order.length) { el("msg").textContent = "Все кадры размечены."; return; }
  const after = order.find(i => i > idx && IMAGES[i].priority === IMAGES[idx].priority);
  idx = after !== undefined ? after : order[0];
  cur = 0; load();
}

function setZoom(z) { zoom = Math.min(8, Math.max(1, z)); persist(); draw(); }

el("prev").onclick = () => go(-1);
el("next").onclick = () => go(1);
el("jump").onclick = nextUnfinished;
el("zin").onclick = () => setZoom(zoom + 1);
el("zout").onclick = () => setZoom(zoom - 1);
el("inv").onclick = () => { invert = !invert; persist(); draw(); drawLupe(); };
el("reset").onclick = () => {
  bright = 0; contrast = 100; invert = false;
  el("bright").value = 0; el("contr").value = 100;
  persist(); draw(); drawLupe();
};
el("bright").oninput = e => { bright = +e.target.value; persist(); draw(); drawLupe(); };
el("contr").oninput = e => { contrast = +e.target.value; persist(); draw(); drawLupe(); };

const HELPKEY = "femur-annot-help-seen-v1";
function showHelp(on) {
  el("help").classList.toggle("hide", !on);
  if (!on) { try { localStorage.setItem(HELPKEY, "1"); } catch (e) {} }
}
el("openhelp").onclick = () => showHelp(true);
el("closehelp").onclick = () => showHelp(false);

document.addEventListener("keydown", e => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const code = e.code;
  if (code === "Escape" || e.key === "Escape") { showHelp(false); e.preventDefault(); return; }
  if (!el("help").classList.contains("hide")) return;
  // Разбор по e.code, а не по e.key: при русской раскладке key даёт «т» и «я».
  if (code === "ArrowRight") go(1);
  else if (code === "ArrowLeft") go(-1);
  else if (code === "KeyN") markNotVisible();
  else if (code === "KeyZ") undo();
  else if (code === "KeyU") markUncertain();
  else if (code === "KeyH") showHelp(true);
  else if (code === "Equal" || code === "NumpadAdd") setZoom(zoom + 1);
  else if (code === "Minus" || code === "NumpadSubtract") setZoom(zoom - 1);
  else if (/^Digit[1-4]$/.test(code)) { cur = +code.slice(5) - 1; draw(); }
  else return;
  e.preventDefault();
});

/* ---------- выгрузка ---------- */
function download(name, text, mime) {
  const blob = new Blob([text], {type: mime});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = name;
  // Firefox не срабатывает на click(), пока ссылки нет в документе.
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { a.remove(); URL.revokeObjectURL(url); }, 1000);
}

function csvCell(v) {
  const s = String(v == null ? "" : v);
  return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
}

el("save").onclick = () => {
  const head = ["image_id"];
  POINTS.forEach(p => head.push(p.key + "_x", p.key + "_y", p.key + "_state"));
  const lines = [head.join(",")];
  let skipped = 0;
  IMAGES.forEach(im => {
    const r = data[im.id];
    if (!started(r)) return;
    if (!full(r)) { skipped++; return; }
    const cells = [im.id];
    r.forEach(p => cells.push(p.x === null ? "" : p.x, p.y === null ? "" : p.y, p.state));
    lines.push(cells.map(csvCell).join(","));
  });
  if (lines.length === 1) {
    el("msg").textContent = "Пока нечего выгружать: ни один кадр не закончен.";
    return;
  }
  download("femur_points_human.csv", lines.join("\n") + "\n", "text/csv;charset=utf-8");
  dirty = false;
  el("msg").textContent = "Выгружено кадров: " + (lines.length - 1)
    + (skipped ? ". Пропущено незаконченных: " + skipped : "");
};

el("drop").onclick = () => {
  download("femur_draft.json", JSON.stringify({version: 1, data}, null, 1),
           "application/json");
  dirty = false;
  el("msg").textContent = "Черновик сохранён в файл.";
};

el("pick").onclick = () => el("file").click();
el("file").onchange = e => {
  const f = e.target.files[0];
  if (!f) return;
  const rd = new FileReader();
  rd.onload = () => {
    try {
      const parsed = JSON.parse(rd.result);
      const incoming = parsed && parsed.data ? parsed.data : parsed;
      let n = 0;
      Object.keys(incoming).forEach(id => {
        if (!IMAGES.some(i => i.id === id)) return;
        const v = incoming[id];
        if (!Array.isArray(v) || v.length !== POINTS.length) return;
        data[id] = v; n++;
      });
      persist(); draw();
      el("msg").textContent = "Загружено кадров из черновика: " + n;
    } catch (err) {
      el("msg").textContent = "Файл черновика не читается.";
    }
  };
  rd.readAsText(f);
  e.target.value = "";
};

window.addEventListener("beforeunload", e => {
  if (!dirty) return;
  e.preventDefault(); e.returnValue = "";
});

/* ---------- старт ---------- */
(function start() {
  el("bright").value = bright; el("contr").value = contrast;
  const fit = Math.floor((window.innerHeight - 60) / MAXROWS);
  if (!localStorage || localStorage.getItem(VIEWKEY) === null) {
    zoom = Math.min(6, Math.max(3, fit));
  }
  let seen = false;
  try { seen = localStorage.getItem(HELPKEY) === "1"; } catch (e) {}
  showHelp(!seen);
  load();
})();
</script></body></html>
"""

if __name__ == "__main__":
    raise SystemExit(main())
