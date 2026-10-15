# Презентация DXA QC на официальном шаблоне ЛЦТ 2026.
# Слайды 1–5 — обязательные слайды шаблона (исходные 7–11), заполнены без изменения
# сетки; со слайда «Команда» удалены лишние карточки. Слайды 6–20 — решение по ТЗ, раздел 4.
# Сборка: python3 build.py  (нужны template.pptx и assets/ рядом со скриптом)
import copy
import os
import sys

from pptx import Presentation
from pptx.enum.dml import MSO_THEME_COLOR
from pptx.enum.shapes import PP_PLACEHOLDER
from pptx.oxml.ns import qn
from pptx.util import Inches as In, Pt
from lxml import etree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *  # noqa: F401,F403

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(HERE, "assets") + os.sep
TEMPLATE = os.path.join(HERE, "template.pptx")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "DXA_QC_presentation.pptx")

prs = Presentation(TEMPLATE)
S = list(prs.slides)
N_TEMPLATE = len(S)
KEEP = [6, 7, 8, 9, 10]  # исходные слайды 7–11


def bg_part(slide):
    bg = slide._element.cSld.find(qn("p:bg"))
    blip = bg.find(".//" + qn("a:blip"))
    return slide.part.related_part(blip.get(qn("r:embed")))


BG_DARK = bg_part(S[8])     # фиолетовый фон с белыми логотипами
BG_LIGHT = bg_part(S[7])    # светлый фон с тёмными логотипами
BG_CITY = bg_part(S[11])    # фиолетовый фон с городом
SLDNUM = copy.deepcopy([sh for sh in S[10].shapes if sh.is_placeholder and sh.placeholder_format.type == PP_PLACEHOLDER.SLIDE_NUMBER][0]._element)


def shape_by_id(slide, sid):
    return [sh for sh in slide.shapes if sh.shape_id == sid][0]


def remove(sh):
    sh._element.getparent().remove(sh._element)


def set_paragraphs_like(tf, items, space_before=None):
    """Заменяет абзацы, копируя формат первого абзаца шаблона. **...** — жирный."""
    import re
    p0 = copy.deepcopy(tf.paragraphs[0]._p)
    r0 = p0.find(qn("a:r"))
    rPr0 = r0.find(qn("a:rPr")) if r0 is not None else None
    txBody = tf._txBody
    for p in list(txBody.findall(qn("a:p"))):
        txBody.remove(p)
    for item in items:
        p = copy.deepcopy(p0)
        if space_before:
            pPr = p.find(qn("a:pPr"))
            if pPr is None:
                pPr = etree.Element(qn("a:pPr"))
                p.insert(0, pPr)
            for e in pPr.findall(qn("a:spcBef")):
                pPr.remove(e)
            sb = etree.Element(qn("a:spcBef"))
            etree.SubElement(sb, qn("a:spcPts")).set("val", str(int(space_before * 100)))
            # spcBef идёт после lnSpc, до buClr/buFont/buChar
            ln = pPr.find(qn("a:lnSpc"))
            if ln is not None:
                ln.addnext(sb)
            else:
                pPr.insert(0, sb)
        for r in p.findall(qn("a:r")) + p.findall(qn("a:br")):
            p.remove(r)
        end = p.find(qn("a:endParaRPr"))
        for part in [t for t in re.split(r"(\*\*[^*]+\*\*)", item) if t]:
            r = etree.SubElement(p, qn("a:r"))
            if rPr0 is not None:
                rp = copy.deepcopy(rPr0)
                r.append(rp)
            else:
                rp = etree.SubElement(r, qn("a:rPr"))
            rp.set("lang", "ru-RU")
            if part.startswith("**"):
                part = part[2:-2]
                rp.set("b", "1")
            t = etree.SubElement(r, qn("a:t"))
            t.text = part
            if end is not None:
                end.addprevious(r)
        txBody.append(p)


# ============================================================ обязательные слайды
# --- 1. Титульный (шаблон, слайд 7)
s = S[6]
ph = {p.placeholder_format.idx: p for p in s.placeholders}
ph[0].text_frame.text = "ТЁМНАЯ МАТЕРИЯ"
ph[12].text_frame.text = "Сервис искусственного интеллекта по оценке качества\nисследований плотности костей человека"
logo = ph[11].insert_picture(A + "cdit_logo_hi.png")
logo.crop_left = logo.crop_right = logo.crop_top = logo.crop_bottom = 0
lw_, lh_ = image_dims(A + "cdit_logo_hi.png")
bx, by, bw, bh = 0.33, 0.36, 5.68, 1.18
h = bh * 0.86
w = h * lw_ / lh_
logo.left, logo.top, logo.width, logo.height = In(bx + 0.12), In(by + (bh - h) / 2), In(w), In(h)

# --- 2. Описание команды (шаблон, слайд 8)
s = S[7]
ph = {p.placeholder_format.idx: p for p in s.placeholders}
geo = (ph[10].left, ph[10].top, ph[10].width, ph[10].height)
pic = ph[10].insert_picture(A + "team_montage.png")
pic.crop_left = pic.crop_right = pic.crop_top = pic.crop_bottom = 0
pic.left, pic.top, pic.width, pic.height = geo
t = ph[0].text_frame
t.text = "Тёмная материя"
r = t.paragraphs[0].runs[0]
r.font.size = Pt(24)
r.font.color.theme_color = MSO_THEME_COLOR.ACCENT_5
tf = shape_by_id(s, 14).text_frame
P = tf.paragraphs
P[0].runs[1].text = "Фёдор Никонец, ML-инженер"
P[1].runs[1].text = "2 человека"
set_text_keep_format(P[3], "два ML-инженера, вклад в проект равный")
P[3].runs[0].font.italic = False
set_text_keep_format(P[4], "оба учатся в РЭУ им. Г. В. Плеханова")
P[4].runs[0].font.italic = False
r_el = copy.deepcopy(P[5].runs[0]._r)
r_el.find(qn("a:t")).text = " Москва"
r_el.find(qn("a:rPr")).set("b", "0")
P[5].runs[-1]._r.addnext(r_el)
set_text_keep_format(shape_by_id(s, 5).text_frame.paragraphs[0],
                     "Локальный сервис проверяет DXA-исследование в DICOM по пяти критериям ТЗ, "
                     "указывает тип нарушения и рисует измерение прямо на снимке. "
                     "F1\u00a00.74 по исследованиям, 0.1\u00a0с на исследование")
set_text_keep_format(shape_by_id(s, 8).text_frame.paragraphs[0],
                     "Не «чёрный ящик»: каждое решение — измерение против порога, нарисованное "
                     "на снимке отдельной DICOM-серией. Работает без GPU и сети, "
                     "все метрики воспроизводятся скриптами")

# --- 3. Команда (шаблон, слайд 9): оставляем две карточки
s = S[8]
s.shapes.title.text_frame.text = "КОМАНДА"
for sid in (59, 60, 61, 62, 63, 64, 65, 66, 67):
    remove(shape_by_id(s, sid))
for sh in list(s.shapes):
    if sh.name in ("Рисунок 3", "Рисунок 4", "Рисунок 5"):
        remove(sh)
members = [
    (15, 9, "Рисунок 1", "Фёдор Никонец", "avatar_fn.png"),
    (58, 57, "Рисунок 2", "Илья Рыжиков", "avatar_ir.png"),
]
for name_id, body_id, pic_name, name, avatar in members:
    set_text_keep_format(shape_by_id(s, name_id).text_frame.paragraphs[0], name)
    bp = shape_by_id(s, body_id).text_frame.paragraphs
    for para, txt in zip(bp, ["ML-инженер", "Telegram: @ВПИСАТЬ", "Тел.: ВПИСАТЬ", "РЭУ им. Г. В. Плеханова"]):
        set_text_keep_format(para, txt)
    ph_pic = [sh for sh in s.shapes if sh.name == pic_name][0]
    geo = (ph_pic.left, ph_pic.top, ph_pic.width, ph_pic.height)
    remove(ph_pic)
    p_ = s.shapes.add_picture(A + avatar, *geo)
    p_.name = pic_name

# --- 4. История команды (шаблон, слайд 10)
s = S[9]
s.shapes.title.text_frame.text = "ИСТОРИЯ КОМАНДЫ"
set_text_keep_format(shape_by_id(s, 37).text_frame.paragraphs[0],
                     "Учимся в РЭУ им. Г. В. Плеханова, оба работаем в ML. "
                     "Вклад равный: данные, правила, модели и сервис делали вдвоём")
set_text_keep_format(shape_by_id(s, 43).text_frame.paragraphs[0],
                     "Задача с быстрым эффектом: брак можно переснять, пока пациент в кабинете. "
                     "Хотели проверку, которую врач перепроверит глазами")
dp = shape_by_id(s, 40).text_frame.paragraphs
set_text_keep_format(dp[0],
                     "Нет контуров и точек, на редкие нарушения — 6–10 примеров, в DICOM нет масштаба. "
                     "Критерии ТЗ перевели в измерения на снимке.")
set_text_keep_format(dp[1],
                     "Пороги брали только с обучающих фолдов, критерий включения модели фиксировали "
                     "заранее — и отклонили вариант без прироста F1.")

# --- 5. Коротко о решении (шаблон, слайд 11)
s = S[10]
ph = {p.placeholder_format.idx: p for p in s.placeholders}
set_paragraphs_like(ph[38].text_frame, [
    "**Вход:** DICOM-файлы, каталог или zip; область и проекция — по геометрии кадра",
    "**Маска кости:** Otsu на 3 класса и гистерезис, масштаб 0.6 мм/пиксель",
    "**5 проверок ТЗ 2.3:** три правила геометрии, логистическая регрессия для артефактов, "
    "правило ротации с двумя границами",
    "**Выход:** таблица ТЗ 2.5 с баллом нарушения, overlay-серия и отчёт DICOM SR",
    "**Запуск:** Docker без сети, HTTP API и веб-интерфейс; ~0.1 с на исследование на CPU",
    "**Качество:** F1 0.60 по снимкам, 0.74 по исследованиям (5 фолдов, 95% ДИ)",
], space_before=7)
set_paragraphs_like(ph[42].text_frame, [
    "**Пользователи:** лаборант, врач-рентгенолог, служба контроля качества",
    "**Ценность:** брак виден, пока пациент в кабинете, — меньше повторных визитов и недостоверных заключений",
    "**Безопасность:** работает в контуре медорганизации, снимки не покидают больницу",
    "**Внедрение:** пилот «в тени» на 1–2 аппаратах, калибровка порогов, затем подсказка лаборанту и аудит",
    "**Развитие:** Th12, точный локализатор вертела, приём исследований из PACS",
    "**Эффект в цифрах** (доля пересъёмок, повторные визиты) измеряем на пилоте",
], space_before=7)

# ============================================================ контентные слайды
LAYOUT = [l for l in prs.slide_layouts if l.name == "Заголовок и объект"][0]
X0, X1, Y0, Y1 = 0.38, 12.96, 1.32, 6.80
new_slides = []


def new_slide(title, dark=True, bg=None, notes=None, pill=True):
    s = prs.slides.add_slide(LAYOUT)
    new_slides.append(s)
    for p in list(s.placeholders):
        if p.placeholder_format.type != PP_PLACEHOLDER.TITLE:
            remove(p)
    set_bg(s, bg or (BG_DARK if dark else BG_LIGHT))
    t = s.shapes.title
    if pill:
        tw = text_width_in(title, 20)
        pl = rrect(s, 0.38, 0.35, tw + 0.62, 0.68, fill=PINK, radius=0.127)
        s.shapes._spTree.remove(pl._element)
        s.shapes._spTree.insert(2, pl._element)
        t.left, t.top, t.width, t.height = In(0.58), In(0.50), In(tw + 0.45), In(0.41)
        t.text_frame.text = title
    else:
        remove(t)
    num = copy.deepcopy(SLDNUM)
    num.find(".//" + qn("p:cNvPr")).set("id", str(s.shapes._next_shape_id))
    if not dark:
        for sc in num.iter(qn("a:schemeClr")):
            sc.set("val", "accent5")
    s.shapes._spTree.append(num)
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def card(s, x, y, w, h, dark):
    return rrect(s, x, y, w, h, fill=WHITE, line=None if dark else LAV, lw=0.75, radius=0.18)


def head(s, text, x, y, w, size=16, color=PURP, h=0.4):
    return tb(s, x, y, w, h, [text], size=size, bold=True, color=color, anchor="m")


def hbars(s, rows, x_lab, w_lab, x_bar, w_max, vmax, y, step, bh=0.32, color=PURP, size=11.5, fmt="{}"):
    for i, (lab, v) in enumerate(rows):
        yy = y + i * step
        tb(s, x_lab, yy, w_lab, bh, [lab], size=size, color=INK, anchor="m")
        bw = max(0.05, w_max * v / vmax)
        rrect(s, x_bar, yy + 0.03, bw, bh - 0.06, fill=color, radius=0.06)
        tb(s, x_bar + bw + 0.08, yy, 0.9, bh, [fmt.format(v)], size=size, bold=True, color=PURP, anchor="m")


def stat(s, x, y, w, h, big, label, big_color=PURP, fill=LAV_L, big_size=24, label_size=10.5):
    box(s, x, y, w, h, [(big, {"size": big_size, "bold": True, "color": big_color, "align": "c"}),
                        (label, {"size": label_size, "color": MUTED, "align": "c"})], fill=fill, radius=0.14)


# --- 6. Задача
s = new_slide("ЗАДАЧА", dark=True, notes=(
    "Задача ТЗ — сервис оценки качества исследований плотности костей. На входе DICOM-исследование, "
    "на выходе таблица ТЗ 2.5 и дополнительная серия, где видно, по какому измерению принято решение. "
    "Главное свойство данных — позитивов единицы, а меток на уровне пикселя нет."))
card(s, 0.38, Y0, 6.05, 5.46, True)
head(s, "Зачем проверять качество", 0.63, 1.5, 5.6)
tb(s, 0.63, 2.02, 5.55, 2.6, [
    "Денситометрия (DXA) измеряет **минеральную плотность кости** поясничного отдела позвоночника "
    "и бедра — по ней оценивают риск остеопороза",
    "Ошибка укладки, наклон оси, ротация бедра или артефакт в зоне интереса **искажают плотность** и заключение",
    "Ошибку выгодно поймать, **пока пациент в кабинете**: снимок переснимают сразу, без повторного визита",
], size=13.5, bullets=True, space_after=14)
box(s, 0.63, 4.78, 5.55, 1.75, [
    ("Что требует ТЗ", {"size": 13, "bold": True, "color": PURP, "space_after": 4}),
    ("На входе — исследование в DICOM. На выходе — класс качества, **тип нарушения** по критериям "
     "ТЗ 2.3, таблица результатов по форме ТЗ 2.5 и визуализация", {"size": 12}),
], fill=LAV_L, anchor="m", margin=0.2, radius=0.14)

card(s, 6.63, Y0, 6.33, 5.46, True)
head(s, "Что есть в обучающих данных", 6.88, 1.5, 5.8)
bw3 = (5.83 - 0.3) / 3
for i, (big, lab, col) in enumerate([("100", "исследований", PURP), ("249", "размеченных\nснимков", PURP),
                                      ("6–36", "нарушений\nна тип", PINK)]):
    stat(s, 6.88 + i * (bw3 + 0.15), 2.02, bw3, 1.2, big, lab, big_color=col, big_size=26)
tb(s, 6.88, 3.48, 5.83, 1.85, [
    "Контуров и ключевых точек нет: метки стоят на уровне **«исследование × область»**",
    "PixelSpacing и теги области **пустые во всех 499 файлах**",
    "На редкие типы нарушений — **6–10 примеров**",
], size=13, bullets=True, space_after=10)
box(s, 6.88, 5.38, 5.83, 1.17, [
    "Обучать сегментацию не на чем → переводим каждый критерий ТЗ 2.3 в **измерение на снимке** "
    "и калибруем порог по разметке эксперта"], fill=PINK, color=WHITE, em=WHITE, size=13, margin=0.2, radius=0.14)

# --- 7. Подход
s = new_slide("ПОДХОД", dark=False, notes=(
    "Три правила — геометрия маски кости, артефакты — один физический признак с обученным порогом, "
    "ротация — две границы по профилю края. Каждое решение можно показать на снимке. "
    "Нейросети проверены на тех же фолдах и не выиграли: выбор порога на единицах примеров съедает их преимущество."))
card(s, X0, Y0, 12.58, 2.2, False)
head(s, "Как строится каждая проверка — пример: наклон оси позвоночника", 0.63, 1.47, 12.0, size=15)
steps = [("Требование ТЗ 2.3", "«ось позвоночника отклонена не более чем на 5°»"),
         ("Признак на маске кости", "угол главной оси маски позвоночника"),
         ("Порог по эксперту", "4.5° — максимум F1 на обучающих фолдах"),
         ("Измерение на снимке", "линия оси и угол на overlay-серии")]
for i, (hname, d) in enumerate(steps):
    x = 0.63 + i * 3.1
    box(s, x, 2.0, 2.75, 0.6, [hname], fill=PURP, color=WHITE, bold=True, size=13, align="c", radius=0.14)
    tb(s, x, 2.72, 2.75, 0.7, [d], size=12, color=INK, align="c")
    if i < 3:
        line(s, x + 2.8, 2.3, x + 3.05, 2.3, color=PINK, w=1.5, arrow=True)

card(s, X0, 3.72, 6.2, 3.08, False)
head(s, "Нарушений в обучающем наборе", 0.63, 3.87, 5.7, size=15)
tb(s, 0.63, 4.25, 5.7, 0.3, ["нарушение есть у 73 снимков из 249"], size=11, color=MUTED)
hbars(s, [("Ротация бедра", 36), ("Артефакты", 17), ("Наклон оси", 10), ("Охват области", 7), ("Укладка", 6)],
      0.63, 1.95, 2.65, 3.1, 36, 4.7, 0.4, bh=0.32, size=12)

card(s, 6.78, 3.72, 6.18, 3.08, False)
head(s, "Что обучено по данным", 7.03, 3.87, 5.7, size=15)
tb(s, 7.03, 4.35, 5.7, 2.4, [
    "Артефакты: **логистическая регрессия** на одном признаке top-hat, веса 1.6 КБ",
    "Ротация: **выбор признака из четырёх и две границы** нормы, 1.2 КБ",
    "Укладка, ось, охват: **пороги геометрии**",
    "Нейросети проверены на тех же фолдах и **не выиграли** (слайд 11); на инференсе нет torch и GPU",
], size=12.5, bullets=True, space_after=8)

# --- 8. Архитектура
s = new_slide("АРХИТЕКТУРА", dark=True, notes=(
    "Командная строка, API и веб-интерфейс вызывают один и тот же batch.run, таблицы совпадают построчно — "
    "это проверяет тест. Ошибка на файле — строка Failure с причиной, остальные файлы обрабатываются. "
    "Базовый образ закреплён по sha256."))
card(s, X0, Y0, 12.58, 2.35, True)
head(s, "Конвейер: один для командной строки, API и веб-интерфейса", 0.63, 1.47, 12.0, size=15)
flow = [("Архив или каталог", "zip с именами в cp866, безопасная распаковка"),
        ("Чтение DICOM", "pydicom и резервный ридер без зависимостей"),
        ("Дедупликация", "решение один раз на уникальный кадр"),
        ("Область и проекция", "по геометрии кадра: 0 ошибок на 252 кадрах"),
        ("Маска кости", "Otsu на 3 класса и ограниченный гистерезис"),
        ("5 проверок ТЗ 2.3", "позвоночник — 3, бедро — 2")]
fw, fg = 1.8, 0.256
for i, (hname, d) in enumerate(flow):
    x = 0.63 + i * (fw + fg)
    last = i == len(flow) - 1
    box(s, x, 1.97, fw, 0.64, [hname], fill=PINK if last else LAV_L, color=WHITE if last else PURP,
        bold=True, size=12, align="c", radius=0.14, margin=0.06)
    tb(s, x, 2.72, fw, 0.85, [d], size=10.5, color=MUTED, align="c")
    if not last:
        line(s, x + fw + 0.03, 2.29, x + fw + fg - 0.03, 2.29, color=PINK, w=1.25, arrow=True)

card(s, X0, 3.87, 6.2, 2.93, True)
head(s, "Что получает пользователь", 0.63, 4.02, 5.7, size=15)
for i, t_ in enumerate(["**result.xlsx, result.csv** — таблица ТЗ 2.5 и балл нарушения, строка на файл",
                        "**overlays.zip** — overlay DICOM и PNG, текстовый отчёт DICOM SR",
                        "**diagnostics.csv** — измерения против порогов для каждого решения"]):
    box(s, 0.63, 4.5 + i * 0.75, 5.7, 0.64, [t_], fill=LAV_L, size=11.5, radius=0.14, margin=0.15)
card(s, 6.78, 3.87, 6.18, 2.93, True)
head(s, "Как запускается", 7.03, 4.02, 5.7, size=15)
for i, t_ in enumerate(["**./run.sh** — пакет: каталог или zip → таблица и overlay",
                        "**POST /batch, /predict** — HTTP API, описание в Swagger /docs",
                        "**Веб-интерфейс** — загрузка, фильтры, просмотр overlay",
                        "**Docker, --network none** — без сети, не root, версии закреплены"]):
    box(s, 7.03, 4.5 + i * 0.56, 5.68, 0.48, [t_], fill=LAV_L, size=11.5, radius=0.14, margin=0.15)

# --- 9. Данные и разбиение
s = new_slide("ДАННЫЕ И РАЗБИЕНИЕ", dark=False, notes=(
    "Разбиение по исследованию, а не по снимку: снимки одного сеанса скоррелированы, иначе была бы утечка. "
    "PatientID обезличен — повторные визиты неотличимы, это в ограничениях. Масштаба в тегах нет, "
    "поэтому источник масштаба выбирается по цепочке, а порог охвата калиброван с запасом."))
cw3 = (12.58 - 0.4) / 3
xs = [X0 + i * (cw3 + 0.2) for i in range(3)]
for x in xs:
    card(s, x, Y0, cw3, 5.48, False)
head(s, "Состав набора", xs[0] + 0.25, 1.47, cw3 - 0.5, size=15)
fun = [("499", "DICOM-файлов", 3.56), ("252", "уникальных кадра", 3.2), ("249", "размеченных снимков", 2.84),
       ("73", "с нарушением", 2.48)]
for i, (b, l, w) in enumerate(fun):
    box(s, xs[0] + (cw3 - w) / 2, 1.97 + i * 0.56, w, 0.46, [f"**{b}**  {l}"], fill=PINK_L if i == 3 else LAV_L,
        size=12, align="c", radius=0.14, margin=0.05)
head(s, "Снимки по областям", xs[0] + 0.25, 4.3, cw3 - 0.5, size=13)
hbars(s, [("Позвоночник", 99), ("Левое бедро", 78), ("Правое бедро", 72)], xs[0] + 0.25, 1.5, xs[0] + 1.8,
      1.65, 99, 4.75, 0.42, bh=0.32, color=LAV, size=11)
tb(s, xs[0] + 0.25, 6.12, cw3 - 0.5, 0.5, ["100 исследований, один аппарат GE Lunar"], size=10.5, color=MUTED)

head(s, "Разбиение: 5 фолдов", xs[1] + 0.25, 1.47, cw3 - 0.5, size=15)
folds = [22, 21, 20, 19, 18]
base_y, hmax = 3.5, 1.05
for i, v in enumerate(folds):
    x = xs[1] + 0.5 + i * 0.66
    hh = hmax * v / 22
    rrect(s, x, base_y - hh, 0.42, hh, fill=LAV, radius=0.06)
    tb(s, x - 0.15, base_y - hh - 0.3, 0.72, 0.28, [str(v)], size=11, bold=True, color=PURP, align="c")
    tb(s, x - 0.15, base_y + 0.04, 0.72, 0.28, [f"фолд {i + 1}"], size=9.5, color=MUTED, align="c")
tb(s, xs[1] + 0.25, 3.84, cw3 - 0.5, 0.28, ["исследований в фолде"], size=10, color=MUTED, align="c")
tb(s, xs[1] + 0.25, 4.25, cw3 - 0.5, 2.5, [
    "Единица — **исследование**: позвоночник и оба бедра пациента в одном фолде",
    "Стратификация по наличию бедра и нарушений каждой области",
    "Пороги и выбор признаков — **только на обучающих фолдах**",
    "Seed 42, split.csv в git — разбиение воспроизводимо",
], size=11.5, bullets=True, space_after=4)

head(s, "Предобработка", xs[2] + 0.25, 1.47, cw3 - 0.5, size=15)
tb(s, xs[2] + 0.25, 1.97, cw3 - 0.5, 4.7, [
    "**Масштаб 0.6 мм/пиксель.** PixelSpacing нет — берём Exposed Area и паспорт аппарата",
    "**Область по геометрии кадра.** Теги области пустые во всех файлах",
    "**Метки на снимки** — паросочетанием внутри исследования, сверено глазами на 252 кадрах",
    "**Один вид пикселей.** MONOCHROME1, 16 бит, Big Endian дают те же решения",
    "**Дедупликация.** 247 файлов — копии; строка в таблице остаётся на каждый файл",
], size=12.5, bullets=True, space_after=12)

# --- 10. Таксономия
s = new_slide("ТАКСОНОМИЯ НАРУШЕНИЙ", dark=True, notes=(
    "Метки один к одному со столбцами разметки организатора. В 3 исследованиях из 100 итог эксперта расходится "
    "с объединением по компонентам — из-за сколиоза и перелома, которых нет в перечне 2.3; мы держимся перечня ТЗ."))
card(s, X0, Y0, 8.1, 5.48, True)
head(s, "Пять типов нарушений по ТЗ 2.3", 0.63, 1.47, 7.6, size=15)
box(s, 3.13, 1.97, 2.6, 0.48, ["DICOM-исследование"], fill=PURP, color=WHITE, bold=True, size=12.5, align="c",
    radius=0.14)
line(s, 3.85, 2.47, 2.55, 2.84, color=PINK, w=1.25, arrow=True)
line(s, 5.01, 2.47, 6.31, 2.84, color=PINK, w=1.25, arrow=True)
box(s, 0.63, 2.86, 3.75, 0.5, ["Поясничный отдел позвоночника"], fill=LAV_L, color=PURP, bold=True, size=11.5,
    align="c", radius=0.14)
box(s, 4.48, 2.86, 3.75, 0.5, ["Проксимальный отдел бедра (левое и правое)"], fill=LAV_L, color=PURP, bold=True,
    size=11.5, align="c", radius=0.14)


def code_box(x, y, code, desc):
    box(s, x, y, 3.75, 0.56, [(code, {"size": 10.5, "bold": True, "color": PURP}),
                              (desc, {"size": 10.5, "color": MUTED})], fill=WHITE, line=LAV, radius=0.14,
        margin=0.15, align="l")


for i, (c, d) in enumerate([("SPINE_POSITIONING", "укладка пациента"), ("SPINE_AXIS_TILT", "наклон оси позвоночника"),
                            ("SPINE_ARTIFACT", "артефакты и наложения")]):
    code_box(0.63, 3.5 + i * 0.66, c, d)
for i, (c, d) in enumerate([("FEMUR_POSITIONING_ROTATION", "позиционирование, ротация"),
                            ("FEMUR_ROI_COVERAGE", "охват области интереса")]):
    code_box(4.48, 3.5 + i * 0.66, c, d)
line(s, 0.63, 5.55, 8.23, 5.55, color=LINE_L, w=1, dash="dash")
tb(s, 0.63, 5.66, 7.6, 1.05, [
    "Снимок: **quality_class = 1**, если сработало хотя бы одно правило его области",
    "**violation_type** — все сработавшие типы через «;» в порядке таксономии",
    "Исследование — с нарушением, если оно есть **хотя бы на одном снимке**",
], size=11, bullets=True, space_after=2)

card(s, 8.68, Y0, 4.28, 5.48, True)
head(s, "Несколько нарушений", 8.93, 1.47, 3.8, size=15)
framed(s, A + "case_multi.png", 8.93, 1.95, 3.78, 3.0)
box(s, 8.93, 5.07, 3.78, 0.78, [("violation_type", {"size": 9.5, "color": MUTED}),
                                ("FEMUR_POSITIONING_ROTATION;", {"size": 10, "bold": True, "color": PINK}),
                                ("FEMUR_ROI_COVERAGE", {"size": 10, "bold": True, "color": PINK})],
    fill=LAV_L, radius=0.14, margin=0.15)
tb(s, 8.93, 5.95, 3.78, 0.8, ["Левое бедро: поле 125 мм при пороге 133 мм и выступ вертела вне нормы. "
                              "В наборе таких снимков 4"], size=10.5, color=MUTED)

# --- 11. Сравнение вариантов
s = new_slide("СРАВНЕНИЕ ВАРИАНТОВ", dark=False, notes=(
    "Все варианты — на одних 249 снимках и одних фолдах. Сервис 0.60 [0.51; 0.68], лучше ответа «всё — нарушение» "
    "на +0.14 [+0.05; +0.24]. ResNet18 на области вертела — 0.63, разница с сервисом +0.03 [−0.04; +0.10], "
    "в пределах шума, и нужен torch в образе. Заполненность маски отклонена: она мерит плотность кости, а не укладку."))
card(s, X0, Y0, 7.9, 5.48, False)
head(s, "F1 «есть нарушение»: 249 снимков, одни и те же 5 фолдов", 0.63, 1.47, 7.4, size=14)
rows = [("ResNet18 на области вертела", 0.626, 0.533, 0.714, "exp"),
        ("Сервис: правила + ротация", 0.598, 0.510, 0.682, "svc"),
        ("Обучаемая голова на геометрии", 0.561, 0.481, 0.638, ""),
        ("Ротация в точке локализатора v2", 0.558, 0.476, 0.638, ""),
        ("Гибрид: геометрия + пиксели", 0.506, 0.422, 0.588, ""),
        ("Правила без ротации", 0.504, 0.397, 0.608, ""),
        ("Классификатор по пикселям", 0.479, 0.400, 0.554, ""),
        ("Всё — нарушение (база)", 0.453, 0.408, 0.500, "base")]
px0, px1, v0, v1, ytop, rh = 4.15, 7.35, 0.35, 0.75, 2.02, 0.5


def X(v):
    return px0 + (v - v0) / (v1 - v0) * (px1 - px0)


for v in (0.4, 0.5, 0.6, 0.7):
    line(s, X(v), ytop - 0.08, X(v), ytop + rh * len(rows) - 0.08, color=GREY, w=0.75)
    tb(s, X(v) - 0.3, ytop + rh * len(rows) - 0.04, 0.6, 0.26, [f"{v:.1f}"], size=9.5, color=MUTED, align="c")
line(s, X(0.453), ytop - 0.08, X(0.453), ytop + rh * len(rows) - 0.08, color=MUTED, w=1, dash="dash")
for i, (name, f, lo, hi, kind) in enumerate(rows):
    yc = ytop + i * rh + rh / 2 - 0.08
    svc = kind == "svc"
    if svc:
        rrect(s, 0.55, yc - 0.23, 7.56, 0.46, fill=PINK_L, radius=0.23)
    tb(s, 0.7, yc - 0.22, 3.4, 0.44, [name], size=11, bold=svc, color=PURP if svc else INK, anchor="m")
    col = PINK if svc else (PURP if kind == "exp" else ("A9A4B8" if kind == "base" else LAV))
    line(s, X(lo), yc, X(hi), yc, color=col, w=2)
    dot = rrect(s, X(f) - 0.09, yc - 0.09, 0.18, 0.18, fill=WHITE if kind == "exp" else col, line=col, lw=1.75,
                shape=MSO_SHAPE.OVAL)
    tb(s, X(hi) + 0.06, yc - 0.18, 0.55, 0.36, [f"{f:.2f}"], size=11, bold=svc or kind == "exp",
       color=PINK if svc else INK, anchor="m")
tb(s, 0.63, 6.38, 7.4, 0.3, ["точка — F1, отрезок — 95% ДИ (бутстреп по исследованиям), пунктир — база"],
   size=9.5, color=MUTED, italic=True)

card(s, 8.48, Y0, 4.48, 5.48, False)
head(s, "По задачам: F1", 8.73, 1.47, 4.0, size=15)
TAGC = {"в сервисе": PINK, "отклонён": "A9A4B8", "не прошёл": "A9A4B8", "эксперимент": LAV}


def task_rows(title, lst, y):
    tb(s, 8.73, y, 4.0, 0.32, [title], size=11.5, bold=True, color=PURP, anchor="m")
    for i, (n, v, st) in enumerate(lst):
        yy = y + 0.36 + i * 0.335
        on = st == "в сервисе"
        tb(s, 8.73, yy, 2.4, 0.3, [n], size=10.5, bold=on, color=INK, anchor="m")
        tb(s, 11.05, yy, 0.45, 0.3, [v], size=10.5, bold=on, color=PINK if on else INK, align="r", anchor="m")
        if st:
            tag(s, st, 11.6, yy + 0.03, 1.12, TAGC[st], size=8.5, h=0.25)


task_rows("Артефакты позвоночника, 17 позитивов",
          [("top-hat, 1 признак", "0.70", "в сервисе"), ("гребень (гессиан)", "0.57", ""),
           ("top-hat + гребень", "0.65", ""), ("HOG, 144 признака", "0.01", "")], 1.95)
task_rows("Ротация бедра, 36 позитивов",
          [("угол оси диафиза", "0.42", ""), ("угол + заполненность", "0.49", "отклонён"),
           ("пиксели бедра, 384", "0.26", ""), ("ResNet18, весь снимок", "0.36", ""),
           ("профиль края", "0.48", "в сервисе"), ("профиль в точке v2", "0.43", "не прошёл"),
           ("ResNet18, вертел", "0.50", "эксперимент")], 3.7)

# --- 12. Анализ ошибок
s = new_slide("АНАЛИЗ ОШИБОК", dark=True, notes=(
    "Сколиоз: признаки кривизны его от наклона не отделяют, поэтому правило не трогаем и пишем в ограничения. "
    "Рёбра: без них F1 артефактов упал бы с 0.71 до 0.26. Ротация: локализатор v2 по 58 точкам врача ставит точку "
    "в 3.7 раза точнее, но правило в этой точке хуже; критерий включения зафиксировали в git до проверки — "
    "и v2 не включили."))
ec = [("case_scol.png", "Наклон оси при сколиозе", PINK, "ложное срабатывание",
       "**4 из 11** ложных срабатываний оси — сколиоз. Все 14 «сколиозов» в разметке — норма по оси: "
       "эксперт считает искривление анатомией"),
      ("err_ribs.png", "Артефакт — чаще всего рёбра", LAV, "верное срабатывание",
       "**9 из 12** верных срабатываний — наложение рёбер: эксперт отмечает **44%** кадров с рёбрами "
       "против 8% без них"),
      ("err_rot.png", "Ротация: промах точки вертела", PINK, "пропуск",
       "Точка измерения в **9 мм** от точки врача (медиана). У 11 из 15 пропусков вертел хорошо виден, "
       "а окно стоит ниже, на диафизе")]
for i, (f, hname, col, tg, d) in enumerate(ec):
    x = xs[i]
    card(s, x, Y0, cw3, 3.98, True)
    framed(s, A + f, x + 0.2, 1.52, cw3 - 0.4, 1.95)
    tag(s, tg, x + 0.2, 3.58, 2.15, col, size=9.5)
    tb(s, x + 0.2, 3.95, cw3 - 0.4, 0.35, [hname], size=13, bold=True, color=PURP, anchor="m")
    tb(s, x + 0.2, 4.33, cw3 - 0.4, 0.95, [d], size=10.5, color=INK)
card(s, X0, 5.5, 12.58, 1.3, True)
tb(s, 0.63, 5.5, 3.9, 1.3, ["Проверили и не включили: точный локализатор малого вертела"], size=13, bold=True,
   color=PURP, anchor="m")
for i, (v, l) in enumerate([("9.0 → 2.5 мм", "медиана ошибки точки"), ("26% → 84%", "точек в пределах 5 мм"),
                            ("0.48 → 0.43", "F1 правила ротации"), ("F1 ≥ 0.53", "порог включения")]):
    stat(s, 4.7 + i * 2.05, 5.66, 1.9, 0.98, v, l, big_color=PINK if i == 2 else PURP, big_size=15, label_size=9.5)

# --- 13. Метрики
s = new_slide("МЕТРИКИ КАЧЕСТВА", dark=False, notes=(
    "Для геометрических правил порог подбирается на четырёх фолдах и применяется к пятому, ротация — вложенная "
    "проверка. Итоговые метрики на закрытых данных считает организатор. Метрика ключевой точки: единственная "
    "точка, которую ставит сервис, — малый вертел на overlay."))
card(s, X0, Y0, 8.75, 5.48, False)
head(s, "Пороги без проверочного фолда · ДИ 95% — бутстреп по исследованиям", 0.63, 1.47, 8.3, size=13)
hdr = ["Срез", "Поз.", "Чувств.", "Спец.", "Сб. т.", "F1 [95% ДИ]", "AUC"]
mrows = [["Позвоночник: укладка", "6", "0.67", "1.00", "0.83", "0.80 [0.50; 1.00]", "0.88"],
         ["Позвоночник: наклон оси", "10", "0.60", "0.80", "0.70", "0.35 [0.18; 0.52]", "0.88"],
         ["Позвоночник: артефакты", "17", "0.71", "0.94", "0.82", "0.71 [0.53; 0.86]", "0.84"],
         ["Бедро: охват ROI", "7", "0.43", "0.99", "0.71", "0.50 [0.00; 0.82]", "0.94"],
         ["Бедро: ротация", "36", "0.56", "0.75", "0.65", "0.48 [0.34; 0.61]", "0.67"],
         ["Позвоночник, есть нарушение", "32", "0.78", "0.76", "0.77", "0.68 [0.57; 0.79]", "0.81"],
         ["Бедро, есть нарушение", "41", "0.59", "0.76", "0.67", "0.53 [0.40; 0.65]", "0.74"],
         ["Все снимки", "73", "0.67", "0.76", "0.72", "0.60 [0.51; 0.68]", "0.78"],
         ["По исследованиям", "50", "0.86", "0.54", "0.70", "0.74 [0.67; 0.81]", "0.71"]]
colw = [2.95, 0.55, 0.78, 0.7, 0.7, 1.82, 0.6]
gt = s.shapes.add_table(len(mrows) + 1, len(hdr), In(0.6), In(1.95), In(sum(colw)), In(0.38 * (len(mrows) + 1)))
tbl = gt.table
tblPr = tbl._tbl.tblPr
for k in ("firstRow", "bandRow"):
    tblPr.set(k, "0")
sty = tblPr.find(qn("a:tableStyleId"))
if sty is not None:
    sty.text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"  # No Style, No Grid
for j, w in enumerate(colw):
    tbl.columns[j].width = In(w)


def cell_set(c, text, bold=False, color=INK, fill=WHITE, align="c", size=10.5):
    c.fill.solid()
    c.fill.fore_color.rgb = rgb(fill)
    c.margin_left = c.margin_right = In(0.06)
    c.margin_top = c.margin_bottom = In(0.02)
    c.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf = c.text_frame
    tf.text = ""
    p = tf.paragraphs[0]
    p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER}[align]
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = rgb(color)


def cell_border(c, color=LINE_L, w=0.75):
    tcPr = c._tc.get_or_add_tcPr()
    for tagn in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
        ln = etree.SubElement(tcPr, qn(tagn))
        ln.set("w", str(int(Pt(w))))
        sf = etree.SubElement(ln, qn("a:solidFill"))
        c_ = etree.SubElement(sf, qn("a:srgbClr"))
        c_.set("val", color)
    # границы должны идти до заливки
    fill = tcPr.find(qn("a:solidFill"))
    if fill is not None:
        tcPr.remove(fill)
        tcPr.append(fill)


for j, h_ in enumerate(hdr):
    cell_set(tbl.cell(0, j), h_, bold=True, color=WHITE, fill=PURP, align="l" if j == 0 else "c")
    cell_border(tbl.cell(0, j))
for i, r_ in enumerate(mrows, start=1):
    agg = i >= 6
    for j, v in enumerate(r_):
        cell_set(tbl.cell(i, j), v, bold=agg and j in (0, 5), color=PINK if (agg and j == 5) else INK,
                 fill=LAV_L if agg else WHITE, align="l" if j == 0 else "c")
        cell_border(tbl.cell(i, j))
for rr in tbl.rows:
    rr.height = In(0.4)
tb(s, 0.63, 6.08, 8.3, 0.65, ["macro-F1 по пяти типам — 0.57. AUC агрегатов — по баллу quality_score (вложенно), "
                              "у ротации — по выходу за границы нормы. Точка малого вертела — в 9.0 мм от точки врача (медиана)"],
   size=10, color=MUTED)
card(s, 9.33, Y0, 3.63, 5.48, False)
head(s, "Главное", 9.58, 1.47, 3.2, size=15)
for i, (b, l, col) in enumerate([("0.60", "F1 по снимкам\n[0.51; 0.68]", PURP), ("0.74", "F1 по исследованиям\n[0.67; 0.81]", PURP),
                                  ("0.78", "ROC-AUC по баллу, снимки\n[0.70; 0.85]", PURP),
                                  ("+0.14", "F1 к базе «всё — нарушение»\n[+0.05; +0.24]", PINK)]):
    stat(s, 9.55, 1.95 + i * 1.19, 3.19, 1.07, b, l, big_color=col, big_size=22, label_size=10)

# --- 14. Скорость и требования
s = new_slide("СКОРОСТЬ И ТРЕБОВАНИЯ", dark=True, notes=(
    "Время на исследование без overlay — около 0.1 с, с overlay-серией 0.3 с. Образ собран под arm64 и amd64 — "
    "на обучающем наборе таблицы совпадают во всех строках. Архив overlay пишется потоком, поэтому память "
    "почти не зависит от размера набора. Стенд организатора с H200 избыточен: GPU не используется."))
card(s, X0, Y0, 6.2, 2.68, True)
head(s, "Весь набор: 100 исследований, 499 файлов", 0.63, 1.47, 5.7, size=14)
hbars(s, [("без overlay", 7), ("overlay PNG", 20), ("overlay PNG + DICOM", 30)], 0.63, 2.2, 2.9, 2.55, 30,
      2.0, 0.46, bh=0.34, color=PURP, size=11, fmt="{} с")
tb(s, 0.63, 3.45, 5.7, 0.4, ["пиковая память: 225 МБ без overlay, 242 МБ с ним · MacBook Air M2, Linux-VM"],
   size=9.5, color=MUTED)
card(s, 6.78, Y0, 6.18, 2.68, True)
head(s, "В цифрах", 7.03, 1.47, 5.7, size=15)
for i, (b, l) in enumerate([("~0.1 с", "на исследование,\n0.3 с с overlay"), ("0.6 с", "самое долгое,\nкадр 2048×2048"),
                            ("3 мин", "лимит ТЗ 2.7")]):
    stat(s, 7.03 + i * 1.92, 1.95, 1.8, 1.22, b, l, big_color=MUTED if i == 2 else PURP, big_size=21, label_size=10)
box(s, 7.03, 3.3, 5.68, 0.5, ["**499 из 499** файлов обработаны со статусом Success"], fill=PINK_L, size=12,
    align="c", radius=0.14)
card(s, X0, 4.2, 7.4, 2.6, True)
head(s, "Системные требования (ТЗ 3.1)", 0.63, 4.33, 6.9, size=15)
req = [["", "минимум", "рекомендуется"], ["Процессор", "1 ядро x86_64 или arm64", "2 ядра и больше"],
       ["Память", "1 ГБ", "2 ГБ"], ["Диск", "образ + 0.3 МБ на снимок", "то же"], ["GPU", "не нужен", "не нужен"],
       ["ПО", "Docker 20+", "Docker 24+"]]
rcw = [1.4, 2.9, 2.6]
gt = s.shapes.add_table(len(req), 3, In(0.63), In(4.8), In(sum(rcw)), In(0.31 * len(req)))
tbl = gt.table
tblPr = tbl._tbl.tblPr
for k in ("firstRow", "bandRow"):
    tblPr.set(k, "0")
sty = tblPr.find(qn("a:tableStyleId"))
if sty is not None:
    sty.text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"
for j, w in enumerate(rcw):
    tbl.columns[j].width = In(w)
for i, r_ in enumerate(req):
    for j, v in enumerate(r_):
        if i == 0:
            cell_set(tbl.cell(i, j), v, bold=True, color=WHITE, fill=PURP, align="l", size=10.5)
        else:
            cell_set(tbl.cell(i, j), v, bold=j == 0, color=INK, fill=WHITE, align="l", size=10.5)
        cell_border(tbl.cell(i, j))
for rr in tbl.rows:
    rr.height = In(0.31)
card(s, 7.98, 4.2, 4.98, 2.6, True)
head(s, "Стресс-тест: 77 файлов", 8.23, 4.33, 4.5, size=15)
tb(s, 8.23, 4.85, 4.5, 1.85, [
    "Пустые, битые, цветные, сжатые, 1×1, 2048×2048 — **0 падений**",
    "MONOCHROME1, 16 бит, Big Endian — **те же решения**",
    "Ошибка файла → строка **Failure** с причиной, остальные обрабатываются",
], size=11.5, bullets=True, space_after=6)

# --- 15. Ограничения
s = new_slide("ОГРАНИЧЕНИЯ", dark=False, notes=(
    "Полный список — 14 пунктов в README, раздел «Ограничения». Из дополнительного функционала ТЗ 2.6 "
    "не реализована только автокоррекция разметки: во входных DICOM разметки нет, исправлять нечего."))
cols = [(0.9, 3.8, "Ограничение"), (4.85, 4.05, "Почему"), (9.05, 3.91, "Что делаем")]
for x, w, t_ in cols:
    box(s, x, Y0, w, 0.46, [t_], fill=PURP, color=WHITE, bold=True, size=13, align="c", radius=0.14)
lim = [("**Th12 не проверяется** — только нижняя граница по гребням подвздошных костей",
        "В наборе **2 позитива**, ни одна из семи мер их не отделяет", "Описано в README; нужна разметка Th12 врачом"),
       ("**Сколиоз** даёт ложный наклон оси", "Эксперт считает искривление анатомией; признаки кривизны его не отделяют",
        "Правило не меняем; при известном сколиозе — перепроверка оси"),
       ("**Ротация — самое слабое правило**, F1 0.48", "Точка вертела в 9 мм от точки врача; без правила бедро ниже базы",
        "Точный локализатор есть; следующий шаг — классификатор вокруг его точки"),
       ("**Охват: порог 133 мм** вместо буквальных 150", "«3 + 3 см» и «2 см сбоку» противоречат разметке (AUC 0.17)",
        "Порог калиброван по эксперту; отступ сбоку — справочно на overlay"),
       ("**Масштаб** восстанавливается косвенно", "PixelSpacing нет ни в одном файле; весь набор — аппарат GE Lunar",
        "Для другого аппарата — калибровка порогов по docs/training.md"),
       ("**100 исследований**, разбиение по исследованию", "PatientID обезличен: повторные визиты неотличимы",
        "Независимую оценку дадут закрытый набор и пилот")]
for i, row in enumerate(lim):
    y = 1.92 + i * 0.81
    tb(s, 0.38, y, 0.45, 0.7, [str(i + 1)], size=18, bold=True, color=PINK, align="c", anchor="m")
    for (x, w, _), t_ in zip(cols, row):
        box(s, x, y, w, 0.7, [t_], fill=LAV_L if x > 9 else WHITE, line=LAV, size=11, radius=0.14, margin=0.14)

# --- 16. Внедрение и эффект
s = new_slide("ВНЕДРЕНИЕ И ЭФФЕКТ", dark=True, notes=(
    "Сейчас сервис принимает пакет из каталога или архива и через HTTP API; приёма по DICOM-протоколу нет — "
    "это доработка для интеграции с PACS. Эффект в цифрах (доля пересъёмок, повторных визитов) — задача пилота."))
card(s, X0, Y0, 4.3, 5.48, True)
head(s, "Как встраивается", 0.63, 1.47, 3.8, size=15)
fl = ["Аппарат DXA", "PACS или папка обмена", "DXA QC в контуре медорганизации", "Overlay-серия в PACS рядом со снимком",
      "Лаборант переснимает сразу"]
for i, t_ in enumerate(fl):
    y = 1.97 + i * 0.9
    hot = i == 2
    box(s, 0.63, y, 3.8, 0.6, [t_], fill=PINK if hot else LAV_L, color=WHITE if hot else PURP, bold=True, size=11.5,
        align="c", radius=0.14)
    if i < len(fl) - 1:
        line(s, 2.53, y + 0.62, 2.53, y + 0.88, color=PINK, w=1.25, arrow=True)
tb(s, 0.63, 6.4, 3.8, 0.3, ["снимки не покидают больницу"], size=10, color=MUTED, italic=True, align="c")

card(s, 4.88, Y0, 4.0, 5.48, True)
head(s, "Кому и зачем", 5.13, 1.47, 3.5, size=15)
who = [("Лаборант", "подсказка «переснять» до ухода пациента; нарушение показано на снимке"),
       ("Врач-рентгенолог", "не описывает заведомо некачественные снимки, видит причину брака"),
       ("Служба качества", "единые критерии ТЗ на всех аппаратах, ежемесячный аудит по таблице результатов")]
for i, (w_, d) in enumerate(who):
    y = 1.97 + i * 1.55
    tb(s, 5.13, y, 3.5, 0.35, [w_], size=13, bold=True, color=PINK)
    tb(s, 5.13, y + 0.38, 3.5, 1.0, [d], size=11.5, color=INK)
    if i < 2:
        line(s, 5.13, y + 1.42, 8.63, y + 1.42, color=LINE_L, w=0.75)

card(s, 9.08, Y0, 3.88, 5.48, True)
head(s, "Эффект", 9.33, 1.47, 3.4, size=15)
tb(s, 9.33, 1.97, 3.4, 3.9, [
    "Меньше **повторных визитов**: брак ловят в день исследования",
    "Меньше **недостоверных заключений** о плотности кости",
    "**Прозрачность**: каждое решение — измерение против порога",
    "**Дёшево в эксплуатации**: CPU, без GPU и облака",
], size=13, bullets=True, space_after=14)
tb(s, 9.33, 6.0, 3.4, 0.65, ["оценка эффекта в цифрах — задача пилота"], size=10, color=MUTED, italic=True)

# --- 17. Планы по развитию (по образцу слайда «Стадии» шаблона)
s = new_slide("ПЛАНЫ ПО РАЗВИТИЮ", dark=False, notes=(
    "План пилота — предложение команды. Для каждого этапа уже есть инструменты: калибровка порогов описана "
    "в docs/training.md и воспроизводится скриптами, точный локализатор вертела обучен и лежит в отчётах."))
stages = [("Пилот «в тени»", "1–2 аппарата, сервис работает параллельно, решения сверяем с врачом"),
          ("Калибровка", "разметка выборки экспертом, пороги и масштаб под каждый аппарат"),
          ("Модели", "разметка Th12; классификатор ротации вокруг точного локализатора вертела"),
          ("Интеграция", "приём исследований из PACS, overlay и отчёт SR обратно в PACS, подсказка лаборанту"),
          ("Масштабирование", "другие аппараты и медорганизации, ежемесячный аудит качества")]
cx = [1.78, 4.21, 6.67, 9.11, 11.54]
for i in range(4):
    line(s, cx[i] + 0.48, 4.22, cx[i + 1] - 0.48, 4.22, color=PINK, w=1.25)
for i, (hname, d) in enumerate(stages):
    box(s, cx[i] - 0.48, 3.84, 0.96, 0.77, [str(i + 1)], fill=WHITE, line=PINK, lw=1.25, color=PINK, bold=True,
        size=22, align="c", radius=0.14)
    top = i % 2 == 0
    y = 1.5 if top else 5.25
    box(s, cx[i] - 1.55, y, 3.1, 1.73, [(hname, {"size": 14, "bold": True, "color": PURP, "space_after": 4}),
                                        (d, {"size": 12, "color": INK})],
        fill=WHITE, line=PINK, lw=1.0, radius=0.16, margin=0.18, anchor="m")
    if top:
        line(s, cx[i], 3.23, cx[i], 3.84, color=PINK, w=1.25)
    else:
        line(s, cx[i], 4.61, cx[i], 5.25, color=PINK, w=1.25)

# --- 18. Кейсы
s = new_slide("КЕЙСЫ", dark=True, notes=(
    "Все примеры — из обучающего набора, прогон текущей версии сервиса. Последний кейс — наша ошибка: "
    "при сколиозе эксперт оценил ось как норму, сервис видит наклон. Overlay — DICOM Secondary Capture в том же "
    "исследовании, открывается в любом просмотрщике и в PACS."))
cases = [("case_norm_spine.png", "норма", LAV, "ось 1.6°, гребни в кадре"),
         ("case_axis.png", "наклон оси", PINK, "ось 6.7° при пороге 4.5°"),
         ("case_pos.png", "укладка", PINK, "гребней подвздошных костей нет"),
         ("case_art.png", "артефакт", PINK, "кружок — максимум отклика"),
         ("case_norm_femur.png", "норма", LAV, "бедро: поле 157 мм, вертел в норме"),
         ("case_multi.png", "2 нарушения", PINK, "ротация и поле 125 мм"),
         ("case_scol.png", "наша ошибка", DPURP, "сколиоз: эксперт — норма")]
cw4 = (12.58 - 3 * 0.2) / 4
ch = 2.64
for i, (f, tg, col, d) in enumerate(cases):
    x = X0 + (i % 4) * (cw4 + 0.2)
    y = Y0 + (i // 4) * (ch + 0.2)
    card(s, x, y, cw4, ch, True)
    framed(s, A + f, x + 0.12, y + 0.12, cw4 - 0.24, 1.98)
    tag(s, tg, x + 0.22, y + 1.78, 1.35, col, size=9.5, h=0.26)
    tb(s, x + 0.12, y + 2.14, cw4 - 0.24, 0.42, [d], size=10.5, color=INK, align="c", anchor="m")
x = X0 + 3 * (cw4 + 0.2)
y = Y0 + ch + 0.2
card(s, x, y, cw4, ch, True)
tb(s, x + 0.22, y + 0.2, cw4 - 0.4, 0.35, ["Как читать overlay"], size=13, bold=True, color=PURP)
leg = [("D9434A", "красное — сработавшее правило"), ("E3B23C", "жёлтое, голубое — измерение без нарушения"),
       ("2FA36B", "зелёное — контур кости"), (PURP, "в шапке — числа против порогов")]
for i, (c_, t_) in enumerate(leg):
    yy = y + 0.7 + i * 0.47
    rrect(s, x + 0.22, yy + 0.08, 0.2, 0.2, fill=c_, radius=0.04)
    tb(s, x + 0.55, yy, cw4 - 0.75, 0.42, [t_], size=10.5, color=INK, anchor="m")

# --- 19. Демонстрация
s = new_slide("ДЕМОНСТРАЦИЯ", dark=False, notes=(
    "Подробный сценарий с идентификаторами исследований — docs/demo_scenario.md. Тот же результат даёт API "
    "и командная строка: конвейер один."))
card(s, X0, Y0, 8.1, 5.48, False)
framed(s, A + "web_ui.png", 0.58, 1.52, 7.7, 4.42, frame=LINE_L, pad=0.04)
box(s, 0.58, 6.08, 7.7, 0.52, [('curl -F file=@архив.zip "localhost:8000/batch?overlays=true" -o result.zip',
                                {"size": 10.5, "color": PURP, "bold": True})], fill=LAV_L, radius=0.14, align="c")
card(s, 8.68, Y0, 4.28, 5.48, False)
head(s, "Сценарий показа, 6–8 минут", 8.93, 1.47, 3.8, size=14)
demo = [("./run.sh --serve", "сервис в контейнере без сети"),
        ("Перетащить архив", "весь набор организатора, 100 исследований"),
        ("Сводка и фильтры", "нарушения по типам, поиск по исследованию"),
        ("Открыть overlay", "шесть кейсов, скачать DICOM SC"),
        ("Скачать результат", "result.xlsx, overlays.zip с отчётами SR")]
for i, (hname, d) in enumerate(demo):
    y = 1.97 + i * 0.93
    box(s, 8.93, y + 0.14, 0.46, 0.46, [str(i + 1)], fill=PINK, color=WHITE, bold=True, size=13, align="c",
        radius=0.23, margin=0)
    tb(s, 9.55, y, 3.25, 0.8, [(hname, {"size": 12, "bold": True, "color": PURP}),
                              (d, {"size": 10.5, "color": MUTED})], anchor="m")

# --- 20. Спасибо
s = new_slide("", dark=True, bg=BG_CITY, pill=False)
tb(s, 0.6, 1.0, 7.0, 0.9, ["СПАСИБО!"], size=44, bold=True, color=WHITE)
pts = [("Всё обязательное по ТЗ", "две области, пять типов нарушений, таблица ТЗ 2.5 с баллом, overlay-серия "
                                  "и отчёт DICOM SR, API и веб-интерфейс, локально без сети"),
       ("Проверяемые цифры", "F1 0.60 по снимкам и 0.74 по исследованиям, ROC-AUC 0.78 на пяти фолдах; "
                             "отчёты воспроизводятся скриптами"),
       ("Объяснимость", "каждое решение — измерение против порога, нарисованное на снимке; ограничения измерены "
                        "и описаны")]
for i, (hname, d) in enumerate(pts):
    y = 2.3 + i * 1.3
    b = rrect(s, 0.6, y, 7.2, 1.1, fill=WHITE, radius=0.18, alpha=0.12)
    tb(s, 0.9, y, 6.7, 1.1, [(hname, {"size": 15, "bold": True, "color": WHITE, "space_after": 2}),
                            (d, {"size": 11.5, "color": "EADFF5"})], anchor="m")
tb(s, 0.6, 6.3, 7.5, 0.4, ["Команда «Тёмная материя»: **Фёдор Никонец, Илья Рыжиков**"], size=13, color=WHITE,
   em=WHITE)

# ============================================================ удаляем служебные слайды шаблона
sldIdLst = prs.slides._sldIdLst
ids = list(sldIdLst)
for i, sldId in enumerate(ids[:N_TEMPLATE]):
    if i not in KEEP:
        prs.part.drop_rel(sldId.rId)
        sldIdLst.remove(sldId)

# разделы: «Обязательные слайды» и «Решение»
pres_el = prs.part._element
P14 = "http://schemas.microsoft.com/office/powerpoint/2010/main"
secLst = pres_el.find(".//{%s}sectionLst" % P14)
if secLst is not None:
    secs = list(secLst)
    kept_ids = [e.get("id") for e in sldIdLst]
    for sec in secs:
        secLst.remove(sec)
    for name, rng, gid in (("Обязательные слайды", kept_ids[:5], "{248E5CD8-1028-4246-90C7-3263F7F286D3}"),
                           ("Решение", kept_ids[5:], "{7C1E3A52-2B8D-4E0A-9C61-5D2F0B7A1E40}")):
        sec = etree.SubElement(secLst, "{%s}section" % P14)
        sec.set("name", name)
        sec.set("id", gid)
        lst = etree.SubElement(sec, "{%s}sldIdLst" % P14)
        for i_ in rng:
            e = etree.SubElement(lst, "{%s}sldId" % P14)
            e.set("id", i_)

prs.core_properties.title = "DXA QC — оценка качества денситометрии"
prs.core_properties.author = "Команда «Тёмная материя»: Фёдор Никонец, Илья Рыжиков"
prs.save(OUT)
print("saved", OUT, "slides:", len(prs.slides))
