# docs/assets/ — картинки README и как они сделаны

[← README](../../README.md) · [Карта документации](../README.md)

В публичной копии нет снимков обучающего набора, поэтому все иллюстрации
сделаны на **синтетических фантомах**: позвоночник и бедро нарисованы
примитивами OpenCV и прогнаны через настоящий конвейер `dxa_qc` — роутер,
маску кости, правила и overlay-серию. Решения сервиса не подгонялись:
подбирались только картинки, на которых сервис выносит нужное для показа
решение (норма или конкретное нарушение).

| файл | что это | источник |
|---|---|---|
| `banner.webp` | баннер README | [`src/banner.html`](src/banner.html) |
| `social-preview.jpg` | превью ссылки на репозиторий (Settings → Social preview) | [`src/banner.html`](src/banner.html) |
| `ui_light.webp`, `ui_dark.webp` | веб-интерфейс в светлой и тёмной теме | [`src/shot.py`](src/shot.py) |
| `overlays.webp` | галерея overlay-серии: норма и четыре типа нарушений | [`src/gallery.html`](src/gallery.html) |

## Как пересобрать

Нужны окружение разработки, Playwright с Chromium и шрифт Inter.

```bash
# 1. шесть синтетических исследований, прогон через сервис, overlay
python docs/assets/src/make_demo.py docs/assets/src/build

# 2. скриншоты веб-интерфейса — сервис должен быть запущен
python -m dxa_qc.service --port 8000 &
python docs/assets/src/shot.py http://localhost:8000 docs/assets/src/build/demo.zip docs/assets/src/build

# 3. баннер, превью и галерея
cd docs/assets/src
python render.py banner.html  build/banner.png  '#b' 1280 400 2
python render.py banner.html  build/social.png  '#b' 1280 640 1 '.b{--h:640px;border-radius:0} .art{height:400px;top:120px}'
python render.py gallery.html build/gallery.png '#g' 1280 600 2
```

PNG переводятся в WebP (качество 88–92) любым конвертером, например
`python -c "from PIL import Image; Image.open('build/banner.png').save('../banner.webp', quality=92, method=6)"`.
Каталог `build/` в git не попадает.

| файл | что делает |
|---|---|
| [`src/phantoms.py`](src/phantoms.py) | фантомы позвоночника и бедра: наклон, инородное тело, размер малого вертела, длина поля |
| [`src/make_demo.py`](src/make_demo.py) | план демо-набора (какое решение ждём на каждом снимке), подбор сидов, прогон, overlay |
| [`src/shot.py`](src/shot.py) | загрузка архива через страницу и скриншоты в двух темах |
| [`src/render.py`](src/render.py) | HTML → PNG через Chromium |
