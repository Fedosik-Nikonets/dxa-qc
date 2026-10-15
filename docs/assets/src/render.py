"""HTML → PNG через Playwright (Chromium): баннер, галерея overlay, social preview.

    python docs/assets/src/render.py banner.html  banner.png  '#b' 1280 400 2
    python docs/assets/src/render.py gallery.html gallery.png '#g' 1280 600 2
    python docs/assets/src/render.py banner.html  social.png  '#b' 1280 640 1 \
        '.b{--h:640px;border-radius:0} .art{height:400px;top:120px}'

Пути к HTML — относительно этого каталога. Шрифт — Inter (должен быть
установлен в системе: страница не ходит в сеть).
"""
import asyncio
import sys
from pathlib import Path

from playwright.async_api import async_playwright

HERE = Path(__file__).resolve().parent


async def main(src, out, sel, w, h, scale, extra_css=""):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": w, "height": h}, device_scale_factor=scale)
        await page.goto((HERE / src).as_uri())
        if extra_css:
            await page.add_style_tag(content=extra_css)
        await page.evaluate("document.fonts.ready")
        await page.wait_for_timeout(300)
        await page.locator(sel).screenshot(path=out, omit_background=True)
        await browser.close()


if __name__ == "__main__":
    a = sys.argv
    asyncio.run(main(a[1], a[2], a[3], int(a[4]), int(a[5]), float(a[6]), a[7] if len(a) > 7 else ""))
