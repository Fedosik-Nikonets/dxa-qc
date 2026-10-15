"""Скриншоты веб-интерфейса в светлой и тёмной теме.

Сервис должен быть запущен (`./run.sh --serve` или
`python -m dxa_qc.service --port 8000`). Скрипт загружает демо-архив через
саму страницу, ждёт таблицу, прогоняет пакет второй раз (без прогрева),
выбирает снимок с артефактом и снимает экран.

    python docs/assets/src/shot.py http://localhost:8000 docs/assets/src/build/demo.zip docs/assets/src/build
"""
import asyncio
import sys

from playwright.async_api import async_playwright


async def main(url: str, demo_zip: str, out: str) -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        for scheme in ("light", "dark"):
            ctx = await browser.new_context(viewport={"width": 1440, "height": 1010}, device_scale_factor=1.5,
                                            color_scheme=scheme, locale="ru-RU")
            page = await ctx.new_page()
            await page.goto(url)
            await page.wait_for_selector(".dot.on")
            await page.set_input_files("#file", demo_zip)
            await page.click("#run")
            await page.wait_for_selector("tr.img", timeout=60000)
            await page.click("#run")
            await page.wait_for_function("document.querySelector('#status').textContent.startsWith('Готово')",
                                         timeout=60000)
            await page.wait_for_timeout(300)
            await page.click('tr.img[data-p="study_03/CR000000.dcm"]')
            await page.wait_for_function("document.querySelector('#viewer img')?.complete")
            await page.evaluate("window.scrollTo(0, 0)")
            await page.wait_for_timeout(500)
            await page.screenshot(path=f"{out}/ui_{scheme}.png")
            await ctx.close()
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main(*sys.argv[1:4]))
