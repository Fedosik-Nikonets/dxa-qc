"""Пути импорта для тестов.

Пакет лежит в `src/`, а тесты запускаются из корня репозитория, поэтому
`dxa_qc` и вспомогательный `synth_dicom` сами по себе не находятся. Раньше
каждый тестовый файл вставлял пути себе сам — это работало, но новый файл,
который так не делает, ломал сбор тестов целиком (`ModuleNotFoundError:
No module named 'dxa_qc'` на этапе collection).

Теперь пути задаются в одном месте: здесь и в `pytest.ini` (`pythonpath`).
Дубли в старых файлах оставлены — они безвредны и держат те файлы
самодостаточными при прямом запуске.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "tests"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
