#!/usr/bin/env python3
"""Минимальный запускатель тестов для окружения без pytest.

В нормальном окружении используется `pytest` (зафиксирован в requirements.txt).
Этот файл нужен только там, где PyPI недоступен: он понимает fixture `tmp_path`,
`@pytest.mark.parametrize` и `pytest.approx`/`pytest.raises` через заглушку.
"""
from __future__ import annotations

import builtins
import importlib
import inspect
import re
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

try:
    import pytest  # noqa: F401
    HAVE_PYTEST = True
except ImportError:
    HAVE_PYTEST = False

    class _Approx:
        """approx для чисел и для последовательностей: настоящий pytest умеет оба."""

        def __init__(self, v, rel=None, abs=None):
            self.v, self.rel, self.abs = v, rel, abs

        def _close(self, a: float, b: float) -> bool:
            tol = self.abs if self.abs is not None else (
                builtins.abs(b) * (self.rel if self.rel is not None else 1e-6) + 1e-12
            )
            return builtins.abs(float(a) - float(b)) <= tol

        def __eq__(self, other):
            try:
                expected = list(self.v)
                got = list(other)
            except TypeError:
                return self._close(other, self.v)
            if len(expected) != len(got):
                return False
            return all(self._close(g, e) for g, e in zip(got, expected))

        def __repr__(self):
            return f"approx({self.v})"

    class _Mark:
        def __init__(self, name, args, kwargs):
            self.name, self.args, self.kwargs = name, args, kwargs

    class _MarkFactory:
        def __getattr__(self, name):
            def deco(*args, **kwargs):
                def wrap(fn):
                    fn.pytestmark = list(getattr(fn, "pytestmark", [])) + [
                        _Mark(name, args, kwargs)
                    ]
                    return fn
                return wrap
            return deco

    class _Raises:
        """Поддерживает match=: без него тест «падает с ValueError» ничего не
        говорит о том, с каким именно."""

        def __init__(self, exc, match=None):
            self.exc, self.match = exc, match
            self.value = None

        def __enter__(self):
            return self

        def __exit__(self, t, v, tb):
            assert t is not None and issubclass(t, self.exc), f"не поднято {self.exc}"
            self.value = v
            if self.match is not None:
                assert re.search(self.match, str(v)), (
                    f"сообщение '{v}' не совпало с шаблоном '{self.match}'")
            return True

    class Skipped(Exception):
        """Тест пропущен: в окружении нет необязательной зависимости."""

    def _skip(reason: str = "") -> None:
        raise Skipped(reason)

    def _importorskip(name: str, reason: str | None = None):
        try:
            return importlib.import_module(name)
        except ImportError:
            raise Skipped(reason or f"нет модуля {name}") from None

    mod = type(sys)("pytest")
    mod.approx = _Approx
    mod.mark = _MarkFactory()
    mod.raises = _Raises
    mod.skip = _skip
    mod.importorskip = _importorskip
    mod.Skipped = Skipped
    sys.modules["pytest"] = mod


class _MonkeyPatch:
    """Минимальная fixture `monkeypatch`: setattr / setenv с откатом после теста."""

    _MISSING = object()

    def __init__(self):
        self._undo: list = []

    def setattr(self, target, name, value, raising=True):
        old = getattr(target, name, self._MISSING)
        if old is self._MISSING and raising:
            raise AttributeError(f"{target!r} не имеет атрибута {name!r}")
        self._undo.append((target, name, old))
        setattr(target, name, value)

    def setenv(self, name, value):
        import os  # noqa: PLC0415
        self._undo.append((os.environ, name, os.environ.get(name, self._MISSING)))
        os.environ[name] = str(value)

    def undo(self):
        for target, name, old in reversed(self._undo):
            if isinstance(target, dict):
                if old is self._MISSING:
                    target.pop(name, None)
                else:
                    target[name] = old
            elif old is self._MISSING:
                delattr(target, name)
            else:
                setattr(target, name, old)
        self._undo.clear()


def _skipped_exc() -> type[BaseException]:
    """Исключение «тест пропущен»: своё в заглушке, штатное в настоящем pytest."""
    import pytest  # noqa: PLC0415
    return getattr(pytest, "Skipped", None) or pytest.skip.Exception


def run_module(name: str) -> tuple[int, int, int]:
    mod = importlib.import_module(name)
    passed = failed = skipped = 0
    tmp_root = Path(tempfile.mkdtemp(prefix=f"{name}_"))
    for fn_name, fn in sorted(vars(mod).items()):
        if not fn_name.startswith("test_") or not callable(fn):
            continue
        argnames: list[str] = []
        cases: list[tuple] = [()]
        for mark in getattr(fn, "pytestmark", []):
            if getattr(mark, "name", "") == "parametrize":
                names = mark.args[0]
                argnames = [n.strip() for n in names.split(",")] if isinstance(names, str) else list(names)
                cases = [v if isinstance(v, tuple) else (v,) for v in mark.args[1]]
        for i, case in enumerate(cases):
            sig = inspect.signature(fn)
            kwargs = dict(zip(argnames, case))
            if "tmp_path" in sig.parameters:
                d = tmp_root / f"{fn_name}_{i}"
                d.mkdir(parents=True, exist_ok=True)
                kwargs["tmp_path"] = d
            mp = _MonkeyPatch() if "monkeypatch" in sig.parameters else None
            if mp is not None:
                kwargs["monkeypatch"] = mp
            label = f"{name}::{fn_name}" + (f"[{case[0]}]" if case else "")
            try:
                fn(**kwargs)
                passed += 1
                print(f"  ok   {label}")
            except _skipped_exc() as exc:
                skipped += 1
                print(f"  skip {label} ({exc})")
            except Exception:
                failed += 1
                print(f"  FAIL {label}")
                traceback.print_exc()
            finally:
                if mp is not None:
                    mp.undo()
    shutil.rmtree(tmp_root, ignore_errors=True)
    return passed, failed, skipped


def main(argv: list[str]) -> int:
    mods = argv[1:] or [
        p.stem for p in sorted((ROOT / "tests").glob("test_*.py"))
    ]
    total_p = total_f = total_s = 0
    for m in mods:
        print(f"[{m}]")
        p, f, s_ = run_module(m)
        total_p += p
        total_f += f
        total_s += s_
    tail = f", {total_s} skipped" if total_s else ""
    print(f"\n{total_p} passed, {total_f} failed{tail}")
    return 1 if total_f else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
