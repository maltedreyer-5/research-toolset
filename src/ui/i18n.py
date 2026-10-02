"""Language of the user interface.

The interface text is written in English in the code and wrapped in
`tr()`; other languages come from catalogs in `src/ui/locales/`:

    [meta]
    name = "Deutsch"            # shown in the language switch

    [strings]
    "Start research" = "Recherche starten"
    "{n} sources" = "{n} Quellen"

The English text is the key (as with gettext), so the code stays readable
and a missing translation simply shows the English text. Placeholders
must match the English key exactly; a catalog where they differ is
rejected at start-up.

    UI_LANGUAGES=en,de          # languages offered (default: all catalogs)
    DEFAULT_UI_LANGUAGE=de      # language of the start page (default: en)

Every language is its own page of the app: the default language at the
root, every other one under `/<code>` (e.g. `/en`). The page sets the
language once while it is built and for every event it handles (see
`bind_events`), so code anywhere below an event handler — including
status messages from the pipeline — can call `tr()` without passing the
language around. A ContextVar keeps parallel sessions apart.

This is the language of the interface. The language of reports is chosen
separately (src.output_language).
"""

from __future__ import annotations

import contextvars
import functools
import inspect
import os
import string
import tomllib
from functools import lru_cache
from pathlib import Path

SOURCE = "en"
_DIR = Path(__file__).parent / "locales"

_CURRENT: "contextvars.ContextVar[str | None]" = contextvars.ContextVar(
    "ui_language", default=None)


def _placeholders(text: str) -> set[str]:
    return {f for _, f, _, _ in string.Formatter().parse(text) if f}


def _read(path: Path) -> tuple[str, dict[str, str]]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    name = str(data.get("meta", {}).get("name", path.stem))
    strings = {str(k): str(v) for k, v in data.get("strings", {}).items()}
    return name, strings


def validate_catalog(code: str, strings: dict[str, str]) -> list[str]:
    """Problems that make a catalog unusable (empty list = fine)."""
    return [
        f"{code}: placeholders of {key!r} differ from the English text "
        f"({sorted(_placeholders(text))} vs {sorted(_placeholders(key))})"
        for key, text in strings.items()
        if _placeholders(text) != _placeholders(key)
    ]


@lru_cache(maxsize=1)
def _state() -> tuple[dict[str, tuple[str, dict]], tuple[str, ...], str]:
    found = {p.stem.lower(): p for p in _DIR.glob("*.toml")}
    raw = os.environ.get("UI_LANGUAGES", "").strip()
    wanted = ([c.strip().lower() for c in raw.split(",") if c.strip()]
              if raw else sorted(found))
    if SOURCE not in wanted:
        wanted.insert(0, SOURCE)
    catalogs = {}
    problems = []
    for code in wanted:
        if code not in found:
            raise ValueError(
                f"UI_LANGUAGES contains {code!r}, but there is no catalog "
                f"src/ui/locales/{code}.toml")
        name, strings = _read(found[code])
        problems += validate_catalog(code, strings)
        catalogs[code] = (name, strings)
    if problems:
        raise ValueError("Invalid interface catalog:\n  " + "\n  ".join(problems))
    default = os.environ.get("DEFAULT_UI_LANGUAGE", SOURCE).strip().lower() or SOURCE
    if default not in catalogs:
        raise ValueError(f"DEFAULT_UI_LANGUAGE={default!r} is not in UI_LANGUAGES")
    # Default first: it is the start page.
    order = (default, *[c for c in wanted if c != default])
    return catalogs, order, default


def reload() -> None:
    """Re-read settings and catalogs (tests, configuration changes)."""
    _state.cache_clear()


def languages() -> tuple[str, ...]:
    """Enabled codes, the default language first."""
    return _state()[1]


def default_language() -> str:
    return _state()[2]


def language_name(code: str) -> str:
    return _state()[0][code][0]


def page_path(code: str) -> str:
    """URL path of a language's page relative to the app root."""
    return "" if code == default_language() else code


def current() -> str:
    """Interface language of the running event (or page build)."""
    return _CURRENT.get() or default_language()


def set_current(code: str | None) -> contextvars.Token:
    return _CURRENT.set(code)


def reset_current(token: contextvars.Token) -> None:
    _CURRENT.reset(token)


def tr(text: str, /, **values) -> str:
    """`text` (English) in the current interface language."""
    code = current()
    if code != SOURCE:
        text = _state()[0].get(code, ("", {}))[1].get(text, text)
    return text.format(**values) if values else text


def translations(code: str) -> dict[str, str]:
    """All entries of a catalog (tests, tools)."""
    return dict(_state()[0][code][1])


def bind(fn, code: str):
    """Wrap an event handler so that it runs in interface language `code`.

    Generators are resumed step by step — possibly in different threads or
    contexts — so the language is set again before every step, not just
    once. The wrapper keeps the kind of function (Gradio decides by it how
    to call the handler) and its signature (functools.wraps).
    """
    if inspect.isasyncgenfunction(fn):
        @functools.wraps(fn)
        async def asyncgen_wrapper(*args, **kwargs):
            _CURRENT.set(code)
            iterator = fn(*args, **kwargs)
            while True:
                _CURRENT.set(code)
                try:
                    item = await iterator.__anext__()
                except StopAsyncIteration:
                    return
                yield item
        return asyncgen_wrapper

    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            _CURRENT.set(code)
            return await fn(*args, **kwargs)
        return async_wrapper

    if inspect.isgeneratorfunction(fn):
        @functools.wraps(fn)
        def gen_wrapper(*args, **kwargs):
            _CURRENT.set(code)
            iterator = fn(*args, **kwargs)
            while True:
                _CURRENT.set(code)
                try:
                    item = next(iterator)
                except StopIteration:
                    return
                yield item
        return gen_wrapper

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        _CURRENT.set(code)
        return fn(*args, **kwargs)
    return wrapper


def bind_events(demo, page: str, code: str) -> None:
    """Run every event handler of one page in its interface language."""
    fns = getattr(demo, "fns", None)
    if isinstance(fns, dict):
        fns = fns.values()
    if not isinstance(fns, (list, tuple, type({}.values()))):
        return
    for block_fn in fns:
        if block_fn.fn is not None and getattr(block_fn, "page", "") == page \
                and not getattr(block_fn, "_ui_language_bound", False):
            block_fn.fn = bind(block_fn.fn, code)
            block_fn._ui_language_bound = True
