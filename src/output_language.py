"""Output languages for reports and exported documents.

English is the core language and always available. An installation can
enable additional output languages:

    OUTPUT_LANGUAGES=en,de,nl        # codes offered in the UI
    DEFAULT_OUTPUT_LANGUAGE=de       # preselected in the UI (default: en)
    OUTPUT_CATALOG_DIR=/etc/tool/locales   # optional: extra <code>.toml catalogs

Each language needs a catalog `<code>.toml` (built-in: `src/locales/`,
extra ones in OUTPUT_CATALOG_DIR, which wins on name clashes):

    [meta]
    name = "Nederlands"        # shown in the UI
    llm_name = "Dutch"         # used in prompts: "Write the report in Dutch."

    [strings]
    footer.heading = "Over dit rapport"
    ...

Keys missing from a catalog fall back to English. A catalog with a key
whose {placeholders} differ from English is rejected at start-up, because
formatting it would fail or silently drop values.

This covers text the tool itself writes into reports and exports. The
report body is written by the LLM, which is told the language by name.
The interface language is separate (src.ui.i18n).
"""

from __future__ import annotations

import logging
import os
import string
import tomllib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

CORE = "en"
_BUILTIN_DIR = Path(__file__).parent / "locales"


@dataclass(frozen=True)
class Catalog:
    code: str
    name: str
    llm_name: str
    strings: dict = field(default_factory=dict)


def _placeholders(text: str) -> set[str]:
    return {f for _, f, _, _ in string.Formatter().parse(text) if f}


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = str(v)
    return out


def _read(path: Path) -> Catalog:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    meta = data.get("meta", {})
    code = path.stem.lower()
    return Catalog(
        code=code,
        name=str(meta.get("name", code)),
        llm_name=str(meta.get("llm_name", meta.get("name", code))),
        strings=_flatten(data.get("strings", {})),
    )


def _catalog_paths() -> dict[str, Path]:
    paths = {p.stem.lower(): p for p in _BUILTIN_DIR.glob("*.toml")}
    extra = os.environ.get("OUTPUT_CATALOG_DIR", "").strip()
    if extra:
        d = Path(extra)
        if not d.is_dir():
            raise ValueError(f"OUTPUT_CATALOG_DIR={extra!r} is not a directory")
        paths.update({p.stem.lower(): p for p in d.glob("*.toml")})
    return paths


def validate_catalog(cat: Catalog, core: Catalog) -> list[str]:
    """Problems that make a catalog unusable (empty list = fine)."""
    problems = []
    for key, text in cat.strings.items():
        if key not in core.strings:
            problems.append(f"{cat.code}: unknown key {key!r}")
        elif _placeholders(text) != _placeholders(core.strings[key]):
            problems.append(
                f"{cat.code}: placeholders of {key!r} differ from English "
                f"({sorted(_placeholders(text))} vs {sorted(_placeholders(core.strings[key]))})"
            )
    return problems


@lru_cache(maxsize=1)
def _state() -> tuple[dict[str, Catalog], tuple[str, ...], str]:
    paths = _catalog_paths()
    core = _read(paths[CORE])
    wanted = [c.strip().lower() for c in os.environ.get("OUTPUT_LANGUAGES", CORE).split(",") if c.strip()]
    if CORE not in wanted:
        wanted.insert(0, CORE)
    catalogs = {CORE: core}
    for code in wanted:
        if code == CORE:
            continue
        if code not in paths:
            raise ValueError(
                f"OUTPUT_LANGUAGES contains {code!r}, but there is no catalog {code}.toml "
                f"(built-in: {sorted(p for p in paths)}; add one via OUTPUT_CATALOG_DIR)"
            )
        cat = _read(paths[code])
        problems = validate_catalog(cat, core)
        if problems:
            raise ValueError("Invalid output-language catalog:\n  " + "\n  ".join(problems))
        missing = set(core.strings) - set(cat.strings)
        if missing:
            logger.warning("Output language %s: %d strings missing, English is used for them",
                           code, len(missing))
        catalogs[code] = cat
    default = os.environ.get("DEFAULT_OUTPUT_LANGUAGE", CORE).strip().lower() or CORE
    if default not in catalogs:
        raise ValueError(f"DEFAULT_OUTPUT_LANGUAGE={default!r} is not in OUTPUT_LANGUAGES")
    return catalogs, tuple(wanted), default


def reload() -> None:
    """Re-read settings and catalogs (tests, configuration changes)."""
    _state.cache_clear()


def enabled_languages() -> tuple[str, ...]:
    return _state()[1]


def default_language() -> str:
    return _state()[2]


def normalize(code: str | None) -> str:
    """An enabled language code; anything else becomes the default."""
    code = (code or "").strip().lower()
    return code if code in _state()[0] else default_language()


def language_choices() -> list[tuple[str, str]]:
    """(display name, code) pairs for the UI."""
    cats = _state()[0]
    return [(cats[c].name, c) for c in enabled_languages()]


def llm_language_name(code: str | None) -> str:
    """Language name for prompts, e.g. 'German'."""
    return _state()[0][normalize(code)].llm_name


def t(key: str, lang: str | None = None, **values) -> str:
    """Text for `key` in `lang`, falling back to English."""
    cats = _state()[0]
    text = cats[normalize(lang)].strings.get(key)
    if text is None:
        text = cats[CORE].strings.get(key)
        if text is None:
            raise KeyError(f"Unknown output-language key {key!r}")
    return text.format(**values) if values else text


# ─── Output language of the current run ──────────────────────────────
# Some text is produced deep inside helpers that have no context object
# (connectors, formatting helpers). The run sets its output language once;
# `tc()` looks keys up in it. A ContextVar keeps parallel sessions apart.
import contextvars as _contextvars

_CURRENT: "_contextvars.ContextVar[str | None]" = _contextvars.ContextVar(
    "current_output_language", default=None)


def set_current(lang: str | None) -> None:
    """Set the output language of the current run (task-local)."""
    _CURRENT.set(lang)


def current() -> str | None:
    """Output language of the current run (None = default)."""
    return _CURRENT.get()


def tc(key: str, **values) -> str:
    """Text for `key` in the output language of the current run."""
    return t(key, _CURRENT.get(), **values)


def prompt_language_line(lang: str | None) -> str:
    """Closing line for prompts whose output ends up in the report."""
    return (f"\n\nOUTPUT LANGUAGE: write all text for the reader in "
            f"{llm_language_name(lang)}. Keep JSON keys, codes, markers and "
            f"search terms exactly as specified above.")
