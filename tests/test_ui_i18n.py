"""Interface language: catalogs, tr(), per-page language of event handlers."""

import ast
import asyncio
import inspect
import os
import unittest
import warnings
from pathlib import Path
from unittest import mock

from src.ui import i18n
from src.ui.i18n import tr

ROOT = Path(__file__).resolve().parent.parent


def interface_keys() -> set[str]:
    """Every English text the interface translates.

    Literal first arguments of tr(...) in src/, plus the texts that live in
    registries and are translated where they are shown (mode and template
    labels, form labels, placeholders and choices, chat action tooltips).
    """
    keys = set()
    for path in (ROOT / "src").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "tr"
                    and node.args and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)):
                keys.add(node.args[0].value)

    from src.pipeline.analysis_pipeline import USE_CASE_REGISTRY
    from src.pipeline.models import OUTPUT_TEMPLATES
    from src.ui.chat_actions import CHAT_ACTIONS
    from src.ui.research_modes import RESEARCH_MODES

    keys.update(label for _, label, _ in RESEARCH_MODES)
    keys.update(t["label"] for t in OUTPUT_TEMPLATES.values())
    keys.update(a.title for a in CHAT_ACTIONS)
    for entry in USE_CASE_REGISTRY.values():
        for req in entry["preflight"].get_requirements():
            keys.add(req.label)
            if req.placeholder:
                keys.add(req.placeholder)
            for c in req.choices or []:
                if isinstance(c, (tuple, list)):
                    keys.add(c[0])
    return keys


class MissingTranslationWarning(UserWarning):
    """Interface texts without a German entry (they show in English)."""


def warn_missing(code: str, missing: set[str]) -> None:
    """Missing entries only warn: contributors need not write German."""
    if missing:
        listed = "\n".join(f"  {k!r}" for k in sorted(missing))
        warnings.warn(
            f"{len(missing)} interface text(s) without a translation in "
            f"src/ui/locales/{code}.toml (shown in English):\n{listed}",
            MissingTranslationWarning, stacklevel=2)


class _Lang:
    """Run a block in one interface language."""

    def __init__(self, code):
        self.code = code

    def __enter__(self):
        self.token = i18n.set_current(self.code)

    def __exit__(self, *exc):
        i18n.reset_current(self.token)


class TestCatalogs(unittest.TestCase):
    def setUp(self):
        i18n.reload()

    def test_german_catalog_is_complete(self):
        # A warning, not a failure: missing texts show in English and the
        # maintainers add the German ones.
        warn_missing("de", interface_keys() - set(i18n.translations("de")))

    def test_missing_entry_warns_without_failing(self):
        with self.assertWarns(MissingTranslationWarning) as w:
            warn_missing("de", {"🔍 Start research"})
        self.assertIn("'🔍 Start research'", str(w.warning))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warn_missing("de", set())  # nothing missing: silent

    def test_german_catalog_has_no_stale_entries(self):
        stale = set(i18n.translations("de")) - interface_keys()
        self.assertEqual(sorted(stale), [],
                         "entries in de.toml that the code no longer uses")

    def test_placeholder_mismatch_is_reported(self):
        problems = i18n.validate_catalog("xx", {"{n} sources": "{count} Quellen"})
        self.assertEqual(len(problems), 1)
        self.assertIn("{n} sources", problems[0])

    def test_unknown_language_stops_start(self):
        with mock.patch.dict(os.environ, {"UI_LANGUAGES": "en,xx"}):
            i18n.reload()
            with self.assertRaises(ValueError):
                i18n.languages()
        i18n.reload()


class TestTr(unittest.TestCase):
    def setUp(self):
        i18n.reload()

    def test_english_is_the_source(self):
        with _Lang("en"):
            self.assertEqual(tr("🔍 Start research"), "🔍 Start research")

    def test_german(self):
        with _Lang("de"):
            self.assertEqual(tr("🔍 Start research"), "🔍 Recherche starten")
            self.assertEqual(tr("📥 Source {n}: {title}", n=3, title="X"),
                             "📥 Quelle 3: X")

    def test_missing_entry_falls_back_to_english(self):
        with _Lang("de"):
            self.assertEqual(tr("not in any catalog {n}", n=1), "not in any catalog 1")

    def test_placeholder_named_text(self):
        with _Lang("de"):
            self.assertEqual(tr("\n**Suggestion:** {text}", text="x"),
                             "\n**Vorschlag:** x")


class TestPages(unittest.TestCase):
    def tearDown(self):
        i18n.reload()

    def test_default_language_is_the_start_page(self):
        with mock.patch.dict(os.environ, {"DEFAULT_UI_LANGUAGE": "de",
                                          "UI_LANGUAGES": "en,de"}):
            i18n.reload()
            self.assertEqual(i18n.languages(), ("de", "en"))
            self.assertEqual(i18n.page_path("de"), "")
            self.assertEqual(i18n.page_path("en"), "en")
            self.assertEqual(i18n.current(), "de")

    def test_single_language(self):
        with mock.patch.dict(os.environ, {"UI_LANGUAGES": "en"}):
            i18n.reload()
            self.assertEqual(i18n.languages(), ("en",))


class TestBind(unittest.TestCase):
    """Handlers run in the language of their page, step by step."""

    def setUp(self):
        i18n.reload()

    def test_plain_function(self):
        bound = i18n.bind(lambda: tr("✖ Cancel"), "de")
        self.assertEqual(bound(), "✖ Abbrechen")

    def test_generator_keeps_language_between_steps(self):
        def handler():
            yield tr("✖ Cancel")
            yield tr("⏹️ Stop")

        bound = i18n.bind(handler, "de")
        self.assertTrue(inspect.isgeneratorfunction(bound))
        it = bound()
        first = next(it)
        # Another session's event runs in between in the same thread.
        with _Lang("en"):
            second = next(it)
        self.assertEqual([first, second], ["✖ Abbrechen", "⏹️ Stopp"])

    def test_async_generator(self):
        async def handler(a, b=2):
            yield tr("✖ Cancel")
            yield tr("⏹️ Stop")

        bound = i18n.bind(handler, "de")
        self.assertTrue(inspect.isasyncgenfunction(bound))
        # Gradio reads the signature (special arguments, number of inputs).
        self.assertEqual(list(inspect.signature(bound).parameters), ["a", "b"])

        async def collect():
            return [x async for x in bound(1)]

        self.assertEqual(asyncio.run(collect()), ["✖ Abbrechen", "⏹️ Stopp"])

    def test_coroutine(self):
        async def handler():
            await asyncio.sleep(0)
            return tr("✖ Cancel")

        bound = i18n.bind(handler, "de")
        self.assertTrue(inspect.iscoroutinefunction(bound))
        self.assertEqual(asyncio.run(bound()), "✖ Abbrechen")


if __name__ == "__main__":
    unittest.main()
