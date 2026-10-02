"""BibTeX export of a reference check.

The check builds the BibTeX itself (best API match per entry) and passes
it on in search_stats["bibtex"]; the export writes exactly that text.
"""

import os
import unittest
from unittest.mock import patch

import tests.conftest  # noqa: F401  (httpx/openai stubs)
from src.pipeline.models import HarvestContext, OutputSchema
from src.ui import gradio_app as ga

BIBTEX = ("% Header\n% 2 entries\n\n"
          "@article{a2020,\n  title = {One},\n}\n\n"
          "@book{b2021,\n  title = {Two},\n}\n")


def _state(search_stats, format_type="literature_check"):
    ctx = HarvestContext(query="Check these references")
    ctx.final_report = "# Bibliography check report"
    ctx.output_schema = OutputSchema(title="Bibliography check report",
                                     format_type=format_type, style="academic")
    ctx.search_stats = search_stats
    st = ga.AppState()
    st.current_research = ctx
    return st


class TestBibtexExport(unittest.TestCase):
    def test_writes_the_checks_bibtex(self):
        state = _state({"type": "literature_check", "bibtex": BIBTEX})
        with patch.object(ga.gr, "Info") as info:
            path = ga.export_bibtex(state)
        self.assertTrue(path and os.path.exists(path))
        self.addCleanup(os.remove, path)
        with open(path, encoding="utf-8") as f:
            self.assertEqual(f.read(), BIBTEX)
        self.assertIn("2 entries", info.call_args.args[0])

    def test_warns_without_entries(self):
        for stats in ({"type": "literature_check"},
                      {"type": "literature_check", "bibtex": "% Header only\n"},
                      None):
            with self.subTest(stats=stats), patch.object(ga.gr, "Warning") as warn:
                self.assertIsNone(ga.export_bibtex(_state(stats)))
                warn.assert_called_once()

    def test_only_for_reference_checks(self):
        state = _state({"bibtex": BIBTEX}, format_type="structured_report")
        with patch.object(ga.gr, "Warning") as warn:
            self.assertIsNone(ga.export_bibtex(state))
        self.assertIn("only available", warn.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
