"""
Regression tests for the research pipeline's wiring.

Passing unit tests do not guarantee that the parts are connected. These
tests guard three such connections:

  1. `create_app()` must boot — every UI module it imports must exist.
  2. The polarity mechanism must be wired into the production harvest,
     not only tested in isolation.
  3. The off-topic filter runs exactly once per round (in
     `_run_search_and_fetch`), not a second time as a DAG node over the
     whole corpus.
"""

import os
import subprocess
import sys
import unittest

import tests.conftest  # noqa: F401  (httpx/openai stubs)

from src.pipeline.harvest_parser import parse_harvest_response_with_polarity


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestCreateAppBoots(unittest.TestCase):
    """create_app() must run through. Executed in a subprocess with a Gradio
    stub, so that NO sys.modules mocks leak into other tests."""

    SCRIPT = r"""
import sys, types
g = types.ModuleType("gradio")
class _C:
    def __init__(s,*a,**k): s.value=k.get("value")
    def __enter__(s): return s
    def __exit__(s,*a): return False
    def __getattr__(s,n): return lambda *a,**k: s
class _B(_C):
    def queue(s,*a,**k): return s
    def launch(s,*a,**k): return s
for n in ["Accordion","BrowserState","Button","Chatbot","Checkbox","Column","Dropdown",
          "File","Group","HTML","Markdown","MultimodalTextbox","Row",
          "State","TabItem","Tabs","Textbox"]:
    setattr(g,n,type(n,(_C,),{}))
g.Blocks=_B
g.Info=g.Warning=lambda *a,**k: None
g.Error=type("Error",(Exception,),{})
g.update=lambda *a,**k: {"__u__":True}
th=types.ModuleType("gradio.themes")
th.Default=th.Base=type("T",(),{"__init__":lambda s,*a,**k: None})
g.themes=th
sys.modules["gradio"]=g; sys.modules["gradio.themes"]=th
import tests.conftest  # httpx/openai stubs
from src.config import AppConfig
from src.ui.gradio_app import create_app
app = create_app(AppConfig())
assert app is not None
print("CREATE_APP_OK")
"""

    def test_create_app_does_not_crash_on_boot(self):
        res = subprocess.run(
            [sys.executable, "-c", self.SCRIPT],
            cwd=REPO, capture_output=True, text=True, timeout=120,
        )
        self.assertIn(
            "CREATE_APP_OK", res.stdout,
            msg=f"create_app() crashes on boot.\n"
                f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}",
        )

    def test_pipeline_run_module_exists_and_renders(self):
        from src.ui.pipeline_run import render_pipeline_run
        from src.pipeline.models import HarvestContext

        self.assertEqual(
            render_pipeline_run(None), "*No research started yet.*"
        )
        ctx = HarvestContext(query="Konsistenz der Web-Recherche")
        ctx.status = "running"
        out = render_pipeline_run(ctx)
        self.assertIn("Pipeline run", out)
        self.assertIn("Konsistenz der Web-Recherche", out)

    def test_import_is_defensive(self):
        """The import in gradio_app is protected against a missing module
        (try/except)."""
        src = open(os.path.join(REPO, "src/ui/gradio_app.py")).read()
        i = src.find("from src.ui.pipeline_run import render_pipeline_run")
        self.assertGreater(i, 0)
        # There must be a try: within the 200-character window before it
        self.assertIn("try:", src[max(0, i - 200):i])


class TestPolarityWiredIntoProductionHarvest(unittest.TestCase):
    """The polarity parser is wired into the production path and the prompt
    asks for the markers."""

    def test_orchestrator_uses_polarity_parser(self):
        src = open(os.path.join(REPO, "src/pipeline/orchestrator.py")).read()
        self.assertIn("parse_harvest_response_with_polarity", src,
                       "the harvest must use the polarity-aware parser")
        # A parser that ignores the marker must not be called in the
        # harvest_one path.
        self.assertNotIn("self._parse_harvest_response(\n", src)

    def test_harvest_prompt_requests_polarity_markers(self):
        src = open(os.path.join(REPO, "src/prompts/research.py")).read()
        for marker in ("[POSITIVE]", "[NEGATIVE]", "[META]"):
            self.assertIn(marker, src,
                          f"HARVEST_PROMPT must ask for {marker}")

    def test_negative_meta_do_not_leak_into_fact_and_are_filtered(self):
        response = (
            "[F1] [POSITIV] Fakt: Das Modell hat 7 Mrd. Parameter\n"
            "     Kontext: Aus dem Datenblatt\n"
            "     Verlässlichkeit: hoch\n"
            "[F2] [NEGATIV] Fakt: Die Quelle nennt keine Trainingsdaten\n"
            "     Kontext: Abschnitt fehlt\n"
            "[F3] [META] Fakt: Die Seite ist Marketing, kein Datenblatt\n"
        )
        extracts = parse_harvest_response_with_polarity(
            response, "http://x", "X"
        )
        by_pol = {e.question_id: e.polarity for e in extracts}
        self.assertEqual(by_pol.get("F1"), "positive")
        self.assertEqual(by_pol.get("F2"), "negative")
        self.assertEqual(by_pol.get("F3"), "meta")

        # the marker must NOT end up in the fact text
        for e in extracts:
            self.assertNotIn("[POSITIV]", e.fact)
            self.assertNotIn("[NEGATIV]", e.fact)
            self.assertNotIn("[META]", e.fact)

        # the positive-only filter (as in HarvestNode) lets exactly F1 through
        positive = [e for e in extracts
                    if getattr(e, "polarity", "positive") == "positive"]
        self.assertEqual([e.question_id for e in positive], ["F1"])


class TestOffTopicFilterRunsOnce(unittest.TestCase):
    """Exactly one off-topic pass per round, over the new sources; no second
    pass over the whole corpus, no double ctx.sources.extend."""

    def setUp(self):
        self.orch = open(
            os.path.join(REPO, "src/pipeline/orchestrator.py")
        ).read()
        self.nodes = open(
            os.path.join(REPO, "src/pipeline/dag_nodes.py")
        ).read()

    def test_no_offtopic_node_in_round_loop(self):
        # run_via_dag builds the round sub-pipeline; OffTopicFilterNode
        # must not be constructed there.
        self.assertNotIn("OffTopicFilterNode(", self.orch,
                          "OffTopicFilterNode runs again in run_via_dag "
                          "over the whole corpus (regression)")

    def test_single_offtopic_call_in_search_and_fetch(self):
        n = self.orch.count("self._filter_off_topic_sources(")
        self.assertEqual(
            n, 1,
            f"_filter_off_topic_sources may be called exactly once, "
            f"found: {n}",
        )

    def test_no_double_ctx_sources_extend(self):
        # _run_search_and_fetch must NOT extend ctx.sources itself
        # (the SearchAndFetchNode does that — otherwise everything ends up twice).
        seg_start = self.orch.find("async def _run_search_and_fetch")
        seg_end = self.orch.find("\n    def _enrich_zis_markdown")
        segment = self.orch[seg_start:seg_end]
        self.assertNotIn("ctx.sources.extend(new_sources)", segment,
                         "duplicate ctx.sources.extend (regression)")
        self.assertIn("ctx.sources.extend(new_sources)", self.nodes,
                      "SearchAndFetchNode must fill ctx.sources")


class TestGitRepoLoopNotNested(unittest.TestCase):
    """The git-repository README loop must not be nested inside the
    direct-URL loop."""

    def test_git_repo_loop_is_sibling_of_direct_loop(self):
        src = open(
            os.path.join(REPO, "src/pipeline/orchestrator.py")
        ).read()
        # position of the two loops
        i_direct = src.find("for direct in ordered_direct:")
        i_repo = src.find("for repo in plan.git_repos:", i_direct)
        self.assertGreater(i_direct, 0)
        self.assertGreater(i_repo, i_direct)
        # The line with the repository loop must be at the 8-space level
        # (method body), not at 12 (inside the direct loop).
        line_start = src.rfind("\n", 0, i_repo) + 1
        indent = len(src[line_start:i_repo])
        self.assertEqual(
            indent, 8,
            f"git repo loop at indentation {indent} instead of 8 — "
            f"it would be nested in the direct loop again (regression)",
        )


if __name__ == "__main__":
    unittest.main()
