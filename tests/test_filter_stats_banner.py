"""
Tests for `src.ui.filter_stats_banner`.

Acceptance:
  - clean run: a short banner with the main figures
  - filter with a high loss rate: critical marker
  - coverage with unanswered/filter_blocked: detail section
  - problematic diagnosis: recommendation block
  - quality findings: list of findings limited to 5
  - fulfilment problems: suggested rework
  - empty ctx: empty banner (no crash)
"""

import unittest
from dataclasses import dataclass, field

from src.ui.filter_stats_banner import (
    CRITICAL_LOSS_RATE,
    HIGH_LOSS_RATE,
    format_filter_stats_banner,
)


@dataclass
class _MinimalCtx:
    """Minimal ctx stub with the fields the banner reads."""
    sources: list = field(default_factory=list)
    extracts: list = field(default_factory=list)
    rounds_completed: int = 0
    filter_stats_per_round: list = field(default_factory=list)
    coverage_per_round: list = field(default_factory=list)
    final_diagnosis: dict = None
    report_quality: dict = None
    query_fulfillment: dict = None


class TestEmptyContext(unittest.TestCase):

    def test_empty_ctx_returns_empty_banner(self):
        """Pipeline aborted before any activity → no banner."""
        ctx = _MinimalCtx()
        out = format_filter_stats_banner(ctx)
        self.assertEqual(out, "")


class TestHeader(unittest.TestCase):

    def test_basic_header_shown(self):
        ctx = _MinimalCtx(
            sources=["s1", "s2", "s3"],
            extracts=["e1", "e2"],
            rounds_completed=2,
        )
        out = format_filter_stats_banner(ctx)
        self.assertIn("Research statistics", out)
        self.assertIn("Rounds:** 2", out)
        self.assertIn("Sources fetched:** 3", out)
        self.assertIn("Extracts obtained:** 2", out)


class TestFilterLosses(unittest.TestCase):

    def test_no_filters_no_section(self):
        ctx = _MinimalCtx(sources=["s1"], extracts=["e1"], rounds_completed=1)
        out = format_filter_stats_banner(ctx)
        self.assertNotIn("Filter statistics", out)

    def test_inactive_filter_shown_as_no_filters(self):
        """The filter was never activated → notice "none active"."""
        ctx = _MinimalCtx(
            sources=["s1"], extracts=["e1"], rounds_completed=1,
            filter_stats_per_round=[{
                "test_filter": {
                    "activated": False, "rejected": 0, "total": 100,
                },
            }],
        )
        out = format_filter_stats_banner(ctx)
        # no filter listing, since not activated
        self.assertNotIn("test_filter", out)

    def test_active_filter_with_loss_shown(self):
        ctx = _MinimalCtx(
            sources=["s1"], extracts=["e1"], rounds_completed=1,
            filter_stats_per_round=[{
                "person_hallucination": {
                    "activated": True, "rejected": 30, "total": 100,
                },
            }],
        )
        out = format_filter_stats_banner(ctx)
        self.assertIn("Filter statistics", out)
        self.assertIn("person_hallucination", out)
        self.assertIn("30/100", out)
        self.assertIn("30%", out)
        # below HIGH_LOSS_RATE → no warning marker
        self.assertNotIn("⚠️", out.split("Filter statistics")[1])

    def test_high_loss_rate_warning_marker(self):
        rate = (HIGH_LOSS_RATE + CRITICAL_LOSS_RATE) / 2  # between 50 % and 90 %
        rejected = int(100 * rate)
        ctx = _MinimalCtx(
            sources=["s1"], extracts=["e1"], rounds_completed=1,
            filter_stats_per_round=[{
                "test": {"activated": True, "rejected": rejected, "total": 100},
            }],
        )
        out = format_filter_stats_banner(ctx)
        # warning marker present, but NOT CRITICAL
        filter_section = out.split("### 🛡️ Filter statistics")[1]
        self.assertIn("⚠️", filter_section)
        self.assertNotIn("CRITICAL", filter_section)

    def test_dgx_critical_loss_rate(self):
        """CRITICAL: a filter discards everything (528/528 = 100 % loss)."""
        ctx = _MinimalCtx(
            sources=["s%d" % i for i in range(50)],
            extracts=[],  # all gone
            rounds_completed=2,
            filter_stats_per_round=[{
                "person_hallucination": {
                    "activated": True, "rejected": 528, "total": 528,
                },
            }],
        )
        out = format_filter_stats_banner(ctx)
        self.assertIn("CRITICAL", out)
        self.assertIn("528/528", out)
        # notice block for critical filters
        self.assertIn("90% loss", out)
        self.assertIn("without this filter", out)

    def test_aggregates_across_rounds(self):
        """Two rounds with losses → values are added up."""
        ctx = _MinimalCtx(
            sources=["s1"], extracts=["e1"], rounds_completed=2,
            filter_stats_per_round=[
                {"f": {"activated": True, "rejected": 10, "total": 50}},
                {"f": {"activated": True, "rejected": 15, "total": 75}},
            ],
        )
        out = format_filter_stats_banner(ctx)
        # aggregate: 25/125
        self.assertIn("25/125", out)


class TestCoverage(unittest.TestCase):

    def test_no_coverage_no_section(self):
        ctx = _MinimalCtx(sources=["s"], extracts=["e"], rounds_completed=1)
        out = format_filter_stats_banner(ctx)
        self.assertNotIn("Coverage", out)

    def test_all_answered(self):
        ctx = _MinimalCtx(
            sources=["s"], extracts=["e"], rounds_completed=1,
            coverage_per_round=[{
                "round": 1,
                "results": {
                    "F1": {"coverage": "answered", "confidence": 0.9},
                    "F2": {"coverage": "answered", "confidence": 0.85},
                },
            }],
        )
        out = format_filter_stats_banner(ctx)
        self.assertIn("Coverage", out)
        self.assertIn("✅", out)
        self.assertIn("2 answered", out)
        # no details with only answered
        self.assertNotIn("F1**", out.split("Coverage")[1])

    def test_unanswered_shows_detail(self):
        ctx = _MinimalCtx(
            sources=["s"], extracts=["e"], rounds_completed=1,
            coverage_per_round=[{
                "round": 1,
                "results": {
                    "F1": {"coverage": "answered", "confidence": 0.9},
                    "F2": {
                        "coverage": "unanswered", "confidence": 0.85,
                        "reasoning": "Keine relevanten Quellen",
                    },
                },
            }],
        )
        out = format_filter_stats_banner(ctx)
        self.assertIn("⛔", out)
        self.assertIn("F2", out)
        self.assertIn("unanswered", out)
        self.assertIn("Keine relevanten Quellen", out)

    def test_filter_blocked_shows_detail(self):
        """The filter_blocked status should be clearly marked."""
        ctx = _MinimalCtx(
            sources=["s"], extracts=[], rounds_completed=1,
            coverage_per_round=[{
                "round": 1,
                "results": {
                    "F1": {
                        "coverage": "filter_blocked", "confidence": 0.92,
                        "reasoning": "Person-Filter hat Extrakte verworfen",
                    },
                },
            }],
        )
        out = format_filter_stats_banner(ctx)
        self.assertIn("🛡️", out)
        self.assertIn("filter_blocked", out)
        self.assertIn("Person-Filter", out)


class TestDiagnosis(unittest.TestCase):

    def test_successful_diagnosis_no_section(self):
        ctx = _MinimalCtx(
            sources=["s"], extracts=["e"], rounds_completed=1,
            final_diagnosis={
                "diagnosis": "successful", "is_successful": True,
                "is_problematic": False, "user_message": "alles ok",
                "remediation": "",
            },
        )
        out = format_filter_stats_banner(ctx)
        # The diagnosis section should be left out with "successful";
        # the header statistics already say so
        self.assertNotIn("### ⚠️ Diagnosis", out)
        self.assertNotIn("### ℹ️ Diagnosis", out)

    def test_problematic_diagnosis_with_recommendation(self):
        ctx = _MinimalCtx(
            sources=["s"], extracts=[], rounds_completed=2,
            final_diagnosis={
                "diagnosis": "filter_too_strict",
                "is_successful": False, "is_problematic": True,
                "user_message": "Alle Extrakte vom Filter verworfen.",
                "remediation": "Recherche ohne Personen-Filter neu starten.",
            },
        )
        out = format_filter_stats_banner(ctx)
        self.assertIn("Diagnosis", out)
        self.assertIn("⚠️", out)
        self.assertIn("Alle Extrakte vom Filter", out)
        self.assertIn("Recommendation:", out)
        self.assertIn("Personen-Filter", out)


class TestQualityAndFulfillment(unittest.TestCase):

    def test_quality_passed_no_section(self):
        ctx = _MinimalCtx(
            sources=["s"], extracts=["e"], rounds_completed=1,
            report_quality={
                "passed": True, "rating": "good",
                "issues": [], "segments": 1,
            },
        )
        out = format_filter_stats_banner(ctx)
        self.assertNotIn("Report quality", out)

    def test_quality_failed_lists_first_5_maengel(self):
        ctx = _MinimalCtx(
            sources=["s"], extracts=["e"], rounds_completed=1,
            report_quality={
                "passed": False, "rating": "poor",
                "segments": 2,
                "issues": [
                    {"art": "hallucinated", "description": f"Mangel {i}"}
                    for i in range(8)
                ],
            },
        )
        out = format_filter_stats_banner(ctx)
        self.assertIn("Report quality", out)
        self.assertIn("poor", out)
        self.assertIn("8 findings", out)
        # the first 5 are in it
        for i in range(5):
            self.assertIn(f"Mangel {i}", out)
        # finding 5 is not (we show 0..4)
        self.assertNotIn("Mangel 5", out)
        # "3 more"
        self.assertIn("3 more", out)

    def test_fulfillment_failed_shows_nacharbeit(self):
        ctx = _MinimalCtx(
            sources=["s"], extracts=["e"], rounds_completed=1,
            query_fulfillment={
                "fulfilled": False,
                "assessment": "Nur Preis, kein Stromverbrauch",
                "rework": "Ergänze TDP-Sektion",
            },
        )
        out = format_filter_stats_banner(ctx)
        self.assertIn("Request fulfilment", out)
        self.assertIn("Stromverbrauch", out)
        self.assertIn("Suggestion:", out)
        self.assertIn("TDP-Sektion", out)


class TestRealisticDgxScenario(unittest.TestCase):
    """End-to-end test of a run in which a filter discarded everything."""

    def test_complete_dgx_banner(self):
        """CRITICAL: all indicators of such a run are visible in the banner."""
        ctx = _MinimalCtx(
            sources=["s%d" % i for i in range(40)],
            extracts=[],  # 528 lost!
            rounds_completed=2,
            filter_stats_per_round=[
                {"person_hallucination": {
                    "activated": True, "rejected": 250, "total": 250,
                }},
                {"person_hallucination": {
                    "activated": True, "rejected": 278, "total": 278,
                }},
            ],
            coverage_per_round=[{
                "round": 2,
                "results": {
                    f"F{i}": {
                        "coverage": "filter_blocked", "confidence": 0.92,
                        "reasoning": "Person-Filter aktiv, alle Extrakte verworfen",
                    }
                    for i in range(1, 5)
                },
            }],
            final_diagnosis={
                "diagnosis": "filter_too_strict",
                "is_successful": False, "is_problematic": True,
                "user_message": "528 Extrakte vom Filter verworfen.",
                "remediation": "Recherche ohne Personen-Filter neu starten.",
            },
        )
        out = format_filter_stats_banner(ctx)
        # Header
        self.assertIn("40", out)
        # Filter
        self.assertIn("528/528", out)  # aggregate over the rounds
        self.assertIn("CRITICAL", out)
        # Coverage
        self.assertIn("filter_blocked", out)
        # Diagnosis: user_message is shown; the internal code
        # `filter_too_strict` is an implementation detail
        self.assertIn("vom Filter verworfen", out)
        self.assertIn("Recommendation:", out)


if __name__ == "__main__":
    unittest.main()
