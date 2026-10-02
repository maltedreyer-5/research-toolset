"""
Research modes: SINGLE SOURCE OF TRUTH.

Every mode is defined here EXACTLY ONCE as (ID, dropdown label, route).
Derived from this table:

  - RESEARCH_MODE_ORDER      — order of the dropdown entries (IDs)
  - DEFAULT_RESEARCH_MODE    — preselected mode
  - ANALYSIS_MODE_MAP        — ID → use_case (analysis modes only)
  - ANALYSIS_USE_CASE_ORDER  — order of the analysis use cases
  - resolve_research_route() — exact ID → route resolution
  - mode_choices()           — (label, ID) pairs for the dropdown

`route` is a tuple (kind, use_case):
  ("web", None)              → web research
  ("institution", None)      → institution research (only with a profile)
  ("literature_check", None) → bibliography check
  ("explainer", None)        → in-depth explanation
  ("analysis", "<uc_name>")  → generic analysis use case

Routing is an exact lookup by ID, never a substring match on labels:
several labels share substrings (three modes contain "Literature"), and
labels may be changed or translated without affecting behaviour.
tests/test_research_modes.py guards this.

This module is deliberately Gradio-free and therefore unit-testable
without UI dependencies.
"""

from __future__ import annotations

from typing import Optional

# Route kind: use_case is only set for "analysis".
Route = tuple[str, Optional[str]]

# Stable mode IDs are what the UI stores and routes on; labels are for
# display only and may be translated freely.
# IMPORTANT: the order here = display order in the dropdown.
RESEARCH_MODES: list[tuple[str, str, Route]] = [
    ("web",               "🌐 Web research",        ("web", None)),
    ("institution",       "🏛️ {institution} research", ("institution", None)),
    ("literature_check",  "📚 Check references",     ("literature_check", None)),
    ("literature_finder", "📑 Find literature",     ("analysis", "literature_finder")),
    ("explainer",         "📖 In-depth explanation",      ("explainer", None)),
    ("peer_review",       "🔍 Peer review",          ("analysis", "peer_review")),
    ("decision_analysis", "⚖️ Decision analysis", ("analysis", "decision_analysis")),
    ("research_design",   "🔬 Research design",     ("analysis", "research_design")),
    ("grant_proposal",    "💰 Grant proposal",    ("analysis", "grant_proposal")),
    ("literature_review", "📚 Literature review",    ("analysis", "literature_review")),
]

# ── Derived structures (do NOT maintain by hand) ──
RESEARCH_MODE_ORDER: list[str] = [mode_id for mode_id, _, _ in RESEARCH_MODES]
DEFAULT_RESEARCH_MODE: str = RESEARCH_MODE_ORDER[0]
_MODE_ROUTING: dict[str, Route] = {mode_id: route for mode_id, _, route in RESEARCH_MODES}

#: Mapping mode ID → use_case name (only the generic analysis modes).
ANALYSIS_MODE_MAP: dict[str, str] = {
    mode_id: uc
    for mode_id, _, (kind, uc) in RESEARCH_MODES
    if kind == "analysis" and uc is not None
}

#: Order in which the components lie in `analysis_components_flat`.
ANALYSIS_USE_CASE_ORDER: list[str] = [
    uc for _, _, (kind, uc) in RESEARCH_MODES if kind == "analysis" and uc is not None
]


def resolve_research_route(mode_id: str) -> Route:
    """Resolve a mode ID EXACTLY to its route.

    Unknown or empty IDs fall back to web research.
    """
    return _MODE_ROUTING.get(mode_id or "", ("web", None))


def mode_choices(profile=None) -> list[tuple[str, str]]:
    """(label, id) pairs for the mode dropdown.

    The institution mode is offered only when an institution profile is
    configured; its label carries the institution's short name. Labels
    are in the current interface language (src.ui.i18n).
    """
    from src.ui.i18n import tr
    if profile is None:
        from src.institution import get_profile
        profile = get_profile()
    out = []
    for mode_id, label, _ in RESEARCH_MODES:
        if mode_id == "institution":
            if not profile.configured:
                continue
            label = tr(label, institution=profile.label)
        else:
            label = tr(label)
        out.append((label, mode_id))
    return out
