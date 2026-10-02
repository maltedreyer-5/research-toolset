"""
Analysis pipeline for the long-form use cases.

Architecture (DAG-based):

  1. **Decomposer** builds a `TaskPlan` from a request (a list of
     `SubTask` nodes with `depends_on` relations). Implemented for all
     analysis use cases (see `Decomposer._decompose_*`).
  2. **TaskPlan.topological_order()** returns layers of tasks that can
     run in parallel.
  3. **AnalysisLayerNode** is a DAG node that runs one layer in parallel
     (with a semaphore, retries and injection of dependency outputs).
  4. **AnalysisPipelineRunner** wraps this in a `PipelineDAG` — one
     layer per node.

Decomposition strategy per use case:

  - **`explainer`**: makes a preliminary LLM call to identify concepts,
    then builds the TaskPlan deterministically (explanation + example
    per concept, then synthesis).
  - **other use cases**: purely deterministic from the preflight inputs
    (fixed phase structure, no preliminary LLM call).

What it provides:

  - layer execution with parallelism, semaphore, stop-signal checks
  - injection of dependency outputs (`{dep:TASK_ID}` in prompt strings)
  - retry logic per task
  - skipping of tasks whose dependencies failed
  - integration as DAG nodes

Tests verify the layer execution with hard-wired plans and the
decomposer logic per use case with a mock LLM.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional, Set

from src.pipeline.dag import (
    FailMode,
    PipelineDAG,
    PipelineNode,
)
from src.pipeline.models import SubTask, TaskPlan, TaskResult, TaskState
from src.pipeline import search_executor
from src.output_language import llm_language_name, normalize as normalize_lang, t as catalog_t
import contextvars

# Output language of the plan being decomposed; task titles (which become
# report headings) are looked up in it.
_PLAN_LANG: contextvars.ContextVar[str] = contextvars.ContextVar("plan_lang", default="en")


def language_instruction(lang: Optional[str]) -> str:
    """Instruction appended to every task prompt: prose follows the output
    language, while search queries, JSON keys and codes stay as the prompt
    asks for them."""
    return (
        f"\n\nWrite all prose in {llm_language_name(lang)}. Keep JSON keys, "
        f"codes and search queries exactly as requested above."
    )


def T(key: str, **values) -> str:
    """Catalog lookup in the output language of the current decomposition."""
    return catalog_t(key, _PLAN_LANG.get(), **values)

if TYPE_CHECKING:
    from src.llm.client import DualLLMClient
    from src.pipeline.models import HarvestContext

logger = logging.getLogger(__name__)


def _year_from(time_window: str) -> Optional[int]:
    """Extract the start year from a time-window input.

    Accepts free input such as "2020-2025", "since 2018" or "last 5
    years" (the latter returns None — the search then simply does not
    filter by year, instead of guessing).
    """
    if not time_window:
        return None
    years = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b",
                                        str(time_window))]
    if not years:
        return None
    return min(years)


def _report_title(raw: Any, max_len: int = 80) -> str:
    """Turn a free-text field into a usable heading.

    First line, whitespace normalised, shortened at a word boundary to
    `max_len`. A heading is a title, not a paragraph — a multi-line
    request as `# ...` looks like an accidentally bold prompt in every
    renderer.
    """
    if not raw:
        return ""
    lines = str(raw).strip().splitlines()
    first = " ".join(lines[0].split()) if lines else ""
    if not first:
        return ""
    if len(first) <= max_len:
        return first
    cut = first[:max_len].rsplit(" ", 1)[0] or first[:max_len]
    return cut.rstrip(" ,;:.–-") + " …"


def _as_text(value: Any, context: str = "") -> str:
    """Force a string — and report when that was necessary.

    Guards against a dict ending up in the report via `str()` as a
    Python repr (`{'empfehlung': '...', 'begruendung': [...]}`). Because
    `str()` never fails, that would go unnoticed.

    A non-string here is ALWAYS an error in the calling path, never a
    valid output. It is therefore logged (ERROR, with its origin) and
    then rendered so that at least something readable appears in the
    report — JSON instead of a Python repr.
    """
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    logger.error(
        "Task output is %s instead of str%s — please check output_format. "
        "Rendered as JSON.",
        type(value).__name__,
        f" ({context})" if context else "",
    )
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(value)


# ── Use-case detection ───────────────────────────────────────────


ANALYSIS_USE_CASES: frozenset = frozenset({
    "explainer",
    "peer_review",
    "decision_analysis",
    "grant_proposal",
    "literature_review",
    "research_design",
    "literature_finder",
})


def is_analysis_use_case(mode: str) -> bool:
    """True if `mode` is one of the analysis use cases.

    Analysis use cases run via the AnalysisPipelineRunner (decomposer →
    DAG layer execution), not via the standard research pipeline.
    """
    return mode in ANALYSIS_USE_CASES


# ── Preflight-Checker ────────────────────────────────────────────


@dataclass
class Requirement:
    """A mandatory input of an analysis use case.

    Attributes:
        field: machine key (e.g. "topic"). Used as the dict key when the
               UI inputs are passed to the decomposer.
        label: human-readable label (for the UI).
        kind: "text" (single line), "textarea" (multi-line),
              "choice" (dropdown).
        choices: if kind=="choice": the allowed values, either plain
                 strings or (label, value) pairs — the value is stored and
                 validated, the label is shown.
        placeholder: optional placeholder text in the UI.
        required: if False, the field is optional (validation allows an
                  empty value). Default True.
        max_length: optional length limit.
        min_length: optional minimum length for text fields.
    """
    field: str
    label: str
    kind: str = "text"
    choices: Optional[list] = None
    placeholder: str = ""
    required: bool = True
    max_length: Optional[int] = None
    min_length: Optional[int] = None


def choice_values(choices) -> list[str]:
    """Stored values of a choice list (plain strings or (label, value) pairs)."""
    return [c[1] if isinstance(c, (tuple, list)) else c for c in (choices or [])]


def choice_label(choices, value: str) -> str:
    """Display label for a stored choice value (the value itself if unknown)."""
    for c in choices or []:
        if isinstance(c, (tuple, list)) and c[1] == value:
            return c[0]
    return value


# Stable choice codes of the analysis forms: (label shown, value stored).
EXPLAINER_LENGTHS = [
    ("short (about 3,000 words)", "short"),
    ("medium (about 8,000 words)", "medium"),
    ("detailed (about 15,000 words)", "detailed"),
]
EXPLAINER_PURPOSES = [
    ("Self-study", "self_study"),
    ("Teaching preparation", "teaching"),
    ("Decision support", "decision_support"),
    ("General understanding", "general_understanding"),
]
REVIEW_FOCUS = [
    ("Methodology", "methodology"),
    ("Theory", "theory"),
    ("Empirical evidence", "empirical"),
    ("Argumentation", "argumentation"),
    ("Complete review", "full"),
]
DESIGN_PREFERENCES = [
    ("Quantitative", "quantitative"),
    ("Qualitative", "qualitative"),
    ("Mixed methods", "mixed_methods"),
    ("Still open", "open"),
]


class PreflightChecker:
    """Validate the mandatory inputs of an analysis use case.

    Interface (see gradio_app.py):
      - `get_requirements() -> list[Requirement]`
      - `validate(inputs: dict) -> tuple[bool, list[str]]`
      - `normalize(inputs: dict) -> dict`
    """

    def __init__(self, requirements: list[Requirement]):
        self._requirements = list(requirements)

    def get_requirements(self) -> list[Requirement]:
        return list(self._requirements)

    def normalize(self, inputs: dict) -> dict:
        """Prepare the validated inputs for the decomposer.

        `gradio_app.py` calls this directly after `validate()`; without
        it every start of an analysis mode would fail with an attribute
        error AFTER validation passed, which would look like a problem
        with the form.

        Deliberately minimal: the decomposers read the raw values
        directly from the dict (`preflight.get("options", "")`) and parse
        lines themselves, so there is nothing to convert here.
        `Requirement` has no `default` field, which is why no defaults
        are filled in.

        Still a real method rather than `dict(inputs)` at the call site:
        the contract belongs to the checker, so that use-case specific
        preparation can be overridden here.
        """
        return dict(inputs)

    def validate(self, inputs: dict) -> tuple[bool, list[str]]:
        """Check an inputs map against the requirements.

        Returns:
            (ok, errors). With `ok=True` errors is empty; with `ok=False`
            errors contains human-readable descriptions of all violations.
        """
        from src.ui.i18n import tr  # messages are shown in the interface

        errors: list[str] = []
        for req in self._requirements:
            label = tr(req.label)
            value = inputs.get(req.field)
            # Treat empty strings as "missing"
            is_empty = value is None or (
                isinstance(value, str) and not value.strip()
            )
            if req.required and is_empty:
                errors.append(tr("Field “{label}” is required", label=label))
                continue
            if is_empty:
                continue  # optional + empty → OK
            # Choice validation
            if req.kind == "choice" and req.choices:
                if value not in choice_values(req.choices):
                    errors.append(
                        tr("“{label}”: value {value} is not one of the "
                           "allowed options {options}", label=label,
                           value=repr(value), options=choice_values(req.choices))
                    )
            # Length validation
            if (req.min_length is not None and isinstance(value, str)
                    and len(value.strip()) < req.min_length):
                errors.append(
                    tr("“{label}”: at least {min} characters (currently {n})",
                       label=label, min=req.min_length, n=len(value.strip()))
                )
            if req.max_length is not None and isinstance(value, str):
                if len(value) > req.max_length:
                    errors.append(
                        tr("“{label}”: at most {max} characters (currently {n})",
                           label=label, max=req.max_length, n=len(value))
                    )
        return (not errors, errors)


# ── USE_CASE_REGISTRY ────────────────────────────────────────────
#
# One entry per analysis use case with at least a PreflightChecker.
# Further keys (e.g. separate decomposer specifications) can be
# added.
#
# The list of mandatory fields is conservatively minimal — domain
# experts can extend it per use case. What matters is that the UI
# reacts generically (renders the fields dynamically), see
# `src/ui/components/preflight_form.py`.


def _explainer_checker() -> PreflightChecker:
    return PreflightChecker([
        Requirement(
            field="topic", label="Topic",
            placeholder="e.g. How does reinforcement learning work?",
            min_length=10, max_length=500,
        ),
        Requirement(
            field="audience", label="Audience",
            placeholder="e.g. computer science undergraduates, no prior ML knowledge",
            min_length=15, max_length=300,
        ),
        Requirement(
            field="length", label="Length",
            kind="choice", choices=EXPLAINER_LENGTHS,
        ),
        Requirement(
            field="purpose", label="Purpose",
            kind="choice", choices=EXPLAINER_PURPOSES, required=False,
        ),
    ])


def _peer_review_checker() -> PreflightChecker:
    return PreflightChecker([
        Requirement(
            field="manuscript_summary", label="Manuscript summary",
            kind="textarea", max_length=3000,
        ),
        Requirement(
            field="discipline", label="Discipline",
            placeholder="e.g. sociology, computer science, biology",
        ),
        Requirement(
            field="review_focus", label="Review focus",
            kind="choice",
            choices=REVIEW_FOCUS,
        ),
    ])


def _decision_analysis_checker() -> PreflightChecker:
    return PreflightChecker([
        Requirement(
            field="decision", label="Decision",
            kind="textarea",
            placeholder="What needs to be decided?",
            max_length=1500,
        ),
        Requirement(
            field="options", label="Options",
            kind="textarea",
            placeholder="Options, one per line",
            max_length=2000,
        ),
        Requirement(
            field="criteria", label="Assessment criteria",
            kind="textarea",
            placeholder="Criteria, one per line",
            required=False, max_length=2000,
        ),
    ])


def _grant_proposal_checker() -> PreflightChecker:
    return PreflightChecker([
        Requirement(
            field="research_question", label="Research question",
            kind="textarea", max_length=1500,
        ),
        Requirement(
            field="funder", label="Funder",
            placeholder="e.g. a national research foundation, EU Horizon",
        ),
        Requirement(
            field="duration", label="Project duration",
            placeholder="e.g. 36 months",
        ),
        Requirement(
            field="discipline", label="Discipline",
            required=False,
        ),
    ])


def _literature_review_checker() -> PreflightChecker:
    return PreflightChecker([
        Requirement(
            field="topic", label="Review topic",
            kind="textarea", max_length=1500,
        ),
        Requirement(
            field="discipline", label="Discipline",
        ),
        Requirement(
            field="time_window", label="Time window",
            placeholder="e.g. 2015-2025",
            required=False,
        ),
        Requirement(
            field="key_questions", label="Key questions",
            kind="textarea",
            placeholder="One question per line",
            required=False, max_length=2000,
        ),
    ])


def _research_design_checker() -> PreflightChecker:
    return PreflightChecker([
        Requirement(
            field="research_question", label="Research question",
            kind="textarea", max_length=1500,
        ),
        Requirement(
            field="discipline", label="Discipline",
        ),
        Requirement(
            field="design_preference", label="Design preference",
            kind="choice",
            choices=DESIGN_PREFERENCES,
            required=False,
        ),
    ])


def _literature_finder_checker() -> PreflightChecker:
    return PreflightChecker([
        Requirement(
            field="topic", label="Search topic",
            kind="textarea", max_length=1500,
        ),
        Requirement(
            field="discipline", label="Discipline",
            required=False,
        ),
        Requirement(
            field="time_window", label="Time window",
            placeholder="e.g. 2020-2025",
            required=False,
        ),
    ])


USE_CASE_REGISTRY: dict[str, dict] = {
    "explainer": {"preflight": _explainer_checker()},
    "peer_review": {"preflight": _peer_review_checker()},
    "decision_analysis": {"preflight": _decision_analysis_checker()},
    "grant_proposal": {"preflight": _grant_proposal_checker()},
    "literature_review": {"preflight": _literature_review_checker()},
    "research_design": {"preflight": _research_design_checker()},
    "literature_finder": {"preflight": _literature_finder_checker()},
}






# ── Decomposer ────────────────────────────────────────────────────


class Decomposer:
    """Break a user request down into a TaskPlan.

    Dispatcher pattern: there is a `_decompose_<use_case>` method for
    every use case. The decomposition strategy can differ per use case:

    - **`explainer`**: makes a preliminary LLM call to identify the
      concepts to explain and builds the TaskPlan from them
      deterministically (explanation + example per concept, then
      synthesis).

    - **`peer_review`**, **`decision_analysis`**, **`grant_proposal`**,
      **`research_design`**, **`literature_review`**,
      **`literature_finder`**: build the TaskPlan purely
      deterministically from the preflight inputs — a fixed phase
      structure, no LLM call for the decomposition. They are less
      flexible than `explainer`, but run reliably without further
      prompt tuning.

    If `decompose` gets an unknown `use_case`, an empty TaskPlan is
    returned with a warning in the log — the caller recognises this by
    `len(plan.tasks) == 0` and raises the appropriate error.
    """

    def __init__(self, llm_client: "DualLLMClient"):
        self.llm_client = llm_client
        self.lang = "en"

    async def decompose(
        self,
        query: str,
        use_case: str,
        preflight_data: Optional[dict] = None,
        lang: Optional[str] = None,
    ) -> TaskPlan:
        # Task titles become headings in the report: they follow the
        # output language, the prompts themselves stay English.
        self.lang = normalize_lang(lang)
        _PLAN_LANG.set(self.lang)
        method = getattr(self, f"_decompose_{use_case}", None)
        if method is None:
            logger.warning(
                "Decomposer: no _decompose_%s method — empty TaskPlan",
                use_case,
            )
            return TaskPlan(use_case=use_case, tasks=[])

        try:
            return await method(query, preflight_data or {})
        except Exception as e:
            logger.error(
                "Decomposer._decompose_%s failed: %s",
                use_case, e, exc_info=True,
            )
            return TaskPlan(use_case=use_case, tasks=[])

    # ── Helper ──

    async def _llm_json(
        self,
        prompt: str,
        expected_keys: list[str],
        default: dict,
    ) -> dict:
        """LLM call with JSON output and a fallback for broken responses.

        `primary_complete_json` already returns a parsed dict (see
        DualLLMClient.primary_complete_json -> dict). This function only
        adds a protective layer:
        - non-dict output → default
        - missing expected keys → default

        `_llm_json` is thus decoupled from the concrete form of the LLM
        output and works against mocks that return dict responses (the
        real behaviour).
        """
        response = await self.llm_client.primary_complete_json(
            [{"role": "user", "content": prompt}],
        )
        if not isinstance(response, dict):
            return default
        if expected_keys and not all(k in response for k in expected_keys):
            return default
        return response

    # ── explainer ──

    async def _decompose_explainer(
        self, query: str, preflight: dict,
    ) -> TaskPlan:
        """Explainer: concepts via LLM, then explanation + example per
        concept, then synthesis."""
        topic = preflight.get("topic") or query
        audience = preflight.get("audience", "")
        length = choice_label(EXPLAINER_LENGTHS, preflight.get("length", "medium"))
        purpose = choice_label(EXPLAINER_PURPOSES, preflight.get("purpose", ""))

        # Step 1: identify the concepts via the LLM
        purpose_line = (
            f"Purpose: {purpose}\n" if purpose else ""
        )
        concepts_prompt = (
            f"You are planning an explanation of the following topic:\n\n"
            f"TOPIC: {topic}\n"
            f"AUDIENCE: {audience}\n"
            f"LENGTH: {length}\n"
            f"{purpose_line}"
            f"\nIdentify 3-5 core concepts the reader has to "
            f"understand to grasp the topic.\n"
            f"Answer as JSON:\n"
            f'{{\n  "concepts": ["Concept A", "Concept B", ...]\n}}'
        )
        parsed = await self._llm_json(
            concepts_prompt + language_instruction(self.lang),
            expected_keys=["concepts"],
            default={"concepts": [topic]},
        )
        concepts = parsed.get("concepts", []) or [topic]
        if not isinstance(concepts, list):
            concepts = [str(concepts)]
        concepts = [str(c).strip() for c in concepts if str(c).strip()]
        if not concepts:
            concepts = [topic]
        # At most 5 concepts
        concepts = concepts[:5]

        # Step 2: build the TaskPlan deterministically
        tasks: list[SubTask] = []
        explanation_ids: list[str] = []
        example_ids: list[str] = []

        for i, concept in enumerate(concepts, start=1):
            exp_id = f"EXP_{i}"
            ex_id = f"EX_{i}"

            tasks.append(SubTask(
                id=exp_id,
                phase="explanation",
                description=T("analysis.explainer.explanation", concept=concept),
                prompt_template=(
                    "Explain {concept} so that {audience} can follow. "
                    "Length: one to two paragraphs. "
                    "Write clearly."
                ),
                prompt_params={
                    "concept": concept,
                    "audience": audience,
                },
            ))
            explanation_ids.append(exp_id)

            tasks.append(SubTask(
                id=ex_id,
                phase="example",
                description=T("analysis.explainer.example", concept=concept),
                prompt_template=(
                    "Give a concrete, vivid example of "
                    "{concept} that illustrates the following explanation:\n\n"
                    "{explanation}\n\n"
                    "Keep the example short and to the point."
                ),
                prompt_params={"concept": concept},
                param_deps={"explanation": exp_id},
                depends_on=[exp_id],
            ))
            example_ids.append(ex_id)

        # Step 3: synthesis
        tasks.append(SubTask(
            id="SYN",
            phase="synthesis",
            description=T("analysis.explainer.synthesis", topic=topic),
            prompt_template=(
                "Synthesise the following explanations and examples "
                "into one coherent explanation of {topic} for "
                "{audience}.\n\n"
                "Explanations:\n{explanations}\n\n"
                "Examples:\n{examples}\n\n"
                "Write a flowing text with an introduction, "
                "the main concepts and a conclusion. "
                "Length: {length}."
            ),
            prompt_params={
                "topic": topic,
                "audience": audience,
                "length": length,
            },
            concat_param_deps={
                "explanations": explanation_ids,
                "examples": example_ids,
            },
            depends_on=explanation_ids + example_ids,
            model_preference="primary",  # the synthesis needs a good model
        ))

        return TaskPlan(
            use_case="explainer",
            tasks=tasks,
            estimated_calls=len(tasks),
        )

    # ── peer_review ──

    async def _decompose_peer_review(
        self, query: str, preflight: dict,
    ) -> TaskPlan:
        """Peer review: four parallel aspect tasks, then a recommendation.

        Deterministic — no preliminary LLM call. The aspects are fixed
        (strengths, weaknesses, methodology, application). With
        review_focus, individual aspects can be dropped.
        """
        manuscript = preflight.get("manuscript_summary", query)
        discipline = preflight.get("discipline", "")
        focus = preflight.get("review_focus", "full")

        # Which aspects are checked?
        if focus == "methodology":
            aspects = [("methodology", T("analysis.peer_review.methodology"))]
        elif focus == "theory":
            aspects = [("theory", T("analysis.peer_review.theory"))]
        elif focus == "empirical":
            aspects = [("empirics", T("analysis.peer_review.empirics"))]
        elif focus == "argumentation":
            aspects = [("argumentation", T("analysis.peer_review.argumentation"))]
        else:
            # "complete review"
            aspects = [
                ("strengths", T("analysis.peer_review.strengths")),
                ("weaknesses", T("analysis.peer_review.weaknesses")),
                ("methodology", T("analysis.peer_review.methodology")),
                ("relevance", T("analysis.peer_review.relevance")),
            ]

        tasks: list[SubTask] = []
        aspect_ids: list[str] = []
        for key, label in aspects:
            aid = f"ASPECT_{key}"
            tasks.append(SubTask(
                id=aid,
                phase="aspect_evaluation",
                description=label,
                prompt_template=(
                    "You are a peer reviewer for a manuscript in the "
                    "field of {discipline}. Assess the following aspect: "
                    "**{label}**.\n\n"
                    "Manuscript summary:\n{manuscript}\n\n"
                    "Write 2-3 concrete, factual paragraphs. "
                    "Support observations with references to the manuscript."
                ),
                prompt_params={
                    "discipline": discipline or "this field",
                    "label": label,
                    "manuscript": manuscript,
                },
            ))
            aspect_ids.append(aid)

        # Empfehlung
        tasks.append(SubTask(
            id="REC",
            phase="recommendation",
            description=T("analysis.peer_review.recommendation"),
            prompt_template=(
                "Based on the following aspect assessments, give a "
                "reasoned reviewer recommendation "
                "(accept / minor revision / major revision / reject) "
                "with concrete requirements.\n\n"
                "{assessments}\n\n"
                "Format: the recommendation as a bold heading, "
                "the reasoning in 2-3 paragraphs, and finally a "
                "numbered list of concrete requirements."
            ),
            concat_param_deps={"assessments": aspect_ids},
            depends_on=aspect_ids,
            model_preference="primary",
        ))

        return TaskPlan(
            use_case="peer_review",
            tasks=tasks,
            estimated_calls=len(tasks),
        )

    # ── decision_analysis ──

    async def _decompose_decision_analysis(
        self, query: str, preflight: dict,
    ) -> TaskPlan:
        """Decision analysis: options × criteria matrix → recommendation.

        Deterministic. Options and criteria are parsed from the preflight
        inputs (one per line).
        """
        decision = preflight.get("decision", query)
        options = self._parse_lines(preflight.get("options", ""))
        criteria = self._parse_lines(preflight.get("criteria", ""))

        if not options:
            options = ["Option A", "Option B"]
        if not criteria:
            criteria = [
                T("analysis.decision.criterion_feasibility"), T("analysis.decision.criterion_cost"),
                T("analysis.decision.criterion_time"), T("analysis.decision.criterion_risks"),
            ]

        tasks: list[SubTask] = []
        cell_ids: list[str] = []

        # One task per option × criterion
        for oi, option in enumerate(options[:6], start=1):
            for ci, criterion in enumerate(criteria[:6], start=1):
                cid = f"CELL_{oi}_{ci}"
                tasks.append(SubTask(
                    id=cid,
                    phase="cell_evaluation",
                    description=f"{option} × {criterion}",
                    prompt_template=(
                        "Assess the option \"{option}\" with regard to "
                        "the criterion \"{criterion}\" for the "
                        "decision: {decision}\n\n"
                        "Give a brief, factual assessment "
                        "(2-3 sentences) and a rating "
                        "(strongly positive / positive / neutral / "
                        "negative / strongly negative)."
                    ),
                    prompt_params={
                        "option": option,
                        "criterion": criterion,
                        "decision": decision,
                    },
                ))
                cell_ids.append(cid)

        # Recommendation from the matrix
        tasks.append(SubTask(
            id="REC",
            phase="recommendation",
            description=T("analysis.decision.recommendation"),
            prompt_template=(
                "You have the following assessment matrix for the "
                "decision \"{decision}\":\n\n"
                "{matrix}\n\n"
                "Synthesise it into a reasoned recommendation. "
                "Start with a clear one-sentence recommendation, "
                "then give the reasoning with 3-5 main arguments, "
                "and finish with a short risk assessment."
            ),
            prompt_params={"decision": decision},
            concat_param_deps={"matrix": cell_ids},
            depends_on=cell_ids,
            model_preference="primary",
        ))

        return TaskPlan(
            use_case="decision_analysis",
            tasks=tasks,
            estimated_calls=len(tasks),
        )

    # ── grant_proposal ──

    async def _decompose_grant_proposal(
        self, query: str, preflight: dict,
    ) -> TaskPlan:
        """Grant proposal: fixed section structure, generated in parallel,
        then a coherence check."""
        question = preflight.get("research_question", query)
        funder = preflight.get("funder", "the funder")
        duration = preflight.get("duration", "36 Monate")
        discipline = preflight.get("discipline", "")

        # Standard sections of a funding proposal
        sections = [
            ("abstract", T("analysis.grant.abstract"),
             "Write a 200-word summary of the proposal."),
            ("background", T("analysis.grant.background"),
             "Outline the current state of research and the "
             "open question this proposal addresses. "
             "Length: 3-4 paragraphs."),
            ("research_question", T("analysis.grant.research_question"),
             "State the research question precisely and derive "
             "hypotheses. Length: 2 paragraphs."),
            ("methodology", T("analysis.grant.methodology"),
             "Describe the approach: data, methods, "
             "analysis strategy. Length: 4-6 paragraphs."),
            ("workplan", T("analysis.grant.workplan"),
             "Draw up a work plan with milestones over "
             "the duration given in the proposal header. Length: 3 paragraphs."),
            ("impact", T("analysis.grant.impact"),
             "Describe the expected contribution of the project. "
             "Length: 2-3 paragraphs."),
        ]

        tasks: list[SubTask] = []
        section_ids: list[str] = []

        for key, label, instr in sections:
            sid = f"SEC_{key}"
            # Only the sections that rest on the literature get the hit
            # list — the work plan and the summary do not need it and would
            # only use up context.
            uses_literature = key in ("background", "research_question")
            tasks.append(SubTask(
                id=sid,
                phase="section_drafting",
                description=label,
                prompt_template=(
                    "Write the section **{label}** of a "
                    "funding proposal to {funder}.\n\n"
                    "Research question: {question}\n"
                    "Discipline: {discipline}\n"
                    "Duration: {duration}\n\n"
                    "{instr}\n\n"
                    + ("Literature found:\n{hits}\n\n"
                       "Refer to these works by title and "
                       "year. Do **not** name sources that are not "
                       "in the list.\n\n"
                       if uses_literature else "")
                    + "Write in a precise academic style."
                ),
                prompt_params={
                    "label": label,
                    "funder": funder,
                    "question": question,
                    "discipline": discipline or "the relevant field",
                    "duration": duration,
                    "instr": instr,
                },
                param_deps=({"hits": "SEARCH"} if uses_literature else {}),
                depends_on=(["SEARCH"] if uses_literature else []),
                model_preference="primary",
                # The sections ARE the proposal. Without this flag the report
                # would only include the coherence check, because it depends
                # on all sections — the proposal itself would be dropped.
                show_in_report=True,
            ))
            section_ids.append(sid)

        # Order in `tasks` = order in the report (execution depends only
        # on depends_on). That is why the search queries and the hit list
        # come AFTER the sections here: first the proposal, then the
        # literature it rests on, then the coherence check.
        # ── Substantiate the state of research instead of asserting it ──
        # Without a real search, the "state of research" would come only
        # from the model's memory. In a proposal that gets reviewed, an
        # invented reference is the most expensive mistake imaginable.
        tasks.append(SubTask(
            id="QUERIES",
            phase="queries",
            description=T("analysis.common.queries"),
            prompt_template=(
                "Formulate 5-8 search queries for bibliographic "
                "databases on the state of research.\n\n"
                "Research question: {question}\n"
                "Discipline: {discipline}\n\n"
                "Important: these queries are run automatically. "
                "Use short, meaningful keyword combinations "
                "in English; no database syntax, no "
                "Boolean operators.\n\n"
                "Answer as JSON: {{\"queries\": [\"...\", \"...\"]}}"
            ),
            prompt_params={"question": question,
                           "discipline": discipline or "the relevant field"},
            output_format="json",
        ))
        tasks.append(SubTask(
            id="SEARCH",
            phase="search",
            description=T("analysis.common.literature_found"),
            prompt_template="",
            depends_on=["QUERIES"],
            executor="literature_search",
            search_config={
                "queries_from": "QUERIES",
                "limit": 20,
                "fallback_query": question,
                "expand_citations": True,
                "max_seeds": 4,
                "max_per_seed": 12,
            },
            show_in_report=True,
        ))


        # Coherence check across all sections
        tasks.append(SubTask(
            id="COHERENCE",
            phase="coherence_check",
            description=T("analysis.grant.coherence"),
            prompt_template=(
                "You are checking a funding proposal for coherence and "
                "scholarly rigour.\n\n"
                "Sections:\n{sections_text}\n\n"
                "Provide:\n"
                "1. Inconsistencies between sections (if any)\n"
                "2. Gaps in the argument\n"
                "3. Concrete suggestions for improvement\n"
            ),
            concat_param_deps={"sections_text": section_ids},
            depends_on=section_ids,
            model_preference="primary",
        ))

        return TaskPlan(
            use_case="grant_proposal",
            tasks=tasks,
            estimated_calls=len(tasks),
        )

    # ── research_design ──

    async def _decompose_research_design(
        self, query: str, preflight: dict,
    ) -> TaskPlan:
        """Research design: research question → literature gap → method →
        hypotheses → limitations → integration."""
        question = preflight.get("research_question", query)
        discipline = preflight.get("discipline", "")
        design_pref = choice_label(
            DESIGN_PREFERENCES, preflight.get("design_preference", "open"))

        common_params = {
            "question": question,
            "discipline": discipline or "the relevant discipline",
            "design_pref": design_pref,
        }

        tasks: list[SubTask] = []

        # 1. Literature gap
        tasks.append(SubTask(
            id="GAP",
            phase="gap_analysis",
            description=T("analysis.design.gap"),
            prompt_template=(
                "Identify the research gap for the question "
                "\"{question}\" in {discipline}. What is known? "
                "What is not? Length: 3-4 paragraphs.\n\n"
                "Important: do **not** invent concrete studies, "
                "authors or years**. Describe "
                "**research directions, concepts and open questions** "
                "in the abstract instead. If you are not sure whether a statement "
                "is supported, phrase it as a conjecture."
            ),
            prompt_params=common_params,
        ))

        # 2. Hypotheses (depend on the gap)
        tasks.append(SubTask(
            id="HYPO",
            phase="hypothesis_formation",
            description=T("analysis.design.hypotheses"),
            prompt_template=(
                "Based on the following gap analysis, formulate "
                "2-4 testable hypotheses for the question "
                "\"{question}\".\n\nGap analysis:\n{gap}"
            ),
            prompt_params=common_params,
            param_deps={"gap": "GAP"},
            depends_on=["GAP"],
        ))

        # 3. Methodology (depends on the hypotheses)
        tasks.append(SubTask(
            id="METHOD",
            phase="methodology",
            description=T("analysis.design.method"),
            prompt_template=(
                "Propose a suitable methodology "
                "(design preference: {design_pref}).\n\n"
                "Hypotheses:\n{hypotheses}\n\n"
                "Scope: data collection, analysis strategy, "
                "sampling considerations. 4-6 paragraphs."
            ),
            prompt_params=common_params,
            param_deps={"hypotheses": "HYPO"},
            depends_on=["HYPO"],
        ))

        # 4. Limitations (in parallel to METHOD, depend on HYPO)
        tasks.append(SubTask(
            id="LIMIT",
            phase="limitations",
            description=T("analysis.design.limitations"),
            prompt_template=(
                "Discuss methodological and conceptual "
                "limitations of a study on "
                "\"{question}\" with the following hypotheses:\n\n"
                "{hypotheses}\n\n"
                "Length: 2-3 paragraphs."
            ),
            prompt_params=common_params,
            param_deps={"hypotheses": "HYPO"},
            depends_on=["HYPO"],
        ))

        # 5. Integration
        tasks.append(SubTask(
            id="DESIGN",
            phase="integration",
            description=T("analysis.design.draft"),
            prompt_template=(
                "Synthesise a consistent research design:\n\n"
                "Research question: {question}\n\n"
                "Literature gap:\n{gap}\n\n"
                "Hypotheses:\n{hypotheses}\n\n"
                "Methodology:\n{method}\n\n"
                "Limitations:\n{limitations}\n\n"
                "Provide a compact design draft with clear "
                "sections."
            ),
            prompt_params={"question": question},
            param_deps={
                "gap": "GAP",
                "hypotheses": "HYPO",
                "method": "METHOD",
                "limitations": "LIMIT",
            },
            depends_on=["GAP", "HYPO", "METHOD", "LIMIT"],
            model_preference="primary",
        ))

        return TaskPlan(
            use_case="research_design",
            tasks=tasks,
            estimated_calls=len(tasks),
        )

    # ── literature_review ──

    async def _decompose_literature_review(
        self, query: str, preflight: dict,
    ) -> TaskPlan:
        """Literature review: one synthesis per key_question, then a
        meta-synthesis.

        If key_questions are empty, it falls back to a deterministic
        standard structure (clarification of terms, main debates, open
        questions).
        """
        topic = preflight.get("topic", query)
        discipline = preflight.get("discipline", "")
        time_window = preflight.get("time_window", "")
        key_questions = self._parse_lines(preflight.get("key_questions", ""))

        if not key_questions:
            # The questions become section headings: output language.
            key_questions = [
                T("analysis.review.default_question_definitions"),
                T("analysis.review.default_question_debates"),
                T("analysis.review.default_question_gaps"),
            ]

        tasks: list[SubTask] = []
        question_ids: list[str] = []

        common = {
            "topic": topic,
            "discipline": discipline or "the relevant discipline",
            "time_window": time_window or "recent years",
        }

        # ── Obtain literature before the synthesis ──
        # Made only of LLM tasks, a "systematic review" would come
        # entirely from the model's memory, without a single publication
        # being read — and the prompt would have to forbid naming authors
        # or titles. With real literature in hand, statements can refer
        # to it with evidence.
        tasks.append(SubTask(
            id="QUERIES",
            phase="queries",
            description=T("analysis.common.queries"),
            prompt_template=(
                "Formulate 5-8 search queries for bibliographic "
                "databases on the topic \"{topic}\".\n\n"
                "Discipline: {discipline}\n"
                "Period: {time_window}\n"
                "Key questions:\n{questions_text}\n\n"
                "Important: these queries are run automatically. "
                "Use short, meaningful keyword combinations "
                "in English; no database syntax, no "
                "Boolean operators, no field names.\n\n"
                "Answer as JSON: {{\"queries\": [\"...\", \"...\"]}}"
            ),
            prompt_params={**common,
                           "questions_text": "\n".join(f"- {q}" for q in
                                               key_questions[:5])},
            output_format="json",
        ))
        tasks.append(SubTask(
            id="SEARCH",
            phase="search",
            description=T("analysis.review.literature_base"),
            prompt_template="",
            depends_on=["QUERIES"],
            executor="literature_search",
            search_config={
                "queries_from": "QUERIES",
                "limit": 25,
                "fallback_query": topic,
                # Citation chasing: for a review, coverage matters, not
                # speed.
                "expand_citations": True,
                "max_seeds": 5,
                "max_per_seed": 15,
                **({"year_from": _year_from(time_window)}
                   if _year_from(time_window) else {}),
            },
            show_in_report=True,
        ))

        for i, q in enumerate(key_questions[:5], start=1):
            qid = f"Q_{i}"
            tasks.append(SubTask(
                id=qid,
                phase="question_synthesis",
                description=q[:60],
                prompt_template=(
                    "Synthesise the state of research on the question:\n"
                    "{question}\n\n"
                    "Overall topic: {topic}\n"
                    "Discipline: {discipline}\n"
                    "Period: {time_window}\n\n"
                    "Literature found:\n{hits}\n\n"
                    "Write 3-5 paragraphs. Base them on the works "
                    "listed above and name them by title "
                    "and year where you refer to them.\n\n"
                    "Do **not** invent authors, titles, years "
                    "or DOIs that are not in the list. Where the "
                    "list does not cover a question, describe the "
                    "research direction in the abstract and mark it "
                    "as not supported."
                ),
                prompt_params={**common, "question": q},
                param_deps={"hits": "SEARCH"},
                depends_on=["SEARCH"],
            ))
            question_ids.append(qid)

        # Meta-synthesis
        tasks.append(SubTask(
            id="META",
            phase="meta_synthesis",
            description=T("analysis.review.meta_synthesis"),
            prompt_template=(
                "Write a meta-synthesis on the topic \"{topic}\" "
                "based on the following question syntheses:\n\n"
                "{syntheses}\n\n"
                "Structure: introduction, main findings, "
                "fields of debate, research outlook."
            ),
            prompt_params={"topic": topic},
            concat_param_deps={"syntheses": question_ids},
            depends_on=question_ids,
            model_preference="primary",
        ))

        return TaskPlan(
            use_case="literature_review",
            tasks=tasks,
            estimated_calls=len(tasks),
        )

    # ── literature_finder ──

    async def _decompose_literature_finder(
        self, query: str, preflight: dict,
    ) -> TaskPlan:
        """Literature finder: strategy → queries → SEARCH → assessment.

        `SEARCH` actually runs the queries against the literature APIs
        (OpenAlex + Semantic Scholar + arXiv), and `ASSESS` evaluates the
        hits against the inclusion and exclusion criteria from the
        strategy — a mode called "find literature" that only proposed
        search terms would leave the actual searching to the user.
        """
        topic = preflight.get("topic", query)
        discipline = preflight.get("discipline", "")
        time_window = preflight.get("time_window", "")

        common = {
            "topic": topic,
            "discipline": discipline or "the relevant discipline",
            "time_window": time_window or "not specified",
        }

        tasks: list[SubTask] = [
            SubTask(
                id="STRATEGY",
                phase="strategy",
                description=T("analysis.finder.strategy"),
                prompt_template=(
                    "Propose a systematic search strategy for "
                    "literature on the topic \"{topic}\".\n\n"
                    "Discipline: {discipline}\n"
                    "Time window: {time_window}\n\n"
                    "Provide:\n"
                    "1. 5-8 search terms\n"
                    "2. Recommended databases\n"
                    "3. Inclusion and exclusion criteria"
                ),
                prompt_params=common,
                # The strategy is a result in its own right; without this it
                # would be dropped because a follow-up task depends on it.
                show_in_report=True,
            ),
            SubTask(
                id="QUERIES",
                phase="queries",
                description=T("analysis.finder.queries"),
                prompt_template=(
                    "Based on the search strategy, formulate 5-8 "
                    "concrete search queries for bibliographic "
                    "databases (OpenAlex, Semantic Scholar, arXiv).\n\n"
                    "Important: these queries are run "
                    "automatically. Use short, meaningful "
                    "keyword combinations in English; "
                    "no database syntax, no field names, no "
                    "quotation marks.\n\n"
                    "Strategy:\n{strategy}\n\n"
                    "Answer as JSON: "
                    "{{\"queries\": [\"...\", \"...\"]}}"
                ),
                prompt_params={},
                param_deps={"strategy": "STRATEGY"},
                depends_on=["STRATEGY"],
                # Structured, so that the search executor can take over the
                # queries unambiguously instead of parsing running text.
                output_format="json",
            ),
            SubTask(
                id="SEARCH",
                phase="search",
                description=T("analysis.finder.found"),
                prompt_template="",  # no LLM call
                depends_on=["QUERIES"],
                executor="literature_search",
                search_config={
                    "queries_from": "QUERIES",
                    "limit": 20,
                    # Fallback if the query task delivers nothing usable —
                    # then at least the topic itself is searched.
                    "fallback_query": topic,
                    **({"year_from": _year_from(time_window)}
                       if _year_from(time_window) else {}),
                },
                # The hit list IS the product of this mode.
                show_in_report=True,
            ),
            SubTask(
                id="ASSESS",
                phase="assessment",
                description=T("analysis.finder.assessment"),
                prompt_template=(
                    "Assess the literature found on the topic "
                    "\"{topic}\" against the search strategy.\n\n"
                    "Strategy (with inclusion/exclusion criteria):\n"
                    "{strategy}\n\n"
                    "Titles found:\n{hits}\n\n"
                    "Select the 5-10 most relevant works and "
                    "give the DOI from the list for each — the "
                    "complete bibliographic details are "
                    "added automatically afterwards, you do "
                    "NOT need to copy them.\n\n"
                    "Important: only DOIs that actually appear above. "
                    "Do not invent works.\n\n"
                    "Answer as JSON:\n"
                    "{{\"selected\": [{{\"doi\": \"...\", "
                    "\"title\": \"...\", \"reason\": \"one sentence\"}}],\n"
                    " \"gaps\": \"noticeable gaps in the hits, "
                    "as prose or a list\",\n"
                    " \"next_round\": \"recommendation for the next "
                    "search round\"}}"
                ),
                prompt_params={"topic": topic},
                param_deps={"strategy": "STRATEGY", "hits": "SEARCH"},
                depends_on=["SEARCH", "STRATEGY"],
                model_preference="primary",
                output_format="json",
            ),
            SubTask(
                id="SELECTION",
                phase="assessment",
                description=T("analysis.finder.assessment"),
                prompt_template="",
                depends_on=["ASSESS", "SEARCH"],
                executor="render_selection",
                search_config={
                    "selection_from": "ASSESS",
                    "results_from": "SEARCH",
                },
            ),
        ]

        return TaskPlan(
            use_case="literature_finder",
            tasks=tasks,
            estimated_calls=len([t for t in tasks if t.executor == "llm"]),
        )

    # ── Helper ──

    @staticmethod
    def _parse_lines(text: str) -> list[str]:
        """Parse a multi-line input into a list of trimmed strings.

        Empty lines and bullet markers (-, *, •) at the start of a line
        are removed. Lines with only special characters are discarded.
        """
        if not text:
            return []
        out: list[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            # Remove bullets
            if line[0] in "-*•":
                line = line[1:].strip()
            # Remove numbered list "1. ..."
            import re as _re
            line = _re.sub(r"^\d+[.)]\s*", "", line)
            if line:
                out.append(line)
        return out


# ── Layer node (runs one layer in parallel) ────────────────


# Pattern for dependency-output injection: `{dep:TASK_ID}`
_DEP_PATTERN = re.compile(r"\{dep:([A-Za-z0-9_\-]+)\}")
# Pattern for simple template variables: `{VAR}` (identifiers only,
# no format specs, no `dep:` prefixes)
_VAR_PATTERN = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


class AnalysisLayerNode(PipelineNode):
    """DAG node that runs one layer of SubTasks in parallel.

    A layer is a list of SubTasks without open dependencies (produced by
    `TaskPlan.topological_order()`). Within the layer the tasks run in
    parallel, limited by a semaphore.

    Before running a task:
      - check whether its dependencies succeeded (otherwise SKIPPED).
      - resolve the prompt template, substituting `{dep:TASK_ID}` with
        the outputs of the preceding tasks.

    During the run:
      - LLM call with retry logic (`SubTask.max_retries`).
      - stop signal before every task.

    After the run:
      - store the TaskResult in `ctx.task_results`.

    Mutations of ctx:
      - ctx.task_results (mapping task_id → TaskResult)
    """

    name = "analysis_layer"
    fail_mode = FailMode.CONTINUE

    def __init__(
        self,
        layer_idx: int,
        tasks: list[SubTask],
        llm_client: "DualLLMClient",
        max_parallel: int = 5,
        search_service: Any = None,
    ):
        # Unique name per layer for clear trace logs
        self.name = f"analysis_layer_{layer_idx + 1}"
        self.layer_idx = layer_idx
        self.tasks = tasks
        self.llm_client = llm_client
        self.max_parallel = max_parallel
        # Optional: service for real literature queries. Created lazily when
        # a task needs it and none was injected — so the analysis path costs
        # nothing as long as no mode searches.
        self._search_service = search_service

    def _get_search_service(self, ctx: Any):
        """Return the literature search service, building one if needed.

        The API client is cached on the context, so that several search
        tasks of one run share an HTTP session.
        """
        if self._search_service is not None:
            return self._search_service
        try:
            from src.connectors.literature_apis import LiteratureAPIClient
            from src.connectors.literature_search import LiteratureSearchService
        except Exception as e:
            logger.error("Literature search unavailable: %s", e)
            return None
        try:
            client = getattr(ctx, "_literature_api_client", None)
            if client is None:
                client = LiteratureAPIClient()
                try:
                    ctx._literature_api_client = client
                except Exception:
                    pass  # context without free attributes — fine too
            self._search_service = LiteratureSearchService(client)
            return self._search_service
        except Exception as e:
            logger.error("Search service could not be created: %s", e)
            return None

    def applies_to(self, ctx: Any) -> tuple[bool, str]:
        if not self.tasks:
            return False, "layer is empty"
        return True, ""

    async def run(self, ctx: Any) -> dict:
        # Initialise ctx.task_results (if not present yet)
        if not hasattr(ctx, "task_results") or ctx.task_results is None:
            ctx.task_results = {}

        semaphore = asyncio.Semaphore(self.max_parallel)
        executable: list[SubTask] = []
        skipped: list[SubTask] = []

        for task in self.tasks:
            failed_deps = self._failed_dependencies(task, ctx.task_results)
            if failed_deps:
                # A dependency failed → skip this task
                ctx.task_results[task.id] = TaskResult(
                    task_id=task.id,
                    state=TaskState.SKIPPED,
                    error=(
                        f"Dependencies failed: "
                        f"{', '.join(sorted(failed_deps))}"
                    ),
                )
                skipped.append(task)
            else:
                executable.append(task)

        if not executable:
            return {"layer": self.layer_idx + 1, "executed": 0,
                    "skipped": len(skipped)}

        async def run_one(task: SubTask):
            async with semaphore:
                await self._execute_task(task, ctx)

        await asyncio.gather(
            *(run_one(t) for t in executable),
            return_exceptions=True,
        )

        done = sum(
            1 for t in executable
            if ctx.task_results.get(t.id, TaskResult(task_id=t.id)).state
            == TaskState.DONE
        )
        failed = sum(
            1 for t in executable
            if ctx.task_results.get(t.id, TaskResult(task_id=t.id)).state
            == TaskState.FAILED
        )
        return {
            "layer": self.layer_idx + 1,
            "executed": len(executable),
            "done": done,
            "failed": failed,
            "skipped": len(skipped),
        }

    @staticmethod
    def _failed_dependencies(
        task: SubTask, results: dict,
    ) -> Set[str]:
        """Return the IDs of dependencies that are NOT DONE."""
        failed = set()
        for dep_id in task.depends_on:
            r = results.get(dep_id)
            if r is None or r.state != TaskState.DONE:
                failed.add(dep_id)
        return failed

    async def _execute_task(self, task: SubTask, ctx: Any) -> None:
        """Run a single SubTask, with retry logic."""
        result = TaskResult(
            task_id=task.id,
            state=TaskState.RUNNING,
            started_at=datetime.now().isoformat(),
        )
        ctx.task_results[task.id] = result

        # Prompt resolution with dependency-output injection
        try:
            prompt_text = self._resolve_prompt(task, ctx.task_results)
        except Exception as e:
            result.state = TaskState.FAILED
            result.error = f"Prompt resolution failed: {e}"
            result.finished_at = datetime.now().isoformat()
            return

        # LLM call with retry. `max_retries` from the SubTask config.
        last_error: Optional[BaseException] = None
        for attempt in range(task.max_retries + 1):
            try:
                executor = getattr(task, "executor", "llm")
                search_executor.OUTPUT_LANG.set(
                    getattr(ctx, "output_language", None) or "en")
                if executor == "literature_search":
                    response, parsed = await self._run_search_task(
                        task, ctx,
                    )
                elif executor == "render_selection":
                    response, parsed = await self._run_render_selection(
                        task, ctx,
                    )
                else:
                    response, parsed = await self._llm_complete(
                        task, prompt_text,
                        lang=getattr(ctx, "output_language", None),
                    )
                # According to the data class, `output` is a string and
                # `parsed_output` its structured counterpart — the dict must
                # not end up in `output`.
                result.output = response
                result.parsed_output = parsed or {}
                result.state = TaskState.DONE
                result.finished_at = datetime.now().isoformat()
                result.retries_used = attempt
                return
            except asyncio.CancelledError:
                # Stop signal — pass it on
                result.state = TaskState.SKIPPED
                result.error = "Cancelled"
                result.finished_at = datetime.now().isoformat()
                raise
            except Exception as e:
                last_error = e
                logger.debug(
                    "Task %s attempt %d/%d failed: %s",
                    task.id, attempt + 1, task.max_retries + 1, e,
                )
                # On the last attempt: give up
                if attempt >= task.max_retries:
                    break

        result.state = TaskState.FAILED
        result.error = str(last_error) if last_error else "Unknown error"
        result.finished_at = datetime.now().isoformat()

    @staticmethod
    def _resolve_prompt(task: SubTask, results: dict) -> str:
        """Resolve the prompt template.

        Deliberately NO str.format() substitution, because it would treat
        `{dep:TASK_ID}` markers as a variable with a format spec.
        Instead, regex substitution in this order:

          1. `{dep:TASK_ID}` → output of the dependency task (raw)
          2. `{VAR}` → value from prompt_params / param_deps /
             concat_param_deps

        - `prompt_params` are inserted directly.
        - `param_deps` are filled with the outputs of preceding tasks.
        - `concat_param_deps` concatenates several outputs.
        """
        # 1. Build the mapping: var → str
        params: dict[str, str] = {}
        for k, v in task.prompt_params.items():
            params[k] = str(v) if v is not None else ""

        # param_deps: single values from preceding outputs
        for var, dep_task_id in task.param_deps.items():
            dep_result = results.get(dep_task_id)
            if dep_result and dep_result.state == TaskState.DONE:
                params[var] = _as_text(
                    dep_result.output, context=f"param_deps {dep_task_id}",
                )
            else:
                params[var] = ""

        # concat_param_deps: concatenated list of preceding outputs
        for var, dep_task_ids in task.concat_param_deps.items():
            outputs = []
            for dep_id in dep_task_ids:
                dep_result = results.get(dep_id)
                if dep_result and dep_result.state == TaskState.DONE:
                    # Without this conversion the join below would raise
                    # "TypeError: sequence item 0: expected str instance, dict
                    # found" — a hard abort instead of merely broken formatting.
                    outputs.append(_as_text(
                        dep_result.output,
                        context=f"concat_param_deps {dep_id}",
                    ))
            params[var] = "\n\n".join(outputs)

        text = task.prompt_template

        # 2. Replace `{dep:TASK_ID}` first
        def replace_dep(m):
            dep_id = m.group(1)
            r = results.get(dep_id)
            if r and r.state == TaskState.DONE:
                return _as_text(r.output, context=f"dep {dep_id}")
            return ""

        text = _DEP_PATTERN.sub(replace_dep, text)

        # 3. Replace `{VAR}` — only simple identifiers, no format specs.
        # Real braces in the outputs (e.g. JSON, code snippets) thus stay
        # untouched.
        def replace_var(m):
            var_name = m.group(1)
            if var_name in params:
                return params[var_name]
            # Variable not defined: leave the marker, so that the caller
            # can see the problem in the output
            return m.group(0)

        text = _VAR_PATTERN.sub(replace_var, text)

        return text

    async def _run_search_task(
        self, task: SubTask, ctx: Any,
    ) -> tuple[str, dict]:
        """Run a task of type `literature_search`.

        The search queries come from the result of a preceding task
        (`search_config["queries_from"]`); if not given, the first entry
        of `depends_on` is used. Its `parsed_output` is preferred — a task
        with `output_format="json"` delivers the queries there
        unambiguously instead of as running text.
        """
        from src.pipeline.search_executor import (
            extract_queries, run_literature_search,
        )

        config = dict(getattr(task, "search_config", None) or {})
        source_id = config.pop("queries_from", None)
        if not source_id and task.depends_on:
            source_id = task.depends_on[0]

        results = getattr(ctx, "task_results", None) or {}
        source = results.get(source_id) if source_id else None

        queries = list(config.pop("queries", []) or [])
        if not queries and source is not None:
            queries = extract_queries(
                getattr(source, "output", "") or "",
                getattr(source, "parsed_output", None) or {},
            )

        # Fallback query: if the query task delivered nothing usable (empty
        # JSON, unparsable answer), the topic itself is searched instead of
        # nothing. A coarser search is worth much more than an aborted run —
        # and the log records that it was the fallback.
        fallback = str(config.pop("fallback_query", "") or "").strip()
        if not queries and fallback:
            logger.warning(
                "Search task %s: no queries from '%s' — using "
                "fallback query %r",
                task.id, source_id, fallback,
            )
            queries = [fallback]

        if not queries:
            logger.warning(
                "Search task %s: no queries determined from '%s'",
                task.id, source_id,
            )

        service = self._get_search_service(ctx)
        markdown, structured = await run_literature_search(
            queries, service, config,
        )

        # ── A failure must not pass as a success ──
        # Otherwise the log says "successful", the report has a thin note,
        # and the downstream assessment task gets an empty hit list — from
        # which it then invents plausible-sounding titles. A FAILED skips
        # the dependent tasks and appears as a gap in the report.
        #
        # Deliberate distinction: a search that ran WITHOUT hits is a valid
        # result, not an error.
        if structured.get("error") == "no_service":
            raise RuntimeError(
                "Literature search service unavailable — the databases "
                "could not be queried"
            )
        if not queries:
            raise RuntimeError(
                f"No search queries determined from '{source_id}' — "
                f"the search could not be run"
            )
        if structured.get("errors") and not structured.get("queries"):
            raise RuntimeError(
                f"All {len(structured['errors'])} search queries "
                f"failed"
            )

        logger.info(
            "Search task %s: %d query(ies) → %d hits",
            task.id, len(queries), structured.get("n_results", 0),
        )
        return markdown, structured

    async def _run_render_selection(
        self, task: SubTask, ctx: Any,
    ) -> tuple[str, dict]:
        """Render a model's selection with real metadata.

        No LLM call, no network activity: joins the assessment task's
        selection with the structured hits of the search task. The
        justification comes from the model, the bibliographic details
        from the API response.
        """
        from src.pipeline.search_executor import render_selected_references

        config = dict(getattr(task, "search_config", None) or {})
        sel_id = config.get("selection_from")
        res_id = config.get("results_from")
        results_map = getattr(ctx, "task_results", None) or {}

        sel_task = results_map.get(sel_id)
        res_task = results_map.get(res_id)
        selection = (getattr(sel_task, "parsed_output", None) or {}) \
            if sel_task else {}
        hits = ((getattr(res_task, "parsed_output", None) or {})
                .get("results", []) if res_task else [])

        markdown = render_selected_references(selection, hits)
        n_sel = len(selection.get("selected") or [])
        logger.info(
            "Render task %s: %d selection entries resolved against %d hits",
            task.id, n_sel, len(hits),
        )
        return markdown, {"n_selected": n_sel, "n_results": len(hits)}

    async def _llm_complete(
        self, task: SubTask, prompt: str, lang: Optional[str] = None,
    ) -> tuple[str, dict]:
        """LLM call with a choice of model AND format.

        Returns:
            (text, parsed). `text` is ALWAYS a string and goes to
            `TaskResult.output`; `parsed` is the parsed dict with
            `output_format="json"`, otherwise empty.

        The two axes are independent: `model_preference` picks the model,
        `output_format` the form of the answer. If the model preference
        decided both, every primary task would get a JSON call although
        these tasks' prompts ask for prose — and the resulting dict would
        end up in the report as a Python repr via `str()`.
        """
        messages = [{"role": "user", "content": prompt + language_instruction(lang)}]
        primary = task.model_preference == "primary"
        want_json = getattr(task, "output_format", "text") == "json"

        if want_json:
            parsed = await (
                self.llm_client.primary_complete_json(messages) if primary
                else self.llm_client.harvest_complete_json(messages)
            )
            if not isinstance(parsed, dict):
                parsed = {}
            # `output` stays a string — serialised, not repr(). So no Python
            # repr can reach the report or a follow-up prompt, whatever the
            # caller does with it.
            text = json.dumps(parsed, ensure_ascii=False, indent=2)
            return text, parsed

        text = await (
            self.llm_client.primary_complete(messages) if primary
            else self.llm_client.harvest_complete(messages)
        )
        return _as_text(text, context=f"Task {task.id}"), {}


# ── Pipeline-Runner ───────────────────────────────────────────────


class AnalysisPipelineRunner:
    """Run an analysis pipeline for a use case.

    Dispatch: decomposer → TaskPlan → one DAG node per layer → run.

    This class is the interface the orchestrator calls. It encapsulates:
      1. decomposition (or skipping it if a plan exists already)
      2. layer execution via PipelineDAG
      3. status aggregation

    Args:
        llm_client: DualLLMClient for LLM calls.
        progress_callback: async callback for pipeline events.
        max_parallel: maximum parallel tasks per layer.
        rate_limit: rate limit (calls/s) — currently ignored, kept for
                    API compatibility.
        stop_check: optional callable for an external stop query. If it
                    returns `True`, open tasks are aborted.
    """

    def __init__(
        self,
        llm_client: "DualLLMClient",
        progress_callback,
        max_parallel: int = 5,
        rate_limit: float = 3.0,
        stop_check=None,
        search_service: Any = None,
    ):
        self.llm_client = llm_client
        self.progress_callback = progress_callback
        self.max_parallel = max_parallel
        self.rate_limit = rate_limit
        self.stop_check = stop_check or (lambda: False)
        # Service for tasks with `executor="literature_search"`. If not
        # given, the layer node builds it itself — the parameter exists so
        # that a caller can pass on an existing API client (or a test can
        # set a stub).
        self.search_service = search_service
        self._decomposer = Decomposer(llm_client)

    async def decompose_only(self, ctx: "HarvestContext") -> "HarvestContext":
        """Plan preview phase: only build the TaskPlan, do not run it.

        Sets `ctx.task_plan`. The caller can show the plan and continue
        later with `run(ctx, skip_decompose=True)`.
        """
        try:
            ctx.task_plan = await self._decomposer.decompose(
                query=ctx.query,
                use_case=ctx.use_case,
                preflight_data=getattr(ctx, "preflight_data", None),
                lang=getattr(ctx, "output_language", None),
            )
            ctx.status = "plan_ready"
        except Exception as e:
            logger.error("Decomposer failed: %s", e, exc_info=True)
            ctx.status = "error"
            ctx.error_message = f"Decomposer: {e}"
        return ctx

    async def run(
        self,
        ctx: "HarvestContext",
        skip_decompose: bool = False,
    ) -> "HarvestContext":
        """Full run: decompose if needed, then run the layer DAG.

        Args:
            ctx: HarvestContext with `query`, `use_case` etc.
            skip_decompose: if True, `ctx.task_plan` is assumed to exist
                            already (resume after plan_only).
        """
        ctx.started_at = ctx.started_at or datetime.now().isoformat()
        ctx.status = "running"

        # 1. Decompose (or skip)
        if not skip_decompose:
            try:
                ctx.task_plan = await self._decomposer.decompose(
                    query=ctx.query,
                    use_case=ctx.use_case,
                    preflight_data=getattr(ctx, "preflight_data", None),
                    lang=getattr(ctx, "output_language", None),
                )
            except Exception as e:
                logger.error("Decomposer failed: %s", e, exc_info=True)
                ctx.status = "error"
                ctx.error_message = f"Decomposer: {e}"
                return ctx

        if ctx.task_plan is None or not ctx.task_plan.tasks:
            ctx.status = "error"
            ctx.error_message = (
                "The decomposer returned no TaskPlan — "
                "the use case is not implemented."
                ""
            )
            return ctx

        # 2. Build the layer DAG
        try:
            layers = ctx.task_plan.topological_order()
        except ValueError as e:
            ctx.status = "error"
            ctx.error_message = f"TaskPlan error: {e}"
            return ctx

        ctx.task_results = {}
        layer_nodes = [
            AnalysisLayerNode(
                layer_idx=i,
                tasks=layer,
                llm_client=self.llm_client,
                max_parallel=self.max_parallel,
                search_service=self.search_service,
            )
            for i, layer in enumerate(layers)
        ]

        pipeline = PipelineDAG(layer_nodes)

        # 3. Run the layer DAG — with a stop-signal bridge.
        # The engine knows our StopSignal, but the runner API only has
        # `stop_check`. We build a periodic bridge: a background task that
        # checks `stop_check()` every 200 ms and calls
        # `signal.request_stop()` once it is True. That way a stop click
        # takes effect even IN THE MIDDLE of a running layer.
        from src.core.stop_signal import StopSignal
        signal = StopSignal()

        async def stop_bridge():
            try:
                while True:
                    if self.stop_check():
                        signal.request_stop("external")
                        return
                    await asyncio.sleep(0.2)
            except asyncio.CancelledError:
                return

        # Preliminary check (synchronous, in case it was stopped before the start)
        if self.stop_check():
            signal.request_stop("external")

        bridge_task = asyncio.create_task(stop_bridge())

        try:
            await pipeline.run(
                ctx,
                stop_signal=signal,
                progress_callback=self.progress_callback,
            )
            ctx.status = "done"
        except asyncio.CancelledError:
            ctx.status = "cancelled"
        except Exception as e:
            logger.error(
                "Analysis pipeline error: %s", e, exc_info=True,
            )
            ctx.status = "error"
            ctx.error_message = str(e)
        finally:
            # Clean up the background task
            bridge_task.cancel()
            try:
                await bridge_task
            except asyncio.CancelledError:
                pass

        # ── Build the final report from the task results ──────────────────────
        # Without this step all outputs would sit only in ctx.task_results
        # and ctx.final_report would stay empty — the user would see
        # "done" without a visible result.
        if ctx.status == "done":
            ctx.final_report = self._build_final_report(ctx)
        ctx.finished_at = datetime.now().isoformat()
        return ctx

    @staticmethod
    def _build_final_report(ctx: "HarvestContext") -> str:
        """Synthesise the final report from the task results.

        Strategy: the final task (a task without dependents) contains the
        actual synthesis — its output is the main part. If there are
        several final tasks (e.g. literature_finder with strategy +
        queries), all are concatenated in plan order.

        Failed/skipped tasks are mentioned in the report as a warning, so
        that the user does not get a "silent gap".
        """
        plan = ctx.task_plan
        results = ctx.task_results or {}
        lang = getattr(ctx, "output_language", None)
        if not plan or not results:
            return ""

        # Identify final tasks = no other tasks depend on them
        all_deps: set[str] = set()
        for t in plan.tasks:
            all_deps.update(t.depends_on)
        final_task_ids = [t.id for t in plan.tasks if t.id not in all_deps]

        # In addition, everything that has explicitly asked to be in the
        # report. The "final tasks only" rule is right for synthesising
        # closing tasks, but not for results that are the product
        # themselves — a literature hit list, for instance, would otherwise
        # be invisible although it is the purpose of the run. The order
        # stays the plan order.
        report_ids: list[str] = []
        for t in plan.tasks:
            if t.id in final_task_ids or getattr(t, "show_in_report", False):
                report_ids.append(t.id)
        final_task_ids = report_ids

        # If there are final tasks: their outputs are the main result.
        # Otherwise fallback: concatenate all DONE tasks.
        parts: list[str] = []

        # Derive the title from preflight_data
        #
        # All free-text fields are shortened by the same rule — for the
        # decision analysis `decision` is the complete request, which would
        # otherwise open the report as an H1 heading.
        pre = getattr(ctx, "preflight_data", None) or {}
        title = _report_title(
            pre.get("topic")
            or pre.get("research_question")
            or pre.get("decision")
            or pre.get("manuscript_summary")
            or ctx.query
        )
        if title:
            parts.append(f"# {title}\n")

        # Main part: outputs of the final tasks
        if final_task_ids:
            # Headings only if several building blocks are in the report —
            # with exactly one, a "## Recommendation" above the only content
            # would just be noise.
            with_headings = len(final_task_ids) > 1
            by_id = {t.id: t for t in plan.tasks}
            for fid in final_task_ids:
                tr = results.get(fid)
                if tr and tr.state.value == "done" and tr.output:
                    if with_headings:
                        desc = getattr(by_id.get(fid), "description", fid)
                        parts.append(f"## {desc}\n")
                    parts.append(_as_text(
                        tr.output, context=f"Final-Task {fid}",
                    ).strip())
                    parts.append("")  # blank line between building blocks
        else:
            # Fallback — should not be reached for the implemented use cases,
            # but defensively helpful
            for t in plan.tasks:
                tr = results.get(t.id)
                if tr and tr.state.value == "done" and tr.output:
                    parts.append(f"## {t.description}\n")
                    parts.append(_as_text(
                        tr.output, context=f"Task {t.id}",
                    ).strip())
                    parts.append("")

        # Note on failed/skipped tasks — the user should rather see gaps
        # than think everything is fine
        problems = [
            t for t in plan.tasks
            if results.get(t.id) and results[t.id].state.value in
            ("failed", "skipped")
        ]
        if problems:
            parts.append("\n---\n")
            parts.append(catalog_t("analysis.report.incomplete", lang))
            for t in problems:
                tr = results[t.id]
                err = (tr.error or "").strip() or tr.state.value
                parts.append(f"> - {t.description}: {err}")

        return "\n".join(parts).strip()
