"""
Helpers for running the analysis pipeline from the UI.

  - `format_analysis_event(event, data)` — formats a pipeline event as a
    Markdown line for the live status display while the analysis
    pipeline runs. Knows the event types of `AnalysisLayerNode` and of
    the DAG engine (PipelineDAG).

  - `validate_explainer_inputs(...)` — validation of the mandatory
    inputs of the `explainer` use case. Wrapper around the generic
    `PreflightChecker`, with explainer-specific defaults.

  - `build_explainer_preflight_data(...)` — builds the `preflight_data`
    dict that the decomposer receives as context.

All three functions are Gradio-agnostic and unit-testable.
"""

from __future__ import annotations

from typing import Any
from src.ui.i18n import tr


def shorten_for_display(text: str, max_len: int = 80) -> str:
    """Shorten a free text to one line and display length.

    Breaks at a word boundary so that no half words appear.
    """
    if not text:
        return ""
    lines = str(text).strip().splitlines()
    first = " ".join(lines[0].split()) if lines else ""
    if not first:
        return ""
    if len(first) <= max_len:
        return first
    cut = first[:max_len].rsplit(" ", 1)[0] or first[:max_len]
    return cut.rstrip(" ,;:.–-") + " …"


def make_chat_summary(use_case: str, inputs: dict) -> str:
    """Produce a short description for the chat display.

    Looks for the first content-bearing field of the use case. The list
    covers all registered modes, including `decision`,
    `research_question` and `manuscript_summary`; without them the
    decision analysis would show its own technical name as the summary
    ("Decision analysis: decision_analysis").
    """
    for key in (
        "question", "topic", "idea",
        "decision",             # decision_analysis
        "research_question",    # research_design, grant_proposal
        "manuscript_summary",   # peer_review
    ):
        value = inputs.get(key)
        if value:
            return shorten_for_display(value, 200)
    if "paper_text" in inputs:
        return tr("Paper ({n} characters)", n=len(str(inputs['paper_text'])))
    # Last resort: any filled text field, rather than the technical
    # use-case name.
    for value in inputs.values():
        if isinstance(value, str) and value.strip():
            return shorten_for_display(value, 200)
    return use_case


# ── Event formatting ───────────────────────────────────────────


def _as_event_dict(data: Any) -> dict:
    """Unify the payload of a pipeline event.

    The DAG engine sends two different shapes:
      - `node_start` → `{"name": <node name>}`
      - `node_done` / `node_failed` → a `NodeResult` OBJECT

    A formatter that only reads `data["node"]` gets neither: with
    `node_start` the key does not match ("name" instead of "node"), and a
    caller that checks `isinstance(data, dict)` drops the `NodeResult`
    entirely — the display would read "▶️ Starting: `?` ✅ `?` done".

    This function accepts both shapes and always returns a dict with
    the key `node`.
    """
    if data is None:
        return {}

    if isinstance(data, dict):
        out = dict(data)
        # "name" is the DAG engine's key, "node" the formatter's —
        # accept both.
        if "node" not in out and out.get("name"):
            out["node"] = out["name"]
        return out

    # NodeResult (or another object with the same attributes)
    out = {}
    node_name = getattr(data, "node_name", "")
    if node_name:
        out["node"] = node_name
    metadata = getattr(data, "metadata", None)
    if isinstance(metadata, dict):
        out["metadata"] = metadata
    duration = getattr(data, "duration_seconds", None)
    if duration:
        out["duration_ms"] = int(duration * 1000)
    error = getattr(data, "error", None)
    if error:
        out["error"] = str(error)
    return out


def format_analysis_event(event: str, data: Any) -> str:
    """Format a pipeline event as a short Markdown line.

    Known event types (from `src.pipeline.dag` and
    `src.pipeline.analysis_pipeline.AnalysisLayerNode`):

      - `node_start`, `node_done`, `node_skipped`, `node_failed`
      - `pipeline_done`, `pipeline_stopped`
      - `status` (free-form status line)
      - `plan_ready` (the decomposer delivered a plan)
      - `error` (pipeline crash)

    Returns:
        Markdown line (with icon) or an empty string if the event has
        nothing worth showing in the UI.
    """
    if not event:
        return ""
    data = _as_event_dict(data)

    if event == "status":
        msg = data.get("message", data.get("text", ""))
        if isinstance(data, str):
            msg = data
        return f"⏳ {msg}" if msg else ""

    if event == "node_start":
        node = data.get("node", "?")
        return tr("▶️ Starting: `{node}`", node=node)

    if event == "node_done":
        node = data.get("node", "?")
        ms = data.get("duration_ms")
        meta = data.get("metadata") or {}
        # Layer metadata if AnalysisLayerNode
        if "done" in meta and "executed" in meta:
            extra = (
                f" — {meta['done']}/{meta['executed']} OK"
                + (tr(", {n} failed", n=meta['failed'])
                   if meta.get("failed") else "")
            )
        else:
            extra = ""
        ms_str = f" ({ms} ms)" if ms is not None else ""
        return tr("✅ `{node}` done", node=node) + f"{extra}{ms_str}"

    if event == "node_skipped":
        node = data.get("node", "?")
        reason = data.get("reason", "")
        return tr("⊘ `{node}` skipped", node=node) + (f": {reason}" if reason else "")

    if event == "node_failed":
        node = data.get("node", "?")
        error = data.get("error", tr("Unknown error"))
        # Shorten long errors
        if len(error) > 200:
            error = error[:200] + "…"
        return tr("❌ `{node}` failed: {error}", node=node, error=error)

    if event == "pipeline_done":
        n = data.get("nodes", 0)
        return tr("🏁 Pipeline finished ({n} nodes)", n=n)

    if event == "pipeline_stopped":
        return tr("⏹ Pipeline stopped")

    if event == "plan_ready":
        n_tasks = data.get("n_tasks", 0)
        return tr("📋 Plan created ({n} tasks)", n=n_tasks)

    if event == "error":
        msg = data.get("message", str(data) if data else tr("Unknown error"))
        return tr("❌ Error: {error}", error=msg)

    # Unknown event — return nothing rather than producing noise
    return ""


# ── Explainer-Inputs ─────────────────────────────────────────────


def validate_explainer_inputs(
    topic: str,
    audience: str,
    length: str,
    purpose: str = "self_study",
) -> tuple[bool, str]:
    """Validate the four mandatory fields of the explainer use case.

    Wrapper around the USE_CASE_REGISTRY checker — offers the tuple
    argument interface that gradio_app.py expects.

    Returns:
        (ok, error_msg). With `ok=True` error_msg is empty; with
        `ok=False` it contains a combined error message listing the
        missing/invalid fields.
    """
    from src.pipeline.analysis_pipeline import USE_CASE_REGISTRY

    checker = USE_CASE_REGISTRY["explainer"]["preflight"]
    inputs = {
        "topic": topic,
        "audience": audience,
        "length": length,
        "purpose": purpose,
    }
    ok, errors = checker.validate(inputs)
    if ok:
        return True, ""
    return False, " · ".join(errors)


def build_explainer_preflight_data(
    topic: str,
    audience: str,
    length: str,
    purpose: str = "self_study",
) -> dict:
    """Build the preflight_data dict from the four mandatory fields.

    Passed to `ctx.preflight_data` and used by the decomposer as
    context. Values are stripped, empty optional fields are dropped.
    """
    data = {
        "topic": (topic or "").strip(),
        "audience": (audience or "").strip(),
        "length": (length or "").strip(),
    }
    purpose_clean = (purpose or "").strip()
    if purpose_clean:
        data["purpose"] = purpose_clean
    return data
