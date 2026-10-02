"""
Markdown formatting of the plan for the plan preview in the chat.

The plan is shown to the user as readable Markdown — not as a
structured editor. To change the plan, the user refines the request in
the chat and triggers a new plan. That is more reliable than a JSON
editor, which the average user cannot operate.

This function is Gradio-agnostic and unit-testable. Wiring it to
Gradio components is UI code in `gradio_app.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.ui.i18n import tr

if TYPE_CHECKING:
    from src.pipeline.models import ResearchPlan, ResearchQuestion


# ───────────────────────────────────────────────────────────────────
# Markdown format (readable, not round-trippable)
# ───────────────────────────────────────────────────────────────────


def format_plan_markdown(plan: "ResearchPlan") -> str:
    """Format a plan as readable Markdown.

    Goal: easy to read in the chat history. To change the plan, the
    user refines the request in the chat, which produces a new plan.
    """
    if plan is None:
        return tr("_(no plan)_")

    lines: list[str] = []
    lines.append(tr("## Research plan"))
    lines.append("")

    summary = (plan.summary or "").strip()
    if summary:
        lines.append(tr("**Summary:** {summary}", summary=summary))
        lines.append("")

    if plan.questions:
        lines.append(tr("### Questions ({n})", n=len(plan.questions)))
        lines.append("")
        for q in plan.questions:
            lines.extend(_format_question_md(q))
            lines.append("")

    if plan.direct_urls:
        lines.append(tr("### Direct URLs"))
        for du in plan.direct_urls:
            url = getattr(du, "url", "")
            reason = getattr(du, "reason", "") or ""
            if reason:
                lines.append(f"- {url} _({reason})_")
            else:
                lines.append(f"- {url}")
        lines.append("")

    if plan.git_repos:
        lines.append(tr("### Git-Repos"))
        for gr in plan.git_repos:
            owner = getattr(gr, "owner", "") or ""
            repo = getattr(gr, "repo", "") or ""
            platform = getattr(gr, "platform", "github") or "github"
            label = f"{platform}:{owner}/{repo}" if owner and repo else tr("(unnamed)")
            lines.append(f"- {label}")
        lines.append("")

    if plan.directory_queries:
        lines.append(tr("### Person directory queries ({n})", n=len(plan.directory_queries)))
        for zq in plan.directory_queries:
            term = getattr(zq, "query", "") or ""
            reason = getattr(zq, "reason", "") or ""
            if reason:
                lines.append(f"- {term} _({reason})_")
            else:
                lines.append(f"- {term}")
        lines.append("")

    return "\n".join(lines).strip()


def _format_question_md(q: "ResearchQuestion") -> list[str]:
    """One question as a Markdown block."""
    qid = q.id or "?"
    priority = q.priority or "—"
    text = q.question or ""
    lines = [f"**[{qid}]** ({priority}) {text}"]

    terms = []
    if q.search_terms_by_lang:
        for lang, lst in sorted(q.search_terms_by_lang.items()):
            terms.append(f"  - `{lang}`: {', '.join(lst)}")
    if not terms and q.search_terms:
        terms.append("  - " + ", ".join(q.search_terms))
    if terms:
        lines.append(tr("  Search terms:"))
        lines.extend(terms)

    if q.source_scope:
        lines.append(tr("  Source scope: {scope}", scope=q.source_scope))

    return lines

