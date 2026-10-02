"""
Pipeline run tab — rendering layer.

Contract:
    render_pipeline_run(ctx) -> str   (Markdown)

UI-agnostic: takes a HarvestContext (or any object with the same
attributes) and returns Markdown. The Gradio wiring lives in
`gradio_app.py` (reactive `app_state.change()` handler).

Robustness: every section is encapsulated on its own. A missing or
broken sub-structure must never empty the whole tab — when in doubt the
section is skipped, not raised.
"""

from __future__ import annotations

import logging
from typing import Any

from src.ui.i18n import tr

logger = logging.getLogger(__name__)


# ─── small helpers ───────────────────────────────────────────────────


def _g(obj: Any, name: str, default: Any = None) -> Any:
    """getattr with a default — tolerates None and missing attributes."""
    if obj is None:
        return default
    return getattr(obj, name, default)


def _short(text: Any, n: int = 300) -> str:
    s = str(text or "").strip()
    return s if len(s) <= n else s[:n] + " …"


def _bool_icon(v: Any) -> str:
    return "✅" if v else "❌"


def _section(title: str, body: str) -> str:
    body = (body or "").strip()
    if not body:
        return ""
    return f"## {title}\n\n{body}\n"


# ─── Section renderers (each defensive) ──────────────────────────────


def _render_header(ctx: Any) -> str:
    status = _g(ctx, "status", "?")
    rounds = _g(ctx, "rounds_completed", 0)
    n_src = len(_g(ctx, "sources", []) or [])
    n_ext = len(_g(ctx, "extracts", []) or [])
    try:
        dur = f"{ctx.duration_seconds:.0f}s"
    except Exception:
        dur = "?"
    lines = [
        tr("**Status:** `{status}`  |  **Rounds:** {rounds}  |  "
           "**Sources:** {sources}  |  **Extracts:** {extracts}  |  "
           "**Duration:** {duration}", status=status, rounds=rounds,
           sources=n_src, extracts=n_ext, duration=dur),
    ]
    q = _short(_g(ctx, "query", ""), 400)
    if q:
        lines.append(f"\n> {q}")
    err = _g(ctx, "error_message", "")
    if err:
        lines.append(tr("\n⚠️ **Error:** {error}", error=_short(err, 500)))
    return _section(tr("🧠 Pipeline run"), "\n".join(lines))


def _render_output_schema(ctx: Any) -> str:
    schema = _g(ctx, "output_schema")
    if not schema:
        return ""
    rows = [
        tr("- **Title:** {title}", title=_g(schema, 'title', '—')),
        tr("- **Format:** `{format}`  |  **Language:** `{language}`",
           format=_g(schema, 'format_type', '—'),
           language=_g(schema, 'language', '—')),
    ]
    secs = _g(schema, "sections", []) or []
    if secs:
        names = []
        for s in secs:
            if isinstance(s, dict):
                names.append(str(s.get("title") or s.get("name") or s))
            else:
                names.append(str(s))
        rows.append(tr("- **Sections:** {names}", names=', '.join(names)))
    guidance = _short(_g(schema, "synthesis_guidance", ""), 400)
    if guidance:
        rows.append(tr("- **Synthesis guidance:** {guidance}", guidance=guidance))
    return _section(tr("📐 Output-Schema"), "\n".join(rows))


def _render_plan(ctx: Any) -> str:
    plan = _g(ctx, "research_plan")
    if not plan:
        return ""
    parts: list[str] = []
    summary = _short(_g(plan, "summary", ""), 600)
    if summary:
        parts.append(summary + "\n")

    questions = _g(plan, "questions", []) or []
    for q in questions:
        qid = _g(q, "id", "?")
        qtext = _g(q, "question", "")
        prio = _g(q, "priority", "—")
        scope = _g(q, "source_scope", "—")
        langs = ", ".join(_g(q, "search_langs", []) or [])
        answered = tr("answered") if _g(q, "answered", False) else tr("open")
        parts.append(
            f"**[{qid}]** {qtext}\n"
            + tr("  · Priority `{priority}` · Scope `{scope}` · "
                 "Languages `{langs}` · {answered}", priority=prio,
                 scope=scope, langs=langs or '—', answered=answered)
        )
        terms = _g(q, "search_terms", []) or []
        if terms:
            preview = ", ".join(f"`{t}`" for t in terms[:8])
            if len(terms) > 8:
                preview += f" (+{len(terms) - 8})"
            parts.append(tr("  · Search terms: {terms}", terms=preview))

    durls = _g(plan, "direct_urls", []) or []
    if durls:
        parts.append(tr("\n**Direct URLs ({n}):**", n=len(durls)))
        for d in durls[:15]:
            parts.append(f"  · {_g(d, 'url', '')} — {_short(_g(d, 'reason', ''), 120)}")

    repos = _g(plan, "git_repos", []) or []
    if repos:
        parts.append(tr("\n**Git-Repos ({n}):**", n=len(repos)))
        for r in repos[:15]:
            parts.append(
                f"  · {_g(r, 'owner', '')}/{_g(r, 'repo', '')} "
                f"({_g(r, 'platform', 'github')})"
            )

    zqs = _g(plan, "directory_queries", []) or []
    if zqs:
        parts.append(tr("\n**Person directory queries:** {n}", n=len(zqs)))

    fqs = _g(plan, "followup_queries", []) or []
    if fqs:
        parts.append(tr("\n**Follow-up queries (from the coverage assessment):** {n}",
                        n=len(fqs)))

    return _section(tr("🗺️ Research plan"), "\n".join(parts))


def _render_query_anchor(ctx: Any) -> str:
    anchor = _g(ctx, "query_anchor")
    if not anchor:
        return ""
    typ = _g(anchor, "type", "?")
    target = _g(anchor, "target", "")
    conf = _g(anchor, "confidence", 0.0)
    fb = _g(anchor, "fallback_used", False)
    reasoning = _short(_g(anchor, "reasoning", ""), 400)
    try:
        is_person = bool(anchor.is_person())
    except Exception:
        is_person = False
    body = (
        tr("- **Type:** `{type}`  |  **Target:** {target}  |  "
           "**Confidence:** {confidence}  |  Fallback: {fallback}\n"
           "- **Person hallucination filter active:** {person}",
           type=typ, target=target or '—', confidence=f"{conf:.2f}",
           fallback=_bool_icon(fb), person=_bool_icon(is_person))
    )
    if reasoning:
        body += tr("\n- **Reasoning:** {reasoning}", reasoning=reasoning)
    return _section(tr("⚓ Query anchor"), body)


def _render_search_log(ctx: Any) -> str:
    stats = _g(ctx, "search_stats", {}) or {}
    if not stats:
        return ""
    rows = []
    langs = stats.get("languages") or []
    if langs:
        rows.append(tr("- **Search languages:** {langs}", langs=', '.join(langs)))
    tbl = stats.get("terms_by_lang") or {}
    for lang, terms in tbl.items():
        terms = terms or []
        preview = ", ".join(f"`{t}`" for t in terms[:6])
        if len(terms) > 6:
            preview += f" (+{len(terms) - 6})"
        rows.append(f"  · **{lang}:** {preview}")
    for k, v in stats.items():
        if k in ("languages", "terms_by_lang"):
            continue
        rows.append(f"- **{k}:** {_short(v, 200)}")
    return _section(tr("🔎 Research history"), "\n".join(rows))


def _render_filter_stats(ctx: Any) -> str:
    rounds = _g(ctx, "filter_stats_per_round", []) or []
    if not rounds:
        return ""
    parts = []
    for i, rnd in enumerate(rounds, 1):
        if not isinstance(rnd, dict) or not rnd:
            continue
        parts.append(tr("**Round {n}:**", n=i))
        for fname, fstats in rnd.items():
            if isinstance(fstats, dict):
                act = fstats.get("activated")
                rej = fstats.get("rejected", 0)
                kept = fstats.get("kept", fstats.get("passed", "—"))
                parts.append(
                    f"  · `{fname}` — "
                    + tr("active {active}, discarded {rejected}, kept {kept}",
                         active=_bool_icon(act), rejected=rej, kept=kept)
                )
            else:
                parts.append(f"  · `{fname}`: {_short(fstats, 120)}")
    return _section(tr("🧪 Filter statistics"), "\n".join(parts))


def _render_coverage(ctx: Any) -> str:
    per_round = _g(ctx, "coverage_per_round", []) or []
    if not per_round:
        return ""
    parts = []
    for i, rnd in enumerate(per_round, 1):
        if not isinstance(rnd, dict) or not rnd:
            continue
        parts.append(tr("**Round {n}:**", n=i))
        for qid, cov in rnd.items():
            if isinstance(cov, dict):
                c = cov.get("coverage", "?")
                conf = cov.get("confidence", 0.0)
                miss = cov.get("missing_aspects") or []
                line = f"  · **{qid}:** `{c}` " + tr("(conf. {conf})", conf=f"{conf:.2f}")
                if miss:
                    line += tr(" — missing: {aspects}",
                               aspects=', '.join(str(m) for m in miss[:4]))
                parts.append(line)
            else:
                parts.append(f"  · **{qid}:** {_short(cov, 120)}")
    return _section(tr("📊 Coverage per question"), "\n".join(parts))


def _render_map_answers(ctx: Any) -> str:
    answers = _g(ctx, "map_answers", []) or []
    if not answers:
        return ""
    parts = []
    for a in answers:
        if not isinstance(a, dict):
            continue
        qid = a.get("question_id", "?")
        q = a.get("question", "")
        n = a.get("n_extracts", 0)
        ans = _short(a.get("answer", ""), 800)
        parts.append(f"**[{qid}]** {q}  " + tr("_(from {n} extracts)_", n=n)
                     + f"\n\n{ans}\n")
    return _section(tr("🧩 Synthesis map (question answers)"), "\n".join(parts))


def _render_diagnosis(ctx: Any) -> str:
    diag = _g(ctx, "final_diagnosis")
    if not diag:
        return ""
    if isinstance(diag, dict):
        body = "\n".join(
            f"- **{k}:** {_short(v, 300)}" for k, v in diag.items()
        )
    else:
        body = _short(diag, 600)
    return _section(tr("🩺 Diagnose"), body)


def _render_quality_fulfillment(ctx: Any) -> str:
    rq = _g(ctx, "report_quality")
    qf = _g(ctx, "query_fulfillment")
    if not rq and not qf:
        return ""
    parts = []
    if isinstance(rq, dict):
        parts.append(
            tr("**Report quality:** passed {passed}", passed=_bool_icon(rq.get('passed')))
            + f" · {_short(rq.get('rating', ''), 300)}"
        )
        issues = rq.get("issues") or []
        for m in issues[:10]:
            if isinstance(m, dict):
                parts.append(
                    f"  · {_short(m.get('description') or m.get('typ'), 200)}"
                )
            else:
                parts.append(f"  · {_short(m, 200)}")
        if rq.get("fallback_used"):
            parts.append(tr("  · _(fallback assessment used)_"))
    if isinstance(qf, dict):
        parts.append(
            tr("\n**Request fulfilment:** fulfilled {fulfilled}",
               fulfilled=_bool_icon(qf.get('fulfilled')))
            + f" · {_short(qf.get('assessment', ''), 300)}"
        )
        if qf.get("rework"):
            parts.append(tr("  · Rework: {rework}", rework=_short(qf.get('rework'), 300)))
        if qf.get("fallback_used"):
            parts.append(tr("  · _(fallback assessment used)_"))
    return _section(tr("✅ Quality & fulfilment"), "\n".join(parts))


def _render_report_revision(ctx: Any) -> str:
    rev = _g(ctx, "report_revision")
    if not rev or not isinstance(rev, dict):
        return ""
    if not rev.get("applied"):
        reason = _short(rev.get("reason", ""), 300)
        return _section(
            tr("✏️ Report revision"),
            tr("Not applied — {reason}",
               reason=reason or tr("no high-confidence contradictions")),
        )
    n = rev.get("n_corrected", 0)
    facts = rev.get("factoids_corrected") or []
    body = [
        tr("Applied — **{n}** statement(s) corrected "
           "(length {before} → {after} characters)", n=n,
           before=rev.get('original_length', '?'),
           after=rev.get('corrected_length', '?')),
    ]
    for f in facts[:10]:
        body.append(f"  · {_short(f, 200)}")
    return _section(tr("✏️ Report revision"), "\n".join(body))


def _render_factoids(ctx: Any) -> str:
    facts = _g(ctx, "factoid_verifications", []) or []
    if not facts:
        return ""
    parts = []
    for f in facts[:40]:
        if not isinstance(f, dict):
            continue
        v = f.get("verified", "?")
        conf = f.get("confidence", 0.0)
        icon = {
            "supported": "✅", "contradicted": "❌",
            "unverifiable": "❓", "unsupported": "⚠️",
        }.get(str(v), "·")
        parts.append(
            f"{icon} `{v}` " + tr("(conf. {conf})", conf=f"{conf:.2f}") + " — "
            f"{_short(f.get('factoid', ''), 220)}"
        )
    return _section(tr("🔬 Factoid verification"), "\n".join(parts))


def _render_classifier_calls(ctx: Any) -> str:
    calls = _g(ctx, "classifier_calls", []) or []
    if not calls:
        return ""
    parts = []
    for c in calls:
        # Accepts ClassifierCall objects or plain dicts
        if hasattr(c, "to_dict"):
            try:
                c = c.to_dict()
            except Exception:
                pass
        if isinstance(c, dict):
            name = c.get("name", "?")
            conf = c.get("confidence", 0.0)
            fb = c.get("fallback_used", False)
            dur = c.get("duration_seconds", 0.0)
            line = (
                f"- `{name}` — " + tr("conf. {conf}", conf=f"{conf:.2f}") + f", {dur:.2f}s"
                f"{', Fallback ⚠️' if fb else ''}"
            )
            if fb and c.get("fallback_reason"):
                line += f" ({_short(c.get('fallback_reason'), 120)})"
            parts.append(line)
        else:
            parts.append(f"- {_short(c, 160)}")
    head = tr("_{n} classifier call(s)_", n=len(calls)) + "\n\n"
    return _section(tr("🧮 Classifier calls"), head + "\n".join(parts))


# ─── public entry point ───────────────────────────────────────────


_RENDERERS = (
    _render_header,
    _render_output_schema,
    _render_plan,
    _render_query_anchor,
    _render_search_log,
    _render_coverage,
    _render_filter_stats,
    _render_map_answers,
    _render_diagnosis,
    _render_quality_fulfillment,
    _render_report_revision,
    _render_factoids,
    _render_classifier_calls,
)


def render_pipeline_run(ctx: Any) -> str:
    """Render a HarvestContext as pipeline-run Markdown.

    Defensive: every section is isolated. A section that raises is
    skipped (with a log warning), the rest stays visible.
    """
    if ctx is None:
        return tr("*No research started yet.*")

    blocks: list[str] = []
    for fn in _RENDERERS:
        try:
            out = fn(ctx)
            if out:
                blocks.append(out)
        except Exception as e:  # a section must never empty the tab
            logger.warning(
                "Pipeline run: section %s failed: %s",
                getattr(fn, "__name__", fn), e,
            )

    if not blocks:
        return tr("*Research running — no intermediate structures yet.*")
    return "\n---\n\n".join(blocks)
