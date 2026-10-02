"""
Research orchestrator — controls the autonomous research pipeline.
"""

import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Awaitable, Callable, Optional

from src.output_language import llm_language_name, prompt_language_line, set_current
from src.pipeline.followup import build_followup_queries
from src.about import TOOL_NAME
from src.output_language import t as catalog_t
from src.config import PipelineConfig
from src.ui.i18n import tr
from src.institution import get_profile, polite_user_agent
from src.connectors.base import ConnectorRegistry, normalize_url, is_url_blocked
from src.llm.client import (
    DualLLMClient, EmptyLLMResponseError, _PLACEHOLDER_RE,
)
from src.pipeline.models import (
    HarvestContext, ResearchPlan, ResearchQuestion, DirectURL,
    DirectoryQuery, OutputSchema, SourceDocument, HarvestResult, SearchResult,
    SourceType, OUTPUT_TEMPLATES, DEFAULT_TEMPLATE,
)
from src.prompts import (
    FORMAT_AGENT_PROMPT, ANALYSIS_PROMPT, HARVEST_PROMPT,
    SYNTHESIS_PROMPT, QUESTION_ANSWER_PROMPT,
    SUMMARY_PROMPT, get_date_text,
)

logger = logging.getLogger(__name__)

# Type of the progress callback
# (async in practice; the orchestrator calls it with `await`)
ProgressCallback = Callable[[str, object], Awaitable[None]]




class ResearchOrchestrator:
    """Run a complete research autonomously.

    IMPORTANT: create a new instance per research run! The orchestrator
    holds per-run state (_seen_urls, _stop_requested) that must not be
    shared between sessions.
    """

    def __init__(
        self,
        llm: DualLLMClient,
        connectors: ConnectorRegistry,
        config: PipelineConfig,
        person_directory=None,
        output_language: str = "en",
    ):
        self.llm = llm
        self.connectors = connectors
        self.config = config
        self.person_directory = person_directory
        from src.output_language import normalize as _norm_lang
        self.output_language = _norm_lang(output_language)
        self._stop_requested = False
        # ── StopSignal ──
        # A real signal object alongside the bool flag. `stop()` sets both
        # synchronously; the StopSignal is passed on to sub-pipelines
        # (AnalysisPipelineRunner), so that stop clicks take effect
        # everywhere (including deeply nested tasks).
        from src.core.stop_signal import StopSignal
        self._stop_signal = StopSignal()
        self._mode = "web"
        self._institution_only = False
        self._academic_only = False
        # Domains from user URLs — filled dynamically per run.
        # Links to these domains are followed preferentially during link
        # following, even if they are not in the static
        # _FOLLOWABLE_DOMAINS allow list.
        self._user_url_domains: set[str] = set()
        # Per-run URL de-duplication (NOT the global ConnectorRegistry._seen_urls)
        self._seen_urls: set[str] = set()
        # ── Classifier layer ──
        # Set once per run; made visible by the UI via `progress_log` /
        # `final_report`.
        self._query_anchor = None              # QueryAnchor | None
        self._search_scope = None              # SearchScope | None
        self._classifier_calls: list = []      # list[ClassifierCall]
        # High-water mark: how many entries of _classifier_calls have already
        # been mirrored into ctx.classifier_calls (DAG visibility for the
        # classifiers run inside the orchestrator: search_scope, off-topic
        # filter). The delta is mirrored in SearchAndFetchNode.
        self._cc_mirrored: int = 0
        self._filter_stats_per_round: list = []  # one dict per round

    def stop(self):
        """Stop the running research (this instance only).

        Sets two stop signals together:
          1. the bool flag `_stop_requested` (for the inline checks in the
             pipeline)
          2. the `StopSignal` object (for the AnalysisPipelineRunner and
             other sub-pipelines)

        Both are set at the same time — no path is left uninformed. Both are
        reset at the start of every run.
        """
        self._stop_requested = True
        self._stop_signal.request_stop("user_request")
        # NOT self.llm.stop() — that would stop ALL running sessions

    # ── Per-Run URL-Dedup ──

    def _is_url_seen(self, url: str) -> bool:
        return normalize_url(url) in self._seen_urls

    def _mark_url_seen(self, url: str):
        self._seen_urls.add(normalize_url(url))

    # ═══════════════════════════════════════════════════════════════
    # DAG-based execution
    # ═══════════════════════════════════════════════════════════════

    async def run_via_dag_plan_only(
        self,
        query: str,
        chat_history: list[dict],
        context_docs: str,
        template_name: str,
        progress_callback: ProgressCallback,
        mode: str = "web",
        use_directory: bool = False,
        institution_only: bool = False,
        academic_only: bool = False,
    ) -> "HarvestContext":
        """Plan phase only, via the DAG.

        Runs FormatAgentNode + AnalysisNode and ends with
        `ctx.status = "plan_ready"`. The UI handler can then show the plan,
        let the user confirm it, and continue the research with
        `existing_ctx + skip_decompose=True`.
        """
        set_current(self.output_language)
        from datetime import datetime

        from src.pipeline.dag import PipelineDAG
        from src.pipeline.dag_nodes import (
            AnalysisNode,
            FormatAgentNode,
        )

        # Reset the per-run state (as in run_via_dag)
        self._stop_requested = False
        self._stop_signal.reset()
        self._mode = mode
        self._institution_only = institution_only and (mode == "institution")
        self._academic_only = academic_only
        self._use_directory = use_directory or (mode == "institution")
        self._user_url_domains = set()
        self._seen_urls = set()
        self._query_anchor = None
        self._search_scope = None
        self._classifier_calls = []
        self._cc_mirrored = 0
        self._filter_stats_per_round = []

        ctx = HarvestContext(query=query, chat_history=chat_history,

                             output_language=self.output_language)
        ctx.status = "running"
        ctx.started_at = datetime.now().isoformat()

        try:
            await PipelineDAG([
                FormatAgentNode(
                    orchestrator=self,
                    context_docs=context_docs,
                    template_name=template_name,
                ),
                AnalysisNode(
                    orchestrator=self,
                    context_docs=context_docs,
                ),
            ]).run(
                ctx, stop_signal=self._stop_signal,
                progress_callback=progress_callback,
            )

            # Success: the plan exists, set the status for the UI handler
            ctx.status = "plan_ready"
            ctx.finished_at = datetime.now().isoformat()
            await progress_callback("plan_ready", ctx)
            return ctx

        except asyncio.CancelledError:
            ctx.status = "cancelled"
            ctx.finished_at = datetime.now().isoformat()
            await progress_callback("cancelled", ctx)
            return ctx

        except Exception as e:
            logger.error(
                "DAG plan-only path failed: %s", e, exc_info=True,
            )
            ctx.status = "error"
            ctx.error_message = str(e)
            ctx.finished_at = datetime.now().isoformat()
            try:
                await progress_callback("error", str(e))
            except Exception:
                pass
            return ctx

    async def run_via_dag(
        self,
        query: str,
        chat_history: list[dict],
        context_docs: str,
        template_name: str,
        progress_callback: ProgressCallback,
        mode: str = "web",
        use_directory: bool = False,
        institution_only: bool = False,
        academic_only: bool = False,
        existing_ctx: Optional["HarvestContext"] = None,
    ) -> "HarvestContext":
        """Pipeline execution via the DAG engine.

        This is the execution path for all standard research runs; `run()`
        delegates here.

        Resume mode: if `existing_ctx` is set, that ctx is reused.
        FormatAgentNode and AnalysisNode skip themselves (`applies_to=False`)
        if `output_schema` or `research_plan` are already set. The pipeline
        thus effectively starts at the research loop.

        Deliberately narrower than `run()`:
          - no plan preview gate (`plan_only`) — separate method
            `run_via_dag_plan_only`
          - no analysis use cases — separate path in `run()`

        Returns:
            HarvestContext with ctx.status in {'done', 'error', 'cancelled'}.
        """
        set_current(self.output_language)
        from datetime import datetime

        from src.pipeline.dag import PipelineDAG
        from src.pipeline.dag_nodes import (
            AnalysisNode,
            ContinueDecisionNode,
            ContradictionCheckNode,
            CoverageNode,
            DiagnosisBannerNode,
            DiagnosisNode,
            FactoidVerificationNode,
            FormatAgentNode,
            HarvestNode,
            QueryAnchorNode,
            QueryFulfillmentNode,
            ReportQualityNode,
            ReportRevisionNode,
            ReportWarningBannerNode,
            SearchAndFetchNode,
            SynthesisNode,
        )

        # Reset the per-run state
        self._stop_requested = False
        self._stop_signal.reset()
        self._mode = mode
        self._institution_only = institution_only and (mode == "institution")
        self._academic_only = academic_only
        self._use_directory = use_directory or (mode == "institution")
        self._user_url_domains = set()
        self._seen_urls = set()
        self._query_anchor = None
        self._search_scope = None
        self._classifier_calls = []
        self._cc_mirrored = 0
        self._filter_stats_per_round = []

        # Resume mode: reuse existing_ctx, otherwise build a new ctx.
        # FormatAgentNode/AnalysisNode skip themselves automatically if ctx
        # is already filled (applies_to hook).
        if existing_ctx is not None:
            ctx = existing_ctx
            logger.info(
                "DAG path resume: %d questions from the existing plan",
                len(ctx.research_plan.questions) if ctx.research_plan else 0,
            )
        else:
            ctx = HarvestContext(query=query, chat_history=chat_history,
                                 output_language=self.output_language)
        ctx.status = "running"
        ctx.started_at = datetime.now().isoformat()

        try:
            # ─── Step 1: plan before the loop ───
            # Format agent + analysis once beforehand — they produce the schema
            # and the plan without which the research loop cannot run.
            await PipelineDAG([
                FormatAgentNode(
                    orchestrator=self,
                    context_docs=context_docs,
                    template_name=template_name,
                ),
                AnalysisNode(
                    orchestrator=self, context_docs=context_docs,
                ),
            ]).run(ctx, stop_signal=self._stop_signal,
                   progress_callback=progress_callback)

            # ─── Step 2: research loop ───
            # The loop itself is not in the DAG — per iteration we build a
            # sub-pipeline from the iterative nodes. The pragmatic solution: no
            # "while nodes" in the engine, but a narrow loop logic here.
            for round_num in range(self.config.max_rounds):
                ctx.rounds_completed = round_num + 1

                # Remember the sources before the search — everything added
                # afterwards counts as "new in this round"
                sources_before = list(ctx.sources)
                extracts_before_count = len(ctx.extracts)

                # (a) Search+fetch (including the off-topic filter over the
                #     complete set of new sources of this round — the filter
                #     sits INSIDE _run_search_and_fetch, exactly once, new sources
                #     only) + query anchor.
                #     OffTopicFilterNode is NOT run here: it would re-judge the
                #     whole accumulated ctx.sources corpus every round
                #     (non-determinism + double LLM cost).
                pre_harvest = PipelineDAG([
                    SearchAndFetchNode(
                        orchestrator=self, round_num=round_num,
                        progress_callback=progress_callback,
                    ),
                    QueryAnchorNode(llm=self.llm),
                ])
                await pre_harvest.run(
                    ctx, stop_signal=self._stop_signal,
                    progress_callback=progress_callback,
                )

                # (b) Identify the fresh sources
                seen_urls_before = {s.url for s in sources_before}
                new_sources = [
                    s for s in ctx.sources if s.url not in seen_urls_before
                ]

                if not new_sources:
                    logger.info(
                        "Round %d: no new sources — ending the loop",
                        round_num + 1,
                    )
                    break

                # (c) Harvest + Coverage + Continue
                post_harvest = PipelineDAG([
                    HarvestNode(
                        orchestrator=self,
                        new_sources=new_sources,
                        progress_callback=progress_callback,
                    ),
                    CoverageNode(
                        llm=self.llm, round_number=round_num + 1
                    ),
                    ContinueDecisionNode(
                        llm=self.llm,
                        round_number=round_num + 1,
                        max_rounds=self.config.max_rounds,
                        new_extracts=len(ctx.extracts) - extracts_before_count,
                        new_sources=len(new_sources)
                    ),
                ])
                await post_harvest.run(
                    ctx, stop_signal=self._stop_signal,
                    progress_callback=progress_callback,
                )

                # If the continue node signalled STOP_PIPELINE, the sub-pipeline
                # has already stopped. We check `ctx.continue_decisions` for
                # whether the last decision was a non-continue.
                decisions = ctx.continue_decisions
                if decisions and decisions[-1]["decision"] != "continue":
                    logger.info(
                        "Round %d: the classifier ends the research (%s)",
                        round_num + 1, decisions[-1]["decision"],
                    )
                    break

                # Next round: search for what the coverage assessment left
                # open. Each round replaces the follow-ups of the previous one.
                coverage = (ctx.coverage_per_round[-1]["results"]
                            if getattr(ctx, "coverage_per_round", None) else {})
                ctx.research_plan.followup_queries = build_followup_queries(
                    ctx.research_plan, coverage,
                )
                if not ctx.research_plan.followup_queries:
                    logger.info(
                        "Round %d: nothing left to search for — ending the loop",
                        round_num + 1,
                    )
                    break
                logger.info(
                    "Round %d: %d follow-up queries for the next round",
                    round_num + 1, len(ctx.research_plan.followup_queries),
                )

            # ─── Step 3: post-research nodes ───
            await PipelineDAG([
                ContradictionCheckNode(orchestrator=self, min_extracts=4),
                DiagnosisNode(
                    llm=self.llm, max_rounds=self.config.max_rounds,
                    use_case=mode,
                ),
                SynthesisNode(
                    orchestrator=self,
                    context_docs=context_docs,
                    progress_callback=progress_callback,
                ),
                DiagnosisBannerNode(),
                FactoidVerificationNode(llm=self.llm),
                # The revision node corrects the report if the factoid verifier
                # found contradicted statements with high confidence (≥0.9).
                # With lower confidence or no contradictions at all, the node
                # skips itself.
                ReportRevisionNode(llm=self.llm),
                # Report quality and request fulfilment — both run AFTER the
                # synthesis and complement the factoid verifier: macro instead
                # of atomic, original query instead of plan questions.
                ReportQualityNode(llm=self.llm),
                QueryFulfillmentNode(llm=self.llm),
                # Must run LAST: writes the findings of the two check nodes into
                # the report, so that a report with complaints is not exported
                # without comment.
                ReportWarningBannerNode(),
            ]).run(ctx, stop_signal=self._stop_signal,
                   progress_callback=progress_callback)

            ctx.status = "done"
            ctx.finished_at = datetime.now().isoformat()
            ctx.llm_usage = self.llm.get_usage_stats()

            try:
                ctx.save(self.config.data_dir)
            except Exception as e:
                logger.warning("DAG path: persistence failed: %s", e)

            await progress_callback("done", ctx)
            return ctx

        except asyncio.CancelledError:
            ctx.status = "cancelled"
            ctx.finished_at = datetime.now().isoformat()
            await progress_callback("cancelled", ctx)
            return ctx

        except Exception as e:
            logger.error("DAG path: pipeline error: %s", e, exc_info=True)
            ctx.status = "error"
            ctx.error_message = str(e)
            ctx.finished_at = datetime.now().isoformat()
            try:
                await progress_callback("error", str(e))
            except Exception:
                pass
            return ctx

    async def run(
        self,
        query: str,
        chat_history: list[dict],
        context_docs: str,
        template_name: str,
        progress_callback: ProgressCallback,
        use_directory: bool = False,
        mode: str = "web",
        institution_only: bool = False,
        academic_only: bool = False,
        preflight_data: Optional[dict] = None,
        plan_only: bool = False,
        existing_ctx: Optional["HarvestContext"] = None,
        skip_decompose: bool = False,
    ) -> HarvestContext:
        """Main method: run the complete research pipeline.

        Args:
            mode: "web" (normal), "institution" (institution research with
                  person directory + site filter), or one of the analysis
                  modes ("explainer", "peer_review", "decision_analysis",
                  "grant_proposal", "literature_review", "research_design",
                  "literature_finder")
            institution_only: only the institution's own sources (website
                              index + person directory + site: filter)
            academic_only: only academic SearXNG engines (Wikipedia,
                           Wikidata, Google Scholar, Semantic Scholar,
                           ArXiv, PubMed). Uses the engines parameter
                           instead of categories="general". Can be combined
                           with institution_only.
            preflight_data: analysis modes only — validated mandatory inputs
            plan_only: for analysis modes: only run the decomposer phase and
                       return the plan. For the plan preview gate.
            existing_ctx: if set, this context is reused instead of building
                          a new one. Only useful together with
                          skip_decompose=True after a previous plan_only call.
            skip_decompose: for analysis modes: skips the decomposer phase
                            and uses ctx.task_plan directly.
        """
        set_current(self.output_language)

        self._stop_requested = False
        self._stop_signal.reset()
        self._mode = mode
        self._institution_only = institution_only and (mode == "institution")
        self._academic_only = academic_only
        self._use_directory = use_directory or (mode == "institution")

        if existing_ctx is not None:
            # Reuse the existing context — its query, chat_history and other
            # fields are already set
            ctx = existing_ctx
        else:
            ctx = HarvestContext(query=query, chat_history=chat_history,
                                 output_language=self.output_language)
        ctx.status = "running"

        # ═══ Branch: analysis pipeline for the analysis modes ══════════════
        # Analysis use cases use the `AnalysisPipelineRunner`, which builds
        # and runs a DAG with one `AnalysisLayerNode` per layer. The
        # decomposer is implemented for all analysis use cases (see
        # analysis_pipeline.py).
        from src.pipeline.analysis_pipeline import (
            AnalysisPipelineRunner,
            is_analysis_use_case,
        )

        if is_analysis_use_case(mode):
            ctx.use_case = mode
            if preflight_data is not None:
                ctx.preflight_data = preflight_data
            # The decomposer finds the LLM client via this attribute
            ctx._llm_client = self.llm

            runner = AnalysisPipelineRunner(
                llm_client=self.llm,
                progress_callback=progress_callback,
                max_parallel=5,
                rate_limit=3.0,
                # The stop check reads BOTH signals (bool and StopSignal), so
                # that neither stop mechanism is left out.
                stop_check=lambda: (
                    self._stop_requested or self._stop_signal.is_stop_requested()
                ),
            )

            if plan_only:
                # Plan preview gate, phase 1: only decompose, return the plan.
                # No finished_at, no execution.
                await progress_callback(
                    "status",
                    tr("📋 Building the execution plan: {mode}", mode=mode),
                )
                await runner.decompose_only(ctx)
                return ctx

            # Normal execution or resume after the gate
            await progress_callback(
                "status",
                tr("🔬 Starting the analysis pipeline: {mode}", mode=mode),
            )
            await runner.run(ctx, skip_decompose=skip_decompose)
            ctx.finished_at = datetime.now().isoformat()
            return ctx

        # Fall-through: classic research pipeline

        # ═══ Research modes via the DAG path ═══
        # Standard research → run_via_dag.
        # Plan preview gate → run_via_dag_plan_only.
        # Resume mode (existing_ctx + skip_decompose) → run_via_dag;
        #   the applies_to hooks of FormatAgentNode/AnalysisNode skip the
        #   plan phase automatically.
        if plan_only:
            try:
                return await self.run_via_dag_plan_only(
                    query=query,
                    chat_history=chat_history,
                    context_docs=context_docs,
                    template_name=template_name,
                    progress_callback=progress_callback,
                    mode=mode,
                    use_directory=use_directory,
                    institution_only=institution_only,
                    academic_only=academic_only,
                )
            except Exception as e:
                logger.error(
                    "DAG plan-only path failed: %s", e, exc_info=True,
                )
                raise

        # Standard and resume both go through run_via_dag
        try:
            return await self.run_via_dag(
                query=query,
                chat_history=chat_history,
                context_docs=context_docs,
                template_name=template_name,
                progress_callback=progress_callback,
                mode=mode,
                use_directory=use_directory,
                institution_only=institution_only,
                academic_only=academic_only,
                existing_ctx=existing_ctx if skip_decompose else None,
            )
        except Exception as e:
            logger.error(
                "DAG path failed — no fallback active: %s",
                e, exc_info=True,
            )
            raise

    async def _run_format_agent(
        self, query: str, chat_history: list[dict],
        context_docs: str, template_name: str,
    ) -> OutputSchema:
        """Create the output schema via the format agent (primary LLM)."""
        template = OUTPUT_TEMPLATES.get(template_name, OUTPUT_TEMPLATES[DEFAULT_TEMPLATE])

        # If a fixed template was chosen → take it directly
        if template["format"] != "free" and template["per_section"]:
            # still ask the LLM for sections based on the request
            pass

        # Format the chat history
        history_text = ""
        for msg in (chat_history or [])[-10:]:
            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")
            if isinstance(content, str):
                history_text += f"{role}: {content[:500]}\n"

        context_summary = context_docs[:2000] if context_docs else "None"

        prompt = FORMAT_AGENT_PROMPT.format(
            query=query,
            chat_history=history_text or "No previous chat",
            template_name=template_name,
            template_description=template.get("description", ""),
            context_summary=context_summary,
        )

        try:
            result = await self.llm.primary_complete_json(
                [{"role": "user", "content": prompt}],
                max_tokens=4096,
            )

            # JSON → OutputSchema via the central conversion
            schema = OutputSchema.from_dict(result, fallback_title=query[:80])

            # Synthesis guidance: first from the template chosen by the user,
            # then fall back to a template matching the format agent's format,
            # finally whatever the LLM output contained. So the standard
            # template is only overridden if the template itself has none.
            guidance = template.get("synthesis_guidance", "")
            if not guidance:
                fmt = schema.format_type
                for t in OUTPUT_TEMPLATES.values():
                    if t.get("format") == fmt and t.get("synthesis_guidance"):
                        guidance = t["synthesis_guidance"]
                        break
            if guidance:
                schema.synthesis_guidance = guidance

            # per_section_fields: if the LLM delivered nothing, use the
            # template default
            if not schema.per_section_fields:
                schema.per_section_fields = template.get("per_section", [])

            return schema
        except Exception as e:
            logger.warning(f"Format agent failed: {e}, using the default")
            # Default guidance
            default_guidance = OUTPUT_TEMPLATES.get(
                DEFAULT_TEMPLATE, {}
            ).get("synthesis_guidance", "")
            return OutputSchema(
                title=query[:80], synthesis_guidance=default_guidance,
            )

    # ═══════════════════════════════════════════════════════════════
    # Phase 1: analysis
    # ═══════════════════════════════════════════════════════════════

    async def _run_analysis(
        self, query: str, context_docs: str, output_schema: OutputSchema,
    ) -> ResearchPlan:
        """Create the research plan (primary LLM)."""
        schema_text = json.dumps({
            "title": output_schema.title,
            "format": output_schema.format_type,
            "sections": output_schema.sections,
        }, ensure_ascii=False, indent=2)

        # Available connectors for the prompt
        connectors_info = []
        elastic = self.connectors.get_connector_by_name("elasticsearch")
        if elastic:
            site = getattr(elastic.config, "site_base_url", "")
            connectors_info.append(
                f"- source_scope='elastic': searches the internal website index"
                f"{' (' + site + ')' if site else ''}. "
                f". Use it for questions about content of our own website."
            )
        if self._use_directory and self.person_directory and self.person_directory.enabled:
            prof = get_profile()
            connectors_info.append(
                f"- Person directory ({prof.directory_name or 'person directory'}"
                f"{' of ' + prof.name if prof.name else ''}) — IMPORTANT, ALWAYS USE:\n"
                "  You MUST generate directory_queries when the request concerns people,\n"
                "  units, expertise or responsibilities, or contains a\n"
                "  person's name.\n"
                "  Format: {\"query\": \"search term\", \"reason\": \"why\"}\n"
                "  Generate SEVERAL directory_queries with different search terms\n"
                "  (name, role, unit) for a better hit rate."
                + (f"\n{prof.directory_planner_hint}" if prof.directory_planner_hint else "")
            )
        available_connectors = "\n".join(connectors_info)

        # Institution mode: use the dedicated prompt
        if self._mode == "institution":
            from src.prompts import INSTITUTION_ANALYSIS_PROMPT
            prof = get_profile()
            prompt = INSTITUTION_ANALYSIS_PROMPT.format(
                institution_name=prof.name,
                institution_domain=prof.primary_domain,
                site_filter=prof.site_filter(),
                institution_guidelines=prof.planner_guidelines,
                query=query,
                output_schema=schema_text,
                context_docs=context_docs[:15000] if context_docs else "None",
                date=get_date_text(),
                available_connectors=available_connectors,
            )
            prompt += prompt_language_line(self.output_language)
        else:
            prompt = ANALYSIS_PROMPT.format(
                query=query,
                output_schema=schema_text,
                context_docs=context_docs[:15000] if context_docs else "None",
                date=get_date_text(),
                available_connectors=available_connectors,
            )
            prompt += prompt_language_line(self.output_language)

        result = await self.llm.primary_complete_json(
            [{"role": "user", "content": prompt}],
            max_tokens=8192,
        )

        # JSON → plan via the central conversion. ResearchPlan.from_dict is
        # defensive: broken entries are discarded, missing fields get
        # defaults. Here we only filter security-relevant parts (URL
        # blocking) and add user-specific safety nets.
        plan = ResearchPlan.from_dict(result)

        # ── Apply URL blocking to direct_urls ──
        # The LLM may propose URLs we do not want to fetch
        # (e.g. internal IPs through hallucination, blocked domains).
        plan.direct_urls = [
            du for du in plan.direct_urls
            if du.url and not is_url_blocked(du.url)
        ]

        # ── Safety net: extract URLs from the user query ──
        # The analysis LLM may overlook URLs in the request. To guarantee
        # that primary sources given by the user end up in direct_urls, the
        # query is re-checked with a regex and missing URLs are added. That
        # is deterministic and robust against LLM failure.
        user_urls = self._extract_urls_from_text(query)
        known_urls = {du.url for du in plan.direct_urls}
        for user_url in user_urls:
            if user_url in known_urls:
                continue
            if is_url_blocked(user_url):
                continue
            plan.direct_urls.append(DirectURL(
                url=user_url,
                reason="From the user request (primary source)",
            ))
            known_urls.add(user_url)
            logger.info(
                f"URL from the request added (overlooked by the LLM): {user_url}"
            )

        # ── Collect user-URL domains for preferential link following ──
        # All domains from user URLs (recognised by the LLM as well as added
        # by the safety net) are added to this run's dynamic followable
        # allow list. Links to these domains are thus accepted by the
        # _is_followable_link check, even if they are not in the static
        # _FOLLOWABLE_DOMAINS list (e.g. huggingface.co).
        from urllib.parse import urlparse
        self._user_url_domains = set()
        for user_url in user_urls:
            try:
                host = (urlparse(user_url).hostname or "").lower()
                host = host.removeprefix("www.")
                if host:
                    self._user_url_domains.add(host)
            except Exception:
                continue
        if self._user_url_domains:
            logger.info(
                f"User URL domains (preferred link following): "
                f"{sorted(self._user_url_domains)}"
            )

        logger.info(f"Analysis: {len(plan.questions)} questions, "
                     f"{len(plan.direct_urls)} URLs, "
                     f"{len(plan.git_repos)} git repos, "
                     f"{len(plan.directory_queries)} directory queries")

        # Person-directory safety net: if the directory is active but the LLM
        # generated no queries, create one automatically from the request
        if (self._use_directory and self.person_directory
                and self.person_directory.enabled
                and not plan.directory_queries):
            logger.info(
                "Person directory active but no directory_queries from the LLM — "
                "creating them from the request"
            )
            plan.directory_queries.append(DirectoryQuery(
                query_type="search",
                reason="Created from the request (the LLM generated none)",
                query=query[:150],
            ))

        if not plan.questions and not plan.direct_urls and not plan.directory_queries:
            logger.warning(
                "The analysis produced no search tasks — "
                "creating a fallback plan from the request"
            )
            # Fallback: use the original query as the search question
            plan.summary = f"Fallback plan for: {query[:200]}"
            # Split the query into 2-3 search terms
            words = query.split()
            if len(words) <= 6:
                search_terms = [query]
            else:
                # First half and second half as separate searches
                mid = len(words) // 2
                search_terms = [
                    " ".join(words[:mid]),
                    " ".join(words[mid:]),
                ]
            plan.questions.append(ResearchQuestion(
                id="F1",
                question=query[:300],
                search_terms=search_terms[:3],
                source_scope="web",
                priority="high",
                search_langs=["de", "en"],
            ))
            # The person directory as a fallback too, if enabled
            if self._use_directory and self.person_directory and self.person_directory.enabled:
                plan.directory_queries.append(DirectoryQuery(
                    query_type="search",
                    reason="Fallback",
                    query=query[:100],
                ))
            logger.info(
                f"Fallback plan: {len(plan.questions)} questions, "
                f"{len(plan.directory_queries)} directory queries"
            )

        return plan

    # ═══════════════════════════════════════════════════════════════
    # Phase 2: search + fetch
    # ═══════════════════════════════════════════════════════════════

    async def _run_search_and_fetch(
        self, ctx: HarvestContext, round_num: int,
        progress_callback: ProgressCallback,
    ) -> list[SourceDocument]:
        """Search and fetch sources — with de-duplication and batching."""
        plan = ctx.research_plan
        all_search_results: list[SearchResult] = []

        search_connector = self.connectors.get_search_connector()
        github_connector = self.connectors.get_connector_by_name("github")
        gitlab_connector = self.connectors.get_connector_by_name("gitlab")
        elastic_connector = self.connectors.get_connector_by_name("elasticsearch")

        # ── Search scope classification — ONCE per run ──
        # Must run BEFORE web_searches is built, because the language
        # selection affects the per-question languages. Fail-open: on
        # error _search_scope stays None ⇒ no restriction at all (plan
        # languages, both additional passes, year).
        if self._search_scope is None and search_connector is not None:
            try:
                from src.llm.classifier_adapter import HarvestModelAdapter
                from src.pipeline.classifiers.search_scope import (
                    classify_search_scope,
                )
                _plan_sum = plan.summary if plan else ""
                self._search_scope, _sc_call = await classify_search_scope(
                    ctx.query, _plan_sum or "",
                    HarvestModelAdapter(self.llm),
                )
                self._classifier_calls.append(_sc_call)
                await progress_callback("classifier", {
                    "name": "search_scope",
                    "time_sensitive": self._search_scope.time_sensitive,
                    "academic": self._search_scope.academic,
                    "languages": self._search_scope.languages,
                    "recency": self._search_scope.recency,
                    "confidence": self._search_scope.confidence,
                    "fallback_used": self._search_scope.fallback_used,
                })
                logger.info(
                    "🎚️ Search scope: time_sensitive=%s recency=%r "
                    "academic=%s languages=%s (conf. %.2f)",
                    self._search_scope.time_sensitive,
                    self._search_scope.recency or "year",
                    self._search_scope.academic,
                    self._search_scope.languages or "(plan languages)",
                    self._search_scope.confidence,
                )
            except Exception as e:
                logger.warning(
                    "Search-scope classifier skipped (%s) — "
                    "no search restriction", type(e).__name__,
                )

        _scope = self._search_scope
        # Apply the language allow list only if the classifier confidently
        # delivered a non-empty list (no fallback). Otherwise: None
        # ⇒ plan languages unchanged (fail-open).
        _scope_langs = (
            _scope.languages
            if (_scope and not _scope.fallback_used and _scope.languages)
            else None
        )

        def _effective_langs(plan_langs: list) -> list:
            """Apply the scope language allow list to a question's plan
            languages. Cuts the plan languages down to those the classifier
            recommends; if the intersection is empty, the classifier wins
            (it judged the topic as a whole). Without an allow list: plan
            languages unchanged."""
            base = plan_langs if plan_langs else ["de", "en"]
            if not _scope_langs:
                return base
            inter = [l for l in base if l in _scope_langs]
            return inter or list(_scope_langs)

        # ── Priority 1: direct URLs (every round — new ones from the gap analysis) ──
        # Fetch user URLs first: they are the primary sources named
        # explicitly by the user and should be processed before the
        # LLM-generated direct_urls and before the web searches. The reason
        # string identifies them unambiguously.
        # User URLs = URLs that appear in the request itself (whether the
        # analysis LLM or the safety net put them into the plan).
        request_urls = {u.rstrip("/") for u in self._extract_urls_from_text(ctx.query or "")}
        user_direct = [
            d for d in plan.direct_urls
            if d.url and d.url.rstrip("/") in request_urls
        ]
        other_direct = [
            d for d in plan.direct_urls if d not in user_direct
        ]
        ordered_direct = user_direct + other_direct

        for direct in ordered_direct:
            if direct.url and not self._is_url_seen(direct.url):
                all_search_results.append(SearchResult(
                    title=direct.url,
                    url=direct.url,
                    snippet=direct.reason,
                    source_type=SourceType.WEB_PAGE
                    if "github" not in direct.url else SourceType.GIT_REPO,
                    connector_name="direct",
                ))

        # Git repository READMEs — a loop of their own (NOT nested in the
        # direct loop: otherwise repos would be created N times with N
        # direct URLs and skipped completely with 0 direct URLs).
        for repo in plan.git_repos:
            url = f"https://github.com/{repo.owner}/{repo.repo}"
            if repo.platform == "gitlab":
                gl = self.connectors.get_connector_by_name("gitlab")
                if gl and hasattr(gl, 'config'):
                    url = f"{gl.config.base_url}/{repo.owner}/{repo.repo}"
            if self._is_url_seen(url):
                continue
            all_search_results.append(SearchResult(
                title=f"{repo.owner}/{repo.repo}",
                url=url,
                snippet=f"README: {repo.search_scope}",
                source_type=SourceType.GIT_REPO,
                connector_name=repo.platform,
            ))

        # ── Priority 2: collect and de-duplicate search terms ──
        seen_terms = set()
        web_searches = []       # (term, max_results, langs)
        github_searches = []    # (term, max_results)
        issue_searches = []     # (repo_fullname, term, max_results)
        elastic_searches = []   # (term, max_results)

        questions = plan.questions if round_num == 0 else []
        followup_queries = plan.followup_queries if round_num > 0 else []

        for q in questions:
            # Filter the plan languages against the scope allow list
            langs = _effective_langs(q.search_langs)
            # Tend towards breadth rather than depth — the reranker picks the
            # best max_sources_per_round from a larger, more varied pool
            # anyway. Fewer results per term, more terms instead (via a
            # higher max_web).
            priority_limits = {"high": 8, "medium": 5, "low": 3}
            max_r = priority_limits.get(q.priority, 5)
            for term in q.search_terms:
                term_key = term.lower().strip()
                if term_key in seen_terms:
                    continue
                seen_terms.add(term_key)

                if q.source_scope in ("web", "github"):
                    web_searches.append((term, max_r, langs))
                if q.source_scope == "github" and github_connector:
                    github_searches.append((term, 3))
                if q.source_scope == "gitlab" and gitlab_connector:
                    github_searches.append((term, 3))
                if q.source_scope == "elastic" and elastic_connector:
                    elastic_searches.append((term, max_r))

        for fq in followup_queries:
            fq_langs = _effective_langs(fq.get("search_langs", ["de", "en"]))
            raw_terms = fq.get("search_terms", [])

            # Follow-up queries (see src/pipeline/followup.py) carry search_terms
            # as a dict per language: {"de": [...], "en": [...]}. A flat list is
            # accepted as well.
            terms_with_langs: list[tuple[str, list[str]]] = []

            if isinstance(raw_terms, dict):
                # Dict format: {"de": [...], "en": [...]}
                # Every term gets the language it is written in as its
                # search language.
                for lang, term_list in raw_terms.items():
                    if isinstance(term_list, list):
                        for t in term_list:
                            if isinstance(t, str) and t.strip():
                                terms_with_langs.append(
                                    (t.strip(), [lang])
                                )
                    elif isinstance(term_list, str) and term_list.strip():
                        terms_with_langs.append(
                            (term_list.strip(), [lang])
                        )
            elif isinstance(raw_terms, list):
                # Flat list: ["term1", "term2"]
                for t in raw_terms:
                    if isinstance(t, str) and t.strip():
                        terms_with_langs.append(
                            (t.strip(), fq_langs)
                        )

            for term, langs in terms_with_langs:
                # Discard language-specific follow-up terms whose language the
                # scope allow list excludes (e.g. a term written in Chinese
                # for a purely German topic).
                if _scope_langs is not None:
                    langs = [l for l in langs if l in _scope_langs]
                    if not langs:
                        continue
                term_key = term.lower().strip()
                if term_key not in seen_terms:
                    seen_terms.add(term_key)
                    web_searches.append((term, 5, langs))

        if round_num == 0:
            for repo in plan.git_repos:
                if github_connector and repo.platform == "github":
                    for term in repo.search_terms[:2]:  # at most 2 terms per repository
                        issue_searches.append(
                            (f"{repo.owner}/{repo.repo}", term, 3)
                        )

        # Limit: at most N searches per connector. Searches are cheap
        # (meta search); the reranker then picks the best
        # max_sources_per_round for fetching. A larger pool ⇒ a better
        # selection, not more cost. Tunable via MAX_WEB_SEARCHES.
        max_web = getattr(self.config, "max_web_searches", 40)
        max_github = 12
        max_issues = 10
        max_elastic = 15

        # Institution mode: additional searches with the institution's site: filter
        if self._mode == "institution" and web_searches:
            institution_extras = []
            site = get_profile().site_filter()
            for term, max_r, langs in web_searches[:10]:
                # only if there is no site: in it yet
                if site and "site:" not in term.lower() and not get_profile().mentions(term):
                    institution_extras.append(
                        (f"{term} {site}", max_r, ["de"])
                    )
            web_searches.extend(institution_extras)
            # Allow more web searches in institution mode (tunable)
            max_web = getattr(self.config, "max_web_searches_institution", 60)

        # "Institution only": disable all external sources
        if self._institution_only:
            # Restrict all web searches to the institution's site: filter
            institution_web = []
            prof = get_profile()
            for term, max_r, langs in web_searches:
                clean = prof.strip_site_filter(term)
                if clean:
                    institution_web.append(
                        (f"{clean} {prof.site_filter()}".strip(), max_r, ["de"])
                    )
            web_searches = institution_web
            # No GitHub/GitLab/Elastic
            github_searches = []
            issue_searches = []
            elastic_searches = []
            # Direct URLs only from the institution's domains
            all_search_results = [
                sr for sr in all_search_results
                if get_profile().matches_url(sr.url or "")
            ]
            logger.info(
                f"Institution-only mode: {len(web_searches)} web searches "
                f"(all with the institution's site: filter), "
                f"{len(all_search_results)} direct institution URLs"
            )

        web_searches = web_searches[:max_web]
        github_searches = github_searches[:max_github]
        issue_searches = issue_searches[:max_issues]
        elastic_searches = elastic_searches[:max_elastic]

        # Collect all languages used, for logging
        all_langs = set()
        for _, _, langs in web_searches:
            all_langs.update(langs)

        logger.info(
            f"Round {round_num + 1}: {len(web_searches)} web searches "
            f"(languages: {', '.join(sorted(all_langs)) or 'all'}), "
            f"{len(github_searches)} GitHub searches, "
            f"{len(issue_searches)} issue searches, "
            f"{len(elastic_searches)} Elastic searches, "
            f"{'Solr active, ' if self.connectors.get_connector_by_name('solr') else ''}"
            f"{len(all_search_results)} direct URLs"
        )

        # ── Priority 3: run the searches (in batches) ──
        search_tasks = []

        # SearXNG (only if available)
        if search_connector:
            searxng_available = True
            if hasattr(search_connector, 'is_available'):
                searxng_available = search_connector.is_available
                if not searxng_available:
                    # check whether it is reachable after all
                    if hasattr(search_connector, 'check_connectivity'):
                        searxng_available = await search_connector.check_connectivity()

            if searxng_available:
                # Academic-only mode: instead of categories="general" use the
                # allow list of academic engines (Wikipedia, Wikidata, Google
                # Scholar, Semantic Scholar, ArXiv, PubMed)
                engines_whitelist: str | None = None
                if self._academic_only:
                    try:
                        from src.connectors.searxng import (
                            ACADEMIC_ENGINES_STRING,
                        )
                        engines_whitelist = ACADEMIC_ENGINES_STRING
                        logger.info(
                            f"🎓 Academic-only mode: SearXNG searches "
                            f"use the engine allow list: {engines_whitelist}"
                        )
                    except ImportError:
                        logger.warning(
                            "ACADEMIC_ENGINES_STRING not available, "
                            "falling back to categories='science'"
                        )
                        engines_whitelist = None

                # One search per search term and per language.
                # De-duplication: the same term in several languages is only
                # searched once per language
                seen_searches = set()
                for term, max_r, langs in web_searches:
                    for lang in langs:
                        key = (term.lower().strip(), lang)
                        if key in seen_searches:
                            continue
                        seen_searches.add(key)
                        search_tasks.append(
                            search_connector.search(
                                term, max_r,
                                language=lang,
                                engines=engines_whitelist,
                            )
                        )

                # ── Additional passes: fan-out control ──────────────────
                # The scope was already classified ONCE above. Here it is only
                # applied:
                #  - recent only with time_sensitive, science only with
                #    academic (fail-open: no scope ⇒ both on).
                #  - additional passes ONLY in round 0 — follow-up rounds fill
                #    gaps and should stay narrow and targeted.
                #  - only the terms of high-priority questions
                #    (max_r == priority_limits['high'] == 8), rather than
                #    simply the first 5.
                #  - time_range from the classifier (week/month/year) rather
                #    than a fixed 'year'.
                _scope = self._search_scope
                _do_recent = (_scope is None) or _scope.time_sensitive
                _do_science = (_scope is None) or _scope.academic
                _recency = (
                    _scope.recency if (_scope and _scope.recency)
                    else "year"
                )
                _hi_searches = [
                    ws for ws in web_searches if ws[1] >= 8
                ] or web_searches[:5]

                if round_num == 0 and _do_recent:
                    for term, max_r, langs in _hi_searches:
                        for lang in langs:
                            key = (term.lower().strip(), lang, "recent")
                            if key in seen_searches:
                                continue
                            seen_searches.add(key)
                            search_tasks.append(
                                search_connector.search(
                                    term, max_r,
                                    language=lang,
                                    time_range=_recency,
                                    engines=engines_whitelist,
                                )
                            )

                # Science category only outside the academic-only mode (there
                # the engines allow list already covers
                # Scholar/arXiv/PubMed/Semantic).
                if (round_num == 0 and not self._academic_only
                        and _do_science):
                    for term, max_r, langs in _hi_searches:
                        key = (term.lower().strip(), "science")
                        if key in seen_searches:
                            continue
                        seen_searches.add(key)
                        search_tasks.append(
                            search_connector.search(
                                term, max_r,
                                categories="science",
                            )
                        )

        # GitHub
        if github_connector:
            for term, max_r in github_searches:
                search_tasks.append(github_connector.search(term, max_r))
            for repo_name, term, max_r in issue_searches:
                search_tasks.append(
                    github_connector.search_issues(repo_name, term, max_r)
                )

        # Elasticsearch
        if elastic_connector:
            for term, max_r in elastic_searches:
                search_tasks.append(elastic_connector.search(term, max_r))

            # Elastic enrichment (web terms automatically against the index)
            # is an additional pass → round 0 only.
            if (round_num == 0 and not elastic_searches
                    and web_searches):
                enrichment_limit = min(3, len(web_searches))
                for term, _, _langs in web_searches[:enrichment_limit]:
                    search_tasks.append(elastic_connector.search(term, 3))

        # Solr (the institution's web index)
        solr_connector = self.connectors.get_connector_by_name("solr")
        if solr_connector:
            # Institution mode: run all search terms against Solr as well.
            # Web mode: only terms that mention the institution.
            solr_terms = []
            if self._mode == "institution":
                # Institution mode: Solr is a PRIMARY source (the
                # institution's web index), not an additional pass — so it
                # keeps running every round (unlike recent/science/Elastic).
                for term, max_r, langs in web_searches[:15]:
                    # remove the site: filter (Solr only indexes the institution)
                    clean_term = get_profile().strip_site_filter(term)
                    if clean_term and clean_term not in solr_terms:
                        solr_terms.append(clean_term)
                        for lang in langs:
                            search_tasks.append(
                                solr_connector.search(
                                    clean_term, max_r, language=lang,
                                )
                            )
            elif round_num == 0:
                # Web mode: Solr only for terms that mention the
                # institution, and round 0 only (incidental additional pass).
                for term, max_r, langs in web_searches[:5]:
                    if get_profile().mentions(term):
                        for lang in langs:
                            search_tasks.append(
                                solr_connector.search(term, max_r, language=lang)
                            )
            if solr_terms:
                logger.info(
                    f"Solr: {len(solr_terms)} search terms "
                    f"against the institution's web index"
                )

        # Run the searches in batches (not all at once!)
        batch_size = 10
        for i in range(0, len(search_tasks), batch_size):
            batch = search_tasks[i:i + batch_size]
            results = await asyncio.gather(*batch, return_exceptions=True)
            for result in results:
                if isinstance(result, list):
                    all_search_results.extend(result)
                elif isinstance(result, Exception):
                    logger.debug(f"Search failed: {result}")

            if self._stop_requested:
                break

        # ── De-duplicate, filter and limit ──
        unique_results = []
        blocked_count = 0
        institution_filtered = 0
        for sr in all_search_results:
            if not sr.url or self._is_url_seen(sr.url):
                continue
            if is_url_blocked(sr.url):
                blocked_count += 1
                logger.debug(f"Blocked: {sr.url}")
                continue
            # "Institution only": let only the institution's URLs through
            if self._institution_only and not get_profile().matches_url(sr.url):
                institution_filtered += 1
                continue
            self._mark_url_seen(sr.url)
            unique_results.append(sr)

        if blocked_count:
            logger.info(f"🛡️ {blocked_count} URL(s) blocked by the domain filter")
        if institution_filtered:
            logger.info(f"🏛️ {institution_filtered} external URL(s) filtered in institution-only mode")

        # ── Relevance ranking: fetch better sources first ──
        def _source_authority(sr: SearchResult) -> int:
            """Higher score = fetched first."""
            score = 0
            url = (sr.url or "").lower()
            # prefer the institution's own sources
            if get_profile().matches_url(url):
                score = 100
            # prefer Solr/person-directory results
            elif sr.connector_name and sr.connector_name.startswith("solr"):
                score = 90
            elif sr.connector_name == "directory":
                score = 95
            # official / academic domains
            elif any(d in url for d in [
                ".edu", ".ac.", ".gov", "scholar.google",
                "doi.org", "arxiv.org", "orcid.org",
            ]):
                score = 80
            # known quality sources
            elif any(d in url for d in [
                "wikipedia.org", "github.com", "researchgate.net",
            ]):
                score = 60
            else:
                score = 40

            # Snippet relevance bonus (keyword density)
            snippet = (sr.snippet or "").lower()
            if snippet and ctx.query:
                query_words = set(
                    w for w in ctx.query.lower().split() if len(w) > 3
                )
                if query_words:
                    hits = sum(1 for w in query_words if w in snippet)
                    # up to +15 bonus for keyword-rich snippets
                    score += min(15, hits * 5)

            return score

        # ── Pre-fetch ranking ──────────────────────────────────
        # _source_authority (domain score + keyword overlap) is a cheap
        # heuristic. If a reranker is configured, IT decides the selection:
        # snippets are ranked against the research questions, the top
        # `max_sources_per_round` go to the fetch. _source_authority stays
        # the tie-break/fallback (fail-open: reranker unreachable ⇒ the
        # heuristic order).
        unique_results.sort(key=_source_authority, reverse=True)

        reranked_applied = False
        if len(unique_results) > self.config.max_sources_per_round:
            try:
                from src.llm.reranker import (
                    build_reranker_from_pipeline_config,
                )
                rr = build_reranker_from_pipeline_config(self.config)
                if rr.available:
                    plan_qs = (
                        ctx.research_plan.questions
                        if ctx.research_plan else []
                    )
                    rerank_query = (
                        " ".join(q.question for q in plan_qs[:5]).strip()
                        or ctx.query
                    )

                    def _doc_text(sr) -> str:
                        return (
                            f"{sr.title or ''}\n{sr.snippet or ''}"
                        )[:1500]

                    ranking = await rr.rank(
                        rerank_query,
                        [_doc_text(sr) for sr in unique_results],
                    )
                    if any(s != 0.0 for _, s in ranking):
                        unique_results = [
                            unique_results[i] for i, _ in ranking
                        ]
                        reranked_applied = True
                        logger.info(
                            "🔝 Reranker: %d candidates reordered "
                            "(selection for fetching)",
                            len(unique_results),
                        )
            except Exception as e:
                logger.warning(
                    "Reranker pre-fetch skipped (%s)",
                    type(e).__name__,
                )

        unique_results = unique_results[:self.config.max_sources_per_round]

        await progress_callback("search_results", {
            "total_found": len(all_search_results),
            "unique": len(unique_results),
            "reranked": reranked_applied,
            "results": [(r.title, r.url) for r in unique_results[:20]],
        })

        # ── Fetch in parallel ──
        semaphore = asyncio.Semaphore(self.config.max_parallel_fetches)

        async def fetch_one(sr: SearchResult) -> SourceDocument | None:
            async with semaphore:
                connector = self.connectors.route_url(sr.url)
                if not connector:
                    logger.warning(f"No connector for: {sr.url}")
                    return None
                try:
                    doc = await connector.fetch(sr.url)
                    await progress_callback("fetch_one", {
                        "url": sr.url, "title": doc.title,
                        "length": doc.content_length,
                    })
                    return doc
                except Exception as e:
                    logger.warning(f"Fetch failed: {sr.url}: {e}")
                    return None

        fetch_tasks = [fetch_one(sr) for sr in unique_results]
        fetch_results = await asyncio.gather(*fetch_tasks)

        new_sources = [doc for doc in fetch_results if doc is not None]

        # ── Content quality filter ──
        filtered_sources = []
        skipped = 0
        thin_content = 0
        for doc in new_sources:
            # Minimum content: 300 characters of meaningful text
            if doc.content_length < 300:
                thin_content += 1
                logger.debug(
                    f"Too little content ({doc.content_length} characters): {doc.url}"
                )
                continue
            if self._is_listing_page(doc):
                logger.info(
                    f"⚠️ Listing page skipped: {doc.url} "
                    f"(contains data on several projects)"
                )
                skipped += 1
                continue
            filtered_sources.append(doc)

        if skipped:
            logger.info(f"🛡️ {skipped} listing page(s) filtered")
        if thin_content:
            logger.info(f"📄 {thin_content} page(s) with too little content (<300 characters) skipped")
        new_sources = filtered_sources

        # ── Content-based de-duplication ──
        # Recognises mirror subdomains with identical content (e.g. a CMS
        # tree served under several host names). The URL de-duplication
        # before it does not apply, because the host names differ.
        from src.pipeline.content_dedup import dedupe_sources_by_content
        pre_dedup_count = len(new_sources)
        new_sources, removed = dedupe_sources_by_content(new_sources)
        if removed:
            logger.info(
                f"🔄 Content dedup: {pre_dedup_count} → {len(new_sources)} "
                f"sources ({removed} mirror duplicates removed)"
            )

        # The off-topic filter does NOT run here — but ONCE at the end of
        # this method (after person directory/OpenAlex/link following), so
        # that the classifier sees the COMPLETE set of new sources of the
        # round. Running it here would be too early (directory/OpenAlex/
        # links unfiltered), and a second run in the DAG over the whole
        # corpus of every round would be non-deterministic and double the
        # LLM cost.

        # ── Person-directory queries (persons/units) ──
        if (round_num == 0 and self._use_directory
                and self.person_directory and self.person_directory.enabled):
            seen_pids = set()  # de-duplication by PID
            for zq in plan.directory_queries:
                try:
                    search_query = zq.query  # Reused field for search text
                    if not search_query:
                        continue

                    directory_results = await self.person_directory.search(
                        search_query, max_results=10,
                        subject_area=zq.subject_area,
                        org_filter=zq.org_filter,
                        query_anchor=getattr(ctx, 'query_anchor', None),
                    )
                    for zr in directory_results:
                        directory_url = zr.real_url
                        # De-duplicate by URL and PID
                        if self._is_url_seen(directory_url):
                            continue
                        if zr.pid in seen_pids:
                            continue
                        self._mark_url_seen(directory_url)
                        seen_pids.add(zr.pid)

                        # Title with the correct academic title
                        display_name = (
                            f"{zr.academic_title} {zr.person_name}".strip()
                            if zr.academic_title else zr.person_name
                        )

                        # Load full person details
                        # (all organisations, roles, contact details)
                        content = self._enrich_directory_markdown(
                            zr, display_name
                        )

                        doc = SourceDocument(
                            url=directory_url,
                            title=f"Directory: {display_name}"
                                  + (f" — {zr.page_title}" if zr.page_title else ""),
                            content=content,
                            source_type=(SourceType.DIRECTORY_PERSON
                                         if not zr.page_id
                                         else SourceType.DIRECTORY_ORG),
                        )
                        new_sources.append(doc)
                        await progress_callback("fetch_one", {
                            "url": directory_url,
                            "title": doc.title,
                            "length": doc.content_length,
                        })

                        # Fetch the homepage URL as a web page too (the
                        # directory Markdown only has metadata, not the
                        # page content itself)
                        hp_url = zr.homepage_url or ""
                        if (hp_url.startswith("http")
                                and not self._is_url_seen(hp_url)
                                and (not self._institution_only
                                     or get_profile().matches_url(hp_url))):
                            self._mark_url_seen(hp_url)
                            connector = self.connectors.route_url(hp_url)
                            if connector:
                                try:
                                    hp_doc = await connector.fetch(hp_url)
                                    if hp_doc and hp_doc.content_length > 50:
                                        hp_doc.title = (
                                            f"Homepage: {display_name}"
                                        )
                                        new_sources.append(hp_doc)
                                        await progress_callback("fetch_one", {
                                            "url": hp_url,
                                            "title": hp_doc.title,
                                            "length": hp_doc.content_length,
                                        })
                                        logger.info(
                                            f"  Directory homepage fetched: "
                                            f"{hp_url}"
                                        )
                                except Exception as e:
                                    logger.debug(
                                        f"Directory homepage fetch "
                                        f"failed: {hp_url}: {e}"
                                    )

                    if directory_results:
                        logger.info(
                            f"Directory search '{search_query}': "
                            f"{len(directory_results)} hits"
                        )

                        # OpenAlex: query publications of the institution's people
                        if self._mode == "institution":
                            for zr2 in directory_results[:5]:
                                if zr2.pid not in seen_pids:
                                    continue  # only already processed ones
                                orcid = ""
                                try:
                                    details = self.person_directory.get_person_details(zr2.pid)
                                    if details and details.get("person"):
                                        orcid = details["person"].get("orcid", "")
                                except Exception:
                                    pass
                                try:
                                    oa_doc = await self._query_openalex_author(
                                        zr2.person_name, orcid,
                                    )
                                    if oa_doc:
                                        new_sources.append(oa_doc)
                                        await progress_callback("fetch_one", {
                                            "url": oa_doc.url,
                                            "title": oa_doc.title,
                                            "length": oa_doc.content_length,
                                        })
                                except Exception as e:
                                    logger.debug(
                                        f"OpenAlex for {zr2.person_name}: {e}"
                                    )
                except Exception as e:
                    logger.warning(f"Directory search failed: {zq} — {e}")

        # ── OpenAlex topic search (institution mode, round 1) ──
        if (round_num == 0 and self._mode == "institution"
                and plan.questions):
            # Extract the main topic from the request (first question)
            main_topic = plan.questions[0].question
            # Reduce to core search terms (max 5 words)
            topic_terms = plan.questions[0].search_terms[:1]
            topic_query = topic_terms[0] if topic_terms else main_topic
            try:
                oa_topic_doc = await self._query_openalex_topic(topic_query)
                if oa_topic_doc:
                    new_sources.append(oa_topic_doc)
                    await progress_callback("fetch_one", {
                        "url": oa_topic_doc.url,
                        "title": oa_topic_doc.title,
                        "length": oa_topic_doc.content_length,
                    })
            except Exception as e:
                logger.debug(f"OpenAlex Topic-Search: {e}")

        # ── Link following: follow links from primary sources ──
        # Primary sources = directory pages, directory homepages, direct_urls.
        # These often contain the most valuable further links.
        primary_sources = [
            s for s in new_sources
            if s.source_type in (
                SourceType.DIRECTORY_PERSON, SourceType.DIRECTORY_ORG, SourceType.WEB_PAGE,
            )
        ]
        if primary_sources and round_num == 0:
            # Relevance query for the link gate: the same logic as for the
            # pre-fetch reranker — plan questions, otherwise the original
            # query. Passed on to _follow_primary_links; fail-open if there
            # is no reranker.
            _plan_qs = (
                ctx.research_plan.questions if ctx.research_plan else []
            )
            _link_rel_q = (
                " ".join(q.question for q in _plan_qs[:5]).strip()
                or ctx.query
            )
            # Identify user-URL sources — they get higher link-following
            # limits, because the user named them explicitly as primary
            # sources.
            request_urls = {u.rstrip("/") for u in self._extract_urls_from_text(ctx.query or "")}
            user_url_set = {
                du.url for du in ctx.research_plan.direct_urls
                if du.url and du.url.rstrip("/") in request_urls
            }
            user_primary = [
                s for s in primary_sources if s.url in user_url_set
            ]
            other_primary = [
                s for s in primary_sources if s.url not in user_url_set
            ]

            # Institution mode: larger limits
            lps = 30 if self._mode == "institution" else 20
            lt = 60 if self._mode == "institution" else 40

            # User-URL sources: even higher limits. The user supplied these
            # pages explicitly — from them, links are followed more broadly
            # and deeply than from web-search hits.
            user_followed: list[SourceDocument] = []
            if user_primary:
                user_followed = await self._follow_primary_links(
                    user_primary, progress_callback,
                    max_links_per_source=50,
                    max_total=100,
                )
                if user_followed:
                    new_sources.extend(user_followed)
                    logger.info(
                        f"🔗 {len(user_followed)} link(s) from "
                        f"user URLs loaded (level 1)"
                    )

                    # Level 2 for user URLs — in web mode too. Follow links
                    # once more from the follow-up pages (e.g. GitHub repo,
                    # README), to reach tutorials, configs and technical
                    # details. Level 2 is already one trust level removed →
                    # relevance gate on (level 1 above stays ungated: the user
                    # named these URLs explicitly).
                    followed_2 = await self._follow_primary_links(
                        user_followed, progress_callback,
                        max_links_per_source=15,
                        max_total=50,
                        relevance_query=_link_rel_q,
                    )
                    if followed_2:
                        new_sources.extend(followed_2)
                        logger.info(
                            f"🔗 {len(followed_2)} link(s) from "
                            f"user-URL follow-up pages loaded (level 2)"
                        )

            # Other primary sources with standard limits. They come from
            # web-search hits (not named by the user) → relevance gate on;
            # without it, dictionary sites and the like leak in here.
            if other_primary:
                followed = await self._follow_primary_links(
                    other_primary, progress_callback,
                    max_links_per_source=lps,
                    max_total=lt,
                    relevance_query=_link_rel_q,
                )
                if followed:
                    new_sources.extend(followed)
                    logger.info(
                        f"🔗 {len(followed)} link(s) from "
                        f"primary sources loaded (level 1)"
                    )

                # Institution mode: level 2 — also follow links from the
                # institution pages just followed (one trust level further)
                if self._mode == "institution" and followed:
                    institution_followed = [
                        s for s in followed
                        if get_profile().matches_url(s.url or "")
                    ]
                    if institution_followed:
                        followed_2 = await self._follow_primary_links(
                            institution_followed, progress_callback,
                            max_links_per_source=10,
                            max_total=30,
                            relevance_query=_link_rel_q,
                        )
                        if followed_2:
                            new_sources.extend(followed_2)
                            logger.info(
                                f"🔗 {len(followed_2)} link(s) from "
                                f"institution pages loaded (level 2)"
                            )

        # ── Off-topic filter — EXACTLY ONCE, over the complete set of new
        # sources of this round (incl. person directory, OpenAlex, followed
        # links). `judge_source_relevance` decides semantically;
        # `_is_listing_page` above was only the cheap first stage
        # (defence in depth). Only NEW sources are judged — sources
        # accepted in earlier rounds stay stable.
        new_sources = await self._filter_off_topic_sources(
            new_sources, ctx.research_plan, progress_callback,
        )

        # ctx.sources is filled by the calling SearchAndFetchNode (DAG node
        # contract). NO extend here — otherwise every source would end up
        # twice in ctx.sources (→ duplicate extracts).
        logger.info(f"Round {round_num + 1}: {len(new_sources)} sources fetched")
        return new_sources

    def _enrich_directory_markdown(self, zr, display_name: str) -> str:
        """Build rich Markdown from a directory search result + full details.

        Loads all position details (roles, organisations) via
        get_person_details() and builds a complete person profile.
        """
        # Base Markdown from the search result
        lines = [f"## {display_name}"]

        # Load full details
        details = None
        if self.person_directory and zr.pid:
            try:
                details = self.person_directory.get_person_details(zr.pid)
            except Exception as e:
                logger.debug(f"Directory details for PID {zr.pid}: {e}")

        if details and details.get("person"):
            p = details["person"]
            academic_title = p.get("academic_title", "")
            lines.append(
                f"**Academic title:** "
                f"{academic_title if academic_title else '(no academic title recorded)'}"
            )
            status = p.get("status", "")
            if status:
                lines.append(f"**Staff status:** {status}")
            email = p.get("email", "")
            if email:
                lines.append(f"**E-mail:** {email}")
            homepage = p.get("homepage_url", "")
            if homepage:
                lines.append(f"**Homepage:** {homepage}")
            orcid = p.get("orcid", "")
            if orcid:
                orcid_url = orcid if orcid.startswith("http") else f"https://orcid.org/{orcid}"
                lines.append(f"**ORCID:** [{orcid}]({orcid_url})")
            lines.append(f"**Profile:** {zr.real_url}")

            # All position details / organisational memberships
            orgs = details.get("orgs", [])
            if orgs:
                lines.append("\n### Organisational assignment")
                for org in orgs:
                    org_name = org.get("org_name", "") or org.get("full_path", "")
                    role = org.get("role", "")
                    subject_area = org.get("subject_area", "")
                    subject_area_en = org.get("subject_area_en", "")
                    phone = org.get("phone", "")
                    room = org.get("room", "")
                    building = org.get("building", "")
                    da_email = org.get("email", "")

                    org_line = f"- **{org_name}**"
                    if role:
                        org_line += f" — {role}"
                    if subject_area:
                        sg = subject_area
                        if subject_area_en:
                            sg += f" ({subject_area_en})"
                        org_line += f"\n  Subject area: {sg}"
                    if phone:
                        org_line += f"\n  Phone: {phone}"
                    if room and building:
                        org_line += f"\n  Location: {building}, room {room}"
                    elif room:
                        org_line += f"\n  Room: {room}"
                    if da_email and da_email != email:
                        org_line += f"\n  E-mail: {da_email}"
                    lines.append(org_line)

                # Full organisation path
                full_paths = [
                    org.get("full_path", "") for org in orgs
                    if org.get("full_path")
                ]
                if full_paths:
                    lines.append(
                        "\n**Organisation path:** "
                        + " | ".join(dict.fromkeys(full_paths))
                    )

            # Pages (CV, publications, homepage)
            pages = details.get("pages", [])
            for page in pages:
                page_title = page.get("title", "")
                page_content = page.get("content", "")
                if page_content:
                    lines.append(f"\n### {page_title}")
                    lines.append(page_content[:5000])

            # Committee memberships
            memberships_raw = p.get("memberships", "")
            if memberships_raw:
                try:
                    memberships_list = json.loads(memberships_raw)
                    if memberships_list:
                        lines.append("\n### Committee memberships")
                        for m in memberships_list:
                            if isinstance(m, dict):
                                name = m.get("name", m.get("gremium", str(m)))
                                lines.append(f"- {name}")
                            elif isinstance(m, str):
                                lines.append(f"- {m}")
                            else:
                                lines.append(f"- {m}")
                except (json.JSONDecodeError, TypeError):
                    if memberships_raw.strip():
                        lines.append(f"\n**Memberships:** {memberships_raw}")
        else:
            # Fallback: base Markdown from the search result
            lines = [zr.to_markdown()]

        return "\n".join(lines)

    @staticmethod
    def _extract_urls_from_text(text: str) -> list[str]:
        """Extract all http(s) URLs from free text.

        Robust against various surroundings of URLs: directly in the text,
        in brackets, at the end of a line with a backslash, after a colon.
        Removes typical trailing characters (full stops, commas, closing
        brackets) that belong grammatically to the sentence, not the URL.

        De-duplication is left to the caller.

        Returns:
            List of the URLs found, in order of appearance.
        """
        if not text:
            return []
        # Matches http:// and https://, stops at whitespace and typical
        # container characters (<, >, ", ', ], ))
        pattern = re.compile(r'https?://[^\s<>"\'\]\)\}]+')
        urls: list[str] = []
        seen: set[str] = set()
        for m in pattern.finditer(text):
            url = m.group(0)
            # Remove trailing punctuation that belongs grammatically to the
            # sentence (full stop, comma, semicolon, colon, closing bracket,
            # backslash at the end of a line)
            url = url.rstrip('.,;:)]}"\'\\')
            if not url:
                continue
            if url in seen:
                continue
            seen.add(url)
            urls.append(url)
        return urls

    @staticmethod
    def _is_listing_page(doc) -> bool:
        """Recognise listing pages that contain data about many projects.

        Such pages lead the harvest LLM to attribute data of other
        projects to the project being researched.
        """
        url = (doc.url or "").lower()
        content = doc.content or ""

        # 1. GitHub listing pages by URL
        github_listing_patterns = [
            "/search?", "/trending", "/explore", "/topics/",
            "/collections/", "?tab=repositories", "?tab=stars",
        ]
        if "github.com" in url:
            for pattern in github_listing_patterns:
                if pattern in url:
                    return True

        # 2. Content-based detection: many different repository links
        if "github.com" in url or "github" in content[:500].lower():
            # Count unique github.com/owner/repo patterns
            repo_urls = set(re.findall(
                r"github\.com/([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)",
                content[:20000]
            ))
            if len(repo_urls) > 8:
                return True

        # 3. "Awesome" lists and collections
        title = (doc.title or "").lower()
        if any(kw in title for kw in [
            "awesome-", "trending", "explore github",
            "top repositories", "popular repos",
        ]):
            return True

        return False

    # ═══════════════════════════════════════════════════════════════
    # Link following: further links from primary sources
    # ═══════════════════════════════════════════════════════════════

    # Academic / relevant domains for link following
    _FOLLOWABLE_DOMAINS = {
        # Academic platforms
        "orcid.org", "scholar.google.com", "scholar.google.de",
        "researchgate.net", "academia.edu", "dblp.org",
        "semanticscholar.org", "arxiv.org", "doi.org",
        "crossref.org", "unpaywall.org", "openalex.org",
        # Scholarly publishers / open access
        "springer.com", "link.springer.com", "wiley.com",
        "sciencedirect.com", "nature.com", "ieee.org",
        "acm.org", "dl.acm.org", "mdpi.com",
        "frontiersin.org", "plos.org", "zenodo.org",
        # Code / tech
        "github.com", "gitlab.com",
    }

    @staticmethod
    def _extract_links(content: str, source_url: str) -> list[str]:
        """Extract HTTP(S) links from text content.

        Finds URLs in:
        - trafilatura output (URLs as plain text)
        - Markdown links [text](url)
        - raw HTML href="url"
        - person-directory Markdown (URLs as text)
        """
        urls = set()
        if not content:
            return []

        # Pattern 1: complete HTTP(S) URLs
        url_pattern = re.compile(
            r'https?://[^\s<>"\')\]\},;]+[^\s<>"\')\]\},;.\)]'
        )
        for match in url_pattern.finditer(content[:50000]):
            url = match.group(0)
            # Clean up trailing punctuation
            url = url.rstrip(".,;:!?)]}>")
            if len(url) > 15 and "." in url:
                urls.add(url)

        # Pattern 2: Markdown-Links [text](url)
        md_pattern = re.compile(r'\]\((https?://[^)]+)\)')
        for match in md_pattern.finditer(content[:50000]):
            urls.add(match.group(1))

        # Pattern 3: href="url"
        href_pattern = re.compile(r'href=["\']?(https?://[^"\'>\s]+)')
        for match in href_pattern.finditer(content[:50000]):
            urls.add(match.group(1))

        return list(urls)

    def _is_followable_link(self, url: str, source_url: str) -> bool:
        """Check whether a link should be followed.

        Criteria:
        - same domain as the source page (e.g. sub-pages)
        - or a known academic domain
        - not blocked
        - not seen yet
        - in "institution only" mode: only the institution's links
        """
        from urllib.parse import urlparse

        if not url or not url.startswith("http"):
            return False

        # Already seen?
        if self._is_url_seen(url):
            return False

        # Blocked?
        if is_url_blocked(url):
            return False

        parsed = urlparse(url)
        host = (parsed.hostname or "").lower().removeprefix("www.")

        # "Institution only": exclusively the institution's domains
        if self._institution_only:
            return get_profile().matches_host(host)
        source_host = (
            urlparse(source_url).hostname or ""
        ).lower().removeprefix("www.")

        # Same domain (incl. subdomains) → always follow
        if host == source_host:
            return True
        # Subdomain of the source domain (e.g. blogs.example.edu)
        if host.endswith(f".{source_host}"):
            return True
        # The source domain is a subdomain (e.g. source = cms.example.edu,
        # link = www.example.edu)
        if source_host.endswith(f".{host}"):
            return True

        # Known academic domain
        for domain in self._FOLLOWABLE_DOMAINS:
            if host == domain or host.endswith(f".{domain}"):
                return True

        # User-URL domains (dynamic per run): if the user supplied a URL on
        # this domain, we follow links on the same domain even if it is
        # not in the static allow list (e.g. huggingface.co, docs.x.y).
        for user_domain in self._user_url_domains:
            if host == user_domain or host.endswith(f".{user_domain}"):
                return True

        # Always follow the institution's (sub)domains
        if get_profile().matches_host(host):
            return True

        return False

    async def _follow_primary_links(
        self,
        primary_sources: list[SourceDocument],
        progress_callback: ProgressCallback,
        max_links_per_source: int = 10,
        max_total: int = 25,
        relevance_query: str = "",
    ) -> list[SourceDocument]:
        """Follow links from primary sources (person directory, homepages).

        Extracts links, filters them to relevant domains, optionally to
        thematic relevance (reranker), fetches the pages and returns them
        as new SourceDocuments.

        relevance_query: if set AND a reranker is configured, the
            candidate links (anchor text + URL) are ranked against this
            query and only the most relevant ones are fetched. Without
            this gate the pipeline would follow links from an arXiv paper
            to dictionary sites and the like — wasted 5-6 s fetches each
            that the off-topic filter only sorts out afterwards (and
            expensively). Fail-open: no reranker / empty query ⇒ no gate.
        """
        all_link_urls: list[tuple[str, str]] = []  # (url, source_title)

        for source in primary_sources:
            links = self._extract_links(source.content, source.url)
            followable = [
                url for url in links
                if self._is_followable_link(url, source.url)
            ]

            # Limit per source
            for url in followable[:max_links_per_source]:
                all_link_urls.append((url, source.title))

            if len(all_link_urls) >= max_total:
                break

        all_link_urls = all_link_urls[:max_total]

        if not all_link_urls:
            return []

        # ── Relevance gate ────────────────────────────────
        # Rerank the candidate links against the research question(s)
        # before starting expensive fetches. Only worthwhile with a
        # noticeable number of candidates. Fail-open: reranker unavailable /
        # empty query ⇒ list unchanged.
        if relevance_query and len(all_link_urls) > 5:
            try:
                from src.llm.reranker import (
                    build_reranker_from_pipeline_config,
                )
                rr = build_reranker_from_pipeline_config(self.config)
                if rr.available:
                    def _link_doc(item) -> str:
                        url, title = item
                        # anchor/source title + meaningful URL path parts
                        from urllib.parse import urlparse, unquote
                        p = urlparse(url)
                        path_words = unquote(
                            p.path.replace("/", " ").replace("-", " ")
                            .replace("_", " ")
                        )
                        return f"{title or ''} {p.netloc} {path_words}"[:300]

                    ranked = await rr.order(
                        relevance_query, all_link_urls, key=_link_doc,
                    )
                    # Conservative: discard the weaker half, but keep at least
                    # 5 (recall protection, fail-open).
                    keep_n = max(5, (len(ranked) + 1) // 2)
                    dropped = len(ranked) - keep_n
                    all_link_urls = ranked[:keep_n]
                    if dropped > 0:
                        logger.info(
                            "🔗 Relevance gate: %d of %d candidate links "
                            "discarded (reranker)",
                            dropped, dropped + keep_n,
                        )
            except Exception as e:
                logger.warning(
                    "Link relevance gate skipped (%s)",
                    type(e).__name__,
                )

        logger.info(
            f"🔗 Link following: {len(all_link_urls)} links "
            f"from {len(primary_sources)} primary sources"
        )

        # Fetch in parallel
        semaphore = asyncio.Semaphore(
            self.config.max_parallel_fetches
        )
        new_docs: list[SourceDocument] = []

        async def fetch_link(url: str, source_title: str):
            async with semaphore:
                # check again (might have been seen in the meantime)
                if self._is_url_seen(url):
                    return
                self._mark_url_seen(url)

                connector = self.connectors.route_url(url)
                if not connector:
                    return

                try:
                    doc = await connector.fetch(url)
                    if not doc or doc.content_length < 50:
                        return

                    # Listing page?
                    if self._is_listing_page(doc):
                        return

                    doc.title = f"🔗 {doc.title}" if doc.title else url
                    new_docs.append(doc)

                    await progress_callback("fetch_one", {
                        "url": url,
                        "title": doc.title,
                        "length": doc.content_length,
                    })
                except Exception as e:
                    logger.debug(f"Link follow failed: {url}: {e}")

        tasks = [
            fetch_link(url, title)
            for url, title in all_link_urls
        ]
        await asyncio.gather(*tasks)

        return new_docs

    # ═══════════════════════════════════════════════════════════════
    # Phase 3: Harvest
    # ═══════════════════════════════════════════════════════════════

    async def _run_harvest(
        self, sources: list[SourceDocument],
        plan: ResearchPlan,
        progress_callback: ProgressCallback,
        query: str = "",
    ) -> list[HarvestResult]:
        """Extract facts from all sources (harvest LLM, in parallel).

        Args:
            sources: sources to harvest
            plan: research plan (for the questions)
            progress_callback: progress callback
            query: the original research query — needed for hallucination
                   detection (person filter).

        Person anchor: `classify_query_anchor` (LLM classifier) decides
        whether the request is about a person; `PersonHallucinationFilter`
        (deterministic token match) is ONLY active if the classifier said
        'person' with confidence >= 0.7.

        The classifier is called exactly once per run (cached in
        `self._query_anchor`). If an earlier call already set the anchor
        (e.g. because the format agent evaluated it), it is not
        recomputed here.
        """
        # Lazy imports to avoid circular imports
        from src.llm.classifier_adapter import HarvestModelAdapter
        from src.pipeline.classifiers.query_anchor import (
            classify_query_anchor,
        )
        from src.pipeline.filters.person_match import (
            PersonHallucinationFilter,
        )
        from src.pipeline.filters.base import run_filter_pipeline

        # Format the research questions
        questions_text = "\n".join(
            f"{q.id}: {q.question}" for q in plan.questions
        )

        # ── Prepare the hallucination filter ──
        # The `classify_query_anchor` classifier decides based on meaning
        # whether the request has a person anchor.
        if self._query_anchor is None:
            classifier_llm = HarvestModelAdapter(self.llm)
            anchor, anchor_call = await classify_query_anchor(
                query=query, plan_summary=plan.summary or "",
                llm=classifier_llm,
            )
            self._query_anchor = anchor
            self._classifier_calls.append(anchor_call)
            # visibility in the UI
            await progress_callback("classifier", {
                "name": "query_anchor",
                "anchor_type": anchor.type,
                "target": anchor.target,
                "confidence": anchor.confidence,
                "reasoning": anchor.reasoning,
                "fallback_used": anchor.fallback_used,
            })
            logger.info(
                "🔍 Query-anchor classification: type=%s, target=%r, "
                "confidence=%.2f — %s",
                anchor.type, anchor.target, anchor.confidence,
                ("person hallucination filter is activated"
                 if anchor.is_person()
                 else "person hallucination filter NOT activated"),
            )
        else:
            anchor = self._query_anchor

        # The filter is active exactly when the classifier recognised a
        # person with sufficient confidence. The threshold (0.7) is in
        # `is_person()`. With an uncertain classification the filter does
        # NOT run, because a filter that does not run does less harm than
        # a false positive.
        person_filter_should_run = anchor.is_person()

        # Adapter context for the filter pipeline. Filters expect a ctx
        # object with `query_anchor`, nothing else. We give them only what
        # they need — not the full HarvestContext. (In the DAG path the
        # filters use the real ctx; this is the path of the _run_harvest
        # helper.)
        class _FilterCtx:
            def __init__(self, anchor):
                self.query_anchor = anchor

        filter_ctx = _FilterCtx(anchor if person_filter_should_run else None)

        # Source-content lookup for the filter
        source_content_by_url = {s.url: s.content for s in sources}
        person_filter = PersonHallucinationFilter(
            source_content_lookup=source_content_by_url.get,
        )

        # Counters for the progress log
        hallucination_stats = {"rejected_extracts": 0, "rejected_sources": 0}

        async def harvest_one(source: SourceDocument) -> HarvestResult:
            # Limit the content (the harvest LLM has ~200k context)
            max_content = 80_000  # ~25k tokens, leaves room for prompt + output
            content = source.content[:max_content]
            if len(source.content) > max_content:
                content += "\n\n[... shortened]"

            prompt = HARVEST_PROMPT.format(
                date=get_date_text(),
                research_questions=questions_text,
                source_url=source.url,
                source_title=source.title,
                source_published_date=(
                    source.published_date if getattr(source, 'published_date', '')
                    else "(not given)"
                ),
                source_content=content,
            )
            prompt += prompt_language_line(self.output_language)

            try:
                response = await self.llm.harvest_complete(
                    [{"role": "user", "content": prompt}],
                    max_tokens=8192,
                )

                if "NO_RELEVANCE" in response or "KEINE_RELEVANZ" in response:
                    return HarvestResult(
                        source_url=source.url,
                        source_title=source.title,
                        is_relevant=False,
                        raw_response=response,
                    )

                # Polarity-aware parser (harvest_parser): it strips the
                # [POSITIVE/NEGATIVE/META] markers from the fact and sets the
                # polarity field, which the NEGATIVE/META filtering in
                # HarvestNode relies on.
                from src.pipeline.harvest_parser import (
                    parse_harvest_response_with_polarity,
                )
                extracts = parse_harvest_response_with_polarity(
                    response, source.url, source.title
                )

                await progress_callback("harvest_one", {
                    "url": source.url,
                    "extracts": len(extracts),
                })

                return HarvestResult(
                    source_url=source.url,
                    source_title=source.title,
                    extracts=extracts,
                    is_relevant=bool(extracts),
                    raw_response=response,
                )

            except Exception as e:
                logger.warning(f"Harvest failed for {source.url}: {e}")
                return HarvestResult(
                    source_url=source.url,
                    source_title=source.title,
                    is_relevant=False,
                    raw_response=f"ERROR: {e}",
                )

        # Run in parallel (the semaphore is in the DualLLMClient)
        tasks = [harvest_one(s) for s in sources]
        results = await asyncio.gather(*tasks)

        # ── Central filter pass ──
        # Collect all extracts from all sources, send them ONCE through the
        # filter pipeline, then write them back. Advantage over per-source
        # filters: filters see the aggregate loss rate, the coverage
        # classifier can later recognise `filter_blocked`, and the
        # statistics are consistent.
        all_extracts = [e for r in results for e in r.extracts]
        if all_extracts:
            kept_extracts, filter_stats_map = await run_filter_pipeline(
                items=all_extracts, ctx=filter_ctx, filters=[person_filter],
            )
            # Index of the surviving extracts for fast re-mapping
            kept_ids = {id(e) for e in kept_extracts}
            for r in results:
                r.extracts = [e for e in r.extracts if id(e) in kept_ids]
                r.is_relevant = bool(r.extracts)

            # Persist statistics for the diagnosis / coverage classifier
            self._filter_stats_per_round.append({
                k: v.to_dict() for k, v in filter_stats_map.items()
            })
            # statistics variable for the log output below
            ph_stats = filter_stats_map.get("person_hallucination")
            if ph_stats is not None and ph_stats.activated:
                hallucination_stats["rejected_extracts"] = ph_stats.rejected
                hallucination_stats["rejected_sources"] = sum(
                    1 for r in results if not r.extracts
                )
                # Visibility: explicit filter statistics
                await progress_callback("filter_stats", ph_stats.to_dict())

        relevant = sum(1 for r in results if r.is_relevant)
        total_extracts = sum(len(r.extracts) for r in results)
        logger.info(f"Harvest: {relevant}/{len(results)} relevant, "
                     f"{total_extracts} extracts")

        # Hallucination statistics as a progress event (for the UI)
        if anchor.is_person() and hallucination_stats["rejected_extracts"] > 0:
            n_ext = hallucination_stats["rejected_extracts"]
            n_src = hallucination_stats["rejected_sources"]
            logger.info(
                f"🛡️ Hallucination filter: {n_ext} extracts from "
                f"{n_src} sources discarded "
                f"(person match on '{anchor.target}' failed)"
            )
            await progress_callback("status", (
                tr("🛡️ Hallucination filter: {extracts} extracts from "
                   "{sources} sources discarded — the sources do not mention "
                   "“{target}”", extracts=n_ext, sources=n_src,
                   target=anchor.target)
            ))

        return list(results)


    # ═══════════════════════════════════════════════════════════════
    # Phase 3a: cross-check against the person directory
    # ═══════════════════════════════════════════════════════════════

    # Pattern for claims about persons: only names WITH an academic title or role.
    # In German all nouns are capitalised → a pure capitalisation heuristic
    # produces false positives ("Maschinelles Lernen", "Die Quelle").
    # Hence: only matches with an explicit title prefix or role description.

    # Pattern 1: academic title + name
    _TITLED_PERSON_RE = re.compile(
        r"(?:Prof\.?\s*(?:Dr\.?\s*(?:-Ing\.?\s*)?)?|Dr\.?\s*(?:-Ing\.?\s*)?|Jun\.-Prof\.?\s*)"
        r"\s*([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+){1,3})"
    )

    # Pattern 2: role + name ("Professor Given Family"); inline (?i:...)
    # only for the role words, the name stays case-sensitive
    _ROLE_PERSON_RE = re.compile(
        r"(?i:Profess(?:or|orin)|Dozent(?:in)?|Forscher(?:in)?|Mitarbeiter(?:in)?|"
        r"Leiter(?:in)?|Direktor(?:in)?|Lehrstuhlinhaber(?:in)?|"
        r"Wissenschaftler(?:in)?|Postdoc)"
        r"\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+){1,3})",
    )

    # Words that are definitely not family names
    _NOT_NAMES = {
        "berlin", "institut", "universität", "forschung", "informatik",
        "machine", "learning", "artificial", "intelligence",
        "natural", "language", "computer", "science",
        "data", "digital", "research", "center",
        "arbeitsgruppe", "visual", "computing", "systems",
        "engineering", "processing", "analytics", "theory",
        "mining", "management", "information", "knowledge",
        "quantum", "software", "distributed", "algorithmic",
        "bioinformatics", "computational", "regulatory",
        "education", "gesellschaft", "mathematik", "physik",
        "exzellenz", "cluster", "graduate", "school",
        "maschinelles", "lernen", "künstliche", "intelligenz",
        "erklärbare", "quelle", "analyse", "bericht",
        "principal", "investigator", "foundations",
    }

    # ═══════════════════════════════════════════════════════════════
    # Phase 2b: off-topic filter (classifier)
    # ═══════════════════════════════════════════════════════════════

    async def _filter_off_topic_sources(
        self,
        sources: list,
        plan,
        progress_callback,
    ) -> list:
        """Remove thematically unsuitable sources via `judge_source_relevance`.

        Complements the pattern heuristic `_is_listing_page`: what the
        heuristic misses, the classifier sorts out semantically here. Only
        the strong statement "off_topic with high confidence" leads to
        exclusion — `should_fetch()` of the `SourceRelevance` class
        encapsulates exactly that. Everything else (high/medium/low/
        uncertain off_topic) is kept.

        Safe defaults:
          - a use case can disable the classifier → the method returns
            the sources unchanged.
          - LLM failure per batch → the batched classifier fills in
            "medium" (= fetch), nothing is lost.
          - plan without questions → the classifier makes no sense →
            unchanged.

        Args:
            sources: list of SourceDocument
            plan: ResearchPlan or None
            progress_callback: for progress messages

        Returns:
            The filtered list; always ⊆ sources, in the same order.
        """
        from src.llm.classifier_adapter import HarvestModelAdapter
        from src.pipeline.classifiers.source_relevance import (
            judge_source_relevance,
        )

        if not sources:
            return sources
        if plan is None or not plan.questions:
            return sources

        # Input format for the classifier
        questions_payload = [
            {"id": q.id, "question": q.question} for q in plan.questions
        ]
        # Snippet from content[:500]
        results_payload = [
            {
                "title": s.title or "",
                "url": s.url,
                "snippet": (s.content or "")[:500],
            }
            for s in sources
        ]

        llm = HarvestModelAdapter(self.llm)
        try:
            judgments, calls = await judge_source_relevance(
                questions=questions_payload,
                search_results=results_payload,
                llm=llm,
            )
        except Exception as e:
            logger.warning(
                "Off-topic classifier failed: %s — "
                "keeping all sources", e,
            )
            return sources

        # Persist the calls for logging
        self._classifier_calls.extend(calls)

        # Apply the filter — only off_topic with high confidence is removed.
        # Positional mapping: judgments[i] belongs to sources[i] (not the
        # index returned by the LLM). NO zip() — with a judgments list that
        # is too short it would silently drop the remaining sources. If a
        # verdict is missing, the source is kept (fail-open, consistent with
        # this method's safe defaults: when in doubt, lose nothing).
        if len(judgments) != len(sources):
            logger.warning(
                "Off-topic filter: %d verdicts for %d sources — "
                "missing ones are kept (fail-open)",
                len(judgments), len(sources),
            )
        kept: list = []
        rejected_count = 0
        for i, src in enumerate(sources):
            judgment = judgments[i] if i < len(judgments) else None
            if judgment is None or judgment.should_fetch():
                kept.append(src)
            else:
                rejected_count += 1
                logger.info(
                    "🎯 Off-topic filter (classifier): %s "
                    "(rel=%s, conf=%.2f) — %s",
                    src.url, judgment.relevance, judgment.confidence,
                    (judgment.reasoning or "")[:80],
                )

        if rejected_count:
            logger.info(
                "🎯 Off-topic filter: %d of %d sources excluded "
                "(classifier path)",
                rejected_count, len(sources),
            )
            try:
                await progress_callback("status", (
                    tr("🎯 Off-topic filter: {n} thematically "
                       "unsuitable source(s) skipped", n=rejected_count)
                ))
            except Exception:
                pass  # progress failures must not stop the pipeline

        return kept

    # ═══════════════════════════════════════════════════════════════
    # (The annotation helper is defined as a module function at the end
    # of the file: `_annotate_unverified_factoids`. It is a pure string
    # operation and does not belong in the class behaviour.)
    # ═══════════════════════════════════════════════════════════════



    # ═══════════════════════════════════════════════════════════════
    # OpenAlex author search (institution mode)
    # ═══════════════════════════════════════════════════════════════

    async def _query_openalex_author(
        self, person_name: str, orcid: str = "",
    ) -> SourceDocument | None:
        """Look up publications of a person of the institution via OpenAlex.

        Uses the ORCID (if present) or the name + the institution's
        affiliation (the profile's `openalex_id`). Returns None if no
        OpenAlex ID is configured.
        """
        import httpx

        INSTITUTION_OPENALEX_ID = get_profile().openalex_id
        if not INSTITUTION_OPENALEX_ID:
            return None
        base = "https://api.openalex.org"
        headers = {"User-Agent": polite_user_agent()}

        async with httpx.AsyncClient(
            timeout=15.0, headers=headers,
        ) as client:
            # 1. Find the author
            author_id = None
            author_data = None

            # Attempt 1: directly via ORCID (the most precise way)
            if orcid:
                clean_orcid = orcid.replace("https://orcid.org/", "")
                try:
                    resp = await client.get(
                        f"{base}/authors/orcid:{clean_orcid}"
                    )
                    if resp.status_code == 200:
                        author_data = resp.json()
                        author_id = author_data.get("id")
                        logger.debug(
                            f"OpenAlex: {person_name} found via ORCID"
                        )
                except Exception:
                    pass

            # Attempt 2: name + institution affiliation
            if not author_id:
                try:
                    resp = await client.get(
                        f"{base}/authors",
                        params={
                            "search": person_name,
                            "filter": (
                                f"affiliations.institution.id:"
                                f"{INSTITUTION_OPENALEX_ID}"
                            ),
                        },
                    )
                    if resp.status_code == 200:
                        results = resp.json().get("results", [])
                        if results:
                            author_data = results[0]
                            author_id = author_data.get("id")
                            logger.debug(
                                f"OpenAlex: {person_name} found via "
                                f"name + affiliation"
                            )
                except Exception:
                    pass

            if not author_id:
                return None

            # 2. Load the most recent works (max 10)
            try:
                resp = await client.get(
                    f"{base}/works",
                    params={
                        "filter": f"authorships.author.id:{author_id}",
                        "sort": "publication_date:desc",
                        "per_page": "10",
                    },
                )
                if resp.status_code != 200:
                    return None
                works = resp.json().get("results", [])
            except Exception:
                return None

            if not works:
                return None

            # 3. Create the Markdown document
            lines = [f"## OpenAlex: publications of {person_name}\n"]

            # Author statistics (if present)
            if author_data:
                works_count = author_data.get("works_count", 0)
                cited_count = author_data.get("cited_by_count", 0)
                h_index = (
                    author_data.get("summary_stats", {})
                    .get("h_index", 0)
                )
                if works_count or cited_count:
                    lines.append(
                        f"**Total:** {works_count} publications, "
                        f"{cited_count} citations"
                        + (f", h-Index: {h_index}" if h_index else "")
                        + "\n"
                    )

            lines.append("### Most recent publications\n")
            for w in works:
                title = w.get("title", "Untitled")
                year = w.get("publication_year", "?")
                doi = w.get("doi", "")
                cited = w.get("cited_by_count", 0)
                venue = (
                    w.get("primary_location", {})
                    .get("source", {})
                    .get("display_name", "")
                )
                oa_type = w.get("type", "")

                line = f"- **{title}** ({year})"
                if venue:
                    line += f"\n  {venue}"
                if doi:
                    line += f"\n  DOI: {doi}"
                details = []
                if cited:
                    details.append(f"cited: {cited}×")
                if oa_type:
                    details.append(oa_type)
                if details:
                    line += f"\n  {', '.join(details)}"
                lines.append(line)

            # URL for the source reference
            oa_url = (
                author_id.replace(
                    "https://openalex.org/", "https://openalex.org/authors/"
                )
                if author_id and "openalex.org" in author_id
                else f"https://openalex.org/authors?search={person_name}"
            )

            logger.info(
                f"OpenAlex: {person_name} — {len(works)} publications"
            )

            return SourceDocument(
                url=oa_url,
                title=(
                    f"OpenAlex: {person_name} — "
                    f"{len(works)} publications"
                ),
                content="\n".join(lines),
                source_type=SourceType.WEB_PAGE,
            )

    async def _query_openalex_topic(
        self, topic: str,
    ) -> SourceDocument | None:
        """Look up researchers of the institution on a topic via OpenAlex.

        Finds the most-cited works by members of the institution on a
        topic and extracts the authors involved. Returns None if no
        OpenAlex ID is configured.
        """
        import httpx

        INSTITUTION_OPENALEX_ID = get_profile().openalex_id
        if not INSTITUTION_OPENALEX_ID:
            return None
        base = "https://api.openalex.org"
        headers = {"User-Agent": polite_user_agent()}

        try:
            async with httpx.AsyncClient(
                timeout=15.0, headers=headers,
            ) as client:
                resp = await client.get(
                    f"{base}/works",
                    params={
                        "search": topic,
                        "filter": (
                            f"authorships.institutions.lineage:"
                            f"{INSTITUTION_OPENALEX_ID}"
                        ),
                        "sort": "cited_by_count:desc",
                        "per_page": "15",
                    },
                )
                if resp.status_code != 200:
                    return None

                works = resp.json().get("results", [])
                if not works:
                    return None

                # Extract authors and their works
                lines = [
                    f"## OpenAlex: research at the institution on {topic}\n",
                    f"**{len(works)} relevant publications found**\n",
                ]

                # Collect authors (who researches this topic?)
                institution_authors: dict[str, int] = {}
                for w in works:
                    for authorship in w.get("authorships", []):
                        insts = [
                            i.get("id", "")
                            for i in authorship.get("institutions", [])
                        ]
                        if any(INSTITUTION_OPENALEX_ID in i for i in insts):
                            name = (
                                authorship.get("author", {})
                                .get("display_name", "")
                            )
                            if name:
                                institution_authors[name] = (
                                    institution_authors.get(name, 0) + 1
                                )

                if institution_authors:
                    top = sorted(
                        institution_authors.items(), key=lambda x: -x[1]
                    )[:10]
                    lines.append(f"### Researchers at {get_profile().label} on this topic\n")
                    for name, count in top:
                        lines.append(f"- **{name}** ({count} publications)")
                    lines.append("")

                lines.append("### Top publications\n")
                for w in works[:10]:
                    title = w.get("title", "?")
                    year = w.get("publication_year", "?")
                    cited = w.get("cited_by_count", 0)
                    doi = w.get("doi", "")
                    line = f"- **{title}** ({year}, cited {cited}×)"
                    if doi:
                        line += f"\n  DOI: {doi}"
                    lines.append(line)

                logger.info(
                    f"OpenAlex topic: '{topic}' — "
                    f"{len(works)} works, {len(institution_authors)} institution authors"
                )

                return SourceDocument(
                    url=(
                        f"https://openalex.org/works?"
                        f"search={topic}&filter=institution:{INSTITUTION_OPENALEX_ID}"
                    ),
                    title=(
                        f"OpenAlex: research on {topic} — "
                        f"{len(institution_authors)} researchers"
                    ),
                    content="\n".join(lines),
                    source_type=SourceType.WEB_PAGE,
                )

        except Exception as e:
            logger.debug(f"OpenAlex Topic-Search '{topic}': {e}")
            return None

    # ═══════════════════════════════════════════════════════════════
    # Phase 3c: contradiction detection
    # ═══════════════════════════════════════════════════════════════

    async def _check_contradictions(
        self, ctx: HarvestContext,
    ) -> list[dict]:
        """Check extracts for contradictions (harvest LLM).

        Groups extracts per question and lets the LLM recognise
        contradictions in content.

        Before the LLM call, purely negative extracts are filtered out by
        reading the `polarity` field set by the harvest LLM: statements
        such as "the source contains no information about X" are not
        positive statements about X and must not create contradictions
        with other sources. META extracts (statements about the source
        situation) are filtered out here as well.
        """
        from src.prompts import CONTRADICTION_PROMPT

        all_contradictions = []
        questions = ctx.research_plan.questions if ctx.research_plan else []

        # Only check questions with >2 POSITIVE extracts
        check_questions = []
        filter_stats = {"total": 0, "non_positive_removed": 0}
        for q in questions:
            relevant = [e for e in ctx.extracts if e.question_id == q.id]
            filter_stats["total"] += len(relevant)
            # Filter by polarity: only "positive" extracts go into the
            # contradiction check. "negative" and "meta" are left out.
            positive = [
                e for e in relevant
                if getattr(e, "polarity", "positive") == "positive"
            ]
            filter_stats["non_positive_removed"] += (
                len(relevant) - len(positive)
            )
            if len(positive) >= 3:
                check_questions.append((q, positive))

        if filter_stats["non_positive_removed"] > 0:
            logger.info(
                f"Contradiction prefilter: {filter_stats['non_positive_removed']} "
                f"of {filter_stats['total']} extracts filtered out as purely negative/meta "
                f"(polarity-based)"
            )

        if not check_questions:
            return []

        # Check in parallel (at most 3 at a time)
        async def check_one(q, extracts):
            extracts_text = ""
            for i, e in enumerate(extracts[:15], 1):  # at most 15 per question
                src = e.source_title or e.source_url or "?"
                extracts_text += (
                    f"[{i}] {e.fact}\n"
                    f"    Source: {src}\n"
                    f"    Reliability: {e.reliability}\n\n"
                )

            prompt = CONTRADICTION_PROMPT.format(
                question=q.question,
                extracts=extracts_text,
            )

            try:
                result = await self.llm.harvest_complete(
                    [{"role": "user", "content": prompt}],
                    max_tokens=2048,
                )
                # Parse JSON
                result = result.strip()
                if result.startswith("```"):
                    result = result.split("\n", 1)[-1].rsplit("```", 1)[0]
                data = json.loads(result)
                return data.get("contradictions", [])
            except Exception as e:
                logger.debug(f"Contradiction check {q.id} failed: {e}")
                return []

        tasks = [check_one(q, exts) for q, exts in check_questions[:5]]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                all_contradictions.extend(result)

        return all_contradictions

    # ═══════════════════════════════════════════════════════════════
    # Phase 4: synthesis
    # ═══════════════════════════════════════════════════════════════

    async def _run_synthesis(
        self, ctx: HarvestContext, context_docs: str,
        progress_callback: ProgressCallback,
    ) -> str:
        """Create the report (primary LLM, streaming)."""
        schema = ctx.output_schema or OutputSchema()

        # Helper: text similarity for de-duplication
        def _normalize_for_dedup(text: str) -> str:
            """Normalise text for similarity comparison."""
            t = text.lower().strip()
            t = re.sub(r'[^\w\s]', '', t)  # remove punctuation
            t = re.sub(r'\s+', ' ', t)      # normalise whitespace
            return t[:200]  # compare only the beginning

        # Sort by reliability (high first)
        reliability_order = {"high": 0, "medium": 1, "low": 2}

        # Build the embedder once per synthesis. Fail-open: unavailable
        # ⇒ word-overlap heuristic.
        from src.llm.embedder import (
            build_embedder_from_pipeline_config, cosine,
        )
        _embedder = build_embedder_from_pipeline_config(self.config)
        _dedup_thr = getattr(
            self.config, "extract_dedup_threshold", 0.93
        )
        # Reranker for the map phase: before the 80k-character cap, order
        # a question's extracts by relevance TO THE QUESTION, not just by
        # reliability. That way the facts most important for the question
        # survive the cap, not merely the "most reliable" ones. Fail-open.
        from src.llm.reranker import build_reranker_from_pipeline_config
        _reranker = build_reranker_from_pipeline_config(self.config)

        def _dedup_wordoverlap(relevant: list) -> list:
            """Heuristic: >0.7 word overlap ⇒ duplicate."""
            seen_facts: list[str] = []
            deduped = []
            for e in relevant:
                norm = _normalize_for_dedup(e.fact)
                is_dup = False
                for seen in seen_facts:
                    words_new = set(norm.split())
                    words_seen = set(seen.split())
                    if not words_new or not words_seen:
                        continue
                    overlap = len(words_new & words_seen) / min(
                        len(words_new), len(words_seen)
                    )
                    if overlap > 0.7:
                        is_dup = True
                        break
                if not is_dup:
                    seen_facts.append(norm)
                    deduped.append(e)
            return deduped

        async def _dedup_extracts(relevant: list) -> list:
            """Embedding-based de-duplication (cosine ≥ threshold).
            Recognises paraphrased syndication that word overlap misses.
            `relevant` is pre-sorted by reliability — so for a duplicate
            the more reliable version is kept. Fail-open: embedder
            unavailable/error ⇒ word overlap."""
            if len(relevant) <= 1 or not _embedder.available:
                return _dedup_wordoverlap(relevant)
            vectors = await _embedder.embed(
                [(e.fact or "")[:512] for e in relevant]
            )
            if vectors is None:
                return _dedup_wordoverlap(relevant)
            kept: list = []
            kept_vecs: list = []
            for e, v in zip(relevant, vectors):
                if any(cosine(v, kv) >= _dedup_thr for kv in kept_vecs):
                    continue
                kept.append(e)
                kept_vecs.append(v)
            return kept

        def _format_extract_block(extracts_for_q: list) -> str:
            """Format a list of extracts as a Markdown block for the map
            prompt. Called once per question."""
            block = ""
            for e in extracts_for_q:
                src_url = e.source_url.strip().rstrip("]}).,:;'\"") if e.source_url else ""
                src_title = (e.source_title or src_url)[:60]
                src_link = f"[{src_title}]({src_url})" if src_url else src_title
                block += (
                    f"- **{e.fact}**\n"
                    f"  Source: {src_link}\n"
                    f"  Reliability: {e.reliability}\n"
                )
                if e.context:
                    block += f"  Context: {e.context}\n"
            return block

        # Token budget per map call. With reasoning models it must cover the
        # reasoning phase TOO (see the comment at the call below), but must
        # not exceed the configured output limit.
        map_budget = min(16384, _primary_output_limit(self.llm))

        async def _summarize_question(q) -> tuple[str, str]:
            """Map phase: condense a question's extracts into a coherent
            answer. Returns (q.id, answer_markdown).

            On error: fall back to the raw list of extracts, so that the
            report can still be created (defence in depth — one hanging
            question must not prevent the whole report).
            """
            relevant = [e for e in ctx.extracts if e.question_id == q.id]
            if not relevant:
                return q.id, ""

            relevant.sort(
                key=lambda e: reliability_order.get(e.reliability, 1)
            )

            # De-duplicate: embedding cosine (fallback: word overlap)
            deduped = await _dedup_extracts(relevant)

            if len(relevant) > len(deduped):
                logger.debug(
                    f"Extract dedup {q.id}: {len(relevant)} → {len(deduped)} "
                    f"({len(relevant) - len(deduped)} duplicates)"
                )

            # Relevance reranking against the question: determines which
            # extracts survive the 80k cap. Only worthwhile if capping would
            # happen anyway (many extracts). Fail-open: order unchanged.
            if _reranker.available and len(deduped) > 8:
                try:
                    def _ext_text(e) -> str:
                        return (
                            f"{e.fact or ''} {e.context or ''}"
                        )[:1200]

                    deduped = await _reranker.order(
                        q.question, deduped, key=_ext_text,
                    )
                    logger.debug(
                        "Map rerank %s: %d extracts ordered by relevance"
                        "", q.id, len(deduped),
                    )
                except Exception as e:
                    logger.warning(
                        "Map rerank %s skipped (%s)",
                        q.id, type(e).__name__,
                    )

            extracts_block = _format_extract_block(deduped)

            # Safety cap: with extremely many extracts for one question
            # (>500 characters per extract × 200 extracts = 100k characters)
            # shorten further, so that the map call itself does not overflow the
            # context. 80k characters ≈ 20k tokens — fits every realistic model context.
            extracts_block = extracts_block[:80_000]

            map_prompt = QUESTION_ANSWER_PROMPT.format(
                question_id=q.id,
                question_text=q.question,
                extracts=extracts_block,
            )
            map_prompt += prompt_language_line(self.output_language)

            try:
                # The map call needs a budget that covers reasoning AND the
                # answer. With thinking mode on and a small max_tokens, the
                # reasoning uses up the whole budget → empty visible answer
                # → ValueError → most questions fall back to raw extracts →
                # a poor report.
                #
                # `enable_thinking=False` alone is NOT enough: the switch is
                # sent as `chat_template_kwargs` — a convention of the Qwen
                # family. Kimi does not know it and reasons anyway. The
                # switch stays (it helps with other models), but the actual
                # safeguard is the larger budget: it must carry reasoning
                # AND the answer.
                answer = await asyncio.wait_for(
                    self.llm.primary_complete(
                        [{"role": "user", "content": map_prompt}],
                        max_tokens=map_budget,
                        enable_thinking=False,
                    ),
                    timeout=300,  # at most 5 minutes per question
                )
                answer = (answer or "").strip()
                if not answer:
                    raise ValueError("LLM returned an empty answer")
                logger.info(
                    "Synthesis map %s: %d extracts → %d characters of answer",
                    q.id, len(deduped), len(answer),
                )
                return q.id, answer
            except (asyncio.TimeoutError, Exception) as e:
                # One retry before falling back to raw extracts. LLM
                # gateways return transient 504s; a single retry rescues
                # most of them without noticeably extending the map phase.
                logger.warning(
                    "Synthesis map %s: first attempt failed (%s) "
                    "— one retry", q.id, type(e).__name__,
                )
                try:
                    answer = await asyncio.wait_for(
                        self.llm.primary_complete(
                            [{"role": "user", "content": map_prompt}],
                            max_tokens=map_budget,
                            enable_thinking=False,
                        ),
                        timeout=300,
                    )
                    answer = (answer or "").strip()
                    if answer:
                        logger.info(
                            "Synthesis map %s (retry): %d characters of answer",
                            q.id, len(answer),
                        )
                        return q.id, answer
                except (asyncio.TimeoutError, Exception) as e2:
                    e = e2
                # Fallback: raw extracts as the "answer" — the reduce call
                # then does the condensing at the end.
                logger.warning(
                    "Synthesis map %s failed permanently (%s) — "
                    "using raw extracts as the fallback",
                    q.id, type(e).__name__,
                )
                return q.id, extracts_block

        # ── Map phase: in parallel per question ──
        plan_questions = (
            ctx.research_plan.questions if ctx.research_plan else []
        )
        if plan_questions:
            await progress_callback(
                "report_stream",
                tr("⏳ *Condensing extracts into {n} question answers "
                   "({extracts} extracts in total)...*",
                   n=len(plan_questions), extracts=len(ctx.extracts)),
            )
            map_results = await asyncio.gather(
                *[_summarize_question(q) for q in plan_questions]
            )
        else:
            map_results = []

        # Assemble the reduce input: question answers as Markdown
        extracts_text = ""
        # Also keep the map answers on ctx, so that the pipeline-run tab
        # can show them.
        ctx.map_answers = []
        for q in plan_questions:
            qid = q.id
            answer = next((a for (i, a) in map_results if i == qid), "")
            if not answer:
                continue
            extracts_text += f"\n### {qid}: {q.question}\n\n{answer}\n"
            n_ext = sum(1 for e in ctx.extracts if e.question_id == qid)
            ctx.map_answers.append({
                "question_id": qid,
                "question": q.question,
                "answer": answer,
                "n_extracts": n_ext,
            })

        # ── Orphan extracts: catch facts not assigned to a question ──
        plan_qids = {
            q.id for q in (ctx.research_plan.questions if ctx.research_plan else [])
        }
        matched_count = sum(
            1 for e in ctx.extracts if e.question_id in plan_qids
        )
        orphans = [e for e in ctx.extracts if e.question_id not in plan_qids]

        if orphans:
            logger.info(
                f"Synthesis: {matched_count} extracts assigned, "
                f"{len(orphans)} orphan extracts "
                f"(IDs: {', '.join(sorted(set(e.question_id for e in orphans)))})"
            )
            extracts_text += "\n### Further results (not assigned to a question)\n"
            for e in orphans[:30]:
                src_url = e.source_url.strip().rstrip("]}).,:;'\"") if e.source_url else ""
                src_title = (e.source_title or src_url)[:60]
                src_link = f"[{src_title}]({src_url})" if src_url else src_title
                extracts_text += (
                    f"- **{e.fact}**\n"
                    f"  Source: {src_link}\n"
                    f"  (original ID: {e.question_id})\n"
                )
        else:
            logger.info(
                f"Synthesis: {matched_count}/{len(ctx.extracts)} extracts "
                f"assigned, no orphans"
            )

        # Append person-directory verification warnings to the extracts
        if ctx.directory_verification_warnings:
            extracts_text += "\n\n### ⚠️ VERIFICATION NOTES (person directory)\n"
            extracts_text += (
                "The following claims about people could not be confirmed in "
                "the person directory. Mark this "
                "information in the report as NOT VERIFIED, but do not "
                "leave it out completely:\n\n"
            )
            for w in ctx.directory_verification_warnings[:30]:
                extracts_text += (
                    f"- **{w['name']}**: {w['directory_result']}\n"
                    f"  Original claim: {w['claim'][:120]}\n"
                )

        # Highlight contradictions between extracts
        if getattr(ctx, 'contradiction_warnings', None):
            extracts_text += "\n\n### ⚠️ CONTRADICTIONS BETWEEN SOURCES\n"
            extracts_text += (
                "The following contradictions were found between different "
                "sources. Present them transparently in the report "
                "and name both sources:\n\n"
            )
            for c in ctx.contradiction_warnings[:10]:
                fact_a = c.get('fact_a', '?')
                fact_b = c.get('fact_b', '?')
                extracts_text += (
                    f"- **Contradiction:** {c.get('nature', '?')}\n"
                    f"  Source A: {c.get('source_a', '?')} — {fact_a}\n"
                    f"  Source B: {c.get('source_b', '?')} — {fact_b}\n\n"
                )

        sections_text = "\n".join(
            f"- {s.get('id', '')}: {s.get('title', '')} — {s.get('description', '')}"
            for s in schema.sections
        ) or "Derive automatically from the request"

        per_section_text = ", ".join(schema.per_section_fields) or "Choose automatically"

        # Synthesis guidance from the schema (set by the format agent)
        guidance = schema.synthesis_guidance
        if not guidance:
            # Fallback: Default-Guidance
            guidance = OUTPUT_TEMPLATES.get(
                DEFAULT_TEMPLATE, {}
            ).get("synthesis_guidance", "Write a readable report.")

        prompt = SYNTHESIS_PROMPT.format(
            query=ctx.query,
            date=get_date_text(),
            title=schema.title,
            format_type=schema.format_type,
            style=schema.style,
            language=llm_language_name(self.output_language),
            sections=sections_text,
            per_section_fields=per_section_text,
            synthesis_guidance=guidance,
            extracts=extracts_text[:200000],
            context_docs=context_docs[:30000] if context_docs else "None",
        )
        prompt += prompt_language_line(self.output_language)

        # Streaming — generate the report.
        # The budget covers reasoning AND the answer: with a reasoning model
        # (Kimi) the reasoning tokens count against max_tokens but cannot be
        # switched off. A fixed 16384 tokens can be used up entirely by the
        # reasoning, so that the answer never starts and the report consists
        # only of header and footer. Hence the full configured output limit
        # instead of a fixed value.
        reduce_budget = max(16384, _primary_output_limit(self.llm))
        report_msgs = [{"role": "user", "content": prompt}]

        full_report = ""
        synthesis_degraded = ""
        try:
            async for chunk in self.llm.primary_stream(
                report_msgs, max_tokens=reduce_budget,
            ):
                full_report = chunk
                await progress_callback("report_stream", chunk)
            full_report = _strip_thinking_placeholder(full_report)
        except EmptyLLMResponseError as e:
            logger.warning("Synthesis reduce: stream without text (%s)", e)
            full_report = ""
        except Exception as e:
            logger.error("Synthesis reduce: stream error %s: %s",
                         type(e).__name__, e)
            full_report = ""

        # ── Fallback 1: repeat without streaming ──
        # A second attempt is worthwhile because the reasoning budget is
        # not used up deterministically (retries often succeed).
        if not full_report.strip():
            logger.warning(
                "Synthesis reduce empty — one retry without streaming"
            )
            await progress_callback(
                "report_stream",
                tr("⏳ *The synthesis returned no text — trying again...*"),
            )
            try:
                full_report = (await asyncio.wait_for(
                    self.llm.primary_complete(
                        report_msgs, max_tokens=reduce_budget,
                    ),
                    timeout=600,
                ) or "").strip()
            except Exception as e:
                logger.error(
                    "Synthesis reduce: retry failed (%s)",
                    type(e).__name__,
                )
                full_report = ""

        # ── Fallback 2: build from the map answers ──
        # The condensed question answers already exist. Delivering them as
        # the report is much better than header + footer without content
        # while thousands of characters of finished answers sit in memory.
        if not full_report.strip():
            fallback_body = _report_from_map_answers(
                getattr(ctx, "map_answers", []) or []
            )
            if fallback_body:
                logger.error(
                    "Synthesis reduce failed permanently — report "
                    "assembled from %d map answers",
                    len(ctx.map_answers),
                )
                synthesis_degraded = (
                    catalog_t("report.degraded_from_map", ctx.output_language)
                )
                full_report = fallback_body
            else:
                logger.error(
                    "Synthesis reduce failed and no map answers "
                    "available as a fallback — the report stays empty"
                )
                synthesis_degraded = (
                    catalog_t("report.no_report", ctx.output_language)
                )

        ctx.synthesis_degraded = synthesis_degraded

        # ── Report header: ALWAYS put title + request first ──
        # (The synthesis output often starts with "# Title" — remove it so
        #  that there is no duplicate title)
        report_body = full_report
        if report_body.startswith("# "):
            # Remove the first heading (replaced by our header)
            first_newline = report_body.find("\n")
            if first_newline > 0:
                report_body = report_body[first_newline:].lstrip("\n")

        header = f"# {schema.title}\n\n"
        header += catalog_t("report.request", ctx.output_language, query=ctx.query)
        if synthesis_degraded:
            header += (
                catalog_t("report.degraded_banner", ctx.output_language, reason=synthesis_degraded)
            )

        # ── Generate the executive summary ──
        summary = ""
        if full_report and len(full_report) > 500:
            try:
                await progress_callback("report_stream",
                                        full_report + "\n\n⏳ *Writing the summary...*")

                summary_prompt = SUMMARY_PROMPT.format(
                    query=ctx.query,
                    report=full_report[:40000],
                    language=llm_language_name(self.output_language),
                )
                summary_prompt += prompt_language_line(self.output_language)
                # Primary LLM but WITHOUT thinking (otherwise endless
                # think tokens for a 3-sentence summary). The switch only
                # works for models with `enable_thinking` in the chat
                # template — Kimi ignores it. Hence an additional budget
                # for the reasoning phase: with 1024 tokens the summary
                # would practically always be empty with a reasoning model.
                summary = await asyncio.wait_for(
                    self.llm.primary_complete(
                        [{"role": "user", "content": summary_prompt}],
                        max_tokens=8192,
                        enable_thinking=False,
                    ),
                    timeout=180,  # at most 3 minutes
                )
                summary = summary.strip()
                if summary and len(summary) > 50:
                    logger.info(
                        f"Executive summary generated "
                        f"({len(summary)} characters)"
                    )
                else:
                    summary = ""
            except asyncio.TimeoutError:
                logger.warning("Summary generation: timeout after 120s")
            except Exception as e:
                logger.warning(f"Summary generation failed: {e}")

        # Assemble the header
        if summary:
            header += (
                catalog_t("report.summary", ctx.output_language, summary=summary)
            )
        else:
            header += "---\n\n"

        # ── LLM metadata footer (transparency) ──
        usage = self.llm.get_usage_stats()
        p_model = usage.get("primary", {}).get("model", "?")
        p_reqs = usage.get("primary", {}).get("requests", 0)
        p_tokens = usage.get("primary", {}).get("total_tokens", 0)
        h_model = usage.get("harvest", {}).get("model", "?")
        h_reqs = usage.get("harvest", {}).get("requests", 0)
        h_tokens = usage.get("harvest", {}).get("total_tokens", 0)

        # Shorten model names (only the last part after /)
        p_short = p_model.rsplit("/", 1)[-1] if "/" in p_model else p_model
        h_short = h_model.rsplit("/", 1)[-1] if "/" in h_model else h_model

        footer = (
            catalog_t("report.footer_line", ctx.output_language, tool=TOOL_NAME, p_model=p_short, p_reqs=p_reqs, p_tokens=p_tokens // 1000, h_model=h_short, h_reqs=h_reqs, h_tokens=h_tokens // 1000, sources=len(ctx.sources), extracts=len(ctx.extracts))
        )

        full_report = header + report_body + footer
        await progress_callback("report_stream", full_report)

        return full_report


# ═══════════════════════════════════════════════════════════════════
# Module functions (outside the class)
# ═══════════════════════════════════════════════════════════════════


def _primary_output_limit(llm, default: int = 16384) -> int:
    """Read the primary LLM's output token limit defensively.

    Test doubles and slim LLM wrappers do not necessarily have a
    `.config.primary` — accessing it directly would break the synthesis
    nodes in tests. Without the configuration, the default applies.
    """
    cfg = getattr(getattr(llm, "config", None), "primary", None)
    try:
        limit = int(getattr(cfg, "max_output_tokens", default) or default)
    except (TypeError, ValueError):
        return default
    return limit if limit > 0 else default


def _strip_thinking_placeholder(text: str) -> str:
    """Remove the thinking placeholder from a streaming result.

    With an unclosed <think> block the stream emits the placeholder
    (THINKING_PLACEHOLDER) so that the UI does not freeze. The caller
    always takes the last chunk seen — if no real text follows, the
    placeholder would be the "report". This function guards against that.
    """
    if not text:
        return ""
    return _PLACEHOLDER_RE.sub("", text).strip()


def _report_from_map_answers(map_answers: list[dict]) -> str:
    """Build an emergency report from the map answers.

    The map phase has already produced a condensed, referenced answer per
    research question. If the reduce phase fails, that is the best
    content available — unedited, but fully referenced and far better
    than an empty report.
    """
    sections = []
    for a in map_answers:
        answer = (a.get("answer") or "").strip()
        if not answer:
            continue
        question = (a.get("question") or a.get("question_id") or "").strip()
        sections.append(f"## {question}\n\n{answer}")
    return "\n\n".join(sections)


def _annotate_unverified_factoids(
    report: str, verifications: list[dict], lang: str | None = None,
) -> str:
    """Append a discreet notice block for UNCERTAIN contradicted
    statements at the end of the report.

    Division of labour with `ReportRevisionNode`:

    - **High-confidence contradictions (≥0.9)** are NOT handled here.
      They go to `ReportRevisionNode`, which has the report corrected
      semantically — no substring heuristics.

    - **Low-confidence contradictions (<0.9)** go into a notice block
      at the end. With low confidence an automatic correction would be
      risky (the verifier could be wrong itself), hence only a notice
      for human review.

    If there are no contradictions or only high-confidence ones, the
    report stays unchanged. The function is deliberately a module
    function (not a class method): a pure string operation, testable on
    its own.
    """
    low_conf_unverified = [
        v for v in verifications
        if v.get("verified") == "false" and v.get("confidence", 0) < 0.9
    ]
    if not low_conf_unverified:
        return report

    block = (
        catalog_t("report.uncertain_block", lang)
    )
    for v in low_conf_unverified:
        block += (
            catalog_t("report.uncertain_item", lang, factoid=v.get('factoid', '?'), type=v.get('type', '?'), confidence=f"{v.get('confidence', 0):.2f}")
        )
        if v.get("supporting_quote"):
            block += catalog_t("report.uncertain_quote", lang, quote=v['supporting_quote'])
    return report + block
