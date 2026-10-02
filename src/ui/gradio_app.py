"""
Gradio interface of the research tool.
Three-column layout: sidebar | chat + input | result panel
"""

import asyncio
import copy
import json
import logging
import os
import re
import tempfile
import uuid
from dataclasses import dataclass, field
from html import escape as html_escape
from datetime import datetime
from pathlib import Path
from typing import Optional

import gradio as gr

from src.config import AppConfig, DOCUMENT_EXTENSIONS, PLAIN_TEXT_EXTENSIONS
from src.ui.chat_state import ChatState
from src.ui.filter_stats_banner import format_filter_stats_banner
from src.llm.client import DualLLMClient
from src.connectors.base import ConnectorRegistry
from src.connectors.searxng import SearXNGConnector
from src.connectors.web_scraper import WebScraperConnector
from src.connectors.github import GitHubConnector
from src.connectors.gitlab import GitLabConnector
from src.connectors.local_files import LocalFileConnector
from src.pipeline.orchestrator import ResearchOrchestrator
from src.pipeline.models import (
    HarvestContext, OutputSchema, DEFAULT_TEMPLATE, template_choices,
)
from src.prompts import SYSTEM_PROMPT_CHAT, get_date_text
from src.prompts.research import render_chat_system_prompt
from src.institution import get_profile
from src.ui.chat_actions import js_replace_rules
from src.llm.client import THINKING_PLACEHOLDER
from src.about import TOOL_NAME
from src.pipeline.analysis_pipeline import EXPLAINER_LENGTHS, EXPLAINER_PURPOSES
from src.output_language import default_language, language_choices

logger = logging.getLogger(__name__)


# ─── Analysis use cases: module constants ──────────────────────────
# Mapping mode ID → use_case name (for the generic analysis modes).
# Used both when building the UI and in run_research(), hence defined
# at module level. The mode constants come from `src.ui.research_modes`
# (single source of truth); they must not be written out a second
# time here.
from src.ui.research_modes import (  # noqa: E402
    ANALYSIS_MODE_MAP,
    ANALYSIS_USE_CASE_ORDER,
    DEFAULT_RESEARCH_MODE,
    mode_choices,
    resolve_research_route,
)


def _split_analysis_args(analysis_args: tuple) -> dict[str, list]:
    """Split the flat analysis_args tuple into {use_case: [values]}.

    Reads the order from the registered PreflightCheckers and slices the
    tuple accordingly.
    """
    from src.pipeline.analysis_pipeline import USE_CASE_REGISTRY

    result = {}
    offset = 0
    for uc_name in ANALYSIS_USE_CASE_ORDER:
        if uc_name not in USE_CASE_REGISTRY:
            continue
        checker = USE_CASE_REGISTRY[uc_name]["preflight"]
        n_fields = len(checker.get_requirements())
        result[uc_name] = list(analysis_args[offset:offset + n_fields])
        offset += n_fields
    return result


# =====================================================================
# Application State
# =====================================================================

def _default_output_language() -> str:
    from src.output_language import default_language
    return default_language()


@dataclass
class AppState:
    config: AppConfig = field(default_factory=AppConfig)
    llm: DualLLMClient = None
    connectors: ConnectorRegistry = None
    orchestrator: ResearchOrchestrator = None
    # Chat
    chat_history: list = field(default_factory=list)
    # Documents
    documents: dict = field(default_factory=dict)
    total_doc_tokens: int = 0
    # Research
    current_research: Optional[HarvestContext] = None
    # Autofill: how many form fields were pre-filled from the chat on the
    # last click on "Start research"? > 0 means: the run was NOT started,
    # the person should review first.
    autofill_pending: int = 0
    autofill_note: str = ""
    research_history: list = field(default_factory=list)
    research_running: bool = False
    # Output language chosen in the UI (code, see src.output_language).
    output_language: str = field(default_factory=lambda: _default_output_language())
    # Plan preview gate
    # If set, the pipeline waits for user confirmation before running.
    # Filled by the analysis handler, read by the confirm handler and then
    # cleared.
    pending_plan_ctx: Optional[HarvestContext] = None
    pending_plan_use_case: Optional[str] = None
    pending_plan_label: Optional[str] = None
    # Mode-specific flags for resuming after the gate (research path): so
    # that the confirm handler uses the same settings as when the plan was
    # built, they must be persisted here.
    pending_institution_only: bool = False
    pending_academic_only: bool = False
    # Copy of the last result in the browser (see save_result_to_browser).
    # The research object that copy belongs to — saved, restored or
    # discarded by "New chat" — so that every result is written once.
    browser_result_ctx: Optional[HarvestContext] = None
    # The stored texts when current_research was restored from the browser
    # (pipeline-run view, appendices for the Word export).
    restored_result: Optional[dict] = None
    # UI
    sidebar_visible: bool = False
    result_panel_visible: bool = False


def init_app_state() -> AppState:
    """Lazy: called by Gradio per session.
    Heavy objects (LLM, connectors) are shared."""
    return AppState()


# Shared singletons — created once, shared by all sessions
_shared_llm: DualLLMClient | None = None
_shared_connectors: ConnectorRegistry | None = None
_shared_orchestrator: ResearchOrchestrator | None = None
_shared_config: AppConfig | None = None
_shared_directory = None
_shared_solr = None


def _ensure_shared_resources():
    """Create the shared resources once."""
    global _shared_llm, _shared_connectors, _shared_orchestrator, _shared_config, _shared_directory, _shared_solr

    if _shared_llm is not None:
        return

    _shared_config = AppConfig.from_env()
    _shared_llm = DualLLMClient(_shared_config.llm)

    _shared_connectors = ConnectorRegistry()
    _shared_connectors.register(
        SearXNGConnector(_shared_config.connectors.searxng),
        is_search_engine=True,
    )
    _shared_connectors.register(
        WebScraperConnector(_shared_config.connectors.web_scraper),
        is_web_scraper=True,
    )
    if _shared_config.connectors.github.enabled:
        _shared_connectors.register(
            GitHubConnector(_shared_config.connectors.github),
            url_patterns=[r"github\.com/", r"raw\.githubusercontent\.com/"],
        )
    if _shared_config.connectors.gitlab.enabled:
        _shared_connectors.register(
            GitLabConnector(_shared_config.connectors.gitlab),
            url_patterns=[
                _shared_config.connectors.gitlab.base_url
                .replace("https://", "").replace("http://", "")
            ],
        )
    _shared_connectors.register(LocalFileConnector())

    if _shared_config.connectors.elasticsearch.enabled:
        from src.connectors.elasticsearch import ElasticsearchConnector
        es_config = _shared_config.connectors.elasticsearch
        _shared_connectors.register(
            ElasticsearchConnector(es_config),
            url_patterns=[
                re.escape(es_config.site_base_url.replace("https://", "")
                           .replace("http://", ""))
            ] if es_config.site_base_url else None,
        )

    # Person directory connector (reads a local SQLite DB)
    _shared_directory = None
    if _shared_config.connectors.directory.enabled:
        from src.connectors.person_directory import PersonDirectoryConnector, DirectorySearchConfig
        directory_cfg = _shared_config.connectors.directory
        pipe = _shared_config.pipeline
        _shared_directory = PersonDirectoryConnector(DirectorySearchConfig(
            db_path=directory_cfg.db_path,
            embedder_base_url=pipe.embedder_base_url,
            embedder_model=pipe.embedder_model,
            embedder_api_key=pipe.embedder_api_key,
            reranker_base_url=pipe.reranker_base_url,
            reranker_model=pipe.reranker_model,
            reranker_api_key=pipe.reranker_api_key,
        ))

    # Solr connector (search of the institution's website)
    _shared_solr = None
    from src.connectors.solr_search import SolrConfig
    solr_cfg = SolrConfig.from_env()
    if solr_cfg.enabled:
        from src.connectors.solr_search import SolrConnector
        _shared_solr = SolrConnector(solr_cfg)
        # Register ONLY as a search, NOT as a URL handler
        # (the web scraper should keep fetching the institution's URLs,
        #  because Solr does not have every page in its index)
        _shared_connectors.register(_shared_solr)
        logger.info(f"Solr: {solr_cfg.base_url} (DE: {solr_cfg.core_de}, EN: {solr_cfg.core_en})")

    _shared_orchestrator = ResearchOrchestrator(
        _shared_llm, _shared_connectors, _shared_config.pipeline,
        person_directory=_shared_directory,
    )


def _get_ready_state(app_state) -> AppState:
    """Make sure the AppState is configured."""
    _ensure_shared_resources()

    # Gradio may pass the factory function instead of its result
    if app_state is None or callable(app_state):
        app_state = AppState()

    app_state.config = _shared_config
    app_state.llm = _shared_llm
    app_state.connectors = _shared_connectors
    app_state.orchestrator = _shared_orchestrator
    return app_state


# =====================================================================
# Helper Functions
# =====================================================================

def _build_doc_list_html(app_state: AppState) -> str:
    if not app_state.documents:
        return "<p style='color: var(--body-text-color-subdued); font-size: 0.8rem;'>No documents.</p>"
    html = ""
    for info in app_state.documents.values():
        name = html_escape(info["filename"], quote=True)
        html += (
            f"<div class='doc-item'>"
            f"<span class='doc-name' title='{name}'>📄 {name}</span>"
            f"</div>"
            f"<div style='font-size: 0.68rem; color: var(--body-text-color-subdued); "
            f"margin: 0 0 4px 8px;'>{info['tokens']:,} tokens</div>"
        )
    return html


def _build_token_bar_html(app_state: AppState) -> str:
    total = app_state.total_doc_tokens
    max_tok = int(app_state.config.llm.primary.max_context_tokens * 0.5)
    usage = (total / max_tok * 100) if max_tok > 0 else 0
    bar_class = "critical" if usage > 90 else "warning" if usage > 70 else "normal"
    return (
        f"<div class='token-bar-container'>"
        f"<div class='token-text'>{total:,} / {max_tok:,} Tokens</div>"
        f"<div class='token-bar'>"
        f"<div class='token-bar-fill {bar_class}' style='width: {min(usage, 100):.1f}%'></div>"
        f"</div></div>"
    )


def _get_doc_context(app_state: AppState) -> str:
    if not app_state.documents:
        return ""
    parts = []
    for doc_id, info in app_state.documents.items():
        parts.append(f"=== DOCUMENT: {info['filename']} ===\n{info['content']}")
    return "\n\n".join(parts)


def _format_progress(phase: str, data) -> str:
    """Format progress updates for the UI."""
    if phase == "format":
        schema = data
        return f"### 📋 Output format\n**{schema.title}**\nType: {schema.format_type}\n"

    elif phase == "plan":
        plan = data
        lines = [f"### 🔍 Research plan\n{plan.summary}\n"]
        for q in plan.questions:
            langs = ", ".join(q.search_langs) if q.search_langs else "de, en"
            lines.append(f"- **{q.id}** [{langs}]: {q.question}")
        if plan.direct_urls:
            lines.append(f"\n**Direct URLs:** {len(plan.direct_urls)}")
        if plan.git_repos:
            lines.append(f"**Git-Repos:** {len(plan.git_repos)}")
        if plan.directory_queries:
            lines.append(f"**👤 Person directory queries:** {len(plan.directory_queries)}")
            for zq in plan.directory_queries:
                lines.append(f'  - „{zq.query}"')
        return "\n".join(lines)

    elif phase == "search_results":
        return (f"### 🌐 Search results\n"
                f"Found: {data['total_found']} → {data['unique']} unique\n")

    elif phase == "harvest_done":
        return (f"### 📝 Round {data['round']}\n"
                f"Sources: {data['total_sources']} | "
                f"Extracts: {data['total_extracts']}\n")

    elif phase == "gaps":
        gap = data
        if gap.should_continue:
            return f"### 🔎 Gaps found\n{gap.reasoning}\n→ Another round..."
        else:
            return f"### ✅ Research complete\n{gap.reasoning}\n"

    return f"*{phase}*\n"


def _format_sources_log(sources: list, fetch_updates: list) -> str:
    """Format the sources overview — grouped by type."""
    if not sources and not fetch_updates:
        return "*No sources yet...*"

    # During the research: only show fetch_updates as a simple list
    if not sources and fetch_updates:
        lines = [f"### 📥 Loading sources... ({len(fetch_updates)} so far)\n"]
        for i, fu in enumerate(fetch_updates[-15:], 1):
            title = (fu.get("title", "") or "")[:60]
            length = fu.get("length", 0)
            suffix = f" · {length:,} characters" if length else ""
            lines.append(f"{i}. {title}{suffix}")
        if len(fetch_updates) > 15:
            lines.append(f"\n*... and {len(fetch_updates) - 15} more*")
        return "\n".join(lines)

    from urllib.parse import urlparse
    from src.connectors.base import clean_url

    # Group by type
    groups: dict[str, list] = {}
    type_labels = {
        "web_search": ("🌐", "Web"),
        "web_page": ("🌐", "Web"),
        "git_repo": ("📦", "Git-Repos"),
        "git_file": ("📄", "Git files"),
        "git_issue": ("🐛", "Issues"),
        "local_file": ("📎", "Local files"),
        "elastic": ("🔎", "Website-Index"),
        "directory_person": ("👤", "Person directory: people"),
        "directory_org": ("🏛️", "Person directory: units"),
    }

    for s in sources:
        type_key = s.source_type.value if hasattr(s.source_type, "value") else str(s.source_type)
        icon, label = type_labels.get(type_key, ("📄", "Other"))
        if label not in groups:
            groups[label] = (icon, [])
        url = clean_url(s.url) if s.url else ""
        # Extract the domain for context
        try:
            domain = urlparse(url).hostname or ""
            domain = domain.removeprefix("www.")
        except Exception:
            domain = ""
        title = (s.title or url)[:80]
        groups[label][1].append((title, url, domain))

    lines = [f"### 🔗 Sources ({len(sources)} in total)\n"]

    for label, (icon, items) in groups.items():
        lines.append(f"\n**{icon} {label}** ({len(items)})\n")
        for i, (title, url, domain) in enumerate(items, 1):
            domain_hint = f" · `{domain}`" if domain else ""
            if url:
                lines.append(f"{i}. [{title}]({url}){domain_hint}\n")
            else:
                lines.append(f"{i}. {title}\n")

    return "\n".join(lines)


def _format_extracts(harvest_results: list, research_plan=None) -> str:
    """Format the extracts — grouped by question, with reliability icons."""
    if not harvest_results:
        return "*No extracts yet...*"

    # Collect all extracts
    all_extracts = []
    for hr in harvest_results:
        if hr.is_relevant:
            all_extracts.extend(hr.extracts)

    if not all_extracts:
        return "*No relevant extracts found.*"

    rel_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}
    total = len(all_extracts)

    # Group by question
    by_question: dict[str, list] = {}
    for ext in all_extracts:
        qid = ext.question_id or "?"
        if qid not in by_question:
            by_question[qid] = []
        by_question[qid].append(ext)

    # Question labels from the plan (if available)
    q_labels = {}
    if research_plan and hasattr(research_plan, 'questions'):
        for q in research_plan.questions:
            q_labels[q.id] = q.question

    lines = [f"### 📝 Extracts ({total} in total)\n"]

    for qid in sorted(by_question.keys()):
        exts = by_question[qid]
        label = q_labels.get(qid, "")
        header = f"{qid}: {label}" if label else qid
        sources = len(set(e.source_url for e in exts if e.source_url))

        lines.append(
            f"<details><summary><b>{header}</b> "
            f"({len(exts)} extracts, {sources} sources)</summary>\n"
        )
        for ext in exts:
            icon = rel_icon.get(ext.reliability, "⚪")
            src = (ext.source_title or ext.source_url or "?")[:50]
            lines.append(f"- {icon} {ext.fact[:200]}")
            lines.append(f"  *{src}*")
        lines.append("\n</details>\n")

    return "\n".join(lines)


# =====================================================================
# Input processing: text + files from the MultimodalTextbox
# =====================================================================

def _extract_filepaths(input_value) -> list[str]:
    """Extract file paths from a MultimodalTextbox value.

    Handles several formats:
    - dict: {"text": "...", "files": [...]}  (change event)
    - list: [filepath, ...]                   (upload event)
    - str: a single path                      (upload event)
    """
    if not input_value:
        return []

    paths = []

    if isinstance(input_value, dict):
        files = input_value.get("files", []) or []
    elif isinstance(input_value, list):
        files = input_value
    elif isinstance(input_value, str):
        files = [input_value]
    else:
        return []

    for f in files:
        if isinstance(f, str):
            fp = f
        elif isinstance(f, dict):
            fp = f.get("path", f.get("name", ""))
        elif hasattr(f, "name"):
            fp = getattr(f, "path", None) or getattr(f, "name", "")
        else:
            fp = str(f) if f else ""

        if fp and os.path.exists(fp):
            paths.append(fp)

    return paths


async def _process_input(app_state: AppState, msg) -> tuple[str, bool]:
    """Process the input: text from the text field AND from pasted files.

    Gradio turns longer pasted texts into temporary .txt files. These are
    recognised here and returned as text.

    Returns:
        (text, docs_added): the complete text and whether new documents were added
    """
    text = ""
    docs_added = False

    if msg:
        if isinstance(msg, dict):
            text = msg.get("text", "") or ""
        elif isinstance(msg, str):
            text = msg

    # Process files
    filepaths = _extract_filepaths(msg)
    if filepaths:
        logger.info(f"_process_input: {len(filepaths)} file(s): "
                     f"{[os.path.basename(f) for f in filepaths]}")

    for filepath in filepaths:
        filename = os.path.basename(filepath)
        ext = Path(filepath).suffix.lower()

        # Detection: does the file come from Gradio's temp directory?
        # If so → paste/drop into the input line → treat as prompt text
        is_paste = ("/tmp/" in filepath.replace("\\", "/")
                     and "gradio" in filepath.replace("\\", "/").lower())

        if ext in PLAIN_TEXT_EXTENSIONS:
            # Plain text: read the content
            raw = ""
            for encoding in ["utf-8", "latin-1", "cp1252"]:
                try:
                    raw = Path(filepath).read_text(encoding=encoding)
                    break
                except (UnicodeDecodeError, Exception):
                    continue

            if not raw.strip():
                continue

            # paste → always prompt text
            # explicit upload + short text → prompt text as well
            threshold = app_state.config.ui.min_paste_doc_length
            if is_paste or len(raw.strip()) < threshold:
                if text:
                    text = f"{text}\n\n{raw.strip()}"
                else:
                    text = raw.strip()
                logger.info(f"  → inline text ({len(raw)} characters, paste={is_paste})")
                continue

            # Long explicit upload → store as a document
            tokens = max(1, len(raw) // 3)
            doc_id = str(uuid.uuid4())[:8]
            app_state.documents[doc_id] = {
                "filename": filename,
                "content": raw,
                "tokens": tokens,
            }
            app_state.total_doc_tokens += tokens
            app_state.sidebar_visible = True
            docs_added = True
            gr.Info(f"✅ {filename} ({tokens:,} Tokens)")

        elif ext in DOCUMENT_EXTENSIONS:
            # Rich documents (PDF, DOCX etc.)
            try:
                from src.documents.processor import DocumentProcessor
                from src.config import ProcessorConfig
                proc = DocumentProcessor(ProcessorConfig())
                processed = proc.process_document(filepath)
                doc_id = str(uuid.uuid4())[:8]
                app_state.documents[doc_id] = {
                    "filename": filename,
                    "content": processed.content,
                    "tokens": processed.token_count,
                }
                app_state.total_doc_tokens += processed.token_count
                app_state.sidebar_visible = True
                docs_added = True
                gr.Info(f"✅ {filename} ({processed.token_count:,} Tokens)")
            except Exception as e:
                logger.error(f"Document processing failed: {filename}: {e}")
                gr.Warning(f"⚠️ {filename}: {e}")
        else:
            gr.Warning(f"⚠️ {filename}: format not supported")

    return text.strip(), docs_added


# =====================================================================
# Event Handlers
# =====================================================================

async def send_chat_message(app_state: AppState, msg: dict, chatbot: list,
                            system_prompt: str):
    """Normal chat — the LLM answers."""
    app_state = _get_ready_state(app_state)

    text, docs_added = await _process_input(app_state, msg)

    if not text:
        # Only documents uploaded, no text → update the UI
        if docs_added:
            yield (app_state, chatbot, gr.update(visible=True),
                   gr.update(visible=False))
        else:
            yield (app_state, chatbot, gr.update(visible=True),
                   gr.update(visible=False))
        return

    # User message
    chatbot = list(chatbot or [])
    chatbot.append({"role": "user", "content": text})
    app_state.chat_history.append({"role": "user", "content": text})

    # Throbber
    chatbot.append({"role": "assistant", "content": THINKING_PLACEHOLDER})
    yield app_state, chatbot, gr.update(visible=False), gr.update(visible=True)

    # API-Call
    had_date = "{date}" in system_prompt or "{datum}" in system_prompt
    system = render_chat_system_prompt(system_prompt.strip(), get_date_text())
    if not had_date:
        system += "\n\n" + get_date_text()
    doc_context = _get_doc_context(app_state)
    if doc_context:
        system += f"\n\nDocument context:\n{doc_context[:5000]}"

    messages = [{"role": "system", "content": system}]
    for m in app_state.chat_history[-20:]:
        messages.append({"role": m["role"], "content": m["content"]})

    assistant_response = ""
    try:
        async for chunk in app_state.llm.primary.stream(messages):
            assistant_response = chunk
            if not assistant_response.strip():
                continue
            display = list(chatbot[:-1])
            display.append({"role": "assistant", "content": assistant_response})
            yield app_state, display, gr.update(visible=False), gr.update(visible=True)
    except Exception as e:
        assistant_response = f"⚠️ Error: {e}"

    app_state.chat_history.append({"role": "assistant", "content": assistant_response})
    chatbot = list(chatbot[:-1])
    chatbot.append({"role": "assistant", "content": assistant_response})
    yield app_state, chatbot, gr.update(visible=True), gr.update(visible=False)


async def _drive_research_pipeline(
    app_state: AppState,
    chatbot: list,
    text: str,
    pipeline_invoker,
):
    """Shared streaming loop of the research pipeline.

    Used by the initial run_research (after the plan preview gate) and by
    the confirm handler, so that the polling logic and the finalisation
    are not duplicated.

    Args:
        app_state: the Gradio session state (mutated)
        chatbot: the chatbot list (mutated)
        text: the original query, for display
        pipeline_invoker: async callable with the signature
                          (on_progress) -> ctx. The caller decides whether
                          it is a first run or a resume.

    Yields:
        Gradio update tuples matching research_outputs (10 items).
    """
    # Progress state for live updates
    progress_md = ""
    sources_md = ""
    report_md = ""
    extracts_md = ""
    all_sources: list = []
    all_harvests: list = []
    fetch_updates: list = []

    live_lines: list[str] = []
    _update_ready = asyncio.Event()

    async def on_progress(phase: str, data):
        nonlocal progress_md, sources_md, report_md, extracts_md
        nonlocal all_sources, all_harvests, fetch_updates

        if phase == "status":
            progress_md += f"\n{data}\n"
            live_lines.append(f"⏳ {data}")
            _update_ready.set()
        elif phase == "format":
            progress_md += _format_progress(phase, data)
            title = data.title if hasattr(data, "title") else "..."
            live_lines.append(f"📋 **Format:** {title}")
            _update_ready.set()
        elif phase == "plan":
            progress_md += _format_progress(phase, data)
            n_q = len(data.questions) if hasattr(data, "questions") else 0
            urls = len(data.direct_urls) if hasattr(data, "direct_urls") else 0
            directory = len(data.directory_queries) if hasattr(data, "directory_queries") else 0
            plan_info = f"🗺️ **Plan:** {n_q} questions, {urls} direct URLs"
            if directory:
                plan_info += f", {directory} person directory queries"
            live_lines.append(plan_info)
            _update_ready.set()
        elif phase == "search_strategy":
            langs = data.get("languages", [])
            lang_names = {
                "de": "🇩🇪 German", "en": "🇬🇧 English",
                "zh": "🇨🇳 Chinese", "pt": "🇧🇷 Portuguese",
                "es": "🇪🇸 Spanish", "fr": "🇫🇷 French",
                "ja": "🇯🇵 Japanese", "ko": "🇰🇷 Korean",
                "ru": "🇷🇺 Russian", "it": "🇮🇹 Italian",
                "nl": "🇳🇱 Dutch", "ar": "🇸🇦 Arabic",
                "pl": "🇵🇱 Polish", "tr": "🇹🇷 Turkish",
            }
            lang_display = ", ".join(lang_names.get(l, l) for l in langs)
            live_lines.append(f"🌍 **Search languages:** {lang_display}")
            terms_by_lang = data.get("terms_by_lang", {})
            for lang, terms in terms_by_lang.items():
                name = lang_names.get(lang, lang)
                terms_preview = ", ".join(f'"{t}"' for t in terms[:4])
                if len(terms) > 4:
                    terms_preview += f" (+{len(terms) - 4})"
                live_lines.append(f"  {name}: {terms_preview}")
            strategy_lines = ["\n### 🌍 Search strategy\n"]
            strategy_lines.append(f"**Languages:** {lang_display}\n")
            for lang, terms in terms_by_lang.items():
                name = lang_names.get(lang, lang)
                strategy_lines.append(f"**{name}:**")
                for t in terms:
                    strategy_lines.append(f"- {t}")
                strategy_lines.append("")
            progress_md += "\n".join(strategy_lines)
            _update_ready.set()
        elif phase == "search_results":
            progress_md += _format_progress(phase, data)
            live_lines.append(
                f"🌐 **Search:** {data.get('unique', '?')} sources found"
            )
            _update_ready.set()
        elif phase == "fetch_one":
            fetch_updates.append(data)
            n = len(fetch_updates)
            title = (data.get("title", "") or "")[:50]
            live_lines.append(f"📥 Source {n}: {title}")
            _update_ready.set()
        elif phase == "harvest_one":
            title = (data.get("title", "") or "")[:50] if isinstance(data, dict) else ""
            live_lines.append(f"📝 Analysing: {title}")
            _update_ready.set()
        elif phase == "harvest_done":
            progress_md += _format_progress(phase, data)
            live_lines.append(
                f"✅ **Round {data.get('round', '?')}:** "
                f"{data.get('total_sources', '?')} sources, "
                f"{data.get('total_extracts', '?')} extracts"
            )
            _update_ready.set()
        elif phase == "gaps":
            progress_md += _format_progress(phase, data)
            if hasattr(data, "should_continue") and data.should_continue:
                live_lines.append("🔎 Gaps found — another round...")
            else:
                live_lines.append(
                    "📊 Research complete — writing the report..."
                )
            _update_ready.set()
        elif phase == "report_stream":
            report_md = data
            _update_ready.set()

    pipeline_result = None
    pipeline_error = None

    async def run_pipeline():
        nonlocal pipeline_result, pipeline_error
        try:
            pipeline_result = await pipeline_invoker(on_progress)
        except Exception as e:
            pipeline_error = e
        finally:
            _update_ready.set()

    task = asyncio.create_task(run_pipeline())

    # Polling-Loop
    while not task.done():
        try:
            await asyncio.wait_for(_update_ready.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pass
        _update_ready.clear()

        if report_md:
            display_md = report_md
        else:
            display_md = (
                "## 🔍 Research running...\n\n"
                + "\n\n".join(live_lines[-15:])
            )

        yield (app_state, chatbot,
               gr.update(visible=False), gr.update(visible=True),
               gr.update(visible=True), gr.update(visible=True),
               display_md,
               _format_sources_log([], fetch_updates) if fetch_updates else "",
               progress_md or "⏳ Running...",
               "")

    # Finalise on error or success
    if pipeline_error:
        logger.error(
            f"Pipeline error: {pipeline_error}", exc_info=pipeline_error
        )
        error_msg = (
            f"⚠️ **Research failed**\n\n"
            f"Error: {str(pipeline_error)[:300]}\n\n"
            f"*Please try again.*"
        )
        chatbot_new = list(chatbot[:-1]) if chatbot else []
        chatbot_new.append({"role": "assistant", "content": error_msg})
        app_state.research_running = False
        yield (app_state, chatbot_new,
               gr.update(visible=True), gr.update(visible=False),
               gr.update(visible=True), gr.update(visible=True),
               f"⚠️ Error: {str(pipeline_error)[:200]}",
               "", progress_md or "", "")
        return

    ctx = pipeline_result
    if not ctx:
        logger.error("Pipeline returned no result (ctx=None)")
        app_state.research_running = False
        yield (app_state, chatbot,
               gr.update(visible=True), gr.update(visible=False),
               gr.update(visible=True), gr.update(visible=True),
               "⚠️ The research returned no result.", "", "", "")
        return

    # Formatting
    try:
        sources_md = _format_sources_log(ctx.sources, fetch_updates)
    except Exception as e:
        logger.warning(f"Formatting the sources failed: {e}")
        sources_md = f"⚠️ Formatting error: {e}"

    try:
        extracts_md = _format_extracts(ctx.harvest_results, ctx.research_plan)
    except Exception as e:
        logger.warning(f"Formatting the extracts failed: {e}")
        extracts_md = f"⚠️ Formatting error: {e}"

    if not report_md:
        report_md = ctx.final_report or "*No report generated.*"

    # Finalisation
    chatbot_new = list(chatbot[:-1]) if chatbot else []
    try:
        lang_names = {
            "de": "🇩🇪", "en": "🇬🇧", "zh": "🇨🇳", "pt": "🇧🇷",
            "es": "🇪🇸", "fr": "🇫🇷", "ja": "🇯🇵", "ko": "🇰🇷",
            "ru": "🇷🇺", "it": "🇮🇹", "nl": "🇳🇱", "ar": "🇸🇦",
            "pl": "🇵🇱", "tr": "🇹🇷",
        }
        search_langs_info = ""
        if ctx.search_stats and ctx.search_stats.get("languages"):
            flags = " ".join(
                lang_names.get(l, l)
                for l in ctx.search_stats["languages"]
            )
            search_langs_info = f"- Search languages: {flags}\n"

        summary_msg = (
            f"✅ **Research completed**\n\n"
            f"- {len(ctx.sources)} sources searched\n"
            f"- {len(ctx.extracts)} facts extracted\n"
            f"- {ctx.rounds_completed} research rounds\n"
            f"{search_langs_info}"
            f"- Duration: {ctx.duration_seconds:.0f} seconds\n\n"
            f"*The result is in the “Report” tab on the right, the export at the top of the result area →*"
        )
        chatbot_new.append({"role": "assistant", "content": summary_msg})
        app_state.chat_history.append(
            {"role": "assistant", "content": summary_msg}
        )
        app_state.current_research = ctx
        app_state.research_history.insert(0, {
            "title": (
                ctx.output_schema.title if ctx.output_schema else text[:40]
            ),
            "date": datetime.now().strftime("%d.%m. %H:%M"),
            "sources": len(ctx.sources),
            "id": ctx.id,
        })
    except Exception as e:
        logger.error(f"Finalisation failed: {e}", exc_info=True)
        app_state.current_research = ctx
        chatbot_new.append({
            "role": "assistant",
            "content": f"✅ Research completed (with warnings: {e})",
        })

    app_state.research_running = False

    yield (app_state, chatbot_new,
           gr.update(visible=True), gr.update(visible=False),
           gr.update(visible=True), gr.update(visible=True),
           report_md, sources_md, progress_md, extracts_md)


async def run_research(app_state: AppState, msg: dict, chatbot: list,
                       system_prompt: str, template_name: str,
                       research_mode: str = DEFAULT_RESEARCH_MODE,
                       institution_only: bool = False,
                       academic_only: bool = False,
                       show_gate: bool = False,
                       explainer_topic: str = "",
                       explainer_audience: str = "",
                       explainer_length: str = "medium",
                       explainer_purpose: str = "self_study",
                       *analysis_args):
    """Start the autonomous research pipeline.

    Args:
        research_mode: ID of one of the supported modes (web, institution,
                       literature check, in-depth explanation or one of
                       the generic analysis modes)
        institution_only: only the institution's own sources (website
                          index + person directory + site: filter)
        academic_only: only academic SearXNG engines (Wikipedia, Wikidata,
                       Google Scholar, Semantic Scholar, ArXiv, PubMed).
                       Can be combined with institution_only.
        explainer_*: only used in the in-depth explanation mode
        analysis_args: tail arguments with the inputs of all analysis
                       panels. Split via _split_analysis_args().
    """
    # Routing over the stable mode ID (never over the display label).
    kind, routed_use_case = resolve_research_route(research_mode)

    # Delegate the generic analysis modes to a helper
    if kind == "analysis":
        async for update in run_analysis_pipeline_generic(
            app_state, chatbot,
            use_case=routed_use_case,
            analysis_args=analysis_args,
            show_gate=show_gate,
        ):
            yield update
        return

    # Delegate the in-depth explanation to its own function
    if kind == "explainer":
        async for update in run_explainer_pipeline(
            app_state, chatbot,
            topic=explainer_topic,
            audience=explainer_audience,
            length=explainer_length,
            purpose=explainer_purpose,
        ):
            yield update
        return

    # Delegate the literature check to its own function
    if kind == "literature_check":
        async for update in run_literature_check(
            app_state, msg, chatbot, system_prompt
        ):
            yield update
        return

    # Determine the mode
    mode = "institution" if kind == "institution" else "web"
    use_directory = (mode == "institution")  # institution mode: person directory always active
    app_state = _get_ready_state(app_state)

    text, docs_added = await _process_input(app_state, msg)

    if not text:
        # No new text → try to use the last relevant message
        for m in reversed(app_state.chat_history):
            content = m.get("content", "")
            if isinstance(content, list):
                parts = [p if isinstance(p, str) else p.get("text", "")
                         for p in content if isinstance(p, (str, dict))]
                content = "\n".join(parts)
            if not isinstance(content, str):
                continue
            content = content.strip()
            if not content or any(content.startswith(p) for p in ("🔍", "✅", "⏳", "🤔")):
                continue
            text = content
            role = "assistant answer" if m["role"] == "assistant" else "message"
            gr.Info(f"📋 Using the last {role} as the research request")
            break

    if not text:
        gr.Warning("Please enter a research request.")
        yield (app_state, chatbot,
               gr.update(visible=True), gr.update(visible=False),
               gr.update(visible=False), gr.update(visible=True),
               "", "", "", "")
        return

    # Research message in the chat
    chatbot = list(chatbot or [])
    if msg and isinstance(msg, dict) and msg.get("text", "").strip():
        chatbot.append({"role": "user", "content": text})
        app_state.chat_history.append({"role": "user", "content": text})

    # Check the connector status and warn
    warnings = []
    searxng = app_state.connectors.get_search_connector()
    if searxng and hasattr(searxng, 'check_connectivity'):
        available = await searxng.check_connectivity()
        if not available:
            warnings.append("⚠️ **SearXNG unreachable** — web search disabled")
    github = app_state.connectors.get_connector_by_name("github")
    if github and hasattr(github, '_has_token') and not github._has_token:
        warnings.append("⚠️ **GitHub without a token** — heavily limited API access (60/h)")

    # Solr: verify the fields on the first run
    if _shared_solr and hasattr(_shared_solr, 'verify_fields'):
        if not getattr(_shared_solr, '_fields_verified', False):
            try:
                result = await _shared_solr.verify_fields()
                _shared_solr._fields_verified = True
                if not result.get("ok") and result.get("missing"):
                    warnings.append(
                        f"⚠️ **Solr:** fields missing in the index: "
                        f"{', '.join(result['missing'])}. "
                        f"Available: {', '.join(result.get('available', [])[:10])}"
                    )
            except Exception as e:
                logger.warning(f"Solr field check: {e}")

    # Clean up person-directory content without consent (institution mode only)
    if use_directory and _shared_directory and hasattr(_shared_directory, 'ensure_consent_fresh'):
        try:
            await _shared_directory.ensure_consent_fresh(max_age_days=7)
        except Exception as e:
            logger.warning(f"Person directory: consent clean-up failed: {e}")

    mode_labels = {
        "institution": f"🏛️ **{get_profile().label} research started...**",
        "web": "🔍 **Research started...**",
    }
    status_msg = mode_labels.get(mode, "🔍 **Research started...**")
    if institution_only and mode == "institution":
        status_msg = (f"🏛️ **{get_profile().label} research started "
                      f"(institution sources only)...**")
    if warnings:
        status_msg += "\n\n" + "\n".join(warnings)
    chatbot.append({"role": "assistant", "content": status_msg})
    app_state.research_running = True
    app_state.result_panel_visible = True

    # Initial yield: open the panel, switch the buttons
    yield (app_state, chatbot,
           gr.update(visible=False), gr.update(visible=True),  # send hidden, stop visible
           gr.update(visible=True),    # result panel
           gr.update(visible=True),    # research btn disabled
           "⏳ *Preparing the research...*", "", "", "")

    # Create a NEW orchestrator per research run
    # (isolates _seen_urls, _stop_requested, _use_directory per session)
    run_orchestrator = ResearchOrchestrator(
        _shared_llm, _shared_connectors, _shared_config.pipeline,
        person_directory=_shared_directory,
        output_language=app_state.output_language,
    )
    app_state.orchestrator = run_orchestrator

    # ── Phase 1: plan-only run for the plan preview gate ──
    # First build the research plan (format agent + analysis phase), show
    # it to the user and let them confirm it. The plan's queries are kept
    # separate per language and are shown that way.
    from src.ui.components.plan_preview import (
        should_show_research_preview,
        format_research_plan_markdown,
    )

    # Silent progress during the plan phase — the chatbot only shows the
    # finished plan, not the intermediate states.
    async def _silent_plan_progress(event, data):
        pass

    try:
        plan_ctx = await run_orchestrator.run(
            query=text,
            chat_history=app_state.chat_history,
            context_docs=_get_doc_context(app_state),
            template_name=template_name or DEFAULT_TEMPLATE,
            progress_callback=_silent_plan_progress,
            use_directory=use_directory,
            mode=mode,
            institution_only=institution_only,
            academic_only=academic_only,
            plan_only=True,
        )
    except Exception as e:
        logger.exception("Building the research plan failed")
        chatbot = list(chatbot[:-1]) if chatbot else []
        chatbot.append({
            "role": "assistant",
            "content": f"❌ Building the plan failed: {e}",
        })
        app_state.research_running = False
        yield (app_state, chatbot,
               gr.update(visible=True), gr.update(visible=False),
               gr.update(visible=True), gr.update(visible=True),
               f"❌ Building the plan failed: {e}", "", "", "")
        return

    if (plan_ctx is None
            or plan_ctx.status == "error"
            or not plan_ctx.research_plan):
        err = (
            plan_ctx.error_message if plan_ctx else "unknown error"
        )
        chatbot = list(chatbot[:-1]) if chatbot else []
        chatbot.append({
            "role": "assistant",
            "content": f"⚠️ Building the plan gave no result: {err}",
        })
        app_state.research_running = False
        yield (app_state, chatbot,
               gr.update(visible=True), gr.update(visible=False),
               gr.update(visible=True), gr.update(visible=True),
               f"⚠️ **Building the plan gave no result**\n\n{err}",
               "", "", "")
        return

    # ── Gate decision ──
    mode_label = (f"{get_profile().label} research" if mode == "institution"
                  else "Web research")

    if show_gate and should_show_research_preview(plan_ctx.research_plan):
        # Show the gate: store the state, render the plan in the chatbot, end
        # the handler. The confirm handler takes over.
        app_state.pending_plan_ctx = plan_ctx
        app_state.pending_plan_use_case = mode  # "web" or "institution"
        app_state.pending_plan_label = mode_label
        app_state.pending_institution_only = institution_only
        app_state.pending_academic_only = academic_only

        plan_md_text = format_research_plan_markdown(
            plan_ctx.research_plan,
            plan_ctx.search_stats,
            mode=mode,
            academic_only=academic_only,
        )
        chatbot = list(chatbot[:-1]) if chatbot else []
        chatbot.append({
            "role": "assistant",
            "content": (
                f"🔍 **{mode_label} — plan preview**\n\n"
                f"{plan_md_text}\n\n"
                f"---\n\n"
                f"👇 **Please confirm the plan below to continue.**"
            ),
        })
        app_state.research_running = False
        yield (app_state, chatbot,
               gr.update(visible=True), gr.update(visible=False),
               gr.update(visible=True), gr.update(visible=True),
               "⏸️ *Plan preview — waiting for confirmation*",
               "", "", "")
        return

    # ── No gate: run the plan directly ──
    async def invoker(on_progress):
        return await run_orchestrator.run(
            query=text,
            chat_history=app_state.chat_history,
            context_docs=_get_doc_context(app_state),
            template_name=template_name or DEFAULT_TEMPLATE,
            progress_callback=on_progress,
            use_directory=use_directory,
            mode=mode,
            institution_only=institution_only,
            academic_only=academic_only,
            existing_ctx=plan_ctx,
            skip_decompose=True,
        )

    async for update in _drive_research_pipeline(
        app_state, chatbot, text, invoker,
    ):
        yield update


def _append_filter_stats_banner(report: str, ctx) -> str:
    """Append the filter statistics banner to a report.

    Returns `report` unchanged if ctx has no data relevant for the banner
    (e.g. an aborted pipeline). Otherwise report + separator + banner.
    Filter losses must be visible, otherwise the user takes an empty
    report for the result.
    """
    if not ctx:
        return report
    try:
        banner = format_filter_stats_banner(ctx)
    except Exception:
        return report
    if not banner.strip():
        return report
    return f"{report}\n\n---\n\n{banner}"


def stop_research(app_state: AppState):
    """Stop the running research or literature check."""
    if app_state and app_state.orchestrator:
        app_state.orchestrator.stop()
    if app_state and hasattr(app_state, '_lit_checker') and app_state._lit_checker:
        app_state._lit_checker.stop()
    return gr.update(visible=True), gr.update(visible=False)


async def run_explainer_pipeline(
    app_state: AppState,
    chatbot: list,
    topic: str,
    audience: str,
    length: str = "medium",
    purpose: str = "self_study",
):
    """Start an in-depth explanation pipeline.

    Uses the AnalysisPipelineRunner via the ResearchOrchestrator
    (mode="explainer"). Streams the analysis pipeline events into the
    progress tab.
    """
    from src.pipeline.orchestrator import ResearchOrchestrator
    from src.ui.analysis_runner import (
        format_analysis_event,
        validate_explainer_inputs,
        build_explainer_preflight_data,
    )

    app_state = _get_ready_state(app_state)

    # Preflight validation (before the pipeline runs)
    ok, error_msg = validate_explainer_inputs(topic, audience, length, purpose)
    if not ok:
        gr.Warning(f"In-depth explanation: {error_msg}")
        yield (app_state, chatbot,
               gr.update(visible=True), gr.update(visible=False),
               gr.update(visible=False), gr.update(visible=True),
               "", "", "", "")
        return

    preflight_data = build_explainer_preflight_data(
        topic, audience, length, purpose,
    )
    topic_clean = preflight_data["topic"]

    # Chat messages
    chatbot = list(chatbot or [])
    chatbot.append({
        "role": "user",
        "content": f"📖 *In-depth explanation:* {topic_clean}",
    })
    chatbot.append({
        "role": "assistant",
        "content": "📖 **Writing the in-depth explanation...**",
    })
    app_state.chat_history.append(
        {"role": "user", "content": f"In-depth explanation: {topic_clean}"}
    )
    app_state.research_running = True

    yield (app_state, chatbot,
           gr.update(visible=False), gr.update(visible=True),
           gr.update(visible=True), gr.update(visible=True),
           "⏳ *Preparing the in-depth explanation...*", "", "", "")

    # Progress-Handling
    progress_md = f"# In-depth explanation: {topic_clean}\n\n"
    progress_md += f"**Audience:** {audience}\n"
    progress_md += f"**Length:** {length}\n\n"
    progress_md += "---\n\n"
    report_md = "⏳ *Decomposition running...*"
    live_lines: list[str] = []
    _update_ready = asyncio.Event()

    async def on_progress(event: str, data):
        nonlocal progress_md, report_md

        # Classic research events do not occur here;
        # we only react to analysis events.
        if event == "status":
            # by the orchestrator before the runner call
            progress_md += f"\n{data}\n"
            live_lines.append(str(data))
            _update_ready.set()
            return

        # Do NOT restrict to dict: node_done/node_failed deliver a
        # NodeResult object, which the formatter evaluates itself.
        line = format_analysis_event(event, data)
        if line:
            live_lines.append(line)
            progress_md += f"\n{line}\n"
            _update_ready.set()

    # Orchestrator instance (isolated per run, as in the research mode)
    run_orchestrator = ResearchOrchestrator(
        _shared_llm, _shared_connectors, _shared_config.pipeline,
        person_directory=_shared_directory,
        output_language=app_state.output_language,
    )
    app_state.orchestrator = run_orchestrator

    # Start the pipeline as a background task
    pipeline_ctx = None
    pipeline_error = None

    async def run_pipeline():
        nonlocal pipeline_ctx, pipeline_error
        try:
            pipeline_ctx = await run_orchestrator.run(
                query=topic_clean,
                chat_history=app_state.chat_history,
                context_docs="",
                template_name="In-depth explanation",
                progress_callback=on_progress,
                mode="explainer",
                preflight_data=preflight_data,
            )
        except Exception as e:
            pipeline_error = e
            logger.exception("Explainer pipeline failed")
        finally:
            _update_ready.set()

    task = asyncio.create_task(run_pipeline())

    # Polling-Loop
    while not task.done():
        try:
            await asyncio.wait_for(_update_ready.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pass
        _update_ready.clear()

        # Live report in the report tab: last lines
        if live_lines:
            report_md = (
                f"# In-depth explanation: {topic_clean}\n\n"
                f"*Running...*\n\n"
                + "\n".join(live_lines[-20:])
            )

        yield (app_state, chatbot,
               gr.update(visible=False), gr.update(visible=True),
               gr.update(visible=True), gr.update(visible=True),
               report_md, "", progress_md, "")

    await task

    # Final
    app_state.research_running = False

    if pipeline_error:
        final_report = (
            f"❌ **Error:** {pipeline_error}\n\n"
            f"Details in the “History” tab."
        )
        ChatState(chatbot).replace_last_assistant(
            f"❌ Error in the in-depth explanation: {pipeline_error}",
        )
    elif pipeline_ctx and pipeline_ctx.status == "done":
        final_report = pipeline_ctx.final_report or "(no result)"
        # Append the filter statistics banner
        final_report = _append_filter_stats_banner(final_report, pipeline_ctx)
        # Session persistence (same name as for research)
        app_state.current_research = pipeline_ctx
        ChatState(chatbot).replace_last_assistant(
            (
                f"✅ **In-depth explanation finished**: {topic_clean}\n\n"
                f"See the result panel for the full text."
            ),
        )
    else:
        err = pipeline_ctx.error_message if pipeline_ctx else "unknown"
        final_report = f"⚠️ **Pipeline finished without a result**\n\n{err}"
        ChatState(chatbot).replace_last_assistant(
            f"⚠️ Pipeline finished: {err}",
        )

    yield (app_state, chatbot,
           gr.update(visible=True), gr.update(visible=False),
           gr.update(visible=True), gr.update(visible=True),
           final_report, "", progress_md, "")


async def _drive_analysis_execution(
    app_state,
    chatbot: list,
    label: str,
    icon: str,
    short_summary: str,
    pipeline_invoker,
):
    """Shared streaming loop of the analysis pipeline run.

    Used by the initial handler (first run after preflight) and by the
    confirm handler (resume after the plan preview gate).

    Args:
        app_state: the Gradio session state
        chatbot: the chatbot list (mutated for the final message)
        label: display label of the use case (e.g. "Peer review")
        icon: emoji icon of the use case
        short_summary: short description for the progress display
        pipeline_invoker: async callable with the signature
                          `(on_progress) -> ctx`. The caller decides
                          whether a first run or a resume is executed and
                          calls the orchestrator with the matching
                          parameters.

    Yields:
        Gradio update tuples matching research_outputs.
    """
    from src.ui.analysis_runner import format_analysis_event

    # Progress setup
    progress_md = f"# {label}: {short_summary}\n\n---\n\n"
    report_md = f"⏳ *{label} running...*"
    live_lines: list[str] = []
    _update_ready = asyncio.Event()

    async def on_progress(event: str, data):
        nonlocal progress_md, report_md
        if event == "status":
            progress_md += f"\n{data}\n"
            live_lines.append(str(data))
            _update_ready.set()
            return
        line = format_analysis_event(
            event, data
        )
        if line:
            live_lines.append(line)
            progress_md += f"\n{line}\n"
            _update_ready.set()

    pipeline_ctx = None
    pipeline_error = None

    async def run_pipeline():
        nonlocal pipeline_ctx, pipeline_error
        try:
            pipeline_ctx = await pipeline_invoker(on_progress)
        except Exception as e:
            pipeline_error = e
            logger.exception(f"{label} pipeline failed")
        finally:
            _update_ready.set()

    task = asyncio.create_task(run_pipeline())

    # Polling
    while not task.done():
        try:
            await asyncio.wait_for(_update_ready.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pass
        _update_ready.clear()

        if live_lines:
            report_md = (
                f"# {label}: {short_summary}\n\n*Running...*\n\n"
                + "\n".join(live_lines[-20:])
            )
        yield (app_state, chatbot,
               gr.update(visible=False), gr.update(visible=True),
               gr.update(visible=True), gr.update(visible=True),
               report_md, "", progress_md, "")

    await task
    app_state.research_running = False

    if pipeline_error:
        final_report = (
            f"❌ **Error:** {pipeline_error}\n\nDetails in the “History” tab."
        )
        ChatState(chatbot).replace_last_assistant(
            f"❌ Error in {label}: {pipeline_error}",
        )
    elif pipeline_ctx and pipeline_ctx.status == "done":
        final_report = pipeline_ctx.final_report or "(no result)"
        # Append the filter statistics banner — filter losses must be visible
        final_report = _append_filter_stats_banner(final_report, pipeline_ctx)
        app_state.current_research = pipeline_ctx
        ChatState(chatbot).replace_last_assistant(
            (
                f"✅ **{label} finished**\n\n"
                f"See the result panel for the full report."
            ),
        )
    else:
        err = pipeline_ctx.error_message if pipeline_ctx else "unknown"
        final_report = f"⚠️ **Pipeline finished without a result**\n\n{err}"
        ChatState(chatbot).replace_last_assistant(
            f"⚠️ Pipeline finished: {err}",
        )

    yield (app_state, chatbot,
           gr.update(visible=True), gr.update(visible=False),
           gr.update(visible=True), gr.update(visible=True),
           final_report, "", progress_md, "")


async def run_analysis_pipeline_generic(
    app_state: AppState,
    chatbot: list,
    use_case: str,
    analysis_args: tuple,
    show_gate: bool = False,
):
    """Start any analysis pipeline (all generic modes).

    Reads the input values from the flat tuple analysis_args, assigns
    them to the right use case (by the order in the registered
    PreflightCheckers), validates via preflight and starts the
    orchestrator with `mode=use_case`.

    Same streaming logic as run_explainer_pipeline, but generic over all
    use cases.
    """
    from src.pipeline.orchestrator import ResearchOrchestrator
    from src.pipeline.analysis_pipeline import USE_CASE_REGISTRY
    from src.ui.analysis_runner import (
        make_chat_summary,
    )

    app_state = _get_ready_state(app_state)

    # Get the use case and its inputs
    if use_case not in USE_CASE_REGISTRY:
        gr.Warning(f"Unknown use case: {use_case}")
        yield (app_state, chatbot,
               gr.update(visible=True), gr.update(visible=False),
               gr.update(visible=False), gr.update(visible=True),
               "", "", "", "")
        return

    config = USE_CASE_REGISTRY[use_case]
    checker = config["preflight"]

    # ── Autofill stop ──
    # `autofill_analysis_form` ran immediately before and pre-filled empty
    # mandatory fields from the chat. The run then does NOT start right
    # away: the suggestions come from an interpretation of the
    # conversation and should be reviewed before a minutes-long analysis
    # builds on them. A second click starts it.
    pending = getattr(app_state, "autofill_pending", 0)
    if pending:
        note = getattr(app_state, "autofill_note", "") or ""
        app_state.autofill_pending = 0
        app_state.autofill_note = ""
        chatbot = list(chatbot) + [{"role": "assistant", "content": note}]
        if pending > 0:
            gr.Info(f"{pending} field(s) pre-filled from the chat — "
                    "please review and start again.")
        else:
            gr.Warning("The mandatory fields could not be derived "
                       "from the chat.")
        yield (app_state, chatbot,
               gr.update(visible=True), gr.update(visible=False),
               gr.update(visible=False), gr.update(visible=True),
               "", "", "", "")
        return

    # Split analysis_args and extract the right slice
    splits = _split_analysis_args(analysis_args)
    values = splits.get(use_case, [])
    field_order = [r.field for r in checker.get_requirements()]
    inputs = dict(zip(field_order, values))

    # Validation
    try:
        ok, errors = checker.validate(inputs)
    except Exception as e:
        ok, errors = False, [f"Validation error: {e}"]

    if not ok:
        gr.Warning(
            f"{_use_case_label(use_case)}: " + " · ".join(errors)
        )
        yield (app_state, chatbot,
               gr.update(visible=True), gr.update(visible=False),
               gr.update(visible=False), gr.update(visible=True),
               "", "", "", "")
        return

    # Normalise (caught here too, in case normalize() raises)
    try:
        preflight_data = checker.normalize(inputs)
    except Exception as e:
        gr.Warning(f"{_use_case_label(use_case)}: input error: {e}")
        yield (app_state, chatbot,
               gr.update(visible=True), gr.update(visible=False),
               gr.update(visible=False), gr.update(visible=True),
               "", "", "", "")
        return

    # Display label
    label = _use_case_label(use_case)
    icon = _use_case_icon(use_case)

    # Chat messages
    short_summary = make_chat_summary(use_case, inputs)
    chatbot = list(chatbot or [])
    chatbot.append({
        "role": "user",
        "content": f"{icon} *{label}:* {short_summary}",
    })
    chatbot.append({
        "role": "assistant",
        "content": f"{icon} **{label} running...**",
    })
    app_state.chat_history.append(
        {"role": "user", "content": f"{label}: {short_summary}"}
    )
    app_state.research_running = True

    yield (app_state, chatbot,
           gr.update(visible=False), gr.update(visible=True),
           gr.update(visible=True), gr.update(visible=True),
           f"⏳ *Preparing {label}...*", "", "", "")

    # Orchestrator instance
    run_orchestrator = ResearchOrchestrator(
        _shared_llm, _shared_connectors, _shared_config.pipeline,
        person_directory=_shared_directory,
        output_language=app_state.output_language,
    )
    app_state.orchestrator = run_orchestrator

    # ── Phase 1: plan-only run for the plan preview gate ──
    # First build the plan (without running it), check whether the gate
    # should be shown, and then decide.
    from src.ui.components.plan_preview import (
        should_show_preview, format_task_plan_markdown,
    )

    # Silent progress callback for the decomposition phase — the plan is
    # only shown when it is complete.
    _plan_status_lines: list[str] = []
    async def _plan_progress(event: str, data):
        if event == "status":
            _plan_status_lines.append(str(data))

    try:
        plan_ctx = await run_orchestrator.run(
            query=short_summary,
            chat_history=app_state.chat_history,
            context_docs="",
            template_name=label,
            progress_callback=_plan_progress,
            mode=use_case,
            preflight_data=preflight_data,
            plan_only=True,
        )
    except Exception as e:
        logger.exception(f"{label} plan build failed")
        plan_ctx = None
        ChatState(chatbot).replace_last_assistant(
            f"❌ Error while building the plan: {e}",
        )
        app_state.research_running = False
        yield (app_state, chatbot,
               gr.update(visible=True), gr.update(visible=False),
               gr.update(visible=True), gr.update(visible=True),
               f"❌ Building the plan failed: {e}", "", "", "")
        return

    if plan_ctx is None or plan_ctx.status == "error" or not plan_ctx.task_plan:
        err = (plan_ctx.error_message if plan_ctx else "unknown error")
        ChatState(chatbot).replace_last_assistant(
            f"⚠️ Plan build ended without a plan: {err}",
        )
        app_state.research_running = False
        yield (app_state, chatbot,
               gr.update(visible=True), gr.update(visible=False),
               gr.update(visible=True), gr.update(visible=True),
               f"⚠️ **Building the plan gave no result**\n\n{err}",
               "", "", "")
        return

    # ── Gate decision ──
    # The "Confirm plan" checkbox decides, nothing else.
    #
    # `should_show_preview(plan, force=show_gate)` alone would not do:
    # `force` is an OR, so with the checkbox unticked the heuristic
    # (>= 3 tasks or >= 2 phases) would still apply — and it applies to
    # practically every analysis, giving users a confirmation gate they
    # never asked for.
    #
    # As in the research path (`if show_gate and
    # should_show_research_preview(...)`), the checkbox is the condition
    # and the heuristic only an additional restriction; both paths
    # behave the same.
    if show_gate and should_show_preview(plan_ctx.task_plan):
        # Show the gate: store the state, render the plan in the chatbot, end
        # the handler. The confirm handler takes over.
        app_state.pending_plan_ctx = plan_ctx
        app_state.pending_plan_use_case = use_case
        app_state.pending_plan_label = label

        plan_md_text = format_task_plan_markdown(plan_ctx.task_plan)
        ChatState(chatbot).replace_last_assistant(
            (
                f"{icon} **{label} — plan preview**\n\n"
                f"{plan_md_text}\n\n"
                f"---\n\n"
                f"👇 **Please confirm the plan below to continue.**"
            ),
        )
        app_state.research_running = False
        # Show start_btn again (for cancel-via-restart), hide stop_btn — the
        # actual confirm button is shown by a subsequent .then() handler.
        yield (app_state, chatbot,
               gr.update(visible=True), gr.update(visible=False),
               gr.update(visible=True), gr.update(visible=True),
               "⏸️ *Plan preview — waiting for confirmation*",
               "", "", "")
        return

    # ── No gate: run the plan directly ──
    async def invoker(on_progress):
        return await run_orchestrator.run(
            query=short_summary,
            chat_history=app_state.chat_history,
            context_docs="",
            template_name=label,
            progress_callback=on_progress,
            mode=use_case,
            existing_ctx=plan_ctx,
            skip_decompose=True,
        )

    async for update in _drive_analysis_execution(
        app_state, chatbot, label, icon, short_summary, invoker,
    ):
        yield update


# ─── Plan-Preview-Gate: Confirm & Cancel Handler ──────────────────

async def confirm_plan_and_run(
    app_state: AppState,
    chatbot: list,
    edited_queries_text: str = "",
):
    """Handler for the confirm button of the plan preview gate.

    Reads the held context from app_state.pending_plan_ctx, clears the
    pending fields and calls the orchestrator with skip_decompose=True,
    so that the plan already built is run directly.

    If the user changed search queries in the query edit field (for
    literature_finder, literature_review, grant_proposal), the
    literature search is repeated with the new queries and a new plan is
    built before the run starts.

    Dispatches between two paths:
    - analysis use cases (explainer, peer_review, ...) → _drive_analysis_execution
    - research modes ("web", "institution") → _drive_research_pipeline
    """
    from src.ui.components.plan_preview import extract_plan_metadata_info

    app_state = _get_ready_state(app_state)

    ctx = app_state.pending_plan_ctx
    use_case = app_state.pending_plan_use_case
    label = app_state.pending_plan_label
    pending_institution_only = app_state.pending_institution_only
    pending_academic_only = app_state.pending_academic_only

    if ctx is None or not use_case:
        # No waiting plan — nothing to do
        yield (app_state, chatbot,
               gr.update(visible=True), gr.update(visible=False),
               gr.update(visible=False), gr.update(visible=True),
               "", "", "", "")
        return

    # ── Check whether queries were edited (literature use cases) ──
    queries_changed = False
    edited_queries: list[str] = []

    if (edited_queries_text
            and use_case in ("literature_finder", "literature_review",
                             "grant_proposal")
            and getattr(ctx, "task_plan", None)):
        # Parse edited queries (one per line, ignore empty lines)
        edited_queries = [
            q.strip() for q in edited_queries_text.strip().split("\n")
            if q.strip()
        ]
        # Get the original queries from the metadata
        meta_info = extract_plan_metadata_info(ctx.task_plan)
        original_queries = (
            (meta_info or {}).get("queries_used")
            or (meta_info or {}).get("generated_queries")
            or (meta_info or {}).get("user_queries")
            or []
        )
        queries_changed = (
            sorted(edited_queries) != sorted(original_queries)
        )

    # Clear the pending fields so that they are not run again
    app_state.pending_plan_ctx = None
    app_state.pending_plan_use_case = None
    app_state.pending_plan_label = None
    app_state.pending_institution_only = False
    app_state.pending_academic_only = False

    app_state.research_running = True

    # ── Queries changed → rebuild the plan with the new queries ──
    if queries_changed and edited_queries:
        logger.info(
            f"Queries edited ({len(edited_queries)} queries) "
            f"— repeating the literature search"
        )
        # Update the preflight data: write the edited queries back as
        # user queries
        if hasattr(ctx, "preflight_data") and ctx.preflight_data:
            ctx.preflight_data["search_queries_list"] = edited_queries
            ctx.preflight_data["search_queries"] = "\n".join(edited_queries)

        chatbot = list(chatbot or [])
        ChatState(chatbot).replace_last_assistant(
            (
                    f"🔄 **Search queries changed** — repeating the literature search "
                    f"with {len(edited_queries)} new queries..."
                ),
            append_if_missing=False,
        )

        yield (app_state, chatbot,
               gr.update(visible=False), gr.update(visible=True),
               gr.update(visible=True), gr.update(visible=True),
               "🔄 *Repeating the literature search with new queries...*",
               "", "", "")

        # Run the plan build again with the new queries
        run_orchestrator = app_state.orchestrator
        if run_orchestrator is None:
            run_orchestrator = ResearchOrchestrator(
                _shared_llm, _shared_connectors, _shared_config.pipeline,
                person_directory=_shared_directory,
                output_language=app_state.output_language,
            )
            app_state.orchestrator = run_orchestrator

        try:
            # Rebuild: plan_only=True, but with updated preflight_data
            # (new queries)
            ctx = await run_orchestrator.run(
                query=ctx.query or label,
                chat_history=app_state.chat_history,
                context_docs="",
                template_name=label,
                progress_callback=lambda *a, **k: None,
                mode=use_case,
                preflight_data=ctx.preflight_data,
                plan_only=True,
            )
        except Exception as e:
            logger.exception("Rebuild with new queries failed")
            ChatState(chatbot).replace_last_assistant(
                f"❌ Error in the repeated literature search: {e}",
            )
            app_state.research_running = False
            yield (app_state, chatbot,
                   gr.update(visible=True), gr.update(visible=False),
                   gr.update(visible=False), gr.update(visible=True),
                   f"❌ Rebuild failed: {e}", "", "", "")
            return

        if ctx is None or not getattr(ctx, "task_plan", None):
            ChatState(chatbot).replace_last_assistant(
                "⚠️ The repeated literature search produced no plan.",
            )
            app_state.research_running = False
            yield (app_state, chatbot,
                   gr.update(visible=True), gr.update(visible=False),
                   gr.update(visible=False), gr.update(visible=True),
                   "⚠️ Rebuild without a result", "", "", "")
            return

    # From here on: the normal confirm flow.

    # Distinguish: research mode or analysis use case?
    is_research_mode = use_case in ("web", "institution")
    icon = "🔍" if is_research_mode else _use_case_icon(use_case)

    # Short description for the chat message
    short_summary = ctx.query or label or ("Research" if is_research_mode else "Analysis")

    # Update the chat message — the "plan preview" entry is replaced by
    # the running status
    ChatState(chatbot).replace_last_assistant(
        f"{icon} **{label} running...** (plan confirmed)",
        append_if_missing=False,
    )

    yield (app_state, chatbot,
           gr.update(visible=False), gr.update(visible=True),
           gr.update(visible=True), gr.update(visible=True),
           f"⏳ *{label} running...*", "", "", "")

    # The user confirmed the plan: recorded in the report footer.
    ctx.plan_confirmed = True

    # Get the orchestrator from app_state (assigned when the plan was built)
    run_orchestrator = app_state.orchestrator
    if run_orchestrator is None:
        # Fallback: create a new one
        run_orchestrator = ResearchOrchestrator(
            _shared_llm, _shared_connectors, _shared_config.pipeline,
            person_directory=_shared_directory,
            output_language=app_state.output_language,
        )
        app_state.orchestrator = run_orchestrator

    if is_research_mode:
        # ── Research path ──
        # use_case is "web" or "institution" here. The persisted pending
        # flags take precedence; in the institution mode without an
        # explicitly set pending_institution_only, institution_only
        # defaults to True.
        institution_only = pending_institution_only or (use_case == "institution")
        use_directory = (use_case == "institution")
        academic_only = pending_academic_only

        async def research_invoker(on_progress):
            return await run_orchestrator.run(
                query=short_summary,
                chat_history=app_state.chat_history,
                context_docs=_get_doc_context(app_state),
                template_name=DEFAULT_TEMPLATE,
                progress_callback=on_progress,
                use_directory=use_directory,
                mode=use_case,
                institution_only=institution_only,
                academic_only=academic_only,
                existing_ctx=ctx,
                skip_decompose=True,
            )

        async for update in _drive_research_pipeline(
            app_state, chatbot, short_summary, research_invoker,
        ):
            yield update
        return

    # ── Analysis path ──
    async def analysis_invoker(on_progress):
        return await run_orchestrator.run(
            query=short_summary,
            chat_history=app_state.chat_history,
            context_docs="",
            template_name=label,
            progress_callback=on_progress,
            mode=use_case,
            existing_ctx=ctx,
            skip_decompose=True,
        )

    async for update in _drive_analysis_execution(
        app_state, chatbot, label, icon, short_summary, analysis_invoker,
    ):
        yield update


async def cancel_pending_plan(app_state: AppState, chatbot: list):
    """Handler for the cancel button of the plan preview gate.

    Discards the held plan and resets the UI state. Returns updates for
    (app_state, chatbot, start_btn, stop_btn, plan_gate_group, plan_md,
    confirm_btn, cancel_btn).
    """
    app_state = _get_ready_state(app_state)

    was_pending = app_state.pending_plan_ctx is not None
    label = app_state.pending_plan_label or "Analysis"

    app_state.pending_plan_ctx = None
    app_state.pending_plan_use_case = None
    app_state.pending_plan_label = None
    app_state.pending_institution_only = False
    app_state.pending_academic_only = False
    app_state.research_running = False

    chatbot = list(chatbot or [])
    if was_pending:
        ChatState(chatbot).replace_last_assistant(
            f"✖ **{label} cancelled** (plan not confirmed)",
            append_if_missing=False,
        )

    return (
        app_state,
        chatbot,
        gr.update(visible=True),   # start_btn (visible again)
        gr.update(visible=False),  # stop_btn (hidden)
        gr.update(visible=False),  # plan_gate_group
        gr.update(visible=False),  # plan_md
        gr.update(visible=False),  # query_edit_box
        gr.update(visible=False),  # confirm_btn
        gr.update(visible=False),  # cancel_btn
    )


def _show_gate_if_pending(app_state: AppState):
    """Helper: read pending_plan_ctx and return updates for the gate UI
    components (plan_gate_group, plan_md, query_edit_box, confirm_btn,
    cancel_btn).

    Called via .then() after run_research. If a plan is waiting, the
    gate components are made visible; otherwise they stay hidden.

    Supports both plan types:
    - analysis plans (TaskPlan in ctx.task_plan) — via format_task_plan_markdown
    - research plans (ResearchPlan in ctx.research_plan) — via
      format_research_plan_markdown
    """
    from src.ui.components.plan_preview import (
        extract_plan_metadata_info,
        format_task_plan_markdown,
        format_research_plan_markdown,
    )

    app_state = _get_ready_state(app_state)

    ctx = app_state.pending_plan_ctx
    if ctx is None:
        return (
            gr.update(visible=False),  # plan_gate_group
            gr.update(visible=False),  # plan_md
            gr.update(visible=False),  # query_edit_box
            gr.update(visible=False),  # confirm_btn
            gr.update(visible=False),  # cancel_btn
        )

    plan_md_text: Optional[str] = None
    use_case = app_state.pending_plan_use_case or ""
    query_box_update = gr.update(visible=False)  # default: hidden

    # Research path: research_plan present and use_case is web/institution
    if use_case in ("web", "institution") and getattr(ctx, "research_plan", None):
        plan_md_text = format_research_plan_markdown(
            ctx.research_plan,
            getattr(ctx, "search_stats", None),
            mode=use_case,
            academic_only=app_state.pending_academic_only,
        )
    # Analysis path: task_plan present
    elif getattr(ctx, "task_plan", None):
        plan_md_text = format_task_plan_markdown(ctx.task_plan)

        # Literature use cases: make the queries editable
        if use_case in ("literature_finder", "literature_review",
                        "grant_proposal"):
            meta_info = extract_plan_metadata_info(ctx.task_plan)
            if meta_info:
                # Extract queries from the metadata
                queries = (
                    meta_info.get("queries_used")
                    or meta_info.get("generated_queries")
                    or meta_info.get("user_queries")
                    or []
                )
                if queries:
                    query_box_update = gr.update(
                        value="\n".join(queries),
                        visible=True,
                    )

    if plan_md_text:
        return (
            gr.update(visible=True),                        # plan_gate_group
            gr.update(value=plan_md_text, visible=True),    # plan_md
            query_box_update,                               # query_edit_box
            gr.update(visible=True),                        # confirm_btn
            gr.update(visible=True),                        # cancel_btn
        )
    return (
        gr.update(visible=False),  # plan_gate_group
        gr.update(visible=False),  # plan_md
        gr.update(visible=False),  # query_edit_box
        gr.update(visible=False),  # confirm_btn
        gr.update(visible=False),  # cancel_btn
    )


def _hide_gate_after_run():
    """Helper: hide the gate UI components again after the confirm
    handler."""
    return (
        gr.update(visible=False),  # plan_gate_group
        gr.update(visible=False),  # plan_md
        gr.update(visible=False),  # query_edit_box
        gr.update(visible=False),  # confirm_btn
        gr.update(visible=False),  # cancel_btn
    )




def _use_case_label(use_case: str) -> str:
    return {
        "peer_review": "Peer Review",
        "decision_analysis": "Decision analysis",
        "research_design": "Research design",
        "grant_proposal": "Grant proposal",
        "literature_review": "Literature Review",
        "explainer": "In-depth explanation",
    }.get(use_case, use_case)


def _use_case_icon(use_case: str) -> str:
    return {
        "peer_review": "🔍",
        "decision_analysis": "⚖️",
        "research_design": "🔬",
        "grant_proposal": "💰",
        "literature_review": "📚",
        "explainer": "📖",
    }.get(use_case, "🔬")


async def run_literature_check(app_state: AppState, msg: dict, chatbot: list,
                               system_prompt: str):
    """Start the bibliography check."""
    from src.pipeline.literature_check import LiteratureChecker
    from src.connectors.literature_apis import LiteratureReport

    app_state = _get_ready_state(app_state)

    text, docs_added = await _process_input(app_state, msg)

    # No text, but documents uploaded → text from the documents
    if not text and docs_added and app_state.documents:
        doc_parts = []
        for doc_id, doc in app_state.documents.items():
            if doc.get("content"):
                doc_parts.append(doc["content"])
        text = "\n\n".join(doc_parts)

    if not text:
        gr.Warning("Please enter a bibliography (text or file upload).")
        yield (app_state, chatbot,
               gr.update(visible=True), gr.update(visible=False),
               gr.update(visible=False), gr.update(visible=True),
               "", "", "", "")
        return

    # Chat message
    chatbot = list(chatbot or [])
    short_preview = text[:200].replace('\n', ' ')
    chatbot.append({"role": "user", "content": f"📚 *Literature check:* {short_preview}..."})
    chatbot.append({"role": "assistant", "content": "📚 **Literature check started...**"})
    app_state.research_running = True

    yield (app_state, chatbot,
           gr.update(visible=False), gr.update(visible=True),
           gr.update(visible=True), gr.update(visible=True),
           "⏳ *Preparing the literature check...*", "", "", "")

    # Progress
    progress_md = ""
    report_md = ""
    live_lines: list[str] = []
    _update_ready = asyncio.Event()

    async def on_progress(phase: str, data):
        nonlocal progress_md, report_md
        if phase == "status":
            live_lines.append(str(data))
            progress_md += f"\n{data}\n"
            _update_ready.set()
        elif phase == "report_stream":
            report_md = data
            _update_ready.set()

    # Create the checker
    searxng = app_state.connectors.get_search_connector() if app_state.connectors else None
    # Reset the LLM counters for clean statistics
    if hasattr(app_state.llm, 'reset_usage'):
        app_state.llm.reset_usage()
    checker = LiteratureChecker(
        llm=app_state.llm,
        searxng=searxng,
        lang=app_state.output_language,
    )
    app_state._lit_checker = checker

    # Pipeline as a background task
    pipeline_result = None
    pipeline_error = None

    async def run_pipeline():
        nonlocal pipeline_result, pipeline_error
        try:
            pipeline_result = await checker.check(
                raw_text=text,
                progress_callback=on_progress,
            )
        except Exception as e:
            pipeline_error = e
        finally:
            await checker.close()
            _update_ready.set()

    task = asyncio.create_task(run_pipeline())

    # Polling-Loop
    while not task.done():
        try:
            await asyncio.wait_for(_update_ready.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pass
        _update_ready.clear()

        # Show the report if present, otherwise the progress
        if report_md:
            display_md = report_md
        else:
            display_md = "## 📚 Literature check running...\n\n" + "\n\n".join(live_lines[-15:])

        yield (app_state, chatbot,
               gr.update(visible=False), gr.update(visible=True),
               gr.update(visible=True), gr.update(visible=True),
               display_md, "", progress_md or "⏳ Running...", "")

    # Result
    if pipeline_error:
        logger.error(f"Literature check failed: {pipeline_error}",
                      exc_info=pipeline_error)
        chatbot = list(chatbot[:-1])
        chatbot.append({"role": "assistant",
                        "content": f"⚠️ Literature check failed: {pipeline_error}"})
        app_state.research_running = False
        yield (app_state, chatbot,
               gr.update(visible=True), gr.update(visible=False),
               gr.update(visible=True), gr.update(visible=True),
               f"⚠️ Error: {pipeline_error}", "", progress_md, "")
        return

    if not pipeline_result:
        app_state.research_running = False
        yield (app_state, chatbot,
               gr.update(visible=True), gr.update(visible=False),
               gr.update(visible=True), gr.update(visible=True),
               report_md or "⚠️ No result.", "", progress_md, "")
        return

    report_text, report_data = pipeline_result

    # Safety check
    if not report_data:
        report_data = LiteratureReport()
    if not report_text:
        report_text = report_md or "*No report generated.*"

    # ── Sources tab: which APIs were queried? ──
    sources_md = "## 🔗 Databases queried\n\n"
    sources_md += (
        "| # | Entry | arXiv | CrossRef | OpenAlex | Sem.Scholar | DBLP | Hits |\n"
        "|---|-------|-------|----------|----------|-------------|------|------|\n"
    )
    for entry in report_data.entries:
        short = f"{str(entry.authors[0]).split(',')[0] if entry.authors else '?'} ({entry.year})"
        apis_checked = entry.checked_sources or []
        apis_matched = [m["source"] for m in entry.api_matches]
        def _icon(api_name):
            if api_name in apis_matched:
                return "✅"
            elif api_name in apis_checked:
                return "❌"
            return "—"
        sources_md += (
            f"| {entry.id} | {short} "
            f"| {_icon('arXiv')} | {_icon('CrossRef')} | {_icon('OpenAlex')} "
            f"| {_icon('Semantic Scholar')} | {_icon('DBLP')} "
            f"| {len(entry.api_matches)} |\n"
        )
    sources_md += "\n✅ = hit, ❌ = searched/not found, — = not queried\n"

    # Links to the entries found
    sources_md += "\n### URLs found\n\n"
    for entry in report_data.entries:
        for match in entry.api_matches:
            url = match.get("url", "")
            if url:
                sources_md += f"- **{match['source']}**: [{entry.title[:60]}...]({url})\n"

    # ── Extracts tab: detailed API matches ──
    extracts_md = "## 📝 API data per entry\n\n"
    for entry in report_data.entries:
        status_icon = {"verified": "✅", "url_verified": "🟡",
                       "deviations": "⚠️",
                       "not_found": "❌", "error": "💥"}.get(entry.status, "❓")
        extracts_md += (
            f"### {entry.id}. {str(entry.authors[0]) if entry.authors else '?'} "
            f"({entry.year}) {status_icon}\n\n"
        )
        extracts_md += f"**Title:** {entry.title}\n\n"

        if entry.api_matches:
            for match in entry.api_matches:
                extracts_md += f"**{match['source']}:**\n"
                extracts_md += f"- Title: {match.get('title', '—')}\n"
                m_authors = match.get('authors', [])
                if isinstance(m_authors, list):
                    extracts_md += f"- Authors: {', '.join(str(a) for a in m_authors[:5])}"
                    if len(m_authors) > 5:
                        extracts_md += f" (+{len(m_authors)-5} more)"
                    extracts_md += "\n"
                extracts_md += f"- Year: {match.get('year', '—')}\n"
                extracts_md += f"- Journal: {match.get('journal', '—')}\n"
                if match.get('doi'):
                    extracts_md += f"- DOI: {match.get('doi')}\n"
                if match.get('url'):
                    extracts_md += f"- URL: [{match['url'][:60]}]({match['url']})\n"
                extracts_md += "\n"

        if entry.deviations:
            extracts_md += "**Deviations:**\n"
            for dev in entry.deviations:
                conf = dev.get("confidence", "")
                conf_tag = f" ({conf})" if conf else ""
                extracts_md += f"- {dev.get('message', dev.get('field', '?'))}{conf_tag}\n"
            extracts_md += "\n"

        if entry.notes:
            extracts_md += f"**Notes:** {'; '.join(entry.notes)}\n\n"

        extracts_md += "---\n\n"

    # ── Chat summary ──
    url_v = sum(1 for e in report_data.entries if e.status == "url_verified")
    summary_lines = [
        "✅ **Literature check completed**\n",
        f"- {report_data.total} {'entry' if report_data.total == 1 else 'entries'} checked",
        f"- {report_data.verified} verified (database) ✅",
    ]
    if url_v:
        summary_lines.append(f"- {url_v} URL confirmed (fields not checked) 🟡")
    summary_lines.extend([
        f"- {report_data.with_deviations} with deviations ⚠️",
        f"- {report_data.not_found} not found ❌",
        "",
        "*The result is in the “Report” tab on the right, the export at the top of the result area.*",
    ])
    summary = "\n".join(summary_lines)
    chatbot = list(chatbot[:-1])
    chatbot.append({"role": "assistant", "content": summary})

    # Minimal research object for the Word/Markdown export
    ctx = HarvestContext(query=f"Literature check ({report_data.total} entries)")
    # Full report with appendices for the Word export
    ctx.final_report = (
        (report_text or "")
        + "\n\n---\n\n"
        + sources_md
        + "\n\n---\n\n"
        + extracts_md
    )
    ctx.output_schema = OutputSchema(
        title="Bibliography check report",
        format_type="literature_check",
        style="academic",
    )
    ctx.status = "done"
    ctx.finished_at = datetime.now().isoformat()
    # Statistics for the metadata appendix
    ctx.search_stats = {
        "type": "literature_check",
        "api_stats": report_data.api_stats,
        "llm_stats": report_data.llm_stats,
        # Built by the check with the best API match per entry
        "bibtex": report_data.bibtex,
    }
    app_state.current_research = ctx

    app_state.research_running = False

    yield (app_state, chatbot,
           gr.update(visible=True), gr.update(visible=False),
           gr.update(visible=True), gr.update(visible=True),
           report_text or "*No report.*",
           sources_md, progress_md, extracts_md)


def toggle_sidebar(app_state: AppState):
    app_state = _get_ready_state(app_state)
    app_state.sidebar_visible = not app_state.sidebar_visible
    return app_state, gr.update(visible=app_state.sidebar_visible)


def toggle_result_panel(app_state: AppState):
    app_state = _get_ready_state(app_state)
    app_state.result_panel_visible = not app_state.result_panel_visible
    return app_state, gr.update(visible=app_state.result_panel_visible)


def _welcome_message() -> dict:
    """Welcome message (built at call time: depends on the institution profile)."""
    return {
        "role": "assistant",
        "content": (
            f"👋 Welcome to **{TOOL_NAME}**\n\n"
            "Choose a mode from the dropdown below, enter your input "
            "and click **🔍 Start research**. For web research you can "
            "sharpen your request with me first via **💬 Discuss the request** "
            "before starting.\n\n"
            "---\n\n"
            "### 🔎 Research modes\n\n"
            "**🌐 Web research** — general online search with an autonomous "
            "pipeline. Delivers a report with sources. Good for current "
            "topics, market information, general knowledge.\n\n"
            + _institution_help_text() +
            "**📚 Check references** — checks existing bibliographies "
            "(entry by entry) against CrossRef, OpenAlex and further "
            "databases. Finds errors, adds DOIs. Good before submitting "
            "manuscripts.\n\n"
            "**📑 Find literature** — searches OpenAlex, Semantic Scholar "
            "and arXiv for your research question, assesses the hits "
            "against inclusion criteria and lists the selection with full "
            "bibliographic details. Good for a first overview "
            "before a review.\n\n"
            "---\n\n"
            "### 🧠 Analysis modes\n\n"
            "**📖 In-depth explanation** — explains a topic for a specific "
            "audience with a structured build-up. Good for teaching preparation "
            "or getting into new fields.\n\n"
            "**🔍 Peer review** — writes a structured review of "
            "a paper, aspect by aspect. Good for "
            "journal reviews, conference reviews, theses.\n\n"
            "**⚖️ Decision analysis** — structured multi-criteria "
            "analysis of 2–8 options against your criteria. Good for "
            "technology choices and strategic decisions.\n\n"
            "**🔬 Research design** — develops a methodological research design "
            "from a research question: literature gap, hypotheses, method, "
            "limitations. Good for project conception and "
            "proposal outlines.\n\n"
            "**💰 Grant proposal** — proposal draft with a real literature "
            "search for the state of research, work plan and "
            "coherence check. Good for proposals in the concept phase.\n\n"
            "**📚 Literature review** — literature review based on a real "
            "literature search (with citation chasing): a synthesis per key "
            "question and a meta-synthesis. Good for thesis chapters and overviews."
        ),
    }



def new_chat(app_state: AppState):
    app_state = _get_ready_state(app_state)
    app_state.chat_history = []
    return app_state, [_welcome_message()]


# ─── Chat history in the browser (survives a page reload) ─────────
#
# The chat lives in gr.State, which is bound to one page load. A copy is
# kept in the browser's localStorage via gr.BrowserState: written at the
# end of every handler chain that changes the chat (not per streamed
# chunk), read back on page load. It stays in this one browser; other
# devices and other people do not see it.
#
# Opt-in: only with BROWSER_STORAGE_SECRET set. Without it nothing is
# stored and the app behaves as before — the safe default on shared
# computers, and no data encrypted with a random per-start key that a
# restart would make unreadable.

CHAT_STORAGE_KEY = "research-toolset-chat"
#: Upper bounds so that localStorage (about 5 MB per origin) never fills up.
CHAT_STORE_MAX_MESSAGES = 200
CHAT_STORE_MAX_HISTORY = 60

_WELCOME_PREFIXES = ("👋 Welcome",)


def _browser_storage_secret() -> Optional[str]:
    """Encryption secret for chat and result in the browser (None = off)."""
    return os.environ.get("BROWSER_STORAGE_SECRET", "").strip() or None


def browser_storage_enabled() -> bool:
    return _browser_storage_secret() is not None


def _message_text(content) -> str:
    """Flatten chatbot message content (str or list of parts) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or "")
    if isinstance(content, (list, tuple)):
        return "\n".join(_message_text(p) for p in content if p)
    return ""


def _is_transient(text: str) -> bool:
    """Welcome message and thinking placeholder are not worth storing."""
    t = text.strip()
    return not t or t == THINKING_PLACEHOLDER or t.startswith(_WELCOME_PREFIXES)


def save_chat_to_browser(chatbot: list, app_state: AppState):
    """Snapshot of the chat for gr.BrowserState (JSON only)."""
    if not browser_storage_enabled():
        return gr.skip()
    messages = []
    for m in chatbot or []:
        if not isinstance(m, dict) or m.get("role") not in ("user", "assistant"):
            continue
        text = _message_text(m.get("content"))
        if _is_transient(text):
            continue
        messages.append({"role": m["role"], "content": text})
    history = []
    for m in getattr(app_state, "chat_history", None) or []:
        text = _message_text(m.get("content"))
        if m.get("role") in ("user", "assistant") and text.strip():
            history.append({"role": m["role"], "content": text})
    return {
        "version": 1,
        "chatbot": messages[-CHAT_STORE_MAX_MESSAGES:],
        "history": history[-CHAT_STORE_MAX_HISTORY:],
    }


def restore_chat_from_browser(stored, app_state: AppState):
    """Rebuild chat display and LLM context from the stored snapshot."""
    app_state = _get_ready_state(app_state)
    chatbot = [_welcome_message()]
    if isinstance(stored, dict) and stored.get("version") == 1:
        for m in stored.get("chatbot") or []:
            if (isinstance(m, dict) and m.get("role") in ("user", "assistant")
                    and isinstance(m.get("content"), str)):
                chatbot.append({"role": m["role"], "content": m["content"]})
        app_state.chat_history = [
            {"role": m["role"], "content": m["content"]}
            for m in stored.get("history") or []
            if isinstance(m, dict) and m.get("role") in ("user", "assistant")
            and isinstance(m.get("content"), str)
        ]
    return app_state, chatbot


# ─── Last result in the browser (survives a page reload) ──────────
#
# Same mechanism and secret as the chat, own storage key. Only the last
# result is kept, as finished text: the tabs as they were shown, the
# pipeline-run view, the report as exported and the BibTeX of a reference
# check. The research object itself (sources, extracts, statistics) is
# not stored; a restored result gets a minimal HarvestContext that is
# enough for the exports (the Word metadata appendices are then empty).

RESULT_STORAGE_KEY = "research-toolset-result"
#: Upper bound for the stored result (UTF-8 bytes of its JSON).
RESULT_STORE_MAX_BYTES = 250_000
#: Shortened first when the result is too big; the report comes last.
_RESULT_TRIM_ORDER = ("pipeline_run", "extracts", "progress", "sources",
                      "bibtex", "report", "export_md")
_RESULT_TEXT_FIELDS = ("report", "sources", "progress", "extracts",
                       "pipeline_run", "bibtex")
_RESULT_MAX_QUERY = 2000


def _json_size(value) -> int:
    return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))


def _fit_result_snapshot(snap: dict) -> dict:
    """Shorten the texts in _RESULT_TRIM_ORDER until the snapshot fits."""
    note = "*… shortened to fit into the browser storage.*"
    for key in _RESULT_TRIM_ORDER:
        over = _json_size(snap) - RESULT_STORE_MAX_BYTES
        if over <= 0:
            break
        text = snap.get(key) or ""
        if not text:
            continue
        # Every dropped character frees at least one byte of JSON.
        keep = len(text) - over - _json_size("\n\n" + note) - _json_size(key) - 2
        # BibTeX only in whole entries, everything else in whole lines
        cut = "\n@" if key == "bibtex" else "\n"
        head = text[:keep].rsplit(cut, 1)[0] if keep > 0 else ""
        snap[key] = f"{head}\n\n{note}" if head else note
        snap["truncated"].append(key)
    return snap


def build_result_snapshot(ctx, report: str, sources: str, progress: str,
                          extracts: str, pipeline_run: str) -> dict:
    """Snapshot of the finished result for gr.BrowserState (JSON only)."""
    schema = getattr(ctx, "output_schema", None)
    report = report or ""
    export_md = ctx.final_report or ""
    is_check = bool(schema and schema.format_type == "literature_check")
    bibtex = (ctx.search_stats or {}).get("bibtex") if is_check else ""
    snap = {
        "version": 1,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "query": (ctx.query or "")[:_RESULT_MAX_QUERY],
        "title": (schema.title if schema else "")[:_RESULT_MAX_QUERY],
        "format_type": schema.format_type if schema else "",
        "style": schema.style if schema else "",
        "output_language": ctx.output_language or "",
        "started_at": ctx.started_at or "",
        "finished_at": ctx.finished_at or "",
        "report": report,
        # Usually the shown report plus banners; stored once if identical.
        "export_md": None if export_md == report else export_md,
        "sources": sources or "",
        "progress": progress or "",
        "extracts": extracts or "",
        "pipeline_run": pipeline_run or "",
        "bibtex": bibtex if isinstance(bibtex, str) else "",
        "truncated": [],
    }
    return _fit_result_snapshot(snap)


def _valid_result_snapshot(stored) -> Optional[dict]:
    """The stored snapshot with every field in shape, or None."""
    if not isinstance(stored, dict) or stored.get("version") != 1:
        return None
    if not isinstance(stored.get("report"), str) or not stored["report"].strip():
        return None
    snap = {k: stored.get(k) if isinstance(stored.get(k), str) else ""
            for k in ("saved_at", "query", "title", "format_type", "style",
                      "output_language", "started_at", "finished_at",
                      *_RESULT_TEXT_FIELDS)}
    export_md = stored.get("export_md")
    snap["export_md"] = export_md if isinstance(export_md, str) else None
    truncated = stored.get("truncated")
    snap["truncated"] = ([t for t in truncated if isinstance(t, str)]
                         if isinstance(truncated, list) else [])
    return snap


def _context_from_snapshot(snap: dict) -> HarvestContext:
    """Minimal research object for the exports of a restored result."""
    ctx = HarvestContext(
        query=snap["query"],
        output_language=snap["output_language"] or _default_output_language(),
    )
    ctx.final_report = (snap["export_md"] if snap["export_md"] is not None
                        else snap["report"])
    if snap["title"] or snap["format_type"]:
        ctx.output_schema = OutputSchema(
            title=snap["title"],
            format_type=snap["format_type"] or "structured_report",
            style=snap["style"] or OutputSchema.style,
            language=ctx.output_language,
        )
    try:
        # duration_seconds parses both; keep the original run times.
        datetime.fromisoformat(snap["started_at"])
        datetime.fromisoformat(snap["finished_at"])
        ctx.started_at, ctx.finished_at = snap["started_at"], snap["finished_at"]
    except ValueError:
        ctx.finished_at = ctx.started_at
    ctx.status = "done"
    return ctx


def _restored_snapshot(app_state) -> Optional[dict]:
    """The stored texts if the shown result was restored from the browser."""
    snap = getattr(app_state, "restored_result", None)
    ctx = getattr(app_state, "current_research", None)
    if snap and ctx is not None and ctx is app_state.browser_result_ctx:
        return snap
    return None


def _render_pipeline_run_text(ctx) -> str:
    try:
        from src.ui.pipeline_run import render_pipeline_run
        return render_pipeline_run(ctx)
    except Exception as e:
        logger.warning("Pipeline-run rendering for the browser copy failed: %s", e)
        return ""


def save_result_to_browser(app_state: AppState, report: str, sources: str,
                           progress: str, extracts: str):
    """Store the last result once, when a run has produced a new one.

    Runs at the end of every research chain. Nothing changes when the
    chain produced no new result (plan gate, autofill stop, error).
    """
    ctx = getattr(app_state, "current_research", None)
    if (not browser_storage_enabled() or ctx is None
            or app_state.research_running
            or ctx is app_state.browser_result_ctx):
        return gr.skip(), gr.skip()
    snap = build_result_snapshot(ctx, report, sources, progress, extracts,
                                 _render_pipeline_run_text(ctx))
    app_state.browser_result_ctx = ctx
    app_state.restored_result = None
    return app_state, snap


def _restored_banner(snap: dict) -> str:
    try:
        when = datetime.fromisoformat(snap["saved_at"]).strftime(
            "%Y-%m-%d %H:%M")
    except ValueError:
        when = snap["saved_at"] or "?"
    text = (f"↩️ **Restored result** from {when}: kept in this browser and "
            "shown again after reloading the page. The Word export has no "
            "metadata appendices.")
    if snap["truncated"]:
        text += " Parts were shortened to fit into the browser storage."
    return f"> {text}\n\n"


def restore_result_from_browser(stored, app_state: AppState):
    """Reopen the result area with the stored last result, if any.

    Returns updates for app_state, the result panel, the four tabs and
    the pipeline-run view.
    """
    snap = _valid_result_snapshot(stored)
    if snap is None:
        return (gr.skip(),) * 7
    app_state = _get_ready_state(app_state)
    ctx = _context_from_snapshot(snap)
    app_state.current_research = ctx
    app_state.browser_result_ctx = ctx
    app_state.restored_result = snap
    app_state.result_panel_visible = True
    return (
        app_state,
        gr.update(visible=True),
        _restored_banner(snap) + snap["report"],
        snap["sources"],
        snap["progress"],
        snap["extracts"],
        snap["pipeline_run"] or "*No research started yet.*",
    )


def _empty_result_texts() -> tuple[str, str, str, str, str]:
    """Report, sources, progress, extracts and pipeline run before a run."""
    return (
        "*Start a research run...*",
        "*No sources yet.*",
        "*Waiting for the research to start...*",
        "*No extracts yet.*",
        "*No research started yet.*",
    )


def clear_result(app_state: AppState):
    """"New chat": forget the result, in the browser and on the page.

    The browser copy is overwritten with an empty dict: Gradio's browser
    side does not write falsy values, so None would leave it in place.
    A running research keeps its result area; it is stored when done.

    Returns updates for app_state, the browser copy, the result panel,
    the four tabs, the pipeline-run view and the download field.
    """
    app_state = _get_ready_state(app_state)
    if app_state.research_running:
        # still the previous result; only the next one gets stored
        app_state.browser_result_ctx = app_state.current_research
        return (app_state, {}) + (gr.skip(),) * 7
    app_state.current_research = None
    app_state.browser_result_ctx = None
    app_state.restored_result = None
    app_state.result_panel_visible = False
    return (app_state, {}, gr.update(visible=False), *_empty_result_texts(),
            gr.update(value=None, visible=False))


def _word_export_context(app_state: AppState):
    """The research object for the Word export.

    A restored result has only its texts: the report plus the sources and
    extracts tabs as appendices, like the reference check does it.
    """
    ctx = app_state.current_research
    snap = _restored_snapshot(app_state)
    if snap is None or (ctx.output_schema and
                        ctx.output_schema.format_type == "literature_check"):
        return ctx  # the reference check report already has them
    parts = [ctx.final_report or ""]
    parts += [snap[k] for k in ("sources", "extracts") if snap[k].strip()]
    word_ctx = copy.copy(ctx)
    word_ctx.final_report = "\n\n---\n\n".join(parts)
    return word_ctx


def store_and_clear(msg, paste_buf):
    """Store the message in the state and clear the input immediately
    (queue=False). paste_buf holds fallback text from the JS paste
    interceptor."""
    if paste_buf and paste_buf.strip():
        # The paste buffer has content → add it to msg as a fallback
        if msg is None:
            msg = {"text": paste_buf.strip(), "files": []}
        elif isinstance(msg, dict):
            current_text = (msg.get("text", "") or "").strip()
            if not current_text:
                msg["text"] = paste_buf.strip()
            elif paste_buf.strip() not in current_text:
                # Pasted text is not in the regular text → append it
                msg["text"] = current_text + "\n\n" + paste_buf.strip()
        elif isinstance(msg, str):
            if not msg.strip():
                msg = paste_buf.strip()
    return msg, gr.MultimodalTextbox(value=None), ""


EXTRACT_QUERY_PROMPT = """Extract ONLY the actual research brief from the following text.

Remove:
- introductory courtesies ("Sure, here is...", "Happy to help...", etc.)
- framing text and explanations ("The revised research request is:", etc.)
- closing hints ("Click Start research", action lines with emojis, etc.)
- meta comments about the brief itself

Keep ONLY the connected, concrete research brief — the substance
that works directly as the input for a web research run. Keep its language.

Answer ONLY with the cleaned-up text, without quotation marks, without explanations.

TEXT:
{text}"""


async def adopt_last_response(app_state: AppState, chatbot: list):
    """Copy the last assistant answer, cleaned up, into the input field.

    Uses the LLM to remove framing text and extract only the research
    request itself.
    """
    logger.info("adopt_last_response called")
    app_state = _get_ready_state(app_state)

    if not chatbot:
        gr.Warning("No chat messages yet.")
        return gr.update()

    # Find the last relevant assistant answer
    content = ""
    for msg in reversed(chatbot):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            raw = msg.get("content", "")
            if isinstance(raw, list):
                parts = [p if isinstance(p, str) else p.get("text", "")
                         for p in raw if isinstance(p, (str, dict))]
                raw = "\n".join(parts)
            if not isinstance(raw, str):
                continue
            raw = raw.strip()
            # Skip status messages
            if not raw or any(raw.startswith(p) for p in (
                "🔍", "✅", "⏳", "🤔", "📚 **Literature", "⚠️",
                "👋 Welcome",
            )):
                continue
            content = raw
            break

    if not content:
        gr.Warning("No suitable assistant answer found.")
        return gr.update()

    # Short texts (< 200 characters) are taken directly — no LLM needed
    if len(content) < 200:
        gr.Info("📋 Copied into the input field")
        return gr.MultimodalTextbox(value={"text": content, "files": []})

    # LLM clean-up: remove framing text (small model = faster)
    try:
        if app_state.llm:
            cleaned = await app_state.llm.harvest_complete(
                [{"role": "user",
                  "content": EXTRACT_QUERY_PROMPT.format(text=content)}],
                max_tokens=2048,
            )
            cleaned = cleaned.strip()
            if cleaned and len(cleaned) > 20:
                gr.Info("📋 Cleaned up and copied into the input field")
                return gr.MultimodalTextbox(value={"text": cleaned, "files": []})
    except Exception as e:
        logger.warning(f"LLM clean-up failed: {e}")

    # Fallback: take it uncleaned
    gr.Info("📋 Copied into the input field (not cleaned up)")
    return gr.MultimodalTextbox(value={"text": content, "files": []})


async def sidebar_upload_files(app_state: AppState, files):
    """Document upload via the sidebar."""
    app_state = _get_ready_state(app_state)
    if not files:
        yield app_state, gr.update(), gr.update(), gr.update()
        return

    file_list = files if isinstance(files, list) else [files]

    for filepath in file_list:
        if not filepath or not os.path.exists(filepath):
            continue
        filename = os.path.basename(filepath)
        ext = Path(filepath).suffix.lower()

        if ext not in DOCUMENT_EXTENSIONS:
            gr.Warning(f"⚠️ {filename}: format not supported")
            continue

        try:
            # Simple text extraction
            content = ""
            for encoding in ["utf-8", "latin-1", "cp1252"]:
                try:
                    content = Path(filepath).read_text(encoding=encoding)
                    break
                except (UnicodeDecodeError, Exception):
                    continue

            if not content:
                # For PDFs etc.: unstructured or pdfminer
                try:
                    from src.documents.processor import DocumentProcessor
                    from src.config import ProcessorConfig
                    proc = DocumentProcessor(ProcessorConfig())
                    processed = proc.process_document(filepath)
                    content = processed.content
                except Exception:
                    content = f"[Could not process {filename}]"

            tokens = max(1, len(content) // 3)
            doc_id = str(uuid.uuid4())[:8]
            app_state.documents[doc_id] = {
                "filename": filename,
                "content": content,
                "tokens": tokens,
            }
            app_state.total_doc_tokens += tokens
            app_state.sidebar_visible = True
            gr.Info(f"✅ {filename} ({tokens:,} Tokens)")

        except Exception as e:
            gr.Warning(f"⚠️ {filename}: {e}")

    yield (
        app_state,
        _build_doc_list_html(app_state),
        _build_token_bar_html(app_state),
        gr.update(value=None),
    )


def export_markdown(app_state: AppState):
    if not app_state or not app_state.current_research:
        gr.Warning("No research to export.")
        return None
    ctx = app_state.current_research
    # Writes to GRADIO_TEMP_DIR (default: tempfile.gettempdir()). Gradio
    # tracks the file and deletes it automatically via delete_cache.
    gradio_temp = os.environ.get("GRADIO_TEMP_DIR") or tempfile.gettempdir()
    os.makedirs(gradio_temp, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", prefix=f"research_{ctx.id}_",
        delete=False, dir=gradio_temp, encoding="utf-8",
    ) as f:
        f.write(ctx.final_report or "")
        filepath = f.name
    gr.Info("✅ Markdown export created.")
    return filepath


def export_word(app_state: AppState):
    if not app_state or not app_state.current_research:
        gr.Warning("No research to export.")
        return None
    try:
        from src.exporters.word_exporter import export_research_to_word
        ctx = _word_export_context(app_state)
        filepath = export_research_to_word(ctx)
        if filepath:
            gr.Info("✅ Word export created.")
            return filepath
        gr.Warning("⚠️ Word export: no file produced.")
        return None
    except Exception as e:
        logger.error(f"Word export error: {e}", exc_info=True)
        gr.Warning(f"⚠️ Word export failed: {e}")
        return None


async def autofill_analysis_form(app_state, msg, research_mode, *analysis_args):
    """Fill empty mandatory fields when "Start research" is clicked.

    Runs as an intermediate step of the click chain, BEFORE
    `run_research`. Gradio passes the updated component values to the
    next step, so the run would use the suggestions directly — which is
    exactly what it must not do here.

    Behaviour:
      - no analysis mode, or all mandatory fields filled
        → do nothing, no LLM call, the run starts as usual.
      - at least one mandatory field empty → suggest from the chat, fill
        ONLY empty fields and set `app_state.autofill_pending`.
        `run_research` then stops and asks for a review.

    Why not start right away: an analysis run takes minutes and costs
    tokens. If a field derived from the chat is off, one would otherwise
    only notice it in the result. Whoever filled in the form themselves
    does not notice this step at all.
    """
    from src.pipeline.analysis_pipeline import USE_CASE_REGISTRY
    from src.ui.preflight_autofill import (
        empty_required_fields, plan_form_autofill, suggest_preflight_values,
    )

    unchanged = tuple(gr.update() for _ in analysis_args)

    use_case = ANALYSIS_MODE_MAP.get(research_mode)
    if not use_case or use_case not in USE_CASE_REGISTRY:
        return unchanged

    app_state = _get_ready_state(app_state)
    app_state.autofill_pending = 0
    app_state.autofill_note = ""

    checker = USE_CASE_REGISTRY[use_case]["preflight"]
    requirements = checker.get_requirements()

    # Position of this use case in the flat component list
    split = _split_analysis_args(analysis_args)
    offset = 0
    for name in ANALYSIS_USE_CASE_ORDER:
        if name == use_case:
            break
        if name in USE_CASE_REGISTRY:
            offset += len(USE_CASE_REGISTRY[name]["preflight"].get_requirements())
    current = split.get(use_case, [])

    if not empty_required_fields(requirements, current):
        return unchanged

    if not getattr(app_state, "llm", None):
        return unchanged

    extra = ""
    if isinstance(msg, dict):
        extra = msg.get("text") or ""
    elif isinstance(msg, str):
        extra = msg

    try:
        values, message = await suggest_preflight_values(
            checker, getattr(app_state, "chat_history", []),
            app_state.llm, extra_text=extra,
        )
    except Exception as e:
        logger.error("Autofill at start failed: %s", e, exc_info=True)
        return unchanged

    # Fill ONLY empty fields — what the person typed stays.
    by_index = plan_form_autofill(requirements, current, values)
    filled = {requirements[i].field: v for i, v in by_index.items()}
    if not filled:
        # Nothing could be derived: stop the run anyway (it would fail
        # validation) and explain why.
        app_state.autofill_pending = -1
        app_state.autofill_note = message or (
            "The mandatory fields could not be derived from the chat so far "
            "— please complete them in the form below."
        )
        return unchanged

    app_state.autofill_pending = len(filled)
    app_state.autofill_note = (
        f"I pre-filled {len(filled)} of {len(requirements)} fields from the "
        f"chat so far: "
        + ", ".join(f"**{_field_label(requirements, f)}**" for f in filled)
        + ". Please review and complete them in the form below, then click "
        "*Start research* again."
    )

    updates = list(unchanged)
    for i, value in by_index.items():
        updates[offset + i] = gr.update(value=value)
    return tuple(updates)


def _field_label(requirements, field: str) -> str:
    for r in requirements:
        if getattr(r, "field", "") == field:
            return getattr(r, "label", field)
    return field


async def autofill_preflight_values(app_state, message_value, use_case: str):
    """Pre-fill the mandatory fields of an analysis mode from the chat.

    Returns one `gr.update()` per form field. Fields for which nothing
    could be substantiated get an empty update — they stay unchanged, and
    a suggestion never overwrites something the person has already
    entered.

    Explicitly NOT submitted: the preflight validator remains the
    authority that decides about completeness.
    """
    from src.pipeline.analysis_pipeline import USE_CASE_REGISTRY
    from src.ui.preflight_autofill import suggest_preflight_values

    entry = USE_CASE_REGISTRY.get(use_case) or {}
    checker = entry.get("preflight")
    if checker is None:
        gr.Warning(f"Unknown use case: {use_case}")
        return ()

    requirements = checker.get_requirements()
    unchanged = tuple(gr.update() for _ in requirements)

    app_state = _get_ready_state(app_state)
    if not getattr(app_state, "llm", None):
        gr.Warning("LLM not ready — no suggestion possible.")
        return unchanged

    # The not yet submitted text in the input field counts too: that is
    # often exactly where the question is.
    extra = ""
    if isinstance(message_value, dict):
        extra = message_value.get("text") or ""
    elif isinstance(message_value, str):
        extra = message_value

    try:
        values, message = await suggest_preflight_values(
            checker,
            getattr(app_state, "chat_history", []),
            app_state.llm,
            extra_text=extra,
        )
    except Exception as e:
        logger.error("Autofill failed: %s", e, exc_info=True)
        gr.Warning(f"Suggestion failed: {type(e).__name__}")
        return unchanged

    if message:
        gr.Warning(message)
    else:
        missing = len(requirements) - len(values)
        info = f"✨ {len(values)} of {len(requirements)} fields suggested"
        info += " — please review and complete." if missing else " — please review."
        gr.Info(info)

    return tuple(
        gr.update(value=values[r.field]) if r.field in values else gr.update()
        for r in requirements
    )


def _bibtex_text(app_state: AppState) -> str:
    """BibTeX of the current reference check (stored text if restored)."""
    snap = _restored_snapshot(app_state)
    if snap is not None:
        return snap["bibtex"]
    return (app_state.current_research.search_stats or {}).get("bibtex") or ""


def export_bibtex(app_state: AppState):
    """Export verified bibliography entries as a BibTeX file."""
    if not app_state or not app_state.current_research:
        gr.Warning("No research to export.")
        return None
    ctx = app_state.current_research
    # Only meaningful for literature checks
    if not (ctx.output_schema and ctx.output_schema.format_type == "literature_check"):
        gr.Warning("BibTeX export is only available for literature checks.")
        return None
    try:
        bibtex = _bibtex_text(app_state)
        n_entries = len(re.findall(r"^@\w+\s*\{", bibtex, re.M))
        if not n_entries:
            gr.Warning("No bibliography entries to export.")
            return None
        gradio_temp = os.environ.get("GRADIO_TEMP_DIR") or tempfile.gettempdir()
        os.makedirs(gradio_temp, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".bib", prefix=f"literature_{ctx.id}_",
            delete=False, dir=gradio_temp, encoding="utf-8",
        ) as f:
            f.write(bibtex)
            filepath = f.name
        gr.Info(f"✅ BibTeX export created ({n_entries} entries).")
        return filepath
    except Exception as e:
        logger.error(f"BibTeX export error: {e}", exc_info=True)
        gr.Warning(f"⚠️ BibTeX export failed: {e}")
        return None


# =====================================================================
# Gradio App
# =====================================================================

# One accent colour (blue) for primary button, checkboxes and links;
# medium radii everywhere.
APP_THEME = gr.themes.Default(
    primary_hue="blue",
    radius_size="md",
    font=["ui-sans-serif", "sans-serif"],
    font_mono=["ui-monospace", "monospace"],
)


def _institution_help_text() -> str:
    """Help paragraph for the institution mode ('' without a profile)."""
    prof = get_profile()
    if not prof.configured:
        return ""
    return (
        f"**🏛️ {prof.label} research** — focuses on sources of {prof.name} "
        f"({prof.directory_name or 'person directory'}, website index, "
        f"{', '.join(prof.domains)}). With the *{prof.label} only* option for strictly "
        f"internal searches. Good for institutional questions.\n\n"
    )


def _footer_markdown() -> str:
    """Footer links from the institution profile ('' without links)."""
    links = get_profile().footer_links
    if not links:
        return ""
    return "---\n" + " |\n".join(f"[{l.label}]({l.url})" for l in links)


def _export_to_file(export_fn):
    """Wrap an export so the download field only shows when there is a file."""
    def _run(app_state: AppState):
        path = export_fn(app_state)
        return gr.update(value=path, visible=bool(path))
    return _run


def create_app(config: AppConfig = None) -> gr.Blocks:
    if config is None:
        config = AppConfig.from_env()

    if browser_storage_enabled():
        logger.info("Browser storage: on — chat and last result are kept "
                    "in the browser, encrypted with BROWSER_STORAGE_SECRET")
    else:
        logger.info("Browser storage: off — set BROWSER_STORAGE_SECRET to enable")

    with gr.Blocks(
        title=TOOL_NAME,
        analytics_enabled=False,    # second line of defence (the first is the env)
        # delete_cache=(frequency, age): Gradio cleans up automatically —
        # checks every hour, files older than 4 hours are deleted. Sized
        # generously, because single research tasks with many sources /
        # factoid verification can run for more than an hour and user
        # uploads must survive meanwhile.
        delete_cache=(3600, 14400),
        # Gradio 6: theme= is not accepted by the Blocks constructor and
        # belongs in demo.launch(theme=APP_THEME, ...). The theme above, with
        # system fonts instead of Google Fonts, is passed at launch (see
        # app.py). Likewise in launch() instead of Blocks():
        #   - css= (the CUSTOM_CSS from src/ui/css.py)
        #   - js= (if app-wide JS is needed)
        #   - head= (e.g. meta tags)
        # Gradio cleans up user uploads and download caches automatically
        # via delete_cache + the component lifecycle.
    ) as demo:

        # --- State ---
        app_state = gr.State(init_app_state())
        stored_message = gr.State(None)
        # Chat and last result in the browser's localStorage (see
        # save_chat_to_browser, save_result_to_browser), only with a fixed
        # secret. Without one, plain session states take their place:
        # nothing reaches the browser and the save handlers do nothing.
        storage_secret = _browser_storage_secret()
        if storage_secret:
            chat_store = gr.BrowserState(None, storage_key=CHAT_STORAGE_KEY,
                                         secret=storage_secret)
            result_store = gr.BrowserState(None, storage_key=RESULT_STORAGE_KEY,
                                           secret=storage_secret)
        else:
            chat_store = gr.State(None)
            result_store = gr.State(None)

        # Hidden elements
        paste_buffer = gr.Textbox(value="", elem_id="paste-buffer", visible=False)

        # =============================================================
        # HEADER
        # =============================================================
        # Labelled buttons instead of bare symbols; "New chat" lives only
        # here (it used to appear in the sidebar and below the input too).
        with gr.Row(elem_id="app-header"):
            sidebar_toggle = gr.Button("☰ Documents", elem_id="sidebar-toggle",
                                       elem_classes=["header-btn"],
                                       scale=0, min_width=0)
            gr.Markdown(f"**🔍 {TOOL_NAME}**", elem_id="header-title")
            new_chat_btn = gr.Button("✨ New chat", elem_id="new-chat-btn",
                                     elem_classes=["header-btn"],
                                     scale=0, min_width=0)
            result_toggle = gr.Button("📊 Result", elem_id="result-toggle",
                                      elem_classes=["header-btn"],
                                      scale=0, min_width=0)
            dark_mode_btn = gr.Button("🌙", elem_id="dark-mode-toggle",
                                      elem_classes=["header-btn"],
                                      scale=0, min_width=0)

        # =============================================================
        # MAIN LAYOUT
        # =============================================================
        with gr.Row(equal_height=True):

            # ─── SIDEBAR (links) ────────────────────────────────────
            with gr.Column(scale=0, min_width=240, visible=False,
                           elem_id="sidebar-column") as sidebar:

                gr.Markdown("📁 **DOCUMENTS**", elem_classes=["section-label"])
                doc_list_display = gr.HTML(
                    value="<p style='color: var(--body-text-color-subdued); font-size: 0.8rem;'>No documents.</p>",
                )
                token_bar_display = gr.HTML(value="")

                gr.Markdown("---")
                sidebar_file_upload = gr.File(
                    label="📄 Upload documents",
                    file_count="multiple",
                    file_types=[".pdf", ".docx", ".doc", ".txt", ".md",
                                ".html", ".htm", ".csv", ".py",
                                ".xlsx", ".xls", ".pptx", ".ppt", ".rtf"],
                    elem_id="sidebar-upload", height=80,
                )

            # ─── CHAT AREA (middle) ──────────────────────────────
            with gr.Column(scale=2, elem_id="chat-column"):

                chatbot = gr.Chatbot(
                    # Gradio 6: no type parameter; "messages" is the only
                    # supported mode.
                    value=[_welcome_message()],
                    buttons=["copy"],        # NO "share" — prevents
                                             # external share mechanisms.
                    allow_tags=False,        # strict sanitisation
                    render_markdown=True,
                    latex_delimiters=[
                        {"left": "$$", "right": "$$", "display": True},
                        {"left": "$", "right": "$", "display": False},
                    ],
                    autoscroll=False,
                    height="calc(100vh - 280px)",
                    elem_id="chatbot",
                    placeholder="What would you like to research?",
                )

                # ─── Input card: text field + toolbar ──────────────
                # One block instead of three loose rows. Left: mode and
                # options; right: discuss (subtle) and start (the only
                # accent button). Enter in the text field still sends to
                # the chat; the field's own arrow is dropped because it
                # duplicated "Discuss the request".
                with gr.Column(elem_id="composer"):
                    message_input = gr.MultimodalTextbox(
                        placeholder="What would you like to research?",
                        file_types=[".pdf", ".docx", ".txt", ".md", ".html",
                                    ".csv", ".py"],
                        show_label=False,
                        lines=1, max_lines=12,
                        submit_btn=False,
                        elem_id="message-input",
                    )

                    with gr.Row(elem_id="composer-toolbar"):
                        research_mode = gr.Dropdown(
                            # (label, stable ID): the component value is the ID.
                            choices=mode_choices(),
                            value=DEFAULT_RESEARCH_MODE,
                            label="",
                            show_label=False,
                            scale=0,
                            min_width=200,
                            elem_id="research-mode",
                        )
                        # Opens/closes the options row below purely in the
                        # browser (see options_btn.click further down).
                        options_btn = gr.Button("⚙️ Options", scale=0,
                                                min_width=0,
                                                elem_id="options-btn")
                        # Own row so the two actions wrap together on
                        # narrow screens.
                        with gr.Row(elem_id="composer-actions"):
                            # Label kept: the chat prompt and the clickable
                            # chat actions refer to it by this name.
                            send_btn = gr.Button("💬 Discuss the request",
                                                 variant="secondary",
                                                 scale=0, min_width=0,
                                                 elem_id="send-btn")
                            start_btn = gr.Button("🔍 Start research",
                                                  variant="primary",
                                                  scale=0, min_width=0,
                                                  elem_id="research-btn")
                            stop_btn = gr.Button("⏹️ Stop", variant="stop",
                                                 scale=0, min_width=0,
                                                 visible=False,
                                                 elem_id="stop-btn")
                        # Invisible button for chat adoption (triggered via JS);
                        # visible=True + CSS display:none so that it stays in the DOM
                        adopt_btn = gr.Button("adopt", size="sm",
                                              elem_id="adopt-btn",
                                              elem_classes=["hidden-btn"])
                        litcheck_mode_btn = gr.Button("litcheck", size="sm",
                                                      elem_id="litcheck-mode-btn",
                                                      elem_classes=["hidden-btn"])

                    # Collapsed by default; _on_mode_change shows only the
                    # options that apply to the selected mode.
                    with gr.Row(elem_id="options-panel"):
                        template_selector = gr.Dropdown(
                            choices=template_choices(),
                            value=DEFAULT_TEMPLATE,
                            label="Report template",
                            scale=1,
                            min_width=200,
                            elem_id="template-selector",
                        )
                        # Report language: only offered when the installation
                        # enables more than one (OUTPUT_LANGUAGES).
                        output_language_selector = gr.Dropdown(
                            choices=language_choices(),
                            value=default_language(),
                            label="Report language",
                            scale=0,
                            min_width=150,
                            visible=len(language_choices()) > 1,
                            elem_id="output-language",
                        )
                        with gr.Column(scale=1, min_width=220,
                                       elem_id="options-checks"):
                            institution_only_checkbox = gr.Checkbox(
                                label=f"{get_profile().label} only" if get_profile().configured else "Institution only",
                                value=False,
                                visible=False,
                                elem_id="institution-only-checkbox",
                            )
                            academic_only_checkbox = gr.Checkbox(
                                label="🎓 Scholarly literature only",
                                value=False,
                                visible=True,
                                elem_id="academic-only-checkbox",
                            )
                            show_gate_checkbox = gr.Checkbox(
                                label="📋 Confirm plan before start",
                                value=False,
                                visible=True,
                                elem_id="show-gate-checkbox",
                            )

                # ─── Plan preview gate ─────────────────────────────
                # Becomes visible when the analysis handler has produced a
                # plan in the plan_only run and the user has to confirm
                # it.
                with gr.Group(visible=False,
                              elem_id="plan-gate") as plan_gate_group:
                    gr.Markdown(
                        "### 📋 Plan preview",
                        elem_id="plan-gate-header",
                    )
                    plan_md = gr.Markdown(
                        value="",
                        visible=False,
                        elem_id="plan-gate-md",
                    )
                    query_edit_box = gr.Textbox(
                        label="🔍 Search queries (one per line — edit, add or remove)",
                        value="",
                        lines=4,
                        max_lines=10,
                        visible=False,
                        interactive=True,
                        elem_id="query-edit-box",
                    )
                    with gr.Row():
                        confirm_btn = gr.Button(
                            "▶️ Run plan",
                            variant="primary",
                            size="sm",
                            visible=False,
                            elem_id="plan-confirm-btn",
                        )
                        cancel_btn = gr.Button(
                            "✖ Cancel",
                            variant="stop",
                            size="sm",
                            visible=False,
                            elem_id="plan-cancel-btn",
                        )

                # ─── In-depth explanation: preflight panel ─────────────
                with gr.Group(visible=False,
                              elem_id="explainer-panel") as explainer_panel:
                    gr.Markdown(
                        "**In-depth explanation — mandatory inputs**",
                        elem_id="explainer-panel-header",
                    )
                    explainer_topic = gr.Textbox(
                        label="Topic",
                        placeholder=(
                            "e.g. How does reinforcement learning work?"
                        ),
                        info=(
                            "Phrase it concretely (at least 10 characters), "
                            "not just a keyword"
                        ),
                        lines=1,
                    )
                    explainer_audience = gr.Textbox(
                        label="Audience",
                        placeholder=(
                            "e.g. computer science undergraduates with maths up to linear "
                            "algebra, no prior ML knowledge"
                        ),
                        info=(
                            "Include prior knowledge (at least 15 characters)"
                        ),
                        lines=2,
                    )
                    with gr.Row():
                        explainer_length = gr.Dropdown(
                            choices=EXPLAINER_LENGTHS,
                            value="medium",
                            label="Length",
                            info="short ≈3k words · medium ≈8k · detailed ≈15k",
                            scale=1,
                        )
                        explainer_purpose = gr.Dropdown(
                            choices=EXPLAINER_PURPOSES,
                            value="self_study",
                            label="Purpose (optional)",
                            scale=1,
                        )

                # ─── Generic analysis panels ───
                # Built dynamically from the registered PreflightCheckers.
                # Every use case gets its own group panel.
                from src.pipeline.analysis_pipeline import (
                    USE_CASE_REGISTRY as _UC_REGISTRY,
                )
                from src.ui.components.preflight_form import (
                    render_preflight_form,
                )

                _ANALYSIS_PANEL_HEADERS = {
                    "literature_finder": "**Find literature — mandatory inputs**",
                    "peer_review": "**Peer review — mandatory inputs**",
                    "decision_analysis": "**Decision analysis — mandatory inputs**",
                    "research_design": "**Research design — mandatory inputs**",
                    "grant_proposal": "**Grant proposal — mandatory inputs**",
                    "literature_review": "**Literature review — mandatory inputs**",
                }

                # Collecting structures
                analysis_panels = {}  # use_case → gr.Group
                analysis_components_flat = []  # all components in one list
                analysis_autofill_btns = {}  # use_case → (Button, [comps])

                # ANALYSIS_USE_CASE_ORDER is a module constant (see above)
                for uc_name in ANALYSIS_USE_CASE_ORDER:
                    if uc_name not in _UC_REGISTRY:
                        continue
                    checker = _UC_REGISTRY[uc_name]["preflight"]
                    header = _ANALYSIS_PANEL_HEADERS.get(
                        uc_name, f"**{uc_name}**"
                    )
                    with gr.Group(
                        visible=False,
                        elem_id=f"{uc_name}-panel",
                    ) as _panel:
                        gr.Markdown(header)
                        _comps, _get_inputs, _validate = render_preflight_form(
                            checker
                        )
                        # Suggestion from the chat so far. Only pre-fills the
                        # fields — nothing is submitted.
                        _autofill_btn = gr.Button(
                            "✨ Suggest from chat", size="sm",
                            variant="secondary",
                        )
                    analysis_panels[uc_name] = _panel
                    analysis_components_flat.extend(_comps)
                    analysis_autofill_btns[uc_name] = (_autofill_btn, _comps)

                # Fixed chat system prompt; not editable in the UI. The
                # {date} placeholder is filled per message by the handlers.
                system_prompt = gr.State(SYSTEM_PROMPT_CHAT)

            # ─── RESULT PANEL (right) ───────────────────────────
            with gr.Column(scale=2, visible=False,
                           elem_id="result-panel") as result_panel:

                # Export belongs to the report, so it sits in the panel
                # header. The download field only appears once a file
                # exists (see _export_to_file).
                with gr.Row(elem_id="result-header"):
                    gr.Markdown("**Result**", elem_id="result-title")
                    word_export_btn = gr.Button(
                        "📄 Word", elem_classes=["export-btn"],
                        scale=0, min_width=0,
                    )
                    md_export_btn = gr.Button(
                        "📝 Markdown", elem_classes=["export-btn"],
                        scale=0, min_width=0,
                    )
                    bib_export_btn = gr.Button(
                        "📚 BibTeX", elem_classes=["export-btn"],
                        scale=0, min_width=0,
                    )
                export_file = gr.File(
                    label="Download",
                    visible=False,
                    interactive=False,
                    elem_id="export-file",
                )

                with gr.Tabs():
                    (empty_report, empty_sources, empty_progress,
                     empty_extracts, empty_pipeline_run) = _empty_result_texts()
                    with gr.TabItem("📄 Report"):
                        report_display = gr.Markdown(
                            value=empty_report,
                            elem_id="report-display",
                        )

                    with gr.TabItem("🔗 Sources"):
                        sources_display = gr.Markdown(
                            value=empty_sources,
                            elem_id="sources-display",
                        )

                    # Progress, extracts and pipeline run are mostly
                    # needed for troubleshooting, so they share one tab.
                    with gr.TabItem("🧭 History", elem_id="history-tab"):
                        with gr.Accordion("📊 Progress", open=True):
                            progress_display = gr.Markdown(
                                value=empty_progress,
                                elem_id="progress-display",
                            )

                        with gr.Accordion("📝 Extracts", open=False):
                            extracts_display = gr.Markdown(
                                value=empty_extracts,
                                elem_id="extracts-display",
                            )

                        with gr.Accordion("🧠 Pipeline run",
                                          open=False) as pipeline_run_acc:
                            # Makes the DAG plan and all intermediate
                            # structures transparent: output schema,
                            # research plan, classifier results, coverage
                            # per question, filter statistics, synthesis
                            # map answers, diagnosis, quality and
                            # fulfilment check, report revision, factoid
                            # verification, classifier calls. Updates
                            # itself when app_state.current_research
                            # changes (via a .change() handler further
                            # down).
                            pipeline_run_display = gr.Markdown(
                                value=empty_pipeline_run,
                                elem_id="pipeline-run-display",
                            )

        # =============================================================
        # FOOTER
        # =============================================================
        gr.Markdown(
            _footer_markdown(),
            elem_id="app-footer",
        )

        # =============================================================
        # EVENTS
        # =============================================================

        # --- Chat Send ---
        chat_outputs = [app_state, chatbot, send_btn, stop_btn]

        # JS: make the action text in the last chat entry clickable
        _CHAT_ACTION_JS = """() => {
            setTimeout(() => {
                const chatbot = document.querySelector('#chatbot');
                if (!chatbot) return;
                // selectors differ between Gradio versions
                let msgs = chatbot.querySelectorAll('.bot .message-content, .bot.message, .message-wrap .bot');
                if (!msgs.length) msgs = chatbot.querySelectorAll('[data-testid="bot"]');
                const lastMsg = msgs[msgs.length - 1];
                if (!lastMsg) return;

                // already processed?
                if (lastMsg.dataset.actionsLinked) return;
                lastMsg.dataset.actionsLinked = 'true';

                const html = lastMsg.innerHTML;
                let changed = html;

                /*CHAT_ACTION_RULES*/

                if (changed !== html) {
                    lastMsg.innerHTML = changed;
                    lastMsg.querySelectorAll('.chat-action').forEach(el => {
                        el.addEventListener('click', (e) => {
                            e.preventDefault();
                            const action = el.dataset.action;
                            if (action === 'adopt') {
                                const wrapper = document.querySelector('#adopt-btn');
                                const btn = wrapper?.querySelector('button') || wrapper;
                                console.log('Adopt click, wrapper:', !!wrapper, 'btn:', !!btn);
                                if (btn) btn.click();
                            } else if (action === 'research') {
                                // start web research
                                document.querySelector('#research-btn')?.click();
                            } else if (action === 'litcheck') {
                                // set the dropdown to the literature check, then start
                                // The dropdown value is a mode ID; set it
                                // server-side via a hidden button.
                                const w = document.querySelector('#litcheck-mode-btn');
                                (w?.querySelector('button') || w)?.click();
                                setTimeout(() => {
                                    document.querySelector('#research-btn')?.click();
                                }, 600);
                            } else if (action === 'focus') {
                                const ta = document.querySelector('#message-input textarea');
                                if (ta) { ta.focus(); ta.select(); }
                            }
                        });
                    });
                }
            }, 300);
        }""".replace("/*CHAT_ACTION_RULES*/", js_replace_rules())

        # Chat: button and Enter key
        send_btn.click(
            store_and_clear,
            inputs=[message_input, paste_buffer],
            outputs=[stored_message, message_input, paste_buffer],
            queue=False,
        ).then(
            send_chat_message,
            inputs=[app_state, stored_message, chatbot, system_prompt],
            outputs=chat_outputs,
        ).then(
            None, js=_CHAT_ACTION_JS, queue=False,
        ).then(
            save_chat_to_browser,
            inputs=[chatbot, app_state],
            outputs=[chat_store],
            queue=False,
        )

        message_input.submit(
            store_and_clear,
            inputs=[message_input, paste_buffer],
            outputs=[stored_message, message_input, paste_buffer],
            queue=False,
        ).then(
            send_chat_message,
            inputs=[app_state, stored_message, chatbot, system_prompt],
            outputs=chat_outputs,
        ).then(
            None, js=_CHAT_ACTION_JS, queue=False,
        ).then(
            save_chat_to_browser,
            inputs=[chatbot, app_state],
            outputs=[chat_store],
            queue=False,
        )

        # --- Research / institution / literature (unified via the dropdown) ---
        research_outputs = [
            app_state, chatbot,
            send_btn, stop_btn,
            result_panel, start_btn,
            report_display, sources_display, progress_display, extracts_display,
        ]

        # ── Pipeline-run tab: reactive from app_state ──
        # Instead of carrying the tab as an output in every single yield,
        # a change handler re-renders it (see _refresh_pipeline_run). A
        # single point of truth, no changes to the yield sites needed.
        #
        # Import defensively: a missing or broken render module must NEVER
        # make create_app() crash at start-up — it would take the whole
        # app down, not just the tab.
        try:
            from src.ui.pipeline_run import render_pipeline_run
        except Exception as _e:  # pragma: no cover - protective path
            logger.error(
                "Pipeline-run module cannot be loaded (%s) — the tab runs in "
                "degraded mode, the app keeps working.", _e,
            )

            def render_pipeline_run(_ctx):  # type: ignore
                return (
                    "*Pipeline-run view not available "
                    "(render module missing).*"
                )

        def _refresh_pipeline_run(state):
            """Read the current ctx from the state and render the tab.

            IMPORTANT: triggered via `progress_display.change()`, NOT via
            `app_state.change()`. `app_state` is a `gr.State` holding one
            and the same `AppState` object; the pipeline MUTATES this
            object in place (`app_state.current_research = ctx`). Gradio
            only fires `.change()` when the *value* of the state component
            changes — with an object mutated in place and identical in
            identity, that NEVER happens, and the tab would stay empty
            although the pipeline ran completely.
            `progress_display`, on the other hand, is a Markdown whose
            string value changes with EVERY yield → its `.change()` fires
            reliably. The tab update hooks in there and still reads ctx
            from `app_state` (as a gr.State input).
            """
            if state is None or not getattr(state, "current_research", None):
                return "*No research started yet.*"
            # A restored result has no research object to render from.
            restored = _restored_snapshot(state)
            if restored:
                return restored["pipeline_run"] or "*No research started yet.*"
            try:
                return render_pipeline_run(state.current_research)
            except Exception as e:
                logger.warning("Pipeline-run rendering failed: %s", e)
                return f"*Rendering error: {e}*"

        # progress_display.change() instead of app_state.change() — see the
        # docstring above. progress_display changes its value with every
        # yield, app_state (a mutated object) does not.
        progress_display.change(
            _refresh_pipeline_run,
            inputs=[app_state],
            outputs=[pipeline_run_display],
            queue=False,
        )
        # Also render when the section is opened: the last progress update
        # of a run can come before the run is stored in the session state,
        # and then no further change event follows.
        pipeline_run_acc.expand(
            _refresh_pipeline_run,
            inputs=[app_state],
            outputs=[pipeline_run_display],
            queue=False,
        )

        # Visibility switches for the mode-dependent panels
        def _on_mode_change(mode: str):
            kind, selected_uc = resolve_research_route(mode)
            is_institution = kind == "institution"
            is_web = kind == "web"
            is_explainer = kind == "explainer"
            is_literature_old = kind == "literature_check"
            # Template dropdown hidden for all analysis modes
            is_analysis = is_explainer or selected_uc is not None
            template_visible = not (is_analysis or is_literature_old)
            # The academic-only checkbox is visible for the two research
            # modes (web and institution) — the literature check, the
            # in-depth explanation and the analysis modes do not need it
            # (they have their own, specific source logic).
            show_academic = is_web or is_institution
            # The plan preview exists for web/institution research and the
            # analysis modes, not for the explanation or literature check.
            show_gate = is_web or is_institution or selected_uc is not None
            # "Options" only makes sense if the panel has something in it.
            has_options = (template_visible or show_academic or show_gate
                           or len(language_choices()) > 1)
            # Panel updates in the same order as ANALYSIS_USE_CASE_ORDER
            panel_updates = tuple(
                gr.update(visible=selected_uc == uc)
                for uc in ANALYSIS_USE_CASE_ORDER
            )
            return (
                gr.update(visible=is_institution),                # hu_only_checkbox
                gr.update(visible=show_academic),        # academic_only_checkbox
                gr.update(visible=show_gate),            # show_gate_checkbox
                gr.update(visible=has_options),          # options_btn
                gr.update(visible=is_explainer),         # explainer_panel
                gr.update(visible=template_visible),     # template_selector
                *panel_updates,
            )

        def _set_output_language(app_state: AppState, code: str):
            app_state = _get_ready_state(app_state)
            from src.output_language import normalize
            app_state.output_language = normalize(code)
            return app_state

        output_language_selector.change(
            _set_output_language,
            inputs=[app_state, output_language_selector],
            outputs=[app_state], queue=False,
        )

        litcheck_mode_btn.click(
            lambda: gr.update(value="literature_check"),
            inputs=None, outputs=[research_mode], queue=False,
        )

        research_mode.change(
            _on_mode_change,
            inputs=[research_mode],
            outputs=[
                institution_only_checkbox, academic_only_checkbox,
                show_gate_checkbox, options_btn,
                explainer_panel, template_selector,
                *[analysis_panels[uc] for uc in ANALYSIS_USE_CASE_ORDER],
            ],
            queue=False,
        )

        # Options row: toggled in the browser only. The flag sits on
        # <body> because Gradio re-renders its own elements' classes.
        options_btn.click(
            None,
            js="() => { document.body.toggleAttribute('data-options-open'); }",
            queue=False,
        )

        start_btn.click(
            store_and_clear,
            inputs=[message_input, paste_buffer],
            outputs=[stored_message, message_input, paste_buffer],
            queue=False,
        ).then(
            # Before the run: pre-fill empty mandatory fields of an
            # analysis mode from the chat. Does nothing if the form is
            # already filled or no analysis mode is selected. Gradio
            # passes the updated values to the next step; it stops via
            # app_state.autofill_pending so that the suggestions are
            # reviewed first.
            autofill_analysis_form,
            inputs=[app_state, stored_message,
                    research_mode] + analysis_components_flat,
            outputs=analysis_components_flat,
        ).then(
            run_research,
            inputs=[app_state, stored_message, chatbot, system_prompt,
                    template_selector, research_mode, institution_only_checkbox,
                    academic_only_checkbox,
                    show_gate_checkbox,
                    explainer_topic, explainer_audience,
                    explainer_length, explainer_purpose] + analysis_components_flat,
            outputs=research_outputs,
        ).then(
            # After run_research: if a plan is waiting for confirmation,
            # make the gate UI components visible.
            _show_gate_if_pending,
            inputs=[app_state],
            outputs=[plan_gate_group, plan_md, query_edit_box,
                     confirm_btn, cancel_btn],
            queue=False,
        ).then(
            save_chat_to_browser,
            inputs=[chatbot, app_state],
            outputs=[chat_store],
            queue=False,
        ).then(
            save_result_to_browser,
            inputs=[app_state, report_display, sources_display,
                    progress_display, extracts_display],
            outputs=[app_state, result_store],
            queue=False,
        )

        # Plan-Preview-Gate: Confirm & Cancel
        confirm_btn.click(
            confirm_plan_and_run,
            inputs=[app_state, chatbot, query_edit_box],
            outputs=research_outputs,
        ).then(
            _hide_gate_after_run,
            outputs=[plan_gate_group, plan_md, query_edit_box,
                     confirm_btn, cancel_btn],
            queue=False,
        ).then(
            save_chat_to_browser,
            inputs=[chatbot, app_state],
            outputs=[chat_store],
            queue=False,
        ).then(
            save_result_to_browser,
            inputs=[app_state, report_display, sources_display,
                    progress_display, extracts_display],
            outputs=[app_state, result_store],
            queue=False,
        )

        cancel_btn.click(
            cancel_pending_plan,
            inputs=[app_state, chatbot],
            outputs=[
                app_state, chatbot, start_btn, stop_btn,
                plan_gate_group, plan_md, query_edit_box,
                confirm_btn, cancel_btn,
            ],
            queue=False,
        ).then(
            save_chat_to_browser,
            inputs=[chatbot, app_state],
            outputs=[chat_store],
            queue=False,
        )

        # Stop
        stop_btn.click(
            stop_research,
            inputs=[app_state],
            outputs=[send_btn, stop_btn],
            queue=False,
        )

        # --- Sidebar ---
        sidebar_toggle.click(
            toggle_sidebar,
            inputs=[app_state],
            outputs=[app_state, sidebar],
            queue=False,
        )

        result_toggle.click(
            toggle_result_panel,
            inputs=[app_state],
            outputs=[app_state, result_panel],
            queue=False,
        )

        # New chat
        new_chat_btn.click(
            new_chat,
            inputs=[app_state],
            outputs=[app_state, chatbot],
            queue=False,
        ).then(
            save_chat_to_browser,
            inputs=[chatbot, app_state],
            outputs=[chat_store],
            queue=False,
        ).then(
            clear_result,
            inputs=[app_state],
            outputs=[app_state, result_store, result_panel,
                     report_display, sources_display, progress_display,
                     extracts_display, pipeline_run_display, export_file],
            queue=False,
        )

        # --- Adopt suggestion (cleaned up by the LLM) ---
        adopt_btn.click(
            adopt_last_response,
            inputs=[app_state, chatbot],
            outputs=[message_input],
            queue=True,  # needs the queue because of the async LLM call
        ).then(
            None,
            js="""() => {
                // Gradio needs time to write the value into the DOM
                // several attempts with increasing delay
                function resizeInput() {
                    const ta = document.querySelector('#message-input textarea');
                    if (!ta) return false;
                    if (!ta.value || ta.value.length < 5) return false;
                    ta.style.height = 'auto';
                    ta.style.height = Math.min(ta.scrollHeight, 400) + 'px';
                    ta.focus();
                    // trigger an input event so that Gradio notices the change
                    ta.dispatchEvent(new Event('input', {bubbles: true}));
                    return true;
                }
                setTimeout(resizeInput, 300);
                setTimeout(resizeInput, 600);
                setTimeout(resizeInput, 1200);
                setTimeout(resizeInput, 2500);
            }""",
            queue=False,
        )

        # Sidebar Upload
        sidebar_file_upload.change(
            sidebar_upload_files,
            inputs=[app_state, sidebar_file_upload],
            outputs=[app_state, doc_list_display, token_bar_display,
                     sidebar_file_upload],
        )

        # Exports — queue=False so that they react immediately,
        # even while a generator is still in the queue
        md_export_btn.click(
            _export_to_file(export_markdown),
            inputs=[app_state],
            outputs=[export_file],
            queue=False,
        )

        word_export_btn.click(
            _export_to_file(export_word),
            inputs=[app_state],
            outputs=[export_file],
            queue=False,
        )

        # ─── Autofill: suggest mandatory fields from the chat ───
        # `partial` instead of a closure: a closure over the loop variable
        # would bind the last use case for all buttons.
        from functools import partial as _partial

        for _uc_name, (_btn, _uc_comps) in analysis_autofill_btns.items():
            _btn.click(
                _partial(autofill_preflight_values, use_case=_uc_name),
                inputs=[app_state, message_input],
                outputs=_uc_comps,
            )

        bib_export_btn.click(
            _export_to_file(export_bibtex),
            inputs=[app_state],
            outputs=[export_file],
            queue=False,
        )

        # Dark-Mode Toggle
        dark_mode_btn.click(
            None,
            js="""() => {
                const body = document.body;
                if (body.classList.contains('dark')) {
                    body.classList.remove('dark');
                    document.documentElement.style.colorScheme = 'light';
                } else {
                    body.classList.add('dark');
                    document.documentElement.style.colorScheme = 'dark';
                }
            }""",
        )

        # Restore the chat stored in this browser, then link its action
        # phrases again (the link script only looks at the last message),
        # then reopen the last result.
        if storage_secret:
            demo.load(
                restore_chat_from_browser,
                inputs=[chat_store, app_state],
                outputs=[app_state, chatbot],
                queue=False,
            ).then(
                None, js=_CHAT_ACTION_JS, queue=False,
            ).then(
                restore_result_from_browser,
                inputs=[result_store, app_state],
                outputs=[app_state, result_panel, report_display, sources_display,
                         progress_display, extracts_display, pipeline_run_display],
                queue=False,
            )

        # Auto Dark-Mode + Paste-Interceptor
        demo.load(
            None,
            js="""() => {
                // --- Dark Mode ---
                if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
                    document.body.classList.add('dark');
                    document.documentElement.style.colorScheme = 'dark';
                }

                // --- Paste Interceptor ---
                // Gradio's MultimodalTextbox converts long pasted texts
                // into temporary .txt files, which then get lost.
                // We intercept the paste event and write the text
                // directly into the textarea.
                
                let _pasteBound = false;
                
                function bindPasteHandler() {
                    if (_pasteBound) return;
                    
                    const container = document.querySelector('#message-input');
                    if (!container) return;
                    
                    const textarea = container.querySelector('textarea');
                    if (!textarea) return;
                    
                    _pasteBound = true;
                    console.log('[Research] Paste-Handler gebunden');
                    
                    // capture phase: BEFORE Gradio's own handlers
                    textarea.addEventListener('paste', function(e) {
                        // only intercept text pastes, not file pastes
                        const clipboardData = e.clipboardData || window.clipboardData;
                        if (!clipboardData) return;
                        
                        // files in the clipboard → let Gradio handle it
                        if (clipboardData.files && clipboardData.files.length > 0) {
                            return;
                        }
                        
                        const pastedText = clipboardData.getData('text/plain');
                        if (!pastedText || pastedText.length < 10) {
                            // short text → let Gradio handle it normally
                            return;
                        }
                        
                        // LONG enough that Gradio might convert it into a file
                        // → we take over
                        e.preventDefault();
                        e.stopPropagation();
                        e.stopImmediatePropagation();
                        
                        // insert the text at the cursor position via execCommand
                        // (the most reliable method for framework compatibility)
                        textarea.focus();
                        
                        // method 1: execCommand (deprecated, but reliable 
                        // for Svelte/React/Vue state sync)
                        let inserted = false;
                        try {
                            inserted = document.execCommand('insertText', false, pastedText);
                        } catch(ex) {
                            inserted = false;
                        }
                        
                        if (!inserted) {
                            // method 2: manual + InputEvent
                            const start = textarea.selectionStart;
                            const end = textarea.selectionEnd;
                            const before = textarea.value.substring(0, start);
                            const after = textarea.value.substring(end);
                            
                            textarea.value = before + pastedText + after;
                            
                            const newPos = start + pastedText.length;
                            textarea.selectionStart = newPos;
                            textarea.selectionEnd = newPos;
                            
                            // InputEvent with the correct inputType for Svelte
                            try {
                                textarea.dispatchEvent(new InputEvent('input', {
                                    bubbles: true, cancelable: true,
                                    inputType: 'insertFromPaste', data: pastedText,
                                }));
                            } catch(ex2) {
                                textarea.dispatchEvent(new Event('input', { bubbles: true }));
                            }
                            textarea.dispatchEvent(new Event('change', { bubbles: true }));
                        }
                        
                        // auto-resize the textarea
                        textarea.style.height = 'auto';
                        textarea.style.height = Math.min(textarea.scrollHeight, 400) + 'px';
                        
                        // fallback: also write into the hidden paste buffer
                        try {
                            const bufferEl = document.querySelector('#paste-buffer textarea, #paste-buffer input');
                            if (bufferEl) {
                                bufferEl.value = pastedText;
                                bufferEl.dispatchEvent(new Event('input', { bubbles: true }));
                                bufferEl.dispatchEvent(new Event('change', { bubbles: true }));
                            }
                        } catch(ex) { /* ignore */ }
                        
                        console.log('[Research] Paste abgefangen: ' + pastedText.length + ' Zeichen');
                        
                    }, true);  // true = capture phase, runs BEFORE Gradio
                }
                
                // retry, because Gradio may re-render the textarea
                const _pasteInterval = setInterval(() => {
                    bindPasteHandler();
                    if (_pasteBound) clearInterval(_pasteInterval);
                }, 500);
                
            }""",
        )

    # No event handler is meant as a public API. "private" removes them
    # from /gradio_api/info and from the Gradio client libraries; the
    # browser UI is unaffected. This hides the endpoints, it is not an
    # access control: authentication in front of the app is what keeps
    # others out (see SECURITY.md).
    fns = getattr(demo, "fns", None)
    if isinstance(fns, dict):
        fns = fns.values()
    for fn in fns if isinstance(fns, (list, tuple, type({}.values()))) else ():
        fn.api_visibility = "private"

    return demo
