"""
Bibliography check.

Multi-stage process:
1. the LLM parses the raw text into structured entries
2. parallel API queries (arXiv, CrossRef, OpenAlex, Semantic Scholar, DBLP)
3. field-by-field comparison
4. web search for entries that were not found
5. report generation
"""

import asyncio
import json
import logging
import re
from difflib import SequenceMatcher
from typing import Callable

from src.about import TOOL_NAME, VERSION
from src.institution import polite_user_agent
from src.ui.i18n import tr
from src.connectors.literature_apis import (
    LiteratureAPIClient, LiteratureEntry, LiteratureReport,
)

logger = logging.getLogger(__name__)

# Type of the progress callback
ProgressCallback = Callable[[str, object], None]

# ─── Prompts ────────────────────────────────────────────────────────

PARSE_PROMPT = """You are an expert for bibliographic references.
Analyse the following bibliography and extract EVERY entry
as structured JSON.

BIBLIOGRAPHY:
{raw_text}

Extract for EVERY entry:
- authors: list of the authors (given and family name, in the original order)
- title: title of the work
- year: year of publication (number only)
- journal: journal, conference or publisher
- volume: volume
- issue: issue/number
- pages: pages
- doi: DOI (if present, without the https://doi.org/ prefix)
- isbn: ISBN (if present)
- url: URL (if present)
- publisher: publisher (for books)
- edition: edition (if given)
- entry_type: "article" | "book" | "inproceedings" | "thesis" | "web" | "other"

Answer ONLY with a JSON array. Each element is an object with the fields
above. Fields that are not present → empty string "".
Start directly with [ and end with ].

IMPORTANT:
- Take over author names exactly as they stand (do not reformat them)
- Separate all authors correctly (also with "&", "and", "und", ";", "," separators)
- DOIs have the format 10.xxxx/yyyy — extract ONLY the DOI, not the URL
- For proceedings: journal = name of the conference
- For arXiv papers: extract the arXiv ID (e.g. "2310.11511") and set
  doi = "10.48550/arXiv.2310.11511" and url = "https://arxiv.org/abs/2310.11511".
  Also if "arXiv preprint" or "arXiv:XXXX.XXXXX" appears in the text.
- Every entry must also have a field "raw_text" with the original text
  of the entry (exactly as in the input).
- Set entry_type correctly:
  * "inproceedings" if "In ... (Hrsg.)" / "In ... (Ed.)" / "In ... (Eds.)" appears
    (= book chapter / contribution to an edited volume)
  * "book" if it is a stand-alone book (ISBN, publisher, no journal)
  * "article" if journal + volume + pages are present
  * "thesis" for a doctoral thesis, master's thesis, etc.
  * "web" for pure web sources without an academic context
  * "other" if unclear
"""


import contextvars

from src.output_language import llm_language_name, set_current, t as _catalog_t

# Output language of the check in progress (set by LiteratureChecker.check).
_LANG: contextvars.ContextVar[str] = contextvars.ContextVar("litcheck_lang", default="en")


def _t(key: str, **values) -> str:
    """Catalog lookup in the output language of the current check."""
    return _catalog_t(key, _LANG.get(), **values)


class LiteratureChecker:
    """Run the multi-stage literature check."""

    def __init__(self, llm, searxng=None, lang: str | None = None):
        self.lang = lang or "en"
        """
        Args:
            llm: DualLLMClient (or an LLMClient with .complete / .stream)
            searxng: SearXNG connector for the web search (optional)
        """
        self.llm = llm
        self.searxng = searxng
        self.api_client = LiteratureAPIClient()
        self._stop_requested = False
        # Statistics
        self.stats = {
            "api_calls": {"arxiv": 0, "crossref": 0, "openalex": 0,
                          "semantic_scholar": 0, "dblp": 0, "web": 0},
            "api_hits": {"arxiv": 0, "crossref": 0, "openalex": 0,
                         "semantic_scholar": 0, "dblp": 0, "web": 0},
            "entries_total": 0,
            "entries_verified": 0,
            "entries_deviations": 0,
            "entries_not_found": 0,
        }

    def stop(self):
        self._stop_requested = True

    async def close(self):
        """Close the API client."""
        try:
            await self.api_client.close()
        except Exception:
            pass

    @staticmethod
    def _join_entry_lines(parts: list[str]) -> str:
        """Join the lines of ONE entry — URL-aware.

        Core problem: PDFs break long URLs at hyphens, e.g.
            .../long-title-of-an-article-in-
            social-media
        Removing EVERY line-end hyphen ("buffer[:-1]") would turn this into
            .../long-title-of-an-article-insocial-media
        → a broken URL, and verification fails. Hyphens in URLs are REAL
        characters and must be kept; the hyphen is only removed for real
        hyphenation of words in running text.

        Rules when appending the next line:
        - the buffer ends in '-' AND the last token looks like a URL/DOI →
          KEEP the hyphen, append without a space.
        - the buffer ends in '-' otherwise (word hyphenation) → remove the
          hyphen, append without a space.
        - the buffer ends in the middle of a URL (…/, no space) → append
          without a space (the URL continues).
        - otherwise → append with a space.
        """
        buf = ""
        for p in parts:
            p = p.strip()
            if not p:
                continue
            if not buf:
                buf = p
                continue
            if buf.endswith("-"):
                tail = buf.rsplit(None, 1)[-1] if " " in buf else buf
                if re.search(r"https?://|doi\.org|www\.|/", tail):
                    buf = buf + p          # keep the URL hyphen
                else:
                    buf = buf[:-1] + p     # resolve hyphenation
            elif re.search(r"https?://\S*$", buf):
                buf = buf + p              # the URL continues without a space
            else:
                buf = buf + " " + p
        return buf

    async def check(
        self,
        raw_text: str,
        progress_callback: ProgressCallback,
    ) -> tuple[str, LiteratureReport]:
        """Run the complete literature check.

        Returns:
            (markdown_report, structured_report)
        """
        _LANG.set(self.lang)
        set_current(self.lang)
        self._stop_requested = False

        # ═══ Stage 1: parsing ═══════════════════════════════════════
        # Line-break/URL repair happens per entry INSIDE the splitter
        # (_split_raw_entries → _join_entry_lines), so that URLs are not
        # destroyed globally.
        await progress_callback(
            "status", tr("📖 Analysing the bibliography...")
        )
        self._parse_error = ""
        entries = await self._parse_entries(raw_text)

        if not entries:
            return (
                _t("lit.parse_failed", error=self._parse_error) if self._parse_error
                else _t("lit.no_entries"),
                LiteratureReport(),
            )

        await progress_callback("status", tr("📖 {n} entries recognised", n=len(entries)))
        logger.info(f"Literature check: {len(entries)} entries parsed")

        if self._stop_requested:
            return _t("lit.cancelled"), LiteratureReport(entries=entries)

        # ═══ Stage 2: API lookup ════════════════════════════════════
        await progress_callback(
            "status",
            tr("🔍 Checking {n} entries in 5 databases...", n=len(entries))
        )

        for i, entry in enumerate(entries):
            if self._stop_requested:
                break

            await progress_callback(
                "status",
                f"🔍 [{i+1}/{len(entries)}] " + tr("Checking:") + " "
                f"{str(entry.authors[0]) if entry.authors else '?'} "
                f"({entry.year}) — {entry.title[:50]}..."
            )

            try:
                # Hard per-entry timeout as a safety net: even if the rate limiter
                # (despite the capped back-off) or an API takes unusually long, ONE
                # entry must never block the whole run. lookup_entry queries 5 APIs
                # in parallel with a 15 s client timeout each + up to 3 back-offs of
                # at most 30 s — we generously allow 90 s and otherwise mark the
                # entry as timed out (fail-open).
                matches = await asyncio.wait_for(
                    self.api_client.lookup_entry(entry),
                    timeout=90.0,
                )
                entry.api_matches = matches
                entry.checked_sources = [
                    "CrossRef", "OpenAlex", "Semantic Scholar", "DBLP"
                ]
                is_arxiv = self.api_client._is_arxiv_paper(entry)
                if is_arxiv:
                    entry.checked_sources.insert(0, "arXiv")

                # API call statistics
                self.stats["api_calls"]["crossref"] += 1
                self.stats["api_calls"]["openalex"] += 1
                self.stats["api_calls"]["semantic_scholar"] += 1
                self.stats["api_calls"]["dblp"] += 1
                if is_arxiv:
                    self.stats["api_calls"]["arxiv"] += 1

                # Count hits
                for m in matches:
                    src = m.get("source", "").lower().replace(" ", "_")
                    if src in self.stats["api_hits"]:
                        self.stats["api_hits"][src] += 1

                if matches:
                    logger.debug(
                        f"  #{entry.id}: {len(matches)} hits in "
                        f"{', '.join(m['source'] for m in matches)}"
                    )
                else:
                    logger.debug(f"  #{entry.id}: not found in the APIs")

            except asyncio.TimeoutError:
                logger.warning(
                    f"API lookup timeout (>90s) for #{entry.id} — "
                    f"skipped, the run continues"
                )
                entry.notes.append(_t("lit.api_timeout"))
            except Exception as e:
                logger.warning(f"API lookup failed for #{entry.id}: {e}")
                entry.notes.append(_t("lit.api_error", e=e))

            # The delay is controlled by the rate limiter
            await asyncio.sleep(0.05)  # only for UI responsiveness

        # ═══ Stage 3: field comparison ═══════════════════════════════
        await progress_callback("status", tr("📊 Comparing fields..."))

        for entry in entries:
            if entry.api_matches:
                # Use the best match (first = highest priority)
                best_match = entry.api_matches[0]
                entry.deviations = self.api_client.compare_fields(
                    entry, best_match
                )

                # Determine the status
                if not entry.deviations:
                    entry.status = "verified"
                else:
                    entry.status = "deviations"
                    # Take corrected fields from the match
                    for dev in entry.deviations:
                        if dev["found"]:
                            entry.corrected_fields[dev["field"]] = dev["found"]

                # Cross-check: when several APIs report the same error
                if len(entry.api_matches) > 1:
                    self._cross_validate(entry)
            else:
                entry.status = "not_found"

        # ═══ Stage 3b: URL verification ═══════════════════════════
        # Extract URLs from raw_text if the url field is empty
        for entry in entries:
            if not entry.url and entry.raw_text:
                # Try to find a URL — also with spaces inserted by the PDF,
                # e.g. "https://researchgate.net/publication/ 334626234_..."
                url_match = re.search(
                    r"https?://\S+(?:\s\S+)?", entry.raw_text
                )
                if url_match:
                    candidate = url_match.group(0).rstrip(".")
                    # Remove spaces and check whether it is a valid URL
                    candidate = candidate.replace(" ", "")
                    # Only take it if it looks like a URL
                    if "/" in candidate and len(candidate) > 15:
                        entry.url = candidate
            # Repair spaces in URLs (PDF copy artefact)
            if entry.url:
                entry.url = entry.url.replace(" ", "")

        # Check entries with a URL but without an API match directly at the source
        url_candidates = [
            e for e in entries
            if e.status == "not_found" and e.url
        ]
        if url_candidates:
            await progress_callback(
                "status",
                tr("🔗 Checking {n} URLs directly...", n=len(url_candidates))
            )
            await self._verify_urls(url_candidates, progress_callback)

        # ═══ Stage 4: web search for entries not found ════════════════
        not_found = [e for e in entries if e.status == "not_found"]
        if not_found and self.searxng:
            await progress_callback(
                "status",
                tr("🌐 Web search for {n} entries not found...", n=len(not_found))
            )
            await self._web_search_unfound(not_found, progress_callback)

        # ═══ Stage 4b: metrics & duplicates ══════════════════════════
        await progress_callback("status", tr("📊 Extracting metrics..."))

        # Take citation counts from the API matches
        for entry in entries:
            if entry.api_matches:
                # Take the highest citation count from all matches
                best_count = 0
                for m in entry.api_matches:
                    count = m.get("cited_by_count", 0)
                    if isinstance(count, int) and count > best_count:
                        best_count = count
                entry.cited_by_count = best_count

        # Detect duplicates
        duplicates = self._detect_duplicates(entries)

        # ═══ Stage 5: report ══════════════════════════════════════
        await progress_callback("status", tr("✍️ Writing the check report..."))

        # Finalise statistics
        self.stats["entries_total"] = len(entries)
        self.stats["entries_verified"] = sum(
            1 for e in entries if e.status == "verified"
        )
        self.stats["entries_url_verified"] = sum(
            1 for e in entries if e.status == "url_verified"
        )
        self.stats["entries_deviations"] = sum(1 for e in entries if e.status == "deviations")
        self.stats["entries_not_found"] = sum(1 for e in entries if e.status == "not_found")

        report = LiteratureReport(
            entries=entries,
            total=len(entries),
            verified=self.stats["entries_verified"],
            with_deviations=self.stats["entries_deviations"],
            not_found=self.stats["entries_not_found"],
            errors=sum(1 for e in entries if e.status == "error"),
            api_stats=self.stats,
        )

        markdown = await self._generate_report(
            entries, report, duplicates, progress_callback,
        )

        # Generate BibTeX
        matches_dict = {}
        for entry in entries:
            if entry.status in ("verified", "deviations") and entry.api_matches:
                matches_dict[entry.id] = entry.api_matches[0]
        bibtex_text = generate_bibtex(entries, matches_dict)
        report.bibtex = bibtex_text

        # Append BibTeX to the report as a collapsible section
        if bibtex_text:
            markdown += (
                _t("lit.bibtex_details", bibtex=bibtex_text)
            )

        # Get the LLM usage AFTER the report was built (incl. individual comments)
        if hasattr(self.llm, 'get_usage_stats'):
            report.llm_stats = self.llm.get_usage_stats()

        logger.info(
            f"Literature check completed: "
            f"{report.verified} verified, "
            f"{report.with_deviations} with deviations, "
            f"{report.not_found} not found"
        )

        return markdown, report

    # ─── Stage 1: parsing ───────────────────────────────────────────

    async def _parse_entries(self, raw_text: str) -> list[LiteratureEntry]:
        """Parse a bibliography with the LLM.

        Long lists are split into chunks so that the LLM does not cut off
        at the token limit.
        """
        # Split entries robustly AND join them per entry
        # (URL repair happens here in _split_raw_entries).
        raw_entries = self._split_raw_entries(raw_text)

        # IMPORTANT: always pass the JOINED entries to the LLM, never the raw
        # text. Passing `raw_text` would forward URLs broken at hyphens and
        # bypass the URL repair — short bibliographies would reach the LLM
        # with broken links.
        if raw_entries and len(raw_entries) > 12:
            return await self._parse_entries_chunked(raw_entries)

        if raw_entries:
            # Few, but cleanly separated entries → one call with the joined
            # entries (separated by \n\n).
            return await self._parse_single_chunk("\n\n".join(raw_entries))

        # Split failed → the raw text as the last resort.
        return await self._parse_single_chunk(raw_text)

    async def _parse_entries_chunked(
        self, raw_entries: list[str],
    ) -> list[LiteratureEntry]:
        """Parse pre-split entries in chunks."""
        logger.info(
            f"Literature parsing: splitting {len(raw_entries)} raw entries "
            f"into chunks of 12"
        )

        all_entries = []
        chunk_size = 12

        for i in range(0, len(raw_entries), chunk_size):
            if self._stop_requested:
                break

            chunk = raw_entries[i:i + chunk_size]
            chunk_text = "\n\n".join(chunk)
            chunk_entries = await self._parse_single_chunk(chunk_text)

            # Number the IDs consecutively
            for entry in chunk_entries:
                entry.id = len(all_entries) + 1
                all_entries.append(entry)

            logger.info(
                f"  Chunk {i // chunk_size + 1}: "
                f"{len(chunk_entries)} entries parsed"
            )

        return all_entries

    # Year detection — deliberately AGNOSTIC of the citation style.
    # Captures years in all common styles, not only APA '(YYYY)':
    #   APA/Nature:  (2020)        Chicago:  2020.        Vancouver: 2020;
    #   IEEE:        , 2020.       MLA:      2020,        bare:      2020
    # A publication year is 19xx/20xx that is NOT part of a longer number
    # (\b…\b) and typically surrounded by brackets/punctuation/space. We
    # additionally require a right context of punctuation OR line/word end,
    # so that page numbers like '2020' in '2018-2020' fire less often
    # (hyphen on the right ⇒ no match on the left year, but the right year
    # counts — harmless for the block heuristic).
    _RE_YEAR = re.compile(
        r"(?<![\d/])(?:19|20)\d{2}[a-z]?"          # 4-digit year (+ optional a/b)
        r"(?=[)\].,;:\s]|$)"                        # right context
    )
    # Stricter variant ONLY for the "is this the start of an entry" check:
    # here we want a year NEAR the author that opens the entry.
    # Accepts (YYYY) OR 'YYYY.' / 'YYYY;' / 'YYYY,' shortly after the start.
    _RE_YEAR_NEAR = re.compile(
        r"\((?:19|20)\d{2}[a-z]?\)"                 # (2020) / (2020a)
        r"|\((?:19|20)\d{2}[a-z]?\s*(?:,|;|/|\.)"   # (2020, …
        r"|(?<![\d/])(?:19|20)\d{2}[a-z]?\s*[.;,)]"  # 2020. / 2020; / 2020,
    )
    # Start of an entry — generic over name/organisation forms of many styles:
    #  • 'Surname, F.' (APA/Chicago)        • 'Surname Firstname' (Chicago)
    #  • 'van der Berg' (lower-case nobiliary particles, NL/DE)
    #  • 'O'Brien' / 'Müller-Lyer' (apostrophe/hyphen)
    #  • 'UNESCO' / 'DA NRW' (acronyms/organisations, upper case)
    #  • '#Hashtag …' (web titles without an author)
    #  • 'Surname I, Surname I' (Vancouver: surname initial without a comma)
    # Negative guard: lines that start with typical continuation/stop words
    # ('In', 'The', 'Proceedings', …) are NOT entry starts.
    _RE_AUTHOR_START = re.compile(
        r"^(?!(?:In|The|See|Available|Retrieved|Proceedings|Vol|No|pp|"
        r"Eds?|Hrsg|And|Und|With|Mit)\b)"
        r"(?:"
        r"#\S"                                       # #hashtag title
        r"|(?:[a-z]{2,5}\s+){1,3}[A-ZÄÖÜ][\w’'.-]+,"  # 'van der Berg,' (particle)
        r"|[A-ZÄÖÜ]{2,}(?:[\s.]|$)"                  # 'UNESCO' / 'DA NRW' (acronym)
        r"|[A-ZÄÖÜ][\w’'.-]+,\s*[A-ZÄÖÜ]\.?"         # 'Surname, F.'
        r"|[A-ZÄÖÜ][\w’'.-]+\s+[A-ZÄÖÜ][a-zäöü]"     # 'Surname Firstname'
        r"|[A-ZÄÖÜ][\w’'.-]+\s+[A-ZÄÖÜ],"            # 'Surname I,' (Vancouver)
        r")"
    )
    # Section headings / page numbers that are NOT entries.
    # Deliberately generic over common German/English bibliography headings.
    _RE_HEADER = re.compile(
        r"^\s*(?:"
        r"\d{1,3}"                                   # page number only
        r"|(?:\d+\.?\s*)?Literatur(?:verzeichnis)?"  # 9. Literatur(verzeichnis) — German heading variants
        r"|(?:\d+\.?\s*)?Quellen(?:verzeichnis)?"
        r"|(?:\d+\.?\s*)?Bibliogra(?:fie|phie)"
        r"|(?:\d+\.?\s*)?References?"
        r"|(?:\d+\.?\s*)?Bibliography"
        r"|(?:\d+\.?\s*)?Works\s+Cited"
        r"|Wissenschaftliche\s+Quellen"
        r"|Nicht[\s-]*wissenschaftliche\s+Quellen"
        r"|Primary\s+Sources|Secondary\s+Sources"
        r"|Cited\s+Literature|Reference\s+List"
        r")\s*:?\s*$",
        re.IGNORECASE,
    )

    @classmethod
    def _is_section_header(cls, s: str) -> bool:
        return bool(cls._RE_HEADER.match(s))

    # End of an entry: ends in a URL, DOI, year + full stop, page range, .pdf …
    _RE_ENTRY_END = re.compile(
        r"(?:https?://\S+|doi\.org/\S+|\d{4}\.|\)\.|\d+[–-]\d+\.|\.pdf|/\d+)$"
    )

    @classmethod
    def _looks_like_entry_start(cls, prev_line: str, line: str) -> bool:
        """True if `line` starts a NEW entry.

        Two valid patterns:
        - author start AND year on the same line (the normal case), OR
        - author start, year NOT (yet) on the line (a long author list
          wrapped over several lines), BUT the previous line looks like
          the END of an entry (URL/DOI/year.).
        """
        s = line.strip()
        if not s or cls._is_section_header(s):
            return False
        if not cls._RE_AUTHOR_START.match(s):
            return False
        # Guard: if the previous line ended in a comma/&/…/-, this is a
        # wrapped author list, NOT a new entry.
        if prev_line and prev_line.rstrip().endswith((",", "&", "…", "-")):
            return False
        if cls._RE_YEAR_NEAR.search(s):
            return True
        # Year not on this line → only a start if the previous entry
        # clearly ended.
        if prev_line and cls._RE_ENTRY_END.search(prev_line.strip()):
            return True
        return False

    @classmethod
    def _blocks_by_blank(cls, text: str) -> list[list[str]]:
        """Split into blocks at blank lines (headings are discarded)."""
        blocks, cur = [], []
        for line in text.split("\n"):
            s = line.strip()
            if not s:
                if cur:
                    blocks.append(cur)
                    cur = []
                continue
            if cls._is_section_header(s):
                if cur:
                    blocks.append(cur)
                    cur = []
                continue
            cur.append(s)
        if cur:
            blocks.append(cur)
        return blocks

    @classmethod
    def _blocks_by_author(cls, text: str) -> list[list[str]]:
        """Split into blocks at author-year starts (for hanging indents
        without blank lines between entries — the typical bibliography case)."""
        entries, cur, last = [], [], ""
        for line in text.split("\n"):
            s = line.strip()
            if not s:
                continue
            if cls._is_section_header(s):
                if cur:
                    entries.append(cur)
                    cur = []
                last = ""
                continue
            if cur and cls._looks_like_entry_start(last, line):
                entries.append(cur)
                cur = [s]
            else:
                cur.append(s)
            last = s
        if cur:
            entries.append(cur)
        return entries

    @classmethod
    def _candidate_numbered(cls, text: str) -> list[str] | None:
        """Segmentation at real, continuous numbering ([1]/1.).
        None if there is no plausible numbering."""
        markers = re.findall(r"\n\s*(?:\[(\d+)\]|(\d+)\.)\s+", text)
        nums = [int(a or b) for a, b in markers]
        if len(nums) < 2:
            return None
        # Plausible if the marker numbers are small and roughly ascending
        # (a real reference list 1..N), not scattered numbers.
        small = sum(1 for n in nums if n <= len(nums) + 5)
        if small < 0.8 * len(nums):
            return None
        pieces = re.split(r"\n\s*(?:\[\d+\]|\d+\.)\s+", text)
        out = [cls._join_entry_lines(p.split("\n"))
               for p in pieces if p.strip()]
        return [e for e in out if len(e) > 25]

    @classmethod
    def _candidate_blank(cls, text: str) -> list[str]:
        """Segmentation at blank lines (joined URL-aware)."""
        return [e for e in (cls._join_entry_lines(b)
                            for b in cls._blocks_by_blank(text))
                if len(e) > 25]

    @classmethod
    def _candidate_author(cls, text: str) -> list[str]:
        """Segmentation at author-year starts (hanging indent)."""
        return [e for e in (cls._join_entry_lines(b)
                            for b in cls._blocks_by_author(text))
                if len(e) > 25]

    @classmethod
    def _score_segmentation(cls, entries: list[str]) -> float:
        """Rate how 'reference-like' a segmentation is.

        A generic quality measure instead of fixed thresholds — the higher,
        the more plausible that every element is EXACTLY ONE reference:
        - share of blocks with an author start  (+)
        - share of blocks with EXACTLY ONE year  (+ ; 0 or >1 years ⇒ penalty)
        - share of blocks with a plausible length (40–600 characters)  (+)
        - penalty for extremely long blocks (several entries glued together)
        Returns 0..1.
        """
        if not entries:
            return 0.0
        n = len(entries)
        author_hits = sum(1 for e in entries
                          if cls._RE_AUTHOR_START.match(e.strip()))
        one_year = 0
        multi_year = 0
        good_len = 0
        for e in entries:
            yc = len(cls._RE_YEAR.findall(e))
            if yc == 1:
                one_year += 1
            elif yc > 1:
                multi_year += 1
            if 40 <= len(e) <= 600:
                good_len += 1
        score = (
            0.35 * (author_hits / n)
            + 0.40 * (one_year / n)
            + 0.25 * (good_len / n)
            - 0.30 * (multi_year / n)   # several years ⇒ entries glued together
        )
        return max(0.0, score)

    @classmethod
    def _split_raw_entries(cls, raw_text: str) -> list[str]:
        """Separate individual bibliography entries robustly from the raw text.

        GENERIC approach (no overfitting to particular PDFs/styles):
        several segmentation candidates are produced — numbered, based on
        blank lines, based on author-year starts — and the candidate with
        the best 'reference score' is chosen. The method thus adapts to
        APA, Vancouver, IEEE, Nature, Chicago, hanging indents and lists
        separated by blank lines, without ever hard-wiring a style.

        Structures covered in practice:
        1. separated by blank lines, URLs wrapped across lines.
        2. hanging indent, almost no blank lines.
        """
        text = raw_text.strip()
        if not text:
            return []

        candidates: list[tuple[str, list[str]]] = []

        numbered = cls._candidate_numbered(text)
        if numbered:
            candidates.append(("numbered", numbered))
        candidates.append(("blank", cls._candidate_blank(text)))
        candidates.append(("author", cls._candidate_author(text)))

        # Rate. On a tie we prefer the FINER segmentation (more entries), but
        # only if its score is not noticeably worse — otherwise glued or
        # over-fragmented variants would win.
        scored = [(name, ents, cls._score_segmentation(ents))
                  for name, ents in candidates if ents]
        if not scored:
            # Last resort: long single lines
            single = [l.strip() for l in text.split("\n") if l.strip()]
            return [l for l in single if len(l) > 50]

        best_score = max(s for _, _, s in scored)
        # All candidates within 90 % of the best value count as ~equal
        near_best = [(name, ents, s) for name, ents, s in scored
                     if s >= 0.9 * best_score]
        # Among the roughly equally good ones: the one with the most entries
        # (finest sensible resolution).
        name, ents, _ = max(near_best, key=lambda t: len(t[1]))
        logger.info(
            "Literature split: strategy '%s' chosen (%d entries, "
            "score %.2f)",
            name, len(ents),
            next(s for n, _, s in scored if n == name),
        )
        return ents

    async def _parse_single_chunk(self, raw_text: str) -> list[LiteratureEntry]:
        """Parse a single chunk with the LLM."""
        prompt = PARSE_PROMPT.format(raw_text=raw_text[:20000])

        try:
            response = await self.llm.primary_complete(
                [{"role": "user", "content": prompt}],
                max_tokens=16384,
            )

            # Extract JSON
            clean = response.strip()
            if not clean:
                logger.warning("Empty LLM answer while parsing the bibliography")
                return []

            # Remove Markdown fences
            clean = re.sub(r"^```(?:json)?\s*", "", clean)
            clean = re.sub(r"\s*```$", "", clean)

            # Find the array
            start = clean.find("[")
            end = clean.rfind("]")
            if start >= 0 and end > start:
                clean = clean[start:end + 1]
            elif start >= 0:
                # Truncated array → repair
                clean = self._repair_truncated_json_array(clean[start:])
                if not clean:
                    logger.warning(
                        f"Truncated JSON array could not be repaired "
                        f"(length={len(response)})"
                    )
                    return []

            items = json.loads(clean)
            if not isinstance(items, list):
                logger.warning("LLM answer is not a JSON array")
                return []

            logger.info(f"Literature chunk: {len(items)} entries parsed")

            entries = []
            for i, item in enumerate(items):
                if not isinstance(item, dict):
                    continue
                entry = LiteratureEntry.from_dict(item)
                entry.id = i + 1
                entries.append(entry)

            return entries

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error while parsing the bibliography: {e}")
            logger.debug(f"Answer (first 500 characters): {response[:500]}")
            # Last repair attempt: everything up to the last valid object
            try:
                # Find the start of the array in the raw answer
                arr_start = response.find("[")
                if arr_start >= 0:
                    repaired = self._repair_truncated_json_array(
                        response[arr_start:]
                    )
                    if repaired:
                        items = json.loads(repaired)
                        if isinstance(items, list) and items:
                            logger.info(
                                f"JSON repair successful: "
                                f"{len(items)} entries rescued"
                            )
                            entries = []
                            for i, item in enumerate(items):
                                if not isinstance(item, dict):
                                    continue
                                entry = LiteratureEntry.from_dict(item)
                                entry.id = i + 1
                                entries.append(entry)
                            return entries
            except Exception:
                pass
            logger.error("JSON repair failed — no entries parsed")
            return []
        except Exception as e:
            logger.error(f"Literature parsing failed: {e}")
            self._parse_error = f"{type(e).__name__}: {e}"
            return []

    @staticmethod
    def _repair_truncated_json_array(text: str) -> str | None:
        """Repair a truncated JSON array.

        Example: [{"a":1}, {"b":2}, {"c": → [{"a":1}, {"b":2}]
        """
        text = text.strip()
        if not text.startswith("["):
            return None

        # Find the last complete object
        depth = 0
        last_complete = -1

        for i, c in enumerate(text):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    last_complete = i

        if last_complete > 0:
            # Up to the last complete object + ]
            candidate = text[:last_complete + 1].rstrip().rstrip(",") + "]"
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass

        return None

    # ─── Stage 3b: URL verification ────────────────────────────────

    async def _verify_urls(
        self,
        entries: list[LiteratureEntry],
        progress_callback: ProgressCallback,
    ):
        """Check URLs directly — for entries without an API match."""
        import httpx
        from urllib.parse import urlparse
        from src.connectors.rate_limiter import get_rate_limiter

        limiter = get_rate_limiter()

        # Domains that block bots with 403 but exist
        _KNOWN_403_DOMAINS = {
            "researchgate.net", "jstor.org", "academia.edu",
            "sciencedirect.com", "wiley.com", "tandfonline.com",
        }

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10, read=15, write=10, pool=10),
            follow_redirects=True,
            headers={"User-Agent": polite_user_agent()},
        ) as client:
            for entry in entries:
                if self._stop_requested:
                    break

                url = entry.url.strip()
                if not url or not url.startswith("http"):
                    continue

                url = url.replace(" ", "")

                await progress_callback(
                    "status",
                    tr("🔗 Checking URL: {url}...", url=url[:60])
                )

                try:
                    # Rate-limited request via the global limiter
                    resp = await limiter.request(
                        "url_check", client, "GET", url,
                    )

                    if resp.status_code == 200:
                        page_text = resp.text[:10000]
                        title_found = self._check_title_on_page(
                            entry.title, page_text
                        )

                        if title_found:
                            entry.status = "url_verified"
                            entry.notes.append(
                                _t("lit.url_ok_title", short=url[:60], url=url)
                            )
                        else:
                            entry.status = "url_verified"
                            entry.notes.append(
                                _t("lit.url_ok", short=url[:60], url=url)
                            )

                        entry.checked_sources.append("URL-Check")
                        self.stats["api_calls"]["web"] += 1
                        self.stats["api_hits"]["web"] += 1

                    elif resp.status_code == 403:
                        domain = urlparse(url).netloc.lower()
                        is_known = any(
                            d in domain for d in _KNOWN_403_DOMAINS
                        )

                        if is_known:
                            entry.status = "url_verified"
                            entry.notes.append(
                                _t("lit.url_restricted", domain=domain, short=url[:60], url=url)
                            )
                            entry.checked_sources.append("URL-Check")
                            self.stats["api_calls"]["web"] += 1
                            self.stats["api_hits"]["web"] += 1
                        else:
                            entry.notes.append(
                                _t("lit.url_403", short=url[:60])
                            )
                            self.stats["api_calls"]["web"] += 1

                    else:
                        entry.notes.append(
                            _t("lit.url_status", status=resp.status_code, short=url[:60])
                        )
                        self.stats["api_calls"]["web"] += 1

                except httpx.TimeoutException:
                    entry.notes.append(f"⏱️ URL-Timeout: {url[:60]}")
                except Exception as e:
                    entry.notes.append(
                        _t("lit.url_unreachable", short=url[:60], e=e)
                    )

    @staticmethod
    def _check_title_on_page(title: str, page_html: str) -> bool:
        """Check whether the title occurs on a web page.

        Checks: body text, <title> tag, <meta> tags (citation_title,
        og:title, DC.title).
        """
        if not title:
            return False

        page_lower = page_html.lower()

        # Extract title words (only significant ones, >3 characters)
        clean_title = re.sub(r"[^\w\s]", "", title.lower())
        title_words = [w for w in clean_title.split() if len(w) > 3][:8]

        if not title_words:
            return False

        # Method 1: words anywhere in the HTML (incl. meta tags)
        matches = sum(1 for w in title_words if w in page_lower)
        threshold = max(2, int(len(title_words) * 0.5))
        if matches >= threshold:
            return True

        # Method 2: exactly in <title> or <meta> tags
        meta_patterns = [
            r'<title[^>]*>([^<]+)</title>',
            r'<meta\s+name=["\'](?:citation_title|DC\.title|dc\.title)["\']\s+content=["\']([^"\']+)',
            r'<meta\s+property=["\']og:title["\']\s+content=["\']([^"\']+)',
            # Reversed order (content before property)
            r'<meta\s+content=["\']([^"\']+)["\']\s+(?:name|property)=["\'](?:citation_title|og:title|DC\.title)',
        ]
        for pattern in meta_patterns:
            m = re.search(pattern, page_html, re.IGNORECASE)
            if m:
                meta_text = m.group(1).lower()
                meta_matches = sum(1 for w in title_words if w in meta_text)
                if meta_matches >= max(2, int(len(title_words) * 0.5)):
                    return True

        return False

    # ─── Stage 4: web search ────────────────────────────────────────

    async def _web_search_unfound(
        self,
        entries: list[LiteratureEntry],
        progress_callback: ProgressCallback,
    ):
        """Web search for entries not found in the APIs."""
        for entry in entries:
            if self._stop_requested:
                break

            # Build the search query
            parts = []
            if entry.authors:
                a0 = str(entry.authors[0])
                parts.append(a0.split()[-1])  # family name
            if entry.title:
                parts.append(f'"{entry.title[:60]}"')
            if entry.year:
                parts.append(entry.year)

            query = " ".join(parts)
            if not query:
                continue

            try:
                results = await self.searxng.search(query, max_results=3)
                if results:
                    entry.notes.append(
                        _t("lit.web_hits", n=len(results), url=results[0].url)
                    )
                    # Status stays "not_found", but with a web hint
                    entry.status = "not_found"  # stays, but with a web hint
                    entry.checked_sources.append(_t("lit.web_search"))
                else:
                    entry.notes.append(_t("lit.web_no_hits"))

            except Exception as e:
                logger.debug(f"Web search failed for #{entry.id}: {e}")

            await asyncio.sleep(0.5)

    # ─── Cross-validation ──────────────────────────────────────────

    @staticmethod
    def _cross_validate(entry: LiteratureEntry):
        """Check whether several APIs report the same deviation."""
        if len(entry.api_matches) < 2:
            return

        def _to_str(val) -> str:
            if val is None:
                return ""
            if isinstance(val, str):
                return val
            if isinstance(val, list):
                return ", ".join(str(x) for x in val)
            if isinstance(val, dict):
                return val.get("text", "") or str(val)
            return str(val)

        # For every deviation: does a second API confirm it?
        for dev in entry.deviations:
            field = dev["field"]
            found_value = _to_str(dev["found"])

            # Check whether other APIs have the same value
            confirming_sources = []
            for match in entry.api_matches[1:]:
                other_value = _to_str(match.get(field, ""))
                if other_value and found_value and \
                        other_value.strip().lower() == found_value.strip().lower():
                    confirming_sources.append(match["source"])

            if confirming_sources:
                dev["confirmed_by"] = confirming_sources
                dev["confidence"] = "high"
                dev["message"] += (
                    _t("lit.confirmed_by", sources=', '.join(confirming_sources))
                )
            else:
                dev["confidence"] = "medium"

    # ─── Stage 5: report generation ────────────────────────────────

    async def _generate_report(
        self,
        entries: list[LiteratureEntry],
        report: LiteratureReport,
        duplicates: list[tuple],
        progress_callback: ProgressCallback | None = None,
    ) -> str:
        """Build the Markdown report deterministically.

        The report is assembled line by line from structured data. Only
        the short individual comments per entry are generated by the LLM
        (1-3 sentences).
        """
        status_labels = {
            "verified": _t("lit.status.verified"),
            "url_verified": _t("lit.status.url_verified"),
            "deviations": _t("lit.status.deviations"),
            "not_found": _t("lit.status.not_found"),
            "error": _t("lit.status.error"),
        }

        parts = []

        # ── Header ──
        parts.append(_t("lit.title"))

        # ── Summary ──
        parts.append(_t("lit.summary"))
        parts.append(_t("lit.sum.total", n=report.total))
        parts.append(_t("lit.sum.verified", n=report.verified))
        url_v = sum(1 for e in entries if e.status == "url_verified")
        if url_v:
            parts.append(_t("lit.sum.url", n=url_v))
        parts.append(_t("lit.sum.deviations", n=report.with_deviations))
        parts.append(_t("lit.sum.not_found", n=report.not_found))
        if duplicates:
            parts.append(_t("lit.sum.duplicates", n=len(duplicates)))
        parts.append("")

        # ── Timeline ──
        parts.append(self._build_timeline(entries))

        # ── Overview table (ALL entries, deterministic) ──
        parts.append(self._build_summary_table(entries, report))

        # ── Duplicates ──
        if duplicates:
            parts.append(self._build_duplicates_section(entries, duplicates))

        # ── Detailed results (deterministic per entry) ──
        parts.append(_t("lit.details"))

        for i, entry in enumerate(entries):
            if self._stop_requested:
                break

            if progress_callback:
                await progress_callback(
                    "status",
                    tr("✍️ Writing report [{i}/{n}]: ", i=i + 1, n=len(entries)) +
                    f"{str(entry.authors[0]) if entry.authors else '?'} "
                    f"({entry.year})..."
                )

            section = self._build_entry_section(entry, status_labels)

            # Generate a short LLM comment for entries with deviations or
            # entries not found
            if entry.status in ("deviations", "not_found"):
                comment = await self._generate_entry_comment(entry)
                if comment:
                    section += f"\n**Bewertung:** {comment}\n"

            parts.append(section)
            parts.append("---\n")

            # Streaming update: show the report step by step
            if progress_callback:
                await progress_callback(
                    "report_stream", "\n".join(parts)
                )

        # ── Corrected bibliography (APA) ──
        parts.append(self._build_corrected_bibliography(entries))

        # ── BibTeX-Export ──
        bibtex_entries = []
        for entry in entries:
            if entry.status in ("verified", "deviations"):
                match = entry.api_matches[0] if entry.api_matches else None
                bibtex_entries.append(
                    format_bibtex_entry(entry, match)
                )
        if bibtex_entries:
            parts.append("\n---\n### BibTeX-Export\n")
            parts.append(
                _t("lit.bibtex_show", n=len(bibtex_entries))
            )
            parts.append("\n\n".join(bibtex_entries))
            parts.append("```\n</details>")

        # ── Sources checked ──
        parts.append(_t("lit.checked_in_all"))

        full_report = "\n".join(parts)

        if progress_callback:
            await progress_callback("report_stream", full_report)

        return full_report

    # ─── Single-entry section ──────────────────────────────────

    @staticmethod
    def _build_entry_section(
        entry: LiteratureEntry,
        status_labels: dict,
    ) -> str:
        """Build the detail section for a single entry."""
        lines = []
        first = str(entry.authors[0]) if entry.authors else "?"
        short_authors = first.split(",")[0] if "," in first else first
        suffix = " et al." if len(entry.authors) > 1 else ""
        lines.append(
            f"### {entry.id}. {short_authors}{suffix} ({entry.year or '?'})"
        )
        lines.append(
            f"**Status:** {status_labels.get(entry.status, entry.status)}"
        )

        # Citations
        if entry.cited_by_count > 0:
            assessment = _assess_citations(entry.cited_by_count, entry.year)
            lines.append(
                _t("lit.citations", n=entry.cited_by_count, assessment=assessment)
            )

        # Original citation
        raw = entry.raw_text or _reconstruct_citation(entry)
        lines.append(f"\n**Original:** {raw}")

        # Deviations
        if entry.deviations:
            lines.append(_t("lit.deviations"))
            for dev in entry.deviations:
                conf = dev.get("confidence", "")
                conf_tag = _t("lit.confidence", conf=conf) if conf else ""
                msg = dev.get("message", "")
                source = dev.get("source", "")
                confirmed = dev.get("confirmed_by", [])

                line = f"- {msg}"
                if source:
                    line += _t("lit.source_open", source=source)
                    if confirmed:
                        line += _t("lit.confirmed_semicolon", sources=', '.join(confirmed))
                    line += ")"
                line += conf_tag
                lines.append(line)

        # Corrected entry (APA)
        if entry.api_matches and entry.status == "deviations":
            best = entry.api_matches[0]
            corrected = _format_apa(entry, best)
            if corrected:
                lines.append(_t("lit.corrected", v=corrected))
                url = best.get("url", "")
                if url:
                    lines.append(_t("lit.available_at", url=url))

        # DOI enrichment
        if not entry.doi and entry.api_matches:
            for m in entry.api_matches:
                if m.get("doi"):
                    lines.append(
                        _t("lit.doi_added", doi=m['doi'], source=m['source'])
                    )
                    break

        # Notes (URL check, web search, etc.)
        if entry.notes and entry.status in ("not_found", "url_verified"):
            for note in entry.notes:
                lines.append(f"\n*{note}*")

        # Sources checked
        if entry.checked_sources:
            lines.append(
                _t("lit.checked_in", sources=', '.join(entry.checked_sources))
            )

        return "\n".join(lines)

    # ─── Individual LLM comment ─────────────────────────────────────

    async def _generate_entry_comment(
        self, entry: LiteratureEntry,
    ) -> str:
        """Generate a short LLM comment for an entry.

        At most 2-3 sentences. On error: an empty string.
        """
        try:
            if entry.status == "not_found":
                context = (
                    f"Bibliography entry not found in academic databases: "
                    f"{entry.title} ({entry.year}). "
                    f"Type: {entry.entry_type}."
                )
            else:
                devs = "; ".join(
                    d.get("message", "") for d in entry.deviations[:5]
                )
                context = (
                    f"Deviations for: {entry.title} ({entry.year}): "
                    f"{devs}"
                )

            prompt = (
                "Assess in 1-2 sentences what this deviation means "
                "for the reliability of the bibliography entry. "
                "Distinguish: harmless formatting differences "
                "(initials vs. first names, journal spelling) vs. "
                "errors of content (wrong title, wrong authors, "
                "wrong year) vs. possible forgery/hallucination "
                "(the entry does not exist at all).\n\n"
                f"Context: {context}"
            )

            comment = await self.llm.primary_complete(
                [{"role": "user", "content": prompt
                  + f"\n\nAnswer in {llm_language_name(self.lang)}."}],
                max_tokens=256,
            )
            return comment.strip()

        except Exception as e:
            logger.debug(f"LLM comment failed for #{entry.id}: {e}")
            return ""

    # ─── Timeline ──────────────────────────────────────────────

    @staticmethod
    def _build_timeline(entries: list[LiteratureEntry]) -> str:
        """Produce a text histogram of the publication years."""
        year_counts: dict[str, int] = {}
        for e in entries:
            y = e.year.strip() if e.year else "?"
            if y and len(y) == 4 and y.isdigit():
                year_counts[y] = year_counts.get(y, 0) + 1
            else:
                year_counts["?"] = year_counts.get("?", 0) + 1

        if not year_counts or (len(year_counts) == 1 and "?" in year_counts):
            return ""

        # Sort
        sorted_years = sorted(
            [(y, c) for y, c in year_counts.items() if y != "?"],
            key=lambda x: x[0],
        )

        if not sorted_years:
            return ""

        max_count = max(c for _, c in sorted_years)
        max_bar = 30  # maximum bar width

        lines = [_t("lit.timeline")]
        lines.append("```")
        for year, count in sorted_years:
            bar_len = int(count / max_count * max_bar) if max_count > 0 else 0
            bar = "█" * max(bar_len, 1)
            lines.append(f"  {year}  {bar} ({count})")

        if "?" in year_counts:
            lines.append(f"   ?    {'█' * 1} ({year_counts['?']})")
        lines.append("```")
        lines.append("")

        return "\n".join(lines)

    # ─── Duplicate detection ──────────────────────────────────────

    @staticmethod
    def _detect_duplicates(
        entries: list[LiteratureEntry],
    ) -> list[tuple[int, int, str]]:
        """Detect duplicate candidates.

        Returns: list of (id_a, id_b, reason)
        """
        duplicates = []
        seen_dois = {}  # doi → entry_id

        for e in entries:
            # DOI duplicates
            if e.doi:
                doi_norm = e.doi.lower().strip()
                if doi_norm in seen_dois:
                    duplicates.append((
                        seen_dois[doi_norm], e.id,
                        _t("lit.dup_doi", doi=e.doi)
                    ))
                else:
                    seen_dois[doi_norm] = e.id

        # Title similarity (O(n²), but bibliographies are small)
        for i, a in enumerate(entries):
            for b in entries[i + 1:]:
                # DOI duplicates already detected
                if a.doi and b.doi and a.doi.lower() == b.doi.lower():
                    continue
                if not a.title or not b.title:
                    continue

                sim = SequenceMatcher(
                    None, a.title.lower(), b.title.lower()
                ).ratio()
                if sim >= 0.95:
                    duplicates.append((
                        a.id, b.id,
                        _t("lit.dup_title", sim=f'{sim:.0%}')
                    ))

        return duplicates

    @staticmethod
    def _build_duplicates_section(
        entries: list[LiteratureEntry],
        duplicates: list[tuple],
    ) -> str:
        """Build the duplicates section of the report."""
        entries_by_id = {e.id: e for e in entries}
        lines = [_t("lit.duplicates")]

        for id_a, id_b, reason in duplicates:
            a = entries_by_id.get(id_a)
            b = entries_by_id.get(id_b)
            if not a or not b:
                continue

            fa = str(a.authors[0]) if a.authors else "?"
            fb = str(b.authors[0]) if b.authors else "?"
            name_a = fa.split(",")[0] if "," in fa else fa
            name_b = fb.split(",")[0] if "," in fb else fb

            lines.append(
                f"- **#{id_a}** {name_a} ({a.year}) ↔ "
                f"**#{id_b}** {name_b} ({b.year}): {reason}"
            )

        lines.append("")
        return "\n".join(lines)

    # ─── Overview table ───────────────────────────────────────

    @staticmethod
    def _build_summary_table(
        entries: list[LiteratureEntry],
        report: LiteratureReport,
    ) -> str:
        """Produce a deterministic Markdown overview table."""
        status_icons = {
            "verified": "✅",
            "url_verified": "🟡",
            "deviations": "⚠️",
            "not_found": "❌",
            "error": "💥",
        }

        def _risk(entry: LiteratureEntry) -> str:
            """Risk assessment."""
            if entry.status == "verified":
                return "🟢 OK"

            if entry.status == "url_verified":
                # URL exists, but fields not confirmed by a database
                return _t("lit.risk.url_confirmed")

            if entry.status == "not_found":
                # Books/theses/web pages are often not in databases
                if entry.entry_type in ("book", "thesis", "web", "other"):
                    return _t("lit.risk.not_indexed")
                # Articles/conference papers should be findable
                return _t("lit.risk.forgery")

            if entry.status == "deviations":
                # The publication was found → it exists.
                # Question: are the deviations merely citation errors,
                # or does the hit point to a different work?
                if not entry.api_matches:
                    return _t("lit.risk.check")

                # Check whether the hit really is the same work:
                # if title AND authors differ strongly → possible mix-up
                title_mismatch = False
                author_mismatch = False
                for d in entry.deviations:
                    if d.get("field") == "title" and d.get("type") == "mismatch":
                        title_mismatch = True
                    if d.get("field") == "authors" and d.get("type") == "mismatch":
                        author_mismatch = True

                # Title AND authors wrong at the same time → could be a
                # completely different paper that merely has a similar name
                if title_mismatch and author_mismatch:
                    return _t("lit.risk.mixup")

                # Only authors or only the title differ, or other field errors
                # → the publication exists, the citation is faulty
                return _t("lit.risk.check")

            return "⚪ ?"

        def _short_authors(entry: LiteratureEntry) -> str:
            if not entry.authors:
                return "?"
            first = entry.authors[0]
            name = first.split(",")[0].strip() if isinstance(first, str) else str(first)
            return f"{name} et al." if len(entry.authors) > 1 else name

        lines = [
            "---",
            "",
            _t("lit.overview"),
            _t("lit.overview_line", total=report.total, verified=report.verified, dev=report.with_deviations, nf=report.not_found),
            _t("lit.table_header"),
            "|---|-----------|------|-------|--------|---------|------------|--------|",
        ]

        for e in entries:
            authors = _short_authors(e)
            year = e.year or "?"
            title = e.title[:40] + "…" if len(e.title or "") > 40 else (e.title or "?")
            status = status_icons.get(e.status, "❓")
            hits = len(e.api_matches)
            cites = str(e.cited_by_count) if e.cited_by_count > 0 else "—"
            risk = _risk(e)

            # Add status details
            if e.status == "verified":
                status = _t("lit.table.db")
            elif e.status == "url_verified":
                status = _t("lit.table.url")
            elif e.status == "not_found":
                status = _t("lit.table.not_found")
            elif e.deviations:
                fields = ", ".join(sorted({d.get("field", "?") for d in e.deviations}))
                status = f"⚠️ {fields}"

            lines.append(
                f"| {e.id} | {authors} | {year} | {title} "
                f"| {status} | {hits}/5 | {cites} | {risk} |"
            )

        lines.extend([
            "",
            _t("lit.legend"),
            "",
        ])

        return "\n".join(lines)

    # ─── Corrected bibliography ────────────────────────────────

    @staticmethod
    def _build_corrected_bibliography(
        entries: list[LiteratureEntry],
    ) -> str:
        """Produce a complete, corrected bibliography in APA format."""
        lines = [
            "---",
            "",
            _t("lit.corrected_title"),
            _t("lit.corrected_note"),
        ]

        for e in entries:
            if e.api_matches:
                best = e.api_matches[0]
                apa = _format_apa(e, best)
            else:
                apa = _format_apa_from_entry(e)

            # Marker
            if e.status == "verified":
                marker = "✅"
            elif e.status == "url_verified":
                marker = "🟡"
            elif e.status == "deviations":
                marker = "⚠️"
            elif e.status == "not_found":
                marker = "❌"
            else:
                marker = "❓"

            lines.append(f"{marker} {apa}\n")

        # DIN 1505-2 version (collapsible)
        lines.append(_t("lit.corrected_din"))
        for e in entries:
            if e.api_matches:
                din = _format_din(e, e.api_matches[0])
            else:
                din = _format_din_from_entry(e)
            if e.status == "verified":
                marker = "✅"
            elif e.status == "deviations":
                marker = "⚠️"
            elif e.status == "not_found":
                marker = "❌"
            else:
                marker = "❓"
            lines.append(f"{marker} {din}\n")
        lines.append("</details>")

        return "\n".join(lines)


def _reconstruct_citation(entry: LiteratureEntry) -> str:
    """Reconstruct a readable citation from parsed fields."""
    parts = []
    if entry.authors:
        parts.append(", ".join(str(a) for a in entry.authors))
    if entry.year:
        parts.append(f"({entry.year})")
    if entry.title:
        parts.append(entry.title)
    if entry.journal:
        parts.append(f"*{entry.journal}*")
    details = []
    if entry.volume:
        details.append(str(entry.volume))
    if entry.issue:
        details.append(f"({entry.issue})")
    if entry.pages:
        details.append(str(entry.pages))
    if details:
        parts.append(", ".join(details))
    if entry.doi:
        parts.append(f"doi:{entry.doi}")
    return ". ".join(parts) + "." if parts else _t("lit.no_text")


def _s(val) -> str:
    """Universal None-safe string converter."""
    if val is None:
        return ""
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, list):
        return ", ".join(str(x) for x in val)
    if isinstance(val, dict):
        return val.get("text", "") or val.get("name", "") or str(val)
    return str(val)


def _format_apa_authors(authors) -> str:
    """Format an author list according to APA 7.

    Accepts various author formats:
      - string "Family, Given" or "Given Family"
      - dict {"family": "...", "given": "..."} (CrossRef/OpenAlex)
      - dict {"name": "Given Family"} (alternative)
      - dict {"display_name": "..."} (OpenAlex)
    """
    if not authors:
        return ""

    # Defensive: if `authors` is a single string (a common bug of LLM
    # parsers), treat it as one entry, not as an iterable over
    # characters.
    if isinstance(authors, str):
        authors = [authors]

    def _name(a) -> str:
        """Turn a single author into the APA form 'Family, G.'."""
        # Dict form from APIs
        if isinstance(a, dict):
            family = (a.get("family") or a.get("last") or "").strip()
            given = (a.get("given") or a.get("first") or "").strip()
            if family and given:
                # "Given" → "G." (every word)
                initials = " ".join(
                    f"{v[0]}." if len(v) >= 1 else v
                    for v in given.split()
                )
                return f"{family}, {initials}"
            if family:
                return family
            # Fallback: "name" or "display_name"
            full = (
                a.get("name")
                or a.get("display_name")
                or ""
            ).strip()
            if full:
                return _name_from_string(full)
            return ""

        # String-Form
        return _name_from_string(str(a).strip())

    names = [_name(a) for a in authors]
    names = [n for n in names if n]
    if not names:
        return ""

    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} & {names[1]}"
    if len(names) <= 20:
        return ", ".join(names[:-1]) + f", & {names[-1]}"
    # More than 20: APA 7 rule: first 19, then "...", then the last
    return ", ".join(names[:19]) + f", ... {names[-1]}"


def _name_from_string(a: str) -> str:
    """String author → 'Family, G.' form."""
    if not a:
        return ""
    # Already in the form "Family, G." → take it
    if "," in a:
        return a
    # "Given Family" → "Family, G."
    parts = a.split()
    if len(parts) >= 2:
        family = parts[-1]
        given_initials = " ".join(
            f"{v[0]}." if len(v) >= 1 else v for v in parts[:-1]
        )
        return f"{family}, {given_initials}"
    return a


def _format_apa(
    entry: LiteratureEntry, match: dict,
) -> str:
    """Format an entry as APA 7, with data from the API match."""
    authors = match.get("authors", entry.authors) or entry.authors
    year = _s(match.get("year")) or entry.year
    title = _s(match.get("title")) or entry.title
    journal = _s(match.get("journal")) or entry.journal
    volume = _s(match.get("volume")) or entry.volume
    issue = _s(match.get("issue")) or entry.issue
    pages = _s(match.get("pages")) or entry.pages
    doi = _s(match.get("doi")) or entry.doi

    return _build_apa_string(authors, year, title, journal,
                             volume, issue, pages, doi)


def _format_apa_from_entry(entry: LiteratureEntry) -> str:
    """Format an entry as APA 7, from parsed fields only."""
    return _build_apa_string(
        entry.authors, entry.year, entry.title, entry.journal,
        entry.volume, entry.issue, entry.pages, entry.doi,
    )


def _build_apa_string(
    authors, year, title, journal,
    volume, issue, pages, doi,
) -> str:
    """Assemble an APA 7 string."""
    parts = []

    # Authors
    author_str = _format_apa_authors(authors)
    if author_str:
        parts.append(author_str)

    # Year
    if year:
        parts.append(f"({_s(year)})")

    # Title (italic NOT for articles, only for books)
    if title:
        t = _s(title).rstrip(".")
        parts.append(f"{t}.")

    # Journal italic + volume italic + issue + pages
    j = _s(journal)
    v = _s(volume)
    iss = _s(issue)
    p = _s(pages)

    if j:
        journal_part = f"*{j}*"
        if v:
            journal_part += f", *{v}*"
            if iss:
                journal_part += f"({iss})"
        if p:
            journal_part += f", {p}"
        journal_part += "."
        parts.append(journal_part)

    # DOI
    d = _s(doi)
    if d:
        if not d.startswith("http"):
            d = f"https://doi.org/{d}"
        parts.append(d)

    return " ".join(parts) if parts else _t("lit.no_text")


# ─── DIN 1505-2 (German citation style) ──────────────────────────

def _format_din(
    entry: LiteratureEntry, match: dict,
) -> str:
    """Format an entry according to DIN 1505-2."""
    authors = match.get("authors", entry.authors) or entry.authors
    year = _s(match.get("year")) or entry.year
    title = _s(match.get("title")) or entry.title
    journal = _s(match.get("journal")) or entry.journal
    volume = _s(match.get("volume")) or entry.volume
    issue = _s(match.get("issue")) or entry.issue
    pages = _s(match.get("pages")) or entry.pages
    doi = _s(match.get("doi")) or entry.doi
    publisher = _s(match.get("publisher")) or entry.publisher
    isbn = _s(match.get("isbn")) or entry.isbn

    return _build_din_string(
        authors, year, title, journal, volume, issue,
        pages, doi, publisher, isbn, entry.entry_type,
    )


def _format_din_from_entry(entry: LiteratureEntry) -> str:
    """Format an entry according to DIN 1505-2, from entry fields only."""
    return _build_din_string(
        entry.authors, entry.year, entry.title, entry.journal,
        entry.volume, entry.issue, entry.pages, entry.doi,
        entry.publisher, entry.isbn, entry.entry_type,
    )


def _build_din_string(
    authors, year, title, journal,
    volume, issue, pages, doi,
    publisher="", isbn="", entry_type="article",
) -> str:
    """Assemble a DIN 1505-2 string.

    Article: Nachname, Vorname: Titel. In: Zeitschrift Jg. (Jahr) H., S. x-y.
    Book:    Nachname, Vorname: Titel. Ort: Verlag, Jahr. — ISBN
    (the German abbreviations are part of the citation style)
    """
    parts = []

    # Authors: Family, Given; Family, Given
    if authors:
        author_parts = []
        for a in (authors if isinstance(authors, list) else [authors]):
            a = a if isinstance(a, str) else str(a)
            a = a.strip().rstrip(".,")
            if a:
                author_parts.append(a)
        if author_parts:
            parts.append("; ".join(author_parts) + ":")

    # Title
    t = _s(title).rstrip(".") if title else ""
    if t:
        parts.append(f"{t}.")

    j = _s(journal)
    v = _s(volume)
    iss = _s(issue)
    p = _s(pages)
    y = _s(year)

    if entry_type in ("article",) and j:
        # Article: In: Zeitschrift Jg. (Jahr) H., S. x-y.
        ref = f"In: {j}"
        if v:
            ref += f" {v}"
        if y:
            ref += f" ({y})"
        if iss:
            ref += f" H. {iss}"
        if p:
            ref += f", S. {p}"
        ref += "."
        parts.append(ref)
    else:
        # Book / other
        if publisher:
            parts.append(f"{publisher},")
        if y:
            parts.append(f"{y}.")

    # DOI / ISBN
    d = _s(doi)
    if d:
        if not d.startswith("http"):
            d = f"https://doi.org/{d}"
        parts.append(f"— {d}")
    elif isbn:
        parts.append(f"— ISBN {isbn}")

    return " ".join(parts) if parts else _t("lit.no_text")


def _assess_citations(count: int, year: str) -> str:
    """Put the citation count in context, taking the age into account."""
    try:
        age = 2026 - int(year) if year and year.isdigit() else 3
    except ValueError:
        age = 3

    if age <= 0:
        age = 1

    # Citations per year
    per_year = count / age

    if count == 0:
        return _t("lit.cites.none")
    elif per_year >= 100:
        return _t("lit.cites.very_high")
    elif per_year >= 30:
        return _t("lit.cites.high")
    elif per_year >= 10:
        return _t("lit.cites.good")
    elif per_year >= 3:
        return _t("lit.cites.moderate")
    elif count >= 1:
        return _t("lit.cites.low")
    else:
        return _t("lit.cites.none")


def _bibtex_escape(text: str) -> str:
    """Escape a value for BibTeX fields.

    - `%` becomes `\\%` (otherwise the rest of the line is treated as a
      comment — serious for titles with percentages).
    - `&` becomes `\\&` (LaTeX special character).
    - unbalanced `{`/`}` are corrected: excess braces are removed,
      otherwise the parser breaks.
    """
    if not text:
        return ""
    s = text
    # Escape the LaTeX special characters first
    s = s.replace("\\", "\\textbackslash{}")  # first, otherwise others get escaped twice
    s = s.replace("%", r"\%")
    s = s.replace("&", r"\&")
    s = s.replace("#", r"\#")
    s = s.replace("$", r"\$")
    s = s.replace("_", r"\_")
    # Catch unbalanced braces
    n_open = s.count("{")
    n_close = s.count("}")
    if n_open != n_close:
        # Pragmatic: replace all braces, better safe than sorry
        s = s.replace("{", "(").replace("}", ")")
    return s


def _bibtex_key_sanitize(s: str) -> str:
    """Reduce a string to characters allowed in BibTeX keys.

    Only ASCII letters, digits, underscore and hyphen are allowed. Other
    characters (umlauts, accents, ß, dots, colons) are transliterated or
    removed. At least one character is returned.
    """
    import unicodedata
    if not s:
        return "anon"
    # Special transliterations that NFD does not handle
    pre = (
        s.replace("ß", "ss").replace("Ø", "O").replace("ø", "o")
         .replace("Æ", "AE").replace("æ", "ae")
         .replace("Œ", "OE").replace("œ", "oe")
         .replace("Ð", "D").replace("ð", "d")
         .replace("Þ", "Th").replace("þ", "th")
    )
    # Unicode → ASCII (NFD normalisation + remove diacritics)
    nfkd = unicodedata.normalize("NFD", pre)
    ascii_str = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Keep only ASCII letters, digits, _ and -
    out = "".join(
        c for c in ascii_str
        if (c.isascii() and (c.isalnum() or c in "_-"))
    )
    return out.lower() if out else "anon"


def format_bibtex_entry(entry: LiteratureEntry, match: dict | None = None) -> str:
    """Format a bibliography entry as BibTeX.

    Uses API match data if present, otherwise the entry fields.
    """
    m = match or {}
    authors = m.get("authors", entry.authors) or entry.authors
    year = _s(m.get("year")) or entry.year or "n.d."
    title = _s(m.get("title")) or entry.title or "Untitled"
    journal = _s(m.get("journal")) or entry.journal
    volume = _s(m.get("volume")) or entry.volume
    issue = _s(m.get("issue")) or entry.issue
    pages = _s(m.get("pages")) or entry.pages
    doi = _s(m.get("doi")) or entry.doi
    publisher = _s(m.get("publisher")) or entry.publisher
    isbn = _s(m.get("isbn")) or entry.isbn
    url = _s(m.get("url")) or entry.url

    # Determine the entry type
    bib_type = {
        "article": "article",
        "book": "book",
        "inproceedings": "inproceedings",
        "thesis": "phdthesis",
        "web": "misc",
    }.get(entry.entry_type, "misc")

    # Generate the BibTeX key: family_year (ASCII only, sanitised)
    key_author = ""
    if authors:
        first = authors[0] if isinstance(authors[0], str) else str(authors[0])
        parts = first.replace(",", " ").split()
        key_author = _bibtex_key_sanitize(parts[0]) if parts else "anon"
    if not key_author:
        key_author = "anon"
    # Sanitise the year for the key (e.g. "n.d." → "nd")
    key_year = _bibtex_key_sanitize(year) or "nd"
    key = f"{key_author}{key_year}"

    # BibTeX fields — escape all values (against %, &, etc.)
    lines = [f"@{bib_type}{{{key},"]
    # Authors: BibTeX format "Family, Given and Family, Given"
    if authors:
        author_strs = []
        for a in authors:
            a = a if isinstance(a, str) else str(a)
            author_strs.append(_bibtex_escape(a.strip()))
        lines.append(f"  author = {{{' and '.join(author_strs)}}},")
    lines.append(f"  title = {{{_bibtex_escape(title)}}},")
    lines.append(f"  year = {{{_bibtex_escape(year)}}},")
    if journal:
        lines.append(f"  journal = {{{_bibtex_escape(journal)}}},")
    if volume:
        lines.append(f"  volume = {{{_bibtex_escape(volume)}}},")
    if issue:
        lines.append(f"  number = {{{_bibtex_escape(issue)}}},")
    if pages:
        lines.append(f"  pages = {{{_bibtex_escape(pages)}}},")
    if doi:
        d = doi.removeprefix("https://doi.org/").removeprefix(
            "http://doi.org/"
        ).removeprefix("https://dx.doi.org/").removeprefix("doi:")
        # Only escape the DOI itself, not the `/`
        lines.append(f"  doi = {{{_bibtex_escape(d)}}},")
    if isbn:
        lines.append(f"  isbn = {{{_bibtex_escape(isbn)}}},")
    if publisher:
        lines.append(f"  publisher = {{{_bibtex_escape(publisher)}}},")
    if url and not doi:
        lines.append(f"  url = {{{_bibtex_escape(url)}}},")
    lines.append("}")

    return "\n".join(lines)


def generate_bibtex(entries: list[LiteratureEntry],
                    matches: dict[int, dict] | None = None) -> str:
    """Generate BibTeX for all entries.

    Args:
        entries: list of bibliography entries
        matches: dict entry.id → best API match (optional)
    """
    parts = [
        _t("lit.bibtex_header", tool=TOOL_NAME, version=VERSION),
        _t("lit.bibtex_count", n=len(entries)),
        "",
    ]
    for entry in entries:
        match = (matches or {}).get(entry.id)
        parts.append(format_bibtex_entry(entry, match))
        parts.append("")

    return "\n".join(parts)
