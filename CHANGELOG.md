# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed

- Calmer layout: text field and toolbar form one input card with a single
  accent button (*Start research*). Report template, report language and
  the check boxes sit in a collapsible options row that shows only what
  applies to the selected mode.
- Export moved into the header of the result panel; the download field
  appears only once there is a file. Progress, Extracts and Pipeline run
  are combined into one *History* tab next to *Report* and *Sources*.
- Labelled header buttons (*Documents*, *New chat*, *Result*); *New chat*
  lives only in the header.

### Fixed

- BibTeX export of a reference check: it looked for entries nothing ever
  stored and always reported "No bibliography entries to export". It now
  writes the BibTeX the check builds itself (best API match per entry).

## [1.0.0] — 2026-09-28

First public release.

### Research

- Web research: plan with research questions and multilingual search terms,
  search via SearXNG (optionally GitHub, GitLab, Elasticsearch, OpenAlex),
  ranking, fetching, link following, fact extraction per question, and a report
  with inline source links.
- Follow-up rounds search for the aspects the coverage assessment reports as
  missing (`MAX_RESEARCH_ROUNDS`).
- Optional plan confirmation before the research starts.
- Institution mode (optional): institution profile, person directory with a
  consent model, Solr or Elasticsearch index of the institution's website.

### Analysis modes

- In-depth explanation, peer review, decision analysis, research design, grant
  proposal (draft) and literature review, run as dependency graphs of sub-tasks;
  the literature-based modes query OpenAlex, Semantic Scholar and arXiv.
- Find literature: assessed selection with bibliographic details taken from the
  APIs.

### Bibliography check

- Entry-by-entry check against CrossRef, OpenAlex, Semantic Scholar, arXiv,
  DBLP and OpenLibrary, URL verification, duplicate detection, corrected
  bibliography in APA and DIN 1505-2, BibTeX export.

### Quality safeguards

- LLM classifiers for query anchor, source relevance, coverage, continue
  decision, search scope and diagnosis, each with a conservative fallback.
- Factoid verification against the collected extracts, revision of
  contradicted statements, report quality and fulfilment checks written into
  the report; pipeline-run view with every decision.

### Output languages

- English interface; reports and exports in English, German or any language
  added as a catalog file (`OUTPUT_LANGUAGES`, `OUTPUT_CATALOG_DIR`).

### Operation

- Word and Markdown export, BibTeX export.
- Privacy defaults: no telemetry, no external fonts, no share links, no public
  API; retention of stored runs (`CLEANUP_MAX_AGE_DAYS`).
- SSRF protection for every fetched URL, TLS verification on by default,
  binding to `127.0.0.1` by default, Docker image running as an unprivileged user.

[Unreleased]: https://github.com/maltedreyer-5/research-toolset/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/maltedreyer-5/research-toolset/releases/tag/v1.0.0
