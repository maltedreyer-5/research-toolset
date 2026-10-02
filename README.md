# research-toolset

[![CI](https://github.com/maltedreyer-5/research-toolset/actions/workflows/ci.yml/badge.svg)](https://github.com/maltedreyer-5/research-toolset/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A self-hosted research assistant that plans a research run, searches the web
and scholarly databases, reads the sources and writes a referenced report —
with every intermediate step visible. It runs against any OpenAI-compatible
LLM endpoint (for example a local vLLM server) and a SearXNG instance, so no
request has to leave your infrastructure.

research-toolset is a Gradio web application written in Python 3.12.

## What it does

**Research modes**

- **Web research** — breaks the request into research questions, searches in
  several languages, fetches and reads the sources, extracts facts per
  question and writes a report with inline source links.
- **Check references** — checks a bibliography entry by entry against
  CrossRef, OpenAlex, Semantic Scholar, arXiv, DBLP and OpenLibrary, flags
  deviations and possible fabrications, adds DOIs and returns a corrected
  bibliography (APA, DIN 1505-2, BibTeX).
- **Find literature** — searches OpenAlex, Semantic Scholar and arXiv for a
  research question and returns an assessed selection with full bibliographic
  details taken from the APIs, not from the model.
- **Institution research** (optional) — restricts the research to one
  organisation's web pages, person directory and search index. Configured
  through an institution profile; hidden without one.

**Analysis modes** — in-depth explanation, peer review, decision analysis,
research design, grant proposal (draft) and literature review. Each mode
breaks the task into sub-tasks, runs them as a dependency graph and assembles
a report; the literature-based modes run a real literature search first.

**Safeguards built into the pipeline**

- A chain of LLM classifiers decides what a plain heuristic would get wrong:
  whether a request is about a person, which sources are off topic, whether a
  question is answered, whether another round is worthwhile, and what went
  wrong if a run ends thin.
- Statements in the finished report are checked against the collected
  extracts; contradicted statements are corrected, uncertain ones are listed.
- A final quality and fulfilment check writes its findings into the report
  instead of hiding them in a log.
- The **Pipeline run** tab shows the plan, every classifier decision, filter
  statistics and the intermediate answers of a run.

## Quick start

Requirements: Python 3.12, an OpenAI-compatible LLM endpoint and a SearXNG
instance with the JSON format enabled (an example configuration is in
[`searxng-config/`](searxng-config/settings.yml)).

```bash
git clone https://github.com/maltedreyer-5/research-toolset.git
cd research-toolset
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # set LLM_API_BASE, LLM_API_KEY, LLM_MODEL_NAME, SEARXNG_BASE_URL
python app.py
```

The interface is then available at <http://127.0.0.1:7860>. By default the
server binds to `127.0.0.1` only; see [deployment](docs/deployment.md) for
serving it to others.

A [`Dockerfile`](Dockerfile) is included; the image runs as an unprivileged
user, binds to `0.0.0.0` inside the container and has a health check.

## Configuration

All settings are environment variables. [`.env.example`](.env.example) lists
every variable the code reads, with its default. Only the LLM endpoint is
required; set `BROWSER_STORAGE_SECRET` to keep the chat and the last result
in the browser across page reloads (off by default). Details:
[configuration](docs/configuration.md).

## Languages

The user interface is available in English and German; a switch in the
header changes it (each language is its own page, e.g. `/de`). See
[interface languages](docs/output-languages.md#interface-languages).

Reports can be written in any language that
has a catalog: English is always available, German ships with the tool, and
further languages are added by dropping a catalog file into a directory —
no code change needed. See [output languages](docs/output-languages.md).

## Privacy

- No telemetry: Gradio analytics, the Hugging Face hub telemetry and similar
  switches are turned off before Gradio is imported.
- No external fonts in the interface (system fonts only), no share links, no public API
  documentation; event handlers are not exposed as an API.
- Only if `BROWSER_STORAGE_SECRET` is set, the chat and the last result are
  kept in the user's own browser (localStorage, encrypted with that key),
  not on the server; *New chat* removes them. Off by default, which suits
  shared computers.
- Uploaded files and download caches are removed by Gradio after a few hours;
  stored research runs are deleted after `CLEANUP_MAX_AGE_DAYS` (default 30).
- Outgoing requests go only to the endpoints you configure. Every fetched URL
  passes an SSRF check (private, loopback and link-local addresses are
  refused, also after DNS resolution and on every redirect).

See [SECURITY.md](SECURITY.md) for the security model and for reporting
vulnerabilities.

## Documentation

- [Architecture](docs/architecture.md) — pipeline, classifiers, analysis modes
- [Configuration](docs/configuration.md) — environment variables
- [Output languages](docs/output-languages.md) — catalogs, adding a language
- [Institution mode](docs/institution-mode.md) — profile, person directory
- [Connectors](docs/connectors.md) — connecting further data sources
- [Deployment](docs/deployment.md) — Docker, reverse proxy, TLS, data retention

## Limitations

- Report quality depends heavily on the model. The prompts were developed
  against large open-weight models (the Qwen3 and Kimi families); small models
  will produce thinner reports.
- The tool reads what search engines return. It does not log in anywhere and
  cannot see pages behind paywalls or logins.
- Web content goes into LLM prompts. Although the pipeline checks statements
  against their sources, a manipulated web page can still influence a report
  (prompt injection). Treat reports as a starting point, not as verified fact.
- There is no user management: access control is a single optional login or
  whatever your reverse proxy provides.

## License

[MIT](LICENSE) © Malte Dreyer
