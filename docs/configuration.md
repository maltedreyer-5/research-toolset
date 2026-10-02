# Configuration

All settings are environment variables, read at start-up. A `.env` file in
the working directory is loaded automatically. [`.env.example`](../.env.example)
lists every variable with its default; this page explains the ones that need
a decision.

## Required

| Variable | Meaning |
|---|---|
| `LLM_API_BASE` | OpenAI-compatible endpoint, e.g. `http://localhost:8000/v1` |
| `LLM_API_KEY` | API key of that endpoint |
| `LLM_MODEL_NAME` | model name as the endpoint expects it |
| `SEARXNG_BASE_URL` | SearXNG instance with the JSON format enabled (default `http://searxng:8080`) |

Set `LLM_MAX_CONTEXT_TOKENS` and `LLM_MAX_OUTPUT_TOKENS` to the limits of your
model; the defaults (260 000 / 32 768) fit large long-context models.

## Second model (optional)

`HARVEST_LLM_API_BASE`, `HARVEST_LLM_API_KEY`, `HARVEST_LLM_MODEL_NAME` select a
smaller, faster model for fact extraction and the classifiers. Without them
the primary model is used for everything. `HARVEST_MAX_PARALLEL` limits
concurrent calls to it.

## Server

| Variable | Default | Meaning |
|---|---|---|
| `GRADIO_SERVER_NAME` | `127.0.0.1` | bind address; the Docker image sets `0.0.0.0` |
| `GRADIO_SERVER_PORT` | `7860` | port |
| `ROOT_PATH` | *(empty)* | path prefix behind a reverse proxy, e.g. `/research` |
| `SERVER_AUTH_USER`, `SERVER_AUTH_PASSWORD` | — | optional single login |
| `SERVER_SSL_CERTFILE`, `SERVER_SSL_KEYFILE`, `SERVER_SSL_PASSWORD` | — | TLS directly in the app; otherwise terminate TLS at a reverse proxy |

## Outgoing connections

| Variable | Default | Meaning |
|---|---|---|
| `TLS_VERIFY` | `true` | certificate verification for the internal services you run yourself: embedder, reranker, person directory, Solr. Connections to public endpoints (web search, fetched pages, literature APIs, LLM) are always verified. Prefer `SSL_CERT_FILE` (path to a CA bundle) for internal certificates over switching verification off |
| `FETCH_ALLOWED_INTERNAL_HOSTS` | *(empty)* | comma-separated hosts that may be fetched although they resolve to private addresses (the SSRF protection refuses private, loopback and link-local targets otherwise) |
| `FETCH_TIMEOUT` | `30` | seconds per fetched page |
| `CONTACT_EMAIL` | — | sent in the User-Agent to Crossref and OpenAlex (polite pool) |

## Research

| Variable | Default | Meaning |
|---|---|---|
| `MAX_RESEARCH_ROUNDS` | `5` | upper bound of research rounds; each further round costs another round of searches, fetches and LLM calls |
| `MAX_SOURCES_PER_ROUND` | `20` | pages fetched per round after ranking |
| `MAX_WEB_SEARCHES` | `40` | web searches per round |
| `MAX_PARALLEL_FETCHES` | `10` | concurrent page fetches |
| `RERANKER_BASE_URL`, `RERANKER_MODEL`, `RERANKER_API_KEY` | — | optional cross-encoder (`/rerank` API) to rank search results, links and extracts; without it a heuristic is used |
| `EMBEDDER_BASE_URL`, `EMBEDDER_MODEL`, `EMBEDDER_API_KEY` | — | optional embedding endpoint for de-duplicating extracts; without it word overlap is used |
| `USE_PLAYWRIGHT` | `false` | JavaScript rendering fallback (needs `requirements-optional.txt` and `playwright install chromium`) |

Reranker and embedder are fail-open: if they are unreachable, the pipeline
continues with the fallback.

## Output language

| Variable | Default | Meaning |
|---|---|---|
| `OUTPUT_LANGUAGES` | `en` | report languages offered in the interface, e.g. `en,de` |
| `DEFAULT_OUTPUT_LANGUAGE` | `en` | preselected language; must be in `OUTPUT_LANGUAGES` |
| `OUTPUT_CATALOG_DIR` | — | directory with additional catalogs |

The start-up fails with a clear message if a language has no catalog or a
catalog is inconsistent. See [output languages](output-languages.md).

## Data and retention

| Variable | Default | Meaning |
|---|---|---|
| `DATA_DIR` | `data/runs` | where completed runs are stored |
| `CLEANUP_MAX_AGE_DAYS` | `30` | stored runs older than this are deleted (at start-up and every 6 hours) |
| `APP_TEMP_DIR` | system temp directory | Gradio's upload and download cache; files are removed after 4 hours |
| `BROWSER_STORAGE_SECRET` | — (off) | turns on keeping the chat and the last result in the browser's localStorage, encrypted with this key, so that they survive a page reload. Without it nothing is stored (safe default for shared computers). Use a fixed random value, e.g. `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`; changing it makes stored data unreadable. The start-up log says whether browser storage is on |

## Optional sources

- **GitHub / GitLab** — `GITHUB_ENABLED`, `GITHUB_TOKEN` (raises the API limit),
  `GITLAB_BASE_URL`, `GITLAB_TOKEN`.
- **Semantic Scholar** — `S2_API_KEY` raises the rate limit.
- **Elasticsearch** index of your own website — `ELASTIC_*`; the field
  mapping (`ELASTIC_FIELD_TITLE`, `…_BODY`, `…_URL`, `…_PATH`) adapts it to
  your index.
- **Institution mode** — `INSTITUTION_PROFILE`, `PERSON_DIRECTORY_DB`,
  `SOLR_*`; see [institution mode](institution-mode.md).

## Advanced tuning

| Variable | Default | Meaning |
|---|---|---|
| `HARVEST_LLM_ENABLE_THINKING` | `false` | thinking mode of the harvest model (sent as `chat_template_kwargs`, a convention of the Qwen family) |
| `EXTRACT_DEDUP_THRESHOLD` | `0.93` | embedding similarity above which two extracts count as duplicates |
| `RATELIMIT_<API>` | per API | limits per external API, as `requests_per_second` or `parallel,min_delay`; `<API>` is one of `CROSSREF`, `OPENALEX`, `S2`, `DBLP`, `ARXIV`, `URL_CHECK`, `WEB_FETCH`, `SEARXNG`. The defaults follow the published limits (arXiv: one request per 3 s) |
| `RATELIMIT_WEBFETCH_PER_DOMAIN` | `0.5` | minimum delay in seconds between two fetches from the same domain |

## Older variable names

These older names are still accepted, with a note at start-up: `RECHERCHE_TEMP_DIR` (now `APP_TEMP_DIR`),
`RECHERCHE_SSL_CERTFILE`, `RECHERCHE_SSL_KEYFILE`, `RECHERCHE_SSL_PASSWORD`
(now `SERVER_SSL_*`), `RECHERCHE_AUTH_USER`, `RECHERCHE_AUTH_PASSWORD`
(now `SERVER_AUTH_*`).
