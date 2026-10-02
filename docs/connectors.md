# Connectors

A connector gives the tool access to one kind of source: a search engine,
a website, a repository host, a search index, a directory. Which connectors an
installation has makes a large difference to the results — an archive that
connects its own catalogue or portal gets different reports than one that
relies on web search alone.

## The interface

Every connector implements `BaseConnector` (`src/connectors/base.py`):

```python
class BaseConnector(ABC):
    name: str = "base"

    async def search(self, query: str, max_results: int = 10, **kwargs) -> list[SearchResult]: ...
    async def fetch(self, url: str) -> SourceDocument: ...
    def can_handle(self, url: str) -> bool: ...
    async def close(self): ...
```

`search` returns `SearchResult`s (title, URL, snippet); `fetch` returns a
`SourceDocument` with the text the fact extraction reads, plus title,
metadata and, if known, the publication date (`published_date`, ISO 8601 —
it helps the model date statements correctly). Both types are in
`src/pipeline/models.py`.

## Registration

Connectors are registered once at start-up in `_ensure_shared_resources()`
in `src/ui/gradio_app.py`:

```python
_shared_connectors.register(
    MyPortalConnector(config),
    url_patterns=[r"portal\.example-archive\.org/"],
)
```

## Two kinds of integration

**Fetching a site better — no further changes needed.** When a URL matches a
connector's `url_patterns`, the registry routes it to that connector instead
of the generic web scraper (`ConnectorRegistry.route_url`). This is the easy
case: for a portal with a structured interface (an API, IIIF manifests, a
record view with clean metadata), a connector that turns a record URL into
clean text and metadata improves every research run that finds such URLs,
whether they come from web search, the plan or followed links.

**A new search source — needs wiring in the pipeline.** Search sources are
called explicitly by the research orchestrator
(`src/pipeline/orchestrator.py`, `_run_search_and_fetch`), and the planner has
to know that the source exists: the plan chooses a `source_scope` per research
question, and the analysis prompt describes the available sources
(`available_connectors`). A new search source therefore needs

1. the connector,
2. a source scope (`VALID_SOURCE_SCOPES` in `src/pipeline/plan_validation.py`)
   and a short description for the planner,
3. a branch in `_run_search_and_fetch` that queries it for questions with that
   scope, following the Elasticsearch connector as the closest example,
4. tests, with the connector replaced by a double (see
   `tests/test_pipeline_smoke.py` for the pattern).

## Rules for connectors

- **Outgoing requests** go through an `httpx.AsyncClient` with the SSRF guard:
  `event_hooks={"request": [make_request_guard(allowed_hosts)]}` from
  `src/core/url_security.py`. Keep TLS verification on; only connectors for
  internal services you run yourself use `verify=tls_verify()` from
  `src/core/tls.py`, so that `TLS_VERIFY=false` can apply to them.
- **Credentials** come from environment variables, never from code; add every
  new variable to `.env.example` and `docs/configuration.md`.
- **Rate limits** of external services belong in `src/connectors/rate_limiter.py`.
- **Failures** are returned, not raised through the pipeline: a connector that is
  down should cost one source, not the run.
