# Security

## Reporting a vulnerability

Please do not open a public issue for security problems. Use GitHub's
private vulnerability reporting: open the *Security* tab of
<https://github.com/maltedreyer-5/research-toolset> and choose
"Report a vulnerability". You will get an answer within two weeks. Please
include the version, a description and, if possible, steps to reproduce.

## Supported versions

Security fixes are made for the latest release only.

## Security model

research-toolset is meant to be run by an organisation for its own users,
behind its own access control. It is not designed as a multi-tenant public
service.

**Network**

- The server binds to `127.0.0.1` by default. Exposing it
  (`GRADIO_SERVER_NAME=0.0.0.0`) requires authentication and TLS in front of
  it — either the built-in single login (`SERVER_AUTH_*`) or a reverse proxy.
- Every URL the tool fetches — from the plan, from search results, from
  followed links — is checked against internal targets: loopback, private
  (RFC 1918), link-local (including cloud metadata addresses), multicast and
  reserved addresses, numeric host forms such as `127.1` or `0x7f000001`, and
  non-HTTP schemes. The check runs again after DNS resolution and on every
  redirect (an httpx request hook). Internal hosts that must be reachable are
  allowed explicitly with `FETCH_ALLOWED_INTERNAL_HOSTS`.
- Outgoing TLS certificates are verified. For internal services with
  certificates from an internal CA, use `SSL_CERT_FILE`; `TLS_VERIFY=false`
  switches verification off for those internal services only (embedder,
  reranker, person directory, Solr), never for public endpoints.
- The GitHub token is only sent to the GitHub API, not with raw file fetches.

**Interface**

- No telemetry (Gradio analytics and related switches are off before Gradio is
  imported), no share links, no public API documentation; event handlers are
  not exposed as an API.
- Uploads are limited to 20 MB per file; `/etc` and `/root` are blocked for
  file serving.
- The Docker image runs as an unprivileged user.

**Data**

- Uploads and downloads are removed from Gradio's cache after 4 hours;
  completed runs are deleted after `CLEANUP_MAX_AGE_DAYS` (default 30).
- In the institution mode, persons without consent in the person directory
  are never returned, and their content is removed from the search tables.
- Requests, source texts and reports are sent to the configured LLM endpoints.
  Choose endpoints you trust with that content.

**Secrets**

- All credentials come from environment variables; `.env` is excluded from
  the repository and from the Docker build context. Never commit it.
- `searxng-config/settings.yml` contains a placeholder `secret_key`; set your
  own value.

## Known limitations

- **Prompt injection.** Fetched web pages are part of the prompts. A page
  written to manipulate language models can influence extracts and reports.
  The factoid verification and quality checks reduce, but do not remove, this
  risk. Do not treat reports as verified facts, and do not connect the tool to
  systems that act on its output automatically.
- **No per-user separation of stored runs.** Stored runs under `DATA_DIR` are
  plain files; anyone with access to the server's file system can read them.
- **Shared budget.** All users share the configured LLM and API quotas; there
  is no per-user rate limit.
