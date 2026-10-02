# Deployment

research-toolset is one Python process. It needs an OpenAI-compatible LLM
endpoint and a SearXNG instance; everything else is optional.

## Local

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # edit
python app.py
```

The server binds to `127.0.0.1:7860`. To make it reachable from other
machines, set `GRADIO_SERVER_NAME=0.0.0.0` — and put authentication and TLS in
front of it (see below).

Optional extras (`requirements-optional.txt`): `unstructured` for legacy
Office formats (`.doc`, `.ppt`, `.xls`) and `playwright` for rendering
JavaScript-heavy pages (`USE_PLAYWRIGHT=true`, then
`playwright install chromium`).

## Windows

The tool runs natively on Windows with Python 3.12: all dependencies are
available as Windows packages. In PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env    # edit
python app.py
```

For Docker on Windows use Docker Desktop with the WSL 2 backend (Linux
containers); the commands below are the same. Keep `.env` with LF line
endings — the repository's `.gitattributes` does this for files you check out;
an editor that saves new files with Windows line endings can add a trailing
character to every value in `--env-file`.

## Docker

```bash
docker build -t research-toolset .
docker run --rm -p 7860:7860 --env-file .env \
    -v research-data:/app/data research-toolset
```

The image runs as an unprivileged user (uid 10001), binds to `0.0.0.0` inside
the container, keeps Gradio's cache in `/var/cache/research-toolset` and has a
health check against the web interface. Mount a volume on `/app/data` to keep
stored runs across container restarts; the retention job deletes them after
`CLEANUP_MAX_AGE_DAYS` either way.

### With SearXNG

An example Compose setup with SearXNG and a Valkey/Redis cache. It is a
starting point, not a tested reference: SearXNG's configuration keys change
between releases (recent versions call the cache `valkey`), so check the
SearXNG documentation for your version.

```yaml
services:
  app:
    build: .
    env_file: .env          # SEARXNG_BASE_URL=http://searxng:8080
    ports: ["127.0.0.1:7860:7860"]
    volumes: ["research-data:/app/data"]
    depends_on: [searxng]
  searxng:
    image: searxng/searxng:latest
    environment:
      SEARXNG_SECRET: "set-a-long-random-value"
    volumes: ["./searxng-config:/etc/searxng"]
    depends_on: [redis]
  redis:
    image: valkey/valkey:8-alpine
volumes:
  research-data:
```

`searxng-config/settings.yml` enables the JSON format the tool needs, selects
web and scholarly engines and disables unrelated ones. Its `secret_key` is a
placeholder: set a random value there or through `SEARXNG_SECRET`.

## Behind a reverse proxy

Serve the tool under a sub-path with `ROOT_PATH`, e.g. `ROOT_PATH=/research`
behind `https://tools.example.org/research/`. The proxy must pass WebSocket
and server-sent-event connections through (Gradio streams results) and should
allow long-running requests — a research run takes minutes.

TLS is best terminated at the proxy. Alternatively the app can serve TLS
itself with `SERVER_SSL_CERTFILE` and `SERVER_SSL_KEYFILE`.

## Access control

The tool has no user management. Options:

- a single login with `SERVER_AUTH_USER` and `SERVER_AUTH_PASSWORD`;
- authentication at the reverse proxy (SSO, basic auth, network restriction).

Everyone with access shares the configured LLM and search budget. Research
results are kept per browser session and are not visible to other users; the
stored run directories under `DATA_DIR` are, however, readable by whoever can
read the server's file system.

## Data handling

| Data | Where | Removed |
|---|---|---|
| Uploaded files, downloads | Gradio cache (`APP_TEMP_DIR`) | by Gradio after 4 hours |
| Completed runs | `DATA_DIR/<id>_<slug>/` | by the retention job after `CLEANUP_MAX_AGE_DAYS` (checked at start-up and every 6 hours) |
| Logs | standard output | by your log collection |

The request, fetched source texts and the report are sent to the configured
LLM endpoint(s) and, for search terms, to SearXNG and the enabled source
APIs. Choose endpoints accordingly if requests may contain personal data.

## Upgrading

Settings are environment variables; older names of a few variables are still
accepted (see [configuration](configuration.md#older-variable-names)).
Stored runs are not read back, so a change of their layout needs no
migration.
