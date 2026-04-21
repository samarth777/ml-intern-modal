"""Modal deployment of the ml-intern FastAPI backend.

Wraps ``backend/main.py:app`` in a ``@modal.asgi_app`` so the entire
agent UI + API can be deployed with::

    modal deploy modal_app.py

Design notes
------------
* The container is pinned to ``min_containers=1, max_containers=1``. The
  backend keeps per-session agent state in process memory (see
  ``backend/session_manager.py``) so we cannot horizontally scale without
  first moving session state to a shared store. One always-warm container
  also avoids cold-start latency on the websocket/SSE endpoints.

* ``backend/main.py`` imports ``routes.agent`` and ``routes.auth`` as
  top-level modules (the Dockerfile sets ``WORKDIR=/app/backend``). To
  preserve that import layout we mount ``backend/`` at ``/root/backend``
  and prepend it to ``sys.path`` inside the container.

* The frontend ``static/`` build is added as a local dir. Build it once
  with ``cd frontend && npm install && npm run build && rm -rf
  ../static && cp -r dist ../static`` before running ``modal deploy``.

* All runtime secrets (HF / Anthropic / OAuth) come from a single
  ``modal.Secret`` named ``ml-intern-secrets``. Create it with::

      modal secret create ml-intern-secrets \\
          ANTHROPIC_API_KEY=... HF_TOKEN=... \\
          OAUTH_CLIENT_ID=... OAUTH_CLIENT_SECRET=... \\
          OPENID_PROVIDER_URL=https://huggingface.co \\
          HF_OAUTH_ORG_ID=698dbf55845d85df163175f1
"""

from __future__ import annotations

from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).parent
STATIC_DIR = PROJECT_ROOT / "static"

# ---- Image -----------------------------------------------------------------
# Install Python deps from pyproject.toml using uv. We deliberately do *not*
# copy the lockfile because Modal images cache by hash of the build steps,
# and we want a single source of truth in pyproject.toml.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl")
    .pip_install("uv")
    .add_local_file(PROJECT_ROOT / "pyproject.toml", "/tmp/pyproject.toml", copy=True)
    .run_commands(
        "cd /tmp && uv pip install --system --no-cache -r pyproject.toml",
        "uv pip install --system --no-cache modal>=0.66",
    )
    # Mount source code as Python modules so imports resolve.
    .add_local_python_source("agent")
    .add_local_dir(PROJECT_ROOT / "backend", "/root/backend")
    .add_local_dir(PROJECT_ROOT / "configs", "/root/configs")
)

# Optionally bundle the prebuilt frontend.
if STATIC_DIR.exists():
    image = image.add_local_dir(STATIC_DIR, "/root/static")

app = modal.App("ml-intern")

secrets = [modal.Secret.from_name("ml-intern-secrets")]


@app.function(
    image=image,
    secrets=secrets,
    min_containers=1,
    max_containers=1,
    timeout=60 * 60,  # 1h per request ceiling (long agent turns + SSE)
    scaledown_window=60 * 20,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def fastapi_app():
    import sys

    # backend/main.py uses `from routes.agent import ...` (no `backend.`
    # prefix), so /root/backend must be on sys.path before importing it.
    sys.path.insert(0, "/root/backend")
    sys.path.insert(0, "/root")

    from backend.main import app as web_app  # noqa: WPS433  (runtime import)

    return web_app
