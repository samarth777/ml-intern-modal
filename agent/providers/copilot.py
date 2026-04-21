"""GitHub Copilot provider for ml-intern.

Copilot is exposed to LiteLLM as a custom OpenAI-compatible endpoint:

    api_base = https://api.githubcopilot.com
    Authorization: Bearer <github_oauth_token>
    + a handful of mandatory editor-identification headers.

Authentication is a single device-code OAuth against github.com (run once
via ``ml-intern copilot login``). The user opens a URL, types a short code,
and we receive a long-lived ``oauth_token`` which is used directly as the
Bearer for every chat request.

Why no session-token exchange?
    The traditional ``api.github.com/copilot_internal/v2/token`` exchange
    is gated to the legacy editor client_id (``Iv1.b507a08c87ecfe98``) and
    returns 404 for the public client we use. The public Copilot endpoint
    accepts the OAuth token directly — this is the same path opencode and
    other modern integrations take.

Credentials live at ``~/.cache/ml-intern/copilot.json`` so the CLI and the
backend share them. Callers obtain a ready-to-use bearer via
:func:`get_oauth_token`.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

# Public Copilot OAuth client-id (matches opencode + every modern editor
# integration). Scope ``read:user`` is sufficient for chat access.
CLIENT_ID = "Ov23li8tweQw6odWQebz"
DEVICE_CODE_URL = "https://github.com/login/device/code"
ACCESS_TOKEN_URL = "https://github.com/login/oauth/access_token"
COPILOT_API_BASE = "https://api.githubcopilot.com"

# Editor headers Copilot's chat endpoint expects. Values are cosmetic but
# Copilot rejects requests missing them — using a plausible VS Code identity
# keeps us out of weird rate-limit buckets.
EDITOR_HEADERS = {
    "Editor-Version": "vscode/1.95.0",
    "Editor-Plugin-Version": "copilot-chat/0.22.0",
    "Copilot-Integration-Id": "vscode-chat",
    "User-Agent": "GitHubCopilotChat/0.22.0",
}


def _cache_path() -> Path:
    """Resolve the on-disk credentials cache path.

    Honours ``ML_INTERN_COPILOT_CACHE`` for tests / non-standard layouts.
    """
    override = os.environ.get("ML_INTERN_COPILOT_CACHE")
    if override:
        return Path(override).expanduser()
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "ml-intern" / "copilot.json"


@dataclass
class CopilotCredentials:
    """Persisted Copilot credentials (just the OAuth token)."""

    oauth_token: str


def load_credentials() -> CopilotCredentials | None:
    """Read credentials from the cache, or return None if not logged in.

    Also honours ``COPILOT_OAUTH_TOKEN`` env var so users can drop a token
    into ``.env`` without running the device flow.
    """
    env_token = os.environ.get("COPILOT_OAUTH_TOKEN")
    if env_token:
        return CopilotCredentials(oauth_token=env_token)

    path = _cache_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Failed to read copilot credentials at %s: %s", path, e)
        return None
    token = data.get("oauth_token", "")
    if not token:
        return None
    return CopilotCredentials(oauth_token=token)


def save_credentials(creds: CopilotCredentials) -> None:
    """Persist credentials to the cache (parents created, mode 600)."""
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(creds), indent=2))
    try:
        path.chmod(0o600)
    except OSError:
        # Non-POSIX filesystems may not support chmod — best effort only.
        pass


def clear_credentials() -> bool:
    """Delete cached credentials. Returns True if a file was removed."""
    path = _cache_path()
    if path.exists():
        path.unlink()
        return True
    return False


# ── OAuth device flow ────────────────────────────────────────────────────


@dataclass
class DeviceCode:
    verification_uri: str
    user_code: str
    device_code: str
    interval: int  # poll interval in seconds


def request_device_code() -> DeviceCode:
    """Initiate the GitHub OAuth device flow."""
    response = requests.post(
        DEVICE_CODE_URL,
        headers={"Accept": "application/json"},
        data={"client_id": CLIENT_ID, "scope": "read:user"},
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()
    return DeviceCode(
        verification_uri=data["verification_uri"],
        user_code=data["user_code"],
        device_code=data["device_code"],
        interval=int(data.get("interval", 5)),
    )


def poll_for_oauth_token(device: DeviceCode, *, max_wait_s: int = 600) -> str:
    """Poll GitHub until the user completes auth; return the OAuth token.

    Raises:
        TimeoutError: User did not authorize within ``max_wait_s``.
        RuntimeError: GitHub returned a non-recoverable error.
    """
    start = time.monotonic()
    interval = device.interval
    while time.monotonic() - start < max_wait_s:
        time.sleep(interval)
        response = requests.post(
            ACCESS_TOKEN_URL,
            headers={"Accept": "application/json"},
            data={
                "client_id": CLIENT_ID,
                "device_code": device.device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
            timeout=10,
        )
        if not response.ok:
            raise RuntimeError(
                f"GitHub token endpoint returned HTTP {response.status_code}"
            )
        data = response.json()
        if token := data.get("access_token"):
            return token
        error = data.get("error")
        if error == "authorization_pending":
            continue
        if error == "slow_down":
            # RFC 8628 §3.5: bump interval by 5s.
            interval = int(data.get("interval", interval + 5))
            continue
        if error:
            raise RuntimeError(
                f"OAuth failed: {error} ({data.get('error_description')})"
            )
    raise TimeoutError("Timed out waiting for user to authorize the device.")


# ── Token retrieval ──────────────────────────────────────────────────────


def get_oauth_token() -> str:
    """Return the cached OAuth token, used directly as the Copilot bearer.

    Raises:
        RuntimeError: Not logged in.
    """
    creds = load_credentials()
    if creds is None or not creds.oauth_token:
        raise RuntimeError(
            "Not logged in to GitHub Copilot. Run `ml-intern copilot login`."
        )
    return creds.oauth_token


def verify_copilot_access(token: str | None = None) -> None:
    """Sanity-check that the token has Copilot access by hitting /models.

    Raises:
        RuntimeError: Token is invalid or the account has no Copilot subscription.
    """
    token = token or get_oauth_token()
    response = requests.get(
        f"{COPILOT_API_BASE}/models",
        headers={"Authorization": f"Bearer {token}", **EDITOR_HEADERS},
        timeout=10,
    )
    if response.status_code == 401:
        raise RuntimeError(
            "Copilot rejected your GitHub token (401). The account likely has "
            "no active Copilot subscription. Verify at https://github.com/settings/copilot."
        )
    if not response.ok:
        raise RuntimeError(
            f"Copilot /models returned HTTP {response.status_code}: {response.text[:200]}"
        )


def status() -> dict[str, Any]:
    """Return a small dict describing current login state (for CLI display)."""
    creds = load_credentials()
    if creds is None:
        return {"logged_in": False}
    return {
        "logged_in": True,
        "cache_path": str(_cache_path()),
        "token_prefix": creds.oauth_token[:8] + "…",
    }
