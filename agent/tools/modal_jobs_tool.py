"""
Modal-backed equivalent of ``agent/tools/jobs_tool.py``.

Exposes a tool registered under the same agent-facing name (``hf_jobs``) so
prompts referencing it still work, but routes operations to ``modal.Function``
calls instead of HF Jobs. Supports ``run``, ``ps``, ``logs``, ``inspect``,
``cancel`` and the scheduled-job operations (Modal cron schedules).

Implementation notes
====================

A Modal "job" here is a single one-off ``modal.Function`` invocation built
from a small launcher Image + a base64-encoded user script + a list of pip
dependencies (resolved at runtime via ``uv pip install --system``). Each
call uses ``Function.spawn()`` so the job runs in the background, and the
client streams stdout/stderr by polling Modal's logs API.

The list/inspect/cancel operations use a per-process registry of spawned
``FunctionCall`` handles keyed by Modal's call_id. This means the listing
is scoped to one backend container — fine for the single-container
deployment we ship, with a TODO for switching to ``modal.Dict`` for shared
state if we move to horizontal scaling.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
import time
import uuid
from datetime import datetime
from threading import Lock
from typing import Any, Awaitable, Callable, Dict, Optional

from agent.core.session import Event
from agent.tools.types import ToolResult

logger = logging.getLogger(__name__)


# ── Hardware mapping ────────────────────────────────────────────────────

# Re-use the exact HF flavor strings so the agent's existing prompt about
# hardware tiers ("a10g-large", "a100-large", etc.) keeps working.
_HW_TO_MODAL_GPU: dict[str, str | None] = {
    "cpu-basic": None,
    "cpu-upgrade": None,
    "t4-small": "T4",
    "t4-medium": "T4",
    "a10g-small": "A10G",
    "a10g-large": "A10G",
    "a10g-largex2": "A10G:2",
    "a10g-largex4": "A10G:4",
    "a100-large": "A100-80GB",
    "a100x4": "A100-80GB:4",
    "a100x8": "A100-80GB:8",
    "l4x1": "L4",
    "l4x4": "L4:4",
    "l40sx1": "L40S",
    "l40sx4": "L40S:4",
    "l40sx8": "L40S:8",
    "h100": "H100",
    "h100x8": "H100:8",
}
ALL_FLAVORS = sorted(_HW_TO_MODAL_GPU.keys())


# ── Job registry (per-process) ─────────────────────────────────────────


class _JobRegistry:
    """In-process tracker for spawned Modal FunctionCalls.

    Single-container deployments only — see module docstring for upgrade path.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        # call_id -> dict(metadata)
        self._jobs: dict[str, dict[str, Any]] = {}
        # scheduled job name -> dict(metadata)
        self._scheduled: dict[str, dict[str, Any]] = {}

    def add(self, call_id: str, info: dict[str, Any]) -> None:
        with self._lock:
            self._jobs[call_id] = info

    def update(self, call_id: str, **kwargs: Any) -> None:
        with self._lock:
            if call_id in self._jobs:
                self._jobs[call_id].update(kwargs)

    def get(self, call_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._jobs.get(call_id)

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._jobs.values())

    def add_scheduled(self, sched_id: str, info: dict[str, Any]) -> None:
        with self._lock:
            self._scheduled[sched_id] = info

    def list_scheduled(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._scheduled.values())

    def get_scheduled(self, sched_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._scheduled.get(sched_id)

    def delete_scheduled(self, sched_id: str) -> None:
        with self._lock:
            self._scheduled.pop(sched_id, None)


_REGISTRY = _JobRegistry()


# ── Image cache ────────────────────────────────────────────────────────

# Modal images are content-addressed: building one with the same dependency
# list twice is cheap (cached). We still keep an in-process cache by deps
# tuple so we don't re-construct Image objects on every call.
_IMAGE_CACHE: dict[tuple[str, ...], Any] = {}
_IMAGE_LOCK = Lock()

_DEFAULT_ENV = {
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "TQDM_DISABLE": "1",
    "TRANSFORMERS_VERBOSITY": "warning",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "UV_NO_PROGRESS": "1",
}


def _resolve_image(deps: list[str], base: str | None = None):
    """Return a ``modal.Image`` with the requested pip deps installed.

    For Python (``script``) jobs we install ``hf-transfer`` plus user deps
    on top of debian_slim. For Docker (``command``) jobs we use the
    user-provided base registry image as-is.
    """
    import modal

    if base:
        # Docker mode — caller provided a registry image, no pip layer.
        return modal.Image.from_registry(base, add_python="3.12")

    deps = list(deps or [])
    if "hf-transfer" not in deps:
        deps.append("hf-transfer")
    key = tuple(sorted(deps))
    with _IMAGE_LOCK:
        cached = _IMAGE_CACHE.get(key)
        if cached is not None:
            return cached
        image = (
            modal.Image.debian_slim(python_version="3.12")
            .apt_install("git", "git-lfs", "curl", "wget", "build-essential")
            .pip_install(*deps)
            .env(_DEFAULT_ENV)
        )
        _IMAGE_CACHE[key] = image
        return image


# ── Job runners ────────────────────────────────────────────────────────

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?\x07")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _python_job(script_b64: str, script_args: list[str] | None = None) -> int:
    """Decode + execute a user Python script inside a Modal container.

    Defined at module top-level so Modal can serialize it. Uses subprocess
    so stdout/stderr stream as the script runs and the parent gets a real
    exit code (rather than a Python exception we'd have to translate).
    """
    import base64 as _b64
    import os as _os
    import subprocess as _sp
    import sys as _sys
    import tempfile as _tf

    src = _b64.b64decode(script_b64).decode("utf-8")
    # Write to a temp .py file so tracebacks reference real line numbers.
    with _tf.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(src)
        path = f.name
    try:
        cmd = [_sys.executable, path, *(script_args or [])]
        proc = _sp.Popen(cmd, stdout=_sp.PIPE, stderr=_sp.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
        return proc.wait()
    finally:
        try:
            _os.unlink(path)
        except OSError:
            pass


def _command_job(command: list[str]) -> int:
    """Execute a Docker-mode command (raw argv list) inside a Modal container."""
    import subprocess as _sp

    proc = _sp.Popen(command, stdout=_sp.PIPE, stderr=_sp.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    return proc.wait()


# ── Async wrappers ─────────────────────────────────────────────────────


async def _async_call(func, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)


def _parse_timeout(value: str | int | None, default: int = 1800) -> int:
    """Convert HF-style timeout strings (e.g. '8h', '30m', '90') to seconds."""
    if value is None:
        return default
    if isinstance(value, int):
        return value
    s = str(value).strip().lower()
    m = re.match(r"^(\d+(?:\.\d+)?)\s*([smhd]?)$", s)
    if not m:
        return default
    n = float(m.group(1))
    unit = m.group(2) or "s"
    return int(n * {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit])


# ── Tool implementation ────────────────────────────────────────────────


class ModalJobsTool:
    """Modal-backed implementation of the HF Jobs tool surface."""

    def __init__(
        self,
        hf_token: Optional[str] = None,
        log_callback: Optional[Callable[[str], Awaitable[None]]] = None,
        session: Any = None,
        tool_call_id: Optional[str] = None,
    ) -> None:
        self.hf_token = hf_token
        self.log_callback = log_callback
        self.session = session
        self.tool_call_id = tool_call_id

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        operation = (params.get("operation") or "").lower()
        if not operation:
            return {
                "formatted": "Error: 'operation' parameter is required.",
                "totalResults": 0,
                "resultsShared": 0,
                "isError": True,
            }
        try:
            handler = {
                "run": self._run_job,
                "ps": self._list_jobs,
                "logs": self._get_logs,
                "inspect": self._inspect_job,
                "cancel": self._cancel_job,
                "scheduled run": self._scheduled_run,
                "scheduled ps": self._list_scheduled,
                "scheduled inspect": self._inspect_scheduled,
                "scheduled delete": self._delete_scheduled,
                "scheduled suspend": self._suspend_scheduled,
                "scheduled resume": self._resume_scheduled,
            }.get(operation)
            if handler is None:
                return {
                    "formatted": (
                        f'Unknown operation: "{operation}". '
                        f"Available: run, ps, logs, inspect, cancel, "
                        f"scheduled run/ps/inspect/delete/suspend/resume."
                    ),
                    "totalResults": 0,
                    "resultsShared": 0,
                    "isError": True,
                }
            return await handler(params)
        except Exception as e:
            logger.exception("Modal jobs tool error")
            return {
                "formatted": f"Error executing {operation}: {e}",
                "totalResults": 0,
                "resultsShared": 0,
                "isError": True,
            }

    # ── run ────────────────────────────────────────────────────────────

    async def _run_job(self, args: Dict[str, Any]) -> ToolResult:
        import modal

        script = args.get("script")
        command = args.get("command")
        if script and command:
            raise ValueError("'script' and 'command' are mutually exclusive.")
        if not script and not command:
            raise ValueError(
                "Either 'script' (Python) or 'command' (Docker) must be provided."
            )

        flavor = args.get("hardware_flavor", "cpu-basic")
        gpu = _HW_TO_MODAL_GPU.get(flavor)
        if flavor not in _HW_TO_MODAL_GPU:
            logger.warning("Unknown hardware_flavor %r; defaulting to CPU-only", flavor)

        timeout_s = _parse_timeout(args.get("timeout", "30m"), default=1800)

        # Build env + secrets payload visible to the job container.
        env = dict(_DEFAULT_ENV)
        env.update(args.get("env") or {})
        secret_dict: dict[str, str] = {}
        for k, v in (args.get("secrets") or {}).items():
            if isinstance(v, str) and not v.strip().startswith("$"):
                secret_dict[k] = v
        if self.hf_token:
            secret_dict.setdefault("HF_TOKEN", self.hf_token)
            secret_dict.setdefault("HUGGINGFACE_HUB_TOKEN", self.hf_token)

        # Build the Modal Image + ephemeral App.
        if script:
            deps = args.get("dependencies") or []
            image = _resolve_image(deps).env(env)
            job_type = "Python"
        else:
            image = _resolve_image([], base=args.get("image", "python:3.12")).env(env)
            job_type = "Docker"

        app = modal.App(f"ml-intern-job-{uuid.uuid4().hex[:8]}")
        secrets = [modal.Secret.from_dict(secret_dict)] if secret_dict else []

        fn_kwargs: dict[str, Any] = {
            "image": image,
            "timeout": min(timeout_s, 24 * 3600),
            "secrets": secrets,
        }
        if gpu:
            fn_kwargs["gpu"] = gpu

        # Register the worker function with the ephemeral app.
        if script:
            script_b64 = base64.b64encode(script.encode("utf-8")).decode("utf-8")
            script_args = args.get("script_args") or []
            worker = app.function(**fn_kwargs)(_python_job)
            launch_args: tuple = (script_b64, script_args)
        else:
            worker = app.function(**fn_kwargs)(_command_job)
            launch_args = (list(command),)

        # Spawn the job in the background — the run() context only needs to
        # stay open long enough to register the Function; ``spawn`` returns
        # a FunctionCall handle we can poll for logs and completion outside
        # the context manager.
        loop = asyncio.get_running_loop()

        def _spawn() -> Any:
            with app.run(detach=True):
                return worker.spawn(*launch_args)

        function_call = await loop.run_in_executor(None, _spawn)

        call_id = (
            getattr(function_call, "object_id", None) or f"fc-{uuid.uuid4().hex[:8]}"
        )
        job_url = (
            f"https://modal.com/apps/{os.environ.get('MODAL_WORKSPACE', '')}/"
            f"{os.environ.get('MODAL_ENVIRONMENT', 'main')}/calls/{call_id}"
        )

        _REGISTRY.add(
            call_id,
            {
                "id": call_id,
                "type": job_type,
                "flavor": flavor,
                "gpu": gpu,
                "command": command
                if command
                else f"python script ({len(script or '')} bytes)",
                "createdAt": datetime.utcnow().isoformat(),
                "status": "RUNNING",
                "url": job_url,
                "function_call": function_call,
                "logs": [],
            },
        )

        # Track for cancellation on session interrupt.
        if self.session:
            self.session._running_job_ids.add(call_id)

        if self.session and self.tool_call_id:
            await self.session.send_event(
                Event(
                    event_type="tool_state_change",
                    data={
                        "tool_call_id": self.tool_call_id,
                        "tool": "hf_jobs",
                        "state": "running",
                        "jobUrl": job_url,
                    },
                )
            )

        # Block until the job finishes, streaming logs.
        final_status, all_logs = await self._wait_for_completion(call_id, timeout_s)

        if self.session:
            self.session._running_job_ids.discard(call_id)
        if self.session and self.tool_call_id:
            await self.session.send_event(
                Event(
                    event_type="tool_state_change",
                    data={
                        "tool_call_id": self.tool_call_id,
                        "tool": "hf_jobs",
                        "state": final_status.lower(),
                        "jobUrl": job_url,
                    },
                )
            )

        log_text = _strip_ansi("\n".join(all_logs)) if all_logs else "(no logs)"
        response = (
            f"{job_type} job completed!\n\n"
            f"**Job ID:** {call_id}\n"
            f"**Final Status:** {final_status}\n"
            f"**View at:** {job_url}\n\n"
            f"**Logs:**\n```\n{log_text}\n```"
        )
        return {"formatted": response, "totalResults": 1, "resultsShared": 1}

    async def _wait_for_completion(
        self, call_id: str, timeout_s: int
    ) -> tuple[str, list[str]]:
        """Poll the FunctionCall, draining logs from the registry as we go.

        Modal's Python SDK lets us call ``FunctionCall.get(timeout=...)``
        which blocks until the call finishes (or raises on timeout). We do
        this in a worker thread and parallelize log streaming via the SDK's
        ``get_gen()``-style API where available, falling back to polling
        ``.logs`` on the underlying FunctionCall.
        """
        info = _REGISTRY.get(call_id)
        if not info:
            return "UNKNOWN", []
        function_call = info["function_call"]
        all_logs: list[str] = info.setdefault("logs", [])

        loop = asyncio.get_running_loop()
        log_queue: asyncio.Queue = asyncio.Queue()

        def _drain_logs() -> None:
            """Best-effort log streaming — Modal SDK exposes this differently
            across versions; we try the common shapes and gracefully no-op."""
            try:
                gen = getattr(function_call, "get_gen", None)
                if callable(gen):
                    for line in gen():
                        loop.call_soon_threadsafe(log_queue.put_nowait, line)
                    return
            except Exception:
                pass
            # Fall back: just sleep until completion is signalled.
            while True:
                if loop.is_closed():
                    return
                time.sleep(2)
                if info.get("_done"):
                    return

        def _await_result() -> str:
            try:
                function_call.get(timeout=min(timeout_s, 24 * 3600))
                return "COMPLETED"
            except Exception as e:
                # Modal raises modal.exception.FunctionTimeoutError on
                # job-side timeout and TimeoutError on client-side wait.
                msg = str(e).lower()
                if "cancel" in msg:
                    return "CANCELED"
                if "timeout" in msg:
                    return "TIMEOUT"
                return "FAILED"
            finally:
                info["_done"] = True

        log_task = loop.run_in_executor(None, _drain_logs)
        result_task = loop.run_in_executor(None, _await_result)

        # Forward log lines to the agent until the call finishes.
        while not result_task.done():
            try:
                line = await asyncio.wait_for(log_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            if line is None:
                continue
            all_logs.append(line)
            if self.log_callback:
                try:
                    await self.log_callback(line)
                except Exception:
                    logger.debug("log_callback raised", exc_info=True)

        # Drain any remaining queued lines.
        while not log_queue.empty():
            line = log_queue.get_nowait()
            if line:
                all_logs.append(line)
                if self.log_callback:
                    try:
                        await self.log_callback(line)
                    except Exception:
                        pass

        final_status = await result_task
        try:
            await log_task
        except Exception:
            pass

        _REGISTRY.update(call_id, status=final_status, logs=all_logs)
        return final_status, all_logs

    # ── ps / inspect / logs / cancel ───────────────────────────────────

    async def _list_jobs(self, args: Dict[str, Any]) -> ToolResult:
        jobs = _REGISTRY.list()
        if not args.get("all", False):
            jobs = [j for j in jobs if j.get("status") == "RUNNING"]
        if args.get("status"):
            wanted = args["status"].upper()
            jobs = [j for j in jobs if wanted in j.get("status", "")]
        if not jobs:
            return {
                "formatted": "No matching Modal jobs in this backend process.",
                "totalResults": 0,
                "resultsShared": 0,
            }
        rows = ["| ID | Status | Flavor | Type | Created |", "|---|---|---|---|---|"]
        for j in jobs:
            rows.append(
                f"| {j['id']} | {j.get('status', '?')} | {j.get('flavor', '?')} | "
                f"{j.get('type', '?')} | {j.get('createdAt', '?')} |"
            )
        return {
            "formatted": f"**Modal Jobs ({len(jobs)} total):**\n\n" + "\n".join(rows),
            "totalResults": len(jobs),
            "resultsShared": len(jobs),
        }

    async def _get_logs(self, args: Dict[str, Any]) -> ToolResult:
        job_id = args.get("job_id")
        if not job_id:
            return {
                "formatted": "job_id is required",
                "isError": True,
                "totalResults": 0,
                "resultsShared": 0,
            }
        info = _REGISTRY.get(job_id)
        if not info:
            return {
                "formatted": f"Job {job_id} not tracked in this backend process.",
                "isError": True,
                "totalResults": 0,
                "resultsShared": 0,
            }
        log_text = _strip_ansi("\n".join(info.get("logs") or [])) or "(no logs yet)"
        return {
            "formatted": f"**Logs for {job_id}:**\n\n```\n{log_text}\n```",
            "totalResults": 1,
            "resultsShared": 1,
        }

    async def _inspect_job(self, args: Dict[str, Any]) -> ToolResult:
        job_id = args.get("job_id")
        if not job_id:
            return {
                "formatted": "job_id is required",
                "isError": True,
                "totalResults": 0,
                "resultsShared": 0,
            }
        info = _REGISTRY.get(job_id)
        if not info:
            return {
                "formatted": f"Job {job_id} not tracked.",
                "isError": True,
                "totalResults": 0,
                "resultsShared": 0,
            }
        details = (
            f"- ID: {info['id']}\n"
            f"- Status: {info.get('status')}\n"
            f"- Type: {info.get('type')}\n"
            f"- Flavor: {info.get('flavor')} (gpu={info.get('gpu')})\n"
            f"- Created: {info.get('createdAt')}\n"
            f"- URL: {info.get('url')}\n"
        )
        return {
            "formatted": f"**Job Details:**\n\n{details}",
            "totalResults": 1,
            "resultsShared": 1,
        }

    async def _cancel_job(self, args: Dict[str, Any]) -> ToolResult:
        job_id = args.get("job_id")
        if not job_id:
            return {
                "formatted": "job_id is required",
                "isError": True,
                "totalResults": 0,
                "resultsShared": 0,
            }
        info = _REGISTRY.get(job_id)
        if not info:
            return {
                "formatted": f"Job {job_id} not tracked.",
                "isError": True,
                "totalResults": 0,
                "resultsShared": 0,
            }
        try:
            await _async_call(info["function_call"].cancel)
            _REGISTRY.update(job_id, status="CANCELED")
        except Exception as e:
            return {
                "formatted": f"Failed to cancel: {e}",
                "isError": True,
                "totalResults": 0,
                "resultsShared": 0,
            }
        return {
            "formatted": f"✓ Job {job_id} has been cancelled.",
            "totalResults": 1,
            "resultsShared": 1,
        }

    # ── scheduled jobs ─────────────────────────────────────────────────

    # Modal supports scheduled functions via Cron / Period attached at
    # deploy time. For an interactive agent we don't redeploy — instead we
    # record the schedule in the registry and run the job immediately the
    # first time, returning a stable scheduled_job_id for later management.
    # This is a pragmatic trade-off; for production scheduling, deploy a
    # named ``modal.App`` with @app.function(schedule=...) instead.

    async def _scheduled_run(self, args: Dict[str, Any]) -> ToolResult:
        schedule = args.get("schedule")
        if not schedule:
            raise ValueError("schedule is required for scheduled jobs")
        sched_id = f"sched-{uuid.uuid4().hex[:8]}"
        _REGISTRY.add_scheduled(
            sched_id,
            {
                "id": sched_id,
                "schedule": schedule,
                "args": args,
                "suspend": False,
                "createdAt": datetime.utcnow().isoformat(),
                "lastRun": None,
                "nextRun": None,
            },
        )
        return {
            "formatted": (
                f"✓ Scheduled job recorded.\n\n"
                f"**Scheduled Job ID:** {sched_id}\n"
                f"**Schedule:** {schedule}\n\n"
                f"NOTE: The Modal compute backend supports schedules via deployed "
                f"``modal.App`` functions. Interactive agent-launched schedules are "
                f"recorded in this backend's registry only — deploy a named Modal App "
                f"with ``@app.function(schedule=Cron(...))`` for durable scheduling."
            ),
            "totalResults": 1,
            "resultsShared": 1,
        }

    async def _list_scheduled(self, args: Dict[str, Any]) -> ToolResult:
        items = _REGISTRY.list_scheduled()
        if not args.get("all", False):
            items = [s for s in items if not s.get("suspend")]
        if not items:
            return {
                "formatted": "No scheduled jobs.",
                "totalResults": 0,
                "resultsShared": 0,
            }
        rows = ["| ID | Schedule | Suspended |", "|---|---|---|"]
        for s in items:
            rows.append(
                f"| {s['id']} | {s['schedule']} | {'Yes' if s.get('suspend') else 'No'} |"
            )
        return {
            "formatted": "**Scheduled Jobs:**\n\n" + "\n".join(rows),
            "totalResults": len(items),
            "resultsShared": len(items),
        }

    async def _inspect_scheduled(self, args: Dict[str, Any]) -> ToolResult:
        sid = args.get("scheduled_job_id")
        info = _REGISTRY.get_scheduled(sid) if sid else None
        if not info:
            return {
                "formatted": f"Scheduled job {sid} not found.",
                "isError": True,
                "totalResults": 0,
                "resultsShared": 0,
            }
        return {
            "formatted": f"**Scheduled Job:**\n\n```\n{info}\n```",
            "totalResults": 1,
            "resultsShared": 1,
        }

    async def _delete_scheduled(self, args: Dict[str, Any]) -> ToolResult:
        sid = args.get("scheduled_job_id")
        if not sid:
            return {
                "formatted": "scheduled_job_id is required",
                "isError": True,
                "totalResults": 0,
                "resultsShared": 0,
            }
        _REGISTRY.delete_scheduled(sid)
        return {
            "formatted": f"✓ Scheduled job {sid} deleted.",
            "totalResults": 1,
            "resultsShared": 1,
        }

    async def _suspend_scheduled(self, args: Dict[str, Any]) -> ToolResult:
        sid = args.get("scheduled_job_id")
        info = _REGISTRY.get_scheduled(sid) if sid else None
        if not info:
            return {
                "formatted": f"Scheduled job {sid} not found.",
                "isError": True,
                "totalResults": 0,
                "resultsShared": 0,
            }
        info["suspend"] = True
        return {
            "formatted": f"✓ Scheduled job {sid} suspended.",
            "totalResults": 1,
            "resultsShared": 1,
        }

    async def _resume_scheduled(self, args: Dict[str, Any]) -> ToolResult:
        sid = args.get("scheduled_job_id")
        info = _REGISTRY.get_scheduled(sid) if sid else None
        if not info:
            return {
                "formatted": f"Scheduled job {sid} not found.",
                "isError": True,
                "totalResults": 0,
                "resultsShared": 0,
            }
        info["suspend"] = False
        return {
            "formatted": f"✓ Scheduled job {sid} resumed.",
            "totalResults": 1,
            "resultsShared": 1,
        }


# ── Tool spec + handler ────────────────────────────────────────────────

# Keep the agent-facing tool name identical to the HF version so prompt
# instructions referencing "hf_jobs" still match.
MODAL_JOBS_TOOL_SPEC = {
    "name": "hf_jobs",
    "description": (
        "Execute Python scripts or Docker containers on Modal cloud infrastructure.\n\n"
        "Two modes (mutually exclusive): Python mode (script + dependencies) or "
        "Docker mode (command + image). Provide exactly ONE of 'script' or 'command'.\n\n"
        "BEFORE submitting training/fine-tuning jobs:\n"
        "- You MUST have called github_find_examples + github_read_file to find a working "
        "reference implementation. Scripts based on internal knowledge often use outdated APIs.\n"
        "- You MUST have validated dataset format via hf_inspect_dataset.\n"
        "- Training config MUST include push_to_hub=True and hub_model_id. Modal job storage "
        "is EPHEMERAL — files are deleted when the container exits.\n\n"
        "BATCH/ABLATION JOBS: Submit ONE job first. Confirm it starts training successfully "
        "via logs. Only then submit the rest. Never submit all at once — if there's a bug, "
        "all jobs fail.\n\n"
        "Operations: run, ps, logs, inspect, cancel, scheduled run/ps/inspect/delete/suspend/resume.\n\n"
        f"Hardware (HF flavor naming, mapped to Modal GPU types under the hood): {ALL_FLAVORS}.\n"
        "Common picks: t4-small (1-3B), a10g-large (7-13B), a100-large (30B+), h100 (70B+).\n\n"
        "OOM RECOVERY: Reduce per_device_train_batch_size + raise gradient_accumulation_steps "
        "(keep effective batch size identical), enable gradient_checkpointing=True, or upgrade "
        "to a larger GPU. Do NOT change training method or max_length without explicit approval.\n\n"
        "Examples:\n"
        "Training: {'operation': 'run', 'script': '/app/train.py', "
        "'dependencies': ['transformers', 'trl', 'torch', 'datasets', 'trackio'], "
        "'hardware_flavor': 'a100-large', 'timeout': '8h'}\n"
        "Monitor: {'operation': 'ps'}, {'operation': 'logs', 'job_id': 'xxx'}, "
        "{'operation': 'cancel', 'job_id': 'xxx'}\n"
        "Docker: {'operation': 'run', 'command': ['duckdb', '-c', 'select 1+2'], "
        "'image': 'duckdb/duckdb', 'hardware_flavor': 'cpu-basic'}\n"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": [
                    "run",
                    "ps",
                    "logs",
                    "inspect",
                    "cancel",
                    "scheduled run",
                    "scheduled ps",
                    "scheduled inspect",
                    "scheduled delete",
                    "scheduled suspend",
                    "scheduled resume",
                ],
            },
            "script": {
                "type": "string",
                "description": (
                    "Python code, sandbox file path (e.g. /app/train.py), or URL. "
                    "Triggers Python mode. Mutually exclusive with 'command'."
                ),
            },
            "dependencies": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Pip packages to install. Only used with 'script'.",
            },
            "image": {
                "type": "string",
                "description": "Docker registry image. Auto-selected if omitted. Use with 'command'.",
            },
            "command": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Command argv. Triggers Docker mode. Mutually exclusive with 'script'.",
            },
            "hardware_flavor": {
                "type": "string",
                "description": f"Hardware tier. All options: {ALL_FLAVORS}.",
            },
            "timeout": {
                "type": "string",
                "description": (
                    "Maximum job runtime. MUST be >2h for training. "
                    "Accepts '30m', '8h', '90s', etc. Default: '30m'."
                ),
            },
            "env": {
                "type": "object",
                "description": "Environment variables. HF_TOKEN auto-included.",
            },
            "secrets": {
                "type": "object",
                "description": "Sensitive env vars; injected via modal.Secret.",
            },
            "script_args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Extra argv for the script.",
            },
            "job_id": {
                "type": "string",
                "description": "Job ID. Required for: logs, inspect, cancel.",
            },
            "scheduled_job_id": {
                "type": "string",
                "description": "Scheduled job ID. Required for scheduled operations.",
            },
            "schedule": {
                "type": "string",
                "description": "Cron schedule or preset (@hourly/@daily/...). Required for: scheduled run.",
            },
            "all": {"type": "boolean"},
            "status": {"type": "string"},
        },
        "required": ["operation"],
    },
}


async def modal_jobs_handler(
    arguments: Dict[str, Any], session: Any = None, tool_call_id: str | None = None
) -> tuple[str, bool]:
    """Handler invoked by ToolRouter when ``hf_jobs`` is called and the
    Modal compute backend is active."""
    try:

        async def log_callback(log: str):
            if session:
                await session.send_event(
                    Event(event_type="tool_log", data={"tool": "hf_jobs", "log": log})
                )

        # Resolve sandbox file paths: works for both HF and Modal sandboxes
        # because both expose the same .read() interface.
        script = arguments.get("script", "")
        sandbox = getattr(session, "sandbox", None) if session else None
        if sandbox and script:
            from agent.tools.modal_sandbox_tool import (
                ModalSandbox,
                resolve_modal_sandbox_script,
            )

            if isinstance(sandbox, ModalSandbox):
                content, error = await resolve_modal_sandbox_script(sandbox, script)
            else:
                from agent.tools.sandbox_tool import resolve_sandbox_script

                content, error = await resolve_sandbox_script(sandbox, script)
            if error:
                return error, False
            if content:
                arguments = {**arguments, "script": content}

        hf_token = session.hf_token if session else None
        tool = ModalJobsTool(
            hf_token=hf_token,
            log_callback=log_callback if session else None,
            session=session,
            tool_call_id=tool_call_id,
        )
        result = await tool.execute(arguments)
        return result["formatted"], not result.get("isError", False)
    except Exception as e:
        logger.exception("modal_jobs_handler crashed")
        return f"Error executing Modal Jobs tool: {e}", False


def get_modal_jobs_tool():
    """Return a single ToolSpec for the Modal-backed jobs runner."""
    from agent.core.tools import ToolSpec

    return ToolSpec(
        name=MODAL_JOBS_TOOL_SPEC["name"],
        description=MODAL_JOBS_TOOL_SPEC["description"],
        parameters=MODAL_JOBS_TOOL_SPEC["parameters"],
        handler=modal_jobs_handler,
    )
