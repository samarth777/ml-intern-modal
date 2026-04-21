"""
Modal Sandbox client + agent tools.

Provides the same agent-facing tool surface as ``agent/tools/sandbox_tool.py``
(``sandbox_create``, ``bash``, ``read``, ``write``, ``edit``) but backed by
``modal.Sandbox`` instead of an HF Space duplicate.

Why a parallel implementation?
    The HF Space sandbox depends on duplicating a template Space and waiting
    for it to come online (1-3 min cold-start, billed by Space hardware tier).
    Modal Sandboxes start in seconds, support GPU per-call, attach
    ``modal.Volume``s, and bill per-second. Swapping the implementation rather
    than rewriting the agent's prompts/schemas keeps tool descriptions stable
    so the LLM does not need re-prompting.

Hardware mapping (HF flavor → Modal gpu= string)::

    cpu-basic, cpu-upgrade           -> None  (CPU-only sandbox)
    t4-small, t4-medium              -> "T4"
    a10g-small, a10g-large           -> "A10G"
    a10g-largex2                     -> "A10G:2"
    a10g-largex4                     -> "A10G:4"
    a100-large                       -> "A100-80GB"
    a100x4                           -> "A100-80GB:4"
    a100x8                           -> "A100-80GB:8"
    l4x1, l4x4                       -> "L4" / "L4:4"
    l40sx1, l40sx4, l40sx8           -> "L40S" / "L40S:4" / "L40S:8"
    h100, h100x8                     -> "H100" / "H100:8"

The agent picks hardware via the existing HF flavor enum so prompts do not
need to change.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

from agent.core.session import Event
from agent.tools.types import ToolResult as ToolResultDict  # noqa: F401  (re-exported for parity)

logger = logging.getLogger(__name__)


# Cap output sizes the same way the HF sandbox does for predictable LLM context use.
OUTPUT_LIMIT = 25_000
DEFAULT_READ_LIMIT = 2_000
LINE_LIMIT = 4_000
DEFAULT_BASH_TIMEOUT = 240
MAX_BASH_TIMEOUT = 1_200
DEFAULT_SANDBOX_TIMEOUT = 60 * 60 * 4  # 4h — sandbox auto-stops after this if idle.

# HF flavor → Modal GPU spec. None means CPU-only.
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
}

# Soft CPU/RAM presets for CPU tiers, matching HF tier vCPU/RAM.
_HW_TO_CPU_MEM: dict[str, tuple[float, int]] = {
    "cpu-basic": (2.0, 16 * 1024),
    "cpu-upgrade": (8.0, 32 * 1024),
}

# Modal app/image are lazily constructed once per process and reused across sandboxes.
_MODAL_APP = None
_MODAL_IMAGE = None
_MODAL_LOCK = threading.Lock()


def _get_modal_app():
    """Lazily build the shared Modal App + base Image used for all sandboxes."""
    global _MODAL_APP, _MODAL_IMAGE
    with _MODAL_LOCK:
        if _MODAL_APP is not None:
            return _MODAL_APP, _MODAL_IMAGE

        import modal

        _MODAL_IMAGE = (
            modal.Image.debian_slim(python_version="3.12")
            .apt_install(
                "git",
                "git-lfs",
                "curl",
                "wget",
                "build-essential",
                "vim",
                "nano",
                "jq",
                "htop",
                "procps",
                "tmux",
            )
            # uv is the preferred installer in HF jobs as well; keep parity.
            .pip_install("uv", "huggingface_hub", "hf-transfer")
            .env(
                {
                    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
                    "TQDM_DISABLE": "1",
                    "HF_HUB_ENABLE_HF_TRANSFER": "1",
                    "UV_NO_PROGRESS": "1",
                    "PYTHONWARNINGS": "ignore::DeprecationWarning",
                }
            )
            .workdir("/app")
        )
        # Lookup-or-create the persistent Modal App holding all sandboxes.
        # ``Sandbox.create`` requires an App handle; using a stable name lets
        # multiple backend processes share telemetry under one App in the
        # Modal dashboard.
        _MODAL_APP = modal.App.lookup("ml-intern-sandbox", create_if_missing=True)
        return _MODAL_APP, _MODAL_IMAGE


@dataclass
class ToolResult:
    success: bool
    output: str = ""
    error: str = ""

    def __str__(self) -> str:
        return self.output or "(no output)" if self.success else f"ERROR: {self.error}"


def _truncate_output(
    output: str, max_chars: int = OUTPUT_LIMIT, head_ratio: float = 0.25
) -> str:
    if len(output) <= max_chars:
        return output
    head_budget = int(max_chars * head_ratio)
    tail_budget = max_chars - head_budget
    head = output[:head_budget]
    tail = output[-tail_budget:]
    omitted = len(output) - max_chars
    meta = (
        f"\n\n... ({omitted:,} of {len(output):,} chars omitted, "
        f"showing first {head_budget:,} + last {tail_budget:,}) ...\n"
    )
    return head + meta + tail


@dataclass
class ModalSandbox:
    """Handle to a running ``modal.Sandbox`` exposing the same surface as
    ``agent.tools.sandbox_client.Sandbox``.

    The agent treats this as duck-typed against the HF ``Sandbox`` — both
    expose ``space_id``, ``url``, ``bash``, ``read``, ``write``, ``edit``,
    ``call_tool``, ``kill_all``, and ``delete``.
    """

    sandbox_id: str  # the Modal-assigned sandbox ID
    hardware: str = "cpu-basic"
    work_dir: str = "/app"
    _modal_sandbox: Any = field(default=None, repr=False)
    _owns_space: bool = field(default=True, repr=False)
    _files_read: set = field(default_factory=set, repr=False)

    # ── Compatibility shims ─────────────────────────────────────────────

    @property
    def space_id(self) -> str:
        """Compat alias for HF code paths that reference ``sandbox.space_id``."""
        return self.sandbox_id

    @property
    def url(self) -> str:
        """Best-effort link to the Modal sandbox in the dashboard."""
        ws = os.environ.get("MODAL_WORKSPACE", "")
        env = os.environ.get("MODAL_ENVIRONMENT", "main")
        if ws:
            return f"https://modal.com/apps/{ws}/{env}/ap-/sandboxes/{self.sandbox_id}"
        return f"modal://sandbox/{self.sandbox_id}"

    @property
    def status(self) -> str:
        try:
            return "RUNNING" if self._modal_sandbox.poll() is None else "STOPPED"
        except Exception:
            return "UNKNOWN"

    # ── Lifecycle ───────────────────────────────────────────────────────

    class Cancelled(Exception):
        """Raised when sandbox creation is cancelled by the user."""

    @classmethod
    def create(
        cls,
        *,
        hardware: str = "cpu-basic",
        timeout: int = DEFAULT_SANDBOX_TIMEOUT,
        secrets: dict[str, str] | None = None,
        log: Any = None,
        cancel_event: Any = None,
        volumes: dict[str, Any] | None = None,
    ) -> "ModalSandbox":
        """Spin up a fresh ``modal.Sandbox``.

        Returns immediately once the sandbox container is created — Modal
        attaches the requested GPU at boot, so the sandbox is ready to
        accept ``exec`` calls right away (no rebuild step needed unlike HF).
        """
        import modal

        _log = log or (lambda msg: None)
        if cancel_event and cancel_event.is_set():
            raise cls.Cancelled("Sandbox creation cancelled before start")

        app, image = _get_modal_app()

        gpu = _HW_TO_MODAL_GPU.get(hardware, None)
        if hardware not in _HW_TO_MODAL_GPU:
            _log(
                f"Unknown hardware {hardware!r}, defaulting to CPU-only. "
                f"Valid: {sorted(_HW_TO_MODAL_GPU)}"
            )

        cpu_mem = _HW_TO_CPU_MEM.get(hardware)
        sb_kwargs: dict[str, Any] = {
            "image": image,
            "app": app,
            "timeout": timeout,
            "workdir": "/app",
        }
        if gpu:
            sb_kwargs["gpu"] = gpu
        if cpu_mem:
            sb_kwargs["cpu"] = cpu_mem[0]
            sb_kwargs["memory"] = cpu_mem[1]
        if volumes:
            sb_kwargs["volumes"] = volumes
        if secrets:
            # Modal expects a list of modal.Secret. Wrap the supplied dict in
            # a one-shot in-memory secret so values like HF_TOKEN are visible
            # to processes inside the sandbox without being persisted.
            sb_kwargs["secrets"] = [modal.Secret.from_dict(secrets)]

        _log(f"Creating Modal sandbox (hardware={hardware}, gpu={gpu or 'none'})...")
        modal_sb = modal.Sandbox.create(**sb_kwargs)

        if cancel_event and cancel_event.is_set():
            try:
                modal_sb.terminate()
            except Exception:
                pass
            raise cls.Cancelled("Sandbox creation cancelled during start")

        sb_id = getattr(modal_sb, "object_id", None) or f"sb-{uuid.uuid4().hex[:8]}"
        _log(f"Sandbox ready: {sb_id}")
        return cls(
            sandbox_id=sb_id,
            hardware=hardware,
            _modal_sandbox=modal_sb,
            _owns_space=True,
        )

    def delete(self) -> None:
        """Terminate the underlying ``modal.Sandbox``. Idempotent."""
        sb = self._modal_sandbox
        if sb is None:
            return
        try:
            sb.terminate()
        finally:
            self._modal_sandbox = None

    # ── Operations ──────────────────────────────────────────────────────

    def _exec(
        self,
        *cmd: str,
        timeout: int = DEFAULT_BASH_TIMEOUT,
        work_dir: str | None = None,
    ) -> ToolResult:
        """Run a command in the sandbox and capture stdout+stderr."""
        sb = self._modal_sandbox
        if sb is None:
            return ToolResult(success=False, error="Sandbox is no longer running.")
        try:
            kwargs: dict[str, Any] = {"timeout": min(timeout, MAX_BASH_TIMEOUT)}
            if work_dir:
                kwargs["workdir"] = work_dir
            proc = sb.exec(*cmd, **kwargs)
            stdout = proc.stdout.read() if proc.stdout else ""
            stderr = proc.stderr.read() if proc.stderr else ""
            returncode = proc.wait()
            output = _truncate_output((stdout or "") + (stderr or ""))
            if returncode == 0:
                return ToolResult(success=True, output=output)
            return ToolResult(
                success=False,
                output=output,
                error=f"Exit code {returncode}",
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def bash(
        self,
        command: str,
        *,
        work_dir: str | None = None,
        timeout: int | None = None,
        description: str | None = None,
    ) -> ToolResult:
        return self._exec(
            "/bin/bash",
            "-lc",
            command,
            timeout=timeout or DEFAULT_BASH_TIMEOUT,
            work_dir=work_dir or self.work_dir,
        )

    def read(
        self, path: str, *, offset: int | None = None, limit: int | None = None
    ) -> ToolResult:
        """Read a file from the sandbox using ``modal.Sandbox.open``.

        Output is line-numbered (``N\\tcontent``) to match the HF sandbox
        format the rest of the agent expects.
        """
        sb = self._modal_sandbox
        if sb is None:
            return ToolResult(success=False, error="Sandbox is no longer running.")
        self._files_read.add(path)
        try:
            with sb.open(path, "r") as f:
                text = f.read()
        except FileNotFoundError:
            return ToolResult(success=False, error=f"File not found: {path}")
        except IsADirectoryError:
            return ToolResult(success=False, error=f"Is a directory: {path}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))

        lines = text.splitlines()
        start = (offset or 1) - 1
        end = start + (limit or DEFAULT_READ_LIMIT)
        selected = lines[start:end]
        # Truncate over-long lines
        truncated = [
            line if len(line) <= LINE_LIMIT else line[:LINE_LIMIT] + " …(truncated)"
            for line in selected
        ]
        numbered = "\n".join(f"{start + i + 1}\t{ln}" for i, ln in enumerate(truncated))
        return ToolResult(success=True, output=numbered)

    def _exists(self, path: str) -> bool:
        result = self._exec(
            "/bin/sh",
            "-lc",
            f"test -e {path!r} && echo true || echo false",
            timeout=10,
        )
        return result.success and "true" in (result.output or "")

    def write(self, path: str, content: str) -> ToolResult:
        sb = self._modal_sandbox
        if sb is None:
            return ToolResult(success=False, error="Sandbox is no longer running.")
        if path not in self._files_read and self._exists(path):
            return ToolResult(
                success=False,
                error=(
                    f"File {path} exists but has not been read this session. "
                    f"Read it first, or use sandbox edit for targeted changes."
                ),
            )
        # Make sure the parent directory exists.
        parent = os.path.dirname(path) or "/"
        self._exec("/bin/sh", "-lc", f"mkdir -p {parent!r}", timeout=10)
        try:
            with sb.open(path, "w") as f:
                f.write(content)
        except Exception as e:
            return ToolResult(success=False, error=str(e))
        self._files_read.add(path)
        return ToolResult(success=True, output=f"Wrote {len(content)} bytes to {path}")

    def edit(
        self,
        path: str,
        old_str: str,
        new_str: str,
        *,
        replace_all: bool = False,
        mode: str = "replace",
    ) -> ToolResult:
        if old_str == new_str:
            return ToolResult(success=False, error="old_str and new_str are identical.")
        if path not in self._files_read:
            return ToolResult(
                success=False,
                error=f"File {path} has not been read this session. Read it first.",
            )
        sb = self._modal_sandbox
        if sb is None:
            return ToolResult(success=False, error="Sandbox is no longer running.")
        try:
            with sb.open(path, "r") as f:
                content = f.read()
        except FileNotFoundError:
            return ToolResult(success=False, error=f"File not found: {path}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))

        if old_str not in content:
            return ToolResult(success=False, error="old_str not found in file.")

        count = content.count(old_str)
        if mode == "replace_all":
            replace_all = True
            mode = "replace"

        if mode == "replace":
            if count > 1 and not replace_all:
                return ToolResult(
                    success=False,
                    error=(
                        f"old_str appears {count} times. Use replace_all=true "
                        f"or provide more context."
                    ),
                )
            new_content = (
                content.replace(old_str, new_str)
                if replace_all
                else content.replace(old_str, new_str, 1)
            )
            replacements = count if replace_all else 1
        elif mode == "append_after":
            idx = content.index(old_str) + len(old_str)
            new_content = content[:idx] + new_str + content[idx:]
            replacements = 1
        elif mode == "prepend_before":
            idx = content.index(old_str)
            new_content = content[:idx] + new_str + content[idx:]
            replacements = 1
        else:
            return ToolResult(success=False, error=f"Unknown mode: {mode}")

        try:
            with sb.open(path, "w") as f:
                f.write(new_content)
        except Exception as e:
            return ToolResult(success=False, error=str(e))
        return ToolResult(
            success=True,
            output=f"Edited {path} ({replacements} replacement{'s' if replacements != 1 else ''})",
        )

    def kill_all(self) -> ToolResult:
        """Kill all processes in the sandbox by terminating + recreating it.

        Modal does not expose process-level kill across multiple ``exec``
        invocations, so we approximate the HF behavior by terminating the
        sandbox. The session manager will recreate one on the next call.
        """
        try:
            self.delete()
            return ToolResult(success=True, output="Sandbox terminated.")
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    # ── Tool dispatch (mirrors Sandbox.TOOLS / call_tool) ──────────────

    TOOLS = {
        "bash": {
            "description": (
                "Run a shell command in the remote Modal sandbox and return stdout/stderr.\n"
                "\n"
                "IMPORTANT: Do NOT use bash for file operations — use the dedicated tools instead:\n"
                "- To read files: use read (not cat/head/tail)\n"
                "- To edit files: use edit (not sed/awk)\n"
                "- To write files: use write (not echo/cat <<EOF)\n"
                "\n"
                "Commands run in a shell at /app. Each invocation is independent — use files in "
                "/app to persist state. pip / uv install work out of the box.\n"
                "Chain dependent commands with &&. Independent commands should be separate "
                "bash calls (they can run in parallel).\n"
                "\n"
                f"Timeout default {DEFAULT_BASH_TIMEOUT}s, max {MAX_BASH_TIMEOUT}s."
            ),
            "parameters": {
                "type": "object",
                "required": ["command"],
                "additionalProperties": False,
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Short description (5-10 words, active voice).",
                    },
                    "work_dir": {
                        "type": "string",
                        "description": "Working directory (default: /app).",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": f"Optional timeout in seconds (default: {DEFAULT_BASH_TIMEOUT}, max: {MAX_BASH_TIMEOUT}).",
                    },
                },
            },
        },
        "read": {
            "description": (
                "Read a file from the Modal sandbox filesystem. Returns contents with line "
                "numbers (cat -n format).\n"
                "\n"
                "Usage:\n"
                "- By default, reads up to 2000 lines from the beginning of the file.\n"
                "- Optionally specify offset and limit for large files.\n"
                "- Lines longer than 4000 chars are truncated.\n"
                "- IMPORTANT: Always read a file before editing or overwriting it. The edit and "
                "write tools will reject operations on files you haven't read."
            ),
            "parameters": {
                "type": "object",
                "required": ["path"],
                "additionalProperties": False,
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file to read.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "1-based starting line number.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of lines to read.",
                    },
                },
            },
        },
        "write": {
            "description": (
                "Write a file to the Modal sandbox filesystem. Overwrites existing files.\n"
                "\n"
                "- If the file already exists, you MUST use the read tool first.\n"
                "- ALWAYS prefer editing existing files with the edit tool over overwriting "
                "with write.\n"
                "- Creates parent directories as needed."
            ),
            "parameters": {
                "type": "object",
                "required": ["path", "content"],
                "additionalProperties": False,
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "The complete file content.",
                    },
                },
            },
        },
        "edit": {
            "description": (
                "Performs string replacements in a file in the Modal sandbox.\n"
                "\n"
                "Usage:\n"
                "- You must read the file at least once before editing.\n"
                "- The edit will FAIL if old_str is not unique in the file. Either provide "
                "a larger string with more surrounding context to make it unique, or set "
                "replace_all to true.\n"
                "- old_str and new_str must differ.\n"
                "- Preserve indentation exactly as it appears in the file.\n"
                "- Do NOT include line number prefixes from read output in old_str or new_str.\n"
                "\n"
                "Modes:\n"
                "- replace (default): replace first occurrence of old_str with new_str.\n"
                "- append_after: insert new_str immediately after old_str.\n"
                "- prepend_before: insert new_str immediately before old_str."
            ),
            "parameters": {
                "type": "object",
                "required": ["path", "old_str", "new_str"],
                "additionalProperties": False,
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file to edit.",
                    },
                    "old_str": {"type": "string", "description": "The text to find."},
                    "new_str": {
                        "type": "string",
                        "description": "The replacement text.",
                    },
                    "replace_all": {"type": "boolean", "default": False},
                    "mode": {
                        "type": "string",
                        "enum": ["replace", "append_after", "prepend_before"],
                        "default": "replace",
                    },
                },
            },
        },
    }

    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        dispatch = {
            "bash": lambda a: self.bash(
                a["command"],
                work_dir=a.get("work_dir"),
                timeout=a.get("timeout"),
                description=a.get("description"),
            ),
            "read": lambda a: self.read(
                a["path"], offset=a.get("offset"), limit=a.get("limit")
            ),
            "write": lambda a: self.write(a["path"], a["content"]),
            "edit": lambda a: self.edit(
                a["path"],
                a["old_str"],
                a["new_str"],
                replace_all=a.get("replace_all", False),
                mode=a.get("mode", "replace"),
            ),
        }
        fn = dispatch.get(name)
        if not fn:
            return ToolResult(success=False, error=f"Unknown tool: {name}")
        return fn(arguments)


# ── Tool handlers (parallel of agent/tools/sandbox_tool.py) ───────────


SANDBOX_CREATE_TOOL_SPEC = {
    "name": "sandbox_create",
    "description": (
        "Create a persistent remote Linux environment for developing and testing scripts.\n\n"
        "Workflow: sandbox_create → write script → pip install → test with small run → "
        "fix errors → hf_jobs at scale.\n"
        "The sandbox persists across tool calls within the session. pip install works out of the box.\n\n"
        "Use this when: you need to develop, test, and iterate on scripts before launching at scale. "
        "Especially for training scripts where you need to verify imports, test on a small subset, "
        "and fix errors interactively.\n\n"
        "Skip this when: the task is a simple one-shot operation (status check, resource search, "
        "quick data query), or the script is copied from a verified working example with minimal changes.\n\n"
        "For ML code that uses CUDA, bf16, or model loading: use GPU hardware (t4-small minimum). "
        "CPU sandboxes cannot run GPU code paths — your test will not catch GPU-related errors.\n\n"
        "Hardware (mapped to Modal GPU types under the hood): "
        + ", ".join(sorted(_HW_TO_MODAL_GPU.keys()))
        + ".\n"
    ),
    "parameters": {
        "type": "object",
        "required": [],
        "additionalProperties": False,
        "properties": {
            "hardware": {
                "type": "string",
                "enum": sorted(_HW_TO_MODAL_GPU.keys()),
                "description": "Hardware tier for the sandbox (default: cpu-basic).",
            },
        },
    },
}


async def _ensure_modal_sandbox(
    session: Any, hardware: str = "cpu-basic"
) -> tuple[ModalSandbox | None, str | None]:
    """Auto-create a Modal sandbox on the session if one isn't already attached."""
    if session and getattr(session, "sandbox", None):
        return session.sandbox, None
    if not session:
        return None, "No session available."

    await session.send_event(
        Event(
            event_type="tool_log",
            data={
                "tool": "sandbox",
                "log": f"Auto-creating Modal sandbox ({hardware})...",
            },
        )
    )

    loop = asyncio.get_running_loop()

    def _log(msg: str) -> None:
        loop.call_soon_threadsafe(
            session.event_queue.put_nowait,
            Event(event_type="tool_log", data={"tool": "sandbox", "log": msg}),
        )

    cancel_flag = threading.Event()

    async def _watch_cancel():
        await session._cancelled.wait()
        cancel_flag.set()

    watcher_task = asyncio.create_task(_watch_cancel())

    secrets: dict[str, str] = {}
    if session.hf_token:
        secrets["HF_TOKEN"] = session.hf_token
        secrets["HUGGINGFACE_HUB_TOKEN"] = session.hf_token

    try:
        sb = await asyncio.to_thread(
            ModalSandbox.create,
            hardware=hardware,
            secrets=secrets or None,
            log=_log,
            cancel_event=cancel_flag,
        )
    except ModalSandbox.Cancelled:
        return None, "Sandbox creation cancelled by user."
    except Exception as e:
        return None, f"Failed to create Modal sandbox: {e}"
    finally:
        watcher_task.cancel()

    session.sandbox = sb
    await session.send_event(
        Event(
            event_type="tool_log",
            data={"tool": "sandbox", "log": f"Modal sandbox ready: {sb.sandbox_id}"},
        )
    )
    return sb, None


async def sandbox_create_handler(
    args: dict[str, Any], session: Any = None
) -> tuple[str, bool]:
    """Handle the explicit ``sandbox_create`` tool call (requires user approval)."""
    if session and getattr(session, "sandbox", None):
        sb = session.sandbox
        return (
            f"Modal sandbox already active: {sb.sandbox_id}\n"
            f"Hardware: {getattr(sb, 'hardware', 'unknown')}\n"
            f"Use bash/read/write/edit to interact with it."
        ), True

    hardware = args.get("hardware", "cpu-basic")
    sb, err = await _ensure_modal_sandbox(session, hardware=hardware)
    if err:
        return err, False
    return (
        f"Modal sandbox created: {sb.sandbox_id}\n"
        f"Hardware: {hardware}\n"
        f"URL: {sb.url}\n"
        f"Use bash/read/write/edit to interact with it."
    ), True


def _make_modal_op_handler(op_name: str):
    async def handler(args: dict[str, Any], session: Any = None) -> tuple[str, bool]:
        if not session or not getattr(session, "sandbox", None):
            return "No sandbox running. Call sandbox_create first to start one.", False
        sb = session.sandbox
        try:
            result = await asyncio.to_thread(sb.call_tool, op_name, args)
            if result.success:
                return result.output or "(no output)", True
            if result.output:
                return f"{result.output}\n\nERROR: {result.error}", False
            return f"ERROR: {result.error}", False
        except Exception as e:
            return f"Sandbox operation failed: {e}", False

    return handler


def get_modal_sandbox_tools():
    """Return the 5 ToolSpecs for the Modal-backed sandbox (matches HF surface)."""
    from agent.core.tools import ToolSpec

    tools = [
        ToolSpec(
            name=SANDBOX_CREATE_TOOL_SPEC["name"],
            description=SANDBOX_CREATE_TOOL_SPEC["description"],
            parameters=SANDBOX_CREATE_TOOL_SPEC["parameters"],
            handler=sandbox_create_handler,
        )
    ]
    for name, spec in ModalSandbox.TOOLS.items():
        tools.append(
            ToolSpec(
                name=name,
                description=spec["description"],
                parameters=spec["parameters"],
                handler=_make_modal_op_handler(name),
            )
        )
    return tools


async def resolve_modal_sandbox_script(
    sandbox: Any, script: str
) -> tuple[str | None, str | None]:
    """Read a file from a Modal sandbox if *script* looks like a path.

    Mirrors ``agent.tools.sandbox_tool.resolve_sandbox_script`` so the jobs
    tool can resolve sandbox paths regardless of which backend created the
    sandbox.
    """
    if not sandbox or not isinstance(script, str):
        return None, None
    s = script.strip()
    if any(c in s for c in "\r\n\0"):
        return None, None
    if not (s.startswith("/") or s.startswith("./") or s.startswith("../")):
        return None, None
    try:
        result = await asyncio.to_thread(sandbox.read, s, limit=100_000)
        if result.success and result.output:
            lines = []
            for line in result.output.split("\n"):
                parts = line.split("\t", 1)
                lines.append(parts[1] if len(parts) == 2 else line)
            return "\n".join(lines), None
        return None, f"Failed to read {s} from sandbox: {result.error}"
    except Exception as e:
        return None, f"Failed to read {s} from sandbox: {e}"
