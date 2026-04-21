"""
Compute backend registry.

Returns the set of "compute" tools (sandbox + jobs) for a given backend:
  - "hf"    : Hugging Face Spaces sandbox + HF Jobs (default; legacy behavior)
  - "modal" : Modal Sandboxes + Modal Functions

The agent-facing tool *names* are kept stable across backends (sandbox_create,
bash, read, write, edit, hf_jobs) so no agent-side prompt or schema changes
are needed when switching backends.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from agent.core.tools import ToolSpec

VALID_BACKENDS = ("hf", "modal")


def get_compute_tools(backend: str, local_mode: bool = False) -> "list[ToolSpec]":
    """Return the sandbox + jobs ToolSpecs for the requested backend.

    Args:
        backend: "hf" or "modal".
        local_mode: If True, override and return local subprocess-based tools
            (sandbox + jobs are skipped — used by the CLI when --local).

    Notes:
        local_mode takes precedence over backend selection — running on the
        user's own machine never uses a remote sandbox or jobs runner.
    """
    if local_mode:
        from agent.tools.local_tools import get_local_tools

        return get_local_tools()

    if backend not in VALID_BACKENDS:
        logger.warning(
            "Unknown compute_backend %r, falling back to 'hf'. Valid: %s",
            backend,
            VALID_BACKENDS,
        )
        backend = "hf"

    if backend == "modal":
        from agent.tools.modal_jobs_tool import get_modal_jobs_tool
        from agent.tools.modal_sandbox_tool import get_modal_sandbox_tools

        tools = get_modal_sandbox_tools()
        tools.append(get_modal_jobs_tool())
        logger.info("Using Modal compute backend (modal.Sandbox + modal.Function)")
        return tools

    # Default: HF backend (existing behavior)
    from agent.core.tools import ToolSpec
    from agent.tools.jobs_tool import HF_JOBS_TOOL_SPEC, hf_jobs_handler
    from agent.tools.sandbox_tool import get_sandbox_tools

    tools = get_sandbox_tools()
    tools.append(
        ToolSpec(
            name=HF_JOBS_TOOL_SPEC["name"],
            description=HF_JOBS_TOOL_SPEC["description"],
            parameters=HF_JOBS_TOOL_SPEC["parameters"],
            handler=hf_jobs_handler,
        )
    )
    logger.info("Using HF compute backend (HF Spaces + HF Jobs)")
    return tools
