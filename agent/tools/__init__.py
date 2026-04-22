"""Agent tools package.

Keep this file *empty of side-effecting imports*. It is loaded whenever
any ``agent.tools.*`` submodule is re-imported inside a Modal job worker
(via ``agent.tools.modal_jobs_tool`` deserialization), and the worker
container only ships the user's pip deps. Eager re-exports here would
pull in ``thefuzz``, ``huggingface_hub``, ``fastmcp``, etc., none of
which the lightweight job image installs.

Importers should use the submodule path directly:
``from agent.tools.jobs_tool import HF_JOBS_TOOL_SPEC``.
"""
