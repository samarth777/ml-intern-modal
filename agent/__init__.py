"""HF Agent — main agent module.

NOTE: keep this file *empty of side-effecting imports*. It is loaded
whenever any ``agent.*`` submodule (e.g. ``agent.tools.modal_jobs_tool``)
is re-imported inside a Modal job worker. Importing ``submission_loop``
here would pull in ``litellm`` / ``fastmcp`` / heavyweight LLM deps that
the lightweight job container does not have installed.

Anything that actually wants ``submission_loop`` should
``from agent.core.agent_loop import submission_loop`` directly.
"""
