"""External LLM provider integrations beyond what LiteLLM ships natively.

Currently only :mod:`agent.providers.copilot` (GitHub Copilot device-flow OAuth
+ session-token exchange) lives here. Standard providers (OpenAI, Anthropic,
Gemini, Groq, Ollama, OpenRouter, …) work directly through LiteLLM by setting
their respective env vars and using the canonical ``provider/model`` string.
"""
