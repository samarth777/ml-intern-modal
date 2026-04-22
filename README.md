<p align="center">
  <img src="frontend/public/smolagents.webp" alt="smolagents logo" width="120" />
  &nbsp;·&nbsp;
  <img src="frontend/public/modal.svg" alt="Modal logo" width="100" />
</p>

# ML Intern · Modal

Fork of [`huggingface/ml-intern`](https://github.com/huggingface/ml-intern)
that adds [Modal](https://modal.com) as a compute backend (sandboxes +
jobs) and a `modal deploy`-able FastAPI app.

An ML intern that autonomously researches, writes, and ships good quality ML
related code using the Hugging Face ecosystem — with deep access to docs,
papers, datasets, and cloud compute. **Compute is powered by [Modal](https://modal.com)
by default** (sandboxes + jobs); the original Hugging Face Spaces backend is
still available behind a config flag.

> See [`examples/peer-review-scorer.md`](examples/peer-review-scorer.md)
> for an end-to-end run: one prompt → finds the latest HF dataset →
> drafts the training script in a Modal sandbox → fine-tunes on a Modal
> A10G job → pushes the model and Trackio dashboard to HF.

## Quick Start

### Installation

```bash
git clone https://github.com/samarth777/ml-intern-modal.git
cd ml-intern-modal
uv sync
uv tool install -e .
```

#### That's it. Now `ml-intern` works from any directory:

```bash
ml-intern
```

Create a `.env` file in the project root (or export these in your shell):

```bash
ANTHROPIC_API_KEY=<your-anthropic-api-key> # if using anthropic models
HF_TOKEN=<your-hugging-face-token>
GITHUB_TOKEN=<github-personal-access-token> 
```
If no `HF_TOKEN` is set, the CLI will prompt you to paste one on first launch. To get a GITHUB_TOKEN follow the tutorial [here](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-fine-grained-personal-access-token).

### Usage

**Interactive mode** (start a chat session):

```bash
ml-intern
```

**Headless mode** (single prompt, auto-approve):

```bash
ml-intern "fine-tune llama on my dataset"
```

**Options:**

```bash
ml-intern --model anthropic/claude-opus-4-6 "your prompt"
ml-intern --max-iterations 100 "your prompt"
ml-intern --no-stream "your prompt"
```

## Model Providers

Set `model_name` in `configs/main_agent_config.json` to switch providers.
Anything LiteLLM understands works; the provider's API key must be in `.env`.

| Provider | `model_name` example | Required env var |
| --- | --- | --- |
| Anthropic | `anthropic/claude-sonnet-4-5-20250929` | `ANTHROPIC_API_KEY` |
| OpenAI | `openai/gpt-4o` | `OPENAI_API_KEY` |
| Google Gemini | `gemini/gemini-2.0-flash` | `GEMINI_API_KEY` |
| Groq | `groq/llama-3.3-70b-versatile` | `GROQ_API_KEY` |
| OpenRouter | `openrouter/anthropic/claude-3.5-sonnet` | `OPENROUTER_API_KEY` |
| Mistral | `mistral/mistral-large-latest` | `MISTRAL_API_KEY` |
| Ollama (local) | `ollama/llama3.1` | — (set `OLLAMA_API_BASE` if remote) |
| HuggingFace Router | `huggingface/novita/moonshotai/kimi-k2.5` | `HF_TOKEN` |
| GitHub Copilot | `copilot/claude-sonnet-4` | (none — uses OAuth, see below) |

### GitHub Copilot

Use Copilot models without an Anthropic/OpenAI API key. Requires an active
Copilot subscription on your GitHub account.

```bash
ml-intern copilot login     # device-flow OAuth, opens a browser
ml-intern copilot status    # show cached login state + session expiry
ml-intern copilot logout    # delete cached credentials
```

Then point `model_name` at any Copilot-hosted model:

```json
{ "model_name": "copilot/claude-sonnet-4" }
```

Other valid IDs include `copilot/gpt-4o`, `copilot/gpt-5`,
`copilot/claude-opus-4`, `copilot/o1`. The full live list is fetched by
GitHub's models endpoint at chat time. Credentials are cached at
`~/.cache/ml-intern/copilot.json`. Set `COPILOT_OAUTH_TOKEN` in `.env` to
skip the device flow if you already have a token.

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         User/CLI                            │
└────────────┬─────────────────────────────────────┬──────────┘
             │ Operations                          │ Events
             ↓ (user_input, exec_approval,         ↑
      submission_queue  interrupt, compact, ...)  event_queue
             │                                          │
             ↓                                          │
┌────────────────────────────────────────────────────┐  │
│            submission_loop (agent_loop.py)         │  │
│  ┌──────────────────────────────────────────────┐  │  │
│  │  1. Receive Operation from queue             │  │  │
│  │  2. Route to handler (run_agent/compact/...) │  │  │
│  └──────────────────────────────────────────────┘  │  │
│                      ↓                             │  │
│  ┌──────────────────────────────────────────────┐  │  │
│  │         Handlers.run_agent()                 │  ├──┤
│  │                                              │  │  │
│  │  ┌────────────────────────────────────────┐  │  │  │
│  │  │  Agentic Loop (max 300 iterations)     │  │  │  │
│  │  │                                        │  │  │  │
│  │  │  ┌──────────────────────────────────┐  │  │  │  │
│  │  │  │ Session                          │  │  │  │  │
│  │  │  │  ┌────────────────────────────┐  │  │  │  │  │
│  │  │  │  │ ContextManager             │  │  │  │  │  │
│  │  │  │  │ • Message history          │  │  │  │  │  │
│  │  │  │  │   (litellm.Message[])      │  │  │  │  │  │
│  │  │  │  │ • Auto-compaction (170k)   │  │  │  │  │  │
│  │  │  │  │ • Session upload to HF     │  │  │  │  │  │
│  │  │  │  └────────────────────────────┘  │  │  │  │  │
│  │  │  │                                  │  │  │  │  │
│  │  │  │  ┌────────────────────────────┐  │  │  │  │  │
│  │  │  │  │ ToolRouter                 │  │  │  │  │  │
│  │  │  │  │  ├─ HF docs & research     │  │  │  │  │  │
│  │  │  │  │  ├─ HF repos, datasets,    │  │  │  │  │  │
│  │  │  │  │  │  jobs, papers           │  │  │  │  │  │
│  │  │  │  │  ├─ GitHub code search     │  │  │  │  │  │
│  │  │  │  │  ├─ Sandbox & local tools  │  │  │  │  │  │
│  │  │  │  │  ├─ Planning               │  │  │  │  │  │
│  │  │  │  │  └─ MCP server tools       │  │  │  │  │  │
│  │  │  │  └────────────────────────────┘  │  │  │  │  │
│  │  │  └──────────────────────────────────┘  │  │  │  │
│  │  │                                        │  │  │  │
│  │  │  ┌──────────────────────────────────┐  │  │  │  │
│  │  │  │ Doom Loop Detector               │  │  │  │  │
│  │  │  │ • Detects repeated tool patterns │  │  │  │  │
│  │  │  │ • Injects corrective prompts     │  │  │  │  │
│  │  │  └──────────────────────────────────┘  │  │  │  │
│  │  │                                        │  │  │  │
│  │  │  Loop:                                 │  │  │  │
│  │  │    1. LLM call (litellm.acompletion)   │  │  │  │
│  │  │       ↓                                │  │  │  │
│  │  │    2. Parse tool_calls[]               │  │  │  │
│  │  │       ↓                                │  │  │  │
│  │  │    3. Approval check                   │  │  │  │
│  │  │       (jobs, sandbox, destructive ops) │  │  │  │
│  │  │       ↓                                │  │  │  │
│  │  │    4. Execute via ToolRouter           │  │  │  │
│  │  │       ↓                                │  │  │  │
│  │  │    5. Add results to ContextManager    │  │  │  │
│  │  │       ↓                                │  │  │  │
│  │  │    6. Repeat if tool_calls exist       │  │  │  │
│  │  └────────────────────────────────────────┘  │  │  │
│  └──────────────────────────────────────────────┘  │  │
└────────────────────────────────────────────────────┴──┘
```

### Agentic Loop Flow

```
User Message
     ↓
[Add to ContextManager]
     ↓
     ╔═══════════════════════════════════════════╗
     ║      Iteration Loop (max 300)             ║
     ║                                           ║
     ║  Get messages + tool specs                ║
     ║         ↓                                 ║
     ║  litellm.acompletion()                    ║
     ║         ↓                                 ║
     ║  Has tool_calls? ──No──> Done             ║
     ║         │                                 ║
     ║        Yes                                ║
     ║         ↓                                 ║
     ║  Add assistant msg (with tool_calls)      ║
     ║         ↓                                 ║
     ║  Doom loop check                          ║
     ║         ↓                                 ║
     ║  For each tool_call:                      ║
     ║    • Needs approval? ──Yes──> Wait for    ║
     ║    │                         user confirm ║
     ║    No                                     ║
     ║    ↓                                      ║
     ║    • ToolRouter.execute_tool()            ║
     ║    • Add result to ContextManager         ║
     ║         ↓                                 ║
     ║  Continue loop ─────────────────┐         ║
     ║         ↑                       │         ║
     ║         └───────────────────────┘         ║
     ╚═══════════════════════════════════════════╝
```

## Events

The agent emits the following events via `event_queue`:

- `processing` - Starting to process user input
- `ready` - Agent is ready for input
- `assistant_chunk` - Streaming token chunk
- `assistant_message` - Complete LLM response text
- `assistant_stream_end` - Token stream finished
- `tool_call` - Tool being called with arguments
- `tool_output` - Tool execution result
- `tool_log` - Informational tool log message
- `tool_state_change` - Tool execution state transition
- `approval_required` - Requesting user approval for sensitive operations
- `turn_complete` - Agent finished processing
- `error` - Error occurred during processing
- `interrupted` - Agent was interrupted
- `compacted` - Context was compacted
- `undo_complete` - Undo operation completed
- `shutdown` - Agent shutting down

## Development

### Adding Built-in Tools

Edit `agent/core/tools.py`:

```python
def create_builtin_tools() -> list[ToolSpec]:
    return [
        ToolSpec(
            name="your_tool",
            description="What your tool does",
            parameters={
                "type": "object",
                "properties": {
                    "param": {"type": "string", "description": "Parameter description"}
                },
                "required": ["param"]
            },
            handler=your_async_handler
        ),
        # ... existing tools
    ]
```

### Adding MCP Servers

Edit `configs/main_agent_config.json`:

```json
{
  "model_name": "anthropic/claude-sonnet-4-5-20250929",
  "mcpServers": {
    "your-server-name": {
      "transport": "http",
      "url": "https://example.com/mcp",
      "headers": {
        "Authorization": "Bearer ${YOUR_TOKEN}"
      }
    }
  }
}
```

Note: Environment variables like `${YOUR_TOKEN}` are auto-substituted from `.env`.

## Deploying on Modal

ml-intern can run with [Modal](https://modal.com) as both the deployment
target *and* the compute backend (sandboxes + jobs), as an alternative to
HF Spaces. The Modal backend is opt-in via a single config flag.

### 1. Install + auth

```bash
pip install modal
modal setup
```

### 2. Create the secret bundle

All runtime secrets are read from a single Modal secret named
`ml-intern-secrets`:

```bash
modal secret create ml-intern-secrets \
    ANTHROPIC_API_KEY=sk-ant-... \
    HF_TOKEN=hf_... \
    GITHUB_TOKEN=ghp_... \
    OAUTH_CLIENT_ID=... \
    OAUTH_CLIENT_SECRET=... \
    OPENID_PROVIDER_URL=https://huggingface.co \
    HF_OAUTH_ORG_ID=... \
```

### 3. Build the frontend

```bash
cd frontend && npm install && npm run build
rm -rf ../static && cp -r dist ../static && cd ..
```

### 4. Compute backend (Modal is the default)

`configs/main_agent_config.json` ships with `"compute_backend": "modal"`,
so agent sandboxes and `hf_jobs` calls run on Modal out of the box. To fall
back to the legacy Hugging Face Spaces backend, set:

```json
{
  "compute_backend": "hf"
}
```

### 5. Deploy

```bash
modal deploy modal_app.py
```

Modal prints a public URL — register it as the OAuth callback in your HF
OAuth app (`<url>/auth/callback`) and you're live.

The deployment is pinned to `min_containers=1, max_containers=1` to
preserve the in-process session manager's semantics. Horizontal scaling
would require moving session state to a shared store first.

## Acknowledgments

Built on top of [`huggingface/ml-intern`](https://github.com/huggingface/ml-intern)
by Hugging Face — all credit for the agent, tools, prompts, and UI goes
to them. This fork only adds the Modal compute backend
(`agent/tools/modal_{sandbox,jobs}_tool.py`, `modal_app.py`) and a
GitHub Copilot provider.
