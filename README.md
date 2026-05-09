# Prism 🔺

**LLM protocol bridge. Any provider, any coding agent, zero config.**

Point Prism between your AI coding tool and any LLM API.
It auto-detects the wire format on both ends, translates transparently,
and preserves message content and tool calls exactly as-is.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What it does

Every LLM provider speaks a slightly different wire format.
Same concepts — messages, tool calls, stop reasons — different JSON shapes.

```
Claude Code (Anthropic format)          Any Provider (OpenAI-compat)
──────────────────────────────          ────────────────────────────
content[{type: text}]          ←──────→  choices[0].message.content
stop_reason: "tool_use"        ←──────→  finish_reason: "tool_calls"
content[{type: tool_use}]      ←──────→  choices[0].message.tool_calls
usage.input_tokens             ←──────→  usage.prompt_tokens
system: "..."                  ←──────→  messages[{role: system}]
```

Prism sits in the middle and handles all of it.
Your coding agent thinks it's talking to Anthropic.
The provider thinks it's getting normal requests.
Neither side needs to know the other exists.

---

## Install

```bash
pip install prism-proxy
```

Or from source:

```bash
git clone https://github.com/Alpha-Legents/Prism
cd Prism
pip install -e .
```

---

## Quick start

### Single model

```bash
prism --provider https://api.mistral.ai/v1 \
      --key sk-... \
      --model mistral-small-latest
```

Then point your coding agent at prism:

```bash
ANTHROPIC_BASE_URL=http://localhost:8000
ANTHROPIC_API_KEY=prism
```

### Model mapping

Map each frontend model endpoint to a specific provider model.
Claude Code uses different models for different tasks (opus for complex,
haiku for cheap background calls). With `--model-map` you can route each
to the right provider model:

```bash
prism --provider https://api.mistral.ai/v1 \
      --key sk-... \
      --model-map
```

Interactive flow:

```
Endpoints to map:
  claude-opus-4-7    → mistral-large-latest      ✓
  claude-sonnet-4-6  → codestral-latest           ✓
  claude-haiku-4-5   → ministral-8b-latest        ✓
  claude-haiku-3-5   → [skipped]
  ...

Fallback model [mistral-large-latest]: _
```

Unmapped models fall back to the fallback. Each model is validated
with a live ping before being accepted.

---

## Supported providers

Any OpenAI-compatible endpoint works out of the box:

| Provider | Base URL |
|---|---|
| Mistral AI | `https://api.mistral.ai/v1` |
| Groq | `https://api.groq.com/openai/v1` |
| NVIDIA NIM | `https://integrate.api.nvidia.com/v1` |
| Cloudflare Workers AI | `https://api.cloudflare.com/client/v4/accounts/{ID}/ai/v1` |
| Together AI | `https://api.together.xyz/v1` |
| OpenRouter | `https://openrouter.ai/api/v1` |
| Ollama | `http://localhost:11434/v1` |

---

## Supported client tools

Prism auto-detects the client format from the first request.
Any tool that speaks Anthropic or OpenAI-compat wire format works:

- **Claude Code** — tested, full agentic loop confirmed
- **Aider** — OpenAI-compat, should work out of the box
- **Cursor** — OpenAI-compat
- **Continue** — OpenAI-compat
- **Cline / Roo** — Anthropic or OpenAI-compat depending on config

---

## What gets translated

**Request (client → provider):**
- Message format (Anthropic content blocks ↔ OpenAI messages array)
- System prompt (top-level field ↔ system role message)
- Tool definitions (input_schema ↔ parameters)
- Tool results (tool_result blocks ↔ tool role messages)
- Model name (client model → your chosen provider model)

**Response (provider → client):**
- Content blocks (choices[0].message ↔ content[])
- Stop reason (finish_reason ↔ stop_reason, with value mapping)
- Tool calls (tool_calls array ↔ tool_use content blocks)
- Token usage (prompt_tokens ↔ input_tokens)
- Thinking blocks (reasoning_content / thinking blocks → Anthropic thinking format)

**Streaming:**
- Text tokens streamed in real time
- Thinking blocks streamed as they arrive
- Tool calls buffered and emitted complete at end of stream

**Never touched:**
- Tool function names and arguments
- Message text content
- Multi-turn conversation history

---

## Architecture

```
prism/
  proxy.py              FastAPI server
  bridge.py             Live state — provider config, model map, learned client
  slots.py              Semantic slot definitions (the translation Rosetta Stone)
  probe/
    provider.py         Probe provider: discover models, learn response shape
    client.py           Learn client format from first request
    capabilities.py     Detect model capabilities (thinking blocks etc)
    frontends.py        Known frontend model lists for --model-map
  translate/
    request.py          Client request → provider request
    response.py         Provider response → client response
    headers.py          Header mapping and injection
    stream.py           Real-time SSE stream translation
```

---

## CLI reference

```
prism --provider URL    Provider base URL
      --key KEY         Provider API key
      --model MODEL     Single model to use
      --model-map       Interactive model mapping mode
      --frontend NAME   Frontend for model-map (default: claude-code)
      --port PORT       Port to listen on (default: 8000)
      --log-level       debug / info / warning (default: info)
```

---

## Environment variables

```bash
PRISM_PROVIDER=https://api.mistral.ai/v1
PRISM_KEY=sk-...
PRISM_MODEL=mistral-small-latest
prism
```

---

## License

MIT — [Alpha-Legents](https://github.com/Alpha-Legents)