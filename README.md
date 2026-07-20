# Prism 🔺

**Universal LLM protocol bridge — use any provider with any coding agent.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)

## What is Prism?

Prism is a proxy that sits between your AI coding tool and any OpenAI-compatible LLM provider. It translates between the Anthropic API format (used by Claude Code, Cline, Roo Code) and the OpenAI API format (used by Cursor, Aider, Continue, Windsurf), so you can use **any model from any provider** with **any coding tool**.

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│  Claude Code │────▶│             │────▶│  OpenRouter      │
│  (Anthropic) │     │             │     │  Groq            │
├─────────────┤     │   Prism     │     │  Mistral         │
│  Cursor      │────▶│   Proxy     │────▶│  Together AI     │
│  (OpenAI)    │     │             │     │  NVIDIA NIM      │
├─────────────┤     │             │     │  Ollama          │
│  Aider       │────▶│             │────▶│  Any OpenAI-compat│
│  (OpenAI)    │     └─────────────┘     └─────────────────┘
└─────────────┘
```

## Quick Start

```bash
# Install
pip install prism-proxy

# Run with interactive model mapping (recommended)
prism --provider https://openrouter.ai/api/v1 --key sk-... --interactive

# Or run with a single model
prism --provider https://api.groq.com/openai/v1 --key sk-... --model llama-3.3-70b-versatile
```

## Connect Your Tool

### Claude Code / Cline / Roo Code
```bash
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_API_KEY=prism  # any value works
claude  # or cline, roo-code
```

### Cursor / Aider / Continue / Windsurf
```bash
export OPENAI_BASE_URL=http://localhost:8000
export OPENAI_API_KEY=prism  # any value works
```

### In Tool Settings
| Tool | Setting | Value |
|------|---------|-------|
| Claude Code | `ANTHROPIC_BASE_URL` | `http://localhost:8000` |
| Cursor | Settings → Models → OpenAI Base URL | `http://localhost:8000/v1` |
| Aider | `--openai-api-base http://localhost:8000/v1` | |
| Cline | API Provider → Custom OpenAI Compatible | `http://localhost:8000` |
| Continue | `models[].apiBase` | `http://localhost:8000` |

## Supported Providers

Any OpenAI-compatible endpoint works:

| Provider | Base URL | Notes |
|----------|----------|-------|
| OpenRouter | `https://openrouter.ai/api/v1` | Best variety, 200+ models |
| Groq | `https://api.groq.com/openai/v1` | Fast inference |
| Mistral AI | `https://api.mistral.ai/v1` | Good code models |
| Together AI | `https://api.together.xyz/v1` | Open source models |
| NVIDIA NIM | `https://integrate.api.nvidia.com/v1` | Enterprise models |
| Ollama | `http://localhost:11434/v1` | Local models |
| vLLM | `http://localhost:8000/v1` | Self-hosted |
| LM Studio | `http://localhost:1234/v1` | Local GUI |

## Supported Clients

| Client | Format | Status | Notes |
|--------|--------|--------|-------|
| Claude Code | Anthropic | ✅ Full | Streaming, tools, thinking |
| Cline / Roo Code | Anthropic | ✅ Full | Streaming, tools |
| Cursor | OpenAI | ✅ Full | Streaming, tools |
| Aider | OpenAI | ✅ Full | Streaming, tools |
| Continue | OpenAI | ✅ Full | Streaming, tools |
| Windsurf | OpenAI | ✅ Full | Streaming |
| GitHub Copilot | OpenAI | ✅ Basic | Chat completions |
| Any OpenAI client | OpenAI | ✅ Full | Auto-detected |
| Any Anthropic client | Anthropic | ✅ Full | Auto-detected |

## Features

- **Auto-format detection** — Works with both Anthropic and OpenAI APIs automatically
- **Streaming** — Full SSE streaming with proper event sequences
- **Tool use** — Complete tool/function calling support
- **Thinking** — Extended thinking / reasoning block support
- **Prompt caching** — Cache control markers preserved
- **Token counting** — `/v1/messages/count_tokens` endpoint
- **Model mapping** — Route any Claude model to any provider model
- **Interactive setup** — Fetches provider models and prompts for mappings
- **Multi-key round-robin** — Distribute load across API keys
- **Error translation** — Provider errors mapped to client-expected format
- **CORS support** — Works with browser-based tools
- **Health dashboard** — `/health` endpoint with status info

## CLI Options

```
prism [OPTIONS]

Options:
  --provider URL        Provider base URL (required)
  --key KEY             Single API key
  --keys KEYS           Comma-separated keys for round-robin
  --model MODEL         Single model to use
  --model-map MAP       Comma-separated client=provider pairs
  --interactive, -i     Interactive model mapping
  --fallback-model M    Default model when no mapping matches
  --host HOST           Bind address (default: 127.0.0.1)
  --port PORT           Server port (default: 8000)
  --log-level LEVEL     debug/info/warning
  --config, -c FILE     YAML config file
```

All flags have `PRISM_*` env var equivalents (e.g. `PRISM_PROVIDER`, `PRISM_KEY`).

## Config File

Create `prism_config.yaml`:

```yaml
provider_url: https://openrouter.ai/api/v1
api_key: sk-or-...
model_map:
  claude-opus-4-6: anthropic/claude-opus-4
  claude-sonnet-4-6: anthropic/claude-sonnet-4
  claude-haiku-4-5: anthropic/claude-3.5-haiku
fallback_model: anthropic/claude-3.5-haiku
host: 127.0.0.1
port: 8000
```

Run with: `prism -c prism_config.yaml`

## Model Mapping

Map Claude model names to your provider's models:

```bash
# Command line
prism --provider https://openrouter.ai/api/v1 --key sk-... \
  --model-map 'claude-opus-4-6=anthropic/claude-opus-4,claude-sonnet-4-6=anthropic/claude-sonnet-4'

# Interactive (recommended)
prism --provider https://openrouter.ai/api/v1 --key sk-... --interactive
```

### Built-in Model Aliases

Prism includes pre-configured mappings for popular providers. When you use a known provider URL, Claude models are automatically mapped:

| Claude Model | OpenRouter | Groq | Mistral |
|-------------|------------|------|---------|
| claude-opus-4-6 | anthropic/claude-opus-4 | llama-3.3-70b-versatile | mistral-large-latest |
| claude-sonnet-4-6 | anthropic/claude-sonnet-4 | llama-3.1-70b-versatile | codestral-latest |
| claude-haiku-4-5 | anthropic/claude-3.5-haiku | llama-3.1-8b-instant | ministral-8b-latest |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/messages` | POST | Anthropic message API |
| `/v1/chat/completions` | POST | OpenAI chat completions API |
| `/v1/models` | GET | List available models |
| `/v1/messages/count_tokens` | POST | Count tokens in messages |
| `/health` | GET | Health check & status |
| `/` | GET | Setup instructions |

## What Gets Translated

**Requests:**
- Content blocks ↔ messages
- System prompts
- Tool definitions / `input_schema`
- Tool results
- Model names (via mapping)
- Thinking blocks
- Images (base64 and URLs)
- Cache control markers

**Responses:**
- `choices[0].message` ↔ `content[]`
- `finish_reason` ↔ `stop_reason`
- Tool calls
- Token usage (including cache tokens)
- Thinking/reasoning blocks

**Streaming:**
- Text deltas
- Tool call chunks (partial JSON)
- Thinking deltas with signatures
- Proper SSE event format for each protocol

**Never touched:** Tool function names/arguments, message text, conversation history, auth.

## Health Dashboard

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "version": "0.7.0",
  "uptime_seconds": 123.4,
  "requests": 42,
  "errors": 0,
  "provider": "https://openrouter.ai/api/v1",
  "model": "anthropic/claude-sonnet-4",
  "model_map_count": 3,
  "keys_configured": 1,
  "endpoints": {
    "anthropic": ["/v1/messages", "/messages"],
    "openai": ["/v1/chat/completions", "/chat/completions"],
    "models": ["/v1/models", "/models"],
    "token_count": ["/v1/messages/count_tokens"]
  }
}
```

## Install

```bash
# From PyPI
pip install prism-proxy

# From source
git clone <url>
cd prism
pip install -e .
```

## License

MIT
