# Prism 🔺

**LLM protocol bridge — use any provider with any AI coding tool.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)

```bash
pip install prism-proxy
prism --provider https://api.mistral.ai/v1 --key sk-... --model mistral-small-latest
```

Point your coding agent at Prism:

```
ANTHROPIC_BASE_URL=http://localhost:8000 ANTHROPIC_API_KEY=prism
```

Tool calls, streaming, multi-turn conversations — all work automatically.

---

## Supported providers

Any OpenAI-compatible endpoint:

Mistral AI, Groq, NVIDIA NIM, Cloudflare Workers AI, Together AI, OpenRouter, Ollama, or your custom URL.

## Supported clients

| Client | Format | Status |
|---|---|---|
| Claude Code | Anthropic | Tested |
| Cline / Roo Code | Anthropic | Tested |
| Aider | OpenAI | Compatible |
| Cursor | OpenAI | Compatible |
| Continue | OpenAI | Compatible |

## Model mapping

Route specific Claude models to specific provider models:

```bash
prism --provider https://api.mistral.ai/v1 --key sk-... --model-map
```

Interactive flow maps `claude-opus-4-7` → your best model, `claude-haiku-4-5` → your cheap model.

---

## CLI

```
prism --provider URL   Provider base URL
      --key KEY        API key
      --keys KEYS      Comma-separated keys (round-robin)
      --model MODEL    Model to use
      --model-map      Interactive model mapping
      --port PORT      Server port (default: 8000)
      --log-level      debug/info/warning
```

All flags have `PRISM_*` env var equivalents.

---

## What gets translated

**Requests:** content blocks ↔ messages, system prompt, tool definitions/`input_schema`, tool results, model names, thinking blocks, images, cache control.

**Responses:** `choices[0].message` ↔ `content[]`, `finish_reason` ↔ `stop_reason`, tool calls, token usage (including cache tokens), thinking blocks.

**Streaming:** Text deltas, tool call chunks (partial JSON), thinking blocks, proper SSE event format.

Never touched: tool function names and arguments, message text, conversation history, auth.

---

## Install

```bash
# From PyPI
pip install prism-proxy

# From source
git clone <url>
cd prism
pip install -e .
```

---

## License

MIT
