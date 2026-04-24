# Prism 🔺

**Destination-inferred protocol translation proxy for LLM APIs.**

Point prism at any target endpoint. It learns the response shape automatically.
Everything coming from any upstream provider (Groq, Together, OpenRouter, Mistral, Gemini...)
gets translated to match — headers, envelope, stop reasons, tool calls.

No manual field mapping. No config files. The target IS the config.

---

## The idea

Every LLM API speaks a slightly different dialect of JSON.  
Same concepts. Different wrappers.

```
Groq returns:          Claude Code expects:
choices[0]             content[]
finish_reason          stop_reason
prompt_tokens          input_tokens
tool_calls[]           content[type=tool_use]
```

Prism sits between them. It probes your target once at startup,
learns the exact shape it expects, then translates everything automatically.

**Message content and tool call internals are never touched.**  
Only the structural envelope gets rewritten.

---

## Quick start

```bash
pip install prism-proxy

# Point at Claude Code (or any Anthropic-compatible endpoint)
prism start --target https://api.anthropic.com/v1/messages \
            --target-key sk-ant-...

# Your Groq/Together/OpenRouter app now hits localhost:8000
# and gets back perfect Anthropic-shaped responses
```

---

## How it works

```
1. startup
   prism fires a minimal probe at your --target
   captures the response shape → that's the translation contract

2. runtime
   upstream response arrives (any provider)
   ↓
   provider auto-detected from response shape
   ↓
   semantic slots extracted (response_text, stop_reason, tokens, tool_calls...)
   ↓
   slots mapped into target schema
   ↓
   headers translated + injected
   ↓
   target-shaped response returned
```

---

## Supported providers (auto-detected)

| Provider | Format detected as |
|---|---|
| Groq | `openai-compat` |
| Together AI | `openai-compat` |
| OpenRouter | `openai-compat` |
| Mistral (direct) | `openai-compat` |
| Anthropic | `anthropic` |
| Gemini | `gemini` |

---

## Supported targets

| Target | Schema type |
|---|---|
| Anthropic / Claude Code | `anthropic` |
| OpenAI-compatible | `openai` |

---

## Tool calls

Tool call **internals** (function name, arguments, tool IDs) are always preserved verbatim.

Only the wrapper format gets translated:

```
OpenAI tool_calls[] ←→ Anthropic content[type=tool_use]
```

---

## Environment variables

| Variable | Description |
|---|---|
| `PRISM_TARGET` | Target endpoint URL |
| `PRISM_TARGET_KEY` | API key for the target |
| `PRISM_UPSTREAM_KEY` | Key to accept from callers (optional) |

---

## Probe a target manually

```bash
prism probe https://api.anthropic.com/v1/messages --key sk-ant-...
```

Prints the learned schema — useful for debugging unknown endpoints.

---

## Kaggle / Colab + localtunnel

```python
# In your notebook:
import subprocess, os

os.environ["PRISM_TARGET"]     = "https://api.anthropic.com/v1/messages"
os.environ["PRISM_TARGET_KEY"] = "sk-ant-..."

subprocess.Popen(["prism", "start", "--port", "8000"])
subprocess.Popen(["lt", "--port", "8000"])
```

---

## Architecture

```
prism/
  slots.py       — semantic slot definitions + provider detection
  learner.py     — target schema probing + learning
  translator.py  — core response translation engine
  headers.py     — header mapping + injection
  proxy.py       — FastAPI server + request forwarding
  __main__.py    — CLI (prism start / prism probe)
```

---

## License

MIT — Alpha (github.com/Alpha-Legents)
