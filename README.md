# Prism 🔺

**LLM protocol bridge. Any provider, any coding agent, zero config.**

Point Prism between your AI coding tool and any LLM API provider.
It auto-detects both ends, translates the wire protocol transparently,
and preserves message content and tool calls exactly as-is.

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
```

Prism sits in the middle and handles all of it. Your coding agent thinks
it's talking to Anthropic. The provider thinks it's getting normal requests.
Neither side needs to know the other exists.

---

## Quick start

```bash
pip install prism-proxy
prism
```

The TUI opens. From there:

1. Enter provider name, URL, and API key
2. Hit **Fetch Models** — pulls the live model list from the provider
3. Select which models to include (space to toggle, `a` for all)
4. Add more providers if you want
5. Hit **Start Bridge**

Then point your coding agent at prism:

```bash
ANTHROPIC_BASE_URL=http://localhost:8000
ANTHROPIC_API_KEY=prism
```

That's it.

---

## Headless mode

For scripts or servers — no TUI:

```bash
# Single provider, multiple models (auto-fallback on rate limits)
prism --provider https://api.mistral.ai/v1 \
      --key sk-... \
      --models mistral-small-latest,codestral-latest,mistral-large-latest

# Multi-provider via config file
prism --config pool.yaml
```

`pool.yaml` example:

```yaml
providers:
  - url: https://api.mistral.ai/v1
    key: your-mistral-key
    models:
      - mistral-small-latest
      - codestral-latest

  - url: https://api.groq.com/openai/v1
    key: your-groq-key
    models:
      - llama-3.3-70b-versatile
      - llama-3.1-8b-instant

  - url: https://integrate.api.nvidia.com/v1
    key: your-nvidia-key
    models:
      - nvidia/llama-3.3-nemotron-super-49b-v1
      - moonshotai/kimi-k2-instruct
```

---

## Pool and fallback

When you add multiple models, Prism builds a pool and rotates through them:

- **429 rate limit** → silently tries next model
- **Timeout** → silently tries next model  
- **3 failures** → that model is disabled for the session
- **All entries exhausted** → returns 503 to the client

Completely transparent. Your coding agent never sees the rotation happening.

---

## What gets translated

**Request (client → provider):**
- Message format (Anthropic content blocks ↔ OpenAI messages array)
- System prompt (top-level Anthropic field ↔ system role message)
- Tool definitions (input_schema ↔ parameters)
- Tool results (tool_result blocks ↔ tool role messages)
- Model name (whatever the client sends → your chosen provider model)

**Response (provider → client):**
- Content blocks (choices[0].message ↔ content[])
- Stop reason (finish_reason ↔ stop_reason, with value mapping)
- Tool calls (tool_calls array ↔ tool_use content blocks)
- Token usage (prompt_tokens ↔ input_tokens)
- Headers (rate limit headers, content-type, encoding)

**Never touched:**
- Tool function names and arguments
- Message text content
- Multi-turn conversation history structure

---

## Tested with

**Client tools:** Claude Code

**Providers:** Mistral AI, NVIDIA NIM, Groq

**Confirmed working:** full agentic loop — multi-turn conversation,
tool calls, bash execution, file writes, error recovery and retry.

---

## Architecture

```
prism/
  proxy.py              FastAPI server, request routing
  bridge.py             Live state — pool + learned client format
  pool.py               Provider pool, rotation, fallback logic
  slots.py              Semantic slot definitions (the Rosetta Stone)
  tui.py                Textual TUI
  probe/
    provider.py         Probe provider: GET /models, learn response shape
    client.py           Learn client format from first request
  translate/
    request.py          Client request → provider request
    response.py         Provider response → client response
    headers.py          Header mapping and injection
```

---

## Requirements

- Python 3.11+
- Provider API key (Mistral, Groq, NVIDIA, or any OpenAI-compatible endpoint)

---

## License

MIT — [Alpha-Legents](https://github.com/Alpha-Legents)