"""Prism CLI entrypoint — colorful, interactive, user-friendly."""

import os
import sys
import asyncio
import argparse
import uvicorn
import yaml

# ── Colors ────────────────────────────────────────────────────────────────

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init()
    C = Fore
    R = Style.RESET_ALL
    B = Style.BRIGHT
except ImportError:
    # Fallback: no colors
    class _NoColor:
        def __getattr__(self, _): return ""
    C = _NoColor()
    R = ""
    B = ""

BANNER = f"""
{C.CYAN}{B}  ██████╗ ██████╗ ██╗███████╗███╗   ███╗
  ██╔══██╗██╔══██╗██║██╔════╝████╗ ████║
  ██████╔╝██████╔╝██║███████╗██╔████╔██║
  ██╔═══╝ ██╔══██╗██║╚════██║██║╚██╔╝██║
  ██║     ██║  ██║██║███████║██║ ╚═╝ ██║
  ╚═╝     ╚═╝  ╚═╝╚═╝╚══════╝╚═╝     ╚═╝{R}
{C.GREEN}        LLM protocol bridge — v0.7.0{R}
"""


# Claude models that frontends may request
CLAUDE_MODELS = [
    ("claude-opus-4-6", "Most capable — complex tasks"),
    ("claude-sonnet-4-6", "Balanced — everyday use"),
    ("claude-haiku-4-5", "Fastest — quick tasks"),
    ("claude-opus-4-1", "Legacy Opus"),
    ("claude-opus-4-20250514", "Legacy Opus (dated)"),
    ("claude-sonnet-4", "Legacy Sonnet"),
    ("claude-sonnet-4-20250514", "Legacy Sonnet (dated)"),
    ("claude-3-7-sonnet", "Legacy Sonnet 3.7"),
    ("claude-3-5-sonnet", "Legacy Sonnet 3.5"),
    ("claude-3-5-haiku", "Legacy Haiku 3.5"),
]


def parse_model_map(raw: str) -> dict:
    """Parse 'client-model=provider-model,...' pairs into a mapping."""
    mapping = {}
    for pair in raw.split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            if k.strip() and v.strip():
                mapping[k.strip()] = v.strip()
    return mapping


async def fetch_provider_models(provider_url: str, api_key: str | None = None) -> list[str]:
    """Fetch available models from the provider."""
    import httpx

    base = provider_url.rstrip("/")
    for suffix in ["/v1/models", "/models"]:
        url = base + suffix if not base.endswith(suffix) else base

        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(url, headers=headers)
                if r.status_code == 200:
                    data = r.json()
                    models = []
                    if isinstance(data, list):
                        models = [m.get("id", str(m)) if isinstance(m, dict) else str(m) for m in data]
                    elif isinstance(data, dict):
                        if "data" in data:
                            models = [m.get("id", "") for m in data["data"] if isinstance(m, dict)]
                        elif "models" in data:
                            models = [m.get("id", "") for m in data["models"] if isinstance(m, dict)]
                    if models:
                        return sorted(models)
        except Exception:
            continue

    return []


def interactive_model_mapping(provider_models: list[str]) -> dict:
    """Interactive prompt to map Claude models to provider models."""
    print()
    print(f"  {C.CYAN}{B}┌─────────────────────────────────────────────────────────────┐{R}")
    print(f"  {C.CYAN}{B}│            Interactive Model Mapping                        │{R}")
    print(f"  {C.CYAN}{B}│  Map each Claude model to a provider model.                │{R}")
    print(f"  {C.CYAN}{B}│  Press Enter to skip, 'done' to finish early.              │{R}")
    print(f"  {C.CYAN}{B}└─────────────────────────────────────────────────────────────┘{R}")
    print()

    if provider_models:
        print(f"  {C.GREEN}Available provider models ({len(provider_models)}):{R}")
        for i, m in enumerate(provider_models[:20], 1):
            print(f"    {C.YELLOW}{i:2d}.{R} {m}")
        if len(provider_models) > 20:
            print(f"    {C.DIM}... and {len(provider_models) - 20} more{R}")
        print()

    mapping = {}

    for model_id, description in CLAUDE_MODELS:
        try:
            print(f"  {C.CYAN}{B}{model_id}{R} {C.DIM}({description}){R}")
            value = input(f"    {C.GREEN}→{R} provider model {C.DIM}(Enter to skip):{R} ").strip()
            if value.lower() == "done":
                break
            if value:
                if value.isdigit() and provider_models:
                    idx = int(value) - 1
                    if 0 <= idx < len(provider_models):
                        mapping[model_id] = provider_models[idx]
                        print(f"    {C.GREEN}✓{R} Mapped to {provider_models[idx]}")
                    else:
                        print(f"    {C.RED}✗{R} Invalid number, skipped")
                else:
                    mapping[model_id] = value
                    print(f"    {C.GREEN}✓{R} Mapped to {value}")
            else:
                print(f"    {C.DIM}· Skipped{R}")
            print()
        except (EOFError, KeyboardInterrupt):
            print()
            break

    return mapping


def print_success_box(model_map, model, fallback_model, host, port):
    """Print a beautiful success box."""
    print()
    print(f"  {C.GREEN}{B}┌─────────────────────────────────────────────────────────────┐{R}")
    print(f"  {C.GREEN}{B}│  {C.CYAN}Prism v0.7.0{C.GREEN} — Ready!                                  │{R}")
    print(f"  {C.GREEN}{B}├─────────────────────────────────────────────────────────────┤{R}")

    if model_map:
        print(f"  {C.GREEN}{B}│{R}  {C.YELLOW}Mode:{R} model-map ({len(model_map)} mappings)                      {C.GREEN}{B}│{R}")
        for k, v in list(model_map.items())[:5]:
            print(f"  {C.GREEN}{B}│{R}    {k:28s} → {v:28s} {C.GREEN}{B}│{R}")
        if len(model_map) > 5:
            print(f"  {C.GREEN}{B}│{R}    {C.DIM}... and {len(model_map) - 5} more{R}                              {C.GREEN}{B}│{R}")
        if fallback_model:
            print(f"  {C.GREEN}{B}│{R}  {C.YELLOW}Fallback:{R} {fallback_model:47s} {C.GREEN}{B}│{R}")
    else:
        print(f"  {C.GREEN}{B}│{R}  {C.YELLOW}Model:{R} {model:52s} {C.GREEN}{B}│{R}")

    print(f"  {C.GREEN}{B}│{R}  {C.YELLOW}Listen:{R} {host}:{port:47s} {C.GREEN}{B}│{R}")
    print(f"  {C.GREEN}{B}├─────────────────────────────────────────────────────────────┤{R}")
    print(f"  {C.GREEN}{B}│{R}  {C.WHITE}Connect your tool:{R}                                        {C.GREEN}{B}│{R}")
    print(f"  {C.GREEN}{B}│{R}                                                              {C.GREEN}{B}│{R}")
    print(f"  {C.GREEN}{B}│{R}  {C.CYAN}Claude Code:{R}                                              {C.GREEN}{B}│{R}")
    print(f"  {C.GREEN}{B}│{R}    ANTHROPIC_BASE_URL=http://localhost:{port:<24s} {C.GREEN}{B}│{R}")
    print(f"  {C.GREEN}{B}│{R}    ANTHROPIC_API_KEY=prism{C.DIM}  (or any value){R}              {C.GREEN}{B}│{R}")
    print(f"  {C.GREEN}{B}│{R}                                                              {C.GREEN}{B}│{R}")
    print(f"  {C.GREEN}{B}│{R}  {C.CYAN}Cursor / Aider / Cline:{R}                                   {C.GREEN}{B}│{R}")
    print(f"  {C.GREEN}{B}│{R}    OPENAI_BASE_URL=http://localhost:{port:<23s} {C.GREEN}{B}│{R}")
    print(f"  {C.GREEN}{B}│{R}    OPENAI_API_KEY=prism{C.DIM}  (or any value){R}               {C.GREEN}{B}│{R}")
    print(f"  {C.GREEN}{B}│{R}                                                              {C.GREEN}{B}│{R}")
    print(f"  {C.GREEN}{B}│{R}  {C.DIM}Status: http://localhost:{port}/health{R}                    {C.GREEN}{B}│{R}")
    print(f"  {C.GREEN}{B}└─────────────────────────────────────────────────────────────┘{R}")
    print()


def main():
    parser = argparse.ArgumentParser(
        prog="prism",
        description="LLM protocol bridge — any provider, any coding agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--provider", default=os.environ.get("PRISM_PROVIDER"),
                        help="Provider base URL (e.g. https://openrouter.ai/api/v1)")
    parser.add_argument("--key", default=os.environ.get("PRISM_KEY"),
                        help="Single API key")
    parser.add_argument("--keys", default=os.environ.get("PRISM_KEYS"),
                        help="Comma-separated API keys for round-robin failover")
    parser.add_argument("--model", default=os.environ.get("PRISM_MODEL"),
                        help="Single model to use")
    parser.add_argument("--model-map", default=os.environ.get("PRISM_MODEL_MAP"),
                        help="Comma-separated client=provider model pairs")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactive model mapping — fetches models and prompts for mappings")
    parser.add_argument("--fallback-model", default=os.environ.get("PRISM_FALLBACK_MODEL"),
                        help="Default model when a requested model has no mapping")
    parser.add_argument("--host", default=os.environ.get("PRISM_HOST", "127.0.0.1"),
                        help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("PRISM_PORT", "8000")))
    parser.add_argument("--log-level", default=os.environ.get("PRISM_LOG_LEVEL", "info"),
                        choices=["debug", "info", "warning"])
    parser.add_argument("--config", "-c",
                        help="Path to YAML config file")

    args = parser.parse_args()

    config = {}
    config_path = args.config or os.environ.get("PRISM_CONFIG")
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

    # CLI args take precedence over config file values
    provider = args.provider or config.get("provider_url")
    api_key = args.key or config.get("api_key")
    model = args.model or config.get("model")
    fallback_model = args.fallback_model or config.get("fallback_model")
    host = args.host or config.get("host", "127.0.0.1")
    port = args.port or config.get("port", 8000)
    log_level = args.log_level or config.get("log_level", "info")

    model_map = {}
    if args.model_map:
        model_map = parse_model_map(args.model_map)
    elif isinstance(config.get("model_map"), dict):
        model_map = config["model_map"]

    if not provider:
        parser.print_help()
        print(f"\n  {C.GREEN}Quick start:{R}")
        print(f"    prism --provider https://openrouter.ai/api/v1 --key sk-... --interactive")
        print(f"    prism --provider https://api.groq.com/openai/v1 --key sk-... --model llama-3.3-70b-versatile")
        print(f"    prism -c prism_config.yaml")
        print(f"\n  {C.DIM}Set PRISM_PROVIDER and PRISM_KEY env vars to skip --provider and --key{R}")
        sys.exit(1)

    if not model and not model_map and not args.interactive:
        print(f"  {C.RED}Error:{R} provide --model, --model-map, or --interactive")
        sys.exit(1)

    print(BANNER)

    from .core.bridge import get_bridge
    from .probe.provider import probe_provider

    print(f"  {C.CYAN}Probing{R} {provider}...")

    # Use first key for probing
    keys = [k.strip() for k in args.keys.split(",") if k.strip()] if args.keys else []
    probe_key = keys[0] if keys else api_key

    # Fetch provider models
    provider_models = []
    try:
        provider_models = asyncio.run(fetch_provider_models(provider, api_key=probe_key or None))
        if provider_models:
            print(f"  {C.GREEN}✓{R} Found {len(provider_models)} models on provider")
    except Exception as e:
        print(f"  {C.YELLOW}⚠{R} Could not fetch models: {e}")

    try:
        probe_schema = asyncio.run(probe_provider(provider, api_key=probe_key or None))
        if probe_schema.models and not provider_models:
            provider_models = probe_schema.models
    except Exception as e:
        print(f"  {C.YELLOW}⚠{R} Provider probe failed: {e}")

    # Interactive model mapping
    if args.interactive and not model_map:
        model_map = interactive_model_mapping(provider_models)
        if not model_map:
            print(f"  {C.RED}No mappings configured. Exiting.{R}")
            sys.exit(1)
        # Save the mapping for next time
        print(f"  {C.DIM}Tip: Save this to a config file with: prism --provider {provider} --key ... --model-map '{','.join(f'{k}={v}' for k,v in model_map.items())}'{R}")

    # Configure bridge
    bridge = get_bridge()
    bridge.configure(
        provider_url=provider,
        api_key=api_key,
        api_keys=keys or None,
        model=model,
        model_map=model_map or None,
        fallback_model=fallback_model,
    )

    print_success_box(model_map, model, fallback_model, host, port)

    from .proxy import app
    uvicorn.run(app, host=host, port=port, log_level=log_level)


if __name__ == "__main__":
    main()
