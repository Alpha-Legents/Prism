"""Prism CLI entrypoint with config file support."""

import os
import sys
import asyncio
import argparse
import uvicorn
import yaml

BANNER = r"""
  ██████╗ ██████╗ ██╗███████╗███╗   ███╗
  ██╔══██╗██╔══██╗██║██╔════╝████╗ ████║
  ██████╔╝██████╔╝██║███████╗██╔████╔██║
  ██╔═══╝ ██╔══██╗██║╚════██║██║╚██╔╝██║
  ██║     ██║  ██║██║███████║██║ ╚═╝ ██║
  ╚═╝     ╚═╝  ╚═╝╚═╝╚══════╝╚═╝     ╚═╝
        LLM protocol bridge — v0.5.0
"""


def parse_model_map(raw: str) -> dict:
    """Parse 'client-model=provider-model,...' pairs into a mapping."""
    mapping = {}
    for pair in raw.split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            if k.strip() and v.strip():
                mapping[k.strip()] = v.strip()
    return mapping


def main():
    parser = argparse.ArgumentParser(
        prog="prism",
        description="LLM protocol bridge — any provider, any coding agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--provider", default=os.environ.get("PRISM_PROVIDER"))
    parser.add_argument("--key", default=os.environ.get("PRISM_KEY"),
                        help="Single API key")
    parser.add_argument("--keys", default=os.environ.get("PRISM_KEYS"),
                        help="Comma-separated list of API keys for round-robin failover")
    parser.add_argument("--model", default=os.environ.get("PRISM_MODEL"),
                        help="Single model override")
    parser.add_argument("--model-map", default=os.environ.get("PRISM_MODEL_MAP"),
                        help="Comma-separated client=provider model pairs, e.g. "
                             "'claude-opus-4-7=mistral-large-latest,"
                             "claude-haiku-4-5=ministral-8b-latest'")
    parser.add_argument("--fallback-model", default=os.environ.get("PRISM_FALLBACK_MODEL"),
                        help="Provider model used when a requested model has no mapping")
    parser.add_argument("--host", default=os.environ.get("PRISM_HOST", "127.0.0.1"),
                        help="Bind address (default: 127.0.0.1; use 0.0.0.0 to expose)")
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
        print("\nExamples:")
        print("  prism --provider https://api.mistral.ai/v1 --key sk-... --model mistral-small-latest")
        print("  prism --provider https://api.groq.com/openai/v1 --key sk-... \\")
        print("        --model-map 'claude-opus-4-7=llama-3.3-70b-versatile' --fallback-model llama-3.1-8b-instant")
        print("  prism -c prism_config.yaml")
        sys.exit(1)

    if not model and not model_map:
        print("Error: provide --model or --model-map")
        sys.exit(1)

    print(BANNER)

    from .core.bridge import get_bridge
    from .probe.provider import probe_provider

    print(f"  Probing {provider}...")

    # Use first key for probing (or single key)
    keys = [k.strip() for k in args.keys.split(",") if k.strip()] if args.keys else []
    probe_key = keys[0] if keys else api_key

    try:
        asyncio.run(probe_provider(provider, api_key=probe_key or None))
    except Exception as e:
        print(f"  ⚠ Provider probe failed: {e}")

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

    # Print summary
    print()
    if model_map:
        print(f"  mode     → model-map ({len(model_map)} mappings)")
        for k, v in model_map.items():
            print(f"             {k} → {v}")
        if fallback_model:
            print(f"  fallback → {fallback_model}")
    else:
        print(f"  model    → {bridge.config.model}")
    print(f"  listen   → {host}:{port}")
    print()
    print(f"  Connect your tool:")
    print(f"    ANTHROPIC_BASE_URL=http://localhost:{port}")
    print(f"    ANTHROPIC_API_KEY=***")
    print()

    from .proxy import app
    uvicorn.run(app, host=host, port=port, log_level=log_level)


if __name__ == "__main__":
    main()
