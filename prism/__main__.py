"""
Prism CLI

Single model:
  prism --provider https://api.mistral.ai/v1 --key sk-... --model mistral-small-latest

Interactive model mapping (one provider model per frontend endpoint):
  prism --provider https://api.mistral.ai/v1 --key sk-... --model-map
  prism --provider https://api.mistral.ai/v1 --key sk-... --model-map --frontend claude-code

Then point your tool at prism:
  ANTHROPIC_BASE_URL=http://localhost:8000
  ANTHROPIC_API_KEY=prism
"""

import os
import sys
import asyncio
import argparse
import uvicorn

BANNER = """
  ██████╗ ██████╗ ██╗███████╗███╗   ███╗
  ██╔══██╗██╔══██╗██║██╔════╝████╗ ████║
  ██████╔╝██████╔╝██║███████╗██╔████╔██║
  ██╔═══╝ ██╔══██╗██║╚════██║██║╚██╔╝██║
  ██║     ██║  ██║██║███████║██║ ╚═╝ ██║
  ╚═╝     ╚═╝  ╚═╝╚═╝╚══════╝╚═╝     ╚═╝
  LLM protocol bridge  v0.3.0
"""


def main():
    parser = argparse.ArgumentParser(
        prog="prism",
        description="LLM protocol bridge — any provider, any coding agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--provider",   default=os.environ.get("PRISM_PROVIDER"))
    parser.add_argument("--key",        default=os.environ.get("PRISM_KEY"))
    parser.add_argument("--model",      default=os.environ.get("PRISM_MODEL"),
                        help="Single model override")
    parser.add_argument("--model-map",  action="store_true",
                        help="Interactive: map one provider model per frontend endpoint")
    parser.add_argument("--frontend",   default="claude-code",
                        help="Frontend tool for model-map (default: claude-code)")
    parser.add_argument("--port",       type=int, default=8000)
    parser.add_argument("--log-level",  default="info",
                        choices=["debug", "info", "warning"])

    args = parser.parse_args()

    if not args.provider:
        parser.print_help()
        print("\nExamples:")
        print("  prism --provider https://api.mistral.ai/v1 --key sk-... --model mistral-small-latest")
        print("  prism --provider https://api.mistral.ai/v1 --key sk-... --model-map")
        sys.exit(1)

    print(BANNER)

    # Probe provider
    print(f"  Probing {args.provider}...")
    from .probe.provider import probe_provider, completion_url

    schema = asyncio.run(probe_provider(args.provider, api_key=args.key or None))
    comp_url = completion_url(args.provider)

    if schema.models:
        print(f"  ✓ {len(schema.models)} models available")
    else:
        print(f"  ⚠ Could not fetch model list — will proceed anyway")

    # Wire bridge base config
    from .bridge import get_bridge
    bridge                 = get_bridge()
    bridge.provider        = schema
    bridge.api_key         = args.key or None
    bridge.provider_format = schema.format if schema.format != "unknown" else "openai-compat"
    bridge.completion_url  = comp_url

    # Mode: single model or model map
    if args.model_map:
        model_map, fallback = _interactive_model_map(
            args.frontend,
            schema.models,
            comp_url,
            args.key,
        )
        bridge.model_map      = model_map
        bridge.fallback_model = fallback
        bridge.model          = fallback  # also set as single fallback
    elif args.model:
        bridge.model = args.model
    else:
        print("\nERROR: provide --model MODEL or --model-map")
        sys.exit(1)

    bridge.mark_ready()

    # Print summary
    print()
    if bridge.model_map:
        print(f"  mode     → model-map ({len(bridge.model_map)} mapped)")
        for frontend_m, provider_m in bridge.model_map.items():
            print(f"    {frontend_m:<40} → {provider_m}")
        if bridge.fallback_model:
            print(f"    (others) → {bridge.fallback_model} [fallback]")
    else:
        print(f"  model    → {bridge.model}")

    print(f"  port     → {args.port}")
    print()
    print(f"  Connect your tool:")
    print(f"    ANTHROPIC_BASE_URL=http://localhost:{args.port}")
    print(f"    ANTHROPIC_API_KEY=prism")
    print()

    from .proxy import app
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level=args.log_level)


def _interactive_model_map(
    frontend: str,
    available_provider_models: list[str],
    completion_url: str,
    api_key: str | None,
) -> tuple[dict[str, str], str | None]:
    """
    Interactive CLI flow to map frontend model endpoints → provider models.
    Returns (model_map, fallback_model).
    """
    from .probe.frontends import detect_frontend_from_flag

    frontend_models = detect_frontend_from_flag(frontend)

    print(f"\n  Frontend: {frontend}")
    print(f"  Endpoints to map ({len(frontend_models)}):\n")

    model_map: dict[str, str] = {}

    # Show available provider models for reference
    if available_provider_models:
        print("  Available provider models:")
        for m in available_provider_models[:15]:
            print(f"    {m}")
        if len(available_provider_models) > 15:
            print(f"    ... and {len(available_provider_models) - 15} more")
        print()

    print("  For each frontend endpoint, enter a provider model name.")
    print("  Press enter to skip (will use fallback), or type 'quit' to cancel.\n")

    for fe_model in frontend_models:
        while True:
            try:
                val = input(f"  {fe_model:<42} → ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Cancelled.")
                sys.exit(0)

            if val.lower() in ("quit", "exit", "q"):
                print("  Cancelled.")
                sys.exit(0)

            if val == "":
                # Skip this endpoint
                print(f"    ↳ skipped")
                break

            # Validate the model exists on provider
            print(f"    ↳ validating {val}...", end=" ", flush=True)
            valid, err = asyncio.run(_validate_model(val, completion_url, api_key))

            if valid:
                print("✓")
                model_map[fe_model] = val
                break
            else:
                print(f"⚠ {err}")
                try:
                    retry = input(f"    Press enter to use anyway, 'r' to re-enter, or 's' to skip: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    sys.exit(0)

                if retry == "r":
                    continue  # re-enter
                elif retry == "s":
                    print(f"    ↳ skipped")
                    break
                else:
                    # Use anyway
                    model_map[fe_model] = val
                    break

    if not model_map:
        print("\n  No models mapped. Exiting.")
        sys.exit(1)

    # Ask for fallback
    print(f"\n  {len(model_map)} model(s) mapped.")
    print(f"  Unmapped endpoints will use a fallback model.")

    first_mapped = list(model_map.values())[0]

    try:
        fallback = input(f"  Fallback model [{first_mapped}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        fallback = ""

    if not fallback:
        fallback = first_mapped

    print(f"  Fallback → {fallback}")

    return model_map, fallback


async def _validate_model(model: str, comp_url: str, api_key: str | None) -> tuple[bool, str]:
    """
    Fire a minimal 1-token request to check if model exists and responds.
    Returns (is_valid, error_message).
    """
    import httpx

    headers = {
        "Content-Type":    "application/json",
        "accept-encoding": "identity",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model":      model,
        "max_tokens": 1,
        "stream":     False,
        "messages":   [{"role": "user", "content": "hi"}],
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(comp_url, json=payload, headers=headers)

        if resp.status_code in (200, 201):
            return True, ""
        if resp.status_code == 404:
            return False, "404 — model not found"
        if resp.status_code == 402:
            return False, "402 — payment required / no access"
        if resp.status_code == 401:
            return False, "401 — unauthorized (check API key)"
        if resp.status_code == 429:
            # Rate limited but model exists
            return True, ""

        return False, f"HTTP {resp.status_code}"

    except httpx.ConnectError:
        return False, "cannot reach provider"
    except httpx.TimeoutException:
        return False, "timeout"
    except Exception as e:
        return False, str(e)


if __name__ == "__main__":
    main()