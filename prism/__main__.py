import os, sys, asyncio, argparse, uvicorn

def main():
    parser = argparse.ArgumentParser(prog="prism")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--provider", default=os.environ.get("PRISM_PROVIDER"))
    parser.add_argument("--provider-key", default=os.environ.get("PRISM_PROVIDER_KEY"))
    parser.add_argument("--model", default=os.environ.get("PRISM_MODEL"))
    args = parser.parse_args()

    if not args.provider or not args.model:
        print("Usage: prism --provider URL --provider-key KEY --model MODEL")
        sys.exit(1)

    from .bridge import get_bridge
    from .probe.provider import probe_provider, _completion_url

    async def setup():
        schema = await probe_provider(args.provider, api_key=args.provider_key)
        bridge = get_bridge()
        bridge.provider = schema
        bridge.provider.api_key = args.provider_key
        bridge.provider.completion_url = _completion_url(args.provider)
        bridge.model = args.model
        bridge.mark_ready()
        print(f"✓ Provider: {schema.format} | Model: {args.model} | Port: {args.port}")

    asyncio.run(setup())

    from .proxy import app
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="debug")

if __name__ == "__main__":
    main()