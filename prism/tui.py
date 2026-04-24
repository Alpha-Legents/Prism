"""
Prism TUI.
Configure provider + client, pick model, start the bridge.
Built with Textual.
"""

import asyncio
import threading
import uvicorn
import logging

from textual.app            import App, ComposeResult
from textual.containers     import Vertical, Horizontal, ScrollableContainer
from textual.widgets        import (
    Header, Footer, Input, Button, Label,
    Select, Static, Log, Rule,
)
from textual.reactive       import reactive
from textual                import work

from .bridge       import get_bridge
from .probe.provider import probe_provider, ProviderSchema, _completion_url

logger = logging.getLogger("prism.tui")


# ── Proxy server runner (background thread) ──────────────────────────────────

class ProxyThread(threading.Thread):
    def __init__(self, port: int):
        super().__init__(daemon=True)
        self.port   = port
        self.server = None

    def run(self):
        from .proxy import app as proxy_app
        config = uvicorn.Config(
            proxy_app,
            host="0.0.0.0",
            port=self.port,
            log_level="warning",
        )
        self.server = uvicorn.Server(config)
        self.server.run()

    def stop(self):
        if self.server:
            self.server.should_exit = True


# ── Widgets ───────────────────────────────────────────────────────────────────

BANNER = """\
  ██████╗ ██████╗ ██╗███████╗███╗   ███╗
  ██╔══██╗██╔══██╗██║██╔════╝████╗ ████║
  ██████╔╝██████╔╝██║███████╗██╔████╔██║
  ██╔═══╝ ██╔══██╗██║╚════██║██║╚██╔╝██║
  ██║     ██║  ██║██║███████║██║ ╚═╝ ██║
  ╚═╝     ╚═╝  ╚═╝╚═╝╚══════╝╚═╝     ╚═╝
  dual-end bridge  v0.2.0\
"""


class StatusBar(Static):
    status = reactive("◉ not configured")

    def render(self) -> str:
        return self.status


# ── Main App ──────────────────────────────────────────────────────────────────

class PrismApp(App):
    CSS = """
    Screen {
        background: #0d0d0d;
        color: #e0e0e0;
    }
    #banner {
        color: #7c6af7;
        text-align: center;
        padding: 1 0;
    }
    #main {
        layout: grid;
        grid-size: 2;
        grid-gutter: 1;
        height: 1fr;
    }
    #left-panel, #right-panel {
        border: solid #2a2a2a;
        padding: 1 2;
        height: 100%;
    }
    .section-title {
        color: #7c6af7;
        text-style: bold;
        margin-bottom: 1;
    }
    .label {
        color: #888888;
        margin-top: 1;
    }
    Input {
        background: #1a1a1a;
        border: solid #333;
        color: #e0e0e0;
        margin-bottom: 1;
    }
    Input:focus {
        border: solid #7c6af7;
    }
    Button {
        margin-top: 1;
        width: 100%;
    }
    #btn-probe {
        background: #2a2a4a;
        color: #7c6af7;
        border: solid #7c6af7;
    }
    #btn-probe:hover {
        background: #3a3a6a;
    }
    #btn-start {
        background: #1a4a1a;
        color: #4af74a;
        border: solid #4af74a;
    }
    #btn-start:hover {
        background: #2a6a2a;
    }
    #btn-start:disabled {
        background: #1a1a1a;
        color: #444;
        border: solid #333;
    }
    #btn-stop {
        background: #4a1a1a;
        color: #f74a4a;
        border: solid #f74a4a;
    }
    Select {
        background: #1a1a1a;
        border: solid #333;
        margin-bottom: 1;
    }
    #status-bar {
        background: #1a1a1a;
        color: #7c6af7;
        padding: 0 2;
        height: 1;
        dock: bottom;
    }
    #log-panel {
        background: #0a0a0a;
        border: solid #2a2a2a;
        height: 12;
        margin-top: 1;
    }
    .info-text {
        color: #4af74a;
        margin-top: 1;
    }
    .warn-text {
        color: #f7a84a;
    }
    .muted {
        color: #555;
    }
    Rule {
        color: #2a2a2a;
        margin: 1 0;
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "Quit"),
        ("ctrl+p", "probe", "Probe"),
        ("ctrl+s", "start_bridge", "Start"),
    ]

    def __init__(self, port: int = 8000):
        super().__init__()
        self.port          = port
        self.proxy_thread  = None
        self.provider_data: ProviderSchema | None = None
        self.bridge_running = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static(BANNER, id="banner")
        yield Rule()

        with Horizontal(id="main"):
            # ── Left: Provider config ─────────────────────────────────────────
            with Vertical(id="left-panel"):
                yield Label("PROVIDER", classes="section-title")

                yield Label("Endpoint URL", classes="label")
                yield Input(
                    placeholder="https://integrate.api.nvidia.com/v1/chat/completions",
                    id="provider-url",
                )

                yield Label("API Key", classes="label")
                yield Input(placeholder="sk-... / nvapi-... / Bearer ...",
                            password=True, id="provider-key")

                yield Button("⚡ Probe Provider", id="btn-probe", variant="default")

                yield Rule()

                yield Label("Model", classes="section-title")
                yield Select(
                    options=[("— probe first —", "__none__")],
                    id="model-select",
                    allow_blank=False,
                )

                yield Rule()

                yield Label("CLIENT TOOL", classes="section-title")
                yield Static(
                    "Format learned automatically\nfrom first incoming request",
                    classes="muted",
                )
                yield Static("", id="client-info")

            # ── Right: Status + log ───────────────────────────────────────────
            with Vertical(id="right-panel"):
                yield Label("BRIDGE", classes="section-title")

                yield Label("Prism Port", classes="label")
                yield Input(str(self.port), id="proxy-port")

                yield Button("▶  Start Bridge", id="btn-start", disabled=True)
                yield Button("■  Stop", id="btn-stop", disabled=True)

                yield Rule()

                yield Label("CONNECT YOUR TOOL TO:", classes="section-title")
                yield Static(
                    f"ANTHROPIC_BASE_URL=http://localhost:{self.port}\n"
                    f"ANTHROPIC_API_KEY=prism",
                    id="connect-info",
                    classes="info-text",
                )

                yield Rule()

                yield Label("LOG", classes="section-title")
                yield Log(id="log-panel", auto_scroll=True)

        yield StatusBar(id="status-bar")
        yield Footer()

    def on_mount(self):
        self.query_one("#log-panel", Log).write_line("Prism ready. Configure provider and probe.")

    # ── Actions ───────────────────────────────────────────────────────────────

    def action_probe(self):
        self.on_button_pressed_probe()

    def action_start_bridge(self):
        self._start_bridge()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-probe":
            self._do_probe()
        elif event.button.id == "btn-start":
            self._start_bridge()
        elif event.button.id == "btn-stop":
            self._stop_bridge()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "proxy-port":
            try:
                self.port = int(event.value)
                self._update_connect_info()
            except ValueError:
                pass

    @work(exclusive=True, thread=True)
    def _do_probe(self):
        url = self.query_one("#provider-url", Input).value.strip()
        key = self.query_one("#provider-key", Input).value.strip()

        if not url:
            self.call_from_thread(self._log, "⚠ Enter a provider URL first")
            return

        self.call_from_thread(self._log, f"Probing {url}...")
        self.call_from_thread(self._set_status, "⟳ probing...")

        async def _probe():
            return await probe_provider(url, api_key=key or None)

        result = asyncio.run(_probe())
        self.provider_data = result

        # Update bridge
        bridge = get_bridge()
        bridge.provider            = result
        bridge.provider.api_key    = key or None
        bridge.provider.completion_url = _completion_url(url)

        self.call_from_thread(self._on_probe_done, result)

    def _on_probe_done(self, result: ProviderSchema):
        self._log(f"✓ Provider: format={result.format} models={len(result.models)} reachable={result.reachable}")

        # Populate model select
        sel = self.query_one("#model-select", Select)
        if result.models:
            options = [(m, m) for m in result.models]
            sel.set_options(options)
            sel.value = result.models[0]
            get_bridge().model = result.models[0]
            self._log(f"  {len(result.models)} models loaded")
        else:
            sel.set_options([("(no models found — type manually below)", "__none__")])
            self._log("  No models found via /models endpoint")

        self.query_one("#btn-start", Button).disabled = False
        self._set_status(f"◉ provider ready [{result.format}]")

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "model-select" and event.value != "__none__":
            get_bridge().model = str(event.value)
            self._log(f"Model selected: {event.value}")

    def _start_bridge(self):
        bridge = get_bridge()
        if not bridge.is_configured():
            self._log("⚠ Probe a provider and select a model first")
            return

        try:
            self.port = int(self.query_one("#proxy-port", Input).value)
        except ValueError:
            self._log("⚠ Invalid port number")
            return

        if self.proxy_thread and self.proxy_thread.is_alive():
            self._log("Bridge already running")
            return

        self.proxy_thread = ProxyThread(self.port)
        self.proxy_thread.start()
        bridge.mark_ready()
        self.bridge_running = True

        self.query_one("#btn-start", Button).disabled = True
        self.query_one("#btn-stop",  Button).disabled = False

        self._update_connect_info()
        self._log(f"▶ Bridge started on :{self.port}")
        self._log(f"  provider → {bridge.provider.format} @ {bridge.provider.completion_url}")
        self._log(f"  model    → {bridge.model}")
        self._log(f"  client format learned on first request")
        self._set_status(f"▶ running on :{self.port} → {bridge.model}")

    def _stop_bridge(self):
        if self.proxy_thread:
            self.proxy_thread.stop()
            self.proxy_thread = None

        self.bridge_running = False
        get_bridge().ready  = False

        self.query_one("#btn-start", Button).disabled = False
        self.query_one("#btn-stop",  Button).disabled = True

        self._log("■ Bridge stopped")
        self._set_status("◉ stopped")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        self.query_one("#log-panel", Log).write_line(msg)

    def _set_status(self, msg: str):
        self.query_one("#status-bar", StatusBar).status = msg

    def _update_connect_info(self):
        self.query_one("#connect-info", Static).update(
            f"ANTHROPIC_BASE_URL=http://localhost:{self.port}\n"
            f"ANTHROPIC_API_KEY=prism"
        )


def run_tui(port: int = 8000):
    app = PrismApp(port=port)
    app.run()
