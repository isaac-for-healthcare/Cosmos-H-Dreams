"""Unified Cosmosh server — keyboard + Quest on one port, one rollout.

The keyboard demo lives at ``/keyboard`` (WebRTC), the Quest demo at
``/quest`` (WebSocket + MJPEG). Both share a single
:class:`CosmoshInferenceRuntime`, so only one driver can be active at a
time: when a keyboard session opens, any active Quest ws is dropped, and
vice versa.

Run with::

    uv run --package flash-cosmosh python -m cosmosh.webrtc.server_unified \
      --config integrations/cosmosh/configs/unified_tabletop.yaml

YAML schema is the union of the keyboard and Quest schemas — see
:mod:`cosmosh.webrtc.config_loader` for fields. Missing sections fall back
to dataclass defaults. WebXR needs HTTPS, so the ``server.cert`` /
``server.key`` paths from the config are used the same way as the Quest
server.
"""

from __future__ import annotations

import argparse
import logging
import socket
import ssl
from pathlib import Path

from aiohttp import web

from cosmosh.webrtc.config_loader import (
    build_runtime_config_unified,
    get_keyboard_settings,
    get_server_settings,
    get_video_settings,
    get_vr_browser_settings,
    load_yaml_config,
    parse_scenes,
)
from cosmosh.webrtc.server_quest import (
    QuestSessionManager,
    _video_stream_handler,
    _viewer_events_handler,
    _ws_handler,
)
from cosmosh.webrtc.session import (
    CosmoshInferenceRuntime,
    CosmoshWebRTCSessionManager,
    SessionBusyError,
)

WEB_DIR = Path(__file__).resolve().parent / "web"
LOGGER = logging.getLogger(__name__)


def get_external_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cosmosh unified server: serves both the keyboard demo "
            "(/keyboard, WebRTC) and the Quest demo (/quest, ws+MJPEG) "
            "on the same port, sharing one rollout. Takeover semantics — "
            "whichever side connects most recently drives. WebXR requires "
            "HTTPS so cert/key must be supplied via the config."
        )
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to a unified-format YAML config. See server_unified.py "
             "docstring or config_loader.py for the schema.",
    )
    parser.add_argument(
        "--host", type=str, default=None,
        help="Override server.host from the config (default 0.0.0.0).",
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="Override server.port from the config (default 8443).",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="DEBUG-level logging (per-event traces).",
    )
    return parser.parse_args()


def create_app(
    *,
    kbd_manager: CosmoshWebRTCSessionManager,
    quest_manager: QuestSessionManager,
    vr_browser_settings: dict,
) -> web.Application:
    """Mount all keyboard + Quest routes on one aiohttp app.

    Both managers must already share a single runtime and have their
    ``on_take_over`` hooks wired before calling this; we only set up
    routing and lifecycle here.
    """
    app = web.Application()
    # Quest route handlers read ``manager`` from the app; keyboard offer
    # handler reads ``session_manager``. Keep both keys so we can reuse the
    # existing handlers unchanged.
    app["manager"] = quest_manager
    app["session_manager"] = kbd_manager
    app["vr_browser_settings"] = vr_browser_settings or {}

    # ---- Pages -----------------------------------------------------------
    async def index_page(_: web.Request) -> web.StreamResponse:
        # Tiny landing page that points at the two demo URLs. Saves anyone
        # at the booth from having to remember which path is which.
        return web.Response(
            content_type="text/html",
            text=(
                "<!doctype html><html><head><meta charset=utf-8>"
                "<title>Cosmosh demo</title>"
                "<style>body{font-family:sans-serif;max-width:40rem;"
                "margin:3rem auto;padding:0 1rem;line-height:1.5}"
                "a{display:block;margin:0.6rem 0}</style>"
                "</head><body><h1>Cosmosh demo</h1>"
                "<p>Pick a driver:</p>"
                "<a href='/keyboard'>Keyboard (browser)</a>"
                "<a href='/quest'>Quest (WebXR — open in Quest browser)</a>"
                "<a href='/viewer'>Spectator viewer</a>"
                "</body></html>"
            ),
        )

    async def keyboard_page(_: web.Request) -> web.StreamResponse:
        return web.FileResponse(WEB_DIR / "request_session.html")

    async def quest_page(_: web.Request) -> web.StreamResponse:
        return web.FileResponse(WEB_DIR / "quest.html")

    async def viewer_page(_: web.Request) -> web.StreamResponse:
        return web.FileResponse(WEB_DIR / "viewer.html")

    # ---- Keyboard WebRTC offer (copy of server.py's inline handler) ------
    async def offer(request: web.Request) -> web.StreamResponse:
        try:
            payload = await request.json()
        except Exception as exc:
            raise web.HTTPBadRequest(reason="Expected JSON offer payload.") from exc
        if not isinstance(payload, dict):
            raise web.HTTPBadRequest(reason="Offer payload must be a JSON object.")
        sdp = payload.get("sdp")
        offer_type = payload.get("type")
        if not isinstance(sdp, str) or not sdp:
            raise web.HTTPBadRequest(reason="Offer payload must include non-empty 'sdp'.")
        if not isinstance(offer_type, str) or not offer_type:
            raise web.HTTPBadRequest(reason="Offer payload must include non-empty 'type'.")
        try:
            answer_payload = await kbd_manager.create_answer(
                offer_sdp=sdp, offer_type=offer_type,
            )
        except SessionBusyError as exc:
            raise web.HTTPConflict(reason=str(exc)) from exc
        except Exception as exc:
            LOGGER.exception("Failed to process WebRTC offer.")
            raise web.HTTPInternalServerError(reason=str(exc)) from exc
        return web.json_response(answer_payload)

    # ---- Healthz / vr_config / scenes ------------------------------------
    async def healthz(_: web.Request) -> web.StreamResponse:
        return web.json_response(
            {
                "status": "ok",
                "runtime_ready": (
                    kbd_manager.is_runtime_ready() or quest_manager.runtime_ready
                ),
                "keyboard_active": kbd_manager.has_active_session(),
                "quest_ws_active": quest_manager.has_active_ws,
                "latest_frame_id": quest_manager.sink.latest_id,
            }
        )

    def _current_driver() -> str:
        # Used by /admin/status. The SSE 'driver' events are the live source
        # of truth once the page is up; this is just so the page renders
        # the correct badge before the first event lands.
        if quest_manager.has_active_ws:
            return "quest"
        if kbd_manager.has_active_session():
            return "keyboard"
        return "idle"

    async def admin_status(_: web.Request) -> web.StreamResponse:
        return web.json_response(
            {
                "driver": _current_driver(),
                "runtime_ready": (
                    kbd_manager.is_runtime_ready() or quest_manager.runtime_ready
                ),
                "active_scene": shared_runtime.active_scene_name,
            }
        )

    async def vr_config(request: web.Request) -> web.StreamResponse:
        return web.json_response(request.app["vr_browser_settings"])

    async def scenes_list(_: web.Request) -> web.StreamResponse:
        # Both managers carry the same scenes list; pick one as the source.
        scenes = quest_manager.scenes or kbd_manager.scenes
        return web.json_response(
            {
                "scenes": [
                    {"name": s.name, "start_frame_idx": s.start_frame_idx}
                    for s in scenes
                ],
                "active": scenes[0].name if scenes else None,
            }
        )

    # ---- Lifecycle -------------------------------------------------------
    async def on_startup(_: web.Application) -> None:
        # Quest manager's preload also pushes the anchor frame to its MJPEG
        # sink and starts the render task; run it first. The keyboard
        # manager's preload just flips its ready flag (init is idempotent).
        LOGGER.info("Preloading Cosmosh runtime (shared)…")
        await quest_manager.preload_runtime()
        await kbd_manager.preload_runtime()
        LOGGER.info("Cosmosh runtime ready.")

    async def on_shutdown(_: web.Application) -> None:
        # Order matters slightly: close keyboard session first (it owns a
        # WebRTC peer connection that should hang up cleanly), then the
        # Quest manager (which closes the runtime + MJPEG sink).
        LOGGER.info("Shutting down Cosmosh unified server.")
        await kbd_manager.shutdown()
        await quest_manager.shutdown()

    # ---- Routes ----------------------------------------------------------
    app.router.add_get("/", index_page)
    # Keyboard
    app.router.add_get("/keyboard", keyboard_page)
    app.router.add_post("/api/webrtc/offer", offer)
    app.router.add_get("/api/scenes", scenes_list)
    # Quest
    app.router.add_get("/quest", quest_page)
    app.router.add_get("/viewer", viewer_page)
    app.router.add_get("/ws", _ws_handler)
    app.router.add_get("/video", _video_stream_handler)
    app.router.add_get("/viewer_events", _viewer_events_handler)
    app.router.add_get("/vr_config", vr_config)
    app.router.add_get("/scenes", scenes_list)
    # Shared
    app.router.add_get("/healthz", healthz)
    app.router.add_get("/admin/status", admin_status)
    app.router.add_static("/static/", WEB_DIR, show_index=False)
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    return app


def _make_ssl_context(cert_path: str, key_path: str) -> ssl.SSLContext | None:
    cert = Path(cert_path)
    key = Path(key_path)
    if cert.exists() and key.exists():
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(cert, key)
        return ctx
    LOGGER.warning(
        "Cert or key not found (%s, %s) — serving plain HTTP. WebXR will "
        "refuse to start on non-localhost origins. Generate a cert with: "
        "openssl req -x509 -newkey rsa:2048 -nodes -keyout key.pem -out "
        "cert.pem -days 365 -subj '/CN=<your-ip>'",
        cert_path, key_path,
    )
    return None


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    cfg = load_yaml_config(args.config)
    scenes = parse_scenes(cfg)
    runtime_config = build_runtime_config_unified(cfg, scenes=scenes)
    server_settings = get_server_settings(cfg)
    video_settings = get_video_settings(cfg)
    vr_browser_settings = get_vr_browser_settings(cfg)
    keyboard_settings = get_keyboard_settings(cfg)

    host = args.host or server_settings.get("host", "0.0.0.0")
    port = args.port or int(server_settings.get("port", 8443))
    cert_path = str(server_settings.get("cert", "cert.pem"))
    key_path = str(server_settings.get("key", "key.pem"))
    jpeg_quality = int(video_settings.get("jpeg_quality", 85))
    light_mode = bool(keyboard_settings.get("light_mode", False))

    # One runtime; both managers point at it.
    shared_runtime = CosmoshInferenceRuntime(config=runtime_config)
    if scenes:
        shared_runtime.set_active_scene_name(scenes[0].name)

    quest_manager = QuestSessionManager(
        runtime_config=runtime_config,
        fps=runtime_config.fps,
        jpeg_quality=jpeg_quality,
        scenes=scenes,
        runtime=shared_runtime,
    )
    # Keyboard manager publishes to the Quest manager's broadcaster so the
    # admin / viewer page sees a unified event stream and can show which
    # side is driving.
    kbd_manager = CosmoshWebRTCSessionManager(
        runtime_config=runtime_config,
        fps=runtime_config.fps,
        light_mode=light_mode,
        scenes=scenes,
        runtime=shared_runtime,
        events_broadcaster=quest_manager.viewer_events,
    )

    # Cross-kick: a new driver on one side closes the other side's active
    # connection so the shared runtime only sees one input stream.
    kbd_manager.on_take_over = quest_manager.kick_active_ws
    quest_manager.on_take_over = kbd_manager.close_active_session

    # Seed an initial "driver: idle" so viewers that connect before the
    # first user attaches see the right state (the broadcaster's ring
    # replays the most recent events on subscribe).
    quest_manager.viewer_events.publish("driver", "idle")

    LOGGER.info(
        "Scenes: %s (initial=%r)",
        [s.name for s in scenes], scenes[0].name,
    )
    if light_mode:
        LOGGER.info("Keyboard light mode enabled.")

    ssl_ctx = _make_ssl_context(cert_path, key_path)
    scheme = "https" if ssl_ctx else "http"
    app = create_app(
        kbd_manager=kbd_manager,
        quest_manager=quest_manager,
        vr_browser_settings=vr_browser_settings,
    )
    print(f"Starting on external IP: {get_external_ip()}")
    LOGGER.info(
        "VR input semantics: body_relative_translate=%s body_relative_rotation=%s",
        vr_browser_settings["body_relative_translate"],
        vr_browser_settings["body_relative_rotation"],
    )
    LOGGER.info(
        "Unified server listening on %s://%s:%d "
        "(keyboard: /keyboard, headset: /quest, spectator: /viewer, "
        "landing: /) — config: %s",
        scheme, host, port, args.config,
    )
    web.run_app(app, host=host, port=port, ssl_context=ssl_ctx)


if __name__ == "__main__":
    main()
