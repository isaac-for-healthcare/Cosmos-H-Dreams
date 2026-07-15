from __future__ import annotations

import argparse
import logging
import socket
from pathlib import Path

from aiohttp import web

from cosmosHDreams.webrtc.config_loader import (
    apply_encoder_settings,
    build_runtime_config,
    get_keyboard_settings,
    get_server_settings,
    load_yaml_config,
    parse_scenes,
)
from cosmosHDreams.webrtc.nvenc.resolver import (
    ENCODER_AUTO,
    ENCODER_CPU_LIBAV,
    ENCODER_NVENC,
)
from cosmosHDreams.webrtc.session import (
    CosmoshWebRTCSessionManager,
    SessionBusyError,
)

WEB_DIR = Path(__file__).resolve().parent / "web"
LOGGER = logging.getLogger(__name__)


def get_external_ip() -> str:
    """Get the external IP address of this machine.

    Uses a UDP socket trick to determine which interface would be used
    to reach an external address. No actual connection is made.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cosmosh WebRTC server: serves /keyboard and streams "
            "action-bound video chunks over a single peer connection. "
            "All experiment parameters live in a YAML config (--config); "
            "CLI overrides are limited to runtime / deployment knobs."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help=(
            "Path to a YAML config (see cosmosHDreams/webrtc/config_loader.py for "
            "the schema; examples in cosmosHDreams/configs/)."
        ),
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Override server.host from the config. Default in config: 0.0.0.0.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override server.port from the config. Default in config: 8080.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Enable DEBUG-level logging — adds per-event traces (each "
            "keydown/keyup, every chunk render). Default level is INFO."
        ),
    )
    parser.add_argument(
        "--encoder",
        choices=[ENCODER_AUTO, ENCODER_NVENC, ENCODER_CPU_LIBAV],
        default=None,
        help=(
            "Override the video encoder. Precedence: this flag > "
            "COSMOSH_VIDEO_ENCODER env > video.encoder in YAML > "
            "code default (cpu_libav). 'auto' picks NVENC when "
            "PyNvVideoCodec + CUDA are present, else falls back to "
            "cpu_libav. 'nvenc' hard-fails if NVENC is unavailable. "
            "'cpu_libav' forces the libavcodec path."
        ),
    )
    return parser.parse_args()


def create_app(
    *,
    session_manager: CosmoshWebRTCSessionManager | None = None,
) -> web.Application:
    manager = session_manager or CosmoshWebRTCSessionManager()
    app = web.Application()
    app["session_manager"] = manager

    async def keyboard_page(_: web.Request) -> web.StreamResponse:
        return web.FileResponse(WEB_DIR / "request_session.html")

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
            raise web.HTTPBadRequest(
                reason="Offer payload must include non-empty 'sdp'."
            )
        if not isinstance(offer_type, str) or not offer_type:
            raise web.HTTPBadRequest(
                reason="Offer payload must include non-empty 'type'."
            )

        manager = request.app["session_manager"]
        try:
            answer_payload = await manager.create_answer(
                offer_sdp=sdp,
                offer_type=offer_type,
            )
        except SessionBusyError as exc:
            raise web.HTTPConflict(reason=str(exc)) from exc
        except Exception as exc:
            LOGGER.exception("Failed to process WebRTC offer.")
            raise web.HTTPInternalServerError(reason=str(exc)) from exc

        return web.json_response(answer_payload)

    async def healthz(request: web.Request) -> web.StreamResponse:
        manager = request.app["session_manager"]
        return web.json_response(
            {
                "status": "ok",
                "runtime_ready": manager.is_runtime_ready(),
                "session_active": manager.has_active_session(),
            }
        )

    async def scenes_list(request: web.Request) -> web.StreamResponse:
        manager = request.app["session_manager"]
        return web.json_response(
            {
                "scenes": [
                    {"name": s.name, "start_frame_idx": s.start_frame_idx}
                    for s in manager.scenes
                ],
                "active": manager.scenes[0].name if manager.scenes else None,
            }
        )

    async def on_startup(app: web.Application) -> None:
        manager = app["session_manager"]
        LOGGER.info("Preloading Cosmosh runtime on startup.")
        await manager.preload_runtime()
        LOGGER.info("Cosmosh runtime preload complete.")

    async def on_shutdown(app: web.Application) -> None:
        manager = app["session_manager"]
        LOGGER.info("Shutting down Cosmosh runtime.")
        await manager.shutdown()

    app.router.add_get("/keyboard", keyboard_page)
    app.router.add_post("/api/webrtc/offer", offer)
    app.router.add_get("/healthz", healthz)
    app.router.add_get("/api/scenes", scenes_list)
    app.router.add_static("/static/", WEB_DIR, show_index=False)
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    return app


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    cfg = load_yaml_config(args.config)
    scenes = parse_scenes(cfg)
    runtime_config = build_runtime_config(cfg, role="keyboard", scenes=scenes)
    # Resolve encoder choice (CLI > env > YAML > code default) and apply
    # to runtime_config. Logs the resolution at INFO. Raises
    # NvencUnavailableError if the user explicitly asked for nvenc on
    # a host without NVENC; let it bubble up so the failure is loud.
    apply_encoder_settings(runtime_config, cfg, cli_value=args.encoder)
    server_settings = get_server_settings(cfg)
    keyboard_settings = get_keyboard_settings(cfg)

    host = args.host or server_settings.get("host", "0.0.0.0")
    port = args.port or int(server_settings.get("port", 8080))
    light_mode = bool(keyboard_settings.get("light_mode", False))

    session_manager = CosmoshWebRTCSessionManager(
        runtime_config=runtime_config,
        fps=runtime_config.fps,
        light_mode=light_mode,
        scenes=scenes,
    )
    LOGGER.info(
        "Scenes: %s (initial=%r)",
        [s.name for s in scenes],
        scenes[0].name,
    )
    if light_mode:
        LOGGER.info("Light mode enabled: rendering only while input is held.")
    app = create_app(session_manager=session_manager)
    print(f"Starting on external IP: {get_external_ip()}")
    LOGGER.info(
        "Keyboard server listening on http://%s:%d — config: %s",
        host,
        port,
        args.config,
    )
    web.run_app(app, host=host, port=port)


if __name__ == "__main__":
    main()
