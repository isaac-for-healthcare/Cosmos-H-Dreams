from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import ssl
import time
from pathlib import Path
from typing import Any

import cv2
import torch
from aiohttp import WSMsgType, web

from cosmosh.webrtc.config_loader import (
    build_runtime_config,
    get_server_settings,
    get_video_settings,
    get_vr_browser_settings,
    load_yaml_config,
)
from cosmosh.webrtc.session import CosmoshInferenceRuntime, CosmoshRuntimeConfig

WEB_DIR = Path(__file__).resolve().parent / "web"
LOGGER = logging.getLogger(__name__)

_PACING_LAG_LOG_MS = 5.0
"""Below this lag the per-frame push pacing re-anchors silently. Above
it the lag is worth a one-line warning so frame drops (the MJPEG sink
is latest-frame-wins, so a back-to-back push burst silently discards
all but the last frame in the chunk) are correlatable in the log."""


def _encode_frame_to_jpeg(
    frame_chw_neg1_pos1: torch.Tensor, jpeg_quality: int
) -> bytes | None:
    """Convert one ``[3, H, W]`` CPU tensor in ``[-1, 1]`` to JPEG bytes.

    Runs off the asyncio loop via :func:`asyncio.to_thread` so the cast +
    cv2 encode don't starve the per-frame pacing in
    :meth:`QuestSessionManager._push_chunk_to_sink`. Returns ``None`` if
    OpenCV fails to encode (rare; corrupt frame data).
    """
    rgb = ((frame_chw_neg1_pos1 * 127.5) + 127.5).clamp(0.0, 255.0).to(torch.uint8)
    rgb_hwc = rgb.permute(1, 2, 0).contiguous().numpy()
    # cv2.imencode wants BGR. Model output is RGB (PIL/torchvision
    # convention from ``_pixel_frame_to_neg1_pos1``).
    bgr = cv2.cvtColor(rgb_hwc, cv2.COLOR_RGB2BGR)
    ok, jpeg = cv2.imencode(
        ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    )
    return jpeg.tobytes() if ok else None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cosmosh Quest server: serves /quest_session, accepts 'vr_input' "
            "on /ws, streams generated frames on /video. All experiment "
            "parameters live in a YAML config (--config). CLI overrides "
            "are limited to runtime / deployment knobs."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help=(
            "Path to a YAML config (see cosmosh/webrtc/config_loader.py for "
            "the schema; examples in integrations/cosmosh/configs/)."
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
        help="Override server.port from the config. Default in config: 8443.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="DEBUG-level logging — per-event traces.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# MJPEG sink (unchanged from Phase 3a).
# ---------------------------------------------------------------------------


class MJPEGSink:
    """Latest-frame-wins JPEG queue feeding ``multipart/x-mixed-replace`` clients."""

    def __init__(self) -> None:
        self._latest: bytes | None = None
        self._frame_id: int = 0
        self._cond = asyncio.Condition()
        self._closed = False

    @property
    def latest_id(self) -> int:
        return self._frame_id

    async def push_jpeg(self, jpeg_bytes: bytes) -> None:
        async with self._cond:
            if self._closed:
                return
            self._latest = jpeg_bytes
            self._frame_id += 1
            self._cond.notify_all()

    async def wait_for_frame_after(
        self, last_id: int
    ) -> tuple[int, bytes] | None:
        async with self._cond:
            while not self._closed and self._frame_id <= last_id:
                await self._cond.wait()
            if self._latest is None or self._frame_id <= last_id:
                return None
            return self._frame_id, self._latest

    async def close(self) -> None:
        async with self._cond:
            self._closed = True
            self._cond.notify_all()


# ---------------------------------------------------------------------------
# Quest session manager — runtime + ws + sink + render loop.
# ---------------------------------------------------------------------------


class QuestSessionManager:
    """Owns the Cosmosh runtime, the active WebSocket, the MJPEG sink, and the render task.

    Single-session contract: at most one ws connected at a time; a new
    connection tears the old one down and resets the runtime. The render
    task starts after the runtime is preloaded and runs until shutdown —
    on idle (no ``vr_input`` since startup or last reset) it blocks on
    ``_first_action_event`` so we don't burn GPU on chunks the user hasn't
    asked for. The current iteration is **continuous** (the loop doesn't
    re-arm the event between chunks); ``--light`` parity can be added later.
    """

    def __init__(
        self,
        *,
        runtime_config: CosmoshRuntimeConfig,
        fps: int,
        jpeg_quality: int,
    ) -> None:
        self.runtime_config = runtime_config
        self.fps = fps
        self.jpeg_quality = jpeg_quality
        self._runtime = CosmoshInferenceRuntime(config=runtime_config)
        self._runtime_ready = False
        self._sink = MJPEGSink()
        self._ws: web.WebSocketResponse | None = None
        self._render_task: asyncio.Task[Any] | None = None
        # One lock serialises: render, reset, ws attach/detach. Keeps the
        # control flow trivially deadlock-free at the cost of waiting up to
        # one chunk's worth of (render + push) latency on reset / attach.
        self._render_lock = asyncio.Lock()
        # Set when a ``vr_input`` arrives; cleared on shutdown / reset /
        # ws detach. Blocks the render loop while idle.
        self._first_action_event = asyncio.Event()
        # ``time.monotonic()`` cutoff after which ``vr_input`` is accepted
        # again. Set on reset so the anchor frame stays in the sink long
        # enough to be visibly distinct from the chunk that the user's
        # in-flight controller motion would otherwise trigger immediately.
        self._reset_cooldown_until: float = 0.0
        self._reset_cooldown_s: float = 1.0
        self._closed = False

    @property
    def sink(self) -> MJPEGSink:
        return self._sink

    @property
    def runtime_ready(self) -> bool:
        return self._runtime_ready

    @property
    def has_active_ws(self) -> bool:
        return self._ws is not None and not self._ws.closed

    async def preload_runtime(self) -> None:
        """Load weights, push the conditional anchor frame, start the render task."""
        if self._runtime_ready:
            return
        LOGGER.info("Preloading Cosmosh runtime…")
        await self._runtime.initialize()
        self._runtime_ready = True
        LOGGER.info("Cosmosh runtime ready.")
        # /video clients connecting before any ws get the anchor frame.
        async with self._render_lock:
            await self._push_chunk_to_sink(self._runtime.initial_frame_chunk())
        self._render_task = asyncio.create_task(self._render_loop())

    async def attach_ws(self, ws: web.WebSocketResponse) -> None:
        """Take over from any previous ws. Does NOT reset runtime state.

        The browser auto-reconnects every 2 s after any drop (heartbeat
        hiccup, Wi-Fi blip, etc.), so an auto-reset here would blow away
        mid-session work each time the network sneezes. Reset is now a
        deliberate user action — Reset button on the page or hold-right-B
        in VR. The render loop's input gate is cleared so we don't keep
        generating from stale state across the reconnect window.
        """
        async with self._render_lock:
            previous_ws = self._ws
            self._ws = ws
            self._first_action_event.clear()
            self._reset_cooldown_until = 0.0
        if previous_ws is not None and not previous_ws.closed:
            LOGGER.info(
                "Kicking previous ws (single-session contract) — a new "
                "client connected. Old ws will be closed.",
            )
            with contextlib.suppress(Exception):
                await previous_ws.close()

    async def detach_ws(self, ws: web.WebSocketResponse) -> None:
        """Drop the ws reference if it's still the one we own.

        Doesn't acquire the render lock — we just clear the input event so
        the render loop idles after its current chunk finishes streaming.
        """
        if self._ws is ws:
            self._ws = None
            self._first_action_event.clear()

    async def handle_message(self, payload: dict[str, Any]) -> None:
        msg_type = payload.get("type")
        if msg_type == "vr_input":
            # Drop input silently during the post-reset cooldown so the
            # anchor frame stays visible. Otherwise the next 11 ms vr_input
            # would wake the render loop and overwrite the anchor before
            # the user could see it.
            if time.monotonic() < self._reset_cooldown_until:
                return
            LOGGER.debug("vr_input %s", payload)
            self._runtime.apply_vr_input(payload)
            self._first_action_event.set()
            return
        if msg_type == "reset":
            async with self._render_lock:
                self._first_action_event.clear()
                if self._runtime_ready:
                    await self._runtime.reset()
                    await self._push_chunk_to_sink(
                        self._runtime.initial_frame_chunk()
                    )
            self._reset_cooldown_until = (
                time.monotonic() + self._reset_cooldown_s
            )
            LOGGER.info(
                "Runtime reset on user request. vr_input ignored for %.1fs.",
                self._reset_cooldown_s,
            )
            return
        if msg_type == "session":
            LOGGER.info("session: %s", payload.get("action"))
            return
        LOGGER.warning("ws msg type=%r ignored", msg_type)

    async def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Wake the render loop so it can see ``_closed`` and exit cleanly.
        self._first_action_event.set()
        if self._render_task is not None and not self._render_task.done():
            self._render_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._render_task
            self._render_task = None
        await self._runtime.close()
        await self._sink.close()

    # ---- render loop ------------------------------------------------------

    async def _render_loop(self) -> None:
        LOGGER.info("Render loop started.")
        try:
            while not self._closed:
                try:
                    await self._first_action_event.wait()
                    if self._closed:
                        break
                    async with self._render_lock:
                        if self._closed:
                            break
                        result = await self._runtime.generate_one_chunk_vr()
                        LOGGER.info(
                            "Rendered VR chunk=%d num_frames=%d",
                            result.chunk_index,
                            result.num_frames,
                        )
                        await self._push_chunk_to_sink(result.video_chunk)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    LOGGER.exception(
                        "VR chunk render failed; sleeping 1s before retrying."
                    )
                    await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            pass
        finally:
            LOGGER.info("Render loop ended.")

    async def _push_chunk_to_sink(self, chunk: torch.Tensor) -> None:
        """Encode each frame of a ``[1, 3, T, H, W]`` ``[-1, 1]`` tensor and push at ``fps``.

        Pacing matters because the MJPEG sink is latest-frame-wins: pushing
        all 12 frames back-to-back would only show the last one. We sleep
        until each frame's wall-clock target so encode latency doesn't
        accumulate. For ``T == 1`` (the conditional anchor) this just pushes
        one frame and returns.
        """
        if chunk.ndim != 5 or chunk.shape[0] != 1 or chunk.shape[1] != 3:
            LOGGER.warning("Unexpected chunk shape %s; skipping push.", tuple(chunk.shape))
            return
        period = 1.0 / float(self.fps)
        t_frames = chunk.shape[2]
        start = time.monotonic()
        for f in range(t_frames):
            # Offload float->uint8 + cv2 JPEG encode to a worker thread.
            # On the asyncio loop this can dominate the 1/fps per-frame
            # budget and force the re-anchor branch below to keep firing.
            jpeg_bytes = await asyncio.to_thread(
                _encode_frame_to_jpeg, chunk[0, :, f], self.jpeg_quality
            )
            if jpeg_bytes is not None:
                await self._sink.push_jpeg(jpeg_bytes)
            target = start + (f + 1) * period
            now = time.monotonic()
            wait_s = target - now
            if wait_s > 0:
                await asyncio.sleep(wait_s)
            else:
                # Deadline is already in the past — encode + push took
                # longer than ``period``. Without re-anchoring, every
                # remaining frame's ``target`` is also in the past and
                # they all push back-to-back; the sink (latest-frame-wins)
                # then drops all but the last, so the spectator sees a
                # missing chunk worth of motion. Shift ``start`` forward
                # so the next frame's deadline is exactly ``now + period``.
                lag_ms = -wait_s * 1000.0
                if lag_ms > _PACING_LAG_LOG_MS:
                    LOGGER.warning(
                        "Quest push lag: f=%d deadline %.1fms behind walltime; "
                        "re-anchoring (MJPEG sink would otherwise drop frames).",
                        f,
                        lag_ms,
                    )
                start = now - (f + 1) * period


# ---------------------------------------------------------------------------
# Route handlers.
# ---------------------------------------------------------------------------


async def _ws_handler(request: web.Request) -> web.WebSocketResponse:
    manager: QuestSessionManager = request.app["manager"]
    ws = web.WebSocketResponse(heartbeat=20)
    await ws.prepare(request)
    peer = request.remote
    LOGGER.info("ws client connected: %s", peer)
    await manager.attach_ws(ws)
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    payload = json.loads(msg.data)
                except json.JSONDecodeError:
                    LOGGER.warning("invalid JSON from %s: %s", peer, msg.data[:120])
                    continue
                if not isinstance(payload, dict):
                    LOGGER.warning("payload not object from %s: %r", peer, payload)
                    continue
                await manager.handle_message(payload)
            elif msg.type == WSMsgType.ERROR:
                LOGGER.error("ws error from %s: %s", peer, ws.exception())
    finally:
        LOGGER.info("ws client disconnected: %s", peer)
        await manager.detach_ws(ws)
    return ws


async def _video_stream_handler(request: web.Request) -> web.StreamResponse:
    manager: QuestSessionManager = request.app["manager"]
    sink = manager.sink
    boundary = "frame"
    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": f"multipart/x-mixed-replace; boundary={boundary}",
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Connection": "close",
        },
    )
    await response.prepare(request)
    LOGGER.info("MJPEG client connected: %s", request.remote)
    last_id = 0
    try:
        while True:
            result = await sink.wait_for_frame_after(last_id)
            if result is None:
                break
            last_id, jpeg = result
            header = (
                f"--{boundary}\r\n"
                f"Content-Type: image/jpeg\r\n"
                f"Content-Length: {len(jpeg)}\r\n\r\n"
            ).encode("ascii")
            await response.write(header)
            await response.write(jpeg)
            await response.write(b"\r\n")
    except (asyncio.CancelledError, ConnectionResetError, ConnectionAbortedError):
        pass
    finally:
        LOGGER.info("MJPEG client disconnected: %s", request.remote)
    return response


# ---------------------------------------------------------------------------
# App wiring.
# ---------------------------------------------------------------------------


def create_app(
    *,
    manager: QuestSessionManager,
    vr_browser_settings: dict[str, Any] | None = None,
) -> web.Application:
    app = web.Application()
    app["manager"] = manager
    app["vr_browser_settings"] = vr_browser_settings or {}

    async def quest_page(_: web.Request) -> web.StreamResponse:
        return web.FileResponse(WEB_DIR / "quest_session.html")

    async def healthz(_: web.Request) -> web.StreamResponse:
        return web.json_response(
            {
                "status": "ok",
                "runtime_ready": manager.runtime_ready,
                "ws_active": manager.has_active_ws,
                "latest_frame_id": manager.sink.latest_id,
            }
        )

    async def vr_config(request: web.Request) -> web.StreamResponse:
        # Browser fetches this on page load to learn which input
        # semantics to apply. Server-side scale knobs aren't included —
        # they're applied after the wire payload lands on the server.
        return web.json_response(request.app["vr_browser_settings"])

    async def on_startup(app: web.Application) -> None:
        await app["manager"].preload_runtime()

    async def on_shutdown(app: web.Application) -> None:
        await app["manager"].shutdown()

    app.router.add_get("/quest_session", quest_page)
    app.router.add_get("/ws", _ws_handler)
    app.router.add_get("/video", _video_stream_handler)
    app.router.add_get("/healthz", healthz)
    app.router.add_get("/vr_config", vr_config)
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
        cert_path,
        key_path,
    )
    return None


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    cfg = load_yaml_config(args.config)
    runtime_config = build_runtime_config(cfg, role="quest")
    server_settings = get_server_settings(cfg)
    video_settings = get_video_settings(cfg)
    vr_browser_settings = get_vr_browser_settings(cfg)

    host = args.host or server_settings.get("host", "0.0.0.0")
    port = args.port or int(server_settings.get("port", 8443))
    cert_path = str(server_settings.get("cert", "cert.pem"))
    key_path = str(server_settings.get("key", "key.pem"))
    jpeg_quality = int(video_settings.get("jpeg_quality", 85))

    manager = QuestSessionManager(
        runtime_config=runtime_config,
        fps=runtime_config.fps,
        jpeg_quality=jpeg_quality,
    )
    ssl_ctx = _make_ssl_context(cert_path, key_path)
    scheme = "https" if ssl_ctx else "http"
    app = create_app(manager=manager, vr_browser_settings=vr_browser_settings)
    LOGGER.info(
        "VR input semantics: body_relative_translate=%s body_relative_rotation=%s",
        vr_browser_settings["body_relative_translate"],
        vr_browser_settings["body_relative_rotation"],
    )
    LOGGER.info(
        "Quest server listening on %s://%s:%d "
        "(page: /quest_session, ws: /ws, video: /video) — config: %s",
        scheme,
        host,
        port,
        args.config,
    )
    web.run_app(app, host=host, port=port, ssl_context=ssl_ctx)


if __name__ == "__main__":
    main()
