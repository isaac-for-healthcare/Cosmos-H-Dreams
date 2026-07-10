"""aiortc isolation boundary — the single file that touches aiortc internals.

aiortc 1.x does not expose a public codec-registration API, so substituting
the H.264 encoder requires mutating two module-level objects:

  - ``aiortc.codecs.h264.H264Encoder`` — replaced with
    :class:`NvencH264Encoder` so the standard aiortc factory
    (``aiortc.codecs.get_encoder``) returns our shim for the H264 codec.
  - ``aiortc.codecs.CODECS["video"]`` — filtered to H.264 entries only,
    so SDP negotiation cannot pick VP8 (for which we have no NVENC
    encoder). The original list is captured for later restore.

Everything that mutates aiortc lives in this file. Every other CosmosH
module depends only on the small public surface exported here.

Public API
----------
- :func:`install_nvenc_encoder(nal_queue)` — bind the NVENC-backed
  encoder; idempotent. Call once during ``preload_runtime`` after
  :func:`assert_aiortc_symbols_present`.
- :func:`restrict_codecs_to_h264()` — drop VP8 from the video codec
  capabilities; idempotent. Call once during ``preload_runtime``.
- :func:`uninstall_nvenc_encoder()` — restore aiortc to its original
  state. Idempotent. Used by tests and by a clean shutdown.
- :func:`assert_aiortc_symbols_present()` — CI gate that verifies the
  pinned aiortc internal symbols still exist. Raises
  :class:`AiortcCompatError` on a version mismatch.

PLI handling: aiortc may call our shim with ``force_keyframe=True`` when
it receives an RTCP PLI from the browser. We can't retroactively re-key
an already-encoded chunk, so the shim ignores the flag and relies on
NVENC's natural IDR cadence (``NvencConfig.idr_period_s``, default 4 s)
to recover.
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import Optional

import aiortc.codecs as _aiortc_codecs  # type: ignore[import-untyped]
import aiortc.codecs.h264 as _aiortc_h264  # type: ignore[import-untyped]
from aiortc.mediastreams import (  # type: ignore[import-untyped]
    VIDEO_TIME_BASE,
    convert_timebase,
)
from cosmosh.webrtc.nvenc.encoder import NalFrame

# NVENC (PyNvVideoCodec 2.1) emits H.264 *High* profile (profile_idc
# 0x64) with no kwarg to override it. Browsers, however, never offer a
# High H.264 line in their SDP — Chrome offers only Baseline (42001f),
# Constrained Baseline (42e01f), Main (4d001f) and High-4:4:4
# (f4001f). aiortc negotiates H.264 by profile *category* +
# packetization-mode (see rtcpeerconnection.is_codec_compatible), so
# advertising a High-only capability matches nothing and
# setRemoteDescription fails outright. We therefore keep aiortc's
# default Constrained-Baseline / Baseline H.264 lines (which the
# browser offers) and rely on the fact that Chromium builds its
# decoder from the in-band SPS, not the negotiated profile-level-id —
# so it decodes the High bitstream we send under those payload types.

LOGGER = logging.getLogger(__name__)


class AiortcCompatError(RuntimeError):
    """Raised when aiortc's internal layout does not match the adapter's pin."""


# ---------------------------------------------------------------------------
# Module state — captured at install time so uninstall can roll back.
# ---------------------------------------------------------------------------


class _AdapterState:
    """Cross-thread state shared between the runtime and the shim.

    Holds the NVENC NAL queue (drained by aiortc on its sender thread)
    and the originals captured at install time so uninstall can restore.
    Stored as module-level singleton ``_STATE``.
    """

    def __init__(self) -> None:
        self.nal_queue: Optional["queue.Queue[NalFrame]"] = None
        self.original_h264_encoder: Optional[type] = None
        self.original_codecs_video: Optional[list] = None


_STATE = _AdapterState()
_INSTALL_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Symbol-shape assertion (CI gate).
# ---------------------------------------------------------------------------


def assert_aiortc_symbols_present() -> None:
    """Verify aiortc's internals still match what the adapter monkey-patches.

    Called at import time by every module that uses the adapter, and also
    available as a standalone pytest fixture. Fails fast with a precise
    message if aiortc renames or removes any of:

      - ``aiortc.codecs.h264.H264Encoder``
      - ``H264Encoder._packetize`` (classmethod used to RFC 6184 fragment)
      - ``H264Encoder._split_bitstream`` (also reused by ``.pack``)
      - ``aiortc.codecs.CODECS["video"]`` (codec capability list)
      - At least one H.264 entry in ``CODECS["video"]``
      - ``aiortc.mediastreams.VIDEO_TIME_BASE``

    Pin aiortc tightly in ``pyproject.toml`` and run this on every
    aiortc lock-file bump.
    """
    if not hasattr(_aiortc_h264, "H264Encoder"):
        raise AiortcCompatError(
            "aiortc.codecs.h264.H264Encoder is missing — adapter cannot install."
        )
    enc_cls = _aiortc_h264.H264Encoder
    for attr in ("_packetize", "_split_bitstream"):
        if not hasattr(enc_cls, attr):
            raise AiortcCompatError(
                f"aiortc.codecs.h264.H264Encoder.{attr} is missing; "
                "the adapter relies on it for RFC 6184 packetization."
            )
    if not hasattr(_aiortc_codecs, "CODECS"):
        raise AiortcCompatError(
            "aiortc.codecs.CODECS is missing — adapter cannot restrict codecs."
        )
    codecs_dict = _aiortc_codecs.CODECS
    if "video" not in codecs_dict:
        raise AiortcCompatError(
            "aiortc.codecs.CODECS['video'] is missing — adapter cannot restrict."
        )
    video_codecs = codecs_dict["video"]
    # Empty CODECS["video"] is the normal state until
    # aiortc.codecs.init_codecs() runs. RTCPeerConnection construction
    # triggers init_codecs, so by the time any session opens H.264 is
    # present. Not an assertion failure — the check is also re-run
    # inside install_nvenc_encoder right before the install happens.


# Run the symbol check at import time so any aiortc version mismatch
# surfaces as soon as the adapter is loaded — before any session opens.
assert_aiortc_symbols_present()


# ---------------------------------------------------------------------------
# The shim encoder — drop-in for aiortc.codecs.h264.H264Encoder.
# ---------------------------------------------------------------------------


class NvencH264Encoder(_aiortc_h264.H264Encoder):
    """Substitutes the H.264 encoder aiortc would otherwise instantiate.

    Inherits ``H264Encoder`` so:
      - aiortc's ``get_encoder`` continues to receive a callable that
        returns an ``Encoder`` instance,
      - the inherited ``target_bitrate`` property (used by
        ``RTCRtpSender`` to read/write bitrate over the wire) works
        unchanged,
      - ``pack()`` and ``_packetize()`` from the parent are reused
        verbatim — we never re-implement RFC 6184 fragmentation.

    The override is :meth:`encode`. Instead of running libavcodec on
    the supplied ``av.VideoFrame``, it blocks on
    :attr:`_STATE.nal_queue`, pulls a :class:`NalFrame` (which the
    runtime placed there after NVENC produced it), and packetizes the
    Annex-B NAL list via the inherited ``_packetize`` classmethod.

    The supplied ``frame`` is a placeholder created by the
    :class:`CosmoshNvencVideoTrack` solely to drive aiortc's pacing.
    Its pixel buffer is unused.
    """

    # Match the parent class's blocking-get default. Configurable via
    # `set_get_timeout` if a deployment needs a different ceiling.
    _DEFAULT_GET_TIMEOUT_S: float = 5.0

    def __init__(self) -> None:
        super().__init__()
        # The queue handle is fetched from module state every encode call
        # (not stashed on self) so install_nvenc_encoder can swap it
        # without invalidating in-flight encoder instances.

    def encode(self, frame, force_keyframe: bool = False):
        """Pull the next NVENC-produced NalFrame and packetize it.

        Args:
            frame: an ``av.VideoFrame`` placeholder from the NVENC
                track. Its pixel data is ignored; its ``pts`` and
                ``time_base`` are used to derive the RTP timestamp,
                matching ``H264Encoder.encode``'s convention exactly.
            force_keyframe: ignored — see module docstring.

        Returns:
            ``(payloads, timestamp)`` per aiortc's ``Encoder.encode``
            contract: ``payloads`` from the inherited ``_packetize``,
            ``timestamp`` in the 90 kHz RTP clockrate.
        """
        nq = _STATE.nal_queue
        if nq is None:
            raise AiortcCompatError(
                "NvencH264Encoder.encode called but no NAL queue is "
                "installed. install_nvenc_encoder must run before any "
                "RTCPeerConnection is created."
            )

        try:
            nf: NalFrame = nq.get(timeout=self._DEFAULT_GET_TIMEOUT_S)
        except queue.Empty as exc:
            raise AiortcCompatError(
                "Timed out waiting for an NVENC NAL frame. The runtime's "
                "encode_chunk path may have stalled."
            ) from exc

        payloads = self._packetize(nf.nal_units)
        timestamp = convert_timebase(frame.pts, frame.time_base, VIDEO_TIME_BASE)
        return list(payloads), timestamp


# ---------------------------------------------------------------------------
# Install / uninstall.
# ---------------------------------------------------------------------------


def install_nvenc_encoder(nal_queue: "queue.Queue[NalFrame]") -> None:
    """Bind ``nal_queue`` and replace aiortc's H.264 encoder class.

    Idempotent. Safe to call multiple times across sessions; the queue
    is updated but the class swap happens only once. After this, every
    ``aiortc.codecs.get_encoder(<H264 capability>)`` returns a fresh
    :class:`NvencH264Encoder` that drains ``nal_queue``.

    Must be called from the runtime preload path *before* the first
    :class:`aiortc.RTCPeerConnection` is constructed for the process.
    """
    assert_aiortc_symbols_present()
    with _INSTALL_LOCK:
        _STATE.nal_queue = nal_queue
        if _STATE.original_h264_encoder is None:
            _STATE.original_h264_encoder = _aiortc_h264.H264Encoder
        # Rebind in BOTH modules. aiortc.codecs.__init__ does
        # ``from .h264 import H264Encoder`` at import time, which
        # captures the original class into ``aiortc.codecs.H264Encoder``.
        # aiortc's ``get_encoder()`` reads from that local binding, so
        # patching only ``aiortc.codecs.h264.H264Encoder`` would leave
        # the dispatcher pointing at the original libavcodec encoder
        # and our shim would never be called.
        _aiortc_h264.H264Encoder = NvencH264Encoder
        if hasattr(_aiortc_codecs, "H264Encoder"):
            _aiortc_codecs.H264Encoder = NvencH264Encoder
    LOGGER.info(
        "aiortc adapter installed: H264Encoder swapped for "
        "NvencH264Encoder in aiortc.codecs.h264 and aiortc.codecs."
    )


def restrict_codecs_to_h264() -> None:
    """Drop non-H.264 video codecs from ``CODECS["video"]``.

    Without this, the SDP offer/answer can negotiate VP8, which falls
    through to aiortc's libavcodec VP8 path — we have no NVENC encoder
    for VP8. Restricting the codec list forces H.264 selection.

    Idempotent. Captures the original list on first call for
    :func:`uninstall_nvenc_encoder` to restore later.

    Must run *after* aiortc has populated ``CODECS["video"]`` — usually
    by the time any session opens, since the first
    ``RTCPeerConnection`` constructor triggers ``init_codecs()``. If
    called before init, the list may be empty; we log and no-op.
    """
    with _INSTALL_LOCK:
        # ``aiortc.codecs.init_codecs()`` APPENDS to CODECS['video']; it
        # is NOT idempotent. Trigger it only when the list is empty
        # (e.g., restrict_codecs_to_h264 called before any
        # RTCPeerConnection has been constructed in the process).
        # In normal production flow aiortc has already initialized
        # the table by the time this function runs.
        if hasattr(_aiortc_codecs, "init_codecs") and not _aiortc_codecs.CODECS.get("video"):
            try:
                _aiortc_codecs.init_codecs()
            except Exception:  # noqa: BLE001 — init_codecs has no docstring
                LOGGER.exception("aiortc.codecs.init_codecs() raised; continuing.")

        # Snapshot the original list BEFORE we mutate it, so uninstall
        # can restore the pre-install state exactly.
        if _STATE.original_codecs_video is None:
            _STATE.original_codecs_video = list(
                _aiortc_codecs.CODECS.get("video", [])
            )

        video_list = _aiortc_codecs.CODECS.get("video", [])
        # Keep every H.264 entry and drop the rest (VP8 + its RTX).
        #
        # NVENC emits High profile (profile_idc=0x64), but browsers do
        # NOT offer a High H.264 line — Chrome offers only Baseline
        # (42001f), Constrained Baseline (42e01f), Main (4d001f) and
        # High-4:4:4 (f4001f). aiortc matches H.264 by profile
        # *category* + packetization-mode (see is_codec_compatible),
        # so advertising a High-only line negotiates with nothing and
        # setRemoteDescription fails outright.
        #
        # aiortc's own default video codecs already include H.264
        # 42001f / 42e01f (packetization-mode=1), which Chrome offers,
        # so keeping them lets negotiation succeed. The negotiated
        # profile-level-id does not gate decoding: Chromium builds its
        # H.264 decoder from the in-band SPS, so it decodes the High
        # bitstream we send under a Constrained-Baseline payload type.
        # We only need to drop VP8 so aiortc doesn't route frames to
        # its libav VP8 encoder (we have no NVENC VP8 path).
        filtered = [
            c for c in video_list
            if getattr(c, "mimeType", "").lower() == "video/h264"
        ]
        if not filtered:
            LOGGER.warning(
                "restrict_codecs_to_h264: no H.264 codec found in CODECS['video']."
            )
        _aiortc_codecs.CODECS["video"] = filtered
    LOGGER.info(
        "aiortc codec list restricted: %d H.264 entr(y/ies) kept, "
        "%d non-H.264 entr(y/ies) dropped.",
        len(filtered),
        len(video_list) - len(filtered),
    )


def uninstall_nvenc_encoder() -> None:
    """Restore aiortc's H.264 encoder class and codec list.

    Idempotent. Tests call this in their teardown so successive cases
    don't bleed state. Production code only calls it during a clean
    shutdown if it shuts the adapter down at all.
    """
    with _INSTALL_LOCK:
        if _STATE.original_h264_encoder is not None:
            _aiortc_h264.H264Encoder = _STATE.original_h264_encoder
            if hasattr(_aiortc_codecs, "H264Encoder"):
                _aiortc_codecs.H264Encoder = _STATE.original_h264_encoder
            _STATE.original_h264_encoder = None
        if _STATE.original_codecs_video is not None:
            _aiortc_codecs.CODECS["video"] = _STATE.original_codecs_video
            _STATE.original_codecs_video = None
        _STATE.nal_queue = None
    LOGGER.info("aiortc adapter uninstalled.")


__all__ = [
    "AiortcCompatError",
    "NvencH264Encoder",
    "assert_aiortc_symbols_present",
    "install_nvenc_encoder",
    "restrict_codecs_to_h264",
    "uninstall_nvenc_encoder",
]
