from __future__ import annotations

import asyncio
import contextlib
import datetime
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mediapy
import numpy as np
import torch

from cosmosh.utils import load_cr1_text_embeddings, pad_actions, pixel_frame_to_neg1_pos1
from cosmosh.webrtc.controls import (
    PSM1_GRIPPER_CLOSED,
    PSM1_GRIPPER_OPEN,
    PSM2_GRIPPER_CLOSED,
    PSM2_GRIPPER_OPEN,
    CosmoshActionIntegrator,
    KeyboardState,
)
from cosmosh.webrtc.controls_quest import (
    VRControllerState,
    compute_action_chunk,
)
from cosmosh.webrtc.media import CosmoshVideoTrack
from cosmosh.webrtc.utils import ACTION_DIM_NORMALISED
from cosmosh.config import COSMOSH_RUNNERS
from flashdreams.infra.config import derive_config

LOGGER = logging.getLogger(__name__)

# Wan2.1 VAE spatial compression ratio (pixel resolution -> latent grid).
WAN_SCR = 8

# ---------------------------------------------------------------------------
# E2E frame-rate profiling
# ---------------------------------------------------------------------------
_PROFILE_FPS_ENV = "COSMOSH_PROFILE_FPS"
_PROFILE_FPS_INTERVAL_S_ENV = "COSMOSH_PROFILE_FPS_INTERVAL_S"
_PROFILE_LATENCY_ENV = "COSMOSH_PROFILE_LATENCY"


def _fps_profile_enabled() -> bool:
    return os.environ.get(_PROFILE_FPS_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def _latency_profile_enabled() -> bool:
    return os.environ.get(_PROFILE_LATENCY_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def _fps_profile_interval_s() -> float:
    raw = os.environ.get(_PROFILE_FPS_INTERVAL_S_ENV, "5").strip()
    try:
        value = float(raw)
    except ValueError:
        return 5.0
    return max(0.5, value)


@dataclass(slots=True)
class _ChunkFpsProfile:
    """Per-session rolling-window FPS accumulator."""

    window_start: float | None = None
    frame_count: int = 0
    chunk_count: int = 0


def _fps_record_chunk(profile: _ChunkFpsProfile, num_frames: int) -> None:
    """Accumulate one chunk and log FPS when the reporting window elapses."""
    if not _fps_profile_enabled():
        return
    now = time.monotonic()
    if profile.window_start is None:
        profile.window_start = now
    profile.frame_count += num_frames
    profile.chunk_count += 1
    window_s = now - profile.window_start
    if window_s < _fps_profile_interval_s():
        return
    chunk_fps = profile.chunk_count / window_s if window_s > 1e-9 else 0.0
    frame_fps = profile.frame_count / window_s if window_s > 1e-9 else 0.0
    LOGGER.info(
        "[profile] fps chunk_fps=%.2f frame_fps=%.2f samples=%d",
        chunk_fps,
        frame_fps,
        profile.chunk_count,
    )
    profile.window_start = now
    profile.frame_count = 0
    profile.chunk_count = 0


def _fps_reset_profile(profile: _ChunkFpsProfile) -> None:
    profile.window_start = None
    profile.frame_count = 0
    profile.chunk_count = 0


# ---------------------------------------------------------------------------
# Per-session latency logger
# ---------------------------------------------------------------------------

class _LatencyLogger:
    """Per-session latency accumulator and JSONL file writer."""

    def __init__(self, mode: str) -> None:
        self._mode = mode
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = f"cosmosh_perf_{ts}.jsonl"
        self._file = open(self._path, "w", encoding="utf-8")
        self._block_records: list[dict] = []
        LOGGER.info("[latency] writing to %s", self._path)

    def log_block(self, record: dict) -> None:
        record = {"mode": self._mode, **record}
        self._file.write(json.dumps(record) + "\n")
        self._block_records.append(record)
        # Build [PERF] log line with only non-None numeric values
        parts = [f"block={record.get('block', '?')}"]
        for key in ("encode_ms", "diffuse_ms", "decode_ms", "finalize_ms", "d2h_ms",
                    "gap_ms", "cast_ms", "recv_wait_ms", "pacing_ms",
                    "jpeg_encode_ms", "mjpeg_drop_rate", "quest_pacing_ms", "input_age_ms"):
            val = record.get(key)
            if val is not None:
                parts.append(f"{key}={val:.2f}")
        LOGGER.info("[PERF] %s", " ".join(parts))

    def log_rollout_summary(self) -> None:
        if not self._block_records:
            return
        # Average every float key across blocks
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for rec in self._block_records:
            for key, val in rec.items():
                if isinstance(val, (int, float)) and key != "block":
                    sums[key] = sums.get(key, 0.0) + val
                    counts[key] = counts.get(key, 0) + 1
        avgs = {k: sums[k] / counts[k] for k in sums}
        summary = {"type": "rollout_summary", "mode": self._mode,
                   "num_blocks": len(self._block_records), **avgs}
        self._file.write(json.dumps(summary) + "\n")
        self._file.flush()
        avg_parts = [f"{k}={v:.2f}" for k, v in avgs.items() if k != "mode"]
        LOGGER.info("[PERF] rollout_summary blocks=%d %s", len(self._block_records), " ".join(avg_parts))
        self._block_records.clear()

    def close(self) -> None:
        try:
            self._file.close()
        except Exception:
            pass


# Default outer block size: 1 conditional + 12 generated pixel frames
# (= 4 latent frames). The 12 is overridable via
# ``CosmoshRuntimeConfig.actions_per_chunk`` — see that field's docstring
# for the divisibility constraint. ``PIXELS_PER_OUTER_BLOCK`` follows by
# adding the conditioning frame.
DEFAULT_ACTIONS_PER_OUTER_BLOCK = 12


class CosmoshRuntimeError(RuntimeError):
    """Raised when the Cosmosh runtime is used incorrectly."""


class SessionBusyError(RuntimeError):
    """Raised when a second peer tries to open a session."""


@dataclass(slots=True, frozen=True)
class Scene:
    """A switchable scene — the per-conditioning inputs the runtime can swap.

    Light-switch contract: ``config_name`` / ``ckpt_path`` / ``resolution`` /
    ``actions_per_chunk`` are model-level and shared across scenes (changing
    any of them would force a ``torch.compile`` recapture). Everything that
    can change cheaply lives on :class:`Scene`.
    """

    name: str
    input_path: str
    stats_path: str
    cr1_embeddings_path: str
    start_frame_idx: int = 0


@dataclass(slots=True)
class CosmoshRuntimeConfig:
    config_name: str = "cosmosh-lightvae-lighttae"
    compile_network: bool = True
    seed: int = 1
    device: str = "cuda:0"
    ckpt_path: str | None = None
    cr1_embeddings_path: str = ""
    # Path to either a video file (any mediapy-readable format) or a still
    # image. Image vs video is auto-detected from the file extension; when
    # an image is supplied, ``start_frame_idx`` is ignored.
    input_path: str = ""
    stats_path: str = ""
    start_frame_idx: int = 0
    # Debug only: path to a recorded actions ``.npy`` ([T, >=20]). When set, the
    # runtime IGNORES live keyboard / VR input and instead drives the model with
    # successive ``actions_per_chunk``-row slices of this file — the same inputs
    # the offline runner / replay use — so the browser shows the deterministic
    # rollout. Only the first ``ACTION_DIM_NORMALISED`` dims are used; the rest
    # are filled with the resting-neutral fill, exactly as the keyboard path
    # does. Pair it with ``input_path`` = the matching episode video so the
    # conditional first frame matches. ``None`` = normal live control.
    debug_action_npy: str | None = None
    # If None, the first frame's native (H, W) is used.
    resolution: tuple[int, int] | None = None
    fps: int = 10
    translate_v_per_frame: float = 0.3
    gripper_v_per_chunk: float = 1.0
    rotate_theta_per_frame: float = 0.017453292519943295  # 1° in radians
    # VR-only: per-arm, per-axis scale applied element-wise to play-space
    # ``dpos`` *before* the empirical camera-frame remap. Each arm's tuple
    # is ``(x, y, z)`` in play-space — index 1 is "vertical hand motion"
    # regardless of which camera-frame dim that maps onto. YAML accepts
    # scalar / 3-vec (broadcast across arms) or ``{right, left}`` dict
    # (asymmetric). Ignored by the keyboard path.
    translate_scale: dict[str, tuple[float, float, float]] = field(
        default_factory=lambda: {
            "right": (500.0, 500.0, 500.0),
            "left": (500.0, 500.0, 500.0),
        }
    )
    # Number of action / generated pixel frames per outer block. Default
    # matches the model's training setup (12). Must be a positive multiple
    # of the pipeline's ``num_action_per_latent_frame`` (typically 4 → valid
    # values 4, 8, 12, 16, ...). Validated at runtime init; an invalid
    # value raises ``CosmoshRuntimeError``. The model was trained with 12;
    # other values may produce degraded quality.
    actions_per_chunk: int = DEFAULT_ACTIONS_PER_OUTER_BLOCK
    # KV-cache rolling window in latent frames. ``None`` keeps the pipeline
    # config's default (``_WINDOW_SIZE_T = 11``). Must satisfy
    # ``(sink_size_t + window_size_t) % len_t == 0``; validated by the
    # transformer config's ``__post_init__`` at init time.
    window_size_t: int | None = None
    # VR-only: per-arm rotation scale. ``omega = drot × rotate_scale`` per
    # arm. ``drot`` is per-browser-frame axis-angle (radians); browser
    # frames are ~90 Hz vs output frames at ``fps`` (typically 10), so
    # ``rotate_scale=1.0`` ≈ keyboard's 1°/output-frame default. YAML
    # accepts scalar (broadcast across arms) or ``{right, left}`` dict
    # (asymmetric). Ignored by the keyboard path.
    rotate_scale: dict[str, float] = field(
        default_factory=lambda: {"right": 1.0, "left": 1.0}
    )


@dataclass(slots=True)
class CosmoshStepResult:
    chunk_index: int
    num_frames: int
    video_chunk: torch.Tensor  # [1, 3, 12, H, W] in [-1, 1] on CPU
    timing: dict[str, float] | None = None  # per-block profiler stats; None when profiling is off


# Common single-image extensions ``mediapy.read_image`` understands. Any
# other extension is treated as a video and read via ``mediapy.read_video``.
_IMAGE_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
)


def _load_conditional_frame(path: str, start_frame_idx: int) -> np.ndarray:
    """Load the conditional first frame from either a still image or a video.

    Image files (extension in :data:`_IMAGE_SUFFIXES`) are read directly via
    ``mediapy.read_image``; ``start_frame_idx`` is ignored. Anything else is
    treated as a video and indexed at ``start_frame_idx``. Returns ``[H, W, 3]``
    uint8.
    """
    suffix = Path(path).suffix.lower()
    if suffix in _IMAGE_SUFFIXES:
        frame = mediapy.read_image(path)
        if frame.ndim == 2:
            # Grayscale → broadcast to RGB.
            frame = np.stack([frame] * 3, axis=-1)
        if frame.ndim == 3 and frame.shape[-1] == 4:
            # Drop alpha channel.
            frame = frame[..., :3]
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(
                f"Image at {path} did not yield a [H, W, 3] frame; got shape "
                f"{frame.shape}"
            )
        return frame
    video = mediapy.read_video(path)
    if start_frame_idx < 0 or start_frame_idx >= video.shape[0]:
        raise ValueError(
            f"start_frame_idx={start_frame_idx} is out of range for video "
            f"with {video.shape[0]} frames at {path}"
        )
    return video[start_frame_idx]


# Gripper dims within the shared 20-dim prefix (arm-A at 9, arm-B at 19).
_PSM1_GRIPPER_IDX = 9
_PSM2_GRIPPER_IDX = 19


def _normalise_endpoint(
    raw: float, mean: float, std: float, *, fallback: float
) -> float:
    """Normalise a raw gripper percentile: ``(raw - mean) / std``.

    Falls back to ``fallback`` (the module-constant default) when ``std`` is
    non-finite or non-positive — guards against a degenerate / constant
    gripper dim (e.g. the all-zero energy dims in the 28-dim CMR stats).
    """
    if not np.isfinite(std) or std <= 0.0:
        return fallback
    return float((raw - mean) / std)


def _gripper_endpoints_from_stats(
    action: dict[str, Any], mean: np.ndarray, std: np.ndarray, stats_path: str
) -> dict[str, float]:
    """Derive per-arm ``[closed, open]`` gripper endpoints in normalised space.

    Closed = the low raw percentile (``q01``, falling back to ``min``); open =
    the high raw percentile (``q99``, falling back to ``max``). Each is
    normalised by the same ``(mean, std)`` the model was trained with, so the
    integrator clips the latched gripper to the dataset's observed range for
    the *active* model rather than to hardcoded dVRK constants. Falls back to
    the module-constant defaults when the percentile arrays are missing.
    """
    lo = action.get("q01") or action.get("min")
    hi = action.get("q99") or action.get("max")
    if lo is None or hi is None:
        LOGGER.warning(
            "stats file %s lacks q01/q99/min/max; using default gripper "
            "endpoints.",
            stats_path,
        )
        return {
            "psm1_gripper_closed": PSM1_GRIPPER_CLOSED,
            "psm1_gripper_open": PSM1_GRIPPER_OPEN,
            "psm2_gripper_closed": PSM2_GRIPPER_CLOSED,
            "psm2_gripper_open": PSM2_GRIPPER_OPEN,
        }
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    return {
        "psm1_gripper_closed": _normalise_endpoint(
            lo[_PSM1_GRIPPER_IDX],
            mean[_PSM1_GRIPPER_IDX],
            std[_PSM1_GRIPPER_IDX],
            fallback=PSM1_GRIPPER_CLOSED,
        ),
        "psm1_gripper_open": _normalise_endpoint(
            hi[_PSM1_GRIPPER_IDX],
            mean[_PSM1_GRIPPER_IDX],
            std[_PSM1_GRIPPER_IDX],
            fallback=PSM1_GRIPPER_OPEN,
        ),
        "psm2_gripper_closed": _normalise_endpoint(
            lo[_PSM2_GRIPPER_IDX],
            mean[_PSM2_GRIPPER_IDX],
            std[_PSM2_GRIPPER_IDX],
            fallback=PSM2_GRIPPER_CLOSED,
        ),
        "psm2_gripper_open": _normalise_endpoint(
            hi[_PSM2_GRIPPER_IDX],
            mean[_PSM2_GRIPPER_IDX],
            std[_PSM2_GRIPPER_IDX],
            fallback=PSM2_GRIPPER_OPEN,
        ),
    }


def _load_action_stats(stats_path: str) -> dict[str, Any]:
    """Load ``stats_cosmos*.json`` and return per-component normalisation slices.

    The file's combined ``"action"`` block holds raw-space mean/std/q01/q99
    over the dataset's action layout. The first 20 dims always follow the
    shared ``[arm-A xyz | arm-A rot6d | arm-A gripper | arm-B xyz |
    arm-B rot6d | arm-B gripper]`` convention — true of both the dVRK 20-dim
    layout (``stats_cosmos.json``) and the Open-H/CMR 28-dim layout
    (``stats_cosmos-28D-exp1.json``). Only the trailing dims differ (energy /
    thumbstick / buttons for CMR; pure zero-padding for dVRK), and the
    integrator doesn't drive those, so we only require the 20-dim prefix.

    Returns the per-arm rot6d mean/std (dims 3:9 / 13:19) for the rotation
    identity baseline, plus the per-arm gripper ``[closed, open]`` endpoints
    derived from dim 9 / 19's percentiles.
    """
    with open(stats_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    action = data.get("action") if isinstance(data, dict) else None
    if not isinstance(action, dict) or "mean" not in action or "std" not in action:
        raise ValueError(
            f"stats file missing 'action.mean' / 'action.std': {stats_path}"
        )
    mean = np.asarray(action["mean"], dtype=np.float64)
    std = np.asarray(action["std"], dtype=np.float64)
    if mean.shape != std.shape:
        raise ValueError(
            f"stats action.mean / action.std shape mismatch: "
            f"{mean.shape} vs {std.shape} in {stats_path}"
        )
    if mean.ndim != 1 or mean.shape[0] < 20:
        raise ValueError(
            f"stats action.mean/std must be 1-D with at least 20 dims (the "
            f"shared [xyz|rot6d|gripper] x2 prefix); got shape {mean.shape} in "
            f"{stats_path}. Supported layouts: 20-dim dVRK, 28-dim Open-H/CMR."
        )
    stats: dict[str, Any] = {
        "psm1_rot6d_mean": mean[3:9].copy(),
        "psm1_rot6d_std": std[3:9].copy(),
        "psm2_rot6d_mean": mean[13:19].copy(),
        "psm2_rot6d_std": std[13:19].copy(),
        # Full-width mean/std for building the resting-neutral fill of the
        # un-driven action dims (energy / thumbstick / buttons for CMR).
        "action_mean": mean.copy(),
        "action_std": std.copy(),
    }
    stats.update(_gripper_endpoints_from_stats(action, mean, std, stats_path))
    return stats


def _resting_neutral_fill(
    mean: np.ndarray, std: np.ndarray, action_dim: int
) -> np.ndarray:
    """Per-dim normalised value of a raw-``0`` (at-rest) action.

    The model consumes actions in mean-std normalised space, so filling an
    un-driven dim with ``0`` feeds the raw dataset *mean*, not a neutral input
    — badly off-centre for the CMR thumbstick dims (e.g. ``thumbstick_y_left``
    has raw mean ≈ -1956, so normalised 0 is a hard-deflected stick). The
    resting value of those controls is raw ``0`` (centred stick, unpressed
    button, zero energy), i.e. ``(0 - mean) / std = -mean / std``.

    Returns a length-``action_dim`` ``float64`` row. Dims with ``std <= 0``
    (degenerate / constant, e.g. the all-zero energy dims) and dims beyond the
    stats width (true zero-padding up to the network's ``action_dim``) are
    left at ``0``.
    """
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    row = np.zeros(action_dim, dtype=np.float64)
    n = min(mean.shape[0], action_dim)
    with np.errstate(divide="ignore", invalid="ignore"):
        row[:n] = np.where(std[:n] > 0.0, -mean[:n] / std[:n], 0.0)
    return row


def _fill_to_action_dim(driven: np.ndarray, neutral_row: np.ndarray) -> np.ndarray:
    """Widen a driven ``[T, D_driven]`` chunk to ``[T, len(neutral_row)]``.

    Starts every frame at ``neutral_row`` (the resting-neutral fill for the
    un-driven dims) and overwrites the leading ``D_driven`` dims with the
    integrator's output. For the dVRK 20-dim stats this is identical to
    zero-padding (dims 20+ of ``neutral_row`` are 0); for CMR it puts the
    thumbstick / button dims at their resting value instead of the raw mean.
    """
    n_frames, n_driven = driven.shape
    out = np.tile(neutral_row.astype(driven.dtype), (n_frames, 1))
    out[:, :n_driven] = driven
    return out


class CosmoshInferenceRuntime:
    """Single-session Cosmosh runtime with action-bound chunk generation."""

    def __init__(self, config: CosmoshRuntimeConfig | None = None) -> None:
        self.config = config or CosmoshRuntimeConfig()

        self.keyboard_state = KeyboardState()
        self.vr_state = VRControllerState()
        self.autoregressive_index = 0

        self._device: torch.device | None = None
        self._dtype: torch.dtype | None = None
        self._pipeline: Any | None = None
        self._action_target_dim: int = 0
        # Populated in _initialize_sync after validating against the model's
        # num_action_per_latent_frame. Exposed publicly for the render loop
        # so it can size its backpressure cap correctly.
        self.actions_per_chunk: int = self.config.actions_per_chunk

        # Action stats (populated in _initialize_sync) — passed to the
        # integrator so rotation is normalised against the dataset stats.
        self._psm1_rot6d_mean: np.ndarray | None = None
        self._psm1_rot6d_std: np.ndarray | None = None
        self._psm2_rot6d_mean: np.ndarray | None = None
        self._psm2_rot6d_std: np.ndarray | None = None

        # Per-arm gripper [closed, open] endpoints derived from the loaded
        # stats (q01/q99 → normalised); None until stats are loaded, in which
        # case the integrator keeps its module-constant defaults.
        self._psm1_gripper_open: float | None = None
        self._psm1_gripper_closed: float | None = None
        self._psm2_gripper_open: float | None = None
        self._psm2_gripper_closed: float | None = None

        # Full-width (= model action_dim) resting-neutral row used to fill the
        # action dims the integrator doesn't drive. None until stats + target
        # dim are known; falls back to zero-padding when unset.
        self._action_neutral_row: np.ndarray | None = None

        self._text_embeddings: torch.Tensor | None = None
        self._initial_cond_pixels: torch.Tensor | None = None
        self._cond_pixels: torch.Tensor | None = None

        # Flat-AR persistent state: the pipeline cache (which owns the KV +
        # decoder caches) is built on the first chunk of each rollout and
        # cleared on reset / scene-switch so the next chunk re-anchors.
        self._cache: Any | None = None
        # Flat AR step counter across all chunks in the current rollout.
        self._global_ar_idx: int = 0
        # Tracked VR arm positions for the Quest path (keyboard positions are
        # managed inside CosmoshActionIntegrator which resets on rebuild).
        self._vr_psm1_pos: np.ndarray = np.zeros(3, dtype=np.float64)
        self._vr_psm2_pos: np.ndarray = np.zeros(3, dtype=np.float64)

        # Debug action-override stream (set when config.debug_action_npy is
        # given): a ``[T, ACTION_DIM_NORMALISED]`` float32 array replayed in
        # ``actions_per_chunk``-row slices instead of live keyboard / VR input.
        self._debug_actions: np.ndarray | None = None
        self._debug_cursor: int = 0

        self._closed = False
        self._step_lock = asyncio.Lock()

        # Name of the currently-loaded scene (set by callers via
        # :meth:`set_active_scene_name` or :meth:`set_scene`). Independent of
        # ``config`` because the runtime is initialised with one scene's
        # fields but doesn't otherwise know about the scene list.
        self._active_scene_name: str | None = None

        # Built lazily after stats are loaded; defaults are safe enough for
        # __init__ time (no rotation, identity baseline).
        self.action_integrator = self._build_integrator()

    def _build_integrator(self) -> CosmoshActionIntegrator:
        """Construct the integrator using whatever stats have been loaded."""
        kwargs: dict[str, Any] = {
            "translate_v_per_frame": self.config.translate_v_per_frame,
            "gripper_v_per_chunk": self.config.gripper_v_per_chunk,
            "rotate_theta_per_frame": self.config.rotate_theta_per_frame,
        }
        if self._psm1_rot6d_mean is not None and self._psm1_rot6d_std is not None:
            kwargs["psm1_rot6d_mean"] = self._psm1_rot6d_mean
            kwargs["psm1_rot6d_std"] = self._psm1_rot6d_std
        if self._psm2_rot6d_mean is not None and self._psm2_rot6d_std is not None:
            kwargs["psm2_rot6d_mean"] = self._psm2_rot6d_mean
            kwargs["psm2_rot6d_std"] = self._psm2_rot6d_std
        if self._psm1_gripper_open is not None and self._psm1_gripper_closed is not None:
            kwargs["gripper_open_psm1"] = self._psm1_gripper_open
            kwargs["gripper_closed_psm1"] = self._psm1_gripper_closed
        if self._psm2_gripper_open is not None and self._psm2_gripper_closed is not None:
            kwargs["gripper_open_psm2"] = self._psm2_gripper_open
            kwargs["gripper_closed_psm2"] = self._psm2_gripper_closed
        return CosmoshActionIntegrator(**kwargs)

    async def initialize(self) -> None:
        if self._pipeline is not None:
            return
        await asyncio.to_thread(self._initialize_sync)

    async def reset_for_new_session(self) -> None:
        if self._closed:
            raise CosmoshRuntimeError("Runtime is closed.")
        if self._pipeline is None:
            raise CosmoshRuntimeError("Runtime is not initialized.")
        await asyncio.to_thread(self._reset_rollout_sync)

    async def reset(self) -> None:
        """In-session reset: clears keyboard / integrator / AR index and re-anchors
        on the initial conditional frame. Serialised against in-flight chunks via
        the step lock."""
        if self._closed:
            raise CosmoshRuntimeError("Runtime is closed.")
        if self._pipeline is None:
            raise CosmoshRuntimeError("Runtime is not initialized.")
        async with self._step_lock:
            await asyncio.to_thread(self._reset_rollout_sync)

    async def set_scene(self, scene: Scene) -> None:
        """Switch to a different scene without re-loading the model.

        Re-loads the conditional frame, action stats, and CR1 embeddings from
        the paths in ``scene``; rebuilds the action integrator; resets the
        rollout state so the next chunk starts from the new anchor frame. The
        model itself (pipeline / encoder / decoder / compiled CUDA graphs) is
        untouched — the switch takes ~one encoder pass plus the file reads,
        no ``torch.compile`` recapture.

        Serialised against in-flight chunks via the step lock. The new scene's
        files are validated before any state is mutated, so a typo can't
        leave the runtime half-switched.
        """
        if self._closed:
            raise CosmoshRuntimeError("Runtime is closed.")
        if self._pipeline is None:
            raise CosmoshRuntimeError("Runtime is not initialized.")
        async with self._step_lock:
            await asyncio.to_thread(self._set_scene_sync, scene)

    @property
    def active_scene_name(self) -> str | None:
        return self._active_scene_name

    @property
    def debug_action_override(self) -> bool:
        """Whether a recorded action stream is replacing live keyboard / VR input."""
        return self._debug_actions is not None

    def set_active_scene_name(self, name: str | None) -> None:
        """Record the initial scene's name (called once at startup by the server)."""
        self._active_scene_name = name

    def initial_frame_chunk(self) -> torch.Tensor:
        """Return the conditional anchor frame as ``[1, 3, 1, H, W]`` on CPU.

        The video track's ``enqueue_chunk`` accepts ``[B, C, T, H, W]``; with
        ``T=1`` it produces a single RGB frame the browser can display while
        the render loop is paused.
        """
        if self._initial_cond_pixels is None:
            raise CosmoshRuntimeError("Runtime is not initialized.")
        # ``_initial_cond_pixels`` is ``[1, 1, 3, H, W]``; permute to
        # ``[1, 3, 1, H, W]`` to match the per-block video-chunk layout.
        return (
            self._initial_cond_pixels.permute(0, 2, 1, 3, 4)
            .contiguous()
            .detach()
            .cpu()
        )

    async def close(self) -> None:
        self._closed = True
        await asyncio.to_thread(self._close_sync)

    async def apply_actions_and_generate(
        self, actions: list[dict[str, Any]]
    ) -> CosmoshStepResult:
        if self._closed:
            raise CosmoshRuntimeError("Session is closed.")
        if self._pipeline is None:
            raise CosmoshRuntimeError("Runtime is not initialized.")

        for action in actions:
            event = str(action.get("event", "keydown")).strip().lower()
            if event == "step":
                LOGGER.debug(
                    "Received step event with active_keys=%s",
                    sorted(self.keyboard_state.snapshot()),
                )
                continue
            raw_key = action.get("key", "")
            key = str(raw_key) if raw_key else ""
            if not key:
                raise CosmoshRuntimeError(
                    "Action payload must include non-empty 'key' for keydown/keyup."
                )

            applied = self.keyboard_state.apply_event(event=event, key=key)
            if not applied:
                raise CosmoshRuntimeError(
                    f"Unsupported action payload: event={event!r}, key={key!r}."
                )
            LOGGER.debug(
                "Applied control event=%s key=%s active_keys=%s "
                "psm1_t=%s psm1_r=%s psm2_t=%s psm2_r=%s",
                event,
                key,
                sorted(self.keyboard_state.snapshot()),
                sorted(self.keyboard_state.psm1_translate_keys()),
                sorted(self.keyboard_state.psm1_rotation_keys()),
                sorted(self.keyboard_state.psm2_translate_keys()),
                sorted(self.keyboard_state.psm2_rotation_keys()),
            )

        async with self._step_lock:
            if self._closed:
                raise CosmoshRuntimeError("Session is closed.")
            return await asyncio.to_thread(self._generate_one_chunk_sync)

    def apply_vr_input(self, payload: dict[str, Any]) -> bool:
        """Update :attr:`vr_state` from a ``vr_input`` payload (Phase 3 VR path).

        Synchronous + cheap: just delegates to :meth:`VRControllerState.apply_vr_input`,
        which replaces per-arm sub-state with fresh dataclass instances. Safe
        to call concurrently with the render loop — the integrator snapshots
        ``vr_state`` at chunk start and operates on the immutable arm
        instances captured at that moment.
        """
        if self._closed:
            return False
        return self.vr_state.apply_vr_input(payload, recv_t_ms=time.perf_counter() * 1000.0)

    async def generate_one_chunk_vr(self) -> CosmoshStepResult:
        """Render one outer block from the latest :class:`VRControllerState`.

        Serialised against keyboard / VR / reset paths via the step lock so
        only one GPU pipeline call is in flight at a time.
        """
        if self._closed:
            raise CosmoshRuntimeError("Session is closed.")
        if self._pipeline is None:
            raise CosmoshRuntimeError("Runtime is not initialized.")
        async with self._step_lock:
            if self._closed:
                raise CosmoshRuntimeError("Session is closed.")
            return await asyncio.to_thread(self._generate_one_chunk_vr_sync)

    async def generate_one_chunk_debug(self) -> CosmoshStepResult | None:
        """Render one chunk from the recorded debug action stream.

        Returns ``None`` once the stream is exhausted. Serialised against the
        keyboard / VR / reset paths via the step lock, same as the live paths.
        """
        if self._closed:
            raise CosmoshRuntimeError("Session is closed.")
        if self._pipeline is None:
            raise CosmoshRuntimeError("Runtime is not initialized.")
        async with self._step_lock:
            if self._closed:
                raise CosmoshRuntimeError("Session is closed.")
            return await asyncio.to_thread(self._generate_one_chunk_debug_sync)

    @torch.inference_mode()
    def _generate_one_chunk_debug_sync(self) -> CosmoshStepResult | None:
        """Slice the next actions from the debug stream and render.

        Mirrors ``cosmosh.webrtc.replay`` exactly: the recorded actions go
        straight into :meth:`_render_chunk_from_actions`, bypassing the keyboard
        integrator — so any divergence from the live keyboard rollout isolates
        to the integrator / action-computation path, not the GPU pipeline.

        The cursor advances by the number of actions actually consumed by the
        pipeline (not by ``actions_per_chunk``): for len_t > 1 AR step 0 needs
        fewer actions than ``actions_per_chunk``, so advancing by the full
        ``actions_per_chunk`` would silently skip recorded actions and feed
        wrong inputs to every subsequent AR step.
        """
        assert self._debug_actions is not None, "debug action stream not loaded"

        consumed = self._count_consumed_actions()
        if consumed == 0 or self._debug_cursor + consumed > self._debug_actions.shape[0]:
            return None  # stream exhausted

        driven = self._debug_actions[self._debug_cursor : self._debug_cursor + consumed]
        block = self._make_action_block(driven, consumed)
        self._debug_cursor += consumed
        return self._render_chunk_from_actions(block)

    def _initialize_sync(self) -> None:
        if self._pipeline is not None:
            return

        if not self.config.cr1_embeddings_path:
            raise CosmoshRuntimeError(
                "CosmoshRuntimeConfig.cr1_embeddings_path must be set."
            )
        if not self.config.input_path:
            raise CosmoshRuntimeError(
                "CosmoshRuntimeConfig.input_path must be set."
            )
        if not self.config.stats_path:
            raise CosmoshRuntimeError(
                "CosmoshRuntimeConfig.stats_path must be set "
                "(stats_cosmos.json — needed for rotation normalisation)."
            )
        if not Path(self.config.cr1_embeddings_path).exists():
            raise FileNotFoundError(
                f"CR1 embeddings not found: {self.config.cr1_embeddings_path}"
            )
        if not Path(self.config.input_path).exists():
            raise FileNotFoundError(
                f"Input not found: {self.config.input_path}"
            )
        if not Path(self.config.stats_path).exists():
            raise FileNotFoundError(
                f"Action stats file not found: {self.config.stats_path}"
            )
        if self.config.config_name not in COSMOSH_RUNNERS:
            supported = ", ".join(sorted(COSMOSH_RUNNERS))
            raise ValueError(
                f"Unknown config_name={self.config.config_name!r}. Supported: {supported}"
            )

        self._device = torch.device(self.config.device)
        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for Cosmosh runtime.")

        # Read the conditional first frame from either an image or a video.
        # Size determines pipeline latent dims unless the user passed an
        # explicit resolution.
        cond_frame = _load_conditional_frame(
            self.config.input_path, self.config.start_frame_idx
        )
        native_h = int(cond_frame.shape[0])
        native_w = int(cond_frame.shape[1])
        if self.config.resolution is None:
            ht, wt = native_h, native_w
        else:
            ht, wt = self.config.resolution
        if ht % WAN_SCR != 0 or wt % WAN_SCR != 0:
            raise ValueError(
                f"Resolution ({ht}, {wt}) must be divisible by Wan VAE spatial "
                f"compression {WAN_SCR}."
            )

        runner_cfg = COSMOSH_RUNNERS[self.config.config_name]
        transformer_overrides: dict[str, Any] = {
            "height": ht // WAN_SCR,
            "width": wt // WAN_SCR,
            "compile_network": self.config.compile_network,
        }
        if self.config.window_size_t is not None:
            transformer_overrides["window_size_t"] = self.config.window_size_t
        if self.config.ckpt_path:
            transformer_overrides["checkpoint_path"] = self.config.ckpt_path
        pipeline_cfg = derive_config(
            runner_cfg.pipeline,
            diffusion_model=dict(
                seed=self.config.seed,
                transformer=transformer_overrides,
            ),
        )
        # derive_config mutates fields via setattr without calling __post_init__,
        # so _pT/_pH/_pW would reflect the literal's defaults rather than the
        # overridden height/width. Re-run it to refresh those derived fields.
        pipeline_cfg.diffusion_model.transformer.__post_init__()
        # Wire per-step CUDA-event profiling (encode/diffuse/decode/finalize)
        # when the env var is set. Adds one torch.cuda.synchronize() per step.
        pipeline_cfg.enable_sync_and_profile = _latency_profile_enabled()
        # The pipeline owns the VAE first-frame encoder + decoder, so there is
        # no separate encoder/decoder to set up here.
        self._pipeline = pipeline_cfg.setup().to(self._device).eval()

        transformer = self._pipeline.diffusion_model.transformer
        cfg = transformer.config
        self._dtype = cfg.dtype
        actions_per_latent = cfg.network.num_action_per_latent_frame
        requested = int(self.config.actions_per_chunk)
        if requested <= 0 or requested % actions_per_latent != 0:
            raise CosmoshRuntimeError(
                f"actions_per_chunk={requested} must be a positive multiple of "
                f"num_action_per_latent_frame={actions_per_latent} "
                f"(valid examples: {actions_per_latent}, {2*actions_per_latent}, "
                f"{3*actions_per_latent}, ...)."
            )
        self.actions_per_chunk = requested
        self._action_target_dim = int(cfg.network.action_dim)

        text_embeddings_cpu = load_cr1_text_embeddings(self.config.cr1_embeddings_path)
        self._text_embeddings = text_embeddings_cpu.to(
            device=self._device, dtype=self._dtype
        )

        stats = _load_action_stats(self.config.stats_path)
        self._psm1_rot6d_mean = stats["psm1_rot6d_mean"]
        self._psm1_rot6d_std = stats["psm1_rot6d_std"]
        self._psm2_rot6d_mean = stats["psm2_rot6d_mean"]
        self._psm2_rot6d_std = stats["psm2_rot6d_std"]
        self._psm1_gripper_open = stats["psm1_gripper_open"]
        self._psm1_gripper_closed = stats["psm1_gripper_closed"]
        self._psm2_gripper_open = stats["psm2_gripper_open"]
        self._psm2_gripper_closed = stats["psm2_gripper_closed"]
        self._action_neutral_row = _resting_neutral_fill(
            stats["action_mean"], stats["action_std"], self._action_target_dim
        )

        if (cond_frame.shape[0], cond_frame.shape[1]) != (ht, wt):
            cond_frame = mediapy.resize_image(cond_frame, (ht, wt))
        cond_pixels = pixel_frame_to_neg1_pos1(
            cond_frame, device=self._device, dtype=self._dtype
        )
        self._initial_cond_pixels = cond_pixels.clone()
        self._cond_pixels = cond_pixels

        # Debug action-override: load the recorded .npy and keep its driven
        # prefix; the render loop replays it instead of live keyboard / VR.
        if self.config.debug_action_npy:
            if not Path(self.config.debug_action_npy).exists():
                raise FileNotFoundError(
                    f"debug_action_npy not found: {self.config.debug_action_npy}"
                )
            debug_actions = np.load(self.config.debug_action_npy)
            if debug_actions.ndim != 2 or debug_actions.shape[1] < ACTION_DIM_NORMALISED:
                raise CosmoshRuntimeError(
                    "debug_action_npy must be [T, >="
                    f"{ACTION_DIM_NORMALISED}]; got shape {debug_actions.shape}."
                )
            self._debug_actions = debug_actions[:, :ACTION_DIM_NORMALISED].astype(
                np.float32
            )
            LOGGER.warning(
                "DEBUG action override active: replaying %d rows from %s "
                "(%d chunks of %d) — live keyboard / VR input is IGNORED.",
                self._debug_actions.shape[0],
                self.config.debug_action_npy,
                self._debug_actions.shape[0] // self.actions_per_chunk,
                self.actions_per_chunk,
            )

        self._reset_rollout_sync()

        LOGGER.info(
            "Cosmosh runtime initialized: config_name=%s resolution=%dx%d "
            "action_dim=%d actions_per_chunk=%d",
            self.config.config_name,
            ht,
            wt,
            self._action_target_dim,
            self.actions_per_chunk,
        )
        LOGGER.info(
            "Gripper endpoints from %s: psm1=[closed=%.3f, open=%.3f] "
            "psm2=[closed=%.3f, open=%.3f]",
            self.config.stats_path,
            self._psm1_gripper_closed,
            self._psm1_gripper_open,
            self._psm2_gripper_closed,
            self._psm2_gripper_open,
        )

    def _reset_rollout_sync(self) -> None:
        if self._pipeline is None:
            raise CosmoshRuntimeError("Runtime pipeline is not initialized.")
        if self._initial_cond_pixels is None:
            raise CosmoshRuntimeError("Runtime input state is not initialized.")

        self.keyboard_state = KeyboardState()
        self.vr_state = VRControllerState()
        self.action_integrator = self._build_integrator()
        self.autoregressive_index = 0
        self._cond_pixels = self._initial_cond_pixels.clone()
        # Drop the flat-AR cache; the next chunk re-encodes and reinitializes.
        self._cache = None
        self._global_ar_idx = 0
        self._vr_psm1_pos = np.zeros(3, dtype=np.float64)
        self._vr_psm2_pos = np.zeros(3, dtype=np.float64)
        # Rewind the debug action stream so a reset replays from the start.
        self._debug_cursor = 0

    def _set_scene_sync(self, scene: Scene) -> None:
        if self._pipeline is None:
            raise CosmoshRuntimeError("Runtime pipeline is not initialized.")
        if (
            self._initial_cond_pixels is None
            or self._device is None
            or self._dtype is None
        ):
            raise CosmoshRuntimeError("Runtime is not initialized.")

        # Validate paths before mutating any state so a typo can't leave the
        # runtime half-switched.
        for path, label in (
            (scene.input_path, "input_path"),
            (scene.stats_path, "stats_path"),
            (scene.cr1_embeddings_path, "cr1_embeddings_path"),
        ):
            if not path or not Path(path).exists():
                raise FileNotFoundError(
                    f"Scene {scene.name!r}: {label} not found: {path}"
                )

        # Resolution is fixed at startup (changing it would force a compile
        # recapture, which violates the light-switch contract). Take it from
        # the existing conditional pixels' shape — ``[1, 1, 3, H, W]``.
        ht = int(self._initial_cond_pixels.shape[-2])
        wt = int(self._initial_cond_pixels.shape[-1])

        cond_frame = _load_conditional_frame(scene.input_path, scene.start_frame_idx)
        if (cond_frame.shape[0], cond_frame.shape[1]) != (ht, wt):
            cond_frame = mediapy.resize_image(cond_frame, (ht, wt))
        cond_pixels = pixel_frame_to_neg1_pos1(
            cond_frame, device=self._device, dtype=self._dtype
        )

        text_embeddings_cpu = load_cr1_text_embeddings(scene.cr1_embeddings_path)
        text_embeddings = text_embeddings_cpu.to(
            device=self._device, dtype=self._dtype
        )

        stats = _load_action_stats(scene.stats_path)

        # Commit. (Above can raise; only here do we touch ``self``.)
        self._psm1_rot6d_mean = stats["psm1_rot6d_mean"]
        self._psm1_rot6d_std = stats["psm1_rot6d_std"]
        self._psm2_rot6d_mean = stats["psm2_rot6d_mean"]
        self._psm2_rot6d_std = stats["psm2_rot6d_std"]
        self._psm1_gripper_open = stats["psm1_gripper_open"]
        self._psm1_gripper_closed = stats["psm1_gripper_closed"]
        self._psm2_gripper_open = stats["psm2_gripper_open"]
        self._psm2_gripper_closed = stats["psm2_gripper_closed"]
        self._action_neutral_row = _resting_neutral_fill(
            stats["action_mean"], stats["action_std"], self._action_target_dim
        )
        self._text_embeddings = text_embeddings
        self._initial_cond_pixels = cond_pixels.clone()
        self._active_scene_name = scene.name

        self._reset_rollout_sync()

        LOGGER.info(
            "Switched scene to %r (input=%s stats=%s); gripper endpoints "
            "psm1=[closed=%.3f, open=%.3f] psm2=[closed=%.3f, open=%.3f].",
            scene.name,
            scene.input_path,
            scene.stats_path,
            self._psm1_gripper_closed,
            self._psm1_gripper_open,
            self._psm2_gripper_closed,
            self._psm2_gripper_open,
        )

    def _close_sync(self) -> None:
        pipeline = self._pipeline
        cache = self._cache
        self._pipeline = None
        self._cache = None
        self._text_embeddings = None
        self._initial_cond_pixels = None
        self._cond_pixels = None

        for obj in (pipeline, cache):
            if obj is not None:
                del obj

        if self._device is not None and self._device.type == "cuda":
            torch.cuda.synchronize(device=self._device)
            torch.cuda.empty_cache()

    def _count_consumed_actions(self) -> int:
        """How many actions the next :meth:`_render_chunk_from_actions` call will consume.

        Simulates the inner while-loop without touching pipeline state so the
        caller can size the action array to exactly what will be read.  For
        configs where AR step 0 needs fewer actions than ``actions_per_chunk``
        (e.g. chunk3: step 0 = 8, step ≥1 = 12) this avoids generating
        ``actions_per_chunk - consumed`` action frames that the pipeline will
        never see.
        """
        assert self._pipeline is not None
        ar_idx = self._global_ar_idx
        consumed = 0
        while True:
            need = self._pipeline.get_num_actions(ar_idx)
            if consumed + need > self.actions_per_chunk:
                break
            consumed += need
            ar_idx += 1
        return consumed

    def _make_action_block(self, driven: np.ndarray, consumed: int) -> np.ndarray:
        """Return an ``(actions_per_chunk, D)`` block for :meth:`_render_chunk_from_actions`.

        ``driven`` has shape ``(consumed, D_driven)`` and covers exactly the
        rows the pipeline will read.  Rows beyond ``consumed`` are zero-padded
        (they are never accessed by the pipeline).
        """
        if consumed == self.actions_per_chunk and driven.shape[0] == self.actions_per_chunk:
            return driven
        block = np.zeros((self.actions_per_chunk, driven.shape[1]), dtype=np.float32)
        block[:consumed] = driven
        return block

    @torch.inference_mode()
    def _generate_one_chunk_sync(self) -> CosmoshStepResult:
        # Variable grippers: hold the open/close key on each arm to step the
        # latched value toward the corresponding endpoint per generated chunk.
        self.action_integrator.step_psm1_gripper(
            self.keyboard_state.psm1_gripper_intent()
        )
        self.action_integrator.step_psm2_gripper(
            self.keyboard_state.psm2_gripper_intent()
        )

        psm1_translate = self.keyboard_state.psm1_translate_keys()
        psm1_rotation = self.keyboard_state.psm1_rotation_keys()
        psm2_translate = self.keyboard_state.psm2_translate_keys()
        psm2_rotation = self.keyboard_state.psm2_rotation_keys()

        consumed = self._count_consumed_actions()
        actions_driven = self.action_integrator.next_action_chunk(
            num_frames=consumed,
            psm1_translate_keys=psm1_translate,
            psm1_rotation_keys=psm1_rotation,
            psm2_translate_keys=psm2_translate,
            psm2_rotation_keys=psm2_rotation,
        )
        actions_np = self._make_action_block(actions_driven, consumed)

        LOGGER.debug(
            "Rendering chunk=%s consumed=%d/%d "
            "psm1_t=%s psm1_r=%s psm1_g=%.3f "
            "psm2_t=%s psm2_r=%s psm2_g=%.3f",
            self.autoregressive_index,
            consumed,
            self.actions_per_chunk,
            sorted(psm1_translate),
            sorted(psm1_rotation),
            self.action_integrator.latched_gripper_psm1,
            sorted(psm2_translate),
            sorted(psm2_rotation),
            self.action_integrator.latched_gripper_psm2,
        )

        return self._render_chunk_from_actions(actions_np)

    @torch.inference_mode()
    def _generate_one_chunk_vr_sync(self) -> CosmoshStepResult:
        """VR variant of :meth:`_generate_one_chunk_sync`.

        Snapshots the current :class:`VRControllerState` (latest-sample-wins
        at chunk boundary) and routes through
        :func:`cosmosh.webrtc.controls_quest.compute_action_chunk`, which
        writes PSM1 translate (xyz), PSM1 rotation (rot6d) and PSM1
        gripper. PSM2 slices stay at zero. GPU body is shared with the
        keyboard path via :meth:`_render_chunk_from_actions`.

        PSM1 rot6d stats live on the keyboard integrator (loaded from
        ``stats_cosmos.json`` at init); we reuse them rather than threading
        a second copy through the runtime.
        """
        state = self.vr_state
        consumed = self._count_consumed_actions()
        # psm1_pos / psm2_pos are mutated in place by compute_action_chunk so
        # the next outer block continues from the arm's current position.
        actions_driven = compute_action_chunk(
            state,
            num_frames=consumed,
            psm1_pos=self._vr_psm1_pos,
            psm2_pos=self._vr_psm2_pos,
            translate_scale=self.config.translate_scale,
            rotate_scale=self.config.rotate_scale,
            psm1_rot6d_mean=self.action_integrator.psm1_rot6d_mean,
            psm1_rot6d_std=self.action_integrator.psm1_rot6d_std,
            psm1_rot6d_identity_norm=self.action_integrator._psm1_identity_rot6d_norm,
            psm2_rot6d_mean=self.action_integrator.psm2_rot6d_mean,
            psm2_rot6d_std=self.action_integrator.psm2_rot6d_std,
            psm2_rot6d_identity_norm=self.action_integrator._psm2_identity_rot6d_norm,
            psm1_gripper_open=self.action_integrator.gripper_open_psm1,
            psm1_gripper_closed=self.action_integrator.gripper_closed_psm1,
            psm2_gripper_open=self.action_integrator.gripper_open_psm2,
            psm2_gripper_closed=self.action_integrator.gripper_closed_psm2,
        )
        actions_np = self._make_action_block(actions_driven, consumed)

        LOGGER.debug(
            "Rendering VR chunk=%s consumed=%d/%d "
            "right(dpos=%s drot=%s trigger=%.3f) "
            "left(dpos=%s drot=%s trigger=%.3f)",
            self.autoregressive_index,
            consumed,
            self.actions_per_chunk,
            state.right.dpos.tolist(),
            state.right.drot.tolist(),
            state.right.trigger,
            state.left.dpos.tolist(),
            state.left.drot.tolist(),
            state.left.trigger,
        )

        t_chunk_start_ms = time.perf_counter() * 1000.0
        result = self._render_chunk_from_actions(actions_np)
        if result.timing is not None and self.vr_state.t_ms > 0:  # t_ms is 0.0 until the first vr_input arrives
            result.timing["input_age_ms"] = t_chunk_start_ms - self.vr_state.t_ms
        return result

    def _render_chunk_from_actions(
        self, actions_np: np.ndarray
    ) -> CosmoshStepResult:
        """Shared flat-AR body for the keyboard and VR paths.

        Keyboard and VR paths differ only in how they compute ``actions_np``;
        from here it's identical work — pad to ``action_target_dim``, then spend
        this chunk's ``actions_per_chunk`` actions across as many flat-AR
        ``pipeline.generate`` steps as they cover. The pipeline owns the VAE
        first-frame encoder + decoder, so each ``generate`` call returns decoded
        pixels; this method just slices per-step action chunks and concatenates
        the results into ``[B=1, C=3, T, H, W]`` on CPU.

        Chunk 0 (``self._cache is None``) builds the per-rollout cache, which
        encodes the conditional first frame and seeds the KV + decoder caches.
        Later chunks reuse the live cache and continue the flat AR index.
        """
        if (
            self._pipeline is None
            or self._text_embeddings is None
            or self._cond_pixels is None
            or self._device is None
            or self._dtype is None
        ):
            raise CosmoshRuntimeError("Runtime is not initialized.")

        assert actions_np.shape == (self.actions_per_chunk, ACTION_DIM_NORMALISED)
        if self._action_neutral_row is not None:
            actions_np = _fill_to_action_dim(actions_np, self._action_neutral_row)
        else:
            actions_np = pad_actions(actions_np, target_dim=self._action_target_dim)
        actions_block = (
            torch.from_numpy(actions_np)
            .to(device=self._device, dtype=self._dtype)
            .unsqueeze(0)
        )  # [1, actions_per_chunk, action_dim]

        if self._cache is None:
            # The pipeline owns the VAE first-frame encoder + decoder:
            # ``initialize_cache`` encodes the conditional first frame internally
            # and seeds the KV + decoder caches.
            if _latency_profile_enabled() and self._device is not None and self._device.type == "cuda":
                torch.cuda.synchronize(self._device)
            _t0_init = time.perf_counter() if _latency_profile_enabled() else 0.0
            self._cache = self._pipeline.initialize_cache(
                text_embeddings=self._text_embeddings,
                image=self._cond_pixels,
            )
            if _latency_profile_enabled():
                if self._device is not None and self._device.type == "cuda":
                    torch.cuda.synchronize(self._device)
                LOGGER.info(
                    "[profile] latency initialize_cache %.2f ms",
                    (time.perf_counter() - _t0_init) * 1000.0,
                )

        # Spend this chunk's action budget across as many flat-AR steps as it
        # covers, slicing each step's chunk via the pipeline's action sizing.
        # ``generate`` returns decoded pixels ``[1, T_pix, 3, H, W]``.
        pixel_frames: list[torch.Tensor] = []
        local_offset = 0
        _block_timing: dict[str, float] = {}
        while True:
            ar_idx = self._global_ar_idx
            need = self._pipeline.get_num_actions(ar_idx)
            if local_offset + need > self.actions_per_chunk:
                break
            chunk = (
                actions_block[:, local_offset : local_offset + need]
                if need > 0
                else None
            )
            pixels = self._pipeline.generate(ar_idx, self._cache, actions=chunk)
            pixels = pixels.clamp(min=-1.0, max=1.0)
            stats = self._pipeline.finalize(ar_idx, self._cache)
            if stats:
                for key, val in stats.items():
                    _block_timing[key] = _block_timing.get(key, 0.0) + val
            local_offset += need
            self._global_ar_idx += 1
            # AR step 0's first decoded frame is the VAE reconstruction of the
            # conditional latent (already shown as the initial anchor); drop it.
            pixel_frames.append(pixels[:, 1:] if ar_idx == 0 else pixels)

        # [1, T_pix, 3, H, W] -> [1, 3, T_pix, H, W] for the video track.
        block_pixels = torch.cat(pixel_frames, dim=1)
        generated_b3thw = block_pixels.permute(0, 2, 1, 3, 4).contiguous()

        if _block_timing and self._device is not None and self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
            _t0_d2h = time.perf_counter()
            video_chunk = generated_b3thw.detach().cpu()
            _block_timing["d2h_ms"] = (time.perf_counter() - _t0_d2h) * 1000.0
        else:
            video_chunk = generated_b3thw.detach().cpu()

        result = CosmoshStepResult(
            chunk_index=self.autoregressive_index,
            num_frames=generated_b3thw.shape[2],
            video_chunk=video_chunk,
            timing=_block_timing if _block_timing else None,
        )
        self.autoregressive_index += 1
        return result


# Cap the video track's outstanding frame buffer at two outer blocks so a
# faster-than-playback render loop doesn't pile up frames (and doesn't run
# more than ~one chunk ahead of what the user is seeing). Sized against the
# default chunk; if a config picks an unusually large chunk it can clamp to
# slightly less than 2× chunks of buffer, which is acceptable for v1.
_MAX_BUFFERED_FRAMES = 2 * DEFAULT_ACTIONS_PER_OUTER_BLOCK
# Polling interval used by the render loop while waiting for the video track
# queue to drain below the cap.
_BACKPRESSURE_POLL_S = 0.05
# How long ``close()`` will wait for an in-flight chunk to finish before
# falling back to ``Task.cancel()``.
_CLOSE_TIMEOUT_S = 30.0


@dataclass(slots=True)
class _ManagedCosmoshSession:
    runtime: CosmoshInferenceRuntime
    video_track: CosmoshVideoTrack
    peer_connection: Any
    control_channel: Any | None = None
    render_task: asyncio.Task[Any] | None = None
    render_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # Set by ``_handle_datachannel_message`` whenever an action arrives. The
    # render loop blocks on it before its first iteration (so the model isn't
    # drifting on idle chunks before the user has done anything). In
    # ``light_mode`` it's also cleared after each chunk if no keys remain held
    # and no events are queued — that's what makes light mode "render only on
    # input" instead of continuous.
    first_action_event: asyncio.Event = field(default_factory=asyncio.Event)
    pending_actions: list[dict[str, Any]] = field(default_factory=list)
    # When True, the render loop idles whenever no input is active. When False
    # (default), the loop runs continuously after the first user action.
    light_mode: bool = False
    closed: bool = False
    fps_profile: _ChunkFpsProfile = field(default_factory=_ChunkFpsProfile)

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        # Unblock the render loop if it's still waiting on the first user
        # action; otherwise close() would idle until the timeout fires.
        self.first_action_event.set()

        # Setting ``closed`` makes the render loop exit at its next checkpoint;
        # we let the in-flight chunk finish so we don't leave the GPU thread
        # mutating runtime state after we return. If the loop overruns, fall
        # back to a hard cancel.
        if self.render_task is not None and not self.render_task.done():
            try:
                await asyncio.wait_for(self.render_task, timeout=_CLOSE_TIMEOUT_S)
            except asyncio.TimeoutError:
                self.render_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self.render_task
            except asyncio.CancelledError:
                pass
            self.render_task = None
        self.pending_actions.clear()

        await self.video_track.close()
        await self.peer_connection.close()


class CosmoshWebRTCSessionManager:
    """Owns one active WebRTC session and forwards actions into the Cosmosh runtime."""

    def __init__(
        self,
        *,
        runtime_config: CosmoshRuntimeConfig | None = None,
        fps: int = 10,
        light_mode: bool = False,
        scenes: list[Scene] | None = None,
        runtime: CosmoshInferenceRuntime | None = None,
        events_broadcaster: Any | None = None,
    ) -> None:
        self.runtime_config = runtime_config or CosmoshRuntimeConfig()
        self.fps = fps
        self.light_mode = light_mode
        self.scenes: list[Scene] = list(scenes) if scenes else []
        self._scenes_by_name: dict[str, Scene] = {s.name: s for s in self.scenes}
        # ``runtime`` lets the unified server share one runtime across the
        # keyboard and Quest managers; otherwise we own a private instance.
        self._runtime = runtime or CosmoshInferenceRuntime(config=self.runtime_config)
        if self.scenes and self._runtime.active_scene_name is None:
            self._runtime.set_active_scene_name(self.scenes[0].name)
        self._runtime_ready = False
        self._active_session: _ManagedCosmoshSession | None = None
        self._session_lock = asyncio.Lock()
        # Optional async hook called immediately before a new keyboard session
        # is accepted. The unified server uses this to close any active Quest
        # ws so only one driver is touching the shared runtime at a time.
        self.on_take_over: Callable[[], Awaitable[None]] | None = None
        # Optional event broadcaster (Quest's ``_ViewerEventBroadcaster``)
        # the unified server passes in so the admin panel can show which
        # side is driving. We only require a ``publish(type, message)``
        # method, so the type is intentionally loose.
        self.events_broadcaster = events_broadcaster

    def has_active_session(self) -> bool:
        return self._active_session is not None and not self._active_session.closed

    def is_runtime_ready(self) -> bool:
        return self._runtime_ready

    def get_scene(self, name: str) -> Scene | None:
        return self._scenes_by_name.get(name)

    async def preload_runtime(self) -> None:
        if self._runtime_ready:
            return
        await self._runtime.initialize()
        self._runtime_ready = True

    async def create_answer(self, *, offer_sdp: str, offer_type: str) -> dict[str, str]:
        try:
            from aiortc import RTCPeerConnection, RTCSessionDescription
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "aiortc is required for WebRTC signaling. Install aiortc dependency."
            ) from exc

        async with self._session_lock:
            if self._active_session is not None and not self._active_session.closed:
                raise SessionBusyError("A Cosmosh session is already active.")

            # Takeover: in the unified server, a new keyboard session means
            # any active Quest ws should be dropped so only one driver is
            # touching the shared runtime at a time. No-op when running
            # keyboard-only.
            if self.on_take_over is not None:
                with contextlib.suppress(Exception):
                    await self.on_take_over()

            if not self._runtime_ready:
                await self._runtime.initialize()
                self._runtime_ready = True
            await self._runtime.reset_for_new_session()

            peer_connection = RTCPeerConnection()
            video_track = CosmoshVideoTrack(fps=self.fps)
            peer_connection.addTrack(video_track)
            managed_session = _ManagedCosmoshSession(
                runtime=self._runtime,
                video_track=video_track,
                peer_connection=peer_connection,
                light_mode=self.light_mode,
            )
            self._active_session = managed_session

            @peer_connection.on("datachannel")
            def on_datachannel(channel: Any) -> None:
                managed_session.control_channel = channel

                @channel.on("message")
                def on_message(message: Any) -> None:
                    asyncio.create_task(
                        self._handle_datachannel_message(
                            managed_session=managed_session,
                            raw_message=message,
                        )
                    )

            @peer_connection.on("connectionstatechange")
            async def on_connectionstatechange() -> None:
                state = peer_connection.connectionState
                if state == "connected" and managed_session.render_task is None:
                    managed_session.render_task = asyncio.create_task(
                        self._render_loop(managed_session=managed_session)
                    )
                    self._publish_driver("keyboard")
                if state in {"failed", "disconnected", "closed"}:
                    await self.close_active_session()

            try:
                offer = RTCSessionDescription(sdp=offer_sdp, type=offer_type)
                await peer_connection.setRemoteDescription(offer)
                answer = await peer_connection.createAnswer()
                await peer_connection.setLocalDescription(answer)
                local_description = peer_connection.localDescription
                if local_description is None:
                    raise RuntimeError(
                        "Peer connection did not produce local description."
                    )
                return {"sdp": local_description.sdp, "type": local_description.type}
            except Exception:
                LOGGER.exception("WebRTC negotiation failed while creating an answer.")
                await managed_session.close()
                self._active_session = None
                raise

    async def reset_active_session(self) -> bool:
        """Run a reset on the currently-active session, if any.

        Public wrapper around :meth:`_handle_reset` for the unified
        server's ``/admin/reset`` endpoint. Returns ``True`` if a reset
        was dispatched; ``False`` if no keyboard session is connected.
        """
        session = self._active_session
        if session is None or session.closed:
            return False
        await self._handle_reset(managed_session=session)
        return True

    async def close_active_session(self) -> None:
        had_active = False
        async with self._session_lock:
            if self._active_session is None:
                return
            had_active = True
            active_session = self._active_session
            self._active_session = None
            await active_session.close()
        if had_active:
            self._publish_driver("idle")

    def _publish_driver(self, name: str) -> None:
        """Broadcast a 'driver' event for the unified server's admin panel.

        ``name`` is one of ``"keyboard" / "quest" / "idle"``. Silent no-op
        when no broadcaster is attached (keyboard-only deployments).
        """
        broadcaster = self.events_broadcaster
        if broadcaster is None:
            return
        with contextlib.suppress(Exception):
            broadcaster.publish("driver", name)

    async def shutdown(self) -> None:
        await self.close_active_session()
        await self._runtime.close()
        self._runtime_ready = False

    async def _handle_datachannel_message(
        self,
        *,
        managed_session: _ManagedCosmoshSession,
        raw_message: Any,
    ) -> None:
        channel = managed_session.control_channel
        if channel is None or managed_session.closed:
            return

        if not isinstance(raw_message, str):
            self._send_json(
                channel, {"type": "error", "message": "Expected text payload."}
            )
            return

        try:
            payload = json.loads(raw_message)
        except json.JSONDecodeError:
            self._send_json(
                channel, {"type": "error", "message": "Invalid JSON payload."}
            )
            return

        if not isinstance(payload, dict):
            self._send_json(
                channel, {"type": "error", "message": "Payload must be a JSON object."}
            )
            return

        message_type = payload.get("type")
        if message_type == "reset":
            await self._handle_reset(managed_session=managed_session)
            return
        if message_type == "set_scene":
            await self._handle_set_scene(
                managed_session=managed_session, payload=payload
            )
            return
        if message_type == "latency_echo":
            if _latency_profile_enabled():
                LOGGER.info(
                    "[PERF] browser chunk_id=%s recv_to_raf_ms=%.1f",
                    payload.get("chunk_id"),
                    float(payload.get("recv_to_raf_ms", 0)),
                )
            return
        if message_type != "action":
            self._send_json(
                channel,
                {
                    "type": "error",
                    "message": (
                        "Unsupported message type, expected "
                        "'action', 'reset', or 'set_scene'."
                    ),
                },
            )
            return

        action_payload = payload.get("action", payload)
        if not isinstance(action_payload, dict):
            self._send_json(
                channel, {"type": "error", "message": "'action' must be an object."}
            )
            return

        # The render loop drains pending_actions at the top of each iteration
        # (under the render lock) — incoming events just update server state.
        managed_session.pending_actions.append(action_payload)
        # Unblock the render loop's first-action gate; subsequent calls are
        # no-ops since asyncio.Event stays set once flipped.
        managed_session.first_action_event.set()

    async def _handle_reset(
        self, *, managed_session: _ManagedCosmoshSession
    ) -> None:
        channel = managed_session.control_channel
        if channel is None or managed_session.closed:
            return

        # The render lock guarantees no chunk is in flight while we mutate
        # runtime state and drain the video track. The lock acquisition will
        # block at most one chunk's worth of inference time.
        try:
            async with managed_session.render_lock:
                managed_session.pending_actions.clear()
                await managed_session.runtime.reset()
                _fps_reset_profile(managed_session.fps_profile)
                dropped = managed_session.video_track.drain_pending()
                # Re-arm the input gate so the render loop pauses on its
                # next iteration, then push the conditional anchor frame so
                # the browser shows the starting pose instead of the last
                # generated frame from the pre-reset rollout.
                managed_session.first_action_event.clear()
                initial_chunk = managed_session.runtime.initial_frame_chunk()
                _ = await managed_session.video_track.enqueue_chunk(initial_chunk)
                # Debug-override: ``runtime.reset()`` rewound the action cursor,
                # so re-arm the loop to replay the recorded stream from the top.
                if managed_session.runtime.debug_action_override:
                    managed_session.first_action_event.set()
        except Exception as exc:
            LOGGER.exception("Cosmosh runtime reset failed.")
            self._send_json(channel, {"type": "error", "message": str(exc)})
            return

        LOGGER.info("Reset: cleared rollout state and %d pending frames.", dropped)
        self._send_json(
            channel,
            {"type": "reset_done", "dropped_frames": dropped},
        )

    async def _handle_set_scene(
        self,
        *,
        managed_session: _ManagedCosmoshSession,
        payload: dict[str, Any],
    ) -> None:
        channel = managed_session.control_channel
        if channel is None or managed_session.closed:
            return
        raw_name = payload.get("name")
        if not isinstance(raw_name, str) or not raw_name:
            self._send_json(
                channel,
                {"type": "error", "message": "set_scene requires non-empty 'name'."},
            )
            return
        scene = self.get_scene(raw_name)
        if scene is None:
            self._send_json(
                channel,
                {"type": "error", "message": f"Unknown scene: {raw_name!r}"},
            )
            return

        # Same render-lock contract as reset: drain pending actions and the
        # video track, swap the runtime's scene, push the new anchor frame.
        try:
            async with managed_session.render_lock:
                managed_session.pending_actions.clear()
                await managed_session.runtime.set_scene(scene)
                dropped = managed_session.video_track.drain_pending()
                managed_session.first_action_event.clear()
                initial_chunk = managed_session.runtime.initial_frame_chunk()
                _ = await managed_session.video_track.enqueue_chunk(initial_chunk)
        except Exception as exc:
            LOGGER.exception("Scene switch to %r failed.", raw_name)
            self._send_json(channel, {"type": "error", "message": str(exc)})
            return

        LOGGER.info(
            "Scene switched to %r; cleared %d pending frames.", scene.name, dropped
        )
        self._send_json(
            channel,
            {
                "type": "scene_set",
                "name": scene.name,
                "dropped_frames": dropped,
            },
        )

    async def _render_loop(
        self, *, managed_session: _ManagedCosmoshSession
    ) -> None:
        """Generate chunks while the session is connected.

        Pushes the conditional anchor frame to the video track immediately so
        the browser has something to display, then pauses on
        ``first_action_event`` until a user action arrives. Each iteration
        drains pending action events, runs one outer block under the render
        lock, enqueues the result, and emits ``chunk_done``. Backpressure is
        enforced outside the lock by polling the video track's queue size —
        when the buffer is at capacity, the loop sleeps and yields the lock
        so reset can interrupt.

        In default (continuous) mode the loop runs back-to-back once the first
        action arrives. In ``light_mode`` it additionally idles after each
        chunk if no keys remain held and no events are queued, waking up the
        next time the user submits an action.
        """
        channel = managed_session.control_channel
        latency_logger = _LatencyLogger("keyboard") if _latency_profile_enabled() else None
        _t_prev_block_end: float | None = None
        try:
            try:
                async with managed_session.render_lock:
                    initial_chunk = managed_session.runtime.initial_frame_chunk()
                    _ = await managed_session.video_track.enqueue_chunk(initial_chunk)
            except Exception:
                LOGGER.exception("Failed to enqueue initial conditional frame.")

            # Debug action-override: replay a recorded .npy continuously,
            # ignoring keyboard input. Lets the browser show the same
            # deterministic rollout the runner / replay produce.
            if managed_session.runtime.debug_action_override:
                await self._render_loop_debug(managed_session=managed_session)
                return

            while not managed_session.closed:
                # Pause until the next user action. Set on action append,
                # cleared on reset.
                await managed_session.first_action_event.wait()
                if managed_session.closed:
                    break

                if latency_logger is not None:
                    _t_iter_start = time.perf_counter() * 1000.0

                # Backpressure: don't run ahead of playback by more than the cap.
                while (
                    not managed_session.closed
                    and managed_session.video_track.qsize() >= _MAX_BUFFERED_FRAMES
                ):
                    await asyncio.sleep(_BACKPRESSURE_POLL_S)
                if managed_session.closed:
                    break

                actions = managed_session.pending_actions
                managed_session.pending_actions = []

                try:
                    async with managed_session.render_lock:
                        if managed_session.closed:
                            break
                        result = await managed_session.runtime.apply_actions_and_generate(
                            actions
                        )
                        enqueued, cast_ms = await managed_session.video_track.enqueue_chunk(
                            result.video_chunk
                        )
                except Exception as exc:
                    LOGGER.exception("Render loop chunk failed.")
                    self._send_json(channel, {"type": "error", "message": str(exc)})
                    break

                if latency_logger is not None:
                    _t_iter_end = time.perf_counter() * 1000.0
                    gap_ms = (_t_iter_start - _t_prev_block_end) if _t_prev_block_end is not None else None
                    record: dict[str, Any] = {"block": result.chunk_index, "gap_ms": gap_ms}
                    if result.timing:
                        record.update({k: v for k, v in result.timing.items()
                                       if k in ("encode_ms", "diffuse_ms", "decode_ms",
                                                "finalize_ms", "d2h_ms", "input_age_ms")})
                    recv_stats = managed_session.video_track.drain_recv_stats()
                    record["cast_ms"] = cast_ms
                    record["recv_wait_ms"] = recv_stats["recv_wait_ms"]
                    record["pacing_ms"] = recv_stats["pacing_ms"]
                    latency_logger.log_block(record)
                    _t_prev_block_end = _t_iter_end
                    if channel is not None:
                        self._send_json(channel, {
                            "type": "frame_ts",
                            "chunk_id": result.chunk_index,
                            "server_ms": time.perf_counter() * 1000.0,
                        })

                _fps_record_chunk(managed_session.fps_profile, result.num_frames)

                # Light mode: idle the loop until the next user action when
                # there's nothing live to render. Continuous mode leaves the
                # event set so the next iteration's ``await`` returns
                # immediately. Safe to mutate the event here without a lock —
                # the datachannel handler can only run at an ``await`` point,
                # and the next iteration's ``wait()`` is the next one.
                if (
                    managed_session.light_mode
                    and not managed_session.closed
                    and not managed_session.runtime.keyboard_state.pressed_keys
                    and not managed_session.pending_actions
                ):
                    managed_session.first_action_event.clear()

                LOGGER.debug(
                    "Rendered chunk=%s num_frames=%s enqueued=%s qsize=%s "
                    "light=%s",
                    result.chunk_index,
                    result.num_frames,
                    enqueued,
                    managed_session.video_track.qsize(),
                    managed_session.light_mode,
                )
                self._send_json(
                    channel,
                    {
                        "type": "chunk_done",
                        "chunk_index": result.chunk_index,
                        "num_frames": result.num_frames,
                        "enqueued_frames": enqueued,
                    },
                )
        except asyncio.CancelledError:
            pass
        finally:
            if latency_logger is not None:
                latency_logger.log_rollout_summary()
                latency_logger.close()

    async def _render_loop_debug(
        self, *, managed_session: _ManagedCosmoshSession
    ) -> None:
        """Render loop for debug action-override mode.

        Plays the recorded stream back-to-back without waiting on keyboard
        input, applying the same backpressure cap as the live loop. When the
        stream is exhausted it idles on ``first_action_event`` instead of
        exiting; a ``reset`` datachannel message rewinds the action cursor and
        re-arms the event (see :meth:`_handle_reset`), replaying from the top.
        The loop is armed immediately so the first play-through needs no reset.
        """
        channel = managed_session.control_channel
        try:
            # Arm immediately so playback starts without a keypress.
            managed_session.first_action_event.set()
            while not managed_session.closed:
                # Wait for the go / restart signal (set here, re-set by reset).
                await managed_session.first_action_event.wait()
                if managed_session.closed:
                    break

                exhausted = False
                while not managed_session.closed:
                    # Backpressure: don't run ahead of playback by the cap.
                    while (
                        not managed_session.closed
                        and managed_session.video_track.qsize() >= _MAX_BUFFERED_FRAMES
                    ):
                        await asyncio.sleep(_BACKPRESSURE_POLL_S)
                    if managed_session.closed:
                        break

                    try:
                        async with managed_session.render_lock:
                            if managed_session.closed:
                                break
                            result = (
                                await managed_session.runtime.generate_one_chunk_debug()
                            )
                            if result is None:
                                exhausted = True
                                break
                            enqueued, _ = await managed_session.video_track.enqueue_chunk(
                                result.video_chunk
                            )
                    except Exception as exc:
                        LOGGER.exception("Debug render loop chunk failed.")
                        self._send_json(channel, {"type": "error", "message": str(exc)})
                        return

                    _fps_record_chunk(managed_session.fps_profile, result.num_frames)

                    LOGGER.debug(
                        "Debug-rendered chunk=%s num_frames=%s enqueued=%s qsize=%s",
                        result.chunk_index,
                        result.num_frames,
                        enqueued,
                        managed_session.video_track.qsize(),
                    )
                    self._send_json(
                        channel,
                        {
                            "type": "chunk_done",
                            "chunk_index": result.chunk_index,
                            "num_frames": result.num_frames,
                            "enqueued_frames": enqueued,
                        },
                    )

                if exhausted and not managed_session.closed:
                    # Idle until a ``reset`` re-arms the event to replay.
                    managed_session.first_action_event.clear()
                    LOGGER.info(
                        "Debug action stream exhausted; send 'reset' to replay."
                    )
                    self._send_json(channel, {"type": "debug_stream_end"})
        except asyncio.CancelledError:
            return

    @staticmethod
    def _send_json(channel: Any, payload: dict[str, Any]) -> None:
        try:
            channel.send(json.dumps(payload))
        except Exception:
            return
