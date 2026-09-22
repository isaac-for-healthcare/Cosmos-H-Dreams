# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""YAML config loader for both servers (keyboard + Quest).

One YAML schema covers both servers; each one only reads the sections it
cares about. Unknown keys log a warning so typos don't silently no-op.

Schema (full)::

    runtime:
      config_name: cosmosHDreams-lightvae-lighttae
      compile_network: true
      device: cuda:0
      seed: 1
      ckpt_path: null                       # null → default checkpoint
      resolution: [288, 512]                # or null for native
      fps: 10
      actions_per_chunk: 12                 # Must divide num_action_per_latent_frame
      window_size_t: 11                     # KV-cache rolling window (latent frames); null → pipeline default
      debug_action_npy: null                # Debug: replay this actions .npy,
                                            # ignoring keyboard/VR (see CosmoshRuntimeConfig)
      # Per-scene fields (input_path / stats_path / cr1_embeddings_path /
      # start_frame_idx) can live here as a backwards-compat shorthand for a
      # single 'default' scene. When ``scenes:`` is present they should live
      # there instead.

    # Optional. List of scenes the user can switch between from the UI.
    # Light switching only: the model (config_name / ckpt_path / resolution /
    # actions_per_chunk) is shared across all scenes — only the conditional
    # input, action stats, and CR1 embeddings change. If omitted, a single
    # 'default' scene is synthesised from the runtime: fields.
    scenes:
      - name: episode_001867
        input_path: /path/to/episode_001867.mp4
        stats_path: /path/to/stats_cosmos.json
        cr1_embeddings_path: /path/to/cr1.pt
        start_frame_idx: 0
      - name: chole
        input_path: /path/to/chole_frame0.png
        stats_path: /path/to/stats_cosmos_chole.json
        cr1_embeddings_path: /path/to/cr1.pt

    # Keyboard server only
    keyboard:
      translate_v_per_frame: 0.1
      gripper_v_per_chunk: 1.0
      rotate_deg_per_frame: 1.0
      light_mode: false

    # Quest server only
    vr:
      # Scalar or 3-element list [x, y, z] in PLAY-SPACE axes. Scalar
      # broadcasts to all three. Use the list form when one axis needs
      # different sensitivity (e.g. vertical hand → scene depth).
      translate_scale: 500.0
      # translate_scale: [500.0, 1500.0, 500.0]
      rotate_scale: 1.0
      # Browser-side input semantics. Both default to false (absolute
      # play-space input — the historical behaviour). Flip to true to
      # decouple from the user's body orientation.
      body_relative_translate: false
      body_relative_rotation: false
      # In-headset display panel (the floating "monitor"). Defaults
      # match the historical hardcoded quad.
      display:
        width_m: 0.4
        height_m: 0.3
        distance_m: 1.2

    # Quest server only
    video:
      jpeg_quality: 85

      # Keyboard server (WebRTC H.264) — opt-in NVENC acceleration.
      # Values: auto | nvenc | cpu_libav. Default: cpu_libav.
      # See cosmosHDreams.webrtc.nvenc.resolver for precedence and the auto probe.
      encoder: cpu_libav
      # NVENC-only sub-block. Ignored when encoder != nvenc.
      nvenc:
        preset: P3
        tuning: ultra_low_latency
        bitrate: 3000000        # bits per second
        idr_period_s: 4.0       # seconds between forced IDR frames

    # Both servers (cert/key only used by Quest)
    server:
      cert: cert.pem
      key: key.pem

CLI flags ``--host`` / ``--port`` / ``--debug`` override the corresponding
config values; everything else comes from the YAML.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import yaml

from cosmosHDreams.webrtc.nvenc.resolver import (
    ENCODER_CPU_LIBAV,
    VALID_ENCODERS,
    normalize_encoder_value,
    read_encoder_env,
    resolve_encoder,
    select_requested_encoder,
)
from cosmosHDreams.webrtc.session import CosmoshRuntimeConfig, Scene

LOGGER = logging.getLogger(__name__)

# Top-level keys recognised by the loader. Anything else triggers a warning
# (typo guard — silently-ignored keys are the worst kind of config bug).
_KNOWN_TOP_LEVEL = frozenset({"runtime", "keyboard", "vr", "video", "server", "scenes"})


class ConfigError(ValueError):
    """Raised on missing/invalid config fields."""


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML config from ``path`` and return it as a dict.

    Empty files load as ``{}``. Top-level keys outside :data:`_KNOWN_TOP_LEVEL`
    log a warning so typos surface during startup rather than as silently
    missing tuning.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(
            f"Top-level YAML must be a mapping; got {type(data).__name__}"
        )
    for key in data.keys():
        if key not in _KNOWN_TOP_LEVEL:
            LOGGER.warning(
                "Unknown top-level config key %r in %s — typo? Known keys: %s",
                key,
                p,
                sorted(_KNOWN_TOP_LEVEL),
            )
    return data


def _require(section: dict[str, Any], key: str, *, section_name: str) -> Any:
    if key not in section:
        raise ConfigError(f"Missing required field: {section_name}.{key}")
    return section[key]


def _parse_translate_scale_vec(value: Any, *, ctx: str) -> tuple[float, float, float]:
    """One arm's translate scale: scalar (broadcast) or 3-element list."""
    if isinstance(value, (int, float)):
        s = float(value)
        return (s, s, s)
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return (float(value[0]), float(value[1]), float(value[2]))
    raise ConfigError(f"{ctx} must be a scalar or 3-element list; got {value!r}")


def _parse_translate_scale(value: Any) -> dict[str, tuple[float, float, float]]:
    """Parse ``vr.translate_scale`` into a per-arm dict.

    Accepted shapes:
    - ``scalar`` or ``[x, y, z]`` → broadcast to both arms.
    - ``{right: <scalar|3-vec>, left: <scalar|3-vec>}`` → asymmetric.
      Both keys must be present (use scalar/list form if you want them equal).
    """
    if isinstance(value, dict):
        for required in ("right", "left"):
            if required not in value:
                raise ConfigError(
                    f"vr.translate_scale dict must include both 'right' and "
                    f"'left' keys; got {sorted(value.keys())}"
                )
        return {
            "right": _parse_translate_scale_vec(
                value["right"], ctx="vr.translate_scale.right"
            ),
            "left": _parse_translate_scale_vec(
                value["left"], ctx="vr.translate_scale.left"
            ),
        }
    vec = _parse_translate_scale_vec(value, ctx="vr.translate_scale")
    return {"right": vec, "left": vec}


def _parse_rotate_scale(value: Any) -> dict[str, float]:
    """Parse ``vr.rotate_scale`` into a per-arm dict.

    Accepted shapes:
    - scalar → broadcast to both arms.
    - ``{right: <scalar>, left: <scalar>}`` → asymmetric (both keys required).
    """
    if isinstance(value, dict):
        for required in ("right", "left"):
            if required not in value:
                raise ConfigError(
                    f"vr.rotate_scale dict must include both 'right' and "
                    f"'left' keys; got {sorted(value.keys())}"
                )
        return {"right": float(value["right"]), "left": float(value["left"])}
    return {"right": float(value), "left": float(value)}


def _parse_resolution(value: Any) -> tuple[int, int] | None:
    """Accept ``[H, W]`` list, ``"H,W"`` string, or null/None → native size."""
    if value is None:
        return None
    if isinstance(value, str):
        if value.lower() in {"", "none", "native"}:
            return None
        try:
            h, w = (int(x) for x in value.split(","))
            return (h, w)
        except ValueError as exc:
            raise ConfigError(
                f"resolution string must be 'H,W'; got {value!r}"
            ) from exc
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ConfigError(f"resolution must be [H, W], 'H,W', or null; got {value!r}")


def parse_scenes(cfg: dict[str, Any]) -> list[Scene]:
    """Parse the optional ``scenes:`` list into ordered :class:`Scene` entries.

    If ``scenes:`` is absent, synthesise a single ``"default"`` scene from
    the ``runtime:`` block's per-scene fields (``input_path``,
    ``stats_path``, ``cr1_embeddings_path``, ``start_frame_idx``). This keeps
    legacy single-scene configs working unchanged.

    Names must be unique. Anything except ``name`` / ``start_frame_idx``
    that's missing or empty is a hard error — the scene wouldn't be
    switchable-to without those files.
    """
    raw = cfg.get("scenes")
    if raw is None:
        runtime = cfg.get("runtime")
        if not isinstance(runtime, dict):
            raise ConfigError("Missing required section: runtime")
        return [
            Scene(
                name="default",
                input_path=str(_require(runtime, "input_path", section_name="runtime")),
                stats_path=str(_require(runtime, "stats_path", section_name="runtime")),
                cr1_embeddings_path=str(
                    _require(runtime, "cr1_embeddings_path", section_name="runtime")
                ),
                start_frame_idx=int(runtime.get("start_frame_idx", 0)),
            )
        ]

    if not isinstance(raw, list) or not raw:
        raise ConfigError("scenes: must be a non-empty list when present.")

    scenes: list[Scene] = []
    seen_names: set[str] = set()
    for idx, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ConfigError(f"scenes[{idx}] must be a mapping; got {entry!r}")
        name = str(_require(entry, "name", section_name=f"scenes[{idx}]")).strip()
        if not name:
            raise ConfigError(f"scenes[{idx}].name must be non-empty.")
        if name in seen_names:
            raise ConfigError(f"Duplicate scene name {name!r}.")
        seen_names.add(name)
        scenes.append(
            Scene(
                name=name,
                input_path=str(
                    _require(entry, "input_path", section_name=f"scenes[{name}]")
                ),
                stats_path=str(
                    _require(entry, "stats_path", section_name=f"scenes[{name}]")
                ),
                cr1_embeddings_path=str(
                    _require(
                        entry, "cr1_embeddings_path", section_name=f"scenes[{name}]"
                    )
                ),
                start_frame_idx=int(entry.get("start_frame_idx", 0)),
            )
        )
    return scenes


def build_runtime_config(
    cfg: dict[str, Any], *, role: str, scenes: list[Scene] | None = None
) -> CosmoshRuntimeConfig:
    """Map a loaded YAML dict onto a :class:`CosmoshRuntimeConfig`.

    ``role`` ∈ ``{"keyboard", "quest"}`` decides which input-tuning section
    feeds the dataclass:
      - ``"keyboard"``: ``keyboard.{translate_v_per_frame, gripper_v_per_chunk,
        rotate_deg_per_frame}`` → translate_v_per_frame, gripper_v_per_chunk,
        rotate_theta_per_frame (radians).
      - ``"quest"``: ``vr.{translate_scale, rotate_scale}`` → translate_scale,
        rotate_scale.

    The first :class:`Scene` in ``scenes`` (or as parsed by
    :func:`parse_scenes` when ``scenes`` is ``None``) seeds the per-scene
    fields. The runtime starts on that scene; subsequent switches go
    through :meth:`CosmoshInferenceRuntime.set_scene`.

    Defaults for missing fields come from :class:`CosmoshRuntimeConfig` itself.
    """
    if role not in {"keyboard", "quest"}:
        raise ValueError(f"role must be 'keyboard' or 'quest'; got {role!r}")

    runtime = cfg.get("runtime")
    if not isinstance(runtime, dict):
        raise ConfigError("Missing required section: runtime")

    if scenes is None:
        scenes = parse_scenes(cfg)
    if not scenes:
        raise ConfigError("Expected at least one scene.")
    initial_scene = scenes[0]

    defaults = CosmoshRuntimeConfig()
    ckpt_path = runtime.get("ckpt_path")
    if ckpt_path in (None, "", "default"):
        ckpt_path = None

    kwargs: dict[str, Any] = dict(
        config_name=_require(runtime, "config_name", section_name="runtime"),
        compile_network=bool(runtime.get("compile_network", defaults.compile_network)),
        seed=int(runtime.get("seed", defaults.seed)),
        device=str(runtime.get("device", defaults.device)),
        ckpt_path=ckpt_path,
        cr1_embeddings_path=initial_scene.cr1_embeddings_path,
        input_path=initial_scene.input_path,
        stats_path=initial_scene.stats_path,
        start_frame_idx=initial_scene.start_frame_idx,
        resolution=_parse_resolution(runtime.get("resolution")),
        fps=int(runtime.get("fps", defaults.fps)),
        actions_per_chunk=int(
            runtime.get("actions_per_chunk", defaults.actions_per_chunk)
        ),
        debug_action_npy=(runtime.get("debug_action_npy") or None),
        window_size_t=(
            int(runtime["window_size_t"]) if "window_size_t" in runtime else None
        ),
    )

    if role == "keyboard":
        _apply_keyboard_kwargs(cfg, defaults, kwargs)
    else:  # quest
        _apply_quest_kwargs(cfg, defaults, kwargs)

    return CosmoshRuntimeConfig(**kwargs)


def _apply_keyboard_kwargs(
    cfg: dict[str, Any],
    defaults: CosmoshRuntimeConfig,
    kwargs: dict[str, Any],
) -> None:
    keyboard = cfg.get("keyboard", {}) or {}
    kwargs["translate_v_per_frame"] = float(
        keyboard.get("translate_v_per_frame", defaults.translate_v_per_frame)
    )
    kwargs["gripper_v_per_chunk"] = float(
        keyboard.get("gripper_v_per_chunk", defaults.gripper_v_per_chunk)
    )
    # Config exposes degrees (human-friendly); dataclass stores radians.
    rotate_deg = float(keyboard.get("rotate_deg_per_frame", 1.0))
    kwargs["rotate_theta_per_frame"] = math.radians(rotate_deg)


def _apply_quest_kwargs(
    cfg: dict[str, Any],
    defaults: CosmoshRuntimeConfig,
    kwargs: dict[str, Any],
) -> None:
    vr = cfg.get("vr", {}) or {}
    if "translate_scale" in vr:
        kwargs["translate_scale"] = _parse_translate_scale(vr["translate_scale"])
    else:
        kwargs["translate_scale"] = defaults.translate_scale
    if "rotate_scale" in vr:
        kwargs["rotate_scale"] = _parse_rotate_scale(vr["rotate_scale"])
    else:
        kwargs["rotate_scale"] = defaults.rotate_scale


def build_runtime_config_unified(
    cfg: dict[str, Any], *, scenes: list[Scene] | None = None
) -> CosmoshRuntimeConfig:
    """Variant of :func:`build_runtime_config` that fills both role's knobs.

    The unified server shares one :class:`CosmoshInferenceRuntime` across the
    keyboard and Quest managers; we populate both keyboard-side
    (``translate_v_per_frame`` / ``gripper_v_per_chunk`` /
    ``rotate_theta_per_frame``) and quest-side (``translate_scale`` /
    ``rotate_scale``) tuning so the runtime can serve either render path
    without reloading. Missing sections fall back to dataclass defaults.
    """
    # Reuse the keyboard path's kwargs (it already validates runtime / scenes)
    # and then layer the quest knobs on top.
    rc = build_runtime_config(cfg, role="keyboard", scenes=scenes)
    defaults = CosmoshRuntimeConfig()
    vr_kwargs: dict[str, Any] = {}
    _apply_quest_kwargs(cfg, defaults, vr_kwargs)
    for key, value in vr_kwargs.items():
        setattr(rc, key, value)
    return rc


def get_server_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    """Extract the ``server:`` section as a plain dict.

    Returns ``{}`` if the section is missing. CLI flags layer over this in
    each server's ``main()``.
    """
    server = cfg.get("server") or {}
    if not isinstance(server, dict):
        raise ConfigError("server section must be a mapping")
    return server


def get_video_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    """Extract the ``video:`` section as a plain dict (Quest only)."""
    video = cfg.get("video") or {}
    if not isinstance(video, dict):
        raise ConfigError("video section must be a mapping")
    return video


def get_keyboard_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    """Extract the ``keyboard:`` section as a plain dict (keyboard server only).

    Only ``light_mode`` is read directly here; the rotation/translate/gripper
    tunables already flow through :func:`build_runtime_config`.
    """
    keyboard = cfg.get("keyboard") or {}
    if not isinstance(keyboard, dict):
        raise ConfigError("keyboard section must be a mapping")
    return keyboard


_DISPLAY_DEFAULTS: dict[str, float] = {
    # Match the historical hardcoded quad in quest.js so flipping
    # this on for the first time doesn't visually change anything.
    "width_m": 0.4,
    "height_m": 0.3,
    "distance_m": 1.2,
}


def _parse_display_section(value: Any) -> dict[str, float]:
    """Parse ``vr.display`` into ``{width_m, height_m, distance_m}`` floats.

    Missing fields fall back to the historical hardcoded defaults so a
    partial spec is fine. Non-positive values raise :class:`ConfigError`
    — a 0 m panel or distance is almost certainly a typo.
    """
    if value is None:
        return dict(_DISPLAY_DEFAULTS)
    if not isinstance(value, dict):
        raise ConfigError(f"vr.display must be a mapping; got {type(value).__name__}")
    out: dict[str, float] = dict(_DISPLAY_DEFAULTS)
    for key in ("width_m", "height_m", "distance_m"):
        if key not in value:
            continue
        try:
            v = float(value[key])
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"vr.display.{key} must be a number; got {value[key]!r}"
            ) from exc
        if v <= 0:
            raise ConfigError(f"vr.display.{key} must be positive; got {v}")
        out[key] = v
    return out


# ---------------------------------------------------------------------------
# Encoder (video.encoder + video.nvenc.*) — used by the keyboard/WebRTC path.
# ---------------------------------------------------------------------------

# Defaults for the NVENC sub-block. Match what the design doc proposes
# for the unified_tabletop profile.
_NVENC_DEFAULTS: dict[str, Any] = {
    "preset": "P3",
    "tuning": "ultra_low_latency",
    "bitrate": 3_000_000,
    "idr_period_s": 4.0,
}
_NVENC_KNOWN: frozenset[str] = frozenset(_NVENC_DEFAULTS.keys())


def _validate_nvenc_block(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate ``video.nvenc.*`` and merge with defaults.

    Unknown keys log a warning (typo guard). Missing keys take the
    defaults in :data:`_NVENC_DEFAULTS`. Bitrate and IDR period are
    validated to be positive — non-positive is almost always a typo.
    """
    if not isinstance(raw, dict):
        raise ConfigError("video.nvenc must be a mapping")
    for key in raw.keys():
        if key not in _NVENC_KNOWN:
            LOGGER.warning(
                "Unknown video.nvenc key %r — typo? Known keys: %s",
                key,
                sorted(_NVENC_KNOWN),
            )
    out: dict[str, Any] = dict(_NVENC_DEFAULTS)
    if "preset" in raw:
        out["preset"] = str(raw["preset"])
    if "tuning" in raw:
        out["tuning"] = str(raw["tuning"])
    if "bitrate" in raw:
        try:
            bitrate = int(raw["bitrate"])
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"video.nvenc.bitrate must be an integer; got {raw['bitrate']!r}"
            ) from exc
        if bitrate <= 0:
            raise ConfigError(f"video.nvenc.bitrate must be > 0; got {bitrate}")
        out["bitrate"] = bitrate
    if "idr_period_s" in raw:
        try:
            idr = float(raw["idr_period_s"])
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                "video.nvenc.idr_period_s must be a number; got "
                f"{raw['idr_period_s']!r}"
            ) from exc
        if idr <= 0:
            raise ConfigError(f"video.nvenc.idr_period_s must be > 0; got {idr}")
        out["idr_period_s"] = idr
    return out


def apply_encoder_settings(
    runtime_config: CosmoshRuntimeConfig,
    cfg: dict[str, Any],
    *,
    cli_value: str | None = None,
) -> str:
    """Resolve the encoder choice and write it onto ``runtime_config``.

    Precedence (highest first):

      1. CLI flag (``--encoder``)
      2. ``COSMOSH_VIDEO_ENCODER`` environment variable
      3. ``video.encoder`` field in the YAML config
      4. Code default (``cpu_libav`` — the safe fallback when no layer specifies)

    ``auto`` is resolved against the NVENC probe at startup; the
    function logs the resolution at INFO so operators always know
    which encoder path is active. Returns the active encoder string
    (one of :data:`ENCODER_NVENC` or :data:`ENCODER_CPU_LIBAV`; never
    ``auto`` — that has already been resolved).

    Raises :class:`cosmosHDreams.webrtc.nvenc.NvencUnavailableError` if the
    user explicitly asked for ``nvenc`` but the probe says NVENC is
    not usable in this process — the call site is expected to let
    that bubble up to ``main()`` so the server fails fast on
    misconfigured rigs.
    """
    encoder_settings = get_encoder_settings(cfg)
    requested = select_requested_encoder(
        cli_value=cli_value,
        env_value=read_encoder_env(),
        yaml_value=encoder_settings["encoder"],
        code_default=ENCODER_CPU_LIBAV,
    )
    # Thread target resolution into the resolver so the probe can call
    # ``GetEncoderCaps`` and reject configurations the GPU cannot encode
    # (e.g., a resolution outside the driver-reported bounds) before
    # ``CreateEncoder`` fails at initialize-time. ``resolution`` is stored
    # as ``[height, width]``; pass a zero-fallback if it's unset so the
    # probe falls back to its shallow (import + CUDA) check.
    res = getattr(runtime_config, "resolution", None) or (0, 0)
    height = int(res[0]) if len(res) >= 1 else 0
    width = int(res[1]) if len(res) >= 2 else 0
    active, reason = resolve_encoder(
        requested,
        gpu_id=0,
        width=width,
        height=height,
    )
    LOGGER.info("Video encoder: %r (%s).", active, reason)

    runtime_config.video_encoder = active
    nvenc_block = encoder_settings["nvenc"]
    runtime_config.nvenc_preset = str(nvenc_block["preset"])
    runtime_config.nvenc_tuning = str(nvenc_block["tuning"])
    runtime_config.nvenc_bitrate = int(nvenc_block["bitrate"])
    runtime_config.nvenc_idr_period_s = float(nvenc_block["idr_period_s"])
    return active


def get_encoder_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    """Extract the video-encoder choice + NVENC tuning from the YAML.

    Returns::

        {
          "encoder": str,   # YAML-level request, one of VALID_ENCODERS.
                            # Defaults to ENCODER_CPU_LIBAV when absent.
                            # The final encoder used at runtime is resolved
                            # by the server's main() against the precedence
                            # chain (CLI > env > this YAML value > code
                            # default), and the auto-probe runs there.
          "nvenc": dict,    # Validated video.nvenc.* sub-block (see
                            # _validate_nvenc_block). Always populated with
                            # defaults so the NVENC encoder wrapper can
                            # consume it unconditionally.
        }

    This function intentionally does NOT call the resolver — it only
    reports what the YAML asks for. The server is responsible for
    applying CLI / env overrides and resolving ``auto`` at startup.
    """
    video = cfg.get("video") or {}
    if not isinstance(video, dict):
        raise ConfigError("video section must be a mapping")

    requested_raw = video.get("encoder")
    requested = normalize_encoder_value(requested_raw)
    if requested is None:
        requested = ENCODER_CPU_LIBAV
    elif requested not in VALID_ENCODERS:
        valid = ", ".join(sorted(VALID_ENCODERS))
        raise ConfigError(
            f"video.encoder must be one of [{valid}]; got {requested_raw!r}"
        )

    nvenc_raw = video.get("nvenc") or {}
    nvenc = _validate_nvenc_block(nvenc_raw)

    return {"encoder": requested, "nvenc": nvenc}


# ---------------------------------------------------------------------------
# VR (browser) settings
# ---------------------------------------------------------------------------


def get_vr_browser_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return the subset of ``vr:`` that the browser needs to know about.

    Includes: the two input-semantics flags + the in-headset display panel
    sizing. ``translate_scale`` / ``rotate_scale`` are server-side only
    (applied after the server receives the wire values) so they don't need
    to leak to the browser.
    """
    vr = cfg.get("vr") or {}
    if not isinstance(vr, dict):
        raise ConfigError("vr section must be a mapping")
    return {
        "body_relative_translate": bool(vr.get("body_relative_translate", False)),
        "body_relative_rotation": bool(vr.get("body_relative_rotation", False)),
        "display": _parse_display_section(vr.get("display")),
        "video_transport": _parse_vr_video_transport(vr.get("video_transport")),
    }


_VALID_VR_VIDEO_TRANSPORTS: frozenset[str] = frozenset({"mjpeg", "webrtc"})


def _parse_vr_video_transport(raw: Any) -> str:
    """Validate ``vr.video_transport`` (``mjpeg`` default, or ``webrtc``).

    ``mjpeg`` is the legacy path — Quest browser fetches multipart-JPEG over
    HTTP. ``webrtc`` switches the headset to a standard WebRTC video
    stream that carries the H.264 bytes NVENC is already producing for the
    keyboard path. Spectator ``/viewer`` always stays on MJPEG; this knob
    only affects the Quest headset's own video path.
    """
    if raw is None:
        return "mjpeg"
    if not isinstance(raw, str):
        raise ConfigError(
            f"vr.video_transport must be a string; got {type(raw).__name__}"
        )
    value = raw.strip().lower()
    if value not in _VALID_VR_VIDEO_TRANSPORTS:
        raise ConfigError(
            "vr.video_transport must be one of "
            f"{sorted(_VALID_VR_VIDEO_TRANSPORTS)}; got {raw!r}"
        )
    return value
