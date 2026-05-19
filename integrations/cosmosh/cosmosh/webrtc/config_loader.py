"""YAML config loader for both servers (keyboard + Quest).

One YAML schema covers both servers; each one only reads the sections it
cares about. Unknown keys log a warning so typos don't silently no-op.

Schema (full)::

    runtime:
      config_name: lightvae_lighttae
      compile_network: true
      device: cuda:0
      seed: 1
      ckpt_path: null                       # null → default checkpoint
      resolution: [288, 512]                # or null for native
      fps: 10
      actions_per_chunk: 12                 # Must divide num_action_per_latent_frame
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
      # decouple from the user's body orientation; see QUEST_PLAN.md.
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

from cosmosh.webrtc.session import CosmoshRuntimeConfig, Scene

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
        raise ConfigError(f"Top-level YAML must be a mapping; got {type(data).__name__}")
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
    raise ConfigError(
        f"{ctx} must be a scalar or 3-element list; got {value!r}"
    )


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
    raise ConfigError(
        f"resolution must be [H, W], 'H,W', or null; got {value!r}"
    )


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
                input_path=str(
                    _require(runtime, "input_path", section_name="runtime")
                ),
                stats_path=str(
                    _require(runtime, "stats_path", section_name="runtime")
                ),
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
    )

    if role == "keyboard":
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
    else:  # quest
        vr = cfg.get("vr", {}) or {}
        if "translate_scale" in vr:
            kwargs["translate_scale"] = _parse_translate_scale(vr["translate_scale"])
        else:
            kwargs["translate_scale"] = defaults.translate_scale
        if "rotate_scale" in vr:
            kwargs["rotate_scale"] = _parse_rotate_scale(vr["rotate_scale"])
        else:
            kwargs["rotate_scale"] = defaults.rotate_scale

    return CosmoshRuntimeConfig(**kwargs)


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
        raise ConfigError(
            f"vr.display must be a mapping; got {type(value).__name__}"
        )
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
            raise ConfigError(
                f"vr.display.{key} must be positive; got {v}"
            )
        out[key] = v
    return out


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
    }
