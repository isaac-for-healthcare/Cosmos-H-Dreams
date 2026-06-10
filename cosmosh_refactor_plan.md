# CosmosH Refactor Plan

## Goal
Move cosmosh from `flashdreams/recipes/cosmosh/` (in-tree builtin) to `integrations/cosmosh/cosmosh/` (workspace plugin), following the pattern of `integrations/cosmos_predict2/`.

---

## Current State

```
flashdreams/recipes/cosmosh/          ← model inference code (to move)
    __init__.py
    config.py                         ← builders + register_runner() loop
    constants.py
    pipeline.py
    runner.py
    encoder/__init__.py
    encoder/action.py
    transformer/__init__.py
    transformer/impl/__init__.py
    transformer/impl/modules.py
    transformer/impl/network.py
    transformer/impl/rope.py

integrations/cosmosh/cosmosh/         ← WebRTC server (stays, gains siblings)
    __init__.py
    webrtc/
        session.py                    ← imports from flashdreams.recipes.cosmosh (!)
        ...
integrations/cosmosh/pyproject.toml  ← no entry-points yet
```

---

## Target State

```
integrations/cosmosh/cosmosh/
    __init__.py                       (update)
    config.py                         (moved + fixed)
    constants.py                      (moved)
    pipeline.py                       (moved + imports fixed)
    runner.py                         (moved + imports fixed)
    encoder/__init__.py               (moved)
    encoder/action.py                 (moved + imports fixed)
    transformer/__init__.py           (moved, relative imports OK)
    transformer/impl/__init__.py      (moved)
    transformer/impl/modules.py       (moved)
    transformer/impl/network.py       (moved)
    transformer/impl/rope.py          (moved)
    webrtc/
        session.py                    (imports fixed)
        ...

integrations/cosmosh/pyproject.toml  (+ 6 entry-points)

flashdreams/recipes/cosmosh/          ← DELETED
```

---

## Steps

### 1. Create directories
```
integrations/cosmosh/cosmosh/encoder/
integrations/cosmosh/cosmosh/transformer/impl/
```

### 2. Copy files (no changes needed yet)
Copy entire `flashdreams/recipes/cosmosh/` tree into `integrations/cosmosh/cosmosh/`.

### 3. Fix imports in moved files

**Bulk replace** `flashdreams.recipes.cosmosh.` → `cosmosh.` in all moved files:
- `config.py`
- `pipeline.py`
- `runner.py`
- `encoder/action.py`
- `transformer/__init__.py` (uses relative imports — likely no change needed)

**Also fix `session.py`** (already in integration):
- `from flashdreams.recipes.cosmosh.config import COSMOSH_CONFIG_BUILDERS`
- `from flashdreams.recipes.cosmosh.constants import AVAILABLE_COSMOSH_CHECKPOINT_PATHS`
→ `from cosmosh.config import ...` / `from cosmosh.constants import ...`

### 4. Fix `recipe_name` → `name` bug in config.py

The public API renamed `StreamInferencePipelineConfig.recipe_name` to `name`.
All calls like `CosmoshPipelineConfig(recipe_name=recipe_name, ...)` must become `CosmoshPipelineConfig(name=recipe_name, ...)`.

Occurrences in `config.py`: lines 223, 372, 457, 511 (all pass-throughs of local var named `recipe_name`).

### 5. Switch runner registration: builtin → plugin

**Remove from config.py:**
```python
from flashdreams.configs.registry import register_runner
...
for _name, _cfg in COSMOSH_RUNNERS.items():
    register_runner(_name, _cfg, source="builtin")
```

**Add to config.py** (after `COSMOSH_RUNNERS` dict):
```python
RUNNER_COSMOSH_VAE_VAE             = COSMOSH_RUNNERS["cosmosh-vae-vae"]
RUNNER_COSMOSH_VAE_LIGHTTAE        = COSMOSH_RUNNERS["cosmosh-vae-lighttae"]
RUNNER_COSMOSH_LIGHTVAE_LIGHTTAE   = COSMOSH_RUNNERS["cosmosh-lightvae-lighttae"]
RUNNER_COSMOSH_2STEPS_VAE_VAE      = COSMOSH_RUNNERS["cosmosh-2steps-vae-vae"]
RUNNER_COSMOSH_2STEPS_VAE_LIGHTTAE = COSMOSH_RUNNERS["cosmosh-2steps-vae-lighttae"]
RUNNER_COSMOSH_2STEPS_LIGHTVAE_LIGHTTAE = COSMOSH_RUNNERS["cosmosh-2steps-lightvae-lighttae"]
```

**Add to `integrations/cosmosh/pyproject.toml`:**
```toml
[project.entry-points."flashdreams.runner_configs"]
"cosmosh-vae-vae"                    = "cosmosh.config:RUNNER_COSMOSH_VAE_VAE"
"cosmosh-vae-lighttae"               = "cosmosh.config:RUNNER_COSMOSH_VAE_LIGHTTAE"
"cosmosh-lightvae-lighttae"          = "cosmosh.config:RUNNER_COSMOSH_LIGHTVAE_LIGHTTAE"
"cosmosh-2steps-vae-vae"             = "cosmosh.config:RUNNER_COSMOSH_2STEPS_VAE_VAE"
"cosmosh-2steps-vae-lighttae"        = "cosmosh.config:RUNNER_COSMOSH_2STEPS_VAE_LIGHTTAE"
"cosmosh-2steps-lightvae-lighttae"   = "cosmosh.config:RUNNER_COSMOSH_2STEPS_LIGHTVAE_LIGHTTAE"
```

### 6. Update pyproject.toml dependencies

Add any new runtime deps the inference code needs (e.g. `loguru`, `numpy`, `torch` — check runner.py imports).

Add SPDX header (missing from current pyproject.toml).

### 7. Delete old recipe

```
rm -rf flashdreams/recipes/cosmosh/
```

### 8. Verify runner_configs.py needs NO changes

`flashdreams/configs/runner_configs.py` currently has NO import for cosmosh (was never wired in). Since cosmosh is now a plugin, discovery happens via entry-points — nothing to add.

### 9. Run tests
```bash
uv run pytest flashdreams/tests/ -m ci_cpu -k "cosmosh or recipe_config"
uv run pytest integrations/cosmosh/tests/ -m ci_cpu
```

---

## Known Issues / Risks

| Issue | Fix |
|-------|-----|
| `recipe_name` vs `name` field | Fixed in step 4 |
| `CosmosHTransformerConfig` has `height`/`width` config fields (not per-rollout) | Leave as-is (separate cleanup) |
| `transformer/impl/rope.py` is a custom RoPE; canonical one is in `flashdreams.core.attention` | Leave as-is (separate cleanup) |
| No cosmosh smoke tests in `flashdreams/tests/` | Tests live in `integrations/cosmosh/tests/` — may need GPU markers |
| `test_recipe_configs.py::test_supported_runners_covers_every_runner_dict` only checks `TEMPLATE_RUNNERS` | Cosmosh was never in that test — no change needed |
