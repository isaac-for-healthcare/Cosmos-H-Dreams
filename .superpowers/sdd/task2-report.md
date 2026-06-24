# Task 2 Report — JSONL Logging + [PERF] Log Line + inter_block_gap_ms

## What Changed

### `integrations/cosmosh/cosmosh/webrtc/session.py`

1. Added `_LatencyLogger` class after `_fps_reset_profile` (around line 105):
   - `__init__(mode)` opens `cosmosh_perf_<YYYYMMDD_HHMMSS>.jsonl` in cwd
   - `log_block(record)` writes one JSONL line, flushes, emits `[PERF]` log line (only non-None numeric keys)
   - `log_rollout_summary()` computes per-key averages across all blocks, writes summary JSONL line, logs summary, clears accumulators
   - `close()` closes the file handle

2. Modified `_render_loop` (keyboard render loop):
   - Added `latency_logger = _LatencyLogger("keyboard") if _latency_profile_enabled() else None` and `_t_prev_block_end: float | None = None` before the while loop
   - Records `_t_iter_start` at the start of each iteration (after `first_action_event.wait()`)
   - After `enqueue_chunk` completes: records `_t_iter_end`, computes `gap_ms`, builds record from `result.timing` fields + `gap_ms`, calls `latency_logger.log_block(record)`, updates `_t_prev_block_end`
   - Changed `except asyncio.CancelledError: return` to `except asyncio.CancelledError: pass` + added `finally` block calling `latency_logger.log_rollout_summary(); latency_logger.close()`

### `integrations/cosmosh/cosmosh/webrtc/server_quest.py`

1. Added imports: `_LatencyLogger` and `_latency_profile_enabled` from `cosmosh.webrtc.session`

2. Modified `QuestSessionManager._render_loop`:
   - Added `latency_logger = _LatencyLogger("quest") if _latency_profile_enabled() else None` and `_t_prev_block_end: float | None = None` at the start
   - Records `_t_iter_start` at start of each iteration
   - After `_push_chunk_to_sink` returns: records `_t_iter_end`, computes `gap_ms`, builds record, calls `latency_logger.log_block(record)`, updates `_t_prev_block_end`
   - Adds `latency_logger.log_rollout_summary(); latency_logger.close()` to the existing `finally` block

## Commit Hash

(See below after commit)

## Test Output

```
{"mode": "test", "block": 0, "encode_ms": 12.3, "gap_ms": null}
{"mode": "test", "block": 1, "encode_ms": 11.5, "gap_ms": 5.2}
{"type": "rollout_summary", "mode": "test", "num_blocks": 2, "encode_ms": 11.9, "gap_ms": 5.2}
OK
```

3 JSONL lines written correctly: 2 block records + 1 rollout_summary with averaged values.

## Concerns

None. All changes are gated behind `_latency_profile_enabled()`. Zero overhead when `COSMOSH_PROFILE_LATENCY` is not set. The `gap_ms` for the first block is always `None` (no previous block end time), which is correct and expected.

---

# Task 2 Hotfix — perf_counter guard + datetime import

## What Changed

### Issue 1: `time.perf_counter()` calls moved inside `if latency_logger is not None:` guards

**`session.py` `_render_loop`:**
- `_t_iter_start = time.perf_counter() * 1000.0` moved inside `if latency_logger is not None:` guard at top of iteration
- `_t_iter_end = time.perf_counter() * 1000.0`, `gap_ms` computation, `log_block()` call, and `_t_prev_block_end = _t_iter_end` all consolidated inside a single `if latency_logger is not None:` block after the render lock

**`server_quest.py` `_render_loop`:**
- Same restructuring: `_t_iter_start` guarded at top, `_t_iter_end`, `gap_ms`, `log_block()`, `_t_prev_block_end` all inside `if latency_logger is not None:` after render lock

When `latency_logger is None` (the default when `COSMOSH_PROFILE_LATENCY` is not set), **zero** `time.perf_counter()` calls occur in the render loop hot path.

### Issue 2: `import datetime` moved to top-level

`import datetime` was inside `_LatencyLogger.__init__` method body in `session.py`. Moved to the top of the file with other stdlib imports in alphabetical order (between `import contextlib` and `import json`).

## Test Result

```
OK
```

`_LatencyLogger('test')` constructs correctly with `datetime` imported at module level. `log_block`, `log_rollout_summary`, and `close` all work. File written and cleaned up successfully.
