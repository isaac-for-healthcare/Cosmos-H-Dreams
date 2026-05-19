# Cosmosh WebRTC perf report — anti-chop gaps vs `lingbot`

This is the audit that produced the `javierg/perf_improvements`
branch. It enumerates concrete differences between
`integrations/lingbot/lingbot/webrtc/` and
`integrations/cosmosh/cosmosh/webrtc/` that account for the
"choppy / bursty / stuttery" playback users have reported on the
cosmosh viewer. Lingbot ships every fix below; cosmosh ships none.

The file diff that matters most is `media.py` (the `*VideoTrack`
class):

- `integrations/lingbot/lingbot/webrtc/media.py` — 230 lines
- `integrations/cosmosh/cosmosh/webrtc/media.py` — 98 lines

The ~150 missing lines on the cosmosh side are anti-chop logic.

---

## Fixes, ranked by visible smoothness impact

### 1. Stall detection + re-anchor on empty-queue wait *(highest impact)*

**Symptom:** burst-of-frames-after-stall. Whenever the producer
briefly falls behind playback (which happens at every chunk boundary
if generation isn't ahead by ≥1 chunk), the queue drains. The next
`get()` waits for the producer; during that wait the playback
deadline drifts behind walltime by the stall duration. When the next
chunk lands the next several `recv` calls see `wait_s < 0` and send
frames back-to-back. The browser jitter buffer responds by *speeding
up* playback to consume the burst. Repeat on every chunk boundary →
sawtooth playback.

**Lingbot** (`media.py:142-179`):

```python
t_get_start = loop.time()
frame_array = await self._frames.get()
get_wait_ms = (loop.time() - t_get_start) * 1000.0
first_frame = self._next_deadline_s is None
just_stalled = (not first_frame) and get_wait_ms > _STALL_THRESHOLD_MS  # 1.0 ms
...
if first_frame or just_stalled:
    self._next_deadline_s = now_s   # ← anchor fresh, don't try to "catch up"
else:
    ...
```

**Cosmosh** (`media.py:69-91`):

```python
frame_array = await self._frames.get()      # no timing
...
self._next_deadline_s += self._frame_interval_s   # always advance
```

**Fix:** copy the stall-detection block from lingbot. ~10 lines.

---

### 2. "Deadline behind walltime → re-anchor" branch

**Symptom:** same burst pattern as (1), just a different trigger
(aiortc's send loop lagging, `asyncio.sleep` over-sleeping, another
task hogging the loop). Without re-anchoring, the deadline stays in
the past for every subsequent frame.

**Lingbot** (`media.py:181-206`):

```python
proposed = self._next_deadline_s + self._frame_interval_s
wait_s = proposed - now_s
if wait_s > 0:
    await asyncio.sleep(wait_s)
    self._next_deadline_s = proposed
else:
    if -wait_s * 1000.0 > _PACING_LAG_LOG_MS:
        LOGGER.warning("Pacing lag: …")
    self._next_deadline_s = now_s     # ← re-anchor on lag
```

**Cosmosh** (`media.py:82-86`):

```python
self._next_deadline_s += self._frame_interval_s
wait_s = self._next_deadline_s - now_s
if wait_s > 0:
    await asyncio.sleep(wait_s)
# else: falls through, _next_deadline_s stays in the past → next frame also burst
```

**Fix:** the `else: self._next_deadline_s = now_s` branch from
lingbot. ~3 lines (+ optional log line).

(1) and (2) are the **two patches that actually fix the visible
choppiness**. They are surgical, ~15 lines combined, and don't
require any signature changes.

---

### 3. Tensor → uint8 conversion on a worker thread

**Symptom:** asyncio loop stalls for the duration of the cast,
holding up `recv`'s 1/fps pacing and creating exactly the stall
condition (1) tries to recover from. Worse at higher resolutions.

**Lingbot** (`media.py:132`):

```python
frames = await asyncio.to_thread(tensor_chunk_to_rgb_frames, video_chunk)
```

**Cosmosh** (`media.py:49`):

```python
frames = tensor_chunk_to_rgb_frames(video_chunk)   # runs on the event loop
```

**Fix:** one-line change to `enqueue_chunk`. Combined with (1)+(2)
this removes the most common cause of those stalls in the first
place, not just papers over them.

---

### 4. Bounded queue → real producer-side backpressure

**Symptom:** producer can run arbitrarily far ahead in bursts, then
sit idle while the consumer drains; queue depth oscillates and
end-to-end latency floats up. The cosmosh render loop tries to
compensate with a soft cap polled every 50 ms, but the polling itself
adds jitter and doesn't pace within a single chunk.

**Lingbot** (`media.py:84-96`):

```python
def __init__(self, *, fps: int, maxsize: int) -> None:
    ...
    self._maxsize = maxsize
    self._frames: asyncio.Queue[…] = asyncio.Queue(maxsize=maxsize)
# constructed in session.py with maxsize = steady-state num_frames per chunk
video_track = LingbotVideoTrack(fps=self.fps, maxsize=num_frames)
```

`enqueue_chunk`'s `await put` blocks once the queue holds one
steady-state chunk → producer is paced exactly to consumer's drain
rate.

**Cosmosh** (`media.py:45`):

```python
self._frames: asyncio.Queue[…] = asyncio.Queue()   # unbounded
```

…plus an external soft cap in `session.py`:

```python
_MAX_BUFFERED_FRAMES = 2 * DEFAULT_ACTIONS_PER_OUTER_BLOCK   # 24 frames
_BACKPRESSURE_POLL_S = 0.05
while managed_session.video_track.qsize() >= _MAX_BUFFERED_FRAMES:
    await asyncio.sleep(_BACKPRESSURE_POLL_S)
```

**Fix:** moderate. Plumb steady-state `num_frames` from the runtime
through to the track constructor; bound the queue; remove the
external soft-cap poll. Note that the lingbot docstring (`media.py:74-80`)
specifically calls out using *steady-state* count, not AR-step-0's
output, because AR 0 produces fewer frames due to causal padding.
Cosmosh has a constant 12 generated frames per outer block, so this
isn't a footgun here, but worth tracing through `peek_next_chunk_num_frames`
to be safe.

---

### 5. Drain-before-sentinel on `close()`

**Symptom:** none on its own. But if you apply (4), `close()`'s
current `await self._frames.put(None)` will deadlock on a full
bounded queue.

**Lingbot** (`media.py:214-230`):

```python
while True:
    try:
        self._frames.get_nowait()
    except asyncio.QueueEmpty:
        break
self._frames.put_nowait(None)
self.stop()
```

**Cosmosh** (`media.py:93-98`):

```python
self._closed = True
await self._frames.put(None)   # deadlocks if queue is bounded + full
self.stop()
```

**Fix:** swap to the drain-then-put_nowait pattern. Required only if
(4) lands.

---

### Bonus: render-loop virtual-clock catch-up *(session.py, not media.py)*

**Symptom:** one transient GPU stall (e.g. first-block warmup) pegs
end-to-end input → pixel latency forever. The render loop keeps
generating against a stale virtual clock and the lag never recovers.

**Lingbot** (`session.py:843-862`):

```python
lag = now - (resampler.next_chunk_start_v + chunk_duration)
if lag > chunk_duration:
    skipped_to = now - chunk_duration
    LOGGER.warning(...)
    resampler.next_chunk_start_v = skipped_to
```

**Cosmosh:** no equivalent. Cosmosh's resampler model is much simpler
(no virtual-clock event resampling — it just consumes the current
keyboard / VR snapshot at chunk start), so this fix may not apply
1:1. Worth a separate evaluation pass after (1)–(4) land.

---

## Suggested patch order

| Order | Fix | Effort | Effect |
|---|---|---|---|
| 1 | (1) Stall detection + re-anchor on empty wait | ~10 LOC in `media.py` | Eliminates per-chunk-boundary burst |
| 2 | (2) Re-anchor on deadline-behind-walltime | ~3 LOC in `media.py` | Eliminates aiortc-jitter-induced burst |
| 3 | (3) Offload uint8 cast to `asyncio.to_thread` | 1 LOC in `media.py` | Removes a major stall source |
| 4 | (4) Bounded queue + remove session.py soft cap | moderate, multi-file | Cleaner pacing, lower latency floor |
| 5 | (5) Drain-then-sentinel in `close()` | ~5 LOC | Required by (4) |
| 6 | Bonus catch-up (session.py) | depends | Bounds worst-case latency |

(1) + (2) + (3) together are <20 lines of code and should be the
first commit on `javierg/perf_improvements`. (4) + (5) should be a
separate commit so the bounded-queue change is reviewable in
isolation.

---

## Files referenced

- `integrations/lingbot/lingbot/webrtc/media.py` (target shape)
- `integrations/lingbot/lingbot/webrtc/session.py` (catch-up branch)
- `integrations/cosmosh/cosmosh/webrtc/media.py` (current; the patch target)
- `integrations/cosmosh/cosmosh/webrtc/session.py:692-695` (`_MAX_BUFFERED_FRAMES` soft cap)
- `integrations/cosmosh/cosmosh/webrtc/session.py:987-994` (soft-cap poll site)
