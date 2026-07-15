"""Unit tests for ``cosmosHDreams.webrtc.nvenc.track.CosmoshNvencVideoTrack``.

Asyncio-based; uses ``pytest-asyncio`` (already in the ``dev`` extra).

Verifies the marker queue semantics + the lockstep contract with
aiortc's pacing layer (``recv`` blocks until a marker arrives;
``close`` wakes a pending ``recv``; ``drain_pending`` empties the
queue idempotently; placeholder ``av.VideoFrame`` has the right
``pts``/``time_base``).
"""
from __future__ import annotations

import asyncio
from fractions import Fraction

import pytest
from aiortc.mediastreams import MediaStreamError  # type: ignore[import-untyped]
from av import VideoFrame  # type: ignore[import-untyped]

from cosmosHDreams.webrtc.nvenc.track import CosmoshNvencVideoTrack

pytestmark = pytest.mark.ci_cpu


# ---------------------------------------------------------------------------
# Constructor + introspection
# ---------------------------------------------------------------------------


def test_constructor_rejects_invalid_fps():
    with pytest.raises(ValueError, match="fps must be > 0"):
        CosmoshNvencVideoTrack(fps=0)
    with pytest.raises(ValueError, match="fps must be > 0"):
        CosmoshNvencVideoTrack(fps=-1)


def test_kind_is_video():
    track = CosmoshNvencVideoTrack(fps=30)
    assert track.kind == "video"


def test_qsize_starts_at_zero():
    track = CosmoshNvencVideoTrack(fps=30)
    assert track.qsize() == 0


# ---------------------------------------------------------------------------
# enqueue_markers — producer side
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enqueue_markers_increases_qsize():
    track = CosmoshNvencVideoTrack(fps=30)
    n = await track.enqueue_markers(num_frames=8)
    assert n == 8
    assert track.qsize() == 8


@pytest.mark.asyncio
async def test_enqueue_markers_noop_on_zero_or_negative():
    track = CosmoshNvencVideoTrack(fps=30)
    assert await track.enqueue_markers(num_frames=0) == 0
    assert await track.enqueue_markers(num_frames=-3) == 0
    assert track.qsize() == 0


@pytest.mark.asyncio
async def test_enqueue_markers_noop_after_close():
    track = CosmoshNvencVideoTrack(fps=30)
    await track.close()
    # close() pushes a sentinel; subsequent enqueue must not add markers.
    pre = track.qsize()
    n = await track.enqueue_markers(num_frames=5)
    assert n == 0
    assert track.qsize() == pre


# ---------------------------------------------------------------------------
# drain_pending
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_drain_pending_clears_queue():
    track = CosmoshNvencVideoTrack(fps=30)
    await track.enqueue_markers(num_frames=12)
    dropped = track.drain_pending()
    assert dropped == 12
    assert track.qsize() == 0


@pytest.mark.asyncio
async def test_drain_pending_empty_queue_returns_zero():
    track = CosmoshNvencVideoTrack(fps=30)
    assert track.drain_pending() == 0


# ---------------------------------------------------------------------------
# recv — consumer side + placeholder frame contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recv_returns_videoframe_with_monotonic_pts():
    track = CosmoshNvencVideoTrack(fps=30)
    await track.enqueue_markers(num_frames=3)
    f0 = await track.recv()
    f1 = await track.recv()
    f2 = await track.recv()
    assert isinstance(f0, VideoFrame)
    assert isinstance(f1, VideoFrame)
    assert isinstance(f2, VideoFrame)
    # Monotonic pts starting at 0.
    assert (f0.pts, f1.pts, f2.pts) == (0, 1, 2)
    # time_base is 1/fps for all three.
    assert f0.time_base == Fraction(1, 30)
    assert f1.time_base == Fraction(1, 30)
    assert f2.time_base == Fraction(1, 30)


@pytest.mark.asyncio
async def test_recv_placeholder_is_rgb24():
    """The placeholder frame is a valid VideoFrame in a format aiortc accepts."""
    track = CosmoshNvencVideoTrack(fps=30)
    await track.enqueue_markers(num_frames=1)
    frame = await track.recv()
    assert frame.format.name == "rgb24"
    # Tiny placeholder — we explicitly do not match the encoder's
    # resolution; the shim discards pixel data.
    assert frame.width == 16 and frame.height == 16


@pytest.mark.asyncio
async def test_recv_paces_at_fps():
    """recv enforces ~1/fps wall-clock between consecutive frames."""
    fps = 60
    track = CosmoshNvencVideoTrack(fps=fps)
    await track.enqueue_markers(num_frames=3)
    # Drain the first frame to anchor the deadline (recv anchors on
    # first frame regardless of when it was enqueued).
    await track.recv()
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    await track.recv()
    t1 = loop.time()
    await track.recv()
    t2 = loop.time()
    # Each frame should be at least ~1/(2*fps) apart — give a wide
    # tolerance because event-loop scheduling jitter dominates the
    # tight 1/60 = 16.7ms target on CI hosts.
    min_gap = 1.0 / (2 * fps)
    assert (t1 - t0) >= min_gap, f"first gap {t1 - t0}s < {min_gap}s"
    assert (t2 - t1) >= min_gap, f"second gap {t2 - t1}s < {min_gap}s"


@pytest.mark.asyncio
async def test_recv_blocks_until_marker_available():
    """recv() with empty queue blocks; producer push unblocks it."""
    track = CosmoshNvencVideoTrack(fps=30)
    # Schedule a delayed enqueue.
    async def producer():
        await asyncio.sleep(0.05)
        await track.enqueue_markers(num_frames=1)
    producer_task = asyncio.create_task(producer())
    frame = await asyncio.wait_for(track.recv(), timeout=1.0)
    await producer_task
    assert isinstance(frame, VideoFrame)


@pytest.mark.asyncio
async def test_close_wakes_pending_recv():
    """close() must unblock a pending recv() with MediaStreamError."""
    track = CosmoshNvencVideoTrack(fps=30)

    async def closer():
        await asyncio.sleep(0.05)
        await track.close()

    closer_task = asyncio.create_task(closer())
    with pytest.raises(MediaStreamError):
        await asyncio.wait_for(track.recv(), timeout=1.0)
    await closer_task


@pytest.mark.asyncio
async def test_recv_after_close_raises():
    track = CosmoshNvencVideoTrack(fps=30)
    await track.close()
    with pytest.raises(MediaStreamError):
        await track.recv()


@pytest.mark.asyncio
async def test_close_is_idempotent():
    track = CosmoshNvencVideoTrack(fps=30)
    await track.close()
    await track.close()  # must not raise / hang


# ---------------------------------------------------------------------------
# Lockstep contract — markers consumed in order
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_marker_count_matches_recv_count():
    """N enqueued markers → exactly N successful recv() calls."""
    track = CosmoshNvencVideoTrack(fps=120)  # high fps so pacing doesn't slow us
    await track.enqueue_markers(num_frames=5)
    frames = [await track.recv() for _ in range(5)]
    assert len(frames) == 5
    assert all(isinstance(f, VideoFrame) for f in frames)
    assert track.qsize() == 0
