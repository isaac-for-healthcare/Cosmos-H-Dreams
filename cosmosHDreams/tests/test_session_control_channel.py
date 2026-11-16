# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The render loops reach the control DataChannel that opens after they start.

The keyboard session starts ``_render_loop`` when the peer connection reports
``connected``; the browser's control DataChannel arrives later through
``on_datachannel``. These tests drive the real loops with stand-in runtime,
track, peer connection and channel objects (no GPU, no network).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import pytest

from cosmosHDreams.webrtc.session import (
    CosmoshStepResult,
    CosmoshWebRTCSessionManager,
    _ManagedCosmoshSession,
)

pytestmark = pytest.mark.ci_cpu

_TIMEOUT_S = 5.0


class FakeChannel:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    def send(self, data: str) -> None:
        self.messages.append(json.loads(data))

    def types(self) -> list[str]:
        return [m["type"] for m in self.messages]


class BrokenChannel:
    def send(self, data: str) -> None:
        raise RuntimeError("channel closed")


class FakeKeyboardState:
    def __init__(self) -> None:
        self.pressed_keys: set[str] = set()


class FakeRuntime:
    def __init__(self, *, debug_action_override: bool = False) -> None:
        self.debug_action_override = debug_action_override
        self.keyboard_state = FakeKeyboardState()
        self.generated = 0
        self.debug_calls = 0
        self.channel_attached = asyncio.Event()

    def initial_frame_chunk(self) -> CosmoshStepResult:
        return CosmoshStepResult(chunk_index=0, num_frames=1)

    async def apply_actions_and_generate(
        self, actions: list[dict[str, Any]]
    ) -> CosmoshStepResult:
        await asyncio.sleep(0)
        self.generated += 1
        return CosmoshStepResult(chunk_index=self.generated, num_frames=12)

    async def generate_one_chunk_debug(self) -> CosmoshStepResult | None:
        self.debug_calls += 1
        if self.debug_calls == 1:
            return CosmoshStepResult(chunk_index=1, num_frames=12)
        if self.debug_calls == 2:
            # Hold the second chunk until the test has attached the channel.
            await self.channel_attached.wait()
            return CosmoshStepResult(chunk_index=2, num_frames=12)
        return None


class FakeTrack:
    def qsize(self) -> int:
        return 0

    async def enqueue_chunk(self, video_chunk: Any) -> tuple[int, float]:
        return 12, 0.0

    async def close(self) -> None:
        pass


class FakePeerConnection:
    async def close(self) -> None:
        pass


async def _no_op_publish(
    video_track: Any, result: CosmoshStepResult
) -> tuple[int, float]:
    return result.num_frames, 0.0


def _make_session(
    runtime: FakeRuntime, *, light_mode: bool
) -> tuple[CosmoshWebRTCSessionManager, _ManagedCosmoshSession]:
    manager = CosmoshWebRTCSessionManager(runtime=runtime)  # ty:ignore[invalid-argument-type]
    # Skip the real track-type dispatch; the fake track is neither NVENC nor CPU.
    manager._publish_step_to_track = _no_op_publish  # ty:ignore[invalid-assignment]
    session = _ManagedCosmoshSession(
        runtime=runtime,  # ty:ignore[invalid-argument-type]
        video_track=FakeTrack(),  # ty:ignore[invalid-argument-type]
        peer_connection=FakePeerConnection(),
        light_mode=light_mode,
    )
    return manager, session


async def _wait_for(condition: Any) -> None:
    async def _poll() -> None:
        while not condition():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(_poll(), timeout=_TIMEOUT_S)


@pytest.mark.asyncio
async def test_render_loop_sends_chunk_done_on_channel_opened_after_start() -> None:
    runtime = FakeRuntime()
    manager, session = _make_session(runtime, light_mode=True)
    assert session.control_channel is None

    # As on ``connectionState == "connected"``: the loop starts with no channel.
    session.render_task = asyncio.create_task(
        manager._render_loop(managed_session=session)
    )
    await asyncio.sleep(0.05)

    # As ``on_datachannel`` then ``_handle_datachannel_message`` do.
    channel = FakeChannel()
    session.control_channel = channel
    session.pending_actions.append({"type": "keydown", "key": "w"})
    session.first_action_event.set()

    try:
        await _wait_for(lambda: "chunk_done" in channel.types())
    finally:
        await session.close()

    chunk_done = [m for m in channel.messages if m["type"] == "chunk_done"]
    assert chunk_done[0] == {
        "type": "chunk_done",
        "chunk_index": 1,
        "num_frames": 12,
        "enqueued_frames": 12,
    }


@pytest.mark.asyncio
async def test_debug_render_loop_sends_on_channel_opened_after_start() -> None:
    runtime = FakeRuntime(debug_action_override=True)
    manager, session = _make_session(runtime, light_mode=False)

    session.render_task = asyncio.create_task(
        manager._render_loop(managed_session=session)
    )
    # Chunk 1 is rendered before any channel exists and is dropped.
    await _wait_for(lambda: runtime.debug_calls >= 2)

    channel = FakeChannel()
    session.control_channel = channel
    runtime.channel_attached.set()

    try:
        await _wait_for(lambda: "debug_stream_end" in channel.types())
    finally:
        await session.close()

    assert channel.types() == ["chunk_done", "debug_stream_end"]
    assert channel.messages[0]["chunk_index"] == 2


def test_send_json_logs_a_failed_send(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="cosmosHDreams.webrtc.session"):
        CosmoshWebRTCSessionManager._send_json(BrokenChannel(), {"type": "chunk_done"})
    assert "Could not send 'chunk_done' message" in caplog.text
    assert "channel closed" in caplog.text


def test_send_json_without_channel_does_not_raise() -> None:
    CosmoshWebRTCSessionManager._send_json(None, {"type": "chunk_done"})
