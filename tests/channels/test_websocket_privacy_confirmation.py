"""WebSocket channel — privacy GateKeeper confirmation bridge.

Exercises the wire protocol (`privacy_confirmation` outbound +
`privacy_confirmation_reply` inbound) and the asyncio.Future correlation
layer that backs `ChannelCapabilities.send_confirmation` /
`await_confirmation_reply`. No real WebSocket server is needed; we test
the dispatch + bridge directly with mock connections.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.channels.websocket import WebSocketChannel
from nanobot.privacy.types import (
    ConfirmationPrompt,
    DetectedEntity,
    EntityType,
    ExecutionPath,
    Linkability,
    Recommendation,
    RiskClass,
)


def _channel(bus: Any = None) -> WebSocketChannel:
    return WebSocketChannel({"enabled": True, "allowFrom": ["*"]}, bus or MagicMock())


def _entity() -> DetectedEntity:
    return DetectedEntity(
        type=EntityType.CREDENTIAL,
        span=(0, 16),
        value="sk-xsadsafsgdrghr",
        risk_class=RiskClass.CATASTROPHIC,
        linkability=Linkability.SINGLE_USE,
        detector="regex:sk_prefixed",
    )


def _prompt(cid: str = "abc123") -> ConfirmationPrompt:
    rec = Recommendation(
        path=ExecutionPath.BLOCKED,
        allowed=frozenset({ExecutionPath.BLOCKED}),
        reason="hard_secret:credential",
        entities=(_entity(),),
    )
    return ConfirmationPrompt(confirmation_id=cid, recommendation=rec, chat_id="chat-1")


@pytest.mark.asyncio
async def test_capabilities_report_interactive_support():
    channel = _channel()
    caps = channel.privacy_capabilities()
    assert caps.supports_interactive_confirm is True
    assert caps.confirmation_max_latency_seconds == 120
    assert caps.send_confirmation is not None
    assert caps.await_confirmation_reply is not None


@pytest.mark.asyncio
async def test_send_confirmation_broadcasts_envelope_to_subscribers():
    channel = _channel()
    conn = AsyncMock()
    channel._attach(conn, "chat-1")

    prompt = _prompt()
    await channel._send_privacy_confirmation("chat-1", prompt)

    assert conn.send.await_count == 1
    raw = conn.send.await_args.args[0]
    payload = json.loads(raw)
    assert payload["event"] == "privacy_confirmation"
    assert payload["chat_id"] == "chat-1"
    assert payload["confirmation_id"] == prompt.confirmation_id
    assert payload["path"] == "blocked"
    assert payload["allowed"] == ["blocked"]
    assert payload["entities"][0]["type"] == "credential"
    assert payload["entities"][0]["risk_class"] == "catastrophic"
    # Future is now pending, waiting for a reply.
    assert prompt.confirmation_id in channel._pending_confirmations


@pytest.mark.asyncio
async def test_send_drops_future_when_no_subscriber():
    channel = _channel()
    # No subscriber attached.
    prompt = _prompt("orphan-1")
    await channel._send_privacy_confirmation("nowhere", prompt)
    # Future must be cleaned up so we don't leak.
    assert "orphan-1" not in channel._pending_confirmations


@pytest.mark.asyncio
async def test_reply_resolves_pending_future():
    channel = _channel()
    conn = AsyncMock()
    channel._attach(conn, "chat-1")
    prompt = _prompt("await-1")

    await channel._send_privacy_confirmation("chat-1", prompt)

    # Schedule the wait alongside an in-loop reply.
    waiter = asyncio.create_task(
        channel._await_privacy_confirmation_reply("await-1", timeout=2.0)
    )
    await asyncio.sleep(0)  # let waiter register
    await channel._dispatch_envelope(
        conn,
        "client",
        {"type": "privacy_confirmation_reply", "confirmation_id": "await-1", "chosen_path": "blocked"},
    )
    reply = await asyncio.wait_for(waiter, timeout=2.0)
    assert reply is not None
    assert reply.confirmation_id == "await-1"
    assert reply.chosen_path == ExecutionPath.BLOCKED
    # Pending map is cleaned.
    assert "await-1" not in channel._pending_confirmations


@pytest.mark.asyncio
async def test_reply_with_null_path_signals_cancel():
    channel = _channel()
    conn = AsyncMock()
    channel._attach(conn, "chat-1")
    await channel._send_privacy_confirmation("chat-1", _prompt("cancel-1"))

    waiter = asyncio.create_task(
        channel._await_privacy_confirmation_reply("cancel-1", timeout=2.0)
    )
    await asyncio.sleep(0)
    await channel._dispatch_envelope(
        conn,
        "client",
        {"type": "privacy_confirmation_reply", "confirmation_id": "cancel-1", "chosen_path": None},
    )
    reply = await asyncio.wait_for(waiter, timeout=2.0)
    assert reply is not None
    assert reply.chosen_path is None


@pytest.mark.asyncio
async def test_wait_times_out_when_no_reply():
    channel = _channel()
    conn = AsyncMock()
    channel._attach(conn, "chat-1")
    await channel._send_privacy_confirmation("chat-1", _prompt("timeout-1"))

    result = await channel._await_privacy_confirmation_reply("timeout-1", timeout=0.1)
    assert result is None
    # The Future has been cleaned up after timeout.
    assert "timeout-1" not in channel._pending_confirmations


@pytest.mark.asyncio
async def test_invalid_chosen_path_reports_error_without_resolving_future():
    channel = _channel()
    conn = AsyncMock()
    channel._attach(conn, "chat-1")
    await channel._send_privacy_confirmation("chat-1", _prompt("bad-1"))

    await channel._dispatch_envelope(
        conn,
        "client",
        {"type": "privacy_confirmation_reply", "confirmation_id": "bad-1", "chosen_path": "not_a_path"},
    )
    # Future still pending — invalid replies don't accidentally resolve as cancel.
    assert "bad-1" in channel._pending_confirmations
    # Error event sent back.
    sent_events = [json.loads(call.args[0]).get("event") for call in conn.send.await_args_list]
    assert "error" in sent_events


@pytest.mark.asyncio
async def test_unknown_confirmation_id_is_silently_ignored():
    channel = _channel()
    conn = AsyncMock()
    channel._attach(conn, "chat-1")
    # No prompt was sent → no future registered.
    await channel._dispatch_envelope(
        conn,
        "client",
        {"type": "privacy_confirmation_reply", "confirmation_id": "ghost", "chosen_path": "blocked"},
    )
    # No error event should be raised for unknown id (avoids enumeration).
    sent_events = [json.loads(call.args[0]).get("event") for call in conn.send.await_args_list]
    assert "error" not in sent_events


@pytest.mark.asyncio
async def test_invalid_confirmation_id_field_returns_error():
    channel = _channel()
    conn = AsyncMock()
    channel._attach(conn, "chat-1")
    await channel._dispatch_envelope(
        conn,
        "client",
        {"type": "privacy_confirmation_reply", "chosen_path": "blocked"},  # missing id
    )
    sent_events = [json.loads(call.args[0]).get("event") for call in conn.send.await_args_list]
    assert "error" in sent_events


@pytest.mark.asyncio
async def test_stop_cancels_pending_confirmations():
    channel = _channel()
    conn = AsyncMock()
    channel._attach(conn, "chat-1")
    await channel._send_privacy_confirmation("chat-1", _prompt("stopping-1"))

    waiter = asyncio.create_task(
        channel._await_privacy_confirmation_reply("stopping-1", timeout=10.0)
    )
    await asyncio.sleep(0)

    channel._running = True
    await channel.stop()

    result = await asyncio.wait_for(waiter, timeout=1.0)
    assert result is None
    assert channel._pending_confirmations == {}
