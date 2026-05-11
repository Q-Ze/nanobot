"""ConfirmationGate unit tests — mode logic, preselect floor, timeout, fallback."""

from __future__ import annotations

from nanobot.privacy.confirmation import ConfirmationGate
from nanobot.privacy.types import (
    ChannelCapabilities,
    ChannelFallback,
    ConfirmationMode,
    ConfirmationPrompt,
    ConfirmationReply,
    DetectedEntity,
    EntityType,
    ExecutionPath,
    Linkability,
    PathSource,
    Recommendation,
    RiskClass,
)


def _rec(path: ExecutionPath, allowed=None, entities=()) -> Recommendation:
    if allowed is None:
        allowed = frozenset({path, ExecutionPath.BLOCKED})
    return Recommendation(path=path, allowed=allowed, reason="t", entities=tuple(entities))


def _high_entity() -> DetectedEntity:
    return DetectedEntity(
        type=EntityType.ID_NUMBER,
        span=(0, 0),
        value="",
        risk_class=RiskClass.HIGH,
        linkability=Linkability.SINGLE_USE,
    )


def _gate(**overrides) -> ConfirmationGate:
    base = dict(
        mode=ConfirmationMode.RISK_THRESHOLD,
        risk_threshold=RiskClass.HIGH,
        timeout_seconds=10,
        on_timeout="block",
        channel_fallback_default=ChannelFallback.FORCED_CONSERVATIVE,
    )
    base.update(overrides)
    return ConfirmationGate(**base)


def _caps_no_interactive() -> ChannelCapabilities:
    return ChannelCapabilities(supports_interactive_confirm=False)


# --- mode: never ----------------------------------------------------------------------


async def test_mode_never_skips_confirmation():
    gate = _gate(mode=ConfirmationMode.NEVER)
    rec = _rec(ExecutionPath.BLOCKED, entities=[_high_entity()])
    d = await gate.confirm(rec, chat_id="c1", channel_name="x", capabilities=_caps_no_interactive())
    assert d.path == ExecutionPath.BLOCKED
    assert d.source == PathSource.SYSTEM_AUTO


# --- mode: risk_threshold -------------------------------------------------------------


async def test_threshold_does_not_ask_when_normal_path_and_no_entities():
    gate = _gate()
    rec = _rec(ExecutionPath.NORMAL, allowed=frozenset({ExecutionPath.NORMAL, ExecutionPath.BLOCKED}))
    d = await gate.confirm(rec, chat_id="c1", channel_name="x", capabilities=_caps_no_interactive())
    assert d.path == ExecutionPath.NORMAL
    assert d.source == PathSource.SYSTEM_AUTO


async def test_threshold_asks_when_recommendation_is_blocked():
    gate = _gate()
    rec = _rec(ExecutionPath.BLOCKED, allowed=frozenset({ExecutionPath.BLOCKED}))
    d = await gate.confirm(rec, chat_id="c1", channel_name="x", capabilities=_caps_no_interactive())
    # No interactive support → forced_conservative → BLOCKED (strictest in allowed).
    assert d.path == ExecutionPath.BLOCKED
    assert d.source == PathSource.FALLBACK_NO_CONFIRM


# --- preselect with floor enforcement -------------------------------------------------


async def test_preselect_violation_falls_back_to_recommendation_silently():
    gate = _gate(mode=ConfirmationMode.NEVER)
    rec = _rec(ExecutionPath.BLOCKED, allowed=frozenset({ExecutionPath.BLOCKED}))
    d = await gate.confirm(
        rec,
        chat_id="c1",
        channel_name="x",
        capabilities=_caps_no_interactive(),
        user_path_preference=ExecutionPath.NORMAL,  # invalid: violates floor
    )
    assert d.path == ExecutionPath.BLOCKED
    assert d.source == PathSource.SYSTEM_AUTO
    assert d.violation_attempt == ExecutionPath.NORMAL


async def test_preselect_valid_is_honoured_in_non_always_mode():
    gate = _gate(mode=ConfirmationMode.NEVER)
    rec = _rec(
        ExecutionPath.NORMAL,
        allowed=frozenset({ExecutionPath.NORMAL, ExecutionPath.BLOCKED}),
    )
    d = await gate.confirm(
        rec,
        chat_id="c1",
        channel_name="x",
        capabilities=_caps_no_interactive(),
        user_path_preference=ExecutionPath.BLOCKED,
    )
    assert d.path == ExecutionPath.BLOCKED
    assert d.source == PathSource.USER_PRESELECTED


# --- interactive ask: success / cancel / timeout ---------------------------------------


def _interactive_caps(reply: ConfirmationReply | None = None, raise_timeout: bool = False) -> ChannelCapabilities:
    sent: dict[str, ConfirmationPrompt] = {}

    async def send(chat_id, prompt: ConfirmationPrompt):
        sent[prompt.confirmation_id] = prompt

    async def wait(confirmation_id: str, timeout: float):
        if raise_timeout:
            import asyncio

            await asyncio.sleep(timeout + 1.0)  # noqa: ASYNC101
            return None
        if reply is None:
            return None
        # Return the reply with the actual confirmation id from the prompt.
        prompt = next(iter(sent.values()))
        return ConfirmationReply(confirmation_id=prompt.confirmation_id, chosen_path=reply.chosen_path)

    return ChannelCapabilities(
        supports_interactive_confirm=True,
        confirmation_max_latency_seconds=5,
        send_confirmation=send,
        await_confirmation_reply=wait,
    )


async def test_interactive_user_chooses_blocked():
    gate = _gate()
    rec = _rec(
        ExecutionPath.NORMAL,
        allowed=frozenset({ExecutionPath.NORMAL, ExecutionPath.BLOCKED}),
        entities=[_high_entity()],
    )
    caps = _interactive_caps(reply=ConfirmationReply(confirmation_id="", chosen_path=ExecutionPath.BLOCKED))
    d = await gate.confirm(rec, chat_id="c", channel_name="x", capabilities=caps)
    assert d.path == ExecutionPath.BLOCKED
    assert d.source == PathSource.USER_CONFIRMED


async def test_interactive_user_cancel_becomes_blocked():
    gate = _gate()
    rec = _rec(ExecutionPath.NORMAL, allowed=frozenset({ExecutionPath.NORMAL, ExecutionPath.BLOCKED}), entities=[_high_entity()])
    caps = _interactive_caps(reply=ConfirmationReply(confirmation_id="", chosen_path=None))
    d = await gate.confirm(rec, chat_id="c", channel_name="x", capabilities=caps)
    assert d.path == ExecutionPath.BLOCKED
    assert d.source == PathSource.USER_CONFIRMED
    assert "cancel" in d.refusal_message.lower()


async def test_interactive_timeout_fails_closed():
    gate = _gate(timeout_seconds=1)
    rec = _rec(ExecutionPath.NORMAL, allowed=frozenset({ExecutionPath.NORMAL, ExecutionPath.BLOCKED}), entities=[_high_entity()])
    caps = _interactive_caps(raise_timeout=True)
    caps.confirmation_max_latency_seconds = 1
    d = await gate.confirm(rec, chat_id="c", channel_name="x", capabilities=caps)
    assert d.path == ExecutionPath.BLOCKED
    assert d.source == PathSource.FALLBACK_TIMEOUT


async def test_interactive_timeout_can_fall_through_to_recommended():
    gate = _gate(timeout_seconds=1, on_timeout="recommended")
    rec = _rec(ExecutionPath.NORMAL, allowed=frozenset({ExecutionPath.NORMAL, ExecutionPath.BLOCKED}), entities=[_high_entity()])
    caps = _interactive_caps(raise_timeout=True)
    caps.confirmation_max_latency_seconds = 1
    d = await gate.confirm(rec, chat_id="c", channel_name="x", capabilities=caps)
    assert d.path == ExecutionPath.NORMAL
    assert d.source == PathSource.FALLBACK_TIMEOUT


# --- channel fallback overrides --------------------------------------------------------


async def test_channel_override_use_recommended():
    gate = _gate(
        channel_fallback_overrides={"webhook": ChannelFallback.USE_RECOMMENDED},
    )
    rec = _rec(ExecutionPath.NORMAL, allowed=frozenset({ExecutionPath.NORMAL, ExecutionPath.BLOCKED}), entities=[_high_entity()])
    d = await gate.confirm(rec, chat_id="c", channel_name="webhook", capabilities=_caps_no_interactive())
    assert d.path == ExecutionPath.NORMAL
    assert d.source == PathSource.FALLBACK_NO_CONFIRM


async def test_channel_override_reject():
    gate = _gate(
        channel_fallback_overrides={"email_bridge": ChannelFallback.REJECT},
    )
    rec = _rec(ExecutionPath.NORMAL, allowed=frozenset({ExecutionPath.NORMAL, ExecutionPath.BLOCKED}), entities=[_high_entity()])
    d = await gate.confirm(rec, chat_id="c", channel_name="email_bridge", capabilities=_caps_no_interactive())
    assert d.path == ExecutionPath.BLOCKED
    assert d.source == PathSource.FALLBACK_NO_CONFIRM
    assert "rejected" in d.refusal_message.lower()


# --- always mode + preselect: still asks ----------------------------------------------


async def test_always_mode_asks_even_when_preselect_provided():
    gate = _gate(mode=ConfirmationMode.ALWAYS)
    rec = _rec(ExecutionPath.NORMAL, allowed=frozenset({ExecutionPath.NORMAL, ExecutionPath.BLOCKED}))
    caps = _interactive_caps(reply=ConfirmationReply(confirmation_id="", chosen_path=ExecutionPath.NORMAL))
    d = await gate.confirm(
        rec,
        chat_id="c",
        channel_name="x",
        capabilities=caps,
        user_path_preference=ExecutionPath.NORMAL,
    )
    # User confirms — source is USER_CONFIRMED (not USER_PRESELECTED).
    assert d.source == PathSource.USER_CONFIRMED
