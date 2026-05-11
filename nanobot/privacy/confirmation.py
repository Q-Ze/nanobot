"""ConfirmationGate — interactive user confirmation per `.agent/privacy_gatekeeper.md` §3.7.

Responsibilities:
1. Honour user's pre-selected path when supplied (subject to safety floor).
2. Decide whether to ask the user (mode + risk threshold).
3. If asking, send a prompt via the channel and await reply with timeout.
4. If channel can't ask, apply the configured fallback policy.
5. Never let the user override below the AllowedPathSet floor.
"""

from __future__ import annotations

import asyncio
import secrets
import time

from nanobot.privacy.types import (
    ChannelCapabilities,
    ChannelFallback,
    ConfirmationMode,
    ConfirmationPrompt,
    Decision,
    ExecutionPath,
    PathSource,
    Recommendation,
    RiskClass,
    is_at_least_as_strict,
    risk_at_least,
)


class ConfirmationGate:
    def __init__(
        self,
        *,
        mode: ConfirmationMode,
        risk_threshold: RiskClass,
        timeout_seconds: int,
        on_timeout: str,                                # "block" | "recommended"
        channel_fallback_default: ChannelFallback,
        channel_fallback_overrides: dict[str, ChannelFallback] | None = None,
    ) -> None:
        self._mode = mode
        self._risk_threshold = risk_threshold
        self._timeout = timeout_seconds
        self._on_timeout = on_timeout
        self._fallback_default = channel_fallback_default
        self._fallback_overrides = channel_fallback_overrides or {}

    # --- public API --------------------------------------------------------------------

    async def confirm(
        self,
        recommendation: Recommendation,
        *,
        chat_id: str,
        channel_name: str,
        capabilities: ChannelCapabilities,
        user_path_preference: ExecutionPath | None = None,
    ) -> Decision:
        """Run the confirmation flow and return the final Decision."""
        # Step 1: handle user pre-selected path.
        if user_path_preference is not None:
            return await self._handle_preselect(
                recommendation,
                user_path_preference,
                chat_id=chat_id,
                channel_name=channel_name,
                capabilities=capabilities,
            )

        # Step 2: decide whether to ask.
        if not self._should_ask(recommendation):
            return Decision(
                path=recommendation.path,
                source=PathSource.SYSTEM_AUTO,
                recommendation=recommendation,
            )

        # Step 3: ask interactively, or fall back per channel policy.
        if not capabilities.supports_interactive_confirm:
            return self._apply_channel_fallback(recommendation, channel_name)
        return await self._ask_user(recommendation, chat_id=chat_id, capabilities=capabilities)

    # --- step helpers -------------------------------------------------------------------

    async def _handle_preselect(
        self,
        recommendation: Recommendation,
        user_pref: ExecutionPath,
        *,
        chat_id: str,
        channel_name: str,
        capabilities: ChannelCapabilities,
    ) -> Decision:
        if user_pref not in recommendation.allowed:
            # Silent floor enforcement (do not enumerate floor rules to attackers).
            chosen = recommendation.path
            return Decision(
                path=chosen,
                source=PathSource.SYSTEM_AUTO,
                recommendation=recommendation,
                violation_attempt=user_pref,
            )

        # Even with valid preselect, ALWAYS mode still asks (using preselect as default).
        if self._mode == ConfirmationMode.ALWAYS:
            if not capabilities.supports_interactive_confirm:
                return self._apply_channel_fallback(
                    recommendation, channel_name, default=user_pref
                )
            return await self._ask_user(
                recommendation, chat_id=chat_id, capabilities=capabilities, default=user_pref
            )
        return Decision(
            path=user_pref,
            source=PathSource.USER_PRESELECTED,
            recommendation=recommendation,
        )

    def _should_ask(self, recommendation: Recommendation) -> bool:
        if self._mode == ConfirmationMode.NEVER:
            return False
        if self._mode == ConfirmationMode.ALWAYS:
            return True
        # risk_threshold mode: ask when path is non-trivial OR any entity meets threshold.
        non_trivial_path = recommendation.path in {
            ExecutionPath.BLOCKED,
            ExecutionPath.K_DECOY,
            ExecutionPath.METRIC_DP,
        }
        threshold_met = any(
            risk_at_least(e.risk_class, self._risk_threshold) for e in recommendation.entities
        )
        return non_trivial_path or threshold_met

    async def _ask_user(
        self,
        recommendation: Recommendation,
        *,
        chat_id: str,
        capabilities: ChannelCapabilities,
        default: ExecutionPath | None = None,
    ) -> Decision:
        send = capabilities.send_confirmation
        wait = capabilities.await_confirmation_reply
        if send is None or wait is None:
            # Capability bug: declared support but missing callbacks.
            return self._apply_channel_fallback(recommendation, channel_name="(unknown)")

        confirmation_id = secrets.token_hex(16)
        prompt = ConfirmationPrompt(
            confirmation_id=confirmation_id,
            recommendation=recommendation,
            chat_id=chat_id,
        )

        timeout = min(self._timeout, capabilities.confirmation_max_latency_seconds)
        await send(chat_id, prompt)
        t0 = time.perf_counter()
        try:
            reply = await asyncio.wait_for(wait(confirmation_id, float(timeout)), timeout=timeout + 1.0)
        except asyncio.TimeoutError:
            reply = None
        latency_ms = int((time.perf_counter() - t0) * 1000)

        if reply is None or reply.confirmation_id != confirmation_id:
            return self._on_timeout_decision(recommendation, latency_ms)

        chosen = reply.chosen_path
        if chosen is None:
            # User cancelled — treat as BLOCKED with explicit refusal.
            return Decision(
                path=ExecutionPath.BLOCKED,
                source=PathSource.USER_CONFIRMED,
                recommendation=recommendation,
                user_choice_latency_ms=latency_ms,
                refusal_message="User cancelled this message via privacy confirmation.",
            )

        # Floor enforcement on user choice.
        if chosen not in recommendation.allowed:
            return Decision(
                path=recommendation.path,
                source=PathSource.SYSTEM_AUTO,
                recommendation=recommendation,
                user_choice_latency_ms=latency_ms,
                violation_attempt=chosen,
            )

        return Decision(
            path=chosen,
            source=PathSource.USER_CONFIRMED,
            recommendation=recommendation,
            user_choice_latency_ms=latency_ms,
        )

    # --- timeout / fallback ------------------------------------------------------------

    def _on_timeout_decision(self, recommendation: Recommendation, latency_ms: int) -> Decision:
        if self._on_timeout == "recommended":
            return Decision(
                path=recommendation.path,
                source=PathSource.FALLBACK_TIMEOUT,
                recommendation=recommendation,
                user_choice_latency_ms=latency_ms,
            )
        # Fail-closed default
        return Decision(
            path=ExecutionPath.BLOCKED,
            source=PathSource.FALLBACK_TIMEOUT,
            recommendation=recommendation,
            user_choice_latency_ms=latency_ms,
            refusal_message="Privacy confirmation timed out; message blocked (fail-closed).",
        )

    def _apply_channel_fallback(
        self,
        recommendation: Recommendation,
        channel_name: str,
        *,
        default: ExecutionPath | None = None,
    ) -> Decision:
        policy = self._fallback_overrides.get(channel_name, self._fallback_default)
        if policy == ChannelFallback.USE_RECOMMENDED:
            return Decision(
                path=default or recommendation.path,
                source=PathSource.FALLBACK_NO_CONFIRM,
                recommendation=recommendation,
            )
        if policy == ChannelFallback.REJECT:
            return Decision(
                path=ExecutionPath.BLOCKED,
                source=PathSource.FALLBACK_NO_CONFIRM,
                recommendation=recommendation,
                refusal_message=(
                    "This channel does not support privacy confirmation; "
                    "message rejected per policy."
                ),
            )
        # forced_conservative: pick the strictest path in allowed set.
        strictest = max(recommendation.allowed, key=_strictness_key)
        return Decision(
            path=strictest,
            source=PathSource.FALLBACK_NO_CONFIRM,
            recommendation=recommendation,
            refusal_message=(
                "This channel does not support privacy confirmation; "
                "blocked per fail-closed policy."
                if strictest == ExecutionPath.BLOCKED
                else ""
            ),
        )


def _strictness_key(p: ExecutionPath) -> int:
    # Reuse the strictness ordering by querying the helper exhaustively.
    rank = 0
    for q in ExecutionPath:
        if is_at_least_as_strict(p, q):
            rank += 1
    return rank
