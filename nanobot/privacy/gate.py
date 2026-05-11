"""GateKeeper facade — single entry point used by AgentLoop.

Per `.agent/privacy_gatekeeper.md` §5.1, AgentLoop calls `detect_and_recommend`,
then `confirm`, then `transform`, then `restore`. M1 implements the first three
non-trivially; `transform` is passthrough for SIMPLE/NORMAL/BLOCKED, and
`restore` is a no-op (the lossless paths don't need restoration).
"""

from __future__ import annotations

from dataclasses import replace

from nanobot.config.schema import PrivacyConfig
from nanobot.privacy.audit import AuditLogger
from nanobot.privacy.confirmation import ConfirmationGate
from nanobot.privacy.decider import DeciderInputs, ExecutionDecider
from nanobot.privacy.detector import PrivacyEntityDetector
from nanobot.privacy.local_model import LocalModelBackend
from nanobot.privacy.local_model import get_default as get_default_backend
from nanobot.privacy.types import (
    AuditView,
    ChannelCapabilities,
    ChannelFallback,
    ConfirmationMode,
    Decision,
    ExecutionPath,
    Recommendation,
    RiskClass,
    TransformOutcome,
)


class GateKeeper:
    """Coordinates detector → decider → confirmation → transform → restore."""

    def __init__(
        self,
        *,
        detector: PrivacyEntityDetector,
        decider: ExecutionDecider,
        confirmation: ConfirmationGate,
        audit: AuditLogger,
        local_model: LocalModelBackend | None = None,
        local_model_available: bool = False,
        k_decoy_supported: bool = False,
        metric_dp_supported: bool = False,
    ) -> None:
        self._detector = detector
        self._decider = decider
        self._confirmation = confirmation
        self._audit = audit
        self._local_model: LocalModelBackend = local_model or get_default_backend()
        self._caps = DeciderInputs(
            entities=(),
            local_model_available=local_model_available and self._local_model.is_available(),
            k_decoy_supported=k_decoy_supported,
            metric_dp_supported=metric_dp_supported,
        )

    @classmethod
    def from_config(
        cls,
        config: PrivacyConfig,
        *,
        semantic_detector: "object | None" = None,
        local_model: LocalModelBackend | None = None,
    ) -> "GateKeeper":
        """Build a GateKeeper from PrivacyConfig.

        Pass ``semantic_detector`` to plug in a local-LM-backed second pass
        (must implement :class:`nanobot.privacy.detector.SemanticDetector`).
        Pass ``local_model`` to override the process-default
        :class:`LocalModelBackend` for this gate.

        M1.5 defaults to NoopSemanticDetector + NullLocalModel — both are
        no-ops, so K_DECOY / METRIC_DP / SIMPLE remain unavailable.
        """
        detector = PrivacyEntityDetector(
            risk_class_overrides=config.risk_class_overrides,
            regex_extensions=config.regex_extensions,
            semantic=semantic_detector,  # None falls back to NoopSemanticDetector inside the class
        )
        decider = ExecutionDecider()
        confirm_cfg = config.confirmation
        confirmation = ConfirmationGate(
            mode=ConfirmationMode(confirm_cfg.mode),
            risk_threshold=RiskClass(confirm_cfg.risk_threshold),
            timeout_seconds=confirm_cfg.timeout_seconds,
            on_timeout=confirm_cfg.on_timeout,
            channel_fallback_default=ChannelFallback(confirm_cfg.channel_fallback_default),
            channel_fallback_overrides={
                k: ChannelFallback(v) for k, v in confirm_cfg.channel_fallback_overrides.items()
            },
        )
        audit = AuditLogger(log_dir=config.audit.log_dir, enabled=config.audit.enabled)
        return cls(
            detector=detector,
            decider=decider,
            confirmation=confirmation,
            audit=audit,
            local_model=local_model,
            local_model_available=False,  # M1.5: SIMPLE/local-only execution not yet wired
            k_decoy_supported=False,   # M2
            metric_dp_supported=False, # M3
        )

    # --- pipeline ----------------------------------------------------------------------

    async def detect_and_recommend(self, raw_message: str) -> Recommendation:
        entities = await self._detector.detect(raw_message)
        inputs = replace(self._caps, entities=tuple(entities))
        return self._decider.decide(inputs)

    async def confirm(
        self,
        recommendation: Recommendation,
        *,
        chat_id: str,
        channel_name: str,
        capabilities: ChannelCapabilities,
        user_path_preference: ExecutionPath | None = None,
    ) -> Decision:
        return await self._confirmation.confirm(
            recommendation,
            chat_id=chat_id,
            channel_name=channel_name,
            capabilities=capabilities,
            user_path_preference=user_path_preference,
        )

    def transform(self, decision: Decision, raw_message: str) -> TransformOutcome:
        """M1.5 paths: NORMAL forwards as-is, BLOCKED replaces with a refusal.

        SIMPLE / K_DECOY / METRIC_DP are accepted by the type system but not
        yet implemented. Hitting them at runtime indicates a config or
        decider bug — we defensively return a refusal rather than silently
        forwarding plaintext to the cloud (which would defeat the gate's
        purpose).
        """
        view = self._audit.build_view(decision, decision.recommendation.entities)
        if decision.path == ExecutionPath.BLOCKED:
            return TransformOutcome(
                decision=decision,
                privacy_message=decision.refusal_message or _default_refusal(decision),
                audit_view=view,
            )
        if decision.path == ExecutionPath.NORMAL:
            return TransformOutcome(decision=decision, privacy_message=raw_message, audit_view=view)
        # SIMPLE / K_DECOY / METRIC_DP — not yet implemented in M1.5.
        return TransformOutcome(
            decision=decision,
            privacy_message=(
                f"Privacy GateKeeper: path '{decision.path.value}' is not implemented yet "
                "(see docs/privacy-gatekeeper-m1.md §5). Message blocked."
            ),
            audit_view=view,
        )

    async def restore(self, response: str, outcome: TransformOutcome) -> str:
        """No-op for M1 paths."""
        return response

    # --- audit -------------------------------------------------------------------------

    def record_audit(self, *, session_key: str, decision: Decision, view: AuditView | None = None) -> None:
        self._audit.record(
            session_key=session_key,
            decision=decision,
            entities=decision.recommendation.entities,
            view=view,
        )


def _default_refusal(decision: Decision) -> str:
    reason = decision.recommendation.reason
    return (
        "Privacy GateKeeper blocked this message. "
        f"Reason: {reason}. Please remove the sensitive content and try again."
    )
