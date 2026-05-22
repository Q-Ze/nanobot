"""GateKeeper facade — single entry point used by AgentLoop.

Per `.agent/privacy_gatekeeper.md` §5.1, AgentLoop calls
``detect_and_recommend`` → ``confirm`` → ``transform`` → (cloud) → ``restore``.

As of M3 step 6 every stage is implemented end-to-end. The METRIC_DP
path threads through the full pipeline:

  detector → decider (with live ε budget check via accountant)
           → confirmation
           → transform (embed + Laplace noise + nearest-neighbour
                         candidate; accountant.consume on success)
           → cloud LLM (sees only anonymized text)
           → restore (Restorer substitutes mapping back)
"""

from __future__ import annotations

from dataclasses import replace

from loguru import logger

from nanobot.config.schema import PrivacyConfig
from nanobot.privacy.accountant import BudgetExceeded, PrivacyAccountant
from nanobot.privacy.audit import AuditLogger
from nanobot.privacy.confirmation import ConfirmationGate
from nanobot.privacy.decider import DeciderInputs, ExecutionDecider
from nanobot.privacy.detector import PrivacyEntityDetector
from nanobot.privacy.k_decoy import KDecoyTransform, resolve_hmac_key
from nanobot.privacy.local_model import LocalModelBackend
from nanobot.privacy.local_model import get_default as get_default_backend
from nanobot.privacy.restorer import Restorer
from nanobot.privacy.semantic_detector import LLMSemanticDetector, is_available_backend
from nanobot.privacy.transform import MetricDPTransform
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
        transform: MetricDPTransform | None = None,
        accountant: PrivacyAccountant | None = None,
        restorer: Restorer | None = None,
        k_decoy: KDecoyTransform | None = None,
        eps_query: float = 8.0,
        local_model_available: bool = False,
        k_decoy_supported: bool = False,
    ) -> None:
        self._detector = detector
        self._decider = decider
        self._confirmation = confirmation
        self._audit = audit
        self._local_model: LocalModelBackend = local_model or get_default_backend()
        self._transform = transform
        self._accountant = accountant
        self._restorer = restorer or Restorer()
        self._k_decoy = k_decoy
        self._eps_query = float(eps_query)
        self._static_caps = DeciderInputs(
            entities=(),
            local_model_available=local_model_available and self._local_model.is_available(),
            k_decoy_supported=k_decoy_supported and k_decoy is not None,
            metric_dp_supported=False,  # decided per-call in detect_and_recommend
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

        When ``local_model`` resolves to a usable embedding backend, both
        :class:`LLMSemanticDetector` (recall booster) and
        :class:`MetricDPTransform` (the anonymisation engine itself) are
        auto-wired so users only need to configure ``privacy.local_model``
        and ``privacy.embedding_model`` once.
        """
        # Auto-wire the LM-backed semantic detector when a usable backend
        # is present and the caller didn't supply one explicitly.
        if semantic_detector is None and is_available_backend(local_model):
            semantic_detector = LLMSemanticDetector(
                backend=local_model,
                risk_class_overrides=config.risk_class_overrides,
                timeout_seconds=config.semantic_timeout_seconds,
            )
        detector = PrivacyEntityDetector(
            risk_class_overrides=config.risk_class_overrides,
            regex_extensions=config.regex_extensions,
            semantic=semantic_detector,
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

        # M3 wiring — accountant + transform + restorer.
        from pathlib import Path

        accountant = PrivacyAccountant(
            eps_session_max=config.metric_dp.eps_session_max,
            eps_user_24h_max=config.metric_dp.eps_user_24h_max,
            persist_path=Path(config.audit.log_dir).expanduser() / "budget.json",
        )
        transform_engine: MetricDPTransform | None = None
        if is_available_backend(local_model):
            transform_engine = MetricDPTransform(
                local_model,
                epsilon=config.metric_dp.eps_query,
            )
        restorer = Restorer()

        # M2 wiring — KDecoyTransform doesn't need a backend, so it's
        # always available as a fallback for METRIC_DP. The HMAC key is
        # resolved from env / persisted file per the config-specified source.
        k_decoy_engine: KDecoyTransform | None = None
        if config.k_decoy.enabled:
            try:
                hmac_key = resolve_hmac_key(
                    source=config.pseudonym_key_source,
                    fallback_path=str(
                        Path(config.audit.log_dir).expanduser() / "pseudo_key"
                    ),
                )
                k_decoy_engine = KDecoyTransform(
                    hmac_key=hmac_key,
                    k_target=max(2, int(config.k_decoy.k_max)),
                )
            except (ValueError, OSError) as exc:
                # Misconfiguration must not crash the agent loop — fall back
                # to no K_DECOY path and let the decider pick BLOCKED for MEDIUM.
                logger.warning("privacy.k_decoy disabled: {}", exc)
                k_decoy_engine = None

        return cls(
            detector=detector,
            decider=decider,
            confirmation=confirmation,
            audit=audit,
            local_model=local_model,
            transform=transform_engine,
            accountant=accountant,
            restorer=restorer,
            k_decoy=k_decoy_engine,
            eps_query=config.metric_dp.eps_query,
            local_model_available=False,  # SIMPLE path still unimplemented
            k_decoy_supported=k_decoy_engine is not None,
        )

    # --- pipeline ----------------------------------------------------------------------

    async def detect_and_recommend(
        self,
        raw_message: str,
        *,
        session_key: str = "",
        user_id: str = "",
    ) -> Recommendation:
        """Detect entities and decide a recommended execution path.

        ``metric_dp_supported`` is computed per-call: it's True iff we have
        a transform engine, a usable backend, AND the accountant (if any)
        still has budget for one query of size ``eps_query``. Otherwise
        the decider routes around METRIC_DP — typically falling back to
        BLOCKED when MEDIUM/HIGH entities are present.
        """
        entities = await self._detector.detect(raw_message)
        backend_ok = self._local_model.is_available()
        transform_ok = self._transform is not None
        budget_ok = (
            self._accountant is None
            or self._accountant.can_afford(session_key, user_id, self._eps_query)
        )
        metric_dp_supported = bool(transform_ok and backend_ok and budget_ok)
        inputs = replace(
            self._static_caps,
            entities=tuple(entities),
            metric_dp_supported=metric_dp_supported,
        )
        recommendation = self._decider.decide(inputs)
        # If decider blocked solely because METRIC_DP is out of budget
        # (transform+backend exist, accountant refused), replace the
        # generic "no path available" reason with a budget-specific one
        # so the user sees *why* and roughly when it'll refresh.
        if (
            recommendation.path == ExecutionPath.BLOCKED
            and recommendation.reason == "no_anonymization_path_available_yet"
            and transform_ok
            and backend_ok
            and self._accountant is not None
            and not budget_ok
        ):
            recommendation = replace(
                recommendation,
                reason=_format_budget_exhausted_reason(
                    self._accountant, session_key, user_id
                ),
            )
        # Visible breadcrumb showing detector + decider outcome.
        # We log counts and types but never raw entity values.
        type_counts: dict[str, int] = {}
        for e in entities:
            type_counts[e.type.value] = type_counts.get(e.type.value, 0) + 1
        logger.info(
            "privacy.detect: {} entit{} {}; decider → {} (reason: {})",
            len(entities),
            "y" if len(entities) == 1 else "ies",
            dict(type_counts) if type_counts else "{}",
            recommendation.path.value,
            recommendation.reason,
        )
        return recommendation

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

    async def transform(
        self,
        decision: Decision,
        raw_message: str,
        *,
        session_key: str = "",
        user_id: str = "",
    ) -> TransformOutcome:
        """Apply the path's transform.

        * BLOCKED → refusal message, no cloud call.
        * NORMAL → original message forwarded as-is.
        * METRIC_DP → MetricDPTransform runs; on success consumes ε from
          the accountant and stashes the restoration mapping for
          :meth:`restore` to use later.
        * K_DECOY → KDecoyTransform runs; deterministic per-session
          pseudonyms from the typed pool; no ε consumed.
        * SIMPLE → not implemented; defensively refuse.
        """
        view = self._audit.build_view(decision, decision.recommendation.entities)
        if decision.path == ExecutionPath.BLOCKED:
            return TransformOutcome(
                decision=decision,
                privacy_message=decision.refusal_message or _default_refusal(decision),
                audit_view=view,
            )
        if decision.path == ExecutionPath.NORMAL:
            return TransformOutcome(
                decision=decision, privacy_message=raw_message, audit_view=view
            )
        if decision.path == ExecutionPath.METRIC_DP and self._transform is not None:
            mdp = await self._transform.transform(
                raw_message, list(decision.recommendation.entities)
            )
            # Pay the budget. consume() can still raise even after can_afford
            # passed at decision time if multiple turns race the same budget.
            if self._accountant is not None and mdp.eps_consumed > 0:
                try:
                    self._accountant.consume(session_key, user_id, mdp.eps_consumed)
                except BudgetExceeded as exc:
                    logger.warning(
                        "privacy.metric_dp: post-transform budget overflow ({}); "
                        "blocking the turn.", exc,
                    )
                    return TransformOutcome(
                        decision=decision,
                        privacy_message=(
                            "Privacy budget exceeded; the message was blocked "
                            "to avoid weakening the cumulative ε guarantee."
                        ),
                        audit_view=replace(view, eps_consumed=mdp.eps_consumed),
                    )
            # Record the mapping so .restore() can reverse it on the way back.
            restoration_plan = {
                "mapping": dict(mdp.mapping),
                "failed_entity_types": [e.type.value for e in mdp.failures],
            }
            # Visible breadcrumb (no raw values; only counts + ε).
            logger.info(
                "privacy.metric_dp: anonymized {} entit{} (ε={:.1f} spent); "
                "{} placeholders for unembeddable values",
                len(mdp.mapping),
                "y" if len(mdp.mapping) == 1 else "ies",
                mdp.eps_consumed,
                len(mdp.failures),
            )
            new_view = replace(
                view,
                fidelity="RESTORED_LOSSY",
                eps_consumed=mdp.eps_consumed,
            )
            return TransformOutcome(
                decision=decision,
                privacy_message=mdp.anonymized_text,
                restoration_plan=restoration_plan,
                audit_view=new_view,
            )
        if decision.path == ExecutionPath.K_DECOY and self._k_decoy is not None:
            kdr = await self._k_decoy.transform(
                raw_message,
                list(decision.recommendation.entities),
                session_key=session_key,
            )
            restoration_plan = {
                "mapping": dict(kdr.mapping),
                "failed_entity_types": [e.type.value for e in kdr.failures],
            }
            logger.info(
                "privacy.k_decoy: pseudonymized {} entit{} (K_eff={}); "
                "{} placeholders for hard-block or empty-pool entities",
                len(kdr.mapping),
                "y" if len(kdr.mapping) == 1 else "ies",
                kdr.k_effective,
                len(kdr.failures),
            )
            new_view = replace(view, fidelity="RESTORED_LOSSY", eps_consumed=0.0)
            return TransformOutcome(
                decision=decision,
                privacy_message=kdr.anonymized_text,
                restoration_plan=restoration_plan,
                audit_view=new_view,
            )
        # SIMPLE or a path whose engine isn't wired up — defensively refuse
        # rather than silently forwarding plaintext.
        return TransformOutcome(
            decision=decision,
            privacy_message=(
                f"Privacy GateKeeper: path '{decision.path.value}' is not implemented yet "
                "(see docs/privacy-gatekeeper-m1.md §5). Message blocked."
            ),
            audit_view=view,
        )

    async def restore(self, response: str, outcome: TransformOutcome) -> str:
        """Reverse any METRIC_DP substitutions on the cloud LLM's response.

        No-op for NORMAL / BLOCKED outcomes — they didn't anonymize
        anything in the first place.
        """
        mapping = (outcome.restoration_plan or {}).get("mapping") if outcome else None
        if not mapping:
            return response
        result = await self._restorer.restore(response, mapping)
        # Visible breadcrumb: how many pseudonyms the cloud actually echoed
        # back, and how many we didn't find.
        logger.info(
            "privacy.restore: substituted {} pseudonym occurrence{} back to original; "
            "{} mapped value{} did not appear in the reply",
            result.replacements_applied,
            "" if result.replacements_applied == 1 else "s",
            len(result.unmatched_keys),
            "" if len(result.unmatched_keys) == 1 else "s",
        )
        return result.restored_text

    # --- audit / introspection ---------------------------------------------------------

    def record_audit(
        self,
        *,
        session_key: str,
        decision: Decision,
        view: AuditView | None = None,
    ) -> None:
        self._audit.record(
            session_key=session_key,
            decision=decision,
            entities=decision.recommendation.entities,
            view=view,
        )

    def reset_session_budget(self, session_key: str) -> None:
        """Called when a chat session ends so its ε counter starts fresh."""
        if self._accountant is not None:
            self._accountant.reset_session(session_key)


def _default_refusal(decision: Decision) -> str:
    reason = decision.recommendation.reason
    return (
        "Privacy GateKeeper blocked this message. "
        f"Reason: {reason}. Please remove the sensitive content and try again."
    )


def _format_budget_exhausted_reason(
    accountant: PrivacyAccountant, session_key: str, user_id: str
) -> str:
    """Build a reason string explaining that the ε accountant is out of budget.

    Mentions the 24-h cap (the one most users will hit during testing)
    and the soonest moment any ε frees up. The session counter is
    included only when it's the limiting factor — otherwise the
    24-h line is the actionable one.
    """
    snap = accountant.snapshot(session_key, user_id)
    refresh_str = _humanize_seconds(accountant.time_until_next_refresh(user_id))
    parts = [
        f"used {snap.eps_user_24h_used:.1f}/{snap.eps_user_24h_max:.1f} ε in 24h"
    ]
    if snap.remaining_session <= 0 < snap.remaining_user_24h:
        parts.append(
            f"session {snap.eps_session_used:.1f}/{snap.eps_session_max:.1f} ε"
        )
    if refresh_str:
        parts.append(f"next refresh in {refresh_str}")
    return "eps_budget_exhausted (" + ", ".join(parts) + ")"


def _humanize_seconds(seconds: float | None) -> str:
    if seconds is None or seconds <= 0:
        return ""
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    return f"{seconds / 3600:.1f}h"
