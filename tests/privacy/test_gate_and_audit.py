"""AuditLogger + GateKeeper facade tests."""

from __future__ import annotations

import json
from pathlib import Path

from nanobot.config.schema import PrivacyConfig
from nanobot.privacy.audit import AuditLogger
from nanobot.privacy.gate import GateKeeper
from nanobot.privacy.types import (
    ChannelCapabilities,
    Decision,
    DetectedEntity,
    EntityType,
    ExecutionPath,
    Linkability,
    PathSource,
    Recommendation,
    RiskClass,
)


def _entity() -> DetectedEntity:
    return DetectedEntity(
        type=EntityType.EMAIL,
        span=(0, 5),
        value="x@y.z",
        risk_class=RiskClass.MEDIUM,
        linkability=Linkability.SINGLE_USE,
    )


def test_audit_writes_jsonl_with_metadata_only(tmp_path: Path):
    logger = AuditLogger(log_dir=tmp_path, enabled=True)
    rec = Recommendation(
        path=ExecutionPath.BLOCKED,
        allowed=frozenset({ExecutionPath.BLOCKED}),
        reason="hard_secret",
        entities=(_entity(),),
    )
    decision = Decision(path=ExecutionPath.BLOCKED, source=PathSource.SYSTEM_AUTO, recommendation=rec)
    logger.record(session_key="alice:web", decision=decision, entities=rec.entities)
    files = list(tmp_path.glob("audit-*.jsonl"))
    assert len(files) == 1
    line = files[0].read_text().strip().splitlines()[0]
    rec_obj = json.loads(line)
    assert rec_obj["path"] == "blocked"
    assert rec_obj["recommended_path"] == "blocked"
    assert rec_obj["entity_counts"] == {"email": 1}
    # Critically: original entity value MUST NOT appear in the audit line.
    assert "x@y.z" not in line


def test_audit_skipped_when_disabled(tmp_path: Path):
    logger = AuditLogger(log_dir=tmp_path, enabled=False)
    rec = Recommendation(path=ExecutionPath.NORMAL, allowed=frozenset({ExecutionPath.NORMAL}), reason="t")
    decision = Decision(path=ExecutionPath.NORMAL, source=PathSource.SYSTEM_AUTO, recommendation=rec)
    logger.record(session_key="k", decision=decision, entities=())
    assert not list(tmp_path.glob("*.jsonl"))


async def test_gatekeeper_facade_blocks_credential(tmp_path: Path):
    cfg = PrivacyConfig(enabled=True)
    cfg.audit.log_dir = str(tmp_path)
    gate = GateKeeper.from_config(cfg)
    rec = await gate.detect_and_recommend(
        "Please use this key: AKIAIOSFODNN7EXAMPLE for the AWS upload."
    )
    assert rec.path == ExecutionPath.BLOCKED
    assert rec.allowed == frozenset({ExecutionPath.BLOCKED})

    decision = await gate.confirm(
        rec,
        chat_id="c",
        channel_name="x",
        capabilities=ChannelCapabilities(),
    )
    assert decision.path == ExecutionPath.BLOCKED
    outcome = await gate.transform(decision, "...")
    assert isinstance(outcome.privacy_message, str)
    assert outcome.audit_view is not None
    gate.record_audit(session_key="alice", decision=decision, view=outcome.audit_view)

    files = list(tmp_path.glob("audit-*.jsonl"))
    assert len(files) == 1


async def test_gatekeeper_facade_passes_low_risk(tmp_path: Path):
    cfg = PrivacyConfig(enabled=True)
    cfg.audit.log_dir = str(tmp_path)
    gate = GateKeeper.from_config(cfg)
    rec = await gate.detect_and_recommend("Hello, how are you today?")
    assert rec.path == ExecutionPath.NORMAL
    decision = await gate.confirm(rec, chat_id="c", channel_name="x", capabilities=ChannelCapabilities())
    assert decision.path == ExecutionPath.NORMAL
    outcome = await gate.transform(decision, "Hello, how are you today?")
    assert outcome.privacy_message == "Hello, how are you today?"


async def test_gatekeeper_routes_medium_pii_to_k_decoy_in_m2(tmp_path: Path):
    """M2 shipped K_DECOY, which the decider prefers over METRIC_DP for MEDIUM
    entities (saves the ε budget for HIGH risk). Email is MEDIUM, so the
    recommendation should be K_DECOY — not BLOCKED (M1 default) nor
    METRIC_DP (which the decider holds in reserve)."""
    cfg = PrivacyConfig(enabled=True)
    cfg.audit.log_dir = str(tmp_path)
    # Confirmation off so we observe the system recommendation directly.
    cfg.confirmation.mode = "never"
    gate = GateKeeper.from_config(cfg)
    rec = await gate.detect_and_recommend("Email me at alice@example.com")
    assert rec.path == ExecutionPath.K_DECOY
    # BLOCKED and METRIC_DP-when-available should both be in the allowed
    # set so the user can escalate via the confirmation prompt.
    assert ExecutionPath.BLOCKED in rec.allowed


async def test_restore_is_identity_for_m1_paths(tmp_path: Path):
    cfg = PrivacyConfig(enabled=True)
    cfg.audit.log_dir = str(tmp_path)
    gate = GateKeeper.from_config(cfg)
    rec = await gate.detect_and_recommend("Hello world")
    decision = await gate.confirm(rec, chat_id="c", channel_name="x", capabilities=ChannelCapabilities())
    outcome = await gate.transform(decision, "Hello world")
    restored = await gate.restore("LLM reply", outcome)
    assert restored == "LLM reply"


async def test_simple_path_is_not_exposed_in_m1_5(tmp_path: Path):
    """M1.5 hard-disables SIMPLE in the allowed set even if a backend is wired."""
    cfg = PrivacyConfig(enabled=True)
    cfg.audit.log_dir = str(tmp_path)
    # local_model defaults to PrivacyLocalModelConfig(provider="null") — already
    # unavailable. SIMPLE must still be absent from the allowed set.
    gate = GateKeeper.from_config(cfg)
    rec = await gate.detect_and_recommend("Hello, my name is Alice.")
    assert ExecutionPath.SIMPLE not in rec.allowed


async def test_transform_refuses_unimplemented_paths(tmp_path: Path):
    """Defensive: SIMPLE has no implementation. If it somehow reaches
    transform (a buggy decider override, a future config), refuse rather
    than silently forward plaintext. K_DECOY and METRIC_DP now have
    transforms, so they're no longer the canary here."""
    cfg = PrivacyConfig(enabled=True)
    cfg.audit.log_dir = str(tmp_path)
    gate = GateKeeper.from_config(cfg)
    rec = Recommendation(
        path=ExecutionPath.SIMPLE,
        allowed=frozenset({ExecutionPath.SIMPLE, ExecutionPath.BLOCKED}),
        reason="t",
        entities=(_entity(),),
    )
    decision = Decision(path=ExecutionPath.SIMPLE, source=PathSource.SYSTEM_AUTO, recommendation=rec)
    outcome = await gate.transform(decision, "raw text")
    assert "not implemented" in outcome.privacy_message.lower()
