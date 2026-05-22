"""End-to-end integration tests for the M3 Metric-DP pipeline.

These tests wire a real :class:`GateKeeper` (built via ``from_config`` so
auto-wiring is exercised) against a *fake* :class:`LocalModelBackend`
that returns deterministic embeddings, then drive a full
detect → confirm → transform → "cloud" → restore loop. No real network
calls.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nanobot.config.schema import PrivacyConfig
from nanobot.privacy.gate import GateKeeper
from nanobot.privacy.types import ChannelCapabilities, ExecutionPath


class _Backend:
    """Deterministic embedding table, large enough to be a plausible
    embedding dim (≥ 8). Values not in the table get a generic vector.
    """

    name = "fake"

    def __init__(self, table: dict[str, list[float]]):
        self._t = table

    def is_available(self) -> bool:
        return True

    async def generate(self, prompt, **_):
        return ""

    async def embed(self, text, **_):
        return list(self._t.get(text, [1.0] + [0.0] * 7))


# Embedding table where alice@x.com sits next to alex.morgan@example.com.
_TABLE = {
    "alice@x.com":              [1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    "alex.morgan@example.com":  [0.95, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    "jordan.lee@example.org":   [0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    "taylor.kim@example.net":   [0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    "casey.nguyen@example.io":  [0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00],
    "sam.patel@example.co":     [0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00],
    "robin.chen@example.com":   [0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00],
    "morgan.davis@example.org": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00],
    "jamie.singh@example.io":   [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],
}


def _silent_cfg(tmp_path: Path) -> PrivacyConfig:
    """Build a PrivacyConfig that won't try to prompt the user."""
    cfg = PrivacyConfig(enabled=True)
    cfg.audit.log_dir = str(tmp_path)
    cfg.confirmation.mode = "never"
    return cfg


def _metric_dp_only_cfg(tmp_path: Path) -> PrivacyConfig:
    """Build a config that disables K_DECOY so MEDIUM PII routes to
    METRIC_DP directly. M2 made K_DECOY the default for MEDIUM (it's
    free of ε), but several tests below specifically exercise the
    METRIC_DP pipeline — they'd be hidden by the K_DECOY fast-path
    otherwise."""
    cfg = _silent_cfg(tmp_path)
    cfg.k_decoy.enabled = False
    return cfg


# --- happy path: end-to-end METRIC_DP -------------------------------------------------


@pytest.mark.asyncio
async def test_full_metric_dp_pipeline_anonymizes_and_restores(tmp_path: Path):
    """The marquee integration test for M3.

    1) Detector finds the email.
    2) Decider routes to METRIC_DP because backend is available and
       accountant has budget.
    3) Transform replaces the email with a candidate from the typed pool.
    4) Pretend the cloud LLM echoes the candidate verbatim.
    5) Restorer swaps the candidate back to the user's real address.
    """
    cfg = _metric_dp_only_cfg(tmp_path)
    # Use high ε so the chosen candidate is the nearest neighbour
    # (alex.morgan@example.com) — gives a deterministic assertion.
    cfg.metric_dp.eps_query = 50.0
    cfg.metric_dp.eps_session_max = 1000.0
    cfg.metric_dp.eps_user_24h_max = 1000.0
    gate = GateKeeper.from_config(cfg, local_model=_Backend(_TABLE))

    raw = "Please email alice@x.com tomorrow."
    rec = await gate.detect_and_recommend(raw, session_key="s1", user_id="user-1")
    assert rec.path == ExecutionPath.METRIC_DP, (
        f"decider should route to METRIC_DP; got {rec.path}"
    )

    decision = await gate.confirm(
        rec, chat_id="c1", channel_name="cli",
        capabilities=ChannelCapabilities(),
    )
    assert decision.path == ExecutionPath.METRIC_DP

    outcome = await gate.transform(
        decision, raw, session_key="s1", user_id="user-1",
    )
    # The anonymized message must not contain the real value.
    assert "alice@x.com" not in outcome.privacy_message
    # And it MUST contain *some* candidate.
    assert "@example" in outcome.privacy_message
    # Audit reflects an anonymising transform.
    assert outcome.audit_view.fidelity == "RESTORED_LOSSY"
    assert outcome.audit_view.eps_consumed == pytest.approx(50.0)
    # Mapping is reversible.
    assert outcome.restoration_plan["mapping"]
    chosen, original = next(iter(outcome.restoration_plan["mapping"].items()))
    assert original == "alice@x.com"
    assert chosen != original

    # Simulate the cloud LLM faithfully using the candidate it saw.
    cloud_reply = f"Got it — I will draft an email to {chosen}. Anything else?"
    final = await gate.restore(cloud_reply, outcome)
    assert chosen not in final
    assert "alice@x.com" in final


@pytest.mark.asyncio
async def test_normal_path_skips_transform_entirely(tmp_path: Path):
    """No-entity message should bypass the transform and restore stages."""
    cfg = _silent_cfg(tmp_path)
    gate = GateKeeper.from_config(cfg, local_model=_Backend(_TABLE))

    raw = "What is the capital of France?"
    rec = await gate.detect_and_recommend(raw, session_key="s2", user_id="user-2")
    assert rec.path == ExecutionPath.NORMAL

    decision = await gate.confirm(
        rec, chat_id="c2", channel_name="cli",
        capabilities=ChannelCapabilities(),
    )
    outcome = await gate.transform(
        decision, raw, session_key="s2", user_id="user-2",
    )
    assert outcome.privacy_message == raw                  # passthrough
    assert not outcome.restoration_plan.get("mapping")
    assert outcome.audit_view.fidelity == "EXACT"
    assert outcome.audit_view.eps_consumed == 0.0
    # Restore is also a noop.
    final = await gate.restore("The capital is Paris.", outcome)
    assert final == "The capital is Paris."


# --- accountant integration ------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_budget_exhaustion_blocks_subsequent_metric_dp(tmp_path: Path):
    """When the accountant runs out of budget mid-session, follow-up METRIC_DP
    requests fall back to BLOCKED instead of forwarding plaintext."""
    cfg = _metric_dp_only_cfg(tmp_path)
    cfg.metric_dp.eps_query = 8.0
    cfg.metric_dp.eps_session_max = 12.0     # room for exactly one request
    cfg.metric_dp.eps_user_24h_max = 100.0
    gate = GateKeeper.from_config(cfg, local_model=_Backend(_TABLE))

    raw = "Email alice@x.com please"
    # Turn 1: METRIC_DP succeeds and consumes 8 ε.
    rec1 = await gate.detect_and_recommend(raw, session_key="s3", user_id="user-3")
    assert rec1.path == ExecutionPath.METRIC_DP
    d1 = await gate.confirm(
        rec1, chat_id="c3", channel_name="cli",
        capabilities=ChannelCapabilities(),
    )
    await gate.transform(d1, raw, session_key="s3", user_id="user-3")

    # Turn 2: only 4 ε left, eps_query=8 — accountant denies, decider
    # falls back to BLOCKED (no other path available for MEDIUM email).
    rec2 = await gate.detect_and_recommend(raw, session_key="s3", user_id="user-3")
    assert rec2.path == ExecutionPath.BLOCKED


@pytest.mark.asyncio
async def test_user_24h_budget_caps_across_sessions(tmp_path: Path):
    """A user can't reset their daily budget by opening a new session."""
    cfg = _metric_dp_only_cfg(tmp_path)
    cfg.metric_dp.eps_query = 8.0
    cfg.metric_dp.eps_session_max = 100.0
    cfg.metric_dp.eps_user_24h_max = 10.0     # tighter than the session cap
    gate = GateKeeper.from_config(cfg, local_model=_Backend(_TABLE))

    raw = "Email alice@x.com please"
    # First session — consumes 8 ε of the 10 daily budget.
    rec1 = await gate.detect_and_recommend(raw, session_key="s-A", user_id="alice")
    assert rec1.path == ExecutionPath.METRIC_DP
    d1 = await gate.confirm(
        rec1, chat_id="cA", channel_name="cli",
        capabilities=ChannelCapabilities(),
    )
    await gate.transform(d1, raw, session_key="s-A", user_id="alice")

    # Fresh session for the same user — only 2 ε left in the day so an
    # 8-ε query is denied even though session budget is untouched.
    rec2 = await gate.detect_and_recommend(raw, session_key="s-B", user_id="alice")
    assert rec2.path == ExecutionPath.BLOCKED


@pytest.mark.asyncio
async def test_reset_session_refills_session_budget(tmp_path: Path):
    cfg = _metric_dp_only_cfg(tmp_path)
    cfg.metric_dp.eps_query = 8.0
    cfg.metric_dp.eps_session_max = 10.0
    cfg.metric_dp.eps_user_24h_max = 100.0
    gate = GateKeeper.from_config(cfg, local_model=_Backend(_TABLE))

    raw = "Email alice@x.com please"
    rec1 = await gate.detect_and_recommend(raw, session_key="s4", user_id="user-4")
    d1 = await gate.confirm(rec1, chat_id="c4", channel_name="cli",
                            capabilities=ChannelCapabilities())
    await gate.transform(d1, raw, session_key="s4", user_id="user-4")

    # Right after, session is exhausted.
    rec_exhausted = await gate.detect_and_recommend(raw, session_key="s4", user_id="user-4")
    assert rec_exhausted.path == ExecutionPath.BLOCKED

    # Reset & we get our session budget back. (user_24h still has room.)
    gate.reset_session_budget("s4")
    rec_refilled = await gate.detect_and_recommend(raw, session_key="s4", user_id="user-4")
    assert rec_refilled.path == ExecutionPath.METRIC_DP


# --- backend absence -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_backend_falls_back_to_k_decoy_for_medium_pii(tmp_path: Path):
    """Without an embedding backend, METRIC_DP is unavailable — but K_DECOY
    (M2) doesn't need a backend, so MEDIUM PII routes there instead of
    BLOCKED. The cloud still never sees the raw value."""
    cfg = _silent_cfg(tmp_path)
    gate = GateKeeper.from_config(cfg)   # no local_model

    raw = "Email alice@x.com please"
    rec = await gate.detect_and_recommend(raw, session_key="s5", user_id="user-5")
    assert rec.path == ExecutionPath.K_DECOY
    assert ExecutionPath.METRIC_DP not in rec.allowed  # truly no METRIC_DP path
    assert ExecutionPath.BLOCKED in rec.allowed         # user can always escalate


@pytest.mark.asyncio
async def test_no_backend_and_k_decoy_disabled_falls_back_to_blocked(tmp_path: Path):
    """When the user wants formal guarantees only (METRIC_DP) and has no
    backend, K_DECOY-disabled is the right way to express that — MEDIUM
    PII fails closed to BLOCKED instead of silently downgrading."""
    cfg = _metric_dp_only_cfg(tmp_path)        # K_DECOY off
    gate = GateKeeper.from_config(cfg)         # no local_model either

    raw = "Email alice@x.com please"
    rec = await gate.detect_and_recommend(raw, session_key="s5b", user_id="user-5b")
    assert rec.path == ExecutionPath.BLOCKED
    assert ExecutionPath.METRIC_DP not in rec.allowed
    assert ExecutionPath.K_DECOY not in rec.allowed


# --- audit metadata --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audit_view_records_eps_for_metric_dp(tmp_path: Path):
    cfg = _metric_dp_only_cfg(tmp_path)
    cfg.metric_dp.eps_query = 4.0
    cfg.metric_dp.eps_session_max = 100.0
    cfg.metric_dp.eps_user_24h_max = 100.0
    gate = GateKeeper.from_config(cfg, local_model=_Backend(_TABLE))

    raw = "alice@x.com please"
    rec = await gate.detect_and_recommend(raw, session_key="s6", user_id="user-6")
    d = await gate.confirm(rec, chat_id="c6", channel_name="cli",
                           capabilities=ChannelCapabilities())
    outcome = await gate.transform(d, raw, session_key="s6", user_id="user-6")
    assert outcome.audit_view.eps_consumed == pytest.approx(4.0)
    assert outcome.audit_view.fidelity == "RESTORED_LOSSY"
    gate.record_audit(session_key="s6", decision=d, view=outcome.audit_view)
    # An audit JSONL line was written.
    files = list(tmp_path.glob("audit-*.jsonl"))
    assert files


# --- A1: budget-exhausted reason override ---------------------------------------------


@pytest.mark.asyncio
async def test_blocked_reason_distinguishes_budget_exhaustion(tmp_path: Path):
    """When METRIC_DP is unavailable *only* because the accountant is out of
    budget (transform + backend still functional), the BLOCKED reason
    must include 'eps_budget_exhausted' so the UX doesn't read like
    'feature not implemented'.

    We use the METRIC_DP-only config so K_DECOY doesn't quietly take over
    when the ε runs out — A1's reason override fires precisely when
    METRIC_DP is the *only* anonymization path and is denied.
    """
    cfg = _metric_dp_only_cfg(tmp_path)
    cfg.metric_dp.eps_query = 8.0
    cfg.metric_dp.eps_session_max = 100.0
    cfg.metric_dp.eps_user_24h_max = 8.0    # exactly one query worth
    gate = GateKeeper.from_config(cfg, local_model=_Backend(_TABLE))

    raw = "Email alice@x.com please"
    # Burn the entire daily cap with one METRIC_DP query.
    rec1 = await gate.detect_and_recommend(raw, session_key="sX", user_id="user-X")
    assert rec1.path == ExecutionPath.METRIC_DP
    d1 = await gate.confirm(rec1, chat_id="cX", channel_name="cli",
                            capabilities=ChannelCapabilities())
    await gate.transform(d1, raw, session_key="sX", user_id="user-X")

    # Second turn: same content, but accountant denies. Reason should
    # tell the user it's a budget thing, not a missing-feature thing.
    rec2 = await gate.detect_and_recommend(raw, session_key="sX", user_id="user-X")
    assert rec2.path == ExecutionPath.BLOCKED
    assert "eps_budget_exhausted" in rec2.reason
    assert "8.0/8.0" in rec2.reason            # used / cap
    assert "next refresh in" in rec2.reason    # refresh ETA included


@pytest.mark.asyncio
async def test_blocked_reason_unchanged_when_no_backend(tmp_path: Path):
    """When the *real* cause of BLOCKED is a missing backend (no transform,
    so accountant never even gets consulted), the original
    no_anonymization_path_available_yet reason must be preserved — the
    A1 override must not falsely accuse the budget. K_DECOY also
    disabled so MEDIUM truly has no anonymization path."""
    cfg = _metric_dp_only_cfg(tmp_path)  # K_DECOY off
    gate = GateKeeper.from_config(cfg)   # no local_model either

    raw = "Email alice@x.com please"
    rec = await gate.detect_and_recommend(raw, session_key="sY", user_id="user-Y")
    assert rec.path == ExecutionPath.BLOCKED
    assert rec.reason == "no_anonymization_path_available_yet"
    assert "eps_budget_exhausted" not in rec.reason


# --- M2 K_DECOY routing ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_k_decoy_is_preferred_for_medium_pii_when_available(tmp_path: Path):
    """The decider prefers K_DECOY > METRIC_DP for MEDIUM. K_DECOY costs
    no ε, so reserving the budget for HIGH risk is the right default."""
    cfg = _silent_cfg(tmp_path)
    gate = GateKeeper.from_config(cfg, local_model=_Backend(_TABLE))

    raw = "Email alice@x.com please"
    rec = await gate.detect_and_recommend(raw, session_key="sK", user_id="user-K")
    assert rec.path == ExecutionPath.K_DECOY
    # Allowed set includes the stricter METRIC_DP as an escalation option.
    assert ExecutionPath.METRIC_DP in rec.allowed
    assert ExecutionPath.BLOCKED in rec.allowed


@pytest.mark.asyncio
async def test_k_decoy_end_to_end_does_not_consume_eps(tmp_path: Path):
    """Full K_DECOY pipeline: detect → confirm → transform → restore.
    The accountant must show zero ε spent because K_DECOY is free."""
    cfg = _silent_cfg(tmp_path)
    cfg.metric_dp.eps_user_24h_max = 16.0   # would be tight if K_DECOY incorrectly spent ε
    gate = GateKeeper.from_config(cfg, local_model=_Backend(_TABLE))

    raw = "Please email alice@x.com about lunch."
    rec = await gate.detect_and_recommend(raw, session_key="sK2", user_id="user-K2")
    assert rec.path == ExecutionPath.K_DECOY
    decision = await gate.confirm(
        rec, chat_id="cK", channel_name="cli",
        capabilities=ChannelCapabilities(),
    )
    assert decision.path == ExecutionPath.K_DECOY
    outcome = await gate.transform(
        decision, raw, session_key="sK2", user_id="user-K2",
    )
    # Cloud should never see the real email.
    assert "alice@x.com" not in outcome.privacy_message
    # eps_consumed exactly zero.
    assert outcome.audit_view.eps_consumed == 0.0
    # Restoration round-trips.
    pseudo, original = next(iter(outcome.restoration_plan["mapping"].items()))
    assert original == "alice@x.com"
    cloud_reply = f"Got it, drafting an email to {pseudo}."
    final = await gate.restore(cloud_reply, outcome)
    assert pseudo not in final
    assert "alice@x.com" in final


@pytest.mark.asyncio
async def test_k_decoy_survives_metric_dp_budget_exhaustion(tmp_path: Path):
    """Even after the user burns through all ε on HIGH-risk work, MEDIUM
    PII can still go via K_DECOY — this is the M2 fallback story.

    Constructed by exhausting the accountant via direct consume(), then
    sending a MEDIUM email. Without K_DECOY (M1), this would BLOCK; with
    K_DECOY (M2), the message still gets anonymized."""
    cfg = _silent_cfg(tmp_path)
    cfg.metric_dp.eps_user_24h_max = 8.0
    cfg.metric_dp.eps_query = 8.0
    gate = GateKeeper.from_config(cfg, local_model=_Backend(_TABLE))

    # Manually exhaust the accountant — simulates a previous HIGH-risk turn.
    gate._accountant.consume("sK3", "user-K3", 8.0)

    raw = "Email alice@x.com about lunch."
    rec = await gate.detect_and_recommend(raw, session_key="sK3", user_id="user-K3")
    # K_DECOY does not consult the accountant — still available.
    assert rec.path == ExecutionPath.K_DECOY
    decision = await gate.confirm(
        rec, chat_id="cK3", channel_name="cli",
        capabilities=ChannelCapabilities(),
    )
    outcome = await gate.transform(
        decision, raw, session_key="sK3", user_id="user-K3",
    )
    assert "alice@x.com" not in outcome.privacy_message
    assert outcome.audit_view.eps_consumed == 0.0


@pytest.mark.asyncio
async def test_k_decoy_pseudonyms_consistent_within_session(tmp_path: Path):
    """A second turn referencing the same entity must yield the same
    pseudonym so the cloud's multi-turn reasoning stays coherent.

    This exercises the deterministic HMAC keying end-to-end (not just
    in the unit test for KDecoyTransform)."""
    cfg = _silent_cfg(tmp_path)
    gate = GateKeeper.from_config(cfg, local_model=_Backend(_TABLE))

    raw1 = "Email alice@x.com tomorrow."
    raw2 = "Also CC alice@x.com on the previous email."
    rec1 = await gate.detect_and_recommend(raw1, session_key="sK4", user_id="user-K4")
    d1 = await gate.confirm(rec1, chat_id="cK4", channel_name="cli",
                            capabilities=ChannelCapabilities())
    o1 = await gate.transform(d1, raw1, session_key="sK4", user_id="user-K4")

    rec2 = await gate.detect_and_recommend(raw2, session_key="sK4", user_id="user-K4")
    d2 = await gate.confirm(rec2, chat_id="cK4", channel_name="cli",
                            capabilities=ChannelCapabilities())
    o2 = await gate.transform(d2, raw2, session_key="sK4", user_id="user-K4")

    p1 = next(iter(o1.restoration_plan["mapping"]))
    p2 = next(iter(o2.restoration_plan["mapping"]))
    assert p1 == p2, "deterministic pseudonym must be stable across turns of the same session"
