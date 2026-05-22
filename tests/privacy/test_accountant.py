"""Tests for PrivacyAccountant — ε budget composition + persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanobot.privacy.accountant import (
    BudgetExceeded,
    BudgetSnapshot,
    PrivacyAccountant,
    _hash,
)


def _mock_clock(now: list[float]):
    """Return a clock function backed by a list[float] so tests can advance time."""
    def clock() -> float:
        return now[0]
    return clock


# --- empty state / shape ---------------------------------------------------------------


def test_empty_state_full_budget():
    acc = PrivacyAccountant(eps_session_max=16.0, eps_user_24h_max=32.0)
    snap = acc.snapshot("s1", "alice")
    assert snap.eps_session_used == 0.0
    assert snap.remaining_session == 16.0
    assert snap.eps_user_24h_used == 0.0
    assert snap.remaining_user_24h == 32.0
    assert snap.can_afford(8.0)
    assert not snap.can_afford(33.0)


def test_construction_rejects_non_positive_caps():
    with pytest.raises(ValueError):
        PrivacyAccountant(eps_session_max=0.0, eps_user_24h_max=10)
    with pytest.raises(ValueError):
        PrivacyAccountant(eps_session_max=10, eps_user_24h_max=-1)


# --- consume / can_afford --------------------------------------------------------------


def test_consume_below_cap_updates_both_budgets():
    acc = PrivacyAccountant(eps_session_max=16.0, eps_user_24h_max=32.0)
    snap = acc.consume("s1", "alice", 5.0)
    assert snap.eps_session_used == 5.0
    assert snap.eps_user_24h_used == 5.0
    assert snap.remaining_session == 11.0
    assert snap.remaining_user_24h == 27.0


def test_consume_exactly_at_cap_is_allowed():
    acc = PrivacyAccountant(eps_session_max=8.0, eps_user_24h_max=8.0)
    acc.consume("s1", "alice", 8.0)
    snap = acc.snapshot("s1", "alice")
    assert snap.remaining_session == 0.0
    assert not snap.can_afford(0.1)


def test_consume_above_cap_raises_with_snapshot():
    acc = PrivacyAccountant(eps_session_max=8.0, eps_user_24h_max=8.0)
    with pytest.raises(BudgetExceeded) as exc:
        acc.consume("s1", "alice", 10.0)
    assert exc.value.requested == 10.0
    assert isinstance(exc.value.snapshot, BudgetSnapshot)
    assert exc.value.snapshot.remaining_session == 8.0  # nothing was consumed


def test_consume_partial_then_overflow_does_not_leak_partial_state():
    """A failed consume() must not double-charge or partially update."""
    acc = PrivacyAccountant(eps_session_max=8.0, eps_user_24h_max=8.0)
    acc.consume("s1", "alice", 5.0)
    with pytest.raises(BudgetExceeded):
        acc.consume("s1", "alice", 4.0)
    snap = acc.snapshot("s1", "alice")
    assert snap.eps_session_used == 5.0  # unchanged
    assert snap.eps_user_24h_used == 5.0


@pytest.mark.parametrize("bad_eps", [-0.1, -1.0, float("inf"), float("nan")])
def test_consume_rejects_invalid_eps(bad_eps):
    acc = PrivacyAccountant(eps_session_max=8.0, eps_user_24h_max=8.0)
    with pytest.raises(ValueError):
        acc.consume("s1", "alice", bad_eps)


def test_user_24h_caps_consume_even_when_session_has_room():
    acc = PrivacyAccountant(eps_session_max=100.0, eps_user_24h_max=8.0)
    acc.consume("s1", "alice", 8.0)
    with pytest.raises(BudgetExceeded):
        acc.consume("s1", "alice", 0.5)


def test_session_cap_blocks_even_when_user_24h_has_room():
    acc = PrivacyAccountant(eps_session_max=4.0, eps_user_24h_max=100.0)
    acc.consume("s1", "alice", 4.0)
    with pytest.raises(BudgetExceeded):
        acc.consume("s1", "alice", 0.5)


# --- reset session ---------------------------------------------------------------------


def test_reset_session_clears_session_budget_only():
    acc = PrivacyAccountant(eps_session_max=16.0, eps_user_24h_max=32.0)
    acc.consume("s1", "alice", 10.0)
    acc.reset_session("s1")
    snap = acc.snapshot("s1", "alice")
    # Session counter is back to 0 …
    assert snap.eps_session_used == 0.0
    # … but the user 24h record still remembers Alice's spend.
    assert snap.eps_user_24h_used == 10.0


def test_reset_unknown_session_is_noop():
    acc = PrivacyAccountant(eps_session_max=8.0, eps_user_24h_max=8.0)
    acc.reset_session("never-existed")  # must not raise
    assert acc.snapshot("s1", "alice").eps_session_used == 0.0


# --- multi-session for the same user ---------------------------------------------------


def test_two_sessions_share_user_24h_budget():
    acc = PrivacyAccountant(eps_session_max=100.0, eps_user_24h_max=10.0)
    acc.consume("s1", "alice", 6.0)
    acc.consume("s2", "alice", 3.0)
    # Each session has its own counter,
    assert acc.snapshot("s1", "alice").eps_session_used == 6.0
    assert acc.snapshot("s2", "alice").eps_session_used == 3.0
    # but they share the user-24h pool.
    assert acc.snapshot("s1", "alice").eps_user_24h_used == 9.0
    with pytest.raises(BudgetExceeded):
        acc.consume("s2", "alice", 2.0)


def test_separate_users_have_separate_user_budgets():
    acc = PrivacyAccountant(eps_session_max=100.0, eps_user_24h_max=4.0)
    acc.consume("s1", "alice", 3.0)
    # Bob is a separate user — Alice's spend does not exhaust him.
    acc.consume("s2", "bob", 4.0)


# --- 24h sliding window ----------------------------------------------------------------


def test_user_24h_window_evicts_old_entries():
    now = [1_000_000.0]
    acc = PrivacyAccountant(
        eps_session_max=100.0,
        eps_user_24h_max=10.0,
        clock=_mock_clock(now),
    )
    # Spend half the budget, then advance 23h59m — still inside the window.
    acc.consume("s1", "alice", 5.0)
    now[0] += 23 * 3600 + 59 * 60
    assert acc.snapshot("s1", "alice").eps_user_24h_used == 5.0
    # Advance past 24 h — old entry must be evicted.
    now[0] += 2 * 60
    assert acc.snapshot("s1", "alice").eps_user_24h_used == 0.0
    # And spending the full budget again is OK now.
    acc.consume("s1", "alice", 10.0)


# --- persistence -----------------------------------------------------------------------


def test_persistence_roundtrip(tmp_path: Path):
    path = tmp_path / "budget.json"
    acc = PrivacyAccountant(
        eps_session_max=100.0, eps_user_24h_max=10.0, persist_path=path,
    )
    acc.consume("s1", "alice", 3.0)
    acc.consume("s1", "bob", 2.0)

    # New instance reads the file: user_24h survives, session does NOT (in-memory only).
    acc2 = PrivacyAccountant(
        eps_session_max=100.0, eps_user_24h_max=10.0, persist_path=path,
    )
    snap_alice = acc2.snapshot("s1", "alice")
    snap_bob = acc2.snapshot("s1", "bob")
    assert snap_alice.eps_user_24h_used == 3.0
    assert snap_bob.eps_user_24h_used == 2.0
    # Session counters started fresh because they aren't persisted.
    assert snap_alice.eps_session_used == 0.0


def test_persistence_disabled_when_no_path(tmp_path: Path):
    # Path not provided → nothing written.
    acc = PrivacyAccountant(eps_session_max=8.0, eps_user_24h_max=8.0)
    acc.consume("s1", "alice", 4.0)
    # tmp_path stays empty
    assert list(tmp_path.iterdir()) == []


def test_user_id_is_hashed_on_disk(tmp_path: Path):
    path = tmp_path / "budget.json"
    acc = PrivacyAccountant(
        eps_session_max=8.0, eps_user_24h_max=8.0, persist_path=path,
    )
    acc.consume("s1", "alice@privatedomain.example", 1.5)
    body = path.read_text()
    # The cleartext user id must NEVER appear in the persisted file.
    assert "alice@privatedomain.example" not in body
    # The hash should be present.
    assert _hash("alice@privatedomain.example") in body


def test_corrupt_persistence_file_does_not_crash(tmp_path: Path):
    path = tmp_path / "budget.json"
    path.write_text("not json {{{")
    acc = PrivacyAccountant(
        eps_session_max=8.0, eps_user_24h_max=8.0, persist_path=path,
    )
    # Should silently start from empty state.
    assert acc.snapshot("s1", "alice").eps_user_24h_used == 0.0


def test_persistence_atomic_temp_file_cleaned_up(tmp_path: Path):
    path = tmp_path / "budget.json"
    acc = PrivacyAccountant(
        eps_session_max=8.0, eps_user_24h_max=8.0, persist_path=path,
    )
    acc.consume("s1", "alice", 1.0)
    # Only the real file remains; the .tmp variant is rename()d away.
    assert path.exists()
    files = list(tmp_path.iterdir())
    assert all(p.suffix != ".tmp" for p in files)


def test_persistence_preserves_timestamps(tmp_path: Path):
    """The 24h window logic depends on timestamps surviving a reload."""
    now = [2_000_000.0]
    path = tmp_path / "budget.json"
    acc = PrivacyAccountant(
        eps_session_max=100.0,
        eps_user_24h_max=10.0,
        persist_path=path,
        clock=_mock_clock(now),
    )
    acc.consume("s1", "alice", 4.0)

    # New instance, advance well past 24 h before its first call.
    now[0] += 25 * 3600
    acc2 = PrivacyAccountant(
        eps_session_max=100.0,
        eps_user_24h_max=10.0,
        persist_path=path,
        clock=_mock_clock(now),
    )
    # Entry should be evicted by the sliding window.
    assert acc2.snapshot("s1", "alice").eps_user_24h_used == 0.0


# --- snapshot purity -----------------------------------------------------------------


def test_snapshot_does_not_mutate():
    acc = PrivacyAccountant(eps_session_max=8.0, eps_user_24h_max=8.0)
    acc.consume("s1", "alice", 3.0)
    snap1 = acc.snapshot("s1", "alice")
    snap2 = acc.snapshot("s1", "alice")
    assert snap1.eps_session_used == snap2.eps_session_used == 3.0
    # The snapshot is frozen — attempting to mutate it raises.
    with pytest.raises(Exception):
        snap1.eps_session_used = 99.0  # type: ignore[misc]


def test_can_afford_rejects_invalid_eps():
    snap = BudgetSnapshot(0.0, 8.0, 0.0, 8.0)
    assert not snap.can_afford(-1.0)
    assert not snap.can_afford(float("inf"))


# --- low-level helper ---------------------------------------------------------------


def test_hash_is_stable_and_truncated():
    h1 = _hash("alice")
    h2 = _hash("alice")
    assert h1 == h2
    assert len(h1) == 16
    assert all(c in "0123456789abcdef" for c in h1)
    assert _hash("bob") != h1


# --- composition guarantee (basic linear) -------------------------------------------


def test_basic_composition_sums_linearly():
    """ε accumulates as Σ εᵢ (basic composition)."""
    acc = PrivacyAccountant(eps_session_max=100.0, eps_user_24h_max=100.0)
    for _ in range(7):
        acc.consume("s1", "alice", 1.5)
    snap = acc.snapshot("s1", "alice")
    assert snap.eps_session_used == pytest.approx(10.5)
    assert snap.eps_user_24h_used == pytest.approx(10.5)


# --- JSON shape (forward-compat for external dashboards) ----------------------------


def test_persisted_json_is_well_formed(tmp_path: Path):
    path = tmp_path / "budget.json"
    acc = PrivacyAccountant(
        eps_session_max=8.0, eps_user_24h_max=8.0, persist_path=path,
    )
    acc.consume("s1", "alice", 1.0)
    parsed = json.loads(path.read_text())
    # Top-level is { user_hash: [[ts, eps], ...] }.
    assert isinstance(parsed, dict)
    entries = next(iter(parsed.values()))
    assert isinstance(entries, list)
    assert all(len(e) == 2 for e in entries)
    assert all(isinstance(e[0], (int, float)) and isinstance(e[1], (int, float))
               for e in entries)


# --- time_until_next_refresh -----------------------------------------------------------


def test_time_until_next_refresh_returns_none_when_empty():
    acc = PrivacyAccountant(eps_session_max=8.0, eps_user_24h_max=8.0)
    assert acc.time_until_next_refresh("alice") is None


def test_time_until_next_refresh_returns_seconds_until_oldest_ages_out():
    now = [1_000_000.0]
    acc = PrivacyAccountant(
        eps_session_max=64.0, eps_user_24h_max=8.0, clock=_mock_clock(now),
    )
    acc.consume("s1", "alice", 4.0)            # at t=0
    now[0] += 3600                              # +1 h
    acc.consume("s1", "alice", 4.0)            # at t=+1 h
    # Oldest entry is the first one — it ages out at t+24h, i.e. 23h from now.
    remaining = acc.time_until_next_refresh("alice")
    assert remaining == pytest.approx(23 * 3600)


def test_time_until_next_refresh_floors_at_zero_when_expired_entries_linger():
    """Even if we somehow query between an entry's expiry and the next
    snapshot call, we never return a negative duration."""
    now = [1_000_000.0]
    acc = PrivacyAccountant(
        eps_session_max=64.0, eps_user_24h_max=8.0, clock=_mock_clock(now),
    )
    acc.consume("s1", "alice", 4.0)
    # snapshot() (called inside time_until_next_refresh) prunes expired
    # entries — push past 24h and the answer must be None (no live entries).
    now[0] += 24 * 3600 + 10
    # Force a snapshot to trigger pruning, then check.
    acc.snapshot("s1", "alice")
    assert acc.time_until_next_refresh("alice") is None
