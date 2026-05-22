"""Privacy accountant — ε budget management for Metric-DP.

Per `.agent/privacy_gatekeeper.md` §3.5 / §4.3 the gate enforces two
budgets in parallel:

  • **ε_session**:  reset when a chat session ends (in-memory only).
  • **ε_user_24h**: rolling 24-hour window per user (disk-persistent).

Both are summed with **basic linear composition** (Σ εᵢ). That bound is
worst-case but trivially correct — high enough for M3 step 4 to ship a
useful accountant without coupling to a specific composition theorem.
A future optimisation can swap in RDP / zCDP-aware accounting and reuse
the same external API.

Per the spec, when the requested ε would exceed either cap the accountant
refuses (raises :class:`BudgetExceeded`); the decider then fails closed
to BLOCKED. Callers SHOULD call :meth:`can_afford` first to decide
whether to attempt Metric-DP versus a cheaper path; :meth:`consume`
is the post-success bookkeeping.

Privacy property: ``user_id`` is **never written to disk in cleartext**.
On-disk persistence keys by SHA-256(user_id)[:16], so even an attacker
with read access to the budget file learns only "some user spent ε at
time t", not who.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

__all__ = [
    "BudgetExceeded",
    "BudgetSnapshot",
    "PrivacyAccountant",
]

_SECONDS_PER_DAY = 86_400


class BudgetExceeded(Exception):  # noqa: N818 — matches existing FileSizeExceeded
    """Raised by :meth:`PrivacyAccountant.consume` when the request would
    push usage past either budget cap. Carries a :class:`BudgetSnapshot`
    so callers can render a useful message without re-querying."""

    def __init__(self, snapshot: "BudgetSnapshot", requested: float) -> None:
        self.snapshot = snapshot
        self.requested = requested
        super().__init__(
            f"requested ε={requested:.2f} exceeds budget; "
            f"session remaining {snapshot.remaining_session:.2f} "
            f"(cap {snapshot.eps_session_max:.2f}), "
            f"user 24h remaining {snapshot.remaining_user_24h:.2f} "
            f"(cap {snapshot.eps_user_24h_max:.2f})"
        )


@dataclass(frozen=True)
class BudgetSnapshot:
    """Read-only view of the budget at a point in time."""

    eps_session_used: float
    eps_session_max: float
    eps_user_24h_used: float
    eps_user_24h_max: float

    @property
    def remaining_session(self) -> float:
        return max(0.0, self.eps_session_max - self.eps_session_used)

    @property
    def remaining_user_24h(self) -> float:
        return max(0.0, self.eps_user_24h_max - self.eps_user_24h_used)

    def can_afford(self, eps: float) -> bool:
        """True if a single request of size ``eps`` fits within both budgets."""
        if eps < 0 or not (eps < float("inf")):
            return False
        return eps <= self.remaining_session and eps <= self.remaining_user_24h


class PrivacyAccountant:
    """Track per-session and per-user-24h ε consumption.

    Construction
    ------------
    ``persist_path`` enables JSON persistence for the user-24h window.
    Session usage is always in-memory (sessions don't survive restarts).
    ``clock`` is injectable so tests can simulate the passage of time
    without ``time.sleep``.
    """

    def __init__(
        self,
        *,
        eps_session_max: float = 32.0,
        eps_user_24h_max: float = 64.0,
        persist_path: Path | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if eps_session_max <= 0 or eps_user_24h_max <= 0:
            raise ValueError("budget caps must be positive")
        self._eps_session_max = float(eps_session_max)
        self._eps_user_24h_max = float(eps_user_24h_max)
        self._clock = clock or time.time
        self._persist_path: Path | None = (
            Path(persist_path).expanduser() if persist_path else None
        )
        # session_key -> cumulative ε
        self._session_usage: dict[str, float] = {}
        # SHA-256(user_id)[:16] -> list of (timestamp, ε) entries
        self._user_entries: dict[str, list[tuple[float, float]]] = {}
        # asyncio is cooperative but the accountant may be called from
        # multiple tasks — RLock is cheap and forward-compatible.
        self._lock = threading.RLock()
        if self._persist_path is not None:
            self._load()

    # --- public API --------------------------------------------------------------------

    def snapshot(self, session_key: str, user_id: str) -> BudgetSnapshot:
        """Return current usage — pure read, never mutates session state.

        Note: the user-24h sum prunes expired entries on read. That's
        an internal optimisation (the pruned entries no longer count
        towards usage anyway); callers see the same number either way.
        """
        with self._lock:
            session_used = self._session_usage.get(session_key, 0.0)
            user_used = self._sum_and_prune_user_24h(user_id)
        return BudgetSnapshot(
            eps_session_used=session_used,
            eps_session_max=self._eps_session_max,
            eps_user_24h_used=user_used,
            eps_user_24h_max=self._eps_user_24h_max,
        )

    def can_afford(self, session_key: str, user_id: str, eps: float) -> bool:
        """Cheap pre-flight check; equivalent to ``snapshot(...).can_afford(eps)``."""
        return self.snapshot(session_key, user_id).can_afford(eps)

    def time_until_next_refresh(self, user_id: str) -> float | None:
        """Seconds until the oldest ε entry ages out of the 24h window.

        Returns None if there are no live entries for the user. The value
        is when *some* budget frees up (the oldest entry), not when the
        cap is fully restored. Useful for UX messages like
        "eps_budget_exhausted, next refresh in 21h".
        """
        h = _hash(user_id)
        with self._lock:
            entries = self._user_entries.get(h)
            if not entries:
                return None
            oldest_ts = min(ts for ts, _ in entries)
        return max(0.0, oldest_ts + _SECONDS_PER_DAY - self._clock())

    def consume(self, session_key: str, user_id: str, eps: float) -> BudgetSnapshot:
        """Record ε spent; raises :class:`BudgetExceeded` if not affordable.

        Returns the post-consumption snapshot so callers don't need a
        second :meth:`snapshot` call.
        """
        if eps < 0 or not (eps < float("inf")):
            raise ValueError(f"eps must be a non-negative finite float, got {eps!r}")
        snap = self.snapshot(session_key, user_id)
        if not snap.can_afford(eps):
            raise BudgetExceeded(snap, eps)
        now = self._clock()
        with self._lock:
            self._session_usage[session_key] = self._session_usage.get(session_key, 0.0) + eps
            self._user_entries.setdefault(_hash(user_id), []).append((now, float(eps)))
        if self._persist_path is not None:
            self._save_atomic()
        # Return the new state.
        return BudgetSnapshot(
            eps_session_used=snap.eps_session_used + eps,
            eps_session_max=self._eps_session_max,
            eps_user_24h_used=snap.eps_user_24h_used + eps,
            eps_user_24h_max=self._eps_user_24h_max,
        )

    def reset_session(self, session_key: str) -> None:
        """Clear the session counter — session ended, budget refreshes for the next one.

        The user-24h window is untouched: ending a session does NOT
        forgive past usage from that user.
        """
        with self._lock:
            self._session_usage.pop(session_key, None)

    # --- internals ---------------------------------------------------------------------

    def _sum_and_prune_user_24h(self, user_id: str) -> float:
        """Sum ε in the last 24 h for *user_id*, pruning expired entries in place."""
        h = _hash(user_id)
        entries = self._user_entries.get(h)
        if not entries:
            return 0.0
        cutoff = self._clock() - _SECONDS_PER_DAY
        kept = [(ts, e) for ts, e in entries if ts >= cutoff]
        if len(kept) != len(entries):
            if kept:
                self._user_entries[h] = kept
            else:
                self._user_entries.pop(h, None)
        return sum(e for _, e in kept)

    def _save_atomic(self) -> None:
        """Write the persistent state via temp-file + rename."""
        path = self._persist_path
        assert path is not None
        data = {h: list(entries) for h, entries in self._user_entries.items()}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, separators=(",", ":")))
        try:
            os.replace(tmp, path)
        except OSError:
            # On the rare platforms where atomic rename across temp files
            # fails, fall back to a non-atomic overwrite. Both lose at most
            # one consume() on a crash; never lose more.
            tmp.replace(path)

    def _load(self) -> None:
        path = self._persist_path
        assert path is not None
        if not path.exists():
            return
        try:
            raw = json.loads(path.read_text() or "{}")
        except (OSError, json.JSONDecodeError):
            # Corrupt / unreadable — start fresh rather than crash.
            return
        if not isinstance(raw, dict):
            return
        result: dict[str, list[tuple[float, float]]] = {}
        for h, entries in raw.items():
            if not isinstance(h, str) or not isinstance(entries, list):
                continue
            parsed: list[tuple[float, float]] = []
            for entry in entries:
                if (
                    isinstance(entry, (list, tuple))
                    and len(entry) == 2
                    and isinstance(entry[0], (int, float))
                    and isinstance(entry[1], (int, float))
                ):
                    parsed.append((float(entry[0]), float(entry[1])))
            if parsed:
                result[h] = parsed
        self._user_entries = result


# --- helpers --------------------------------------------------------------------------


def _hash(value: str) -> str:
    """Stable, length-bounded user id digest — same format as the audit log."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
