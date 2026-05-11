"""Local-model abstraction for the privacy GateKeeper.

M1.5 ships the interface only — no concrete backend (Ollama / LM Studio /
llama.cpp) is wired in. M2/M3 will plug real backends behind the same
contract; until then the default is :class:`NullLocalModel`, which advertises
itself as unavailable and refuses every call.

Why an interface now? Three downstream features depend on the same shape:

* **Semantic detection** (`SemanticDetector` in `detector.py`) — needs
  ``generate`` to classify spans missed by regex.
* **K-decoy generation** (M2) — needs ``generate`` to synthesize decoys
  drawn from the same distribution as real entities.
* **Metric-DP restoration** (M3) — needs ``generate`` (and possibly
  ``embed``) to reconstruct a coherent response from the noised query.

Locking the contract early prevents three concurrent ad-hoc rewires later.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LocalModelBackend(Protocol):
    """Contract every local-model backend must satisfy.

    Implementations are expected to be thread-safe (callers may invoke
    methods from multiple asyncio tasks). They should NOT raise on
    transient failures — return an empty string / empty list and log
    instead, so the privacy pipeline can fail closed rather than crash.
    """

    @property
    def name(self) -> str:
        """Human-readable backend identity, e.g. ``"ollama:qwen2.5:0.5b"``."""
        ...

    def is_available(self) -> bool:
        """Cheap, side-effect-free readiness check. False when no backend is wired."""
        ...

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> str:
        """Single-shot completion. Implementations decide whether to use
        chat templates internally; callers pass a fully-formed prompt.
        """
        ...

    async def embed(self, text: str) -> list[float]:
        """Token / sentence embedding. Returns an empty list when the backend
        cannot embed (e.g. completion-only models)."""
        ...


class NullLocalModel:
    """Default backend — advertises unavailable, refuses all calls.

    The privacy pipeline checks ``is_available()`` before invoking the
    backend, so a NullLocalModel never produces traffic. M1.5 always
    uses this; M2 swaps in a real backend behind the same Protocol.
    """

    name: str = "null"

    def is_available(self) -> bool:
        return False

    async def generate(self, prompt: str, **_: object) -> str:
        return ""

    async def embed(self, text: str) -> list[float]:
        return []


# Module-level registry so power users / tests can inject a backend without
# touching GateKeeper.from_config every time. Convention:
#   from nanobot.privacy import local_model
#   local_model.set_default(MyOllamaBackend())
_default: LocalModelBackend = NullLocalModel()


def get_default() -> LocalModelBackend:
    return _default


def set_default(backend: LocalModelBackend) -> None:
    """Replace the process-wide default backend. Idempotent."""
    global _default
    if not isinstance(backend, LocalModelBackend):
        raise TypeError(
            f"{backend!r} does not satisfy LocalModelBackend protocol "
            "(missing name/is_available/generate/embed?)"
        )
    _default = backend


def reset_default() -> None:
    """Restore the NullLocalModel default (mainly for tests)."""
    global _default
    _default = NullLocalModel()
