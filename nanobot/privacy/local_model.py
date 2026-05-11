"""Local-model abstraction for the privacy GateKeeper.

M1.5 ships the interface plus an :class:`LLMProviderBackend` adapter that
reuses nanobot's existing LLM provider registry. To wire a real backend
(Ollama / LM Studio / OpenAI-compatible / Anthropic …) you set
``privacy.local_model = "ollama/qwen2.5:0.5b"`` in config — the providers
block already configured for the main agent is consulted via
:func:`nanobot.providers.factory.make_provider`.

Why a Protocol on top of LLMProvider? Three downstream features depend on a
narrow shape that doesn't need the full chat-completion surface:

* **Semantic detection** (``SemanticDetector`` in ``detector.py``) — needs
  ``generate`` to classify spans missed by regex.
* **K-decoy generation** (M2) — needs ``generate`` to synthesize decoys
  drawn from the same distribution as real entities.
* **Metric-DP restoration** (M3) — needs ``generate`` (and possibly
  ``embed``) to reconstruct a coherent response from the noised query.

The narrower contract lets us test with cheap fakes and keeps the
privacy modules from importing the full provider stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider


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
    backend, so a NullLocalModel never produces traffic.
    """

    name: str = "null"

    def is_available(self) -> bool:
        return False

    async def generate(self, prompt: str, **_: object) -> str:
        return ""

    async def embed(self, text: str) -> list[float]:
        return []


class LLMProviderBackend:
    """Adapt any nanobot :class:`LLMProvider` as a :class:`LocalModelBackend`.

    Built by ``GateKeeper.from_root_config`` when ``privacy.local_model``
    is configured. Calls ``provider.chat()`` for generation and returns
    an empty embedding (cloud chat providers rarely expose embeddings;
    M3 will add a separate embedding backend when needed).
    """

    def __init__(self, provider: "LLMProvider", model: str) -> None:
        self._provider = provider
        self._model = model

    @property
    def name(self) -> str:
        return f"llm:{self._model}"

    def is_available(self) -> bool:
        return self._provider is not None

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> str:
        try:
            response = await self._provider.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self._model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception:
            # Privacy pipeline fails closed elsewhere; do not propagate.
            return ""
        return _coerce_text(getattr(response, "content", None))

    async def embed(self, text: str) -> list[float]:
        # Most chat providers don't expose embeddings via .chat(); leave to M3.
        return []


def _coerce_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    # Some providers return a list of content blocks.
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


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


def build_from_root_config(root_config: Any) -> "LocalModelBackend | None":
    """Construct an :class:`LLMProviderBackend` from a root :class:`Config`.

    Returns None when ``privacy.enabled`` is False or ``privacy.local_model``
    is unset / unresolvable — callers should fall back to NullLocalModel.
    """
    privacy = getattr(root_config, "privacy", None)
    if privacy is None or not getattr(privacy, "enabled", False):
        return None
    model = getattr(privacy, "local_model", None)
    if not model:
        return None
    try:
        from nanobot.providers.factory import make_provider

        provider = make_provider(root_config, model_override=model)
    except Exception:
        # Misconfigured local model: log via NullLocalModel fallback instead
        # of crashing the agent loop.
        return None
    return LLMProviderBackend(provider=provider, model=model)

