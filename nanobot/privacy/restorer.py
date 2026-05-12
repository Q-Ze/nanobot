"""Restorer — reverse :class:`MetricDPTransform` substitutions on a cloud response.

The Metric-DP transform replaces each sensitive entity with a typed
candidate (e.g. ``alice@x.com`` → ``alex.morgan@example.com``) and emits a
mapping that the Restorer consults when the cloud LLM's response comes
back. The result is a final answer that *reads as if* the cloud had seen
the user's real values — except no real value was ever sent.

Pipeline (M3 step 5):

    cloud response  →  Restorer.restore(text, mapping)  →  final answer

Algorithm — two-pass deterministic substitution
-----------------------------------------------

1. **Pass 1**: Replace every anonymized value with a unique **sentinel**
   token (e.g. ``\\x00SENTINEL_3\\x00``). Sentinels prevent A→B→C chain
   replacements: with mapping ``{"Bob": "Alice", "Alice": "Eve"}``,
   naively running ``str.replace`` for "Bob" first would later rewrite
   that "Alice" to "Eve" — wrong. Sentinels are not valid response
   characters, so pass 2 can safely swap them for the originals.

2. **Pass 2**: Replace each sentinel with the corresponding **original
   value** from the mapping.

Word boundaries
~~~~~~~~~~~~~~~

ASCII keys are matched with ``(?<!\\w)…(?!\\w)`` lookaround so that a
key like ``"Bob"`` does **not** clobber the substring inside ``"Bobby"``.
Plain ``\\b…\\b`` doesn't work for keys that start with non-word
characters (``"+1 415…"`` — both space-before-``+`` and ``+``-itself are
``\\W``, so ``\\b`` finds no transition). The negative lookarounds, by
contrast, just say "the char *immediately* before/after must not be a
word character", which covers both ``"Bob"`` and ``"+1 415 555 0101"``.
Non-ASCII keys (Chinese names, addresses, …) skip the boundary check —
Chinese has no word delimiters — and just do plain substring replace.

What this module deliberately does NOT do (yet)
-----------------------------------------------

* **LM-assisted variant rewriting.** A future revision will let a local
  small model paraphrase the response, catching cases where the cloud
  inflected the assigned candidate (``"Bob's email"`` → ``"Alice's email"``
  is already handled by ``\\b`` boundaries; ``"Bob → Robert"`` style
  expansion is not). The hook is reserved — pass ``backend=`` to the
  constructor and ``use_llm_rewrite=True`` — but the M3 step 5 first cut
  keeps things deterministic.
* **Fuzzy / case-insensitive matching.** The cloud almost always echoes
  the exact candidate we sent it. Adding fuzzy matching by default
  invites silent over-replacement (``Bob`` matching inside arbitrary
  text). Callers who need it can preprocess the mapping.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from nanobot.privacy.local_model import LocalModelBackend

__all__ = ["Restorer", "RestoreResult"]


@dataclass(frozen=True)
class RestoreResult:
    """Outcome of :meth:`Restorer.restore`."""

    restored_text: str
    replacements_applied: int = 0
    unmatched_keys: list[str] = field(default_factory=list)
    """Anonymized values from *mapping* that did NOT appear in the cloud
    response. Common causes: the cloud paraphrased the candidate (rare)
    or simply didn't mention it (e.g. user only asked for a greeting).
    Surface this to logs so silent drift is debuggable."""


class Restorer:
    """Apply the inverse of MetricDPTransform's substitutions."""

    def __init__(
        self,
        *,
        backend: LocalModelBackend | None = None,
        use_llm_rewrite: bool = False,
    ) -> None:
        # Backend is reserved for the LM-assisted rewrite path. Step 5
        # first cut never invokes it; the kwargs are present so callers
        # already pin a future-stable signature.
        self._backend = backend
        self._use_llm_rewrite = use_llm_rewrite

    async def restore(
        self,
        response_text: str,
        mapping: dict[str, str],
    ) -> RestoreResult:
        if not response_text or not mapping:
            return RestoreResult(response_text, 0, [])

        # Drop entries with empty keys/values defensively.
        cleaned = {
            k: v for k, v in mapping.items()
            if isinstance(k, str) and isinstance(v, str) and k
        }
        if not cleaned:
            return RestoreResult(response_text, 0, [])

        # Longest first: prevents shorter keys from claiming substrings
        # of longer keys before we get to them. E.g. with
        # {"Alex": "X", "alex.morgan@example.com": "Y"} we want the long
        # email to match first.
        keys = sorted(cleaned.keys(), key=len, reverse=True)

        # Unique sentinel per key. \x00 is never a valid response char.
        sentinels = {k: f"\x00SENTINEL_{i}\x00" for i, k in enumerate(keys)}

        text = response_text
        applied = 0
        unmatched: list[str] = []

        # Pass 1: anonymized value → sentinel
        for k in keys:
            count = 0
            if _is_ascii_token(k):
                # Negative lookbehind/ahead — works for keys that start
                # or end with non-word characters too (e.g. "+1 415…").
                pattern = re.compile(
                    r"(?<!\w)" + re.escape(k) + r"(?!\w)"
                )
                text, count = pattern.subn(sentinels[k], text)
            else:
                # Chinese / non-ASCII strings have no word delimiters; a
                # plain substring replace is the right semantic.
                count = text.count(k)
                if count:
                    text = text.replace(k, sentinels[k])
            if count == 0:
                unmatched.append(k)
            applied += count

        # Pass 2: sentinel → original
        for k in keys:
            text = text.replace(sentinels[k], cleaned[k])

        return RestoreResult(
            restored_text=text,
            replacements_applied=applied,
            unmatched_keys=unmatched,
        )


# --- helpers --------------------------------------------------------------------------


def _is_ascii_token(s: str) -> bool:
    """True if *s* should use ``(?<!\\w)…(?!\\w)`` lookarounds for matching.

    Heuristic: pure-ASCII strings benefit from negative-lookaround
    boundaries (avoids "Bob" matching inside "Bobby" while still
    allowing "+1 415 555 0101" to match next to spaces and dots).
    Strings with any non-ASCII character fall back to plain substring
    replace — Chinese text has no word delimiters and word-boundary
    regexes behave unhelpfully there.
    """
    return bool(s) and s.isascii()
