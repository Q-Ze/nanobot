# Privacy GateKeeper — Release Notes & Usage Guide

> **Status:** M3 shipped 2026-05-12. The full Metric-DP pipeline runs
> end-to-end: a configured nanobot agent can now anonymise sensitive
> entities, send the anonymised text to a cloud LLM, and restore the
> original values in the reply — with a provable (ε)-dχ-privacy
> guarantee on the anonymisation step.
>
> Design spec: [`.agent/privacy_gatekeeper.md`](../.agent/privacy_gatekeeper.md).
> Changes since M3 (M2 K_DECOY + cloud-boundary fixes + UX): [`privacy-gatekeeper-changelog.md`](./privacy-gatekeeper-changelog.md).

This document describes what's shipped, how to enable it, how to test
it locally, and what's intentionally left for future milestones.

---

## 1. What's in the box

### 1.1 The `nanobot/privacy/` module

| File | Purpose |
|---|---|
| `types.py` | Shared enums + dataclasses (`ExecutionPath`, `RiskClass`, `EntityType`, `Decision`, `Recommendation`, `ChannelCapabilities`, `TransformOutcome`, `AuditView`, …) and strictness ordering. |
| `detector.py` | `PrivacyEntityDetector` — regex layer (email, phone E.164 + CN, CN ID checksum, bank card Luhn, IP, cloud credentials, JWT, SSH/PEM, high-entropy) + pluggable semantic-detector hook. |
| `semantic_detector.py` | `LLMSemanticDetector` — second-pass LM-backed detector for names / addresses / medical terms that regex can't catch. Auto-wired when a backend is configured. |
| `decider.py` | `ExecutionDecider` — pure function inputs → `Recommendation(path, allowed_set, reason)`. Implements §3.2 decision tree and §4.5 safety floor. |
| `confirmation.py` | `ConfirmationGate` — three modes (`always` / `risk_threshold` / `never`), user pre-selection (with silent floor enforcement), interactive ask with timeout, channel-fallback policies. |
| `audit.py` | `AuditLogger` — append-only JSONL, metadata only (no raw entity values), atomic + fsync. |
| `local_model.py` | `LocalModelBackend` protocol, `NullLocalModel`, `LLMProviderBackend` adapter that wraps any nanobot `LLMProvider` for chat + embeddings. |
| `metric_dp.py` | Multivariate Laplace sampler `laplace_noise(d, ε)` — polar decomposition (uniform direction on S^(d-1) × Gamma(d, 1/ε) radius) gives density ∝ exp(-ε‖η‖₂). |
| `transform.py` | `MetricDPTransform` — for each entity: embed → +Laplace noise → nearest neighbour in a typed candidate pool → swap; emits a restoration mapping. |
| `accountant.py` | `PrivacyAccountant` — basic linear composition of ε across `(ε_session, ε_user_24h)`; raises `BudgetExceeded` on overflow; user-id hashed on disk. |
| `restorer.py` | `Restorer` — two-pass deterministic anti-substitution with sentinels and ASCII-vs-non-ASCII boundary handling. |
| `gate.py` | `GateKeeper` facade — `detect_and_recommend` → `confirm` → `transform` → `restore`. Auto-wires the components from `PrivacyConfig`. |
| `cli_channel_caps.py` | CLI-channel `ChannelCapabilities` factory — interactive confirmation via stdin/stdout. |

### 1.2 Touched files outside the module

| File | What changed |
|---|---|
| `nanobot/config/schema.py` | `PrivacyConfig` (+ 4 sub-configs: `confirmation`, `audit`, `k_decoy`, `metric_dp`), `local_model`, `embedding_model`, `semantic_timeout_seconds`. |
| `nanobot/agent/loop.py` | New `TurnState.GATE` between COMMAND and BUILD; `_state_gate` runs the GateKeeper, `_state_run` calls `gate.restore` on the cloud response. `TurnContext` gains `privacy_outcome`. |
| `nanobot/channels/base.py` | `privacy_capabilities()` hook + two flag attributes. Default returns non-interactive. |
| `nanobot/channels/websocket.py` | Implements `privacy_capabilities()` with a real `send_confirmation` / `await_confirmation_reply` bridge backed by `asyncio.Future` + a new `privacy_confirmation` / `privacy_confirmation_reply` envelope pair. |
| `nanobot/providers/base.py` | `LLMProvider.embed(text, model=...) → list[float]`; default returns `[]`. |
| `nanobot/providers/openai_compat_provider.py` | Overrides `embed()` to call `/v1/embeddings`; transparent httpx fallback for non-canonical JSON responses (e.g. OpenRouter free-tier keep-alive prefix). |
| `nanobot/providers/factory.py` | `make_provider(config, *, model_override=…)` so a second provider can be built for the privacy model without disturbing the main agent. |
| `nanobot/cli/commands.py` | `nanobot agent` registers CLI `ChannelCapabilities` so the gate's confirmation prompt renders in the terminal. |

### 1.3 Tests

| Module | Cases |
|---|---|
| `tests/privacy/test_detector.py` | 19 (regex coverage, Luhn / CN-ID checksum, high-entropy, overrides, regex extensions, overlap merge, semantic injection). |
| `tests/privacy/test_decider.py` | 9 (recommendation table + AllowedPathSet floor). |
| `tests/privacy/test_confirmation.py` | 11 (three modes, preselect floor, timeout, channel fallback overrides). |
| `tests/privacy/test_gate_and_audit.py` | 8 (audit no-raw, audit disabled, façade end-to-end). |
| `tests/privacy/test_local_model.py` | 16 (protocol, registry, LLMProviderBackend incl. embedding-model override, factory build paths). |
| `tests/privacy/test_semantic_detector.py` | 22 (parser, span resolution, hallucination guard, fail-closed branches, auto-wiring through GateKeeper). |
| `tests/privacy/test_metric_dp.py` | 18 (input validation, Gamma moments, isotropy, off-diagonal covariance, ε-vs-noise inverse, **empirical dχ-privacy bound**, determinism). |
| `tests/privacy/test_transform.py` | 11 (no-op, high-ε convergence, low-ε spread, placeholder fail-closed paths, original-excluded-from-pool, multi-entity span splicing, UTF-8). |
| `tests/privacy/test_accountant.py` | 28 (empty / consume / overflow without partial state / sliding 24h window / persistence / on-disk user-id hashed / atomic temp file / forward-compat JSON shape). |
| `tests/privacy/test_restorer.py` | 20 (chain-replacement protection, lookaround boundary, Chinese, case-sensitive, unmatched reporting, **roundtrip with MetricDPTransform**). |
| `tests/privacy/test_gate_integration.py` | 7 (full METRIC_DP pipeline; NORMAL skips transform; session budget exhaustion; user_24h cap across sessions; reset_session refills; no-backend falls back; audit fidelity & eps_consumed). |
| `tests/channels/test_websocket_privacy_confirmation.py` | 10 (wire protocol broadcast, reply, cancel, timeout, invalid path, unknown id, stop-cancels-pending). |
| `tests/providers/test_embed.py` | 11 (embed happy path, default-model fallback, transport errors, OpenRouter whitespace-prefix httpx fallback). |

**Total: ~190 dedicated privacy tests, all green; 1726 in the full suite.**

---

## 2. Enable & use

### 2.1 Minimal opt-in (default: off)

```json
{
  "privacy": { "enabled": true }
}
```

What that alone gives you (no backend → no LM detection, no Metric-DP):
- CATASTROPHIC entities (API keys, SSH/PEM, JWT, …) **hard-blocked**.
- HIGH entities (CN ID, bank card, medical, high-entropy strings) **blocked**.
- MEDIUM entities (email, phone, name, address) **blocked** — fail-closed because there's no anonymisation path wired.
- LOW entities and entity-free messages pass through unchanged.
- Audit JSONL written to `~/.nanobot/privacy_audit/audit-YYYYMMDD.jsonl`.

### 2.2 Full configuration with Metric-DP enabled

```json
{
  "agents": {
    "defaults": { "model": "anthropic/claude-opus-4-5" }
  },
  "providers": {
    "anthropic": { "api_key": "sk-ant-..." },
    "openrouter": { "api_key": "sk-or-..." }
  },
  "privacy": {
    "enabled": true,
    "local_model": "minimax/minimax-m2.5:free",
    "embedding_model": "openai/text-embedding-3-small",
    "semantic_timeout_seconds": 60,
    "risk_class_overrides": { "ip": "low" },
    "regex_extensions": ["INTERNAL-\\d{6}"],
    "routing_mode": "conservative",
    "confirmation": {
      "mode": "risk_threshold",
      "risk_threshold": "high",
      "timeout_seconds": 60,
      "on_timeout": "block",
      "channel_fallback_default": "forced_conservative"
    },
    "audit": {
      "enabled": true,
      "log_dir": "~/.nanobot/privacy_audit"
    },
    "metric_dp": {
      "eps_query": 8.0,
      "eps_session_max": 32.0,
      "eps_user_24h_max": 64.0
    }
  }
}
```

| Key | Meaning |
|---|---|
| `local_model` | Model id (e.g. `"ollama/qwen2.5:0.5b"` or `"minimax/minimax-m2.5:free"`) used by `LLMSemanticDetector` for second-pass entity detection. Resolves through the existing `providers.*` blocks. |
| `embedding_model` | Model id used by `MetricDPTransform` for embeddings (e.g. `"ollama/nomic-embed-text"`, `"openai/text-embedding-3-small"`). Most chat models cannot embed — set a dedicated embedding model. When unset, `LLMProviderBackend.embed()` reuses the chat model and most servers reject it; the gate then falls back to BLOCKED for MEDIUM/HIGH entities. |
| `semantic_timeout_seconds` | Per-call wall-clock cap for the LM-backed `SemanticDetector`. Default 15 s — bump to 30–60 s for slow / cloud / free-tier endpoints. Visible in logs as `privacy.semantic: LM call exceeded timeout`. |
| `metric_dp.eps_query` | ε spent per Metric-DP transform. Default 8 (utility-leaning); set to 1 for strong-DP. |
| `metric_dp.eps_session_max` | Total ε a session can spend before further METRIC_DP turns block. |
| `metric_dp.eps_user_24h_max` | Same, rolling 24-hour window per user (persists to disk). |
| `risk_class_overrides` | Demote / promote a category's risk class to fix systematic false positives. |
| `regex_extensions` | Extra Python regex patterns flagging user-defined sensitive strings. |
| `confirmation.mode` | `always` / `risk_threshold` (default) / `never`. |
| `confirmation.on_timeout` | `block` (default, fail-closed) or `recommended`. |
| `channel_fallback_*` | Behaviour on channels that can't ask interactively. |

**Local can be cloud.** `local_model` and `embedding_model` are *roles*, not hard locality requirements. Pointing them at OpenAI/Anthropic/OpenRouter works; the privacy guarantee then assumes the embedding provider is part of your trust boundary.

### 2.3 User-supplied path preference (SDK / power users)

Set `metadata["privacy_path"]` on the inbound message to one of
`"normal" | "simple" | "k_decoy" | "metric_dp" | "blocked"` — the gate
honours the preference if it satisfies the safety floor, otherwise silently
overrides to the system recommendation and records `violation_attempt` in
the audit log.

---

## 3. Runtime behaviour

### 3.1 A blocked turn (CATASTROPHIC)

```
$ nanobot agent -m "my api key is sk-ant-test12345"

🛡  Privacy GateKeeper
  Detected entities:
    • credential catastrophic 'sk-ant-test12345'
  Recommended: blocked (hard_secret:credential)
  Options:
    → [1] blocked — do not send to cloud LLM
    [c]  cancel (don't send this message)
Choose [number / c / Enter]: ↵
[blocked] Privacy GateKeeper blocked this message. Reason: hard_secret:credential.
```

The cloud LLM is **never called**.

### 3.2 A METRIC_DP turn (MEDIUM entity, full pipeline)

```
$ nanobot agent -m "Please email alice@x.com about tomorrow."

🛡  Privacy GateKeeper
  Detected entities:
    • email medium 'alice@x.com'
  Recommended: metric_dp (recommended_metric_dp)
  Options:
    → [1] metric_dp — send with metric-DP noise
       [2] blocked   — do not send to cloud LLM
Choose [number / Enter]: ↵
[transform] alice@x.com → jamie.singh@example.io  (ε=8.0, fidelity=RESTORED_LOSSY)
[cloud]     "Sure, I'll draft an email to jamie.singh@example.io about ..."
[restore]   "Sure, I'll draft an email to alice@x.com about ..."
[audit]     path=metric_dp eps_consumed=8.0 session_remaining=24.0
```

The cloud LLM saw `jamie.singh@example.io`; the user sees `alice@x.com`.

### 3.3 A passing turn (no entities or LOW only)

The gate is invisible — no prompt, no rewrite. The message metadata
acquires a tiny `privacy: {path: "normal", source: "system_auto", …}`
hint and an audit JSONL line records "no entity detected".

### 3.4 Audit log entry

```jsonl
{
  "ts": "2026-05-12T04:06:41.945+00:00",
  "session_id_hash": "1a2b3c4d5e6f7890",
  "path": "metric_dp",
  "recommended_path": "metric_dp",
  "path_source": "user_confirmed",
  "user_choice_latency_ms": 2200,
  "violation_attempt": null,
  "user_downgrade": false,
  "entity_counts": {"email": 1},
  "risk_counts": {"medium": 1},
  "decision_reason": "recommended_metric_dp",
  "fidelity": "RESTORED_LOSSY",
  "eps_consumed": 8.0
}
```

Original entity value (`alice@x.com`) is **not** in this line.

---

## 4. How to test

### 4.0 Quick triage when "it's not triggering"

Run the diagnostic — walks the pipeline layer by layer:

```bash
python scripts/debug_privacy.py \
  --config ~/.nanobot/config.json \
  -m "Hello, my api key is sk-xsadsafsgdrghr"
```

It prints each layer's state (config → backend → smoke test → detector
→ semantic-LM raw call → decider → confirmation → transformer → embeddings).
Common findings and fixes:

| Diagnostic output | Cause | Fix |
|---|---|---|
| `privacy.enabled = False` at step 1 | Config not opted in | Set `privacy.enabled = true` |
| step 2 `✗ build_from_root_config returned None` | Provider not configured | Configure the matching `providers.<name>` block |
| step 2 smoke test returns `''` | Model not pulled / wrong id | `ollama pull <model>` or fix model id |
| step 3 empty + step 3b shows error response | Free-tier rate limit / wrong model | Switch model or add your own key |
| step 3 empty + step 3b empty (silent) | LM timed out | Bump `privacy.semantic_timeout_seconds` |
| step 7 returns `[]` | `embedding_model` unset / can't embed | Set `embedding_model` to an embedding-capable id |

### 4.1 Visualise the algorithm pieces (zero-dep)

```bash
# Multivariate Laplace sampler: histogram + 2-D scatter + isotropy check
python scripts/visualize_metric_dp.py

# Token-replacement distribution across ε values
python scripts/visualize_token_replacement.py
```

The token-replacement script run with `{0.2, 1, 5, 50}` shows the full
trade-off in 200-trial batches:

```
ε = 0.2   →  near-uniform across 7/8 candidates    (strong privacy)
ε = 1     →  top candidate 22 %                     (intermediate)
ε = 5     →  top candidate 48 %                     (weak privacy)
ε = 50    →  top candidate 100 % (always nearest)   (no privacy)
```

### 4.2 Run the unit and integration suites

```bash
pytest tests/privacy/ -v                      # ~190 cases, < 6 s
pytest tests/agent/ tests/config/ tests/channels/ tests/providers/ -q
ruff check nanobot/privacy/ tests/privacy/
```

Expected: all green.

### 4.3 End-to-end smoke without a real LLM

```python
import asyncio, random
from nanobot.config.schema import PrivacyConfig
from nanobot.privacy.gate import GateKeeper
from nanobot.privacy.types import ChannelCapabilities

class FakeBackend:
    name = "fake"
    def is_available(self): return True
    async def generate(self, p, **_): return ""
    async def embed(self, t, **_):
        # alice and a candidate sit at neighbouring vectors;
        # everything else is on a different axis
        return {"alice@x.com": [1,0,0,0,0,0,0,0],
                "alex.morgan@example.com": [0.95,0.05,0,0,0,0,0,0]}.get(t, [1]+[0]*7)

async def main():
    cfg = PrivacyConfig(enabled=True)
    cfg.confirmation.mode = "never"
    cfg.metric_dp.eps_query = 50.0   # high ε → nearest neighbour wins
    cfg.metric_dp.eps_session_max = 1000
    gate = GateKeeper.from_config(cfg, local_model=FakeBackend())

    raw = "Please email alice@x.com tomorrow."
    rec = await gate.detect_and_recommend(raw, session_key="s", user_id="u")
    print(f"path: {rec.path.value}")
    d = await gate.confirm(rec, chat_id="c", channel_name="cli",
                           capabilities=ChannelCapabilities())
    out = await gate.transform(d, raw, session_key="s", user_id="u")
    print(f"anonymised: {out.privacy_message}")
    cloud = f"Got it — drafting an email to {list(out.restoration_plan['mapping'])[0]} now."
    final = await gate.restore(cloud, out)
    print(f"restored:   {final}")

asyncio.run(main())
```

Output:
```
path: metric_dp
anonymised: Please email alex.morgan@example.com tomorrow.
restored:   Got it — drafting an email to alice@x.com now.
```

### 4.4 Live end-to-end with a real backend

```bash
# Enable privacy + configure an embedding-capable model in config.json,
# then send a message that contains a MEDIUM entity:
nanobot agent --logs --config ~/.nanobot/config.json \
    -m "Please email alice@x.com about the demo tomorrow."

# Check what the audit log captured:
tail -1 ~/.nanobot/privacy_audit/audit-*.jsonl | python -m json.tool

# Inspect the ε budget file:
python -m json.tool ~/.nanobot/privacy_audit/budget.json
```

### 4.5 WebSocket wire protocol (for custom clients / WebUI authors)

Server → client (alongside `message`, `delta`, etc.):

```jsonc
{
  "event": "privacy_confirmation",
  "chat_id": "<chat>",
  "confirmation_id": "<32-char hex>",
  "path": "metric_dp",
  "reason": "recommended_metric_dp",
  "allowed": ["metric_dp", "blocked"],
  "entities": [
    {"type": "email", "risk_class": "medium",
     "value": "alice@x.com", "span": [13, 24]}
  ]
}
```

Client → server:

```jsonc
{
  "type": "privacy_confirmation_reply",
  "confirmation_id": "<echo the prompt's id>",
  "chosen_path": "metric_dp"   // or null to cancel the message
}
```

Rules:
- `chosen_path` not in `allowed` → silently overridden to the
  recommendation. The server doesn't reveal *why* a choice was rejected
  to avoid leaking floor-rule structure.
- Unknown `confirmation_id` → silently ignored.
- Default timeout is 120 s; on timeout the gate falls back to BLOCKED.
- The server cancels pending confirmations when the channel stops, so
  clients should treat disconnect as implicit cancel.

---

## 5. Limits & roadmap

### 5.1 What's solid right now

- **NORMAL** (no entities or LOW only) — passthrough.
- **BLOCKED** (CATASTROPHIC, hard-blocked types, budget exhausted) — informational refusal; cloud never called.
- **METRIC_DP** (MEDIUM/HIGH with backend configured) — provable (ε)-dχ-privacy on the anonymised tokens; restored on the way back.
- **CLI + WebSocket** interactive confirmation.
- **Audit log** with metadata-only JSONL (no raw entities).
- **ε budget management** across `(ε_session, ε_user_24h)` with on-disk persistence and user-id hashing.

### 5.2 What's intentionally NOT implemented yet

- **K-decoy (M2)** — Generate k-1 "self-consistent" decoys and send all k to the cloud. We documented why this gives only *computational* plausible deniability (vs Metric-DP's formal guarantee), so it's a lower-priority "fast path" rather than a strict requirement. The decider has a `k_decoy_supported` flag that's hard-wired to False; turning it on is a small follow-up PR once we want it.
- **LM-assisted Restorer rewriting** — The Restorer currently does deterministic substitution. If the cloud paraphrases the assigned pseudonym (e.g. "Bob" → "Robert"), the swap misses. Hook is reserved (`backend=`, `use_llm_rewrite=`) but not invoked.
- **Tool-output filtering (M4)** — GateKeeper only inspects the user's inbound message. Filesystem reads, MCP outputs, web fetches, etc. flow to the cloud unfiltered.
- **Envelope-encrypted audit (M4)** — Current logs are plaintext metadata-only JSONL. A KMS / age recipient wrap is reserved (`audit.kms_recipient` config field is already there but not honoured at write time).
- **WebUI rendering** of the `privacy_confirmation` envelope — protocol is wired and tested; the React side renders nothing yet. Adding the popup is a frontend-only PR.
- **Routing side-channel mitigation** (`routing_mode: balanced` / cover-traffic randomisation) — config field exists; the conservative-only path is the only one wired.

### 5.3 Milestones

| ID | Status | Adds |
|---|---|---|
| M1 | ✅ | Detector + decider + confirmation + audit; BLOCKED + NORMAL paths. |
| M1.5 | ✅ | SIMPLE stub closed; `LocalModelBackend` abstraction; CLI + WebSocket interactive confirmation; `local_model` via providers registry; `LLMSemanticDetector` auto-wired. |
| M3 | ✅ | Multivariate Laplace; MetricDPTransform; PrivacyAccountant; Restorer; full GateKeeper / AgentLoop integration. |
| M2 | not started | K-decoy + pseudonym layer. Optional "fast path"; weaker guarantees than M3 — most users will prefer Metric-DP. |
| M4 | not started | Tool-output PII scanning; envelope-encrypted audit; routing-side-channel mitigation; MyTool guard. |
| WebUI | not started | Render the `privacy_confirmation` envelope as a popup in the React UI. |

---

## 6. File map (for reviewers)

```
nanobot/privacy/
  __init__.py
  types.py               # enums + dataclasses
  detector.py            # regex + plug-in semantic hook
  semantic_detector.py   # LM-backed second pass + auto-wire helper
  decider.py             # recommendation tree + allowed-set
  confirmation.py        # mode + preselect + timeout + fallback
  audit.py               # JSONL append-only
  local_model.py         # LocalModelBackend protocol + LLMProviderBackend
  metric_dp.py           # Laplace sampler (Andrés 2013 / Feyisetan 2020)
  transform.py           # MetricDPTransform: embed + noise + nearest neighbour
  accountant.py          # ε budget (ε_session, ε_user_24h)
  restorer.py            # de-anonymize cloud responses
  gate.py                # façade, wired by from_config
  cli_channel_caps.py    # CLI interactive confirmation

nanobot/agent/loop.py        # TurnState.GATE + _state_gate + _state_run restore
nanobot/channels/base.py     # privacy_capabilities() hook
nanobot/channels/websocket.py # privacy_confirmation / _reply envelopes + Future bridge
nanobot/providers/base.py    # LLMProvider.embed() default
nanobot/providers/openai_compat_provider.py  # /v1/embeddings + httpx fallback
nanobot/providers/factory.py # make_provider(model_override=…)
nanobot/config/schema.py     # PrivacyConfig + sub-configs
nanobot/cli/commands.py      # CLI capability registration

tests/privacy/
  test_detector.py
  test_decider.py
  test_confirmation.py
  test_gate_and_audit.py
  test_local_model.py
  test_semantic_detector.py
  test_metric_dp.py
  test_transform.py
  test_accountant.py
  test_restorer.py
  test_gate_integration.py
tests/channels/test_websocket_privacy_confirmation.py
tests/providers/test_embed.py

scripts/
  debug_privacy.py             # pipeline diagnostic (config → … → embeddings)
  visualize_metric_dp.py       # Laplace sampler histogram + scatter
  visualize_token_replacement.py  # ε-vs-distribution trade-off

.agent/privacy_gatekeeper.md   # design spec
docs/privacy-gatekeeper-m1.md  # this file
```

Production code added across M1–M3: ~3,500 lines.
Test code added: ~3,000 lines (190+ dedicated privacy cases).

---

## 7. References

- Sweeney, L. (2002). *k-anonymity: A model for protecting privacy.* IJUFKS.
- Andrés, M. E. et al. (2013). *Geo-indistinguishability: Differential privacy for location-based systems.* CCS.
- Feyisetan, O. et al. (2020). *Privacy- and utility-preserving textual analysis via calibrated multivariate perturbations.* WSDM (MADLIB).
- Yue, X. et al. (2021). *Differential privacy for text analytics via natural text sanitization.* ACL (SANTEXT).
- Dwork, C. & Roth, A. (2014). *The Algorithmic Foundations of Differential Privacy.*
