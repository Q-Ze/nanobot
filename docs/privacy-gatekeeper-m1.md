# Privacy GateKeeper — M1 Release Notes

> **Status:** M1 shipped 2026-05-11. See `.agent/privacy_gatekeeper.md` for the full
> design spec, `add_new_fuction.md` for the original requirements.

This document summarizes what M1 added to the codebase, how to enable it, and
how to test it locally. M1 is the **foundation milestone**: detector, decider,
confirmation flow, audit, and AgentLoop integration. It does **not** include
K-decoy or Metric-DP transformations — those arrive in M2/M3.

---

## 1. What was added

### 1.1 New module: `nanobot/privacy/`

| File | Purpose |
|---|---|
| `types.py` | Shared enums + dataclasses (`ExecutionPath`, `RiskClass`, `EntityType`, `Decision`, `Recommendation`, `ChannelCapabilities`, …) and strictness-ordering helpers. |
| `detector.py` | `PrivacyEntityDetector`: regex layer (email, phone E.164 + CN, CN ID number with checksum, bank card with Luhn, IP, cloud credentials, JWT, SSH/PEM, high-entropy strings) + pluggable semantic LM hook (defaults to no-op). |
| `decider.py` | `ExecutionDecider`: pure function `inputs → Recommendation(path, allowed_set, reason)`. Implements the §3.2 decision tree and the §4.5 safety-floor matrix. |
| `confirmation.py` | `ConfirmationGate`: handles `mode ∈ {always, risk_threshold, never}`, user pre-selection, interactive ask with timeout, channel-fallback policies. |
| `audit.py` | `AuditLogger`: append-only JSONL. **Never writes raw entity values.** Stores `path / recommended_path / source / entity_counts / risk_counts / latency / downgrade_flag`. |
| `gate.py` | `GateKeeper` facade — single import surface used by `AgentLoop`. |

### 1.2 Modified files

| File | Change |
|---|---|
| `nanobot/config/schema.py` | New `PrivacyConfig` (+ four sub-configs) added to root `Config`. Defaults are off; nothing changes for existing users until they set `privacy.enabled = true`. |
| `nanobot/channels/base.py` | Two new attributes (`privacy_supports_interactive_confirm`, `privacy_confirmation_max_latency_seconds`) and `privacy_capabilities()` hook for channels to declare interactive-confirmation support. Default is non-interactive. |
| `nanobot/agent/loop.py` | Added `TurnState.GATE` between `COMMAND` and `BUILD`, new `_state_gate` handler, transitions, and `privacy_config` constructor parameter. Existing state handlers are untouched. |

### 1.3 Tests (`tests/privacy/`)

43 unit tests across four files: detector, decider, confirmation, gate+audit.
All run synchronously without network or LLM calls.

---

## 2. Enable & use

### 2.1 Opt-in (default: off)

Privacy GateKeeper is **disabled by default**. Edit `~/.nanobot/config.json`:

```json
{
  "privacy": {
    "enabled": true
  }
}
```

That alone gives you:
- All catastrophic entities (API keys, SSH keys, JWT, internal instructions) **hard-blocked** before reaching the cloud LLM.
- All HIGH-risk entities (CN ID, bank card, medical, high-entropy strings) **blocked** in M1 (M3 will route them to Metric-DP).
- All MEDIUM-risk entities (email, phone, name, address) **blocked** in M1 (M2 will route them to K-decoy).
- LOW-risk and entity-free messages pass through unchanged.
- All decisions logged to `~/.nanobot/privacy_audit/audit-YYYYMMDD.jsonl`.

### 2.2 Full configuration surface

```json
{
  "privacy": {
    "enabled": true,
    "local_model": "ollama/qwen2.5:0.5b",
    "risk_class_overrides": {
      "email": "low",
      "ip": "medium"
    },
    "regex_extensions": [
      "INTERNAL-\\d{6}"
    ],
    "routing_mode": "conservative",
    "confirmation": {
      "mode": "risk_threshold",
      "risk_threshold": "high",
      "timeout_seconds": 60,
      "on_timeout": "block",
      "channel_fallback_default": "forced_conservative",
      "channel_fallback_overrides": {
        "webhook": "use_recommended",
        "email_bridge": "reject"
      }
    },
    "audit": {
      "enabled": true,
      "log_dir": "~/.nanobot/privacy_audit"
    }
  }
}
```

| Key | Meaning |
|---|---|
| `local_model` | A model identifier in the same shape as `agents.defaults.model` (e.g. `"ollama/qwen2.5:0.5b"`, `"lm_studio/Qwen2.5-1.5B"`, `"anthropic/claude-haiku-4-5"`). The provider is resolved through the existing `providers.*` config blocks — no separate endpoint/auth needs to be configured here. M1.5 instantiates the backend but doesn't yet use it (SemanticDetector / K-decoy / Metric-DP arrive in M2/M3). |
| `risk_class_overrides` | Demote/promote a specific entity type's risk class. Use this to fix systematic false positives instead of per-message overrides. |
| `regex_extensions` | Extra Python regex patterns that flag user-defined sensitive strings (mapped to `EntityType.OTHER`, risk class `LOW`). |
| `confirmation.mode` | `always` = ask every relevant turn; `risk_threshold` (default) = only when risk ≥ threshold or path is non-trivial; `never` = silent automated decisions. |
| `confirmation.on_timeout` | `block` (default, fail-closed) or `recommended` (use system suggestion when user is slow). |
| `channel_fallback_*` | What to do on channels that can't ask interactively. `forced_conservative` = pick the strictest path in the allowed set; `use_recommended` = trust the system suggestion; `reject` = bounce the message. |

**Local model selection.** If `local_model` is set, the GateKeeper builds
an `LLMProviderBackend` using the same `make_provider` machinery that
constructs the main agent provider. You configure auth/endpoint exactly
where you already do — e.g. `providers.ollama.api_base` for an Ollama
endpoint. Pointing `local_model` at a cloud model (Anthropic, OpenAI…)
is supported too; "local" here is a role, not a hard locality requirement.

### 2.3 User-supplied path preference (SDK / power users)

Any caller can supply `metadata["privacy_path"]` in the inbound message to
pre-select a path. Allowed values: `"normal" | "simple" | "k_decoy" | "metric_dp" | "blocked"`.

**Safety floor still applies:** if your pre-selection violates the safety floor
(e.g. you ask for `normal` on a message containing a CATASTROPHIC entity),
the GateKeeper silently overrides to the system recommendation and records
`violation_attempt` in the audit log — it will not raise or leak which rule
you tripped.

Example (Python SDK):

```python
from nanobot.bus.events import InboundMessage

msg = InboundMessage(
    channel="cli",
    sender_id="alice",
    chat_id="direct",
    content="My email is alice@example.com — help me draft a reply.",
    metadata={"privacy_path": "blocked"},   # user explicitly chooses to NOT send
)
```

---

## 3. What you'll see at runtime

### 3.1 A blocked turn

User input: `"My API key is sk-ant-abc123… please test it"`

Outbound message content:
```
Privacy GateKeeper blocked this message. Reason: hard_secret:credential.
Please remove the sensitive content and try again.
```

The cloud LLM is **never called**. The decision goes through `_state_gate → DONE`.

### 3.2 A passing turn (no entities or LOW only)

User input: `"What's the capital of France?"`

GateKeeper attaches `msg.metadata["privacy"] = {path: "normal", recommended_path: "normal", source: "system_auto", fidelity: "EXACT"}` and forwards the message unchanged. The cloud LLM call proceeds normally.

### 3.3 Audit log entry

```jsonl
{"decision_reason": "hard_secret:credential", "entity_counts": {"credential": 1}, "eps_consumed": 0.0, "fidelity": "EXACT", "path": "blocked", "path_source": "system_auto", "recommended_path": "blocked", "risk_counts": {"catastrophic": 1}, "session_id_hash": "1a2b3c…", "ts": "2026-05-11T12:34:56.789+00:00", "user_choice_latency_ms": null, "user_downgrade": false, "violation_attempt": null}
```

Note: **`value` of the credential is NOT in the log line.** Only counts and types are.

---

## 4. How to test

### 4.0 Try it from the CLI in one minute

Enable GateKeeper in your config (or use `NANOBOT_PRIVACY__ENABLED=true`):

```json
{ "privacy": { "enabled": true } }
```

Then send a message that contains an obvious credential:

```bash
nanobot agent -m "Hello, my api key is sk-xsadsafsgdrghr" \
              --config ~/.nanobot/config.json
```

What you should see:

1. The CLI prints a **🛡 Privacy GateKeeper** banner listing the detected
   entity (`credential / catastrophic`), the recommended path (`blocked`),
   and your options.
2. The default mode is `risk_threshold` — because the entity is
   `catastrophic`, the gate asks you to confirm. Press Enter to accept the
   recommendation (block), or `[c]` to cancel.
3. The cloud LLM is **never called** when the path is `blocked`. You get a
   refusal message instead.
4. An entry appears in `~/.nanobot/privacy_audit/audit-YYYYMMDD.jsonl`.

If you want fully automated runs without interactive prompts, set
`privacy.confirmation.mode = "never"` — the gate then applies its
recommendation silently.

### 4.1 Run the privacy unit tests

```bash
pytest tests/privacy/ -v
```

Expected: 43 tests pass.

### 4.2 Regression-check the rest of the suite

```bash
pytest tests/agent/ tests/config/ -q
```

Expected: 796 + 30 tests pass (the loop.py integration adds a state but does
not change any existing handler behavior).

### 4.3 Lint

```bash
ruff check nanobot/privacy/ tests/privacy/ nanobot/agent/loop.py nanobot/channels/base.py nanobot/config/schema.py
```

Expected: `All checks passed!`.

### 4.4 End-to-end smoke (no LLM call needed)

```python
import asyncio
from nanobot.config.schema import PrivacyConfig
from nanobot.privacy import GateKeeper
from nanobot.privacy.types import ChannelCapabilities

async def main():
    cfg = PrivacyConfig(enabled=True)
    # Don't ask interactively for the smoke test
    cfg.confirmation.mode = "never"
    gate = GateKeeper.from_config(cfg)

    for text in [
        "Hello, how are you?",                                  # NORMAL
        "Email me at alice@example.com",                         # BLOCKED (medium, no path in M1)
        "Use this AWS key: AKIAIOSFODNN7EXAMPLE for upload",     # BLOCKED (catastrophic)
        "My CN ID is 110101199003079577 for the form",           # BLOCKED (high, no DP in M1)
    ]:
        rec = await gate.detect_and_recommend(text)
        decision = await gate.confirm(rec, chat_id="c", channel_name="x",
                                       capabilities=ChannelCapabilities())
        out = gate.transform(decision, text)
        print(f"{text[:50]:50}  →  {decision.path.value:10}  (reason: {rec.reason})")

asyncio.run(main())
```

Expected output:
```
Hello, how are you?                                →  normal      (reason: no_privacy_entity_detected)
Email me at alice@example.com                      →  blocked     (reason: no_anonymization_path_available_yet)
Use this AWS key: AKIAIOSFODNN7EXAMPLE for upload  →  blocked     (reason: hard_secret:credential)
My CN ID is 110101199003079577 for the form        →  blocked     (reason: no_anonymization_path_available_yet)
```

### 4.5 Full agent loop test

Enable GateKeeper in your config, start nanobot, and send a message containing
sensitive content through any channel (CLI, Telegram, WebSocket, …). You
should see the refusal message returned and an entry appear in
`~/.nanobot/privacy_audit/audit-*.jsonl`.

```bash
# Enable GateKeeper
cat > /tmp/privacy_demo.json <<'EOF'
{"privacy": {"enabled": true, "confirmation": {"mode": "never"}}}
EOF

# Merge it into your config (or set NANOBOT_PRIVACY__ENABLED=true)
nanobot --config /tmp/privacy_demo.json chat "test my key sk-ant-test123 please"

# Check the audit log
tail -1 ~/.nanobot/privacy_audit/audit-*.jsonl | python -m json.tool
```

### 4.6 WebSocket wire protocol (for custom clients / WebUI authors)

When `privacy.enabled = true` and a WebSocket client is subscribed to a
chat, the GateKeeper asks for confirmation by broadcasting an outbound
envelope on the existing connection. The client replies with an inbound
envelope; AgentLoop's turn is suspended until either the reply arrives or
the configured timeout expires (default 120 s for WebSocket, fail-closed
to BLOCKED on timeout).

**Server → client** (outbound event, alongside `message` / `delta` etc.):

```jsonc
{
  "event": "privacy_confirmation",
  "chat_id": "<chat>",
  "confirmation_id": "<32-char hex>",
  "path": "blocked",                       // system-recommended path
  "reason": "hard_secret:credential",       // machine-friendly explanation
  "allowed": ["blocked"],                   // user may only choose from this set
  "entities": [
    {
      "type": "credential",
      "risk_class": "catastrophic",
      "linkability": "single_use",
      "value": "sk-xsadsafsgdrghr",         // user's own data — show in the UI
      "span": [21, 38],
      "confidence": 1.0,
      "detector": "regex:sk_prefixed"
    }
  ]
}
```

**Client → server** (inbound envelope, sharing the same channel):

```jsonc
{
  "type": "privacy_confirmation_reply",
  "confirmation_id": "<32-char hex>",       // must echo the prompt's id
  "chosen_path": "blocked"                   // or null to cancel the message
}
```

Rules clients should rely on:

- Any value in `chosen_path` that is **not** in the prompt's `allowed`
  list is silently overridden to the system recommendation
  (`violation_attempt` is recorded in the audit log). This is
  intentional — the server does not reveal *why* a choice was rejected
  to avoid leaking floor-rule structure to attackers.
- `chosen_path: null` means "cancel this turn" (the message is treated
  as BLOCKED with a cancellation refusal).
- Replies with an **unknown** `confirmation_id` are silently ignored
  (same reasoning). Malformed envelopes (missing `confirmation_id`,
  unrecognized `chosen_path` string) generate an `error` event back.
- The server cancels every pending confirmation when the channel stops,
  so clients should treat connection loss as an implicit cancel.

---

## 5. Limits & roadmap

### 5.1 What M1 cannot do (yet)

- **Anonymize and forward.** Any MEDIUM/HIGH entity blocks the message in M1.
  This is intentional fail-closed behavior pending the K-decoy (M2) and
  Metric-DP (M3) transformers.
- **Built-in semantic (small-model) detection.** M1 ships a regex layer
  only; the `SemanticDetector` slot is a no-op. Two ways to fill it:
  - **Wait for M2/M3** — a packaged local-LM-based detector is on the
    roadmap. The `privacy.local_model` config field is reserved for it.
  - **Plug your own now**: implement the protocol and inject it:

    ```python
    from nanobot.privacy import GateKeeper
    from nanobot.privacy.detector import SemanticDetector

    class MySmallLM:
        async def detect(self, raw_message, regex_hits):
            # call your local model here (Ollama, LM Studio, llama.cpp, …)
            # return a list of nanobot.privacy.types.DetectedEntity
            return []

    gate = GateKeeper.from_config(config.privacy, semantic_detector=MySmallLM())
    ```
- **Interactive confirmation in non-CLI channels.** CLI and **WebSocket**
  are wired (see §4.0 for CLI; §4.6 below for the WebSocket wire
  protocol). Telegram, Discord, Slack all fall back to
  `forced_conservative` until each implements
  `BaseChannel.privacy_capabilities()` with real `send_confirmation` /
  `await_confirmation_reply` callbacks. The WebUI frontend rendering on
  top of the WebSocket protocol is a follow-up PR.
- **Tool-output filtering.** GateKeeper only inspects the user's inbound
  message. Filesystem reads, MCP tool outputs, etc. are not scanned. M4 adds
  catastrophic-entity scanning on outbound tool results.
- **Envelope-encrypted audit.** M1 audit logs are plaintext JSONL (metadata
  only — no raw entities). M4 wraps the file with a KMS / age recipient.

### 5.2 Upcoming milestones

| Milestone | Adds |
|---|---|
| **M2** | Pseudonym layer (session-keyed HMAC), `K_DECOY` transformer + restorer, `PrivacyAccountant` skeleton, first interactive-channel implementation (WebSocket). |
| **M3** | `METRIC_DP` (dχ-privacy) on token embeddings, restorer with local LM, UI fidelity labels, ε budget enforcement. |
| **M4** | Routing-side-channel mitigation, envelope-encrypted audit, tool-output catastrophic scanning, MyTool guard. |

---

## 6. File map (for reviewers)

```
nanobot/privacy/
  __init__.py
  types.py              # 200 lines  — enums + dataclasses
  detector.py           # 280 lines  — regex + Luhn + CN-ID checksum + entropy + merge
  decider.py            # 110 lines  — recommendation tree + allowed-set
  confirmation.py       # 200 lines  — mode + preselect + timeout + fallback
  audit.py              # 100 lines  — JSONL append-only
  gate.py               # 140 lines  — facade

nanobot/config/schema.py   # +60 lines (PrivacyConfig + 4 sub-configs)
nanobot/channels/base.py   # +30 lines (capability flags + hook)
nanobot/agent/loop.py      # +130 lines (TurnState.GATE + handler + wiring)

tests/privacy/
  __init__.py
  test_detector.py        # 15 cases
  test_decider.py         # 9 cases
  test_confirmation.py    # 11 cases
  test_gate_and_audit.py  # 8 cases

.agent/privacy_gatekeeper.md   # design spec (v1.1)
```

Total: ~1,030 lines of production code + ~530 lines of test code.
