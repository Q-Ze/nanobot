# Privacy GateKeeper — Changelog (post-M3)

> Companion to [`privacy-gatekeeper-m1.md`](./privacy-gatekeeper-m1.md).
> Tracks the seven commits that landed after the M3-complete tag
> ([`7c584d3`](https://github.com/anthropics/nanobot/commit/7c584d3)).
> All work in this changelog is on `main`; M1, M1.5 and M3 themselves
> are described in the main release notes.

The changes fall into three buckets:

1. **M2 K_DECOY shipped** — the third anonymisation path is now live, ~free of ε.
2. **Four cloud-trust-boundary fixes** — every place where data crosses
   the cloud boundary now anonymises or restores explicitly. Three were
   leaks surfaced by a live `nanobot agent` run; one was a UX leak where
   the streaming display bypassed the post-cloud restore.
3. **Two UX improvements** — the CLI confirmation prompt and the
   BLOCKED reason string both got clearer.

---

## 1. M2 v1 — K_DECOY anonymisation path

**Commit:** [`89118b6`](#) `feat(privacy): M2 v1 — KDecoyTransform (HMAC pseudonym + typed pool)`

### What it does

For each detected entity, replace the value with a deterministic
pseudonym drawn from a typed candidate pool:

```
idx       = HMAC_SHA256(deployment_key, session_key ‖ type ‖ value) mod len(pool)
pseudonym = pool[idx]
```

The mapping (`pseudonym → original`) is recorded in the same
`restoration_plan["mapping"]` shape that METRIC_DP uses, so the
Restorer, tool-arg restore and session-history restore all work
unchanged.

### Why a separate path

| | METRIC_DP (M3) | K_DECOY (M2) |
|---|---|---|
| ε consumed per call | `eps_query` (default 8) | **0** |
| Needs embedding backend | yes | no |
| Formal guarantee | ε-dχ-privacy (Andrés 2013) | computational k-deniability `≤ 1/K_eff − 1/2 + adv(prior)` |
| Default routing | HIGH risk | MEDIUM risk (saves ε) |
| Cross-turn consistency | nondeterministic (Laplace noise) | deterministic per (session_key, value) |

K_DECOY is the natural fallback when METRIC_DP isn't usable: ε budget
exhausted, embedding model down, or simply MEDIUM-risk content where
the formal guarantee is overkill. The decider now prefers it for
MEDIUM and reserves METRIC_DP for HIGH.

### Honest math

K_DECOY is **not** k-anonymity in the Sweeney sense (which is a
property of tabular releases, not free-text messages). It gives:

* To an attacker with **no prior**, the posterior on the true value
  is bounded by `1/K_effective` (where `K_eff = min(k_target, pool_size − 1)`).
* To an attacker **with prior** (chat context, account info, surrounding
  text), that bound collapses to roughly `1/K_eff − 1/2 + adv(prior)`.
* Cross-session unlinkability holds: different `session_key` → different
  pseudonym for the same value (HMAC keying).
* Cross-turn coherence holds: same `session_key` + same value → same
  pseudonym, so the cloud's multi-turn reasoning stays intact.

If you want a formal mathematical guarantee, use METRIC_DP. If you
want the ε-free fallback or have no embedding model, K_DECOY is the
right tool.

### How to disable

```json
{ "privacy": { "k_decoy": { "enabled": false } } }
```

Disabling K_DECOY routes MEDIUM PII back to METRIC_DP-or-BLOCKED.
Use this if you want the formal guarantee or nothing at all.

### Deployment-key resolution

`KDecoyTransform.__init__` requires a ≥16-byte `hmac_key`. The gate
resolves it via `resolve_hmac_key(source=...)` in this order:

1. `NANOBOT_PRIVACY_KEY` env var — either 32+ hex chars, or any
   string (gets SHA-256'd).
2. Persisted file at `~/.nanobot/privacy_audit/pseudo_key` — written
   `0o600` on first run with `os.urandom(32)`.

This means:
* Different deployments have different keys (cross-deployment
  unlinkability).
* The same deployment is stable across restarts (multi-turn coherence
  survives a process restart).

### Verification

```
nanobot agent --logs -m "Please email alice@x.com about the demo tomorrow."
```

You should see:

```
privacy.detect:    1 entity {'email': 1}; decider → k_decoy (reason: recommended_k_decoy)
privacy.k_decoy:   pseudonymized 1 entity (K_eff=3); 0 placeholders
privacy.restore:   substituted 1 pseudonym occurrence back to original
```

…and the terminal's final response contains `alice@x.com` (real
value, restored), not `robin.chen@example.com` (or whichever pool
slot the HMAC happened to land on).

### Tests added

* `tests/privacy/test_k_decoy.py` — 27 cases: determinism, cross-session
  unlinkability, cross-deployment unlinkability, hard-block defense,
  pool exclusion, right-to-left splicing, `resolve_hmac_key` envelope.
* `tests/privacy/test_gate_integration.py` — 4 K_DECOY end-to-end cases
  including the "METRIC_DP budget exhausted → K_DECOY still works"
  fallback.

---

## 2. Cloud-trust-boundary fixes (the most important section)

The M3 milestone shipped a working METRIC_DP pipeline but had four
leaks where data crossed the cloud boundary without being
anonymised / restored. Three surfaced on the very first live
`nanobot agent` run; one (streaming) was found by visually inspecting
the terminal output. All four are now closed.

### 2.1 Tool-call argument restoration

**Commit:** [`17bc0bf`](#) `fix(privacy): restore pseudonyms in tool-call arguments before execution`

**Bug**: After METRIC_DP rewrote `alice@x.com` to `casey.nguyen@example.io`,
the cloud LLM emitted `cron.add({"message": "Email casey.nguyen@example.io ..."})`.
The runner passed those args verbatim to the local cron tool, which then
persisted the pseudonym on disk. The user reading their own cron jobs
later saw fake email addresses they never typed.

**Fix**: Add `AgentRunSpec.privacy_mapping: dict[pseudonym → real]`,
plumb it from `_state_gate` through `_run_agent_loop`, and walk every
tool-call's arguments through `_restore_tool_arguments` (dict / list /
tuple / string recursion) before invoking the tool. The cron file
now reads `Email alice@x.com ...`.

### 2.2 Tool-result re-anonymisation + session history restore

**Commit:** [`56c6a22`](#) `fix(privacy): re-anonymize tool results & restore session history on save`

Two separate leaks fixed together:

**Forward leak**: After 2.1, the tool result string contained the
real `alice@x.com`. The runner then sent that result back to the
cloud on the next iteration, so on multi-iteration turns the cloud
ended up seeing the real value anyway — breaking dχ-privacy.

**Fix**: In the runner, build the inverse mapping (`real → pseudonym`)
and re-anonymise every tool result before appending it to the
message list. The cloud sees pseudonyms on every iteration.

**Reverse leak**: `_state_save` persisted the anonymised
`ctx.all_messages` to `~/.nanobot/workspace/sessions/<id>/history.json`.
The user reading their own chat log later saw a parade of
`casey.nguyen@example.io`-style pseudonyms instead of the email they
actually typed.

**Fix**: In `_state_save`, walk `ctx.all_messages` through
`_walk_restore(mapping)` before calling `_save_turn`. The session
log now reads as the user's view of the conversation.

### 2.3 Streaming display bypassed the restore

**Commits:** [`56d0f72`](#) (METRIC_DP) + [`1c2b139`](#) (K_DECOY)
`fix(privacy): suppress streaming on {METRIC_DP, K_DECOY} turns to prevent visible leak`

**Bug**: Without `--logs`, the user saw `jamie.singh@example.io`
flow onto the terminal token-by-token as the cloud streamed, even
though `_state_run` later called `gate.restore` against the
aggregated response. By the time restore ran, the pseudonym had
already been on screen for several seconds. Worse: the on-screen
text and the persisted session log disagreed.

**Fix**: In `_state_gate`, when `decision.path in {METRIC_DP, K_DECOY}`,
set `ctx.on_stream = None` and `ctx.on_stream_end = None`. The only
remaining display path is `_print_agent_response`, which receives
the post-restore `ctx.final_content`. The trade-off is no
incremental token rendering for anonymised turns — judged worth it
over leaking pseudonyms on screen.

### 2.4 K_DECOY forwarding bug

**Commit:** [`1c2b139`](#) `fix(privacy): forward K_DECOY turns to the cloud and suppress streaming`

**Bug**: `_state_gate` had
`allowed_to_forward = {NORMAL, METRIC_DP}` — K_DECOY was not in the
set, so K_DECOY turns short-circuited to the BLOCKED branch.
The "response" the user saw was the anonymised version of their own
input, echoed back as if from the model. The cloud LLM was never
actually called.

**Fix**: Add K_DECOY to `allowed_to_forward`. All other restore
plumbing was already generic (keys on `restoration_plan["mapping"]`),
so K_DECOY now inherits the full pipeline.

### Summary: every cloud-boundary now closed

| Boundary | Mechanism | Commit |
|---|---|---|
| User input → cloud | `MetricDPTransform` / `KDecoyTransform` | M3, M2 |
| Cloud → tool-call args | `_restore_tool_arguments` | 17bc0bf |
| Tool result → cloud (next iter) | inverse `_walk_restore` | 56c6a22 |
| Cloud → terminal (streaming) | suppress `on_stream` | 56d0f72, 1c2b139 |
| Cloud → session history | `_walk_restore` on `ctx.all_messages` | 56c6a22 |
| Cloud → final display | `gate.restore` on `ctx.final_content` | M3 |

---

## 3. UX improvements

### 3.1 CLI confirmation labels + pipeline breadcrumbs

**Commit:** [`a605f22`](#) `fix(privacy): visible CLI confirmation labels + pipeline breadcrumbs`

* `[c]` in the CLI prompt was being parsed by rich as a markup tag and
  rendering empty. Escaped to `\[c\]`.
* Per-path labels like "send with metric-DP noise (M3, ε-dχ-privacy)"
  in `_path_label` — was just "metric_dp" before.
* Three new info-level `loguru` lines so `--logs` shows the GateKeeper's
  state:
  * `privacy.detect: N entit{y,ies} {type_counts}; decider → <path> (reason: <r>)`
  * `privacy.metric_dp: anonymized N entit{y,ies} (ε=X spent); M placeholders`
  * `privacy.restore: substituted N pseudonym occurrence(s) back to original; M mapped values did not appear in the reply`

Entity counts and ε numbers are logged — raw values never are.

### 3.2 Budget-exhausted reason distinction

**Commit:** [`b1d507f`](#) `fix(privacy): distinguish budget-exhausted from path-unavailable in BLOCKED`

**Bug**: When METRIC_DP was the only viable anonymisation path
(K_DECOY disabled, no other fallback) but the ε accountant denied
the request, the decider correctly fell back to BLOCKED — but the
generic reason `no_anonymization_path_available_yet` made users
think the feature had regressed.

**Fix**: When `transform + backend` are present and only the
accountant is refusing, the gate rewrites the reason to:

```
eps_budget_exhausted (used 64.0/64.0 ε in 24h, next refresh in 21.5h)
```

Adds `PrivacyAccountant.time_until_next_refresh(user_id)` for the
oldest-entry timestamp lookup that powers the ETA. The confirmation
prompt and the audit log both pick up the new reason verbatim.

---

## 4. Updated math-guarantee table

The spec's §6 "honest guarantee" table now reads:

| Path | Formal guarantee | Status | Assumptions |
|---|---|---|---|
| BLOCKED | Data does not leave the device | ✅ shipped (M1) | local side not compromised |
| SIMPLE | Data does not leave the device | ⚠️ stub only, hard-disabled in decider | — |
| METRIC_DP | ε-dχ-privacy on transformed tokens; `(ε_session, ε_user_24h)` enforced by `PrivacyAccountant` | ✅ shipped (M3) | attacker has no local key; pool & user value share the embedding model |
| K_DECOY | Computational k-deniability — adversary advantage `≤ 1/K_eff − 1/2 + adv(prior)` | ✅ shipped (M2 v1) | poly-time attacker without strong prior; HMAC key not leaked; `K_eff = pool_size − 1` |
| NORMAL | **No protection** (by definition) | ✅ shipped (M1) | already determined no PII; routing-side-channel accepted |

---

## 5. What's still open

Not blocking anyone right now, but worth listing so the open issues
stay visible:

* **M4 — Tool output PII scanning**: filesystem / web-fetch / MCP
  results bypass the detector. Reading a file with PII and sending
  the bytes to the cloud is still a gap.
* **M4 — Routing side-channel**: NORMAL turns are observably different
  from anonymised turns (no transform call, different latency). Spec
  §4.4 describes `routing_mode: balanced` + cover-traffic; not built.
* **M4 — MyTool self-modification guard**: an agent rewriting its own
  prompt could route around the gate.
* **M4 — Envelope-encrypted audit**: spec §3.6 reserves
  `audit.kms_recipient`; not yet consumed.
* **SIMPLE path**: still hard-disabled in `from_config`. Wiring the
  branch is ~150 lines, but the bigger question is the
  task-difficulty classifier. Deferred until there's user demand.
* **K_DECOY v2 (multi-message)**: the original §3.3.1 proposal that
  sends K parallel cloud calls and discards K−1 responses; held back
  pending evidence the v1 anonymity set is too weak.
* **LM-assisted Restorer rewriting**: cloud rewriting `"Bob" → "Robert"`
  defeats string-level restore. Hook (`use_llm_rewrite=`) reserved.
* **Multi-process budget sync**: `PrivacyAccountant` is a single-process
  `RLock`. Multi-agent deployments could double-spend ε.
* **Chinese NER recall**: detector still leans on regex + English-leaning
  LLM detector prompt. Chinese names rely on the LM second pass alone.

---

## 6. Total cost so far

| Component | Lines (added) | Tests | Notes |
|---|---|---|---|
| M1 / M1.5 / M3 core | ~3.5k | 168 | shipped before this changelog |
| M2 v1 (this changelog) | ~950 | 31 | `nanobot/privacy/k_decoy.py` + tests + gate wiring |
| Cloud-boundary fixes | ~250 | 22 | tool-arg restore, tool-result re-anon, history restore, streaming suppression |
| UX fixes (A1 + breadcrumbs) | ~100 | 5 | reason override, helper, CLI label escapes |
| **Total** | **~4.8k** | **226** | full privacy suite green |

Full suite still runs in ~6 s.
