"""Additional cloud-trust-boundary tests around tool-result + session history.

After M3 step 6 wired Metric-DP end-to-end, two leak vectors remained:

1. **Forward direction** — when the runner restored tool-call arguments
   so a local tool (cron / write_file / …) acts on real values, the
   tool's return string then contained those real values. If that
   string is appended to the message list and passed back to the cloud
   for the next iteration, the cloud now observes the user's real PII.
   This breaks dχ-privacy on multi-iteration turns.

2. **Reverse direction** — session history was persisted with the
   anonymized text from the cloud, so the user reading their own chat
   log later sees a parade of `casey.nguyen@example.io`-style
   pseudonyms instead of the email they actually typed.

The runner re-anonymizes tool results before passing them to the next
cloud call, and the AgentLoop restores `ctx.all_messages` before
`_save_turn`. These tests exercise the building block (`_walk_restore`)
that both fixes share, plus a couple of end-to-end shapes.
"""

from __future__ import annotations

from nanobot.agent.runner import _sync_restore, _walk_restore

# --- inverse-direction substitution: real → pseudonym -------------------------------


def test_inverse_mapping_anonymizes_tool_result():
    """A tool that returned `Send email to alice@x.com` must, when seen
    by the cloud on the next iteration, look like the pseudonym."""
    forward = {"casey.nguyen@example.io": "alice@x.com"}  # pseudo → real
    inverse = {v: k for k, v in forward.items()}           # real → pseudo
    tool_result = "Created job 'Send email to alice@x.com about tomorrow'"
    out = _walk_restore(tool_result, inverse)
    assert "alice@x.com" not in out
    assert "casey.nguyen@example.io" in out


def test_inverse_mapping_walks_nested_tool_payload():
    """Tool results may be JSON-shaped; the walker handles arbitrary nesting."""
    forward = {"morgan.davis@example.org": "alice@x.com"}
    inverse = {v: k for k, v in forward.items()}
    tool_result = {
        "ok": True,
        "log": ["sent draft to alice@x.com", "alice@x.com queued"],
        "meta": {"recipient": "alice@x.com", "count": 2},
    }
    out = _walk_restore(tool_result, inverse)
    assert "alice@x.com" not in str(out)
    assert "morgan.davis@example.org" in str(out)
    assert out["ok"] is True
    assert out["meta"]["count"] == 2


# --- forward direction: anon → real (session history restore) ----------------------


def test_session_history_restore_walks_message_list():
    """Mirror of what `_state_save` does on `ctx.all_messages`."""
    mapping = {"casey.nguyen@example.io": "alice@x.com"}
    all_messages = [
        {"role": "user",      "content": "Please email casey.nguyen@example.io tomorrow."},
        {"role": "assistant", "content": "Sure, drafting a note to casey.nguyen@example.io."},
        {"role": "tool",      "content": "Created job 'Send email to casey.nguyen@example.io'"},
        {"role": "assistant", "content": "I've scheduled a reminder."},
    ]
    out = _walk_restore(all_messages, mapping)
    # Original message body, the way the user actually typed it:
    assert out[0]["content"].endswith("alice@x.com tomorrow.")
    # Assistant's mention restored too:
    assert "alice@x.com" in out[1]["content"]
    # Even tool result is restored — it had the pseudonym after our
    # earlier re-anonymization step in the runner, so the restore at
    # save time brings it back to the real value for the local log.
    assert "alice@x.com" in out[2]["content"]
    # Original list is not mutated.
    assert "casey.nguyen@example.io" in all_messages[0]["content"]


def test_walk_restore_preserves_tool_call_args_in_assistant_messages():
    """Assistant messages can carry `tool_calls` with arbitrary args."""
    mapping = {"casey.nguyen@example.io": "alice@x.com"}
    msg = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "1",
                "type": "function",
                "function": {
                    "name": "cron.add",
                    "arguments": '{"message": "Send email to casey.nguyen@example.io"}',
                },
            }
        ],
    }
    out = _walk_restore(msg, mapping)
    args_str = out["tool_calls"][0]["function"]["arguments"]
    # JSON-string is treated as a leaf string and substring-replaced; the
    # JSON shape survives because pseudonyms never collide with quotes/braces.
    assert "alice@x.com" in args_str
    assert "casey.nguyen@example.io" not in args_str


# --- round-trip: real → anon → real -----------------------------------------------


def test_real_anon_real_round_trip_via_runner_helpers():
    """Verify the round-trip the M3 pipeline does within a single turn:

      raw user message → anonymize → cloud sees pseudo →
      tool runs on real (after restore) → tool returns real →
      runner re-anonymizes for next cloud iteration → cloud only sees pseudo →
      eventually save_turn restores everything back for the local log.
    """
    forward = {"casey.nguyen@example.io": "alice@x.com"}  # how Restorer uses it
    inverse = {v: k for k, v in forward.items()}           # how re-anon uses it

    user_raw = "email alice@x.com tomorrow"
    # Step A: anonymize for cloud
    for_cloud = _walk_restore(user_raw, inverse)
    assert "alice@x.com" not in for_cloud
    assert "casey.nguyen@example.io" in for_cloud

    # Step B: cloud emits tool call referencing the pseudo (we don't
    # model it here — just assume tool got the real value via the M3
    # tool-arg restore that already shipped).
    tool_result_real = "Created job 'Send email to alice@x.com'"

    # Step C: runner re-anonymizes the tool result before next cloud call
    tool_result_for_cloud = _walk_restore(tool_result_real, inverse)
    assert "alice@x.com" not in tool_result_for_cloud

    # Step D: cloud final reply mentions only the pseudo
    cloud_reply = "I've scheduled an email to casey.nguyen@example.io."

    # Step E: Restorer + session-history restore give the user the real value
    user_visible = _walk_restore(cloud_reply, forward)
    assert user_visible == "I've scheduled an email to alice@x.com."


# --- regression: _sync_restore unchanged --------------------------------------------


def test_sync_restore_still_works_after_the_re_anonymization_addition():
    out = _sync_restore("Bob and Bobby", {"Bob": "Alice"})
    assert out == "Alice and Bobby"
