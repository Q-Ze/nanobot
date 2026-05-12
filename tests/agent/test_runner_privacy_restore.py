"""Tests for the tool-argument restoration helper used by AgentRunner.

When a turn goes through METRIC_DP the cloud LLM sees pseudonyms. Tool
calls it emits reference those pseudonyms. The runner must restore the
arguments back to real values before invoking the local tool, otherwise
side effects (cron jobs, file writes, MCP calls, …) get persisted with
pseudonyms — visible in the Restorer-applied response text but wrong on
disk.
"""

from __future__ import annotations

from nanobot.agent.runner import _restore_tool_arguments, _sync_restore

# --- _sync_restore (the synchronous twin of Restorer.restore) -------------------------


def test_sync_restore_basic_replace():
    out = _sync_restore(
        "Send email to bob@gmail.com tomorrow.",
        {"bob@gmail.com": "alice@example.com"},
    )
    assert out == "Send email to alice@example.com tomorrow."


def test_sync_restore_word_boundary_blocks_partial():
    """ASCII keys must use word-boundary semantics so 'Bob' doesn't
    consume the prefix of 'Bobby'."""
    out = _sync_restore("Bobby is not Bob.", {"Bob": "Alice"})
    assert out == "Bobby is not Alice."


def test_sync_restore_chain_protection():
    out = _sync_restore(
        "Bob met Alice.",
        {"Bob": "Alice", "Alice": "Eve"},
    )
    assert out == "Alice met Eve."


def test_sync_restore_chinese_substring():
    out = _sync_restore("我是李四的朋友。", {"李四": "张三"})
    assert out == "我是张三的朋友。"


def test_sync_restore_phone_number_with_leading_plus():
    """Keys that start with '+' would silently fail under naive \\b\\b
    (space-then-plus is non-word↔non-word, no boundary). Lookarounds
    catch this."""
    out = _sync_restore(
        "Call +1 415 555 0101 please.",
        {"+1 415 555 0101": "13800138000"},
    )
    assert out == "Call 13800138000 please."


def test_sync_restore_noop_when_empty():
    assert _sync_restore("", {"x": "y"}) == ""
    assert _sync_restore("hello", {}) == "hello"


# --- _restore_tool_arguments (recursive walk over dict / list / tuple) ---------------


def test_restore_walks_nested_dict():
    args = {
        "action": "add",
        "every_seconds": 86400,
        "message": "Send email to bob@gmail.com about demo",
    }
    out = _restore_tool_arguments(args, {"bob@gmail.com": "alice@example.com"})
    assert out["action"] == "add"
    assert out["every_seconds"] == 86400  # non-string preserved exactly
    assert out["message"] == "Send email to alice@example.com about demo"


def test_restore_walks_lists():
    args = {"recipients": ["bob@gmail.com", "carol@x.com"]}
    out = _restore_tool_arguments(
        args, {"bob@gmail.com": "alice@example.com",
               "carol@x.com": "diana@x.com"}
    )
    assert out["recipients"] == ["alice@example.com", "diana@x.com"]


def test_restore_walks_tuples():
    out = _restore_tool_arguments(
        ("hello bob@gmail.com",), {"bob@gmail.com": "alice@example.com"}
    )
    assert isinstance(out, tuple)
    assert out == ("hello alice@example.com",)


def test_restore_handles_deeply_nested_payloads():
    args = {
        "outer": {
            "schedule": {"kind": "every", "every_ms": 86400000},
            "payload": {
                "message": "Send to bob@gmail.com",
                "options": ["urgent", "ping bob@gmail.com again"],
            },
        }
    }
    out = _restore_tool_arguments(
        args, {"bob@gmail.com": "alice@example.com"}
    )
    assert out["outer"]["payload"]["message"] == "Send to alice@example.com"
    assert "ping alice@example.com again" in out["outer"]["payload"]["options"]


def test_restore_returns_input_when_mapping_empty():
    args = {"message": "Send to bob@gmail.com"}
    assert _restore_tool_arguments(args, {}) is args


def test_restore_returns_input_when_mapping_is_none():
    args = {"message": "x"}
    assert _restore_tool_arguments(args, None) is args


def test_restore_returns_input_when_arguments_is_none():
    assert _restore_tool_arguments(None, {"x": "y"}) is None


def test_restore_preserves_non_string_leaves():
    args = {"count": 3, "ratio": 0.5, "active": True, "tags": None}
    out = _restore_tool_arguments(args, {"x": "y"})
    assert out == args


def test_restore_real_world_cron_payload():
    """The exact shape the cron tool received in the user's M3 live run."""
    cron_args = {
        "action": "add",
        "message": "Send email to casey.nguyen@example.io about the demo tomorrow",
        "every_seconds": 86400,
    }
    out = _restore_tool_arguments(
        cron_args, {"casey.nguyen@example.io": "alice@x.com"}
    )
    # Side-effect (the on-disk cron message) now references the REAL email.
    assert out["message"] == "Send email to alice@x.com about the demo tomorrow"
    # Other args untouched.
    assert out["action"] == "add"
    assert out["every_seconds"] == 86400


def test_restore_is_failsoft_on_internal_error():
    """If the walker hits something pathological, return inputs unchanged
    rather than crash the tool call."""
    # Pass a non-string, non-container, non-None object — shouldn't blow up.
    class _Weird:
        pass

    args = _Weird()
    assert _restore_tool_arguments(args, {"a": "b"}) is args
