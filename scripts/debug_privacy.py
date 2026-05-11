"""Privacy GateKeeper diagnostic — quickly find where the pipeline drops the message.

Run with the same config nanobot uses:
    python scripts/debug_privacy.py --config /root/.nanobot/config.json \
        -m "我是张三，住在朝阳区芍药居北里 23 号，最近在治糖尿病"

Prints each layer's state so you can spot where the message stops triggering:
  1. Is privacy enabled in the loaded config?
  2. Did `local_model` resolve to a real backend?
  3. Does the backend actually answer when called?
  4. What does the detector return on this text?
  5. What does the decider recommend?
  6. What does the confirmation gate decide?
  7. What does the transformer hand back to AgentLoop?
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from nanobot.config.loader import load_config
from nanobot.privacy.gate import GateKeeper
from nanobot.privacy.local_model import build_from_root_config
from nanobot.privacy.types import ChannelCapabilities


def _line(label: str = "") -> None:
    print(("─── " + label + " ").ljust(72, "─"))


async def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("-m", "--message", required=True)
    args = p.parse_args()

    _line("1. config")
    cfg = load_config(Path(args.config) if args.config else None)
    print(f"  privacy.enabled        = {cfg.privacy.enabled}")
    print(f"  privacy.local_model    = {cfg.privacy.local_model!r}")
    print(f"  privacy.confirmation   = mode={cfg.privacy.confirmation.mode}, "
          f"threshold={cfg.privacy.confirmation.risk_threshold}, "
          f"timeout={cfg.privacy.confirmation.timeout_seconds}s, "
          f"on_timeout={cfg.privacy.confirmation.on_timeout}")
    print(f"  privacy.audit          = enabled={cfg.privacy.audit.enabled}, "
          f"dir={cfg.privacy.audit.log_dir}")
    if not cfg.privacy.enabled:
        print("\n  ✗ privacy.enabled is FALSE — GateKeeper never runs. "
              "Set `privacy.enabled = true` in config.json.")
        return 1

    _line("2. local_model backend")
    backend = build_from_root_config(cfg)
    if backend is None:
        print("  ✗ build_from_root_config returned None.")
        if not cfg.privacy.local_model:
            print("    → privacy.local_model is empty; SemanticDetector will not run.")
            print("      Set e.g. privacy.local_model = 'ollama/qwen2.5:0.5b' and configure")
            print("      providers.ollama.api_base = 'http://localhost:11434'.")
        else:
            print(f"    → make_provider({cfg.privacy.local_model!r}) raised. The factory swallows")
            print("      the exception. Re-run with --logs for the underlying error, or call")
            print("      `from nanobot.providers.factory import make_provider; "
                  "make_provider(cfg, model_override=...)` directly to see the traceback.")
    else:
        print(f"  ✓ backend       = {backend.name}")
        print(f"    is_available  = {backend.is_available()}")
        try:
            response = await backend.generate(
                "Reply with only the word OK.", max_tokens=8, temperature=0.0
            )
            print(f"    smoke test    = {response!r}")
            if not response:
                print("    ⚠ empty smoke-test response — provider may be unreachable or model not pulled.")
        except Exception as exc:  # noqa: BLE001
            print(f"    ✗ generate() raised: {exc!r}")

    _line("3. detector")
    gate = GateKeeper.from_config(cfg.privacy, local_model=backend)
    rec = await gate.detect_and_recommend(args.message)
    if not rec.entities:
        print("  ✗ no entities detected.")
        print("    → If you expected the regex layer to hit (email/phone/key…),")
        print("      check that the substring matches the patterns in")
        print("      nanobot/privacy/detector.py.")
        print("    → If you expected the SemanticDetector (names/addresses…),")
        print("      verify the backend smoke-test above returned text.")
    else:
        for e in rec.entities:
            print(f"  • {e.type.value:18} {e.risk_class.value:13} "
                  f"conf={e.confidence:.2f}  detector={e.detector:18}  value={e.value!r}")

    if backend is not None and backend.is_available():
        _line("3b. semantic-LM raw call")
        from nanobot.privacy.semantic_detector import _PROMPT_TEMPLATE, _parse_json_array

        prompt = _PROMPT_TEMPLATE.format(text=args.message)
        try:
            lm_raw = await backend.generate(prompt, max_tokens=256, temperature=0.0)
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ generate() raised: {exc!r}")
            lm_raw = ""
        print(f"  prompt length    = {len(prompt)} chars")
        print(f"  raw response     = {lm_raw!r}")
        parsed = _parse_json_array(lm_raw)
        print(f"  parsed JSON      = {parsed}")
        if not lm_raw:
            print("  ⚠ LM returned empty — model may not understand the task. Try a stronger model.")
        elif not parsed:
            print("  ⚠ LM returned text but parser found no JSON array. Either the model didn't")
            print("    follow the JSON format (try a stronger model / different prompt), or you")
            print("    need to extend nanobot/privacy/semantic_detector._parse_json_array.")
        elif parsed and not rec.entities:
            print("  ⚠ LM produced JSON but every value was hallucinated (not found in source text).")
            print("    Inspect the values above — small models sometimes paraphrase Chinese names.")

    _line("4. decider recommendation")
    print(f"  recommended path = {rec.path.value}")
    print(f"  reason           = {rec.reason}")
    print(f"  allowed set      = {sorted(p.value for p in rec.allowed)}")

    _line("5. confirmation (channel: cli, no interactive)")
    decision = await gate.confirm(
        rec, chat_id="diag", channel_name="cli",
        capabilities=ChannelCapabilities(),
    )
    print(f"  final path       = {decision.path.value}")
    print(f"  source           = {decision.source.value}")
    if decision.refusal_message:
        print(f"  refusal          = {decision.refusal_message[:120]}")

    _line("6. transformer")
    outcome = gate.transform(decision, args.message)
    msg = outcome.privacy_message
    if isinstance(msg, list):
        print(f"  privacy_message  = list of {len(msg)} items")
    else:
        print(f"  privacy_message  = {msg[:120]!r}{'…' if len(msg) > 120 else ''}")

    _line("done")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
