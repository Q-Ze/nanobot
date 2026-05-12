"""CLI channel capability for interactive privacy confirmation.

Bridges `ConfirmationGate` with stdin/stdout so the user can choose an
execution path from a terminal session. Used by ``nanobot agent`` (both
single-shot and interactive modes).

This module is intentionally small — channels with rich UIs (WebUI,
Telegram, Slack) should ship their own implementations of the same
ChannelCapabilities contract.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from nanobot.privacy.types import (
    ChannelCapabilities,
    ConfirmationPrompt,
    ConfirmationReply,
    ExecutionPath,
)


@dataclass
class _PendingPrompt:
    prompt: ConfirmationPrompt
    options: list[ExecutionPath]


@dataclass
class _CLIConfirmationState:
    """Shared state between the send_confirmation and await_confirmation_reply hooks."""

    console: Any                                 # rich.console.Console or anything with .print()
    pending: dict[str, _PendingPrompt] = field(default_factory=dict)

    async def send(self, chat_id: str, prompt: ConfirmationPrompt) -> None:
        rec = prompt.recommendation
        options = sorted(rec.allowed, key=lambda p: p.value)
        self.pending[prompt.confirmation_id] = _PendingPrompt(prompt=prompt, options=options)

        self.console.print()
        self.console.print("[bold yellow]🛡  Privacy GateKeeper[/bold yellow]")
        if rec.entities:
            self.console.print("  Detected entities:")
            for e in rec.entities:
                preview = e.value if len(e.value) <= 40 else e.value[:37] + "..."
                self.console.print(
                    f"    • [cyan]{e.type.value}[/cyan] "
                    f"[{_risk_color(e.risk_class.value)}]{e.risk_class.value}[/{_risk_color(e.risk_class.value)}] "
                    f"[dim]{preview!r}[/dim]"
                )
        self.console.print(
            f"  Recommended: [bold]{rec.path.value}[/bold] "
            f"[dim]({rec.reason})[/dim]"
        )
        self.console.print("  Options:")
        for i, p in enumerate(options, 1):
            marker = "[bold green]→[/bold green]" if p == rec.path else " "
            label = _path_label(p)
            # Escape the [i] / [c] tokens so rich doesn't read them as markup
            # tags. Without escaping, e.g. "[1]" or "[c]" would either render
            # empty or distort the colour state of the rest of the line.
            self.console.print(
                f"    {marker} \\[{i}] [bold]{p.value}[/bold] — {label}"
            )
        self.console.print("    \\[c] cancel (don't send this message)")
        self.console.print("[dim]Press Enter to accept the recommendation.[/dim]")

    async def wait(self, confirmation_id: str, timeout: float) -> ConfirmationReply | None:
        entry = self.pending.get(confirmation_id)
        if entry is None:
            return None
        try:
            raw = await asyncio.wait_for(
                asyncio.to_thread(_safe_input, "Choose [number / c / Enter]: "),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return None
        finally:
            self.pending.pop(confirmation_id, None)

        line = (raw or "").strip().lower()
        if line == "":
            return ConfirmationReply(
                confirmation_id=confirmation_id, chosen_path=entry.prompt.recommendation.path
            )
        if line in {"c", "cancel"}:
            return ConfirmationReply(confirmation_id=confirmation_id, chosen_path=None)
        # By number
        try:
            idx = int(line)
            if 1 <= idx <= len(entry.options):
                return ConfirmationReply(
                    confirmation_id=confirmation_id, chosen_path=entry.options[idx - 1]
                )
        except ValueError:
            pass
        # By name
        for p in entry.options:
            if line == p.value:
                return ConfirmationReply(confirmation_id=confirmation_id, chosen_path=p)
        # Unrecognized → treat as cancel (safer than silently accepting recommendation)
        self.console.print(f"[red]Unrecognized choice {raw!r} — treating as cancel.[/red]")
        return ConfirmationReply(confirmation_id=confirmation_id, chosen_path=None)


def make_cli_channel_caps(console: Any, max_latency_seconds: int = 120) -> ChannelCapabilities:
    """Build a ChannelCapabilities that uses *console* + stdin to confirm.

    The channel reports support for interactive confirmation; AgentLoop will
    suspend the current turn and call us during the GATE state.
    """
    state = _CLIConfirmationState(console=console)
    return ChannelCapabilities(
        supports_interactive_confirm=True,
        confirmation_max_latency_seconds=max_latency_seconds,
        send_confirmation=state.send,
        await_confirmation_reply=state.wait,
    )


# --- helpers ----------------------------------------------------------------


def _safe_input(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        return ""


def _risk_color(name: str) -> str:
    return {
        "catastrophic": "bold red",
        "high": "red",
        "medium": "yellow",
        "low": "green",
    }.get(name, "white")


def _path_label(p: ExecutionPath) -> str:
    return {
        ExecutionPath.BLOCKED: "do not send to cloud LLM",
        ExecutionPath.SIMPLE: "answer locally (no cloud call; requires local model — not yet wired)",
        ExecutionPath.METRIC_DP: "send with metric-DP noise (M3, ε-dχ-privacy)",
        ExecutionPath.K_DECOY: "send k decoys + truth (M2, not yet wired)",
        ExecutionPath.NORMAL: "send as-is to cloud LLM",
    }.get(p, "")
