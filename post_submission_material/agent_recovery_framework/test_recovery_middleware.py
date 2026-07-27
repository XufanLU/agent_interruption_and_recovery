from __future__ import annotations

import pytest

from agent_recovery_framework.recovery_middleware import InjectedInterruption, recover_after_interruption


def test_recovery_middleware_restarts_baseline_after_interruption():
    events: list[str] = []

    def run_fresh(allow_interrupt: bool) -> str:
        events.append(f"fresh:{allow_interrupt}")
        if allow_interrupt:
            raise InjectedInterruption("stop")
        return "restarted"

    result = recover_after_interruption(
        behavior="baseline_restart_from_zero",
        run_fresh=run_fresh,
        resume_from_checkpoint=lambda: "resumed",
        clear_checkpoint=lambda: events.append("clear"),
        on_recovery_decision=events.append,
    )

    assert result == "restarted"
    assert events == [
        "fresh:True",
        "discard_checkpoint_and_restart",
        "clear",
        "fresh:False",
    ]


def test_recovery_middleware_resumes_checkpoint_after_interruption():
    events: list[str] = []

    def run_fresh(allow_interrupt: bool) -> str:
        events.append(f"fresh:{allow_interrupt}")
        raise InjectedInterruption("stop")

    result = recover_after_interruption(
        behavior="checkpoint_resume",
        run_fresh=run_fresh,
        resume_from_checkpoint=lambda: events.append("resume") or "resumed",
        clear_checkpoint=lambda: events.append("clear"),
        on_recovery_decision=events.append,
    )

    assert result == "resumed"
    assert events == ["fresh:True", "load_checkpoint_and_resume", "resume"]


def test_recovery_middleware_rejects_unknown_behavior():
    with pytest.raises(ValueError, match="Unsupported behavior"):
        recover_after_interruption(
            behavior="unknown",
            run_fresh=lambda _: "fresh",
            resume_from_checkpoint=lambda: "resumed",
            clear_checkpoint=lambda: None,
        )
