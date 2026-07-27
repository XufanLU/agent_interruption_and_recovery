from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar


T = TypeVar("T")

RecoveryDecision = str
InterruptionHandler = Callable[["InjectedInterruption"], None]
RecoveryDecisionHandler = Callable[[RecoveryDecision], None]


class InjectedInterruption(RuntimeError):
    """Raised when an experiment injects an interruption at a recovery point."""


def validate_recovery_behavior(behavior: str) -> None:
    if behavior not in {"baseline_restart_from_zero", "checkpoint_resume"}:
        raise ValueError(f"Unsupported behavior: {behavior}")


def recover_after_interruption(
    *,
    behavior: str,
    run_fresh: Callable[[bool], T],
    resume_from_checkpoint: Callable[[], T],
    clear_checkpoint: Callable[[], None],
    on_interruption: InterruptionHandler | None = None,
    on_recovery_decision: RecoveryDecisionHandler | None = None,
) -> T:
    """Run a workflow with restart-vs-resume recovery semantics.

    The workflow supplies domain-specific execution callbacks. This middleware
    owns the cross-cutting interruption policy shared by framework examples.
    """

    validate_recovery_behavior(behavior)
    try:
        return run_fresh(True)
    except InjectedInterruption as exc:
        if on_interruption is not None:
            on_interruption(exc)

        if behavior == "baseline_restart_from_zero":
            decision = "discard_checkpoint_and_restart"
            if on_recovery_decision is not None:
                on_recovery_decision(decision)
            clear_checkpoint()
            return run_fresh(False)

        decision = "load_checkpoint_and_resume"
        if on_recovery_decision is not None:
            on_recovery_decision(decision)
        return resume_from_checkpoint()


async def recover_after_interruption_async(
    *,
    behavior: str,
    run_fresh: Callable[[bool], Awaitable[T]],
    resume_from_checkpoint: Callable[[], Awaitable[T]],
    clear_checkpoint: Callable[[], None],
    on_interruption: InterruptionHandler | None = None,
    on_recovery_decision: RecoveryDecisionHandler | None = None,
) -> T:
    """Async variant of recover_after_interruption for real agent runners."""

    validate_recovery_behavior(behavior)
    try:
        return await run_fresh(True)
    except InjectedInterruption as exc:
        if on_interruption is not None:
            on_interruption(exc)

        if behavior == "baseline_restart_from_zero":
            decision = "discard_checkpoint_and_restart"
            if on_recovery_decision is not None:
                on_recovery_decision(decision)
            clear_checkpoint()
            return await run_fresh(False)

        decision = "load_checkpoint_and_resume"
        if on_recovery_decision is not None:
            on_recovery_decision(decision)
        return await resume_from_checkpoint()
