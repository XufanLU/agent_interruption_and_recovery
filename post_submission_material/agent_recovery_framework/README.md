# Agent Recovery Framework

Shared middleware for interruption/recovery experiments.

`recovery_middleware.py` contains the reusable restart-vs-resume policy:

- run once with interruption enabled
- on interruption, either clear checkpoint state and restart from scratch
- or resume from the existing checkpoint

Workflows provide domain-specific callbacks for fresh execution, checkpoint
resume, checkpoint clearing, and optional logging. This keeps recovery policy
outside individual example flows.
