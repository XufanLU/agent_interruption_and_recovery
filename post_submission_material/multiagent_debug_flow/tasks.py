from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DebugTask:
    task_id: str
    bug_report: str
    module_path: str
    test_path: str
    buggy_source: str
    patch_v1_source: str
    patch_v2_source: str
    test_source: str
    reviewer_feedback: tuple[str, ...]


NORMALIZE_SCORES = DebugTask(
    task_id="normalize_scores",
    bug_report=(
        "Fix normalize_scores so it returns normalized proportions for positive "
        "inputs, returns [] for an empty list, returns zeros for all-zero input, "
        "and rejects negative values."
    ),
    module_path="src/scoring.py",
    test_path="tests/test_scoring.py",
    buggy_source='''"""Small scoring helpers used by the debug workflow."""


def normalize_scores(scores):
    total = sum(scores)
    return [score / total for score in scores]
''',
    patch_v1_source='''"""Small scoring helpers used by the debug workflow."""


def normalize_scores(scores):
    if not scores:
        return []
    total = sum(scores)
    return [score / total for score in scores]
''',
    patch_v2_source='''"""Small scoring helpers used by the debug workflow."""


def normalize_scores(scores):
    if any(score < 0 for score in scores):
        raise ValueError("scores must be non-negative")
    if not scores:
        return []
    total = sum(scores)
    if total == 0:
        return [0 for _ in scores]
    return [score / total for score in scores]
''',
    test_source='''import pytest

from src.scoring import normalize_scores


def test_normalize_scores_regular_case():
    assert normalize_scores([2, 2, 6]) == [0.2, 0.2, 0.6]


def test_normalize_scores_empty_list():
    assert normalize_scores([]) == []


def test_normalize_scores_all_zero():
    assert normalize_scores([0, 0]) == [0, 0]


def test_normalize_scores_rejects_negative():
    with pytest.raises(ValueError):
        normalize_scores([1, -1, 2])
''',
    reviewer_feedback=(
        "Patch v1 handles the empty-list case but still divides by zero for all-zero input.",
        "Patch v1 does not reject negative scores with ValueError.",
    ),
)


PARSE_DURATION = DebugTask(
    task_id="parse_duration",
    bug_report=(
        "Fix parse_duration so it accepts positive values ending in s, m, or h, "
        "converts them to seconds, strips whitespace, and rejects malformed or "
        "negative durations."
    ),
    module_path="src/duration.py",
    test_path="tests/test_duration.py",
    buggy_source='''"""Duration parsing helpers used by the debug workflow."""


def parse_duration(text):
    value = int(text[:-1])
    unit = text[-1]
    if unit == "s":
        return value
    if unit == "m":
        return value * 60
    if unit == "h":
        return value * 3600
    return value
''',
    patch_v1_source='''"""Duration parsing helpers used by the debug workflow."""


def parse_duration(text):
    text = text.strip()
    value = int(text[:-1])
    unit = text[-1]
    if unit == "s":
        return value
    if unit == "m":
        return value * 60
    if unit == "h":
        return value * 3600
    raise ValueError("unsupported duration unit")
''',
    patch_v2_source='''"""Duration parsing helpers used by the debug workflow."""


def parse_duration(text):
    text = text.strip()
    if len(text) < 2:
        raise ValueError("duration must include a value and unit")
    value_text = text[:-1]
    unit = text[-1]
    if not value_text.isdigit():
        raise ValueError("duration value must be non-negative")
    value = int(value_text)
    if unit == "s":
        return value
    if unit == "m":
        return value * 60
    if unit == "h":
        return value * 3600
    raise ValueError("unsupported duration unit")
''',
    test_source='''import pytest

from src.duration import parse_duration


def test_parse_duration_seconds():
    assert parse_duration("15s") == 15


def test_parse_duration_minutes_with_whitespace():
    assert parse_duration(" 2m ") == 120


def test_parse_duration_hours():
    assert parse_duration("1h") == 3600


def test_parse_duration_rejects_negative():
    with pytest.raises(ValueError):
        parse_duration("-1m")


def test_parse_duration_rejects_unknown_unit():
    with pytest.raises(ValueError):
        parse_duration("4d")
''',
    reviewer_feedback=(
        "Patch v1 strips whitespace and rejects unknown units, but negative values still pass through.",
        "Patch v1 relies on int conversion errors instead of a clear ValueError contract for malformed values.",
    ),
)


TASKS = {
    task.task_id: task
    for task in (
        NORMALIZE_SCORES,
        PARSE_DURATION,
    )
}


def get_task(task_id: str) -> DebugTask:
    try:
        return TASKS[task_id]
    except KeyError as exc:
        available = ", ".join(sorted(TASKS))
        raise ValueError(f"Unknown debug task {task_id!r}. Available tasks: {available}") from exc


def task_metadata(task_id: str) -> dict:
    task = get_task(task_id)
    return {
        "source": "local_synthetic",
        "task_id": task.task_id,
        "benchmark": None,
        "swe_bench_instance_id": None,
        "module_path": task.module_path,
        "test_path": task.test_path,
        "description": task.bug_report,
    }
