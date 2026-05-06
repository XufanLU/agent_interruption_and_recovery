from __future__ import annotations

from typing import Any


def _empty_token_usage() -> dict[str, int]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "requests": 0,
    }


def _coerce_int(value: Any) -> int:
    try:
        if value is None:
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def _read_usage_value(data: Any, *names: str) -> int:
    if data is None:
        return 0
    for name in names:
        if isinstance(data, dict) and name in data:
            return _coerce_int(data.get(name))
        if hasattr(data, name):
            return _coerce_int(getattr(data, name))
    return 0


def normalize_token_usage(usage: Any) -> dict[str, int]:
    if usage is None:
        return _empty_token_usage()

    input_tokens = _read_usage_value(usage, "input_tokens", "prompt_tokens")
    output_tokens = _read_usage_value(usage, "output_tokens", "completion_tokens")
    total_tokens = _read_usage_value(usage, "total_tokens")
    requests = _read_usage_value(usage, "requests", "request_count")
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens
    if requests == 0 and total_tokens > 0:
        requests = 1

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "requests": requests,
    }


def extract_model_token_usage(result: Any) -> dict[str, int]:
    for candidate in (
        getattr(result, "usage", None),
        getattr(getattr(result, "context_wrapper", None), "usage", None),
    ):
        normalized = normalize_token_usage(candidate)
        if normalized["total_tokens"] > 0:
            return normalized

    raw_responses = getattr(result, "raw_responses", None) or getattr(result, "responses", None) or []
    total = _empty_token_usage()
    for response in raw_responses:
        usage = normalize_token_usage(getattr(response, "usage", None))
        for key in total:
            total[key] += usage.get(key, 0)
    return total
