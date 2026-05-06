from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pytest

from fixture_pairs import get_fixture_pair, load_fixture_pairs, resolve_fixture_path


AB_TEST_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = AB_TEST_DIR / "fixtures"
DEFAULT_PAIR_ID = "cuo_mp-14549_nims-cu-k"
DEFAULT_RESULTS_LOG_PATH = AB_TEST_DIR / "live_ab_results.jsonl"
STRUCTURED_LOGGER = logging.getLogger("ab_test.live_agent")


@dataclass
class LiveAgentABMetrics:
    wall_time_s: float
    token_usage: dict[str, int]
    tool_calls: dict[str, Any]
    data_loads: dict[str, int]


class InjectedLiveInterruption(BaseException):
    """Abort the Agents SDK run instead of returning a recoverable tool error."""


def _live_pair_ids() -> list[str]:
    if os.getenv("LIVE_AB_ALL_PAIRS") == "1":
        return [str(pair["id"]) for pair in load_fixture_pairs()]
    return [os.getenv("LIVE_AB_PAIR_ID", DEFAULT_PAIR_ID)]


def _require_live_ab_env(pair_id: str) -> tuple[dict[str, str], str, str]:
    if os.getenv("RUN_REAL_AGENT_AB_TEST") != "1":
        pytest.skip("Set RUN_REAL_AGENT_AB_TEST=1 to run the live first-shell agent A/B test.")

    pair = get_fixture_pair(pair_id)
    if not pair:
        pytest.skip("LIVE_AB_PAIR_ID does not match a fixture pair.")

    if os.getenv("LIVE_AB_ALL_PAIRS") == "1":
        material_id = str(pair["material_id"]).strip()
        xas_path = str(resolve_fixture_path(str(pair["xas_path"]))).strip()
    else:
        material_id = os.getenv("LIVE_AB_MATERIAL_ID", str(pair["material_id"])).strip()
        xas_path = os.getenv("LIVE_AB_XAS_PATH", str(resolve_fixture_path(str(pair["xas_path"])))).strip()

    resolved_xas = Path(xas_path).expanduser()
    if not resolved_xas.exists():
        pytest.skip(f"LIVE_AB_XAS_PATH does not exist: {resolved_xas}")

    return pair, material_id, str(resolved_xas)


def _clear_fit_cache(checkpoint_dir: Path) -> None:
    for cache_file in checkpoint_dir.glob("fit_cache_*.json"):
        cache_file.unlink(missing_ok=True)


def _configure_structured_logger() -> None:
    if STRUCTURED_LOGGER.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    STRUCTURED_LOGGER.addHandler(handler)
    STRUCTURED_LOGGER.setLevel(logging.INFO)
    STRUCTURED_LOGGER.propagate = False


def _read_value(data: Any, *names: str) -> Any:
    for name in names:
        if isinstance(data, dict) and name in data:
            return data.get(name)
        if hasattr(data, name):
            return getattr(data, name)
    return None


def _iter_agent_items(result: Any, collection_names: tuple[str, ...]):
    for collection_name in collection_names:
        collection = _read_value(result, collection_name) or []
        for item in collection:
            yield item
            raw_item = _read_value(item, "raw_item", "item")
            if raw_item is not None and raw_item is not item:
                yield raw_item
            for nested_name in ("output", "content"):
                for nested_item in _read_value(item, nested_name) or []:
                    yield nested_item


def _summarize_tool_calls(items) -> dict[str, Any]:
    by_name: dict[str, int] = {}
    seen: set[int] = set()
    total = 0

    for item in items:
        item_id = id(item)
        if item_id in seen:
            continue
        seen.add(item_id)

        item_type = str(_read_value(item, "type", "item_type") or "").lower()
        class_name = item.__class__.__name__.lower()
        is_tool_output = (
            "output" in item_type
            or "result" in item_type
            or "output" in class_name
            or "result" in class_name
        )
        if is_tool_output:
            continue

        is_tool_call = (
            "tool_call" in item_type
            or item_type == "function_call"
            or "toolcall" in class_name
            or "tool_call" in class_name
        )
        if not is_tool_call:
            continue

        function_data = _read_value(item, "function") or {}
        tool_name = (
            _read_value(item, "name", "tool_name")
            or _read_value(function_data, "name")
        )
        if not tool_name:
            continue

        tool_name = str(tool_name)
        by_name[tool_name] = by_name.get(tool_name, 0) + 1
        total += 1

    return {
        "total": total,
        "by_name": by_name,
    }


def _extract_tool_call_summary(result: Any) -> dict[str, Any]:
    summary = _summarize_tool_calls(_iter_agent_items(result, ("new_items", "items")))
    if summary["total"] > 0:
        return summary
    return _summarize_tool_calls(_iter_agent_items(result, ("raw_responses", "responses")))


def _emit_structured_pair_log(summary: dict[str, Any]) -> None:
    _configure_structured_logger()
    line = json.dumps(summary, sort_keys=True)
    STRUCTURED_LOGGER.info("LIVE_AGENT_AB_PAIR_RESULT=%s", line)

    log_path = os.getenv("LIVE_AB_LOG_PATH")
    destination = Path(log_path).expanduser() if log_path else DEFAULT_RESULTS_LOG_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as log_file:
        log_file.write(line + "\n")


def _log_path_from_ab_test(path: str | Path) -> str:
    path_obj = Path(path).expanduser()
    try:
        return str(path_obj.resolve().relative_to(AB_TEST_DIR))
    except ValueError:
        return str(path)


def _pair_file_names(pair: dict[str, Any], xas_path: str) -> dict[str, str]:
    return {
        "cif_file": Path(str(pair.get("cif_path", ""))).name,
        "xas_file": Path(xas_path).name,
    }


def _pair_log_context(pair: dict[str, Any], material_id: str, xas_path: str) -> dict[str, Any]:
    context = {
        "id": pair["id"],
        "formula": pair["formula"],
        "material_id": material_id,
        "xas_path": _log_path_from_ab_test(xas_path),
    }
    context.update(_pair_file_names(pair, xas_path))
    return context


async def _run_first_shell_agent_once(material_id: str, xas_path: str):
    from agents import Runner
    from first_shell_agent import create_first_shell_agent

    agent = await create_first_shell_agent(material_id=material_id, xas_path=xas_path)
    prompt = os.getenv(
        "LIVE_AB_PROMPT",
        "Run a first-shell EXAFS fit using the current material and XAS spectrum. "
        "Use the provided runtime context and return a concise fitting summary.",
    )
    return await Runner.run(agent, prompt)


@pytest.mark.parametrize("pair_id", _live_pair_ids())
def test_live_first_shell_agent_ab_real_time_and_tokens(
    pair_id: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """
    Opt-in live A/B test using the real `create_agent_first_shell()` path.

    This spends real model/API tokens and real fitting time.
    """

    pair, material_id, xas_path = _require_live_ab_env(pair_id)

    from function_calling import feff as feff_module
    from function_calling import artifacts as artifacts_module
    from function_calling import fit as fit_module
    from usage_tracking import extract_model_token_usage

    checkpoint_dir = tmp_path / "live-checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(fit_module, "checkpoints_dir", lambda *args, **kwargs: checkpoint_dir)
    monkeypatch.setattr(feff_module, "online_cif_data_dir", lambda *args, **kwargs: FIXTURES_DIR)

    original_execute_first_shell_fit = fit_module.execute_first_shell_fit
    original_load_fit_group = artifacts_module.load_fit_group
    should_interrupt = {"armed": True}
    data_load_counts = {
        "raw_xas_imports": 0,
        "processed_payload_loads": 0,
    }

    def execute_first_shell_fit_with_one_interruption(*args, **kwargs):
        result = original_execute_first_shell_fit(*args, **kwargs)
        if should_interrupt["armed"]:
            should_interrupt["armed"] = False
            raise InjectedLiveInterruption("Injected live interruption after the first real fitting step")
        return result

    def load_fit_group_with_counts(*, xas_path=None, xas_ref=None):
        if xas_ref or (xas_path and artifacts_module.load_processed_payload(xas_path) is not None):
            data_load_counts["processed_payload_loads"] += 1
        else:
            data_load_counts["raw_xas_imports"] += 1
        return original_load_fit_group(xas_path=xas_path, xas_ref=xas_ref)

    monkeypatch.setattr(fit_module, "execute_first_shell_fit", execute_first_shell_fit_with_one_interruption)
    monkeypatch.setattr(artifacts_module, "load_fit_group", load_fit_group_with_counts)

    async def _run_with_one_interruption(*, keep_cache_after_interrupt: bool) -> LiveAgentABMetrics:
        should_interrupt["armed"] = True
        data_load_counts.update({"raw_xas_imports": 0, "processed_payload_loads": 0})
        started = time.perf_counter()
        try:
            await _run_first_shell_agent_once(material_id, xas_path)
            pytest.fail("Expected injected interruption in live first-shell agent run")
        except InjectedLiveInterruption:
            if not keep_cache_after_interrupt:
                _clear_fit_cache(checkpoint_dir)

        completed_result = await _run_first_shell_agent_once(material_id, xas_path)
        elapsed = time.perf_counter() - started
        return LiveAgentABMetrics(
            wall_time_s=elapsed,
            token_usage=extract_model_token_usage(completed_result),
            tool_calls=_extract_tool_call_summary(completed_result),
            data_loads=dict(data_load_counts),
        )

    _clear_fit_cache(checkpoint_dir)
    baseline = asyncio.run(_run_with_one_interruption(keep_cache_after_interrupt=False))

    _clear_fit_cache(checkpoint_dir)
    checkpointed = asyncio.run(_run_with_one_interruption(keep_cache_after_interrupt=True))

    baseline_summary = asdict(baseline)
    checkpointed_summary = asdict(checkpointed)
    summary = {
        "schema_version": 1,
        "result_log_path": _log_path_from_ab_test(os.getenv("LIVE_AB_LOG_PATH", DEFAULT_RESULTS_LOG_PATH)),
        "pair": _pair_log_context(pair, material_id, xas_path),
        "arms": {
            "A": {
                "behavior": "baseline_restart_from_zero",
                **baseline_summary,
            },
            "B": {
                "behavior": "checkpoint_resume",
                **checkpointed_summary,
            },
        },
        "B_minus_A_wall_time_s": checkpointed.wall_time_s - baseline.wall_time_s,
        "B_minus_A_total_tokens": checkpointed.token_usage["total_tokens"] - baseline.token_usage["total_tokens"],
        "B_minus_A_raw_xas_imports": (
            checkpointed.data_loads["raw_xas_imports"] - baseline.data_loads["raw_xas_imports"]
        ),
    }
    _emit_structured_pair_log(summary)
    print("\nLIVE_AGENT_AB_METRICS=" + json.dumps(summary, indent=2, sort_keys=True))

    assert baseline.wall_time_s > 0
    assert checkpointed.wall_time_s > 0
