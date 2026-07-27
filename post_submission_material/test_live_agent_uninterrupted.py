from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from fixture_pairs import get_fixture_pair, load_fixture_pairs, resolve_fixture_path
from test_live_agent_ab import (
    FIXTURES_DIR,
    _json_safe,
    _run_first_shell_agent_once,
    _serialize_fit_result,
)


AB_TEST_DIR = Path(__file__).resolve().parent
DEFAULT_PAIR_ID = "cuo_mp-14549_nims-cu-k"
DEFAULT_RESULTS_LOG_PATH = AB_TEST_DIR / "live_uninterrupted_results.jsonl"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _live_pair_ids() -> list[str]:
    if os.getenv("LIVE_UNINTERRUPTED_ALL_PAIRS") == "1":
        return [str(pair["id"]) for pair in load_fixture_pairs()]
    return [os.getenv("LIVE_UNINTERRUPTED_PAIR_ID", DEFAULT_PAIR_ID)]


def _require_live_env(pair_id: str) -> tuple[dict[str, Any], str, str]:
    if os.getenv("RUN_REAL_UNINTERRUPTED_TEST") != "1":
        pytest.skip(
            "Set RUN_REAL_UNINTERRUPTED_TEST=1 to run the live uninterrupted test."
        )

    pair = get_fixture_pair(pair_id)
    if not pair:
        pytest.skip("LIVE_UNINTERRUPTED_PAIR_ID does not match a fixture pair.")

    material_id = str(pair["material_id"]).strip()
    xas_path = str(resolve_fixture_path(str(pair["xas_path"]))).strip()
    if not Path(xas_path).exists():
        pytest.skip(f"Fixture XAS path does not exist: {xas_path}")

    return pair, material_id, xas_path


def _append_jsonl(summary: dict[str, Any]) -> None:
    configured_path = os.getenv("LIVE_UNINTERRUPTED_LOG_PATH")
    destination = (
        Path(configured_path).expanduser()
        if configured_path
        else DEFAULT_RESULTS_LOG_PATH
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(summary, sort_keys=True) + "\n")


@pytest.mark.parametrize("pair_id", _live_pair_ids())
def test_live_first_shell_agent_uninterrupted(
    pair_id: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Run one clean, uninterrupted first-shell agent flow and log its fit."""

    pair, material_id, xas_path = _require_live_env(pair_id)

    from agents.tracing.processors import default_processor
    from function_calling import artifacts as artifacts_module
    from function_calling import feff as feff_module
    from function_calling import fit as fit_module
    from function_calling import tools as tools_module

    checkpoint_dir = tmp_path / "uninterrupted-checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        fit_module,
        "checkpoints_dir",
        lambda *args, **kwargs: checkpoint_dir,
    )
    monkeypatch.setattr(
        feff_module,
        "online_cif_data_dir",
        lambda *args, **kwargs: FIXTURES_DIR,
    )

    original_execute_first_shell_fit = fit_module.execute_first_shell_fit
    original_viz_first_shell = tools_module.viz_first_shell
    fit_results: list[dict[str, Any]] = []
    visualization_completions = {"count": 0}

    def execute_first_shell_fit_with_capture(*args, **kwargs):
        result = original_execute_first_shell_fit(*args, **kwargs)
        cache_data = args[0] if args else kwargs.get("cache_data") or {}
        fit_results.append(_serialize_fit_result(result[0], cache_data, fit_module))
        return result

    def viz_first_shell_with_counts(*args, **kwargs):
        result = original_viz_first_shell(*args, **kwargs)
        visualization_completions["count"] += 1
        return result

    monkeypatch.setattr(
        fit_module,
        "execute_first_shell_fit",
        execute_first_shell_fit_with_capture,
    )
    monkeypatch.setattr(
        tools_module,
        "viz_first_shell",
        viz_first_shell_with_counts,
    )

    request_records: list[dict[str, Any]] = []
    trace_records: list[dict[str, Any]] = []
    run_id = f"live-uninterrupted-run-{uuid.uuid4()}"
    trace_group_id = f"live-uninterrupted-group-{uuid.uuid4()}"
    started_at = _utc_now()
    started = time.perf_counter()
    result = asyncio.run(
        _run_first_shell_agent_once(
            material_id,
            xas_path,
            pair_id=pair_id,
            arm="U",
            phase="uninterrupted",
            trace_group_id=trace_group_id,
            trace_records=trace_records,
            request_records=request_records,
        )
    )
    wall_time_s = time.perf_counter() - started
    default_processor().force_flush()

    fit_status = (
        "completed"
        if len(fit_results) == 1 and visualization_completions["count"] == 1
        else "failed"
    )
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "pair": {
            "id": str(pair["id"]),
            "formula": str(pair.get("formula") or ""),
            "material_id": material_id,
            "cif_file": str(pair.get("cif_file") or ""),
            "xas_file": str(pair.get("xas_file") or ""),
        },
        "started_at": started_at,
        "completed_at": _utc_now(),
        "fit_status": fit_status,
        "wall_time_s": wall_time_s,
        "fit_attempts": len(fit_results),
        "visualization_completions": visualization_completions["count"],
        "parameters": fit_results[0] if fit_results else None,
        "assistant_output": _json_safe(getattr(result, "final_output", None)),
    }
    _append_jsonl(summary)

    assert fit_status == "completed"
    assert summary["parameters"] is not None

