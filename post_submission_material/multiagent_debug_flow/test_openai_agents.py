from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest


def test_openai_agents_debug_flow(tmp_path: Path):
    if os.getenv("RUN_REAL_DEBUG_FLOW_AGENT_TEST") != "1":
        pytest.skip("Set RUN_REAL_DEBUG_FLOW_AGENT_TEST=1 to spend tokens on the OpenAI debug-flow agent test.")
    if importlib.util.find_spec("agents") is None:
        pytest.skip("OpenAI Agents SDK is not installed. Install ab_test/requirements.txt first.")

    from multiagent_debug_flow.workflow import run_ab_experiment
    from multiagent_debug_flow.results import append_result, display_log_path, resolve_results_log_path
    from multiagent_debug_flow.process_logging import resolve_process_log_path

    process_log_path = resolve_process_log_path(os.getenv("DEBUG_FLOW_PROCESS_LOG_PATH"))
    experiment_id = f"pytest_openai_debug_flow_{os.getenv('DEBUG_FLOW_TASK_ID', 'normalize_scores')}"
    summary = run_ab_experiment(
        task_id=os.getenv("DEBUG_FLOW_TASK_ID", "normalize_scores"),
        root=tmp_path,
        process_log_path=process_log_path,
        experiment_id=experiment_id,
    )
    log_path = resolve_results_log_path(os.getenv("DEBUG_FLOW_AB_LOG_PATH"))
    summary["result_log_path"] = display_log_path(log_path)
    append_result(summary, log_path)
    baseline = summary["arms"]["A"]
    checkpointed = summary["arms"]["B"]

    assert baseline["approved"]
    assert checkpointed["approved"]
    assert checkpointed["metrics"]["agent_runs"] < baseline["metrics"]["agent_runs"]
    print("\nLIVE_DEBUG_FLOW_AB_METRICS=" + __import__("json").dumps(summary, indent=2, sort_keys=True))
