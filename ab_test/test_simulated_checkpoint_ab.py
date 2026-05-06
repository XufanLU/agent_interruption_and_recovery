from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest


@dataclass
class ABMetrics:
    wall_time_s: float
    preprocessing_calls: int
    raw_import_calls: int
    processed_load_calls: int
    fit_calls: int
    token_usage: dict[str, int]


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def sleep(self, seconds: float) -> None:
        self.t += float(seconds)


def test_simulated_ab_stress_restart_vs_checkpoint_resume(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """
    Simulated A/B stress test for n=1.

    - Group A: interruption -> delete cache -> restart from Step 0.
    - Group B: interruption -> keep cache -> resume from checkpoint.
    """

    from function_calling import fit as fit_module
    from usage_tracking import normalize_token_usage

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(fit_module, "checkpoints_dir", lambda *args, **kwargs: checkpoint_dir)

    clock = FakeClock()
    call_counts = {"pre": 0, "raw_import": 0, "processed_load": 0, "fit": 0}
    should_interrupt = {"armed": True}

    async def preprocessing_fn(material_id: str):
        call_counts["pre"] += 1
        clock.sleep(2.0)
        return {"path1": "dummy_feff_path.dat"}

    def load_fit_group_fn(*, xas_path=None, xas_ref=None):
        if xas_ref:
            call_counts["processed_load"] += 1
            clock.sleep(0.1)
            return object(), f"{xas_ref}.json"
        call_counts["raw_import"] += 1
        clock.sleep(1.0)
        return object(), xas_path

    class DummyResult:
        chi2_reduced = 1.0
        rfactor = 0.01

    def execute_first_shell_fit_interruptible(cache_data, *, data, load_fit_group_fn):
        call_counts["fit"] += 1
        if data is None:
            data, _ = load_fit_group_fn(
                xas_path=cache_data.get("xas_path"),
                xas_ref=cache_data.get("fit_xas_ref"),
            )
        for _ in range(20):
            clock.sleep(1.0)
            if should_interrupt["armed"] and clock.t >= 12.0:
                should_interrupt["armed"] = False
                raise TimeoutError("Injected interruption at t=12s during Step 2 fitting")
        return DummyResult(), {"path1": "ok"}, data

    monkeypatch.setattr(fit_module, "execute_first_shell_fit", execute_first_shell_fit_interruptible)
    monkeypatch.setattr(
        fit_module,
        "_persist_fit_group_for_resume",
        lambda data, *, fit_target, source_file: "dummy_xas",
    )

    async def check_if_paused_fn():
        return None

    async def save_checkpoint_from_tool_fn(*args, **kwargs):
        return None

    def extract_fitted_parameters_fn(_result):
        class _FittedParameters:
            def model_dump(self):
                return {}

        return _FittedParameters()

    def extract_path_parameters_fn(_result):
        class _PathParameters:
            def model_dump(self):
                return {}

        return [_PathParameters()]

    def viz_first_shell_fn(*args, **kwargs):
        return None

    def build_fit_report_from_cache_fn(*, cache_data: dict, fitted_parameters=None, path_parameters=None):
        return {"ok": True, "cache": cache_data}

    params = SimpleNamespace(amp=0.8, e0=0.0, sigma2=0.003, deltar=0.0)

    async def _run_once() -> None:
        await fit_module.orchestrate_first_shell_fit_with_checkpoints(
            params=params,
            material_id="mp-ab",
            xas_path="dummy_xas.dat",
            preprocessing_fn=preprocessing_fn,
            check_if_paused_fn=check_if_paused_fn,
            save_checkpoint_from_tool_fn=save_checkpoint_from_tool_fn,
            calibrate_xas_with_foil_data_fn=lambda **kwargs: None,
            load_fit_group_fn=load_fit_group_fn,
            extract_fitted_parameters_fn=extract_fitted_parameters_fn,
            extract_path_parameters_fn=extract_path_parameters_fn,
            viz_first_shell_fn=viz_first_shell_fn,
            build_fit_report_from_cache_fn=build_fit_report_from_cache_fn,
        )

    def _token_usage_from_counts() -> dict[str, int]:
        input_tokens = (
            call_counts["pre"] * 400
            + call_counts["raw_import"] * 250
            + call_counts["processed_load"] * 25
            + call_counts["fit"] * 100
        )
        output_tokens = call_counts["fit"] * 50
        return normalize_token_usage(
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "requests": call_counts["fit"],
            }
        )

    clock.t = 0.0
    call_counts.update({"pre": 0, "raw_import": 0, "processed_load": 0, "fit": 0})
    should_interrupt["armed"] = True
    try:
        asyncio.run(_run_once())
        pytest.fail("Expected injected interruption in baseline run")
    except TimeoutError:
        for cache_file in checkpoint_dir.glob("fit_cache_*.json"):
            cache_file.unlink(missing_ok=True)
    asyncio.run(_run_once())
    metrics_a = ABMetrics(
        wall_time_s=clock.t,
        preprocessing_calls=call_counts["pre"],
        raw_import_calls=call_counts["raw_import"],
        processed_load_calls=call_counts["processed_load"],
        fit_calls=call_counts["fit"],
        token_usage=_token_usage_from_counts(),
    )

    clock.t = 0.0
    call_counts.update({"pre": 0, "raw_import": 0, "processed_load": 0, "fit": 0})
    should_interrupt["armed"] = True
    for cache_file in checkpoint_dir.glob("fit_cache_*.json"):
        cache_file.unlink(missing_ok=True)
    try:
        asyncio.run(_run_once())
        pytest.fail("Expected injected interruption in checkpoint run")
    except TimeoutError:
        pass
    asyncio.run(_run_once())
    metrics_b = ABMetrics(
        wall_time_s=clock.t,
        preprocessing_calls=call_counts["pre"],
        raw_import_calls=call_counts["raw_import"],
        processed_load_calls=call_counts["processed_load"],
        fit_calls=call_counts["fit"],
        token_usage=_token_usage_from_counts(),
    )

    assert metrics_b.wall_time_s < metrics_a.wall_time_s
    assert metrics_b.preprocessing_calls <= metrics_a.preprocessing_calls
    assert metrics_b.raw_import_calls < metrics_a.raw_import_calls
    assert metrics_b.processed_load_calls > metrics_a.processed_load_calls
    assert metrics_b.token_usage["total_tokens"] < metrics_a.token_usage["total_tokens"]
