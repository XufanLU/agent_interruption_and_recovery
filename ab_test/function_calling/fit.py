from __future__ import annotations

"""EXAFS fitting and foil-calibration workflow helpers.

This module contains the reusable fitting-side logic: calibration against foil
references, fit-cache/report construction, parameter extraction, and the lower-
level data preparation steps used by the higher-level tool wrappers.
"""

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import larch
import numpy as np
from larch.fitting import param, param_group
from larch.xafs import autobk, pre_edge, xftf, xftr
from larch.xafs import feffit, feffit_dataset, feffit_transform, feffpath
from data_paths import checkpoints_dir
from .artifacts import artifact_key, group_to_payload, persist_xas_payload, safe_float

from .artifacts import load_processed_payload, persist_preprocessed_payload, resolve_processed_payload
from .edge_identifier import EdgeIdentifier
from .lookups import prepare_feff_paths_data
from .models import EnergyCalibrationResult, FittedParameter, PathParameter, Report
from .xas_io import load_embedded_foil_reference


def build_fit_report_from_cache(
    *,
    cache_data: dict,
    fitted_parameters,
    path_parameters,
):
    fit_xas_ref = cache_data.get("fit_xas_ref")
    fit_source_file = None
    fit_xas_url = None
    if fit_xas_ref:
        fit_xas_url = f"processed_xas/{artifact_key(fit_xas_ref)}.json"
        payload = load_processed_payload(fit_xas_ref)
        if isinstance(payload, dict):
            fit_source_file = payload.get("source_file")

    return Report(
        fitted_parameter=fitted_parameters,
        path_parameter=path_parameters,
        xas_ref=fit_xas_ref,
        source_file=fit_source_file,
        xas_url=fit_xas_url,
    )

def lookup_reference_edge_energy(
    element: str | None,
    edge: str | None,
) -> float | None:
    if not element:
        return None

    identifier = EdgeIdentifier()
    edge_key = str(edge or "K").strip().upper()
    if edge_key == "L":
        edge_key = "L3"

    table = {
        "K": getattr(identifier, "k_edges", {}),
        "L1": getattr(identifier, "l1_edges", {}),
        "L2": getattr(identifier, "l2_edges", {}),
        "L3": getattr(identifier, "l3_edges", {}),
    }.get(edge_key, {})
    return safe_float(table.get(element))


def detect_edge_energy_and_reference(
    energy: np.ndarray,
    mu: np.ndarray,
    *,
    accepted_edge_energy: float | None = None,
    foil_element: str | None = None,
    foil_edge: str | None = None,
) -> tuple[float, float, str | None, str | None]:
    identifier = EdgeIdentifier()
    measured_edge_energy = safe_float(identifier.find_edge_energy(energy.tolist(), mu.tolist()))
    if measured_edge_energy is None:
        raise ValueError("Could not determine the foil edge energy from the derivative peak.")

    resolved_element = foil_element
    resolved_edge = str(foil_edge or "K").strip().upper()
    if resolved_edge == "L":
        resolved_edge = "L3"

    resolved_reference = safe_float(accepted_edge_energy)
    if resolved_reference is None and resolved_element:
        resolved_reference = lookup_reference_edge_energy(resolved_element, resolved_edge)

    if resolved_reference is None:
        matches = identifier.identify_element(measured_edge_energy)
        if matches:
            best_match = matches[0]
            resolved_element = resolved_element or best_match.get("Element")
            resolved_edge = best_match.get("Edge") or resolved_edge
            resolved_reference = safe_float(best_match.get("Energy"))

    if resolved_reference is None:
        raise ValueError(
            "Could not determine the accepted foil edge energy. "
            "Provide accepted_edge_energy or specify foil_element/foil_edge."
        )

    return measured_edge_energy, resolved_reference, resolved_element, resolved_edge


def calibrate_xas_with_foil_data(
    *,
    xas_path: str | None = None,
    xas_ref: str | None = None,
    foil_path: str | None = None,
    foil_xas_ref: str | None = None,
    accepted_edge_energy: float | None = None,
    foil_element: str | None = None,
    foil_edge: str = "K",
    allow_self_calibration: bool = False,
    spectrum_is_foil: bool = False,
):
    sample_key, sample_payload = resolve_processed_payload(xas_path=xas_path, xas_ref=xas_ref)

    sample_energy = np.asarray(sample_payload.get("energy", []), dtype=float)
    sample_mu = np.asarray(sample_payload.get("mu", []), dtype=float)

    if sample_energy.size == 0 or sample_mu.size == 0:
        raise ValueError("Sample spectrum does not contain usable energy/mu data.")

    foil_key: str | None = None
    foil_payload: dict | None = None
    foil_source_file: str | None = None
    calibration_source = "separate"

    if foil_path or foil_xas_ref:
        foil_key, foil_payload = resolve_processed_payload(xas_path=foil_path, xas_ref=foil_xas_ref)
        foil_source_file = foil_payload.get("source_file") if foil_payload else None
    else:
        embedded_target = sample_payload.get("source_file") or xas_path or xas_ref
        if embedded_target:
            embedded_group, embedded_filename = load_embedded_foil_reference(str(embedded_target))
            if embedded_group is not None:
                foil_key = f"{sample_key}_embedded_foil"
                foil_payload = group_to_payload(embedded_group, source_file=str(embedded_filename))
                if not isinstance(foil_payload, dict) or not foil_payload.get("energy") or not foil_payload.get("mu"):
                    foil_payload = {
                        "energy": [float(value) for value in np.asarray(embedded_group.energy, dtype=float).tolist()],
                        "mu": [float(value) for value in np.asarray(embedded_group.mu, dtype=float).tolist()],
                        "source_file": str(embedded_filename),
                    }
                foil_source_file = str(embedded_filename)
                calibration_source = "embedded"

    if foil_payload is None and allow_self_calibration and spectrum_is_foil:
        foil_key = sample_key
        foil_payload = sample_payload
        foil_source_file = sample_payload.get("source_file")
        calibration_source = "self"

    if foil_payload is None:
        raise ValueError(
            "Foil calibration requested, but no foil data was found. "
            "Provide foil_path/foil_xas_ref or upload a scan that includes It and Iref channels."
        )

    foil_energy = np.asarray(foil_payload.get("energy", []), dtype=float)
    foil_mu = np.asarray(foil_payload.get("mu", []), dtype=float)
    if foil_energy.size == 0 or foil_mu.size == 0:
        raise ValueError("Foil spectrum does not contain usable energy/mu data.")

    measured_edge_energy, resolved_reference, resolved_element, resolved_edge = detect_edge_energy_and_reference(
        foil_energy,
        foil_mu,
        accepted_edge_energy=accepted_edge_energy,
        foil_element=foil_element,
        foil_edge=foil_edge,
    )
    energy_offset = resolved_reference - measured_edge_energy

    calibrated = larch.Group(
        name=f"{sample_key}_calibrated",
        energy=sample_energy + energy_offset,
        mu=sample_mu,
    )
    pre_edge(calibrated)
    autobk(calibrated, rbkg=1.0, kweight=2)
    xftf(calibrated, kweight=2)
    xftr(calibrated)

    calibrated_ref = f"{sample_key}_calibrated"
    calibrated_payload = group_to_payload(calibrated, source_file=sample_payload.get("source_file"))
    if not isinstance(calibrated_payload, dict) or not calibrated_payload.get("energy") or not calibrated_payload.get("mu"):
        calibrated_payload = {
            "energy": [float(value) for value in np.asarray(calibrated.energy, dtype=float).tolist()],
            "mu": [float(value) for value in np.asarray(calibrated.mu, dtype=float).tolist()],
            "source_file": sample_payload.get("source_file"),
        }
    calibrated_payload["calibration"] = {
        "mode": "foil",
        "source": calibration_source,
        "foil_ref": foil_key,
        "foil_source_file": foil_source_file or foil_payload.get("source_file"),
        "foil_element": resolved_element,
        "foil_edge": resolved_edge,
        "measured_foil_edge_energy": measured_edge_energy,
        "accepted_foil_edge_energy": resolved_reference,
        "energy_offset_ev": energy_offset,
    }

    persisted = persist_preprocessed_payload(calibrated_ref, calibrated_payload)
    return EnergyCalibrationResult(
        success=True,
        xas_ref=persisted.xas_ref,
        source_file=persisted.source_file,
        foil_ref=foil_key,
        foil_source_file=foil_source_file or foil_payload.get("source_file"),
        foil_element=resolved_element,
        foil_edge=resolved_edge,
        measured_foil_edge_energy=measured_edge_energy,
        accepted_foil_edge_energy=resolved_reference,
        energy_offset_ev=energy_offset,
        e0=persisted.e0,
        n_points=persisted.n_points,
        energy_min=persisted.energy_min,
        energy_max=persisted.energy_max,
        processed_path=persisted.processed_path,
        s3_key=persisted.s3_key,
        error=None,
    )


async def preprocessing(
    cif_file: str,
) -> dict:
    prepared = await asyncio.to_thread(prepare_feff_paths_data, cif_file)
    if not prepared.success or not prepared.paths:
        raise ValueError(prepared.error or "Failed to prepare FEFF paths.")
    return {"path1": prepared.paths[0].path}


def _build_params_group(cache_data: dict):
    params_group = param_group(
        amp=param(cache_data["params_dict"]["amp"], vary=True),
        degen=param(1, vary=False),
        e0=param(cache_data["params_dict"]["e0"], vary=True),
        sigma2=param(cache_data["params_dict"]["sigma2"], vary=True),
        deltar=param(cache_data["params_dict"]["deltar"], vary=True),
    )
    params_group["degen"].vary = False
    params_group["degen"].value = 1
    return params_group


def _build_paths_dict(cache_data: dict):
    paths_dict = {}
    for path_key, path_str in cache_data["paths_list"]:
        path = feffpath(
            path_str,
            degen=1,
            s02="amp * degen",
            e0="e0",
            deltar="deltar",
            sigma2="sigma2",
        )
        paths_dict[path_key] = path
    return paths_dict


def execute_first_shell_fit(
    cache_data: dict,
    *,
    data,
    load_fit_group_fn,
):
    # Core numerical fit step: build one FEFF dataset and run the optimizer.
    params_group = _build_params_group(cache_data)
    paths_dict = _build_paths_dict(cache_data)
    transform = feffit_transform(
        kmin=3,
        kmax=10,
        kweight=2,
        dk=4,
        window="kaiser",
        rmin=1.0,
        rmax=3.0,
    )
    if data is None:
        data, _ = load_fit_group_fn(
            xas_path=cache_data.get("xas_path"),
            xas_ref=cache_data.get("fit_xas_ref"),
        )
    if not paths_dict:
        raise ValueError("No paths provided in paths_dict")

    first_path = next(iter(paths_dict.values()))
    dataset = feffit_dataset(data=data, pathlist=[first_path], transform=transform)
    result = feffit(params_group, [dataset])
    return result, paths_dict, data


def _persist_fit_group_for_resume(data, *, fit_target: str, source_file: Any) -> str:
    payload = group_to_payload(data, source_file=str(source_file or fit_target))
    bundle = persist_xas_payload(
        fit_target,
        payload,
        upload=False,
        remote_key="processed_xas/{xas_id}.json",
    )
    return bundle.xas_id


def _build_fit_cache_data(
    *,
    params,
    material_id: str,
    xas_path: str | None,
    xas_ref: str | None,
    calibrate_with_foil: bool,
    foil_path: str | None,
    foil_xas_ref: str | None,
    accepted_edge_energy: float | None,
    foil_element: str | None,
    foil_edge: str,
    paths: dict[str, str],
) -> dict:
    return {
        "params_dict": {
            "amp": params.amp,
            "e0": params.e0,
            "sigma2": params.sigma2,
            "deltar": params.deltar,
        },
        "paths_list": [(key, path_str) for key, path_str in paths.items()],
        "material_id": material_id,
        "xas_path": xas_path,
        "xas_ref": xas_ref,
        "fit_xas_ref": xas_ref,
        "calibrate_with_foil": calibrate_with_foil,
        "foil_path": foil_path,
        "foil_xas_ref": foil_xas_ref,
        "accepted_edge_energy": accepted_edge_energy,
        "foil_element": foil_element,
        "foil_edge": foil_edge,
    }


async def run_first_shell_fit_workflow(
    *,
    params,
    material_id: str,
    xas_path: str | None = None,
    xas_ref: str | None = None,
    calibrate_with_foil: bool = False,
    foil_path: str | None = None,
    foil_xas_ref: str | None = None,
    accepted_edge_energy: float | None = None,
    foil_element: str | None = None,
    foil_edge: str = "K",
    preprocessing_fn,
    calibrate_xas_with_foil_data_fn,
    load_fit_group_fn,
    extract_fitted_parameters_fn,
    extract_path_parameters_fn,
    viz_first_shell_fn,
    build_fit_report_from_cache_fn,
):
    # Default workflow: run the full fit once without checkpoint resume state.
    fit_target = xas_ref or xas_path
    if not fit_target:
        raise ValueError("Provide xas_path or xas_ref for fitting.")

    paths = await preprocessing_fn(material_id)
    cache_data = _build_fit_cache_data(
        params=params,
        material_id=material_id,
        xas_path=xas_path,
        xas_ref=xas_ref,
        calibrate_with_foil=calibrate_with_foil,
        foil_path=foil_path,
        foil_xas_ref=foil_xas_ref,
        accepted_edge_energy=accepted_edge_energy,
        foil_element=foil_element,
        foil_edge=foil_edge,
        paths=paths,
    )

    # Foil calibration is intentionally not part of the fitting checkpoint
    # sequence. If needed, run `calibrate_xas_energy` before fitting and pass the
    # resulting xas_ref here.
    cache_data["fit_xas_ref"] = cache_data.get("xas_ref")

    data, _ = load_fit_group_fn(
        xas_path=cache_data.get("xas_path"),
        xas_ref=cache_data.get("fit_xas_ref"),
    )
    result, paths_dict, _ = execute_first_shell_fit(
        cache_data,
        data=data,
        load_fit_group_fn=load_fit_group_fn,
    )

    fitted_parameters = extract_fitted_parameters_fn(result)
    path_parameters = extract_path_parameters_fn(result)
    cache_data["fitted_parameters"] = fitted_parameters.model_dump()
    cache_data["path_parameters"] = [p.model_dump() for p in path_parameters]

    viz_first_shell_fn(
        paths_dict,
        result,
        xas_path=cache_data.get("xas_path"),
        xas_ref=cache_data.get("fit_xas_ref") or fit_target,
    )

    return build_fit_report_from_cache_fn(
        cache_data=cache_data,
        fitted_parameters=fitted_parameters,
        path_parameters=path_parameters,
    )


async def orchestrate_first_shell_fit_with_checkpoints(
    *,
    params,
    material_id: str,
    xas_path: str | None = None,
    xas_ref: str | None = None,
    calibrate_with_foil: bool = False,
    foil_path: str | None = None,
    foil_xas_ref: str | None = None,
    accepted_edge_energy: float | None = None,
    foil_element: str | None = None,
    foil_edge: str = "K",
    preprocessing_fn,
    check_if_paused_fn,
    save_checkpoint_from_tool_fn,
    calibrate_xas_with_foil_data_fn,
    load_fit_group_fn,
    extract_fitted_parameters_fn,
    extract_path_parameters_fn,
    viz_first_shell_fn,
    build_fit_report_from_cache_fn,
):
    # Checkpoint-aware workflow: persists step state and can resume after interruption.
    fit_target = xas_ref or xas_path
    if not fit_target:
        raise ValueError("Provide xas_path or xas_ref for fitting.")

    print(f"Starting fit_ffef_first_shell for {fit_target}")

    cache_identity = {
        "material_id": material_id,
        "xas_path": xas_path,
        "xas_ref": xas_ref,
        "calibrate_with_foil": calibrate_with_foil,
        "foil_path": foil_path,
        "foil_xas_ref": foil_xas_ref,
        "accepted_edge_energy": accepted_edge_energy,
        "foil_element": foil_element,
        "foil_edge": foil_edge,
    }
    cache_key = hashlib.md5(json.dumps(cache_identity, sort_keys=True).encode("utf-8")).hexdigest()
    cache_file = checkpoints_dir() / f"fit_cache_{cache_key}.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    start_step = 0
    cache_data = None

    if cache_file.exists():
        cache_data = json.loads(cache_file.read_text())
        if cache_data.get("results_extracted") and cache_data.get("visualization_complete"):
            print("All steps already complete, returning cached results")
            return build_fit_report_from_cache_fn(cache_data=cache_data)
        elif cache_data.get("results_extracted"):
            start_step = 4
            print("Resuming from step 4 (visualization)")
        elif cache_data.get("fit_completed"):
            start_step = 3
            print("Resuming from step 3 (extraction)")
        elif cache_data.get("data_imported"):
            start_step = 2
            print("Resuming from step 2 (fitting)")
        elif cache_data.get("paths_prepared"):
            start_step = 1
            print("Resuming from step 1 (import)")
        else:
            print("Starting from step 0 (prepare)")
    else:
        print("No cache found, starting from step 0")

    if cache_data is None:
        paths = await preprocessing_fn(material_id)
        cache_data = _build_fit_cache_data(
            params=params,
            material_id=material_id,
            xas_path=xas_path,
            xas_ref=xas_ref,
            calibrate_with_foil=calibrate_with_foil,
            foil_path=foil_path,
            foil_xas_ref=foil_xas_ref,
            accepted_edge_energy=accepted_edge_energy,
            foil_element=foil_element,
            foil_edge=foil_edge,
            paths=paths,
        )

    paths_dict = {}
    data = None
    result = None
    fitted_parameters = None
    path_parameters = None

    await check_if_paused_fn()
    if start_step <= 0:
        cache_data["paths_prepared"] = True
        cache_file.write_text(json.dumps(cache_data))
        await save_checkpoint_from_tool_fn("fit_ffef_first_shell_step_0")

    await check_if_paused_fn()
    if start_step <= 1:
        cache_data["fit_xas_ref"] = cache_data.get("xas_ref")
        data, source_file = load_fit_group_fn(
            xas_path=cache_data.get("xas_path"),
            xas_ref=cache_data.get("fit_xas_ref"),
        )
        if not cache_data.get("fit_xas_ref"):
            cache_data["fit_xas_ref"] = _persist_fit_group_for_resume(
                data,
                fit_target=fit_target,
                source_file=source_file,
            )
        cache_data["fit_source_file"] = str(source_file or fit_target)
        cache_data["data_imported"] = True
        cache_file.write_text(json.dumps(cache_data, default=str))
        await save_checkpoint_from_tool_fn("fit_ffef_first_shell_step_1")

    if start_step <= 2:
        await check_if_paused_fn()
        result, paths_dict, data = execute_first_shell_fit(
            cache_data,
            data=data,
            load_fit_group_fn=load_fit_group_fn,
        )
        cache_data["result_chi2"] = float(result.chi2_reduced) if result.chi2_reduced else None
        cache_data["result_rfactor"] = float(result.rfactor) if result.rfactor else None
        cache_data["fit_completed"] = True
        cache_file.write_text(json.dumps(cache_data, default=str))
        await save_checkpoint_from_tool_fn("fit_ffef_first_shell_step_2")

    if start_step <= 3:
        await check_if_paused_fn()
        if result is None:
            result, paths_dict, data = execute_first_shell_fit(
                cache_data,
                data=None,
                load_fit_group_fn=load_fit_group_fn,
            )

        fitted_parameters = extract_fitted_parameters_fn(result)
        path_parameters = extract_path_parameters_fn(result)
        cache_data["fitted_parameters"] = fitted_parameters.model_dump()
        cache_data["path_parameters"] = [p.model_dump() for p in path_parameters]
        cache_data["results_extracted"] = True
        cache_file.write_text(json.dumps(cache_data, default=str))
        await save_checkpoint_from_tool_fn("fit_ffef_first_shell_step_3")

    if start_step <= 4:
        await check_if_paused_fn()
        if result is None:
            result, paths_dict, data = execute_first_shell_fit(
                cache_data,
                data=None,
                load_fit_group_fn=load_fit_group_fn,
            )
        if fitted_parameters is None or path_parameters is None:
            fitted_parameters = extract_fitted_parameters_fn(result)
            path_parameters = extract_path_parameters_fn(result)

        viz_first_shell_fn(
            paths_dict,
            result,
            xas_path=cache_data.get("xas_path"),
            xas_ref=cache_data.get("fit_xas_ref") or fit_target,
        )

        cache_data["visualization_complete"] = True
        cache_file.write_text(json.dumps(cache_data, default=str))
        await save_checkpoint_from_tool_fn("fit_ffef_first_shell_step_4")

    return build_fit_report_from_cache_fn(
        cache_data=cache_data,
        fitted_parameters=fitted_parameters,
        path_parameters=path_parameters,
    )

def extract_fitted_parameters(result) -> FittedParameter:
    params = result.params
    tr = result.datasets[0].transform
    s02, s02_err = 0, 0

    dataset = result.datasets[0]
    hashkey = dataset.hashkey
    paths = dataset.paths

    def safe_get(name: str):
        param = params.get(name)
        if param is None:
            return float("nan"), None
        return float(param.value), float(param.stderr) if param.stderr is not None else None

    for _, path in paths.items():
        path_hash = path.hashkey
        s02, s02_err = safe_get(f"s02_{hashkey}_{path_hash}")
        break

    return FittedParameter(
        nvar=result.nvarys,
        kmin=tr.kmin if tr else float("nan"),
        kmax=tr.kmax if tr else float("nan"),
        rmin=tr.rmin if tr else float("nan"),
        rmax=tr.rmax if tr else float("nan"),
        s02=s02,
        s02_err=s02_err,
        deltae=params["e0"].value if "e0" in params else float("nan"),
        errore=params["e0"].stderr if "e0" in params else float("nan"),
        reduced_chi2=result.chi2_reduced if result.chi2_reduced is not None else float("nan"),
        rfactor=result.rfactor if result.rfactor is not None else float("nan"),
    )


def extract_path_parameters(result) -> List[PathParameter]:
    params = result.params
    dataset = result.datasets[0]
    hashkey = dataset.hashkey
    paths = dataset.paths

    path_summaries = []

    def safe_get(name: str):
        param = params.get(name)
        if param is None:
            return float("nan"), None
        return float(param.value), float(param.stderr) if param.stderr is not None else None

    for label, path in paths.items():
        path_hash = path.hashkey
        reff = float(path.reff)

        deltar, deltar_err = safe_get(f"deltar_{hashkey}_{path_hash}")
        sigma2, sigma2_err = safe_get(f"sigma2_{hashkey}_{path_hash}")
        path_summaries.append(
            PathParameter(
                path_label=label,
                deltar=deltar,
                deltar_err=deltar_err,
                R=reff + (deltar if deltar is not None else 0.0),
                sigma2=sigma2,
                sigma2_err=sigma2_err,
            )
        )
    return path_summaries
