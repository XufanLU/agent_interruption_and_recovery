from __future__ import annotations

"""Agent tool wrappers for the function-calling workflow layer.

This module exposes the higher-level tool functions that agents call directly,
bridging lookup, preprocessing, plotting, matching, calibration, and fitting
helpers into user-facing operations.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, List

import larch
import numpy as np
from agents import function_tool
from larch.fitting import param, param_group
from larch.xafs import autobk, feffit, feffit_dataset, feffit_transform, feffpath, pre_edge, xftf, xftr
from .artifacts import group_to_payload, safe_float

from . import artifacts, lookups, reports
from .edge_identifier import EdgeIdentifier
from .fit import (
    build_fit_report_from_cache,
    calibrate_xas_with_foil_data,
    detect_edge_energy_and_reference,
    extract_fitted_parameters,
    extract_path_parameters,
    lookup_reference_edge_energy,
    orchestrate_first_shell_fit_with_checkpoints,
    preprocessing,
    run_first_shell_fit_workflow,
)
from checkpoint import check_if_paused, save_checkpoint_from_tool
from .models import (
    ElementDetectionResult,
    EnergyCalibrationResult,
    FEFFPathEntry,
    FEFFPathsResult,
    FittedParameter,
    MaterialStructureLookupResult,
    MultiSpectrumReportResult,
    Param_first_shell,
    PathParameter,
    PlotArtifact,
    PlotResult,
    PreprocessedXASResult,
    PreviewGroupResult,
    PreviewReportItem,
    Report,
    SpectraMatchEntry,
    SpectraMatchResult,
    SpectrumDatabaseSearchResult,
    SpectrumDatasetFetchResult,
    SpectrumDatasetRecord,
)
from .reports import viz_first_shell
from .spectra_matching import extract_element_from_filename, load_user_spectrum, match_spectrum_for_agent

@function_tool
async def prepare_feff_paths(
    material_id: str,
    edge: str = "K",
    radius: float = 5.0,
    amp_ratio: float | None = None,
    r_max: float | None = None,
) -> FEFFPathsResult:
    """
    Prepare FEFF paths for the provided material. Use this when the user asks for
    FEFF path generation or when a partial fitting workflow needs only path setup.
    """
    try:
        return await asyncio.to_thread(
            lookups.prepare_feff_paths_data,
            material_id,
            edge,
            radius,
            amp_ratio,
            r_max,
        )
    except Exception as e:
        return FEFFPathsResult(success=False, material_id=material_id, paths=[], error=str(e))


@function_tool
def lookup_material_structure(query: str) -> MaterialStructureLookupResult:
    """
    Resolve a material formula or Materials Project id to a local CIF path.
    Use this when the user wants a structure from the material database instead of uploading a CIF.
    """
    try:
        return lookups.lookup_material_structure_data(query)
    except Exception as e:
        return MaterialStructureLookupResult(success=False, query=str(query or ""), error=str(e))


@function_tool
def search_xafs_database(
    query: str,
    limit: int = 10,
) -> SpectrumDatabaseSearchResult:
    """
    Search the online XAFS dataset catalog by title, specimen, or dataset id.
    Use this before downloading a spectrum from the MDR/NIMS XAFS database.
    """
    try:
        return lookups.search_xafs_database_data(query, limit=limit)
    except Exception as e:
        return SpectrumDatabaseSearchResult(success=False, query=str(query or ""), matches=[], error=str(e))


@function_tool
def fetch_xafs_dataset(dataset_id: str) -> SpectrumDatasetFetchResult:
    """
    Download an online XAFS dataset and return the local spectrum files.
    Use the returned primary_path for downstream preprocessing, matching, or fitting tools.
    """
    try:
        return lookups.fetch_xafs_dataset_data(dataset_id)
    except Exception as e:
        return SpectrumDatasetFetchResult(success=False, dataset_id=str(dataset_id or ""), file_paths=[], error=str(e))


@function_tool
def load_and_preprocess_xas(xas_path: str) -> PreprocessedXASResult:
    """
    Load and preprocess XAS data, saving a reusable processed artifact. Use this for
    requests such as normalization, preprocessing summaries, and as the first step
    before plotting or matching.
    """
    try:
        return artifacts.load_and_preprocess_xas_data(xas_path)
    except Exception as e:
        return PreprocessedXASResult(success=False, error=str(e))


@function_tool
def manipulate_xas_data(
    xas_path: str | None = None,
    xas_ref: str | None = None,
    energy_min: float | None = None,
    energy_max: float | None = None,
) -> PreprocessedXASResult:
    """
    Apply simple data manipulation to an XAS spectrum. Currently supports energy-range
    cropping and writes the manipulated spectrum as a new reusable artifact.
    """
    try:
        key, payload = artifacts.resolve_processed_payload(xas_path=xas_path, xas_ref=xas_ref)
        energy = np.asarray(payload.get("energy", []), dtype=float)
        mu = np.asarray(payload.get("mu", []), dtype=float)
        if energy.size == 0 or mu.size == 0:
            raise ValueError("No energy/mu data available for manipulation.")

        mask = np.ones_like(energy, dtype=bool)
        if energy_min is not None:
            mask &= energy >= energy_min
        if energy_max is not None:
            mask &= energy <= energy_max
        if not np.any(mask):
            raise ValueError("No points remain after applying the requested energy range.")

        cropped = larch.Group(
            name=f"{key}_manipulated",
            energy=energy[mask],
            mu=mu[mask],
        )
        pre_edge(cropped)
        autobk(cropped, rbkg=1.0, kweight=2)
        xftf(cropped, kweight=2)
        xftr(cropped)

        manipulated_ref = f"{key}_manipulated"
        manipulated_payload = group_to_payload(cropped, source_file=payload.get("source_file"))
        return artifacts.persist_preprocessed_payload(manipulated_ref, manipulated_payload)
    except Exception as e:
        return PreprocessedXASResult(success=False, error=str(e))


@function_tool
def plot_preprocessed_xas(
    xas_path: str | None = None,
    xas_ref: str | None = None,
) -> PlotResult:
    """
    Plot preprocessed XAS data without running fitting or matching.
    """
    try:
        return reports.plot_preprocessed_xas_data(xas_path=xas_path, xas_ref=xas_ref)
    except Exception as e:
        return PlotResult(success=False, xas_ref=xas_ref, plots=[], error=str(e))


@function_tool
def calibrate_xas_energy(
    xas_path: str | None = None,
    xas_ref: str | None = None,
    foil_path: str | None = None,
    foil_xas_ref: str | None = None,
    accepted_edge_energy: float | None = None,
    foil_element: str | None = None,
    foil_edge: str = "K",
    spectrum_is_foil: bool = False,
) -> EnergyCalibrationResult:
    """
    Calibrate a spectrum energy axis against a foil reference.
    Use this only when the user explicitly asks for foil-based calibration or
    provides a foil reference to align the energy scale.
    """
    try:
        return calibrate_xas_with_foil_data(
            xas_path=xas_path,
            xas_ref=xas_ref,
            foil_path=foil_path,
            foil_xas_ref=foil_xas_ref,
            accepted_edge_energy=accepted_edge_energy,
            foil_element=foil_element,
            foil_edge=foil_edge,
            allow_self_calibration=True,
            spectrum_is_foil=spectrum_is_foil,
        )
    except Exception as e:
        return EnergyCalibrationResult(success=False, error=str(e))


@function_tool
def prepare_multi_spectrum_reports(xas_paths: List[str]) -> MultiSpectrumReportResult:
    """
    Prepare right-side preview reports for multiple spectra at once.
    Use this when the user wants to compare, inspect, preprocess, or plot several
    attached spectra together instead of working on only one spectrum.
    """
    preview_groups: List[PreviewGroupResult] = []
    errors: List[str] = []
    preprocessed_results: List[PreprocessedXASResult] = []

    for xas_path in xas_paths:
        try:
            preprocessed = artifacts.load_and_preprocess_xas_data(xas_path)
            if not preprocessed.success or not preprocessed.xas_ref:
                errors.append(f"{xas_path}: {preprocessed.error or 'preprocessing failed'}")
                continue

            plots = reports.plot_preprocessed_xas_data(xas_ref=preprocessed.xas_ref)
        except Exception as exc:
            errors.append(f"{xas_path}: {exc}")
            continue

        preprocessed_results.append(preprocessed)
        preview_groups.append(reports.build_preview_group_from_preprocessing(preprocessed, plots))

    overlay_group = reports.build_overlay_preview_group(preprocessed_results)
    if overlay_group is not None:
        preview_groups.insert(0, overlay_group)

    if not preview_groups:
        return MultiSpectrumReportResult(
            success=False,
            preview_groups=[],
            error="; ".join(errors) or "Failed to prepare reports for the spectra.",
        )

    summary = (
        f"Prepared interactive reports for {len(preprocessed_results)} spectra. "
        "Use the report switcher on the right to compare them or inspect each spectrum."
    )
    if errors:
        summary += " Some spectra could not be prepared: " + "; ".join(errors)

    return MultiSpectrumReportResult(
        success=True,
        preview_groups=preview_groups,
        summary=summary,
        error="; ".join(errors) if errors else None,
    )


@function_tool
async def prepare_multi_fit_reports(
    material_id: str,
    xas_paths: List[str],
    amp: float = 0.8,
    e0: float = 0.0,
    sigma2: float = 0.003,
    deltar: float = 0.0,
    max_parallel: int = 5,
) -> MultiSpectrumReportResult:
    """
    Run first-shell EXAFS fitting for multiple spectra against the same CIF/structure
    and return one grouped right-panel report per spectrum.
    Use this when the user asks to fit both/all/multiple attached spectra with one CIF.
    """
    if not material_id:
        return MultiSpectrumReportResult(
            success=False,
            preview_groups=[],
            error="Missing CIF structure context for multi-spectrum fitting.",
        )

    normalized_xas_paths = [path for path in xas_paths if path]
    if not normalized_xas_paths:
        return MultiSpectrumReportResult(
            success=False,
            preview_groups=[],
            error="No XAS spectra were provided for multi-spectrum fitting.",
        )

    prepared_paths = await asyncio.to_thread(lookups.prepare_feff_paths_data, material_id)
    if not prepared_paths.success or not prepared_paths.paths:
        return MultiSpectrumReportResult(
            success=False,
            preview_groups=[],
            error=prepared_paths.error or f"Failed to prepare FEFF paths for {material_id}.",
        )

    first_path_str = prepared_paths.paths[0].path
    params = Param_first_shell(amp=amp, e0=e0, sigma2=sigma2, deltar=deltar)

    concurrency = max(1, min(int(max_parallel), len(normalized_xas_paths), 4))
    semaphore = asyncio.Semaphore(concurrency)
    errors: List[str] = []

    async def _single_path_preprocessing(_: str) -> dict[str, str]:
        return {"path1": first_path_str}

    async def run_fit(xas_path: str) -> PreviewGroupResult | None:
        async with semaphore:
            try:
                preprocessed = await asyncio.to_thread(artifacts.load_and_preprocess_xas_data, xas_path)
                if not preprocessed.success:
                    raise ValueError(preprocessed.error or f"Failed to preprocess {xas_path}.")

                report = await run_first_shell_fit_workflow(
                    params=params,
                    material_id=material_id,
                    xas_path=xas_path,
                    xas_ref=preprocessed.xas_ref,
                    preprocessing_fn=_single_path_preprocessing,
                    calibrate_xas_with_foil_data_fn=calibrate_xas_with_foil_data,
                    load_fit_group_fn=artifacts.load_fit_group,
                    extract_fitted_parameters_fn=extract_fitted_parameters,
                    extract_path_parameters_fn=extract_path_parameters,
                    viz_first_shell_fn=viz_first_shell,
                    build_fit_report_from_cache_fn=build_fit_report_from_cache,
                )
                return reports.build_preview_group_from_fit(
                    material_id=material_id,
                    xas_path=xas_path,
                    preprocessed=preprocessed,
                    report=report,
                )
            except Exception as exc:
                errors.append(f"{xas_path}: {exc}")
                return None

    preview_groups = [
        group
        for group in await asyncio.gather(*(run_fit(xas_path) for xas_path in normalized_xas_paths))
        if group is not None
    ]

    if not preview_groups:
        return MultiSpectrumReportResult(
            success=False,
            preview_groups=[],
            error="; ".join(errors) or "Failed to prepare multi-spectrum fitting reports.",
        )

    summary = (
        f"Completed first-shell EXAFS fitting for {len(preview_groups)} spectra using the same structure. "
        "Use the report switcher on the right to compare the fit outputs."
    )
    if errors:
        summary += " Some spectra could not be fitted: " + "; ".join(errors)

    return MultiSpectrumReportResult(
        success=True,
        preview_groups=preview_groups,
        summary=summary,
        error="; ".join(errors) if errors else None,
    )


@function_tool
def detect_xas_element(
    xas_path: str | None = None,
    xas_ref: str | None = None,
) -> ElementDetectionResult:
    """
    Detect the absorbing element from a preprocessed or raw XAS spectrum.
    """
    try:
        if xas_ref:
            payload = artifacts.load_processed_payload(xas_ref)
            if payload:
                energy = np.asarray(payload.get("energy", []), dtype=float)
                mu_raw = np.asarray(payload.get("mu", []), dtype=float)
                source_name = payload.get("source_file") or xas_ref
            else:
                energy, mu_raw, _, _ = load_user_spectrum(xas_ref)
                source_name = xas_ref
        else:
            energy, mu_raw, _, _ = load_user_spectrum(xas_path)
            source_name = xas_path

        if energy is None or mu_raw is None:
            raise ValueError("Could not load spectrum for element detection.")

        identifier = EdgeIdentifier()
        edge_energy = identifier.find_edge_energy(energy, mu_raw)
        matches = identifier.identify_element(edge_energy)
        element = matches[0]["Element"] if matches else extract_element_from_filename(Path(str(source_name)).name)
        if not element:
            raise ValueError("Could not detect the absorbing element.")

        return ElementDetectionResult(
            success=True,
            element=element,
            e0_detected=safe_float(edge_energy),
            error=None,
        )
    except Exception as e:
        return ElementDetectionResult(success=False, error=str(e))


@function_tool
def match_xas_spectra(
    xas_path: str,
    element: str | None = None,
    top_n: int = 5,
    save_plot: bool = True,
) -> SpectraMatchResult:
    """
    Match a user XAS spectrum against the reference database to find similar materials.

    Use this when the user wants to identify or compare their XAS spectrum against known
    reference spectra (e.g., from LISA, SSHADE, XASDB). 
    """
    raw = match_spectrum_for_agent(
        spectrum_file_path=xas_path,
        element=element,
        top_n=top_n,
        save_plot=save_plot,
    )
    return SpectraMatchResult(
        success=raw["success"],
        element=raw["element"],
        e0_detected=raw["e0_detected"],
        top_matches=[SpectraMatchEntry(**match) for match in raw["top_matches"]],
        plot_path=raw.get("plot_path"),
        error=raw["error"],
    )


@function_tool
async def fit_ffef_first_shell(
    params: Param_first_shell,
    material_id: str,
    xas_path: str | None = None,
    xas_ref: str | None = None,
) -> Report:
    """
    Fit XAFS data using the provided parameters to FEFF paths.
    This is the checkpoint-aware workflow for a single first-shell fit.
    """
    def _build_report_from_cache(
        *,
        cache_data: dict,
        fitted_parameters: FittedParameter | None = None,
        path_parameters: List[PathParameter] | None = None,
    ) -> Report:
        if fitted_parameters is None and cache_data.get("fitted_parameters") is not None:
            fitted_parameters = FittedParameter(**cache_data["fitted_parameters"])
        if path_parameters is None and cache_data.get("path_parameters") is not None:
            path_parameters = [PathParameter(**item) for item in cache_data["path_parameters"]]
        return build_fit_report_from_cache(
            cache_data=cache_data,
            fitted_parameters=fitted_parameters,
            path_parameters=path_parameters,
        )

    return await orchestrate_first_shell_fit_with_checkpoints(
        params=params,
        material_id=material_id,
        xas_path=xas_path,
        xas_ref=xas_ref,
        preprocessing_fn=preprocessing,
        check_if_paused_fn=check_if_paused,
        save_checkpoint_from_tool_fn=save_checkpoint_from_tool,
        calibrate_xas_with_foil_data_fn=calibrate_xas_with_foil_data,
        load_fit_group_fn=artifacts.load_fit_group,
        extract_fitted_parameters_fn=extract_fitted_parameters,
        extract_path_parameters_fn=extract_path_parameters,
        viz_first_shell_fn=viz_first_shell,
        build_fit_report_from_cache_fn=_build_report_from_cache,
    )
