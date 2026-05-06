from __future__ import annotations

"""Public function-calling surface for XAS workflows.

This module gathers the main models, helpers, and tool-facing functions used by
the agent layer so callers can import a single package entrypoint for
preprocessing, fitting, lookup, plotting, and reporting workflows.
"""

import asyncio
import hashlib
import json
from pathlib import Path

import larch
from dotenv import load_dotenv
from larch.fitting import param, param_group
from larch.plot.plotly_xafsplots import plot_chifit
from larch.xafs import autobk, feffit, feffit_dataset, feffit_transform, feffpath, pre_edge, xftf, xftr

from checkpoint import check_if_paused, save_checkpoint_from_tool
from material_database import get_material_by_id, search_materials
from .artifacts import (
    artifact_key,
    build_overlay_payload,
    generate_xas_artifact_bundle,
    group_to_payload,
    persist_xas_payload,
    safe_float,
)
from spectrum_database import get_data_by_id, get_datasets
from storage import upload_file

from . import artifacts, fit, lookups, reports, tools
from .edge_identifier import EdgeIdentifier
from .feff import load_paths, make_and_run_feff
from .fit import (
    build_fit_report_from_cache,
    calibrate_xas_with_foil_data,
    detect_edge_energy_and_reference,
    lookup_reference_edge_energy,
)
from .models import (
    ElementDetectionResult,
    EnergyCalibrationResult,
    FEFF_Path,
    FEFFPathEntry,
    FEFFPathsResult,
    FittedParameter,
    MaterialStructureLookupResult,
    MultiSpectrumReportResult,
    Param,
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
from .xas_parser import XASParser
from .spectra_matching import extract_element_from_filename, load_user_spectrum, match_spectrum_for_agent
from .xas_io import load_embedded_foil_reference, load_prj_first_shell

load_dotenv()


def _build_fit_report_from_cache(
    *,
    cache_data: dict,
    fitted_parameters: FittedParameter | None,
    path_parameters: list[PathParameter] | None,
) -> Report:
    return build_fit_report_from_cache(
        cache_data=cache_data,
        fitted_parameters=fitted_parameters,
        path_parameters=path_parameters,
    )


_processed_dir = artifacts.processed_dir
_plot_dir = artifacts.plot_dir
_processed_payload_path = artifacts.processed_payload_path
_normalize_lookup_text = lookups.normalize_lookup_text


def _resolve_material_cif_path(cif_path: str | None) -> str | None:
    return lookups.resolve_material_cif_path(cif_path, backend_root=Path(__file__).resolve().parents[1])


_lookup_material_structure_data = lookups.lookup_material_structure_data
_search_xafs_database_data = lookups.search_xafs_database_data
_choose_primary_spectrum_path = lookups.choose_primary_spectrum_path
_fetch_xafs_dataset_data = lookups.fetch_xafs_dataset_data
_persist_preprocessed_payload = artifacts.persist_preprocessed_payload
_load_processed_payload = artifacts.load_processed_payload
_resolve_processed_payload = artifacts.resolve_processed_payload
_load_group_from_payload = artifacts.load_group_from_payload
_load_fit_group = artifacts.load_fit_group


_lookup_reference_edge_energy = lookup_reference_edge_energy
_detect_edge_energy_and_reference = detect_edge_energy_and_reference


def _calibrate_xas_with_foil_data(
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
) -> EnergyCalibrationResult:
    return calibrate_xas_with_foil_data(
        xas_path=xas_path,
        xas_ref=xas_ref,
        foil_path=foil_path,
        foil_xas_ref=foil_xas_ref,
        accepted_edge_energy=accepted_edge_energy,
        foil_element=foil_element,
        foil_edge=foil_edge,
        allow_self_calibration=allow_self_calibration,
        spectrum_is_foil=spectrum_is_foil,
    )


_build_line_plot = reports.build_line_plot
_upload_if_possible = reports.upload_if_possible
_prepare_feff_paths_data = lookups.prepare_feff_paths_data
_load_and_preprocess_xas_data = artifacts.load_and_preprocess_xas_data
_plot_preprocessed_xas_data = reports.plot_preprocessed_xas_data
_readable_plot_title = reports.readable_plot_title
_build_preview_group_from_preprocessing = reports.build_preview_group_from_preprocessing
_build_overlay_preview_group = reports.build_overlay_preview_group
_serialize_fit_report_message = reports.serialize_fit_report_message


def _build_preview_group_from_fit(
    *,
    material_id: str,
    xas_path: str,
    preprocessed: PreprocessedXASResult,
    report: Report,
) -> PreviewGroupResult:
    return reports.build_preview_group_from_fit(
        material_id=material_id,
        xas_path=xas_path,
        preprocessed=preprocessed,
        report=report,
    )
from .tools import (
    calibrate_xas_energy,
    detect_xas_element,
    fetch_xafs_dataset,
    fit_ffef_first_shell,
    load_and_preprocess_xas,
    lookup_material_structure,
    manipulate_xas_data,
    match_xas_spectra,
    plot_preprocessed_xas,
    prepare_feff_paths,
    prepare_multi_fit_reports,
    prepare_multi_spectrum_reports,
    search_xafs_database,
)
from .fit import extract_fitted_parameters, extract_path_parameters, preprocessing
from .reports import viz_first_shell


__all__ = [
    "Param",
    "Param_first_shell",
    "FittedParameter",
    "PathParameter",
    "Report",
    "FEFFPathEntry",
    "FEFF_Path",
    "FEFFPathsResult",
    "PreprocessedXASResult",
    "PlotArtifact",
    "PlotResult",
    "EnergyCalibrationResult",
    "PreviewReportItem",
    "PreviewGroupResult",
    "MultiSpectrumReportResult",
    "ElementDetectionResult",
    "SpectraMatchEntry",
    "SpectraMatchResult",
    "MaterialStructureLookupResult",
    "SpectrumDatabaseSearchResult",
    "SpectrumDatasetRecord",
    "SpectrumDatasetFetchResult",
    "XASParser",
    "prepare_feff_paths",
    "lookup_material_structure",
    "search_xafs_database",
    "fetch_xafs_dataset",
    "load_and_preprocess_xas",
    "manipulate_xas_data",
    "plot_preprocessed_xas",
    "calibrate_xas_energy",
    "prepare_multi_spectrum_reports",
    "prepare_multi_fit_reports",
    "detect_xas_element",
    "match_xas_spectra",
    "preprocessing",
    "fit_ffef_first_shell",
    "viz_first_shell",
    "extract_fitted_parameters",
    "extract_path_parameters",
]
