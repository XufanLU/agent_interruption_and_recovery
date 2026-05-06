from __future__ import annotations

"""Pydantic models shared across function-calling workflows.

This module defines the typed request/response and report models exchanged
between lookup, preprocessing, plotting, matching, and fitting helpers.
"""

from typing import Dict, List

from pydantic import BaseModel, Field


class Param(BaseModel):
    amp: float
    e0: float
    alpha: float
    sigma2: float
    sigma2_2: float
    sigma2_4: float


class Param_first_shell(BaseModel):
    amp: float
    e0: float
    sigma2: float
    deltar: float


class FittedParameter(BaseModel):
    nvar: int | None
    kmin: float
    kmax: float
    rmin: float
    rmax: float
    s02: float
    s02_err: float | None
    deltae: float
    errore: float | None
    reduced_chi2: float
    rfactor: float


class PathParameter(BaseModel):
    path_label: str
    deltar: float
    deltar_err: float | None
    R: float
    sigma2: float
    sigma2_err: float | None


class FEFFPathEntry(BaseModel):
    name: str
    path: str


class FEFF_Path(BaseModel):
    entries: List[FEFFPathEntry]

    def items(self):
        return [(entry.name, entry.path) for entry in self.entries]


class Report(BaseModel):
    fitted_parameter: FittedParameter | None = None
    path_parameter: List[PathParameter] | None = None
    xas_ref: str | None = None
    source_file: str | None = None
    xas_url: str | None = None


class PlotArtifact(BaseModel):
    name: str
    path: str
    s3_key: str | None = None


class FEFFPathsResult(BaseModel):
    success: bool
    material_id: str
    paths: List[FEFFPathEntry]
    error: str | None = None


class PreprocessedXASResult(BaseModel):
    success: bool
    xas_ref: str | None = None
    source_file: str | None = None
    e0: float | None = None
    n_points: int | None = None
    energy_min: float | None = None
    energy_max: float | None = None
    processed_path: str | None = None
    s3_key: str | None = None
    error: str | None = None


class PlotResult(BaseModel):
    success: bool
    xas_ref: str | None = None
    plots: List[PlotArtifact]
    error: str | None = None


class EnergyCalibrationResult(BaseModel):
    success: bool
    xas_ref: str | None = None
    source_file: str | None = None
    foil_ref: str | None = None
    foil_source_file: str | None = None
    foil_element: str | None = None
    foil_edge: str | None = None
    measured_foil_edge_energy: float | None = None
    accepted_foil_edge_energy: float | None = None
    energy_offset_ev: float | None = None
    e0: float | None = None
    n_points: int | None = None
    energy_min: float | None = None
    energy_max: float | None = None
    processed_path: str | None = None
    s3_key: str | None = None
    error: str | None = None


class PreviewReportItem(BaseModel):
    kind: str
    title: str
    source: str | None = None


class PreviewGroupResult(BaseModel):
    id: str
    title: str
    message: str | None = None
    material_url: str | None = None
    xas_url: str | None = None
    fitting_result_url: str | None = None
    chifit_result1_url: str | None = None
    chifit_result2_url: str | None = None
    matching_result_url: str | None = None
    preprocessing_plot_urls: List[Dict[str, str]] = Field(default_factory=list)
    report_items: List[PreviewReportItem] = Field(default_factory=list)
    is_fitting: bool = False


class MultiSpectrumReportResult(BaseModel):
    success: bool
    preview_groups: List[PreviewGroupResult] = Field(default_factory=list)
    summary: str | None = None
    error: str | None = None


class ElementDetectionResult(BaseModel):
    success: bool
    element: str | None = None
    e0_detected: float | None = None
    error: str | None = None


class SpectraMatchEntry(BaseModel):
    material_name: str
    source_database: str
    score: float


class SpectraMatchResult(BaseModel):
    success: bool
    element: str
    e0_detected: float | None
    top_matches: List[SpectraMatchEntry]
    plot_path: str | None = None
    error: str | None


class MaterialStructureLookupResult(BaseModel):
    success: bool
    query: str
    material_id: str | None = None
    cif_path: str | None = None
    error: str | None = None


class SpectrumDatasetRecord(BaseModel):
    title: str
    dataset_id: str
    specimen: str | None = None


class SpectrumDatabaseSearchResult(BaseModel):
    success: bool
    query: str
    matches: List[SpectrumDatasetRecord] = Field(default_factory=list)
    error: str | None = None


class SpectrumDatasetFetchResult(BaseModel):
    success: bool
    dataset_id: str
    file_paths: List[str] = Field(default_factory=list)
    primary_path: str | None = None
    error: str | None = None
