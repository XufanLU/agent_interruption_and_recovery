"""Lookup helpers for resolving materials, datasets, and FEFF path inputs.

This module contains read-oriented helpers used by agent tools to:
- resolve material queries like formulas or Materials Project ids to CIF files
- search the XAFS dataset catalog
- fetch dataset file paths from dataset identifiers
- prepare FEFF path metadata from a structure identifier

These functions are lower-level building blocks for the tool workflows in
`backend/function_calling/tools.py`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

from material_database import get_material_by_id, search_materials
from spectrum_database import get_data_by_id, get_datasets

from .models import (
    FEFFPathEntry,
    FEFFPathsResult,
    MaterialStructureLookupResult,
    SpectrumDatabaseSearchResult,
    SpectrumDatasetFetchResult,
    SpectrumDatasetRecord,
)
from .feff import load_paths, make_and_run_feff


def normalize_lookup_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def resolve_material_cif_path(cif_path: str | None, *, backend_root: Path | None = None) -> str | None:
    if not cif_path:
        return None

    candidate = Path(cif_path)
    if candidate.is_absolute():
        return str(candidate)

    root = backend_root or Path(__file__).resolve().parents[1]
    return str((root / candidate).resolve())


def lookup_material_structure_data(query: str) -> MaterialStructureLookupResult:
    """Resolve a material query into a structured local CIF lookup result.

    The query can be a formula, chemical system, or Materials Project id.
    This helper finds the best matching material record, resolves its CIF
    path, and returns the normalized lookup result object.
    """
    normalized_query = str(query or "").strip()
    if not normalized_query:
        raise ValueError("Provide a material formula, chemical system, or Materials Project id.")

    material_id = normalized_query if normalized_query.lower().startswith("mp-") else search_materials(normalized_query)
    if not material_id:
        raise ValueError(f"No material match found for '{normalized_query}'.")

    material_result = get_material_by_id(material_id)
    cif_path = material_result.get("file_path") if isinstance(material_result, dict) else material_result
    resolved_path = resolve_material_cif_path(cif_path)
    if not resolved_path:
        raise ValueError(f"Failed to fetch a CIF for '{material_id}'.")

    return MaterialStructureLookupResult(
        success=True,
        query=normalized_query,
        material_id=material_id,
        cif_path=resolved_path,
        error=None,
    )


def search_xafs_database_data(
    query: str,
    limit: int = 10,
) -> SpectrumDatabaseSearchResult:
    normalized_query = str(query or "").strip()
    if not normalized_query:
        raise ValueError("Provide a dataset search query such as a material name, specimen, or dataset id.")

    datasets = get_datasets() or {}
    query_text = normalize_lookup_text(normalized_query)
    query_tokens = [token for token in query_text.split() if token]
    ranked: List[tuple[int, str, str, str]] = []

    for title, raw_value in datasets.items():
        dataset_id = ""
        specimen = ""
        if isinstance(raw_value, (list, tuple)):
            if len(raw_value) > 0:
                dataset_id = str(raw_value[0] or "")
            if len(raw_value) > 1:
                specimen = str(raw_value[1] or "")
        else:
            dataset_id = str(raw_value or "")

        haystack = normalize_lookup_text(f"{title} {specimen} {dataset_id}")
        score = 0
        if query_text and dataset_id.lower() == normalized_query.lower():
            score += 100
        if query_text and query_text in haystack:
            score += 25
        score += sum(5 for token in query_tokens if token in haystack)

        if score <= 0:
            continue

        ranked.append((score, str(title), dataset_id, specimen))

    ranked.sort(key=lambda item: (-item[0], item[1].lower(), item[2].lower()))
    matches = [
        SpectrumDatasetRecord(title=title, dataset_id=dataset_id, specimen=specimen or None)
        for _, title, dataset_id, specimen in ranked[: max(1, int(limit))]
    ]

    return SpectrumDatabaseSearchResult(
        success=True,
        query=normalized_query,
        matches=matches,
        error=None,
    )


def choose_primary_spectrum_path(file_paths: List[str]) -> str | None:
    if not file_paths:
        return None

    preferred_suffixes = {".txt", ".dat", ".csv", ".xdi"}
    for file_path in file_paths:
        suffix = Path(file_path).suffix.lower()
        if suffix in preferred_suffixes or suffix.startswith(".0"):
            return file_path
    return file_paths[0]


def fetch_xafs_dataset_data(dataset_id: str) -> SpectrumDatasetFetchResult:
    normalized_dataset_id = str(dataset_id or "").strip()
    if not normalized_dataset_id:
        raise ValueError("Provide a dataset id to download.")

    raw_paths = get_data_by_id(normalized_dataset_id) or []
    file_paths = [str(path) for path in raw_paths if path]
    if not file_paths:
        raise ValueError(f"No spectrum files were downloaded for dataset '{normalized_dataset_id}'.")

    return SpectrumDatasetFetchResult(
        success=True,
        dataset_id=normalized_dataset_id,
        file_paths=file_paths,
        primary_path=choose_primary_spectrum_path(file_paths),
        error=None,
    )


def prepare_feff_paths_data(
    material_id: str,
    edge: str = "K",
    radius: float = 5.0,
    amp_ratio: float | None = None,
    r_max: float | None = None,
) -> FEFFPathsResult:
    feff_dir = make_and_run_feff(material_id, radius=radius, edge=edge)
    paths = load_paths(feff_dir, amp_ratio=amp_ratio, r_max=r_max) or {}
    return FEFFPathsResult(
        success=True,
        material_id=material_id,
        paths=[FEFFPathEntry(name=name, path=path) for name, path in paths.items()],
        error=None,
    )
