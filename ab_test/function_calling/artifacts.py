from __future__ import annotations

"""Helpers for persisted preprocessing artifacts and plot inputs.

This module centralizes the processed-XAS artifact lifecycle: converting Larch
groups into JSON-friendly payloads, optionally adding wavelet views, persisting
and resolving processed spectra, and reconstructing the saved artifacts for
later plotting and fitting steps.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import larch
import numpy as np
from larch.xafs import autobk, cauchy_wavelet, pre_edge, xftf, xftr
from data_paths import ensure_dir, processed_xas_dir, project_relative_path, viz_kind_dir
from storage import upload_file

from .models import PreprocessedXASResult
from .xas_io import load_prj_first_shell


@dataclass
class XASArtifactBundle:
    xas_id: str
    local_path: str
    remote_key: str
    payload: dict[str, Any]


def artifact_key(value: str) -> str:
    base = Path(str(value)).name or str(value)
    stem = Path(base).stem or base
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in stem)


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        cast_value = float(value)
        if np.isnan(cast_value):
            return None
        return cast_value
    except Exception:
        return None


def processed_dir(*, base_dir: Path | None = None) -> Path:
    return ensure_dir(processed_xas_dir(base_dir))


def plot_dir(kind: str, *, base_dir: Path | None = None) -> Path:
    return ensure_dir(viz_kind_dir(kind, base_dir))


def group_to_payload(data: Any, source_file: str | None = None) -> dict[str, Any]:
    normalized_source_file = project_relative_path(source_file)
    payload: dict[str, Any] = {
        "energy": np.asarray(getattr(data, "energy", []), dtype=float).tolist(),
        "mu": np.asarray(getattr(data, "mu", []), dtype=float).tolist(),
        "norm": np.asarray(getattr(data, "norm", []), dtype=float).tolist()
        if hasattr(data, "norm")
        else [],
        "flat": np.asarray(getattr(data, "flat", []), dtype=float).tolist()
        if hasattr(data, "flat")
        else [],
        "pre_edge": np.asarray(getattr(data, "pre_edge", []), dtype=float).tolist()
        if hasattr(data, "pre_edge")
        else [],
        "post_edge": np.asarray(getattr(data, "post_edge", []), dtype=float).tolist()
        if hasattr(data, "post_edge")
        else [],
        "e0": safe_float(getattr(data, "e0", None)),
        "edge_step": safe_float(getattr(data, "edge_step", None)),
        "source_file": normalized_source_file,
        "metadata": {
            "filename": Path(normalized_source_file).stem if normalized_source_file else None,
        },
    }
    if hasattr(data, "k"):
        payload["k"] = np.asarray(getattr(data, "k"), dtype=float).tolist()
    if hasattr(data, "chi"):
        payload["chi"] = np.asarray(getattr(data, "chi"), dtype=float).tolist()
    if hasattr(data, "r"):
        payload["r"] = np.asarray(getattr(data, "r"), dtype=float).tolist()
    if hasattr(data, "chir_mag"):
        payload["chir_mag"] = np.asarray(getattr(data, "chir_mag"), dtype=float).tolist()
    return payload


def add_wavelet_payload(
    payload: dict[str, Any],
    data: Any,
    *,
    weights: list[int] | None = None,
    r_min: float = 0.0,
    r_max: float = 6.0,
) -> dict[str, Any]:
    chi_unweighted = np.asarray(data.chi)
    resolved_weights = weights or [0, 1, 2, 3]

    k_arrays: list[list[float]] = []
    r_arrays: list[list[float]] = []
    z_arrays: list[list[list[float]]] = []
    top_traces: list[list[float]] = []
    left_traces: list[list[float]] = []

    for weight in resolved_weights:
        data.chi = chi_unweighted * (data.k ** weight)
        cauchy_wavelet(data, kweight=0)

        k_axis = np.asarray(data.k)
        r_axis = np.asarray(data.wcauchy_r)
        z_full = np.asarray(data.wcauchy_mag)

        mask = (r_axis >= r_min) & (r_axis <= r_max)
        r_display = r_axis[mask]
        z_display = z_full[mask, :]

        k_arrays.append(k_axis.tolist())
        r_arrays.append(r_display.tolist())
        z_arrays.append(z_display.tolist())
        top_traces.append(np.asarray(data.chi).tolist())
        left_traces.append((-z_display.max(axis=1)).tolist())

    data.chi = chi_unweighted

    payload.update(
        {
            "weights": resolved_weights,
            "k_arrays": k_arrays,
            "r_arrays": r_arrays,
            "z_arrays": z_arrays,
            "top_traces": top_traces,
            "left_traces": left_traces,
        }
    )
    return payload


def persist_xas_payload(
    xas_id: str,
    payload: dict[str, Any],
    *,
    bucket_name: str = "test-dr-xas",
    upload: bool = True,
    remote_key: str | None = None,
) -> XASArtifactBundle:
    normalized_id = artifact_key(xas_id)
    output_path = processed_dir() / f"{normalized_id}.json"
    output_path.write_text(json.dumps(payload))

    if remote_key:
        resolved_remote_key = remote_key.format(xas_id=normalized_id)
    else:
        resolved_remote_key = normalized_id
    if upload:
        upload_file(str(output_path), bucket_name, resolved_remote_key)

    return XASArtifactBundle(
        xas_id=normalized_id,
        local_path=project_relative_path(output_path) or str(output_path),
        remote_key=resolved_remote_key,
        payload=payload,
    )


def generate_xas_artifact_bundle(
    xas_path: str,
    *,
    bucket_name: str = "test-dr-xas",
    upload: bool = True,
    include_wavelet: bool = True,
    remote_key: str | None = None,
) -> XASArtifactBundle:
    data, filename = load_prj_first_shell(xas_path)
    source_file = str(filename) if filename else xas_path
    xas_id = artifact_key(filename.name if filename else xas_path)
    payload = group_to_payload(data, source_file=source_file)
    if include_wavelet:
        add_wavelet_payload(payload, data)
    return persist_xas_payload(
        xas_id,
        payload,
        bucket_name=bucket_name,
        upload=upload,
        remote_key=remote_key,
    )


def build_overlay_payload(
    payloads: list[dict[str, Any]],
    *,
    mode: str = "norm",
    title: str = "Spectra Overlay",
) -> dict[str, Any]:
    series: list[dict[str, Any]] = []

    for index, payload in enumerate(payloads):
        source_file = payload.get("source_file")
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        label = (
            metadata.get("filename")
            or (Path(str(source_file)).stem if source_file else None)
            or f"Spectrum {index + 1}"
        )
        series.append(
            {
                "label": label,
                "source_file": source_file,
                "energy": np.asarray(payload.get("energy", []), dtype=float).tolist(),
                "mu": np.asarray(payload.get("mu", []), dtype=float).tolist(),
                "norm": np.asarray(payload.get("norm", []), dtype=float).tolist(),
                "flat": np.asarray(payload.get("flat", []), dtype=float).tolist(),
                "e0": safe_float(payload.get("e0")),
                "calibration": payload.get("calibration"),
            }
        )

    return {
        "kind": "xas_overlay",
        "title": title,
        "mode": mode,
        "series": series,
    }


def processed_payload_path(
    xas_ref: str,
    *,
    base_dir: Path | None = None,
) -> Path:
    return processed_dir(base_dir=base_dir) / f"{artifact_key(xas_ref)}.json"


def persist_preprocessed_payload(
    xas_ref: str,
    payload: dict,
    *,
    base_dir: Path | None = None,
) -> PreprocessedXASResult:
    normalized_ref = artifact_key(xas_ref)
    output_path = processed_payload_path(normalized_ref, base_dir=base_dir)
    bundle = persist_xas_payload(
        normalized_ref,
        payload,
        remote_key=f"processed_xas/{output_path.name}",
    )

    energy = np.asarray(payload.get("energy", []), dtype=float)
    return PreprocessedXASResult(
        success=True,
        xas_ref=normalized_ref,
        source_file=payload.get("source_file"),
        e0=safe_float(payload.get("e0")),
        n_points=int(len(energy)),
        energy_min=safe_float(np.min(energy)) if energy.size else None,
        energy_max=safe_float(np.max(energy)) if energy.size else None,
        processed_path=bundle.local_path,
        s3_key=bundle.remote_key,
        error=None,
    )


def load_processed_payload(
    xas_ref: str,
    *,
    base_dir: Path | None = None,
) -> dict | None:
    path = processed_payload_path(xas_ref, base_dir=base_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def load_and_preprocess_xas_data(xas_path: str) -> PreprocessedXASResult:
    bundle = generate_xas_artifact_bundle(
        xas_path,
        include_wavelet=True,
        remote_key="processed_xas/{xas_id}.json",
    )

    energy = np.asarray(bundle.payload.get("energy", []), dtype=float)
    return PreprocessedXASResult(
        success=True,
        xas_ref=bundle.xas_id,
        source_file=bundle.payload.get("source_file"),
        e0=safe_float(bundle.payload.get("e0")),
        n_points=int(len(energy)),
        energy_min=safe_float(np.min(energy)) if energy.size else None,
        energy_max=safe_float(np.max(energy)) if energy.size else None,
        processed_path=bundle.local_path,
        s3_key=bundle.remote_key,
        error=None,
    )


def resolve_processed_payload(
    *,
    xas_path: str | None = None,
    xas_ref: str | None = None,
) -> tuple[str, dict]:
    key = artifact_key(xas_ref or xas_path or "")
    if not key:
        raise ValueError("Provide xas_path or xas_ref.")

    payload = load_processed_payload(key)
    if payload is not None:
        return key, payload

    if not xas_path:
        raise FileNotFoundError(f"No processed artifact found for xas_ref={xas_ref}.")

    preprocessed = load_and_preprocess_xas_data(xas_path=xas_path)
    if not preprocessed.success or not preprocessed.xas_ref:
        raise ValueError(preprocessed.error or "Failed to preprocess XAS data.")
    payload = load_processed_payload(preprocessed.xas_ref)
    if payload is None:
        raise FileNotFoundError(f"Processed payload missing for {preprocessed.xas_ref}.")
    return preprocessed.xas_ref, payload


def load_group_from_payload(
    payload: dict,
    *,
    name: str,
):
    energy = np.asarray(payload.get("energy", []), dtype=float)
    mu = np.asarray(payload.get("mu", []), dtype=float)
    if energy.size == 0 or mu.size == 0:
        raise ValueError("No energy/mu data available.")
    data = larch.Group(name=name, energy=energy, mu=mu)
    pre_edge(data)
    autobk(data, rbkg=1.0, kweight=2)
    xftf(data, kweight=2)
    xftr(data)
    return data


def load_fit_group(
    *,
    xas_path: str | None = None,
    xas_ref: str | None = None,
):
    if xas_ref:
        payload = load_processed_payload(xas_ref)
        if payload is None:
            raise FileNotFoundError(f"No processed artifact found for xas_ref={xas_ref}.")
        data = load_group_from_payload(payload, name=xas_ref)
        return data, payload.get("source_file") or xas_ref
    if not xas_path:
        raise ValueError("Provide xas_path or xas_ref.")
    processed_payload = load_processed_payload(xas_path)
    if processed_payload is not None:
        data = load_group_from_payload(processed_payload, name=xas_path)
        return data, processed_payload.get("source_file") or xas_path
    return load_prj_first_shell(xas_path=xas_path)
