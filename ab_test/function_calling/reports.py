from __future__ import annotations

"""Plot and preview builders for XAS workflow outputs.

This module generates plot artifacts, overlay payloads, and preview/report
groups that the API and frontend use to present preprocessing, fitting, and
matching results.
"""

import json
from pathlib import Path
from typing import List

import numpy as np
import plotly.graph_objects as go
from larch.plot.plotly_xafsplots import plot_chifit
from data_paths import ensure_dir, project_relative_path, viz_dir
from .artifacts import artifact_key, build_overlay_payload, persist_xas_payload
from storage import upload_file
from plotly.subplots import make_subplots

from .artifacts import plot_dir, resolve_processed_payload
from .models import PlotArtifact, PlotResult, PreviewGroupResult, PreviewReportItem, PreprocessedXASResult, Report


##### Shared Plot Helpers #####


def build_line_plot(x, y, title: str, x_label: str, y_label: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines"))
    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=450,
    )
    return fig


def upload_if_possible(local_path: Path, remote_path: str) -> None:
    try:
        upload_file(local_path, "test-dr-xas", remote_path)
    except Exception:
        pass


def readable_plot_title(name: str) -> str:
    return {
        "mu": "mu(E)",
        "norm": "Normalized mu(E)",
        "chi_k": "chi(k)",
        "chi_r": "chi(R)",
    }.get(name, name)


##### Preprocessing Plots #####


def plot_preprocessed_xas_data(
    *,
    xas_path: str | None = None,
    xas_ref: str | None = None,
) -> PlotResult:
    resolved_ref, payload = resolve_processed_payload(xas_path=xas_path, xas_ref=xas_ref)
    plots: List[PlotArtifact] = []
    energy = payload.get("energy", [])
    mu = payload.get("mu", [])
    norm = payload.get("norm", [])
    k = payload.get("k", [])
    chi = payload.get("chi", [])
    r = payload.get("r", [])
    chir_mag = payload.get("chir_mag", [])

    plot_specs = []
    if energy and mu:
        plot_specs.append(("mu", build_line_plot(energy, mu, "Raw XAS", "Energy (eV)", "mu")))
    if energy and norm:
        plot_specs.append(
            ("norm", build_line_plot(energy, norm, "Normalized XAS", "Energy (eV)", "Normalized mu"))
        )
    if k and chi:
        plot_specs.append(("chi_k", build_line_plot(k, chi, "chi(k)", "k", "chi(k)")))
    if r and chir_mag:
        plot_specs.append(("chi_r", build_line_plot(r, chir_mag, "chi(R)", "R", "|chi(R)|")))

    if not plot_specs:
        raise ValueError("No plottable preprocessing data found.")

    out_dir = plot_dir("preprocessed")
    for plot_name, fig in plot_specs:
        plot_path = out_dir / f"{resolved_ref}_{plot_name}.json"
        fig.write_json(str(plot_path))
        s3_key = f"viz/preprocessed/{plot_path.name}"
        upload_if_possible(plot_path, s3_key)
        plots.append(
            PlotArtifact(
                name=plot_name,
                path=project_relative_path(plot_path) or str(plot_path),
                s3_key=s3_key,
            )
        )

    return PlotResult(success=True, xas_ref=resolved_ref, plots=plots, error=None)


def build_preview_group_from_preprocessing(
    preprocessed: PreprocessedXASResult,
    plots: PlotResult,
) -> PreviewGroupResult:
    xas_ref = preprocessed.xas_ref or ""
    source_label = Path(str(preprocessed.source_file or xas_ref)).stem or xas_ref or "Spectrum"
    preprocessing_plot_urls: List[Dict[str, str]] = []
    report_items: List[PreviewReportItem] = []

    for plot in plots.plots:
        if not plot.s3_key:
            continue
        preprocessing_plot_urls.append(
            {
                "name": plot.name,
                "url": plot.s3_key,
            }
        )
        if plot.name in {"chi_k", "chi_r"}:
            report_items.append(
                PreviewReportItem(
                    kind="chi_fit",
                    title=readable_plot_title(plot.name),
                    source=plot.s3_key,
                )
            )

    summary = f"Prepared the report for {source_label}."
    if preprocessed.e0 is not None:
        summary += f" Estimated E0 is {float(preprocessed.e0):.2f} eV."

    return PreviewGroupResult(
        id=artifact_key(xas_ref or source_label),
        title=source_label,
        message=summary,
        xas_url=preprocessed.s3_key,
        preprocessing_plot_urls=preprocessing_plot_urls,
        report_items=report_items,
        is_fitting=False,
    )


##### Multi-Spectrum Overlay Plot #####


def build_overlay_preview_group(
    preprocessed_results: List[PreprocessedXASResult],
) -> PreviewGroupResult | None:
    if len(preprocessed_results) < 2:
        return None

    overlay_payloads: List[dict] = []
    labels: List[str] = []
    for preprocessed in preprocessed_results:
        if not preprocessed.xas_ref:
            continue
        _, payload = resolve_processed_payload(xas_ref=preprocessed.xas_ref)
        overlay_payloads.append(payload)
        labels.append(Path(str(preprocessed.source_file or preprocessed.xas_ref)).stem or preprocessed.xas_ref)

    if len(overlay_payloads) < 2:
        return None

    overlay_id = artifact_key("__".join(labels[:4]) or "overlay")
    overlay_title = "Normalized μ(E) Overlay"
    bundle = persist_xas_payload(
        f"{overlay_id}_overlay",
        build_overlay_payload(overlay_payloads, mode="norm", title=overlay_title),
        remote_key="processed_xas/{xas_id}.json",
    )

    return PreviewGroupResult(
        id=f"{overlay_id}_comparison",
        title="Comparison",
        message=(
            f"Prepared a shared overlay for {len(overlay_payloads)} spectra. "
            "Open the overlay view to compare them on a single axis."
        ),
        report_items=[
            PreviewReportItem(
                kind="xas_overlay",
                title=overlay_title,
                source=bundle.remote_key,
            )
        ],
        is_fitting=False,
    )


##### Spectra Matching Plot #####


def plot_matches(user_energy, user_raw_mu, user_norm_mu, match_records, e0, filename, limit=3):
    top_matches = match_records[:limit]
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Normalized Data", "Raw Data (Scaled)"))

    fig.add_trace(
        go.Scatter(
            x=user_energy,
            y=user_norm_mu,
            mode="lines",
            name="User Sample",
            line=dict(color="black", width=2.5),
            legendgroup="user",
        ),
        row=1,
        col=1,
    )

    colors = ["#d62728", "#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd"]

    for index, match in enumerate(top_matches):
        color = colors[index % len(colors)]
        label = f"{match['material_name']} ({match['source_database']}) R={match['score']:.4f}"
        fig.add_trace(
            go.Scatter(
                x=match["ref_energy"],
                y=match["ref_norm_mu"],
                mode="lines",
                name=label,
                line=dict(color=color),
                opacity=0.7,
                legendgroup=f"match_{index}",
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=user_energy,
            y=user_raw_mu,
            mode="lines",
            name="User Sample (Raw)",
            line=dict(color="black", width=2.5),
            legendgroup="user",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    user_min = np.min(user_raw_mu)
    user_max = np.max(user_raw_mu)
    user_amp = user_max - user_min if (user_max - user_min) > 0 else 1.0

    for index, match in enumerate(top_matches):
        color = colors[index % len(colors)]
        current_raw = match["ref_raw_mu"]
        if len(current_raw) == 0:
            continue

        current_min = np.min(current_raw)
        current_amp = np.max(current_raw) - current_min
        if current_amp > 0:
            current_projected = (current_raw - current_min) / current_amp * user_amp + user_min
        else:
            current_projected = current_raw

        fig.add_trace(
            go.Scatter(
                x=match["ref_energy"],
                y=current_projected,
                mode="lines",
                name=match["material_name"],
                line=dict(color=color),
                opacity=0.7,
                legendgroup=f"match_{index}",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    if e0:
        x_range = [e0 - 50, e0 + 200]
        fig.update_xaxes(range=x_range, row=1, col=1)
        fig.update_xaxes(range=x_range, row=2, col=1)

    fig.update_layout(
        height=800,
        autosize=True,
        template="plotly_white",
        hovermode="x unified",
        font=dict(family="Avenir, Avenir Next, Helvetica Neue, Helvetica, Arial, sans-serif"),
        legend=dict(
            font=dict(size=9, family="Avenir, Avenir Next, Helvetica Neue, Helvetica, Arial, sans-serif"),
            orientation="h",
            yanchor="top",
            y=-0.12,
            xanchor="center",
            x=0.5,
        ),
    )

    fig.update_xaxes(title_text="Energy (eV)", row=1, col=1)
    fig.update_yaxes(title_text="Normalized Absorption", range=[-0.1, 1.5], row=1, col=1)
    fig.update_xaxes(title_text="Energy (eV)", row=2, col=1)
    fig.update_yaxes(title_text="Raw Absorption (fluo_DT_corr / I0)", row=2, col=1)
    return fig


##### First-Shell Fit Plots #####


def serialize_fit_report_message(source_label: str, report: Report) -> str:
    fitted_parameter = (
        report.fitted_parameter.model_dump() if report.fitted_parameter is not None else {}
    )
    path_parameter = [
        item.model_dump() for item in (report.path_parameter or [])
    ]
    explanation = (
        f"Completed first-shell EXAFS fitting for {source_label}. "
        "Use the report switcher on the right to compare spectra."
    )
    return json.dumps(
        {
            "fitted_parameter": fitted_parameter,
            "path_parameter": path_parameter,
            "explanation": explanation,
        }
    )


def build_preview_group_from_fit(
    *,
    material_id: str,
    xas_path: str,
    preprocessed: PreprocessedXASResult,
    report: Report,
    cwd: Path | None = None,
) -> PreviewGroupResult:
    plot_root = cwd or viz_dir()
    xas_ref = preprocessed.xas_ref or artifact_key(xas_path)
    source_label = Path(str(preprocessed.source_file or xas_path)).stem or xas_ref or "Spectrum"
    xas_url = preprocessed.s3_key or f"processed_xas/{xas_ref}.json"
    fit_key = artifact_key(xas_path or xas_ref or source_label)
    chifit1_local = plot_root / f"{fit_key}_chifit1.json"
    chifit2_local = plot_root / f"{fit_key}_chifit2.json"

    return PreviewGroupResult(
        id=xas_ref,
        title=source_label,
        message=serialize_fit_report_message(source_label, report),
        material_url=f"{material_id}.cif" if material_id else None,
        xas_url=xas_url,
        fitting_result_url=f"fit/{fit_key}",
        chifit_result1_url=f"viz/{fit_key}_chifit1.json" if chifit1_local.exists() else None,
        chifit_result2_url=f"viz/{fit_key}_chifit2.json" if chifit2_local.exists() else None,
        is_fitting=True,
    )


def viz_first_shell(
    path_list,
    result,
    *,
    xas_path: str | None = None,
    xas_ref: str | None = None,
):
    plot_id = artifact_key(xas_path or xas_ref or "fit_result")
    fig6, fig7 = plot_chifit(result.datasets[0])
    fig6_json = json.dumps(fig6.fig.to_dict())
    fig7_json = json.dumps(fig7.fig.to_dict())

    out_dir = ensure_dir(viz_dir())

    json_path1 = out_dir / f"{plot_id}_chifit1.json"
    json_path2 = out_dir / f"{plot_id}_chifit2.json"
    json_path1.write_text(fig6_json)
    json_path2.write_text(fig7_json)

    upload_if_possible(json_path1, f"viz/{plot_id}_chifit1.json")
    upload_if_possible(json_path2, f"viz/{plot_id}_chifit2.json")
