from __future__ import annotations

import os

from agents import Agent

from function_calling import (
    fit_ffef_first_shell,
    load_and_preprocess_xas,
    lookup_material_structure,
    plot_preprocessed_xas,
    prepare_feff_paths,
)


def _first_shell_agent_instructions(
    params: dict,
    material_id: str | None,
    xas_path: str | None,
    xas_ref: str | None = None,
) -> str:
    """Minimal first-shell fitting agent instructions."""
    structure_context = material_id if material_id else "<missing>"
    xas_context = xas_path if xas_path else "<missing>"
    xas_ref_line = f"- current processed XAS artifact ref: {xas_ref}\n" if xas_ref else ""
    return f"""
You are a helpful assistant for XAS analysis.

Runtime context:
- default first-shell fitting parameters: {params}
- current structure context for FEFF/path-related tasks: {structure_context}
- current XAS path for spectrum-related tasks: {xas_context}
{xas_ref_line}

- If the user asks for a first-shell EXAFS fit and both structure and XAS context are available, call `fit_ffef_first_shell` directly.
- Use the runtime material_id and xas_path/xas_ref without asking the user to upload files again.
- Keep the response concise.
- Do not expose raw local filesystem paths in the user-facing answer.
"""


async def create_first_shell_agent(
    material_id: str | None = None,
    xas_path: str | None = None,
    xas_ref: str | None = None,
) -> Agent:
    """Create the minimal first-shell agent required by the live test."""
    params = {
        "amp": 0.8,
        "e0": 0.0,
        "sigma2": 0.003,
        "deltar": 0,
    }

    agent_kwargs = {
        "name": "First-Shell Assistant",
        "instructions": _first_shell_agent_instructions(
            params=params,
            material_id=material_id,
            xas_path=xas_path,
            xas_ref=xas_ref,
        ),
        "tools": [
            lookup_material_structure,
            fit_ffef_first_shell,
            load_and_preprocess_xas,
            plot_preprocessed_xas,
            prepare_feff_paths,
        ],
    }
    model_name = os.getenv("AB_TEST_MODEL") or os.getenv("OPENAI_MODEL")
    if model_name:
        agent_kwargs["model"] = model_name

    return Agent(**agent_kwargs)
