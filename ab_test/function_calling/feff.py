from __future__ import annotations

"""Helpers for preparing and running FEFF calculations.

This module resolves CIF sources, determines the absorber element, writes FEFF
input decks, executes FEFF, and loads filtered path outputs for downstream
EXAFS fitting workflows.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional

from data_paths import feff_dir, online_cif_data_dir, user_uploaded_cif_dir
from larch.xafs.feffrunner import feffrunner

from .edge_identifier import EdgeIdentifier


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _feff_output_root() -> Path:
    return feff_dir()


def get_absorber_from_cif(
    cif_file: str,
    edge: str = "K",
    elements: Optional[Dict[str, float]] = None,
) -> str:
    """Identify the absorber element from CIF composition using edge tables."""
    if elements is None:
        from pymatgen.io.cif import CifParser

        struct = CifParser(cif_file).get_structures()[0]
        elements = struct.composition.get_el_amt_dict()

    if not elements:
        raise ValueError(f"No elements found in CIF: {cif_file}")

    identifier = EdgeIdentifier()
    edge_tables = {
        "K": identifier.k_edges,
        "L1": identifier.l1_edges,
        "L2": identifier.l2_edges,
        "L3": identifier.l3_edges,
    }

    edge_key = str(edge).strip().upper()
    if edge_key == "L":
        edge_key = "L3"
    table = edge_tables.get(edge_key, identifier.k_edges)

    candidates = [el for el in elements if el in table]
    if candidates:
        return max(candidates, key=lambda el: table[el])

    return max(elements, key=elements.get)


def make_and_run_feff(cif_file_name, radius=5.0, edge="K"):
    """Run FEFF on a single CIF file and return the output directory."""
    origin = online_cif_data_dir()

    cif_file = origin / f"{cif_file_name}.cif"
    if not cif_file.exists():
        origin = user_uploaded_cif_dir()
        cif_file = origin / f"{cif_file_name}.cif"

    output_dir = _feff_output_root() / cif_file_name
    _make_and_run_feff(str(cif_file), str(output_dir), radius=radius, edge=edge)
    return output_dir


def _resolve_feff_executable(feff_exe: Optional[str] = None) -> Optional[str]:
    """Resolve an explicit FEFF executable override if one was configured."""
    candidate = feff_exe or os.environ.get("FEFF_EXE") or os.environ.get("XAS_FEFF_EXE")
    if not candidate:
        return None

    resolved = shutil.which(candidate)
    if resolved:
        return resolved

    candidate_path = Path(candidate).expanduser()
    if candidate_path.is_file() and os.access(candidate_path, os.X_OK):
        return str(candidate_path)

    raise FileNotFoundError(
        f"Configured FEFF executable '{candidate}' could not be found or is not executable."
    )


def _run_feff(out_dir: str, inp_path: str, feff_exe: Optional[str] = None) -> None:
    """Run FEFF using an explicit override or Larch's bundled executable discovery."""
    resolved_exe = _resolve_feff_executable(feff_exe)
    if resolved_exe is not None:
        subprocess.run([resolved_exe], cwd=out_dir, check=True)
        return

    runner = feffrunner(folder=out_dir, feffinp=inp_path, verbose=False)
    result = runner.run()
    if isinstance(result, Exception):
        raise result


def _make_and_run_feff(cif_file, out_dir, radius=5.0, edge="K", feff_exe=None):
    from pymatgen.io.cif import CifParser
    from pymatgen.io.feff.sets import FEFFDictSet

    os.makedirs(out_dir, exist_ok=True)

    struct = CifParser(cif_file).get_structures()[0]
    composition = struct.composition
    elements = composition.get_el_amt_dict()
    if not elements:
        raise ValueError(f"No elements found in CIF: {cif_file}")

    absorber = get_absorber_from_cif(cif_file, edge=edge, elements=elements)
    main_element = absorber if absorber in elements else max(elements, key=elements.get)

    feff_set = FEFFDictSet(
        absorbing_atom=main_element,
        structure=struct,
        radius=radius,
        edge=edge,
        config_dict={},
        user_tag_settings={"CONTROL": {"ff2chi": 1}},
    )
    feff_set.write_input(out_dir)

    inp_path = os.path.join(out_dir, "feff.inp")
    new_lines = []
    saw_control = False

    for line in open(inp_path):
        if line.strip().startswith("CONTROL") and not saw_control:
            new_lines.append(
                "*         pot    xsph  fms   paths genfmt ff2chi\n"
                "CONTROL   1      1     1     1     1      1\n"
            )
            new_lines.append("PRINT     1      0     0     0     0      3\n")
            saw_control = True
            continue

        if not saw_control and line.strip().startswith("POTENTIALS"):
            new_lines.append(
                "*         pot    xsph  fms   paths genfmt ff2chi\n"
                "CONTROL   1      1     1     1     1      1\n"
                "PRINT     1      0     0     0     0      3\n"
            )
            saw_control = True

        new_lines.append(line)

    with open(inp_path, "w") as handle:
        handle.writelines(new_lines)

    _run_feff(out_dir, inp_path, feff_exe=feff_exe)


def load_paths(feff_dir, amp_ratio=None, r_max=None, verbose=False):
    """
    Scan a FEFF run directory, filter by amp_ratio and r_max, and return
    a dict mapping 'path<index>' to the corresponding feffNNNN.dat filepath.
    """
    try:
        list_file = os.path.join(feff_dir, "list.dat")
        if not os.path.isfile(list_file):
            raise FileNotFoundError(f"No list.dat found in {feff_dir}")

        lines = Path(list_file).read_text().splitlines()
        start = next(
            (i + 1 for i, line in enumerate(lines) if "pathindex" in line.lower()),
            None,
        )
        if start is None:
            start = next(
                (i + 1 for i, line in enumerate(lines) if line.strip().startswith("-----")),
                None,
            )
        if start is None:
            raise ValueError("Couldn't find table in list.dat")

        entries = []
        for line in lines[start:]:
            parts = line.split()
            if not parts or not parts[0].isdigit():
                continue
            idx = int(parts[0])
            amp = float(parts[2])
            deg = float(parts[3])
            nlegs = int(parts[4])
            r_eff = float(parts[5])
            entries.append((idx, amp, r_eff, deg, nlegs))

        selected = [
            (idx, amp, r_eff, deg, nlegs)
            for idx, amp, r_eff, deg, nlegs in entries
            if (amp_ratio is None or amp >= amp_ratio)
            and (r_max is None or r_eff <= r_max)
        ]

        if verbose:
            _ = (
                f"{'Path':>4}  {'Bond':<7}  {'Amp (%)':>8}  "
                f"{'R_eff (Å)':>9}  {'Deg':>4}  {'Nlegs':>5}"
            )

        paths = {}
        for idx, amp, r_eff, deg, nlegs in selected:
            fname = os.path.join(feff_dir, f"feff{idx:04d}.dat")
            if not os.path.exists(fname):
                alt = fname.replace(".dat", ".data")
                if os.path.exists(alt):
                    fname = alt
                else:
                    continue

            bond = ""
            with open(fname) as handle:
                datlines = handle.readlines()
            for j, line in enumerate(datlines):
                if "pot at#" in line.lower():
                    atom_lines = [ln for ln in datlines[j + 1 : j + 4] if ln.strip()]
                    if len(atom_lines) >= 2:
                        el0 = atom_lines[0].split()[5]
                        el1 = atom_lines[1].split()[5]
                        bond = f"{el0}-{el1}"
                    break

            if verbose:
                _ = (bond, amp, r_eff, deg, nlegs)

            paths[f"path{idx}"] = fname

        return paths
    except Exception as exc:
        print(f"load_paths: {exc}")


__all__ = [
    "_make_and_run_feff",
    "_resolve_feff_executable",
    "_run_feff",
    "get_absorber_from_cif",
    "load_paths",
    "make_and_run_feff",
]
