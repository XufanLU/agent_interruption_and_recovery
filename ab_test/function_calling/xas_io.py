from __future__ import annotations

"""XAS file loading helpers for uploaded and reference spectra.

This module resolves local XAS filenames, parses supported file formats,
extracts embedded foil/reference channels when present, and normalizes the
loaded spectra into Larch groups for downstream workflows.
"""

from pathlib import Path

import larch
import numpy as np
import pandas as pd
from larch.xafs import autobk, pre_edge, xftf, xftr
from data_paths import inputs_root

from .xas_parser import XASParser


def _sanitize_path_component(name: str) -> str:
    """Replace spaces to avoid loading/path issues."""
    return name.replace(" ", "_") if " " in name else name


def _backend_data_dir(*parts: str) -> Path:
    return inputs_root().joinpath(*parts)


def _resolve_xas_filename(xas_path: str) -> Path:
    direct_path = Path(xas_path)
    if direct_path.is_file():
        return direct_path

    foldername = _backend_data_dir("user_uploaded_xas")
    xas_safe = _sanitize_path_component(xas_path)
    filenames = (
        list(foldername.glob(f"{xas_path}.txt"))
        + list(foldername.glob(f"{xas_path}.csv"))
        + list(foldername.glob(f"{xas_path}.xdi"))
        + list(foldername.glob(f"{xas_path}.dat"))
        + list(foldername.glob(f"{xas_path}.prj"))
    )
    if len(filenames) == 0 and xas_safe != xas_path:
        filenames = (
            list(foldername.glob(f"{xas_safe}.txt"))
            + list(foldername.glob(f"{xas_safe}.csv"))
            + list(foldername.glob(f"{xas_safe}.xdi"))
            + list(foldername.glob(f"{xas_safe}.dat"))
            + list(foldername.glob(f"{xas_safe}.prj"))
        )
    for path in foldername.glob(f"{xas_path}.*"):
        if path.suffix and len(path.suffix) > 1 and path.suffix[1:].isdigit():
            if path not in filenames:
                filenames.append(path)
    exact = foldername / xas_path
    if exact.is_file() and exact not in filenames:
        filenames.insert(0, exact)

    if len(filenames) == 0:
        foldername = _backend_data_dir("online_xas_data", Path(xas_path).name)
        if not foldername.exists():
            try:
                from spectrum_database import get_data_by_id

                fetched_files = get_data_by_id(Path(xas_path).name)
                if fetched_files:
                    filenames = [
                        Path(file_path)
                        for file_path in fetched_files
                        if Path(file_path).exists()
                    ]
            except Exception as exc:
                print(f"Could not fetch online XAS dataset {xas_path}: {exc}")

        if len(filenames) == 0:
            if not foldername.exists():
                raise FileNotFoundError(f"Folder {foldername} does not exist.")
            filenames = (
                list(foldername.glob("*.txt"))
                + list(foldername.glob("*.dat"))
                + list(foldername.glob("*.csv"))
                + list(foldername.glob("*.xdi"))
            )
        for path in list(filenames):
            if " " in path.name:
                safe_name = path.name.replace(" ", "_")
                dst = path.parent / safe_name
                if path != dst:
                    path.rename(dst)
                    filenames = [dst if candidate == path else candidate for candidate in filenames]

    return filenames[0]


def further_process_data(data):
    pre_edge(data)
    autobk(data, rbkg=1.0, kweight=2)
    xftf(data, kweight=2)
    xftr(data)
    return data


def load_embedded_foil_reference(xas_path: str):
    """
    Load an embedded foil/reference spectrum from the same scan when It and Iref
    channels are present. Returns (data, filename) or (None, filename) if no
    embedded reference is available.
    """
    filename = _resolve_xas_filename(xas_path)
    if filename.suffix.lower() == ".prj":
        return None, filename

    energy, mu, metadata = XASParser().parse_file_with_metadata(str(filename))
    ref_energy = metadata.get("reference_energy") if metadata else None
    ref_mu = metadata.get("reference_mu") if metadata else None
    if not ref_energy or not ref_mu:
        return None, filename

    data = larch.Group(
        name="embedded_reference",
        energy=np.array(ref_energy),
        mu=np.array(ref_mu),
    )
    further_process_data(data)
    return data, filename


def load_prj_first_shell(xas_path: str):
    """Load a project file, supporting Athena .prj and plain text/ascii formats."""
    filename = _resolve_xas_filename(xas_path)
    suffix = filename.suffix.lower()

    if suffix == ".prj":
        data = larch.io.read_athena(
            filename,
            match=None,
            do_preedge=True,
            do_bkg=True,
            do_fft=True,
            use_hashkey=False,
        )
        further_process_data(data)
        return data, filename

    if suffix == ".dat":
        energy, mu = XASParser().parse_file(str(filename))
        if energy is not None and len(energy) > 5:
            data = larch.Group(name="xas_data", energy=np.array(energy), mu=np.array(mu))
        else:
            data = larch.io.read_ascii(
                filename,
                labels=("ang_c", "ang_o", "time", "i0", "itrans"),
            )
            hc = 12398.42
            d = 1.63747
            theta = np.radians(data.ang_c)
            energy = hc / (2 * d * np.sin(theta))

            data.energy = energy
            data.mu = -np.log(data.itrans / data.i0)
        further_process_data(data)
        return data, filename

    if suffix == ".txt":
        energy, mu = XASParser().parse_file(str(filename))
        if energy is not None and len(energy) > 5:
            data = larch.Group(name="xas_data", energy=np.array(energy), mu=np.array(mu))
            further_process_data(data)
        else:
            data = larch.io.read_ascii(filename, labels=("energy", "mu"))
            further_process_data(data)
        return data, filename

    if suffix == ".csv":
        energy, mu = XASParser().parse_file(str(filename))
        if energy is not None and len(energy) > 5:
            data = larch.Group(name="xas_data", energy=np.array(energy), mu=np.array(mu))
            further_process_data(data)
        else:
            df = pd.read_csv(filename)
            energy = df["energy"].values.astype(float)
            if "IpreKB" in df.columns and "fluo_DT_corrected" in df.columns:
                i_0 = df["IpreKB"].values.copy().astype(float)
                i_0[i_0 < 1.0] = 1.0
                mu = df["fluo_DT_corrected"].values.astype(float) / i_0
            else:
                mu = df["mu"].values.astype(float)
            data = larch.Group(name="xas_data", energy=energy, mu=mu)
            further_process_data(data)
        return data, filename

    if suffix == ".xdi":
        energy, mu = XASParser().parse_file(str(filename))
        if energy is not None and len(energy) > 5:
            data = larch.Group(name="xas_data", energy=np.array(energy), mu=np.array(mu))
            further_process_data(data)
        else:
            data = larch.io.read_xdi(filename)
            further_process_data(data)
        return data, filename

    energy, mu = XASParser().parse_file(str(filename))
    if energy is not None and len(energy) > 5:
        data = larch.Group(name="xas_data", energy=np.array(energy), mu=np.array(mu))
        further_process_data(data)
        return data, filename

    raise ValueError(
        "Unsupported file format. Please provide a .prj, .dat, .txt, .csv, or .xdi file."
    )


__all__ = [
    "_backend_data_dir",
    "_resolve_xas_filename",
    "_sanitize_path_component",
    "further_process_data",
    "load_embedded_foil_reference",
    "load_prj_first_shell",
]
