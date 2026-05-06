from __future__ import annotations

"""Reference-spectrum matching helpers for user-uploaded XAS data.

This module loads processed or raw spectra, reads the reference SQLite
database, computes similarity against normalized reference spectra, and writes
matching plot artifacts for presentation.
"""

import json
import os
import re
import sqlite3
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from data_paths import matching_dir, processed_xas_dir, xas_spectra_db_path

from .reports import plot_matches

try:
    import larch
    from larch.xafs import pre_edge

    LARCH_AVAILABLE = True
except ImportError:
    LARCH_AVAILABLE = False


CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = CURRENT_DIR.parent
PROJECT_ROOT = BACKEND_ROOT.parent
DB_PATH = xas_spectra_db_path()
LEGACY_DB_PATH = PROJECT_ROOT / "XAS_databases" / "xas_spectra.db"
DB_PATH_FALLBACK = CURRENT_DIR / "xas_spectra.db"


def _resolve_db_path() -> str:
    candidates = [
        str(DB_PATH),
        str(LEGACY_DB_PATH),
        str(DB_PATH_FALLBACK),
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return str(DB_PATH)


def _validate_db_path(path: str) -> str:
    if not path:
        raise FileNotFoundError(
            "Spectra reference database path is not configured."
        )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Spectra reference database not found at '{path}'. "
            f"Expected the checked-in database at '{DB_PATH}', or a fallback copy under "
            f"'{LEGACY_DB_PATH}' or '{CURRENT_DIR}'."
        )
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Spectra reference database path '{path}' is not a file.")
    return path


def extract_element_from_filename(filename):
    valid_elements = [
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Zr",
        "Nb",
        "Mo",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "Pt",
        "Au",
        "Hg",
        "Pb",
        "Sb",
        "Sn",
    ]

    for element in valid_elements:
        if re.search(f"[_-]{element}[_-]", filename) or re.search(f"[_-]{element}\\d", filename):
            return element

    for element in valid_elements:
        if f"_{element}_" in filename or f"-{element}-" in filename:
            return element

    return None


def load_user_spectrum(file_path):
    def _normalize_arrays(energy_arr, mu_arr):
        energy = np.asarray(energy_arr, dtype=float)
        mu_raw = np.asarray(mu_arr, dtype=float)

        valid = np.isfinite(energy) & np.isfinite(mu_raw)
        energy = energy[valid]
        mu_raw = mu_raw[valid]
        if energy.size < 6:
            return None, None, None, None

        order = np.argsort(energy)
        energy = energy[order]
        mu_raw = mu_raw[order]

        _, unique_idx = np.unique(energy, return_index=True)
        unique_idx = np.sort(unique_idx)
        energy = energy[unique_idx]
        mu_raw = mu_raw[unique_idx]

        mu_norm = None
        e0 = np.nan

        if LARCH_AVAILABLE:
            try:
                group = larch.Group(name="user_spectrum", energy=energy, mu=mu_raw)
                pre_edge(group)
                if hasattr(group, "norm"):
                    mu_norm = np.asarray(group.norm, dtype=float)
                if hasattr(group, "e0"):
                    e0 = float(group.e0)
            except Exception:
                pass

        if mu_norm is None or mu_norm.size != energy.size:
            n = max(3, min(25, energy.size // 10))
            pre_level = np.nanmedian(mu_raw[:n])
            post_level = np.nanmedian(mu_raw[-n:])
            denom = post_level - pre_level
            if not np.isfinite(denom) or abs(denom) < 1e-12:
                denom = np.nanstd(mu_raw)
            if not np.isfinite(denom) or abs(denom) < 1e-12:
                denom = 1.0
            mu_norm = (mu_raw - pre_level) / denom

        if not np.isfinite(e0):
            try:
                grad = np.gradient(mu_raw, energy)
                e0 = float(energy[np.nanargmax(grad)])
            except Exception:
                e0 = float(np.nanmedian(energy))

        return energy, mu_raw, mu_norm, e0

    origin = processed_xas_dir()
    file_str = str(file_path)
    file_obj = Path(file_str)
    candidate_keys = list(
        dict.fromkeys(
            [
                key
                for key in [file_obj.name, file_obj.stem, file_str]
                if key
            ]
        )
    )

    for key in candidate_keys:
        json_path = origin / f"{Path(key).name}.json"
        if not json_path.exists():
            continue
        try:
            data = json.loads(json_path.read_text())
            energy = np.asarray(data["energy"], dtype=float)
            mu_raw = np.asarray(data["mu"], dtype=float)
            mu_norm = np.asarray(data["norm"], dtype=float)
            e0 = float(data["e0"])
            return energy, mu_raw, mu_norm, e0
        except Exception:
            continue

    raw_path = Path(file_str)
    if raw_path.exists() and raw_path.is_file():
        try:
            from .xas_parser import XASParser

            energy, mu = XASParser().parse_file(str(raw_path))
            if energy is not None and len(energy) > 5:
                normalized = _normalize_arrays(energy, mu)
                if normalized[0] is not None:
                    return normalized
        except Exception:
            pass

    try:
        from .xas_io import load_prj_first_shell

        for key in candidate_keys:
            try:
                data, _ = load_prj_first_shell(key)
                energy = getattr(data, "energy", None)
                mu_raw = getattr(data, "mu", None)
                mu_norm = getattr(data, "norm", None)
                e0 = getattr(data, "e0", None)
                if energy is None or mu_raw is None:
                    continue
                if mu_norm is None or e0 is None:
                    normalized = _normalize_arrays(energy, mu_raw)
                    if normalized[0] is not None:
                        return normalized
                    continue
                return (
                    np.asarray(energy, dtype=float),
                    np.asarray(mu_raw, dtype=float),
                    np.asarray(mu_norm, dtype=float),
                    float(e0),
                )
            except Exception:
                continue
    except Exception:
        pass

    return None, None, None, None


def get_normalized_db_spectra(element="Ti", db_path=None):
    path = _validate_db_path(db_path or _resolve_db_path())
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, material_name, source_database, energy_json, mu_json, norm_mu_json
        FROM spectra
        WHERE element=? AND norm_mu_json IS NOT NULL
        """,
        (element,),
    )
    rows = cursor.fetchall()
    conn.close()

    spectra = []
    for row in rows:
        try:
            energy = json.loads(row["energy_json"])
            norm_mu = json.loads(row["norm_mu_json"])
            raw_mu = json.loads(row["mu_json"]) if row["mu_json"] else None
            if energy and norm_mu and len(energy) == len(norm_mu):
                spectra.append(
                    {
                        "id": row["id"],
                        "material_name": row["material_name"],
                        "source_database": row["source_database"],
                        "energy": np.array(energy),
                        "norm_mu": np.array(norm_mu),
                        "raw_mu": np.array(raw_mu) if raw_mu else np.array(norm_mu),
                    }
                )
        except Exception:
            continue
    return spectra


def match_spectra(user_energy, user_norm_mu, db_records, e0):
    results = []
    target_min = e0 - 50
    target_max = e0 + 100

    for record in db_records:
        ref_energy = record["energy"]
        ref_mu = record["norm_mu"]

        ref_min = np.min(ref_energy)
        ref_max = np.max(ref_energy)
        user_min = np.min(user_energy)
        user_max = np.max(user_energy)

        overlap_min = max(target_min, ref_min, user_min)
        overlap_max = min(target_max, ref_max, user_max)

        if overlap_max - overlap_min < 20:
            continue

        mask = (user_energy >= overlap_min) & (user_energy <= overlap_max)
        if np.sum(mask) < 10:
            continue

        common_energy = user_energy[mask]
        user_signal = user_norm_mu[mask]

        try:
            ref_interp = interp1d(ref_energy, ref_mu, kind="linear", bounds_error=False, fill_value=0)
            ref_signal = ref_interp(common_energy)
            if np.std(user_signal) == 0 or np.std(ref_signal) == 0:
                continue
            corr = np.corrcoef(user_signal, ref_signal)[0, 1]
            if not np.isnan(corr):
                results.append(
                    {
                        "id": record["id"],
                        "material_name": record["material_name"],
                        "source_database": record["source_database"],
                        "score": corr,
                        "ref_energy": record["energy"],
                        "ref_norm_mu": record["norm_mu"],
                        "ref_raw_mu": record["raw_mu"],
                    }
                )
        except Exception:
            continue

    results.sort(key=lambda item: item["score"], reverse=True)
    return results


def match_spectrum_for_agent(
    spectrum_file_path: str,
    element: str | None = None,
    top_n: int = 5,
    save_plot: bool = True,
    output_dir: str | None = None,
) -> dict:
    result = {
        "success": False,
        "element": element or "unknown",
        "e0_detected": None,
        "top_matches": [],
        "plot_path": None,
        "error": None,
    }

    try:
        file_name = os.path.basename(spectrum_file_path)
        energy, mu_raw, mu_norm, e0 = load_user_spectrum(spectrum_file_path)

        if energy is None:
            result["error"] = (
                "Failed to load spectrum. Provide a processed spectrum key, online dataset id, "
                "or a parsable raw file (CSV/TXT/XDI/DAT with energy + detector or mu columns)."
            )
            return result

        if mu_norm is None:
            result["error"] = "Spectrum normalization failed."
            return result

        if element is None:
            from .edge_identifier import EdgeIdentifier

            identifier = EdgeIdentifier()
            edge_energy = identifier.find_edge_energy(energy, mu_raw)
            matches = identifier.identify_element(edge_energy)
            if matches:
                element = matches[0]["Element"]
            else:
                element = extract_element_from_filename(file_name)

        if not element:
            result["error"] = (
                "Could not detect element from spectrum or filename. "
                "Please provide element parameter (e.g., 'Cu', 'Fe', 'Mn')."
            )
            return result

        result["element"] = element

        db_records = get_normalized_db_spectra(element, db_path=_resolve_db_path())
        if not db_records:
            result["error"] = f"No reference spectra found for element {element} in database."
            return result

        result["e0_detected"] = float(e0)
        matches = match_spectra(energy, mu_norm, db_records, e0)
        result["top_matches"] = [
            {
                "material_name": match["material_name"],
                "source_database": match["source_database"],
                "score": round(float(match["score"]), 4),
            }
            for match in matches[:top_n]
        ]
        result["success"] = True

        if save_plot and matches:
            out_dir = Path(output_dir) if output_dir is not None else matching_dir()
            out_dir.mkdir(parents=True, exist_ok=True)
            plot_filename = f"{Path(spectrum_file_path).stem}_matching.json"
            plot_path = out_dir / plot_filename
            fig = plot_matches(energy, mu_raw, mu_norm, matches, e0, file_name, limit=min(5, top_n))
            fig.write_json(str(plot_path))
            from storage import upload_file

            upload_file(plot_path, "test-dr-xas", f"matching/{plot_filename}")
            result["plot_path"] = str(plot_path)

    except Exception as exc:
        result["error"] = str(exc)

    return result
