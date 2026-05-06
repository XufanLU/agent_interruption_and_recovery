from __future__ import annotations

from pathlib import Path


AB_TEST_ROOT = Path(__file__).resolve().parent
DATA_DIRNAME = "data"


def backend_root(root_dir: str | Path | None = None) -> Path:
    return Path(root_dir) if root_dir is not None else AB_TEST_ROOT


def project_root(root_dir: str | Path | None = None) -> Path:
    return backend_root(root_dir)


def data_root(root_dir: str | Path | None = None) -> Path:
    return backend_root(root_dir) / DATA_DIRNAME


def inputs_root(root_dir: str | Path | None = None) -> Path:
    return data_root(root_dir) / "inputs"


def artifacts_root(root_dir: str | Path | None = None) -> Path:
    return data_root(root_dir) / "artifacts"


def state_root(root_dir: str | Path | None = None) -> Path:
    return data_root(root_dir) / "state"


def ops_root(root_dir: str | Path | None = None) -> Path:
    return data_root(root_dir) / "ops"


def user_uploaded_xas_dir(root_dir: str | Path | None = None) -> Path:
    return inputs_root(root_dir) / "user_uploaded_xas"


def user_uploaded_cif_dir(root_dir: str | Path | None = None) -> Path:
    return inputs_root(root_dir) / "user_uploaded_cif"


def online_xas_data_dir(root_dir: str | Path | None = None) -> Path:
    return inputs_root(root_dir) / "online_xas_data"


def online_cif_data_dir(root_dir: str | Path | None = None) -> Path:
    return AB_TEST_ROOT / "fixtures"


def xas_spectra_db_path(root_dir: str | Path | None = None) -> Path:
    return inputs_root(root_dir) / "xas_spectra.db"


def processed_xas_dir(root_dir: str | Path | None = None) -> Path:
    return artifacts_root(root_dir) / "processed_xas"


def viz_dir(root_dir: str | Path | None = None) -> Path:
    return artifacts_root(root_dir) / "viz"


def viz_kind_dir(kind: str, root_dir: str | Path | None = None) -> Path:
    return viz_dir(root_dir) / kind


def plot_dir(root_dir: str | Path | None = None) -> Path:
    return artifacts_root(root_dir) / "plots"


def matching_dir(root_dir: str | Path | None = None) -> Path:
    return artifacts_root(root_dir) / "matching"


def feff_dir(root_dir: str | Path | None = None) -> Path:
    return artifacts_root(root_dir) / "feff"


def conversation_context_dir(root_dir: str | Path | None = None) -> Path:
    return state_root(root_dir) / "conversation_context"


def activity_history_dir(root_dir: str | Path | None = None) -> Path:
    return state_root(root_dir) / "activity_history"


def checkpoints_dir(root_dir: str | Path | None = None) -> Path:
    return state_root(root_dir) / "checkpoints"


def bug_reports_dir(root_dir: str | Path | None = None) -> Path:
    return ops_root(root_dir) / "bug_reports"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def project_relative_path(path: str | Path | None, root_dir: str | Path | None = None) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(backend_root(root_dir).resolve()))
    except Exception:
        return str(candidate)
