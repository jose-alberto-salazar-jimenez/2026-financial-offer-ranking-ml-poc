"""Project path helpers."""
import os
from pathlib import Path


def get_project_root() -> Path:
    """Repo root (parent of src/)."""
    return Path(__file__).resolve().parent.parent.parent


def get_data_path() -> Path:
    """data/ directory."""
    return get_project_root() / "data"


def get_artifacts_path() -> Path:
    """artifacts/ directory."""
    return get_project_root() / "artifacts"


def get_config_path() -> Path:
    """configs/ directory."""
    return get_project_root() / "configs"


def get_raw_data_dir() -> Path:
    """data/raw/."""
    return get_data_path() / "raw"


def get_processed_data_dir() -> Path:
    """data/processed/."""
    return get_data_path() / "processed"


def get_model_dir() -> Path:
    """artifacts/model/."""
    return get_artifacts_path() / "model"


def get_metrics_dir() -> Path:
    """artifacts/metrics/."""
    return get_artifacts_path() / "metrics"


def get_baselines_dir() -> Path:
    """artifacts/baselines/."""
    return get_artifacts_path() / "baselines"


def artifacts_path_from_env() -> Path:
    """Artifacts base path; use ARTIFACTS_DIR env if set."""
    root = os.environ.get("ARTIFACTS_DIR")
    if root:
        return Path(root)
    return get_artifacts_path()
