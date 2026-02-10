"""Load YAML configs from configs/."""
import os
from pathlib import Path
from typing import Any

import yaml

from src.utils.paths import get_config_path


def _load_yaml(name: str) -> dict[str, Any]:
    path = get_config_path() / name
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_cached: dict[str, dict[str, Any]] = {}


def load_config(name: str) -> dict[str, Any]:
    """Load a config file by name (e.g. 'app.yaml' -> app config)."""
    if name not in _cached:
        _cached[name] = _load_yaml(name)
    return _cached[name]


def get_app_config() -> dict[str, Any]:
    """App/serving config (app.yaml)."""
    return load_config("app.yaml")


def get_model_config() -> dict[str, Any]:
    """Model and feature config (model.yaml)."""
    return load_config("model.yaml")


def get_monitoring_config() -> dict[str, Any]:
    """Monitoring/drift config (monitoring.yaml)."""
    return load_config("monitoring.yaml")
