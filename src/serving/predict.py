"""Single inference API: load artifacts and predict propensity."""
from pathlib import Path
from typing import Any, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

from src.pipelines.features import load_preprocessor, transform
from src.utils.paths import get_model_dir
from src.utils.paths import artifacts_path_from_env

_model_cache: Optional[Any] = None
_preprocessor_cache: Optional[Any] = None
_feature_names_cache: Optional[List[str]] = None


def _get_artifacts_dir() -> Path:
    return artifacts_path_from_env() / "model"


def load_model() -> Any:
    """Load model and preprocessor (cached)."""
    global _model_cache, _preprocessor_cache, _feature_names_cache
    if _model_cache is not None:
        return _model_cache, _preprocessor_cache, _feature_names_cache
    model_dir = _get_artifacts_dir()
    if not (model_dir / "model.joblib").exists():
        raise FileNotFoundError(f"Model not found at {model_dir}. Run 'make train' first.")
    _model_cache = joblib.load(model_dir / "model.joblib")
    _preprocessor_cache, _feature_names_cache = load_preprocessor(model_dir)
    return _model_cache, _preprocessor_cache, _feature_names_cache


def predict(features: Union[pd.DataFrame, dict]) -> float:
    """Return propensity score in [0, 1] for one or more rows."""
    model, preprocessor, _ = load_model()
    if isinstance(features, dict):
        df = pd.DataFrame([features])
    else:
        df = features
    X = transform(preprocessor, df)
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.atleast_2d(X)
    proba = model.predict_proba(X)[:, 1]
    return float(proba[0]) if proba.size == 1 else proba.tolist()


def predict_batch(features: pd.DataFrame) -> np.ndarray:
    """Return array of propensity scores for a DataFrame."""
    model, preprocessor, _ = load_model()
    X = transform(preprocessor, features)
    if hasattr(X, "toarray"):
        X = X.toarray()
    return model.predict_proba(X)[:, 1]
