"""Feature engineering: encode categoricals, scale numericals."""
from pathlib import Path
from typing import Any, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.config import get_model_config
from src.utils.logging import get_logger
from src.utils.paths import get_model_dir

logger = get_logger(__name__)


def get_feature_columns() -> Tuple[List[str], List[str]]:
    """Return (numerical_cols, categorical_cols) from config."""
    cfg = get_model_config()
    fc = cfg.get("feature_columns", {})
    num = fc.get("numerical", [])
    cat = fc.get("categorical", [])
    return num, cat


def build_preprocessor(
    df: pd.DataFrame,
    numerical: Optional[List[str]] = None,
    categorical: Optional[List[str]] = None,
) -> Tuple[ColumnTransformer, List[str]]:
    """Build and fit a ColumnTransformer; return transformer and feature names out."""
    num_cols, cat_cols = get_feature_columns()
    if numerical is not None:
        num_cols = [c for c in numerical if c in df.columns]
    if categorical is not None:
        cat_cols = [c for c in categorical if c in df.columns]
    num_cols = [c for c in num_cols if c in df.columns]
    cat_cols = [c for c in cat_cols if c in df.columns]

    transformer = ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )
    X = df[num_cols + cat_cols]
    transformer.fit(X)
    # Get output feature names
    num_names = num_cols
    cat_enc = transformer.named_transformers_["cat"]
    cat_names = cat_enc.get_feature_names_out(cat_cols).tolist()
    feature_names = num_names + cat_names
    logger.info("Preprocessor fitted: %d features", len(feature_names))
    return transformer, feature_names


def transform(
    preprocessor: ColumnTransformer,
    df: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
) -> np.ndarray:
    """Transform raw DataFrame to model input matrix."""
    num_cols, cat_cols = get_feature_columns()
    num_cols = [c for c in num_cols if c in df.columns]
    cat_cols = [c for c in cat_cols if c in df.columns]
    X = df[num_cols + cat_cols]
    return preprocessor.transform(X)


def save_preprocessor(
    preprocessor: ColumnTransformer,
    feature_names: List[str],
    path: Optional[Path] = None,
) -> None:
    """Save preprocessor and feature list."""
    path = path or get_model_dir()
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path / "preprocessor.joblib")
    joblib.dump(feature_names, path / "feature_names.joblib")
    logger.info("Saved preprocessor to %s", path)


def load_preprocessor(path: Optional[Path] = None) -> Tuple[Any, List[str]]:
    """Load preprocessor and feature names."""
    path = path or get_model_dir()
    preprocessor = joblib.load(path / "preprocessor.joblib")
    feature_names = joblib.load(path / "feature_names.joblib")
    return preprocessor, feature_names
