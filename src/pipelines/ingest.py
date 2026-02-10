"""Load and validate raw UCI Bank Marketing data."""
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.config import get_model_config
from src.utils.logging import get_logger
from src.utils.paths import get_raw_data_dir

logger = get_logger(__name__)

# Expected columns for bank-additional-full.csv
EXPECTED_COLUMNS = [
    "age", "job", "marital", "education", "default", "balance",
    "housing", "loan", "contact", "day", "month", "duration",
    "campaign", "pdays", "previous", "poutcome", "y",
]


def get_raw_csv_path() -> Path:
    """Path to raw CSV (bank-additional-full.csv)."""
    return get_raw_data_dir() / "bank-additional-full.csv"


def load_raw(
    path: Optional[Path] = None,
    validate: bool = True,
) -> pd.DataFrame:
    """Load raw CSV and optionally validate schema."""
    p = path or get_raw_csv_path()
    if not p.exists():
        raise FileNotFoundError(f"Raw data not found: {p}. Run 'make data' first.")
    df = pd.read_csv(p, sep=";", encoding="utf-8")
    logger.info("Loaded %s rows from %s", len(df), p)
    if validate:
        _validate_schema(df)
    return df


def _validate_schema(df: pd.DataFrame) -> None:
    """Check required columns and basic dtypes/ranges."""
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    # Basic range check for age
    if "age" in df.columns:
        if (df["age"] < 0).any() or (df["age"] > 120).any():
            logger.warning("Some 'age' values outside [0, 120]")
    logger.info("Schema validation passed")
