"""Basic data quality checks: missing rate, value ranges, forbidden values."""
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.config import get_monitoring_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def run_checks(df: pd.DataFrame) -> Dict[str, Any]:
    """Run configured data quality checks; return pass/fail and summary."""
    cfg = get_monitoring_config()
    dq = cfg.get("data_quality", {})
    max_missing = dq.get("max_missing_rate", 0.05)
    age_min = dq.get("age_min", 0)
    age_max = dq.get("age_max", 120)
    balance_min = dq.get("balance_min", -10000)
    balance_max = dq.get("balance_max", 10_000_000)

    results = {"passed": True, "checks": [], "summary": []}
    n = len(df)

    missing = df.isnull().mean()
    for col in missing.index[missing > max_missing]:
        results["passed"] = False
        results["checks"].append({"check": "missing_rate", "column": col, "value": float(missing[col]), "threshold": max_missing})
        results["summary"].append(f"{col}: missing rate {missing[col]:.2%} > {max_missing:.2%}")
    if "age" in df.columns:
        out_of_range = (df["age"] < age_min) | (df["age"] > age_max)
        if out_of_range.any():
            results["passed"] = False
            results["checks"].append({"check": "age_range", "violations": int(out_of_range.sum()), "expected": [age_min, age_max]})
            results["summary"].append(f"age: {out_of_range.sum()} values outside [{age_min}, {age_max}]")
    if "balance" in df.columns:
        out_of_range = (df["balance"] < balance_min) | (df["balance"] > balance_max)
        if out_of_range.any():
            results["passed"] = False
            results["checks"].append({"check": "balance_range", "violations": int(out_of_range.sum()), "expected": [balance_min, balance_max]})
            results["summary"].append(f"balance: {out_of_range.sum()} values outside [{balance_min}, {balance_max}]")

    if not results["summary"]:
        results["summary"].append("All checks passed")
    return results
