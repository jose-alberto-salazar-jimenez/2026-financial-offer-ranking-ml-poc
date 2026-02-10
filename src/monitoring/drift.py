"""PSI and KS drift checks vs baseline."""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.config import get_monitoring_config
from src.utils.logging import get_logger
from src.utils.paths import get_baselines_dir

logger = get_logger(__name__)


def _psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index between two 1d arrays."""
    breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1)[1:-1])
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        breakpoints = np.percentile(np.concatenate([expected, actual]), np.linspace(0, 100, n_bins + 1)[1:-1])
    bins = np.clip(np.searchsorted(breakpoints, expected, side="right"), 0, len(breakpoints))
    bin_expected = np.bincount(bins, minlength=len(breakpoints) + 1) / len(expected)
    bins_actual = np.clip(np.searchsorted(breakpoints, actual, side="right"), 0, len(breakpoints))
    bin_actual = np.bincount(bins_actual, minlength=len(breakpoints) + 1) / len(actual)
    bin_expected = np.where(bin_expected == 0, 1e-6, bin_expected)
    bin_actual = np.where(bin_actual == 0, 1e-6, bin_actual)
    return float(np.sum((bin_actual - bin_expected) * np.log(bin_actual / bin_expected)))


def _ks(expected: np.ndarray, actual: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic (max CDF difference)."""
    from scipy import stats
    return float(stats.ks_2samp(expected, actual).statistic)


def compute_baseline_stats(df: pd.DataFrame, scores: Optional[np.ndarray] = None) -> dict:
    """Compute feature and score stats for drift baseline (used by package_model)."""
    cfg = get_monitoring_config()
    monitor_cols = cfg.get("drift", {}).get("monitor_features", ["age", "balance", "duration", "campaign", "pdays"])
    monitor_cols = [c for c in monitor_cols if c in df.columns]
    feature_stats = {}
    for c in monitor_cols:
        feature_stats[c] = {
            "mean": float(df[c].mean()),
            "std": float(df[c].std()),
            "min": float(df[c].min()),
            "max": float(df[c].max()),
        }
    out = {"feature_stats": feature_stats}
    if scores is not None and len(scores):
        out["score_mean"] = float(np.mean(scores))
        out["score_std"] = float(np.std(scores))
    else:
        out["score_mean"] = 0.0
        out["score_std"] = 0.0
    return out


def load_baseline(path: Optional[Path] = None) -> dict:
    """Load baseline_stats.json."""
    path = path or get_baselines_dir() / "baseline_stats.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compute_drift(
    current_df: pd.DataFrame,
    current_scores: Optional[np.ndarray] = None,
    baseline_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Compare current data/scores to baseline; return metrics and drift_detected."""
    baseline = load_baseline(baseline_path)
    cfg = get_monitoring_config()
    drift_cfg = cfg.get("drift", {})
    psi_threshold = drift_cfg.get("psi_threshold", 0.2)
    ks_threshold = drift_cfg.get("ks_threshold", 0.1)
    score_psi_threshold = drift_cfg.get("score_psi_threshold", 0.15)
    monitor_cols = drift_cfg.get("monitor_features", ["age", "balance", "duration", "campaign", "pdays"])
    monitor_cols = [c for c in monitor_cols if c in current_df.columns]

    results = {"feature_psi": {}, "feature_ks": {}, "score_psi": None, "drift_detected": False}
    try:
        from scipy import stats
        has_scipy = True
    except ImportError:
        has_scipy = False

    for col in monitor_cols:
        current = current_df[col].dropna().values
        if not len(current):
            continue
        fs = baseline.get("feature_stats", {}).get(col, {})
        base_mean = fs.get("mean")
        base_std = fs.get("std")
        if base_mean is None or base_std is None:
            continue
        # Simulate baseline sample from normal for PSI (we don't have raw baseline sample)
        np.random.seed(42)
        n = min(len(current), 5000)
        expected = np.random.normal(base_mean, max(base_std, 1e-6), n)
        psi = _psi(expected, current)
        results["feature_psi"][col] = psi
        if has_scipy:
            ks = _ks(expected, current)
            results["feature_ks"][col] = ks
            if ks > ks_threshold:
                results["drift_detected"] = True
        if psi > psi_threshold:
            results["drift_detected"] = True

    if current_scores is not None and len(current_scores) and "score_mean" in baseline:
        base_mean = baseline["score_mean"]
        base_std = max(baseline.get("score_std", 0), 1e-6)
        expected = np.random.normal(base_mean, base_std, min(len(current_scores), 5000))
        score_psi = _psi(expected, current_scores)
        results["score_psi"] = score_psi
        if score_psi > score_psi_threshold:
            results["drift_detected"] = True

    return results


def main() -> None:
    """CLI: run drift on recent data (e.g. test set)."""
    from src.utils.paths import get_processed_data_dir
    import joblib
    from src.utils.paths import get_model_dir
    from src.pipelines.features import load_preprocessor, transform

    proc_dir = get_processed_data_dir()
    model_dir = get_model_dir()
    if not (proc_dir / "train_raw.parquet").exists() or not (model_dir / "model.joblib").exists():
        logger.warning("No training data or model; skipping drift check")
        return
    df = pd.read_parquet(proc_dir / "train_raw.parquet")
    # Use last 20% as "current" for demo
    n = len(df)
    current_df = df.iloc[-int(n * 0.2) :]
    model = joblib.load(model_dir / "model.joblib")
    preprocessor, _ = load_preprocessor(model_dir)
    X = transform(preprocessor, current_df)
    if hasattr(X, "toarray"):
        X = X.toarray()
    current_scores = model.predict_proba(X)[:, 1]
    results = compute_drift(current_df, current_scores)
    logger.info("Drift results: %s", results)
    from src.utils.paths import get_metrics_dir
    report_path = get_metrics_dir() / "drift_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    if results["drift_detected"]:
        logger.warning("Drift detected above threshold")


if __name__ == "__main__":
    main()
