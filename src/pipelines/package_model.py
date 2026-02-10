"""Package model artifacts and baseline stats for drift detection."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.utils.config import get_monitoring_config
from src.utils.logging import get_logger
from src.utils.paths import get_artifacts_path, get_baselines_dir, get_model_dir, get_processed_data_dir

logger = get_logger(__name__)


def _compute_baseline_stats(df: pd.DataFrame, scores: np.ndarray) -> dict:
    """Compute feature and score stats for drift baseline."""
    cfg = get_monitoring_config()
    monitor_cols = cfg.get("drift", {}).get(
        "monitor_features", ["age", "balance", "duration", "campaign", "pdays"]
    )
    monitor_cols = [c for c in monitor_cols if c in df.columns]
    feature_stats = {}
    for c in monitor_cols:
        feature_stats[c] = {
            "mean": float(df[c].mean()),
            "std": float(df[c].std()),
            "min": float(df[c].min()),
            "max": float(df[c].max()),
        }
    out = {
        "feature_stats": feature_stats,
        "score_mean": float(np.mean(scores)) if len(scores) else 0.0,
        "score_std": float(np.std(scores)) if len(scores) else 0.0,
    }
    return out


def main() -> None:
    model_dir = get_model_dir()
    baselines_dir = get_baselines_dir()
    proc_dir = get_processed_data_dir()
    baselines_dir.mkdir(parents=True, exist_ok=True)

    model = joblib.load(model_dir / "model.joblib")
    X_train_path = proc_dir / "X_train.npy"
    if X_train_path.exists():
        X_train = np.load(X_train_path)
        train_scores = model.predict_proba(X_train)[:, 1]
    else:
        train_scores = np.array([])

    raw_path = proc_dir / "train_raw.parquet"
    if raw_path.exists():
        df = pd.read_parquet(raw_path)
        baseline = _compute_baseline_stats(df, train_scores)
    else:
        baseline = {
            "score_mean": float(np.mean(train_scores)) if len(train_scores) else 0.0,
            "score_std": float(np.std(train_scores)) if len(train_scores) else 0.0,
            "feature_stats": {},
        }

    baseline_path = baselines_dir / "baseline_stats.json"
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2)
    logger.info("Baseline stats written to %s", baseline_path)

    try:
        from src.governance.model_card import generate_model_card
        card_path = get_artifacts_path() / "model_card.md"
        generate_model_card(card_path)
        logger.info("Model card written to %s", card_path)
    except Exception as e:
        logger.warning("Could not generate model card: %s", e)
