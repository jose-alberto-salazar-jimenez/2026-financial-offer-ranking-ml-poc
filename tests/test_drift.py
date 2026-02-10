"""Test drift detection: no drift when identical, drift when shifted."""
import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift import compute_drift, _psi, load_baseline


def test_psi_identical():
    x = np.random.RandomState(42).randn(500)
    assert _psi(x, x) < 0.01


def test_psi_shifted():
    x = np.random.RandomState(42).randn(500)
    y = x + 2  # clear shift
    assert _psi(x, y) > 0.2


def test_compute_drift_no_baseline():
    df = pd.DataFrame({"age": [30, 40], "balance": [100, 200], "duration": [100, 200], "campaign": [1, 2], "pdays": [-1, 5]})
    result = compute_drift(df, current_scores=np.array([0.3, 0.5]), baseline_path=None)
    # With no baseline file, we get empty feature_psi or no drift
    assert "drift_detected" in result


def test_compute_drift_with_mock_baseline():
    import json
    from pathlib import Path
    base = Path(__file__).resolve().parent
    tmp = base / "_drift_test_tmp"
    tmp.mkdir(exist_ok=True)
    baseline_path = tmp / "baseline_stats.json"
    try:
        baseline = {
            "feature_stats": {
                "age": {"mean": 35, "std": 10, "min": 18, "max": 90},
                "balance": {"mean": 500, "std": 1000, "min": -1000, "max": 50000},
                "duration": {"mean": 250, "std": 100, "min": 0, "max": 1000},
                "campaign": {"mean": 2, "std": 2, "min": 1, "max": 10},
                "pdays": {"mean": 0, "std": 100, "min": -1, "max": 500},
            },
            "score_mean": 0.1,
            "score_std": 0.05,
        }
        baseline_path.write_text(json.dumps(baseline))
        df = pd.DataFrame({
            "age": np.random.RandomState(1).normal(35, 10, 200),
            "balance": np.random.RandomState(2).normal(500, 1000, 200),
            "duration": np.random.RandomState(3).normal(250, 100, 200),
            "campaign": np.random.RandomState(4).randint(1, 5, 200),
            "pdays": np.random.RandomState(5).choice([-1, 0, 50], 200),
        })
        result = compute_drift(df, current_scores=np.random.RandomState(6).uniform(0.05, 0.2, 200), baseline_path=baseline_path)
        assert "feature_psi" in result or "score_psi" in result
        assert "drift_detected" in result
    finally:
        try:
            if baseline_path.exists():
                baseline_path.unlink()
            if tmp.exists():
                tmp.rmdir()
        except OSError:
            pass
