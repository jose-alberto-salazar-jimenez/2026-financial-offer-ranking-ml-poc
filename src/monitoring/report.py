"""Consume drift + data quality results; output markdown or JSON for Actions."""
import json
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.paths import get_baselines_dir, get_metrics_dir


def write_drift_report(drift_results: Dict[str, Any], output_path: Optional[Path] = None) -> str:
    """Format drift results as markdown; optionally write to path. Returns markdown string."""
    path = output_path or get_metrics_dir() / "drift_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Drift Report", ""]
    lines.append(f"**Drift detected:** {drift_results.get('drift_detected', False)}")
    lines.append("")
    if drift_results.get("feature_psi"):
        lines.append("## Feature PSI")
        lines.append("| Feature | PSI |")
        lines.append("|---------|-----|")
        for col, psi in drift_results["feature_psi"].items():
            lines.append(f"| {col} | {psi:.4f} |")
        lines.append("")
    if drift_results.get("score_psi") is not None:
        lines.append(f"## Score PSI: {drift_results['score_psi']:.4f}")
        lines.append("")
    md = "\n".join(lines)
    path.write_text(md, encoding="utf-8")
    return md


def write_quality_report(quality_results: Dict[str, Any], output_path: Optional[Path] = None) -> str:
    """Format data quality results as markdown. Returns markdown string."""
    path = output_path or get_metrics_dir() / "quality_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Data Quality Report", ""]
    lines.append(f"**Passed:** {quality_results.get('passed', False)}")
    lines.append("")
    for s in quality_results.get("summary", []):
        lines.append(f"- {s}")
    md = "\n".join(lines)
    path.write_text(md, encoding="utf-8")
    return md


def write_combined_json(drift_results: Dict[str, Any], quality_results: Optional[Dict[str, Any]] = None, output_path: Optional[Path] = None) -> Path:
    """Write drift + optional quality to a single JSON for Actions."""
    path = output_path or get_metrics_dir() / "monitoring_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {"drift": drift_results}
    if quality_results is not None:
        out["data_quality"] = quality_results
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return path
