"""Generate model card from config + metrics."""
from pathlib import Path
from typing import Optional

from src.utils.paths import get_artifacts_path, get_metrics_dir


def generate_model_card(output_path: Optional[Path] = None) -> str:
    """Write model_card.md from metrics and config. Returns content."""
    path = output_path or get_artifacts_path() / "model_card.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = get_metrics_dir() / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        import json
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
    lines = [
        "# Model Card: Financial Offer Propensity",
        "",
        "## Overview",
        "Binary classification model predicting P(customer accepts offer) using UCI Bank Marketing–style features.",
        "",
        "## Dataset",
        "UCI Bank Marketing (bank-additional-full.csv). Target: subscription to term deposit (yes/no).",
        "",
        "## Metrics",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"| {k} | {v:.4f} |")
        else:
            lines.append(f"| {k} | {v} |")
    lines.extend([
        "",
        "## Limitations",
        "- Trained on historical campaign data; may not reflect current product mix.",
        "- Not evaluated for fairness across segments in this PoC.",
        "",
        "## Intended use",
        "Demonstration of propensity scoring and offer ranking for financial personalization.",
        "",
    ])
    content = "\n".join(lines)
    path.write_text(content, encoding="utf-8")
    return content
