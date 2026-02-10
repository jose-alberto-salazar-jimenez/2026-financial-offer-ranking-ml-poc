"""Simple append-only audit log for 'who ran what' demo."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.utils.paths import get_artifacts_path


def get_audit_log_path() -> Path:
    return get_artifacts_path() / "audit_log.jsonl"


def log_action(action: str, outcome: str = "success", metadata: Optional[dict] = None) -> None:
    """Append one audit entry."""
    path = get_audit_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "action": action,
        "outcome": outcome,
        "user": os.environ.get("USER", os.environ.get("USERNAME", "unknown")),
        "metadata": metadata or {},
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
