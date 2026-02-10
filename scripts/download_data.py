"""Download UCI Bank Marketing dataset into data/raw/."""
import zipfile
from pathlib import Path

import requests

from src.utils.paths import get_raw_data_dir, get_project_root

# UCI ML Repository
UCI_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
FILENAME_CSV = "bank-additional-full.csv"


def main() -> None:
    root = get_project_root()
    raw_dir = get_raw_data_dir()
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = root / "bank-marketing.zip"

    print("Downloading UCI Bank Marketing dataset...")
    resp = requests.get(UCI_URL, timeout=60)
    resp.raise_for_status()
    zip_path.write_bytes(resp.content)
    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)
    zip_path.unlink(missing_ok=True)
    csv_path = raw_dir / FILENAME_CSV
    if not csv_path.exists():
        for f in raw_dir.rglob("*.csv"):
            if "full" in f.name.lower():
                if f.resolve() != (raw_dir / FILENAME_CSV).resolve():
                    f.rename(raw_dir / FILENAME_CSV)
                break
        csv_path = raw_dir / FILENAME_CSV
    if csv_path.exists():
        print(f"Data saved to {csv_path}")
    else:
        raise FileNotFoundError(f"Expected {FILENAME_CSV} not found under {raw_dir}")


if __name__ == "__main__":
    main()
