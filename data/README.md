# Data

## Source: UCI Bank Marketing Dataset

This project uses the [UCI Bank Marketing](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) dataset. It simulates customer response to financial marketing (phone campaigns) and is used for **offer acceptance propensity** modeling.

- **Target**: `y` — whether the client subscribed to a term deposit (yes/no).
- **Features**: Demographics (age, job, marital, education), financial (balance, default, housing, loan), contact (contact, day, month, duration, campaign, pdays, previous, poutcome).

## How to populate

1. Run **`make data`** from the repo root, or
2. Run **`python scripts/download_data.py`**

This downloads the dataset from the UCI ML repository and extracts it into `data/raw/`. Processed feature tables are written to `data/processed/` by the training pipeline.

## Directory layout

- `raw/` — Raw CSV (e.g. `bank-additional-full.csv`). Gitignored.
- `processed/` — Feature-engineered outputs used by training. Gitignored.

Only this README is committed.
