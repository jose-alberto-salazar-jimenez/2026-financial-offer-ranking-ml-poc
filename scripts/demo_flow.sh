#!/usr/bin/env bash
# One-command: train -> evaluate -> drift -> run UI (assumes data already downloaded)
set -e
cd "$(dirname "$0")/.."
echo "=== Train ==="
make train
echo "=== Evaluate ==="
make evaluate
echo "=== Package (baselines + model card) ==="
make package
echo "=== Drift check ==="
make drift
echo "=== Starting Streamlit (Ctrl+C to stop) ==="
make run
