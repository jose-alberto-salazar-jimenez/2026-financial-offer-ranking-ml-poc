#!/usr/bin/env bash
# Quick sanity test: run one prediction inside the container or env
set -e
cd "$(dirname "$0")/.."
python -c "
from src.serving.predict import predict
score = predict({'age': 40, 'job': 'admin.', 'marital': 'married', 'education': 'university.degree', 'default': 'no', 'balance': 1000, 'housing': 'yes', 'loan': 'no', 'contact': 'cellular', 'day': 15, 'month': 'may', 'duration': 300, 'campaign': 2, 'pdays': -1, 'previous': 0, 'poutcome': 'unknown'})
assert 0 <= score <= 1
print('Smoke test OK: propensity =', score)
"
