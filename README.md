# Production-Style Demo — Financial Offer Propensity & Ranking ML System

## Overview

This repository demonstrates a production-style machine learning system that predicts customer propensity to accept financial offers and serves ranked recommendations via an interactive application.

The system uses the **UCI Bank Marketing** dataset to simulate a realistic financial personalization workflow — similar to systems used by consumer financial platforms.

This project focuses on the **full ML lifecycle**, not just model training:

- Customer segmentation & feature engineering
- Propensity modeling
- Offer ranking logic
- Batch + interactive inference
- Drift & data-quality monitoring
- Governance artifacts
- CI/CD automation
- Containerized deployment

The goal is to showcase how classical ML models are integrated into a production-ready pipeline.

---

## System Architecture

```
Raw Data → Feature Engineering → Model Training/Evaluation
              ↓
         Model Artifacts + Baselines
              ↓
     Dockerized Streamlit Application
              ↓
       Monitoring + Drift Checks
              ↓
       CI/CD Automation (GitHub Actions)
```

Key characteristics:

- Reproducible ML pipeline
- Shared inference layer
- Monitoring hooks
- Automated retraining workflow
- Containerized app deployment

---

## Dataset

**Source:** [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)  
Simulates customer response to financial offers (phone campaigns).

Features represent:

- Demographics (age, job, marital, education)
- Financial profile (balance, default, housing, loan)
- Contact strategy (contact, day, month, duration, campaign, pdays, previous, poutcome)

Target:

- Offer acceptance (yes/no) — P(customer accepts offer)

This enables:

- Propensity modeling
- Segmentation analysis
- Ranking simulation

---

## Project Structure

```
├── src/
│   ├── pipelines/        # ingest → features → train → evaluate → package
│   ├── serving/          # shared inference logic
│   ├── monitoring/       # drift + data quality checks
│   ├── governance/       # model documentation + audit helpers
│   └── app/              # Streamlit UI
├── configs/              # model + monitoring configs
├── artifacts/            # model + evaluation outputs
├── tests/                 # unit tests
├── scripts/              # automation helpers
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/     # CI/CD automation
```

---

## ML Pipeline

### 1. Data ingestion

- Load raw dataset from UCI (via `make data`)
- Validate schema and columns

### 2. Feature engineering

- Categorical encoding (one-hot)
- Numerical scaling (StandardScaler)
- ColumnTransformer saved with model

### 3. Training

Models:

- Logistic Regression (baseline)
- Gradient Boosting (primary)

Evaluation:

- ROC-AUC, precision/recall, F1
- Calibration (binned reliability)
- Metrics and optional segment performance

Artifacts saved:

- model, preprocessing pipeline, feature list
- metrics, baseline distributions for drift

---

## Ranking Logic

Predictions are interpreted as:

**P(customer accepts offer)**

The app ranks synthetic offers per customer based on predicted acceptance probability for demonstration.

---

## Interactive Demo (Streamlit)

The Streamlit app allows:

- Customer profile simulation (UCI-style inputs)
- Real-time propensity scoring
- Ranked offer display
- Optional prediction explanations

Run locally:

```bash
make run
```

or

```bash
docker-compose up
```

---

## Monitoring & Drift Detection

The system includes lightweight production-style monitoring:

- Feature distribution drift (PSI, optional KS)
- Prediction score shift detection
- Data quality checks (missing rate, value ranges)

Drift reports can trigger automated alerts via GitHub Actions (e.g. open an Issue).

---

## Governance & Documentation

Artifacts include:

- Model card generation
- Evaluation reports
- Version tracking
- Audit log (who ran what)

These simulate responsible ML deployment practices.

---

## CI/CD Automation

GitHub Actions workflows:

### CI pipeline

- Tests, optional lint (ruff)
- Optional smoke training on small sample

### Build image

- Build Docker image, smoke test, push to GHCR

### Drift check

- Scheduled (e.g. weekly) monitoring
- Create GitHub Issue if drift above threshold

### Retraining workflow

- Manual trigger: full pipeline (ingest → train → evaluate → package)
- Upload model and metrics as workflow artifacts

### Release

- On version tag: create release (optional model artifact attach)

---

## Running the Demo

Use the project from the repo root. Optionally activate a conda environment first, e.g.:

```bash
conda activate 2026_02_experian_ml_eng
```

### 1. Setup

```bash
make setup
```

### 2. Download dataset

```bash
make data
```

### 3. Train model

```bash
make train
```

### 4. Evaluate

```bash
make evaluate
```

### 5. Package (baselines + model card)

```bash
make package
```

### 6. Run app

```bash
make run
```

Or run the full flow then start UI:

```bash
./scripts/demo_flow.sh
```

---

## What This Demonstrates

- End-to-end ML lifecycle design
- Feature engineering pipeline
- Classical ML modeling
- Ranking system logic
- Monitoring & drift detection
- Governance artifacts
- Containerized deployment
- CI/CD integration

The focus is on engineering maturity and production readiness.

---

## Interview Narrative

This system simulates a financial personalization platform where customer attributes are used to predict offer acceptance likelihood. The architecture shows how ML models move from training into monitored, automated deployment workflows with governance considerations.

---

## Future Extensions

- Feature store abstraction
- A/B experimentation framework
- Online feedback loop
- Fairness analysis
- Explainability dashboards

---

## License

Proof of concept / demo project.
