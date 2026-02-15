# Notebooks

## demo_build_and_evaluate.ipynb

Demonstrates the **exploratory phase** of the project as if it was built first; the rest of the codebase then productionized this workflow.

**Contents:**

1. **Setup and load data** — Load UCI Bank Marketing CSV from `data/raw/`.
2. **Data analysis (EDA)** — Schema, target distribution, numerical summaries, categorical counts, simple plots (age/balance by subscription).
3. **Feature engineering** — Same as production: numerical scaling, categorical one-hot encoding; train/test split.
4. **Model training** — Logistic Regression (baseline) and Gradient Boosting (primary), same hyperparameters as `configs/model.yaml`.
5. **Evaluation** — ROC-AUC, precision, recall, F1, calibration curve, confusion matrix.
6. **From prototype to production** — Short summary of how this maps to `src/pipelines/`, serving, and the Streamlit app.

**How to run:**

- From repo root, ensure data exists: `make data`.
- Open the notebook in Jupyter or VS Code; run from **repo root** (the notebook changes to repo root so imports and `data/raw/` path work).
- Or start Jupyter from repo root: `jupyter notebook notebooks/demo_build_and_evaluate.ipynb`.
