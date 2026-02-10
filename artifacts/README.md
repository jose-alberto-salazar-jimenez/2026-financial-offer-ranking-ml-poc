# Artifacts

Model and evaluation outputs produced by the ML pipeline. Most contents are gitignored; only this README (and optionally small `metrics/` and model card files) are committed.

## Directories

- **model/** — Trained model binary (`model.joblib`), preprocessor (`preprocessor.joblib`), and feature list. Produced by `make train` and `package_model`.
- **metrics/** — Evaluation metrics (`metrics.json`) and optional eval report. Produced by `make evaluate` and the package step.
- **baselines/** — Baseline feature and score distributions used for drift detection. Produced by `package_model`.

## Generating artifacts

From repo root:

```bash
make data    # download data
make train   # train and save model + preprocessor
make evaluate
make package # write baselines + optional model card
```

Pipeline entry point: `src.pipelines.package_model` (also invoked as part of retrain workflows).
