# Changelog

## [0.1.0] — Initial PoC

- End-to-end ML pipeline: ingest, feature engineering, train (Logistic Regression + Gradient Boosting), evaluate, package
- UCI Bank Marketing dataset integration and download script
- Streamlit app for propensity scoring and ranked offers
- Serving layer: shared inference API (predict / predict_batch)
- Monitoring: PSI/KS drift checks, data quality checks, report generation
- Governance: model card generation, audit logging
- Docker and docker-compose for containerized app
- Makefile targets: setup, data, train, evaluate, package, run, drift, build
- GitHub Actions: CI, build/push image, drift check (with Issue on alert), retrain workflow, release on tag
- Unit tests: features, predict, drift
