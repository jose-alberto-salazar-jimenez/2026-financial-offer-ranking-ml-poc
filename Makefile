# Financial Offer Propensity & Ranking ML PoC
# One-liners for README "Running the Demo"

PYTHON ?= python
PIP ?= pip

.PHONY: setup data train evaluate run drift build

setup:
	$(PIP) install -e ".[dev]"

data:
	$(PYTHON) scripts/download_data.py

train:
	$(PYTHON) -m src.pipelines.train

evaluate:
	$(PYTHON) -m src.pipelines.evaluate

package:
	$(PYTHON) -m src.pipelines.package_model

run:
	streamlit run src/app/streamlit_app.py --server.port=8501

drift:
	$(PYTHON) -m src.monitoring.drift

build:
	docker build -t financial-offer-ranking-ml-poc:latest .
