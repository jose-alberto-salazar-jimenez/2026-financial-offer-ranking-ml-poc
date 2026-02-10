FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
RUN pip install --no-cache-dir -e .

# Create dirs for data and artifacts (can be mounted)
RUN mkdir -p data/raw data/processed artifacts/model artifacts/metrics artifacts/baselines

# Build model at image build time so image runs without mount
RUN python scripts/download_data.py && \
    python -m src.pipelines.train && \
    python -m src.pipelines.evaluate && \
    python -m src.pipelines.package_model

EXPOSE 8501
CMD ["streamlit", "run", "src/app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
