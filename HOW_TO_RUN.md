# How to Run This PoC — Step-by-Step Guide

This guide explains, in plain language, how to **containerize the model**, run the **Streamlit app** for inference, and how to **train**, **evaluate**, and **showcase** the PoC. No prior Docker or ML experience required.

---

## What You Need First

- **Python 3.9+** (e.g. 3.11). Check with: `python --version`
- **Docker** (only if you want to run the app in a container). Check with: `docker --version`
- **Git** (to clone the repo). Check with: `git --version`

Optional but recommended: a **conda** environment (e.g. `2026_02_experian_ml_eng`) so dependencies don’t conflict with other projects.

---

## Two Ways to Run the PoC

| Goal | How |
|------|-----|
| **Run everything in a container** (easiest for “just show the app”) | Use **Docker**: build the image, run the container, open the app. The model is trained **inside** the image. |
| **Train and evaluate on your machine, then run the app** | Use **local Python**: download data, train, evaluate, then run Streamlit (or run the same app inside Docker and point it at your local model). |

Below we cover both.

---

## Option A: Run Everything with Docker (Containerized)

This is the **simplest way to showcase the PoC**: one container that already has the data, trained model, and Streamlit app.

### Step 1: Open a terminal in the project folder

```bash
cd path\to\2026-financial-offer-ranking-ml-poc
```

(Use your actual path to the repo.)

### Step 2: Build the Docker image

This step:

- Installs Python and dependencies inside the image.
- Downloads the UCI Bank Marketing dataset.
- Trains the model (Gradient Boosting).
- Evaluates it and saves metrics and baselines.
- Packages the model so the app can use it.

So **the model is containerized**: it lives inside the image.

**Command:**

```bash
docker build -t financial-offer-ranking-ml-poc:latest .
```

- `-t financial-offer-ranking-ml-poc:latest` gives the image a name and tag.
- The dot (`.`) means “use the current folder as build context.”

**Time:** Several minutes (download + training). Wait until you see something like “Successfully built” / “Successfully tagged”.

### Step 3: Run the container

**Command:**

```bash
docker run -p 8501:8501 financial-offer-ranking-ml-poc:latest
```

- `-p 8501:8501` maps port 8501 on your machine to port 8501 in the container (where Streamlit runs).

### Step 4: Open the Streamlit app in your browser

- Open: **http://localhost:8501**
- You should see the **Financial Offer Propensity & Ranking** app.

### Step 5: Use the app for inference

- In the **sidebar**, change the customer profile (age, job, balance, contact, etc.).
- Click **“Get propensity & ranking”**.
- The app will:
  - Send that profile to the **model inside the container**.
  - Get back a **propensity score** (probability that this customer accepts an offer).
  - Show **ranked offers** (synthetic list) based on that score.

That’s **inference**: the containerized model is doing predictions for the profiles you enter.

### Optional: Run with Docker Compose

If you prefer `docker-compose`:

```bash
docker-compose up --build
```

Then open **http://localhost:8501**.  
If you have a local `artifacts` folder (from training on your machine), the compose file can mount it so the container uses **your** model instead of the one built in the image.

---

## Option B: Run Locally (Train, Evaluate, Then Use the App)

Use this when you want to **see and control** training and evaluation on your machine, then run the same Streamlit app (locally or in Docker).

### Step 1: Open a terminal in the project folder

```bash
cd path\to\2026-financial-offer-ranking-ml-poc
```

### Step 2: (Optional) Activate a conda environment

```bash
conda activate 2026_02_experian_ml_eng
```

(Or whatever env you use for this project.)

### Step 3: Install the project

```bash
pip install -e ".[dev]"
```

This installs the code and dependencies (pandas, scikit-learn, streamlit, etc.) in “editable” mode so changes in the repo are picked up.

### Step 4: Download the dataset

```bash
make data
```

Or:

```bash
python scripts/download_data.py
```

This fetches the UCI Bank Marketing dataset and puts it in `data/raw/`. You only need to do this once.

### Step 5: Train the model

```bash
make train
```

Or:

```bash
python -m src.pipelines.train
```

What happens:

- Loads the raw data.
- Splits into train/test.
- Builds a preprocessor (scaling, encoding).
- Trains a **Gradient Boosting** classifier (and keeps a Logistic Regression as baseline).
- Saves the model and preprocessor under `artifacts/model/` (e.g. `model.joblib`, `preprocessor.joblib`).

So **the model is saved on disk**; the Streamlit app will load it from `artifacts/` for inference.

### Step 6: Evaluate the model

```bash
make evaluate
```

Or:

```bash
python -m src.pipelines.evaluate
```

What happens:

- Loads the trained model and the test set.
- Computes **ROC-AUC**, precision, recall, F1, calibration.
- Writes results to `artifacts/metrics/metrics.json` and an eval report (e.g. `artifacts/metrics/eval_report.md`).

**To see how the model works** from a performance perspective, open:

- `artifacts/metrics/metrics.json` — numbers (AUC, precision, etc.)
- `artifacts/metrics/eval_report.md` — same in a readable table.

### Step 7: Package the model (baselines + model card)

```bash
make package
```

Or:

```bash
python -m src.pipelines.package_model
```

What happens:

- Saves **baseline stats** (feature and score distributions) under `artifacts/baselines/` for drift checks.
- Generates a **model card** (e.g. `artifacts/model_card.md`) describing the model, data, and metrics.

**To see how the model is documented**, open `artifacts/model_card.md`.

### Step 8: Run the Streamlit app locally

```bash
make run
```

Or:

```bash
streamlit run src/app/streamlit_app.py --server.port=8501
```

Then open **http://localhost:8501**.

The app:

- Loads the model from `artifacts/model/` (or from `ARTIFACTS_DIR` if you set it).
- Uses it for **inference**: you enter a customer profile, click the button, and get propensity + ranked offers.

So **the same app** runs locally or in Docker; the only difference is where the `artifacts/` folder lives (your disk vs inside the image).

### Optional: Run drift check

```bash
make drift
```

Or:

```bash
python -m src.monitoring.drift
```

This compares recent data (e.g. last 20% of training data) to the baseline and writes a drift report. Useful to show “monitoring” in the PoC.

---

## How the Model Is Used for Inference in the App

- The **model** and **preprocessor** are loaded once when the app starts (from `artifacts/model/` or the path given by `ARTIFACTS_DIR`).
- When you click **“Get propensity & ranking”**:
  1. The values from the sidebar are sent to the **serving** layer (`src/serving/predict.py`).
  2. The preprocessor turns them into the same features the model was trained on.
  3. The model outputs a probability (propensity).
  4. The app shows that score and a synthetic list of ranked offers.

So **containerizing the model** means: the same `artifacts/model/` (or an image that was built with that step) is used inside a container to serve these predictions. The Streamlit app is just the UI that calls that inference code.

---

## One-Command Local Demo (Train → Eval → Package → Drift → App)

If you already have the data and want to run the full pipeline and then the app in one go:

```bash
./scripts/demo_flow.sh
```

On Windows (PowerShell), you can run the same steps in order:

```powershell
make train
make evaluate
make package
make drift
make run
```

Then open http://localhost:8501.

---

## How to Showcase the PoC

1. **Containerized app**
   - Show: `docker build ...` then `docker run ...`, then open the app and change a profile and click “Get propensity & ranking”. Explain: “The model runs inside the container and serves predictions to the UI.”

2. **Training and evaluation**
   - Show: `make data`, `make train`, `make evaluate`. Then open `artifacts/metrics/metrics.json` or `eval_report.md` and briefly explain AUC, precision, recall.

3. **Model card and governance**
   - Open `artifacts/model_card.md` and say: “We document the model, data, and metrics for governance.”

4. **Drift monitoring**
   - Run `make drift` and show that a report is produced (and that in CI we could open a GitHub Issue if drift is high).

5. **Re-training**
   - Run `make train` again (and optionally `make evaluate` and `make package`), then run the app again. Explain: “Same pipeline can be run on a schedule or trigger to retrain and update the model.”

---

## Quick Reference: Commands

| What you want | Command |
|---------------|---------|
| Install deps | `pip install -e ".[dev]"` or `make setup` |
| Download data | `make data` |
| Train model | `make train` |
| Evaluate model | `make evaluate` |
| Package (baselines + model card) | `make package` |
| Run Streamlit app (local) | `make run` |
| Run drift check | `make drift` |
| Build Docker image (trains model inside) | `docker build -t financial-offer-ranking-ml-poc:latest .` |
| Run container + app | `docker run -p 8501:8501 financial-offer-ranking-ml-poc:latest` |
| Run with Docker Compose | `docker-compose up --build` |

---

## Troubleshooting

- **“Model not found” in the app**  
  Train first: `make train` (and `make package` if you use drift). The app reads from `artifacts/model/`. If you use Docker without building the image, ensure the container has those artifacts (e.g. built in the image or mounted via a volume).

- **“Raw data not found”**  
  Run `make data` (or `python scripts/download_data.py`) so `data/raw/` contains the CSV.

- **Docker build fails (e.g. network)**  
  The build downloads data from the internet. Check your connection and any corporate proxy/firewall. If needed, download data locally first and adjust the Dockerfile to copy from `data/raw/` instead of running the download script.

- **Port 8501 already in use**  
  Stop any other Streamlit or app using 8501, or run the app on another port, e.g. `streamlit run src/app/streamlit_app.py --server.port=8502` and open http://localhost:8502.

- **Make not found (Windows)**  
  Use the equivalent commands shown above (e.g. `python -m src.pipelines.train` instead of `make train`), or install Make (e.g. via Chocolatey or WSL).

---

## Summary

- **Containerize the model:** Build the Docker image; the image runs download → train → evaluate → package, so the model lives inside the image. Running the container starts Streamlit and uses that model for inference.
- **Use it in the Streamlit app:** The app loads the model from `artifacts/model/` and runs inference when you click “Get propensity & ranking.”
- **Evaluate / train / see how it works:** Use `make train`, `make evaluate`, `make package`, then read `artifacts/metrics/` and `artifacts/model_card.md`. Run `make run` to try the app locally, or run the same app in Docker for the showcase.
