# Credit Risk Analysis

[![License](https://img.shields.io/badge/license-Apache-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.12.12-blue)](https://www.python.org/)
![Status](https://img.shields.io/badge/status-complete-brightgreen)

## Table of Contents
- [Overview](#overview)
- [Live Demo](#live-demo)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Repository Structure](#repository-structure)
- [Model Performance Snapshot](#model-performance-snapshot)
- [Reproducibility](#reproducibility)
- [Security & Privacy](#security--privacy)
- [Limitations](#limitations)
- [License](#license)

---

## Overview
This project predicts the probability of loan approval from applicant and loan attributes, returns per-feature explanation values (SHAP contributions), and can recommend a revised interest rate to improve approval probability where applicable. It is intended as a demonstration and research codebase for interpretable ML and simple deployment patterns.

Core components:
- Notebook pipeline to reproduce dataset preparation, feature engineering, model training and explainability.
- Serialized model + explainer artifacts.
- Inference service: `FastAPI` (`app/api.py`).
- Web UI: `Streamlit` (`app/main.py`).

---

## Live Demo

[https://credit-risk-analysis-ui.onrender.com/](https://credit-risk-analysis-ui.onrender.com/)

---

## Quick Start

**Tested environment:** Python `3.12.12` (use a venv or equivalent).

### 1. Create & activate a virtual environment

- Unix / macOS:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

- Windows (PowerShell):
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2. Install dependencies

- For development (notebooks, training, explainers, app):
```bash
pip install -r requirements-dev.txt
```

- For runtime (only serving the app):
```bash
pip install -r requirements.txt
```

### 3. Run everything (single command)

```bash
bash run.sh
```

- FastAPI: http://localhost:8000

- FastAPI Swagger UI: http://localhost:8000/docs

- Streamlit UI: http://localhost:8501

### 4. Run services manually (optional)

- Backend

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

- Frontend

```bash
export API_URL="http://localhost:8000/"
streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0
```

---

### API Reference

Endpoint: `POST /`

Accepts: `application/json` — a single applicant + loan record.

Returns: model prediction probability, per-feature explanation contributions, and optional recommendation fields.

### Example request
```json
{
  "person_age": 35,
  "person_education": "Bachelor",
  "person_income": 50000.0,
  "person_home_ownership": "Mortgage",
  "loan_amount": 10000.0,
  "loan_intent": "Personal",
  "loan_interest_rate": 12.0,
  "credit_score": 650,
  "previous_loan_defaults": "No"
}
```

### Example response
```json
{
  "person_age":1.93,
  "person_education":5.88,
  "person_income":6.97,
  "person_home_ownership":-0.44,
  "loan_amount":1.94,
  "loan_intent":12.61,
  "loan_interest_rate":0.39,
  "credit_score":6.05,
  "previous_loan_defaults":22.83,
  "current_pred":0.5817,
  "new_pred":0.8374,
  "rec_rate":14.02
}
```

**Note**
  - Feature contributions are explanation values (SHAP style), not raw feature values.
  - `new_pred` and `rec_rate` may be null when recommendation is not applicable.

---

## Repository Structure

```text
Credit-Risk-Analysis/
├── app/
│   ├── api.py                     # FastAPI backend (prediction + explanations)
│   └── main.py                    # Streamlit frontend
├── datasets/
│   ├── raw_data.csv
│   ├── cleaned_data.csv
│   ├── train_data.csv
│   ├── validation_set.csv
│   ├── test_set.csv
│   └── sampled_train_data.csv     # SMOTE-balanced training set
├── models/
│   ├── best_model.pkl             # Final LightGBM model
│   ├── power_transformer.pkl
│   ├── shap_explainer.pkl
│   └── lime_config.pkl
├── notebooks/
│   ├── 01_dataset_selection.ipynb
│   ├── 02_cleaning_and_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_model_selection.ipynb
│   └── 06_explainability.ipynb
├── utils/helpers.py
├── requirements.txt
├── requirements-dev.txt
├── run.sh
└── LICENSE
```

---

## Model Performance Snapshot

From `notebooks/05_model_selection.ipynb`:

- Validation (tuned models):
  - `LightGBM`: weighted F1 `0.92`, accuracy `0.92`
  - `XGBoost`: weighted F1 `0.92`, accuracy `0.92`

`LightGBM` selected for efficiency/deployment fit

- Held-out test set (`n=4391`):
  - Accuracy: `0.93`
  - Weighted F1: `0.93`
  - Class 1 (approved) F1: `0.84`

---

## Reproducibility
- Data source: [Loan Approval Classification Dataset (Kaggle)](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data)
- Training seed: 100 (Configurable via environment variable `RANDOM_STATE`)
- Training pipeline: notebooks 01 → 06
- Artifacts: `models/` directory

To retrain:
1. Run notebooks in order
2. Export final model and transformers
3. Replace artifacts in `models/`

---

## Security & Privacy
```md
This project uses public dataset(s) for demonstration.  
Do not use for real lending decisions without compliance and fairness review.  
Remove PII if adapting to real data.
```

---

## Limitations
- Educational/demo project
- Dataset bias possible
- Recommendation logic heuristic

---

## License

Distributed under the terms of the `LICENSE` file in this repository.

---