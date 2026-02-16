# Credit Risk Analysis

Interpretable loan approval prediction system built with a full ML pipeline and a deployable app stack (`FastAPI` + `Streamlit`).

## Overview

This project predicts loan approval probability from applicant and loan attributes, explains each prediction with SHAP/LIME, and can suggest an interest-rate adjustment for low-probability cases.

The repository includes:

- End-to-end notebooks for data prep, modeling, and explainability
- Versioned datasets and serialized model artifacts
- A FastAPI inference service (`app/api.py`)
- A Streamlit frontend (`app/main.py`)

**Live Demo:** `https://credit-risk-analysis-ui.onrender.com/`

## Key Features

- Binary loan approval prediction (`loan_status`)
- Feature-level explanation for every request
- Optional recommendation flow for `loan_interest_rate` adjustments
- Reproducible notebook pipeline from raw Kaggle data to deployed artifacts
- Local one-command startup script (`run.sh`)

## Model Performance Snapshot

From `notebooks/05_model_selection.ipynb`:

- Validation (tuned models):
  - `LightGBM`: weighted F1 `0.92`, accuracy `0.92`
  - `XGBoost`: weighted F1 `0.92`, accuracy `0.92`
  - `LightGBM` selected for efficiency/deployment fit
- Held-out test set (`n=4391`):
  - Accuracy: `0.93`
  - Weighted F1: `0.93`
  - Class 1 (approved) F1: `0.84`

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

## Data Source

- Primary dataset: [Loan Approval Classification Dataset (Kaggle)](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data)
- The project notebook explicitly treats the data as synthetic and educational.

## Quick Start

Dependency file intent:

- `requirements-dev.txt`: complete dependency set for the full project (notebooks, training, explainability, app)
- `requirements.txt`: minimal dependency set for deploying/running only the web app (`FastAPI` + `Streamlit`), to reduce redundancy

### 1. Create environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements-dev.txt
```

### 3. Run backend + frontend

```bash
bash run.sh
```

App endpoints:

- FastAPI: `http://localhost:8000`
- FastAPI docs: `http://localhost:8000/docs`
- Streamlit UI: `http://localhost:8501`

## Manual Run (optional)

Run services separately if needed.

Backend:

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

Frontend:

```bash
export API_URL="http://localhost:8000/"
streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0
```

## API Contract

`POST /`

Example request:

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

Example response:

```json
{
  "person_age": -3.41,
  "person_education": 2.87,
  "person_income": 6.14,
  "person_home_ownership": -1.02,
  "loan_amount": -4.35,
  "loan_intent": 0.73,
  "loan_interest_rate": 5.22,
  "credit_score": 11.03,
  "previous_loan_defaults": -8.56,
  "current_pred": 0.71,
  "new_pred": 0.84,
  "rec_rate": 13.25
}
```

Notes:

- Feature keys in the response are explanation contributions.
- `new_pred` and `rec_rate` are `null` when no recommendation is produced.

## Notebook Workflow

1. `notebooks/01_dataset_selection.ipynb`
2. `notebooks/02_cleaning_and_eda.ipynb`
3. `notebooks/03_feature_engineering.ipynb`
4. `notebooks/04_model_training.ipynb`
5. `notebooks/05_model_selection.ipynb`
6. `notebooks/06_explainability.ipynb`

Important pipeline outputs:

- `datasets/raw_data.csv`
- `datasets/cleaned_data.csv`
- `datasets/train_data.csv`, `datasets/validation_set.csv`, `datasets/test_set.csv`
- `datasets/sampled_train_data.csv` (SMOTE-balanced)
- `models/power_transformer.pkl`
- `models/best_model.pkl`
- `models/shap_explainer.pkl`
- `models/lime_config.pkl`

## Modeling and Data Notes

- `person_gender` is intentionally dropped in feature engineering to reduce direct gender bias.
- Highly correlated fields (`person_employment_experience`, `credit_history_length`) are removed to limit multicollinearity.
- Numeric features are transformed with a saved `PowerTransformer`.
- SMOTE is applied only on the training split.

## Important Limitation

This repository is built as a demonstration/learning project and should not be used as an official lending decision engine without production-grade governance, compliance controls, and external validation.

## License

Distributed under the terms of the `LICENSE` file in this repository.
