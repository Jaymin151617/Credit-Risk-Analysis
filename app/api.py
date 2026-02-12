"""
Lightweight API that loads a trained LightGBM model and explainability
artifacts (SHAP & LIME) at application startup and exposes endpoints
to explain predictions for loan applications.

Classes
-------

Row()
    Pydantic model that validates and documents a single loan application record
    expected by the API endpoints.

Functions
---------

lifespan()
    Async context manager used as FastAPI 'lifespan' hook.

process()
    End-to-end pipeline function that prepares an input row for inference by
    validation, encoding, transformation, and any other preprocessing
    steps required by the model and returns SHAP values of the features.

convert_to_raw()
    Convert user-friendly input values into the original raw representation
    expected by the training pipeline.

apply_cat_to_num_mapping()
    Map categorical variables to their numeric encodings 
    using precomputed mappings.

apply_power_transformation()
    Apply the fitted PowerTransformer to numeric columns to match training-time
    scaling; returns transformed features suitable for model input.

generate_explanations()
    Produce explainability artifacts (SHAP values, summary) for a processed
    input row using the preloaded SHAP explainer.

_get_recommended_interest_rate()
    Internal helper that uses LIME to suggest a minimal adjustment 
    to the loan interest rate that could push the approval probability 
    above a given threshold.

return_predictions()
    Primary API endpoint handler that accepts validated input, runs the
    preprocessing and inference pipeline and returns prediction metadata.

"""


# ---------------------------------------------------------------------
# Standard Library Imports
# ---------------------------------------------------------------------
import asyncio                              # Async utilities for event-driven code
import os                                   # Operating system utilities (file paths, env)
import warnings                             # Warning control for cleaner outputs
from contextlib import asynccontextmanager  # Async context manager helper
from pathlib import Path                    # Path manipulation with pathlib
from typing import Optional                 # Type hinting for optional values


# ---------------------------------------------------------------------
# Web / API Framework
# ---------------------------------------------------------------------
from fastapi import FastAPI              # Web framework for building APIs
from pydantic import BaseModel           # Data validation / settings management


# ---------------------------------------------------------------------
# Serialization / I/O
# ---------------------------------------------------------------------
import joblib                            # Save/load fitted models and large objects


# ---------------------------------------------------------------------
# Third-Party Scientific Computing Libraries
# ---------------------------------------------------------------------
import numpy as np                       # Numerical computing (arrays, math ops)
import pandas as pd                      # DataFrame-based data manipulation


# ---------------------------------------------------------------------
# Machine Learning Libraries
# ---------------------------------------------------------------------
from sklearn.preprocessing import PowerTransformer
# Used to stabilize variance and make data more Gaussian-like.
# Helpful for skewed distributions.

from lightgbm import LGBMClassifier
# Gradient-boosted decision tree classifier (efficient for tabular data).


# ---------------------------------------------------------------------
# Model Explainability
# ---------------------------------------------------------------------
from shap import TreeExplainer
# SHAP TreeExplainer for fast SHAP value computation on tree-based models.

from lime.lime_tabular import LimeTabularExplainer
# LIME tabular explainer for local surrogate explanations of model predictions.


# =====================================================================
# Warning configuration
# =====================================================================

# Suppress all warnings to keep terminal clean
warnings.filterwarnings("ignore", category=FutureWarning)


# =====================================================================
# Configuration & Constants
# =====================================================================

# Reproducible random seed (from environment or default)
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 100))

# Categorical and numerical feature lists (used by explainers / preprocessing)
CAT_VARIABLES = [
    "person_education",
    "person_home_ownership",
    "loan_intent",
    "previous_loan_defaults"
]

NUM_VARIABLES = [
    "person_age",
    "person_income",
    "loan_amount",
    "loan_interest_rate",
    "credit_score"
]

# Resolve models directory relative to repository root.
# `root_dir` is parent of the package containing this file.
root_dir = Path(__file__).resolve().parents[1]
models_dir = root_dir / "models"


# =====================================================================
# Module-level placeholders for loaded artifacts
# =====================================================================

model: Optional[LGBMClassifier] = None
transformer: Optional[PowerTransformer] = None
shap_explainer: Optional[TreeExplainer] = None
lime_explainer: Optional[LimeTabularExplainer] = None


# =====================================================================
# Application lifespan: load heavy artifacts once at startup
# =====================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Async context manager used as FastAPI 'lifespan' hook.

    Loads heavy ML artifacts (model, transformer, explainers) once during
    application startup so endpoints can rely on them being available.

    Notes:
    - joblib.load is used for deserialization. If files are missing or corrupt,
      startup will fail (and the app won't serve requests) — consider adding
      try/except with logging and graceful shutdown if desired.
    - Loading inside an async context is okay because joblib.load is blocking;
      consider running it inside a threadpool (asyncio.to_thread) for strict
      non-blocking behavior in an async event loop.
    """
    global model, transformer, shap_explainer, lime_explainer

    # Load model and preprocessing artifacts from disk
    # These calls are synchronous and can be heavy; they run once at startup.
    model = joblib.load(models_dir / "best_model.pkl")
    transformer = joblib.load(models_dir / "power_transformer.pkl")
    shap_explainer = joblib.load(models_dir / "shap_explainer.pkl")

    # LIME requires configuration (training data, feature names, categorical
    # feature indices, etc.). We expect lime_config.pkl to hold the kwargs
    # necessary to reconstruct the LimeTabularExplainer.
    lime_config = joblib.load(models_dir / "lime_config.pkl")
    lime_explainer = LimeTabularExplainer(**lime_config, random_state=RANDOM_STATE)

    # Yield control back to FastAPI so it can start serving requests.
    yield

    # Optional teardown logic could go here (if any resources need cleanup).


# =====================================================================
# FastAPI application factory
# =====================================================================

app = FastAPI(
    title="Credit Risk Analysis",
    description="This API takes loan application data and returns SHAP values of features.",
    lifespan=lifespan
)


# =====================================================================
# Request models
# =====================================================================

class Row(BaseModel):
    """
    Pydantic model describing a single loan application row expected by endpoints.
    """
    person_age: int
    person_education: str
    person_income: float
    person_home_ownership: str
    loan_amount: float
    loan_intent: str
    loan_interest_rate: float
    credit_score: int
    previous_loan_defaults: str


# =====================================================================
# API Helpers
# =====================================================================

def process(row: dict) -> dict:
    """
    Execute the end-to-end inference and explainability pipeline.

    This function performs the following steps:

    1. Convert user-friendly input data into the original raw format
       expected by the model.
    2. Encode categorical variables into numerical representations
       using predefined mappings.
    3. Apply power transformation using a pre-fitted transformer.
    4. Generate model explanations using SHAP (and optionally LIME).
       - If predicted probability is below threshold (e.g., 0.75),
         a recommended feature adjustment (e.g., interest rate)
         may be generated.
    5. Return a JSON-serializable dictionary containing:
       - Prediction outputs
       - SHAP values
       - Visualization-ready data
       - Optional recommended adjustments

    Parameters
    ----------
    row : dict
        User-provided input data in UI-friendly format.

    Returns
    -------
    dict
        Structured response containing prediction results,
        explanation data, and optional recommendations.

    Notes
    -----
    - Assumes global objects `model`, `transformer`,
      `shap_explainer`, and `lime_explainer` are initialized.
    - Intended for real-time inference usage (e.g., API layer).
    - All preprocessing steps must match training pipeline exactly.
    """

    # Convert user-friendly input to raw model format
    raw_data = convert_to_raw(row)

    # Convert categorical features to numeric using mapping
    num_data = apply_cat_to_num_mapping(raw_data)

    # Apply power transformation using pre-fitted transformer
    transformed_data = apply_power_transformation(num_data, transformer)

    # Generate explanations and optional recommendations
    explanations = generate_explanations(
        transformed_data,
        model,
        transformer,
        shap_explainer,
        lime_explainer
    )

    # Return structured output for UI consumption
    return explanations


def convert_to_raw(row: dict) -> pd.DataFrame:
    """
    Convert user-friendly input into model-ready raw format.

    This method:
    - Normalizes string values by:
        Replacing spaces with underscores
        Converting to lowercase
    - Wraps the processed dictionary into a single-row DataFrame.

    Parameters
    ----------
    row : dict
        User-provided input data (typically from UI or API request).

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame formatted for downstream preprocessing
        and model inference.
    """

    # Normalize string values to match model training format
    # (e.g., "High School" -> "high_school")
    for key, val in row.items():
        if isinstance(val, str):
            row[key] = val.replace(" ", "_").lower()

    # Convert dictionary into a single-row DataFrame
    return pd.DataFrame([row])


def apply_cat_to_num_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert predefined categorical variables into numerical encodings.

    This method applies fixed ordinal mappings to specific categorical
    columns to match the model's training-time feature representation.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing categorical columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with mapped categorical values converted to integers.

    Notes
    -----
    - Assumes categorical values are already normalized
      (e.g., lowercase, underscores).
    - Columns to transform are defined in `CAT_VARIABLES`.
    - Mapping is deterministic and must match training configuration.
    - Unseen categories will result in NaN values.
    """

    # Predefined categorical → numeric mappings
    mappings = {
        "person_education": {
            "high_school": 0,
            "associate": 1,
            "bachelor": 2,
            "master": 3,
            "doctorate": 4,
        },
        "person_home_ownership": {
            "other": 0,
            "rent": 1,
            "mortgage": 2,
            "own": 3,
        },
        "loan_intent": {
            "venture": 0,
            "medical": 1,
            "personal": 2,
            "education": 3,
            "home_improvement": 4,
            "debt_consolidation": 5,
        },
        "previous_loan_defaults": {
            "no": 0,
            "yes": 1,
        },
    }

    # Apply mapping to each categorical column defined in CAT_VARIABLES
    for col in CAT_VARIABLES:
        map_dict = mappings.get(col)

        # Convert categorical values to numeric codes
        df[col] = df[col].map(map_dict)

    return df


def apply_power_transformation(
        df: pd.DataFrame,
        transformer: PowerTransformer
) -> pd.DataFrame:
    """
    Apply a pre-fitted PowerTransformer to numeric feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing numeric columns to transform.
    transformer : sklearn.preprocessing.PowerTransformer
        Pre-fitted transformer (must already be trained on training data).

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with transformed numeric columns.

    Notes
    -----
    - The transformer must already be fitted.
    - The transformation must match training-time preprocessing exactly.
    - Does not modify the original DataFrame.
    """

    # Create a copy to avoid mutating the original input
    df_out = df.copy()

    # Apply transformation to predefined numeric columns
    # `.values` ensures 2D array input for sklearn transformer
    df_out.loc[:, NUM_VARIABLES] = transformer.transform(
        df_out[NUM_VARIABLES].values
    )

    return df_out


def generate_explanations(
        df: pd.DataFrame,
        model: LGBMClassifier,
        transformer: PowerTransformer,
        shap_explainer: TreeExplainer,
        lime_explainer: LimeTabularExplainer
) -> dict:
    """
    Generate prediction explanation and optional recommendation.

    This function:
    1. Computes SHAP values for the input row.
    2. Converts SHAP contributions into percentage-scale values
       (baseline redistributed across features).
    3. Calculates the current predicted probability (class 1).
    4. If probability is below threshold (0.75), attempts to
       recommend a new interest rate using LIME-based local search.
    5. Returns a JSON-serializable dictionary containing:
        - Feature SHAP contributions (percentage)
        - Current prediction
        - New prediction (if recommendation applied)
        - Recommended interest rate (original scale)

    Parameters
    ----------
    df : pd.DataFrame
        Single-row transformed DataFrame (model-ready format).
    model : LGBMClassifier
        Trained LightGBM classification model.
    transformer : PowerTransformer
        Pre-fitted transformer used during preprocessing.
    shap_explainer : shap.TreeExplainer
        SHAP explainer for tree-based models.
    lime_explainer : LimeTabularExplainer
        LIME explainer for local surrogate modeling.

    Returns
    -------
    dict
        Dictionary containing SHAP contributions and optional recommendation.

    Notes
    -----
    - Assumes `df` contains exactly one row.
    - Assumes binary classification (class index 1).
    - SHAP baseline is redistributed across features for percentage-style UI.
    - Interest rate feature is assumed to be column index 6.
    - NUM_VARIABLES must match transformer fitting order.
    """

    # SHAP explanation (single row)
    index = 0  # Only one row expected
    shap_object = shap_explainer(df)

    features = list(shap_object[index].feature_names)
    vals = np.array(shap_object[index].values, dtype=float).flatten()
    base = float(shap_object[index].base_values)

    # Convert SHAP values to percentage scale
    # Baseline is redistributed evenly across features
    shifted_shap = (vals + base / len(features)) * 100.0

    # Compute current predicted probability (class 1)
    current_pred = round(model.predict_proba(df.values)[0, 1], 4)

    new_pred = None
    rec_rate = None

    # If prediction is below threshold, attempt recommendation
    if current_pred < 0.75:

        rec_val = _get_recommended_interest_rate(
            df.values.flatten(),
            model,
            lime_explainer,
            prob_threshold=min(max(current_pred + 0.25, 0.50), 0.90)
        )

        if rec_val is not None:
            # Create modified copy with recommended interest rate
            df_rec = df.copy()

            # Assumes interest rate column index is 6
            df_rec.iloc[0, 6] = rec_val

            # Compute new prediction
            new_pred = round(model.predict_proba(df_rec)[0, 1], 4)

            # Convert transformed numeric columns back to original scale
            df_rec.loc[:, NUM_VARIABLES] = transformer.inverse_transform(
                df_rec[NUM_VARIABLES].values
            )

            # Extract recommended interest rate (original scale)
            rec_rate = round(df_rec.iloc[0, 6], 2)

    # Build output dictionary
    output_dict = {}

    for idx, col in enumerate(features):
        output_dict[col] = round(float(shifted_shap[idx]), 2)

    output_dict["current_pred"] = current_pred
    output_dict["new_pred"] = new_pred
    output_dict["rec_rate"] = rec_rate

    return output_dict


def _get_recommended_interest_rate(
        data: np.ndarray,
        model: LGBMClassifier,
        lime_explainer: LimeTabularExplainer,
        feature_idx: int = 6,
        prob_threshold: float = 0.75,
        min_limit: float = -2.2492310680997027,
        max_limit: float = 2.696269297384873,
        window: float = 2.0,
        step: float = 0.01
) -> Optional[float]:
    """
    Recommend a new interest rate (transformed scale) to increase
    loan approval probability beyond a specified threshold.

    This function:
    1. Uses LIME to build a local linear approximation of the model.
    2. Solves analytically for the feature value required to reach
       the desired probability threshold.
    3. Searches within bounded limits around that solution.
    4. Validates candidate values against the real model.
    5. Returns the smallest value that satisfies the threshold.

    Parameters
    ----------
    data : np.ndarray
        Single input instance (transformed feature space).
    model : LGBMClassifier
        Trained binary classification model with `predict_proba()`.
    lime_explainer : LimeTabularExplainer
        Pre-configured LIME explainer instance.
    feature_idx : int, optional
        Index of the interest rate feature (default = 6).
    prob_threshold : float, optional
        Desired minimum probability for approval (default = 0.75).
    min_limit : float, optional
        Lower bound of feature search range (5% in transformed scale).
    max_limit : float, optional
        Upper bound of feature search range (20% in transformed scale).
    window : float, optional
        Search window centered around analytical solution.
    step : float, optional
        Increment size for brute-force search.

    Returns
    -------
    Optional[float]
        Recommended feature value (transformed scale).
        Returns None if no suitable value found.

    Notes
    -----
    - Assumes binary classification (class index 1).
    - Feature space must match model training scale.
    - Analytical solution is based on LIME's local linear model,
      but final validation uses the actual model.
    """

    # Generate local explanation using LIME
    exp = lime_explainer.explain_instance(
        data_row=data,
        predict_fn=model.predict_proba
    )

    # Extract local slope (weight) for interest rate feature
    slope = None
    for idx, weight in exp.local_exp[1]:  # class 1
        if idx == feature_idx:
            slope = float(weight)
            break

    # Feature not present in explanation → zero local influence
    if slope is None:
        slope = 0.0

    # If slope ≈ 0, changing feature won't affect probability locally
    if np.isclose(slope, 0.0):
        return None

    # Extract intercept of local surrogate model
    # Local linear model: prob ≈ intercept + slope * feature_value
    intercept = float(exp.intercept[1])

    # Analytical solution:
    # intercept + slope * x = prob_threshold
    # => x = (prob_threshold - intercept) / slope
    center_value = (prob_threshold - intercept) / slope

    # Define bounded search region
    start = max(min_limit, center_value - window)
    end = min(max_limit, center_value + window)

    # Ensure current value is included in search range
    current_val = float(data[feature_idx])
    if current_val < start:
        start = current_val - step
    if current_val > end:
        end = current_val + step

    # Generate candidate values (rounded to 2 decimals)
    raw_vals = np.arange(start, end + (step / 2.0), step)
    values = np.unique(raw_vals.round(2))

    # Validate candidates using actual model predictions
    passing = []

    for val in values:
        row = data.copy()
        row[feature_idx] = val

        prob = float(model.predict_proba(row.reshape(1, -1))[0, 1])

        if prob >= prob_threshold:
            passing.append(val)

    # No valid value found
    if not passing:
        return None

    # Return smallest valid adjustment
    return round(float(min(passing)), 2)


# =====================================================================
# API Endpoint
# =====================================================================

@app.post("/")
async def return_predictions(row: Row):
    """
    Handle prediction and explanation requests.

    This endpoint:
    1. Accepts user input validated via the `Row` schema.
    2. Converts the request body into a dictionary.
    3. Executes the synchronous `process()` pipeline in a thread pool
       to avoid blocking the event loop.
    4. Returns prediction results and explanations as JSON.

    Parameters
    ----------
    row : Row
        Pydantic model representing validated user input.

    Returns
    -------
    dict
        JSON-serializable dictionary containing:
        - Prediction probability
        - SHAP contributions
        - Optional recommended interest rate
        - Optional updated prediction
    """

    # Retrieve current asyncio event loop
    loop = asyncio.get_running_loop()

    # Run CPU-bound synchronous `process` function in thread pool
    # to prevent blocking the async event loop
    result = await loop.run_in_executor(
        None,               # Default ThreadPoolExecutor
        process,            # Blocking function
        row.model_dump()    # Convert Pydantic model to dict
    )

    # Return structured response to client
    return result
