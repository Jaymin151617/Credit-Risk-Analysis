from fastapi import FastAPI, Response
from pydantic import BaseModel
import json
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer
from lightgbm import LGBMClassifier
from shap import TreeExplainer
from lime.lime_tabular import LimeTabularExplainer
from typing import Optional
from pathlib import Path

root_dir = Path(__file__).resolve().parents[1]
models_dir = root_dir / "models"

with open(models_dir / "best_model.pkl", "rb") as f:
    model = pickle.load(f)
with open(models_dir / "power_transformer.pkl", "rb") as f:
    transformer = pickle.load(f)
with open(models_dir / "shap_explainer.pkl", "rb") as f:
    shap_explainer = pickle.load(f)
with open(models_dir / "lime_config.pkl", "rb") as f:
    lime_config = pickle.load(f)


RANDOM_STATE = int(os.getenv("RANDOM_STATE", 100))

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


lime_explainer = LimeTabularExplainer(**lime_config, random_state=RANDOM_STATE)

app = FastAPI(
    title="Credit Risk Analysis",
    description="This API takes loan application data and returns SHAP values of features."
)

class Row(BaseModel):
    person_age: int
    person_education: str
    person_income: float
    person_home_ownership: str
    loan_amount: float
    loan_intent: str
    loan_interest_rate: float
    credit_score: int
    previous_loan_defaults: str

class APIBase:
    def process(self, row: dict) -> dict:
        """
        Pipeline function
        1. Convert data from user friendly to OG raw data equivalent
        2. Convert categorical variables to numeric value using the mapping
        3. Apply power transformation to all columns
        4. Explain row using SHAP explainer and return SHAP values of features
            4.1. (Optional) If prob < 0.75 then generate recommended interest rate
        5. Return json with data necessary to create chart in UI along with recommended rate.
        """
        
        raw_data = self.convert_to_raw(row)

        num_data = self.apply_cat_to_num_mapping(raw_data)

        transformed_data = self.apply_power_transformation(num_data, transformer)
    
        explanations = self.generate_explanations(
            transformed_data,
            model,
            transformer,
            shap_explainer,
            lime_explainer
        )

        return explanations


    def convert_to_raw(self, row: dict) -> pd.DataFrame:
        """Convert input data from user friendly to OG raw data equivalent"""
        
        for key, val in row.items():
            try:
                row[key] = val.replace(" ", "_").lower()
            except TypeError:
                continue

        df = pd.DataFrame([row])
        return df

    
    def apply_cat_to_num_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert categorical variables to numeric value using the mapping"""
        
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

        for col in CAT_VARIABLES:
            map_dict = mappings.get(col)
            df[col] = df[col].map(map_dict)

        return df

    
    def apply_power_transformation(
            self, 
            df: pd.DataFrame, 
            transformer: PowerTransformer
    ) -> pd.DataFrame:
        """Apply power transformation to all numeric columns"""
        
        df_out = df.copy()
        df_out.loc[:, NUM_VARIABLES] = transformer.transform(df_out[NUM_VARIABLES].values)
        return df_out

    
    def generate_explanations(
            self, 
            df: pd.DataFrame, 
            model: LGBMClassifier, 
            transformer: PowerTransformer, 
            shap_explainer: TreeExplainer,
            lime_explainer: LimeTabularExplainer
    ) -> dict:
        """
        Explain row using SHAP explainer and return SHAP values of features
        (Optional) If prob < 0.75 then generate recommended interest rate
        """
        index = 0       # Only 1 row in df
        shap_object = shap_explainer(df)

        features = list(shap_object[index].feature_names)
        vals = np.array(shap_object[index].values, dtype=float).flatten()
        base = float(shap_object[index].base_values)

        # Shift each contribution so base is effectively 0 and convert to percent
        shifted_shap = (vals + base / len(features)) * 100.0  # percentage-scale contributions

        current_pred = model.predict_proba(df)[0,1]

        new_pred = None
        rec_rate = None
        if current_pred < 0.75:
            rec_val = self._get_recommended_interest_rate(
                df.values.flatten(),
                model,
                lime_explainer
            )

            if rec_val is not None:
                df_rec = df.copy()
                df_rec.iloc[0, 6] = rec_val

                new_pred = model.predict_proba(df_rec)[0,1]

                df_rec.loc[:, NUM_VARIABLES] = transformer.inverse_transform(df[NUM_VARIABLES].values)
                rec_rate = df_rec.iloc[0, 6]

        output_dict = {}

        for idx, col in enumerate(features):
            output_dict[col] = shifted_shap[idx]

        output_dict["current_pred"] = current_pred
        output_dict["new_pred"] = new_pred
        output_dict["rec_rate"] = rec_rate

        return output_dict


    def _get_recommended_interest_rate(
            self,
            data: np.ndarray,
            model: LGBMClassifier,
            lime_explainer: dict,
            feature_idx: int = 6,
            prob_threshold: float = 0.75,
            min_limit: float = -2.0,
            max_limit: float = 2.0,
            window: float = 1.0,
            step: float = 0.1
    ) -> Optional[float]:
        """Get recommended interest rate to push loan approval chance beyond the given threshold"""
        pass


base = APIBase()

@app.post("/")
async def return_predictions(row: Row) -> Response:
    try:
        result = base.process(row)
        return Response(json.dumps(result), 200)
    except Exception as e:
        return Response(str(e), 400)
