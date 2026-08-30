from pathlib import Path

import joblib
import pandas as pd
import shap

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "productivity_model.pkl"

FEATURES = [
    "study_hours_per_day",
    "focus_score",
    "sleep_hours",
    "phone_usage_hours",
    "stress_level",
]


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Student Productivity API",
    version="3.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOAD MODEL
# ============================================================

try:
    model = joblib.load(MODEL_PATH)
    explainer = shap.TreeExplainer(model)

    print("=" * 60)
    print("PRODUCTIVITY API")
    print("=" * 60)
    print("MODEL PATH:", MODEL_PATH)
    print("MODEL EXISTS:", MODEL_PATH.exists())
    print("MODEL SIZE:", MODEL_PATH.stat().st_size)
    print("MODEL TYPE:", type(model))

    if hasattr(model, "feature_names_in_"):
        print("MODEL FEATURES:", list(model.feature_names_in_))

    print("MODEL LOADED: TRUE")
    print("SHAP LOADED: TRUE")

except Exception as e:
    model = None
    explainer = None

    print("MODEL LOADING ERROR:", e)


# ============================================================
# REQUEST
# ============================================================

class StudentData(BaseModel):
    study_hours_per_day: float
    focus_score: float
    sleep_hours: float
    phone_usage_hours: float
    stress_level: float


# ============================================================
# INPUT
# ============================================================

def create_input(data: StudentData) -> pd.DataFrame:

    return pd.DataFrame(
        [{
            "study_hours_per_day": data.study_hours_per_day,
            "focus_score": data.focus_score,
            "sleep_hours": data.sleep_hours,
            "phone_usage_hours": data.phone_usage_hours,
            "stress_level": data.stress_level,
        }],
        columns=FEATURES,
    )


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Student Productivity API is running",
        "version": "3.0.0",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "model_loaded": model is not None,
        "explainer_loaded": explainer is not None,
        "model_file": MODEL_PATH.name,
        "model_exists": MODEL_PATH.exists(),
    }


# ============================================================
# MODEL TEST
# ============================================================

@app.get("/debug-model")
def debug_model():

    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model is not loaded",
        )

    test = pd.DataFrame(
        [{
            "study_hours_per_day": 8,
            "focus_score": 95,
            "sleep_hours": 8,
            "phone_usage_hours": 1,
            "stress_level": 2,
        }],
        columns=FEATURES,
    )

    prediction = float(
        model.predict(test)[0]
    )

    return {
        "model_file": MODEL_PATH.name,
        "model_size": MODEL_PATH.stat().st_size,
        "prediction": prediction,
        "features": FEATURES,
    }


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
def predict(data: StudentData):

    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model is not loaded",
        )

    try:

        input_data = create_input(data)

        prediction = float(
            model.predict(input_data)[0]
        )

        print("INPUT:", input_data.to_dict(orient="records")[0])
        print("RAW PREDICTION:", prediction)

        return {
            "productivity_score": round(
                prediction,
                2,
            )
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}",
        )


# ============================================================
# SHAP
# ============================================================

@app.post("/explainer")
def explain(data: StudentData):

    if explainer is None:
        raise HTTPException(
            status_code=500,
            detail="SHAP explainer is not loaded",
        )

    try:

        input_data = create_input(data)

        shap_values = explainer.shap_values(
            input_data
        )

        values = shap_values[0]

        contributions = {
            feature: round(float(value), 4)
            for feature, value in zip(
                FEATURES,
                values,
            )
        }

        return {
            "shap_values": contributions
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"SHAP error: {str(e)}",
        )
