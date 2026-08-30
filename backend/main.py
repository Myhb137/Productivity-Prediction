from pathlib import Path

import joblib
import pandas as pd
import shap

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# ============================================================
# CONFIGURATION
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
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Student Productivity API",
    description="Machine learning API for predicting student productivity.",
    version="2.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# GLOBAL MODEL / EXPLAINER
# ============================================================

model = None
explainer = None


# ============================================================
# LOAD MODEL
# ============================================================

try:

    print("=" * 60)
    print("STARTING STUDENT PRODUCTIVITY API")
    print("=" * 60)

    print("Model path:")
    print(MODEL_PATH)

    print("Model exists:")
    print(MODEL_PATH.exists())

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    print("✅ Productivity model loaded successfully.")

    explainer = shap.TreeExplainer(model)

    print("✅ SHAP explainer created successfully.")

    print("=" * 60)


except Exception as e:

    print("=" * 60)
    print("❌ MODEL LOADING ERROR")
    print(str(e))
    print("=" * 60)


# ============================================================
# REQUEST SCHEMA
# ============================================================

class StudentData(BaseModel):

    study_hours_per_day: float
    focus_score: float
    sleep_hours: float
    phone_usage_hours: float
    stress_level: float


# ============================================================
# CREATE MODEL INPUT
# ============================================================

def create_input(data: StudentData) -> pd.DataFrame:

    return pd.DataFrame(
        [
            {
                "study_hours_per_day": data.study_hours_per_day,
                "focus_score": data.focus_score,
                "sleep_hours": data.sleep_hours,
                "phone_usage_hours": data.phone_usage_hours,
                "stress_level": data.stress_level,
            }
        ],
        columns=FEATURES,
    )


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Student Productivity API is running",
        "version": "2.0.0",
        "model": "XGBoost",
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "version": "2.0.0",
        "model_loaded": model is not None,
        "explainer_loaded": explainer is not None,
        "model_file": MODEL_PATH.name,
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(data: StudentData):

    if model is None:

        raise HTTPException(
            status_code=500,
            detail="Productivity model is not loaded.",
        )

    try:

        input_data = create_input(data)

        prediction = model.predict(input_data)[0]

        prediction = float(prediction)

        return {
            "productivity_score": round(prediction, 2)
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}",
        )


# ============================================================
# SHAP EXPLANATION ENDPOINT
# ============================================================

@app.post("/explainer")
def explain(data: StudentData):

    if explainer is None:

        raise HTTPException(
            status_code=500,
            detail="SHAP explainer is not loaded.",
        )

    try:

        input_data = create_input(data)

        shap_values = explainer.shap_values(input_data)

        # SHAP returns an array for this regression model.
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
            detail=f"SHAP explanation error: {str(e)}",
        )
