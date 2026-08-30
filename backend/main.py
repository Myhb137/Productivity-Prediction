from pathlib import Path

import joblib
import pandas as pd
import shap

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


app = FastAPI(
    title="Student Productivity API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR /"model"/"productivity_model.pkl"


try:
    model = joblib.load(MODEL_PATH)
    explainer = shap.TreeExplainer(model)
except Exception as e:
    model = None
    explainer = None
    print(f"Loading error: {e}")


class StudentData(BaseModel):
    study_hours_per_day: float
    focus_score: float
    sleep_hours: float
    phone_usage_hours: float
    stress_level: float


def create_input(data: StudentData):
    return pd.DataFrame([{
        "study_hours_per_day": data.study_hours_per_day,
        "focus_score": data.focus_score,
        "sleep_hours": data.sleep_hours,
        "phone_usage_hours": data.phone_usage_hours,
        "stress_level": data.stress_level
    }])


@app.get("/")
def home():
    return {
        "message": "Student Productivity API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "explainer_loaded": explainer is not None,
        "model_path": str(MODEL_PATH)
    }


@app.post("/predict")
def predict(data: StudentData):
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model could not be loaded"
        )

    try:
        input_data = create_input(data)
        prediction = model.predict(input_data)[0]

        return {
            "productivity_score": round(float(prediction), 2)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post("/explainer")
def explain(data: StudentData):
    if explainer is None:
        raise HTTPException(
            status_code=500,
            detail="SHAP explainer could not be loaded"
        )

    try:
        input_data = create_input(data)
        shap_values = explainer.shap_values(input_data)
        values = shap_values[0]

        contributions = {
            feature: round(float(value), 4)
            for feature, value in zip(
                input_data.columns,
                values
            )
        }

        return {
            "shap_values": contributions
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )