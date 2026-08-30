from pathlib import Path

import joblib
import pandas as pd
import shap

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel


app = FastAPI(
    title="Student Productivity Prediction API",
    description="API for predicting student productivity scores",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "best_model.pkl"
EXPLAINER_PATH = BASE_DIR / "model" / "shap_explainer.pkl"
PLOT_PATH = BASE_DIR / "model" / "shap_summary_plot.png"


try:
    model = joblib.load(MODEL_PATH)
except Exception:
    model = None


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Invalid request data.",
            "errors": exc.errors(),
        },
    )


class StudentData(BaseModel):
    study_hours_per_day: float
    focus_score: float
    sleep_hours: float
    phone_usage_hours: float
    stress_level: float


@app.get("/")
def home():
    return {
        "message": "Student Productivity Prediction API is running",
        "status": "ok",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(data: StudentData):
    if model is None:
        raise HTTPException(
            status_code=500,
            detail=f"Model could not be loaded from {MODEL_PATH}",
        )

    try:
        input_data = pd.DataFrame([
            {
                "study_hours_per_day": data.study_hours_per_day,
                "focus_score": data.focus_score,
                "sleep_hours": data.sleep_hours,
                "phone_usage_hours": data.phone_usage_hours,
                "stress_level": data.stress_level,
            }
        ])

        prediction = model.predict(input_data)[0]

        return {
            "productivity_score": round(float(prediction), 2)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post("/explainer")
def explain(data: StudentData):
    if not EXPLAINER_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Explainer not found at {EXPLAINER_PATH}",
        )

    try:
        input_data = pd.DataFrame([
            {
                "study_hours_per_day": data.study_hours_per_day,
                "focus_score": data.focus_score,
                "sleep_hours": data.sleep_hours,
                "phone_usage_hours": data.phone_usage_hours,
                "stress_level": data.stress_level,
            }
        ])

        explainer = joblib.load(EXPLAINER_PATH)
        shap_values = explainer.shap_values(input_data)

        import matplotlib.pyplot as plt

        shap.summary_plot(
            shap_values,
            input_data,
            plot_type="bar",
            show=False,
        )

        plt.tight_layout()
        plt.savefig(PLOT_PATH, bbox_inches="tight")
        plt.close()

        return {
            "message": "Explanation generated successfully",
            "plot_path": str(PLOT_PATH),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Explanation failed: {str(e)}",
        )

