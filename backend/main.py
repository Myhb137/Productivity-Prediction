from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import shap


app = FastAPI(
    title="Student Productivity Prediction API",
    description="API for predicting student productivity scores",
    version="1.0.0"
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "best_model.pkl"
EXPLAINER_PATH = BASE_DIR / "model" / "shap_explainer.pkl"


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    detail = exc.errors()
    json_error_types = {"json_invalid", "model_attributes_type", "string_type", "list_type"}
    is_json_error = any(error.get("type") in json_error_types for error in detail)

    if is_json_error:
        return JSONResponse(
            status_code=400,
            content={"detail": "Invalid JSON request body. Please send valid JSON matching the expected schema."},
        )

    return JSONResponse(status_code=422, content={"detail": detail})


try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None


class StudentData(BaseModel):

    study_hours_per_day: float
    focus_score: float
    sleep_hours: float
    phone_usage_hours: float
    stress_level: float


@app.get("/")
def home():
    return {
        "message": "Student Productivity Prediction API is running"
    }


@app.post("/predict")
def predict(data: StudentData):
    if model is None:
        raise HTTPException(
            status_code=500,
            detail=f"Model file not found at {MODEL_PATH}. Train the model first or ensure the file exists."
        )

    input_data = pd.DataFrame([{
        "study_hours_per_day": data.study_hours_per_day,
        "focus_score": data.focus_score,
        "sleep_hours": data.sleep_hours,
        "phone_usage_hours": data.phone_usage_hours,
        "stress_level": data.stress_level
    }])

    prediction = model.predict(input_data)[0]

    return {
        "productivity_score": round(float(prediction), 2)
    }


@app.post("/explainer")
def explain(data: StudentData):
    if not EXPLAINER_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"SHAP explainer file not found at {EXPLAINER_PATH}."
        )

    input_data = pd.DataFrame([{
        "study_hours_per_day": data.study_hours_per_day,
        "focus_score": data.focus_score,
        "sleep_hours": data.sleep_hours,
        "phone_usage_hours": data.phone_usage_hours,
        "stress_level": data.stress_level
    }])

    explainer = joblib.load(str(EXPLAINER_PATH))

    shap_values = explainer.shap_values(input_data)

    import matplotlib.pyplot as plt
    shap.summary_plot(shap_values, input_data, plot_type="bar")
    plt.tight_layout()
    plt.savefig("shap_summary_plot.png")
    plt.close()

    return {
        "message": "SHAP summary plot generated and saved as 'shap_summary_plot.png'"
    }