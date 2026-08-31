from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import shap
import os

app = FastAPI(
    title="Productivity Prediction API",
    description="API for predicting productivity score and explaining predictions",
    version="1.0.0"
)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "productivity_model.pkl"
)

model = joblib.load(MODEL_PATH)
explainer = shap.TreeExplainer(model)

FEATURES = [
    "study_hours_per_day",
    "focus_score",
    "sleep_hours",
    "phone_usage_hours",
    "stress_level"
]


class PredictionInput(BaseModel):
    study_hours_per_day: float
    focus_score: float
    sleep_hours: float
    phone_usage_hours: float
    stress_level: float


def create_input(data: PredictionInput):
    X = pd.DataFrame([{
        "study_hours_per_day": data.study_hours_per_day,
        "focus_score": data.focus_score,
        "sleep_hours": data.sleep_hours,
        "phone_usage_hours": data.phone_usage_hours,
        "stress_level": data.stress_level
    }])

    return X[FEATURES]


@app.get("/")
def home():
    return {
        "message": "Productivity Prediction API is running",
        "status": "success",
        "endpoints": ["/predict", "/explain", "/docs"]
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "shap_loaded": True
    }


@app.post("/predict")
def predict(data: PredictionInput):
    X = create_input(data)

    prediction = model.predict(X)[0]

    return {
        "productivity_score": round(float(prediction), 2)
    }


@app.post("/explain")
def explain(data: PredictionInput):
    X = create_input(data)

    shap_values = explainer.shap_values(X)

    if hasattr(shap_values, "tolist"):
        shap_values = shap_values.tolist()

    base_value = explainer.expected_value

    if hasattr(base_value, "tolist"):
        base_value = base_value.tolist()

    if isinstance(base_value, list):
        base_value = base_value[0]

    return {
        "productivity_score": round(float(model.predict(X)[0]), 2),
        "base_value": round(float(base_value), 2),
        "features": {
            FEATURES[i]: round(float(shap_values[0][i]), 4)
            for i in range(len(FEATURES))
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000))
    )