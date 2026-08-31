from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI(
    title="Productivity Prediction API",
    description="API for predicting productivity score",
    version="1.0.0"
)

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "model",
    "productivity_model.pkl"
)

print("=" * 60)
print("PRODUCTIVITY API")
print("=" * 60)

print("MODEL PATH:", MODEL_PATH)
print("MODEL EXISTS:", os.path.exists(MODEL_PATH))

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

print("MODEL SIZE:", os.path.getsize(MODEL_PATH))
print("MODEL TYPE:", type(model))

FEATURES = [
    "study_hours_per_day",
    "focus_score",
    "sleep_hours",
    "phone_usage_hours",
    "stress_level"
]

print("MODEL FEATURES:", FEATURES)
print("MODEL LOADED: TRUE")


class PredictionInput(BaseModel):
    study_hours_per_day: float
    focus_score: float
    sleep_hours: float
    phone_usage_hours: float
    stress_level: float


@app.get("/")
def home():
    return {
        "message": "Productivity Prediction API is running",
        "status": "success",
        "endpoint": "/predict",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }


@app.post("/predict")
def predict(data: PredictionInput):

    input_data = {
        "study_hours_per_day": data.study_hours_per_day,
        "focus_score": data.focus_score,
        "sleep_hours": data.sleep_hours,
        "phone_usage_hours": data.phone_usage_hours,
        "stress_level": data.stress_level
    }

    print("=" * 60)
    print("NEW PREDICTION REQUEST")
    print("INPUT DATA:", input_data)

    X = pd.DataFrame([input_data])
    X = X[FEATURES]

    print("DATAFRAME:")
    print(X)

    print("FEATURE ORDER:")
    print(list(X.columns))

    print("VALUES:")
    print(X.iloc[0].tolist())

    prediction = model.predict(X)[0]
    prediction = float(prediction)

    print("MODEL PREDICTION:", prediction)

    return {
        "productivity_score": round(prediction, 2)
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000))
    )