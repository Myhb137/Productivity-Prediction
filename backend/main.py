from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd


app = FastAPI(
    title="Student Productivity Prediction API",
    description="API for predicting student productivity scores",
    version="1.0.0"
)


# Load trained model
model = joblib.load("../model/best_model.pkl")


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