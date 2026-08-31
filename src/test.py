import joblib
import pandas as pd

model = joblib.load("backend/productivity_model.pkl")

X = pd.DataFrame({
    "study_hours_per_day": [7],
    "focus_score": [95],
    "sleep_hours": [8],
    "phone_usage_hours": [2],
    "stress_level": [2]
})

print(model.feature_names_in_)
print(model.predict(X))