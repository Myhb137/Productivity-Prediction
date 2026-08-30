import pandas as pd 


def fe(data):
    
    selected_features = data[
    [
        "study_hours_per_day",
        "focus_score",
        "sleep_hours",
        "phone_usage_hours",
        "stress_level",
        "productivity_score"
    ] 
    ]   
    print("Selected features : ", selected_features.columns.tolist())
    return selected_features
