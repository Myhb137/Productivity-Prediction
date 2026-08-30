from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV


def tune_xgboost(X_train, y_train):

    model = XGBRegressor(
        random_state=42
    )

    param_grid = {
        "n_estimators": [100, 200, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [2, 3, 4, 5, 6],
        "min_child_weight": [1, 3, 5],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0]
    }

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=30,
        scoring="neg_root_mean_squared_error",
        cv=5,
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    print("Best parameters:")
    print(search.best_params_)

    print("\nBest CV RMSE:")
    print(-search.best_score_)

    return search.best_estimator_