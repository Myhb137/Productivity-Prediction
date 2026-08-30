from src.data_loader import load_data
from src.feature_selction import fe
from src.overall_analysis import overall_analysis
from src.model import train_model
from src.split_data import split_data
from src.model_tuning import tune_xgboost
from src.evaluate import evaluate_model
from src.modl_pkl import save_model
from src.explainer import create_explainer


def main():

    data = load_data()

    data = fe(data)

    overall_analysis(data)

    X = data.drop("productivity_score", axis=1)
    y = data["productivity_score"]

    X_train, X_test, y_train, y_test = split_data(X, y)

    best_params = tune_xgboost(
        X_train,
        y_train
    )

    model = train_model(
        X_train,
        y_train,
        params=best_params.get_params()
    )

    results = evaluate_model(
        model,
        X_test,
        y_test
    )

    print("\n" + "=" * 35)
    print("       FINAL MODEL RESULTS")
    print("=" * 35)
    print(f"RMSE : {results[0]:.4f}")
    print(f"MAE  : {results[1]:.4f}")
    print(f"R²   : {results[2]:.4f}")
    print("=" * 35)

    best_model = train_model(
        X_train,
        y_train,
        params=best_params.get_params()
    )

    explainer = create_explainer(
        best_model,
        X_train
    )

    save_model(
        best_model,
        "../Productivity Prediction/model/best_model.pkl"
    )

    save_model(
        explainer,
        "../Productivity Prediction/model/explainer.pkl"
    )

    return data


if __name__ == "__main__":
    main()

