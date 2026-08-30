from xgboost import XGBRegressor


def train_model(X_train, y_train, params=None):

    if params is None:
        model = XGBRegressor()
    else:
        model = XGBRegressor(**params)

    model.fit(X_train, y_train)

    return model