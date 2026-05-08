from sklearn.linear_model import LinearRegression


def train_forecast_model(x_train, y_train):
    """Обучение базовой прогнозной модели."""
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model
