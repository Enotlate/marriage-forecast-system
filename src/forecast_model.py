import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


def train_model(x: pd.DataFrame, y: pd.Series) -> LinearRegression:
    """Обучает линейную модель прогнозирования коэффициента брачности."""
    model = LinearRegression()
    model.fit(x, y)
    return model


def predict_marriage_rate(model: LinearRegression, x: pd.DataFrame):
    """Возвращает прогноз коэффициента брачности."""
    return model.predict(x)


def build_forecast_result(data: pd.DataFrame, predictions) -> pd.DataFrame:
    """Добавляет к данным колонку с прогнозом."""
    result = data.copy()
    result["predicted_marriage_rate"] = predictions
    return result


def calculate_mae(actual, predicted) -> float:
    """Считает среднюю абсолютную ошибку прогноза."""
    return float(mean_absolute_error(actual, predicted))


def run_forecast(data: pd.DataFrame, x: pd.DataFrame, y: pd.Series):
    """Обучает модель, выполняет прогноз и возвращает результат с MAE."""
    model = train_model(x, y)
    predictions = predict_marriage_rate(model, x)
    result = build_forecast_result(data, predictions)
    mae = calculate_mae(y, predictions)
    return result, mae, model
