import pandas as pd

from src.feature_builder import build_features
from src.forecast_model import run_forecast


def test_forecast_returns_predictions_for_each_row():
    data = pd.DataFrame(
        {
            "region": ["Москва", "Санкт-Петербург", "Республика Татарстан"],
            "year": [2022, 2022, 2022],
            "marriage_rate": [6.7, 7.0, 7.5],
            "income": [105000, 81500, 51500],
            "unemployment": [2.0, 2.3, 3.1],
            "population": [13010000, 5600000, 3920000],
        }
    )

    x, y = build_features(data)
    result, mae, model = run_forecast(data, x, y)

    assert model is not None
    assert mae >= 0
    assert "predicted_marriage_rate" in result.columns
    assert len(result["predicted_marriage_rate"]) == len(data)
