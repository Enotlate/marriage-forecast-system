import pandas as pd

from src.feature_builder import build_features
from src.forecast_model import run_forecast


def test_forecast_returns_predictions_for_each_row():
    data = pd.DataFrame(
        {
            "region": ["Москва", "Санкт-Петербург", "Республика Татарстан"],
            "year": [2022, 2022, 2022],
            "marriages": [103530, 58240, 27612],
            "population": [13104200, 5600000, 4001600],
            "marriage_rate": [7.9, 10.4, 6.9],
        }
    )

    x, y = build_features(data)
    result, mae, model = run_forecast(data, x, y)

    assert model is not None
    assert mae >= 0
    assert "predicted_marriage_rate" in result.columns
    assert len(result["predicted_marriage_rate"]) == len(data)
