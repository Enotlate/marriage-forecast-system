import pandas as pd


FEATURE_COLUMNS = ["income", "unemployment", "population"]
TARGET_COLUMN = "marriage_rate"


def build_features(data: pd.DataFrame):
    """Формирует матрицу признаков X и целевую переменную y."""
    x = data[FEATURE_COLUMNS].copy()
    y = data[TARGET_COLUMN].copy()
    return x, y
