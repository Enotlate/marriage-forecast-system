from pathlib import Path

import pandas as pd


def load_csv_data(file_path) -> pd.DataFrame:
    """Загружает региональные данные из CSV-файла."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError as exc:
        path = Path(file_path)
        raise FileNotFoundError(f"Файл с данными не найден: {path}") from exc
    except Exception as exc:
        raise ValueError(f"Не удалось прочитать CSV-файл: {exc}") from exc
