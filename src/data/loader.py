import pandas as pd


def load_dataset(path: str) -> pd.DataFrame:
    """Загрузка исходного набора данных из CSV или XLSX файла."""
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".xlsx"):
        return pd.read_excel(path)
    raise ValueError("Неподдерживаемый формат файла")
