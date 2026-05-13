import pandas as pd


REQUIRED_COLUMNS = [
    "region",
    "year",
    "marriage_rate",
]

NUMERIC_COLUMNS = ["year", "marriage_rate"]
OPTIONAL_NUMERIC_COLUMNS = ["marriages"]


def validate_data(data: pd.DataFrame) -> list[str]:
    """Проверяет данные и возвращает список найденных ошибок."""
    errors: list[str] = []

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        errors.append("Отсутствуют обязательные колонки: " + ", ".join(missing_columns))
        return errors

    if data["region"].isna().any() or (data["region"].astype(str).str.strip() == "").any():
        errors.append("Колонка region не должна содержать пустые значения.")

    years = pd.to_numeric(data["year"], errors="coerce")
    if years.isna().any():
        errors.append("Колонка year должна быть числовой.")
    elif ((years < 1990) | (years > 2100)).any():
        errors.append("Колонка year должна находиться в диапазоне от 1990 до 2100.")

    for column in NUMERIC_COLUMNS + [
        column for column in OPTIONAL_NUMERIC_COLUMNS if column in data.columns
    ]:
        values = pd.to_numeric(data[column], errors="coerce")
        if values.isna().any():
            errors.append(f"Колонка {column} должна быть числовой.")
        elif (values < 0).any():
            errors.append(f"Колонка {column} не должна содержать отрицательные значения.")

    return errors
