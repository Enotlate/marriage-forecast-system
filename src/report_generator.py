import pandas as pd


def generate_text_report(data: pd.DataFrame, mae: float | None = None) -> str:
    """Формирует простой текстовый отчет по результатам анализа."""
    rows_count = len(data)
    regions_count = data["region"].nunique() if "region" in data.columns else 0
    avg_actual = data["marriage_rate"].mean() if "marriage_rate" in data.columns else None
    avg_predicted = (
        data["predicted_marriage_rate"].mean()
        if "predicted_marriage_rate" in data.columns
        else None
    )

    lines = [
        "Отчет по прогнозированию коэффициента брачности",
        f"Количество строк данных: {rows_count}",
        f"Количество регионов: {regions_count}",
    ]

    if avg_actual is not None:
        lines.append(f"Средний фактический коэффициент брачности: {avg_actual:.2f}")

    if avg_predicted is not None:
        lines.append(f"Средний прогнозный коэффициент брачности: {avg_predicted:.2f}")

    if mae is not None:
        lines.append(f"MAE модели: {mae:.3f}")

    lines.append(
        "Вывод: система позволяет загрузить региональные данные, проверить их, "
        "сгруппировать регионы и получить базовый прогноз коэффициента брачности."
    )

    return "\n".join(lines)
