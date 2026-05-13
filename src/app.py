from pathlib import Path

import streamlit as st

try:
    from cluster_analyzer import add_region_clusters
    from data_loader import load_csv_data
    from data_validator import validate_data
    from feature_builder import build_features
    from forecast_model import run_forecast
    from report_generator import generate_text_report
except ImportError:
    from src.cluster_analyzer import add_region_clusters
    from src.data_loader import load_csv_data
    from src.data_validator import validate_data
    from src.feature_builder import build_features
    from src.forecast_model import run_forecast
    from src.report_generator import generate_text_report


BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_DATA_PATH = BASE_DIR / "data" / "sample_marriage_data.csv"


def main() -> None:
    st.title("Программная система прогнозирования соотношения браков по регионам РФ")

    uploaded_file = st.file_uploader("Загрузите CSV-файл", type=["csv"])
    data_source = uploaded_file if uploaded_file is not None else SAMPLE_DATA_PATH

    try:
        data = load_csv_data(data_source)
    except Exception as exc:
        st.error(str(exc))
        return

    st.subheader("Исходные данные")
    st.dataframe(data)

    st.subheader("Проверка данных")
    errors = validate_data(data)
    if errors:
        for error in errors:
            st.error(error)
        return

    st.success("Данные прошли проверку.")

    x, y = build_features(data)
    forecast_data, mae, _ = run_forecast(data, x, y)
    clustered_data = add_region_clusters(forecast_data)

    st.subheader("Результаты прогноза и кластеризации")
    st.dataframe(clustered_data)

    st.subheader("Текстовый отчет")
    st.text(generate_text_report(clustered_data, mae))


if __name__ == "__main__":
    main()
