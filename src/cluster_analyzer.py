import pandas as pd
from sklearn.cluster import KMeans


CLUSTER_COLUMNS = ["income", "unemployment", "population", "marriage_rate"]


def add_region_clusters(data: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    """Добавляет к данным номер кластера региона."""
    result = data.copy()
    cluster_count = min(n_clusters, len(result))

    if cluster_count < 1:
        result["cluster"] = []
        return result

    model = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
    result["cluster"] = model.fit_predict(result[CLUSTER_COLUMNS])
    return result
