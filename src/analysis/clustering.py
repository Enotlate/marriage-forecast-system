from sklearn.cluster import KMeans


def build_clusters(data, n_clusters: int = 3):
    """Кластеризация регионов по подготовленным признакам."""
    model = KMeans(n_clusters=n_clusters, random_state=42)
    return model.fit_predict(data)
