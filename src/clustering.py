from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib


def train_clustering_model(rfm):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    k_values = list(range(2, 11))
    silhouette_scores = []
    inertias = []

    for k in k_values:
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )
        labels = model.fit_predict(rfm_scaled)
        
        score = silhouette_score(rfm_scaled, labels)
        silhouette_scores.append(score)
        
        inertias.append(model.inertia_)

    metrics = {
        "k_values": k_values,
        "silhouette_scores": silhouette_scores,
        "inertias": inertias
    }
    joblib.dump(metrics, "models/clustering_metrics.pkl")

    final_model = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    clusters = final_model.fit_predict(rfm_scaled)
    rfm["Cluster"] = clusters

    joblib.dump(final_model, "models/kmeans_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    joblib.dump(rfm, "models/rfm_with_clusters.pkl")

    return rfm, final_model, scaler
