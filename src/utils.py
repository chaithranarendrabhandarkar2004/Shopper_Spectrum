import numpy as np


def get_dynamic_cluster_labels(kmeans, scaler):
    """
    Dynamically profile and assign correct semantic labels to the 4 clusters
    based on their RFM characteristics, matching the exact required PDF naming:
    - High-Value (High R, High F, High M)
    - Regular (Medium F, Medium M)
    - Occasional (Low F, Low M, older R)
    - At-Risk (High R, Low F, Low M)
    """
    centers_scaled = kmeans.cluster_centers_
    centers_original = scaler.inverse_transform(centers_scaled)

    # Sort cluster IDs by their mean monetary value (index 2)
    monetary_values = centers_original[:, 2]
    sorted_cluster_ids = np.argsort(monetary_values)

    labels = {}
    if len(sorted_cluster_ids) == 4:
        # Map based on ascending monetary rank
        labels[sorted_cluster_ids[0]] = "Occasional"
        labels[sorted_cluster_ids[1]] = "At-Risk"
        labels[sorted_cluster_ids[2]] = "Regular"
        labels[sorted_cluster_ids[3]] = "High-Value"
    else:
        # Fallback if different number of clusters is trained
        for rank, cid in enumerate(sorted_cluster_ids):
            if rank == 0:
                labels[cid] = "Occasional"
            elif rank == 1:
                labels[cid] = "At-Risk"
            elif rank == 2:
                labels[cid] = "Regular"
            else:
                labels[cid] = "High-Value"

    return labels