import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage


# ── K-Means ───────────────────────────────────────────────────────────────────
def run_kmeans(X_scaled, n_clusters=4, random_state=42):
    """Run K-Means clustering."""
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X_scaled)
    return labels, model


def elbow_method(X_scaled, max_k=10):
    """Calculate inertia for different K values."""
    inertias, silhouettes = [], []
    k_range = range(2, min(max_k + 1, len(X_scaled)))

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        if len(set(labels)) > 1:
            silhouettes.append(silhouette_score(X_scaled, labels))
        else:
            silhouettes.append(0)

    # Find optimal K using silhouette
    optimal_k = list(k_range)[np.argmax(silhouettes)]

    return {
        "k_range": list(k_range),
        "inertias": inertias,
        "silhouettes": silhouettes,
        "optimal_k": optimal_k
    }


# ── DBSCAN ────────────────────────────────────────────────────────────────────
def run_dbscan(X_scaled, eps=0.8, min_samples=3):
    """Run DBSCAN clustering."""
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    return labels, model, n_clusters, n_noise


def tune_dbscan(X_scaled):
    """Find best eps and min_samples for DBSCAN."""
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=4).fit(X_scaled)
    distances, _ = nbrs.kneighbors(X_scaled)
    distances = np.sort(distances[:, 3])
    # Use knee point as eps
    eps_candidates = np.percentile(distances, [50, 65, 75, 85])

    best_score, best_eps, best_min = -1, 0.8, 3
    for eps in eps_candidates:
        for min_s in [3, 4, 5]:
            labels, _, n_cl, _ = run_dbscan(X_scaled, eps=eps, min_samples=min_s)
            if n_cl >= 2 and n_cl <= 8:
                valid = labels != -1
                if valid.sum() > 10:
                    score = silhouette_score(X_scaled[valid], labels[valid])
                    if score > best_score:
                        best_score, best_eps, best_min = score, eps, min_s

    return best_eps, best_min


# ── Hierarchical Clustering ───────────────────────────────────────────────────
def run_hierarchical(X_scaled, n_clusters=4, linkage_method='ward'):
    """Run Agglomerative Hierarchical Clustering."""
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    labels = model.fit_predict(X_scaled)
    return labels, model


def get_dendrogram_data(X_scaled, method='ward'):
    """Get linkage matrix for dendrogram."""
    Z = linkage(X_scaled[:min(50, len(X_scaled))], method=method)
    return Z


# ── Gaussian Mixture Model ────────────────────────────────────────────────────
def run_gmm(X_scaled, n_components=4, random_state=42):
    """Run Gaussian Mixture Model clustering."""
    model = GaussianMixture(
        n_components=n_components,
        random_state=random_state,
        covariance_type='full',
        n_init=3
    )
    model.fit(X_scaled)
    labels = model.predict(X_scaled)
    probs = model.predict_proba(X_scaled)
    return labels, model, probs


def tune_gmm(X_scaled, max_components=8):
    """Find optimal number of components using BIC score."""
    bic_scores, aic_scores = [], []
    comp_range = range(2, min(max_components + 1, len(X_scaled) // 5))

    for n in comp_range:
        gmm = GaussianMixture(n_components=n, random_state=42, n_init=3)
        gmm.fit(X_scaled)
        bic_scores.append(gmm.bic(X_scaled))
        aic_scores.append(gmm.aic(X_scaled))

    optimal_n = list(comp_range)[np.argmin(bic_scores)]
    return {
        "comp_range": list(comp_range),
        "bic_scores": bic_scores,
        "aic_scores": aic_scores,
        "optimal_n": optimal_n
    }


# ── Evaluation Metrics ────────────────────────────────────────────────────────
def evaluate_clustering(X_scaled, labels, algorithm_name):
    """Calculate all clustering evaluation metrics."""
    # Filter out noise points for DBSCAN
    valid_mask = labels != -1
    X_valid = X_scaled[valid_mask]
    labels_valid = labels[valid_mask]

    n_clusters = len(set(labels_valid))

    if n_clusters < 2 or len(X_valid) < n_clusters + 1:
        return {
            "algorithm": algorithm_name,
            "n_clusters": n_clusters,
            "silhouette_score": None,
            "davies_bouldin_score": None,
            "calinski_harabasz_score": None,
            "noise_points": int((labels == -1).sum())
        }

    sil = silhouette_score(X_valid, labels_valid)
    db = davies_bouldin_score(X_valid, labels_valid)
    ch = calinski_harabasz_score(X_valid, labels_valid)

    return {
        "algorithm": algorithm_name,
        "n_clusters": n_clusters,
        "silhouette_score": round(sil, 4),
        "davies_bouldin_score": round(db, 4),
        "calinski_harabasz_score": round(ch, 2),
        "noise_points": int((labels == -1).sum())
    }


def get_best_algorithm(metrics_list):
    """Recommend best algorithm based on silhouette score."""
    valid = [m for m in metrics_list if m["silhouette_score"] is not None]
    if not valid:
        return metrics_list[0]["algorithm"]
    best = max(valid, key=lambda x: x["silhouette_score"])
    return best["algorithm"]


# ── Run All Algorithms ────────────────────────────────────────────────────────
def run_all_algorithms(X_scaled, n_clusters=4):
    """Run all 4 clustering algorithms and return results."""
    results = {}

    # 1. K-Means
    km_labels, km_model = run_kmeans(X_scaled, n_clusters)
    results["K-Means"] = {
        "labels": km_labels,
        "model": km_model,
        "metrics": evaluate_clustering(X_scaled, km_labels, "K-Means")
    }

    # 2. DBSCAN
    best_eps, best_min = tune_dbscan(X_scaled)
    db_labels, db_model, n_cl, n_noise = run_dbscan(X_scaled, best_eps, best_min)
    results["DBSCAN"] = {
        "labels": db_labels,
        "model": db_model,
        "metrics": evaluate_clustering(X_scaled, db_labels, "DBSCAN"),
        "eps": best_eps,
        "min_samples": best_min
    }

    # 3. Hierarchical
    hc_labels, hc_model = run_hierarchical(X_scaled, n_clusters)
    results["Hierarchical"] = {
        "labels": hc_labels,
        "model": hc_model,
        "metrics": evaluate_clustering(X_scaled, hc_labels, "Hierarchical")
    }

    # 4. GMM
    gmm_labels, gmm_model, gmm_probs = run_gmm(X_scaled, n_clusters)
    results["GMM"] = {
        "labels": gmm_labels,
        "model": gmm_model,
        "probs": gmm_probs,
        "metrics": evaluate_clustering(X_scaled, gmm_labels, "GMM")
    }

    return results