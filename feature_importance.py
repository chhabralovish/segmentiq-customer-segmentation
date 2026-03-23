import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA


def get_pca_feature_importance(pca, feature_names, n_components=2):
    """Get feature importance from PCA component loadings."""
    n_comp = min(n_components, len(pca.components_))
    loadings = pd.DataFrame(
        pca.components_[:n_comp].T,
        index=feature_names,
        columns=[f"PC{i+1}" for i in range(n_comp)]
    )
    # Overall importance = sum of absolute loadings weighted by explained variance
    weights = pca.explained_variance_ratio_[:n_comp]
    loadings["importance"] = np.abs(loadings.values * weights).sum(axis=1)
    loadings = loadings.sort_values("importance", ascending=False)
    return loadings


def get_mutual_information(X_scaled, labels, feature_names):
    """Calculate mutual information between features and cluster labels."""
    # Filter noise points
    valid = labels != -1
    X_valid = X_scaled[valid]
    labels_valid = labels[valid]

    if len(set(labels_valid)) < 2:
        return pd.DataFrame({"feature": feature_names, "mutual_info": [0] * len(feature_names)})

    mi_scores = mutual_info_classif(X_valid, labels_valid, random_state=42)
    mi_df = pd.DataFrame({
        "feature": feature_names,
        "mutual_info": mi_scores
    }).sort_values("mutual_info", ascending=False)
    return mi_df


def get_random_forest_importance(X_scaled, labels, feature_names):
    """Use Random Forest to get feature importance for cluster prediction."""
    valid = labels != -1
    X_valid = X_scaled[valid]
    labels_valid = labels[valid]

    if len(set(labels_valid)) < 2:
        return pd.DataFrame({"feature": feature_names, "rf_importance": [0] * len(feature_names)})

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_valid, labels_valid)

    rf_df = pd.DataFrame({
        "feature": feature_names,
        "rf_importance": rf.feature_importances_
    }).sort_values("rf_importance", ascending=False)
    return rf_df


def get_shap_importance(X_scaled, labels, feature_names):
    """Calculate SHAP values for cluster explainability."""
    try:
        import shap
        valid = labels != -1
        X_valid = X_scaled[valid]
        labels_valid = labels[valid]

        if len(set(labels_valid)) < 2:
            return None, None

        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(X_valid, labels_valid)

        # Use TreeExplainer for speed
        explainer = shap.TreeExplainer(rf)
        sample = X_valid[:min(100, len(X_valid))]
        shap_values = explainer.shap_values(sample)

        # Mean absolute SHAP per feature
        if isinstance(shap_values, list):
            mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            mean_shap = np.abs(shap_values).mean(axis=0)

        shap_df = pd.DataFrame({
            "feature": feature_names,
            "shap_importance": mean_shap
        }).sort_values("shap_importance", ascending=False)

        return shap_df, shap_values

    except ImportError:
        return None, None
    except Exception:
        return None, None


def get_cluster_feature_means(clean_df, labels, feature_names):
    """Get mean value of each feature per cluster — for radar charts."""
    df = clean_df[feature_names].copy()
    df["cluster"] = labels
    cluster_means = df.groupby("cluster")[feature_names].mean()
    return cluster_means


def combined_importance(pca_imp, mi_imp, rf_imp, feature_names):
    """Combine all importance scores into a single ranked DataFrame."""
    df = pd.DataFrame({"feature": feature_names})

    if pca_imp is not None:
        pca_norm = pca_imp["importance"] / pca_imp["importance"].sum()
        df = df.merge(
            pca_imp[["importance"]].rename(columns={"importance": "pca_score"}).reset_index().rename(columns={"index": "feature"}),
            on="feature", how="left"
        )

    if mi_imp is not None:
        mi_norm = mi_imp.copy()
        if mi_norm["mutual_info"].sum() > 0:
            mi_norm["mi_score"] = mi_norm["mutual_info"] / mi_norm["mutual_info"].sum()
        else:
            mi_norm["mi_score"] = 0
        df = df.merge(mi_norm[["feature", "mi_score"]], on="feature", how="left")

    if rf_imp is not None:
        df = df.merge(
            rf_imp[["feature", "rf_importance"]].rename(columns={"rf_importance": "rf_score"}),
            on="feature", how="left"
        )

    # Fill NaN with 0
    df = df.fillna(0)

    # Combined score
    score_cols = [c for c in ["pca_score", "mi_score", "rf_score"] if c in df.columns]
    if score_cols:
        df["combined_score"] = df[score_cols].mean(axis=1)
        df = df.sort_values("combined_score", ascending=False)

    return df