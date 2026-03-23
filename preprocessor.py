import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA


def load_and_preprocess(file_or_path):
    """Load CSV and preprocess for clustering."""
    if hasattr(file_or_path, 'read'):
        df = pd.read_csv(file_or_path)
    else:
        df = pd.read_csv(file_or_path)

    original_df = df.copy()

    # Drop ID-like columns
    id_cols = [c for c in df.columns if any(x in c.lower() for x in ['id', 'name', 'email', 'phone'])]
    df_clean = df.drop(columns=id_cols, errors='ignore')

    # Encode categorical columns
    label_encoders = {}
    cat_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))
        label_encoders[col] = le

    # Fill missing values
    df_clean = df_clean.fillna(df_clean.median(numeric_only=True))

    # Scale features
    scaler = StandardScaler()
    feature_names = df_clean.columns.tolist()
    X_scaled = scaler.fit_transform(df_clean)

    return {
        "original_df": original_df,
        "clean_df": df_clean,
        "X_scaled": X_scaled,
        "feature_names": feature_names,
        "scaler": scaler,
        "label_encoders": label_encoders,
        "id_cols": id_cols
    }


def apply_pca(X_scaled, n_components=3):
    """Apply PCA for dimensionality reduction and visualization."""
    n_comp = min(n_components, X_scaled.shape[1])
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X_scaled)

    return {
        "pca": pca,
        "X_pca": X_pca,
        "explained_variance": pca.explained_variance_ratio_,
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
        "components": pca.components_
    }


def get_feature_summary(clean_df, feature_names):
    """Get basic summary stats per feature."""
    summary = []
    for feat in feature_names:
        if feat in clean_df.columns:
            summary.append({
                "feature": feat,
                "mean": clean_df[feat].mean(),
                "std": clean_df[feat].std(),
                "min": clean_df[feat].min(),
                "max": clean_df[feat].max()
            })
    return pd.DataFrame(summary)