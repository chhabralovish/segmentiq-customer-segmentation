import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


COLORS = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D",
          "#3B1F2B", "#44BBA4", "#E94F37", "#393E41"]


# ── PCA Scatter Plot ──────────────────────────────────────────────────────────
def plot_pca_2d(X_pca, labels, profiles, title="PCA 2D Cluster Visualization"):
    """2D scatter plot of clusters in PCA space."""
    df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "Cluster": labels.astype(str)
    })

    # Add segment names
    name_map = {str(p["cluster_id"]): f"{p['icon']} {p['segment_name']}" for p in profiles}
    df["Segment"] = df["Cluster"].map(name_map).fillna("Noise")

    fig = px.scatter(
        df, x="PC1", y="PC2", color="Segment",
        title=title,
        color_discrete_sequence=COLORS,
        template="plotly_dark",
        opacity=0.8
    )
    fig.update_traces(marker=dict(size=8))
    fig.update_layout(height=500, legend_title="Segment")
    return fig


def plot_pca_3d(X_pca, labels, profiles):
    """3D scatter plot of clusters in PCA space."""
    if X_pca.shape[1] < 3:
        return None

    df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "PC3": X_pca[:, 2],
        "Cluster": labels.astype(str)
    })
    name_map = {str(p["cluster_id"]): f"{p['icon']} {p['segment_name']}" for p in profiles}
    df["Segment"] = df["Cluster"].map(name_map).fillna("Noise")

    fig = px.scatter_3d(
        df, x="PC1", y="PC2", z="PC3",
        color="Segment",
        title="PCA 3D Cluster Visualization",
        color_discrete_sequence=COLORS,
        template="plotly_dark",
        opacity=0.8
    )
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(height=600)
    return fig


# ── Elbow Curve ───────────────────────────────────────────────────────────────
def plot_elbow(elbow_data):
    """Plot elbow curve with inertia and silhouette scores."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=elbow_data["k_range"], y=elbow_data["inertias"],
        name="Inertia", line=dict(color="#2E86AB", width=2),
        marker=dict(size=8)
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=elbow_data["k_range"], y=elbow_data["silhouettes"],
        name="Silhouette Score", line=dict(color="#F18F01", width=2, dash="dot"),
        marker=dict(size=8)
    ), secondary_y=True)

    # Highlight optimal K
    opt_k = elbow_data["optimal_k"]
    fig.add_vline(
        x=opt_k, line_dash="dash", line_color="red",
        annotation_text=f"Optimal K={opt_k}"
    )

    fig.update_layout(
        title="Elbow Method — Finding Optimal K",
        template="plotly_dark", height=400,
        xaxis_title="Number of Clusters (K)"
    )
    fig.update_yaxes(title_text="Inertia", secondary_y=False)
    fig.update_yaxes(title_text="Silhouette Score", secondary_y=True)
    return fig


# ── Algorithm Comparison ──────────────────────────────────────────────────────
def plot_algorithm_comparison(metrics_list):
    """Compare all algorithms on silhouette, DB, and CH scores."""
    df = pd.DataFrame(metrics_list).dropna(subset=["silhouette_score"])

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Silhouette Score (↑ better)",
                        "Davies-Bouldin (↓ better)",
                        "Calinski-Harabasz (↑ better)"]
    )

    colors = COLORS[:len(df)]

    fig.add_trace(go.Bar(
        x=df["algorithm"], y=df["silhouette_score"],
        marker_color=colors, name="Silhouette", showlegend=False
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=df["algorithm"], y=df["davies_bouldin_score"],
        marker_color=colors, name="Davies-Bouldin", showlegend=False
    ), row=1, col=2)

    fig.add_trace(go.Bar(
        x=df["algorithm"], y=df["calinski_harabasz_score"],
        marker_color=colors, name="Calinski-Harabasz", showlegend=False
    ), row=1, col=3)

    fig.update_layout(
        title="Algorithm Comparison — All Metrics",
        template="plotly_dark", height=400
    )
    return fig


# ── Radar Chart ───────────────────────────────────────────────────────────────
def plot_radar(cluster_means, feature_names, profiles):
    """Spider/Radar chart per segment showing feature profiles."""
    # Normalize to 0-1 for radar
    means_df = cluster_means.copy()
    for col in means_df.columns:
        col_range = means_df[col].max() - means_df[col].min()
        if col_range > 0:
            means_df[col] = (means_df[col] - means_df[col].min()) / col_range

    fig = go.Figure()
    name_map = {p["cluster_id"]: f"{p['icon']} {p['segment_name']}" for p in profiles}

    for i, cluster_id in enumerate(means_df.index):
        if cluster_id == -1:
            continue
        values = means_df.loc[cluster_id, feature_names].tolist()
        values += values[:1]  # Close the radar
        categories = feature_names + [feature_names[0]]

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=name_map.get(cluster_id, f"Segment {cluster_id}"),
            line_color=COLORS[i % len(COLORS)],
            opacity=0.7
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Feature Radar Chart — Segment Profiles",
        template="plotly_dark", height=500
    )
    return fig


# ── Cluster Size Chart ────────────────────────────────────────────────────────
def plot_cluster_sizes(profiles):
    """Pie + bar showing cluster size distribution."""
    names = [f"{p['icon']} {p['segment_name']}" for p in profiles]
    sizes = [p["size"] for p in profiles]

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "bar"}]])

    fig.add_trace(go.Pie(
        labels=names, values=sizes,
        marker_colors=COLORS[:len(profiles)],
        hole=0.4, name="Distribution"
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=names, y=sizes,
        marker_color=COLORS[:len(profiles)],
        name="Count", showlegend=False
    ), row=1, col=2)

    fig.update_layout(
        title="Segment Size Distribution",
        template="plotly_dark", height=400
    )
    return fig


# ── Feature Importance Bar Chart ──────────────────────────────────────────────
def plot_feature_importance(importance_df, title="Feature Importance"):
    """Horizontal bar chart of feature importance scores."""
    df = importance_df.head(15)
    score_col = "combined_score" if "combined_score" in df.columns else df.columns[-1]

    fig = px.bar(
        df.sort_values(score_col),
        x=score_col, y="feature",
        orientation='h',
        title=title,
        color=score_col,
        color_continuous_scale="Blues",
        template="plotly_dark"
    )
    fig.update_layout(height=450, showlegend=False)
    return fig


# ── Box Plots per Feature per Cluster ────────────────────────────────────────
def plot_feature_distributions(clean_df, labels, feature_names, profiles, top_n=6):
    """Box plots showing feature distribution per cluster."""
    df = clean_df[feature_names].copy()
    df["Segment"] = labels

    name_map = {p["cluster_id"]: f"{p['icon']} {p['segment_name']}" for p in profiles}
    df["Segment"] = df["Segment"].map(name_map).fillna("Noise")
    df = df[df["Segment"] != "Noise"]

    top_features = feature_names[:min(top_n, len(feature_names))]
    n_cols = 3
    n_rows = (len(top_features) + n_cols - 1) // n_cols

    fig = make_subplots(rows=n_rows, cols=n_cols,
                        subplot_titles=top_features)

    for i, feat in enumerate(top_features):
        row = i // n_cols + 1
        col = i % n_cols + 1
        for j, seg in enumerate(df["Segment"].unique()):
            seg_data = df[df["Segment"] == seg][feat]
            fig.add_trace(go.Box(
                y=seg_data, name=seg,
                marker_color=COLORS[j % len(COLORS)],
                showlegend=(i == 0)
            ), row=row, col=col)

    fig.update_layout(
        title="Feature Distributions per Segment",
        template="plotly_dark",
        height=300 * n_rows,
        boxmode="group"
    )
    return fig


# ── Dendrogram ────────────────────────────────────────────────────────────────
def plot_dendrogram(Z):
    """Plot hierarchical clustering dendrogram."""
    from scipy.cluster.hierarchy import dendrogram as sp_dendrogram
    import plotly.figure_factory as ff

    fig = ff.create_dendrogram(
        np.array([[0]] * len(Z)),
        linkagefun=lambda x: Z,
        colorscale=COLORS[:5]
    )
    fig.update_layout(
        title="Hierarchical Clustering Dendrogram",
        template="plotly_dark",
        height=400
    )
    return fig


# ── Correlation Heatmap ───────────────────────────────────────────────────────
def plot_correlation_heatmap(clean_df, feature_names):
    """Correlation heatmap of all features."""
    corr = clean_df[feature_names].corr()
    fig = px.imshow(
        corr,
        title="Feature Correlation Heatmap",
        color_continuous_scale="RdBu_r",
        template="plotly_dark",
        aspect="auto"
    )
    fig.update_layout(height=500)
    return fig