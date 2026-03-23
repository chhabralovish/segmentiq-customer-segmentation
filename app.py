import streamlit as st
import pandas as pd
import numpy as np
import os
from preprocessor import load_and_preprocess, apply_pca, get_feature_summary
from segmentation import run_all_algorithms, elbow_method, tune_gmm, get_dendrogram_data, get_best_algorithm
from profiler import profile_segments, get_segment_comparison_df
from feature_importance import (
    get_pca_feature_importance, get_mutual_information,
    get_random_forest_importance, get_shap_importance,
    get_cluster_feature_means, combined_importance
)
from visualizer import (
    plot_pca_2d, plot_pca_3d, plot_elbow, plot_algorithm_comparison,
    plot_radar, plot_cluster_sizes, plot_feature_importance,
    plot_feature_distributions, plot_correlation_heatmap
)
from report_generator import generate_pdf_report

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SegmentIQ — Customer Segmentation",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Dark Theme CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2E86AB;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-value { font-size: 32px; font-weight: bold; color: #2E86AB; }
    .metric-label { font-size: 12px; color: #aaaaaa; margin-top: 4px; }
    .segment-card {
        background: #1a1a2e;
        border-left: 4px solid #2E86AB;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }
    .priority-badge {
        background: #F18F01;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: bold;
    }
    h1, h2, h3 { color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 20px 0'>
    <h1 style='font-size:3em; margin:0'>🎯 SegmentIQ</h1>
    <p style='color:#aaaaaa; font-size:1.1em'>
        AI-Powered Customer Segmentation | K-Means · DBSCAN · Hierarchical · GMM
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    uploaded_file = st.file_uploader(
        "Upload Customer CSV",
        type=["csv"],
        help="Upload customer data CSV file"
    )

    use_sample = st.checkbox("Use sample cruise dataset", value=True)

    st.divider()
    st.subheader("Clustering Settings")

    n_clusters = st.slider("Number of Clusters (K)", min_value=2, max_value=10, value=4)
    auto_k = st.checkbox("Auto-detect optimal K", value=True)

    st.divider()
    st.subheader("Feature Importance")
    run_shap = st.checkbox("Run SHAP Analysis (slower)", value=False)

    st.divider()
    st.markdown("Built by [Lovish Chhabra](https://www.linkedin.com/in/lovish-chhabra/)")

# ── Load Data ─────────────────────────────────────────────────────────────────
data_path = None
if uploaded_file:
    data_path = uploaded_file
elif use_sample:
    data_path = "sample_data/customers.csv"

if not data_path:
    st.info("👈 Upload a CSV file or enable the sample dataset to get started.")
    st.stop()

# ── Preprocess ────────────────────────────────────────────────────────────────
with st.spinner("Loading and preprocessing data..."):
    try:
        preprocessed = load_and_preprocess(data_path)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

X_scaled = preprocessed["X_scaled"]
feature_names = preprocessed["feature_names"]
clean_df = preprocessed["clean_df"]
original_df = preprocessed["original_df"]

# ── Auto K Detection ──────────────────────────────────────────────────────────
with st.spinner("Finding optimal K..."):
    elbow_data = elbow_method(X_scaled)
    if auto_k:
        n_clusters = elbow_data["optimal_k"]

# ── PCA ───────────────────────────────────────────────────────────────────────
pca_result = apply_pca(X_scaled, n_components=3)

# ── Run All Algorithms ────────────────────────────────────────────────────────
with st.spinner("Running all clustering algorithms..."):
    algo_results = run_all_algorithms(X_scaled, n_clusters)

metrics_list = [v["metrics"] for v in algo_results.values()]
best_algo = get_best_algorithm(metrics_list)
best_labels = algo_results[best_algo]["labels"]

# ── Profile Segments ──────────────────────────────────────────────────────────
profiles = profile_segments(original_df, clean_df, best_labels, feature_names)
cluster_means = get_cluster_feature_means(clean_df, best_labels, feature_names)

# ── Feature Importance ────────────────────────────────────────────────────────
with st.spinner("Calculating feature importance..."):
    pca_imp = get_pca_feature_importance(pca_result["pca"], feature_names)
    mi_imp = get_mutual_information(X_scaled, best_labels, feature_names)
    rf_imp = get_random_forest_importance(X_scaled, best_labels, feature_names)

    shap_df, shap_vals = None, None
    if run_shap:
        shap_df, shap_vals = get_shap_importance(X_scaled, best_labels, feature_names)

    combined_imp = combined_importance(pca_imp, mi_imp, rf_imp, feature_names)

# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "🔬 Clustering",
    "👥 Segments",
    "🔍 Feature Importance",
    "📈 Visualizations",
    "📄 Export Report"
])

# ── TAB 1: Overview ───────────────────────────────────────────────────────────
with tab1:
    st.subheader("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{len(original_df):,}</div>
            <div class='metric-label'>Total Customers</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{len(feature_names)}</div>
            <div class='metric-label'>Features</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{len(profiles)}</div>
            <div class='metric-label'>Segments Found</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        best_sil = next((m["silhouette_score"] for m in metrics_list if m["algorithm"] == best_algo), "N/A")
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>{best_sil}</div>
            <div class='metric-label'>Best Silhouette Score</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Raw Data Preview")
        st.dataframe(original_df.head(10), use_container_width=True)
    with col2:
        st.subheader("Feature Summary")
        summary = get_feature_summary(clean_df, feature_names)
        st.dataframe(summary, use_container_width=True)

    st.divider()
    st.subheader("Feature Correlation Heatmap")
    st.plotly_chart(
        plot_correlation_heatmap(clean_df, feature_names),
        use_container_width=True
    )

# ── TAB 2: Clustering ─────────────────────────────────────────────────────────
with tab2:
    st.subheader("Algorithm Comparison")
    st.success(f"🏆 Best Algorithm: **{best_algo}** (highest silhouette score)")

    metrics_df = pd.DataFrame(metrics_list)
    st.dataframe(metrics_df, use_container_width=True)

    st.plotly_chart(
        plot_algorithm_comparison(metrics_list),
        use_container_width=True
    )

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Elbow Method")
        st.plotly_chart(plot_elbow(elbow_data), use_container_width=True)
    with col2:
        st.subheader("PCA 2D Visualization")
        st.plotly_chart(
            plot_pca_2d(pca_result["X_pca"], best_labels, profiles),
            use_container_width=True
        )

    st.divider()
    st.subheader("PCA Explained Variance")
    exp_var = pca_result["explained_variance"]
    cum_var = pca_result["cumulative_variance"]
    var_df = pd.DataFrame({
        "Component": [f"PC{i+1}" for i in range(len(exp_var))],
        "Explained Variance": [f"{v*100:.1f}%" for v in exp_var],
        "Cumulative Variance": [f"{v*100:.1f}%" for v in cum_var]
    })
    st.dataframe(var_df, use_container_width=True)

# ── TAB 3: Segments ───────────────────────────────────────────────────────────
with tab3:
    st.subheader("Segment Profiles")
    st.plotly_chart(plot_cluster_sizes(profiles), use_container_width=True)

    st.divider()
    comparison_df = get_segment_comparison_df(profiles, feature_names)
    st.dataframe(comparison_df, use_container_width=True)

    st.divider()
    for p in profiles:
        with st.expander(f"{p['icon']} {p['segment_name']} — {p['size']} customers ({p['percentage']}%)"):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown(f"**Description:** {p['description']}")
                st.markdown(f"**Priority:** `{p['priority']}`")
                st.markdown("**Key Metrics:**")
                for feat in feature_names[:6]:
                    if feat in p["means"]:
                        st.markdown(f"- {feat.replace('_',' ').title()}: **{p['means'][feat]:.1f}**")
            with col2:
                st.markdown("**Recommended Actions:**")
                for action in p["actions"]:
                    st.markdown(f"✅ {action}")

# ── TAB 4: Feature Importance ─────────────────────────────────────────────────
with tab4:
    st.subheader("Feature Importance Analysis")

    imp_tab1, imp_tab2, imp_tab3, imp_tab4 = st.tabs([
        "Combined Score", "PCA Loadings", "Mutual Information", "Random Forest"
    ])

    with imp_tab1:
        st.info("Combined score = average of PCA, Mutual Information, and Random Forest importance.")
        st.plotly_chart(
            plot_feature_importance(combined_imp, "Combined Feature Importance"),
            use_container_width=True
        )
        st.dataframe(combined_imp, use_container_width=True)

    with imp_tab2:
        pca_display = pca_imp.reset_index().rename(columns={"index": "feature"})
        st.plotly_chart(
            plot_feature_importance(pca_display, "PCA Component Loadings"),
            use_container_width=True
        )

    with imp_tab3:
        st.plotly_chart(
            plot_feature_importance(mi_imp.rename(columns={"mutual_info": "combined_score"}),
                                    "Mutual Information Score"),
            use_container_width=True
        )

    with imp_tab4:
        st.plotly_chart(
            plot_feature_importance(rf_imp.rename(columns={"rf_importance": "combined_score"}),
                                    "Random Forest Feature Importance"),
            use_container_width=True
        )

    if run_shap and shap_df is not None:
        st.divider()
        st.subheader("SHAP Feature Importance")
        st.plotly_chart(
            plot_feature_importance(shap_df.rename(columns={"shap_importance": "combined_score"}),
                                    "SHAP Values"),
            use_container_width=True
        )
    elif run_shap:
        st.warning("SHAP analysis failed. Make sure `shap` is installed: pip install shap")

    st.divider()
    st.subheader("Radar Chart — Feature Profiles per Segment")
    st.plotly_chart(
        plot_radar(cluster_means, feature_names, profiles),
        use_container_width=True
    )

# ── TAB 5: Visualizations ─────────────────────────────────────────────────────
with tab5:
    st.subheader("3D PCA Visualization")
    fig_3d = plot_pca_3d(pca_result["X_pca"], best_labels, profiles)
    if fig_3d:
        st.plotly_chart(fig_3d, use_container_width=True)
    else:
        st.info("3D visualization requires at least 3 features.")

    st.divider()
    st.subheader("Feature Distributions per Segment")
    st.plotly_chart(
        plot_feature_distributions(clean_df, best_labels, feature_names, profiles),
        use_container_width=True
    )

# ── TAB 6: Export ─────────────────────────────────────────────────────────────
with tab6:
    st.subheader("Export Report")
    st.markdown("""
    Generate a professional PDF report with:
    - Executive Summary
    - Algorithm Comparison Table
    - Segment Profiles with Recommendations
    - Feature Importance Rankings
    - Key Metrics per Segment
    """)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📄 Generate PDF Report", type="primary", use_container_width=True):
            with st.spinner("Generating PDF report..."):
                try:
                    pdf_buffer = generate_pdf_report(
                        profiles=profiles,
                        metrics_list=metrics_list,
                        best_algorithm=best_algo,
                        feature_importance_df=combined_imp,
                        n_customers=len(original_df),
                        feature_names=feature_names
                    )
                    st.download_button(
                        label="⬇️ Download PDF Report",
                        data=pdf_buffer,
                        file_name=f"segmentiq_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    st.success("✅ PDF report generated successfully!")
                except Exception as e:
                    st.error(f"Error generating PDF: {str(e)}")
                    st.info("Make sure reportlab is installed: pip install reportlab")

    with col2:
        if st.button("📊 Export Segment Data (CSV)", use_container_width=True):
            export_df = original_df.copy()
            name_map = {p["cluster_id"]: p["segment_name"] for p in profiles}
            export_df["segment_id"] = best_labels
            export_df["segment_name"] = export_df["segment_id"].map(name_map)
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download CSV",
                data=csv,
                file_name="segmented_customers.csv",
                mime="text/csv",
                use_container_width=True
            )