import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Text Clustering Topology Dashboard", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .main-header { font-size:2.2rem !important; color: #2c3e50; font-weight: 700; }
    .metric-card { background-color: #f8f9fa; border-left: 5px solid #f1c40f; padding: 1rem; border-radius: 4px; margin-bottom: 1rem; }
    .metric-title { font-size: 0.85rem; color: #7f8c8d; font-weight: 600; text-transform: uppercase; }
    .metric-value { font-size: 1.6rem; color: #2c3e50; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📊 Cross-Dataset Clustering Feature Topology App</div>', unsafe_allow_html=True)
st.write("Exploratory interface analyzing model-specific text vector configurations at K=20.")

# Load Data Block
@st.cache_data
def load_data():
    try:
        return pd.read_csv('comprehensive_clustering_experiments.csv')
    except Exception:
        st.warning("⚠️ Experiment file data matrix not found. Rendering fallback simulation rows...")
        models, datasets = ['K-Means', 'GMM', 'Hierarchical'], ['20 Newsgroups', 'Wikipedia Corpus']
        mock_data = []
        for ds in datasets:
            for m in models:
                for mx in [0.7, 0.9]:
                    for mn in [2, 5]:
                        for f in [200, 500, 1000, 10000]:
                            val = 0.45 if f <= 500 and m == 'K-Means' else 0.38
                            if m == 'Hierarchical' and f == 10000: val = 0.49
                            mock_data.append({'dataset': ds, 'model_family': m, 'max_df': mx, 'min_df': mn, 'max_features': f, 'silhouette_score': val + np.random.uniform(-0.02, 0.02)})
        return pd.DataFrame(mock_data)

df = load_data()

# Sidebar Configuration
st.sidebar.header("🕹️ Filter Options")
dataset_choice = st.sidebar.selectbox("Select Active Corpus", options=df['dataset'].unique() if not df.empty else ["No Data Found"])

df_filtered = df[df['dataset'] == dataset_choice] if not df.empty else pd.DataFrame()

st.markdown(f"### Current Corpus View: **{dataset_choice}**")

# Metric Summary Cards Layout
if not df_filtered.empty:
    col1, col2, col3 = st.columns(3)
    for idx, model in enumerate(['K-Means', 'GMM', 'Hierarchical']):
        df_m = df_filtered[df_filtered['model_family'] == model]
        if not df_m.empty:
            best_row = df_m.loc[df_m['silhouette_score'].idxmax()]
            cols = [col1, col2, col3]
            with cols[idx]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">{model} Peak Silhouette</div>
                    <div class="metric-value">{best_row['silhouette_score']:.4f}</div>
                    <p style="margin:0.2rem 0 0 0; font-size:0.85rem; color:#34495e;">
                        max_df: {best_row['max_df']} | min_df: {best_row['min_df']} <br>
                        <b>Optimal Features: <span style="color:#b7950b;">{best_row['max_features']}</span></b>
                    </p>
                </div>
                """, unsafe_allow_html=True)

st.markdown("---")
tab1, tab2 = st.tabs(["📈 Parameter Influences", "📋 Comprehensive Raw Logs"])

with tab1:
    if not df_filtered.empty:
        g_col1, g_col2, g_col3 = st.columns(3)
        g_cols = [g_col1, g_col2, g_col3]
        params = ['max_df', 'min_df', 'max_features']
        
        for idx, p in enumerate(params):
            with g_cols[idx]:
                stats = df_filtered.groupby(p)['silhouette_score'].mean().reset_index()
                stats[p] = stats[p].astype(str)
                
                max_pos = stats['silhouette_score'].idxmax()
                colors = ['#2c3e50'] * len(stats)
                colors[max_pos] = '#f1c40f' # Highlight winning bar in gold
                
                fig = px.bar(stats, x=p, y='silhouette_score', text_auto='.4f', title=f"Mean Score vs {p}")
                fig.update_traces(marker_color=colors, marker_line_color='black', marker_line_width=1)
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)

with tab2:
    if not df_filtered.empty:
        st.dataframe(df_filtered.sort_values(by='silhouette_score', ascending=False), use_container_width=True)