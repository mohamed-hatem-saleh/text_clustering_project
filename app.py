import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

st.set_page_config(page_title="Text Clustering Topology Dashboard", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .main-header { font-size:2.2rem !important; color: #2c3e50; font-weight: 700; }
    .section-title { font-size:1.3rem !important; font-weight:600; color: #34495e; margin-top:1rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📊 Text Clustering Feature Topology Dashboard</div>', unsafe_allow_html=True)

# Safety loading function
@st.cache_data
def load_data_package():
    try:
        k_df = pd.read_csv("metrics_optimal_k.csv")
        grid_df = pd.read_csv("metrics_tfidf_grid.csv")
        tune_df = pd.read_csv("metrics_model_tuning.csv")
        spatial_df = pd.read_csv("metrics_spatial_layout.csv")
        return k_df, grid_df, tune_df, spatial_df
    except Exception:
            st.error("❌ Metrics files not found. Please run 'python run_pipeline.py' in your console first to execute calculations.")
            st.stop()

k_df, grid_df, tune_df, spatial_df = load_data_package()


# Sidebar controller filters
st.sidebar.header("🕹️ Filter Controls")
target_dataset = st.sidebar.selectbox("Select Target Corpus Domain Matrix:", options=k_df['Dataset'].unique())

# Filter master variables
k_sub = k_df[k_df['Dataset'] == target_dataset]
grid_sub = grid_df[grid_df['Dataset'] == target_dataset]
tune_sub = tune_df[tune_df['Dataset'] == target_dataset]
spatial_sub = spatial_df[spatial_df['Dataset'] == target_dataset]

# --- USER TABS STRUCTURE ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 1. Optimal K Diagnostics", 
    "🔮 2. TF-IDF Hyperparameter Grid", 
    "⚙️ 3. Base Model Parameter Tuning", 
    "🗺️ 4. t-SNE Maps & Dendrograms"
])

# -----------------------------------------------------------------
# TAB 1: ELBOW MATRIX PLOTS (DUAL AXIS VISUALS)
# -----------------------------------------------------------------
# --- TAB 1: ELBOW MATRIX PLOTS (DUAL AXIS VISUALS) ---
# --- TAB 1: ELBOW MATRIX PLOTS (DUAL AXIS VISUALS) ---
with tab1:
    st.markdown('<div class="section-title">Pre-Experiment Diagnostic: Elbow Method vs Silhouette Peaks</div>', unsafe_allow_html=True)
    st.write("Analyzes localized variance drops (Inertia) synchronized with absolute partitioning score sweeps.")
    
    fig_k = go.Figure()
    
    # 1. Left axis: Inertia (Electric Cyan)
    fig_k.add_trace(go.Scatter(
        x=k_sub['K'], 
        y=k_sub['Inertia'], 
        name="Inertia (Left Axis)", 
        mode="lines+markers",
        marker=dict(color='#00e5ff', size=8), 
        line=dict(width=3, color='#00e5ff')
    ))
    
    # 2. Dynamic Highlighting for the Optimal Silhouette Peak
    sil_values = k_sub['Silhouette'].tolist()
    k_values = k_sub['K'].tolist()
    
    if sil_values:
        max_sil = max(sil_values)
        max_idx = sil_values.index(max_sil)
        optimal_k = k_values[max_idx]
        
        # Base points get hot pink; the peak gets a high-contrast neon yellow
        color_array = ['#ff007f'] * len(sil_values)
        size_array = [8] * len(sil_values)
        
        # Override the peak point properties
        color_array[max_idx] = '#ffff00'
        size_array[max_idx] = 16  
        
        # 3. Right axis: Silhouette (Hot Pink with Neon Yellow Peak)
        fig_k.add_trace(go.Scatter(
            x=k_sub['K'], 
            y=k_sub['Silhouette'], 
            name="Silhouette (Right Axis)", 
            mode="lines+markers",
            marker=dict(
                color=color_array, 
                size=size_array,
                line=dict(width=2, color='#ffffff')  # White border for crisp separation
            ), 
            yaxis="y2",
            line=dict(width=3, color='#ff007f')  # Connecting line is hot pink
        ))
        
        # Helpful callout message in Streamlit
        st.info(f"💡 **Optimization Peak Detected:** The highest clustering quality is achieved at **K = {optimal_k}** with a Silhouette score of **{max_sil:.4f}**.")

    fig_k.update_layout(
        title=dict(
            text="Elbow Curve Optimization Trajectory Metrics",
            font=dict(color="#ffffff", size=16),
            y=0.95 # Shift title up slightly to make room
        ),
        xaxis=dict(
            title=dict(text="Number of Clusters (K)", font=dict(color="#ffffff")),
            tickfont=dict(color="#ffffff"),
            gridcolor="rgba(255, 255, 255, 0.1)",
            zeroline=False
        ),
        yaxis=dict(
            title=dict(text="Inertia / WCSS Value", font=dict(color="#00e5ff")),
            tickfont=dict(color="#00e5ff"),
            gridcolor="rgba(255, 255, 255, 0.1)",
            zeroline=False
        ),
        yaxis2=dict(
            title=dict(text="Silhouette Metrics Coefficient", font=dict(color="#ff007f")),
            tickfont=dict(color="#ff007f"),
            anchor="x",
            overlaying="y",
            side="right",
            zeroline=False
        ),
        # 🛡️ Relocating the legend completely outside the plot area
        legend=dict(
            font=dict(color="#ffffff"),
            bgcolor="rgba(0,0,0,0)",
            orientation="h",       # Horizontal layout
            yanchor="bottom",
            y=1.05,                # Places it cleanly above the chart grid lines
            xanchor="center",
            x=0.5                  # Centers it horizontally
        ),
        margin=dict(t=100),         # Adds top margin padding so the legend doesn't hit the title
        plot_bgcolor="#111111",
        paper_bgcolor="#111111"
    )
    st.plotly_chart(fig_k, use_container_width=True)

# -----------------------------------------------------------------
# TAB 2: VECTOR VOCABULARY OVER TUNING CHARTS
# -----------------------------------------------------------------
with tab2:
    st.markdown('<div class="section-title">TF-IDF Vectorizer Hyperparameter Grid Sweep</div>', unsafe_allow_html=True)
    st.write("Evaluates tokenization limits across various max_df bounds and feature cap dimensions.")
    
    # 🛡️ Swapped grid_search_records for your grid_df variable
    if not grid_df.empty:
        grid_sub = grid_df[grid_df['Dataset'] == target_dataset]
        
        # Custom color map using our signature dark-theme palette
        color_map = {
            'K-Means': '#00e5ff',      # Electric Cyan
            'Hierarchical': '#ff007f',  # Hot Pink
            'GMM': '#ffff00'           # Neon Yellow
        }
        
        fig_grid = px.bar(
            grid_sub,
            x='max_features',
            y='Silhouette',
            color='Model',
            barmode='group',
            facet_col='max_df',
            color_discrete_map=color_map,
            title="Silhouette Clustering Performance Across NLP Vector Extractions",
            labels={'max_features': 'Vocabulary Max Features Capping', 'Silhouette': 'Silhouette Score'}
        )
        
        # Apply premium dark mode overrides to the express object
        fig_grid.update_layout(
            plot_bgcolor="#111111",
            paper_bgcolor="#111111",
            font=dict(color="#ffffff"),
            title=dict(font=dict(color="#ffffff", size=16), y=0.95),
            legend=dict(
                font=dict(color="#ffffff"),
                bgcolor="rgba(0,0,0,0)",
                orientation="h",
                yanchor="bottom",
                y=1.1,
                xanchor="center",
                x=0.5
            ),
            margin=dict(t=110)
        )
        
        fig_grid.update_xaxes(showgrid=False, tickfont=dict(color="#ffffff"), title_font=dict(color="#ffffff"))
        fig_grid.update_yaxes(gridcolor="rgba(255, 255, 255, 0.1)", tickfont=dict(color="#ffffff"), title_font=dict(color="#ffffff"))
        
        st.plotly_chart(fig_grid, use_container_width=True)
    else:
        st.warning("No pipeline metrics dataset found for Phase 2 Grid Search logs.")

# -----------------------------------------------------------------
# TAB 3: INTERNAL MODEL PARAMETERS PERFORMANCE TUNING
# -----------------------------------------------------------------
with tab3:
    st.markdown('<div class="section-title">Internal Architectural Hyperparameter Tuning Profiles</div>', unsafe_allow_html=True)
    st.write("Isolates internal mathematical execution states across specific algorithmic parameter environments.")
    
    # 🛡️ Swapped model_tuning_records for your tune_df variable
    if not tune_df.empty:
        tune_sub = tune_df[tune_df['Dataset'] == target_dataset]
        
        color_map_tune = {
            'K-Means': '#00e5ff',
            'Hierarchical': '#ff007f',
            'GMM': '#ffff00'
        }
        
        fig_tune = px.bar(
            tune_sub,
            x='Parameter',
            y='Silhouette',
            color='Model',
            color_discrete_map=color_map_tune,
            text_auto='.4f', 
            title="Algorithmic Internal Component Parameter Comparison Sweeps",
            labels={'Parameter': 'Tested Parameter Configuration Setting', 'Silhouette': 'Clustering Quality (Silhouette)'}
        )
        
        fig_tune.update_layout(
            plot_bgcolor="#111111",
            paper_bgcolor="#111111",
            font=dict(color="#ffffff"),
            title=dict(font=dict(color="#ffffff", size=16), y=0.95),
            legend=dict(
                font=dict(color="#ffffff"),
                bgcolor="rgba(0,0,0,0)",
                orientation="h",
                yanchor="bottom",
                y=1.1,
                xanchor="center",
                x=0.5
            ),
            margin=dict(t=110)
        )
        
        fig_tune.update_xaxes(showgrid=False, tickfont=dict(color="#ffffff"), title_font=dict(color="#ffffff"))
        fig_tune.update_yaxes(gridcolor="rgba(255, 255, 255, 0.1)", tickfont=dict(color="#ffffff"), title_font=dict(color="#ffffff"))
        fig_tune.update_traces(textfont_color="#ffffff", textposition="outside")
        
        st.plotly_chart(fig_tune, use_container_width=True)
    else:
        st.warning("No hyperparameter tuning log files found for Phase 3 configuration matrices.")

# -----------------------------------------------------------------
# TAB 4: GEOMETRIC CLOUD PROJECTIONS & STRUCTURAL LEAF TREES
# -----------------------------------------------------------------
with tab4:
    st.markdown('### Definitive 2D Dimension Spatial Projections')
    
    # Selection widget to swap color shading fields on the fly
    chosen_model = st.selectbox(
        "Select Model Clustering Results to Display on t-SNE Map:",
        options=['K-Means_Cluster', 'GMM_Cluster', 'Hierarchical_Cluster']
    )
    
    fig_tsne = px.scatter(
        spatial_sub, x='tsne_x', y='tsne_y', 
        color=spatial_sub[chosen_model].astype(str),
        hover_data=['Snippet'],
        title=f"2D t-SNE Topological Document Map Shaded by {chosen_model}",
        labels={'tsne_x': 't-SNE Axis X', 'tsne_y': 't-SNE Axis Y'}
    )
    st.plotly_chart(fig_tsne, use_container_width=True)