import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from core_pipeline import TextCleanerTransformer

# --- LOAD DATASETS ---
print("📊 Loading Datasets...")
data_news = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
df_news = pd.DataFrame({'text': data_news.data})
df_wiki = pd.read_csv("people_wiki.csv")

print("🚀 Starting the Text Cleaning Process...")
cleaner = TextCleanerTransformer(use_lemmatizer=True)
clean_news = cleaner.transform(df_news['text'])
clean_wiki = cleaner.transform(df_wiki['text'])

# Master lists for storing metrics dataframes
optimal_k_records = []
grid_search_records = []
model_tuning_records = []
spatial_records = []

for dataset_name, clean_texts in [("20 Newsgroups", clean_news), ("Wikipedia Corpus", clean_wiki)]:
    print(f"⚙️ Running Experiment Cycles for: {dataset_name}")
    
    # Baseline text representations
    vec_base = TfidfVectorizer(max_df=0.8, min_df=2, max_features=1000, norm='l2')
    X_tfidf = vec_base.fit_transform(clean_texts)
    svd = TruncatedSVD(n_components=10, random_state=42)
    X_dense = svd.fit_transform(X_tfidf)
    
    print("🔍 Phase 1: Optimal K Diagnostics...")
    # 1. PHASE ONE: GENERATE OPTIMAL K DIAGNOSTIC LOGS
    for k in range(5, 26, 5):
        km_diag = KMeans(n_clusters=k, init='k-means++', n_init=5, random_state=42).fit(X_tfidf)
        score = silhouette_score(X_tfidf, km_diag.labels_, random_state=42)
        optimal_k_records.append({
            'Dataset': dataset_name, 'K': k, 'Inertia': km_diag.inertia_, 'Silhouette': score
        })

    print("🔍 Phase 2: TF-IDF Grid Search Sweeps...")
    # 2. PHASE TWO: TF-IDF GRID SEARCH SWEEPS (Fixed K=20)
    for max_df in [0.7, 0.9]:
        for max_features in [200, 500, 1000, 10000]:
            vec_grid = TfidfVectorizer(max_df=max_df, min_df=2, max_features=max_features)
            X_grid = vec_grid.fit_transform(clean_texts)
            
            # K-Means Grid Check (Runs on sparse matrix)
            km_grid = KMeans(n_clusters=20, n_init=5, random_state=42).fit(X_grid)
            s_km = silhouette_score(X_grid, km_grid.labels_, random_state=42)
            grid_search_records.append({'Dataset': dataset_name, 'Model': 'K-Means', 'max_df': max_df, 'max_features': max_features, 'Silhouette': s_km})
            
            # Hierarchical & GMM Grid Checks (Require Dense Components)
            if X_grid.shape[1] >= 10:
                X_d_grid = TruncatedSVD(n_components=10, random_state=42).fit_transform(X_grid)
                
                # Hierarchical
                agg_grid = AgglomerativeClustering(n_clusters=20, linkage='ward').fit(X_d_grid)
                s_agg = silhouette_score(X_d_grid, agg_grid.labels_, random_state=42)
                grid_search_records.append({'Dataset': dataset_name, 'Model': 'Hierarchical', 'max_df': max_df, 'max_features': max_features, 'Silhouette': s_agg})
                
                # Gaussian Mixture Models (GMM) added here safely!
                gmm_grid = GaussianMixture(n_components=20, covariance_type='diag', random_state=42).fit(X_d_grid)
                gmm_labs = gmm_grid.predict(X_d_grid)
                s_gmm = silhouette_score(X_d_grid, gmm_labs, random_state=42)
                grid_search_records.append({'Dataset': dataset_name, 'Model': 'GMM', 'max_df': max_df, 'max_features': max_features, 'Silhouette': s_gmm})

    print("🔍 Phase 3: Base Model Hyperparameter Tuning...")
    # 3. PHASE THREE: BASE MODEL INTERNAL HYPERPARAMETER TUNING
    # Tuning K-Means initialization strategies
    for init_strategy in ['k-means++']:
        km_tune = KMeans(n_clusters=20, init=init_strategy, n_init=5, random_state=42).fit(X_tfidf)
        s = silhouette_score(X_tfidf, km_tune.labels_, random_state=42)
        model_tuning_records.append({'Dataset': dataset_name, 'Model': 'K-Means', 'Parameter': f'init={init_strategy}', 'Silhouette': s})
        
    # Tuning Hierarchical linkage algorithms
    for linkage_type in ['ward', 'average', 'complete']:
        agg_tune = AgglomerativeClustering(n_clusters=20, linkage=linkage_type).fit(X_dense)
        s = silhouette_score(X_dense, agg_tune.labels_, random_state=42)
        model_tuning_records.append({'Dataset': dataset_name, 'Model': 'Hierarchical', 'Parameter': f'linkage={linkage_type}', 'Silhouette': s})

    # Tuning GMM Covariance matrix types added here safely!
    for cov_type in ['diag', 'spherical', 'tied']:
        gmm_tune = GaussianMixture(n_components=20, covariance_type=cov_type, random_state=42).fit(X_dense)
        gmm_tune_labs = gmm_tune.predict(X_dense)
        s = silhouette_score(X_dense, gmm_tune_labs, random_state=42)
        model_tuning_records.append({'Dataset': dataset_name, 'Model': 'GMM', 'Parameter': f'covariance={cov_type}', 'Silhouette': s})

    print("🔍 Phase 4: Computing Spatial Topologies for All Models...")
    # 4. PHASE FOUR: COMPUTE GRAPHIC SPATIAL TOPOLOGIES
    tsne = TSNE(n_components=2, perplexity=30, max_iter=300, random_state=42)
    X_tsne = tsne.fit_transform(X_dense)
    
    # Calculate final predictions for all 3 models to enable dashboard toggling
    km_final = KMeans(n_clusters=20, n_init=5, random_state=42).fit_predict(X_tfidf)
    gmm_final = GaussianMixture(n_components=20, covariance_type='diag', random_state=42).fit_predict(X_dense)
    agg_final = AgglomerativeClustering(n_clusters=20, linkage='ward').fit_predict(X_dense)
    
    for idx in range(X_tsne.shape[0]):
        spatial_records.append({
            'Dataset': dataset_name,
            'tsne_x': X_tsne[idx, 0], 
            'tsne_y': X_tsne[idx, 1],
            'K-Means_Cluster': km_final[idx],
            'GMM_Cluster': gmm_final[idx],
            'Hierarchical_Cluster': agg_final[idx],
            'Snippet': clean_texts[idx][:35] + "..."
        })

# Export data matrices to CSV files
print("💾 Saving metrics files to disk...")
pd.DataFrame(optimal_k_records).to_csv("metrics_optimal_k.csv", index=False)
pd.DataFrame(grid_search_records).to_csv("metrics_tfidf_grid.csv", index=False)
pd.DataFrame(model_tuning_records).to_csv("metrics_model_tuning.csv", index=False)
pd.DataFrame(spatial_records).to_csv("metrics_spatial_layout.csv", index=False)
print("💾 Success: All validation optimization datasets saved safely with GMM metrics included!")