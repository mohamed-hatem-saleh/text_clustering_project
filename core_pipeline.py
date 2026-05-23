import numpy as np
import pandas as pd
import re
import string
import matplotlib.pyplot as plt
import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import linkage, dendrogram
import contractions


# --- TEXT CLEANING TRANSFORMER ---
class TextCleanerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, use_lemmatizer=True, custom_stop_words=None):
        self.use_lemmatizer = use_lemmatizer
        self.custom_stop_words = custom_stop_words if custom_stop_words else set()
        if self.use_lemmatizer:
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        else:
            self.nlp = None

    def fit(self, X, y=None):
        return self

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        
        # Lowercase and basic contractions clean
        text = text.lower()
        text = contractions.fix(text)
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text) # Strip URLs
        text = re.sub(r'@\w+', ' ', text)                  # Strip handles
        text = text.translate(str.maketrans({char: ' ' for char in string.punctuation}))
        text = re.sub(r'[^a-z\s]', ' ', text)              # Keep characters only

        if self.nlp:
            doc = self.nlp(text)
            cleaned_tokens = [
                token.lemma_ for token in doc
                if not token.is_stop and token.lemma_ not in self.custom_stop_words and token.lemma_.strip()
            ]
            text = " ".join(cleaned_tokens)
        return " ".join(text.split())

    def transform(self, X):
        if isinstance(X, pd.Series):
            return X.apply(self.clean_text).tolist()
        elif isinstance(X, list):
            return [self.clean_text(text) for text in X]
        else:
            raise TypeError("Expected a Pandas Series or a List of strings.")

# --- DECOUPLED INFLECTION PASSES ---
def get_filtered_features(df, text_column, pipeline):
    print("⏳ Processing text through custom modular pipeline...")
    
    # Generate features using the pipeline
    X_tfidf_all = pipeline.fit_transform(df[text_column])
    
    # SAFETY CHECK: If the pipeline returned a list, run it through a standalone vectorizer
    if isinstance(X_tfidf_all, list):
        print("💡 Pipeline returned text lists. Applying dynamic vectorization step...")
        # Fallback vectorizer using your preset notebook configurations
        fallback_vec = TfidfVectorizer(max_df=0.8, min_df=5, max_features=10000, norm='l2')
        X_tfidf_all = fallback_vec.fit_transform(X_tfidf_all)
    
    # Now X_tfidf_all is guaranteed to be a sparse matrix, so .sum(axis=1) works perfectly!
    row_sums = np.array(X_tfidf_all.sum(axis=1)).flatten()
    X_tfidf_filtered = X_tfidf_all[row_sums > 0]
    
    print(f"🛡️  Dropped {np.sum(row_sums <= 0)} empty documents. Active shape: {X_tfidf_filtered.shape}")
    return X_tfidf_filtered

def execute_k_diagnostic_plots(X_features, k_min=5, k_max=25, seed=42):
    k_values = list(range(k_min, k_max + 1, 2))
    inertia_scores, silhouette_scores = [], []
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=seed)
        cluster_labels = kmeans.fit_predict(X_features)
        inertia_scores.append(kmeans.inertia_)
        
        sample_size = min(3000, X_features.shape[0])
        np.random.seed(seed)
        idx = np.random.choice(X_features.shape[0], sample_size, replace=False)
        score = silhouette_score(X_features[idx], cluster_labels[idx], random_state=seed)
        silhouette_scores.append(score)
        
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(k_values, inertia_scores, marker='o', color='#1f77b4', label='Inertia')
    ax1.set_xlabel('Number of Clusters (K)', fontweight='bold')
    ax1.set_ylabel('Inertia', color='#1f77b4')
    
    ax2 = ax1.twinx()
    ax2.plot(k_values, silhouette_scores, marker='s', linestyle='--', color='#2ca02c', label='Silhouette')
    ax2.set_ylabel('Silhouette Score', color='#2ca02c')
    plt.title("Pre-Experiment Optimization Sweep (Synced)", fontweight='bold')
    plt.show()

# --- MODEL SPECIFIC HYPERPARAMETER SWEEPS ---
def run_model_specific_grid_search(dataset_name, clean_texts, max_dfs, min_dfs, max_features_range, fixed_k=20, seed=42):
    print(f"🚀 Launching Grid Search for Dataset: {dataset_name}...")
    records = []
    
    for m_df in max_dfs:
        for n_df in min_dfs:
            for max_f in max_features_range:
                vec = TfidfVectorizer(max_df=m_df, min_df=n_df, max_features=max_f, norm='l2')
                X_raw = vec.fit_transform(clean_texts)
                row_sums = np.array(X_raw.sum(axis=1)).flatten()
                X_tfidf = X_raw[row_sums > 0]
                
                if X_tfidf.shape[0] < fixed_k: continue
                
                svd = TruncatedSVD(n_components=min(50, max_f), random_state=seed)
                X_dense = svd.fit_transform(X_tfidf)
                
                # Deterministic Evaluation sample
                sample_sz = min(2000, X_tfidf.shape[0])
                np.random.seed(seed)
                idx = np.random.choice(X_tfidf.shape[0], sample_sz, replace=False)
                
                # K-Means
                km = KMeans(n_clusters=fixed_k, init='k-means++', n_init=5, random_state=seed).fit(X_tfidf)
                km_s = silhouette_score(X_tfidf[idx], km.labels_[idx], random_state=seed)
                records.append({'dataset': dataset_name, 'model_family': 'K-Means', 'max_df': m_df, 'min_df': n_df, 'max_features': max_f, 'silhouette_score': km_s})
                
                # GMM
                gmm = GaussianMixture(n_components=fixed_k, covariance_type='diag', random_state=seed)
                gmm_labs = gmm.fit_predict(X_dense)
                gmm_s = silhouette_score(X_dense[idx], gmm_labs[idx], random_state=seed)
                records.append({'dataset': dataset_name, 'model_family': 'GMM', 'max_df': m_df, 'min_df': n_df, 'max_features': max_f, 'silhouette_score': gmm_s})
                
                # Hierarchical
                agg = AgglomerativeClustering(n_clusters=fixed_k, linkage='ward')
                agg_labs = agg.fit_predict(X_dense)
                agg_s = silhouette_score(X_dense[idx], agg_labs[idx], random_state=seed)
                records.append({'dataset': dataset_name, 'model_family': 'Hierarchical', 'max_df': m_df, 'min_df': n_df, 'max_features': max_f, 'silhouette_score': agg_s})
                
    return pd.DataFrame(records)

# --- PERFORMANCE ANALYSIS VISUALS ---
def plot_parameter_performance_subplots(df_results):
    params = ['max_df', 'min_df', 'max_features']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for idx, p in enumerate(params):
        stats = df_results.groupby(p)['silhouette_score'].mean().reset_index()
        max_i = stats['silhouette_score'].idxmax()
        colors = ['#2c3e50' if i != max_i else '#f1c40f' for i in range(len(stats))]
        
        axes[idx].bar(stats[p].astype(str), stats['silhouette_score'], color=colors, edgecolor='black')
        axes[idx].set_title(f"Marginal Influence of {p}", fontweight='bold')
        axes[idx].grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()