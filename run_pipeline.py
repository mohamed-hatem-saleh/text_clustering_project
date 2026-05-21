import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from core_pipeline import (
    TextCleanerTransformer, get_filtered_features, 
    execute_k_diagnostic_plots, run_model_specific_grid_search, 
    plot_parameter_performance_subplots
)

# 1. Instantiate Dummy/Sample Datasets representing your corporate repositories
# Substitute these rows with: df_news = pd.read_csv("your_20newsgroups.csv")
data_news = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes')).data
data_wiki = pd.read_csv("people_wiki.csv")['text']  

df_news = pd.DataFrame({'text': data_news})
df_wiki = pd.DataFrame({'text': data_wiki})

# 2. Declare Pipeline Framework Structure
processing_pipeline = Pipeline([
    ('cleaner', TextCleanerTransformer(use_lemmatizer=True))
])

print("--- EXECUTING PHASE 1: DIAGNOSTIC TESTS ---")
X_news_features = get_filtered_features(df_news, 'text', processing_pipeline)
execute_k_diagnostic_plots(X_news_features, k_min=5, k_max=15, seed=42)

# Transform raw string formats to clean lists for grid compliance
clean_news = processing_pipeline.named_steps['cleaner'].transform(df_news['text'])
clean_wiki = processing_pipeline.named_steps['cleaner'].transform(df_wiki['text'])

print("\n--- EXECUTING PHASE 2: PARAMETER GRID SEARCH PARALLELS ---")
max_dfs = [0.7, 0.9]
min_dfs = [2, 5]
max_features_list = [200, 500, 1000, 10000]

df_news_results = run_model_specific_grid_search("20 Newsgroups", clean_news, max_dfs, min_dfs, max_features_list)
df_wiki_results = run_model_specific_grid_search("Wikipedia Corpus", clean_wiki, max_dfs, min_dfs, max_features_list)

# Combine and serialize to disk for Streamlit to consume
df_comprehensive = pd.concat([df_news_results, df_wiki_results], ignore_index=True)
df_comprehensive.to_csv("comprehensive_clustering_experiments.csv", index=False)
print("\n💾 System Success: Results logged to 'comprehensive_clustering_experiments.csv'")

# Render global performance subplots for local validation checks
plot_parameter_performance_subplots(df_news_results)