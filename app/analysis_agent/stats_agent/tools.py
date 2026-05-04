import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import zscore, pearsonr
import os

SANDBOX_DIR = "/Users/josephsong/Desktop/Projects/Personal/adk-stats-agent-demo-v2/_sandbox"

def load_parquet(filepath: str) -> pd.DataFrame:
    """Helper to load a parquet file"""
    basename = os.path.basename(filepath)
    
    # Auto-correct if the agent hallucinates the CSV extension
    if basename.endswith(".csv"):
        basename = basename.replace(".csv", ".parquet")
        if not basename.startswith("clean_"):
            basename = "clean_" + basename
            
    actual_path = os.path.join(SANDBOX_DIR, basename)
    if not os.path.exists(actual_path):
        raise FileNotFoundError(f"File {basename} not found in sandbox. You MUST use the .parquet files provided by the Data Engineer.")
    
    try:
        return pd.read_parquet(actual_path)
    except Exception as e:
        raise ValueError(f"Failed to read {basename} as parquet. Is it a valid Parquet file? Error: {str(e)}")

def forecast_time_series(filepath: str, target_col: str, periods: int) -> dict:
    """Uses statsmodels (ARIMA) to forecast metrics like revenue or profit over time. Returns forecast and confidence intervals."""
    df = load_parquet(filepath)
    if 'date' in df.columns:
        df = df.set_index('date')
    if target_col not in df.columns:
        return {"error": f"Column '{target_col}' not found. Available columns: {df.columns.tolist()}"}
    series = df[target_col].dropna()
    
    # Simple ARIMA (1,1,1) for demonstration
    model = ARIMA(series, order=(1, 1, 1))
    fit_model = model.fit()
    forecast = fit_model.get_forecast(steps=periods)
    
    return {
        "forecast": forecast.predicted_mean.tolist(),
        "conf_int": forecast.conf_int().values.tolist(),
        "periods": periods,
        "target_col": target_col
    }

def predict_probability(filepath: str, target_event: str) -> dict:
    """Uses xgboost.XGBClassifier to output a probability score (0.0 to 1.0) for discrete events."""
    df = load_parquet(filepath)
    if target_event not in df.columns:
        return {"error": f"Target column '{target_event}' not found."}
    
    # Create dummy numeric features for demonstration
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_event in numeric_cols:
        numeric_cols.remove(target_event)
        
    X = df[numeric_cols].fillna(0)
    # Encode target_event safely (e.g. "Yes"/"No" to 1/0)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(df[target_event].astype(str))
    
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)
    probs = model.predict_proba(X)[:, 1]
    
    return {
        "average_probability": float(np.mean(probs)),
        "max_probability": float(np.max(probs)),
        "min_probability": float(np.min(probs)),
        "target_event": target_event
    }

def cluster_entities(filepath: str, feature_cols: list[str], k_clusters: int) -> dict:
    """Uses sklearn.cluster.KMeans to group entities based on numeric features."""
    df = load_parquet(filepath)
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        return {"error": f"Missing columns: {missing_cols}"}
        
    X = df[feature_cols].fillna(0)
    kmeans = KMeans(n_clusters=k_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    
    counts = np.bincount(labels)
    return {
        "cluster_sizes": counts.tolist(),
        "cluster_centers": kmeans.cluster_centers_.tolist(),
        "features_used": feature_cols
    }

def calculate_rfm_scores(filepath: str, customer_col: str, date_col: str, revenue_col: str) -> dict:
    """Uses Pandas to segment customers by Recency, Frequency, and Monetary value."""
    df = load_parquet(filepath)
    if not all(col in df.columns for col in [customer_col, date_col, revenue_col]):
        return {"error": f"Missing required columns. Found: {df.columns.tolist()}"}
    
    max_date = df[date_col].max()
    rfm = df.groupby(customer_col).agg({
        date_col: lambda x: (max_date - x.max()).days,
        customer_col: 'count',
        revenue_col: 'sum'
    })
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    
    return {
        "summary": rfm.describe().to_dict()
    }

def detect_anomalies_isolation_forest(filepath: str, metric_col: str) -> dict:
    """Uses sklearn.ensemble.IsolationForest to flag multivariate outliers."""
    df = load_parquet(filepath)
    if metric_col not in df.columns:
        return {"error": f"Column '{metric_col}' not found."}
        
    X = df[[metric_col]].fillna(0)
    iso = IsolationForest(contamination=0.05, random_state=42)
    preds = iso.fit_predict(X)
    
    anomalies = df[preds == -1]
    return {
        "anomaly_count": len(anomalies),
        "total_rows": len(df),
        "metric": metric_col
    }

def calculate_z_scores(filepath: str, metric_col: str) -> dict:
    """Uses scipy.stats.zscore to flag data points >3 standard deviations from the mean."""
    df = load_parquet(filepath)
    if metric_col not in df.columns:
        return {"error": f"Column '{metric_col}' not found."}
        
    series = df[metric_col].fillna(0)
    z_scores = zscore(series)
    outliers = len(np.where(np.abs(z_scores) > 3)[0])
    
    return {
        "outlier_count_gt_3_std": outliers,
        "total_rows": len(df),
        "metric": metric_col
    }

def extract_topics(filepath: str, text_column: str) -> dict:
    """Uses sklearn.decomposition.LatentDirichletAllocation (LDA) to group text entries into thematic categories."""
    df = load_parquet(filepath)
    if text_column not in df.columns:
        return {"error": f"Column '{text_column}' not found."}
        
    texts = df[text_column].dropna().astype(str).tolist()
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    X = vectorizer.fit_transform(texts)
    
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(X)
    
    feature_names = vectorizer.get_feature_names_out()
    topics = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-6:-1]]
        topics[f"Topic {topic_idx + 1}"] = top_words
        
    return {"topics": topics}

def get_top_ngrams(filepath: str, text_column: str, n: int) -> dict:
    """Uses CountVectorizer to find the most frequent multi-word phrases (ngrams)."""
    df = load_parquet(filepath)
    if text_column not in df.columns:
        return {"error": f"Column '{text_column}' not found."}
        
    texts = df[text_column].dropna().astype(str).tolist()
    vectorizer = CountVectorizer(ngram_range=(n, n), stop_words='english')
    X = vectorizer.fit_transform(texts)
    
    sum_words = X.sum(axis=0)
    words_freq = [(word, int(sum_words[0, idx])) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    
    return {"top_ngrams": words_freq[:10]}

def calculate_correlation_matrix(filepath: str, col_a: str, col_b: str) -> dict:
    """Uses scipy.stats.pearsonr to calculate correlation coefficients and returns the p-value."""
    df = load_parquet(filepath)
    if col_a not in df.columns or col_b not in df.columns:
        return {"error": "Columns not found."}
        
    clean_df = df[[col_a, col_b]].dropna()
    corr, p_value = pearsonr(clean_df[col_a], clean_df[col_b])
    
    return {
        "correlation_coefficient": float(corr),
        "p_value": float(p_value),
        "significant_at_0_05": bool(p_value < 0.05)
    }

def run_linear_regression(filepath: str, independent_var: str, dependent_var: str) -> dict:
    """Uses statsmodels.api.OLS to determine driver impact."""
    df = load_parquet(filepath)
    if independent_var not in df.columns or dependent_var not in df.columns:
        return {"error": "Columns not found."}
        
    clean_df = df[[independent_var, dependent_var]].dropna()
    X = sm.add_constant(clean_df[independent_var])
    y = clean_df[dependent_var]
    
    model = sm.OLS(y, X).fit()
    
    return {
        "r_squared": float(model.rsquared),
        "p_values": model.pvalues.to_dict(),
        "coefficients": model.params.to_dict()
    }

def generate_cohort_retention_matrix(filepath: str, acquisition_date_col: str, activity_date_col: str) -> dict:
    """Uses Pandas pivot tables to generate a time-decay matrix for cohort retention analysis."""
    df = load_parquet(filepath)
    if acquisition_date_col not in df.columns or activity_date_col not in df.columns:
        return {"error": "Date columns not found."}
        
    df[acquisition_date_col] = pd.to_datetime(df[acquisition_date_col])
    df[activity_date_col] = pd.to_datetime(df[activity_date_col])
    
    df['CohortMonth'] = df[acquisition_date_col].dt.to_period('M')
    df['ActivityMonth'] = df[activity_date_col].dt.to_period('M')
    
    df['CohortIndex'] = (df['ActivityMonth'] - df['CohortMonth']).apply(lambda x: x.n)
    
    cohort_data = df.groupby(['CohortMonth', 'CohortIndex'])['customer_id'].nunique().reset_index()
    cohort_pivot = cohort_data.pivot(index='CohortMonth', columns='CohortIndex', values='customer_id')
    
    # Convert to dict for JSON serialization, converting periods to strings
    pivot_dict = {str(k): v.dropna().to_dict() for k, v in cohort_pivot.iterrows()}
    
    return {"cohort_matrix": pivot_dict}
