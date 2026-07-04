"""
Shopper Spectrum - Data Pipeline (Low-Memory Optimized)
Cleans data, builds RFM features, trains KMeans, and computes 
an optimized item similarity matrix for Streamlit Cloud deployment.
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import timedelta

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Pulls directly from your public Google Drive file link
DATA_PATH = "https://drive.google.com/uc?export=download&id=1ecuj3vs7I-7AB5wUq84vftxApmMvvaur"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def load_and_clean(path=DATA_PATH):
    df = pd.read_csv(path)
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)

    # Drop missing CustomerID
    df = df.dropna(subset=["CustomerID"])

    # Exclude cancelled invoices (InvoiceNo starting with 'C')
    df = df[~df["InvoiceNo"].str.startswith("C")]

    # Remove negative/zero quantities and prices
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    # Drop rows with missing description
    df = df.dropna(subset=["Description"])
    df["Description"] = df["Description"].str.strip()

    df["CustomerID"] = df["CustomerID"].astype(int)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    return df.reset_index(drop=True)


def build_rfm(df):
    snapshot_date = df["InvoiceDate"].max() + timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    ).reset_index()

    return rfm


def label_clusters(rfm_with_clusters, cluster_col="Cluster"):
    profile = rfm_with_clusters.groupby(cluster_col)[["Recency", "Frequency", "Monetary"]].mean()
    score = profile["Frequency"].rank() + profile["Monetary"].rank() - profile["Recency"].rank()
    ordered = score.sort_values(ascending=False).index.tolist()

    labels = ["High-Value", "Regular", "Occasional", "At-Risk"]
    while len(labels) < len(ordered):
        labels.append(f"Segment-{len(labels)+1}")
    labels = labels[:len(ordered)]

    mapping = {cluster: labels[i] for i, cluster in enumerate(ordered)}
    return mapping, profile


def run_clustering(rfm, forced_k=4):
    rfm_log = rfm.copy()
    for col in ["Recency", "Frequency", "Monetary"]:
        rfm_log[col] = np.log1p(rfm_log[col].clip(lower=0))

    X = rfm_log[["Recency", "Frequency", "Monetary"]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    final_km = KMeans(n_clusters=forced_k, random_state=42, n_init=10)
    rfm["Cluster"] = final_km.fit_predict(X_scaled)

    return rfm, scaler, final_km


def build_similarity_matrix(df, top_n_products=1500):
    """
    Optimized for low-memory environments like Streamlit Cloud.
    Filters down to the top N most frequent items to prevent 1GB RAM crash.
    """
    # Find the top products sold most frequently
    top_items = df["Description"].value_counts().head(top_n_products).index
    df_filtered = df[df["Description"].isin(top_items)]

    # Group aggregates efficiently
    basket = df_filtered.groupby(["CustomerID", "Description"])["Quantity"].sum().unstack(fill_value=0)
    
    # Cast to float32 immediately to reduce matrix sizes by 50%
    item_matrix = basket.T.astype(np.float32) 
    
    sim = cosine_similarity(item_matrix).astype(np.float32)
    sim_df = pd.DataFrame(sim, index=item_matrix.index, columns=item_matrix.index)
    return sim_df


def main():
    print("Loading & cleaning data directly from Google Drive...")
    df = load_and_clean()

    print("Building RFM table...")
    rfm = build_rfm(df)

    print("Running clustering...")
    rfm, scaler, kmeans = run_clustering(rfm)
    cluster_map, _ = label_clusters(rfm)
    rfm["Segment"] = rfm["Cluster"].map(cluster_map)

    print("Building optimized item similarity matrix (Low Memory Mode)...")
    sim_df = build_similarity_matrix(df, top_n_products=1500)

    product_lookup = (
        df[["StockCode", "Description"]]
        .drop_duplicates(subset="Description")
        .set_index("Description")["StockCode"]
        .to_dict()
    )

    print("Saving artifacts...")
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODEL_DIR, "kmeans_model.pkl"), "wb") as f:
        pickle.dump(kmeans, f)
    with open(os.path.join(MODEL_DIR, "cluster_labels.pkl"), "wb") as f:
        pickle.dump(cluster_map, f)
    with open(os.path.join(MODEL_DIR, "similarity_matrix.pkl"), "wb") as f:
        pickle.dump(sim_df, f)
    with open(os.path.join(MODEL_DIR, "product_lookup.pkl"), "wb") as f:
        pickle.dump(product_lookup, f)
    with open(os.path.join(MODEL_DIR, "rfm_table.pkl"), "wb") as f:
        pickle.dump(rfm, f)

    print("Done! Artifacts saved successfully.")


if __name__ == "__main__":
    main()
