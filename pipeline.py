"""
Shopper Spectrum - Data Pipeline
Cleans the online retail transaction data, engineers RFM features,
trains a KMeans customer-segmentation model, and builds an item-based
collaborative-filtering similarity matrix for product recommendations.

Outputs (saved into ./models/):
    - scaler.pkl            StandardScaler fit on log-transformed RFM
    - kmeans_model.pkl      Trained KMeans model
    - cluster_labels.pkl    dict mapping cluster number -> segment label
    - similarity_matrix.pkl DataFrame of item-item cosine similarity
    - product_lookup.pkl    dict mapping StockCode -> Description (and reverse)
    - rfm_table.pkl         Full RFM table (for EDA / reference in app)
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import timedelta

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "/mnt/user-data/uploads/online_retail.csv"
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

    # Drop rows with missing description (can't recommend a nameless product)
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
    """Rank clusters by their RFM profile and assign business-friendly labels."""
    profile = rfm_with_clusters.groupby(cluster_col)[["Recency", "Frequency", "Monetary"]].mean()

    # Build a composite score: high F & M and low R = best customers
    score = profile["Frequency"].rank() + profile["Monetary"].rank() - profile["Recency"].rank()
    ordered = score.sort_values(ascending=False).index.tolist()

    labels = ["High-Value", "Regular", "Occasional", "At-Risk"]
    # If there are more/fewer clusters than 4 labels, extend generically
    while len(labels) < len(ordered):
        labels.append(f"Segment-{len(labels)+1}")
    labels = labels[:len(ordered)]

    mapping = {cluster: labels[i] for i, cluster in enumerate(ordered)}
    return mapping, profile


def run_clustering(rfm, k_range=range(2, 9), forced_k=4):
    """Runs the elbow/silhouette scan across k_range for reporting purposes,
    but trains the final model with `forced_k` clusters so the resulting
    segments line up with the four business-defined labels
    (High-Value / Regular / Occasional / At-Risk) requested in the brief.
    Set forced_k=None to let silhouette score pick k automatically."""
    rfm_log = rfm.copy()
    for col in ["Recency", "Frequency", "Monetary"]:
        rfm_log[col] = np.log1p(rfm_log[col].clip(lower=0))

    X = rfm_log[["Recency", "Frequency", "Monetary"]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    inertias, sil_scores = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        preds = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, preds))

    print("Silhouette scores:", dict(zip(k_range, sil_scores)))

    if forced_k is not None:
        best_k = forced_k
    else:
        best_k = list(k_range)[int(np.argmax(sil_scores))]
    print("Chosen k:", best_k)

    final_km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    rfm["Cluster"] = final_km.fit_predict(X_scaled)

    return rfm, scaler, final_km, {"k_range": list(k_range), "inertias": inertias, "sil_scores": sil_scores}


def build_similarity_matrix(df, min_customers_per_item=1, top_n_products=None):
    """Item-based collaborative filtering via cosine similarity on the
    CustomerID x Description quantity matrix."""
    basket = df.groupby(["CustomerID", "Description"])["Quantity"].sum().unstack(fill_value=0)

    if top_n_products:
        top_items = df.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(top_n_products).index
        basket = basket[top_items]

    item_matrix = basket.T  # items x customers
    sim = cosine_similarity(item_matrix).astype(np.float32)
    sim_df = pd.DataFrame(sim, index=item_matrix.index, columns=item_matrix.index)
    return sim_df


def main():
    print("Loading & cleaning data...")
    df = load_and_clean()
    print("Clean shape:", df.shape)

    print("Building RFM table...")
    rfm = build_rfm(df)

    print("Running clustering...")
    rfm, scaler, kmeans, elbow_info = run_clustering(rfm)

    cluster_map, profile = label_clusters(rfm)
    rfm["Segment"] = rfm["Cluster"].map(cluster_map)
    print(profile)
    print(cluster_map)

    print("Building item similarity matrix (this can take a minute)...")
    sim_df = build_similarity_matrix(df)

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
    with open(os.path.join(MODEL_DIR, "elbow_info.pkl"), "wb") as f:
        pickle.dump(elbow_info, f)

    # Also save the cleaned transactional data for EDA reference
    df.to_csv(os.path.join(MODEL_DIR, "clean_transactions.csv"), index=False)

    print("Done. Artifacts saved to", MODEL_DIR)


if __name__ == "__main__":
    main()
