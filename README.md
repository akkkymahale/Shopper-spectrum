# 🛒 Shopper Spectrum: Customer Segmentation & Product Recommendations

A Streamlit web app for e-commerce analytics: item-based product recommendations
(collaborative filtering) and customer segmentation via RFM + KMeans clustering.

## Project structure

```
shopper_spectrum/
├── app.py                 # Streamlit app (2 modules: recommendations + segmentation)
├── pipeline.py             # Data cleaning, RFM, clustering, similarity matrix builder
├── requirements.txt
├── models/                 # Generated artifacts (created by pipeline.py)
│   ├── scaler.pkl
│   ├── kmeans_model.pkl
│   ├── cluster_labels.pkl
│   ├── similarity_matrix.pkl
│   ├── product_lookup.pkl
│   ├── rfm_table.pkl
│   ├── elbow_info.pkl
│   └── clean_transactions.csv
└── README.md
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Only needed once, or if you change the raw data)
#    Regenerate the models from the raw transaction CSV.
#    Edit DATA_PATH in pipeline.py to point at your online_retail.csv
python pipeline.py

# 3. Launch the app
streamlit run app.py
```

The app reads all trained artifacts from `./models/` — no need to retrain
every time you launch it, only when the underlying data changes.

## Data preprocessing (`pipeline.py`)

1. Drop rows with missing `CustomerID`.
2. Exclude cancelled invoices (`InvoiceNo` starting with `C`).
3. Remove rows with non-positive `Quantity` or `UnitPrice`.
4. Engineer RFM features per customer:
   - **Recency** — days since the customer's last purchase (relative to the day after the most recent invoice in the dataset).
   - **Frequency** — number of distinct invoices.
   - **Monetary** — total amount spent (`Quantity × UnitPrice`, summed).
5. Log-transform + `StandardScaler` the RFM features, then fit **KMeans (k=4)**.
   The elbow/silhouette scan across k=2–8 is stored in `models/elbow_info.pkl`
   for reference; k=4 is used for the final model so cluster labels line up
   with the four business-defined segments below.
6. Clusters are automatically ranked by their average RFM profile and mapped to:

   | Segment | Profile |
   |---|---|
   | **High-Value** | Recent, frequent, big spenders |
   | **Regular** | Steady purchasers, moderate spend |
   | **Occasional** | Infrequent, lower spend |
   | **At-Risk** | Haven't purchased in a long time |

7. Item-based collaborative filtering: builds a `CustomerID × Description`
   purchase-quantity matrix, transposes to `Description × CustomerID`, and
   computes cosine similarity between products.

## App modules

### 🎯 Product Recommendations
Type a product name → returns the top 5 most similar products (by cosine
similarity of co-purchase patterns).

### 👥 Customer Segmentation
Enter Recency / Frequency / Monetary → predicts the customer's segment using
the trained KMeans model.

## Notes

- If you want the model to auto-select the number of clusters instead of a
  fixed k=4, call `run_clustering(rfm, forced_k=None)` in `pipeline.py`.
- The product-name lookup in the app does an exact (case-insensitive) match
  first, falling back to a substring match, so partial names still work.
