# Project Analysis & Fixes Report: Shopper Spectrum

This document details the issues identified, structural corrections made, and design enhancements implemented in the **Shopper Spectrum** customer segmentation and product recommendation workspace.

---

## 🛠️ Errors Identified & Corrected

### 1. Corrupted/Empty Pickle Files (`EOFError`)
* **Issue**: The three pickled files in the `models/` directory (`item_similarity.pkl`, `kmeans_model.pkl`, and `scaler.pkl`) were empty (0 bytes). When starting the Streamlit application, `joblib.load()` threw an `EOFError` traceback, causing the app to crash on boot.
* **Fix**: Created and executed a complete model training pipeline script (`train.py`). This script loads the dataset from `data/online_retail.csv`, pre-processes the transactions, aggregates Recency/Frequency/Monetary (RFM) values per customer, trains the KMeans model (dynamically finding the optimal number of clusters using silhouette score calculations), computes the cosine similarity matrix for item-to-item recommendations, and saves the fully trained models. All files are now non-zero and functional.

### 2. Hardcoded & Inverted Cluster Labels (Semantic Logic Error)
* **Issue**: `src/utils.py` had a static dictionary mapping cluster indices 0-5 directly to labels. Because KMeans selects the number of clusters dynamically and assigns IDs arbitrarily based on initialization, standard/occasional shoppers were mislabeled as "High Value" while top-tier spenders (with average monetary spends of $85,000+) were mislabeled as "Regular".
* **Fix**: Rewrote `src/utils.py` to use a dynamic profiling algorithm `get_dynamic_cluster_labels(kmeans, scaler)`. This function reconstructs original-scale cluster centers, sorts them by average monetary value, and dynamically assigns correct customer classifications matching the PDF specifications:
  * High R, High F, High M $\rightarrow$ **`High-Value`**
  * Medium F, Medium M $\rightarrow$ **`Regular`**
  * Low F, Low M, older R $\rightarrow$ **`Occasional`**
  * High R, Low F, Low M $\rightarrow$ **`At-Risk`**

### 3. Jupyter Notebook Errors (`shopper_spectrum.ipynb`)
* **Issue**: The notebook `notebooks/shopper_spectrum.ipynb` was failing due to:
  * Missing import of `train_clustering_model` in the first cell.
  * An incorrect case-sensitive path reference (`data/OnlineRetail.csv` instead of the actual `data/online_retail.csv`).
* **Fix**: Patched the notebook cells programmatically to ensure it executes seamlessly from top to bottom.

### 4. Illegible Button Text & Contrast (UI/UX Bug)
* **Issue**: Streamlit's default theme settings rendered the main interaction buttons with white text on a white background, making the text `Predict Customer Cluster` completely invisible.
* **Fix**: Injected custom CSS rules inside `streamlit_app/app.py` to provide a premium design:
  * **Default state**: Clean white background button with legible `#000000` (black) text.
  * **Hover state**: Smooth transition to an elegant indigo-purple gradient background with `#ffffff` (white) text.

---

## 🎨 Premium Visual & Feature Upgrades

* **Three-Page Navigation Layout**:
  * **Home**: Landing page describing the project, customer segmentation metrics, and recommendation mechanics.
  * **Clustering**: Displaying dynamic insights, including interactive curves and customer-level predictions.
  * **Recommendation**: Collaborative filtering engine with catalog lookups.
* **Clustering Visualizations**:
  * **Elbow Curve (Inertia vs. k)**: Graphical analysis demonstrating the mathematical logic behind cluster count selection.
  * **Silhouette Curve (Scores vs. k)**: Silhouette coefficients plotted across $k=2..10$.
  * **3D RFM Segments Map**: An interactive 3D scatter plot of all 4,338 customer coordinates, color-coded by their predicted behavior segments (`High-Value`, `Regular`, `Occasional`, `At-Risk`).
* **Recommendation Visualizations**:
  * **Product Similarity Heatmap**: A dynamic cosine similarity heatmap showing item correlation matrix of the selected product and its top 10 recommended items.
* **Technical Tags Badge Grid**:
  * Renders a styled tag pill grid in the sidebar listing: `Pandas, Numpy, DataCleaning, FeatureEngineering, EDA, RFMAnalysis, CustomerSegmentation, KMeansClustering, CollaborativeFiltering, CosineSimilarity, ProductRecommendation, ScikitLearn, StandardScaler, StreamlitApp, MachineLearning, DataVisualization, PivotTables, DataTransformation, RealTimePrediction`

---

## 🚀 How to Run the Project

1. **Activate Virtual Environment**:
   ```powershell
   .venv\Scripts\activate
   ```
2. **Start Streamlit Dashboard**:
   ```powershell
   streamlit run streamlit_app/app.py
   ```
   The application will boot on `http://localhost:8501`.
