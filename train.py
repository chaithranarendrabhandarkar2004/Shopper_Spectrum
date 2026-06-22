import os
import sys

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

print("Importing modules...")
from src.preprocessing import load_and_clean_data
from src.rfm_analysis import create_rfm
from src.clustering import train_clustering_model
from src.recommendation import build_recommendation_model

data_path = "data/online_retail.csv"
if not os.path.exists(data_path):
    if os.path.exists("data/OnlineRetail.csv"):
        data_path = "data/OnlineRetail.csv"
    else:
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)

print(f"Loading and cleaning data from {data_path}...")
df = load_and_clean_data(data_path)
print(f"Data loaded. Shape: {df.shape}")

print("Performing RFM analysis...")
rfm = create_rfm(df)
print(f"RFM dataframe shape: {rfm.shape}")

print("Training clustering model (calculating silhouette scores)...")
rfm, model, scaler = train_clustering_model(rfm)
print("Clustering model trained and saved.")

print("Building recommendation model (this may take a minute)...")
similarity_df = build_recommendation_model(df)
print("Recommendation model built and saved.")

print("All models successfully trained and saved!")
