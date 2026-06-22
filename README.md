# Shopper Spectrum

## Customer Segmentation and Product Recommendation System

### Project Overview

Shopper Spectrum is an E-Commerce Analytics project that performs:

- Customer Segmentation using RFM Analysis
- K-Means Clustering
- Product Recommendation using Item-Based Collaborative Filtering
- Interactive Streamlit Dashboard

---

## Problem Statement

The project analyzes online retail transaction data to understand customer purchasing behavior.

Objectives:

1. Segment customers using RFM Analysis.
2. Identify High-Value, Regular, Occasional, and At-Risk customers.
3. Build a Product Recommendation System.
4. Deploy the solution using Streamlit.

---

## Dataset Features

| Column | Description |
|----------|-------------|
| InvoiceNo | Transaction Number |
| StockCode | Product Code |
| Description | Product Name |
| Quantity | Quantity Purchased |
| InvoiceDate | Transaction Date |
| UnitPrice | Product Price |
| CustomerID | Customer Identifier |
| Country | Customer Country |

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Streamlit
- UV Package Manager

---

## Project Structure

```text
shopper_spectrum/
│
├── data/
├── notebooks/
├── models/
├── src/
├── streamlit_app/
├── outputs/
├── pyproject.toml
├── README.md
└── .gitignore
```

---

## Machine Learning Workflow

### Data Preprocessing

- Remove Missing CustomerID
- Remove Cancelled Invoices
- Remove Invalid Quantity
- Remove Invalid Prices

### Feature Engineering

RFM Metrics:

- Recency
- Frequency
- Monetary

### Clustering

- StandardScaler
- KMeans Clustering
- Elbow Method
- Silhouette Score

### Recommendation System

- Customer Product Matrix
- Cosine Similarity
- Item-Based Collaborative Filtering

---

## Streamlit Features

### Product Recommendation

Input:

- Product Name

Output:

- Top 5 Similar Products

### Customer Segmentation

Input:

- Recency
- Frequency
- Monetary

Output:

- Customer Segment

---

## Model Files

Generated Models:

- models/kmeans_model.pkl
- models/scaler.pkl
- models/item_similarity.pkl

---

## Installation

Create Environment

```bash
uv venv
```

Activate Environment

```bash
.venv\Scripts\activate
```

Install Dependencies

```bash
uv sync
```

---

## Run Notebook

```bash
jupyter notebook
```

---

## Run Streamlit Application

```bash
streamlit run streamlit_app/app.py
```

---

## Author

Leena Devadiga

Artificial Intelligence and Machine Learning Engineering Student