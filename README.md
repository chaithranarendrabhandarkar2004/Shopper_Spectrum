# 🛍️ Shopper Spectrum
### Customer Segmentation and Product Recommendation System

<p align="center">

**Turning E-Commerce Data into Customer Insights**

</p>

---

## 📌 Project Overview

Shopper Spectrum is an E-Commerce Analytics and Machine Learning project designed to understand customer purchasing behaviour and provide personalized product recommendations.

The project combines customer segmentation, RFM analysis, K-Means clustering, and item-based collaborative filtering into an interactive Streamlit dashboard.

### ✨ Key Features

- 👥 Customer Segmentation using RFM Analysis
- 📊 K-Means Clustering
- 🛒 Product Recommendation using Item-Based Collaborative Filtering
- 📈 Interactive Data Visualizations
- 🎯 Customer Behaviour Analysis
- 🌐 Streamlit Web Application

---

## 🎯 Problem Statement

The project analyzes online retail transaction data to understand customer purchasing patterns and improve customer-focused decision-making.

### Main Objectives

1. Segment customers using RFM Analysis.
2. Identify High-Value, Regular, Occasional, and At-Risk customers.
3. Build a Product Recommendation System.
4. Visualize customer segments and product similarities.
5. Deploy the solution using Streamlit.

---

## 📂 Dataset Features

The project uses online retail transaction data with the following columns:

| Column | Description |
|---|---|
| InvoiceNo | Transaction Number |
| StockCode | Product Code |
| Description | Product Name |
| Quantity | Quantity Purchased |
| InvoiceDate | Transaction Date |
| UnitPrice | Product Price |
| CustomerID | Customer Identifier |
| Country | Customer Country |

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| Pandas | Data Manipulation |
| NumPy | Numerical Computing |
| Scikit-Learn | Machine Learning |
| Matplotlib | Data Visualization |
| Seaborn | Statistical Visualization |
| Plotly | Interactive Visualizations |
| Streamlit | Web Application |
| Joblib | Model Serialization |
| UV | Python Package Management |

---

## 🏗️ Project Structure

```text
Shopper_Spectrum/
│
├── data/
│   └── online_retail.csv
│
├── notebooks/
│
├── models/
│   ├── kmeans_model.pkl
│   ├── scaler.pkl
│   ├── rfm_with_clusters.pkl
│   ├── clustering_metrics.pkl
│   └── item_similarity.pkl  (Generated during training)
│
├── src/
│   ├── preprocessing.py
│   ├── rfm_analysis.py
│   ├── clustering.py
│   └── recommendation.py
│
├── streamlit_app/
│   └── app.py
│
├── outputs/
│
├── train.py
├── pyproject.toml
├── uv.lock
├── README.md
└── .gitignore
```

---

## ⚙️ Machine Learning Workflow

### 1️⃣ Data Preprocessing

- Remove missing CustomerID values.
- Remove cancelled invoices.
- Remove invalid quantities.
- Remove invalid prices.
- Prepare clean transaction data.

### 2️⃣ Feature Engineering – RFM Analysis

RFM analysis evaluates customers using three important metrics:

- **Recency:** How recently a customer made a purchase.
- **Frequency:** How frequently a customer purchases.
- **Monetary:** How much a customer spends.

### 3️⃣ Customer Segmentation

- StandardScaler for feature scaling.
- K-Means Clustering for customer segmentation.
- Elbow Method to evaluate cluster counts.
- Silhouette Score to assess clustering quality.
- Dynamic cluster labeling.

### 4️⃣ Product Recommendation

- Create a Customer-Product Matrix.
- Calculate Cosine Similarity.
- Apply Item-Based Collaborative Filtering.
- Generate the Top 5 Similar Product Recommendations.

---

## 📊 Streamlit Dashboard Features

### 🏠 Home

- Project overview.
- Introduction to customer segmentation.
- Recommendation system overview.

### 👥 Customer Clustering

- Interactive 3D RFM visualization.
- Elbow Method chart.
- Silhouette Score visualization.
- New Customer Segment Prediction.
- Customer Cluster Profiles.

### 🛒 Product Recommendation

- Select a product.
- Generate Top 5 similar product recommendations.
- Explore product similarity through a heatmap.

---

## 🤖 Generated Model Files

The project generates the following machine learning artifacts:

| Model File | Purpose |
|---|---|
| `kmeans_model.pkl` | Trained K-Means model |
| `scaler.pkl` | Feature scaling |
| `rfm_with_clusters.pkl` | Customer segmentation data |
| `clustering_metrics.pkl` | Clustering evaluation metrics |
| `item_similarity.pkl` | Product similarity matrix |

**Note:** The large `item_similarity.pkl` file is generated automatically during model training and is not included in this GitHub repository.

---

## 🚀 Installation and Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/chaithranarendrabhandarkar2004/Shopper_Spectrum.git
```

```bash
cd Shopper_Spectrum
```

### 2️⃣ Create Virtual Environment

```bash
uv venv
```

### 3️⃣ Activate Virtual Environment (Windows)

```bash
.venv\Scripts\activate
```

### 4️⃣ Install Dependencies

```bash
uv sync
```

### 5️⃣ Train Machine Learning Models

```bash
python train.py
```

This command:

- Loads and preprocesses the dataset.
- Performs RFM Analysis.
- Trains the K-Means clustering model.
- Builds the product recommendation model.
- Saves the generated model files inside the `models/` directory.

**Important:** Run model training before starting the Streamlit application. The recommendation model is generated locally and is not stored in GitHub because of its large file size.

### 6️⃣ Run Streamlit Application

```bash
streamlit run streamlit_app/app.py
```

The application will open in your browser.

### 7️⃣ Run Jupyter Notebook (Optional)

```bash
jupyter notebook
```

---

## 📈 Expected Outcomes

- Better understanding of customer purchasing behaviour.
- Identification of different customer segments.
- Data-driven customer segmentation.
- Product recommendations based on similarity.
- Interactive visualization of retail analytics.

---

## 🔮 Future Improvements

- Improve recommendation performance.
- Explore additional recommendation algorithms.
- Add more interactive dashboard features.
- Enhance customer segmentation analysis.
- Deploy the application online.

---

## 👩‍💻 Author

**Leena Devadiga**

Artificial Intelligence and Machine Learning Engineering Student

---

## ⭐ Support

If you find this project interesting, consider giving this repository a ⭐ on GitHub!

**Thank you for visiting Shopper Spectrum!** 🚀
