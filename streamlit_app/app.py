import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.recommendation import recommend_products
from src.utils import get_dynamic_cluster_labels

st.set_page_config(
    page_title="Shopper Spectrum - Analytics Dashboard",
    page_icon="🛒",
    layout="wide"
)

# Custom Premium Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Outfit', sans-serif;
    background: linear-gradient(135deg, #0e1117 0%, #161b22 100%);
    color: #e6edf3;
}

/* Header style */
.main-header {
    background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 3rem;
    margin-bottom: 0.5rem;
    text-align: center;
}

.sub-header {
    font-size: 1.2rem;
    color: #8b949e;
    text-align: center;
    margin-bottom: 2rem;
}

/* Glassmorphism cards */
.glass-card {
    background: rgba(255, 255, 255, 0.03);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
}

/* Product recommendation cards */
.product-card {
    background: rgba(79, 70, 229, 0.08);
    border: 1px solid rgba(79, 70, 229, 0.2);
    border-radius: 12px;
    padding: 16px;
    margin: 8px 0;
    transition: all 0.3s ease;
}

.product-card:hover {
    transform: translateY(-2px);
    background: rgba(79, 70, 229, 0.15);
    border-color: rgba(79, 70, 229, 0.4);
    box-shadow: 0 4px 20px rgba(79, 70, 229, 0.2);
}

.product-name {
    font-weight: 600;
    color: #f0f6fc;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #0b0e14 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

/* Custom Alert Card */
.alert-card {
    padding: 16px;
    border-radius: 12px;
    margin-top: 15px;
    border: 1px solid;
}
.alert-success {
    background: rgba(46, 160, 67, 0.15);
    border-color: rgba(46, 160, 67, 0.4);
    color: #3fb950;
}
.alert-info {
    background: rgba(56, 139, 253, 0.15);
    border-color: rgba(56, 139, 253, 0.4);
    color: #58a6ff;
}

/* Tech tag pills */
.tech-tag {
    display: inline-block;
    background: rgba(79, 70, 229, 0.12);
    border: 1px solid rgba(79, 70, 229, 0.25);
    border-radius: 20px;
    padding: 4px 10px;
    margin: 4px;
    font-size: 0.75rem;
    color: #c9d1d9;
    font-weight: 500;
}

/* Button styling */
div.stButton > button {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 1px solid #d0d7de !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
}

div.stButton > button p, div.stButton > button span {
    color: #000000 !important;
    transition: all 0.3s ease !important;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%) !important;
    color: #ffffff !important;
    border: 1px solid transparent !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(124, 58, 237, 0.4) !important;
}

div.stButton > button:hover p, div.stButton > button:hover span {
    color: #ffffff !important;
}

/* Active and focus styles to stay clean */
div.stButton > button:active, div.stButton > button:focus {
    background-color: #ffffff !important;
    color: #000000 !important;
    border-color: #d0d7de !important;
}

div.stButton > button:active p, div.stButton > button:active span,
div.stButton > button:focus p, div.stButton > button:focus span {
    color: #000000 !important;
}

div.stButton > button:focus:hover {
    background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%) !important;
    color: #ffffff !important;
}

div.stButton > button:focus:hover p, div.stButton > button:focus:hover span {
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

# App Logo / Header
st.markdown("<div class='main-header'>🛒 Shopper Spectrum</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>AI-Powered Customer Segmentation & Product Recommendations</div>", unsafe_allow_html=True)

# Load Models
@st.cache_resource
def load_models():
    similarity_df = joblib.load("models/item_similarity.pkl")
    kmeans = joblib.load("models/kmeans_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    rfm_df = joblib.load("models/rfm_with_clusters.pkl")
    metrics = joblib.load("models/clustering_metrics.pkl")
    return similarity_df, kmeans, scaler, rfm_df, metrics

similarity_df, kmeans, scaler, rfm_df, metrics = load_models()

# Get dynamic cluster labels based on actual trained profiles (matching exact labels)
cluster_labels = get_dynamic_cluster_labels(kmeans, scaler)

# Sidebar Navigation
st.sidebar.markdown("### 🧭 Navigation")
option = st.sidebar.radio(
    "Go To Page",
    [
        "Home",
        "Clustering",
        "Recommendation"
    ]
)

# Sidebar Quick Stats / Info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Overview")
st.sidebar.metric("Unique Products", len(similarity_df.columns))
st.sidebar.metric("Customer Segments", kmeans.n_clusters)

# Sidebar Technical Tags
st.sidebar.markdown("---")
st.sidebar.markdown("### 🏷️ Technical Tags")
tags_html = ""
tags = [
    "Pandas", "Numpy", "DataCleaning", "FeatureEngineering", "EDA", 
    "RFMAnalysis", "CustomerSegmentation", "KMeansClustering", 
    "CollaborativeFiltering", "CosineSimilarity", "ProductRecommendation", 
    "ScikitLearn", "StandardScaler", "StreamlitApp", "MachineLearning", 
    "DataVisualization", "PivotTables", "DataTransformation", "RealTimePrediction"
]
for tag in tags:
    tags_html += f"<span class='tech-tag'>{tag}</span>"
st.sidebar.markdown(f"<div>{tags_html}</div>", unsafe_allow_html=True)


# ------------------ 🏠 HOME PAGE ------------------
if option == "Home":
    st.markdown("""
    <div class='glass-card'>
        <h2>👋 Welcome to Shopper Spectrum</h2>
        <p style='font-size:1.15rem; line-height:1.6; color:#c9d1d9;'>
            Shopper Spectrum is an intelligent, double-engine retail analytics dashboard designed to solve two core challenges in e-commerce:
        </p>
        <ol style='font-size:1.1rem; line-height:1.8; color:#c9d1d9;'>
            <li><strong>Who are our customers?</strong> We employ RFM Analysis and unsupervised KMeans clustering to segment our shoppers into distinct behavior profiles.</li>
            <li><strong>What should we sell to them?</strong> We use item-to-item Collaborative Filtering based on purchase history to recommend relevant products in real-time.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='glass-card' style='height: 380px;'>
            <h3 style='color:#7c3aed;'>👥 Part 1: Customer Segmentation</h3>
            <p style='color:#c9d1d9; font-size:1rem; line-height:1.6;'>
                By assessing customer transactions across three core dimensions:
            </p>
            <ul style='color:#c9d1d9; font-size:0.95rem; line-height:1.6;'>
                <li><strong>Recency (R)</strong>: How many days ago did the customer buy?</li>
                <li><strong>Frequency (F)</strong>: How many distinct orders did they place?</li>
                <li><strong>Monetary (M)</strong>: How much total currency did they spend?</li>
            </ul>
            <p style='color:#c9d1d9; font-size:1rem; line-height:1.6;'>
                KMeans groups customers into four operational segments: 
                <span class='tech-tag'>High-Value</span>, 
                <span class='tech-tag'>Regular</span>, 
                <span class='tech-tag'>Occasional</span>, and 
                <span class='tech-tag'>At-Risk</span>.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class='glass-card' style='height: 380px;'>
            <h3 style='color:#db2777;'>🎯 Part 2: Product Recommendations</h3>
            <p style='color:#c9d1d9; font-size:1rem; line-height:1.6;'>
                Leverages matrix transformations and Cosine Similarity to capture product buying similarities.
            </p>
            <p style='color:#c9d1d9; font-size:1rem; line-height:1.6;'>
                If a customer is interested in a product, the system immediately recommends 
                the top 5 items most frequently purchased in tandem across all invoice histories.
            </p>
            <p style='color:#c9d1d9; font-size:1rem; line-height:1.6;'>
                Explore this system using the navigation options on the left sidebar!
            </p>
        </div>
        """, unsafe_allow_html=True)


# ------------------ 📊 CLUSTERING PAGE ------------------
elif option == "Clustering":
    st.markdown("<div class='glass-card'><h2>📊 Clustering Insights & Analytics</h2>"
                "Review the metrics, decision pathways, and 3D maps generated by our KMeans clustering engine.</div>", unsafe_allow_html=True)
    
    # 3D RFM Scatter Plot
    st.subheader("🌐 3D RFM Customer Segments Map")
    
    # Map the Cluster ID to Segment Label for clean plotting
    rfm_df["Segment"] = rfm_df["Cluster"].map(cluster_labels)
    
    fig_3d = px.scatter_3d(
        rfm_df,
        x='Recency',
        y='Frequency',
        z='Monetary',
        color='Segment',
        color_discrete_map={
            'High-Value': '#2ea043',
            'Regular': '#58a6ff',
            'Occasional': '#8b949e',
            'At-Risk': '#ff7b72'
        },
        labels={'Recency': 'Recency (Days)', 'Frequency': 'Frequency (Invoices)', 'Monetary': 'Monetary ($)'},
        opacity=0.7,
        height=600
    )
    fig_3d.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, b=0, t=30),
        scene=dict(
            xaxis=dict(backgroundcolor="#0e1117", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(backgroundcolor="#0e1117", gridcolor="rgba(255,255,255,0.05)"),
            zaxis=dict(backgroundcolor="#0e1117", gridcolor="rgba(255,255,255,0.05)"),
        )
    )
    st.plotly_chart(fig_3d, use_container_width=True)
    
    # Decision curves (Elbow Curve & Silhouette Curve)
    st.subheader("📈 Clustering Decision Metrics")
    
    col_elbow, col_sil = st.columns(2)
    
    with col_elbow:
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=metrics["k_values"],
            y=metrics["inertias"],
            mode='lines+markers',
            line=dict(color='#7c3aed', width=3),
            marker=dict(size=8, symbol='circle')
        ))
        fig_elbow.update_layout(
            title="Elbow Method (Inertia vs. k)",
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Inertia (SSE)",
            template="plotly_dark"
        )
        st.plotly_chart(fig_elbow, use_container_width=True)
        
    with col_sil:
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Scatter(
            x=metrics["k_values"],
            y=metrics["silhouette_scores"],
            mode='lines+markers',
            line=dict(color='#db2777', width=3),
            marker=dict(size=8, symbol='diamond')
        ))
        fig_sil.update_layout(
            title="Silhouette Scores vs. k",
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Silhouette Coefficient",
            template="plotly_dark"
        )
        st.plotly_chart(fig_sil, use_container_width=True)

    # Single Customer Predictor Form inside Clustering
    st.markdown("---")
    st.subheader("🔮 Predict Segment for a New Customer")
    
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        in_recency = st.number_input("Recency (Days since last purchase)", min_value=0, value=30)
    with col_p2:
        in_frequency = st.number_input("Frequency (Total Invoices)", min_value=1, value=5)
    with col_p3:
        in_monetary = st.number_input("Monetary (Total Currency Spend)", min_value=0.0, value=500.0)
        
    if st.button("Predict Segment"):
        sample = pd.DataFrame(
            [[in_recency, in_frequency, in_monetary]],
            columns=["Recency", "Frequency", "Monetary"]
        )
        sample_scaled = scaler.transform(sample)
        pred_cluster = kmeans.predict(sample_scaled)[0]
        pred_label = cluster_labels.get(pred_cluster, f"Segment {pred_cluster}")
        
        st.markdown(f"""
        <div class='glass-card' style='border-left: 6px solid #7c3aed;'>
            <h4 style='margin: 0; color: #8b949e;'>Result</h4>
            <div style='font-size: 2.2rem; font-weight: 700; color: #7c3aed; margin: 10px 0;'>
                🎯 {pred_label}
            </div>
            <p style='color: #f0f6fc;'>
                This customer fits the profile of <strong>{pred_label}</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Segment Archetype Profiles
    st.markdown("### 📊 Segment Profiles")
    col_c1, col_c2 = st.columns(2)
    
    centers_scaled = kmeans.cluster_centers_
    centers_original = scaler.inverse_transform(centers_scaled)
    
    for cid in sorted(cluster_labels.keys()):
        c_name = cluster_labels[cid]
        c_vals = centers_original[cid]
        target_col = col_c1 if cid % 2 == 0 else col_c2
        
        target_col.markdown(f"""
        <div class='glass-card' style='padding: 15px; margin-bottom: 10px;'>
            <strong style='color: #58a6ff; font-size: 1.1rem;'>{c_name} Profile</strong>
            <div style='font-size: 0.9rem; color: #8b949e; margin-top: 8px;'>
                • Avg Recency: {c_vals[0]:.1f} days<br/>
                • Avg Frequency: {c_vals[1]:.1f} orders<br/>
                • Avg Monetary: ${c_vals[2]:.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)


# ------------------ 🎯 RECOMMENDATION PAGE ------------------
elif option == "Recommendation":
    st.markdown("<div class='glass-card'><h3>🔍 Product Recommendation Engine</h3>"
                "Discover products bought together based on item similarity matrices.</div>", unsafe_allow_html=True)
    
    all_products = sorted(list(similarity_df.columns))
    default_product = "WHITE HANGING HEART T-LIGHT HOLDER"
    default_index = all_products.index(default_product) if default_product in all_products else 0
    
    product_name = st.selectbox(
        "Select a product from the catalog",
        all_products,
        index=default_index
    )
    
    col_btn, _ = st.columns([1, 4])
    with col_btn:
        get_rec = st.button("Generate Recommendations")
        
    if get_rec or product_name:
        recommendations = recommend_products(product_name, similarity_df)
        
        if recommendations:
            st.markdown(f"<div class='alert-card alert-success'>Top 5 Recommended Products for <strong>{product_name}</strong>:</div>", unsafe_allow_html=True)
            
            # Display recommendation cards
            for idx, item in enumerate(recommendations):
                st.markdown(f"""
                <div class='product-card'>
                    <span style='color: #8b949e; font-weight: 600; margin-right: 10px;'>#{idx+1}</span>
                    <span class='product-name'>{item}</span>
                </div>
                """, unsafe_allow_html=True)
                
            # Render the Item Similarity Matrix / Heatmap
            st.markdown("---")
            st.subheader("🔥 Product Similarity Heatmap")
            
            # Sub-matrix of similarities among the product and its top recommendations
            all_heatmap_items = [product_name] + recommendations
            sub_matrix = similarity_df.loc[all_heatmap_items, all_heatmap_items]
            
            fig_heatmap = px.imshow(
                sub_matrix,
                labels=dict(x="Product Name", y="Product Name", color="Cosine Similarity"),
                x=sub_matrix.columns,
                y=sub_matrix.index,
                color_continuous_scale='Viridis',
                height=550
            )
            fig_heatmap.update_layout(
                template="plotly_dark",
                margin=dict(l=10, r=10, b=10, t=30)
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.error("No recommendation profiles found for this product.")