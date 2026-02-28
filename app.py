import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64
import json
from datetime import datetime
import time

# Import custom modules
from data_generator import RetailDataGenerator
from models import RetailMLModels
from recommendation_engine import RetailRecommendationEngine
from utils import (
    DataVisualizationUtils, ModelEvaluationUtils, DataProcessingUtils,
    ReportGenerator, ValidationUtils, set_custom_css, format_number,
    create_download_link
)
import numpy as np

# Configure Streamlit page with custom theme
st.set_page_config(
    page_title="🏪 Smart Retail Analytics Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown(set_custom_css(), unsafe_allow_html=True)

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}
if 'recommendation_engine' not in st.session_state:
    st.session_state.recommendation_engine = None
if 'cluster_labels' not in st.session_state:
    st.session_state.cluster_labels = None

def create_animated_metric_card(title, value, delta=None, color="blue"):
    """Create animated metric card."""
    delta_html = f"<span style='color: {'green' if delta and delta > 0 else 'red'};'>{delta:+.2f}</span>" if delta else ""
    
    st.markdown(f"""
    <div class="metric-card" style="border-left-color: {color};">
        <h3 style="margin: 0; color: #333;">{title}</h3>
        <p style="font-size: 2rem; margin: 0.5rem 0; font-weight: bold;">{value}</p>
        {f"<p>{delta_html}</p>" if delta else ""}
    </div>
    """, unsafe_allow_html=True)

def home_page():
    """Clean project overview page with Streamlit native components."""
    
    # Hero Section
    st.markdown("""
    <div class="css-1avcm2n">
        <h1 style="margin: 0; font-size: 3rem; font-weight: 700; text-align: center;">
            🏪 Smart Retail Analytics Platform
        </h1>
        <p style="margin: 1rem 0 0 0; font-size: 1.3rem; text-align: center; opacity: 0.9;">
            AI-Powered Business Intelligence for Modern Retail Enterprises
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Welcome Section
    st.markdown("---")
    st.markdown("### 🚀 Welcome to the Future of Retail Analytics")
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <span style="font-size: 3rem;">🚀</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Problem Statement
    st.markdown("### 🎯 Problem Statement")
    st.markdown("""
    <div class="section-container">
        <h4 style="color: var(--text-primary); margin-bottom: 1rem;">🧩 The Challenge</h4>
        <p style="margin: 0; line-height: 1.7; color: var(--text-secondary);">
            Retail businesses struggle with <strong>predicting customer behavior</strong>, 
            <strong>optimizing inventory</strong>, and <strong>personalizing marketing campaigns</strong>. 
            Traditional analytics tools are complex, expensive, and require data science expertise.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Objectives
    st.markdown("### 🚀 Objectives")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="professional-card">
            <h4 style="color: var(--primary); margin-bottom: 1rem;">📈 Predictive Analytics</h4>
            <ul style="margin: 0; padding-left: 1.5rem; color: var(--text-secondary); line-height: 1.6;">
                <li>🎯 Forecast sales with 85%+ accuracy</li>
                <li>⚡ Predict purchase decisions in real-time</li>
                <li>⭐ Classify customer loyalty levels</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="professional-card">
            <h4 style="color: var(--primary); margin-bottom: 1rem;">👥 Customer Intelligence</h4>
            <ul style="margin: 0; padding-left: 1.5rem; color: var(--text-secondary); line-height: 1.6;">
                <li>🔍 Segment customers using K-Means</li>
                <li>🤖 Generate AI-powered recommendations</li>
                <li>💰 Optimize marketing spend</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Algorithms
    st.markdown("### 🤖 Machine Learning Algorithms")
    
    col1, col2, col3, col4 = st.columns(4)
    
    algorithms = [
        ("📈 Linear Regression", "Sales Prediction", "Forecast monthly sales using customer demographics"),
        ("🌳 Decision Tree", "Purchase Classification", "Classify purchase intent with decision rules"),
        ("🎯 K-Nearest Neighbors", "Loyalty Prediction", "Categorize customers into loyalty tiers"),
        ("📊 K-Means", "Customer Segmentation", "Group customers by behavior for marketing")
    ]
    
    for i, (icon, name, description) in enumerate(algorithms):
        with [col1, col2, col3, col4][i]:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-icon">{icon}</div>
                <div class="feature-title">{name}</div>
                <div class="feature-description">{description}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Team Information
    st.markdown("### 👥 Team Information")
    
    col1, col2, col3 = st.columns(3)
    
    team_roles = [
        ("🧠 Data Science", "ML Model Development & Validation"),
        ("📊 Business Analytics", "Requirements Analysis & KPI Definition"),
        ("🎨 UI/UX Design", "User Experience & Interface Design")
    ]
    
    for i, (role, description) in enumerate(team_roles):
        with [col1, col2, col3][i]:
            st.markdown(f"""
            <div class="professional-card" style="text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">
                    {"👨‍💻" if i == 0 else "👩‍💼" if i == 1 else "🎨"}
                </div>
                <h4 style="color: var(--primary); margin-bottom: 0.5rem;">{role}</h4>
                <p style="margin: 0; color: var(--text-secondary);">{description}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Key Metrics
    st.markdown("### 📊 Platform Capabilities")
    
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = [
        ("ML Models", "4", "🤖"),
        ("Predictions", "3 Types", "🔮"),
        ("Accuracy", "85%+", "📈"),
        ("Segments", "4-6", "👥")
    ]
    
    for i, (label, value, icon) in enumerate(metrics):
        with [col1, col2, col3, col4][i]:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">
                    {icon}
                </div>
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer" style="text-align: center;">
        <div style="font-size: 2.5rem; margin-bottom: 1rem;">
            🏆
        </div>
        <h3 style="color: var(--text-primary); margin: 0 0 1rem 0;">
            Built for Internal Data Science Hackathon 2026
        </h3>
        <p style="margin: 0; color: var(--text-secondary);">
            🚀 Powered by Advanced Machine Learning & Modern Web Design
        </p>
    </div>
    """, unsafe_allow_html=True)

def dataset_page():
    """Dataset management and exploration page."""
    st.markdown("### 📊 Dataset Management")
    st.markdown("---")
    
    # Data generation section
    st.markdown("### 🎲 Generate Synthetic Data")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="section-container">
            <h4 style="color: var(--text-primary); margin-bottom: 1rem;">📋 Dataset Configuration</h4>
            <p style="margin: 0; line-height: 1.6; color: var(--text-secondary);">
                Generate a comprehensive retail dataset with customer demographics, 
                purchase behavior, and loyalty information.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("🔄 Generate Dataset", type="primary", use_container_width=True):
            with st.spinner("Generating synthetic retail data..."):
                generator = RetailDataGenerator(num_records=1500)
                df = generator.generate_complete_dataset()
                st.session_state.data = df
                st.success("✅ Dataset generated successfully!")
                st.info(f"📋 Generated {len(df)} records with {len(df.columns)} features")
    
    # Dataset preview
    if st.session_state.data is not None:
        st.markdown("---")
        st.markdown("### 📈 Dataset Preview")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("""
            <div class="section-container">
                <h4 style="color: var(--text-primary); margin-bottom: 1rem;">🔍 Data Overview</h4>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(st.session_state.data.head(10), use_container_width=True)
        
        with col2:
            st.markdown("""
            <div class="professional-card">
                <h4 style="color: var(--primary); margin-bottom: 1rem;">📊 Statistics</h4>
            </div>
            """, unsafe_allow_html=True)
            
            df = st.session_state.data
            st.metric("Total Records", f"{len(df):,}")
            st.metric("Total Features", f"{len(df.columns)}")
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        
        # Dataset summary
        st.markdown("---")
        st.markdown("### 📋 Dataset Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="professional-card">
                <h4 style="color: var(--primary); margin-bottom: 1rem;">📈 Numerical Features</h4>
            </div>
            """, unsafe_allow_html=True)
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            for col in numeric_cols[:5]:  # Show first 5
                st.write(f"• {col}")
        
        with col2:
            st.markdown("""
            <div class="professional-card">
                <h4 style="color: var(--primary); margin-bottom: 1rem;">🏷️ Categorical Features</h4>
            </div>
            """, unsafe_allow_html=True)
            
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            for col in categorical_cols[:5]:  # Show first 5
                st.write(f"• {col}")
        
        # Download option
        st.markdown("---")
        st.markdown("### 💾 Export Dataset")
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="retail_dataset.csv",
            mime="text/csv",
            type="primary"
        )
    
    else:
        st.warning("⚠️ No dataset available. Please generate data first.")

def model_training_page():
    """Professional model training page."""
    st.markdown("### 🤖 Model Training")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please generate dataset first!")
        return
    
    st.markdown("""
    <div class="section-container">
        <h4 style="color: var(--primary); margin-bottom: 1rem;">🚀 Training Pipeline</h4>
        <p style="margin: 0; line-height: 1.6; color: var(--text-secondary);">
            Our platform automatically trains four machine learning models when you visit this page. 
            Each model is optimized for specific retail analytics tasks.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Training Progress
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="professional-card">
            <h4 style="color: var(--primary); margin-bottom: 1rem;">📈 Models to Train</h4>
            <ul style="margin: 0; padding-left: 1.5rem; color: var(--text-secondary); line-height: 1.6;">
                <li><strong>Linear Regression</strong> - Sales Prediction</li>
                <li><strong>Decision Tree</strong> - Purchase Classification</li>
                <li><strong>K-Nearest Neighbors</strong> - Loyalty Prediction</li>
                <li><strong>K-Means</strong> - Customer Segmentation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            with st.spinner("Training models... This may take up to 30 seconds."):
                models = RetailMLModels()
                df = st.session_state.data
                
                # Train all models and store in session state
                sales_model, sales_metrics = models.train_sales_prediction_model(df)
                purchase_model, purchase_metrics = models.train_purchase_decision_model(df)
                loyalty_model, loyalty_metrics = models.train_loyalty_prediction_model(df)
                segmentation_model, segmentation_metrics, cluster_labels = models.train_customer_segmentation_model(df)
                
                # Store models in session state
                st.session_state.models = {
                    'sales': sales_model,
                    'purchase': purchase_model,
                    'loyalty': loyalty_model,
                    'segmentation': segmentation_model
                }
                
                # Store metrics in session state
                st.session_state.metrics = {
                    'sales': sales_metrics,
                    'purchase': purchase_metrics,
                    'loyalty': loyalty_metrics,
                    'segmentation': segmentation_metrics
                }
                
                # Store other components
                st.session_state.cluster_labels = cluster_labels
                st.session_state.models_object = models  # Keep for predictions
                
                st.success("✅ All models trained successfully!")
                st.info(f"📊 Trained 4 models with performance metrics")
                st.info("📊 Models are ready for predictions!")
    
    # Training Status
    if st.session_state.models is not None:
        st.markdown("---")
        st.markdown("### 📊 Training Status")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Sales Model</div>
                    <div class="metric-value">✅</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Purchase Model</div>
                    <div class="metric-value">✅</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Loyalty Model</div>
                    <div class="metric-value">✅</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Segmentation</div>
                    <div class="metric-value">✅</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Performance Metrics
        st.markdown("---")
        st.markdown("### 📈 Model Performance")
        
        if 'sales' in st.session_state.metrics:
            metrics = st.session_state.metrics['sales']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                    <div class="professional-card">
                        <h4 style="color: var(--primary); margin-bottom: 1rem;">📈 Sales Model</h4>
                        <p><strong>R² Score:</strong> {metrics['test_r2']:.4f}</p>
                        <p><strong>RMSE:</strong> ${np.sqrt(metrics['test_mse']):.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                if 'purchase' in st.session_state.metrics:
                    metrics = st.session_state.metrics['purchase']
                    st.markdown(f"""
                        <div class="professional-card">
                            <h4 style="color: var(--primary); margin-bottom: 1rem;">🎯 Purchase Model</h4>
                            <p><strong>Accuracy:</strong> {metrics['test_accuracy']:.4f}</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            with col3:
                if 'loyalty' in st.session_state.metrics:
                    metrics = st.session_state.metrics['loyalty']
                    st.markdown(f"""
                        <div class="professional-card">
                            <h4 style="color: var(--primary); margin-bottom: 1rem;">⭐ Loyalty Model</h4>
                            <p><strong>Accuracy:</strong> {metrics['test_accuracy']:.4f}</p>
                        </div>
                        """, unsafe_allow_html=True)

def prediction_page():
    """Prediction interface page."""
    st.markdown("### 🔮 Prediction Interface")
    st.markdown("---")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please generate dataset first!")
        return
    
    if st.session_state.models is None:
        st.warning("⚠️ Please train models first!")
        return
    
    # Prediction tabs
    tab1, tab2, tab3 = st.tabs(["📈 Sales Prediction", "🎯 Purchase Decision", "⭐ Loyalty Prediction"])
    
    with tab1:
        st.markdown("### 📈 Sales Prediction")
        st.markdown("""
        <div class="section-container">
            <h4 style="color: var(--text-primary); margin-bottom: 1rem;">🔮 Predict Monthly Sales</h4>
            <p style="margin: 0; line-height: 1.6; color: var(--text-secondary);">
                Enter customer details to predict their monthly sales amount.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Input form
        with st.form("sales_prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                customer_age = st.number_input("Customer Age", min_value=18, max_value=80, value=35)
                annual_income = st.number_input("Annual Income ($)", min_value=10000, max_value=200000, value=50000)
                purchase_frequency = st.number_input("Purchase Frequency", min_value=1, max_value=20, value=5)
            
            with col2:
                discount_offered = st.number_input("Discount Offered (%)", min_value=0.0, max_value=50.0, value=10.0)
                marketing_spend = st.number_input("Marketing Spend ($)", min_value=0, max_value=1000, value=50)
                previous_amount = st.number_input("Previous Purchase Amount ($)", min_value=0, max_value=10000, value=500)
            
            # Categorical inputs
            col3, col4 = st.columns(2)
            
            with col3:
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                product_category = st.selectbox("Product Category", ["Electronics", "Clothing", "Food", "Books", "Sports"])
            
            with col4:
                store_location = st.selectbox("Store Location", ["Downtown", "Mall", "Suburb", "Online"])
                customer_tenure = st.number_input("Customer Tenure (years)", min_value=0, max_value=20, value=3)
            
            submit_button = st.form_submit_button("🔮 Predict Sales", type="primary")
            
            if submit_button:
                input_data = {
                    'Customer_Age': customer_age,
                    'Gender': gender,
                    'Annual_Income': annual_income,
                    'Purchase_Frequency': purchase_frequency,
                    'Discount_Offered': discount_offered,
                    'Product_Category': product_category,
                    'Marketing_Spend': marketing_spend,
                    'Seasonal_Demand_Index': 1.0,
                    'Store_Location': store_location,
                    'Previous_Purchase_Amount': previous_amount,
                    'Customer_Tenure': customer_tenure
                }
                
                try:
                    prediction = st.session_state.models.predict_sales(input_data)
                    st.success(f"🎯 Predicted Monthly Sales: ${prediction:.2f}")
                    
                    # Display in a nice card
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Predicted Sales</div>
                        <div class="metric-value">${prediction:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")
    
    with tab2:
        st.markdown("### 🎯 Purchase Decision")
        st.markdown("""
        <div class="section-container">
            <h4 style="color: var(--text-primary); margin-bottom: 1rem;">🎯 Predict Purchase Intent</h4>
            <p style="margin: 0; line-height: 1.6; color: var(--text-secondary);">
                Predict whether a customer will make a purchase.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("purchase_prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                customer_age = st.number_input("Customer Age", min_value=18, max_value=80, value=35, key="purchase_age")
                annual_income = st.number_input("Annual Income ($)", min_value=10000, max_value=200000, value=50000, key="purchase_income")
                discount_offered = st.number_input("Discount Offered (%)", min_value=0.0, max_value=50.0, value=10.0, key="purchase_discount")
            
            with col2:
                purchase_frequency = st.number_input("Purchase Frequency", min_value=1, max_value=20, value=5, key="purchase_freq")
                marketing_spend = st.number_input("Marketing Spend ($)", min_value=0, max_value=1000, value=50, key="purchase_marketing")
                previous_amount = st.number_input("Previous Purchase Amount ($)", min_value=0, max_value=10000, value=500, key="purchase_previous")
            
            col3, col4 = st.columns(2)
            
            with col3:
                gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="purchase_gender")
                product_category = st.selectbox("Product Category", ["Electronics", "Clothing", "Food", "Books", "Sports"], key="purchase_category")
            
            with col4:
                store_location = st.selectbox("Store Location", ["Downtown", "Mall", "Suburb", "Online"], key="purchase_location")
                customer_tenure = st.number_input("Customer Tenure (years)", min_value=0, max_value=20, value=3, key="purchase_tenure")
            
            submit_button = st.form_submit_button("🎯 Predict Purchase", type="primary")
            
            if submit_button:
                input_data = {
                    'Customer_Age': customer_age,
                    'Gender': gender,
                    'Annual_Income': annual_income,
                    'Purchase_Frequency': purchase_frequency,
                    'Discount_Offered': discount_offered,
                    'Product_Category': product_category,
                    'Marketing_Spend': marketing_spend,
                    'Seasonal_Demand_Index': 1.0,
                    'Store_Location': store_location,
                    'Previous_Purchase_Amount': previous_amount,
                    'Customer_Tenure': customer_tenure
                }
                
                try:
                    prediction = st.session_state.models.predict_purchase_decision(input_data)
                    st.success(f"🎯 Purchase Decision: {prediction}")
                    
                    # Display result
                    color = "var(--success)" if prediction == "Yes" else "var(--warning)"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Purchase Decision</div>
                        <div class="metric-value">{prediction}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")
    
    with tab3:
        st.markdown("### ⭐ Loyalty Prediction")
        st.markdown("""
        <div class="section-container">
            <h4 style="color: var(--text-primary); margin-bottom: 1rem;">⭐ Predict Customer Loyalty</h4>
            <p style="margin: 0; line-height: 1.6; color: var(--text-secondary);">
                Predict customer loyalty category (Gold, Silver, Bronze).
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("loyalty_prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                customer_age = st.number_input("Customer Age", min_value=18, max_value=80, value=35, key="loyalty_age")
                annual_income = st.number_input("Annual Income ($)", min_value=10000, max_value=200000, value=50000, key="loyalty_income")
                purchase_frequency = st.number_input("Purchase Frequency", min_value=1, max_value=20, value=5, key="loyalty_freq")
            
            with col2:
                discount_offered = st.number_input("Discount Offered (%)", min_value=0.0, max_value=50.0, value=10.0, key="loyalty_discount")
                marketing_spend = st.number_input("Marketing Spend ($)", min_value=0, max_value=1000, value=50, key="loyalty_marketing")
                previous_amount = st.number_input("Previous Purchase Amount ($)", min_value=0, max_value=10000, value=500, key="loyalty_previous")
            
            col3, col4 = st.columns(2)
            
            with col3:
                gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="loyalty_gender")
                product_category = st.selectbox("Product Category", ["Electronics", "Clothing", "Food", "Books", "Sports"], key="loyalty_category")
            
            with col4:
                store_location = st.selectbox("Store Location", ["Downtown", "Mall", "Suburb", "Online"], key="loyalty_location")
                customer_tenure = st.number_input("Customer Tenure (years)", min_value=0, max_value=20, value=3, key="loyalty_tenure")
            
            submit_button = st.form_submit_button("⭐ Predict Loyalty", type="primary")
            
            if submit_button:
                input_data = {
                    'Customer_Age': customer_age,
                    'Gender': gender,
                    'Annual_Income': annual_income,
                    'Purchase_Frequency': purchase_frequency,
                    'Discount_Offered': discount_offered,
                    'Product_Category': product_category,
                    'Marketing_Spend': marketing_spend,
                    'Seasonal_Demand_Index': 1.0,
                    'Store_Location': store_location,
                    'Previous_Purchase_Amount': previous_amount,
                    'Customer_Tenure': customer_tenure
                }
                
                try:
                    prediction = st.session_state.models.predict_loyalty_category(input_data)
                    st.success(f"⭐ Loyalty Category: {prediction}")
                    
                    # Display result with appropriate styling
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Loyalty Category</div>
                        <div class="metric-value">{prediction}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")

def visualization_page():
    """Visualization and insights page."""
    st.markdown("### 📈 Visualizations & Insights")
    st.markdown("---")
    
    # Debug session state
    st.write("🔍 Session State Debug:")
    st.write(f"- Data exists: {st.session_state.data is not None}")
    st.write(f"- Data type: {type(st.session_state.data)}")
    
    # Safety check
    if st.session_state.data is None:
        st.warning("⚠️ Please generate dataset first!")
        st.info("📋 Go to '📊 Data Generator' to create synthetic data")
        return
    
    try:
        df = st.session_state.data
        st.write("🔍 Debug: Data loaded:", df.shape)
        st.write("📊 Columns:", list(df.columns))
        
        # Data Overview Section
        st.markdown("### 📊 Data Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="section-container">
                <h4 style="color: var(--text-primary); margin-bottom: 1rem;">📈 Sales Distribution</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Sales distribution histogram
            if 'Monthly_Sales' in df.columns:
                fig = px.histogram(df, x='Monthly_Sales', nbins=30, 
                                  title='Monthly Sales Distribution',
                                  color_discrete_sequence=['var(--primary)'])
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Monthly_Sales column not found")
                st.write("Available columns:", list(df.columns))
        
        with col2:
            st.markdown("""
            <div class="section-container">
                <h4 style="color: var(--text-primary); margin-bottom: 1rem;">👥 Customer Demographics</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Age distribution
            if 'Customer_Age' in df.columns:
                fig = px.histogram(df, x='Customer_Age', nbins=20,
                                  title='Customer Age Distribution',
                                  color_discrete_sequence=['var(--secondary)'])
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Customer_Age column not found")
                st.write("Available columns:", list(df.columns))
        
        # Customer Analysis
        st.markdown("---")
        st.markdown("### 👥 Customer Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="section-container">
                <h4 style="color: var(--text-primary); margin-bottom: 1rem;">🎯 Purchase Decision Analysis</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Purchase decision pie chart
            if 'Purchase_Decision' in df.columns:
                purchase_counts = df['Purchase_Decision'].value_counts()
                fig = px.pie(values=purchase_counts.values, names=purchase_counts.index,
                             title='Purchase Decision Distribution',
                             color_discrete_sequence=['var(--success)', 'var(--warning)'])
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Purchase_Decision column not found")
        
        with col2:
            st.markdown("""
            <div class="section-container">
                <h4 style="color: var(--text-primary); margin-bottom: 1rem;">⭐ Loyalty Categories</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Loyalty category distribution
            if 'Loyalty_Category' in df.columns:
                loyalty_counts = df['Loyalty_Category'].value_counts()
                fig = px.bar(x=loyalty_counts.index, y=loyalty_counts.values,
                            title='Loyalty Category Distribution',
                            color_discrete_sequence=['var(--primary)'])
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ Loyalty_Category column not found")
        
        # Correlation Analysis
        st.markdown("---")
        st.markdown("### 🔗 Correlation Analysis")
        
        st.markdown("""
        <div class="section-container">
            <h4 style="color: var(--text-primary); margin-bottom: 1rem;">📊 Feature Correlations</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Create correlation matrix for numerical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        st.write(f"🔍 Debug: Found {len(numeric_cols)} numerical columns:", list(numeric_cols))
        
        if len(numeric_cols) > 0:
            corr_matrix = df[numeric_cols].corr()
            
            fig = px.imshow(corr_matrix, 
                           title='Feature Correlation Matrix',
                           color_continuous_scale='RdBu_r',
                           aspect="auto")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ No numerical columns found for correlation")
        
        # Business Insights
        st.markdown("---")
        st.markdown("### 💼 Business Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="professional-card">
                <h4 style="color: var(--primary); margin-bottom: 1rem;">📈 Key Metrics</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Calculate metrics safely
            if 'Monthly_Sales' in df.columns:
                avg_sales = df['Monthly_Sales'].mean()
                st.metric("Average Monthly Sales", f"${avg_sales:.2f}")
            
            total_customers = len(df)
            st.metric("Total Customers", f"{total_customers:,}")
            
            if 'Purchase_Decision' in df.columns:
                purchase_rate = (df['Purchase_Decision'] == 'Yes').mean() * 100
                st.metric("Purchase Rate", f"{purchase_rate:.1f}%")
        
        with col2:
            st.markdown("""
            <div class="professional-card">
                <h4 style="color: var(--primary); margin-bottom: 1rem;">🎯 Top Insights</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Find most valuable customer segment
            if 'Monthly_Sales' in df.columns and 'Annual_Income' in df.columns:
                high_value_customers = df[df['Monthly_Sales'] > df['Monthly_Sales'].quantile(0.8)]
                avg_income_high = high_value_customers['Annual_Income'].mean()
                
                st.write(f"💎 **High-value customers** earn ${avg_income_high:,.0f} on average")
                st.write(f"🎯 **Top 20% customers** represent {len(high_value_customers)} people")
                st.write(f"📈 **Sales variance**: ${df['Monthly_Sales'].std():.2f}")
            else:
                st.warning("⚠️ Required columns for insights not found")
    
    except Exception as e:
        st.error(f"❌ Error in visualization page: {str(e)}")
        st.write("🔍 Debug info:")
        st.write(f"- Data type: {type(st.session_state.data)}")
        if st.session_state.data is not None:
            st.write(f"- Data shape: {st.session_state.data.shape}")
            st.write(f"- Data columns: {list(st.session_state.data.columns)}")

def model_performance_page():
    """Professional model performance page."""
    st.markdown("### 📊 Model Performance Dashboard")
    st.markdown("---")
    
    # Debug session state
    st.write("🔍 Session State Debug:")
    st.write(f"- Models trained: {len(st.session_state.models)}")
    st.write(f"- Metrics available: {len(st.session_state.metrics)}")
    st.write(f"- Data loaded: {st.session_state.data is not None}")
    
    # Safety check
    if not st.session_state.models or not st.session_state.metrics:
        st.warning("⚠️ Please train models first!")
        st.info("📋 Go to '🤖 Model Training' to train all models")
        return
    
    try:
        # Debug info
        st.write("🔍 Debug: Models loaded:", list(st.session_state.models.keys()))
        st.write("📊 Available metrics:", list(st.session_state.metrics.keys()))
        
        # Performance Overview
        st.markdown("### 📈 Performance Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="section-container">
                <h4 style="color: var(--text-primary); margin-bottom: 1rem;">🎯 Model Comparison</h4>
                <p style="margin: 0; line-height: 1.6; color: var(--text-secondary);">
                    Compare performance metrics across all trained models
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Create performance comparison chart using session state metrics
            performance_data = []
            model_names = []
            
            for model_name, metrics in st.session_state.metrics.items():
                if 'test_r2' in metrics:
                    performance_data.append(metrics['test_r2'])
                    model_names.append(model_name.replace('_', ' ').title())
                elif 'test_accuracy' in metrics:
                    performance_data.append(metrics['test_accuracy'])
                    model_names.append(model_name.replace('_', ' ').title())
            
            if performance_data:
                fig = go.Figure(data=[
                    go.Bar(x=model_names, y=performance_data, 
                           marker_color='var(--primary)',
                           text=[f'{val:.3f}' for val in performance_data],
                           textposition='auto')
                ])
                
                fig.update_layout(
                    title="Model Performance Comparison",
                    xaxis_title="Models",
                    yaxis_title="Performance Score",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ No performance metrics available")
        
        # Detailed Metrics
        st.markdown("---")
        st.markdown("### 📋 Detailed Performance Metrics")
        
        for model_name, metrics in st.session_state.metrics.items():
            with st.expander(f"🔍 {model_name.replace('_', ' ').title()} Details"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div class="professional-card">
                        <h4 style="color: var(--primary); margin-bottom: 1rem;">📊 Metrics</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for metric_name, metric_value in metrics.items():
                        if isinstance(metric_value, (int, float)):
                            st.metric(metric_name.replace('_', ' ').title(), f"{metric_value:.4f}")
                        else:
                            st.write(f"**{metric_name.replace('_', ' ').title()}:** {metric_value}")
                
                with col2:
                    # Feature importance for applicable models
                    if 'feature_importance' in metrics:
                        importance = metrics['feature_importance']
                        if importance:
                            fig = go.Figure(data=[
                                go.Bar(
                                    x=list(importance.keys())[:10],  # Top 10 features
                                    y=list(importance.values())[:10],
                                    marker_color='var(--primary)'
                                )
                            ])
                            
                            fig.update_layout(
                                title="Feature Importance",
                                xaxis_title="Features",
                                yaxis_title="Importance",
                                height=300
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
        
        # Overall Summary
        st.markdown("---")
        st.markdown("### 📊 Overall Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_models = len(st.session_state.models)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Total Models</div>
                <div class="metric-value">{total_models}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Calculate average performance
            performances = []
            for metrics in st.session_state.metrics.values():
                if 'test_accuracy' in metrics:
                    performances.append(metrics['test_accuracy'])
                elif 'test_r2' in metrics:
                    performances.append(metrics['test_r2'])
            
            avg_performance = sum(performances) / len(performances) if performances else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Avg Performance</div>
                <div class="metric-value">{avg_performance:.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Status</div>
                <div class="metric-value">✅ Ready</div>
            </div>
            """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"❌ Error in model performance page: {str(e)}")
        st.write("🔍 Debug info:")
        st.write(f"- Models type: {type(st.session_state.models)}")
        if st.session_state.models is not None:
            st.write(f"- Performance metrics keys: {list(st.session_state.metrics.keys())}")
    """Dataset management and exploration page."""
    st.title("�� Dataset Management")
    st.markdown("---")
    
    # Data generation section
    st.markdown("### 🎲 Generate Synthetic Data")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        num_records = st.slider("Number of Records", min_value=500, max_value=5000, value=1500, step=100)
        
        if st.button("🔄 Generate Dataset", type="primary"):
            with st.spinner("Generating synthetic retail data..."):
                generator = RetailDataGenerator(num_records=num_records)
                df = generator.generate_complete_dataset()
                st.session_state.data = df
                
                # Show success message
                st.success(f"✅ Successfully generated {num_records} records!")
                
                # Display data preview
                st.markdown("#### 📋 Data Preview")
                st.dataframe(df.head(10))
                
                # Data summary
                summary = generator.get_data_summary(df)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    create_animated_metric_card("Total Records", summary['total_records'])
                with col2:
                    create_animated_metric_card("Features", summary['total_features'])
                with col3:
                    create_animated_metric_card("Target Variables", len(summary['target_variables']))
                with col4:
                    create_animated_metric_card("Missing Values", summary['missing_values'])
    
    with col2:
        st.markdown("#### 📥 Upload CSV")
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.data = df
                st.success("✅ File uploaded successfully!")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"❌ Error uploading file: {str(e)}")
    
    # Data exploration section
    if st.session_state.data is not None:
        st.markdown("---")
        st.markdown("### 🔍 Data Exploration")
        
        df = st.session_state.data
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Summary Statistics", "📊 Visualizations", "🔍 Data Quality", "💾 Export"])
        
        with tab1:
            st.markdown("#### 📋 Dataset Information")
            
            # Basic info
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Dataset Shape:**", df.shape)
                st.write("**Data Types:**")
                st.write(df.dtypes)
            
            with col2:
                st.write("**Numeric Columns Summary:**")
                numeric_df = df.select_dtypes(include=[np.number])
                st.write(numeric_df.describe())
            
            # Categorical columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                st.markdown("#### 📝 Categorical Variables Distribution")
                for col in categorical_cols:
                    if col != 'Customer_ID':
                        fig = DataVisualizationUtils.create_pie_chart(df, col)
                        st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("#### 📊 Data Visualizations")
            
            # Correlation heatmap
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                fig = DataVisualizationUtils.create_correlation_heatmap(df)
                st.plotly_chart(fig, use_container_width=True)
            
            # Distribution plots
            st.markdown("#### 📈 Feature Distributions")
            selected_col = st.selectbox("Select feature for distribution", numeric_cols)
            if selected_col:
                fig = DataVisualizationUtils.create_distribution_plot(df, selected_col)
                st.plotly_chart(fig, use_container_width=True)
            
            # Scatter plots
            if len(numeric_cols) >= 2:
                st.markdown("#### 🔗 Feature Relationships")
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("X-axis", numeric_cols, key="scatter_x")
                with col2:
                    y_col = st.selectbox("Y-axis", numeric_cols, key="scatter_y")
                
                if x_col and y_col and x_col != y_col:
                    fig = DataVisualizationUtils.create_scatter_plot(df, x_col, y_col)
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("#### 🔍 Data Quality Assessment")
            
            # Missing values
            missing_data = df.isnull().sum()
            if missing_data.sum() > 0:
                st.write("**Missing Values:**")
                missing_df = missing_data[missing_data > 0].reset_index()
                missing_df.columns = ['Column', 'Missing Count']
                st.dataframe(missing_df)
            else:
                st.success("✅ No missing values found!")
            
            # Outlier detection
            st.markdown("#### 🚨 Outlier Detection")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            selected_outlier_col = st.selectbox("Select column for outlier analysis", numeric_cols)
            
            if selected_outlier_col:
                outliers = DataProcessingUtils.detect_outliers(df, selected_outlier_col)
                st.write(f"**Outliers in {selected_outlier_col}:** {len(outliers)} records")
                
                if len(outliers) > 0:
                    st.dataframe(outliers.head())
        
        with tab4:
            st.markdown("#### 💾 Export Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📥 Download as CSV"):
                    csv = df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="retail_dataset.csv">Download CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)
            
            with col2:
                if st.button("📄 Generate Data Report"):
                    report = ReportGenerator.generate_data_profile_report(df)
                    st.json(report)

def prediction_page():
    """Prediction page with all ML models."""
    st.title("🔮 Predictions & Analytics")
    st.markdown("---")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please generate or upload a dataset first!")
        return
    
    if not st.session_state.models or 'models_object' not in st.session_state:
        st.info("🔄 Training ML models... This may take a moment.")
        with st.spinner("Training models..."):
            models = RetailMLModels()
            df = st.session_state.data
            
            # Train all models
            models.train_sales_prediction_model(df)
            models.train_purchase_decision_model(df)
            models.train_loyalty_prediction_model(df)
            models.train_customer_segmentation_model(df)
            
            st.session_state.models_object = models
            st.success("✅ Models trained successfully!")
    
    models = st.session_state.models_object
    
    # Prediction tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💰 Sales Prediction", "🎯 Purchase Decision", "⭐ Loyalty Category", "📊 Model Performance"])
    
    with tab1:
        st.markdown("### 💰 Monthly Sales Prediction")
        st.markdown("Predict monthly sales using Linear Regression model.")
        
        # Input form
        col1, col2 = st.columns(2)
        
        with col1:
            customer_age = st.number_input("Customer Age", min_value=18, max_value=100, value=35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            annual_income = st.number_input("Annual Income ($)", min_value=20000, max_value=200000, value=50000)
            purchase_frequency = st.number_input("Purchase Frequency (per month)", min_value=1, max_value=30, value=5)
        
        with col2:
            discount_offered = st.slider("Discount Offered (%)", min_value=0.0, max_value=50.0, value=10.0)
            marketing_spend = st.number_input("Marketing Spend ($)", min_value=5.0, max_value=500.0, value=50.0)
            seasonal_demand = st.slider("Seasonal Demand Index", min_value=0.5, max_value=2.0, value=1.0)
            previous_purchase = st.number_input("Previous Purchase Amount ($", min_value=100, max_value=5000, value=500)
        
        # Additional inputs
        col1, col2 = st.columns(2)
        with col1:
            product_category = st.selectbox("Product Category", 
                ["Electronics", "Clothing", "Groceries", "Home & Garden", "Sports", "Books", "Beauty", "Toys"])
            store_location = st.selectbox("Store Location", 
                ["Downtown", "Suburban", "Mall", "Airport", "Online"])
        with col2:
            customer_tenure = st.number_input("Customer Tenure (years)", min_value=1, max_value=20, value=3)
        
        if st.button("🔮 Predict Sales", type="primary"):
            # Prepare input data
            input_data = {
                'Customer_Age': customer_age,
                'Gender': gender,
                'Annual_Income': annual_income,
                'Purchase_Frequency': purchase_frequency,
                'Discount_Offered': discount_offered,
                'Product_Category': product_category,
                'Marketing_Spend': marketing_spend,
                'Seasonal_Demand_Index': seasonal_demand,
                'Store_Location': store_location,
                'Previous_Purchase_Amount': previous_purchase,
                'Customer_Tenure': customer_tenure
            }
            
            try:
                prediction = models.predict_sales(input_data)
                st.success(f"💰 **Predicted Monthly Sales: ${prediction:,.2f}**")
                
                # Feature importance
                if 'sales_prediction' in models.performance_metrics:
                    feature_importance = models.performance_metrics['sales_prediction']['feature_importance']
                    fig = ModelEvaluationUtils.plot_feature_importance(feature_importance, "Sales Prediction Feature Importance")
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
        
        # Model performance
        if 'sales_prediction' in models.performance_metrics:
            st.markdown("#### 📊 Model Performance")
            metrics = models.performance_metrics['sales_prediction']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                create_animated_metric_card("R² Score", f"{metrics['test_r2']:.4f}")
            with col2:
                create_animated_metric_card("RMSE", f"${np.sqrt(metrics['test_mse']):.2f}")
            with col3:
                create_animated_metric_card("MAE", f"${metrics['test_mae']:.2f}")
    
    with tab2:
        st.markdown("### 🎯 Purchase Decision Prediction")
        st.markdown("Predict whether a customer will make a purchase using Decision Tree model.")
        
        # Input form (similar to sales prediction)
        col1, col2 = st.columns(2)
        
        with col1:
            customer_age = st.number_input("Customer Age", min_value=18, max_value=100, value=35, key="pd_age")
            gender = st.selectbox("Gender", ["Male", "Female"], key="pd_gender")
            annual_income = st.number_input("Annual Income ($)", min_value=20000, max_value=200000, value=50000, key="pd_income")
            purchase_frequency = st.number_input("Purchase Frequency", min_value=1, max_value=30, value=5, key="pd_freq")
        
        with col2:
            discount_offered = st.slider("Discount Offered (%)", min_value=0.0, max_value=50.0, value=10.0, key="pd_discount")
            marketing_spend = st.number_input("Marketing Spend ($)", min_value=5.0, max_value=500.0, value=50.0, key="pd_marketing")
            previous_purchase = st.number_input("Previous Purchase ($)", min_value=100, max_value=5000, value=500, key="pd_previous")
            customer_tenure = st.number_input("Customer Tenure (years)", min_value=1, max_value=20, value=3, key="pd_tenure")
        
        # Additional inputs
        col1, col2 = st.columns(2)
        with col1:
            product_category = st.selectbox("Product Category", 
                ["Electronics", "Clothing", "Groceries", "Home & Garden", "Sports", "Books", "Beauty", "Toys"], 
                key="pd_category")
            store_location = st.selectbox("Store Location", 
                ["Downtown", "Suburban", "Mall", "Airport", "Online"], key="pd_location")
        
        if st.button("🎯 Predict Purchase Decision", type="primary"):
            input_data = {
                'Customer_Age': customer_age,
                'Gender': gender,
                'Annual_Income': annual_income,
                'Purchase_Frequency': purchase_frequency,
                'Discount_Offered': discount_offered,
                'Product_Category': product_category,
                'Marketing_Spend': marketing_spend,
                'Store_Location': store_location,
                'Previous_Purchase_Amount': previous_purchase,
                'Customer_Tenure': customer_tenure
            }
            
            try:
                prediction = models.predict_purchase_decision(input_data)
                if prediction == "Yes":
                    st.success(f"✅ **Purchase Decision: YES** - Customer is likely to make a purchase!")
                else:
                    st.warning(f"❌ **Purchase Decision: NO** - Customer is unlikely to make a purchase.")
                
                # Feature importance
                if 'purchase_decision' in models.performance_metrics:
                    feature_importance = models.performance_metrics['purchase_decision']['feature_importance']
                    fig = ModelEvaluationUtils.plot_feature_importance(feature_importance, "Purchase Decision Feature Importance")
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
        
        # Model performance
        if 'purchase_decision' in models.performance_metrics:
            st.markdown("#### 📊 Model Performance")
            metrics = models.performance_metrics['purchase_decision']
            
            col1, col2 = st.columns(2)
            with col1:
                create_animated_metric_card("Accuracy", f"{metrics['test_accuracy']:.4f}")
            with col2:
                create_animated_metric_card("Optimal Depth", "10")
            
            # Confusion Matrix
            if 'confusion_matrix' in metrics:
                fig = ModelEvaluationUtils.plot_confusion_matrix(
                    metrics['confusion_matrix'], 
                    models.encoders['Purchase_Decision'].classes_,
                    "Purchase Decision Confusion Matrix"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### ⭐ Loyalty Category Prediction")
        st.markdown("Predict customer loyalty category using K-Nearest Neighbors model.")
        
        # Input form
        col1, col2 = st.columns(2)
        
        with col1:
            customer_age = st.number_input("Customer Age", min_value=18, max_value=100, value=35, key="loyalty_age")
            gender = st.selectbox("Gender", ["Male", "Female"], key="loyalty_gender")
            annual_income = st.number_input("Annual Income ($)", min_value=20000, max_value=200000, value=50000, key="loyalty_income")
            purchase_frequency = st.number_input("Purchase Frequency", min_value=1, max_value=30, value=5, key="loyalty_freq")
        
        with col2:
            discount_offered = st.slider("Discount Offered (%)", min_value=0.0, max_value=50.0, value=10.0, key="loyalty_discount")
            marketing_spend = st.number_input("Marketing Spend ($)", min_value=5.0, max_value=500.0, value=50.0, key="loyalty_marketing")
            previous_purchase = st.number_input("Previous Purchase ($)", min_value=100, max_value=5000, value=500, key="loyalty_previous")
            customer_tenure = st.number_input("Customer Tenure (years)", min_value=1, max_value=20, value=3, key="loyalty_tenure")
        
        # Additional inputs
        col1, col2 = st.columns(2)
        with col1:
            product_category = st.selectbox("Product Category", 
                ["Electronics", "Clothing", "Groceries", "Home & Garden", "Sports", "Books", "Beauty", "Toys"], 
                key="loyalty_category")
            store_location = st.selectbox("Store Location", 
                ["Downtown", "Suburban", "Mall", "Airport", "Online"], key="loyalty_location")
        
        if st.button("⭐ Predict Loyalty Category", type="primary"):
            input_data = {
                'Customer_Age': customer_age,
                'Gender': gender,
                'Annual_Income': annual_income,
                'Purchase_Frequency': purchase_frequency,
                'Discount_Offered': discount_offered,
                'Product_Category': product_category,
                'Marketing_Spend': marketing_spend,
                'Store_Location': store_location,
                'Previous_Purchase_Amount': previous_purchase,
                'Customer_Tenure': customer_tenure
            }
            
            try:
                prediction = models.predict_loyalty_category(input_data)
                
                # Display result with appropriate styling
                if prediction == "Gold":
                    st.success(f"🏆 **Loyalty Category: GOLD** - Premium customer!")
                elif prediction == "Silver":
                    st.info(f"🥈 **Loyalty Category: SILVER** - Valued customer!")
                else:
                    st.warning(f"🥉 **Loyalty Category: BRONZE** - Developing customer!")
                
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
        
        # Model performance
        if 'loyalty_prediction' in models.performance_metrics:
            st.markdown("#### 📊 Model Performance")
            metrics = models.performance_metrics['loyalty_prediction']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                create_animated_metric_card("Accuracy", f"{metrics['test_accuracy']:.4f}")
            with col2:
                create_animated_metric_card("Optimal K", metrics['optimal_k'])
            with col3:
                create_animated_metric_card("Classes", "3")
            
            # K selection visualization
            if 'k_scores' in metrics:
                k_scores = metrics['k_scores']
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(k_scores.keys()),
                    y=list(k_scores.values()),
                    mode='lines+markers',
                    name='Cross-Validation Score'
                ))
                fig.update_layout(
                    title="K-Selection for KNN Model",
                    xaxis_title="Number of Neighbors (K)",
                    yaxis_title="Cross-Validation Accuracy"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("### 📊 Model Performance Comparison")
        st.markdown("Compare performance metrics across all trained models.")
        
        # Create comparison metrics
        comparison_metrics = {}
        
        for model_name, metrics in models.performance_metrics.items():
            comparison_metrics[model_name] = {}
            if 'test_accuracy' in metrics:
                comparison_metrics[model_name]['accuracy'] = metrics['test_accuracy']
            if 'test_r2' in metrics:
                comparison_metrics[model_name]['r2_score'] = metrics['test_r2']
        
        if comparison_metrics:
            fig = ModelEvaluationUtils.plot_model_comparison(comparison_metrics, "Model Performance Comparison")
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed metrics table
        st.markdown("#### 📋 Detailed Performance Metrics")
        
        for model_name, metrics in models.performance_metrics.items():
            with st.expander(f"🔍 {model_name.replace('_', ' ').title()} Details"):
                st.json(metrics)

def segmentation_page():
    """Customer segmentation page."""
    st.title("👥 Customer Segmentation")
    st.markdown("---")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please generate or upload a dataset first!")
        return
    
    if st.session_state.models is None:
        st.warning("⚠️ Please train models first in the Prediction page!")
        return
    
    if st.session_state.cluster_labels is None:
        st.info("🔄 Running customer segmentation analysis...")
        with st.spinner("Segmenting customers..."):
            models = st.session_state.models
            df = st.session_state.data
            
            # Train clustering model
            _, _, cluster_labels = models.train_customer_segmentation_model(df)
            st.session_state.cluster_labels = cluster_labels
            
            # Initialize recommendation engine
            if st.session_state.recommendation_engine is None:
                st.session_state.recommendation_engine = RetailRecommendationEngine()
            
            st.success("✅ Customer segmentation completed!")
    
    df = st.session_state.data
    cluster_labels = st.session_state.cluster_labels
    models = st.session_state.models
    
    # Add cluster labels to dataframe
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = cluster_labels
    
    # Segmentation tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Cluster Analysis", "🎯 Segment Profiles", "🤖 AI Recommendations", "📈 Visualizations"])
    
    with tab1:
        st.markdown("### 📊 Cluster Analysis")
        
        # Elbow method and silhouette analysis
        if 'customer_segmentation' in models.performance_metrics:
            metrics = models.performance_metrics['customer_segmentation']
            
            col1, col2 = st.columns(2)
            with col1:
                create_animated_metric_card("Optimal Clusters", metrics['optimal_clusters'])
            with col2:
                create_animated_metric_card("Silhouette Score", f"{metrics['final_silhouette_score']:.4f}")
            
            # Elbow method visualization
            fig = ModelEvaluationUtils.plot_elbow_method(
                metrics['inertias'], 
                metrics['silhouette_scores'],
                "Cluster Analysis: Elbow Method & Silhouette Score"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Cluster distribution
        st.markdown("#### 🎯 Cluster Distribution")
        cluster_counts = df_with_clusters['Cluster'].value_counts().sort_index()
        
        fig = DataVisualizationUtils.create_pie_chart(df_with_clusters, 'Cluster', 'Customer Segment Distribution')
        st.plotly_chart(fig, use_container_width=True)
        
        # Cluster size table
        cluster_stats = pd.DataFrame({
            'Cluster': cluster_counts.index,
            'Customer Count': cluster_counts.values,
            'Percentage': (cluster_counts.values / len(df) * 100).round(2)
        })
        st.dataframe(cluster_stats)
    
    with tab2:
        st.markdown("### 🎯 Segment Profiles")
        
        # Analyze segments
        if st.session_state.recommendation_engine:
            engine = st.session_state.recommendation_engine
            segment_analysis = engine.analyze_customer_segments(df, cluster_labels)
            
            # Display segment profiles
            for segment_id, profile in segment_analysis.items():
                with st.expander(f"📋 {segment_id} - {profile['segment_type'].replace('_', ' ').title()}"):
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Customers", profile['size'])
                        st.metric("Percentage", f"{profile['percentage']:.1f}%")
                    
                    with col2:
                        st.metric("Avg Income", f"${profile['avg_income']:,.0f}")
                        st.metric("Avg Sales", f"${profile['avg_sales']:,.0f}")
                    
                    with col3:
                        st.metric("Purchase Rate", f"{profile['purchase_decision_rate']:.1f}%")
                        st.metric("Avg Tenure", f"{profile['avg_tenure']:.1f} years")
                    
                    # Loyalty distribution
                    st.markdown("**Loyalty Distribution:**")
                    loyalty_df = pd.DataFrame(list(profile['loyalty_distribution'].items()), 
                                            columns=['Loyalty', 'Count'])
                    fig = px.bar(loyalty_df, x='Loyalty', y='Count', title=f"Loyalty Distribution - {segment_id}")
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### 🤖 AI-Powered Recommendations")
        
        if st.session_state.recommendation_engine:
            engine = st.session_state.recommendation_engine
            
            # Generate recommendations
            recommendations = engine.generate_recommendations()
            executive_summary = engine.generate_executive_summary()
            
            # Executive summary
            st.markdown("#### 📊 Executive Summary")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Segments", executive_summary['total_segments'])
                st.metric("Largest Segment", executive_summary['largest_segment'])
            
            with col2:
                st.metric("Highest Value Segment", executive_summary['highest_value_segment'])
                st.metric("Priority Actions", len(executive_summary['priority_actions']))
            
            # Key insights
            if executive_summary['key_insights']:
                st.markdown("**🔍 Key Insights:**")
                for insight in executive_summary['key_insights']:
                    st.write(f"• {insight}")
            
            # Priority actions
            if executive_summary['priority_actions']:
                st.markdown("**🚀 Priority Actions:**")
                for action in executive_summary['priority_actions'][:5]:  # Show top 5
                    st.markdown(f"""
                    **{action['segment']}**: {action['action']}  
                    *Expected Impact: {action['impact']}*
                    """)
            
            # Detailed recommendations by segment
            st.markdown("#### 📋 Detailed Recommendations")
            
            for segment_id, recs in recommendations.items():
                with st.expander(f"🎯 {segment_id} Recommendations"):
                    for i, rec in enumerate(recs, 1):
                        priority_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}[rec['priority']]
                        
                        st.markdown(f"""
                        **{priority_color} Recommendation {i}: {rec['action']}**  
                        **Type:** {rec['type'].title()} | **Priority:** {rec['priority'].title()}  
                        **Expected Impact:** {rec['expected_impact']}  
                        
                        **Implementation:** {rec['implementation']}
                        """)
            
            # Action plan
            if st.button("📋 Generate Action Plan"):
                action_plan = engine.create_action_plan()
                
                st.markdown("#### 📅 Implementation Action Plan")
                
                for initiative in action_plan['initiatives'][:10]:  # Show top 10
                    with st.expander(f"🚀 {initiative['action']}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Segment:** {initiative['segment']}")
                            st.write(f"**Type:** {initiative['type'].title()}")
                            st.write(f"**Priority:** {initiative['priority'].title()}")
                        
                        with col2:
                            st.write(f"**Timeline:** {initiative['timeline']}")
                            st.write(f"**Team:** {initiative['responsible_team']}")
                        
                        st.write(f"**Expected Impact:** {initiative['expected_impact']}")
                        st.write(f"**Implementation:** {initiative['implementation_steps']}")
                        
                        st.write("**Success Metrics:**")
                        for metric in initiative['success_metrics']:
                            st.write(f"• {metric}")
    
    with tab4:
        st.markdown("### 📈 Segment Visualizations")
        
        # Cluster scatter plots
        if 'customer_segmentation' in models.performance_metrics:
            features_used = models.performance_metrics['customer_segmentation']['features_used']
            
            if len(features_used) >= 2:
                st.markdown("#### 🎯 Customer Segment Scatter Plots")
                
                # Create scatter plots for different feature combinations
                feature_pairs = [
                    (features_used[0], features_used[1]) if len(features_used) >= 2 else None,
                    (features_used[0], features_used[2]) if len(features_used) >= 3 else None,
                    (features_used[1], features_used[2]) if len(features_used) >= 3 else None
                ]
                
                for i, (x_col, y_col) in enumerate(filter(None, feature_pairs)):
                    if x_col and y_col:
                        fig = ModelEvaluationUtils.plot_cluster_scatter(
                            df_with_clusters, x_col, y_col, 'Cluster',
                            f"Customer Segments: {x_col} vs {y_col}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        # Feature distributions by cluster
        st.markdown("#### 📊 Feature Distributions by Cluster")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        selected_feature = st.selectbox("Select feature to analyze by cluster", numeric_cols)
        
        if selected_feature:
            fig = make_subplots(
                rows=1, cols=len(df_with_clusters['Cluster'].unique()),
                subplot_titles=[f"Cluster {i}" for i in sorted(df_with_clusters['Cluster'].unique())],
                shared_yaxes=True
            )
            
            for i, cluster in enumerate(sorted(df_with_clusters['Cluster'].unique())):
                cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster][selected_feature]
                
                fig.add_trace(
                    go.Histogram(x=cluster_data, name=f"Cluster {cluster}", opacity=0.7),
                    row=1, col=i+1
                )
            
            fig.update_layout(
                title=f"Distribution of {selected_feature} by Cluster",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)

def visualization_page():
    """Visualization and insights page."""
    st.title("📊 Visualizations & Insights")
    st.markdown("---")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please generate or upload a dataset first!")
        return
    
    df = st.session_state.data
    
    # Visualization tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Sales Analysis", "💰 Marketing Insights", 
        "👥 Customer Behavior", "🎯 Product Analysis", "📊 Business KPIs"
    ])
    
    with tab1:
        st.markdown("### 📈 Sales Analysis")
        
        # Sales vs Marketing Spend
        fig = DataVisualizationUtils.create_scatter_plot(
            df, 'Marketing_Spend', 'Monthly_Sales', 
            color_col='Loyalty_Category',
            title="Sales vs Marketing Spend by Loyalty Category"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Sales by Product Category
        col1, col2 = st.columns(2)
        
        with col1:
            sales_by_category = df.groupby('Product_Category')['Monthly_Sales'].mean().sort_values(ascending=False)
            fig = px.bar(
                x=sales_by_category.index, 
                y=sales_by_category.values,
                title="Average Sales by Product Category"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            sales_by_location = df.groupby('Store_Location')['Monthly_Sales'].mean().sort_values(ascending=False)
            fig = px.bar(
                x=sales_by_location.index, 
                y=sales_by_location.values,
                title="Average Sales by Store Location"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Sales trends
        st.markdown("#### 📊 Sales Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = DataVisualizationUtils.create_distribution_plot(df, 'Monthly_Sales', 'Monthly Sales Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Sales by loyalty category
            fig = px.box(
                df, x='Loyalty_Category', y='Monthly_Sales',
                title="Sales Distribution by Loyalty Category"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### 💰 Marketing Insights")
        
        # Marketing effectiveness
        col1, col2 = st.columns(2)
        
        with col1:
            fig = DataVisualizationUtils.create_scatter_plot(
                df, 'Marketing_Spend', 'Purchase_Frequency',
                title="Marketing Spend vs Purchase Frequency"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Discount effectiveness
            fig = DataVisualizationUtils.create_scatter_plot(
                df, 'Discount_Offered', 'Monthly_Sales',
                color_col='Purchase_Decision',
                title="Discount Impact on Sales"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # ROI Analysis
        st.markdown("#### 📊 Marketing ROI Analysis")
        
        # Calculate simple ROI (Sales/Marketing Spend)
        df['Marketing_ROI'] = df['Monthly_Sales'] / (df['Marketing_Spend'] + 1)  # +1 to avoid division by zero
        
        roi_by_category = df.groupby('Product_Category')['Marketing_ROI'].mean().sort_values(ascending=False)
        
        fig = px.bar(
            x=roi_by_category.index,
            y=roi_by_category.values,
            title="Marketing ROI by Product Category"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Marketing spend distribution
        fig = DataVisualizationUtils.create_distribution_plot(df, 'Marketing_Spend', 'Marketing Spend Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### 👥 Customer Behavior Analysis")
        
        # Income vs Purchase Frequency
        col1, col2 = st.columns(2)
        
        with col1:
            fig = DataVisualizationUtils.create_scatter_plot(
                df, 'Annual_Income', 'Purchase_Frequency',
                color_col='Loyalty_Category',
                title="Income vs Purchase Frequency"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = DataVisualizationUtils.create_scatter_plot(
                df, 'Customer_Tenure', 'Previous_Purchase_Amount',
                color_col='Loyalty_Category',
                title="Customer Tenure vs Previous Purchase Amount"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Customer demographics
        st.markdown("#### 📊 Customer Demographics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Age distribution
            fig = DataVisualizationUtils.create_distribution_plot(df, 'Customer_Age', 'Customer Age Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Gender distribution
            fig = DataVisualizationUtils.create_pie_chart(df, 'Gender', 'Gender Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        # Loyalty analysis
        st.markdown("#### ⭐ Loyalty Analysis")
        
        loyalty_stats = df.groupby('Loyalty_Category').agg({
            'Annual_Income': 'mean',
            'Purchase_Frequency': 'mean',
            'Monthly_Sales': 'mean',
            'Customer_Tenure': 'mean'
        }).round(2)
        
        st.dataframe(loyalty_stats)
        
        # Loyalty trends
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Income by Loyalty', 'Purchase Frequency by Loyalty', 
                          'Sales by Loyalty', 'Tenure by Loyalty']
        )
        
        for i, col in enumerate(['Annual_Income', 'Purchase_Frequency', 'Monthly_Sales', 'Customer_Tenure']):
            row = (i // 2) + 1
            col_idx = (i % 2) + 1
            
            for loyalty in df['Loyalty_Category'].unique():
                data = df[df['Loyalty_Category'] == loyalty][col]
                fig.add_trace(
                    go.Histogram(x=data, name=loyalty, opacity=0.7),
                    row=row, col=col_idx
                )
        
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("### 🎯 Product Analysis")
        
        # Product performance
        product_stats = df.groupby('Product_Category').agg({
            'Monthly_Sales': ['mean', 'sum', 'count'],
            'Purchase_Frequency': 'mean',
            'Discount_Offered': 'mean'
        }).round(2)
        
        product_stats.columns = ['Avg Sales', 'Total Sales', 'Transaction Count', 'Avg Frequency', 'Avg Discount']
        st.dataframe(product_stats)
        
        # Product preferences by customer segment
        st.markdown("#### 🎯 Product Preferences by Customer Segment")
        
        # Create crosstab of product category vs loyalty
        product_loyalty = pd.crosstab(df['Product_Category'], df['Loyalty_Category'], normalize='index') * 100
        
        fig = px.imshow(
            product_loyalty.values,
            x=product_loyalty.columns,
            y=product_loyalty.index,
            title="Product Category Preference by Loyalty Segment (%)",
            labels=dict(x="Loyalty Category", y="Product Category", color="Percentage")
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Discount analysis by product
        col1, col2 = st.columns(2)
        
        with col1:
            discount_by_product = df.groupby('Product_Category')['Discount_Offered'].mean().sort_values(ascending=False)
            fig = px.bar(
                x=discount_by_product.index,
                y=discount_by_product.values,
                title="Average Discount by Product Category"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Purchase decision by product
            purchase_rate = df.groupby('Product_Category')['Purchase_Decision'].apply(
                lambda x: (x == 'Yes').mean() * 100
            ).sort_values(ascending=False)
            
            fig = px.bar(
                x=purchase_rate.index,
                y=purchase_rate.values,
                title="Purchase Rate by Product Category (%)"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.markdown("### 📊 Business KPIs & Metrics")
        
        # Key business metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_sales = df['Monthly_Sales'].sum()
            create_animated_metric_card("Total Sales", f"${total_sales:,.0f}")
        
        with col2:
            avg_order_value = df['Monthly_Sales'].mean()
            create_animated_metric_card("Avg Order Value", f"${avg_order_value:.2f}")
        
        with col3:
            purchase_rate = (df['Purchase_Decision'] == 'Yes').mean() * 100
            create_animated_metric_card("Purchase Rate", f"{purchase_rate:.1f}%")
        
        with col4:
            total_customers = len(df)
            create_animated_metric_card("Total Customers", f"{total_customers:,}")
        
        # Revenue breakdown
        st.markdown("#### 💰 Revenue Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            revenue_by_category = df.groupby('Product_Category')['Monthly_Sales'].sum()
            fig = px.pie(
                values=revenue_by_category.values,
                names=revenue_by_category.index,
                title="Revenue by Product Category"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            revenue_by_location = df.groupby('Store_Location')['Monthly_Sales'].sum()
            fig = px.pie(
                values=revenue_by_location.values,
                names=revenue_by_location.index,
                title="Revenue by Store Location"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Customer lifetime value estimation
        st.markdown("#### 💎 Customer Lifetime Value Analysis")
        
        # Simple CLV calculation
        df['Estimated_CLV'] = df['Monthly_Sales'] * 12 * df['Customer_Tenure']
        
        clv_by_loyalty = df.groupby('Loyalty_Category')['Estimated_CLV'].mean()
        
        fig = px.bar(
            x=clv_by_loyalty.index,
            y=clv_by_loyalty.values,
            title="Estimated Customer Lifetime Value by Loyalty Category"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Business insights summary
        st.markdown("#### 🎯 Key Business Insights")
        
        insights = []
        
        # Top performing category
        top_category = df.groupby('Product_Category')['Monthly_Sales'].sum().idxmax()
        insights.append(f"🏆 **Top Product Category:** {top_category}")
        
        # Best performing location
        top_location = df.groupby('Store_Location')['Monthly_Sales'].mean().idxmax()
        insights.append(f"📍 **Best Location:** {top_location}")
        
        # Most valuable customer segment
        valuable_segment = df.groupby('Loyalty_Category')['Estimated_CLV'].mean().idxmax()
        insights.append(f"💎 **Most Valuable Segment:** {valuable_segment}")
        
        # Optimal discount range
        optimal_discount = df.loc[df['Purchase_Decision'] == 'Yes', 'Discount_Offered'].mean()
        insights.append(f"💰 **Optimal Discount:** {optimal_discount:.1f}%")
        
        for insight in insights:
            st.markdown(insight)

def main():
    """Main application function with professional navigation."""
    # Professional Sidebar Navigation
    with st.sidebar:
        st.markdown("""
        <div style="background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%); 
                    padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem; 
                    text-align: center; color: white;">
            <h3 style="margin: 0; font-size: 1.25rem; font-weight: 600;">🧭 Navigation</h3>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 0.9rem;">Choose Your Analytics Journey</p>
        </div>
        """, unsafe_allow_html=True)
        
        page = st.selectbox(
            "",
            ["⭐ Project Overview", "📊 Data Generator", "🤖 Model Training", 
             "� Prediction Interface", "📈 Visualizations & Insights", "📊 Model Performance"],
            index=0,
            key="navigation"
        )
        
        # System Status Section
        st.markdown("### 📊 System Status")
        
        # Dataset Status
        if st.session_state.data is not None:
            st.success("✅ Dataset Loaded")
            st.info(f"📋 Records: `{len(st.session_state.data):,}`")
        else:
            st.warning("⚠️ No Dataset")
            st.info("📋 Records: 0")
        
        # Models Status
        if st.session_state.models is not None:
            st.success("✅ Models Trained")
        else:
            st.warning("⚠️ Models Not Trained")
        
        # Segmentation Status
        if st.session_state.cluster_labels is not None:
            st.success("✅ Segmentation Complete")
        else:
            st.warning("⚠️ No Segmentation")
        
        # Platform Info
        st.markdown("---")
        st.markdown("""
        <div style="background: var(--surface); border: 1px solid var(--border); 
                    border-radius: 8px; padding: 1rem; text-align: center;">
            <h4 style="margin: 0 0 0.5rem 0; color: var(--primary);">ℹ️ Platform Info</h4>
            <p style="margin: 0; font-size: 0.8rem; color: var(--text-secondary);">Version 1.0.0</p>
            <p style="margin: 0.25rem 0 0 0; font-size: 0.8rem; color: var(--text-secondary);">Enterprise Ready</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Page Routing
    if page == "⭐ Project Overview":
        home_page()
    elif page == "📊 Data Generator":
        dataset_page()
    elif page == "🤖 Model Training":
        model_training_page()
    elif page == "🔮 Prediction Interface":
        prediction_page()
    elif page == "� Visualizations & Insights":
        visualization_page()
    elif page == "📊 Model Performance":
        model_performance_page()

if __name__ == "__main__":
    main()
