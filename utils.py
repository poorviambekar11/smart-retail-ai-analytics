import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime
import json

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DataVisualizationUtils:
    """Utility class for creating data visualizations and charts."""
    
    @staticmethod
    def create_correlation_heatmap(df, title="Feature Correlation Heatmap"):
        """Create correlation heatmap for numeric features."""
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Create heatmap using plotly
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Features",
            yaxis_title="Features",
            width=800,
            height=700
        )
        
        return fig
    
    @staticmethod
    def create_scatter_plot(df, x_col, y_col, color_col=None, title=None):
        """Create interactive scatter plot."""
        if title is None:
            title = f"{y_col} vs {x_col}"
        
        if color_col and color_col in df.columns:
            fig = px.scatter(
                df, x=x_col, y=y_col, color=color_col,
                title=title, hover_data=df.columns.tolist()
            )
        else:
            fig = px.scatter(
                df, x=x_col, y=y_col,
                title=title, hover_data=df.columns.tolist()
            )
        
        fig.update_layout(
            width=700,
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_distribution_plot(df, column, title=None):
        """Create distribution plot for a column."""
        if title is None:
            title = f"Distribution of {column}"
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Histogram', 'Box Plot'],
            vertical_spacing=0.1
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(x=df[column], name='Histogram', nbinsx=30),
            row=1, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(y=df[column], name='Box Plot'),
            row=2, col=1
        )
        
        fig.update_layout(
            title=title,
            height=600,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def create_pie_chart(df, column, title=None):
        """Create pie chart for categorical data."""
        if title is None:
            title = f"Distribution of {column}"
        
        value_counts = df[column].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=value_counts.index,
            values=value_counts.values,
            hole=0.3,
            textinfo='label+percent',
            textposition='outside'
        )])
        
        fig.update_layout(
            title=title,
            width=600,
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_bar_chart(df, x_col, y_col, title=None):
        """Create bar chart."""
        if title is None:
            title = f"{y_col} by {x_col}"
        
        fig = px.bar(
            df, x=x_col, y=y_col,
            title=title
        )
        
        fig.update_layout(
            width=700,
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_line_plot(df, x_col, y_col, title=None):
        """Create line plot."""
        if title is None:
            title = f"{y_col} over {x_col}"
        
        fig = px.line(
            df, x=x_col, y=y_col,
            title=title, markers=True
        )
        
        fig.update_layout(
            width=700,
            height=500
        )
        
        return fig

class ModelEvaluationUtils:
    """Utility class for model evaluation and performance visualization."""
    
    @staticmethod
    def plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
        """Create confusion matrix visualization."""
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 12},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Predicted",
            yaxis_title="Actual",
            width=500,
            height=500
        )
        
        return fig
    
    @staticmethod
    def plot_feature_importance(feature_importance, title="Feature Importance"):
        """Create feature importance plot."""
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        features, importance = zip(*sorted_features)
        
        fig = go.Figure(data=[
            go.Bar(x=list(importance), y=list(features), orientation='h')
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=max(400, len(features) * 30)
        )
        
        return fig
    
    @staticmethod
    def plot_model_comparison(metrics_dict, title="Model Performance Comparison"):
        """Create model performance comparison chart."""
        models = list(metrics_dict.keys())
        
        # Extract different metrics
        metric_types = set()
        for model_metrics in metrics_dict.values():
            metric_types.update(model_metrics.keys())
        
        metric_types = list(metric_types)
        
        fig = go.Figure()
        
        for metric in metric_types:
            values = []
            for model in models:
                if metric in metrics_dict[model]:
                    values.append(metrics_dict[model][metric])
                else:
                    values.append(0)
            
            fig.add_trace(go.Bar(
                name=metric,
                x=models,
                y=values
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Models",
            yaxis_title="Score",
            barmode='group',
            width=800,
            height=500
        )
        
        return fig
    
    @staticmethod
    def plot_elbow_method(inertias, silhouette_scores, title="Elbow Method and Silhouette Analysis"):
        """Create elbow method visualization for K-means clustering."""
        k_values = list(inertias.keys())
        inertia_values = list(inertias.values())
        silhouette_values = list(silhouette_scores.values())
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Elbow Method (Inertia)', 'Silhouette Score'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Elbow method
        fig.add_trace(
            go.Scatter(x=k_values, y=inertia_values, mode='lines+markers', name='Inertia'),
            row=1, col=1
        )
        
        # Silhouette score
        fig.add_trace(
            go.Scatter(x=k_values, y=silhouette_values, mode='lines+markers', name='Silhouette Score'),
            row=1, col=2
        )
        
        fig.update_layout(
            title=title,
            height=400,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def plot_cluster_scatter(df, x_col, y_col, cluster_col, title="Customer Segments"):
        """Create cluster visualization."""
        fig = px.scatter(
            df, x=x_col, y=y_col, color=cluster_col,
            title=title,
            hover_data=df.columns.tolist()
        )
        
        fig.update_layout(
            width=700,
            height=500
        )
        
        return fig

class DataProcessingUtils:
    """Utility class for data processing and manipulation."""
    
    @staticmethod
    def clean_data(df):
        """Clean and preprocess data."""
        # Make a copy
        cleaned_df = df.copy()
        
        # Remove duplicates
        cleaned_df = cleaned_df.drop_duplicates()
        
        # Handle missing values
        numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
        cleaned_df[numeric_columns] = cleaned_df[numeric_columns].fillna(
            cleaned_df[numeric_columns].median()
        )
        
        categorical_columns = cleaned_df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0] if not cleaned_df[col].mode().empty else 'Unknown')
        
        return cleaned_df
    
    @staticmethod
    def encode_categorical(df, columns=None):
        """Encode categorical variables."""
        encoded_df = df.copy()
        
        if columns is None:
            columns = encoded_df.select_dtypes(include=['object']).columns
        
        for col in columns:
            if col in encoded_df.columns:
                encoded_df[col] = pd.Categorical(encoded_df[col]).codes
        
        return encoded_df
    
    @staticmethod
    def create_summary_statistics(df):
        """Create comprehensive summary statistics."""
        summary = {
            'dataset_info': {
                'total_records': len(df),
                'total_columns': len(df.columns),
                'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
                'categorical_columns': len(df.select_dtypes(include=['object']).columns),
                'missing_values': df.isnull().sum().sum()
            },
            'numeric_summary': df.describe().to_dict(),
            'categorical_summary': {}
        }
        
        # Categorical summaries
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            summary['categorical_summary'][col] = {
                'unique_values': df[col].nunique(),
                'most_frequent': df[col].mode()[0] if not df[col].mode().empty else None,
                'frequency': df[col].value_counts().to_dict()
            }
        
        return summary
    
    @staticmethod
    def detect_outliers(df, column, method='iqr'):
        """Detect outliers in a column."""
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        
        elif method == 'zscore':
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            outliers = df[z_scores > 3]
        
        else:
            raise ValueError("Method must be 'iqr' or 'zscore'")
        
        return outliers

class ReportGenerator:
    """Utility class for generating reports."""
    
    @staticmethod
    def generate_data_profile_report(df):
        """Generate comprehensive data profile report."""
        report = {
            'generated_at': datetime.now().isoformat(),
            'dataset_overview': DataProcessingUtils.create_summary_statistics(df),
            'data_quality': {
                'duplicate_rows': df.duplicated().sum(),
                'missing_values_by_column': df.isnull().sum().to_dict(),
                'data_types': df.dtypes.to_dict()
            },
            'statistical_summary': {
                'correlation_matrix': df.select_dtypes(include=[np.number]).corr().to_dict(),
                'distributions': {}
            }
        }
        
        # Add distribution information for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            report['statistical_summary']['distributions'][col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'skewness': df[col].skew(),
                'kurtosis': df[col].kurtosis()
            }
        
        return report
    
    @staticmethod
    def generate_model_performance_report(models_metrics):
        """Generate model performance report."""
        report = {
            'generated_at': datetime.now().isoformat(),
            'model_performance': models_metrics,
            'best_models': {},
            'recommendations': []
        }
        
        # Find best models for each task
        for model_name, metrics in models_metrics.items():
            if 'accuracy' in str(metrics).lower():
                # Classification model
                if 'test_accuracy' in metrics:
                    report['best_models'][model_name] = {
                        'metric': 'accuracy',
                        'value': metrics['test_accuracy']
                    }
            elif 'r2' in str(metrics).lower():
                # Regression model
                if 'test_r2' in metrics:
                    report['best_models'][model_name] = {
                        'metric': 'r2_score',
                        'value': metrics['test_r2']
                    }
        
        return report
    
    @staticmethod
    def export_to_csv(df, filename):
        """Export dataframe to CSV."""
        df.to_csv(filename, index=False)
        return f"Data exported to {filename}"
    
    @staticmethod
    def export_to_json(data, filename):
        """Export data to JSON."""
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return f"Data exported to {filename}"

class ValidationUtils:
    """Utility class for data validation."""
    
    @staticmethod
    def validate_input_data(df, required_columns=None):
        """Validate input data format and required columns."""
        errors = []
        warnings = []
        
        # Check if dataframe is empty
        if df.empty:
            errors.append("DataFrame is empty")
            return errors, warnings
        
        # Check for required columns
        if required_columns:
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                errors.append(f"Missing required columns: {missing_columns}")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        high_missing = missing_values[missing_values > len(df) * 0.5]
        if not high_missing.empty:
            warnings.append(f"Columns with >50% missing values: {high_missing.index.tolist()}")
        
        # Check data types
        expected_types = {
            'Customer_Age': 'int64',
            'Annual_Income': 'int64',
            'Monthly_Sales': 'float64'
        }
        
        for col, expected_type in expected_types.items():
            if col in df.columns:
                if str(df[col].dtype) != expected_type:
                    warnings.append(f"Column {col} has unexpected type: {df[col].dtype}")
        
        return errors, warnings
    
    @staticmethod
    def validate_model_input(input_dict, required_fields):
        """Validate model input parameters."""
        errors = []
        
        for field in required_fields:
            if field not in input_dict:
                errors.append(f"Missing required field: {field}")
            elif input_dict[field] is None:
                errors.append(f"Field {field} cannot be None")
        
        # Validate numeric ranges
        if 'Customer_Age' in input_dict:
            age = input_dict['Customer_Age']
            if not isinstance(age, (int, float)) or age < 18 or age > 100:
                errors.append("Customer_Age must be between 18 and 100")
        
        if 'Annual_Income' in input_dict:
            income = input_dict['Annual_Income']
            if not isinstance(income, (int, float)) or income < 0:
                errors.append("Annual_Income must be a positive number")
        
        return errors

# Utility functions for Streamlit
def set_custom_css():
    """Set custom CSS for stunning, aesthetic, hackathon-winning design."""
    css = """
    <style>
    /* Global Styles & Theme */
    .main {
        padding: 2rem;
        background: linear-gradient(135deg, #1e3a8a 0%, #7c3aed 50%, #ec4899 100%);
        color: #ffffff;
        min-height: 100vh;
    }
    
    /* Professional Color Palette */
    :root {
        --primary: #8b5cf6;
        --primary-dark: #7c3aed;
        --secondary: #ec4899;
        --accent: #f59e0b;
        --success: #10b981;
        --warning: #f59e0b;
        --error: #ef4444;
        --background: linear-gradient(135deg, #1e3a8a 0%, #7c3aed 50%, #ec4899 100%);
        --surface: rgba(255, 255, 255, 0.15);
        --surface-hover: rgba(255, 255, 255, 0.25);
        --text-primary: #ffffff;
        --text-secondary: rgba(255, 255, 255, 0.9);
        --text-muted: rgba(255, 255, 255, 0.7);
        --border: rgba(255, 255, 255, 0.3);
        --shadow: rgba(0, 0, 0, 0.4);
        --glow: rgba(139, 92, 246, 0.6);
    }
    
    /* Animated Background */
    .main::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: 
            radial-gradient(circle at 20% 80%, rgba(168, 85, 247, 0.3) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(236, 72, 153, 0.3) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(99, 102, 241, 0.2) 0%, transparent 50%);
        z-index: -1;
        animation: float 20s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        33% { transform: translateY(-20px) rotate(1deg); }
        66% { transform: translateY(10px) rotate(-1deg); }
    }
    
    /* Sidebar Styles */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border-right: 1px solid var(--border);
        box-shadow: 4px 0 30px var(--shadow);
    }
    
    /* Header & Title Styles */
    .css-1avcm2n {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 50%, var(--accent) 100%);
        color: white;
        border-radius: 20px;
        padding: 3rem;
        margin-bottom: 3rem;
        box-shadow: 0 10px 40px var(--shadow), 0 0 60px var(--glow);
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .css-1avcm2n::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%) translateY(-100%); }
        100% { transform: translateX(100%) translateY(100%); }
    }
    
    /* Glassmorphism Card Styles */
    .professional-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px var(--shadow), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        height: 100%;
        position: relative;
        overflow: hidden;
    }
    
    .professional-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, transparent 100%);
        pointer-events: none;
    }
    
    .professional-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 20px 60px var(--shadow), 0 0 80px var(--glow);
        border-color: var(--primary);
        background: rgba(255, 255, 255, 0.15);
    }
    
    /* Feature Card Styles */
    .feature-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border);
        border-radius: 25px;
        padding: 2.5rem;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 8px 32px var(--shadow), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.1) 0%, transparent 70%);
        opacity: 0;
        transition: opacity 0.4s ease;
        pointer-events: none;
    }
    
    .feature-card:hover::before {
        opacity: 1;
    }
    
    .feature-card:hover {
        transform: translateY(-15px) scale(1.05) rotate(1deg);
        box-shadow: 0 25px 80px var(--shadow), 0 0 100px var(--glow);
        border-color: var(--secondary);
    }
    
    .feature-icon {
        font-size: 4rem;
        margin-bottom: 1.5rem;
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        filter: drop-shadow(0 0 20px var(--glow));
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.1); }
    }
    
    .feature-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 1rem;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
    }
    
    .feature-description {
        color: var(--text-secondary);
        line-height: 1.8;
        font-size: 1.1rem;
        font-weight: 500;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 50%, var(--accent) 100%);
        color: white;
        border-radius: 25px;
        padding: 2rem;
        box-shadow: 0 10px 40px var(--shadow), 0 0 60px var(--glow);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        animation: shimmer 3s infinite;
    }
    
    .metric-card:hover {
        transform: translateY(-10px) scale(1.05);
        box-shadow: 0 20px 60px var(--shadow), 0 0 80px var(--glow);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { box-shadow: 0 10px 40px var(--shadow), 0 0 60px var(--glow); }
        to { box-shadow: 0 20px 60px var(--shadow), 0 0 100px var(--glow); }
    }
    
    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        margin: 1rem 0;
        text-shadow: 0 4px 20px var(--shadow);
        animation: countUp 2s ease-out;
    }
    
    @keyframes countUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 600;
    }
    
    /* Container Styles */
    .section-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px var(--shadow), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .section-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, transparent 100%);
        pointer-events: none;
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 50%, var(--accent) 100%);
        color: white;
        border: none;
        border-radius: 15px;
        padding: 1rem 2.5rem;
        font-weight: 700;
        font-size: 1.1rem;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 8px 32px var(--shadow), 0 0 40px var(--glow);
        text-transform: uppercase;
        letter-spacing: 1px;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        transition: left 0.5s ease;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-5px) scale(1.05);
        box-shadow: 0 15px 50px var(--shadow), 0 0 60px var(--glow);
        animation: buttonGlow 2s ease-in-out infinite alternate;
    }
    
    @keyframes buttonGlow {
        from { box-shadow: 0 8px 32px var(--shadow), 0 0 40px var(--glow); }
        to { box-shadow: 0 15px 50px var(--shadow), 0 0 80px var(--glow); }
    }
    
    /* Input Styles */
    .stSelectbox > div > div > select,
    .stNumberInput > div > div > input,
    .stSlider > div > div > div {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        color: white;
        border: 2px solid var(--border);
        border-radius: 12px;
        transition: all 0.3s ease;
        font-size: 1rem;
    }
    
    .stSelectbox > div > div > select:focus,
    .stNumberInput > div > div > input:focus,
    .stSlider > div > div > div:focus {
        border-color: var(--primary);
        box-shadow: 0 0 20px var(--glow);
        background: rgba(255, 255, 255, 0.15);
    }
    
    /* Tab Styles */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border);
        border-radius: 15px;
        padding: 0.75rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.1);
        color: var(--text-secondary);
        border: 1px solid var(--border);
        border-radius: 10px;
        margin: 0 0.5rem;
        transition: all 0.3s ease;
        font-weight: 600;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: white;
        border-color: var(--primary);
        box-shadow: 0 4px 20px var(--glow);
    }
    
    /* Dataframe Styles */
    .dataframe {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border);
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 32px var(--shadow);
    }
    
    .dataframe th {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: white;
        font-weight: 700;
        text-shadow: 0 2px 10px var(--shadow);
    }
    
    /* Success/Info/Warning/Error Messages */
    .element-container .stSuccess {
        background: linear-gradient(135deg, var(--success) 0%, #059669 100%);
        border: 1px solid var(--success);
        border-radius: 15px;
        color: white;
        font-weight: 600;
        box-shadow: 0 8px 32px var(--shadow), 0 0 40px rgba(16, 185, 129, 0.3);
    }
    
    .element-container .stInfo {
        background: linear-gradient(135deg, var(--secondary) 0%, var(--primary) 100%);
        border: 1px solid var(--secondary);
        border-radius: 15px;
        color: white;
        font-weight: 600;
        box-shadow: 0 8px 32px var(--shadow), 0 0 40px rgba(168, 85, 247, 0.3);
    }
    
    .element-container .stWarning {
        background: linear-gradient(135deg, var(--warning) 0%, #D97706 100%);
        border: 1px solid var(--warning);
        border-radius: 15px;
        color: white;
        font-weight: 600;
        box-shadow: 0 8px 32px var(--shadow), 0 0 40px rgba(245, 158, 11, 0.3);
    }
    
    .element-container .stError {
        background: linear-gradient(135deg, var(--error) 0%, #DC2626 100%);
        border: 1px solid var(--error);
        border-radius: 15px;
        color: white;
        font-weight: 600;
        box-shadow: 0 8px 32px var(--shadow), 0 0 40px rgba(239, 68, 68, 0.3);
    }
    
    /* Loading Animation */
    .loading-spinner {
        border: 4px solid var(--border);
        border-top: 4px solid var(--primary);
        border-radius: 50%;
        width: 50px;
        height: 50px;
        animation: spin 1s linear infinite;
        margin: 2rem auto;
        box-shadow: 0 0 20px var(--glow);
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Footer Styles */
    .footer {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border-top: 1px solid var(--border);
        padding: 2rem;
        text-align: center;
        color: var(--text-secondary);
        margin-top: 3rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px var(--shadow);
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main {
            padding: 1rem;
        }
        
        .feature-card {
            padding: 1.5rem;
        }
        
        .metric-card {
            padding: 1.5rem;
        }
        
        .section-container {
            padding: 1.5rem;
        }
        
        .feature-icon {
            font-size: 3rem;
        }
        
        .metric-value {
            font-size: 2rem;
        }
    }
    
    /* Additional Animations for Moving Elements */
    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-30px); }
        60% { transform: translateY(-15px); }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-50px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(50px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes zoomIn {
        from { opacity: 0; transform: scale(0.5); }
        to { opacity: 1; transform: scale(1); }
    }
    
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        10%, 30%, 50%, 70%, 90% { transform: translateX(-10px); }
        20%, 40%, 60%, 80% { transform: translateX(10px); }
    }
    
    @keyframes swing {
        0%, 100% { transform: rotate(0deg); }
        20% { transform: rotate(15deg); }
        40% { transform: rotate(-10deg); }
        60% { transform: rotate(5deg); }
        80% { transform: rotate(-5deg); }
    }
    
    @keyframes tada {
        0% { transform: scale(1); }
        10%, 20% { transform: scale(0.9) rotate(-3deg); }
        30%, 50%, 70%, 90% { transform: scale(1.1) rotate(3deg); }
        40%, 60%, 80% { transform: scale(1.1) rotate(-3deg); }
        100% { transform: scale(1) rotate(0); }
    }
    
    @keyframes wobble {
        0% { transform: translateX(0%); }
        15% { transform: translateX(-25%) rotate(-5deg); }
        30% { transform: translateX(20%) rotate(3deg); }
        45% { transform: translateX(-15%) rotate(-3deg); }
        60% { transform: translateX(10%) rotate(2deg); }
        75% { transform: translateX(-5%) rotate(-1deg); }
        100% { transform: translateX(0%); }
    }
    
    @keyframes flip {
        from { transform: perspective(400px) rotate3d(0, 1, 0, -360deg); }
        to { transform: perspective(400px) rotate3d(0, 1, 0, 0deg); }
    }
    
    @keyframes flipInX {
        from { transform: perspective(400px) rotate3d(1, 0, 0, 90deg); }
        to { transform: perspective(400px) rotate3d(1, 0, 0, 0deg); }
    }
    
    @keyframes flipInY {
        from { transform: perspective(400px) rotate3d(0, 1, 0, 90deg); }
        to { transform: perspective(400px) rotate3d(0, 1, 0, 0deg); }
    }
    
    @keyframes lightSpeedIn {
        from { transform: translate3d(100%, 0, 0) skewX(-30deg); }
        60% { transform: translate3d(-20%, 0, 0) skewX(20deg); }
        80% { transform: translate3d(0%, 0, 0) skewX(-5deg); }
        to { transform: none; }
    }
    
    @keyframes heartBeat {
        0% { transform: scale(1); }
        14% { transform: scale(1.3); }
        28% { transform: scale(1); }
        42% { transform: scale(1.3); }
        70% { transform: scale(1); }
    }
    
    @keyframes rubberBand {
        0% { transform: scale(1); }
        30% { transform: scale(1.25, 0.75); }
        40% { transform: scale(0.75, 1.25); }
        60% { transform: scale(1.15, 0.85); }
        100% { transform: scale(1); }
    }
    
    @keyframes jello {
        0%, 11.1%, 100% { transform: none; }
        22.2% { transform: skewX(-12.5deg) skewY(-12.5deg); }
        33.3% { transform: skewX(6.25deg) skewY(6.25deg); }
        44.4% { transform: skewX(-3.125deg) skewY(-3.125deg); }
        55.5% { transform: skewX(1.5625deg) skewY(1.5625deg); }
        66.6% { transform: skewX(-0.78125deg) skewY(-0.78125deg); }
        77.7% { transform: skewX(0.390625deg) skewY(0.390625deg); }
        88.8% { transform: skewX(-0.1953125deg) skewY(-0.1953125deg); }
    }
    
    @keyframes hinge {
        0% { transform-origin: top left; }
        20%, 60% { transform: rotate3d(0, 0, 1, 80deg); }
        40%, 80% { transform: rotate3d(0, 0, 1, 60deg); }
        100% { transform: translate3d(0, 700px, 0); opacity: 0; }
    }
    
    @keyframes jackInTheBox {
        from { transform: scale(0.1) rotate(30deg); }
        50% { transform: rotate(-10deg); }
        70% { transform: rotate(3deg); }
        to { transform: scale(1) rotate(0); }
    }
    
    @keyframes rollIn {
        from { opacity: 0; transform: translate3d(-100%, 0, 0) rotate3d(0, 0, 1, -120deg); }
        to { opacity: 1; transform: none; }
    }
    
    @keyframes rotateIn {
        from { transform-origin: center; transform: rotate3d(0, 0, 1, -200deg); }
        to { transform-origin: center; transform: none; }
    }
    
    @keyframes zoomInDown {
        from { opacity: 0; transform: scale3d(0.1, 0.1, 0.1) translate3d(0, -1000px, 0); }
        60% { opacity: 1; transform: scale3d(0.475, 0.475, 0.475) translate3d(0, 60px, 0); }
        to { transform: scale3d(1, 1, 1) translate3d(0, 0, 0); }
    }
    
    @keyframes zoomInLeft {
        from { opacity: 0; transform: scale3d(0.1, 0.1, 0.1) translate3d(-1000px, 0, 0); }
        60% { opacity: 1; transform: scale3d(0.475, 0.475, 0.475) translate3d(10px, 0, 0); }
        to { transform: scale3d(1, 1, 1) translate3d(0, 0, 0); }
    }
    
    @keyframes zoomInRight {
        from { opacity: 0; transform: scale3d(0.1, 0.1, 0.1) translate3d(1000px, 0, 0); }
        60% { opacity: 1; transform: scale3d(0.475, 0.475, 0.475) translate3d(-10px, 0, 0); }
        to { transform: scale3d(1, 1, 1) translate3d(0, 0, 0); }
    }
    
    @keyframes zoomInUp {
        from { opacity: 0; transform: scale3d(0.1, 0.1, 0.1) translate3d(0, 1000px, 0); }
        60% { opacity: 1; transform: scale3d(0.475, 0.475, 0.475) translate3d(0, -60px, 0); }
        to { transform: scale3d(1, 1, 1) translate3d(0, 0, 0); }
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        border-radius: 5px;
        box-shadow: 0 0 10px var(--glow);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, var(--secondary) 0%, var(--accent) 100%);
    }
    </style>
    """
    return css

def format_number(num):
    """Format numbers for display."""
    if isinstance(num, (int, float)):
        if num >= 1000000:
            return f"{num/1000000:.1f}M"
        elif num >= 1000:
            return f"{num/1000:.1f}K"
        else:
            return f"{num:.2f}"
    return str(num)

def create_download_link(data, filename, link_text):
    """Create download link for data."""
    if isinstance(data, pd.DataFrame):
        csv = data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
        return href
    return ""

# Example usage
if __name__ == "__main__":
    # Test utilities
    from data_generator import RetailDataGenerator
    
    generator = RetailDataGenerator(num_records=100)
    df = generator.generate_complete_dataset()
    
    # Test data processing
    cleaned_df = DataProcessingUtils.clean_data(df)
    summary = DataProcessingUtils.create_summary_statistics(cleaned_df)
    
    print("Data Summary:")
    print(f"Total records: {summary['dataset_info']['total_records']}")
    print(f"Total columns: {summary['dataset_info']['total_columns']}")
    
    # Test visualization
    viz = DataVisualizationUtils()
    fig = viz.create_correlation_heatmap(cleaned_df)
    print("Correlation heatmap created successfully!")
    
    # Test validation
    errors, warnings = ValidationUtils.validate_input_data(cleaned_df)
    print(f"Validation errors: {errors}")
    print(f"Validation warnings: {warnings}")
