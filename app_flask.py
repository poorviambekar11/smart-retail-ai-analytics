from flask import Flask, render_template, request, jsonify, session, send_file
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import json
import io
import base64
from datetime import datetime
import os

# Import custom modules
from data_generator import RetailDataGenerator
from models import RetailMLModels
from recommendation_engine import RetailRecommendationEngine
from utils import (
    DataVisualizationUtils, ModelEvaluationUtils, DataProcessingUtils,
    ReportGenerator, ValidationUtils, format_number
)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Global variables for session state
current_data = None
current_models = None
current_recommendation_engine = None
current_cluster_labels = None
training_feature_columns = None  # Add this to store training columns
saved_encoders = {}  # Add this to store encoders
saved_scalers = {}  # Add this to store scalers

@app.route('/')
def index():
    """Home page with project overview."""
    return render_template('index.html')

@app.route('/data_generator')
def data_generator():
    """Data generator page."""
    global current_data
    return render_template('data_generator.html', data=current_data)

@app.route('/generate_data', methods=['POST'])
def generate_data():
    """Generate synthetic data endpoint."""
    global current_data
    try:
        generator = RetailDataGenerator(num_records=1500)
        df = generator.generate_complete_dataset()
        current_data = df
        
        # Store in session
        session['data_generated'] = True
        session['data_shape'] = df.shape
        session['data_columns'] = list(df.columns)
        
        return jsonify({
            'success': True,
            'message': f'Generated {len(df)} records with {len(df.columns)} features',
            'data_preview': df.head(10).to_html(classes='table table-striped'),
            'data_shape': df.shape,
            'data_summary': {
                'total_records': len(df),
                'total_features': len(df.columns),
                'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB"
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/model_training')
def model_training():
    """Model training page."""
    global current_data, current_models
    return render_template('model_training.html', 
                       data_exists=current_data is not None,
                       models_trained=current_models is not None)

@app.route('/train_models', methods=['POST'])
def train_models():
    """Train all models endpoint."""
    global current_data, current_models, training_feature_columns, saved_encoders, saved_scalers
    try:
        print("Debug: Training models endpoint called")
        
        if current_data is None:
            print("Debug: No data available")
            return jsonify({'success': False, 'error': 'No data available'}), 400
        
        print("Debug: Creating RetailMLModels instance")
        models = RetailMLModels()
        
        print("Debug: Preparing data")
        (X_train_sales, X_test_sales, y_train_sales, y_test_sales,
         X_train_purchase, X_test_purchase, y_train_purchase, y_test_purchase,
         X_train_loyalty, X_test_loyalty, y_train_loyalty, y_test_loyalty) = models.prepare_data(current_data)
        
        # Save training feature columns
        training_feature_columns = X_train_sales.columns.tolist()
        print(f"Debug: Saved training columns: {training_feature_columns}")
        
        # Save encoders and scalers from the trained models
        saved_encoders = models.encoders.copy()
        saved_scalers = models.scalers.copy()
        print(f"Debug: Saved encoders: {list(saved_encoders.keys())}")
        print(f"Debug: Saved scalers: {list(saved_scalers.keys())}")
        
        print("Debug: Starting model training")
        # Train all models
        models.train_sales_prediction_model(current_data)
        print("Debug: Sales model trained")
        
        models.train_purchase_decision_model(current_data)
        print("Debug: Purchase model trained")
        
        models.train_loyalty_prediction_model(current_data)
        print("Debug: Loyalty model trained")
        
        models.train_customer_segmentation_model(current_data)
        print("Debug: Segmentation model trained")
        
        current_models = models
        print("Debug: Models stored in global variable")
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_metrics = {}
        for model_name, metrics in models.performance_metrics.items():
            serializable_metrics[model_name] = {}
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    serializable_metrics[model_name][key] = value.tolist()
                elif isinstance(value, (np.int64, np.int32, np.float64, np.float32)):
                    serializable_metrics[model_name][key] = float(value)
                elif hasattr(value, '__dict__'):  # Skip objects that can't be serialized
                    continue
                else:
                    serializable_metrics[model_name][key] = value
        
        # Store in session
        session['models_trained'] = True
        session['performance_metrics'] = serializable_metrics
        print("Debug: Session updated")
        
        print("Debug: Returning success response")
        
        return jsonify({
            'success': True,
            'message': 'All models trained successfully!',
            'performance_metrics': serializable_metrics
        })
        
    except Exception as e:
        print(f"Debug: Error in train_models: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Training error: {str(e)}'}), 500

@app.route('/prediction_interface')
def prediction_interface():
    """Prediction interface page."""
    global current_data, current_models
    return render_template('prediction_interface.html',
                       data_exists=current_data is not None,
                       models_trained=current_models is not None)

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions endpoint."""
    global current_models, training_feature_columns, saved_encoders, saved_scalers
    try:
        if current_models is None:
            return jsonify({'success': False, 'error': 'Models not trained'}), 400
        
        if training_feature_columns is None:
            return jsonify({'success': False, 'error': 'Training features not available'}), 400
        
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
            
        prediction_type = data.get('type')
        input_data = data.get('input_data')
        
        if not prediction_type or not input_data:
            return jsonify({'success': False, 'error': 'Missing prediction type or input data'}), 400
        
        print(f"Debug: Making {prediction_type} prediction with data: {input_data}")
        print(f"Debug: Training columns: {training_feature_columns}")
        
        # Create prediction dataframe
        input_df = pd.DataFrame([input_data])
        print(f"Debug: Input dataframe columns: {input_df.columns.tolist()}")
        
        # Add missing columns with default values
        for col in training_feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0  # Default value for missing columns
        
        # Ensure correct column order
        input_df = input_df[training_feature_columns]
        print(f"Debug: Final dataframe columns: {input_df.columns.tolist()}")
        
        # Apply the same preprocessing as training
        input_processed = preprocess_input_data(input_df, prediction_type)
        print(f"Debug: Preprocessed data shape: {input_processed.shape}")
        
        if prediction_type == 'sales':
            # Use the model directly with preprocessed data
            prediction = current_models.models['sales_prediction'].predict(input_processed)[0]
            return jsonify({
                'success': True,
                'prediction': float(prediction),
                'formatted': f"₹{prediction:.2f}"
            })
        elif prediction_type == 'purchase':
            # Use the model directly with preprocessed data
            prediction_encoded = current_models.models['purchase_decision'].predict(input_processed)[0]
            # Use saved encoder for inverse transform
            try:
                prediction = saved_encoders['Purchase_Decision'].inverse_transform([prediction_encoded])[0]
            except ValueError as e:
                print(f"Warning: Inverse transform error: {e}")
                prediction = "Unknown"
            return jsonify({
                'success': True,
                'prediction': prediction
            })
        elif prediction_type == 'loyalty':
            # Use the model directly with preprocessed data
            prediction_encoded = current_models.models['loyalty_prediction'].predict(input_processed)[0]
            # Use saved encoder for inverse transform
            try:
                prediction = saved_encoders['Loyalty_Category'].inverse_transform([prediction_encoded])[0]
            except ValueError as e:
                print(f"Warning: Inverse transform error: {e}")
                prediction = "Unknown"
            return jsonify({
                'success': True,
                'prediction': prediction
            })
        else:
            return jsonify({'success': False, 'error': 'Invalid prediction type'}), 400
            
    except ValueError as e:
        print(f"Debug: ValueError in prediction: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        print(f"Debug: General error in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Prediction error: {str(e)}'}), 500

def preprocess_input_data(input_df, prediction_type):
    """Apply the same preprocessing as training using saved encoders and scalers."""
    global saved_encoders, saved_scalers
    
    # Make a copy to avoid modifying original
    processed_df = input_df.copy()
    
    # Convert all columns to string first to handle any numeric issues
    for col in processed_df.columns:
        processed_df[col] = processed_df[col].astype(str)
    
    # Apply categorical encoding using saved encoders
    target_vars = ['Monthly_Sales', 'Purchase_Decision', 'Loyalty_Category']
    categorical_cols = ['Gender', 'Product_Category', 'Store_Location']
    
    for col in categorical_cols:
        if col in processed_df.columns and col in saved_encoders:
            encoder = saved_encoders[col]
            print(f"Debug: Encoder classes for {col}: {encoder.classes_}")
            print(f"Debug: Input values for {col}: {processed_df[col].values}")
            
            # Check for unseen labels
            unique_values = processed_df[col].unique()
            unseen_values = [val for val in unique_values if val not in encoder.classes_]
            
            if unseen_values:
                print(f"Debug: Unseen values in {col}: {unseen_values}")
                # Replace unseen values with most frequent class
                most_frequent = encoder.classes_[0]
                processed_df[col] = processed_df[col].apply(
                    lambda x: most_frequent if x not in encoder.classes_ else x
                )
                print(f"Debug: Replaced unseen values with: {most_frequent}")
            
            # Apply encoding
            try:
                processed_df[col] = encoder.transform(processed_df[col])
                print(f"Debug: Successfully encoded {col}")
            except Exception as e:
                print(f"Debug: Error encoding {col}: {e}")
                # Fallback to most frequent class
                most_frequent = encoder.classes_[0]
                processed_df[col] = encoder.transform([most_frequent] * len(processed_df))
    
    # Apply scaling using saved scalers
    if prediction_type == 'sales' and 'sales_prediction' in saved_scalers:
        scaler = saved_scalers['sales_prediction']
        processed_df = pd.DataFrame(scaler.transform(processed_df), 
                                  columns=processed_df.columns, 
                                  index=processed_df.index)
        print("Debug: Applied sales prediction scaler")
    elif prediction_type == 'purchase' and 'purchase_decision' in saved_scalers:
        scaler = saved_scalers['purchase_decision']
        processed_df = pd.DataFrame(scaler.transform(processed_df), 
                                  columns=processed_df.columns, 
                                  index=processed_df.index)
        print("Debug: Applied purchase decision scaler")
    elif prediction_type == 'loyalty' and 'loyalty_prediction' in saved_scalers:
        scaler = saved_scalers['loyalty_prediction']
        processed_df = pd.DataFrame(scaler.transform(processed_df), 
                                  columns=processed_df.columns, 
                                  index=processed_df.index)
        print("Debug: Applied loyalty prediction scaler")
    
    # Return the processed dataframe directly (not as dict)
    return processed_df

@app.route('/visualizations')
def visualizations():
    """Visualizations page."""
    global current_data
    return render_template('visualizations.html', data=current_data)

@app.route('/get_visualizations')
def get_visualizations():
    """Get visualization data endpoint."""
    global current_data
    try:
        if current_data is None:
            return jsonify({'success': False, 'error': 'No data available'})
        
        # Sales distribution
        sales_hist = px.histogram(current_data, x='Monthly_Sales', nbins=30,
                               title='Monthly Sales Distribution')
        sales_json = json.dumps(sales_hist, cls=PlotlyJSONEncoder)
        
        # Age distribution
        age_hist = px.histogram(current_data, x='Customer_Age', nbins=20,
                             title='Customer Age Distribution')
        age_json = json.dumps(age_hist, cls=PlotlyJSONEncoder)
        
        # Purchase decision pie chart
        purchase_counts = current_data['Purchase_Decision'].value_counts()
        purchase_pie = px.pie(values=purchase_counts.values, names=purchase_counts.index,
                              title='Purchase Decision Distribution')
        purchase_json = json.dumps(purchase_pie, cls=PlotlyJSONEncoder)
        
        # Loyalty categories
        loyalty_counts = current_data['Loyalty_Category'].value_counts()
        loyalty_bar = px.bar(x=loyalty_counts.index, y=loyalty_counts.values,
                            title='Loyalty Category Distribution')
        loyalty_json = json.dumps(loyalty_bar, cls=PlotlyJSONEncoder)
        
        # Correlation matrix
        numeric_cols = current_data.select_dtypes(include=[np.number]).columns
        corr_matrix = current_data[numeric_cols].corr()
        corr_heatmap = px.imshow(corr_matrix, title='Feature Correlation Matrix',
                                color_continuous_scale='RdBu_r')
        corr_json = json.dumps(corr_heatmap, cls=PlotlyJSONEncoder)
        
        # Business insights
        insights = {
            'avg_sales': float(current_data['Monthly_Sales'].mean()),
            'total_customers': len(current_data),
            'purchase_rate': float((current_data['Purchase_Decision'] == 'Yes').mean() * 100),
            'high_value_income': float(current_data[current_data['Monthly_Sales'] > current_data['Monthly_Sales'].quantile(0.8)]['Annual_Income'].mean()),
            'sales_variance': float(current_data['Monthly_Sales'].std())
        }
        
        return jsonify({
            'success': True,
            'visualizations': {
                'sales_distribution': sales_json,
                'age_distribution': age_json,
                'purchase_decision': purchase_json,
                'loyalty_categories': loyalty_json,
                'correlation_matrix': corr_json
            },
            'insights': insights
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/model_performance')
def model_performance():
    """Model performance page."""
    global current_models
    return render_template('model_performance.html', models=current_models)

@app.route('/get_performance')
def get_performance():
    """Get model performance data endpoint."""
    global current_models
    try:
        if current_models is None:
            return jsonify({'success': False, 'error': 'Models not trained'})
        
        # Performance comparison chart
        performance_data = []
        model_names = []
        
        for model_name, metrics in current_models.performance_metrics.items():
            if 'test_accuracy' in metrics:
                performance_data.append(metrics['test_accuracy'])
                model_names.append(model_name.replace('_', ' ').title())
            elif 'test_r2' in metrics:
                performance_data.append(metrics['test_r2'])
                model_names.append(model_name.replace('_', ' ').title())
        
        comparison_chart = go.Figure(data=[
            go.Bar(x=model_names, y=performance_data,
                   marker_color='#8b5cf6',
                   text=[f'{val:.3f}' for val in performance_data],
                   textposition='auto')
        ])
        
        comparison_chart.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Models",
            yaxis_title="Performance Score",
            height=400
        )
        
        comparison_json = json.dumps(comparison_chart, cls=PlotlyJSONEncoder)
        
        # Convert performance metrics to JSON-serializable format
        serializable_metrics = {}
        for model_name, metrics in current_models.performance_metrics.items():
            serializable_metrics[model_name] = {}
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    serializable_metrics[model_name][key] = value.tolist()
                elif isinstance(value, (np.int64, np.int32, np.float64, np.float32)):
                    serializable_metrics[model_name][key] = float(value)
                elif hasattr(value, '__dict__'):  # Skip objects that can't be serialized
                    continue
                else:
                    serializable_metrics[model_name][key] = value
        
        return jsonify({
            'success': True,
            'performance_metrics': serializable_metrics,
            'comparison_chart': comparison_json,
            'summary': {
                'total_models': len(current_models.performance_metrics),
                'avg_performance': sum(performance_data) / len(performance_data) if performance_data else 0
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/download_data')
def download_data():
    """Download data as CSV endpoint."""
    global current_data
    try:
        if current_data is None:
            return jsonify({'success': False, 'error': 'No data available'})
        
        # Create CSV in memory
        output = io.StringIO()
        current_data.to_csv(output, index=False)
        output.seek(0)
        
        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name='retail_dataset.csv'
        )
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
