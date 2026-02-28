import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error,
                            accuracy_score, classification_report, confusion_matrix,
                            silhouette_score)
import pickle
import warnings
warnings.filterwarnings('ignore')

class RetailMLModels:
    """
    Implements machine learning models for retail analytics:
    - Linear Regression for Sales Prediction
    - Decision Tree for Purchase Decision Classification
    - KNN for Loyalty Category Prediction
    - K-Means for Customer Segmentation
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = {}
        self.performance_metrics = {}
        
    def prepare_data(self, df):
        """Prepare data for all models and return train/test splits."""
        print("Preparing data for all models...")
        
        # Prepare sales prediction data
        X_sales, y_sales = self.preprocess_data(df, 'Monthly_Sales')
        X_train_sales, X_test_sales, y_train_sales, y_test_sales = train_test_split(
            X_sales, y_sales, test_size=0.2, random_state=42
        )
        
        # Prepare purchase decision data
        X_purchase, y_purchase = self.preprocess_data(df, 'Purchase_Decision')
        X_train_purchase, X_test_purchase, y_train_purchase, y_test_purchase = train_test_split(
            X_purchase, y_purchase, test_size=0.2, random_state=42, stratify=y_purchase
        )
        
        # Prepare loyalty data
        X_loyalty, y_loyalty = self.preprocess_data(df, 'Loyalty_Category')
        X_train_loyalty, X_test_loyalty, y_train_loyalty, y_test_loyalty = train_test_split(
            X_loyalty, y_loyalty, test_size=0.2, random_state=42, stratify=y_loyalty
        )
        
        return (X_train_sales, X_test_sales, y_train_sales, y_test_sales,
                X_train_purchase, X_test_purchase, y_train_purchase, y_test_purchase,
                X_train_loyalty, X_test_loyalty, y_train_loyalty, y_test_loyalty)
        
    def preprocess_data(self, df, target_column):
        """Preprocess data for ML models."""
        # Make a copy to avoid modifying original dataframe
        data = df.copy()
        
        # Separate features and target
        if target_column in data.columns:
            y = data[target_column]
            # Remove all target variables from features
            target_vars = ['Monthly_Sales', 'Purchase_Decision', 'Loyalty_Category']
            X = data.drop(columns=[target_column] + [col for col in target_vars if col != target_column and col in data.columns])
        else:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        # Remove Customer_ID if present (not a feature)
        if 'Customer_ID' in X.columns:
            X = X.drop(columns=['Customer_ID'])
        
        # Handle categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        numerical_columns = X.select_dtypes(include=[np.number]).columns
        
        # Encode categorical features
        X_encoded = X.copy()
        for col in categorical_columns:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
            X_encoded[col] = self.encoders[col].fit_transform(X[col].astype(str))
        
        # Store feature columns for later use
        self.feature_columns[target_column] = X_encoded.columns.tolist()
        
        return X_encoded, y
    
    def train_sales_prediction_model(self, df):
        """Train Linear Regression model for Monthly Sales prediction."""
        print("Training Sales Prediction Model (Linear Regression)...")
        
        # Preprocess data
        X, y = self.preprocess_data(df, 'Monthly_Sales')
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'train_mse': mean_squared_error(y_train, y_train_pred),
            'test_mse': mean_squared_error(y_test, y_test_pred),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'feature_names': X.columns.tolist(),
            'feature_importance': dict(zip(X.columns, np.abs(model.coef_)))
        }
        
        # Store model and components
        self.models['sales_prediction'] = model
        self.scalers['sales_prediction'] = scaler
        self.performance_metrics['sales_prediction'] = metrics
        
        print(f"Sales Prediction Model trained successfully!")
        print(f"Test R² Score: {metrics['test_r2']:.4f}")
        print(f"Test RMSE: {np.sqrt(metrics['test_mse']):.2f}")
        
        return model, metrics
    
    def train_purchase_decision_model(self, df):
        """Train Decision Tree model for Purchase Decision classification."""
        print("Training Purchase Decision Model (Decision Tree)...")
        
        # Preprocess data
        X, y = self.preprocess_data(df, 'Purchase_Decision')
        
        # Encode target variable
        if 'Purchase_Decision' not in self.encoders:
            self.encoders['Purchase_Decision'] = LabelEncoder()
        y_encoded = self.encoders['Purchase_Decision'].fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = DecisionTreeClassifier(random_state=42, max_depth=10)
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # Calculate metrics with safe classification report
        unique_classes = np.unique(y_test)
        unique_labels = self.encoders['Purchase_Decision'].classes_[np.isin(self.encoders['Purchase_Decision'].classes_, unique_classes)]
        
        # Handle empty sequence case
        if len(unique_labels) == 0:
            classification_rep = "No valid classes in test data for classification report"
        else:
            classification_rep = classification_report(y_test, y_test_pred, 
                                                       target_names=unique_labels,
                                                       labels=unique_classes)
        
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'classification_report': classification_rep,
            'confusion_matrix': confusion_matrix(y_test, y_test_pred),
            'feature_names': X.columns.tolist(),
            'feature_importance': dict(zip(X.columns, model.feature_importances_))
        }
        
        # Store model and components
        self.models['purchase_decision'] = model
        self.scalers['purchase_decision'] = scaler
        self.performance_metrics['purchase_decision'] = metrics
        
        print(f"Purchase Decision Model trained successfully!")
        print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
        
        return model, metrics
    
    def train_loyalty_prediction_model(self, df):
        """Train KNN model for Loyalty Category prediction."""
        print("Training Loyalty Prediction Model (KNN)...")
        
        # Preprocess data
        X, y = self.preprocess_data(df, 'Loyalty_Category')
        
        # Encode target variable
        if 'Loyalty_Category' not in self.encoders:
            self.encoders['Loyalty_Category'] = LabelEncoder()
        y_encoded = self.encoders['Loyalty_Category'].fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Find optimal K
        k_range = range(1, 21)
        k_scores = []
        
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
            k_scores.append(scores.mean())
        
        optimal_k = k_range[np.argmax(k_scores)]
        
        # Train model with optimal K
        model = KNeighborsClassifier(n_neighbors=optimal_k)
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        # Calculate metrics with safe classification report
        unique_classes = np.unique(y_test)
        unique_labels = self.encoders['Loyalty_Category'].classes_[np.isin(self.encoders['Loyalty_Category'].classes_, unique_classes)]
        
        # Handle empty sequence case
        if len(unique_labels) == 0:
            classification_rep = "No valid classes in test data for classification report"
        else:
            classification_rep = classification_report(y_test, y_test_pred,
                                                       target_names=unique_labels,
                                                       labels=unique_classes)
        
        metrics = {
            'optimal_k': optimal_k,
            'k_scores': dict(zip(k_range, k_scores)),
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'classification_report': classification_rep,
            'confusion_matrix': confusion_matrix(y_test, y_test_pred),
            'feature_names': X.columns.tolist()
        }
        
        # Store model and components
        self.models['loyalty_prediction'] = model
        self.scalers['loyalty_prediction'] = scaler
        self.performance_metrics['loyalty_prediction'] = metrics
        
        print(f"Loyalty Prediction Model trained successfully!")
        print(f"Optimal K: {optimal_k}")
        print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
        
        return model, metrics
    
    def train_customer_segmentation_model(self, df, n_clusters=4):
        """Train K-Means model for customer segmentation."""
        print("Training Customer Segmentation Model (K-Means)...")
        
        # Select relevant features for clustering
        clustering_features = ['Annual_Income', 'Purchase_Frequency', 
                              'Previous_Purchase_Amount', 'Customer_Tenure',
                              'Monthly_Sales', 'Marketing_Spend']
        
        # Filter available features
        available_features = [col for col in clustering_features if col in df.columns]
        
        if len(available_features) < 3:
            raise ValueError("Not enough features for clustering. Need at least 3 numeric features.")
        
        X = df[available_features].copy()
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Find optimal number of clusters using elbow method
        inertias = []
        silhouette_scores = []
        k_range = range(2, 11)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))
        
        # Use specified n_clusters or find optimal
        if n_clusters is None:
            # Find elbow point
            diffs = np.diff(inertias)
            diffs2 = np.diff(diffs)
            optimal_k = k_range[np.argmax(diffs2) + 1]
        else:
            optimal_k = n_clusters
        
        # Train final model
        final_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = final_model.fit_predict(X_scaled)
        
        # Calculate metrics
        metrics = {
            'optimal_clusters': optimal_k,
            'inertias': dict(zip(k_range, inertias)),
            'silhouette_scores': dict(zip(k_range, silhouette_scores)),
            'final_silhouette_score': silhouette_score(X_scaled, cluster_labels),
            'cluster_centers': final_model.cluster_centers_,
            'features_used': available_features,
            'cluster_labels': cluster_labels
        }
        
        # Store model and components
        self.models['customer_segmentation'] = final_model
        self.scalers['customer_segmentation'] = scaler
        self.performance_metrics['customer_segmentation'] = metrics
        
        print(f"Customer Segmentation Model trained successfully!")
        print(f"Optimal number of clusters: {optimal_k}")
        print(f"Silhouette Score: {metrics['final_silhouette_score']:.4f}")
        
        return final_model, metrics, cluster_labels
    
    def predict_sales(self, input_data):
        """Predict monthly sales for new customer data."""
        if 'sales_prediction' not in self.models:
            raise ValueError("Sales prediction model not trained. Call train_sales_prediction_model first.")
        
        # Preprocess input data
        input_df = pd.DataFrame([input_data])
        
        # Encode categorical variables (exclude target variables)
        target_vars = ['Monthly_Sales', 'Purchase_Decision', 'Loyalty_Category']
        for col in self.encoders:
            if col in input_df.columns and col not in target_vars:
                input_df[col] = self.encoders[col].transform(input_df[col].astype(str))
        
        # Select features used during training
        features = self.feature_columns['Monthly_Sales']
        input_processed = input_df[features]
        
        # Scale features
        input_scaled = self.scalers['sales_prediction'].transform(input_processed)
        
        # Make prediction
        prediction = self.models['sales_prediction'].predict(input_scaled)[0]
        
        return prediction
    
    def predict_purchase_decision(self, input_data):
        """Predict purchase decision for new customer data."""
        if 'purchase_decision' not in self.models:
            raise ValueError("Purchase decision model not trained. Call train_purchase_decision_model first.")
        
        # Preprocess input data
        input_df = pd.DataFrame([input_data])
        
        # Encode categorical variables (exclude target variables)
        target_vars = ['Monthly_Sales', 'Purchase_Decision', 'Loyalty_Category']
        for col in self.encoders:
            if col in input_df.columns and col not in target_vars:
                try:
                    input_df[col] = self.encoders[col].transform(input_df[col].astype(str))
                except ValueError as e:
                    # Handle unseen labels in categorical encoding
                    print(f"Warning: Unseen label in {col}: {e}")
                    # Use the most frequent label as fallback
                    most_frequent = self.encoders[col].classes_[0]
                    input_df[col] = self.encoders[col].transform([most_frequent] * len(input_df))
        
        # Select features used during training
        features = self.feature_columns['Purchase_Decision']
        input_processed = input_df[features]
        
        # Scale features
        input_scaled = self.scalers['purchase_decision'].transform(input_processed)
        
        # Make prediction
        prediction_encoded = self.models['purchase_decision'].predict(input_scaled)[0]
        
        # Handle inverse transform safely
        try:
            prediction = self.encoders['Purchase_Decision'].inverse_transform([prediction_encoded])[0]
        except ValueError as e:
            # Fallback if prediction class wasn't seen during training
            prediction = "Unknown"
            print(f"Warning: {e}")
        
        return prediction
    
    def predict_loyalty_category(self, input_data):
        """Predict loyalty category for new customer data."""
        if 'loyalty_prediction' not in self.models:
            raise ValueError("Loyalty prediction model not trained. Call train_loyalty_prediction_model first.")
        
        # Preprocess input data
        input_df = pd.DataFrame([input_data])
        
        # Encode categorical variables (exclude target variables)
        target_vars = ['Monthly_Sales', 'Purchase_Decision', 'Loyalty_Category']
        for col in self.encoders:
            if col in input_df.columns and col not in target_vars:
                try:
                    input_df[col] = self.encoders[col].transform(input_df[col].astype(str))
                except ValueError as e:
                    # Handle unseen labels in categorical encoding
                    print(f"Warning: Unseen label in {col}: {e}")
                    # Use the most frequent label as fallback
                    most_frequent = self.encoders[col].classes_[0]
                    input_df[col] = self.encoders[col].transform([most_frequent] * len(input_df))
        
        # Select features used during training
        features = self.feature_columns['Loyalty_Category']
        input_processed = input_df[features]
        
        # Scale features
        input_scaled = self.scalers['loyalty_prediction'].transform(input_processed)
        
        # Make prediction
        prediction_encoded = self.models['loyalty_prediction'].predict(input_scaled)[0]
        
        # Handle inverse transform safely
        try:
            prediction = self.encoders['Loyalty_Category'].inverse_transform([prediction_encoded])[0]
        except ValueError as e:
            # Fallback if prediction class wasn't seen during training
            prediction = "Unknown"
            print(f"Warning: {e}")
        
        return prediction
    
    def get_customer_segments(self, df):
        """Get customer segments for the dataset."""
        if 'customer_segmentation' not in self.models:
            raise ValueError("Customer segmentation model not trained. Call train_customer_segmentation_model first.")
        
        # Select features used for clustering
        features = self.performance_metrics['customer_segmentation']['features_used']
        X = df[features].copy()
        X = X.fillna(X.mean())
        
        # Scale features
        X_scaled = self.scalers['customer_segmentation'].transform(X)
        
        # Predict clusters
        cluster_labels = self.models['customer_segmentation'].predict(X_scaled)
        
        return cluster_labels
    
    def save_models(self, filepath='models/'):
        """Save trained models to disk."""
        import os
        os.makedirs(filepath, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            with open(f"{filepath}{name}.pkl", 'wb') as f:
                pickle.dump(model, f)
        
        # Save scalers and encoders
        with open(f"{filepath}preprocessors.pkl", 'wb') as f:
            pickle.dump({'scalers': self.scalers, 'encoders': self.encoders}, f)
        
        # Save feature columns and metrics
        with open(f"{filepath}metadata.pkl", 'wb') as f:
            pickle.dump({
                'feature_columns': self.feature_columns,
                'performance_metrics': self.performance_metrics
            }, f)
        
        print(f"Models saved to {filepath}")
    
    def load_models(self, filepath='models/'):
        """Load trained models from disk."""
        # Load models
        model_files = ['sales_prediction.pkl', 'purchase_decision.pkl', 
                      'loyalty_prediction.pkl', 'customer_segmentation.pkl']
        
        for model_file in model_files:
            if os.path.exists(f"{filepath}{model_file}"):
                name = model_file.replace('.pkl', '')
                with open(f"{filepath}{model_file}", 'rb') as f:
                    self.models[name] = pickle.load(f)
        
        # Load preprocessors
        if os.path.exists(f"{filepath}preprocessors.pkl"):
            with open(f"{filepath}preprocessors.pkl", 'rb') as f:
                data = pickle.load(f)
                self.scalers = data['scalers']
                self.encoders = data['encoders']
        
        # Load metadata
        if os.path.exists(f"{filepath}metadata.pkl"):
            with open(f"{filepath}metadata.pkl", 'rb') as f:
                data = pickle.load(f)
                self.feature_columns = data['feature_columns']
                self.performance_metrics = data['performance_metrics']
        
        print(f"Models loaded from {filepath}")

# Add missing import
from sklearn.model_selection import cross_val_score

# Example usage
if __name__ == "__main__":
    # Generate sample data
    from data_generator import RetailDataGenerator
    
    generator = RetailDataGenerator(num_records=1500)
    df = generator.generate_complete_dataset()
    
    # Initialize and train models
    ml_models = RetailMLModels()
    
    # Train all models
    ml_models.train_sales_prediction_model(df)
    ml_models.train_purchase_decision_model(df)
    ml_models.train_loyalty_prediction_model(df)
    ml_models.train_customer_segmentation_model(df)
    
    # Save models
    ml_models.save_models()
    
    print("\nAll models trained and saved successfully!")
