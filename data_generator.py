import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class RetailDataGenerator:
    """
    Generates synthetic retail dataset for machine learning models.
    Creates realistic customer data with logical relationships between features.
    """
    
    def __init__(self, num_records=1500):
        self.num_records = num_records
        self.np_random = np.random.RandomState(42)
        
    def generate_customer_demographics(self):
        """Generate customer demographic data with realistic distributions."""
        # Age distribution (18-70, with concentration around 25-45)
        ages = np.clip(
            np.random.normal(35, 12, self.num_records).astype(int),
            18, 70
        )
        
        # Gender distribution
        genders = np.random.choice(['Male', 'Female'], self.num_records, p=[0.48, 0.52])
        
        # Annual income distribution (correlated with age)
        base_income = 25000 + (ages - 18) * 800
        income_noise = np.random.normal(0, 15000, self.num_records)
        annual_income = np.clip(base_income + income_noise, 20000, 150000).astype(int)
        
        # Customer tenure (years with the company)
        tenure = np.clip(
            np.random.exponential(3, self.num_records).astype(int),
            1, np.minimum(ages - 18, 20)
        )
        
        return pd.DataFrame({
            'Customer_Age': ages,
            'Gender': genders,
            'Annual_Income': annual_income,
            'Customer_Tenure': tenure
        })
    
    def generate_purchase_behavior(self, demographics_df):
        """Generate purchase behavior data based on demographics."""
        # Purchase frequency (higher income and tenure = more frequent)
        base_frequency = (demographics_df['Annual_Income'] / 50000 + 
                         demographics_df['Customer_Tenure'] / 10)
        purchase_frequency = np.clip(
            np.random.poisson(base_frequency * 2, self.num_records),
            1, 30
        )
        
        # Previous purchase amount (correlated with income and frequency)
        prev_purchase_base = demographics_df['Annual_Income'] * 0.1
        prev_purchase = np.clip(
            prev_purchase_base * (1 + purchase_frequency / 20) + 
            np.random.normal(0, 500, self.num_records),
            100, 5000
        ).astype(int)
        
        # Product categories
        categories = ['Electronics', 'Clothing', 'Groceries', 'Home & Garden', 
                      'Sports', 'Books', 'Beauty', 'Toys']
        product_category = np.random.choice(categories, self.num_records)
        
        return pd.DataFrame({
            'Purchase_Frequency': purchase_frequency,
            'Previous_Purchase_Amount': prev_purchase,
            'Product_Category': product_category
        })
    
    def generate_business_factors(self):
        """Generate business and market factors."""
        # Store locations
        locations = ['Downtown', 'Suburban', 'Mall', 'Airport', 'Online']
        store_location = np.random.choice(locations, self.num_records, 
                                        p=[0.25, 0.30, 0.20, 0.10, 0.15])
        
        # Marketing spend per customer
        marketing_spend = np.random.exponential(50, self.num_records)
        marketing_spend = np.clip(marketing_spend, 5, 500).round(2)
        
        # Discount offered (0-50%)
        discount_offered = np.random.beta(2, 5, self.num_records) * 50
        discount_offered = np.round(discount_offered, 1)
        
        # Seasonal demand index (0.5 to 2.0)
        seasonal_demand = np.random.beta(3, 3, self.num_records) * 1.5 + 0.5
        seasonal_demand = np.round(seasonal_demand, 2)
        
        return pd.DataFrame({
            'Store_Location': store_location,
            'Marketing_Spend': marketing_spend,
            'Discount_Offered': discount_offered,
            'Seasonal_Demand_Index': seasonal_demand
        })
    
    def generate_target_variables(self, df):
        """Generate target variables based on features with logical relationships."""
        # Monthly Sales (target for regression)
        # Base calculation from multiple factors
        sales_base = (df['Annual_Income'] * 0.02 + 
                     df['Previous_Purchase_Amount'] * 0.3 +
                     df['Marketing_Spend'] * 2 +
                     df['Purchase_Frequency'] * 50)
        
        # Adjust for seasonal demand and discount
        sales_adjusted = sales_base * df['Seasonal_Demand_Index'] * (1 - df['Discount_Offered']/200)
        
        # Add noise and ensure positive values
        monthly_sales = np.clip(
            sales_adjusted + np.random.normal(0, 200, self.num_records),
            50, 5000
        ).round(2)
        
        # Purchase Decision (Yes/No target for classification)
        # Higher probability for customers with good history
        purchase_prob = (df['Previous_Purchase_Amount'] / 5000 + 
                        df['Purchase_Frequency'] / 30 +
                        df['Marketing_Spend'] / 500 +
                        df['Discount_Offered'] / 50) / 4
        
        purchase_decision = (np.random.random(self.num_records) < purchase_prob).astype(int)
        purchase_decision = np.where(purchase_decision == 1, 'Yes', 'No')
        
        # Loyalty Category (Gold/Silver/Bronze target for classification)
        # Based on tenure, income, and purchase frequency
        loyalty_score = (df['Customer_Tenure'] / 20 * 0.3 +
                        df['Annual_Income'] / 150000 * 0.4 +
                        df['Purchase_Frequency'] / 30 * 0.3)
        
        loyalty_category = []
        for score in loyalty_score:
            if score > 0.7:
                loyalty_category.append('Gold')
            elif score > 0.4:
                loyalty_category.append('Silver')
            else:
                loyalty_category.append('Bronze')
        
        return pd.DataFrame({
            'Monthly_Sales': monthly_sales,
            'Purchase_Decision': purchase_decision,
            'Loyalty_Category': loyalty_category
        })
    
    def generate_complete_dataset(self):
        """Generate the complete synthetic dataset."""
        print(f"Generating {self.num_records} synthetic retail records...")
        
        # Generate different components
        demographics = self.generate_customer_demographics()
        purchase_behavior = self.generate_purchase_behavior(demographics)
        business_factors = self.generate_business_factors()
        
        # Combine all features
        features_df = pd.concat([demographics, purchase_behavior, business_factors], axis=1)
        
        # Generate target variables
        targets = self.generate_target_variables(features_df)
        
        # Combine features and targets
        complete_df = pd.concat([features_df, targets], axis=1)
        
        # Add customer ID
        complete_df.insert(0, 'Customer_ID', range(1, self.num_records + 1))
        
        # Shuffle the dataset
        complete_df = complete_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"Dataset generated successfully with {len(complete_df)} records and {len(complete_df.columns)} columns.")
        return complete_df
    
    def get_data_summary(self, df):
        """Generate comprehensive data summary."""
        summary = {
            'total_records': len(df),
            'total_features': len(df.columns) - 3,  # Excluding target variables
            'target_variables': ['Monthly_Sales', 'Purchase_Decision', 'Loyalty_Category'],
            'numeric_features': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_features': df.select_dtypes(include=['object']).columns.tolist(),
            'missing_values': df.isnull().sum().sum(),
            'data_types': df.dtypes.to_dict()
        }
        
        # Basic statistics for numeric columns
        numeric_stats = df.describe().to_dict()
        summary['statistics'] = numeric_stats
        
        return summary

# Example usage and testing
if __name__ == "__main__":
    generator = RetailDataGenerator(num_records=1500)
    dataset = generator.generate_complete_dataset()
    
    print("\nDataset Preview:")
    print(dataset.head())
    
    print("\nDataset Info:")
    dataset.info()
    
    print("\nTarget Variable Distributions:")
    print("Purchase Decision:")
    print(dataset['Purchase_Decision'].value_counts())
    print("\nLoyalty Category:")
    print(dataset['Loyalty_Category'].value_counts())
    
    # Save to CSV
    dataset.to_csv('retail_dataset.csv', index=False)
    print(f"\nDataset saved to 'retail_dataset.csv'")
