import pandas as pd
import numpy as np
from datetime import datetime
import json

class RetailRecommendationEngine:
    """
    AI-powered recommendation engine for retail business decisions.
    Generates intelligent business suggestions based on customer segments,
    purchase patterns, and predictive analytics.
    """
    
    def __init__(self):
        self.recommendation_rules = self._initialize_recommendation_rules()
        self.cluster_profiles = {}
        self.business_impact_scores = {}
        
    def _initialize_recommendation_rules(self):
        """Initialize rule-based recommendation logic."""
        return {
            'budget_customers': {
                'characteristics': {
                    'income_threshold': 40000,
                    'purchase_frequency_low': True,
                    'price_sensitive': True
                },
                'recommendations': [
                    {
                        'type': 'marketing',
                        'action': 'Launch targeted discount campaigns',
                        'priority': 'high',
                        'expected_impact': '15-25% increase in purchase frequency',
                        'implementation': 'Email campaigns with 10-20% discounts on frequently purchased items'
                    },
                    {
                        'type': 'pricing',
                        'action': 'Implement bundle pricing',
                        'priority': 'medium',
                        'expected_impact': '10-15% increase in average order value',
                        'implementation': 'Create product bundles with 5-10% savings'
                    },
                    {
                        'type': 'loyalty',
                        'action': 'Introduce tiered loyalty program',
                        'priority': 'medium',
                        'expected_impact': '20% improvement in customer retention',
                        'implementation': 'Free entry tier with points, paid premium tier with exclusive benefits'
                    }
                ]
            },
            'premium_customers': {
                'characteristics': {
                    'income_threshold': 80000,
                    'purchase_frequency_high': True,
                    'brand_loyal': True
                },
                'recommendations': [
                    {
                        'type': 'loyalty',
                        'action': 'Offer exclusive VIP membership',
                        'priority': 'high',
                        'expected_impact': '30% increase in customer lifetime value',
                        'implementation': 'Premium membership with early access, personal shopper, exclusive events'
                    },
                    {
                        'type': 'product',
                        'action': 'Curate premium product selections',
                        'priority': 'high',
                        'expected_impact': '25% increase in average order value',
                        'implementation': 'Personalized product recommendations based on purchase history'
                    },
                    {
                        'type': 'service',
                        'action': 'Provide white-glove customer service',
                        'priority': 'medium',
                        'expected_impact': '40% improvement in customer satisfaction',
                        'implementation': 'Dedicated account managers, priority support, free shipping'
                    }
                ]
            },
            'seasonal_buyers': {
                'characteristics': {
                    'seasonal_pattern': True,
                    'purchase_frequency_medium': True,
                    'occasion_driven': True
                },
                'recommendations': [
                    {
                        'type': 'marketing',
                        'action': 'Implement seasonal marketing campaigns',
                        'priority': 'high',
                        'expected_impact': '35% increase in seasonal sales',
                        'implementation': 'Holiday-themed promotions, back-to-school campaigns, summer sales'
                    },
                    {
                        'type': 'inventory',
                        'action': 'Optimize seasonal inventory management',
                        'priority': 'high',
                        'expected_impact': '20% reduction in carrying costs',
                        'implementation': 'Predictive inventory planning based on seasonal patterns'
                    },
                    {
                        'type': 'communication',
                        'action': 'Send timely seasonal reminders',
                        'priority': 'medium',
                        'expected_impact': '25% improvement in conversion rates',
                        'implementation': 'Automated email/SMS reminders before peak seasons'
                    }
                ]
            },
            'high_value_customers': {
                'characteristics': {
                    'high_clv': True,
                    'frequent_purchases': True,
                    'high_average_order': True
                },
                'recommendations': [
                    {
                        'type': 'retention',
                        'action': 'Create exclusive membership program',
                        'priority': 'high',
                        'expected_impact': '40% reduction in churn rate',
                        'implementation': 'Invite-only club with premium benefits and recognition'
                    },
                    {
                        'type': 'cross_sell',
                        'action': 'Implement sophisticated cross-selling',
                        'priority': 'high',
                        'expected_impact': '30% increase in revenue per customer',
                        'implementation': 'AI-powered product recommendations based on browsing and purchase history'
                    },
                    {
                        'type': 'feedback',
                        'action': 'Establish customer advisory board',
                        'priority': 'medium',
                        'expected_impact': '15% improvement in product development',
                        'implementation': 'Quarterly meetings with top customers for feedback and insights'
                    }
                ]
            },
            'at_risk_customers': {
                'characteristics': {
                    'declining_frequency': True,
                    'low_recent_purchases': True,
                    'low_engagement': True
                },
                'recommendations': [
                    {
                        'type': 'retention',
                        'action': 'Launch re-engagement campaign',
                        'priority': 'high',
                        'expected_impact': '20-30% recovery of at-risk customers',
                        'implementation': 'Personalized "we miss you" offers with significant discounts'
                    },
                    {
                        'type': 'incentives',
                        'action': 'Provide win-back incentives',
                        'priority': 'high',
                        'expected_impact': '25% increase in return purchases',
                        'implementation': 'Limited-time special offers and loyalty point bonuses'
                    },
                    {
                        'type': 'feedback',
                        'action': 'Conduct exit interviews',
                        'priority': 'medium',
                        'expected_impact': 'Valuable insights for improvement',
                        'implementation': 'Short surveys with incentive for completion'
                    }
                ]
            }
        }
    
    def analyze_customer_segments(self, df, cluster_labels):
        """Analyze customer segments and create profiles."""
        segment_analysis = {}
        
        # Add cluster labels to dataframe
        df_with_clusters = df.copy()
        df_with_clusters['Cluster'] = cluster_labels
        
        for cluster_id in np.unique(cluster_labels):
            cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster_id]
            
            # Calculate segment characteristics
            profile = {
                'size': len(cluster_data),
                'percentage': (len(cluster_data) / len(df)) * 100,
                'avg_income': cluster_data['Annual_Income'].mean(),
                'avg_purchase_frequency': cluster_data['Purchase_Frequency'].mean(),
                'avg_sales': cluster_data['Monthly_Sales'].mean(),
                'loyalty_distribution': cluster_data['Loyalty_Category'].value_counts().to_dict(),
                'purchase_decision_rate': (cluster_data['Purchase_Decision'] == 'Yes').mean() * 100,
                'avg_tenure': cluster_data['Customer_Tenure'].mean(),
                'avg_discount_sensitivity': cluster_data['Discount_Offered'].mean()
            }
            
            segment_analysis[f'Cluster_{cluster_id}'] = profile
            
            # Classify segment type
            segment_type = self._classify_segment_type(profile)
            segment_analysis[f'Cluster_{cluster_id}']['segment_type'] = segment_type
            
        self.cluster_profiles = segment_analysis
        return segment_analysis
    
    def _classify_segment_type(self, profile):
        """Classify customer segment based on characteristics."""
        avg_income = profile['avg_income']
        avg_frequency = profile['avg_purchase_frequency']
        avg_sales = profile['avg_sales']
        purchase_rate = profile['purchase_decision_rate']
        
        # Classification logic
        if avg_income < 40000 and avg_frequency < 5:
            return 'budget_customers'
        elif avg_income > 80000 and avg_frequency > 10:
            return 'premium_customers'
        elif avg_sales > 2000 and avg_frequency > 8:
            return 'high_value_customers'
        elif purchase_rate < 50 and avg_frequency < 3:
            return 'at_risk_customers'
        else:
            return 'seasonal_buyers'
    
    def generate_recommendations(self, segment_id=None):
        """Generate business recommendations for specific segment or all segments."""
        recommendations = {}
        
        if segment_id:
            # Generate recommendations for specific segment
            if segment_id in self.cluster_profiles:
                segment_type = self.cluster_profiles[segment_id]['segment_type']
                recommendations[segment_id] = self._get_segment_recommendations(segment_type)
        else:
            # Generate recommendations for all segments
            for segment_id, profile in self.cluster_profiles.items():
                segment_type = profile['segment_type']
                recommendations[segment_id] = self._get_segment_recommendations(segment_type)
        
        return recommendations
    
    def _get_segment_recommendations(self, segment_type):
        """Get recommendations for a specific segment type."""
        if segment_type in self.recommendation_rules:
            return self.recommendation_rules[segment_type]['recommendations']
        else:
            # Default recommendations
            return [
                {
                    'type': 'general',
                    'action': 'Monitor customer behavior patterns',
                    'priority': 'medium',
                    'expected_impact': 'Improved understanding of customer needs',
                    'implementation': 'Regular analysis of purchase patterns and feedback'
                }
            ]
    
    def calculate_business_impact(self, recommendations, segment_profile):
        """Calculate potential business impact of recommendations."""
        impact_scores = {}
        
        for rec in recommendations:
            priority_multiplier = {'high': 1.0, 'medium': 0.7, 'low': 0.4}[rec['priority']]
            segment_size = segment_profile['size']
            
            # Estimate impact based on recommendation type
            if rec['type'] == 'marketing':
                base_impact = 0.15  # 15% base impact
            elif rec['type'] == 'pricing':
                base_impact = 0.12
            elif rec['type'] == 'loyalty':
                base_impact = 0.20
            elif rec['type'] == 'retention':
                base_impact = 0.25
            else:
                base_impact = 0.10
            
            # Calculate overall impact score
            impact_score = base_impact * priority_multiplier * (segment_size / 100)
            impact_scores[rec['action']] = impact_score
        
        return impact_scores
    
    def generate_executive_summary(self):
        """Generate executive summary of recommendations."""
        if not self.cluster_profiles:
            return "No customer segments analyzed. Please run analyze_customer_segments first."
        
        summary = {
            'total_segments': len(self.cluster_profiles),
            'largest_segment': max(self.cluster_profiles.items(), 
                                  key=lambda x: x[1]['size'])[0],
            'highest_value_segment': max(self.cluster_profiles.items(), 
                                       key=lambda x: x[1]['avg_sales'])[0],
            'key_insights': [],
            'priority_actions': []
        }
        
        # Generate key insights
        for segment_id, profile in self.cluster_profiles.items():
            if profile['percentage'] > 30:
                summary['key_insights'].append(
                    f"{segment_id} represents {profile['percentage']:.1f}% of customer base"
                )
            if profile['avg_sales'] > 2000:
                summary['key_insights'].append(
                    f"{segment_id} has high average sales (${profile['avg_sales']:.2f})"
                )
        
        # Get priority actions
        all_recommendations = self.generate_recommendations()
        for segment_id, recs in all_recommendations.items():
            for rec in recs:
                if rec['priority'] == 'high':
                    summary['priority_actions'].append({
                        'segment': segment_id,
                        'action': rec['action'],
                        'impact': rec['expected_impact']
                    })
        
        return summary
    
    def create_action_plan(self, timeframe='quarterly'):
        """Create actionable plan with timeline and responsibilities."""
        recommendations = self.generate_recommendations()
        
        action_plan = {
            'timeframe': timeframe,
            'initiatives': []
        }
        
        for segment_id, recs in recommendations.items():
            for rec in recs:
                initiative = {
                    'segment': segment_id,
                    'action': rec['action'],
                    'type': rec['type'],
                    'priority': rec['priority'],
                    'expected_impact': rec['expected_impact'],
                    'implementation_steps': rec['implementation'],
                    'timeline': self._estimate_timeline(rec['priority']),
                    'responsible_team': self._assign_team(rec['type']),
                    'success_metrics': self._define_success_metrics(rec['type'])
                }
                action_plan['initiatives'].append(initiative)
        
        # Sort by priority
        action_plan['initiatives'].sort(
            key=lambda x: {'high': 0, 'medium': 1, 'low': 2}[x['priority']]
        )
        
        return action_plan
    
    def _estimate_timeline(self, priority):
        """Estimate implementation timeline based on priority."""
        timelines = {
            'high': '4-6 weeks',
            'medium': '6-8 weeks',
            'low': '8-12 weeks'
        }
        return timelines.get(priority, '8-10 weeks')
    
    def _assign_team(self, recommendation_type):
        """Assign responsible team based on recommendation type."""
        team_assignments = {
            'marketing': 'Marketing Team',
            'pricing': 'Pricing Strategy Team',
            'loyalty': 'Customer Retention Team',
            'retention': 'Customer Success Team',
            'product': 'Product Management Team',
            'service': 'Customer Service Team',
            'inventory': 'Supply Chain Team',
            'communication': 'Communications Team',
            'cross_sell': 'Sales Team',
            'feedback': 'Customer Insights Team',
            'incentives': 'Marketing Team',
            'general': 'Operations Team'
        }
        return team_assignments.get(recommendation_type, 'Cross-functional Team')
    
    def _define_success_metrics(self, recommendation_type):
        """Define success metrics for different recommendation types."""
        metrics = {
            'marketing': ['Conversion rate', 'Campaign ROI', 'Customer acquisition cost'],
            'pricing': ['Average order value', 'Gross margin', 'Price elasticity'],
            'loyalty': ['Customer retention rate', 'Repeat purchase rate', 'CLV'],
            'retention': ['Churn rate reduction', 'Customer satisfaction score', 'NPS'],
            'product': ['Product adoption rate', 'Customer feedback score', 'Return rate'],
            'service': ['Customer satisfaction', 'First response time', 'Resolution rate'],
            'inventory': ['Inventory turnover', 'Stockout rate', 'Carrying cost reduction'],
            'communication': ['Open rate', 'Click-through rate', 'Engagement score'],
            'cross_sell': ['Cross-sell revenue', 'Product basket size', 'Recommendation acceptance'],
            'feedback': ['Response rate', 'Insight quality', 'Implementation rate'],
            'incentives': ['Redemption rate', 'Incremental revenue', 'Customer reactivation'],
            'general': ['Overall satisfaction', 'Operational efficiency', 'Cost savings']
        }
        return metrics.get(recommendation_type, ['Performance improvement', 'Customer satisfaction'])
    
    def export_recommendations(self, format='json'):
        """Export recommendations in specified format."""
        if not self.cluster_profiles:
            return "No recommendations available. Please generate recommendations first."
        
        data = {
            'analysis_date': datetime.now().isoformat(),
            'segment_analysis': self.cluster_profiles,
            'recommendations': self.generate_recommendations(),
            'executive_summary': self.generate_executive_summary(),
            'action_plan': self.create_action_plan()
        }
        
        if format == 'json':
            return json.dumps(data, indent=2, default=str)
        elif format == 'dict':
            return data
        else:
            raise ValueError("Format must be 'json' or 'dict'")
    
    def get_recommendation_dashboard_data(self):
        """Get data formatted for dashboard visualization."""
        dashboard_data = {
            'segment_overview': [],
            'recommendation_summary': {},
            'impact_forecast': {}
        }
        
        # Segment overview
        for segment_id, profile in self.cluster_profiles.items():
            dashboard_data['segment_overview'].append({
                'segment': segment_id,
                'size': profile['size'],
                'percentage': profile['percentage'],
                'avg_sales': profile['avg_sales'],
                'type': profile['segment_type']
            })
        
        # Recommendation summary by type
        all_recs = self.generate_recommendations()
        rec_types = {}
        for segment_recs in all_recs.values():
            for rec in segment_recs:
                rec_type = rec['type']
                if rec_type not in rec_types:
                    rec_types[rec_type] = {'count': 0, 'high_priority': 0}
                rec_types[rec_type]['count'] += 1
                if rec['priority'] == 'high':
                    rec_types[rec_type]['high_priority'] += 1
        
        dashboard_data['recommendation_summary'] = rec_types
        
        # Impact forecast
        for segment_id, profile in self.cluster_profiles.items():
            segment_type = profile['segment_type']
            recommendations = self._get_segment_recommendations(segment_type)
            impact_scores = self.calculate_business_impact(recommendations, profile)
            dashboard_data['impact_forecast'][segment_id] = impact_scores
        
        return dashboard_data

# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    from data_generator import RetailDataGenerator
    from models import RetailMLModels
    
    # Generate data and train models
    generator = RetailDataGenerator(num_records=1500)
    df = generator.generate_complete_dataset()
    
    ml_models = RetailMLModels()
    _, _, cluster_labels = ml_models.train_customer_segmentation_model(df)
    
    # Initialize recommendation engine
    engine = RetailRecommendationEngine()
    
    # Analyze segments and generate recommendations
    segment_analysis = engine.analyze_customer_segments(df, cluster_labels)
    recommendations = engine.generate_recommendations()
    executive_summary = engine.generate_executive_summary()
    action_plan = engine.create_action_plan()
    
    print("Customer Segment Analysis:")
    for segment, profile in segment_analysis.items():
        print(f"{segment}: {profile['segment_type']} - {profile['size']} customers")
    
    print("\nExecutive Summary:")
    print(json.dumps(executive_summary, indent=2))
