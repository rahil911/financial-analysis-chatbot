import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime, timedelta, date
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
import json
warnings.filterwarnings('ignore')

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle timestamps, dates, and numpy types"""
    def default(self, obj):
        if isinstance(obj, (datetime, date, pd.Timestamp)):
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        return super().default(obj)

class FinancialAnalyzer:
    def __init__(self, data_path: str = "DB"):
        """Initialize the Financial Analyzer"""
        self.data_path = Path(data_path)
        self.reports_path = Path("analysis/reports/financial")
        self.reports_path.mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.gl_transactions = None
        self.ar_details = None
        self.ar_header = None
        self.ap_details = None
        self.forecasts = None
        
        # Analysis results
        self.results = {
            'cash_flow': {},
            'ar_aging': {},
            'revenue_forecast': {},
            'profitability': {},
            'anomalies': {},
            'working_capital': {}
        }

    def load_data(self) -> None:
        """Load all relevant financial data"""
        print("Loading financial data...")
        
        try:
            # Load GL Transactions
            gl_path = self.data_path / "dbo_F_GL Transaction Detail.csv"
            self.gl_transactions = pd.read_csv(gl_path, low_memory=False)
            print(f"Loaded GL Transactions: {len(self.gl_transactions):,} rows")
            
            # Load AR Details
            ar_detail_path = self.data_path / "dbo_F_AR Detail.csv"
            self.ar_details = pd.read_csv(ar_detail_path, low_memory=False)
            print(f"Loaded AR Details: {len(self.ar_details):,} rows")
            
            # Load AR Header
            ar_header_path = self.data_path / "dbo_F_AR Header.csv"
            self.ar_header = pd.read_csv(ar_header_path, low_memory=False)
            print(f"Loaded AR Header: {len(self.ar_header):,} rows")
            
            # Load AP Details
            ap_detail_path = self.data_path / "dbo_F_AP Detail.csv"
            self.ap_details = pd.read_csv(ap_detail_path, low_memory=False)
            print(f"Loaded AP Details: {len(self.ap_details):,} rows")
            
            # Load GL Forecasts
            forecast_path = self.data_path / "dbo_F_GL Forecast.csv"
            self.forecasts = pd.read_csv(forecast_path, low_memory=False)
            print(f"Loaded GL Forecasts: {len(self.forecasts):,} rows")
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def analyze_cash_flow(self) -> Dict:
        """Analyze cash flow patterns and trends"""
        print("\nAnalyzing cash flow...")
        
        # Prepare GL transactions
        gl_df = self.gl_transactions.copy()
        gl_df['Posting Date'] = pd.to_datetime(gl_df['Posting Date'])
        
        # Calculate daily cash flows
        cash_flows = gl_df.groupby('Posting Date')['Txn Amount'].sum().reset_index()
        cash_flows['Cumulative Cash Flow'] = cash_flows['Txn Amount'].cumsum()
        
        # Calculate key metrics
        metrics = {
            'total_inflow': float(gl_df[gl_df['Txn Amount'] > 0]['Txn Amount'].sum()),
            'total_outflow': float(gl_df[gl_df['Txn Amount'] < 0]['Txn Amount'].sum()),
            'net_cash_flow': float(gl_df['Txn Amount'].sum()),
            'daily_average': float(cash_flows['Txn Amount'].mean()),
            'daily_volatility': float(cash_flows['Txn Amount'].std())
        }
        
        # Identify seasonal patterns
        cash_flows['Month'] = cash_flows['Posting Date'].dt.month
        seasonal_patterns = cash_flows.groupby('Month')['Txn Amount'].agg(['mean', 'std']).to_dict()
        
        # Detect anomalies
        isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        cash_flows['is_anomaly'] = isolation_forest.fit_predict(cash_flows[['Txn Amount']])
        anomalies = cash_flows[cash_flows['is_anomaly'] == -1]
        
        # Generate visualizations
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cash_flows['Posting Date'],
            y=cash_flows['Cumulative Cash Flow'],
            name='Cumulative Cash Flow'
        ))
        fig.add_trace(go.Scatter(
            x=anomalies['Posting Date'],
            y=anomalies['Txn Amount'],
            mode='markers',
            name='Anomalies',
            marker=dict(color='red', size=10)
        ))
        fig.update_layout(title='Cash Flow Analysis')
        fig.write_html(self.reports_path / 'cash_flow_analysis.html')
        
        self.results['cash_flow'] = {
            'metrics': metrics,
            'seasonal_patterns': seasonal_patterns,
            'anomalies': anomalies.to_dict(orient='records')
        }
        
        return self.results['cash_flow']

    def analyze_ar_aging(self) -> Dict:
        """Analyze accounts receivable aging and payment patterns"""
        print("\nAnalyzing AR aging...")
        
        ar_df = self.ar_header.copy()
        ar_df['Posting Date'] = pd.to_datetime(ar_df['Posting Date'])
        current_date = ar_df['Posting Date'].max()
        
        # Calculate aging buckets
        ar_df['Days Outstanding'] = (current_date - ar_df['Posting Date']).dt.days
        ar_df['Aging Bucket'] = pd.cut(
            ar_df['Days Outstanding'],
            bins=[-float('inf'), 30, 60, 90, 120, float('inf')],
            labels=['0-30', '31-60', '61-90', '91-120', '120+']
        )
        
        # Calculate aging metrics
        aging_summary = ar_df.groupby('Aging Bucket').agg({
            'Balance Due Amount': ['sum', 'count'],
            'Days Outstanding': 'mean'
        }).round(2)
        
        # Convert MultiIndex columns to flat format
        aging_summary.columns = [f"{col[0]}_{col[1]}" for col in aging_summary.columns]
        
        # Calculate DSO (Days Sales Outstanding)
        total_ar = ar_df['Balance Due Amount'].sum()
        total_sales = ar_df['Invoice Amount'].sum()  # Using actual invoice amount
        dso = (total_ar / (total_sales / 365)) if total_sales != 0 else 0
        
        # Analyze payment patterns
        payment_patterns = ar_df.groupby('Customer Key').agg({
            'Days Outstanding': ['mean', 'std'],
            'Balance Due Amount': 'sum',
            'Days to Pay': 'mean',
            'Number Days Late': 'mean',
            'Number Payments Late': 'sum',
            'Number Payments': 'sum'
        }).round(2)
        
        # Convert MultiIndex columns to flat format
        payment_patterns.columns = [f"{col[0]}_{col[1]}" if isinstance(col, tuple) else col for col in payment_patterns.columns]
        
        # Calculate late payment ratio
        payment_patterns['late_payment_ratio'] = (
            payment_patterns['Number Payments Late_sum'] / 
            payment_patterns['Number Payments_sum']
        ).fillna(0)
        
        # Generate visualization
        fig = px.bar(
            ar_df.groupby('Aging Bucket')['Balance Due Amount'].sum().reset_index(),
            x='Aging Bucket',
            y='Balance Due Amount',
            title='AR Aging Analysis'
        )
        fig.write_html(self.reports_path / 'ar_aging_analysis.html')
        
        # Generate payment behavior visualization
        payment_behavior = ar_df.groupby('Customer Key').agg({
            'Days to Pay': 'mean',
            'Balance Due Amount': 'sum'
        }).reset_index()
        
        fig2 = px.scatter(
            payment_behavior,
            x='Days to Pay',
            y='Balance Due Amount',
            title='Customer Payment Behavior Analysis'
        )
        fig2.write_html(self.reports_path / 'payment_behavior_analysis.html')
        
        self.results['ar_aging'] = {
            'aging_summary': aging_summary.to_dict(orient='index'),
            'dso': float(dso),
            'payment_patterns': payment_patterns.to_dict(orient='index'),
            'high_risk_customers': payment_patterns[
                (payment_patterns['Days Outstanding_mean'] > 90) |
                (payment_patterns['late_payment_ratio'] > 0.5)
            ].to_dict(orient='index'),
            'total_overdue': float(ar_df[ar_df['Days Outstanding'] > 30]['Balance Due Amount'].sum()),
            'total_severely_overdue': float(ar_df[ar_df['Days Outstanding'] > 90]['Balance Due Amount'].sum()),
            'payment_statistics': {
                'avg_days_to_pay': float(ar_df['Days to Pay'].mean()),
                'avg_days_late': float(ar_df['Number Days Late'].mean()),
                'late_payment_percentage': float(
                    (ar_df['Number Payments Late'].sum() / ar_df['Number Payments'].sum()) * 100
                    if ar_df['Number Payments'].sum() > 0 else 0
                )
            }
        }
        
        return self.results['ar_aging']

    def forecast_revenue(self) -> Dict:
        """Forecast revenue using multiple models and ensemble methods"""
        print("\nForecasting revenue...")
        
        # Prepare data
        gl_df = self.gl_transactions.copy()
        gl_df['Posting Date'] = pd.to_datetime(gl_df['Posting Date'])
        daily_revenue = gl_df[gl_df['Txn Amount'] > 0].groupby('Posting Date')['Txn Amount'].sum()
        
        # Create features
        X = pd.DataFrame({
            'day_of_week': daily_revenue.index.dayofweek,
            'month': daily_revenue.index.month,
            'year': daily_revenue.index.year,
            'day_of_month': daily_revenue.index.day
        })
        y = daily_revenue.values
        
        # Train multiple models
        models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'xgb': XGBRegressor(n_estimators=100, random_state=42),
            'lgbm': LGBMRegressor(n_estimators=100, random_state=42)
        }
        
        predictions = {}
        for name, model in models.items():
            model.fit(X, y)
            predictions[name] = model.predict(X)
        
        # Ensemble predictions
        ensemble_pred = np.mean([pred for pred in predictions.values()], axis=0)
        
        # Calculate forecast accuracy
        mape = np.mean(np.abs((y - ensemble_pred) / y)) * 100
        rmse = np.sqrt(np.mean((y - ensemble_pred) ** 2))
        
        # Generate visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_revenue.index,
            y=daily_revenue.values,
            name='Actual Revenue'
        ))
        fig.add_trace(go.Scatter(
            x=daily_revenue.index,
            y=ensemble_pred,
            name='Forecasted Revenue'
        ))
        fig.update_layout(title='Revenue Forecast Analysis')
        fig.write_html(self.reports_path / 'revenue_forecast.html')
        
        self.results['revenue_forecast'] = {
            'accuracy_metrics': {
                'mape': float(mape),
                'rmse': float(rmse)
            },
            'model_importance': {
                name: dict(zip(X.columns, model.feature_importances_))
                for name, model in models.items()
            }
        }
        
        return self.results['revenue_forecast']

    def analyze_profitability(self) -> Dict:
        """Analyze profitability by various dimensions"""
        print("\nAnalyzing profitability...")
        
        gl_df = self.gl_transactions.copy()
        
        # Calculate profitability metrics
        profitability = {
            'overall': {
                'revenue': float(gl_df[gl_df['Txn Amount'] > 0]['Txn Amount'].sum()),
                'expenses': float(gl_df[gl_df['Txn Amount'] < 0]['Txn Amount'].sum()),
                'profit': float(gl_df['Txn Amount'].sum())
            }
        }
        
        # Analyze by dimension
        for dim in ['Company Code', 'Business Unit Key', 'Department Key']:
            if dim in gl_df.columns:
                dim_profit = gl_df.groupby(dim)['Txn Amount'].agg([
                    ('revenue', lambda x: x[x > 0].sum()),
                    ('expenses', lambda x: x[x < 0].sum()),
                    ('profit', 'sum')
                ]).round(2)
                
                # Convert column names to strings
                dim_profit.columns = [str(col) for col in dim_profit.columns]
                profitability[f'by_{dim.lower().replace(" ", "_")}'] = dim_profit.to_dict(orient='index')
        
        # Generate visualization
        if 'Business Unit Key' in gl_df.columns:
            fig = px.treemap(
                gl_df,
                path=['Business Unit Key'],
                values='Txn Amount',
                title='Profitability by Business Unit'
            )
            fig.write_html(self.reports_path / 'profitability_analysis.html')
        
        self.results['profitability'] = profitability
        return self.results['profitability']

    def generate_report(self) -> None:
        """Generate a comprehensive financial analysis report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.reports_path / f"financial_analysis_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=4, cls=CustomJSONEncoder)
        
        print(f"\nFinancial analysis report generated: {report_path}")
        
        # Generate executive summary
        summary = {
            'cash_flow_health': {
                'net_position': self.results['cash_flow']['metrics']['net_cash_flow'],
                'daily_volatility': self.results['cash_flow']['metrics']['daily_volatility'],
                'anomaly_count': len(self.results['cash_flow']['anomalies'])
            },
            'ar_health': {
                'dso': self.results['ar_aging']['dso'],
                'high_risk_customers': len(self.results['ar_aging']['high_risk_customers'])
            },
            'forecast_accuracy': self.results['revenue_forecast']['accuracy_metrics'],
            'profitability': {
                'overall_profit_margin': (
                    self.results['profitability']['overall']['profit'] /
                    self.results['profitability']['overall']['revenue']
                    if self.results['profitability']['overall']['revenue'] != 0 else 0
                )
            }
        }
        
        with open(self.reports_path / f"executive_summary_{timestamp}.json", 'w') as f:
            json.dump(summary, f, indent=4, cls=CustomJSONEncoder)

    def run_analysis(self) -> None:
        """Run the complete financial analysis pipeline"""
        self.load_data()
        self.analyze_cash_flow()
        self.analyze_ar_aging()
        self.forecast_revenue()
        self.analyze_profitability()
        self.generate_report()

def main():
    analyzer = FinancialAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main() 