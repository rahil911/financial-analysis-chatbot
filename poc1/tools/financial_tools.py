import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import json
from datetime import datetime
import os
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("financial_tools.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FinancialTools")

class FinancialTools:
    """Financial analysis tools that can be called by the chatbot"""
    
    def __init__(self, db_path="financial.db"):
        """Initialize with database connection"""
        # Use absolute path for database
        self.db_path = Path(db_path)
        logger.info(f"Database path set to: {self.db_path} (absolute: {self.db_path.absolute()})")
        
        # Verify database exists
        if not self.db_path.exists():
            logger.error(f"Database file not found at: {self.db_path.absolute()}")
            
            # Try to find the database at common locations
            possible_paths = [
                Path("data/financial.db"),
                Path("../data/financial.db"),
                Path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "financial.db")),
                Path("/Users/rahilharihar/Projects/Bicycle/poc1/data/financial.db")
            ]
            
            for path in possible_paths:
                if path.exists():
                    logger.info(f"Found database at alternate location: {path.absolute()}")
                    self.db_path = path.absolute()
                    break
        else:
            logger.info(f"Database file exists at: {self.db_path.absolute()}")
            
        # Test connection
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            # List all tables to log
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            logger.info(f"Tables in database: {[t[0] for t in tables]}")
            conn.close()
        except Exception as e:
            logger.error(f"Error connecting to database: {str(e)}")
        
        # Use absolute path for reports
        module_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        project_root = module_dir.parent
        self.reports_path = project_root / "data" / "reports"
        self.reports_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Reports path set to: {self.reports_path}")
    
    def _get_connection(self):
        """Get a database connection"""
        try:
            conn = sqlite3.connect(self.db_path)
            return conn
        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            raise
    
    def _execute_query(self, query, params=None):
        """Execute a SQL query and return the results as a DataFrame"""
        logger.info(f"Executing SQL query: {query}")
        if params:
            logger.info(f"With parameters: {params}")
            
        try:
            conn = self._get_connection()
            df = pd.read_sql_query(query, conn, params=params)
            logger.info(f"Query returned {len(df)} rows")
            conn.close()
            return df
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            # Provide more details to debug table issues
            try:
                conn = self._get_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                logger.info(f"Available tables: {[t[0] for t in tables]}")
                conn.close()
            except Exception as inner_e:
                logger.error(f"Could not list tables: {str(inner_e)}")
            raise
    
    def cash_flow_analysis(self, start_date=None, end_date=None):
        """
        Analyze cash flow patterns and trends
        
        Args:
            start_date (str, optional): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format
        
        Returns:
            dict: Cash flow analysis results
        """
        # Construct date filters if provided
        date_filter = ""
        params = {}
        
        logger.info(f"Cash flow analysis requested with start_date={start_date}, end_date={end_date}")
        
        # If no dates provided, use the last 3 months by default
        if not start_date and not end_date:
            logger.info("No date range specified, defaulting to last 3 months")
            query = """
            SELECT MIN("Posting Date") as earliest_date, MAX("Posting Date") as latest_date
            FROM dbo_F_GL_Transaction_Detail
            """
            try:
                date_range = self._execute_query(query)
                if not date_range.empty:
                    latest_date = pd.to_datetime(date_range['latest_date'].iloc[0])
                    # Default to 3 months before the latest date
                    earliest_date = latest_date - pd.DateOffset(months=3)
                    start_date = earliest_date.strftime('%Y-%m-%d')
                    logger.info(f"Using default date range: {start_date} to {latest_date.strftime('%Y-%m-%d')}")
            except Exception as e:
                logger.error(f"Error determining default date range: {str(e)}")
        
        if start_date:
            date_filter += " AND \"Posting Date\" >= :start_date"
            params['start_date'] = start_date
        
        if end_date:
            date_filter += " AND \"Posting Date\" <= :end_date"
            params['end_date'] = end_date
        
        # Query GL transactions
        query = f"""
        SELECT * FROM dbo_F_GL_Transaction_Detail
        WHERE 1=1 {date_filter}
        """
        
        logger.info(f"Executing cash flow query with filter: {date_filter}")
        
        try:
            gl_transactions = self._execute_query(query, params)
            
            if len(gl_transactions) == 0:
                logger.warning(f"No cash flow data found for the specified period (start_date={start_date}, end_date={end_date})")
                return {"error": f"No data found for the specified period. Please try a different date range. Available data may not include this period."}
            
            logger.info(f"Found {len(gl_transactions)} transactions for cash flow analysis")
            
            # Convert posting date to datetime
            gl_transactions['Posting Date'] = pd.to_datetime(gl_transactions['Posting Date'])
            
            # Calculate daily cash flows
            cash_flows = gl_transactions.groupby('Posting Date')['Txn Amount'].sum().reset_index()
            cash_flows['Cumulative Cash Flow'] = cash_flows['Txn Amount'].cumsum()
            
            # Calculate key metrics
            metrics = {
                'total_inflow': float(gl_transactions[gl_transactions['Txn Amount'] > 0]['Txn Amount'].sum()),
                'total_outflow': float(gl_transactions[gl_transactions['Txn Amount'] < 0]['Txn Amount'].sum()),
                'net_cash_flow': float(gl_transactions['Txn Amount'].sum()),
                'daily_average': float(cash_flows['Txn Amount'].mean()),
                'daily_volatility': float(cash_flows['Txn Amount'].std())
            }
            
            # Identify seasonal patterns
            cash_flows['Month'] = cash_flows['Posting Date'].dt.month
            seasonal_patterns_df = cash_flows.groupby('Month')['Txn Amount'].agg(['mean', 'std'])
            
            # Convert seasonal patterns to JSON-serializable format
            seasonal_patterns = {}
            for month, row in seasonal_patterns_df.iterrows():
                seasonal_patterns[int(month)] = {
                    'mean': float(row['mean']),
                    'std': float(row['std'])
                }
            
            # Detect anomalies (only if we have enough data)
            anomalies = []
            if len(cash_flows) > 10:
                isolation_forest = IsolationForest(contamination=0.1, random_state=42)
                cash_flows['is_anomaly'] = isolation_forest.fit_predict(cash_flows[['Txn Amount']])
                anomalies_df = cash_flows[cash_flows['is_anomaly'] == -1]
                
                # Convert anomalies to JSON-serializable format
                anomalies = []
                for _, row in anomalies_df.iterrows():
                    anomalies.append({
                        'Posting Date': row['Posting Date'].strftime('%Y-%m-%d'),
                        'Txn Amount': float(row['Txn Amount']),
                        'Cumulative Cash Flow': float(row['Cumulative Cash Flow'])
                    })
            
            # Generate visualization
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=cash_flows['Posting Date'],
                y=cash_flows['Cumulative Cash Flow'],
                name='Cumulative Cash Flow'
            ))
            
            if anomalies:
                anomalies_dates = [pd.to_datetime(anomaly['Posting Date']) for anomaly in anomalies]
                anomalies_amounts = [anomaly['Txn Amount'] for anomaly in anomalies]
                
                fig.add_trace(go.Scatter(
                    x=anomalies_dates,
                    y=anomalies_amounts,
                    mode='markers',
                    name='Anomalies',
                    marker=dict(color='red', size=10)
                ))
            
            fig.update_layout(title='Cash Flow Analysis')
            
            # Save visualization
            viz_path = str(self.reports_path / f'cash_flow_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html')
            fig.write_html(viz_path)
            
            return {
                'metrics': metrics,
                'seasonal_patterns': seasonal_patterns,
                'anomalies': anomalies,
                'visualization_path': viz_path
            }
            
        except Exception as e:
            error_msg = f"Error during cash flow analysis: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def ar_aging_analysis(self):
        """
        Analyze accounts receivable aging
        
        Returns:
            dict: Accounts receivable aging analysis results
        """
        # Query AR data
        query = """
        SELECT * FROM dbo_F_AR_Header
        """
        
        ar_data = self._execute_query(query)
        
        if len(ar_data) == 0:
            return {"error": "No accounts receivable data found"}
        
        # Make sure 'Posting Date' and 'Due Date' are datetime
        ar_data['Posting Date'] = pd.to_datetime(ar_data['Posting Date'])
        ar_data['Due Date'] = pd.to_datetime(ar_data['Due Date'])
        
        # Calculate days overdue (as of most recent data)
        max_date = ar_data['Posting Date'].max()
        ar_data['Days Overdue'] = (max_date - ar_data['Due Date']).dt.days
        ar_data.loc[ar_data['Days Overdue'] < 0, 'Days Overdue'] = 0
        
        # Define aging buckets
        aging_buckets = {
            'current': (0, 30),
            '31-60_days': (31, 60),
            '61-90_days': (61, 90),
            '91-120_days': (91, 120),
            'over_120_days': (121, float('inf'))
        }
        
        # Calculate totals for each bucket
        bucket_totals = {}
        for bucket, (min_days, max_days) in aging_buckets.items():
            bucket_data = ar_data[(ar_data['Days Overdue'] >= min_days) & (ar_data['Days Overdue'] <= max_days)]
            bucket_totals[bucket] = {
                'count': int(len(bucket_data)),
                'amount': float(bucket_data['Balance Due Amount'].sum()),
                'percentage': float(bucket_data['Balance Due Amount'].sum() / ar_data['Balance Due Amount'].sum()) if ar_data['Balance Due Amount'].sum() > 0 else 0
            }
        
        # Calculate overall metrics
        total_overdue = float(ar_data[ar_data['Days Overdue'] > 30]['Balance Due Amount'].sum())
        severely_overdue = float(ar_data[ar_data['Days Overdue'] > 90]['Balance Due Amount'].sum())
        
        # Calculate Days Sales Outstanding (DSO)
        # Use 90-day average revenue as the denominator
        # Query GL data for revenue
        revenue_query = """
        SELECT * FROM dbo_F_GL_Transaction_Detail
        WHERE "Txn Amount" > 0
        """
        gl_data = self._execute_query(revenue_query)
        
        if len(gl_data) > 0:
            gl_data['Posting Date'] = pd.to_datetime(gl_data['Posting Date'])
            
            # Calculate average daily revenue over the last 90 days
            last_date = gl_data['Posting Date'].max()
            start_date = last_date - pd.Timedelta(days=90)
            recent_revenue = gl_data[gl_data['Posting Date'] >= start_date]
            
            if len(recent_revenue) > 0:
                avg_daily_revenue = recent_revenue['Txn Amount'].sum() / 90
                total_ar = ar_data['Balance Due Amount'].sum()
                dso = total_ar / avg_daily_revenue if avg_daily_revenue > 0 else 0
            else:
                dso = 0
        else:
            dso = 0
        
        # Generate visualization
        labels = list(aging_buckets.keys())
        values = [bucket_totals[bucket]['amount'] for bucket in labels]
        
        fig = px.pie(
            names=labels,
            values=values,
            title='Accounts Receivable Aging',
            hole=0.4
        )
        
        # Save visualization
        viz_path = str(self.reports_path / f'ar_aging_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html')
        fig.write_html(viz_path)
        
        return {
            'aging_buckets': bucket_totals,
            'total_receivables': float(ar_data['Balance Due Amount'].sum()),
            'total_overdue': total_overdue,
            'total_severely_overdue': severely_overdue,
            'dso': float(dso),
            'visualization_path': viz_path
        }
    
    def revenue_forecast(self, days_ahead=30):
        """
        Forecast revenue for future periods
        
        Args:
            days_ahead (int): Number of days to forecast
        
        Returns:
            dict: Revenue forecast results
        """
        logger.info(f"Revenue forecast requested for {days_ahead} days ahead")
        
        # Query GL transactions (only positive amounts represent revenue)
        query = """
            SELECT * FROM dbo_F_GL_Transaction_Detail
            WHERE "Txn Amount" > 0
            """
        
        gl_transactions = self._execute_query(query)
        
        if len(gl_transactions) == 0:
            logger.warning("No revenue data found for forecasting")
            return {"error": "No revenue data found"}
        
        # Convert posting date to datetime
        gl_transactions['Posting Date'] = pd.to_datetime(gl_transactions['Posting Date'])
        
        # Aggregate by posting date to get daily revenue
        daily_revenue = gl_transactions.groupby('Posting Date')['Txn Amount'].sum().reset_index()
        logger.info(f"Aggregated {len(gl_transactions)} transactions into {len(daily_revenue)} daily revenue entries")
        
        if len(daily_revenue) < 30:
            logger.warning(f"Insufficient data for forecasting: only {len(daily_revenue)} days available")
            return {"error": "Insufficient data for forecasting (need at least 30 days)"}
        
        # Handle outliers (clip values above 3 std dev from mean to reduce their impact)
        revenue_mean = daily_revenue['Txn Amount'].mean()
        revenue_std = daily_revenue['Txn Amount'].std()
        outlier_threshold = revenue_mean + 3 * revenue_std
        
        original_max = daily_revenue['Txn Amount'].max()
        outlier_count = len(daily_revenue[daily_revenue['Txn Amount'] > outlier_threshold])
        
        if outlier_count > 0:
            logger.info(f"Detected {outlier_count} outliers above {outlier_threshold:.2f} (max value: {original_max:.2f})")
            daily_revenue.loc[daily_revenue['Txn Amount'] > outlier_threshold, 'Txn Amount'] = outlier_threshold
            logger.info(f"Clipped outliers to max value of {outlier_threshold:.2f}")
        
        # Create features based on the posting date
        daily_revenue['day_of_week'] = daily_revenue['Posting Date'].dt.dayofweek
        daily_revenue['month'] = daily_revenue['Posting Date'].dt.month
        daily_revenue['year'] = daily_revenue['Posting Date'].dt.year
        daily_revenue['day_of_month'] = daily_revenue['Posting Date'].dt.day
        daily_revenue['day_of_year'] = daily_revenue['Posting Date'].dt.dayofyear
        
        # Add lag features
        daily_revenue = daily_revenue.sort_values('Posting Date')
        daily_revenue['lag_1'] = daily_revenue['Txn Amount'].shift(1)
        daily_revenue['lag_7'] = daily_revenue['Txn Amount'].shift(7)
        daily_revenue['lag_14'] = daily_revenue['Txn Amount'].shift(14)
        
        # Drop rows with NaN values
        daily_revenue = daily_revenue.dropna()
        logger.info(f"Created features with lag values, {len(daily_revenue)} days after dropping NaN values")
        
        # Prepare data for training
        X = daily_revenue[['day_of_week', 'month', 'year', 'day_of_month', 'day_of_year', 'lag_1', 'lag_7', 'lag_14']]
        y = daily_revenue['Txn Amount']
        
        # Initialize models
        models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'xgb': XGBRegressor(n_estimators=100, random_state=42)
        }
        
        # Cross-validation
        from sklearn.model_selection import KFold
        from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
        
        n_splits = 5
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        cv_results = {model_name: {'mape': [], 'rmse': []} for model_name in models}
        
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            for model_name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mape = mean_absolute_percentage_error(y_test, y_pred) * 100
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                cv_results[model_name]['mape'].append(mape)
                cv_results[model_name]['rmse'].append(rmse)
        
        # Average CV results
        for model_name in models:
            cv_results[model_name]['mape'] = float(np.mean(cv_results[model_name]['mape']))
            cv_results[model_name]['rmse'] = float(np.mean(cv_results[model_name]['rmse']))
        
        logger.info(f"Cross-validation results: {cv_results}")
        
        # Train final models on all data
        for model_name, model in models.items():
            models[model_name].fit(X, y)
        
        # Get feature importance
        feature_importance = {}
        for model_name, model in models.items():
            if hasattr(model, 'feature_importances_'):
                importances = {}
                for i, feature in enumerate(X.columns):
                    # Convert numpy float32 to Python float
                    importances[feature] = float(model.feature_importances_[i])
                feature_importance[model_name] = importances
        
        # Calculate accuracy metrics based on the in-sample performance
        y_pred_rf = models['rf'].predict(X)
        mape = mean_absolute_percentage_error(y, y_pred_rf) * 100
        rmse = np.sqrt(mean_squared_error(y, y_pred_rf))
        
        logger.info(f"Final model accuracy - MAPE: {mape:.2f}%, RMSE: {rmse:.2f}")
        
        # Generate future dates for prediction
        last_date = daily_revenue['Posting Date'].max()
        future_dates = [last_date + pd.DateOffset(days=i) for i in range(1, days_ahead + 1)]
        
        # Create future features
        future_data = []
        for i, date in enumerate(future_dates):
            # For the very first prediction, use actual values from the data
            if i == 0:
                lag_1 = daily_revenue.iloc[-1]['Txn Amount']
                lag_7 = daily_revenue.iloc[-7]['Txn Amount'] if len(daily_revenue) > 7 else daily_revenue['Txn Amount'].mean()
                lag_14 = daily_revenue.iloc[-14]['Txn Amount'] if len(daily_revenue) > 14 else daily_revenue['Txn Amount'].mean()
            else:
                # For subsequent predictions, use predicted values
                lag_1 = future_data[i-1]['predicted']
                lag_7 = daily_revenue.iloc[-7+i]['Txn Amount'] if i < 7 else future_data[i-7]['predicted']
                lag_14 = daily_revenue.iloc[-14+i]['Txn Amount'] if i < 14 else future_data[i-14]['predicted']
            
            future_data.append({
                'Posting Date': date,
                'day_of_week': date.dayofweek,
                'month': date.month,
                'year': date.year,
                'day_of_month': date.day,
                'day_of_year': date.dayofyear,
                'lag_1': lag_1,
                'lag_7': lag_7,
                'lag_14': lag_14,
                'predicted': 0.0  # Initialize with 0.0
            })
        
        # Convert to DataFrame
        future_df = pd.DataFrame(future_data)
        
        # Make predictions
        X_future = future_df[['day_of_week', 'month', 'year', 'day_of_month', 'day_of_year', 'lag_1', 'lag_7', 'lag_14']]
        
        predictions = {}
        for model_name, model in models.items():
            predictions[model_name] = model.predict(X_future)
        
        # Ensemble predictions (average of models)
        future_df['predicted'] = 0
        for model_name in models:
            # Convert numpy float32 to Python float for each prediction
            future_df['predicted'] += np.array([float(x) for x in predictions[model_name]]) / len(models)
        
        # Generate visualization
        # Combine historical data with forecast
        historical = daily_revenue[['Posting Date', 'Txn Amount']].rename(columns={'Txn Amount': 'actual'})
        forecast = future_df[['Posting Date', 'predicted']]
        
        # Create fitted values for historical data
        historical['fitted'] = np.nan
        for model_name, model in models.items():
            historical['fitted'] += model.predict(X) / len(models)
        
        fig = go.Figure()
        
        # Plot historical data
        fig.add_trace(go.Scatter(
            x=historical['Posting Date'],
            y=historical['actual'],
            mode='lines',
            name='Actual Revenue'
        ))
        
        # Plot fitted values
        fig.add_trace(go.Scatter(
            x=historical['Posting Date'],
            y=historical['fitted'],
            mode='lines',
            name='Fitted Values',
            line=dict(dash='dot')
        ))
        
        # Plot forecast
        fig.add_trace(go.Scatter(
            x=forecast['Posting Date'],
            y=forecast['predicted'],
            mode='lines',
            name='Forecast',
            line=dict(dash='dash')
        ))
        
        fig.update_layout(title='Revenue Forecast', showlegend=True)
        
        # Save visualization
        viz_path = str(self.reports_path / f'revenue_forecast_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html')
        fig.write_html(viz_path)
        
        # Prepare forecast results
        forecast_data = []
        for _, row in future_df.iterrows():
            forecast_data.append({
                'Posting Date': row['Posting Date'].strftime('%Y-%m-%d'),
                'predicted': float(row['predicted'])
            })
        
        return {
            'forecast': forecast_data,
            'forecast_total': float(future_df['predicted'].sum()),
            'accuracy_metrics': {
                'mape': float(mape),
                'rmse': float(rmse)
            },
            'model_importance': feature_importance,
            'visualization_path': viz_path
        }
    
    def profitability_analysis(self, dimension=None):
        """
        Analyze profitability by various dimensions
        
        Args:
            dimension (str, optional): Dimension to analyze by (e.g., 'Company Code', 'Business Unit Key')
        
        Returns:
            dict: Profitability analysis results
        """
        # Query GL transactions
        query = """
        SELECT * FROM dbo_F_GL_Transaction_Detail
        """
        
        gl_transactions = self._execute_query(query)
        
        if len(gl_transactions) == 0:
            return {"error": "No data found"}
        
        # Calculate overall profitability
        profitability = {
            'overall': {
                'revenue': float(gl_transactions[gl_transactions['Txn Amount'] > 0]['Txn Amount'].sum()),
                'expenses': float(gl_transactions[gl_transactions['Txn Amount'] < 0]['Txn Amount'].sum()),
                'profit': float(gl_transactions['Txn Amount'].sum())
            }
        }
        
        # Calculate profit margin
        if profitability['overall']['revenue'] != 0:
            profitability['overall']['profit_margin'] = profitability['overall']['profit'] / profitability['overall']['revenue']
        else:
            profitability['overall']['profit_margin'] = 0
        
        # Analyze by dimension if provided
        if dimension and dimension in gl_transactions.columns:
            dim_profit = gl_transactions.groupby(dimension).agg({
                'Txn Amount': [
                    ('revenue', lambda x: x[x > 0].sum()),
                    ('expenses', lambda x: x[x < 0].sum()),
                    ('profit', 'sum')
                ]
            })
            
            # Flatten MultiIndex columns
            dim_profit.columns = [col[1] for col in dim_profit.columns]
            
            # Calculate profit margin
            dim_profit['profit_margin'] = dim_profit['profit'] / dim_profit['revenue'].replace(0, float('nan'))
            dim_profit['profit_margin'] = dim_profit['profit_margin'].fillna(0)
            
            profitability[f'by_{dimension}'] = dim_profit.to_dict(orient='index')
            
            # Generate visualization
            if len(dim_profit) <= 20:  # Only visualize if we don't have too many categories
                fig = px.bar(
                    dim_profit.reset_index(),
                    x=dimension,
                    y=['revenue', 'expenses', 'profit'],
                    barmode='group',
                    title=f'Profitability by {dimension}'
                )
                
                # Save visualization
                viz_path = str(self.reports_path / f'profitability_{dimension}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html')
                fig.write_html(viz_path)
                profitability['visualization_path'] = viz_path
        
        return profitability
    
    def customer_analysis(self, top_n=10):
        """
        Analyze customer metrics
        
        Args:
            top_n (int): Number of top customers to analyze
        
        Returns:
            dict: Customer analysis results
        """
        # First, check the available columns in the D_Customer table
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(dbo_D_Customer)")
            columns = cursor.fetchall()
            customer_columns = [col[1] for col in columns]
            logger.info(f"Available columns in dbo_D_Customer: {customer_columns}")
            conn.close()
        except Exception as e:
            logger.error(f"Error checking columns in dbo_D_Customer: {str(e)}")
            customer_columns = ["Customer Key", "Customer Name"]  # Default if we can't check
        
        # Query AR data with only columns we know exist
        query = """
        SELECT 
            h.*,
            c."Customer Name" as CustomerName
        FROM dbo_F_AR_Header h
        LEFT JOIN dbo_D_Customer c ON h."Customer Key" = c."Customer Key"
        """
        
        ar_data = self._execute_query(query)
        
        if len(ar_data) == 0:
            return {"error": "No customer data found"}
        
        # Calculate customer metrics
        customer_metrics = ar_data.groupby('Customer Key').agg({
            'Invoice Amount': ['sum', 'count'],
            'Balance Due Amount': ['sum', 'mean']
        })
        
        # Flatten MultiIndex columns
        customer_metrics.columns = [f"{col[0]}_{col[1]}" for col in customer_metrics.columns]
        
        # Rename columns for clarity
        customer_metrics = customer_metrics.rename(columns={
            'Invoice Amount_sum': 'total_revenue',
            'Invoice Amount_count': 'transaction_count',
            'Balance Due Amount_sum': 'total_outstanding',
            'Balance Due Amount_mean': 'average_outstanding'
        })
        
        # Calculate additional metrics
        customer_metrics['outstanding_ratio'] = customer_metrics['total_outstanding'] / customer_metrics['total_revenue']
        customer_metrics = customer_metrics.fillna(0)
        
        # Identify top customers by revenue
        top_customers = customer_metrics.sort_values('total_revenue', ascending=False).head(top_n)
        
        # Generate visualization
        customer_names = ar_data.groupby('Customer Key')['CustomerName'].first()
        
        # Map customer names if available
        if not customer_names.empty:
            top_customers_with_names = top_customers.join(customer_names)
            
            fig = px.bar(
                top_customers_with_names.reset_index(),
                x='CustomerName',
                y='total_revenue',
                title=f'Top {top_n} Customers by Revenue'
            )
        else:
            fig = px.bar(
                top_customers.reset_index(),
                x='Customer Key',
                y='total_revenue',
                title=f'Top {top_n} Customers by Revenue'
            )
        
        # Save visualization
        viz_path = str(self.reports_path / f'customer_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html')
        fig.write_html(viz_path)
        
        # Convert to JSON-serializable format
        top_customers_dict = {}
        for customer_key, row in top_customers.iterrows():
            customer_name = customer_names.get(customer_key, "Unknown") if not customer_names.empty else "Unknown"
            top_customers_dict[str(customer_key)] = {
                'customer_name': customer_name,
                'total_revenue': float(row['total_revenue']),
                'transaction_count': int(row['transaction_count']),
                'total_outstanding': float(row['total_outstanding']),
                'average_outstanding': float(row['average_outstanding']),
                'outstanding_ratio': float(row['outstanding_ratio'])
            }
        
        return {
            'top_customers': top_customers_dict,
            'customer_count': len(customer_metrics),
            'total_customer_revenue': float(customer_metrics['total_revenue'].sum()),
            'total_outstanding': float(customer_metrics['total_outstanding'].sum()),
            'visualization_path': viz_path
        } 