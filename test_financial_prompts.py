#!/usr/bin/env python
# Test script for financial analysis prompts

import os
import sys
import json
from pathlib import Path
import pandas as pd
import logging
import sqlite3

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_prompts.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TestPrompts")

# Add parent directory to path for imports
parent_dir = Path(__file__).resolve().parent
poc1_dir = parent_dir / "poc1"
sys.path.append(str(poc1_dir))

try:
    from poc1.tools.financial_tools import FinancialTools
    from poc1.utils.llm_processor import LLMProcessor
    logger.info("Successfully imported required modules")
except Exception as e:
    logger.error(f"Error importing modules: {str(e)}")
    sys.exit(1)

def test_cash_flow_analysis():
    logger.info("==== TESTING CASH FLOW ANALYSIS PROMPTS ====")
    try:
        # Initialize financial tools
        db_path = poc1_dir / "data" / "financial.db"
        logger.info(f"Using database at: {db_path}")
        financial_tools = FinancialTools(db_path=str(db_path))
        
        # 1. "Analyze our cash flow trends over the past year"
        # Need to use dates in the available data range (2019-01-31 to 2021-06-30)
        logger.info("PROMPT: Analyze our cash flow trends over the past year")
        result = financial_tools.cash_flow_analysis(
            start_date="2020-06-30", 
            end_date="2021-06-30"
        )
        
        # Test JSON serializability
        try:
            json_result = json.dumps(result)
            logger.info("✅ Cash flow result is JSON serializable")
            
            # Log a sample of the result structure
            if "metrics" in result:
                logger.info(f"Cash flow metrics: {result['metrics']}")
            if "seasonal_patterns" in result:
                logger.info(f"Seasonal patterns sample: {list(result['seasonal_patterns'].items())[:2]}")
            if "anomalies" in result:
                logger.info(f"Found {len(result['anomalies'])} anomalies")
                
        except TypeError as e:
            logger.error(f"❌ Cash flow result is NOT JSON serializable: {str(e)}")
            return False

        # 2. "Are there any unusual patterns or anomalies in our recent cash flow?"
        logger.info("PROMPT: Are there any unusual patterns or anomalies in our recent cash flow?")
        result = financial_tools.cash_flow_analysis(
            start_date="2021-01-01", 
            end_date="2021-06-30"
        )
        if "anomalies" in result and result["anomalies"]:
            logger.info(f"Found {len(result['anomalies'])} anomalies")
        else:
            logger.info("No anomalies found or error occurred")

        # 3. "What was our net cash position at the end of last quarter?"
        logger.info("PROMPT: What was our net cash position at the end of last quarter?")
        # Using Q1 2021 as "last quarter" for the test
        result = financial_tools.cash_flow_analysis(
            start_date="2021-01-01", 
            end_date="2021-03-31"
        )
        if "metrics" in result and "net_cash_flow" in result["metrics"]:
            logger.info(f"Net cash position: ${result['metrics']['net_cash_flow']:,.2f}")
        else:
            logger.info(f"Error or no data: {result}")

        # 4. "Show me the daily cash flow volatility"
        logger.info("PROMPT: Show me the daily cash flow volatility")
        result = financial_tools.cash_flow_analysis()  # Using default date range
        if "metrics" in result and "daily_volatility" in result["metrics"]:
            logger.info(f"Daily volatility: ${result['metrics']['daily_volatility']:,.2f}")
        else:
            logger.info(f"Error or no data: {result}")

        return True
    except Exception as e:
        logger.error(f"Error in cash flow analysis tests: {str(e)}")
        return False

def test_ar_aging_analysis():
    logger.info("==== TESTING ACCOUNTS RECEIVABLE PROMPTS ====")
    try:
        # Initialize financial tools
        db_path = poc1_dir / "data" / "financial.db"
        financial_tools = FinancialTools(db_path=str(db_path))
        
        # 1. "What is our current accounts receivable aging?"
        logger.info("PROMPT: What is our current accounts receivable aging?")
        result = financial_tools.ar_aging_analysis()
        
        # Test JSON serializability
        try:
            json_result = json.dumps(result)
            logger.info("✅ AR aging result is JSON serializable")
        except TypeError as e:
            logger.error(f"❌ AR aging result is NOT JSON serializable: {str(e)}")
            return False
            
        logger.info(f"AR aging buckets: {list(result['aging_buckets'].keys())}")

        # 2. "How much is currently overdue beyond 90 days?"
        logger.info("PROMPT: How much is currently overdue beyond 90 days?")
        result = financial_tools.ar_aging_analysis()
        if "total_severely_overdue" in result:
            logger.info(f"Total severely overdue: ${result['total_severely_overdue']:,.2f}")
        else:
            logger.info(f"Error or no data: {result}")

        # 3. "What is our Days Sales Outstanding (DSO)?"
        logger.info("PROMPT: What is our Days Sales Outstanding (DSO)?")
        result = financial_tools.ar_aging_analysis()
        if "dso" in result:
            logger.info(f"DSO: {result['dso']:.2f} days")
        else:
            logger.info(f"Error or no data: {result}")

        return True
    except Exception as e:
        logger.error(f"Error in AR aging analysis tests: {str(e)}")
        return False

def test_revenue_forecast():
    logger.info("==== TESTING REVENUE FORECASTING PROMPTS ====")
    try:
        # Initialize financial tools
        db_path = poc1_dir / "data" / "financial.db"
        financial_tools = FinancialTools(db_path=str(db_path))
        
        # 1. "Forecast our revenue for the next 30 days"
        logger.info("PROMPT: Forecast our revenue for the next 30 days")
        result = financial_tools.revenue_forecast(days_ahead=30)
        
        # Test JSON serializability
        try:
            json_result = json.dumps(result)
            logger.info("✅ Revenue forecast result is JSON serializable")
        except TypeError as e:
            logger.error(f"❌ Revenue forecast result is NOT JSON serializable: {str(e)}")
            return False
        
        if "forecast_total" in result and "accuracy_metrics" in result:
            logger.info(f"Forecast total: ${result['forecast_total']:,.2f}")
            logger.info(f"MAPE: {result['accuracy_metrics']['mape']:.2f}%")
            logger.info(f"RMSE: ${result['accuracy_metrics']['rmse']:,.2f}")
        else:
            logger.info(f"Error or no data: {result}")
            
        # 2. "What factors are most important in predicting our revenue?"
        logger.info("PROMPT: What factors are most important in predicting our revenue?")
        if "model_importance" in result:
            for model, importance in result["model_importance"].items():
                logger.info(f"Model {model} feature importance:")
                sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                for feature, value in sorted_importance:
                    logger.info(f"  {feature}: {value:.4f}")
        else:
            logger.info(f"Error or no data: {result}")

        return True
    except Exception as e:
        logger.error(f"Error in revenue forecast tests: {str(e)}")
        return False

def test_profitability_analysis():
    logger.info("==== TESTING PROFITABILITY ANALYSIS PROMPTS ====")
    try:
        # Initialize financial tools
        db_path = poc1_dir / "data" / "financial.db"
        financial_tools = FinancialTools(db_path=str(db_path))
        
        # 1. "What is our overall profit margin?"
        logger.info("PROMPT: What is our overall profit margin?")
        result = financial_tools.profitability_analysis()
        
        # Test JSON serializability
        try:
            json_result = json.dumps(result)
            logger.info("✅ Profitability analysis result is JSON serializable")
        except TypeError as e:
            logger.error(f"❌ Profitability analysis result is NOT JSON serializable: {str(e)}")
            return False
        
        if "overall" in result and "profit_margin" in result["overall"]:
            logger.info(f"Overall profit margin: {result['overall']['profit_margin']*100:.2f}%")
        else:
            logger.info(f"Error or no data: {result}")
            
        # 2. "Which business units are most profitable?"
        logger.info("PROMPT: Which business units are most profitable?")
        result = financial_tools.profitability_analysis(dimension="Business Unit Key")
        logger.info(f"Result structure: {list(result.keys())}")
        
        # 3. "Analyze profitability by company code"
        logger.info("PROMPT: Analyze profitability by company code")
        result = financial_tools.profitability_analysis(dimension="Company Code")
        logger.info(f"Result structure: {list(result.keys())}")

        return True
    except Exception as e:
        logger.error(f"Error in profitability analysis tests: {str(e)}")
        return False

def test_customer_analysis():
    logger.info("==== TESTING CUSTOMER ANALYSIS PROMPTS ====")
    try:
        # Initialize financial tools
        db_path = poc1_dir / "data" / "financial.db"
        financial_tools = FinancialTools(db_path=str(db_path))
        
        # 1. "Who are our top 10 customers by revenue?"
        logger.info("PROMPT: Who are our top 10 customers by revenue?")
        result = financial_tools.customer_analysis(top_n=10)
        
        # Test JSON serializability
        try:
            json_result = json.dumps(result)
            logger.info("✅ Customer analysis result is JSON serializable")
        except TypeError as e:
            logger.error(f"❌ Customer analysis result is NOT JSON serializable: {str(e)}")
            return False
        
        if "top_customers" in result:
            logger.info(f"Found {len(result['top_customers'])} top customers")
            # Log first customer details as sample
            first_customer = list(result['top_customers'].items())[0]
            logger.info(f"Sample customer: {first_customer}")
            logger.info(f"Total customer revenue: ${result['total_customer_revenue']:,.2f}")
        else:
            logger.info(f"Error or no data: {result}")

        return True
    except Exception as e:
        logger.error(f"Error in customer analysis tests: {str(e)}")
        return False

def test_custom_sql_queries():
    logger.info("==== TESTING CUSTOM SQL QUERIES ====")
    try:
        # Initialize financial tools
        db_path = poc1_dir / "data" / "financial.db"
        financial_tools = FinancialTools(db_path=str(db_path))
        
        # 1. "Run a query to show transaction amounts by month"
        logger.info("PROMPT: Run a query to show transaction amounts by month")
        query = """
        SELECT strftime('%Y-%m', "Posting Date") as Month, 
               SUM("Txn Amount") as TotalAmount 
        FROM dbo_F_GL_Transaction_Detail 
        GROUP BY Month 
        ORDER BY Month
        """
        try:
            result = financial_tools._execute_query(query)
            logger.info(f"Query returned {len(result)} rows")
            if not result.empty:
                logger.info(f"First few rows: \n{result.head().to_string()}")
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
        
        # 2. "Query for transactions over $10,000 in the last quarter"
        logger.info("PROMPT: Query for transactions over $10,000 in the last quarter")
        # Using Q1 2021 as "last quarter" for the test
        query = """
        SELECT * 
        FROM dbo_F_GL_Transaction_Detail 
        WHERE "Txn Amount" > 10000 
        AND "Posting Date" BETWEEN '2021-01-01' AND '2021-03-31'
        ORDER BY "Txn Amount" DESC
        """
        try:
            result = financial_tools._execute_query(query)
            logger.info(f"Query returned {len(result)} transactions over $10,000")
            if not result.empty:
                logger.info(f"First few transactions: \n{result.head().to_string()}")
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")

        return True
    except Exception as e:
        logger.error(f"Error in custom SQL query tests: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Starting financial prompts test after fixes")
    logger.info(f"Current working directory: {os.getcwd()}")
    
    # Database information
    db_path = poc1_dir / "data" / "financial.db"
    logger.info(f"Database path: {db_path}")
    logger.info(f"Database exists: {db_path.exists()}")
    
    if db_path.exists():
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            logger.info(f"Tables in database: {[t[0] for t in tables]}")
            conn.close()
        except Exception as e:
            logger.error(f"Error connecting to database: {str(e)}")
    
    test_results = {}
    test_results["cash_flow"] = test_cash_flow_analysis()
    test_results["ar_aging"] = test_ar_aging_analysis()
    test_results["revenue_forecast"] = test_revenue_forecast()
    test_results["profitability"] = test_profitability_analysis()
    test_results["customer_analysis"] = test_customer_analysis()
    test_results["custom_sql"] = test_custom_sql_queries()
    
    logger.info("==== TEST SUMMARY ====")
    for test, result in test_results.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"{test}: {status}")
        
    if all(test_results.values()):
        logger.info("All tests PASSED! ✅")
    else:
        logger.info("Some tests FAILED. ❌ Check the log for details.") 