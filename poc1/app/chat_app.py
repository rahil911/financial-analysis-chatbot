import streamlit as st
import sys
import os
import json
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import logging
import sqlite3
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chat_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ChatApp")

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
logger.info(f"Added parent directory to path: {parent_dir}")

try:
    from tools.financial_tools import FinancialTools
    from utils.llm_processor import LLMProcessor
    logger.info("Successfully imported required modules")
except Exception as e:
    logger.error(f"Error importing modules: {str(e)}")
    st.error(f"Failed to import required modules: {str(e)}")

# Page configuration
st.set_page_config(
    page_title="Financial Analysis Chatbot",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "llm_processor" not in st.session_state:
    st.session_state.llm_processor = None

if "financial_tools" not in st.session_state:
    st.session_state.financial_tools = None

if "debug_info" not in st.session_state:
    st.session_state.debug_info = []

def initialize_app():
    """Initialize the application components"""
    # Setup database path using absolute path
    app_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    project_root = app_dir.parent
    db_path = project_root / "data" / "financial.db"
    
    # Log the paths for debugging
    logger.info(f"App directory: {app_dir}")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Database path: {db_path}")
    
    # Add to debug info
    st.session_state.debug_info.append(f"App directory: {app_dir}")
    st.session_state.debug_info.append(f"Database path: {db_path}")
    
    if not db_path.exists():
        error_msg = f"Database not found at {db_path}. Please run the database setup script first."
        logger.error(error_msg)
        st.warning(error_msg)
        st.session_state.debug_info.append(f"ERROR: {error_msg}")
        
        # Try to find the database by scanning common locations
        possible_paths = [
            Path("poc1/data/financial.db"),
            Path("data/financial.db"),
            Path("../data/financial.db"),
            Path("/Users/rahilharihar/Projects/Bicycle/poc1/data/financial.db")
        ]
        
        for path in possible_paths:
            if path.exists():
                st.session_state.debug_info.append(f"Found database at: {path}")
                logger.info(f"Found database at alternate location: {path}")
                db_path = path
                st.success(f"Found database at alternate location: {path}")
                break
            else:
                st.session_state.debug_info.append(f"Checked path (not found): {path}")
    else:
        logger.info(f"Database found at {db_path}")
        st.session_state.debug_info.append(f"Database found at {db_path}")
    
    # Initialize financial tools
    try:
        st.session_state.financial_tools = FinancialTools(db_path=str(db_path))
        logger.info("Financial tools initialized successfully")
        
        # Verify database connection
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            conn.close()
            
            table_list = [t[0] for t in tables]
            logger.info(f"Database tables: {table_list}")
            st.session_state.debug_info.append(f"Database tables: {table_list}")
        except Exception as e:
            error_msg = f"Error connecting to database: {str(e)}"
            logger.error(error_msg)
            st.session_state.debug_info.append(f"ERROR: {error_msg}")
    except Exception as e:
        error_msg = f"Error initializing financial tools: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        st.session_state.debug_info.append(f"ERROR: {error_msg}")
    
    # Initialize LLM processor
    try:
        st.session_state.llm_processor = LLMProcessor(provider="openai")
        logger.info("LLM processor initialized successfully")
    except Exception as e:
        error_msg = f"Error initializing LLM processor: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        st.session_state.debug_info.append(f"ERROR: {error_msg}")
    
    # Display welcome message
    if not st.session_state.messages:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hello! I'm your financial analysis assistant. I can help you analyze cash flow, AR aging, forecast revenue, and more. What would you like to know about your financial data?"
        })

def display_visualizations(viz_paths):
    """Display visualizations from the tool results"""
    for viz_path in viz_paths:
        if os.path.exists(viz_path) and viz_path.endswith('.html'):
            try:
                # Read the HTML content
                with open(viz_path, 'r') as f:
                    html_content = f.read()
                
                # Display in Streamlit
                st.components.v1.html(html_content, height=500)
                logger.info(f"Successfully displayed visualization: {viz_path}")
            except Exception as e:
                error_msg = f"Error displaying visualization {viz_path}: {str(e)}"
                logger.error(error_msg)
                st.error(error_msg)
                st.session_state.debug_info.append(f"ERROR: {error_msg}")

def format_tool_results(results):
    """Format tool results for display"""
    formatted_results = {}
    viz_paths = []
    
    for tool_name, result in results.items():
        try:
            if isinstance(result, dict) and "error" in result:
                error_msg = f"Error in {tool_name}: {result['error']}"
                logger.error(error_msg)
                st.session_state.debug_info.append(f"ERROR: {error_msg}")
                formatted_results[tool_name] = f"Error: {result['error']}"
            elif isinstance(result, dict) and "visualization_path" in result:
                # Extract visualization path
                viz_paths.append(result["visualization_path"])
                
                # Format the numerical results
                if tool_name == "cash_flow_analysis":
                    metrics = result.get("metrics", {})
                    formatted_results[tool_name] = {
                        "Total Inflow": f"${metrics.get('total_inflow', 0):,.2f}",
                        "Total Outflow": f"${metrics.get('total_outflow', 0):,.2f}",
                        "Net Cash Flow": f"${metrics.get('net_cash_flow', 0):,.2f}",
                        "Daily Average": f"${metrics.get('daily_average', 0):,.2f}"
                    }
                    
                    if result.get("anomalies"):
                        formatted_results[tool_name]["Anomalies Detected"] = len(result["anomalies"])
                    
                elif tool_name == "ar_aging_analysis":
                    formatted_results[tool_name] = {
                        "Days Sales Outstanding (DSO)": f"{result.get('dso', 0):.1f} days",
                        "Total Overdue": f"${result.get('total_overdue', 0):,.2f}",
                        "Severely Overdue (90+ days)": f"${result.get('total_severely_overdue', 0):,.2f}"
                    }
                    
                elif tool_name == "revenue_forecast":
                    metrics = result.get("accuracy_metrics", {})
                    formatted_results[tool_name] = {
                        "Forecast Total (Next Period)": f"${result.get('forecast_total', 0):,.2f}",
                        "Mean Absolute Percentage Error": f"{metrics.get('mape', 0):.2f}%",
                        "Root Mean Square Error": f"${metrics.get('rmse', 0):,.2f}"
                    }
                    
                elif tool_name == "profitability_analysis":
                    overall = result.get("overall", {})
                    formatted_results[tool_name] = {
                        "Total Revenue": f"${overall.get('revenue', 0):,.2f}",
                        "Total Expenses": f"${overall.get('expenses', 0):,.2f}",
                        "Net Profit": f"${overall.get('profit', 0):,.2f}",
                        "Profit Margin": f"{overall.get('profit_margin', 0) * 100:.2f}%"
                    }
                    
                elif tool_name == "customer_analysis":
                    formatted_results[tool_name] = {
                        "Total Customers": result.get("customer_count", 0),
                        "Total Customer Revenue": f"${result.get('total_customer_revenue', 0):,.2f}",
                        "Total Outstanding": f"${result.get('total_outstanding', 0):,.2f}"
                    }
                    
            elif isinstance(result, list):
                # Handle query_database results (list of records)
                if len(result) > 0:
                    formatted_results[tool_name] = pd.DataFrame(result)
                else:
                    formatted_results[tool_name] = "No data returned from query"
        except Exception as e:
            error_msg = f"Error formatting results for {tool_name}: {str(e)}"
            logger.error(error_msg)
            st.session_state.debug_info.append(f"ERROR: {error_msg}")
            formatted_results[tool_name] = f"Error formatting results: {str(e)}"
    
    return formatted_results, viz_paths

def process_user_query(query):
    """Process a user query through the LLM and financial tools"""
    try:
        # Process with LLM
        llm_response = st.session_state.llm_processor.process_query(query)
        logger.info(f"Received response from LLM: {llm_response}")
        
        # Check for errors in LLM response
        if "error" in llm_response:
            error_msg = f"Error from LLM: {llm_response['error']}"
            logger.error(error_msg)
            st.session_state.debug_info.append(f"ERROR: {error_msg}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": llm_response.get("content", "I encountered an error processing your request.")
            })
            return
        
        # Check for tool calls
        tool_calls = llm_response.get("tool_calls", [])
        logger.info(f"Tool calls from LLM: {tool_calls}")
        
        # Execute tool calls
        results = {}
        if tool_calls:
            try:
                results = st.session_state.llm_processor.execute_tool_calls(
                    tool_calls, 
                    st.session_state.financial_tools
                )
                logger.info(f"Tool execution results: {results}")
                
                # Generate follow-up response based on tool results if needed
                if results and not llm_response.get("content"):
                    follow_up_response = generate_followup_response(results)
                    if follow_up_response:
                        llm_response["content"] = follow_up_response
            except Exception as e:
                error_msg = f"Error executing tools: {str(e)}"
                logger.error(error_msg)
                st.session_state.debug_info.append(f"ERROR: {error_msg}")
                results = {"error": str(e)}
        
        # Format results and get visualization paths
        formatted_results, viz_paths = format_tool_results(results)
        
        # Get assistant response text
        assistant_response = llm_response.get("content", "I'm sorry, I couldn't process your request.")
        
        # Add tool results to the message
        if formatted_results:
            st.session_state.messages.append({
                "role": "assistant",
                "content": assistant_response,
                "tool_results": formatted_results,
                "visualizations": viz_paths
            })
        else:
            st.session_state.messages.append({
                "role": "assistant",
                "content": assistant_response
            })
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        st.session_state.debug_info.append(f"ERROR: {error_msg}")
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"I'm sorry, I encountered an error: {str(e)}"
        })

def generate_followup_response(results):
    """Generate a follow-up response based on tool results"""
    try:
        # Create a summary message based on the results
        summary_parts = []
        
        for tool_name, result in results.items():
            if isinstance(result, dict) and "error" in result:
                summary_parts.append(f"I had an issue with the {tool_name}: {result['error']}")
                continue
                
            if tool_name == "cash_flow_analysis":
                metrics = result.get("metrics", {})
                anomalies = result.get("anomalies", [])
                summary_parts.append(
                    f"Based on the cash flow analysis, I found a net cash flow of ${metrics.get('net_cash_flow', 0):,.2f} "
                    f"with ${metrics.get('total_inflow', 0):,.2f} inflows and ${metrics.get('total_outflow', 0):,.2f} outflows. "
                    f"I detected {len(anomalies)} anomalies in the cash flow data."
                )
                
            elif tool_name == "ar_aging_analysis":
                summary_parts.append(
                    f"The accounts receivable analysis shows a Days Sales Outstanding (DSO) of {result.get('dso', 0):.1f} days, "
                    f"with ${result.get('total_overdue', 0):,.2f} in overdue invoices and "
                    f"${result.get('total_severely_overdue', 0):,.2f} severely overdue (90+ days)."
                )
                
            elif tool_name == "revenue_forecast":
                metrics = result.get("accuracy_metrics", {})
                summary_parts.append(
                    f"The revenue forecast for the next period is ${result.get('forecast_total', 0):,.2f}. "
                    f"The forecast model has a Mean Absolute Percentage Error of {metrics.get('mape', 0):.2f}% "
                    f"and a Root Mean Square Error of ${metrics.get('rmse', 0):,.2f}."
                )
                
            elif tool_name == "profitability_analysis":
                overall = result.get("overall", {})
                summary_parts.append(
                    f"The profitability analysis shows a profit margin of {overall.get('profit_margin', 0) * 100:.2f}% "
                    f"with total revenue of ${overall.get('revenue', 0):,.2f}, expenses of ${overall.get('expenses', 0):,.2f}, "
                    f"and a net profit of ${overall.get('profit', 0):,.2f}."
                )
                
            elif tool_name == "customer_analysis":
                summary_parts.append(
                    f"The customer analysis identified {result.get('customer_count', 0)} customers "
                    f"with a total revenue of ${result.get('total_customer_revenue', 0):,.2f} and "
                    f"${result.get('total_outstanding', 0):,.2f} in outstanding invoices."
                )
                
            elif tool_name == "query_database":
                if isinstance(result, list):
                    summary_parts.append(f"The database query returned {len(result)} records.")
                    
        if summary_parts:
            return " ".join(summary_parts) + " Is there anything specific about these results you'd like me to explain further?"
        
        return None
    except Exception as e:
        logger.error(f"Error generating follow-up response: {str(e)}")
        return "I've processed your request, but had some difficulty summarizing the results. Would you like more details on a specific aspect?"

def main():
    # Title
    st.title("💼 Financial Analysis Chatbot")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        initialize_app()
        
        st.subheader("About")
        st.markdown("""
        This chatbot uses machine learning and financial analysis tools to:
        - Analyze cash flow patterns
        - Review accounts receivable aging
        - Forecast revenue
        - Analyze profitability
        - Perform customer analysis
        
        Simply ask questions in natural language!
        """)
        
        if st.button("Clear Conversation"):
            st.session_state.messages = []
            st.rerun()
        
        # Debug expander
        with st.expander("Debug Information"):
            st.write("#### System Information")
            st.code(f"Python version: {sys.version}")
            st.code(f"Current working directory: {os.getcwd()}")
            
            # Add database schema information
            st.write("#### Database Tables")
            try:
                if st.session_state.financial_tools:
                    conn = st.session_state.financial_tools._get_connection()
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = cursor.fetchall()
                    
                    if tables:
                        for table in tables:
                            if table[0] != 'sqlite_sequence' and table[0] != 'table_mapping':
                                st.write(f"**{table[0]}**")
                                # Sample 5 rows from each table
                                try:
                                    sample_df = pd.read_sql_query(f"SELECT * FROM '{table[0]}' LIMIT 5", conn)
                                    st.dataframe(sample_df, use_container_width=True)
                                    st.code(f"Table columns: {', '.join(sample_df.columns)}")
                                except Exception as e:
                                    st.error(f"Error sampling table {table[0]}: {str(e)}")
                    conn.close()
            except Exception as e:
                st.error(f"Error fetching database schema: {str(e)}")
            
            st.write("#### Debug Log")
            for info in st.session_state.debug_info:
                st.code(info)
            
            if st.button("Clear Debug Info"):
                st.session_state.debug_info = []
                st.rerun()
    
    # Chat container
    chat_container = st.container()
    
    # Input container
    with st.container():
        user_input = st.chat_input("Ask about your financial data...")
        if user_input:
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Process the query
            with st.spinner("Analyzing financial data..."):
                process_user_query(user_input)
    
    # Display chat messages
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.markdown(message["content"])
                    
                    # Display tool results if available
                    if "tool_results" in message:
                        for tool_name, result in message["tool_results"].items():
                            with st.expander(f"{tool_name.replace('_', ' ').title()} Results", expanded=True):
                                if isinstance(result, dict):
                                    # For dictionary results, display as a table
                                    items = []
                                    for k, v in result.items():
                                        items.append({"Metric": k, "Value": v})
                                    st.table(pd.DataFrame(items))
                                elif isinstance(result, pd.DataFrame):
                                    # For DataFrame results, display as a table
                                    st.dataframe(result, use_container_width=True)
                                else:
                                    # For other results, display as text
                                    st.write(result)
                    
                    # Display visualizations if available
                    if "visualizations" in message:
                        display_visualizations(message["visualizations"])

if __name__ == "__main__":
    main() 