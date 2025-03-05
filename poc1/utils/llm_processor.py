import os
import json
import requests
from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd
import logging
import re

logger = logging.getLogger("LLMProcessor")

class LLMProcessor:
    """Process natural language queries using an LLM and select appropriate tools"""
    
    def __init__(self, provider="openai"):
        """Initialize the LLM processor"""
        self.provider = "openai"  # Always use OpenAI regardless of provider parameter
        self.api_key = self._get_api_key()
        self.tools_schema = self._load_tools_schema()
        self.conversation_history = []
        
        # Add database schema information to help with queries
        self._add_schema_information()
    
    def _get_api_key(self) -> str:
        """Get the API key for the LLM provider"""
        return os.environ.get("OPENAI_API_KEY", "")
    
    def _add_schema_information(self):
        """Add database schema information to the conversation history"""
        system_message = {
            "role": "system", 
            "content": """
            You are a financial analysis assistant that can query and analyze financial data.
            
            IMPORTANT - Here are the actual table names and columns in the database (use these exact names in SQL queries):
            
            1. dbo_F_GL_Transaction_Detail - Contains GL transactions
               Columns:
               - "GL Agg Detail Key" (INT, PRIMARY KEY): Unique identifier for each GL transaction
               - "Business Unit Key" (INT): Business unit identifier
               - "Company Key" (INT): Company identifier
               - "Company Code" (TEXT): Company code (e.g. 'US', 'CA', 'MX')
               - "Posting Date" (DATE): Date when transaction was posted, format 'YYYY-MM-DD'
               - "Txn Amount" (DECIMAL): Transaction amount (positive = revenue, negative = expense)
               - "Currency Code" (TEXT): Currency of transaction
               - "Department Key" (INT): Department identifier
               - "Customer Key" (INT): Reference to customer in dbo_D_Customer table
            
            2. dbo_F_AR_Header - Contains accounts receivable header data
               Columns:
               - "AR Header Key" (INT, PRIMARY KEY): Unique identifier for AR invoice
               - "Customer Key" (INT): Reference to customer in dbo_D_Customer table
               - "Posting Date" (DATE): Date when invoice was posted, format 'YYYY-MM-DD'
               - "Due Date" (DATE): Date when payment is due
               - "Invoice Amount" (DECIMAL): Total invoice amount
               - "Balance Due Amount" (DECIMAL): Amount still owed on invoice
               - "Document Status" (TEXT): Status of invoice (e.g. 'OPEN', 'PAID')
            
            3. dbo_F_AR_Detail - Contains accounts receivable detail data
               Columns:
               - "AR Detail Key" (INT, PRIMARY KEY): Unique identifier for AR line item
               - "AR Header Key" (INT): Reference to header in dbo_F_AR_Header
               - "Line Amount" (DECIMAL): Amount for this line item
               - "Item Description" (TEXT): Description of the item
            
            4. dbo_D_Customer - Contains customer information
               Columns:
               - "Customer Key" (INT, PRIMARY KEY): Unique identifier for customer
               - "Customer Name" (TEXT): Name of the customer
               - "Customer Type" (TEXT): Type of customer
               - "Country" (TEXT): Country where customer is located
            
            5. dbo_F_GL_Forecast - Contains forecast data
               Columns:
               - "Forecast Key" (INT, PRIMARY KEY): Unique identifier for forecast record
               - "Forecast Date" (DATE): Date of forecast
               - "Business Unit Key" (INT): Business unit identifier
               - "Forecast Amount" (DECIMAL): Forecasted amount
               - "Actual Amount" (DECIMAL): Actual amount (if available)
            
            IMPORTANT SYNTAX NOTES:
            - This is a SQLite database, so use SQLite syntax
            - Use LIMIT instead of TOP (e.g., "SELECT * FROM table LIMIT 10" not "SELECT TOP 10 * FROM table")
            - Use double-quotes for column names with spaces (e.g., "Customer Name")
            - Date operations use date functions like DATE(), strftime()
            - String concatenation uses the || operator, not +
            - JOIN syntax is standard (INNER JOIN, LEFT JOIN, etc.)
            
            NEVER use generic table names like "transactions" or "customers" - they don't exist in the database.
            Instead, always use the specific table names listed above in your SQL queries.
            
            When working with dates, use the "Posting Date" field in the format 'YYYY-MM-DD'.
            """
        }
        self.conversation_history.append(system_message)
    
    def _load_tools_schema(self) -> List[Dict]:
        """Load the schema for available financial analysis tools"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "cash_flow_analysis",
                    "description": "Analyze cash flow patterns and trends",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "start_date": {
                                "type": "string",
                                "description": "Start date in YYYY-MM-DD format"
                            },
                            "end_date": {
                                "type": "string",
                                "description": "End date in YYYY-MM-DD format"
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "ar_aging_analysis",
                    "description": "Analyze accounts receivable aging and payment patterns",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "revenue_forecast",
                    "description": "Forecast revenue using machine learning models",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "days_ahead": {
                                "type": "integer",
                                "description": "Number of days to forecast ahead"
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "profitability_analysis",
                    "description": "Analyze profitability by various dimensions",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "dimension": {
                                "type": "string",
                                "description": "Dimension to analyze by (e.g., 'Company Code', 'Business Unit Key')"
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "customer_analysis",
                    "description": "Analyze customer metrics",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "top_n": {
                                "type": "integer",
                                "description": "Number of top customers to analyze"
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "query_database",
                    "description": "Run a custom SQL query on the financial database",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sql_query": {
                                "type": "string",
                                "description": "SQL query to execute (always use actual table names like dbo_F_GL_Transaction_Detail, not generic names)"
                            }
                        },
                        "required": ["sql_query"]
                    }
                }
            }
        ]
    
    def process_query(self, query: str) -> Dict:
        """
        Process a natural language query and determine which tools to use
        
        Args:
            query (str): User's natural language query
            
        Returns:
            Dict: LLM response with tool calls
        """
        # Add user query to conversation history
        self.conversation_history.append({"role": "user", "content": query})
        
        # Process with OpenAI
        return self._process_with_openai(query)
    
    def _process_with_openai(self, query: str) -> Dict:
        """Process query with OpenAI"""
        url = "https://api.openai.com/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-4o",
            "messages": self.conversation_history,
            "tools": self.tools_schema,
            "temperature": 0
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            
            # Add assistant response to conversation history
            assistant_message = result["choices"][0]["message"]
            self.conversation_history.append(assistant_message)
            
            return {
                "id": result["id"],
                "content": assistant_message.get("content", ""),
                "tool_calls": assistant_message.get("tool_calls", [])
            }
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            return {"error": str(e), "content": "I'm sorry, I encountered an error while processing your request."}
    
    def execute_tool_calls(self, tool_calls: List[Dict], tools_instance: Any) -> Dict:
        """
        Execute the tool calls identified by the LLM
        
        Args:
            tool_calls (List[Dict]): List of tool calls from the LLM
            tools_instance (Any): Instance of the financial tools class
            
        Returns:
            Dict: Results from executing the tools
        """
        results = {}
        
        for tool_call in tool_calls:
            function_name = tool_call.get("function", {}).get("name")
            function_args = tool_call.get("function", {}).get("arguments", "{}")
            tool_id = tool_call.get("id", "")
            
            try:
                # Parse arguments
                args = json.loads(function_args)
                
                # Log the function call for debugging
                logger.info(f"Executing {function_name} with args: {args}")
                
                # Execute the appropriate tool
                if function_name == "cash_flow_analysis":
                    tool_result = tools_instance.cash_flow_analysis(**args)
                elif function_name == "ar_aging_analysis":
                    tool_result = tools_instance.ar_aging_analysis()
                elif function_name == "revenue_forecast":
                    tool_result = tools_instance.revenue_forecast(**args)
                elif function_name == "profitability_analysis":
                    tool_result = tools_instance.profitability_analysis(**args)
                elif function_name == "customer_analysis":
                    tool_result = tools_instance.customer_analysis(**args)
                elif function_name == "query_database":
                    # For security, we would normally validate custom SQL queries
                    # Sanitize the SQL query for SQLite compatibility
                    sanitized_query = self._sanitize_sql_query(args["sql_query"])
                    logger.info(f"Executing SQL query: {sanitized_query}")
                    conn = tools_instance._get_connection()
                    df = pd.read_sql_query(sanitized_query, conn)
                    conn.close()
                    tool_result = df.to_dict(orient='records')
                else:
                    tool_result = {"error": f"Unknown tool: {function_name}"}
                
                # Add tool results to conversation history
                tool_result_str = self._format_tool_result_for_context(function_name, tool_result)
                self.conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "name": function_name,
                    "content": tool_result_str
                })
                
                # Save results for return
                results[function_name] = tool_result
                    
            except Exception as e:
                logger.error(f"Error executing {function_name}: {str(e)}")
                results[function_name] = {"error": str(e)}
                
                # Add error to conversation history
                self.conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "name": function_name,
                    "content": f"Error: {str(e)}"
                })
        
        return results
    
    def _format_tool_result_for_context(self, function_name: str, result: Any) -> str:
        """Format tool results for inclusion in conversation context"""
        try:
            # Convert complex results to a readable summary string
            if isinstance(result, dict):
                if "error" in result:
                    return f"Error: {result['error']}"
                
                # Create a simplified summary based on tool type
                if function_name == "cash_flow_analysis":
                    metrics = result.get("metrics", {})
                    anomalies = result.get("anomalies", [])
                    return (
                        f"Cash Flow Analysis: Total Inflow: ${metrics.get('total_inflow', 0):,.2f}, "
                        f"Total Outflow: ${metrics.get('total_outflow', 0):,.2f}, "
                        f"Net Cash Flow: ${metrics.get('net_cash_flow', 0):,.2f}, "
                        f"Anomalies detected: {len(anomalies)}"
                    )
                
                elif function_name == "ar_aging_analysis":
                    return (
                        f"AR Aging Analysis: Days Sales Outstanding (DSO): {result.get('dso', 0):.1f} days, "
                        f"Total Overdue: ${result.get('total_overdue', 0):,.2f}, "
                        f"Severely Overdue (90+ days): ${result.get('total_severely_overdue', 0):,.2f}"
                    )
                
                elif function_name == "revenue_forecast":
                    metrics = result.get("accuracy_metrics", {})
                    return (
                        f"Revenue Forecast: Forecast Total (Next Period): ${result.get('forecast_total', 0):,.2f}, "
                        f"MAPE: {metrics.get('mape', 0):.2f}%, RMSE: ${metrics.get('rmse', 0):,.2f}"
                    )
                
                elif function_name == "profitability_analysis":
                    overall = result.get("overall", {})
                    return (
                        f"Profitability Analysis: Total Revenue: ${overall.get('revenue', 0):,.2f}, "
                        f"Total Expenses: ${overall.get('expenses', 0):,.2f}, "
                        f"Net Profit: ${overall.get('profit', 0):,.2f}, "
                        f"Profit Margin: {overall.get('profit_margin', 0) * 100:.2f}%"
                    )
                
                elif function_name == "customer_analysis":
                    return (
                        f"Customer Analysis: Total Customers: {result.get('customer_count', 0)}, "
                        f"Total Customer Revenue: ${result.get('total_customer_revenue', 0):,.2f}, "
                        f"Total Outstanding: ${result.get('total_outstanding', 0):,.2f}"
                    )
                
                # Default formatting for other dictionary results
                return json.dumps(result, default=str)
            
            elif isinstance(result, list):
                # For database query results
                summary = f"Query returned {len(result)} records. "
                if len(result) > 0:
                    first_item = result[0]
                    columns = ", ".join(first_item.keys())
                    summary += f"Columns: {columns}"
                return summary
            
            else:
                return str(result)
                
        except Exception as e:
            logger.error(f"Error formatting tool result: {str(e)}")
            return f"Error formatting result: {str(e)}"
    
    def _sanitize_sql_query(self, query):
        """
        Sanitize SQL query for SQLite compatibility
        
        Args:
            query (str): Original SQL query
            
        Returns:
            str: Sanitized SQL query
        """
        # Replace TOP n with LIMIT n
        # Log the original query
        logger.info(f"Original SQL query: {query}")
        
        # Replace TOP n with LIMIT n - place at the end of the query
        top_pattern = re.compile(r'SELECT\s+TOP\s+(\d+)', re.IGNORECASE)
        match = top_pattern.search(query)
        if match:
            limit_value = match.group(1)
            new_query = top_pattern.sub('SELECT', query)
            
            # Add LIMIT at the end if not already present
            if not re.search(r'LIMIT\s+\d+', new_query, re.IGNORECASE):
                # Check if query ends with semicolon
                if new_query.rstrip().endswith(';'):
                    new_query = new_query.rstrip()[:-1] + f" LIMIT {limit_value};"
                else:
                    new_query = new_query.rstrip() + f" LIMIT {limit_value}"
                
            logger.info(f"Modified SQL query: {new_query}")
            return new_query
        
        return query 