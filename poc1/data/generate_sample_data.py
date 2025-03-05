#!/usr/bin/env python
"""
Sample Database Generator for Financial Analysis Chatbot

This script generates sample financial data for the chatbot's database.
It creates a SQLite database with realistic but fictional financial transactions,
customers, invoices, and forecast data.

Usage:
    python generate_sample_data.py [--output financial.db] [--records 1000]
"""

import os
import sys
import sqlite3
import random
import argparse
import datetime
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('GenerateData')

# Get the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate sample financial data')
    parser.add_argument('--output', default='financial.db', help='Output database file')
    parser.add_argument('--records', type=int, default=1000, help='Number of GL records to generate')
    return parser.parse_args()

def create_database_schema(conn):
    """Create the database schema."""
    logger.info("Creating database schema...")
    
    # Create tables
    conn.executescript('''
    -- Create GL Transaction Detail table
    CREATE TABLE IF NOT EXISTS dbo_F_GL_Transaction_Detail (
        TransactionID INTEGER PRIMARY KEY,
        "Posting Date" TEXT,
        "Document Number" TEXT,
        "Account Number" TEXT,
        "Account Description" TEXT,
        "Department Code" TEXT,
        "Department Description" TEXT,
        "Txn Amount" REAL,
        "Currency Code" TEXT,
        "Txn Description" TEXT,
        "Customer ID" TEXT
    );

    -- Create AR Header table
    CREATE TABLE IF NOT EXISTS dbo_F_AR_Header (
        InvoiceID INTEGER PRIMARY KEY,
        "Posting Date" TEXT,
        "Document Number" TEXT, 
        "Customer ID" TEXT,
        "Total Amount" REAL,
        "Currency Code" TEXT,
        "Due Date" TEXT,
        "Status" TEXT
    );

    -- Create AR Detail table
    CREATE TABLE IF NOT EXISTS dbo_F_AR_Detail (
        LineID INTEGER PRIMARY KEY,
        InvoiceID INTEGER,
        "Posting Date" TEXT,
        "Item Description" TEXT,
        "Quantity" INTEGER,
        "Unit Price" REAL,
        "Line Amount" REAL,
        "Currency Code" TEXT,
        FOREIGN KEY (InvoiceID) REFERENCES dbo_F_AR_Header(InvoiceID)
    );

    -- Create Customer dimension table
    CREATE TABLE IF NOT EXISTS dbo_D_Customer (
        "Customer ID" TEXT PRIMARY KEY,
        "Customer Name" TEXT,
        "Customer Type" TEXT,
        "Industry" TEXT,
        "Region" TEXT,
        "Country" TEXT,
        "Credit Limit" REAL,
        "Status" TEXT
    );

    -- Create GL Forecast table
    CREATE TABLE IF NOT EXISTS dbo_F_GL_Forecast (
        ForecastID INTEGER PRIMARY KEY,
        "Forecast Date" TEXT,
        "Account Number" TEXT,
        "Account Description" TEXT,
        "Department Code" TEXT,
        "Forecast Amount" REAL,
        "Currency Code" TEXT
    );

    -- Create table_mapping table for metadata
    CREATE TABLE IF NOT EXISTS table_mapping (
        table_name TEXT,
        column_name TEXT,
        description TEXT,
        PRIMARY KEY (table_name, column_name)
    );
    ''')
    
    # Insert metadata
    conn.executescript('''
    INSERT OR REPLACE INTO table_mapping VALUES
        ('dbo_F_GL_Transaction_Detail', 'Posting Date', 'Date when the transaction was posted'),
        ('dbo_F_GL_Transaction_Detail', 'Document Number', 'Unique identifier for the transaction document'),
        ('dbo_F_GL_Transaction_Detail', 'Account Number', 'GL account number'),
        ('dbo_F_GL_Transaction_Detail', 'Account Description', 'Description of the GL account'),
        ('dbo_F_GL_Transaction_Detail', 'Department Code', 'Department code for the transaction'),
        ('dbo_F_GL_Transaction_Detail', 'Department Description', 'Description of the department'),
        ('dbo_F_GL_Transaction_Detail', 'Txn Amount', 'Amount of the transaction'),
        ('dbo_F_GL_Transaction_Detail', 'Currency Code', 'Currency code for the transaction'),
        ('dbo_F_GL_Transaction_Detail', 'Txn Description', 'Description of the transaction'),
        ('dbo_F_GL_Transaction_Detail', 'Customer ID', 'Customer ID associated with the transaction'),
        
        ('dbo_F_AR_Header', 'Posting Date', 'Date when the invoice was posted'),
        ('dbo_F_AR_Header', 'Document Number', 'Invoice number'),
        ('dbo_F_AR_Header', 'Customer ID', 'Customer ID associated with the invoice'),
        ('dbo_F_AR_Header', 'Total Amount', 'Total amount of the invoice'),
        ('dbo_F_AR_Header', 'Currency Code', 'Currency code for the invoice'),
        ('dbo_F_AR_Header', 'Due Date', 'Date when the invoice is due'),
        ('dbo_F_AR_Header', 'Status', 'Status of the invoice (Open, Paid, Overdue)'),
        
        ('dbo_F_AR_Detail', 'Posting Date', 'Date when the line item was posted'),
        ('dbo_F_AR_Detail', 'Item Description', 'Description of the line item'),
        ('dbo_F_AR_Detail', 'Quantity', 'Quantity of items'),
        ('dbo_F_AR_Detail', 'Unit Price', 'Price per unit'),
        ('dbo_F_AR_Detail', 'Line Amount', 'Total amount for the line item'),
        ('dbo_F_AR_Detail', 'Currency Code', 'Currency code for the line item'),
        
        ('dbo_D_Customer', 'Customer ID', 'Unique identifier for the customer'),
        ('dbo_D_Customer', 'Customer Name', 'Name of the customer'),
        ('dbo_D_Customer', 'Customer Type', 'Type of customer (Corporate, Individual, Government)'),
        ('dbo_D_Customer', 'Industry', 'Industry the customer belongs to'),
        ('dbo_D_Customer', 'Region', 'Geographical region of the customer'),
        ('dbo_D_Customer', 'Country', 'Country of the customer'),
        ('dbo_D_Customer', 'Credit Limit', 'Credit limit assigned to the customer'),
        ('dbo_D_Customer', 'Status', 'Status of the customer (Active, Inactive)'),
        
        ('dbo_F_GL_Forecast', 'Forecast Date', 'Date of the forecast'),
        ('dbo_F_GL_Forecast', 'Account Number', 'GL account number'),
        ('dbo_F_GL_Forecast', 'Account Description', 'Description of the GL account'),
        ('dbo_F_GL_Forecast', 'Department Code', 'Department code'),
        ('dbo_F_GL_Forecast', 'Forecast Amount', 'Forecasted amount'),
        ('dbo_F_GL_Forecast', 'Currency Code', 'Currency code for the forecast');
    ''')
    
    conn.commit()
    logger.info("Database schema created successfully")

def generate_customers(conn, num_customers=50):
    """Generate sample customer data."""
    logger.info(f"Generating {num_customers} customers...")
    
    customers = []
    customer_types = ['Corporate', 'Individual', 'Government']
    industries = ['Technology', 'Healthcare', 'Finance', 'Manufacturing', 'Retail', 'Energy', 'Education']
    regions = ['North', 'South', 'East', 'West', 'Central']
    countries = ['USA', 'Canada', 'UK', 'Germany', 'France', 'Japan', 'Australia', 'Brazil', 'China', 'India']
    statuses = ['Active', 'Inactive']
    
    for i in range(1, num_customers + 1):
        customer_id = f'CUST{i:05d}'
        customer_name = f'Customer {i}'
        customer_type = random.choice(customer_types)
        industry = random.choice(industries)
        region = random.choice(regions)
        country = random.choice(countries)
        credit_limit = random.randint(10000, 1000000)
        status = 'Active' if random.random() < 0.9 else 'Inactive'
        
        customers.append((
            customer_id, customer_name, customer_type, industry,
            region, country, credit_limit, status
        ))
    
    conn.executemany('''
    INSERT INTO dbo_D_Customer (
        "Customer ID", "Customer Name", "Customer Type", "Industry",
        "Region", "Country", "Credit Limit", "Status"
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', customers)
    
    conn.commit()
    logger.info(f"Generated {num_customers} customers")
    return [c[0] for c in customers]  # Return list of customer IDs

def generate_gl_transactions(conn, customer_ids, num_records=1000):
    """Generate sample GL transaction data."""
    logger.info(f"Generating {num_records} GL transactions...")
    
    transactions = []
    accounts = [
        ('1010', 'Cash'),
        ('1020', 'Accounts Receivable'),
        ('1030', 'Inventory'),
        ('1040', 'Prepaid Expenses'),
        ('2010', 'Accounts Payable'),
        ('3010', 'Common Stock'),
        ('4010', 'Sales Revenue'),
        ('5010', 'Cost of Goods Sold'),
        ('6010', 'Salaries Expense'),
        ('6020', 'Rent Expense')
    ]
    
    departments = [
        ('D001', 'Sales'),
        ('D002', 'Marketing'),
        ('D003', 'Operations'),
        ('D004', 'Finance'),
        ('D005', 'Human Resources'),
        ('D006', 'Information Technology'),
        ('D007', 'Customer Support')
    ]
    
    descriptions = [
        'Monthly subscription', 'Annual fee', 'Professional services',
        'Product purchase', 'Software license', 'Hardware purchase',
        'Consulting services', 'Training services', 'Maintenance fee'
    ]
    
    # Start date 6 months ago
    start_date = datetime.datetime.now() - datetime.timedelta(days=180)
    
    for i in range(1, num_records + 1):
        transaction_id = i
        
        # Generate date within last 6 months, more recent dates more likely
        days_ago = int(random.betavariate(1, 2) * 180)
        posting_date = (start_date + datetime.timedelta(days=days_ago)).strftime('%Y-%m-%d')
        
        document_number = f'DOC{i:06d}'
        
        account = random.choice(accounts)
        account_number, account_description = account
        
        department = random.choice(departments)
        department_code, department_description = department
        
        # Revenue accounts have positive amounts, expense accounts have negative
        if account_number.startswith('4'):  # Revenue
            txn_amount = random.uniform(100, 50000)
        elif account_number.startswith(('5', '6')):  # Expense
            txn_amount = -random.uniform(100, 30000)
        else:  # Other accounts - can be either
            txn_amount = random.uniform(-20000, 20000)
        
        currency_code = 'USD'
        txn_description = random.choice(descriptions)
        
        # Only assign customers to revenue transactions
        customer_id = random.choice(customer_ids) if account_number.startswith('4') else None
        
        transactions.append((
            transaction_id, posting_date, document_number,
            account_number, account_description,
            department_code, department_description,
            txn_amount, currency_code, txn_description, customer_id
        ))
    
    conn.executemany('''
    INSERT INTO dbo_F_GL_Transaction_Detail (
        TransactionID, "Posting Date", "Document Number",
        "Account Number", "Account Description",
        "Department Code", "Department Description",
        "Txn Amount", "Currency Code", "Txn Description", "Customer ID"
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', transactions)
    
    conn.commit()
    logger.info(f"Generated {num_records} GL transactions")

def generate_ar_data(conn, customer_ids, num_invoices=200):
    """Generate sample AR (Accounts Receivable) data."""
    logger.info(f"Generating {num_invoices} invoices...")
    
    # Start date 6 months ago
    start_date = datetime.datetime.now() - datetime.timedelta(days=180)
    
    invoices = []
    invoice_details = []
    status_options = ['Paid', 'Open', 'Overdue']
    status_weights = [0.6, 0.3, 0.1]  # 60% paid, 30% open, 10% overdue
    
    for invoice_id in range(1, num_invoices + 1):
        # Generate date within last 6 months
        days_ago = int(random.betavariate(1, 2) * 180)
        posting_date = (start_date + datetime.timedelta(days=days_ago)).strftime('%Y-%m-%d')
        
        document_number = f'INV{invoice_id:06d}'
        customer_id = random.choice(customer_ids)
        
        # Generate due date (15-45 days after posting)
        posting_date_obj = datetime.datetime.strptime(posting_date, '%Y-%m-%d')
        due_days = random.randint(15, 45)
        due_date = (posting_date_obj + datetime.timedelta(days=due_days)).strftime('%Y-%m-%d')
        
        # Determine status based on dates and probability
        current_date = datetime.datetime.now()
        if posting_date_obj + datetime.timedelta(days=due_days) < current_date:
            # Past due date - mostly paid, some overdue
            status = random.choices(['Paid', 'Overdue'], weights=[0.85, 0.15])[0]
        else:
            # Not yet due - mostly open, some paid early
            status = random.choices(['Open', 'Paid'], weights=[0.7, 0.3])[0]
        
        # Generate 1-5 line items per invoice
        num_lines = random.randint(1, 5)
        total_amount = 0
        
        for line_id in range(1, num_lines + 1):
            detail_id = (invoice_id - 1) * 5 + line_id
            quantity = random.randint(1, 10)
            unit_price = random.uniform(100, 1000)
            line_amount = quantity * unit_price
            total_amount += line_amount
            
            item_descriptions = [
                'Software license', 'Hardware component', 'Consulting services',
                'Training session', 'Support subscription', 'Custom development',
                'Cloud storage', 'Data processing', 'Professional services'
            ]
            item_description = random.choice(item_descriptions)
            
            invoice_details.append((
                detail_id, invoice_id, posting_date, item_description,
                quantity, unit_price, line_amount, 'USD'
            ))
        
        invoices.append((
            invoice_id, posting_date, document_number, customer_id,
            total_amount, 'USD', due_date, status
        ))
    
    # Insert AR header data
    conn.executemany('''
    INSERT INTO dbo_F_AR_Header (
        InvoiceID, "Posting Date", "Document Number", "Customer ID",
        "Total Amount", "Currency Code", "Due Date", "Status"
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', invoices)
    
    # Insert AR detail data
    conn.executemany('''
    INSERT INTO dbo_F_AR_Detail (
        LineID, InvoiceID, "Posting Date", "Item Description",
        "Quantity", "Unit Price", "Line Amount", "Currency Code"
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', invoice_details)
    
    conn.commit()
    logger.info(f"Generated {num_invoices} invoices with {len(invoice_details)} line items")

def generate_forecasts(conn, num_forecasts=100):
    """Generate sample forecast data."""
    logger.info(f"Generating {num_forecasts} forecast records...")
    
    forecasts = []
    accounts = [
        ('4010', 'Sales Revenue'),
        ('5010', 'Cost of Goods Sold'),
        ('6010', 'Salaries Expense'),
        ('6020', 'Rent Expense')
    ]
    
    departments = [
        ('D001', 'Sales'),
        ('D002', 'Marketing'),
        ('D003', 'Operations'),
        ('D004', 'Finance')
    ]
    
    # Generate forecasts for next 12 months
    current_date = datetime.datetime.now()
    current_month = current_date.replace(day=1)
    
    for i in range(1, num_forecasts + 1):
        forecast_id = i
        
        # Random month in the next 12 months
        month_offset = random.randint(0, 11)
        forecast_date = (current_month + datetime.timedelta(days=30*month_offset)).strftime('%Y-%m-%d')
        
        account = random.choice(accounts)
        account_number, account_description = account
        
        department = random.choice(departments)
        department_code, department_description = department
        
        # Generate forecast amount (positive for revenue, negative for expenses)
        if account_number.startswith('4'):  # Revenue
            forecast_amount = random.uniform(50000, 500000)
        else:  # Expense
            forecast_amount = -random.uniform(20000, 300000)
        
        currency_code = 'USD'
        
        forecasts.append((
            forecast_id, forecast_date, account_number, account_description,
            department_code, forecast_amount, currency_code
        ))
    
    conn.executemany('''
    INSERT INTO dbo_F_GL_Forecast (
        ForecastID, "Forecast Date", "Account Number", "Account Description",
        "Department Code", "Forecast Amount", "Currency Code"
    ) VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', forecasts)
    
    conn.commit()
    logger.info(f"Generated {num_forecasts} forecast records")

def main():
    args = parse_args()
    
    # Define database path
    db_path = os.path.join(script_dir, args.output)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    logger.info(f"Generating sample database at {db_path}")
    
    # Connect to database (creates it if it doesn't exist)
    conn = sqlite3.connect(db_path)
    
    try:
        # Create schema
        create_database_schema(conn)
        
        # Generate data
        customer_ids = generate_customers(conn, num_customers=50)
        generate_gl_transactions(conn, customer_ids, num_records=args.records)
        generate_ar_data(conn, customer_ids, num_invoices=args.records // 5)
        generate_forecasts(conn, num_forecasts=100)
        
        logger.info("Sample database generation completed successfully.")
    except Exception as e:
        logger.error(f"Error generating sample database: {e}")
        return 1
    finally:
        conn.close()
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 