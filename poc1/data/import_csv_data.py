#!/usr/bin/env python
"""
CSV Data Importer for Financial Analysis Chatbot

This script imports CSV files into the SQLite database used by the chatbot.
It's designed to work with the CSV files stored in the Google Drive folder.

Usage:
    python import_csv_data.py --csv_dir /path/to/csv/files --output financial.db
"""

import os
import sys
import sqlite3
import argparse
import pandas as pd
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ImportCSV')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Import CSV files into SQLite database')
    parser.add_argument('--csv_dir', required=True, help='Directory containing CSV files')
    parser.add_argument('--output', default='financial.db', help='Output database file')
    return parser.parse_args()

def create_database_schema(conn):
    """Create the database schema."""
    cursor = conn.cursor()
    
    # Create GL Transaction Detail table
    cursor.execute('''
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
    )
    ''')
    
    # Create AR Header table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS dbo_F_AR_Header (
        InvoiceID INTEGER PRIMARY KEY,
        "Posting Date" TEXT,
        "Document Number" TEXT, 
        "Customer ID" TEXT,
        "Total Amount" REAL,
        "Currency Code" TEXT,
        "Due Date" TEXT,
        "Status" TEXT
    )
    ''')
    
    # Create AR Detail table
    cursor.execute('''
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
    )
    ''')
    
    # Create Customer dimension table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS dbo_D_Customer (
        "Customer ID" TEXT PRIMARY KEY,
        "Customer Name" TEXT,
        "Customer Type" TEXT,
        "Industry" TEXT,
        "Region" TEXT,
        "Country" TEXT,
        "Credit Limit" REAL,
        "Status" TEXT
    )
    ''')
    
    # Create GL Forecast table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS dbo_F_GL_Forecast (
        ForecastID INTEGER PRIMARY KEY,
        "Forecast Date" TEXT,
        "Account Number" TEXT,
        "Account Description" TEXT,
        "Department Code" TEXT,
        "Forecast Amount" REAL,
        "Currency Code" TEXT
    )
    ''')
    
    # Create table_mapping table for metadata
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS table_mapping (
        table_name TEXT,
        column_name TEXT,
        description TEXT,
        PRIMARY KEY (table_name, column_name)
    )
    ''')
    
    # Insert sample metadata
    cursor.execute('''
    INSERT OR IGNORE INTO table_mapping VALUES
        ('dbo_F_GL_Transaction_Detail', 'Posting Date', 'Date when the transaction was posted'),
        ('dbo_F_GL_Transaction_Detail', 'Txn Amount', 'Amount of the transaction'),
        ('dbo_F_AR_Header', 'Posting Date', 'Date when the invoice was posted'),
        ('dbo_F_AR_Header', 'Due Date', 'Date when the invoice is due'),
        ('dbo_D_Customer', 'Customer ID', 'Unique identifier for the customer'),
        ('dbo_D_Customer', 'Customer Name', 'Name of the customer')
    ''')
    
    conn.commit()

def import_csv_files(conn, csv_dir):
    """Import CSV files from directory into database."""
    csv_dir_path = Path(csv_dir)
    
    # Map of CSV filenames to table names
    file_to_table = {
        'dbo_F_GL_Transaction_Detail.csv': 'dbo_F_GL_Transaction_Detail',
        'dbo_F_AR_Header.csv': 'dbo_F_AR_Header',
        'dbo_F_AR_Detail.csv': 'dbo_F_AR_Detail',
        'dbo_D_Customer.csv': 'dbo_D_Customer',
        'dbo_F_GL_Transaction.csv': 'dbo_F_GL_Transaction',
        # Add more mappings as needed
    }
    
    for csv_file, table_name in file_to_table.items():
        file_path = csv_dir_path / csv_file
        
        if not file_path.exists():
            logger.warning(f"CSV file {file_path} not found. Skipping.")
            continue
        
        logger.info(f"Importing {csv_file} into {table_name}...")
        
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Handle any necessary data transformations here
            # For example, ensure date formats are correct
            date_columns = [col for col in df.columns if 'Date' in col]
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%Y-%m-%d')
            
            # Write to SQLite
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            logger.info(f"Successfully imported {len(df)} rows into {table_name}")
            
        except Exception as e:
            logger.error(f"Error importing {csv_file}: {str(e)}")

def main():
    """Main function to execute the script."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Connect to SQLite database
    conn = sqlite3.connect(args.output)
    
    try:
        # Create database schema
        create_database_schema(conn)
        
        # Import CSV files
        import_csv_files(conn, args.csv_dir)
        
        logger.info(f"Database created successfully at {args.output}")
        
    except Exception as e:
        logger.error(f"Error creating database: {str(e)}")
        
    finally:
        conn.close()

if __name__ == "__main__":
    main() 