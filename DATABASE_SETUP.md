# Database Setup Guide

This guide explains how to set up a sample database for the Financial Analysis Chatbot. The chatbot expects a SQLite database with specific tables and schema.

## Expected Database Location

The database should be located at:
```
poc1/data/financial.db
```

## Required Database Schema

The database should have the following tables:

1. **dbo_F_GL_Transaction_Detail** - General Ledger transactions
2. **dbo_F_AR_Header** - Accounts Receivable headers
3. **dbo_F_AR_Detail** - Accounts Receivable details
4. **dbo_D_Customer** - Customer dimension table
5. **dbo_F_GL_Forecast** - General Ledger forecasts
6. **table_mapping** - Table and column descriptions

## Sample Database Schema

Below is a SQL script to create a minimal sample database structure:

```sql
-- Create GL Transaction Detail table
CREATE TABLE dbo_F_GL_Transaction_Detail (
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
CREATE TABLE dbo_F_AR_Header (
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
CREATE TABLE dbo_F_AR_Detail (
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
CREATE TABLE dbo_D_Customer (
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
CREATE TABLE dbo_F_GL_Forecast (
    ForecastID INTEGER PRIMARY KEY,
    "Forecast Date" TEXT,
    "Account Number" TEXT,
    "Account Description" TEXT,
    "Department Code" TEXT,
    "Forecast Amount" REAL,
    "Currency Code" TEXT
);

-- Create table_mapping table for metadata
CREATE TABLE table_mapping (
    table_name TEXT,
    column_name TEXT,
    description TEXT,
    PRIMARY KEY (table_name, column_name)
);

-- Insert sample metadata
INSERT INTO table_mapping VALUES
    ('dbo_F_GL_Transaction_Detail', 'Posting Date', 'Date when the transaction was posted'),
    ('dbo_F_GL_Transaction_Detail', 'Txn Amount', 'Amount of the transaction'),
    ('dbo_F_AR_Header', 'Posting Date', 'Date when the invoice was posted'),
    ('dbo_F_AR_Header', 'Due Date', 'Date when the invoice is due'),
    ('dbo_D_Customer', 'Customer ID', 'Unique identifier for the customer'),
    ('dbo_D_Customer', 'Customer Name', 'Name of the customer');
```

## Generating Sample Data

To generate sample data for testing, you can run the following Python script included in the repository:

```bash
python poc1/data/generate_sample_data.py
```

This will create a sample database with random but realistic financial data that you can use for testing the chatbot.

## Using Your Own Data

If you want to use your own financial data, you should ensure it follows the schema described above. You can:

1. Export your data to CSV files
2. Use a script to import the CSV files into a SQLite database
3. Ensure the column names match those expected by the chatbot
4. Place the database file at `poc1/data/financial.db`

## Important Notes

- The chatbot expects dates in ISO format (YYYY-MM-DD)
- Currency amounts should be stored as decimal numbers
- Foreign key relationships should be maintained where specified
- Customer IDs should be consistent across all tables 