import pandas as pd
import sqlite3
from pathlib import Path
import os
import logging

# Configure logging
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
project_root = script_dir.parent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(script_dir / "db_setup.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DBSetup")

def setup_database():
    """
    Load only the necessary CSV files from the DB directory into an SQLite database
    """
    # Define paths
    db_dir = Path("/Users/rahilharihar/Projects/Bicycle/DB")
    sqlite_path = script_dir / "financial.db"
    
    # Make sure the paths are absolute
    db_dir = db_dir.absolute()
    sqlite_path = sqlite_path.absolute()
    
    logger.info(f"DB directory: {db_dir}")
    logger.info(f"SQLite database path: {sqlite_path}")
    
    # Create the database directory if it doesn't exist
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Connect to SQLite database
    conn = sqlite3.connect(sqlite_path)
    
    # Define the specific files we need for our financial analysis
    required_files = [
        "dbo_F_GL Transaction Detail.csv",  # For cash flow, revenue forecast, profitability
        "dbo_F_AR Header.csv",             # For AR aging analysis
        "dbo_F_AR Detail.csv",             # For deeper AR analysis
        "dbo_D_Customer.csv",              # For customer analysis
        "dbo_F_GL Forecast.csv"            # For forecast comparison
    ]
    
    # Create a mapping of original filenames to sanitized table names
    # This will help us track the exact table names used
    filename_to_table_map = {}
    
    # Count found files
    found_files = []
    for required_file in required_files:
        file_path = db_dir / required_file
        if file_path.exists():
            found_files.append(file_path)
            # Create sanitized table name
            table_name = os.path.splitext(file_path.name)[0].replace(" ", "_").replace("-", "_")
            # Clean table name for SQL
            table_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in table_name)
            filename_to_table_map[required_file] = table_name
        else:
            logger.warning(f"Required file {required_file} not found. Some analyses may not work correctly.")
            print(f"Warning: Required file {required_file} not found. Some analyses may not work correctly.")
    
    # Log the filename to table name mapping
    logger.info("File to table mapping:")
    for original, table in filename_to_table_map.items():
        logger.info(f"  {original} -> {table}")
    
    print(f"Found {len(found_files)} out of {len(required_files)} required files")
    
    # Process each found file
    for csv_file in found_files:
        table_name = filename_to_table_map.get(csv_file.name, None)
        if not table_name:
            # Fallback if not in mapping
            table_name = os.path.splitext(csv_file.name)[0].replace(" ", "_").replace("-", "_")
            table_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in table_name)
        
        print(f"Loading {csv_file.name} into table {table_name}...")
        logger.info(f"Loading {csv_file.name} into table {table_name}...")
        
        try:
            # Read CSV
            df = pd.read_csv(csv_file, low_memory=False)
            
            # Write to SQLite
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            print(f"Successfully loaded {len(df)} rows into {table_name}")
            logger.info(f"Successfully loaded {len(df)} rows into {table_name}")
            
        except Exception as e:
            error_msg = f"Error loading {csv_file.name}: {str(e)}"
            print(error_msg)
            logger.error(error_msg)
    
    # Create views for common queries
    print("Creating database views...")
    
    # First, verify table names that actually exist in the database
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    existing_tables = [t[0] for t in tables]
    logger.info(f"Tables in database: {existing_tables}")
    
    # Make sure we use the actual table names that exist in the database
    ar_header_table = filename_to_table_map.get("dbo_F_AR Header.csv", "dbo_F_AR_Header")
    
    # Example: AR Aging view - only create if the table exists
    if ar_header_table in existing_tables:
        try:
            ar_aging_view = f"""
            CREATE VIEW IF NOT EXISTS ar_aging AS
            SELECT 
                h."Customer Key",
                h."Invoice Number",
                h."Posting Date",
                h."Balance Due Amount",
                JULIANDAY('now') - JULIANDAY(h."Posting Date") as days_outstanding,
                CASE
                    WHEN JULIANDAY('now') - JULIANDAY(h."Posting Date") <= 30 THEN '0-30'
                    WHEN JULIANDAY('now') - JULIANDAY(h."Posting Date") <= 60 THEN '31-60'
                    WHEN JULIANDAY('now') - JULIANDAY(h."Posting Date") <= 90 THEN '61-90'
                    WHEN JULIANDAY('now') - JULIANDAY(h."Posting Date") <= 120 THEN '91-120'
                    ELSE '120+'
                END as aging_bucket
            FROM "{ar_header_table}" h
            WHERE h."Balance Due Amount" > 0
            """
            conn.execute(ar_aging_view)
            print(f"Created AR aging view using table: {ar_header_table}")
            logger.info(f"Created AR aging view using table: {ar_header_table}")
        except Exception as e:
            error_msg = f"Error creating AR aging view: {str(e)}"
            print(error_msg)
            logger.error(error_msg)
    else:
        error_msg = f"Cannot create AR aging view: Table {ar_header_table} does not exist"
        print(error_msg)
        logger.error(error_msg)
    
    # Create a mapping table to help with table name lookups
    try:
        cursor.execute("CREATE TABLE IF NOT EXISTS table_mapping (original_name TEXT, table_name TEXT)")
        cursor.executemany(
            "INSERT INTO table_mapping VALUES (?, ?)",
            [(k, v) for k, v in filename_to_table_map.items()]
        )
        conn.commit()
        logger.info("Created table_mapping table with filename-to-table mappings")
    except Exception as e:
        error_msg = f"Error creating table mapping: {str(e)}"
        print(error_msg)
        logger.error(error_msg)
    
    # Close connection
    conn.close()
    print("Database setup complete")
    logger.info("Database setup complete")

if __name__ == "__main__":
    setup_database() 