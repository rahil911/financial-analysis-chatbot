# Database Setup for Vercel Deployment

This directory should contain the SQLite database file (`financial.db`) used by the application. 

## Setting Up the Database for Vercel Deployment

### Option 1: Using Google Drive CSV Files (Recommended)

1. Download the required CSV files from the Google Drive folder shared with you
2. Create a temporary directory for the CSV files:
   ```bash
   mkdir -p ~/csv_import
   ```
3. Copy or move the downloaded CSV files to this directory
4. Run the import script:
   ```bash
   python import_csv_data.py --csv_dir ~/csv_import --output financial.db
   ```
5. Verify the database was created successfully

### Option 2: Generate Sample Data (For Testing Only)

If you don't have access to the Google Drive CSV files, you can generate sample data for testing:

```bash
python generate_sample_data.py
```

## Important Notes for Vercel Deployment

Vercel has some limitations when working with file-based databases:

1. **Build-time vs Runtime**: The database must be created during the build phase, as the file system is read-only during runtime
2. **Size Limitations**: Vercel has a size limit for deployments, large databases may cause issues
3. **Database Location**: The database must be located in this directory (`poc1/data/financial.db`) for the application to find it

## CSV Files Expected

The following CSV files should be downloaded from Google Drive:

- `dbo_F_GL_Transaction_Detail.csv` - General Ledger transactions
- `dbo_F_AR_Header.csv` - Accounts Receivable headers 
- `dbo_F_AR_Detail.csv` - Accounts Receivable details
- `dbo_D_Customer.csv` - Customer data
- `dbo_F_GL_Transaction.csv` - GL transactions

For detailed information about the database schema and setup process, see the `DATABASE_SETUP.md` file in the root of the repository. 