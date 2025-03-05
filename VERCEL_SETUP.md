# Vercel Deployment Guide

This guide covers how to deploy the Financial Analysis Chatbot to Vercel, with particular attention to setting up the database from Google Drive CSV files.

## Prerequisites

- [Vercel account](https://vercel.com/signup)
- [Vercel CLI](https://vercel.com/cli) installed
- Access to the Google Drive folder containing the financial CSV data files
- OpenAI API key

## Step 1: Prepare Your Repository

1. Clone the repository (if you haven't already):
   ```bash
   git clone https://github.com/rahil911/financial-analysis-chatbot.git
   cd financial-analysis-chatbot
   ```

2. Ensure the `vercel.json` file is in the root directory with the correct configuration

## Step 2: Set Up the Database

### Option A: Import During Development

1. Download all CSV files from the Google Drive folder to your local machine
2. Create a directory for the CSV files:
   ```bash
   mkdir -p data_import
   ```
3. Move the downloaded CSV files to this directory
4. Run the import script to create the SQLite database:
   ```bash
   python poc1/data/import_csv_data.py --csv_dir ./data_import --output poc1/data/financial.db
   ```
5. Verify the database was created:
   ```bash
   ls -lh poc1/data/financial.db
   ```
   
### Option B: Configure Build-time Database Creation

Since Vercel has a read-only filesystem at runtime, you'll need to create the database during the build phase:

1. Add build commands to your `vercel.json` file to download the CSV files and create the database during deployment
2. Make the CSV files accessible via a secure URL (e.g., private S3 bucket, password-protected download)
3. Update your build script to download the files and run the import script

## Step 3: Environment Variables

Configure your environment variables in the Vercel dashboard:

1. Go to your Vercel project
2. Navigate to "Settings" > "Environment Variables"
3. Add the following variables:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `DATABASE_PATH`: `poc1/data/financial.db`
   - (Optional) Any other environment variables you might need

## Step 4: Deploy to Vercel

### Using Vercel Dashboard

1. Push your code to GitHub
2. In the Vercel dashboard, click "New Project"
3. Import your repository
4. Configure your project settings
5. Click "Deploy"

### Using Vercel CLI

1. Ensure you're logged in to Vercel CLI:
   ```bash
   vercel login
   ```
2. Deploy your project:
   ```bash
   vercel
   ```

## Step 5: Handle Database Updates

Since Vercel deployments are immutable, to update your database:

1. Make the necessary changes to your CSV files or database
2. Create a new deployment, which will run the build process again
3. Alternatively, set up a workflow to load data from an external database service

## Special Considerations for Vercel Deployments

1. **Filesystem Limitations**: Vercel's filesystem is read-only during runtime, so your app can't write to the database after deployment
2. **Database Size**: Large databases may exceed Vercel's size limits (check current [limits](https://vercel.com/docs/concepts/limits/overview))
3. **Build Duration**: Importing large CSV files during build may exceed build time limits
4. **Serverless Architecture**: Be aware of cold starts and connection handling in serverless environments

## Alternative Options

If the SQLite file-based approach becomes problematic on Vercel, consider:

1. Using a cloud database service (e.g., Supabase, Neon, PlanetScale)
2. Setting up a simple API to access your database hosted elsewhere
3. Using Vercel Storage solutions for more dynamic data handling

## Troubleshooting

- **Build Failures**: Check the build logs in Vercel for specific errors
- **Missing Data**: Verify that your CSV import script is running correctly during build
- **API Key Issues**: Ensure your environment variables are correctly set
- **Performance Problems**: Consider optimizing your database or queries if the app is slow