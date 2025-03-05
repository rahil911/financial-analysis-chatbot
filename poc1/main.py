import os
import sys
import argparse
from pathlib import Path
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("main.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Main")

def setup_environment():
    """Set up the environment variables and dependencies"""
    logger.info("Setting up environment")
    
    # Set up directories using correct relative paths
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    for dir_path in ["data", "data/reports", "logs"]:
        full_path = script_dir / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {full_path.absolute()}")
    
    # Check for required packages
    required_packages = [
        "pandas", "streamlit", "plotly", "sklearn", 
        "xgboost", "lightgbm", "sqlite3", "requests"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"Package {package} is installed")
        except ImportError as e:
            missing_packages.append(package)
            logger.error(f"Missing required package: {package}")
    
    if missing_packages:
        error_msg = f"Missing required packages: {', '.join(missing_packages)}"
        logger.error(error_msg)
        print(error_msg)
        print("Please install all dependencies with: pip install -r requirements.txt")
        sys.exit(1)
    
    # Check for API keys
    if not os.environ.get("OPENAI_API_KEY"):
        logger.warning("No API key found for OpenAI")
        print("Warning: No API key found for OpenAI.")
        print("Please set the OPENAI_API_KEY environment variable.")

def setup_database():
    """Set up the SQLite database from CSV files"""
    logger.info("Setting up the financial database")
    print("Setting up the financial database...")
    
    try:
        # Import the database setup module
        script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        sys.path.append(str(script_dir))
        from data.db_setup import setup_database
        
        # Run the database setup
        setup_database()
        logger.info("Database setup complete")
        print("Database setup complete.")
    except Exception as e:
        error_msg = f"Error setting up database: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        print(f"Error setting up database: {str(e)}")
        sys.exit(1)

def run_chatbot():
    """Run the Streamlit chatbot application"""
    logger.info("Starting the financial analysis chatbot")
    print("Starting the financial analysis chatbot...")
    
    # Use correct path for app directory
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    app_path = script_dir / "app"
    
    logger.info(f"App directory: {app_path}")
    
    if not app_path.exists():
        error_msg = f"Error: App directory not found at {app_path.absolute()}"
        logger.error(error_msg)
        print(error_msg)
        sys.exit(1)
    
    # Verify database exists
    db_path = script_dir / "data" / "financial.db"
    if not db_path.exists():
        error_msg = f"Error: Database not found at {db_path.absolute()}. Run setup first."
        logger.error(error_msg)
        print(error_msg)
        sys.exit(1)
    else:
        logger.info(f"Database found at {db_path.absolute()}")
    
    # Run Streamlit with detailed command
    chat_app_path = app_path / "chat_app.py"
    streamlit_cmd = f"streamlit run {chat_app_path}"
    logger.info(f"Running command: {streamlit_cmd}")
    os.system(streamlit_cmd)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Financial Analysis Chatbot")
    parser.add_argument("--setup", action="store_true", help="Set up the environment and database")
    parser.add_argument("--run", action="store_true", help="Run the chatbot application")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Enable debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug mode enabled")
    
    # If no arguments provided, show help
    if not args.setup and not args.run:
        print("Please specify an action: --setup or --run")
        logger.warning("No action specified")
        sys.exit(1)
    
    # Setup environment
    if args.setup:
        try:
            setup_environment()
            setup_database()
            logger.info("Setup completed successfully")
        except Exception as e:
            error_msg = f"Setup failed: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            print(f"Setup failed: {str(e)}")
            sys.exit(1)
    
    # Run the chatbot
    if args.run:
        try:
            run_chatbot()
            logger.info("Chatbot exited")
        except Exception as e:
            error_msg = f"Error running chatbot: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            print(f"Error running chatbot: {str(e)}")
            sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        error_msg = f"Unhandled exception: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        print(f"An error occurred: {str(e)}")
        sys.exit(1) 