#!/bin/bash

# Setup script for the Financial Analysis Chatbot

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Installing required packages..."
pip install -r requirements.txt

# Create .env file template if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file template..."
    echo "# Add your OpenAI API key here" > .env
    echo "export OPENAI_API_KEY=''" >> .env
    echo ".env file created. Please edit it to add your OpenAI API key."
else
    echo ".env file already exists."
fi

# Create reports directory if it doesn't exist
if [ ! -d "poc1/data/reports" ]; then
    echo "Creating reports directory..."
    mkdir -p poc1/data/reports
fi

echo "Setup complete! Follow these steps to run the chatbot:"
echo "1. Add your OpenAI API key to the .env file"
echo "2. Run 'source .env' to load environment variables"
echo "3. Run 'python poc1/main.py --run' to start the chatbot"
echo "4. Open your browser and navigate to http://localhost:8501" 