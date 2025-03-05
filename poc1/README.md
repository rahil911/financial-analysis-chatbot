# Financial Analysis Chatbot

A conversational AI interface for comprehensive financial data analysis, forecasting, and visualization.

## Overview

This project creates an intelligent chatbot that combines:

1. **Financial Analysis Tools**: Cash flow, accounts receivable, revenue forecasting, profitability, and customer analysis
2. **Machine Learning Models**: Time series forecasting, anomaly detection, and pattern recognition
3. **Natural Language Processing**: Conversational interface powered by OpenAI or Anthropic LLMs
4. **Interactive Visualizations**: Plotly-generated charts and graphs embedded in chat

The chatbot allows financial analysts, business leaders, and other stakeholders to explore and analyze financial data naturally through conversation, while leveraging sophisticated analytical tools in the background.

## Features

- **Conversational Interface**: Ask questions in natural language about your financial data
- **Cash Flow Analysis**: Analyze inflows, outflows, and identify anomalies
- **AR Aging Analysis**: Evaluate accounts receivable health and customer payment patterns
- **Revenue Forecasting**: ML-powered revenue projections using ensemble methods
- **Profitability Analysis**: Segment profitability by dimensions like business unit or product
- **Customer Analysis**: Identify top customers, evaluate their behavior, and assess risk
- **Visualizations**: Interactive charts and graphs embedded directly in the chat
- **SQL Database Integration**: Structured financial data for efficient querying and analysis

## Setup Instructions

### Prerequisites

- Python 3.8+
- Financial data in CSV format (located at `/Users/rahilharihar/Projects/Bicycle/DB`)
- API keys for either OpenAI or Anthropic

### Installation

1. Clone this repository or navigate to the directory:

```bash
cd poc1
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Set up your API keys as environment variables:

```bash
# For Anthropic Claude
export ANTHROPIC_API_KEY=your_api_key_here

# OR for OpenAI
export OPENAI_API_KEY=your_api_key_here
```

4. Set up the database and environment:

```bash
python main.py --setup
```

### Running the Application

1. Start the Streamlit application:

```bash
python main.py --run
```

2. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

## Usage Examples

You can interact with the chatbot by asking questions in natural language, such as:

- "How has our cash flow trended over the last quarter?"
- "Show me our accounts receivable aging analysis"
- "What is our revenue forecast for the next 30 days?"
- "Which business units are most profitable?"
- "Who are our top 10 customers by revenue?"
- "Are there any anomalies in our recent cash flow?"

The chatbot will process your query, run the appropriate analyses, and respond with both textual insights and visualizations.

## Architecture

- **Data Layer**: Loads CSV financial data into SQLite for efficient querying
- **Analysis Engine**: Modular financial analysis tools exposed as API endpoints
- **LLM Integration**: Connects to OpenAI or Anthropic APIs for natural language processing
- **Visualization Layer**: Creates dynamic visualizations using Plotly
- **User Interface**: Streamlit-based chat interface for easy interaction

## Customization

- **Add New Analysis Tools**: Extend the `FinancialTools` class in `poc1/tools/financial_tools.py`
- **Modify LLM Integration**: Change settings in `poc1/utils/llm_processor.py`
- **Adjust Visualizations**: Customize charts in the tool methods
- **UI Customization**: Modify the Streamlit interface in `poc1/app/chat_app.py`

## License

This project is licensed under the MIT License. See the LICENSE file for details. 