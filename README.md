# Financial Analysis Chatbot

An intelligent financial analysis chatbot powered by OpenAI's GPT models that helps analyze financial data and generate insights from SQL databases.

## Features

- Interactive chat interface built with Streamlit
- Real-time financial analysis and data visualization
- Natural language processing for financial queries
- Supports various financial analyses:
  - Revenue forecasting
  - Cash flow analysis
  - Accounts receivable aging
  - Customer analysis
  - Custom SQL queries

## Requirements

- Python 3.8+
- SQLite database with financial data
- OpenAI API key

## Installation

1. Clone this repository:
```bash
git clone https://github.com/[USERNAME]/financial-analysis-chatbot.git
cd financial-analysis-chatbot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Set up your environment variables:
```bash
echo "export OPENAI_API_KEY='your-api-key-here'" > .env
source .env
```

## Running the Application

1. Ensure your financial database is in `poc1/data/financial.db`

2. Run the application:
```bash
python poc1/main.py --run
```

3. Open your browser and navigate to `http://localhost:8501`

## Project Structure

```
poc1/
├── app/
│   └── chat_app.py          # Streamlit chat interface
├── data/
│   └── financial.db         # SQLite database with financial data
│   └── reports/             # Generated reports and visualizations
├── tools/
│   └── financial_tools.py   # Financial analysis tools
├── utils/
│   └── llm_processor.py     # LLM processing utilities
└── main.py                  # Main application entry point
```

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key

## Customization

You can customize the chatbot by modifying the following files:
- `poc1/utils/llm_processor.py`: Change the LLM model or add more tools
- `poc1/tools/financial_tools.py`: Add more financial analysis tools
- `poc1/app/chat_app.py`: Modify the Streamlit UI

## License

MIT License 