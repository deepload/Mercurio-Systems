# Mercurio AI Trading Platform

An open-source, modular and extensible platform for AI-driven stock trading.

## Overview

Mercurio AI is a comprehensive trading platform built with Python, featuring:

- A robust FastAPI backend
- Pluggable strategy system for custom trading algorithms
- Market data integration with IEX Cloud and Alpaca
- Real-time and paper trading capabilities
- Backtesting system for strategy evaluation
- Asynchronous task processing with Celery
- PostgreSQL database for data persistence
- Docker support for easy deployment

## Project Structure

```
mercurio_ai/
├── app/
│   ├── main.py               # FastAPI application entry point
│   ├── api/                  # API endpoints
│   │   ├── routes.py         # API route definitions
│   │   └── schemas.py        # Pydantic models for requests/responses
│   ├── services/             # Business logic services
│   │   ├── market_data.py    # Service for fetching market data
│   │   ├── trading.py        # Service for executing trades
│   │   └── backtesting.py    # Service for backtesting strategies
│   ├── strategies/           # Trading strategies
│   │   ├── base.py           # Base strategy class
│   │   ├── moving_average.py # Moving Average Crossover strategy
│   │   └── lstm_predictor.py # LSTM-based prediction strategy
│   ├── db/                   # Database modules
│   │   ├── database.py       # Database connection
│   │   └── models.py         # SQLAlchemy models
│   ├── tasks/                # Celery tasks
│   │   ├── celery_app.py     # Celery configuration
│   │   ├── training.py       # Tasks for model training
│   │   ├── trading.py        # Tasks for automated trading
│   │   └── data.py           # Tasks for data collection
│   └── utils/                # Utility functions
├── tests/                    # Tests directory
├── docker-compose.yml        # Docker Compose configuration
├── Dockerfile                # Docker configuration
├── requirements.txt          # Python dependencies
├── .env.example              # Example environment variables
└── README.md                 # This file
```

## Getting Started

### Prerequisites

- Python 3.11 or later
- Docker and Docker Compose (for containerized deployment)
- Alpaca API key and secret (for trading)
- IEX Cloud API key (for market data)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mercurio-ai.git
   cd mercurio-ai
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file based on `.env.example`:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

### Running with Docker

1. Start the services:
   ```bash
   docker-compose up -d
   ```

2. Access the API at http://localhost:8000
   
3. Access Flower (Celery monitoring) at http://localhost:5555

### Running Locally

1. Start a PostgreSQL database and Redis server

2. Run the FastAPI application:
   ```bash
   uvicorn app.main:app --reload
   ```

3. In a separate terminal, run Celery workers:
   ```bash
   celery -A app.tasks.celery_app worker --loglevel=info
   ```

## API Endpoints

The API provides the following endpoints:

- `GET /api/strategies` - List available trading strategies
- `GET /api/strategies/{strategy_name}` - Get details about a specific strategy
- `POST /api/predictions/{strategy_name}/{symbol}` - Get a prediction for a symbol
- `POST /api/backtests` - Run a backtest for a strategy
- `GET /api/backtests/{backtest_id}` - Get backtest results
- `POST /api/models/train` - Train a strategy model
- `GET /api/models/{model_id}` - Get model details
- `POST /api/trades` - Execute a trade
- `GET /api/account` - Get account information
- `GET /api/market/status` - Check market status

## Adding Custom Strategies

To create a new trading strategy:

1. Create a new Python file in the `app/strategies` directory
2. Extend the `BaseStrategy` class
3. Implement the required methods:
   - `load_data`
   - `preprocess_data`
   - `predict`
   - Optionally `train` and other methods

Example:

```python
from app.strategies.base import BaseStrategy

class MyCustomStrategy(BaseStrategy):
    """My custom trading strategy"""
    
    def __init__(self, custom_param=10):
        self.custom_param = custom_param
        
    async def load_data(self, symbol, start_date, end_date):
        # Implementation
        pass
        
    async def preprocess_data(self, data):
        # Implementation
        pass
        
    async def predict(self, data):
        # Implementation
        pass
```

## Testing

Run the test suite with:

```bash
pytest
```

## License

This project is open-source and available under the MIT License.

## Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/)
- [SQLAlchemy](https://www.sqlalchemy.org/)
- [Alpaca API](https://alpaca.markets/)
- [IEX Cloud](https://iexcloud.io/)
- [Backtrader](https://www.backtrader.com/)
