# MercurioAI Trading Platform Documentation

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Table of Contents

- [Project Overview](#project-overview)
- [Core Features](#core-features)
- [Architecture](#architecture)
- [Installation and Setup](#installation-and-setup)
- [Configuration](#configuration)
- [Core Components](#core-components)
  - [Market Data Service](#market-data-service)
  - [Trading Service](#trading-service)
  - [Backtesting Service](#backtesting-service)
  - [Strategy Manager](#strategy-manager)
- [Trading Strategies](#trading-strategies)
  - [Moving Average Strategy](#moving-average-strategy)
  - [LSTM Predictor Strategy](#lstm-predictor-strategy)
- [Creating Custom Strategies](#creating-custom-strategies)
- [Backtesting](#backtesting)
  - [Standard Backtesting](#standard-backtesting)
  - [Long-term Backtesting](#long-term-backtesting)
  - [Performance Metrics](#performance-metrics)
- [Running the Platform](#running-the-platform)
  - [Demo Mode](#demo-mode)
  - [Paper Trading](#paper-trading)
  - [Live Trading](#live-trading)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [Contributing](#contributing)

## Project Overview

MercurioAI is a comprehensive algorithmic trading platform built in Python. It's designed for developing, testing, and deploying trading strategies for stocks and cryptocurrencies. The platform integrates with Alpaca for execution and provides a modular architecture that supports multiple data sources and strategy types.

MercurioAI is particularly well-suited for quantitative traders and researchers who want to:
- Develop and test trading strategies using historical data
- Implement both technical analysis and machine learning approaches
- Backtest strategies over different time periods and market conditions
- Execute strategies through paper trading and live trading
- Analyze performance using industry-standard metrics

## Core Features

- **Multiple Data Sources**: Integration with Polygon.io, Yahoo Finance, and Alpaca for market data with automatic fallback mechanisms
- **Modular Strategy Framework**: Easily create and test various trading strategies
- **Advanced Backtesting**: Comprehensive backtesting capabilities with transaction costs and detailed performance metrics
- **Machine Learning Integration**: Built-in support for ML-based trading strategies (LSTM and Random Forest)
- **Paper Trading**: Test strategies with real-time data without risking capital
- **Live Trading**: Execute strategies in live markets through Alpaca
- **Performance Analytics**: Track and analyze strategy performance
- **Data Persistence**: Store market data, trades, and model information in PostgreSQL

## Architecture

MercurioAI follows a modular architecture with several key components:

1. **Services**: Core functionalities encapsulated in service classes (Market Data, Trading, Backtesting)
2. **Strategies**: Trading strategy implementations that inherit from a common base class
3. **Data Models**: Database models for persistent storage
4. **Utilities**: Helper functions and common tools
5. **API Layer**: Optional REST API for monitoring and control

The platform uses asynchronous programming (async/await) for efficient data processing and API interactions.

## Installation and Setup

### Prerequisites

- Python 3.9 or higher
- PostgreSQL (optional, for data persistence)
- Redis (optional, for caching)

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/yourusername/mercurioai.git
cd mercurioai

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Database Setup (Optional)

For full functionality, set up a PostgreSQL database:

```bash
# Create database
createdb mercurio

# Run database migrations (if applicable)
python -m app.db.migrate
```

## Configuration

MercurioAI uses environment variables for configuration. Create a `.env` file in the project root with the following settings:

```
# API Keys for data and trading
# === Market Data Providers ===
# Configure one or more of the following providers:

# Polygon.io (primary recommended market data provider)
POLYGON_API_KEY=your_polygon_api_key_here

# Alpaca (can be used for both market data and trading)
ALPACA_KEY=your_alpaca_key_here
ALPACA_SECRET=your_alpaca_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # or https://api.alpaca.markets for live trading

# Additional market data providers (uncomment and configure as needed)
# Note: Yahoo Finance is available as a free provider without an API key
# FMP_API_KEY=your_financial_modeling_prep_api_key_here
# TRADIER_API_KEY=your_tradier_api_key_here

# Database configuration
POSTGRES_DB=mercurio
POSTGRES_USER=mercurio_user
POSTGRES_PASSWORD=mercurio_password
POSTGRES_HOST=db
POSTGRES_PORT=5432
DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}

# Redis configuration
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_URL=redis://${REDIS_HOST}:${REDIS_PORT}/0

# Application settings
MODEL_DIR=./models
LOG_LEVEL=INFO
ENVIRONMENT=development
```

## Core Components

### Market Data Service

The `MarketDataService` provides access to historical and real-time market data through various providers.

```python
from app.services.market_data import MarketDataService
from datetime import datetime, timedelta

# Initialize the service
market_data = MarketDataService()

# Get historical data
start_date = datetime.now() - timedelta(days=30)
end_date = datetime.now()
data = await market_data.get_historical_data(
    symbol="AAPL", 
    start_date=start_date, 
    end_date=end_date
)

# Get latest price
price = await market_data.get_latest_price("MSFT")
```

The service supports multiple data providers with automatic fallback:

1. Polygon.io (recommended primary source, requires API key)
2. Alpaca (if you have an Alpaca account)
3. Yahoo Finance (free alternative)
4. Sample Data Generator (fallback for testing)

### Trading Service

The `TradingService` handles order execution and account management through Alpaca.

```python
from app.services.trading import TradingService
from app.db.models import TradeAction

# Initialize with paper trading
trading = TradingService(is_paper=True)

# Check market status
status = await trading.check_market_status()
print(f"Market is {'open' if status['is_open'] else 'closed'}")

# Execute a trade
result = await trading.execute_trade(
    symbol="AAPL",
    action=TradeAction.BUY,
    quantity=10,
    strategy_name="MovingAverageStrategy"
)
```

### Backtesting Service

The `BacktestingService` provides comprehensive backtesting capabilities.

```python
from app.services.backtesting import BacktestingService
from app.services.strategy_manager import StrategyManager
from datetime import datetime, timedelta

# Initialize services
backtesting = BacktestingService()
strategy_manager = StrategyManager()

# Get a strategy
strategy = await strategy_manager.get_strategy(
    "MovingAverageStrategy", 
    {"short_window": 20, "long_window": 50}
)

# Define backtest parameters
symbol = "AAPL"
end_date = datetime.now()
start_date = end_date - timedelta(days=365)
initial_capital = 10000.0

# Run backtest
results = await backtesting.run_backtest(
    strategy=strategy,
    symbol=symbol,
    start_date=start_date,
    end_date=end_date,
    initial_capital=initial_capital
)

# Access results
print(f"Total Return: {results['total_return']*100:.2f}%")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
```

### Strategy Manager

The `StrategyManager` handles strategy registration, instantiation, and management.

```python
from app.services.strategy_manager import StrategyManager

# Initialize the manager
strategy_manager = StrategyManager()

# List available strategies
strategies = await strategy_manager.list_strategies()
for strategy in strategies:
    print(f"{strategy['name']}: {strategy['description']}")

# Get a specific strategy with parameters
ma_strategy = await strategy_manager.get_strategy(
    "MovingAverageStrategy", 
    {"short_window": 10, "long_window": 30}
)
```

## Trading Strategies

### Moving Average Strategy

A classic technical analysis strategy based on moving average crossovers.

**Parameters**:
- `short_window`: Period for short-term moving average (default: 20)
- `long_window`: Period for long-term moving average (default: 50)
- `use_ml`: Whether to enhance with machine learning (default: False)

**Logic**:
- Buy when short MA crosses above long MA
- Sell when short MA crosses below long MA
- Optional ML enhancement using Random Forest

```python
# Initialize the strategy
strategy = await strategy_manager.get_strategy(
    "MovingAverageStrategy", 
    {
        "short_window": 20, 
        "long_window": 50,
        "use_ml": True
    }
)

# Train if using ML
if strategy.use_ml:
    await strategy.train(data)

# Get prediction
action, confidence = await strategy.predict(data)
```

### LSTM Predictor Strategy

An advanced machine learning strategy using Long Short-Term Memory neural networks.

**Parameters**:
- `sequence_length`: Number of previous time steps to use (default: 60)
- `lstm_units`: Number of LSTM units in the model (default: 50)
- `dropout_rate`: Dropout rate for regularization (default: 0.2)
- `epochs`: Number of training epochs (default: 50)
- `batch_size`: Batch size for training (default: 32)

**Logic**:
- Preprocess data with technical indicators
- Create sequences for LSTM input
- Train neural network to predict price direction
- Generate signals based on predicted direction

```python
# Initialize the strategy
lstm_strategy = await strategy_manager.get_strategy(
    "LSTMPredictorStrategy", 
    {
        "sequence_length": 30,
        "epochs": 50
    }
)

# Train the model (required)
await lstm_strategy.train(data)

# Get prediction
action, confidence = await lstm_strategy.predict(data)
```

## Creating Custom Strategies

To create a custom strategy:

1. Create a new Python file in `app/strategies/`
2. Inherit from `BaseStrategy`
3. Implement required methods
4. Register your strategy

Example:

```python
# app/strategies/my_custom_strategy.py
from app.strategies.base import BaseStrategy
from app.db.models import TradeAction
import pandas as pd
from datetime import datetime

class MyCustomStrategy(BaseStrategy):
    """My custom trading strategy"""
    
    def __init__(self, param1=10, param2=20, **kwargs):
        super().__init__(**kwargs)
        self.param1 = param1
        self.param2 = param2
    
    async def load_data(self, symbol, start_date, end_date):
        # You can use the market data service
        from app.services.market_data import MarketDataService
        market_data = MarketDataService()
        return await market_data.get_historical_data(symbol, start_date, end_date)
    
    async def preprocess_data(self, data):
        # Calculate your indicators
        data = data.copy()
        # Example: Simple moving averages
        data['sma1'] = data['close'].rolling(window=self.param1).mean()
        data['sma2'] = data['close'].rolling(window=self.param2).mean()
        return data
    
    async def train(self, data):
        # Implement if your strategy requires training
        return {"status": "success"}
    
    async def predict(self, data):
        # Generate trading signals
        if data['sma1'].iloc[-1] > data['sma2'].iloc[-1]:
            return TradeAction.BUY, 0.8
        elif data['sma1'].iloc[-1] < data['sma2'].iloc[-1]:
            return TradeAction.SELL, 0.8
        else:
            return TradeAction.HOLD, 0.5
    
    async def backtest(self, data, initial_capital=10000.0):
        # You can use the default implementation or customize
        return await super().backtest(data, initial_capital)
```

Then register your strategy in the `StrategyManager`:

```python
# In app/services/strategy_manager.py
from app.strategies.my_custom_strategy import MyCustomStrategy

# Add to the strategies dictionary
self.strategies = {
    "MovingAverageStrategy": MovingAverageStrategy,
    "LSTMPredictorStrategy": LSTMPredictorStrategy,
    "MyCustomStrategy": MyCustomStrategy
}
```

## Backtesting

### Standard Backtesting

MercurioAI provides comprehensive backtesting capabilities through the `BacktestingService`.

```python
# Run a standard backtest
results = await backtesting.run_backtest(
    strategy=strategy,
    symbol="AAPL",
    start_date=datetime(2022, 1, 1),
    end_date=datetime(2022, 12, 31),
    initial_capital=10000.0
)

# Results include performance metrics and equity curve
print(f"Total Return: {results['total_return']*100:.2f}%")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']*100:.2f}%")
```

### Long-term Backtesting

For more comprehensive backtesting, use the `long_term_backtest.py` script:

```bash
# Basic usage
python long_term_backtest.py

# Specify symbols and date range
python long_term_backtest.py --symbols AAPL,MSFT,GOOGL --start_date 2020-01-01 --end_date 2023-12-31

# Include transaction fees
python long_term_backtest.py --fee_percentage 0.1
```

The script supports:
- Testing multiple symbols
- Comparing multiple strategies
- Incorporating transaction fees
- Generating detailed reports and visualizations

### Performance Metrics

MercurioAI calculates a range of performance metrics for comprehensive strategy evaluation:

- **Total Return**: Overall percentage gain/loss
- **Annualized Return**: Return normalized to yearly rate
- **Sharpe Ratio**: Risk-adjusted return (volatility)
- **Max Drawdown**: Largest peak-to-trough decline
- **Number of Trades**: Total trade count
- **Win Rate**: Percentage of profitable trades (for strategies that generate specific trades)
- **Equity Curve**: Visualization of performance over time

## Running the Platform

### Demo Mode

To explore the platform's capabilities without real data or trades:

```bash
python run_demo.py
```

This script:
1. Initializes all core services
2. Checks market status and account information
3. Demonstrates loading market data
4. Runs sample backtests with built-in strategies
5. Simulates paper trades

### Paper Trading

For testing with real market data but no actual money:

```bash
# Ensure your .env has ALPACA_KEY and ALPACA_SECRET configured
# and ALPACA_BASE_URL set to paper trading URL

# Run with default configuration
python run_paper_trading.py

# Specify a strategy
python run_paper_trading.py --strategy MovingAverageStrategy
```

### Live Trading

For live trading with real funds:

```bash
# Ensure your .env has ALPACA_KEY and ALPACA_SECRET configured
# and ALPACA_BASE_URL set to live trading URL

# Run with caution!
python run_live_trading.py --risk_limit 0.02 --symbols AAPL,MSFT
```

**Important**: Always thoroughly test strategies in paper mode before deploying with real capital.

## Troubleshooting

### Common Issues

**API Connection Problems**:
- Verify your API keys are correct in the `.env` file
- Check if you're hitting rate limits
- Ensure your account has the proper permissions

**Data Issues**:
- Check internet connectivity
- Verify the requested symbol exists
- Ensure date ranges are valid

**Strategy Errors**:
- Confirm inputs match expected format
- Check for NaN values in your data
- Ensure sufficient data for indicators (e.g., enough bars for moving averages)

### Logging

MercurioAI uses Python's standard logging module. Set the `LOG_LEVEL` in your `.env` file:

```
LOG_LEVEL=DEBUG  # Options: DEBUG, INFO, WARNING, ERROR
```

Logs are written to the `./logs` directory with separate files for different components.

## FAQ

**Q: Do I need paid API keys to use MercurioAI?**

A: No, the platform supports free data sources like Yahoo Finance and has fallback mechanisms. However, for the best experience and reliable real-time data, we recommend using Polygon.io or Alpaca.

**Q: Can I use MercurioAI with cryptocurrency exchanges?**

A: Currently, the platform focuses on stocks through Alpaca, but the modular architecture allows for adding cryptocurrency exchange support.

**Q: How much historical data do I need for the LSTM strategy?**

A: For optimal results, we recommend at least 1-2 years of data, although the LSTM strategy can work with less. More data generally leads to better model training.

**Q: Can I deploy MercurioAI on a cloud server?**

A: Yes, the platform supports deployment on any Python-compatible environment. For production use, we recommend using Docker with the provided Dockerfile and docker-compose configuration.

**Q: How do I add a new data provider?**

A: Create a new provider class in `app/services/providers/` that implements the `MarketDataProvider` interface, then register it in the provider factory.

## Contributing

Contributions to MercurioAI are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature-name`)
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

For major changes, please open an issue first to discuss what you would like to change.

---

## Disclaimer

This software is for educational and research purposes only. Trading involves risk of financial loss. Always conduct thorough testing and consider consulting with a financial advisor before trading with real capital.

---

*Last updated: April 25, 2025*
