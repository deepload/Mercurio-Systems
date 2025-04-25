# MercurioAI

![Version](https://img.shields.io/badge/version-2.1.0-blue)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**MercurioAI** is a powerful, modular Python framework for algorithmic trading across cryptocurrency exchanges and traditional markets.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Strategy System](#strategy-system)
- [Built-in Strategies](#built-in-strategies)
- [Creating Custom Strategies](#creating-custom-strategies)
- [Running the Trading Bot](#running-the-trading-bot)
- [Backtesting](#backtesting)
- [Performance Metrics](#performance-metrics)
- [FAQ & Troubleshooting](#faq--troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

MercurioAI provides a comprehensive framework for developing, testing, and deploying algorithmic trading strategies. It supports multiple exchanges through unified APIs, allowing traders to:

- Automate trading with customizable strategies
- Backtest strategies against historical data
- Analyze performance metrics in real-time
- Implement risk management protocols
- Scale from simple strategies to complex machine learning models

### Use Cases

- **Retail Traders**: Implement personal trading strategies without 24/7 manual monitoring
- **Quantitative Researchers**: Rapidly test trading hypotheses against historical data
- **Fund Managers**: Deploy multiple strategies across various markets simultaneously
- **Data Scientists**: Integrate machine learning models into trading workflows

### Technologies

- **Python 3.9+**: Core programming language
- **ccxt**: Unified API for cryptocurrency exchanges
- **pandas/numpy**: Data manipulation and numerical operations
- **TA-Lib**: Technical analysis indicators
- **scikit-learn**: Machine learning capabilities (optional)
- **Redis**: Message queue for trade execution
- **PostgreSQL/TimescaleDB**: Time-series data storage
- **FastAPI**: REST API for monitoring and control
- **Docker**: Containerization for easy deployment

## Features

- 🔄 **Multi-Exchange Support**: Trade on Binance, Bybit, Coinbase, FTX, and more
- 📊 **Advanced Technical Analysis**: Over 100+ technical indicators through TA-Lib
- 📈 **Backtesting Engine**: Test strategies against historical data with realistic simulation
- 🤖 **Strategy Templates**: Pre-built strategies for common trading approaches
- 📉 **Risk Management**: Position sizing, stop-loss, and risk allocation tools
- 📊 **Performance Analytics**: Detailed metrics and reporting
- 🔌 **Extensible Architecture**: Easy to add new exchanges, indicators, or strategies
- 📱 **Notification System**: Telegram, email, and webhook alerts
- 🔒 **Secure Credential Management**: Environment variables and vault integration

## Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git
- Docker (optional, for containerized deployment)

### Standard Installation

```bash
# Clone the repository
git clone https://github.com/mercurioai/mercurioai.git
cd mercurioai

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt

# Install TA-Lib (may require additional steps based on your OS)
# See: https://github.com/mrjbq7/ta-lib
```

### Docker Installation

```bash
# Clone the repository
git clone https://github.com/mercurioai/mercurioai.git
cd mercurioai

# Build and run the Docker container
docker-compose up -d
```

## Configuration

TradingQuant uses environment variables for configuration. Create a `.env` file in the project root:

```
# Exchange API configuration
EXCHANGE_NAME=binance
EXCHANGE_API_KEY=your_api_key_here
EXCHANGE_API_SECRET=your_api_secret_here
EXCHANGE_TESTNET=True  # Set to False for live trading

# Database configuration
DB_CONNECTION_STRING=postgresql://user:password@localhost:5432/mercurioai

# Trading parameters
DEFAULT_QUOTE_CURRENCY=USDT
POSITION_SIZE_PERCENTAGE=5
MAX_OPEN_TRADES=3
STOP_LOSS_PERCENTAGE=2

# Notification configuration
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

### Exchange API Keys

To obtain API keys:

1. Create an account on your chosen exchange (e.g., Binance, Bybit)
2. Navigate to API Management in your account settings
3. Create a new API key (enable trading permissions, limit IP addresses for security)
4. Copy the API key and secret to your `.env` file
5. Enable testnet/sandbox mode for initial testing

## Strategy System

TradingQuant's strategy system is built around a modular, event-driven architecture that separates:

1. **Data Collection**: Market data retrieval from exchanges
2. **Signal Generation**: Technical analysis and strategy logic
3. **Position Management**: Entry/exit rules and risk parameters
4. **Order Execution**: Interaction with exchange APIs

### Core Concepts

- **Strategy**: A class implementing trading logic with entry/exit rules
- **Indicator**: Technical analysis functions that process price data
- **Exchange**: Wrapper around exchange API for market data and order execution
- **Portfolio**: Tracks open positions and manages risk
- **Backtest**: Simulates strategy performance on historical data

### Strategy Lifecycle

1. **Initialization**: Strategy loads with configuration parameters
2. **Data Retrieval**: Market data is fetched from exchange or database
3. **Indicator Calculation**: Technical indicators are computed
4. **Signal Generation**: Buy/sell signals are created based on indicators
5. **Risk Assessment**: Position size is calculated based on risk parameters
6. **Order Execution**: Orders are placed on the exchange
7. **Position Monitoring**: Open positions are tracked and managed
8. **Performance Tracking**: Results are recorded for analysis

## Built-in Strategies

TradingQuant includes several pre-built strategies that can be used immediately or as templates for custom development.

### 1. Moving Average Crossover

**Description**: Classic strategy based on the crossover of fast and slow moving averages.

**Indicators**:
- Short-term Simple Moving Average (SMA)
- Long-term Simple Moving Average (SMA)

**Parameters**:
- `short_window`: Period for short-term moving average (default: 20)
- `long_window`: Period for long-term moving average (default: 50)

**Logic**:
- **Buy Signal**: When short-term MA crosses above long-term MA
- **Sell Signal**: When short-term MA crosses below long-term MA

**Ideal For**: Trend-following in markets with clear momentum

```python
# Example configuration
{
    "strategy": "MovingAverageCrossover",
    "parameters": {
        "short_window": 10,
        "long_window": 30
    }
}
```

### 2. RSI Reversal

**Description**: Counter-trend strategy based on Relative Strength Index (RSI) overbought/oversold conditions.

**Indicators**:
- Relative Strength Index (RSI)

**Parameters**:
- `rsi_period`: Period for RSI calculation (default: 14)
- `overbought_threshold`: RSI threshold for overbought condition (default: 70)
- `oversold_threshold`: RSI threshold for oversold condition (default: 30)

**Logic**:
- **Buy Signal**: When RSI crosses above oversold threshold from below
- **Sell Signal**: When RSI crosses below overbought threshold from above

**Ideal For**: Range-bound markets with regular reversion to mean

```python
# Example configuration
{
    "strategy": "RSIReversal",
    "parameters": {
        "rsi_period": 14,
        "overbought_threshold": 75,
        "oversold_threshold": 25
    }
}
```

### 3. MACD Momentum

**Description**: Trend and momentum detection using Moving Average Convergence Divergence (MACD).

**Indicators**:
- MACD Line
- Signal Line
- MACD Histogram

**Parameters**:
- `fast_ema`: Period for fast EMA (default: 12)
- `slow_ema`: Period for slow EMA (default: 26)
- `signal_period`: Period for signal line (default: 9)

**Logic**:
- **Buy Signal**: When MACD line crosses above signal line with positive momentum
- **Sell Signal**: When MACD line crosses below signal line with negative momentum

**Ideal For**: Capturing medium-term trends with momentum confirmation

```python
# Example configuration
{
    "strategy": "MACDMomentum",
    "parameters": {
        "fast_ema": 12,
        "slow_ema": 26,
        "signal_period": 9
    }
}
```

### 4. Bollinger Band Breakout

**Description**: Volatility-based strategy using Bollinger Bands to identify breakouts.

**Indicators**:
- Bollinger Bands (Middle, Upper, Lower)

**Parameters**:
- `bb_period`: Period for middle band calculation (default: 20)
- `bb_std_dev`: Standard deviations for bands (default: 2)

**Logic**:
- **Buy Signal**: When price closes above upper band after being below middle band
- **Sell Signal**: When price closes below lower band after being above middle band

**Ideal For**: Volatile markets with strong breakout movements

```python
# Example configuration
{
    "strategy": "BollingerBreakout",
    "parameters": {
        "bb_period": 20,
        "bb_std_dev": 2.5
    }
}
```

## Creating Custom Strategies

Creating your own trading strategy involves subclassing the `BaseStrategy` class and implementing your own signal generation logic.

### Step 1: Create a Strategy File

Create a new Python file in the strategies directory (e.g., `my_strategy.py`):

```python
from mercurioai.core.strategy import BaseStrategy
from mercurioai.indicators import calculate_rsi, calculate_sma

class MyCustomStrategy(BaseStrategy):
    """
    My custom trading strategy combining RSI and SMA
    """
    
    def __init__(self, rsi_period=14, sma_period=50, **kwargs):
        """Initialize strategy with parameters"""
        super().__init__(**kwargs)
        self.rsi_period = rsi_period
        self.sma_period = sma_period
        self.required_history = max(rsi_period, sma_period) + 10  # Data points needed
        
    def preprocess_data(self, dataframe):
        """Calculate indicators needed for the strategy"""
        # Add RSI
        dataframe = calculate_rsi(dataframe, period=self.rsi_period)
        
        # Add SMA
        dataframe = calculate_sma(dataframe, period=self.sma_period, column='close')
        
        return dataframe
        
    def generate_signals(self, dataframe):
        """Generate buy/sell signals based on indicators"""
        signals = {}
        
        # Current candle
        current = dataframe.iloc[-1]
        # Previous candle
        previous = dataframe.iloc[-2]
        
        # Define buy conditions
        price_above_sma = current['close'] > current['sma']
        rsi_oversold_recovery = previous['rsi'] < 30 and current['rsi'] > 30
        
        # Define sell conditions
        price_below_sma = current['close'] < current['sma']
        rsi_overbought_decline = previous['rsi'] > 70 and current['rsi'] < 70
        
        # Generate signals
        if price_above_sma and rsi_oversold_recovery:
            signals['side'] = 'buy'
            signals['confidence'] = min(1.0, (current['rsi'] - 30) / 20)
        elif price_below_sma and rsi_overbought_decline:
            signals['side'] = 'sell'
            signals['confidence'] = min(1.0, (70 - current['rsi']) / 20)
        else:
            signals['side'] = 'hold'
            signals['confidence'] = 0.0
            
        return signals
```

### Step 2: Register Your Strategy

Add your strategy to the `strategy_registry.py` file:

```python
from strategies.my_strategy import MyCustomStrategy

STRATEGY_REGISTRY = {
    # ... existing strategies
    'MyCustomStrategy': MyCustomStrategy,
}
```

### Step 3: Configure Your Strategy

Create a configuration file for your strategy or add parameters to your environment file:

```json
{
    "strategy": "MyCustomStrategy",
    "parameters": {
        "rsi_period": 21,
        "sma_period": 100
    },
    "risk_management": {
        "position_size_percentage": 5,
        "stop_loss_percentage": 3,
        "take_profit_percentage": 9
    }
}
```

### Strategy Development Best Practices

1. **Keep it simple**: Start with a clear concept before adding complexity
2. **Validate on different timeframes**: Ensure strategy works on various timeframes
3. **Test on multiple pairs/assets**: Avoid overfitting to a specific market
4. **Consider transaction costs**: Include realistic fees in your logic
5. **Handle edge cases**: Plan for missing data, extreme volatility, etc.
6. **Document thoroughly**: Comment your code and explain your logic
7. **Use risk management**: Always implement position sizing and stop losses

## Running the Trading Bot

TradingQuant can be run in several modes depending on your needs.

### Command Line Interface

```bash
# Run with default configuration
python -m tradingquant run --config configs/my_strategy_config.json

# Run in paper trading mode
python -m tradingquant run --paper --config configs/my_strategy_config.json

# Run with specific symbols
python -m tradingquant run --symbols BTC/USDT,ETH/USDT --config configs/my_strategy_config.json

# Run with specific timeframe
python -m tradingquant run --timeframe 1h --config configs/my_strategy_config.json
```

### Docker Deployment

```bash
# Run using Docker Compose
docker-compose -f docker-compose.prod.yml up -d

# View logs
docker-compose -f docker-compose.prod.yml logs -f bot
```

### Web Dashboard

TradingQuant includes a web dashboard for monitoring and controlling your trading bot:

1. Start the API server: `python -m tradingquant api`
2. Access the dashboard at `http://localhost:8000`
3. Login with your credentials
4. View active strategies, open positions, and performance metrics
5. Start/stop trading or adjust parameters in real-time

## Backtesting

Backtesting allows you to evaluate strategy performance on historical data before risking real capital.

### Running a Backtest

```bash
# Basic backtest
python -m tradingquant backtest --strategy MyCustomStrategy --symbol BTC/USDT --timeframe 1h --start-date 2023-01-01 --end-date 2023-12-31

# Backtest with custom parameters
python -m tradingquant backtest --config configs/my_backtest_config.json
```

### Backtest Configuration

```json
{
    "strategy": "MyCustomStrategy",
    "parameters": {
        "rsi_period": 14,
        "sma_period": 50
    },
    "backtest_settings": {
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "symbols": ["BTC/USDT", "ETH/USDT"],
        "timeframe": "1h",
        "initial_capital": 10000,
        "fee_percentage": 0.1
    }
}
```

### Analyzing Results

After running a backtest, TradingQuant generates comprehensive performance metrics:

- **Profit/Loss**: Total and percentage returns
- **Sharpe Ratio**: Risk-adjusted return
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Risk/Reward Ratio**: Average profit vs. average loss
- **Monthly/Daily Returns**: Performance breakdown by time period
- **Equity Curve**: Visual representation of capital growth

Results are saved to the `backtest_results` directory with CSV exports and interactive HTML charts.

### Optimization

TradingQuant supports parameter optimization to find the best settings for your strategy:

```bash
python -m tradingquant optimize --strategy MyCustomStrategy --symbol BTC/USDT --parameters rsi_period:7:28:7 sma_period:20:200:20
```

This will test combinations of parameters (RSI periods from 7 to 28 in steps of 7, and SMA periods from 20 to 200 in steps of 20) and rank them by performance.

## Performance Metrics

TradingQuant provides detailed analytics for both backtests and live trading:

- **Profit Factor**: Gross profit divided by gross loss
- **Calmar Ratio**: Annual return divided by maximum drawdown
- **Sortino Ratio**: Return adjusted for downside risk
- **Omega Ratio**: Probability-weighted ratio of gains vs. losses
- **Expectancy**: Expected return per trade
- **Recovery Factor**: Net profit divided by maximum drawdown
- **Trade Analysis**: Duration, frequency, win/loss streaks
- **Exposure**: Time invested in market vs. total time

## FAQ & Troubleshooting

### General Questions

**Q: Is TradingQuant suitable for beginners?**  
A: While TradingQuant is designed to be user-friendly, basic knowledge of Python and trading concepts is recommended. Start with the built-in strategies and tutorials before developing custom strategies.

**Q: Can I use TradingQuant for traditional markets (stocks, forex)?**  
A: Yes, TradingQuant supports traditional markets through brokers with API access. Additional configuration may be required for market-specific features.

**Q: How much capital do I need to start?**  
A: TradingQuant can be used with any amount of capital. Start with paper trading to practice without risk, then begin with a small amount you can afford to lose.

### Technical Issues

**Q: API connection errors with exchange**

```
ERROR: Connection to exchange failed: {'code': -1022, 'msg': 'Signature for this request is not valid.'}
```

**A**: This usually indicates an API key issue. Check:
1. API key and secret are correct in your `.env` file
2. API key has trading permissions enabled
3. IP restrictions match your current IP
4. System time is synchronized (NTP)

**Q: Strategy not generating trades**

**A**: Troubleshoot with these steps:
1. Check logs for warning/error messages
2. Verify your data feed is working (print recent candles)
3. Ensure your strategy parameters aren't too restrictive
4. Confirm minimum order size meets exchange requirements
5. Check for sufficient balance in your account

**Q: "ModuleNotFoundError" when running the bot**

**A**: Make sure:
1. All dependencies are installed: `pip install -r requirements.txt`
2. TA-Lib is properly installed for your OS
3. You're running from the project root directory
4. Your virtual environment is activated

### Best Practices

**Q: How can I prevent significant losses?**  
A: Always use risk management:
1. Set stop-loss orders for all positions
2. Limit position size to a small percentage of your portfolio (1-5%)
3. Diversify across multiple strategies and assets
4. Use the `max_open_trades` setting to limit exposure
5. Start with paper trading to validate strategies

**Q: How often should I update my strategies?**  
A: Markets evolve, so regular review is essential:
1. Monitor performance metrics weekly
2. Re-optimize parameters monthly
3. Review strategy logic quarterly
4. Backtest against recent data before making changes

## Contributing

Contributions to TradingQuant are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add some amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

For major changes, please open an issue first to discuss what you would like to change.

## License

TradingQuant is released under the MIT License. See `LICENSE` file for details.

---

**Disclaimer**: Trading cryptocurrencies and other financial instruments involves significant risk of loss. TradingQuant is provided for educational and informational purposes only. The developers are not responsible for any financial losses incurred while using this software. Always consult a financial advisor before trading with real money.

---

*This documentation was last updated on April 25, 2025.*

## Disclaimer

This documentation was created for the MercurioAI project based on current functionality and planned features. Always refer to the official repository for the most up-to-date information.
