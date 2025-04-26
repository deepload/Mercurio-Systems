# Chapter 3: Understanding the Platform

Welcome to Chapter 3! Now that you're set up and have run your first script, it's time to understand the core components of Mercurio AI in more detail. This chapter will give you a comprehensive overview of the platform's architecture and how the different parts work together.

## The Mercurio AI Architecture

Mercurio AI follows a modular, service-oriented architecture. This design provides several advantages:

- **Flexibility**: Components can be used independently or together
- **Extensibility**: Easy to add new features or strategies
- **Resilience**: Multiple fallback mechanisms ensure the system keeps working
- **Scalability**: Components can be scaled independently based on needs

Let's explore each of the major components in detail:

## 1. Market Data Service

The Market Data Service is your gateway to financial data. It's responsible for:

- Fetching historical price data
- Accessing real-time market information
- Preprocessing data for analysis
- Providing fallback data when needed

### Key Features

- **Multiple Data Providers**: Supports various data sources with automatic selection
- **Transparent Fallback**: Seamlessly switches to alternative sources when needed
- **Sample Data Generation**: Creates realistic data for testing when no external data is available
- **Normalized Interface**: Consistent data format regardless of the source

### How it Works

```python
from app.services.market_data import MarketDataService

# Initialize the service
market_data = MarketDataService()

# Get historical data with automatic fallback
async def get_data():
    data = await market_data.get_historical_data(
        symbol="BTC-USD",
        start_date="2024-01-01",
        end_date="2024-04-25",
        timeframe="1d"  # Daily timeframe
    )
    return data
```

The service follows this sequence when retrieving data:

1. Try configured primary data provider (e.g., paid API)
2. If that fails, try free alternative providers
3. If all external sources fail, generate sample data
4. Normalize the data to a consistent pandas DataFrame format

### Data Format

Regardless of the source, data is always provided in a standardized format:

| Column    | Description                    | Example       |
|-----------|--------------------------------|---------------|
| timestamp | Date and time of the data point| 2024-01-01    |
| open      | Opening price                  | 42,000.00     |
| high      | Highest price in the period    | 43,500.00     |
| low       | Lowest price in the period     | 41,800.00     |
| close     | Closing price                  | 43,200.00     |
| volume    | Trading volume                 | 28,500        |

## 2. Trading Service

The Trading Service handles all aspects of order execution and management. It's responsible for:

- Placing buy and sell orders
- Tracking open positions
- Managing order execution
- Simulating trades in paper trading mode

### Key Features

- **Paper Trading**: Simulated trading for risk-free practice
- **Live Trading**: Real-money trading with supported brokers
- **Order Types**: Market, limit, stop, and other order types
- **Position Tracking**: Keeps track of current holdings and performance

### How it Works

```python
from app.services.trading import TradingService

# Initialize in paper trading mode
trading = TradingService(mode="paper")

# Place an order
async def buy_stock():
    order = await trading.place_order(
        symbol="AAPL",
        quantity=10,
        side="buy",
        order_type="market"
    )
    return order
```

The trading service operates in two primary modes:

1. **Paper Trading Mode**: Simulates trades without using real money
   - Great for testing strategies
   - Uses current market prices for simulation
   - Tracks theoretical positions and performance

2. **Live Trading Mode**: Executes real trades with actual money
   - Connects to supported brokers
   - Requires API keys and proper configuration
   - Includes additional safety checks

## 3. Backtesting Service

The Backtesting Service allows you to test trading strategies against historical data. It's responsible for:

- Simulating strategy performance on past data
- Calculating performance metrics
- Visualizing results
- Comparing different strategies

### Key Features

- **Historical Simulation**: Tests how strategies would have performed in the past
- **Performance Metrics**: Calculates key indicators like returns, drawdowns, and Sharpe ratio
- **Trade Analysis**: Detailed breakdown of individual trades
- **Visualization**: Charts and graphs to understand performance

### How it Works

Most strategies in Mercurio AI include their own backtesting methods, which leverage the Backtesting Service:

```python
from app.strategies.moving_average import MovingAverageStrategy

# Create a strategy
strategy = MovingAverageStrategy(short_window=10, long_window=30)

# Run a backtest
async def backtest_strategy(data):
    # Preprocess the data for the strategy
    processed_data = await strategy.preprocess_data(data)
    
    # Run the backtest
    result = await strategy.backtest(
        data=processed_data,
        initial_capital=10000
    )
    
    return result
```

The backtesting process follows these steps:

1. Preprocess the data (calculate indicators, etc.)
2. Simulate trades based on strategy signals
3. Track portfolio value throughout the simulation
4. Calculate performance metrics
5. Generate visualizations and reports

## 4. Strategy Manager

The Strategy Manager oversees all trading strategies. It's responsible for:

- Managing strategy lifecycle
- Loading and initializing strategies
- Coordinating strategy execution
- Handling strategy-specific configurations

### Key Features

- **Multiple Strategy Types**: From simple to advanced ML-based strategies
- **Strategy Creation**: Tools for creating custom strategies
- **Parameter Management**: Handles strategy-specific settings
- **Optimization**: Tools for finding optimal strategy parameters

### Available Strategy Types

Mercurio AI comes with several built-in strategy types:

1. **Moving Average Strategy**
   - Uses crossovers of short and long-term moving averages
   - Available in both traditional and ML-enhanced versions
   - Simple to understand but effective in trending markets

2. **LSTM Predictor Strategy**
   - Uses Long Short-Term Memory neural networks
   - Good at identifying patterns in time series data
   - More complex but potentially more powerful

3. **Transformer Strategy**
   - Uses transformer neural network architecture
   - Excellent at capturing long-range dependencies
   - State-of-the-art performance for certain assets

4. **LLM Strategy**
   - Leverages Large Language Models for prediction
   - Can incorporate text data and market sentiment
   - Cutting-edge approach to market analysis

### How Strategies Work

All strategies in Mercurio AI follow a common interface:

```python
# General pattern for using any strategy
async def use_strategy(strategy, data):
    # 1. Preprocess the data
    processed_data = await strategy.preprocess_data(data)
    
    # 2. Get a prediction (for the latest data point)
    signal, confidence = await strategy.predict(processed_data)
    
    # 3. Backtest (for historical data)
    backtest_result = await strategy.backtest(processed_data, initial_capital=10000)
    
    return signal, confidence, backtest_result
```

## How Components Work Together

These components work together to create a complete trading system:

1. **Market Data Service** provides data to the strategies
2. **Strategy Manager** uses this data to generate trading signals
3. **Backtesting Service** tests strategies on historical data
4. **Trading Service** executes trades based on strategy signals

### Example Workflow

Here's a typical workflow that shows how these components interact:

```python
import asyncio
from app.services.market_data import MarketDataService
from app.strategies.moving_average import MovingAverageStrategy
from app.services.trading import TradingService

async def run_trading_cycle():
    # Initialize services
    market_data = MarketDataService()
    trading = TradingService(mode="paper")
    
    # Create strategy
    strategy = MovingAverageStrategy(short_window=10, long_window=30)
    
    # Get data
    data = await market_data.get_historical_data("AAPL", "2024-01-01", "2024-04-25")
    
    # Preprocess data
    processed_data = await strategy.preprocess_data(data)
    
    # Get trading signal
    signal, confidence = await strategy.predict(processed_data)
    
    # Execute trade based on signal
    if signal == "BUY":
        await trading.place_order(symbol="AAPL", quantity=10, side="buy")
    elif signal == "SELL":
        await trading.place_order(symbol="AAPL", quantity=10, side="sell")
    
    print(f"Signal: {signal}, Confidence: {confidence:.2f}")

# Run the trading cycle
asyncio.run(run_trading_cycle())
```

## Fallback Mechanisms in Detail

Mercurio AI's fallback system is one of its most powerful features. Let's look at how it works in more detail:

### Market Data Fallbacks

1. **Primary API** (e.g., paid provider)
   ↓ (if unavailable)
2. **Secondary APIs** (e.g., free alternatives)
   ↓ (if unavailable)
3. **Sample Data Provider** (generates realistic data)

### Trading Fallbacks

1. **Live Trading** (with configured broker)
   ↓ (if unavailable)
2. **Paper Trading** (simulated trading)

### Strategy Fallbacks

1. **Full ML Models** (if enough data and computing resources)
   ↓ (if unavailable)
2. **Simplified Models** (less resource-intensive)
   ↓ (if unavailable)
3. **Traditional Approaches** (non-ML algorithms)

This multi-layered approach ensures you can always work with the platform, regardless of your environment or resources.

## Configuration System

Mercurio AI uses a flexible configuration system that allows you to customize various aspects of the platform:

- **Environment Variables**: For sensitive information like API keys
- **Configuration Files**: For persistent settings
- **Code-Level Configuration**: For runtime adjustments

Here's a simple example of how to configure the Market Data Service:

```python
from app.services.market_data import MarketDataService

# Configure with specific providers
market_data = MarketDataService(
    primary_provider="alpaca",
    api_key="your_api_key",
    api_secret="your_api_secret",
    enable_fallback=True
)
```

## Next Steps

Now that you understand the core components of Mercurio AI, you're ready to start using the platform more effectively. In the next chapter, we'll explore paper trading, which allows you to practice trading strategies without risking real money.

Continue to [Chapter 4: Paper Trading Basics](./04-paper-trading.md) to learn how to simulate trades in a risk-free environment.

---

**Key Takeaways:**
- Mercurio AI consists of four main components: Market Data Service, Trading Service, Backtesting Service, and Strategy Manager
- The Market Data Service provides data with automatic fallbacks to ensure availability
- The Trading Service handles order execution in both paper and live trading modes
- The Backtesting Service allows testing strategies against historical data
- The Strategy Manager oversees multiple strategy types from simple to advanced ML-based
- All components work together through a consistent interface for a complete trading workflow
- The platform's multi-layered fallback system ensures it works in any environment
