# Chapter 4: Paper Trading Basics

Welcome to Chapter 4! In this chapter, we'll explore paper trading with Mercurio AI, which allows you to practice trading without risking real money. This is an essential step before moving to live trading.

## What is Paper Trading?

Paper trading (also called virtual trading or demo trading) is a simulation of real trading that lets you:

- Practice trading strategies in real-time
- Test your trading ideas with current market data
- Experience the emotional aspects of trading
- Build confidence without financial risk
- Fine-tune strategies before using real money

With Mercurio AI, paper trading uses the same code and infrastructure as real trading, just without actual money changing hands.

## Setting Up Paper Trading

Setting up paper trading in Mercurio AI is straightforward:

```python
from app.services.trading import TradingService

# Initialize in paper trading mode with $10,000 starting capital
trading_service = TradingService(
    mode="paper",
    initial_capital=10000,
    commission=0.001  # 0.1% commission per trade (optional)
)
```

The paper trading environment includes:

- Virtual balance (initial_capital)
- Position tracking
- Order management
- Optional simulated commissions and slippage
- Performance metrics

## Running Your First Paper Trading Session

Let's create a simple script that runs a complete paper trading session using a moving average strategy:

```python
"""
Paper Trading Example
This script runs a paper trading session with the Moving Average strategy.
"""
import asyncio
import pandas as pd
import time
from app.strategies.moving_average import MovingAverageStrategy
from app.services.market_data import MarketDataService
from app.services.trading import TradingService

async def paper_trading_session():
    # 1. Initialize services
    market_data = MarketDataService()
    trading = TradingService(mode="paper", initial_capital=10000)
    
    # 2. Create strategy
    strategy = MovingAverageStrategy(short_window=10, long_window=30)
    
    # 3. Set trading parameters
    symbol = "AAPL"
    position_size = 0.2  # Use 20% of capital per position
    
    # 4. Main trading loop
    print(f"Starting paper trading session for {symbol}")
    print(f"Initial capital: ${trading.get_balance():.2f}")
    
    try:
        for i in range(10):  # Run for 10 iterations (in real use, this would run continuously)
            # Get latest data
            data = await market_data.get_historical_data(
                symbol=symbol,
                lookback_days=60  # Get 60 days of data for analysis
            )
            
            # Preprocess data for strategy
            processed_data = await strategy.preprocess_data(data)
            
            # Get trading signal
            signal, confidence = await strategy.predict(processed_data)
            
            print(f"\nIteration {i+1}:")
            print(f"Current price: ${processed_data['close'].iloc[-1]:.2f}")
            print(f"Signal: {signal}, Confidence: {confidence:.2f}")
            
            # Execute trade based on signal
            if signal == "BUY" and not trading.has_position(symbol):
                # Calculate quantity based on position size
                price = processed_data['close'].iloc[-1]
                available_capital = trading.get_balance() * position_size
                quantity = int(available_capital / price)
                
                if quantity > 0:
                    order = await trading.place_order(
                        symbol=symbol,
                        quantity=quantity,
                        side="buy",
                        order_type="market"
                    )
                    print(f"Bought {quantity} shares at ${price:.2f}")
            
            elif signal == "SELL" and trading.has_position(symbol):
                # Sell entire position
                position = trading.get_position(symbol)
                order = await trading.place_order(
                    symbol=symbol,
                    quantity=position.quantity,
                    side="sell",
                    order_type="market"
                )
                print(f"Sold {position.quantity} shares at ${processed_data['close'].iloc[-1]:.2f}")
            
            else:
                print("No action taken")
            
            # Print current portfolio status
            print(f"Current balance: ${trading.get_balance():.2f}")
            print(f"Current positions: {trading.get_positions()}")
            print(f"Portfolio value: ${trading.get_portfolio_value():.2f}")
            
            # In a real scenario, we would wait for market updates
            # Here we'll just wait a second to simulate time passing
            await asyncio.sleep(1)
        
        # Final portfolio summary
        print("\n--- Final Portfolio Summary ---")
        print(f"Starting capital: $10,000.00")
        print(f"Final portfolio value: ${trading.get_portfolio_value():.2f}")
        print(f"Total return: {(trading.get_portfolio_value() / 10000 - 1) * 100:.2f}%")
        print(f"Positions: {trading.get_positions()}")
        
    except Exception as e:
        print(f"Error during paper trading session: {e}")

if __name__ == "__main__":
    asyncio.run(paper_trading_session())
```

Save this as `paper_trading_example.py` and run it to see paper trading in action.

### Understanding the Paper Trading Process

The paper trading process involves several key steps:

1. **Initialization**: Set up market data and trading services
2. **Strategy Creation**: Define the trading strategy to use
3. **Trading Loop**: Continuously fetch data, get signals, and execute trades
4. **Position Management**: Track and manage open positions
5. **Performance Monitoring**: Track balance, portfolio value, and returns

## Paper Trading Best Practices

To get the most out of paper trading with Mercurio AI, follow these best practices:

### 1. Treat Paper Trading as Real

The most common mistake with paper trading is not taking it seriously. To avoid this:

- Set a realistic initial capital amount
- Stick to your trading plan and rules
- Include commissions and slippage in your simulation
- Note your emotional responses to trades

### 2. Use Realistic Position Sizing

Position sizing is crucial in real trading, so practice it in paper trading:

```python
# Calculate position size based on risk percentage
def calculate_position_size(capital, risk_percentage, entry_price, stop_loss_price):
    risk_amount = capital * (risk_percentage / 100)
    risk_per_share = abs(entry_price - stop_loss_price)
    shares = int(risk_amount / risk_per_share)
    return shares
```

### 3. Monitor Multiple Metrics

Don't just focus on returns. Track multiple performance metrics:

- Win/loss ratio
- Average win vs. average loss
- Maximum drawdown
- Sharpe ratio
- Number of trades

### 4. Test Different Market Conditions

Markets behave differently in various conditions. Test your strategy across:

- Bull markets (uptrends)
- Bear markets (downtrends)
- Sideways/ranging markets
- High volatility periods
- Low volatility periods

Mercurio AI's data generation capabilities make this easy:

```python
# Generate data for different market conditions
async def test_different_markets():
    market_data = MarketDataService()
    
    # Test in bull market
    bull_data = await market_data.get_historical_data(
        symbol="AAPL",
        start_date="2020-03-23",  # Start of a strong bull market
        end_date="2021-01-01"
    )
    
    # Test in bear market
    bear_data = await market_data.get_historical_data(
        symbol="AAPL",
        start_date="2022-01-01",
        end_date="2022-06-15"  # Bear market period
    )
    
    # Test in sideways market
    sideways_data = await market_data.get_historical_data(
        symbol="AAPL",
        start_date="2023-05-01",
        end_date="2023-08-01"  # Period of consolidation
    )
```

### 5. Journal Your Paper Trades

Keep a trading journal during paper trading:

```python
# Simple trade journaling
def log_trade(signal, price, quantity, reason, confidence):
    with open("trading_journal.csv", "a") as f:
        timestamp = pd.Timestamp.now()
        f.write(f"{timestamp},{signal},{price},{quantity},{reason},{confidence}\n")
```

## Advanced Paper Trading Features

Mercurio AI offers several advanced paper trading features:

### Multi-Asset Trading

Trade multiple assets simultaneously:

```python
async def multi_asset_paper_trading():
    market_data = MarketDataService()
    trading = TradingService(mode="paper", initial_capital=10000)
    
    # Create strategies for different assets
    aapl_strategy = MovingAverageStrategy(short_window=10, long_window=30)
    btc_strategy = LSTMPredictorStrategy(lookback_periods=30)
    
    # Trading universe
    symbols = ["AAPL", "BTC-USD"]
    strategies = {
        "AAPL": aapl_strategy,
        "BTC-USD": btc_strategy
    }
    
    # Main loop
    for i in range(10):
        for symbol in symbols:
            # Get data for this asset
            data = await market_data.get_historical_data(symbol=symbol, lookback_days=60)
            
            # Get signal from the appropriate strategy
            strategy = strategies[symbol]
            processed_data = await strategy.preprocess_data(data)
            signal, confidence = await strategy.predict(processed_data)
            
            # Execute trade
            # ... trading logic here ...
```

### Scheduled Trading

Run paper trading on a schedule:

```python
import schedule
import time

def scheduled_trading_job():
    # Run the trading logic
    asyncio.run(paper_trading_session())

# Schedule trading at market open (9:30 AM Eastern)
schedule.every().monday.at("09:30").do(scheduled_trading_job)
schedule.every().tuesday.at("09:30").do(scheduled_trading_job)
schedule.every().wednesday.at("09:30").do(scheduled_trading_job)
schedule.every().thursday.at("09:30").do(scheduled_trading_job)
schedule.every().friday.at("09:30").do(scheduled_trading_job)

# Run the scheduler
while True:
    schedule.run_pending()
    time.sleep(1)
```

### Different Trade Types

Experiment with different order types:

```python
# Market order (immediate execution at market price)
await trading.place_order(symbol="AAPL", quantity=10, side="buy", order_type="market")

# Limit order (execution only at specified price or better)
await trading.place_order(
    symbol="AAPL",
    quantity=10,
    side="buy",
    order_type="limit",
    limit_price=150.00
)

# Stop order (becomes market order when price reaches stop_price)
await trading.place_order(
    symbol="AAPL",
    quantity=10,
    side="sell",
    order_type="stop",
    stop_price=145.00
)
```

## Analyzing Paper Trading Performance

After running paper trading sessions, analyze your performance:

```python
from app.analysis.performance import analyze_trading_performance

# Analyze performance from trading service
def analyze_paper_trading_results(trading_service):
    # Get trade history
    trades = trading_service.get_trade_history()
    
    # Get equity curve
    equity_curve = trading_service.get_equity_curve()
    
    # Calculate performance metrics
    performance = analyze_trading_performance(trades, equity_curve)
    
    print("--- Performance Analysis ---")
    print(f"Total Return: {performance['total_return']:.2f}%")
    print(f"Annualized Return: {performance['annualized_return']:.2f}%")
    print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {performance['max_drawdown']:.2f}%")
    print(f"Win Rate: {performance['win_rate']:.2f}%")
    print(f"Profit Factor: {performance['profit_factor']:.2f}")
    
    # Plot equity curve
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve)
    plt.title("Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True)
    plt.show()
```

## From Paper Trading to Live Trading

When you're ready to transition from paper to live trading, the process is simple in Mercurio AI:

```python
# Paper trading
paper_trading = TradingService(mode="paper", initial_capital=10000)

# Live trading (when you're ready)
live_trading = TradingService(
    mode="live",
    broker="alpaca",  # Example broker
    api_key="your_api_key",
    api_secret="your_api_secret"
)
```

The beauty of Mercurio AI's design is that your strategy code doesn't change when moving from paper to live trading. The same strategy that worked in paper trading can be used directly for live trading.

## Next Steps

Now that you understand how to use paper trading in Mercurio AI, you're ready to explore data sources and management in more detail. In the next chapter, we'll dive into how Mercurio AI handles market data from various sources.

Continue to [Chapter 5: Data Sources and Management](./05-data-management.md) to learn about working with market data in Mercurio AI.

---

**Key Takeaways:**
- Paper trading allows you to practice trading without financial risk
- Setting up paper trading in Mercurio AI is as simple as specifying "paper" mode
- The paper trading process involves data acquisition, signal generation, and trade execution
- Best practices include treating paper trading seriously and using realistic position sizing
- Advanced features include multi-asset trading, scheduled trading, and various order types
- The transition from paper to live trading is seamless due to Mercurio AI's consistent API
