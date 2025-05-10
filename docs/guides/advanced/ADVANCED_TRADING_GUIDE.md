# Advanced Trading Guide for MercurioAI

This guide provides detailed instructions for using MercurioAI's trading capabilities, from paper trading simulation to live trading with real capital. It includes best practices, risk management strategies, and guidance on accounting for transaction costs.

## Table of Contents

1. [Trading Overview](#trading-overview)
2. [Paper Trading](#paper-trading)
3. [Live Trading](#live-trading)
4. [Transaction Costs](#transaction-costs)
5. [Strategy Training and Validation](#strategy-training-and-validation)
6. [Risk Management](#risk-management)
7. [Monitoring and Logging](#monitoring-and-logging)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

## Trading Overview

MercurioAI supports two primary trading modes:

- **Paper Trading**: Simulated trading using real market data but without actual money
- **Live Trading**: Real trading using actual capital through Alpaca

Both modes use the same underlying strategy engine, allowing for consistent behavior across simulation and live environments.

## Paper Trading

Paper trading allows you to test your strategies with real-time market data without risking actual capital. This is an essential step before deploying any strategy to live trading.

### Setup for Paper Trading

1. **Configure Alpaca API Keys**:
   
   In your `.env` file, configure both paper and live trading credentials:

   ```
   # Paper trading configuration
   ALPACA_PAPER_KEY=your_paper_key_here
   ALPACA_PAPER_SECRET=your_paper_secret_here
   ALPACA_PAPER_URL=https://paper-api.alpaca.markets
   
   # Live trading configuration
   ALPACA_LIVE_KEY=your_live_key_here
   ALPACA_LIVE_SECRET=your_live_secret_here
   ALPACA_LIVE_URL=https://api.alpaca.markets
   
   # Data API URL for both modes
   ALPACA_DATA_URL=https://data.alpaca.markets
   
   # Set the active trading mode (paper or live)
   ALPACA_MODE=paper
   ```

   You can obtain paper trading keys by signing up at [Alpaca](https://app.alpaca.markets/signup).

2. **Create run_paper_trading.py**:

   We'll create a dedicated script for paper trading that uses the same structure as `run_live_trading.py` but with paper trading mode enabled:

   ```python
   # Create a copy of run_live_trading.py named run_paper_trading.py
   # Then modify the TradingService initialization to use paper trading:
   
   # The trading service now automatically detects the mode from ALPACA_MODE in .env
   # So both scripts use the same initialization:
   self.trading_service = TradingService()

   # To check the current mode:
   print(f"Current trading mode: {self.trading_service.get_trading_mode()}")
   
   # Also modify the confirmation prompt:
   confirmation = input("Type 'CONFIRM' to start paper trading or anything else to abort: ")
   ```

3. **Run Paper Trading**:

   ```bash
   python run_paper_trading.py --strategy MovingAverageStrategy --symbols AAPL,MSFT,GOOGL
   ```

### Paper Trading with Transaction Costs

To make your paper trading more realistic, you should incorporate transaction costs. Here's how to simulate them:

```python
# Add this to the execute_trading_signal method in your run_paper_trading.py script

async def execute_trading_signal(self, symbol: str, action, confidence: float):
    # ...existing code...
    
    # Execute the trade
    result = await self.trading_service.execute_trade(
        symbol=symbol,
        action=action,
        quantity=quantity,
        strategy_name=self.strategy_name
    )
    
    # Apply simulated transaction costs (add after successful trade)
    if result.get("status") == "success":
        # Typical costs: percentage fee + fixed fee
        percentage_fee = 0.0010  # 0.1% fee (adjust based on your broker)
        fixed_fee = 0.0  # Fixed fee per trade
        min_fee = 0.0  # Minimum fee per trade
        
        # Calculate the transaction cost
        price = float(result.get("order", {}).get("filled_avg_price", 0))
        trade_value = price * quantity
        cost = max(min_fee, fixed_fee + (trade_value * percentage_fee))
        
        # Log the transaction cost
        logger.info(f"Applied simulated transaction cost: ${cost:.2f} for {symbol}")
        
        # You could track these costs in a separate variable for performance analysis
        # self.total_transaction_costs += cost
```

## Live Trading

Live trading executes strategies with real money. Only proceed to this step after thorough backtesting and successful paper trading.

### Prerequisites for Live Trading

1. **Funded Alpaca Account**: You need a funded Alpaca brokerage account.

2. **Live API Keys**: Obtain live API keys from Alpaca.

3. **Configure Live API Keys**:
   
   In your `.env` file:

   ```
   ALPACA_KEY=your_live_alpaca_key_here
   ALPACA_SECRET=your_live_alpaca_secret_here
   ALPACA_BASE_URL=https://api.alpaca.markets
   ```

### Running Live Trading

The `run_live_trading.py` script handles live trading with several safety mechanisms:

```bash
# Basic usage
python run_live_trading.py

# With specific strategy and symbols
python run_live_trading.py --strategy MovingAverageStrategy --symbols AAPL,MSFT,GOOGL

# With risk management parameters
python run_live_trading.py --risk_limit 0.01 --interval 300
```

### Command Line Options

The script supports several command line parameters:

| Parameter       | Description                                           | Default              |
|-----------------|-------------------------------------------------------|----------------------|
| `--strategy`    | Strategy name to use                                  | MovingAverageStrategy|
| `--params`      | JSON string of strategy parameters                    | {}                   |
| `--symbols`     | Comma-separated list of symbols to trade              | AAPL,MSFT,GOOGL      |
| `--risk_limit`  | Maximum percentage of portfolio per position (0-1)    | 0.02 (2%)            |
| `--interval`    | Check interval in seconds                             | 60                   |
| `--lookback`    | Lookback period in days for historical data           | 90                   |
| `--config`      | Path to JSON configuration file                       | None                 |

### Configuration File

For more complex setups, use a JSON configuration file:

```json
{
    "strategy": "MovingAverageStrategy",
    "strategy_params": {
        "short_window": 20,
        "long_window": 50,
        "use_ml": true
    },
    "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN"],
    "risk_limit": 0.01,
    "check_interval": 300,
    "data_lookback": 120,
    "transaction_costs": {
        "percentage_fee": 0.0010,
        "fixed_fee": 0.0,
        "min_fee": 0.0
    }
}
```

Then run with:

```bash
python run_live_trading.py --config my_trading_config.json
```

## Transaction Costs

Understanding and accounting for transaction costs is crucial for realistic performance assessment.

### Types of Transaction Costs

1. **Percentage Fee**: A percentage of the trade value (e.g., 0.1%)
2. **Fixed Fee**: A flat fee per trade (e.g., $1.00)
3. **Minimum Fee**: A minimum fee threshold (e.g., at least $1.00 per trade)
4. **Spread Cost**: The difference between bid and ask prices
5. **Slippage**: The difference between expected execution price and actual execution price

### Simulating Transaction Costs in Backtesting

```python
# Example of incorporating transaction costs in backtesting
def apply_transaction_costs(results, percentage_fee=0.001, fixed_fee=0.0, min_fee=0.0):
    """Apply transaction costs to backtest results"""
    data = results["backtest_data"].copy()
    trades = (data['position'].diff() != 0).sum()
    
    # Calculate average trade size
    avg_position_value = results["initial_capital"] * results["risk_limit"] if "risk_limit" in results else results["initial_capital"] * 0.1
    
    # Calculate total cost
    total_cost = 0
    for i in range(1, len(data)):
        # Check if a trade occurred
        if data['position'].iloc[i] != data['position'].iloc[i-1]:
            # Calculate trade value
            trade_value = avg_position_value
            # Calculate cost for this trade
            cost = max(min_fee, fixed_fee + (trade_value * percentage_fee))
            total_cost += cost
    
    # Adjust final capital and returns
    adjusted_final_capital = results["final_capital"] - total_cost
    adjusted_total_return = (adjusted_final_capital / results["initial_capital"]) - 1
    
    # Return adjusted results
    return {
        **results,
        "transaction_costs": total_cost,
        "adjusted_final_capital": adjusted_final_capital,
        "adjusted_total_return": adjusted_total_return,
        "original_final_capital": results["final_capital"],
        "original_total_return": results["total_return"]
    }
```

### Adding Transaction Costs to Long-Term Backtesting

In `long_term_backtest.py`, you can modify the script to include transaction costs:

```python
# Add this to your command line parameters
parser.add_argument("--fee_percentage", type=float, default=0.001,
                   help="Percentage fee per trade (e.g., 0.001 for 0.1%)")
parser.add_argument("--fee_fixed", type=float, default=0.0,
                   help="Fixed fee per trade")
parser.add_argument("--fee_minimum", type=float, default=0.0,
                   help="Minimum fee per trade")

# Then apply these costs to each backtest result
backtest_results = await backtesting_service.run_backtest(...)
adjusted_results = apply_transaction_costs(
    backtest_results,
    percentage_fee=args.fee_percentage,
    fixed_fee=args.fee_fixed,
    min_fee=args.fee_minimum
)
```

## Strategy Training and Validation

### Training Machine Learning Strategies

ML strategies like the LSTM Predictor require proper training before deployment:

```python
async def train_strategy(strategy_name, symbols, start_date, end_date):
    """Train a strategy on historical data before deployment"""
    from app.services.market_data import MarketDataService
    from app.services.strategy_manager import StrategyManager
    
    market_data = MarketDataService()
    strategy_manager = StrategyManager()
    
    # Get strategy instance
    strategy = await strategy_manager.get_strategy(strategy_name)
    
    for symbol in symbols:
        # Get historical data
        data = await market_data.get_historical_data(symbol, start_date, end_date)
        
        # Preprocess data
        processed_data = await strategy.preprocess_data(data)
        
        # Train strategy
        training_metrics = await strategy.train(processed_data)
        
        print(f"Trained {strategy_name} on {symbol}:")
        print(f"  Metrics: {training_metrics}")
        
    # Save the trained model
    model_path = await strategy.save_model("./models")
    print(f"Model saved to: {model_path}")
    
    return strategy
```

### Walk-Forward Validation

For more robust validation, implement walk-forward testing:

```python
async def walk_forward_validation(strategy_name, symbol, start_date, end_date, window_size=90, step_size=30):
    """Perform walk-forward validation"""
    from app.services.market_data import MarketDataService
    from app.services.strategy_manager import StrategyManager
    from app.services.backtesting import BacktestingService
    
    market_data = MarketDataService()
    strategy_manager = StrategyManager()
    backtesting = BacktestingService()
    
    current_date = start_date
    results = []
    
    while current_date + timedelta(days=window_size) <= end_date:
        train_start = current_date
        train_end = current_date + timedelta(days=window_size)
        test_start = train_end
        test_end = min(test_start + timedelta(days=step_size), end_date)
        
        # Get strategy instance
        strategy = await strategy_manager.get_strategy(strategy_name)
        
        # Get training data
        train_data = await market_data.get_historical_data(symbol, train_start, train_end)
        processed_train_data = await strategy.preprocess_data(train_data)
        
        # Train strategy
        await strategy.train(processed_train_data)
        
        # Test on out-of-sample data
        test_results = await backtesting.run_backtest(
            strategy=strategy,
            symbol=symbol,
            start_date=test_start,
            end_date=test_end
        )
        
        # Apply transaction costs
        adjusted_results = apply_transaction_costs(test_results)
        
        results.append({
            "train_period": (train_start, train_end),
            "test_period": (test_start, test_end),
            "results": adjusted_results
        })
        
        # Move to next window
        current_date += timedelta(days=step_size)
    
    return results
```

## Risk Management

Effective risk management is crucial for successful trading.

### Position Sizing

The `--risk_limit` parameter controls position sizing:

```
# Limit each position to 1% of portfolio
python run_live_trading.py --risk_limit 0.01
```

### Setting Stop Losses

Add stop-loss functionality to your trading script:

```python
# Add this parameter to your script
parser.add_argument("--stop_loss_pct", type=float, default=0.02,
                   help="Stop loss percentage (e.g., 0.02 for 2%)")

# Add this method to the LiveTrader class
async def place_stop_loss(self, symbol, entry_price, quantity, side):
    """Place a stop loss order"""
    stop_price = entry_price * (1 - self.stop_loss_pct) if side == "buy" else entry_price * (1 + self.stop_loss_pct)
    
    # Round stop price to appropriate decimal
    stop_price = round(stop_price, 2)
    
    try:
        # Place the stop order
        stop_order = self.trading_service.alpaca_client.submit_order(
            symbol=symbol,
            qty=quantity,
            side='sell' if side == 'buy' else 'buy',
            type='stop',
            stop_price=stop_price,
            time_in_force='gtc'
        )
        
        logger.info(f"Placed stop loss for {symbol} at ${stop_price}")
        return stop_order
    except Exception as e:
        logger.error(f"Error placing stop loss: {str(e)}")
        return None
```

### Diversification

Trade multiple symbols to diversify risk:

```bash
python run_live_trading.py --symbols AAPL,MSFT,GOOGL,AMZN,V,JNJ
```

## Monitoring and Logging

MercurioAI includes comprehensive logging for monitoring trading activity.

### Log Files

The trading scripts log to the `logs` directory:

- `logs/live_trading.log`: Records all live trading activity
- `logs/paper_trading.log`: Records paper trading simulation

### Adding Email Notifications

Add email notifications for important events:

```python
import smtplib
from email.mime.text import MIMEText

async def send_notification(subject, message):
    """Send email notification for important trading events"""
    sender = "your_email@example.com"
    recipient = "your_email@example.com"
    password = os.getenv("EMAIL_PASSWORD")
    
    if not password:
        logger.error("Email password not set in environment variables")
        return
    
    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = recipient
    
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)
        server.quit()
        logger.info(f"Notification sent: {subject}")
    except Exception as e:
        logger.error(f"Failed to send notification: {str(e)}")
```

Then call this function for important events:

```python
# For significant trades
await send_notification(
    f"MercurioAI: New {action.name} Order for {symbol}",
    f"Executed {action.name} of {quantity} {symbol} at ${price:.2f}"
)

# For large P&L changes
await send_notification(
    "MercurioAI: Significant P&L Change",
    f"Current P&L: ${unrealized_pl:.2f} ({unrealized_plpc*100:.2f}%)"
)
```

## Best Practices

### Before Going Live

1. **Validate with Historical Data**: Always backtest extensively before deploying.

2. **Paper Trade First**: Run strategies in paper trading mode for at least 1-2 months.

3. **Start Small**: Begin with smaller position sizes than you eventually plan to use.

4. **Monitor Carefully**: Check results daily and be prepared to intervene.

### Production Operations

1. **Use Dedicated Hardware**: Run your trading system on a reliable, always-on server.

2. **Implement Redundancy**: Consider multiple internet connections and power backup.

3. **Monitor System Health**: Track CPU, memory, and network performance.

4. **Review Performance Daily**: Analyze trades and overall strategy performance.

### Example Trading Workflow

1. **Develop Strategy**:
   ```bash
   # Implement your strategy in app/strategies/
   ```

2. **Backtest**:
   ```bash
   python long_term_backtest.py --strategy YourStrategy --symbols AAPL,MSFT
   ```

3. **Paper Trade**:
   ```bash
   python run_paper_trading.py --strategy YourStrategy --symbols AAPL,MSFT --risk_limit 0.01
   ```

4. **Review Results**:
   Analyze logs, charts, and performance metrics

5. **Go Live (if results are good)**:
   ```bash
   python run_live_trading.py --strategy YourStrategy --symbols AAPL,MSFT --risk_limit 0.01
   ```

## Troubleshooting

### Common Issues

#### API Connection Problems

**Issue**: `Error checking market status: APIError: 401 Client Error`
**Solution**: Check your API keys in the `.env` file. Ensure they have appropriate permissions.

#### Strategy Prediction Errors

**Issue**: `Error in strategy prediction: NoneType object has no attribute 'iloc'`
**Solution**: Ensure your data preprocessing is handling null values correctly.

#### Order Execution Failures

**Issue**: `Order execution failed: insufficient buying power`  
**Solution**: Check your account balance and reduce position sizes.

#### Data Provider Issues

**Issue**: `No data received for symbol`  
**Solution**: MercurioAI will automatically try alternative data providers. Check that at least one is configured.

### Getting Help

For additional assistance:

1. Check the logs in the `logs/` directory
2. Review the MercurioAI documentation
3. Examine the exception traceback for specific error information

---

## Disclaimer

Trading involves significant risk of loss. MercurioAI is provided for educational and research purposes only. Always conduct thorough testing and consider consulting a financial advisor before trading with real capital.

---

*Last updated: April 25, 2025*
