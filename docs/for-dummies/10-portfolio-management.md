# Chapter 10: Portfolio Management

Welcome to Chapter 10! Now that you've learned how to create, backtest, and optimize individual trading strategies, it's time to explore how to combine them into a cohesive portfolio. Portfolio management is crucial for diversification and risk management.

## The Importance of Portfolio Management

Managing a portfolio of strategies gives you several advantages:

- **Diversification**: Reduces risk by spreading investments across multiple strategies and assets
- **Consistent Returns**: Smoothes performance across different market conditions
- **Risk Control**: Manages overall portfolio risk more effectively
- **Opportunity Expansion**: Capitalizes on more market opportunities

## Creating a Multi-Strategy Portfolio

Let's start by creating a portfolio that combines multiple strategies:

```python
import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from app.services.market_data import MarketDataService
from app.portfolio.portfolio_manager import PortfolioManager
from app.strategies.moving_average import MovingAverageStrategy
from app.strategies.lstm_predictor import LSTMPredictorStrategy

async def create_multi_strategy_portfolio():
    """Create a portfolio with multiple strategies."""
    # Initialize market data service
    market_data = MarketDataService()
    
    # Create strategies
    ma_strategy = MovingAverageStrategy(short_window=10, long_window=30)
    lstm_strategy = LSTMPredictorStrategy(sequence_length=30, hidden_units=64)
    
    # Create portfolio manager
    portfolio = PortfolioManager(initial_capital=10000)
    
    # Add strategies with allocation weights
    portfolio.add_strategy(ma_strategy, symbol="AAPL", allocation=0.5)
    portfolio.add_strategy(lstm_strategy, symbol="MSFT", allocation=0.5)
    
    # Get data for backtesting
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    # Run portfolio backtest
    backtest_result = await portfolio.backtest(
        start_date=start_date,
        end_date=end_date,
        market_data_service=market_data
    )
    
    # Print results
    print(f"Portfolio Backtest Results:")
    print(f"Initial Capital: $10,000.00")
    print(f"Final Capital: ${backtest_result['final_equity']:.2f}")
    print(f"Total Return: {((backtest_result['final_equity'] / 10000) - 1) * 100:.2f}%")
    print(f"Sharpe Ratio: {backtest_result['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {backtest_result['max_drawdown']:.2f}%")
    
    # Plot portfolio equity curve
    plt.figure(figsize=(12, 6))
    plt.plot(backtest_result['equity_curve'])
    plt.title('Portfolio Equity Curve')
    plt.xlabel('Time')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    plt.show()
    
    return backtest_result
```

## Understanding the PortfolioManager

The `PortfolioManager` in Mercurio AI handles multiple strategies:

```python
# Creating a portfolio manager
portfolio = PortfolioManager(
    initial_capital=10000,
    rebalance_frequency="monthly",  # How often to rebalance allocations
    risk_management=True            # Enable risk management features
)

# Adding strategies with allocations
portfolio.add_strategy(
    strategy=moving_avg_strategy,
    symbol="AAPL",
    allocation=0.4,                 # 40% of capital
    max_allocation=0.5              # Never allocate more than 50%
)

portfolio.add_strategy(
    strategy=lstm_strategy,
    symbol="MSFT",
    allocation=0.3,
    stop_loss=0.05                  # 5% stop loss
)

portfolio.add_strategy(
    strategy=transformer_strategy,
    symbol="GOOGL",
    allocation=0.3,
    take_profit=0.15                # 15% take profit
)
```

## Multi-Asset Portfolios

You can also create portfolios that trade multiple assets with the same strategy:

```python
async def create_multi_asset_portfolio():
    """Create a portfolio trading multiple assets with the same strategy."""
    # Initialize market data service
    market_data = MarketDataService()
    
    # Create a single strategy
    ma_strategy = MovingAverageStrategy(short_window=10, long_window=30)
    
    # Create portfolio manager
    portfolio = PortfolioManager(initial_capital=10000)
    
    # Add multiple assets with the same strategy
    assets = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    allocation_per_asset = 1.0 / len(assets)
    
    for asset in assets:
        # Create a new instance of the strategy for each asset
        strategy_instance = MovingAverageStrategy(short_window=10, long_window=30)
        portfolio.add_strategy(strategy_instance, symbol=asset, allocation=allocation_per_asset)
    
    # Get data for backtesting
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    # Run portfolio backtest
    backtest_result = await portfolio.backtest(
        start_date=start_date,
        end_date=end_date,
        market_data_service=market_data
    )
    
    # Print results
    print(f"Multi-Asset Portfolio Results:")
    print(f"Initial Capital: $10,000.00")
    print(f"Final Capital: ${backtest_result['final_equity']:.2f}")
    print(f"Total Return: {((backtest_result['final_equity'] / 10000) - 1) * 100:.2f}%")
    
    return backtest_result
```

## Portfolio Allocation Strategies

Different allocation strategies can significantly impact performance:

### Equal Weighting

```python
def equal_weight_allocation(strategies, total_allocation=1.0):
    """Allocate capital equally among strategies."""
    allocation_per_strategy = total_allocation / len(strategies)
    return {strategy: allocation_per_strategy for strategy in strategies}
```

### Performance-Based Weighting

```python
async def performance_based_allocation(strategies, symbols, start_date, end_date, market_data):
    """Allocate based on historical performance."""
    # Test each strategy's performance
    performance = {}
    
    for i, strategy in enumerate(strategies):
        # Get data
        symbol = symbols[i]
        data = await market_data.get_historical_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        # Preprocess and backtest
        processed_data = await strategy.preprocess_data(data)
        backtest_result = await strategy.backtest(
            data=processed_data,
            initial_capital=10000
        )
        
        # Calculate Sharpe ratio
        equity_curve = pd.Series(backtest_result['equity_curve'])
        daily_returns = equity_curve.pct_change().dropna()
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        
        performance[i] = max(sharpe_ratio, 0.01)  # Ensure positive weight
    
    # Normalize to sum to total_allocation
    total_performance = sum(performance.values())
    allocations = {i: (perf / total_performance) for i, perf in performance.items()}
    
    return allocations
```

### Risk Parity Allocation

```python
async def risk_parity_allocation(strategies, symbols, start_date, end_date, market_data):
    """Allocate to achieve equal risk contribution from each strategy."""
    # Calculate volatility for each strategy
    volatilities = {}
    
    for i, strategy in enumerate(strategies):
        # Get data
        symbol = symbols[i]
        data = await market_data.get_historical_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        # Preprocess and backtest
        processed_data = await strategy.preprocess_data(data)
        backtest_result = await strategy.backtest(
            data=processed_data,
            initial_capital=10000
        )
        
        # Calculate volatility
        equity_curve = pd.Series(backtest_result['equity_curve'])
        daily_returns = equity_curve.pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252)
        
        volatilities[i] = max(volatility, 0.001)  # Ensure positive value
    
    # Allocate inversely proportional to volatility
    inverse_vol = {i: 1/vol for i, vol in volatilities.items()}
    total_inverse_vol = sum(inverse_vol.values())
    allocations = {i: (inv_vol / total_inverse_vol) for i, inv_vol in inverse_vol.items()}
    
    return allocations
```

## Dynamic Portfolio Rebalancing

Rebalancing keeps your portfolio aligned with your intended allocations:

```python
async def perform_rebalancing(portfolio, current_positions, target_allocations, market_data):
    """Rebalance portfolio to target allocations."""
    # Get current portfolio value
    total_value = sum(pos['value'] for pos in current_positions.values())
    
    # Calculate target position values
    target_values = {symbol: total_value * alloc for symbol, alloc in target_allocations.items()}
    
    # Determine trades needed
    trades = []
    
    for symbol, target_value in target_values.items():
        current_value = current_positions.get(symbol, {'value': 0})['value']
        value_difference = target_value - current_value
        
        if abs(value_difference) > total_value * 0.01:  # 1% threshold
            # Get current price
            latest_data = await market_data.get_recent_data(symbol=symbol, bars=1)
            current_price = latest_data['close'].iloc[-1]
            
            # Calculate quantity
            quantity = int(value_difference / current_price)
            
            if quantity != 0:
                trades.append({
                    'symbol': symbol,
                    'side': 'buy' if quantity > 0 else 'sell',
                    'quantity': abs(quantity),
                    'value': abs(value_difference)
                })
    
    return trades
```

## Portfolio Risk Management

### Position Sizing

```python
def calculate_position_size(capital, risk_per_trade, stop_loss_percent):
    """
    Calculate position size based on risk.
    
    Args:
        capital: Total capital available
        risk_per_trade: Percentage of capital to risk per trade (e.g., 0.01 for 1%)
        stop_loss_percent: Stop loss percentage (e.g., 0.05 for 5%)
    
    Returns:
        Position size in dollar amount
    """
    # Calculate dollar risk amount
    risk_amount = capital * risk_per_trade
    
    # Calculate position size
    position_size = risk_amount / stop_loss_percent
    
    return position_size
```

### Drawdown Protection

```python
def implement_drawdown_protection(portfolio, max_drawdown_limit=0.1):
    """
    Implement drawdown protection by reducing exposure when drawdown exceeds limit.
    
    Args:
        portfolio: Portfolio manager instance
        max_drawdown_limit: Maximum allowable drawdown (e.g., 0.1 for 10%)
    """
    # Get current drawdown
    equity_curve = portfolio.get_equity_curve()
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    current_drawdown = drawdown[-1]
    
    if abs(current_drawdown) > max_drawdown_limit:
        # Calculate reduction factor
        reduction = 1.0 - (abs(current_drawdown) / (2 * max_drawdown_limit))
        reduction = max(0.25, min(1.0, reduction))  # Limit between 25% and 100%
        
        # Reduce all allocations
        for strategy_id in portfolio.get_strategies():
            current_allocation = portfolio.get_allocation(strategy_id)
            portfolio.set_allocation(strategy_id, current_allocation * reduction)
        
        print(f"Drawdown protection activated. Reducing exposure to {reduction:.0%}")
```

### Correlation Management

```python
async def analyze_strategy_correlations(strategies, symbols, start_date, end_date, market_data):
    """
    Analyze correlations between strategies to improve diversification.
    """
    # Get returns for each strategy
    strategy_returns = {}
    
    for i, strategy in enumerate(strategies):
        # Get data
        symbol = symbols[i]
        data = await market_data.get_historical_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        # Preprocess and backtest
        processed_data = await strategy.preprocess_data(data)
        backtest_result = await strategy.backtest(
            data=processed_data,
            initial_capital=10000
        )
        
        # Get daily returns
        equity_curve = pd.Series(backtest_result['equity_curve'])
        daily_returns = equity_curve.pct_change().dropna()
        
        strategy_returns[f"{strategy.__class__.__name__}_{symbol}"] = daily_returns
    
    # Create returns DataFrame
    returns_df = pd.DataFrame(strategy_returns)
    
    # Calculate correlation matrix
    correlation_matrix = returns_df.corr()
    
    # Visualize correlations
    plt.figure(figsize=(10, 8))
    plt.imshow(correlation_matrix, cmap='coolwarm')
    plt.colorbar()
    
    # Add labels
    labels = correlation_matrix.columns
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    
    # Add correlation values
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(i, j, f"{correlation_matrix.iloc[i, j]:.2f}", 
                     ha="center", va="center", color="black")
    
    plt.title('Strategy Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    return correlation_matrix
```

## Creating an Advanced Portfolio

Let's put it all together with an advanced portfolio:

```python
async def create_advanced_portfolio(start_date, end_date):
    """Create an advanced portfolio with multiple strategies, assets, and risk management."""
    # Initialize services
    market_data = MarketDataService()
    
    # Create strategies with optimized parameters
    ma_strategy_aapl = MovingAverageStrategy(short_window=10, long_window=30)
    ma_strategy_msft = MovingAverageStrategy(short_window=15, long_window=45)
    lstm_strategy = LSTMPredictorStrategy(sequence_length=30, hidden_units=64)
    transformer_strategy = TransformerStrategy(sequence_length=60, num_layers=2)
    
    # Get performance-based allocations
    strategies = [ma_strategy_aapl, ma_strategy_msft, lstm_strategy, transformer_strategy]
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    
    allocations = await performance_based_allocation(
        strategies=strategies,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        market_data=market_data
    )
    
    # Create portfolio with risk management
    portfolio = PortfolioManager(
        initial_capital=10000,
        rebalance_frequency="monthly",
        risk_management=True,
        max_drawdown=0.15  # 15% maximum drawdown
    )
    
    # Add strategies with allocations
    for i, strategy in enumerate(strategies):
        portfolio.add_strategy(
            strategy=strategy,
            symbol=symbols[i],
            allocation=allocations[i],
            stop_loss=0.05,  # 5% stop loss
            take_profit=0.15  # 15% take profit
        )
    
    # Run portfolio backtest
    backtest_result = await portfolio.backtest(
        start_date=start_date,
        end_date=end_date,
        market_data_service=market_data
    )
    
    # Analyze correlations for future improvement
    correlation_matrix = await analyze_strategy_correlations(
        strategies=strategies,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        market_data=market_data
    )
    
    # Print detailed results
    print(f"Advanced Portfolio Results:")
    print(f"Initial Capital: $10,000.00")
    print(f"Final Capital: ${backtest_result['final_equity']:.2f}")
    print(f"Total Return: {((backtest_result['final_equity'] / 10000) - 1) * 100:.2f}%")
    print(f"Sharpe Ratio: {backtest_result['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {backtest_result['max_drawdown']:.2f}%")
    print(f"Win Rate: {backtest_result['win_rate']:.2f}%")
    
    return backtest_result, correlation_matrix
```

## Implementing a Portfolio Trading System

To use your portfolio in an actual trading system:

```python
async def run_portfolio_trading_system():
    """Run a complete portfolio trading system."""
    # Initialize services
    market_data = MarketDataService()
    trading_service = TradingService(mode="paper")  # Paper trading mode
    
    # Load optimized portfolio configuration
    portfolio_config = load_portfolio_config("optimized_portfolio.json")
    
    # Create portfolio
    portfolio = PortfolioManager(
        initial_capital=10000,
        rebalance_frequency=portfolio_config["rebalance_frequency"],
        risk_management=True
    )
    
    # Add strategies from configuration
    for strategy_config in portfolio_config["strategies"]:
        # Create strategy instance
        strategy_class = get_strategy_class(strategy_config["type"])
        strategy_instance = strategy_class(**strategy_config["parameters"])
        
        # Add to portfolio
        portfolio.add_strategy(
            strategy=strategy_instance,
            symbol=strategy_config["symbol"],
            allocation=strategy_config["allocation"],
            stop_loss=strategy_config.get("stop_loss"),
            take_profit=strategy_config.get("take_profit")
        )
    
    # Trading loop
    while True:
        try:
            # Check if market is open
            if not is_market_open():
                print("Market is closed. Sleeping...")
                await asyncio.sleep(3600)  # Sleep for an hour
                continue
            
            # Get portfolio signals
            signals = await portfolio.generate_signals(market_data)
            
            # Execute trades
            for signal in signals:
                if signal["action"] in ["BUY", "SELL"]:
                    await trading_service.place_order(
                        symbol=signal["symbol"],
                        quantity=signal["quantity"],
                        side=signal["action"].lower(),
                        order_type="market"
                    )
                    print(f"Executed {signal['action']} for {signal['symbol']}, "
                          f"Quantity: {signal['quantity']}")
            
            # Check for rebalancing
            if portfolio.should_rebalance():
                rebalance_trades = await portfolio.calculate_rebalance_trades(
                    trading_service.get_positions(),
                    market_data
                )
                
                # Execute rebalancing trades
                for trade in rebalance_trades:
                    await trading_service.place_order(
                        symbol=trade["symbol"],
                        quantity=trade["quantity"],
                        side=trade["side"],
                        order_type="market"
                    )
                    print(f"Rebalancing: {trade['side']} {trade['quantity']} of {trade['symbol']}")
            
            # Update portfolio status
            portfolio.update_status(trading_service.get_positions())
            
            # Wait before next cycle
            await asyncio.sleep(300)  # 5 minutes
            
        except Exception as e:
            print(f"Error in trading system: {e}")
            await asyncio.sleep(60)  # Wait a minute before trying again
```

## Next Steps

Now that you understand portfolio management in Mercurio AI, you're ready to learn about monitoring and analytics. In the next chapter, we'll explore how to monitor your strategies and analyze their performance.

Continue to [Chapter 11: Monitoring and Analytics](./11-monitoring.md) to learn about keeping track of your strategies and gaining insights from performance data.

---

**Key Takeaways:**
- Portfolio management combines multiple strategies and assets for better diversification
- Different allocation methods (equal weight, performance-based, risk parity) serve different goals
- Dynamic rebalancing maintains your desired portfolio allocations
- Risk management techniques help protect capital during adverse market conditions
- Correlation analysis helps build truly diversified portfolios
- Advanced portfolios combine optimized strategies, smart allocations, and risk management
