# Chapter 8: Backtesting Your Strategies

Welcome to Chapter 8! Now that you understand various trading strategies, it's time to learn how to properly backtest them using Mercurio AI. Backtesting is the process of testing a strategy against historical data to evaluate its performance before risking real money.

## The Importance of Backtesting

Proper backtesting gives you:

- Confidence in your strategy's performance
- Insight into risk and return characteristics
- Understanding of how a strategy behaves in different market conditions
- Identification of potential issues before real-world deployment

## Backtesting Basics in Mercurio AI

All strategies in Mercurio AI include a `backtest` method that simulates trading based on the strategy's signals:

```python
import asyncio
from app.strategies.moving_average import MovingAverageStrategy
from app.services.market_data import MarketDataService

async def basic_backtest():
    # Get data
    market_data = MarketDataService()
    data = await market_data.get_historical_data(
        symbol="AAPL",
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    
    # Create strategy
    strategy = MovingAverageStrategy(short_window=10, long_window=30)
    
    # Preprocess data
    processed_data = await strategy.preprocess_data(data)
    
    # Run backtest
    backtest_result = await strategy.backtest(
        data=processed_data,
        initial_capital=10000,
        commission=0.001  # 0.1% commission
    )
    
    # Print results
    print(f"Initial Capital: $10,000.00")
    print(f"Final Capital: ${backtest_result['final_equity']:.2f}")
    print(f"Total Return: {((backtest_result['final_equity'] / 10000) - 1) * 100:.2f}%")
    print(f"Number of Trades: {len(backtest_result['trades'])}")
    
    return backtest_result
```

## Understanding Backtest Results

The backtest result typically includes:

- `final_equity`: Final portfolio value
- `equity_curve`: Array of portfolio values over time
- `trades`: List of all trades executed during the simulation
- `metrics`: Various performance metrics

## Key Performance Metrics

### Return Metrics

```python
def calculate_return_metrics(backtest_result):
    initial_capital = 10000  # Adjust as needed
    final_equity = backtest_result['final_equity']
    equity_curve = pd.Series(backtest_result['equity_curve'])
    
    # Total return
    total_return = (final_equity / initial_capital - 1) * 100
    
    # Annualized return
    days = len(equity_curve)
    years = days / 252  # Trading days in a year
    annualized_return = ((final_equity / initial_capital) ** (1 / years) - 1) * 100
    
    # Daily returns
    daily_returns = equity_curve.pct_change().dropna()
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'avg_daily_return': daily_returns.mean() * 100,
        'daily_return_std': daily_returns.std() * 100
    }
```

### Risk Metrics

```python
def calculate_risk_metrics(backtest_result):
    equity_curve = pd.Series(backtest_result['equity_curve'])
    daily_returns = equity_curve.pct_change().dropna()
    
    # Sharpe Ratio (assuming risk-free rate of 0)
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    
    # Maximum Drawdown
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100
    
    # Sortino Ratio (downside risk only)
    negative_returns = daily_returns[daily_returns < 0]
    sortino_ratio = (daily_returns.mean() / negative_returns.std()) * np.sqrt(252)
    
    # Calmar Ratio (return / max drawdown)
    annualized_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (252 / len(equity_curve)) - 1
    calmar_ratio = annualized_return / (abs(max_drawdown) / 100)
    
    return {
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio
    }
```

### Trade Metrics

```python
def calculate_trade_metrics(trades):
    """Calculate metrics related to individual trades."""
    if not trades:
        return {"win_rate": 0, "profit_factor": 0, "avg_win": 0, "avg_loss": 0}
    
    # Calculate profits/losses for each trade
    profits = []
    for trade in trades:
        if trade['type'] == 'buy':
            entry_price = trade['price']
            entry_quantity = trade['quantity']
        elif trade['type'] == 'sell' and entry_quantity > 0:
            exit_price = trade['price']
            profit = (exit_price - entry_price) * entry_quantity
            profits.append(profit)
            entry_quantity = 0
    
    # Winning and losing trades
    winning_trades = [p for p in profits if p > 0]
    losing_trades = [p for p in profits if p < 0]
    
    # Metrics
    win_rate = len(winning_trades) / len(profits) * 100 if profits else 0
    profit_factor = abs(sum(winning_trades) / sum(losing_trades)) if sum(losing_trades) < 0 else float('inf')
    avg_win = np.mean(winning_trades) if winning_trades else 0
    avg_loss = np.mean(losing_trades) if losing_trades else 0
    
    return {
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'total_trades': len(profits)
    }
```

## Visualizing Backtest Results

Visualization is crucial for understanding backtest performance:

```python
def visualize_backtest(backtest_result, strategy_name="Strategy"):
    """Create comprehensive visualization of backtest results."""
    equity_curve = pd.Series(backtest_result['equity_curve'])
    trades = backtest_result['trades']
    
    # Create figure with subplots
    fig, axs = plt.subplots(3, 1, figsize=(12, 15), gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Plot 1: Equity Curve
    axs[0].plot(equity_curve)
    axs[0].set_title(f"{strategy_name} - Equity Curve")
    axs[0].set_ylabel("Portfolio Value ($)")
    axs[0].grid(True)
    
    # Add buy/sell markers
    buy_dates = [t['date'] for t in trades if t['type'] == 'buy']
    sell_dates = [t['date'] for t in trades if t['type'] == 'sell']
    
    for i, date in enumerate(buy_dates):
        idx = equity_curve.index.get_loc(date)
        axs[0].plot(date, equity_curve.iloc[idx], 'g^', markersize=8)
    
    for i, date in enumerate(sell_dates):
        idx = equity_curve.index.get_loc(date)
        axs[0].plot(date, equity_curve.iloc[idx], 'rv', markersize=8)
    
    # Plot 2: Drawdown
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max * 100
    axs[1].fill_between(drawdown.index, drawdown, 0, color='r', alpha=0.3)
    axs[1].set_title("Drawdown")
    axs[1].set_ylabel("Drawdown (%)")
    axs[1].grid(True)
    
    # Plot 3: Returns Distribution
    returns = equity_curve.pct_change().dropna() * 100
    axs[2].hist(returns, bins=50, alpha=0.75)
    axs[2].set_title("Daily Returns Distribution")
    axs[2].set_xlabel("Daily Return (%)")
    axs[2].set_ylabel("Frequency")
    axs[2].grid(True)
    
    plt.tight_layout()
    plt.show()
```

## Walk-Forward Testing

Walk-forward testing is a more robust approach that prevents overfitting:

```python
async def walk_forward_test(strategy_class, symbol, start_date, end_date, **strategy_params):
    """
    Perform walk-forward testing with periodic retraining.
    
    This simulates how the strategy would perform in real-world conditions
    by periodically retraining on recent data.
    """
    market_data = MarketDataService()
    
    # Get full dataset
    full_data = await market_data.get_historical_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date
    )
    
    # Convert to dataframe with datetime index
    full_df = pd.DataFrame(full_data)
    
    # Set up walk-forward parameters
    training_window = 180  # 6 months of training data
    testing_window = 30    # 1 month of testing
    
    start_idx = 0
    results = []
    equity_curves = []
    
    while start_idx + training_window + testing_window <= len(full_df):
        # Extract training and testing data
        train_end_idx = start_idx + training_window
        test_end_idx = train_end_idx + testing_window
        
        training_data = full_df.iloc[start_idx:train_end_idx]
        testing_data = full_df.iloc[train_end_idx:test_end_idx]
        
        # Initialize strategy with parameters
        strategy = strategy_class(**strategy_params)
        
        # Train on training data
        processed_training = await strategy.preprocess_data(training_data)
        
        # Test on testing data
        processed_testing = await strategy.preprocess_data(testing_data)
        
        # Backtest on testing period
        backtest_result = await strategy.backtest(
            data=processed_testing,
            initial_capital=10000
        )
        
        # Store results
        performance = {
            'period_start': testing_data.index[0],
            'period_end': testing_data.index[-1],
            'return': (backtest_result['final_equity'] / 10000 - 1) * 100,
            'trades': len(backtest_result['trades'])
        }
        
        results.append(performance)
        equity_curves.append(backtest_result['equity_curve'])
        
        # Move window forward
        start_idx += testing_window
    
    # Combine results
    results_df = pd.DataFrame(results)
    
    # Calculate overall performance
    total_return = (1 + results_df['return'] / 100).prod() - 1
    avg_return = results_df['return'].mean()
    
    print(f"Walk-Forward Test Results for {symbol}:")
    print(f"Periods tested: {len(results_df)}")
    print(f"Average period return: {avg_return:.2f}%")
    print(f"Compounded total return: {total_return * 100:.2f}%")
    
    return results_df, equity_curves
```

## Monte Carlo Simulation

Monte Carlo simulation helps understand the range of possible outcomes:

```python
def monte_carlo_simulation(backtest_result, simulations=1000):
    """
    Perform Monte Carlo simulation by resampling returns.
    
    This helps understand the range of possible outcomes and
    the robustness of the strategy.
    """
    equity_curve = pd.Series(backtest_result['equity_curve'])
    daily_returns = equity_curve.pct_change().dropna()
    
    # Number of days in the simulation
    days = len(daily_returns)
    
    # Run simulations
    simulated_returns = np.zeros((simulations, days))
    
    for i in range(simulations):
        # Resample returns with replacement
        sampled_returns = np.random.choice(daily_returns, size=days, replace=True)
        
        # Calculate cumulative returns
        cumulative_returns = (1 + sampled_returns).cumprod()
        
        # Store the equity curve
        simulated_returns[i] = 10000 * cumulative_returns
    
    # Calculate statistics
    final_values = simulated_returns[:, -1]
    
    results = {
        'median': np.median(final_values),
        'mean': np.mean(final_values),
        'std': np.std(final_values),
        'min': np.min(final_values),
        'max': np.max(final_values),
        'percentile_5': np.percentile(final_values, 5),
        'percentile_95': np.percentile(final_values, 95)
    }
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot all simulations
    for i in range(simulations):
        plt.plot(simulated_returns[i], 'b-', alpha=0.05)
    
    # Plot original equity curve
    plt.plot(backtest_result['equity_curve'], 'r-', linewidth=2, label='Original Backtest')
    
    # Plot 5th and 95th percentiles
    percentile_5 = np.percentile(simulated_returns, 5, axis=0)
    percentile_95 = np.percentile(simulated_returns, 95, axis=0)
    
    plt.plot(percentile_5, 'g--', linewidth=2, label='5th Percentile')
    plt.plot(percentile_95, 'g--', linewidth=2, label='95th Percentile')
    
    plt.title('Monte Carlo Simulation')
    plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    
    return results
```

## The Dangers of Overfitting

Overfitting occurs when a strategy is too closely tailored to historical data and fails in real-world conditions. Here are techniques to avoid it:

1. **Out-of-Sample Testing**: Always hold out some data that wasn't used in strategy development
2. **Walk-Forward Testing**: Periodically retrain on recent data
3. **Cross-Validation**: Test on multiple periods
4. **Simplicity**: Prefer simpler strategies with fewer parameters
5. **Parameter Robustness**: Check performance across a range of parameters

## Creating a Comprehensive Backtest Report

Let's create a function to generate a comprehensive backtest report:

```python
async def generate_backtest_report(strategy, symbol, start_date, end_date, initial_capital=10000):
    """Generate a comprehensive backtest report."""
    # Get data
    market_data = MarketDataService()
    data = await market_data.get_historical_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date
    )
    
    # Preprocess data
    processed_data = await strategy.preprocess_data(data)
    
    # Run backtest
    backtest_result = await strategy.backtest(
        data=processed_data,
        initial_capital=initial_capital
    )
    
    # Calculate metrics
    return_metrics = calculate_return_metrics(backtest_result)
    risk_metrics = calculate_risk_metrics(backtest_result)
    trade_metrics = calculate_trade_metrics(backtest_result['trades'])
    
    # Generate report
    report = {
        'strategy': strategy.__class__.__name__,
        'symbol': symbol,
        'period': f"{start_date} to {end_date}",
        'initial_capital': initial_capital,
        'final_equity': backtest_result['final_equity'],
        'return_metrics': return_metrics,
        'risk_metrics': risk_metrics,
        'trade_metrics': trade_metrics
    }
    
    # Print summary
    print(f"=== BACKTEST REPORT: {strategy.__class__.__name__} on {symbol} ===")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Capital: ${initial_capital:.2f}")
    print(f"Final Equity: ${backtest_result['final_equity']:.2f}")
    print(f"Total Return: {return_metrics['total_return']:.2f}%")
    print(f"Annualized Return: {return_metrics['annualized_return']:.2f}%")
    print(f"Sharpe Ratio: {risk_metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {risk_metrics['max_drawdown']:.2f}%")
    print(f"Win Rate: {trade_metrics['win_rate']:.2f}%")
    print(f"Profit Factor: {trade_metrics['profit_factor']:.2f}")
    print(f"Total Trades: {trade_metrics['total_trades']}")
    
    # Create visualization
    visualize_backtest(backtest_result, strategy.__class__.__name__)
    
    return report, backtest_result
```

## Comparing Multiple Strategies

To find the best strategy, we often need to compare several:

```python
async def compare_strategies(strategies, symbol, start_date, end_date, initial_capital=10000):
    """Compare multiple strategies on the same asset and time period."""
    # Get data
    market_data = MarketDataService()
    data = await market_data.get_historical_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date
    )
    
    # Results storage
    results = []
    equity_curves = {}
    
    # Test each strategy
    for strategy in strategies:
        strategy_name = strategy.__class__.__name__
        
        # Preprocess data
        processed_data = await strategy.preprocess_data(data)
        
        # Run backtest
        backtest_result = await strategy.backtest(
            data=processed_data,
            initial_capital=initial_capital
        )
        
        # Calculate metrics
        return_metrics = calculate_return_metrics(backtest_result)
        risk_metrics = calculate_risk_metrics(backtest_result)
        trade_metrics = calculate_trade_metrics(backtest_result['trades'])
        
        # Store results
        results.append({
            'strategy': strategy_name,
            'total_return': return_metrics['total_return'],
            'annualized_return': return_metrics['annualized_return'],
            'sharpe_ratio': risk_metrics['sharpe_ratio'],
            'max_drawdown': risk_metrics['max_drawdown'],
            'win_rate': trade_metrics['win_rate'],
            'profit_factor': trade_metrics['profit_factor'],
            'trade_count': trade_metrics['total_trades']
        })
        
        # Store equity curve
        equity_curves[strategy_name] = backtest_result['equity_curve']
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Plot comparison
    plt.figure(figsize=(12, 8))
    
    for strategy, curve in equity_curves.items():
        plt.plot(pd.Series(curve), label=strategy)
    
    plt.title(f"Strategy Comparison - {symbol}")
    plt.xlabel("Trading Days")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Print comparison table
    print(results_df)
    
    return results_df, equity_curves
```

## Stress Testing

Stress testing helps understand how strategies perform in extreme market conditions:

```python
async def stress_test_strategy(strategy, symbol, start_date, end_date, initial_capital=10000):
    """
    Stress test a strategy against various market conditions.
    """
    market_data = MarketDataService()
    
    # Get base data
    base_data = await market_data.get_historical_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date
    )
    
    # Convert to DataFrame
    df = pd.DataFrame(base_data)
    
    # Define stress scenarios
    scenarios = {
        'Base Case': df.copy(),
        'Bear Market': apply_bear_market(df.copy()),  # Gradual downtrend
        'Bull Market': apply_bull_market(df.copy()),  # Gradual uptrend
        'Market Crash': apply_market_crash(df.copy()),  # Sudden sharp drop
        'Volatile Market': apply_volatility(df.copy()),  # Increased volatility
        'Low Volatility': apply_low_volatility(df.copy()),  # Decreased volatility
    }
    
    # Test each scenario
    results = []
    
    for scenario_name, scenario_data in scenarios.items():
        # Preprocess data
        processed_data = await strategy.preprocess_data(scenario_data)
        
        # Run backtest
        backtest_result = await strategy.backtest(
            data=processed_data,
            initial_capital=initial_capital
        )
        
        # Calculate metrics
        return_metrics = calculate_return_metrics(backtest_result)
        risk_metrics = calculate_risk_metrics(backtest_result)
        
        # Store results
        results.append({
            'scenario': scenario_name,
            'total_return': return_metrics['total_return'],
            'sharpe_ratio': risk_metrics['sharpe_ratio'],
            'max_drawdown': risk_metrics['max_drawdown']
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df
```

## Next Steps

Now that you understand how to properly backtest strategies, you're ready to learn about strategy optimization in Mercurio AI. In the next chapter, we'll explore methods to fine-tune your strategies for better performance.

Continue to [Chapter 9: Strategy Optimization](./09-optimization.md) to learn how to optimize your trading strategies.

---

**Key Takeaways:**
- Backtesting is essential for evaluating strategy performance before using real money
- Mercurio AI provides comprehensive backtesting capabilities for all strategy types
- Key performance metrics include return metrics, risk metrics, and trade metrics
- Visualization helps understand strategy behavior and performance
- Advanced techniques like walk-forward testing and Monte Carlo simulation provide deeper insights
- Comparing multiple strategies helps identify the best approach for specific assets and market conditions
