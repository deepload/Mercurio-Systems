# Chapter 9: Strategy Optimization

Welcome to Chapter 9! Now that you understand how to backtest your strategies, it's time to learn how to optimize them for better performance. Strategy optimization is the process of finding the best parameters and configurations for your trading strategies.

## The Importance of Optimization

Optimization helps you:

- Improve strategy performance
- Find the most effective parameter combinations
- Avoid overfitting to historical data
- Create more robust trading systems

## Optimization Basics

Let's start with a simple example of optimizing a Moving Average strategy:

```python
import asyncio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from app.strategies.moving_average import MovingAverageStrategy
from app.services.market_data import MarketDataService

async def optimize_ma_strategy(symbol, start_date, end_date):
    """Find optimal Moving Average parameters."""
    # Get data
    market_data = MarketDataService()
    data = await market_data.get_historical_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date
    )
    
    # Define parameter ranges
    short_windows = range(5, 31, 5)  # 5, 10, 15, 20, 25, 30
    long_windows = range(20, 101, 10)  # 20, 30, 40, ..., 100
    
    # Store results
    results = []
    
    # Test each combination
    for short in short_windows:
        for long in long_windows:
            if short >= long:
                continue  # Short must be less than long
            
            # Create strategy with these parameters
            strategy = MovingAverageStrategy(
                short_window=short,
                long_window=long,
                use_ml=False
            )
            
            # Run backtest
            processed_data = await strategy.preprocess_data(data)
            backtest_result = await strategy.backtest(
                data=processed_data,
                initial_capital=10000
            )
            
            # Calculate metrics
            total_return = (backtest_result['final_equity'] / 10000 - 1) * 100
            
            # Store results
            results.append({
                'short_window': short,
                'long_window': long,
                'total_return': total_return,
                'trade_count': len(backtest_result['trades'])
            })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Find best combination
    best_row = results_df.loc[results_df['total_return'].idxmax()]
    
    print(f"Best parameters for {symbol}:")
    print(f"Short Window: {best_row['short_window']}")
    print(f"Long Window: {best_row['long_window']}")
    print(f"Total Return: {best_row['total_return']:.2f}%")
    print(f"Trade Count: {best_row['trade_count']}")
    
    # Visualize results
    plt.figure(figsize=(12, 8))
    
    # Create pivot table for heatmap
    pivot = results_df.pivot(index='short_window', columns='long_window', values='total_return')
    
    # Plot heatmap
    plt.imshow(pivot, cmap='hot')
    plt.colorbar(label='Total Return (%)')
    
    # Add labels
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    
    plt.xlabel('Long Window')
    plt.ylabel('Short Window')
    plt.title(f'Moving Average Parameter Optimization for {symbol}')
    
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            if not np.isnan(pivot.iloc[i, j]):
                plt.text(j, i, f"{pivot.iloc[i, j]:.1f}%", 
                         ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    return results_df, best_row
```

## Optimization Techniques

### Grid Search

The above example uses grid search, which tests all combinations in a predefined grid of parameters:

- **Pros**: Thorough, easy to visualize
- **Cons**: Computationally expensive, especially with many parameters

### Random Search

Random search samples parameter combinations randomly:

```python
async def random_search_optimization(strategy_class, param_ranges, symbol, start_date, end_date, samples=30):
    """Optimize strategy parameters using random search."""
    market_data = MarketDataService()
    data = await market_data.get_historical_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date
    )
    
    # Store results
    results = []
    
    # Generate random samples
    for _ in range(samples):
        # Generate random parameters
        params = {}
        for param, param_range in param_ranges.items():
            if isinstance(param_range[0], int):
                params[param] = np.random.randint(param_range[0], param_range[1])
            else:
                params[param] = np.random.uniform(param_range[0], param_range[1])
        
        # Create strategy with these parameters
        strategy = strategy_class(**params)
        
        # Run backtest
        processed_data = await strategy.preprocess_data(data)
        backtest_result = await strategy.backtest(
            data=processed_data,
            initial_capital=10000
        )
        
        # Calculate metrics
        total_return = (backtest_result['final_equity'] / 10000 - 1) * 100
        
        # Store results
        result = {'total_return': total_return, 'trade_count': len(backtest_result['trades'])}
        result.update(params)  # Add parameters to result
        results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Find best combination
    best_row = results_df.loc[results_df['total_return'].idxmax()]
    
    print(f"Best parameters found:")
    for param in param_ranges.keys():
        print(f"{param}: {best_row[param]}")
    print(f"Total Return: {best_row['total_return']:.2f}%")
    
    return results_df, best_row
```

Example usage:

```python
# Define parameter ranges for LSTM Strategy
lstm_param_ranges = {
    'sequence_length': (10, 50),  # Between 10 and 50
    'hidden_units': (32, 128),    # Between 32 and 128
    'dropout': (0.1, 0.5),        # Between 0.1 and 0.5
    'epochs': (20, 100)           # Between 20 and 100
}

# Run random search
from app.strategies.lstm_predictor import LSTMPredictorStrategy
results, best = await random_search_optimization(
    LSTMPredictorStrategy,
    lstm_param_ranges,
    "AAPL",
    "2022-01-01",
    "2023-12-31",
    samples=20
)
```

### Bayesian Optimization

Bayesian optimization is more efficient than grid or random search:

```python
# Requires installation of scikit-optimize
!pip install scikit-optimize

from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

async def bayesian_optimization(strategy_class, param_spaces, symbol, start_date, end_date, n_calls=30):
    """Optimize strategy parameters using Bayesian optimization."""
    market_data = MarketDataService()
    data = await market_data.get_historical_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date
    )
    
    # Define the objective function
    @use_named_args(param_spaces)
    async def objective(**params):
        # Create strategy with these parameters
        strategy = strategy_class(**params)
        
        # Run backtest
        processed_data = await strategy.preprocess_data(data)
        backtest_result = await strategy.backtest(
            data=processed_data,
            initial_capital=10000
        )
        
        # Calculate negative return (we minimize, so negate)
        total_return = (backtest_result['final_equity'] / 10000 - 1) * 100
        return -total_return
    
    # Run optimization
    result = gp_minimize(
        lambda x: asyncio.run(objective(x)),
        param_spaces,
        n_calls=n_calls,
        random_state=42
    )
    
    # Get best parameters
    best_params = dict(zip([param.name for param in param_spaces], result.x))
    
    print(f"Best parameters found:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print(f"Best Return: {-result.fun:.2f}%")
    
    return result, best_params
```

Example usage:

```python
# Define parameter spaces for RSI Strategy
from skopt.space import Real, Integer

rsi_param_spaces = [
    Integer(5, 30, name='period'),
    Integer(60, 85, name='overbought'),
    Integer(15, 40, name='oversold')
]

# Run Bayesian optimization
result, best_params = await bayesian_optimization(
    RSIStrategy,
    rsi_param_spaces,
    "AAPL",
    "2022-01-01",
    "2023-12-31",
    n_calls=30
)
```

## Best Practices for Optimization

### 1. Train-Test Split

Always split your data to avoid overfitting:

```python
async def train_test_optimization(strategy_class, param_ranges, symbol, start_date, end_date, samples=30):
    """Optimize with train-test split to avoid overfitting."""
    market_data = MarketDataService()
    data = await market_data.get_historical_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date
    )
    
    # Split into training and testing sets
    split_idx = int(len(data) * 0.7)  # 70% training, 30% testing
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    # Store results
    results = []
    
    # Generate random samples
    for _ in range(samples):
        # Generate random parameters
        params = {}
        for param, param_range in param_ranges.items():
            if isinstance(param_range[0], int):
                params[param] = np.random.randint(param_range[0], param_range[1])
            else:
                params[param] = np.random.uniform(param_range[0], param_range[1])
        
        # Create strategy with these parameters
        strategy = strategy_class(**params)
        
        # Train on training data
        processed_train = await strategy.preprocess_data(train_data)
        
        # Test on testing data
        processed_test = await strategy.preprocess_data(test_data)
        
        # Run backtest on both sets
        train_result = await strategy.backtest(data=processed_train, initial_capital=10000)
        test_result = await strategy.backtest(data=processed_test, initial_capital=10000)
        
        # Calculate metrics
        train_return = (train_result['final_equity'] / 10000 - 1) * 100
        test_return = (test_result['final_equity'] / 10000 - 1) * 100
        
        # Store results
        result = {
            'train_return': train_return, 
            'test_return': test_return,
            'train_trades': len(train_result['trades']),
            'test_trades': len(test_result['trades'])
        }
        result.update(params)  # Add parameters to result
        results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Find best combination based on test set
    best_row = results_df.loc[results_df['test_return'].idxmax()]
    
    print(f"Best parameters found:")
    for param in param_ranges.keys():
        print(f"{param}: {best_row[param]}")
    print(f"Training Return: {best_row['train_return']:.2f}%")
    print(f"Testing Return: {best_row['test_return']:.2f}%")
    
    return results_df, best_row
```

### 2. Cross-Validation

Use time-series cross-validation for more robust results:

```python
async def time_series_cv_optimization(strategy_class, params, symbol, start_date, end_date, num_folds=5):
    """Optimize using time-series cross-validation."""
    market_data = MarketDataService()
    data = await market_data.get_historical_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date
    )
    
    # Calculate fold size
    fold_size = len(data) // num_folds
    
    # Store results for each fold
    fold_returns = []
    
    # Run cross-validation
    for fold in range(num_folds - 1):  # Use last fold as final test
        # Calculate indices
        train_end = (fold + 1) * fold_size
        test_end = train_end + fold_size
        
        # Split data
        train_data = data.iloc[:train_end]
        test_data = data.iloc[train_end:test_end]
        
        # Create strategy
        strategy = strategy_class(**params)
        
        # Preprocess data
        processed_train = await strategy.preprocess_data(train_data)
        processed_test = await strategy.preprocess_data(test_data)
        
        # Run backtest on test set
        test_result = await strategy.backtest(
            data=processed_test,
            initial_capital=10000
        )
        
        # Calculate return
        test_return = (test_result['final_equity'] / 10000 - 1) * 100
        fold_returns.append(test_return)
    
    # Calculate average and standard deviation
    avg_return = np.mean(fold_returns)
    std_return = np.std(fold_returns)
    
    print(f"Cross-Validation Results:")
    print(f"Average Return: {avg_return:.2f}%")
    print(f"Standard Deviation: {std_return:.2f}%")
    print(f"Return by Fold: {fold_returns}")
    
    return avg_return, std_return, fold_returns
```

### 3. Multi-Asset Optimization

Test parameters across multiple assets:

```python
async def multi_asset_optimization(strategy_class, params, symbols, start_date, end_date):
    """Test strategy parameters across multiple assets."""
    market_data = MarketDataService()
    
    # Store results for each asset
    asset_returns = {}
    
    # Test on each symbol
    for symbol in symbols:
        # Get data
        data = await market_data.get_historical_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        # Create strategy
        strategy = strategy_class(**params)
        
        # Run backtest
        processed_data = await strategy.preprocess_data(data)
        backtest_result = await strategy.backtest(
            data=processed_data,
            initial_capital=10000
        )
        
        # Calculate return
        total_return = (backtest_result['final_equity'] / 10000 - 1) * 100
        asset_returns[symbol] = total_return
    
    # Calculate average return
    avg_return = np.mean(list(asset_returns.values()))
    
    print(f"Multi-Asset Results:")
    print(f"Average Return: {avg_return:.2f}%")
    for symbol, ret in asset_returns.items():
        print(f"{symbol}: {ret:.2f}%")
    
    return asset_returns, avg_return
```

### 4. Objective Functions

Consider different objective functions beyond just returns:

```python
def calculate_sharpe_ratio(backtest_result):
    """Calculate Sharpe ratio from backtest result."""
    equity_curve = pd.Series(backtest_result['equity_curve'])
    daily_returns = equity_curve.pct_change().dropna()
    
    # Calculate Sharpe ratio (assuming risk-free rate of 0)
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    return sharpe_ratio

async def optimize_for_sharpe(strategy_class, param_ranges, symbol, start_date, end_date, samples=30):
    """Optimize for Sharpe ratio instead of total return."""
    market_data = MarketDataService()
    data = await market_data.get_historical_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date
    )
    
    # Store results
    results = []
    
    # Generate random samples
    for _ in range(samples):
        # Generate random parameters
        params = {}
        for param, param_range in param_ranges.items():
            if isinstance(param_range[0], int):
                params[param] = np.random.randint(param_range[0], param_range[1])
            else:
                params[param] = np.random.uniform(param_range[0], param_range[1])
        
        # Create strategy with these parameters
        strategy = strategy_class(**params)
        
        # Run backtest
        processed_data = await strategy.preprocess_data(data)
        backtest_result = await strategy.backtest(
            data=processed_data,
            initial_capital=10000
        )
        
        # Calculate metrics
        total_return = (backtest_result['final_equity'] / 10000 - 1) * 100
        sharpe_ratio = calculate_sharpe_ratio(backtest_result)
        
        # Store results
        result = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'trade_count': len(backtest_result['trades'])
        }
        result.update(params)  # Add parameters to result
        results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Find best combination based on Sharpe ratio
    best_row = results_df.loc[results_df['sharpe_ratio'].idxmax()]
    
    print(f"Best parameters for Sharpe ratio:")
    for param in param_ranges.keys():
        print(f"{param}: {best_row[param]}")
    print(f"Sharpe Ratio: {best_row['sharpe_ratio']:.2f}")
    print(f"Total Return: {best_row['total_return']:.2f}%")
    
    return results_df, best_row
```

## Optimizing Different Strategy Types

### Optimizing Moving Average Strategy

```python
# Define parameter ranges
ma_param_ranges = {
    'short_window': (5, 30),
    'long_window': (20, 100),
    'use_ml': (True, False)  # Binary choice
}

# Convert use_ml to boolean during parameter generation
async def optimize_ma_strategy_random(symbol, start_date, end_date, samples=30):
    market_data = MarketDataService()
    data = await market_data.get_historical_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date
    )
    
    # Store results
    results = []
    
    # Generate random samples
    for _ in range(samples):
        # Generate random parameters
        short_window = np.random.randint(ma_param_ranges['short_window'][0], ma_param_ranges['short_window'][1])
        long_window = np.random.randint(ma_param_ranges['long_window'][0], ma_param_ranges['long_window'][1])
        use_ml = np.random.choice([True, False])
        
        # Skip invalid combinations
        if short_window >= long_window:
            continue
        
        # Create strategy
        strategy = MovingAverageStrategy(
            short_window=short_window,
            long_window=long_window,
            use_ml=use_ml
        )
        
        # Run backtest
        processed_data = await strategy.preprocess_data(data)
        backtest_result = await strategy.backtest(
            data=processed_data,
            initial_capital=10000
        )
        
        # Calculate metrics
        total_return = (backtest_result['final_equity'] / 10000 - 1) * 100
        
        # Store results
        results.append({
            'short_window': short_window,
            'long_window': long_window,
            'use_ml': use_ml,
            'total_return': total_return,
            'trade_count': len(backtest_result['trades'])
        })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Find best combination
    best_row = results_df.loc[results_df['total_return'].idxmax()]
    
    print(f"Best MA parameters for {symbol}:")
    print(f"Short Window: {best_row['short_window']}")
    print(f"Long Window: {best_row['long_window']}")
    print(f"Use ML: {best_row['use_ml']}")
    print(f"Total Return: {best_row['total_return']:.2f}%")
    
    return results_df, best_row
```

### Optimizing LSTM Strategy

```python
from app.strategies.lstm_predictor import LSTMPredictorStrategy

async def optimize_lstm_strategy(symbol, start_date, end_date, samples=15):
    """Optimize LSTM strategy parameters."""
    # Define parameter ranges
    lstm_param_ranges = {
        'sequence_length': (10, 50),
        'hidden_units': (32, 128),
        'dropout': (0.1, 0.5),
        'epochs': (20, 50)
    }
    
    # Run optimization with train-test split
    results_df, best_row = await train_test_optimization(
        LSTMPredictorStrategy,
        lstm_param_ranges,
        symbol,
        start_date,
        end_date,
        samples=samples
    )
    
    return results_df, best_row
```

## Saving and Loading Optimized Parameters

Save your optimized parameters for future use:

```python
import json

def save_optimized_parameters(params, strategy_name, symbol, filename=None):
    """Save optimized parameters to a file."""
    if filename is None:
        filename = f"optimized_{strategy_name}_{symbol}.json"
    
    with open(filename, 'w') as f:
        json.dump(params, f, indent=4)
    
    print(f"Saved parameters to {filename}")

def load_optimized_parameters(strategy_name, symbol, filename=None):
    """Load optimized parameters from a file."""
    if filename is None:
        filename = f"optimized_{strategy_name}_{symbol}.json"
    
    try:
        with open(filename, 'r') as f:
            params = json.load(f)
        print(f"Loaded parameters from {filename}")
        return params
    except FileNotFoundError:
        print(f"File {filename} not found")
        return None
```

## Optimization Workflow

Here's a complete optimization workflow:

```python
async def complete_optimization_workflow(strategy_class, param_ranges, symbol, start_date, end_date):
    """Complete optimization workflow."""
    print(f"Starting optimization for {strategy_class.__name__} on {symbol}")
    
    # Step 1: Initial optimization with train-test split
    print("\nStep 1: Initial Optimization")
    results_df, best_params = await train_test_optimization(
        strategy_class,
        param_ranges,
        symbol,
        start_date,
        end_date,
        samples=30
    )
    
    # Step 2: Cross-validation with best parameters
    print("\nStep 2: Cross-Validation")
    avg_return, std_return, fold_returns = await time_series_cv_optimization(
        strategy_class,
        best_params,
        symbol,
        start_date,
        end_date,
        num_folds=5
    )
    
    # Step 3: Test on other assets
    print("\nStep 3: Multi-Asset Testing")
    other_symbols = ["MSFT", "GOOGL", "AMZN", "TSLA"]  # Example symbols
    if symbol in other_symbols:
        other_symbols.remove(symbol)
    
    asset_returns, avg_asset_return = await multi_asset_optimization(
        strategy_class,
        best_params,
        other_symbols,
        start_date,
        end_date
    )
    
    # Step 4: Save optimized parameters
    print("\nStep 4: Saving Parameters")
    save_optimized_parameters(
        best_params,
        strategy_class.__name__,
        symbol
    )
    
    # Final report
    print("\n===== Optimization Complete =====")
    print(f"Strategy: {strategy_class.__name__}")
    print(f"Symbol: {symbol}")
    print(f"Best Parameters: {best_params}")
    print(f"Cross-Validation Return: {avg_return:.2f}% (±{std_return:.2f}%)")
    print(f"Multi-Asset Average Return: {avg_asset_return:.2f}%")
    
    return {
        'best_params': best_params,
        'cv_return': avg_return,
        'cv_std': std_return,
        'asset_returns': asset_returns
    }
```

## Next Steps

Now that you understand how to optimize your trading strategies in Mercurio AI, you're ready to learn about portfolio management. In the next chapter, we'll explore how to combine multiple strategies and assets into a cohesive portfolio.

Continue to [Chapter 10: Portfolio Management](./10-portfolio-management.md) to learn about managing multiple strategies and assets.

---

**Key Takeaways:**
- Strategy optimization helps find the best parameters for improved performance
- Different optimization techniques include grid search, random search, and Bayesian optimization
- Best practices include train-test splitting, cross-validation, and multi-asset testing
- Different objective functions (return, Sharpe ratio, etc.) can be used depending on your goals
- A complete optimization workflow includes initial optimization, validation, multi-asset testing, and parameter saving
