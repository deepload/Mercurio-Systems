# Chapter 5: Data Sources and Management

Welcome to Chapter 5! In this chapter, we'll explore how Mercurio AI handles market data, which is the foundation of any successful trading strategy. You'll learn about different data sources, how to access them, and how to prepare data for your trading strategies.

## Understanding Market Data

Market data is the raw material that trading strategies use to make decisions. There are several key types of market data:

### Price Data (OHLCV)

The most common type of market data is price data, typically in OHLCV format:

- **O**pen: Price at the beginning of the period
- **H**igh: Highest price during the period
- **L**ow: Lowest price during the period
- **C**lose: Price at the end of the period
- **V**olume: Trading volume during the period

This data forms the basis of most technical analysis and trading strategies.

### Timeframes

Market data is organized into different timeframes:

- **Intraday**: 1-minute, 5-minute, 15-minute, 1-hour, etc.
- **Daily**: One data point per trading day
- **Weekly**: One data point per trading week
- **Monthly**: One data point per month

Different strategies work best with different timeframes. Mercurio AI supports multiple timeframes depending on the data source.

### Fundamental Data

Beyond price data, some strategies may use fundamental data:

- Financial statements
- Economic indicators
- Company news and events
- Earnings reports

While Mercurio AI's primary focus is on price data, some strategies (like the LLM strategy) can incorporate fundamental data.

## Data Sources in Mercurio AI

Mercurio AI supports multiple data sources, arranged in a fallback hierarchy:

### Real-Time and Historical Data Providers

These external APIs provide market data, and may require API keys:

1. **AlphaVantage**: Stocks, forex, cryptocurrencies
2. **Alpaca**: US stocks, ETFs, and cryptocurrencies (BTC-USD, ETH-USD, etc.)
3. **Binance**: Cryptocurrencies
4. **Yahoo Finance**: Stocks, ETFs, mutual funds (used as a free fallback)

### Sample Data Provider

When external data sources aren't available, Mercurio AI generates realistic sample data:

- Based on statistical properties of real assets
- Includes trend, cycle, and random components
- Customizable to different market conditions
- Great for testing and development

## Accessing Market Data

Let's see how to access market data using the MarketDataService:

```python
from app.services.market_data import MarketDataService

async def get_market_data():
    # Initialize the service
    market_data = MarketDataService()
    
    # Get historical daily data for Apple (AAPL)
    aapl_data = await market_data.get_historical_data(
        symbol="AAPL",
        start_date="2024-01-01",
        end_date="2024-04-25",
        timeframe="1d"  # Daily data
    )
    
    # Get historical hourly data for Bitcoin
    btc_data = await market_data.get_historical_data(
        symbol="BTC-USD",
        start_date="2024-04-01",
        end_date="2024-04-25",
        timeframe="1h"  # Hourly data
    )
    
    # Get most recent data for Tesla
    tesla_recent = await market_data.get_recent_data(
        symbol="TSLA",
        bars=100,  # Last 100 data points
        timeframe="15min"  # 15-minute data
    )
    
    return aapl_data, btc_data, tesla_recent
```

### Configuring Data Sources

You can configure which data sources to use and provide API keys:

```python
# Configuring specific data sources
market_data = MarketDataService(
    primary_provider="alpaca",  # Use Alpaca as primary
    api_key="your_alpaca_api_key",
    api_secret="your_alpaca_api_secret",
    backup_provider="yahoo",  # Use Yahoo Finance as backup
    enable_sample_data=True  # Allow fallback to sample data
)
```

### Working with Different Asset Types

Mercurio AI supports multiple asset types with a consistent interface:

```python
# Stocks (US)
stock_data = await market_data.get_historical_data("AAPL")

# Cryptocurrencies
crypto_data = await market_data.get_historical_data("BTC-USD")

# ETFs
etf_data = await market_data.get_historical_data("SPY")

# Forex
forex_data = await market_data.get_historical_data("EUR/USD")
```

## Generating Sample Data

When external data sources aren't available, Mercurio AI can generate realistic sample data:

```python
from app.utils.simulation_utils import generate_simulation_data
import pandas as pd
from datetime import datetime, timedelta

# Generate one year of daily data for AAPL
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

sample_data = generate_simulation_data(
    symbol="AAPL",
    start_date=start_date,
    end_date=end_date,
    freq="1d"  # Daily frequency
)

print(f"Generated {len(sample_data)} days of sample data")
print(sample_data.head())
```

### Understanding Sample Data Generation

The `generate_simulation_data` function creates realistic price data with these components:

1. **Initial Price**: Based on the typical price range of the asset
2. **Trend Component**: Long-term direction (bull or bear market)
3. **Cyclical Component**: Medium-term cycles (like quarterly patterns)
4. **Seasonal Component**: Recurring patterns (like monthly effects)
5. **Random Component**: Day-to-day price fluctuations
6. **Volatility**: Varying based on the asset type

This creates data that preserves the statistical properties of real market data while being generated on demand.

## Data Preprocessing

Raw market data often needs preprocessing before it can be used by trading strategies:

### Common Preprocessing Steps

1. **Cleaning**: Handling missing values, outliers, etc.
2. **Feature Engineering**: Creating indicators like moving averages, RSI, etc.
3. **Normalization**: Scaling data for machine learning algorithms
4. **Time Series Transformations**: Converting to returns, log returns, etc.

Most strategies in Mercurio AI include their own preprocessing methods:

```python
from app.strategies.moving_average import MovingAverageStrategy

# Create a strategy
strategy = MovingAverageStrategy(short_window=10, long_window=30)

async def process_data(raw_data):
    # Strategy-specific preprocessing
    processed_data = await strategy.preprocess_data(raw_data)
    
    # The processed data now includes indicators needed by the strategy
    # For example, short and long moving averages
    print("Available columns after preprocessing:")
    print(processed_data.columns)
    
    return processed_data
```

### Creating Custom Indicators

You can also create custom indicators for your strategies:

```python
def add_custom_indicators(data):
    """Add custom technical indicators to OHLCV data."""
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Example: Hull Moving Average (HMA)
    period = 20
    half_period = period // 2
    sqrt_period = int(period ** 0.5)
    
    # Step 1: Calculate WMA with period/2
    df['wma_half'] = df['close'].rolling(window=half_period).apply(
        lambda x: sum((i+1) * x.iloc[i] for i in range(len(x))) / sum(i+1 for i in range(len(x)))
    )
    
    # Step 2: Calculate WMA with period
    df['wma_full'] = df['close'].rolling(window=period).apply(
        lambda x: sum((i+1) * x.iloc[i] for i in range(len(x))) / sum(i+1 for i in range(len(x)))
    )
    
    # Step 3: Calculate 2*WMA(half period) - WMA(full period)
    df['hma_raw'] = 2 * df['wma_half'] - df['wma_full']
    
    # Step 4: Calculate WMA of raw HMA with sqrt(period)
    df['hma'] = df['hma_raw'].rolling(window=sqrt_period).apply(
        lambda x: sum((i+1) * x.iloc[i] for i in range(len(x))) / sum(i+1 for i in range(len(x)))
    )
    
    # Clean up intermediate columns
    df = df.drop(['wma_half', 'wma_full', 'hma_raw'], axis=1)
    
    return df
```

## Data Visualization

Visualizing market data is crucial for understanding market behavior and strategy performance:

```python
import matplotlib.pyplot as plt
import pandas as pd

def visualize_market_data(data):
    """Visualize OHLCV data with volume."""
    plt.figure(figsize=(12, 8))
    
    # Create two subplots - price and volume
    ax1 = plt.subplot(2, 1, 1)  # Price plot
    ax2 = plt.subplot(2, 1, 2)  # Volume plot
    
    # Plot price data
    ax1.plot(data.index, data['close'], label='Close Price')
    
    # If we have moving averages, plot them
    if 'MA_10' in data.columns:
        ax1.plot(data.index, data['MA_10'], label='10-day MA')
    if 'MA_30' in data.columns:
        ax1.plot(data.index, data['MA_30'], label='30-day MA')
    
    ax1.set_title('Price Chart')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    # Plot volume data
    ax2.bar(data.index, data['volume'])
    ax2.set_title('Volume')
    ax2.set_ylabel('Volume')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
```

## Working with Different Timeframes

Different trading strategies work best with different timeframes. Mercurio AI makes it easy to work with various timeframes:

```python
async def compare_timeframes():
    market_data = MarketDataService()
    
    # Same asset, different timeframes
    daily_data = await market_data.get_historical_data(
        symbol="AAPL",
        start_date="2024-01-01",
        end_date="2024-04-25",
        timeframe="1d"
    )
    
    weekly_data = await market_data.get_historical_data(
        symbol="AAPL",
        start_date="2024-01-01",
        end_date="2024-04-25",
        timeframe="1w"
    )
    
    monthly_data = await market_data.get_historical_data(
        symbol="AAPL",
        start_date="2024-01-01",
        end_date="2024-04-25",
        timeframe="1mo"
    )
    
    print(f"Daily data points: {len(daily_data)}")
    print(f"Weekly data points: {len(weekly_data)}")
    print(f"Monthly data points: {len(monthly_data)}")
    
    return daily_data, weekly_data, monthly_data
```

### Converting Between Timeframes

You can also convert data from one timeframe to another:

```python
def resample_timeframe(data, new_timeframe):
    """
    Resample OHLCV data to a new timeframe.
    
    Parameters:
    - data: DataFrame with OHLCV data
    - new_timeframe: Target timeframe (e.g., 'W' for weekly, 'M' for monthly)
    
    Returns:
    - Resampled DataFrame
    """
    # Make sure the index is a datetime index
    if not isinstance(data.index, pd.DatetimeIndex):
        data = data.set_index('timestamp')
    
    # Resample rules
    # 'W' - week start
    # 'M' - month end
    # 'D' - day
    # 'H' - hour
    
    # Resample the data
    resampled = data.resample(new_timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })
    
    return resampled
```

## Data Storage and Management

For larger projects, you might want to store market data locally:

```python
import os
import pandas as pd

def save_market_data(data, symbol, timeframe):
    """Save market data to a CSV file."""
    # Create the data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Create a filename
    filename = f"data/{symbol}_{timeframe}.csv"
    
    # Save to CSV
    data.to_csv(filename)
    print(f"Saved {len(data)} records to {filename}")

def load_market_data(symbol, timeframe):
    """Load market data from a CSV file."""
    filename = f"data/{symbol}_{timeframe}.csv"
    
    if os.path.exists(filename):
        data = pd.read_csv(filename, index_col=0, parse_dates=True)
        print(f"Loaded {len(data)} records from {filename}")
        return data
    else:
        print(f"File not found: {filename}")
        return None
```

## Advanced Data Topics

### Handling Multiple Symbols

Trading across multiple assets requires careful data management:

```python
async def get_multi_symbol_data(symbols, start_date, end_date, timeframe="1d"):
    """Get data for multiple symbols."""
    market_data = MarketDataService()
    
    # Dictionary to hold data for each symbol
    data_dict = {}
    
    for symbol in symbols:
        try:
            data = await market_data.get_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe
            )
            data_dict[symbol] = data
            print(f"Retrieved {len(data)} records for {symbol}")
        except Exception as e:
            print(f"Error retrieving data for {symbol}: {e}")
    
    return data_dict
```

### Creating a Market Calendar

For proper backtesting and simulation, understanding market calendars is important:

```python
from datetime import datetime, timedelta

def generate_market_calendar(start_date, end_date, market_type="stock"):
    """
    Generate a simple market calendar.
    
    Parameters:
    - start_date: Start date
    - end_date: End date
    - market_type: 'stock' or 'crypto'
    
    Returns:
    - List of trading days
    """
    # Create a date range
    curr_date = start_date
    trading_days = []
    
    while curr_date <= end_date:
        # For stock markets, only include weekdays
        if market_type == "stock":
            if curr_date.weekday() < 5:  # 0-4 are Monday to Friday
                trading_days.append(curr_date)
        # For crypto markets, include all days
        else:
            trading_days.append(curr_date)
        
        curr_date += timedelta(days=1)
    
    return trading_days
```

## Best Practices for Data Management

### 1. Always Validate Your Data

Before using data in a strategy, always validate it:

```python
def validate_market_data(data):
    """Validate market data for common issues."""
    issues = []
    
    # Check for missing values
    missing = data.isnull().sum()
    if missing.sum() > 0:
        issues.append(f"Missing values detected: {missing}")
    
    # Check for duplicate timestamps
    if data.index.duplicated().any():
        issues.append("Duplicate timestamps detected")
    
    # Check for negative prices
    if (data[['open', 'high', 'low', 'close']] < 0).any().any():
        issues.append("Negative prices detected")
    
    # Check for high-low inconsistency
    if (data['low'] > data['high']).any():
        issues.append("Found instances where low is greater than high")
    
    # Check for open-close outside high-low range
    outside_range = ((data['open'] > data['high']) | 
                     (data['open'] < data['low']) | 
                     (data['close'] > data['high']) | 
                     (data['close'] < data['low']))
    if outside_range.any():
        issues.append("Found prices outside the high-low range")
    
    # Check for large gaps between days
    if isinstance(data.index, pd.DatetimeIndex) and len(data) > 1:
        gaps = data.index.to_series().diff().dt.days
        large_gaps = gaps[gaps > 5]
        if not large_gaps.empty:
            issues.append(f"Found {len(large_gaps)} large gaps (>5 days) in the data")
    
    return issues
```

### 2. Use Caching for Performance

Retrieving data repeatedly can be slow. Use caching to improve performance:

```python
class SimpleDataCache:
    """A simple cache for market data."""
    
    def __init__(self):
        self.cache = {}
    
    def get(self, key):
        """Get data from cache if it exists."""
        return self.cache.get(key)
    
    def set(self, key, data):
        """Store data in the cache."""
        self.cache[key] = data
    
    def clear(self):
        """Clear the cache."""
        self.cache = {}

# Example usage
data_cache = SimpleDataCache()

async def get_cached_data(symbol, start_date, end_date, timeframe="1d"):
    """Get data with caching."""
    # Create a cache key
    cache_key = f"{symbol}_{start_date}_{end_date}_{timeframe}"
    
    # Check if in cache
    cached_data = data_cache.get(cache_key)
    if cached_data is not None:
        print(f"Using cached data for {symbol}")
        return cached_data
    
    # Not in cache, fetch from service
    market_data = MarketDataService()
    data = await market_data.get_historical_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe
    )
    
    # Store in cache
    data_cache.set(cache_key, data)
    print(f"Cached data for {symbol}")
    
    return data
```

### 3. Handle Data Consistency Across Assets

When trading multiple assets, ensure data consistency:

```python
def align_multi_symbol_data(data_dict):
    """
    Align data for multiple symbols to have the same dates.
    
    Parameters:
    - data_dict: Dictionary with symbol as key and DataFrame as value
    
    Returns:
    - Dictionary with aligned DataFrames
    """
    # Find common dates
    common_dates = None
    
    for symbol, data in data_dict.items():
        if common_dates is None:
            common_dates = set(data.index)
        else:
            common_dates = common_dates.intersection(set(data.index))
    
    # Convert to sorted list
    common_dates = sorted(list(common_dates))
    
    # Filter data to include only common dates
    aligned_data = {}
    for symbol, data in data_dict.items():
        aligned_data[symbol] = data.loc[common_dates]
    
    return aligned_data
```

## Next Steps

Now that you understand how to work with market data in Mercurio AI, you're ready to explore basic trading strategies. In the next chapter, we'll dive into simple strategies that you can use to start trading.

Continue to [Chapter 6: Basic Trading Strategies](./06-basic-strategies.md) to learn about implementing your first strategies in Mercurio AI.

---

**Key Takeaways:**
- Market data is the foundation of trading strategies, typically in OHLCV format
- Mercurio AI supports multiple data sources with automatic fallbacks
- Sample data generation provides realistic data when external sources aren't available
- Data preprocessing is crucial for strategy development
- Working with different timeframes allows for various trading approaches
- Best practices include data validation, caching, and ensuring consistency across assets
