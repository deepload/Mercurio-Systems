# Chapter 6: Basic Trading Strategies

Welcome to Chapter 6! Now that you understand the platform and how to work with market data, it's time to explore basic trading strategies in Mercurio AI. This chapter will focus on traditional strategies that are easy to understand and implement.

## Understanding Trading Strategies

A trading strategy is a set of rules that determine when to buy, sell, or hold an asset. Good strategies typically include:

- **Entry Rules**: Conditions for opening a position
- **Exit Rules**: Conditions for closing a position
- **Position Sizing**: How much to invest in each trade
- **Risk Management**: How to protect capital from significant losses

Let's start with the most fundamental strategy in Mercurio AI: the Moving Average Strategy.

## Moving Average Strategy

The Moving Average (MA) strategy is one of the oldest and most widely used trading strategies. It's based on the crossover of two moving averages of different periods.

### How Moving Averages Work

A moving average smooths out price data by creating a constantly updated average price over a specific time period:

- **Simple Moving Average (SMA)**: Average of prices over a period
- **Exponential Moving Average (EMA)**: Weighted average giving more importance to recent prices

When a shorter-period MA crosses above a longer-period MA, it's considered a bullish signal (buy). When it crosses below, it's considered a bearish signal (sell).

### Implementing a Moving Average Strategy

In Mercurio AI, the MovingAverageStrategy is already implemented:

```python
from app.strategies.moving_average import MovingAverageStrategy
import asyncio
from app.services.market_data import MarketDataService

async def run_moving_average_strategy():
    # Initialize market data service
    market_data = MarketDataService()
    
    # Get historical data
    data = await market_data.get_historical_data(
        symbol="AAPL",
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    
    # Create a moving average strategy
    # Short window: 10 days, Long window: 30 days
    strategy = MovingAverageStrategy(
        short_window=10,
        long_window=30,
        use_ml=False  # Start with simple strategy without ML
    )
    
    # Preprocess the data
    processed_data = await strategy.preprocess_data(data)
    
    # Generate a signal for the latest data point
    signal, confidence = await strategy.predict(processed_data)
    
    print(f"Signal: {signal}, Confidence: {confidence:.2f}")
    
    # Run a backtest
    backtest_result = await strategy.backtest(
        data=processed_data,
        initial_capital=10000
    )
    
    # Print backtest results
    print(f"Initial Capital: $10,000.00")
    print(f"Final Capital: ${backtest_result['final_equity']:.2f}")
    print(f"Total Return: {((backtest_result['final_equity'] / 10000) - 1) * 100:.2f}%")
    print(f"Number of Trades: {len(backtest_result['trades'])}")
    
    return backtest_result

# Run the strategy
if __name__ == "__main__":
    result = asyncio.run(run_moving_average_strategy())
```

### Visualizing Moving Average Signals

Let's visualize the moving average strategy to better understand it:

```python
import matplotlib.pyplot as plt
import pandas as pd

def visualize_ma_strategy(data):
    """Visualize Moving Average strategy signals."""
    plt.figure(figsize=(12, 8))
    
    # Price and moving averages
    plt.subplot(2, 1, 1)
    plt.plot(data.index, data['close'], label='Close Price')
    plt.plot(data.index, data['MA_short'], label=f'Short MA')
    plt.plot(data.index, data['MA_long'], label=f'Long MA')
    
    # Add buy/sell signals
    buy_signals = data[data['signal'] == 1]
    sell_signals = data[data['signal'] == -1]
    
    plt.scatter(buy_signals.index, buy_signals['close'], 
                marker='^', color='green', s=100, label='Buy')
    plt.scatter(sell_signals.index, sell_signals['close'], 
                marker='v', color='red', s=100, label='Sell')
    
    plt.title('Moving Average Strategy')
    plt.ylabel('Price')
    plt.legend()
    
    # Volume
    plt.subplot(2, 1, 2)
    plt.bar(data.index, data['volume'])
    plt.title('Volume')
    plt.ylabel('Volume')
    
    plt.tight_layout()
    plt.show()
```

### Optimizing the Moving Average Strategy

Different assets and market conditions may require different moving average periods. Let's create a function to find optimal parameters:

```python
async def optimize_ma_strategy(symbol, start_date, end_date):
    """Find optimal MA parameters for a given asset and time period."""
    market_data = MarketDataService()
    
    # Get data
    data = await market_data.get_historical_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date
    )
    
    # Ranges to test
    short_windows = range(5, 21, 5)  # 5, 10, 15, 20
    long_windows = range(20, 61, 10)  # 20, 30, 40, 50, 60
    
    best_return = -float('inf')
    best_params = None
    results = []
    
    # Test each combination
    for short in short_windows:
        for long in long_windows:
            if short >= long:
                continue  # Short must be less than long
            
            # Create and test strategy
            strategy = MovingAverageStrategy(
                short_window=short,
                long_window=long,
                use_ml=False
            )
            
            # Preprocess and backtest
            processed_data = await strategy.preprocess_data(data)
            backtest_result = await strategy.backtest(
                data=processed_data,
                initial_capital=10000
            )
            
            # Calculate return
            total_return = (backtest_result['final_equity'] / 10000 - 1) * 100
            
            # Save result
            results.append({
                'short_window': short,
                'long_window': long,
                'total_return': total_return,
                'trade_count': len(backtest_result['trades'])
            })
            
            # Check if this is the best so far
            if total_return > best_return:
                best_return = total_return
                best_params = (short, long)
    
    # Convert results to DataFrame for analysis
    results_df = pd.DataFrame(results)
    
    print(f"Best Parameters: short_window={best_params[0]}, long_window={best_params[1]}")
    print(f"Best Return: {best_return:.2f}%")
    
    return best_params, results_df
```

## Mean Reversion Strategy

Mean reversion is based on the idea that prices tend to return to their average over time. When prices deviate significantly from their mean, they're likely to revert back.

### Relative Strength Index (RSI) Strategy

The RSI is a momentum oscillator that measures the speed and change of price movements. It ranges from 0 to 100:

- RSI > 70: Asset may be overbought (sell signal)
- RSI < 30: Asset may be oversold (buy signal)

Let's implement a simple RSI strategy:

```python
import pandas as pd
import numpy as np

class RSIStrategy:
    """A simple RSI mean reversion strategy."""
    
    def __init__(self, period=14, overbought=70, oversold=30):
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
    
    async def preprocess_data(self, data):
        """Add RSI indicator to the data."""
        df = data.copy()
        
        # Calculate price changes
        df['price_change'] = df['close'].diff()
        
        # Calculate gains and losses
        df['gain'] = df['price_change'].apply(lambda x: x if x > 0 else 0)
        df['loss'] = df['price_change'].apply(lambda x: abs(x) if x < 0 else 0)
        
        # Calculate average gains and losses
        df['avg_gain'] = df['gain'].rolling(window=self.period).mean()
        df['avg_loss'] = df['loss'].rolling(window=self.period).mean()
        
        # Calculate relative strength (RS)
        df['rs'] = df['avg_gain'] / df['avg_loss']
        
        # Calculate RSI
        df['rsi'] = 100 - (100 / (1 + df['rs']))
        
        # Generate signals
        df['signal'] = 0
        df.loc[df['rsi'] < self.oversold, 'signal'] = 1  # Buy signal
        df.loc[df['rsi'] > self.overbought, 'signal'] = -1  # Sell signal
        
        return df
    
    async def predict(self, data):
        """Generate trading signal for the latest data point."""
        # Get the latest RSI value
        latest_rsi = data['rsi'].iloc[-1]
        
        # Determine signal
        if latest_rsi < self.oversold:
            signal = "BUY"
            confidence = (self.oversold - latest_rsi) / self.oversold
        elif latest_rsi > self.overbought:
            signal = "SELL"
            confidence = (latest_rsi - self.overbought) / (100 - self.overbought)
        else:
            signal = "HOLD"
            confidence = 0.5
        
        return signal, min(confidence, 1.0)
    
    async def backtest(self, data, initial_capital=10000):
        """Backtest the RSI strategy."""
        df = data.copy()
        
        # Initialize variables
        capital = initial_capital
        position = 0
        equity_curve = [capital]
        trades = []
        
        # Iterate through data points
        for i in range(1, len(df)):
            date = df.index[i]
            price = df['close'].iloc[i]
            signal = df['signal'].iloc[i]
            
            # Execute trades
            if signal == 1 and position == 0:  # Buy signal
                # Calculate position size (invest all capital)
                position = capital / price
                entry_price = price
                entry_date = date
                capital = 0
                trades.append({
                    'type': 'buy',
                    'date': date,
                    'price': price,
                    'quantity': position
                })
            
            elif signal == -1 and position > 0:  # Sell signal
                # Sell entire position
                capital = position * price
                position = 0
                trades.append({
                    'type': 'sell',
                    'date': date,
                    'price': price,
                    'quantity': position
                })
            
            # Update equity
            current_equity = capital + (position * price)
            equity_curve.append(current_equity)
        
        # Final equity calculation
        final_equity = capital + (position * df['close'].iloc[-1])
        
        return {
            'final_equity': final_equity,
            'equity_curve': equity_curve,
            'trades': trades
        }
```

### Using the RSI Strategy

Now let's use our RSI strategy:

```python
async def run_rsi_strategy():
    # Initialize market data service
    market_data = MarketDataService()
    
    # Get historical data
    data = await market_data.get_historical_data(
        symbol="AAPL",
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    
    # Create RSI strategy
    strategy = RSIStrategy(period=14, overbought=70, oversold=30)
    
    # Preprocess the data
    processed_data = await strategy.preprocess_data(data)
    
    # Generate a signal for the latest data point
    signal, confidence = await strategy.predict(processed_data)
    
    print(f"RSI Signal: {signal}, Confidence: {confidence:.2f}")
    
    # Run a backtest
    backtest_result = await strategy.backtest(
        data=processed_data,
        initial_capital=10000
    )
    
    # Print backtest results
    print(f"Initial Capital: $10,000.00")
    print(f"Final Capital: ${backtest_result['final_equity']:.2f}")
    print(f"Total Return: {((backtest_result['final_equity'] / 10000) - 1) * 100:.2f}%")
    print(f"Number of Trades: {len(backtest_result['trades'])}")
    
    return backtest_result
```

## Bollinger Bands Strategy

Bollinger Bands measure market volatility and can identify overbought or oversold conditions.

### How Bollinger Bands Work

Bollinger Bands consist of:
- A middle band (typically a 20-day simple moving average)
- An upper band (middle band + 2 standard deviations)
- A lower band (middle band - 2 standard deviations)

When the price touches the upper band, it may be overbought; when it touches the lower band, it may be oversold.

### Implementing a Bollinger Bands Strategy

```python
class BollingerBandsStrategy:
    """A strategy based on Bollinger Bands."""
    
    def __init__(self, window=20, num_std=2):
        self.window = window
        self.num_std = num_std
    
    async def preprocess_data(self, data):
        """Add Bollinger Bands to the data."""
        df = data.copy()
        
        # Calculate middle band (SMA)
        df['middle_band'] = df['close'].rolling(window=self.window).mean()
        
        # Calculate standard deviation
        df['std'] = df['close'].rolling(window=self.window).std()
        
        # Calculate upper and lower bands
        df['upper_band'] = df['middle_band'] + (df['std'] * self.num_std)
        df['lower_band'] = df['middle_band'] - (df['std'] * self.num_std)
        
        # Generate signals
        df['signal'] = 0
        df['distance'] = (df['close'] - df['middle_band']) / (df['upper_band'] - df['middle_band'])
        
        # Buy when price touches lower band
        df.loc[df['close'] <= df['lower_band'], 'signal'] = 1
        
        # Sell when price touches upper band
        df.loc[df['close'] >= df['upper_band'], 'signal'] = -1
        
        return df
    
    async def predict(self, data):
        """Generate trading signal for the latest data point."""
        # Get the latest values
        latest_close = data['close'].iloc[-1]
        latest_upper = data['upper_band'].iloc[-1]
        latest_lower = data['lower_band'].iloc[-1]
        latest_distance = data['distance'].iloc[-1]
        
        # Determine signal
        if latest_close <= latest_lower:
            signal = "BUY"
            # Confidence increases as price drops below lower band
            confidence = min(1.0, abs(latest_distance))
        elif latest_close >= latest_upper:
            signal = "SELL"
            # Confidence increases as price rises above upper band
            confidence = min(1.0, abs(latest_distance))
        else:
            signal = "HOLD"
            # Neutral zone
            confidence = 0.5
        
        return signal, confidence
    
    async def backtest(self, data, initial_capital=10000):
        """Backtest the Bollinger Bands strategy."""
        # Similar to RSI backtest implementation
        # ...
```

## Pattern Recognition Strategy

Pattern recognition strategies identify specific price patterns that may indicate future price movements.

### Implementing a Simple Pattern Recognition Strategy

Let's implement a strategy that identifies double bottoms:

```python
class DoubleBottomStrategy:
    """A strategy that identifies double bottom patterns."""
    
    def __init__(self, window=20, threshold=0.03):
        self.window = window
        self.threshold = threshold  # % difference allowed between bottoms
    
    async def preprocess_data(self, data):
        """Identify double bottom patterns."""
        df = data.copy()
        
        # Find local minima
        df['is_min'] = 0
        
        for i in range(self.window, len(df) - self.window):
            # Current price window
            window_prices = df['low'].iloc[i-self.window:i+self.window+1]
            
            # If current price is the minimum in the window
            if df['low'].iloc[i] == window_prices.min():
                df['is_min'].iloc[i] = 1
        
        # Identify double bottoms
        df['double_bottom'] = 0
        
        for i in range(2*self.window, len(df)):
            # Find two recent minima
            recent_mins = df[df['is_min'] == 1].iloc[i-4*self.window:i]
            
            if len(recent_mins) >= 2:
                # Get the two most recent minima
                last_two_mins = recent_mins.iloc[-2:]['low'].values
                
                # Calculate percentage difference
                min1, min2 = last_two_mins
                diff_pct = abs(min1 - min2) / min1
                
                # If bottoms are within threshold % of each other
                if diff_pct <= self.threshold:
                    # If current price is higher than both bottoms
                    if df['close'].iloc[i] > max(last_two_mins):
                        df['double_bottom'].iloc[i] = 1
        
        # Generate signals
        df['signal'] = 0
        df.loc[df['double_bottom'] == 1, 'signal'] = 1  # Buy on double bottom
        
        return df
```

## Combining Multiple Strategies

One of the most powerful approaches is to combine multiple strategies to create a more robust system.

### Creating a Combined Strategy

```python
class CombinedStrategy:
    """A strategy that combines multiple sub-strategies."""
    
    def __init__(self, strategies, weights=None):
        self.strategies = strategies
        
        # Equal weights if not specified
        if weights is None:
            self.weights = [1/len(strategies)] * len(strategies)
        else:
            self.weights = weights
    
    async def preprocess_data(self, data):
        """Preprocess data for all sub-strategies."""
        processed_data = {}
        
        for i, strategy in enumerate(self.strategies):
            processed_data[f"strategy_{i}"] = await strategy.preprocess_data(data)
        
        return processed_data
    
    async def predict(self, processed_data):
        """Generate a weighted signal from all sub-strategies."""
        signals = []
        confidences = []
        
        for i, strategy in enumerate(self.strategies):
            strategy_data = processed_data[f"strategy_{i}"]
            signal, confidence = await strategy.predict(strategy_data)
            
            # Convert signal to numeric
            if signal == "BUY":
                numeric_signal = 1
            elif signal == "SELL":
                numeric_signal = -1
            else:
                numeric_signal = 0
            
            signals.append(numeric_signal)
            confidences.append(confidence)
        
        # Calculate weighted signal
        weighted_signal = sum(s * c * w for s, c, w in zip(signals, confidences, self.weights))
        
        # Determine final signal
        if weighted_signal > 0.2:
            return "BUY", abs(weighted_signal)
        elif weighted_signal < -0.2:
            return "SELL", abs(weighted_signal)
        else:
            return "HOLD", abs(weighted_signal)
```

## Risk Management Strategies

Effective risk management is crucial for long-term trading success.

### Position Sizing

One common approach is the percentage risk model:

```python
def calculate_position_size(capital, risk_percentage, entry_price, stop_loss_price):
    """
    Calculate position size based on risk percentage.
    
    Args:
        capital: Available capital
        risk_percentage: Percentage of capital to risk (e.g., 1 for 1%)
        entry_price: Entry price
        stop_loss_price: Stop loss price
    
    Returns:
        Quantity to buy
    """
    # Calculate risk amount
    risk_amount = capital * (risk_percentage / 100)
    
    # Calculate risk per share
    risk_per_share = abs(entry_price - stop_loss_price)
    
    # Calculate position size
    position_size = risk_amount / risk_per_share
    
    return int(position_size)
```

### Stop Loss and Take Profit

Implementing stop loss and take profit levels in your strategy:

```python
def add_stop_loss_take_profit(data, stop_loss_pct=0.05, take_profit_pct=0.1):
    """
    Add stop loss and take profit levels to trading signals.
    
    Args:
        data: DataFrame with trading signals
        stop_loss_pct: Stop loss percentage
        take_profit_pct: Take profit percentage
    
    Returns:
        DataFrame with stop loss and take profit levels
    """
    df = data.copy()
    
    # Initialize columns
    df['stop_loss'] = None
    df['take_profit'] = None
    
    # Find buy signals
    buy_signals = df[df['signal'] == 1].index
    
    # Set stop loss and take profit for each buy signal
    for signal_date in buy_signals:
        entry_price = df.loc[signal_date, 'close']
        
        # Calculate levels
        stop_loss = entry_price * (1 - stop_loss_pct)
        take_profit = entry_price * (1 + take_profit_pct)
        
        # Add to dataframe
        df.loc[signal_date, 'stop_loss'] = stop_loss
        df.loc[signal_date, 'take_profit'] = take_profit
    
    return df
```

## Next Steps

Now that you understand the basics of trading strategies in Mercurio AI, you're ready to explore more advanced strategies powered by machine learning. In the next chapter, we'll dive into how Mercurio AI leverages machine learning for more sophisticated trading approaches.

Continue to [Chapter 7: Advanced Trading Strategies](./07-advanced-strategies.md) to learn about ML-powered strategies including LSTM and Transformer models.

---

**Key Takeaways:**
- Basic trading strategies include Moving Average, Mean Reversion, and Pattern Recognition
- Mercurio AI includes pre-built implementations of common strategies
- Strategy optimization helps find the best parameters for specific assets and time periods
- Combining multiple strategies can create more robust trading systems
- Risk management is essential for long-term trading success
- The strategy interface in Mercurio AI is consistent across all strategy types
