# Chapter 7: Advanced Trading Strategies

Welcome to Chapter 7! Now that you've mastered basic trading strategies, it's time to explore Mercurio AI's advanced, machine learning-powered strategies. These strategies can potentially capture complex patterns that traditional approaches might miss.

## Introduction to ML-Based Trading

Machine learning (ML) strategies differ from traditional approaches in several ways:

- They can identify non-linear patterns in market data
- They can adapt to changing market conditions
- They can process multiple data inputs simultaneously
- They often require more data and computational resources

Let's explore the main ML-based strategies available in Mercurio AI.

## ML-Enhanced Moving Average Strategy

Before diving into the most advanced strategies, let's start with a familiar one: the Moving Average strategy with ML enhancement.

### How It Works

The standard MovingAverageStrategy in Mercurio AI has an `use_ml` parameter that enables an ML enhancement layer:

```python
from app.strategies.moving_average import MovingAverageStrategy
import asyncio
from app.services.market_data import MarketDataService

async def run_ml_enhanced_ma_strategy():
    # Initialize market data service
    market_data = MarketDataService()
    
    # Get historical data
    data = await market_data.get_historical_data(
        symbol="AAPL",
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    
    # Create a moving average strategy with ML enhancement
    strategy = MovingAverageStrategy(
        short_window=10,
        long_window=30,
        use_ml=True  # Enable ML enhancement
    )
    
    # Preprocess the data
    processed_data = await strategy.preprocess_data(data)
    
    # Generate a signal for the latest data point
    signal, confidence = await strategy.predict(processed_data)
    
    print(f"ML-Enhanced MA Signal: {signal}, Confidence: {confidence:.2f}")
    
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

### What the ML Enhancement Does

When `use_ml=True`, the strategy:

1. Uses the traditional moving average crossover as initial signals
2. Adds additional features like volatility, momentum, and trend strength
3. Trains a simple machine learning model to refine the signal
4. Provides a confidence score based on the model's prediction

This creates a "best of both worlds" approach - the reliability of moving averages combined with the pattern recognition of machine learning.

## LSTM Predictor Strategy

Long Short-Term Memory (LSTM) networks are a type of recurrent neural network (RNN) well-suited for sequence prediction problems like price forecasting.

### Understanding LSTMs

LSTMs have several advantages for financial time series:

- They can remember patterns over long sequences
- They're good at identifying recurring patterns
- They can learn to ignore irrelevant information
- They're relatively robust to noise in the data

### Implementing the LSTM Predictor Strategy

In Mercurio AI, the LSTMPredictorStrategy is pre-built and ready to use:

```python
from app.strategies.lstm_predictor import LSTMPredictorStrategy

async def run_lstm_strategy():
    # Initialize market data service
    market_data = MarketDataService()
    
    # Get historical data (we need more data for training)
    data = await market_data.get_historical_data(
        symbol="AAPL",
        start_date="2022-01-01",  # More data for LSTM
        end_date="2023-12-31"
    )
    
    # Create an LSTM strategy
    strategy = LSTMPredictorStrategy(
        sequence_length=30,  # Look back 30 days for patterns
        epochs=50,  # Training epochs
        hidden_units=64  # Complexity of the model
    )
    
    # Preprocess the data
    processed_data = await strategy.preprocess_data(data)
    
    # Generate a signal for the latest data point
    signal, confidence = await strategy.predict(processed_data)
    
    print(f"LSTM Signal: {signal}, Confidence: {confidence:.2f}")
    
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

### How LSTM Makes Predictions

The LSTM strategy works as follows:

1. **Data Preprocessing**: Normalizes price data and calculates technical indicators
2. **Sequence Creation**: Forms training sequences of historical data
3. **Model Training**: Trains the LSTM model to predict future price movements
4. **Signal Generation**: Converts predictions into BUY/SELL/HOLD signals
5. **Confidence Calculation**: Estimates confidence based on prediction strength

### Customizing the LSTM Strategy

You can customize several aspects of the LSTM strategy:

```python
# Customize the LSTM strategy
lstm_strategy = LSTMPredictorStrategy(
    sequence_length=30,  # How many past days to consider
    epochs=50,           # Training iterations
    hidden_units=64,     # Model complexity
    dropout=0.2,         # Regularization to prevent overfitting
    features=[           # Custom features to include
        'close',
        'volume',
        'ma_10',
        'ma_30',
        'rsi_14'
    ]
)
```

## Transformer Strategy

Transformer models have revolutionized many machine learning fields and are now being applied to financial markets in Mercurio AI.

### Understanding Transformers

Transformers have several advantages for market prediction:

- They can capture relationships between distant time points
- They process sequences in parallel (faster than RNNs)
- They use attention mechanisms to focus on relevant parts of data
- They're state-of-the-art in many sequence modeling tasks

### Implementing the Transformer Strategy

In Mercurio AI, the TransformerStrategy is ready to use:

```python
from app.strategies.transformer_strategy import TransformerStrategy

async def run_transformer_strategy():
    # Initialize market data service
    market_data = MarketDataService()
    
    # Get historical data (we need more data for training)
    data = await market_data.get_historical_data(
        symbol="AAPL",
        start_date="2022-01-01",
        end_date="2023-12-31"
    )
    
    # Create a Transformer strategy
    strategy = TransformerStrategy(
        sequence_length=60,     # Look back 60 days
        num_layers=2,           # Transformer layers
        d_model=64,             # Embedding dimension
        num_heads=4,            # Attention heads
        epochs=50               # Training epochs
    )
    
    # Preprocess the data
    processed_data = await strategy.preprocess_data(data)
    
    # Generate a signal for the latest data point
    signal, confidence = await strategy.predict(processed_data)
    
    print(f"Transformer Signal: {signal}, Confidence: {confidence:.2f}")
    
    # Run a backtest
    backtest_result = await strategy.backtest(
        data=processed_data,
        initial_capital=10000
    )
    
    # Print backtest results
    print(f"Initial Capital: $10,000.00")
    print(f"Final Capital: ${backtest_result['final_equity']:.2f}")
    print(f"Total Return: {((backtest_result['final_equity'] / 10000) - 1) * 100:.2f}%")
    
    return backtest_result
```

### How Transformer Makes Predictions

The Transformer strategy works as follows:

1. **Feature Engineering**: Creates a rich set of features from price data
2. **Sequence Formation**: Prepares sequences for the transformer model
3. **Self-Attention**: Uses attention mechanisms to weigh the importance of different time points
4. **Prediction**: Forecasts future price movements
5. **Signal Generation**: Converts predictions to actionable trading signals

### When to Use Transformer Strategy

The Transformer strategy tends to perform best when:

- You have substantial historical data (at least 1 year)
- You're trading liquid assets with clear patterns
- You have sufficient computational resources for training
- The market has complex, non-linear relationships to capture

## LLM Strategy

The LLM (Large Language Model) strategy represents the cutting edge of AI-powered trading, leveraging natural language understanding alongside price data.

### Understanding LLM-Based Trading

LLM-based trading has several unique advantages:

- It can incorporate textual data (news, social media, etc.)
- It can understand market sentiment
- It can recognize complex narratives and themes
- It can adapt to new market conditions quickly

### Implementing the LLM Strategy

In Mercurio AI, the LLMStrategy is available for advanced users:

```python
from app.strategies.llm_strategy import LLMStrategy

async def run_llm_strategy():
    # Initialize market data service
    market_data = MarketDataService()
    
    # Get historical data
    data = await market_data.get_historical_data(
        symbol="AAPL",
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    
    # Create an LLM strategy
    strategy = LLMStrategy(
        model_name="mercurio-mini",  # Default smaller model
        include_news=True,           # Incorporate news data
        sentiment_analysis=True      # Analyze market sentiment
    )
    
    # Preprocess the data
    processed_data = await strategy.preprocess_data(data)
    
    # Generate a signal for the latest data point
    signal, confidence = await strategy.predict(processed_data)
    
    print(f"LLM Signal: {signal}, Confidence: {confidence:.2f}")
    print(f"Reasoning: {strategy.get_reasoning()}")  # Get model's reasoning
    
    # Run a backtest
    backtest_result = await strategy.backtest(
        data=processed_data,
        initial_capital=10000
    )
    
    # Print backtest results
    print(f"Initial Capital: $10,000.00")
    print(f"Final Capital: ${backtest_result['final_equity']:.2f}")
    print(f"Total Return: {((backtest_result['final_equity'] / 10000) - 1) * 100:.2f}%")
    
    return backtest_result
```

### How LLM Strategy Works

The LLM strategy operates differently from other strategies:

1. **Data Collection**: Gathers both price data and relevant textual data
2. **Contextual Analysis**: Uses LLM to understand market context
3. **Pattern Recognition**: Identifies both technical and narrative patterns
4. **Prediction Generation**: Creates forecasts based on comprehensive analysis
5. **Reasoning**: Provides human-readable explanations for its decisions

### LLM Strategy Modes

The LLM strategy can operate in different modes:

```python
# Technical mode - focus on price patterns
llm_technical = LLMStrategy(
    model_name="mercurio-mini",
    mode="technical",
    confidence_threshold=0.7
)

# Sentiment mode - focus on market sentiment
llm_sentiment = LLMStrategy(
    model_name="mercurio-mini",
    mode="sentiment",
    include_news=True,
    news_sources=["bloomberg", "reuters"]
)

# Hybrid mode - combine technical and fundamental analysis
llm_hybrid = LLMStrategy(
    model_name="mercurio-mini",
    mode="hybrid",
    include_fundamentals=True,
    include_news=True
)
```

## Multi-Strategy Ensemble

One of the most powerful approaches in Mercurio AI is to combine multiple ML strategies into an ensemble.

### Creating a Strategy Ensemble

```python
from app.strategies.ensemble import StrategyEnsemble

async def run_ensemble_strategy():
    # Initialize market data service
    market_data = MarketDataService()
    
    # Get historical data
    data = await market_data.get_historical_data(
        symbol="AAPL",
        start_date="2022-01-01",
        end_date="2023-12-31"
    )
    
    # Create individual strategies
    ma_strategy = MovingAverageStrategy(short_window=10, long_window=30, use_ml=True)
    lstm_strategy = LSTMPredictorStrategy(sequence_length=30)
    transformer_strategy = TransformerStrategy(sequence_length=60)
    
    # Create ensemble
    ensemble = StrategyEnsemble(
        strategies=[ma_strategy, lstm_strategy, transformer_strategy],
        weights=[0.3, 0.3, 0.4],  # Weight transformer higher
        voting_method="weighted"  # Use weighted voting
    )
    
    # Preprocess the data
    processed_data = await ensemble.preprocess_data(data)
    
    # Generate a signal for the latest data point
    signal, confidence = await ensemble.predict(processed_data)
    
    print(f"Ensemble Signal: {signal}, Confidence: {confidence:.2f}")
    
    # Run a backtest
    backtest_result = await ensemble.backtest(
        data=processed_data,
        initial_capital=10000
    )
    
    # Print backtest results
    print(f"Initial Capital: $10,000.00")
    print(f"Final Capital: ${backtest_result['final_equity']:.2f}")
    print(f"Total Return: {((backtest_result['final_equity'] / 10000) - 1) * 100:.2f}%")
    
    return backtest_result
```

### Ensemble Voting Methods

The ensemble can use different voting methods:

- **Majority**: Goes with the most common signal
- **Weighted**: Weights signals by both strategy weight and confidence
- **Confidence**: Selects the signal with highest confidence
- **Consensus**: Only acts when all strategies agree

## Hyperparameter Optimization

Advanced strategies have many parameters that can be optimized for better performance.

### Using Grid Search

```python
async def optimize_lstm_hyperparameters(symbol, start_date, end_date):
    """Find optimal LSTM parameters for a given asset."""
    market_data = MarketDataService()
    
    # Get data
    data = await market_data.get_historical_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date
    )
    
    # Parameters to test
    sequence_lengths = [10, 20, 30]
    hidden_units = [32, 64, 128]
    epochs = [30, 50]
    
    best_return = -float('inf')
    best_params = None
    results = []
    
    # Test each combination
    for seq_len in sequence_lengths:
        for units in hidden_units:
            for ep in epochs:
                # Create and test strategy
                strategy = LSTMPredictorStrategy(
                    sequence_length=seq_len,
                    hidden_units=units,
                    epochs=ep
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
                    'sequence_length': seq_len,
                    'hidden_units': units,
                    'epochs': ep,
                    'total_return': total_return
                })
                
                # Check if this is the best so far
                if total_return > best_return:
                    best_return = total_return
                    best_params = (seq_len, units, ep)
    
    print(f"Best Parameters: sequence_length={best_params[0]}, hidden_units={best_params[1]}, epochs={best_params[2]}")
    print(f"Best Return: {best_return:.2f}%")
    
    return best_params, pd.DataFrame(results)
```

## Advanced Feature Engineering

ML strategies benefit greatly from well-engineered features.

### Creating Advanced Features

```python
def create_advanced_features(data):
    """Create advanced features for ML strategies."""
    df = data.copy()
    
    # Price-based features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Volatility features
    df['volatility_5d'] = df['returns'].rolling(5).std()
    df['volatility_20d'] = df['returns'].rolling(20).std()
    
    # Trend features
    df['trend_strength'] = abs(df['close'].rolling(20).mean() - df['close'].rolling(5).mean()) / df['close']
    
    # Volume features
    df['volume_change'] = df['volume'].pct_change()
    df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    # Price pattern features
    df['higher_high'] = (df['high'] > df['high'].shift(1)) & (df['high'].shift(1) > df['high'].shift(2))
    df['lower_low'] = (df['low'] < df['low'].shift(1)) & (df['low'].shift(1) < df['low'].shift(2))
    
    # Technical indicators
    df['rsi_14'] = calculate_rsi(df['close'], 14)
    df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['close'])
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = calculate_bollinger_bands(df['close'])
    
    return df
```

## Practical Considerations for ML Strategies

When using ML strategies, keep these practical considerations in mind:

### Data Requirements

ML strategies typically need more data:

- LSTM: At least 1 year of historical data
- Transformer: At least 2 years of historical data
- LLM: Varies based on model size, but generally 1+ years

### Computational Requirements

Different strategies have different computational needs:

- ML-Enhanced MA: Low (can run on most computers)
- LSTM: Medium (CPU is sufficient, GPU helpful)
- Transformer: High (GPU recommended)
- LLM: Very High (GPU required for full-scale models)

### Fallback Mechanisms

Mercurio AI includes fallback mechanisms for when computational resources are limited:

```python
# Create transformer with fallback options
transformer_strategy = TransformerStrategy(
    sequence_length=60,
    num_layers=2,
    d_model=64,
    enable_fallback=True,  # Enable fallbacks
    fallback_strategy="moving_average"  # Use MA if transformer fails
)
```

### Overfitting Prevention

ML strategies can be prone to overfitting. Mercurio AI includes several techniques to prevent this:

- Train/test splitting
- Regularization (dropout, L1/L2)
- Early stopping
- Cross-validation

## Evaluating ML Strategy Performance

Evaluate ML strategies thoroughly before using them:

```python
async def evaluate_ml_strategy(strategy, symbol, start_date, end_date):
    """Comprehensive evaluation of an ML strategy."""
    market_data = MarketDataService()
    
    # Get data
    data = await market_data.get_historical_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date
    )
    
    # Split into training and testing periods
    split_date = pd.Timestamp(start_date) + (pd.Timestamp(end_date) - pd.Timestamp(start_date)) * 0.7
    train_data = data[data.index <= split_date]
    test_data = data[data.index > split_date]
    
    # Preprocess all data
    processed_data = await strategy.preprocess_data(data)
    
    # Backtest on testing period only
    test_period = processed_data[processed_data.index > split_date]
    backtest_result = await strategy.backtest(
        data=test_period,
        initial_capital=10000
    )
    
    # Calculate performance metrics
    total_return = (backtest_result['final_equity'] / 10000 - 1) * 100
    
    # Calculate Sharpe ratio
    returns = pd.Series(backtest_result['equity_curve']).pct_change().dropna()
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)  # Annualized
    
    # Calculate maximum drawdown
    equity_curve = pd.Series(backtest_result['equity_curve'])
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100
    
    # Print results
    print(f"Strategy: {strategy.__class__.__name__}")
    print(f"Symbol: {symbol}")
    print(f"Testing Period: {split_date} to {end_date}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")
    print(f"Number of Trades: {len(backtest_result['trades'])}")
    
    return {
        'strategy': strategy.__class__.__name__,
        'symbol': symbol,
        'period': f"{split_date} to {end_date}",
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'trade_count': len(backtest_result['trades'])
    }
```

## Next Steps

Now that you understand advanced trading strategies in Mercurio AI, you're ready to explore backtesting in more detail. In the next chapter, we'll dive into how to thoroughly test your strategies against historical data.

Continue to [Chapter 8: Backtesting Your Strategies](./08-backtesting.md) to learn how to evaluate strategy performance with historical data.

---

**Key Takeaways:**
- Mercurio AI includes several advanced ML-based strategies: ML-Enhanced Moving Average, LSTM, Transformer, and LLM
- Each strategy has different data and computational requirements
- ML strategies can capture complex, non-linear patterns in market data
- Strategy ensembles combine multiple strategies for more robust performance
- Advanced feature engineering and hyperparameter optimization are critical for ML strategy success
- Mercurio AI includes fallback mechanisms for when resources are limited
