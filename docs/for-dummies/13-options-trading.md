# Chapter 13: Options Trading Strategies

## Introduction to Options Trading with Mercurio AI

Options trading can seem complex at first, but Mercurio AI simplifies the process with its advanced strategies and automated tools. This chapter will guide you through the basics of options trading within the platform and how to get started even if you're new to options.

### What Are Options?

Options are financial derivatives that give you the right—but not the obligation—to buy or sell an underlying asset at a predetermined price before a specific date. The two basic types of options are:

- **Call Options**: Give the holder the right to *buy* the underlying asset
- **Put Options**: Give the holder the right to *sell* the underlying asset

### Key Options Terminology

Before diving deeper, let's understand some essential options trading terms:

- **Strike Price**: The price at which you can buy (for calls) or sell (for puts) the underlying asset
- **Expiration Date**: The date when the option contract expires
- **Premium**: The price you pay to purchase an option contract
- **In-the-Money (ITM)**: When an option has intrinsic value (call option's strike price < market price; put option's strike price > market price)
- **Out-of-the-Money (OTM)**: When an option has no intrinsic value (call option's strike price > market price; put option's strike price < market price)
- **Greeks**: Metrics that measure different dimensions of risk in options (Delta, Gamma, Theta, Vega)

## Options Strategies in Mercurio AI

Mercurio AI supports a wide range of options strategies, from simple to complex, across all options trading levels:

### Level 1 Strategies

These basic strategies are perfect for beginners:

#### Long Call Strategy

**How it works**: You purchase a call option, giving you the right to buy the underlying asset at the strike price.

**When to use it**: When you expect the price of the underlying asset to rise significantly.

**Configuration in Mercurio AI**:
```bash
python -m scripts.options.run_daily_options_trader --strategy LONG_CALL --symbols AAPL --capital 10000 --profit-target 0.5 --stop-loss 0.5
```

#### Long Put Strategy

**How it works**: You purchase a put option, giving you the right to sell the underlying asset at the strike price.

**When to use it**: When you expect the price of the underlying asset to fall significantly.

**Configuration in Mercurio AI**:
```bash
python -m scripts.options.run_daily_options_trader --strategy LONG_PUT --symbols AAPL --capital 10000 --profit-target 0.5 --stop-loss 0.5
```

#### Covered Call Strategy

**How it works**: You own the underlying asset and sell a call option against it.

**When to use it**: When you own stocks and want to generate additional income while expecting the stock price to remain stable or slightly increase.

**Configuration in Mercurio AI**:
```bash
python -m scripts.options.run_daily_options_trader --strategy COVERED_CALL --symbols MSFT --capital 10000 --allocation-per-trade 0.1
```

#### Cash-Secured Put Strategy

**How it works**: You sell a put option and set aside enough cash to purchase the underlying asset if the option is exercised.

**When to use it**: When you're willing to buy the asset at a lower price than the current market price and want to generate income while waiting.

**Configuration in Mercurio AI**:
```bash
python -m scripts.options.run_daily_options_trader --strategy CASH_SECURED_PUT --symbols AAPL --capital 10000 --allocation-per-trade 0.1
```

### Level 2 Strategies

These intermediate strategies are suitable for traders with some options experience:

#### Iron Condor Strategy

**How it works**: Combines four options at different strike prices to create a range where you can profit.

**When to use it**: When you expect the underlying asset to remain within a specific price range with low volatility.

**Configuration in Mercurio AI**:
```bash
python -m scripts.options.run_daily_options_trader --strategy IRON_CONDOR --symbols SPY --capital 10000 --allocation-per-trade 0.05
```

#### Butterfly Spread Strategy

**How it works**: Uses three strike prices with four option contracts to create a position that profits when the underlying price stays near the middle strike price.

**When to use it**: When you expect the underlying asset's price to remain stable around a specific target price.

**Configuration in Mercurio AI**:
```bash
python -m scripts.options.run_daily_options_trader --strategy BUTTERFLY --symbols AAPL --capital 10000 --allocation-per-trade 0.05
```

## ML-Powered Options Trading

One of Mercurio AI's unique features is the ability to combine machine learning predictions with options trading strategies:

### Using ML for Options Signal Generation

Mercurio AI can leverage its existing ML models to generate more accurate options trading signals:

```bash
python -m scripts.options.run_ml_options_trader --ml-strategy LSTM --options-strategy AUTO --symbols AAPL MSFT GOOGL --confidence-threshold 0.75
```

In this example:
- The LSTM model generates price movement predictions
- The AUTO parameter selects the optimal options strategy based on the prediction
- A confidence threshold of 0.75 ensures only high-confidence signals trigger trades

### Available ML Models for Options Trading

Mercurio AI offers several ML strategies for options trading:

1. **LSTM**: Recognizes patterns in time-series data, ideal for trend identification
2. **Transformer**: Captures long-term dependencies in market data (similar to GPT models)
3. **LLM (Large Language Model)**: Analyzes market sentiment from news and social media
4. **MSI (Multi-Source Intelligence)**: Combines multiple data sources for comprehensive analysis
5. **Ensemble**: Uses a combination of models for the most robust predictions

## High-Volume Options Trading

For traders looking to scale their options strategies across multiple symbols, Mercurio AI provides specialized tools:

```bash
python -m scripts.options.run_high_volume_options_trader --strategy COVERED_CALL --max-symbols 50 --use-threads --use-custom-symbols
```

This script is optimized to handle up to 50 symbols simultaneously with multi-threading for maximum performance.

## Crypto Options Trading

Mercurio AI also supports options trading for cryptocurrencies:

```bash
python -m scripts.options.run_crypto_options_trader --strategy LONG_CALL --symbols BTC ETH --capital 50000
```

This specialized script accounts for the higher volatility and unique characteristics of cryptocurrency markets.

## Options Backtesting

Before trading with real money, it's essential to backtest your options strategies:

```python
from app.services.options_backtester import OptionsBacktester
from app.strategies.options.covered_call import CoveredCallStrategy
import asyncio

async def backtest_covered_call():
    backtester = OptionsBacktester()
    
    strategy_params = {
        "max_position_size": 0.05,
        "days_to_expiration": 30,
        "profit_target_pct": 0.5,
        "stop_loss_pct": 0.5
    }
    
    results = await backtester.run_backtest(
        strategy_class=CoveredCallStrategy,
        symbols=["AAPL", "MSFT"],
        strategy_params=strategy_params,
        timeframe="1d",
        report_name="covered_call_backtest"
    )
    
    print(f"Total return: {results['total_return']:.2f}%")
    print(f"Sharpe ratio: {results['sharpe_ratio']:.2f}")

asyncio.run(backtest_covered_call())
```

## Risk Management for Options Trading

Options trading carries unique risks that differ from standard stock trading. Mercurio AI provides several risk management tools:

### Position Sizing

Control your exposure by adjusting the `max_position_size` parameter (recommended: start with 0.02-0.05 or 2-5% of your portfolio per position).

### Stop-Loss and Profit Targets

Set appropriate exit points with the `stop_loss_pct` and `profit_target_pct` parameters.

### Delta Targeting

For more advanced control, use the `delta-target` parameter to select options with specific sensitivity to price movements.

### Maximum Allocation

Limit your overall options exposure with the `allocation-per-trade` parameter.

## Testing Your Options Strategies

Mercurio AI includes a comprehensive testing framework for options strategies:

```bash
python -m scripts.options.test_options_strategies --test-all
```

This script validates:
- Proper initialization of strategies
- Entry and exit conditions
- Trade execution
- Risk management parameters
- Edge case handling

## Getting Started: A Step-by-Step Guide

1. **Start with Paper Trading**:
   ```bash
   python -m scripts.options.run_daily_options_trader --strategy COVERED_CALL --symbols AAPL --capital 10000 --paper-trading
   ```

2. **Begin with Level 1 Strategies**:
   - Long Call and Long Put are simpler to understand
   - Covered Call and Cash Secured Put have more favorable risk profiles for beginners

3. **Backtest Before Trading**:
   - Use the OptionsBacktester to validate your strategy
   - Aim for a Sharpe ratio above 1.0 in backtests

4. **Start Small**:
   - Use a small allocation-per-trade (0.02-0.05)
   - Focus on highly liquid options on major stocks

5. **Monitor and Learn**:
   - Use the built-in reporting tools to track performance
   - Gradually introduce more complex strategies as you gain experience

## Conclusion

Options trading with Mercurio AI combines the power of machine learning with sophisticated options strategies to potentially improve your trading outcomes. Start with the basics, focus on risk management, and gradually expand your options trading knowledge for the best results.

Remember that options involve risk, including the potential loss of your investment. Always start with paper trading and smaller position sizes until you're comfortable with how the strategies perform.
