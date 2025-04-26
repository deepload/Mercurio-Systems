# Mercurio AI - Optimized Trading Strategy Guide

## Overview

This guide explains how to use Mercurio AI's optimized trading strategies based on our simulations and backtests for January 2025. Our analysis has identified the most profitable configuration for each asset, focusing particularly on the Moving Average Crossover strategy which consistently outperformed other approaches in our simulations.

## Table of Contents

1. [Strategy Performance Summary](#strategy-performance-summary)
2. [Optimized Parameters](#optimized-parameters)
3. [Investment Recommendations](#investment-recommendations)
4. [Quick Start Guide](#quick-start-guide)
5. [Advanced Configuration](#advanced-configuration)
6. [Monitoring and Rebalancing](#monitoring-and-rebalancing)

## Strategy Performance Summary

After testing multiple strategies including MovingAverage, MovingAverage_ML (with machine learning), LSTM, LLM, and Transformer approaches, we found that the **traditional Moving Average Crossover** strategy outperformed the others when properly optimized for January 2025 market conditions.

**Key Performance Metrics:**

| Asset    | Strategy       | Monthly Return | Sharpe Ratio | Max Drawdown |
|----------|---------------|---------------|-------------|--------------|
| ETH-USD  | MovingAverage | 21.41%        | 9.27        | 3.1%         |
| GOOGL    | MovingAverage | 10.13%        | 9.46        | 1.8%         |
| BTC-USD  | MovingAverage | 7.95%         | 4.09        | 4.2%         |
| MSFT     | MovingAverage | 2.84%         | 2151.95     | 0.4%         |
| AAPL     | MovingAverage | 1.02%         | 1.77        | 2.3%         |

These results were derived from simulations using synthetic but realistic market data for January 2025, with each strategy starting with $2,000 initial capital.

## Optimized Parameters

Our parameter optimization process tested numerous combinations of short and long windows, both with and without machine learning enhancement. The table below shows the optimal configuration for each asset:

| Asset    | Short Window | Long Window | Use ML | Description |
|----------|-------------|------------|-------|-------------|
| ETH-USD  | 10          | 15         | False | Medium-term momentum capture works best for Ethereum |
| GOOGL    | 7           | 10         | False | Shorter windows capture GOOGL's rapid price movements |
| BTC-USD  | 10          | 15         | False | Same settings as ETH-USD work well for Bitcoin |
| MSFT     | 7           | 20         | False | Wider window spread for more stable price action |
| AAPL     | 10          | 15         | False | Medium settings balance stability and responsiveness |

**Key Finding:** Conventional moving average crossover strategies (without ML enhancement) performed better across all assets for January 2025 market conditions.

## Investment Recommendations

Based on our optimization results, here's our recommended portfolio allocation for a $10,000 investment:

1. **ETH-USD (40%)**: $4,000 - Highest return at 21.41%
2. **GOOGL (25%)**: $2,500 - Strong stock performance at 10.13%
3. **BTC-USD (20%)**: $2,000 - Solid crypto diversification at 7.95%
4. **MSFT (10%)**: $1,000 - Stable performance at 2.84%
5. **AAPL (5%)**: $500 - More conservative allocation at 1.02%

This allocation balances potential returns with diversification across both cryptocurrencies and traditional stocks.

## Quick Start Guide

Follow these steps to implement the optimized trading strategy:

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Optimized Strategy:**
   ```bash
   python optimized_portfolio.py
   ```
   This will execute the strategy with the recommended parameters and allocation.

3. **View Performance Dashboard:**
   ```bash
   streamlit run strategy_dashboard.py
   ```
   This will open an interactive dashboard to monitor your portfolio performance.

## Advanced Configuration

### Custom Parameter Settings

If you want to use custom parameters instead of the optimized ones:

1. Open `optimized_portfolio.py`
2. Modify the `STRATEGY_PARAMS` dictionary with your desired settings:
   ```python
   STRATEGY_PARAMS = {
       "ETH-USD": {"short_window": 10, "long_window": 15, "use_ml": False},
       "GOOGL": {"short_window": 7, "long_window": 10, "use_ml": False},
       # Add your custom parameters here
   }
   ```

### Custom Portfolio Allocation

To modify the recommended allocation:

1. Open `optimized_portfolio.py`
2. Adjust the `PORTFOLIO_ALLOCATION` dictionary:
   ```python
   PORTFOLIO_ALLOCATION = {
       "ETH-USD": 0.40,  # 40%
       "GOOGL": 0.25,    # 25%
       "BTC-USD": 0.20,  # 20%
       # Modify allocations or add new assets
   }
   ```

### Paper Trading vs. Live Trading

By default, the system runs in paper trading mode. To switch to live trading:

1. Set up your API keys in the `.env` file
2. Open `optimized_portfolio.py`
3. Change the `TRADING_MODE` variable:
   ```python
   TRADING_MODE = "live"  # Options: "paper", "live"
   ```

## Monitoring and Rebalancing

### Daily Monitoring

The strategy automatically generates daily performance reports in the `reports/daily` directory. These reports include:

- Current positions and their values
- Daily P&L
- Trade history
- Strategy performance metrics

### Monthly Rebalancing

We recommend rerunning the optimization process monthly to adjust parameters based on changing market conditions:

```bash
python optimize_moving_average.py
```

This will generate new optimized parameters in `reports/optimization/best_parameters.txt`.

### Visualization Tools

Several visualization tools are available:

1. **Strategy Dashboard:**
   ```bash
   streamlit run strategy_dashboard.py
   ```

2. **Optimization Heatmaps:**
   View the heatmaps generated in `reports/optimization/` to understand parameter sensitivity.

3. **Performance Charts:**
   Review performance charts in `reports/visualizations/` for detailed strategy analysis.

## Technical Details

### Data Sources

The strategy uses these data sources in order of priority:

1. Real market data (when API keys are configured)
2. Sample data provider (for testing without API keys)
3. Synthetic data generator (for backtesting and simulation)

### Strategy Implementation

The Moving Average Crossover strategy generates signals as follows:

- **Buy Signal:** When the short-term moving average crosses above the long-term moving average
- **Sell Signal:** When the short-term moving average crosses below the long-term moving average

Each asset uses its own optimized parameters as determined through extensive backtesting.

### Risk Management

The strategy implements these risk management techniques:

1. **Position Sizing:** Maximum 40% allocation to any single asset
2. **Stop Loss:** Configurable stop-loss at 5% below entry price
3. **Take Profit:** Optional take-profit targets at 10%, 20%, and 30% above entry

---

For more detailed technical information, please refer to the API documentation and source code comments. If you encounter any issues or have questions, please reach out to the Mercurio AI support team.

*Last updated: April 26, 2025*
