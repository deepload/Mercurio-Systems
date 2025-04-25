# MercurioAI Quick Start Trading Guide

This guide provides the essential steps to quickly begin trading with MercurioAI, first in paper trading mode to practice without risk, then transitioning to live trading when you're ready.

## Setup (One-Time)

1. **Configure API Keys**

   Edit your `.env` file to include your API credentials:

   ```
   # For paper trading (Alpaca)
   ALPACA_KEY=your_paper_key_here
   ALPACA_SECRET=your_paper_secret_here
   ALPACA_BASE_URL=https://paper-api.alpaca.markets
   
   # For live trading (uncomment when ready)
   # ALPACA_KEY=your_live_key_here
   # ALPACA_SECRET=your_live_secret_here
   # ALPACA_BASE_URL=https://api.alpaca.markets
   ```

   > **Note**: If you don't have API keys, MercurioAI can still run in demo mode with sample data.

2. **Install Dependencies** (if not already done)

   ```bash
   pip install -r requirements.txt
   ```

## Paper Trading

### Step 1: Quick Start (Basic)

Run this command to start paper trading with default settings:

```bash
python run_paper_trading.py
```

This will:
- Use the default moving average strategy
- Trade AAPL, MSFT, and GOOGL
- Check for signals every 60 seconds
- Use a 2% risk limit per position

### Step 2: Customized Paper Trading

For more control, use command-line options:

```bash
python run_paper_trading.py \
  --strategy MovingAverageStrategy \
  --symbols AAPL,TSLA,AMZN \
  --risk_limit 0.01 \
  --interval 300 \
  --fee_percentage 0.001
```

Common options:
- `--strategy`: Strategy to use (MovingAverageStrategy, LSTMPredictorStrategy)
- `--symbols`: Comma-separated list of symbols to trade
- `--risk_limit`: Maximum portfolio percentage per position (0.01 = 1%)
- `--interval`: Check frequency in seconds (300 = 5 minutes)
- `--fee_percentage`: Simulated transaction fee percentage

### Step 3: Strategy Configuration

For advanced strategy parameters, use the `--params` option with JSON:

```bash
python run_paper_trading.py \
  --strategy MovingAverageStrategy \
  --symbols AAPL,MSFT \
  --params '{"short_window": 20, "long_window": 50, "use_ml": true}'
```

### Step 4: Monitor Performance

1. Watch the terminal output for:
   - Trading signals and confidence levels
   - Order executions and fill prices
   - Position updates and P&L
   - Transaction costs

2. Check the log file for detailed information:
   ```
   logs/paper_trading.log
   ```

3. Review the performance report after each trading session

## Live Trading

> **WARNING**: Only proceed to live trading after:
> - Successful paper trading for at least 2-4 weeks
> - Verifying strategy performance matches expectations
> - Setting appropriate risk parameters

### Step 1: Update API Keys

Edit your `.env` file to use live trading credentials:

```
# Live trading keys
ALPACA_KEY=your_live_key_here
ALPACA_SECRET=your_live_secret_here
ALPACA_BASE_URL=https://api.alpaca.markets
```

### Step 2: Start Small

Begin live trading with minimal risk:

```bash
python run_live_trading.py \
  --strategy MovingAverageStrategy \
  --symbols AAPL \
  --risk_limit 0.005
```

Notice the changes:
- Using `run_live_trading.py` instead of `run_paper_trading.py`
- Starting with a single symbol
- Using a lower risk limit (0.5% of portfolio)

### Step 3: Scale Gradually

As you gain confidence, gradually increase your parameters:

```bash
python run_live_trading.py \
  --strategy MovingAverageStrategy \
  --symbols AAPL,MSFT,GOOGL \
  --risk_limit 0.01
```

### Step 4: Use Configuration Files

For complex setups, create a JSON configuration file:

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
  "check_interval": 300
}
```

Then run with:

```bash
python run_live_trading.py --config my_trading_config.json
```

## Daily Operations

### 1. Morning Check

```bash
# Check account status before market open
python run_live_trading.py --check_account
```

### 2. Start Trading

```bash
# Start the trading algorithm for the day
python run_live_trading.py --config my_config.json
```

### 3. Evening Review

- Check the logs for performance
- Note any issues or anomalies
- Review positions and P&L

## Emergency Procedures

### Stop Trading

Press `Ctrl+C` in the terminal running the script to gracefully shut down.

### Check Status

```bash
# View current positions and account status
python run_live_trading.py --status
```

### Force Close Positions

```bash
# Emergency: close all positions
python run_live_trading.py --close_all_positions
```

## Troubleshooting

### API Connection Issues

- Verify your API keys are correct
- Check internet connection
- Ensure Alpaca services are operational

### Strategy Issues

- Run a backtest to verify strategy logic
- Check for recent market condition changes
- Review strategy parameters

### Performance Problems

- Check system resources (CPU, memory)
- Review log files for errors or warnings
- Reduce the number of symbols or check frequency

---

Remember: successful algorithmic trading requires patience, disciplined risk management, and continuous learning. Start small, learn from each trade, and scale up gradually.

*Last updated: April 25, 2025*
