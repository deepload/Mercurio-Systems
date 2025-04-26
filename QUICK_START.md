# Mercurio AI Quick Start

This guide will help you quickly run the strategy timeframe comparison script and understand its output.

## 1. Requirements
- Python 3.8+
- All dependencies listed in `requirements.txt` (install with `pip install -r requirements.txt`)

## 2. Running the Strategy Timeframe Comparison

The script `strategy_timeframe_comparison.py` runs all available trading strategies (including LSTM, Moving Average, LLM, Transformer, and MSI) for both day trading and week trading. It compares results side-by-side for each symbol and strategy.

### To run the script:

```bash
python strategy_timeframe_comparison.py
```

- The script will generate synthetic/demo data if no real data is available (no API keys required for demo mode).
- Results will be saved to `reports/strategy_timeframe_comparison.csv` and printed in the console in a table format.

## 3. Output
- The results table shows initial/final prices, total return, and any errors for each strategy, symbol, and timeframe.
- If a strategy cannot run (e.g., not enough data for LSTM), the error column will explain why.

## 4. Troubleshooting
- If you see errors like `Not enough data after preprocessing` or `No valid data for initial/final price`, this means the dataset was too small for that strategy's requirements.
- For further diagnostics, check the logs printed in the console.

## 5. Next Steps
- You can modify the script to add/remove strategies, change symbols, or adjust timeframes as needed.
- For more advanced usage, see the main documentation or explore other scripts like `strategy_simulator_final.py`.

---

For more information, see the main `README.md` or contact the Mercurio AI team.
