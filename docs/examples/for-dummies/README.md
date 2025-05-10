# Mercurio AI For Dummies - Example Scripts

This directory contains working example scripts that accompany the "Mercurio AI For Dummies" documentation series. Each script demonstrates key concepts from the corresponding chapter and has been tested to work with Mercurio AI's paper trading and fallback mechanisms.

## Chapter-by-Chapter Examples

### Chapter 1-2: Getting Started
- [01_first_script.py](01_first_script.py) - Basic script to run a Moving Average strategy on sample data

### Chapter 3: Understanding the Platform
- [02_market_data_service.py](02_market_data_service.py) - Demonstrates fetching and analyzing market data
- [03_trading_service.py](03_trading_service.py) - Shows how to use the Trading Service for paper trading
- [04_backtesting_service.py](04_backtesting_service.py) - Runs backtests with different strategies
- [05_strategy_manager.py](05_strategy_manager.py) - Explores available strategies and their configuration

### Chapter 4: Paper Trading
- [06_paper_trading_session.py](06_paper_trading_session.py) - Runs a complete paper trading session
- [07_position_sizing.py](07_position_sizing.py) - Demonstrates different position sizing techniques

## Running the Examples

All examples can be run directly from this directory. Each script:

1. Adds the project root to the Python path
2. Properly imports Mercurio AI components
3. Uses fallback mechanisms for sample data when API keys aren't available

To run an example:

```bash
# Navigate to the project root
cd /path/to/MercurioAI

# Run the example
python docs/examples/for-dummies/01_first_script.py
```

## Important Notes

- **Fallback Mechanism**: All examples work with Mercurio AI's fallback system. If API keys aren't available or valid, the system automatically uses sample data.
- **Paper Trading**: Examples use paper trading mode, so no real money is at risk.
- **Customization**: Feel free to modify parameters and strategies to experiment with different approaches.
- **Dependencies**: Make sure you've installed all dependencies from requirements.txt before running examples.

These examples complement the concepts explained in the "For Dummies" guide and provide a hands-on way to understand how Mercurio AI works.

## Additional Resources

For more examples including advanced strategies and multi-asset trading, see the [parent examples directory](../).
