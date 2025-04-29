# MercurioAI Example Scripts

This directory contains working examples for all trading strategies and features described in the Mercurio AI documentation. 
These scripts have been tested and verified to work with the paper trading mode.

## Basic Examples

- [Basic Moving Average Strategy](basic_ma_strategy.py) - Simple example demonstrating how to use the MovingAverageStrategy
- [Multi-Strategy Portfolio](multi_strategy_portfolio.py) - Example of combining multiple strategies in a diversified portfolio

## Paper Trading Examples

- [Basic Paper Trading](paper_trading_basic.py) - Paper trading with default settings
- [Comprehensive Strategy Testing](paper_trading_comprehensive.py) - Test and compare multiple strategies simultaneously
- [Customized Paper Trading](paper_trading_customized.py) - Paper trading with custom parameters
- [Strategy Configuration](paper_trading_strategy_config.py) - Advanced strategy configuration with JSON parameters

## Advanced Strategy Examples

- [LLM Strategy Test](llm_strategy_test.py) - Testing LLM-based trading strategies
- [Transformer Strategy Test](transformer_strategy_test.py) - Testing transformer-based trading strategies
- [MSI Strategy Test](msi_strategy_test.py) - Testing Multi-Source Intelligence strategies

## Running the Examples

All examples can be run directly from this directory. For example:

```bash
# Run the basic moving average strategy example
python basic_ma_strategy.py

# Run paper trading with custom parameters
python paper_trading_customized.py --strategy MovingAverageStrategy --symbols AAPL,MSFT --risk_limit 0.02

# Test LLM-based strategies
python llm_strategy_test.py --symbols BTC/USDT --duration 24
```

Each script includes help documentation accessible with the `--help` flag.

## Notes

- All examples work with both real API keys and Mercurio AI's fallback mechanisms (sample data)
- Scripts prepend the project root to the Python path so they can be run from this directory
- Examples with ML models may require training before they produce meaningful signals
