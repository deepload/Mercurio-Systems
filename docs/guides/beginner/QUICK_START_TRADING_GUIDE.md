# MercurioAI Quick Start Trading Guide

This guide provides the essential steps to quickly begin trading with MercurioAI, first in paper trading mode to practice without risk, then transitioning to live trading when you're ready.

## Setup (One-Time)

1. **Configure API Keys**

   Edit your `.env` file to include your API credentials:

   ```
   # Alpaca Trading Mode - 'paper' or 'live'
   ALPACA_MODE=paper
   
   # Paper Trading Credentials
   ALPACA_PAPER_KEY=your_paper_key_here
   ALPACA_PAPER_SECRET=your_paper_secret_here
   ALPACA_PAPER_URL=https://paper-api.alpaca.markets
   
   # Live Trading Credentials (configure but keep ALPACA_MODE=paper until ready)
   ALPACA_LIVE_KEY=your_live_key_here
   ALPACA_LIVE_SECRET=your_live_secret_here
   ALPACA_LIVE_URL=https://api.alpaca.markets
   
   # Market Data URL (used for both paper and live)
   ALPACA_DATA_URL=https://data.alpaca.markets
   ```
   
   > **Note**: Switching between paper and live trading is now as simple as changing `ALPACA_MODE=paper` to `ALPACA_MODE=live`

   > **Note**: If you don't have API keys, MercurioAI can still run in demo mode with sample data.

2. **Install Dependencies** (if not already done)

   ```bash
   pip install -r requirements.txt
   ```

## Paper Trading

### Method 1: Quick Start (Basic)

Run this command to start paper trading with default settings:

```bash
python run_paper_trading.py
```

This will:
- Use the default moving average strategy
- Trade AAPL, MSFT, and GOOGL
- Check for signals every 60 seconds
- Use a 2% risk limit per position

### Method 2: Comprehensive Strategy Testing

To test and compare multiple strategies simultaneously in paper trading mode:

```bash
python paper_trading_test.py
```

This will:
- Automatically discover and test all available strategies
- Run them with the same initial capital and time period
- Generate performance comparisons and reports
- Help you identify the best-performing strategies

For customized testing:

```bash
python paper_trading_test.py \
  --capital 50000 \
  --duration 48 \
  --symbols BTC-USD ETH-USD \
  --risk moderate \
  --strategies TransformerStrategy LLMStrategy
```

Common options:
- `--capital`: Initial capital amount (e.g., 50000 = $50,000)
- `--duration`: Test duration in hours (e.g., 48 = 2 days)
- `--symbols`: Space-separated list of symbols to trade
- `--risk`: Risk profile to use (conservative, moderate, aggressive)
- `--strategies`: Space-separated list of specific strategies to test (e.g., MovingAverageStrategy, LLMStrategy, MultiSourceIntelligenceStrategy)
- `--output`: Path for the output report file

### Method 3: Customized Paper Trading

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
- `--strategy`: Strategy to use (MovingAverageStrategy, LSTMPredictorStrategy, LLMStrategy, TransformerStrategy, MultiSourceIntelligenceStrategy)
- `--symbols`: Comma-separated list of symbols to trade
- `--risk_limit`: Maximum portfolio percentage per position (0.01 = 1%)
- `--interval`: Check frequency in seconds (300 = 5 minutes)
- `--fee_percentage`: Simulated transaction fee percentage

### Method 4: Strategy Configuration

For advanced strategy parameters, use the `--params` option with JSON:

```bash
python run_paper_trading.py \
  --strategy MovingAverageStrategy \
  --symbols AAPL,MSFT \
  --params '{"short_window": 20, "long_window": 50, "use_ml": true}'
```

### Method 5: Monitor Performance

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

## Advanced Strategy Testing

### Testing LLM Strategies

MercurioAI includes advanced LLM-based strategies that can analyze market sentiment and make trading decisions using natural language understanding. Here's how to test and utilize them:

```bash
# Test the LLM strategy with specific parameters
python paper_trading_test.py --strategies LLMStrategy --duration 24 --symbols BTC/USDT,ETH/USDT
```

Key parameters for LLM strategies:

```json
{
  "strategy_params": {
    "LLMStrategy": {
      "model_path": "models/llama-2-7b-chat.gguf",
      "context_window": 72,
      "temperature": 0.7,
      "max_tokens": 512
    }
  }
}
```

### Testing Transformer Strategies

Transformer-based models can identify complex patterns in financial time series:

```bash
python paper_trading_test.py --strategies TransformerStrategy --duration 24
```

Customizing transformer parameters:

```bash
python run_paper_trading.py \
  --strategy TransformerStrategy \
  --symbols BTC/USDT \
  --params '{"sequence_length": 30, "d_model": 64, "nhead": 4, "num_layers": 2}'
```

### Comparative Strategy Testing

To compare multiple strategies head-to-head, including traditional and LLM-based approaches:

```bash
python paper_trading_test.py \
  --strategies MovingAverageStrategy,RSIStrategy,LLMStrategy,TransformerStrategy,MultiSourceIntelligenceStrategy \
  --duration 48 \
  --symbols BTC/USDT,ETH/USDT \
  --risk moderate
```

This will generate comprehensive performance metrics for all strategies, including:
- Total return
- Annualized return
- Maximum drawdown
- Sharpe ratio
- Win rate
- Number of trades

### Multi-Source Intelligence (MSI) Strategy

The Multi-Source Intelligence strategy is a professional-grade trading strategy that only makes decisions when it has fresh and validated market data from multiple sources:

```bash
python paper_trading_test.py --strategies MultiSourceIntelligenceStrategy --duration 24 --symbols BTC/USDT,ETH/USDT
```

Key features:
- Rigorous data freshness verification before each trade decision
- Multi-source sentiment analysis (Twitter, Reddit, news)
- Potential market manipulation detection
- Smart caching system to optimize API calls
- Continuous position reassessment

Example configuration:

```json
{
  "strategy_params": {
    "MultiSourceIntelligenceStrategy": {
      "max_data_age_seconds": 30,
      "sentiment_lookback_minutes": 30,
      "confidence_threshold": 0.75,
      "sentiment_weight": 0.4,
      "technical_weight": 0.4,
      "volume_weight": 0.2,
      "debounce_interval_seconds": 15
    }
  }
}
```

### LLM Strategy Configuration

For optimal LLM strategy performance, you can customize these parameters in the configuration file:

1. **Model Selection**: Choose between different LLM models in the `model_path` parameter
2. **Context Window**: Adjust the `context_window` parameter to control how much historical data is analyzed
3. **Temperature**: Control randomness with the `temperature` parameter (lower is more deterministic)
4. **Prompt Templates**: Customize analysis prompts in the strategy file

Example configuration in `config/paper_test_config.json`:

```json
{
  "strategy_params": {
    "LLMStrategy": {
      "model_path": "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
      "context_window": 48,
      "temperature": 0.5,
      "strategy_type": "sentiment",
      "data_sources": ["price", "volume", "news"]
    }
  }
}
```

### Method 6: Crypto Trading

For cryptocurrency trading, Mercurio AI now fully supports the Alpaca crypto API with proper formatting:

```bash
python run_crypto_paper_trading.py \
  --duration_minutes 60 \
  --initial_capital 1000
```

This will run paper trading on multiple cryptocurrencies using Alpaca's crypto API and compare the performance of different strategies.

> **Note**: Cryptocurrency symbols use the format `BTC-USD`, `ETH-USD`, etc. in Mercurio AI.

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

*Last updated: April 26, 2025 - Added Multi-Source Intelligence (MSI) Strategy*

## Guide de démarrage rapide pour la stratégie MSI (Français)

La stratégie Multi-Source Intelligence (MSI) est une stratégie de trading professionnelle qui ne prend des décisions que lorsqu'elle dispose de données de marché fraîches et validées provenant de sources multiples.

### Caractéristiques principales

- Vérification rigoureuse de la fraîcheur des données avant chaque décision
- Analyse de sentiment multi-sources (Twitter, Reddit, actualités)
- Détection de manipulations potentielles du marché
- Système de mise en cache intelligent pour optimiser les appels API
- Réévaluation continue des positions ouvertes

### Test rapide avec la stratégie MSI

```bash
python run_paper_trading.py --strategy MultiSourceIntelligenceStrategy --symbols BTC/USDT
```

### Test comparatif avec plusieurs stratégies

```bash
python paper_trading_test.py --strategies MovingAverageStrategy,MultiSourceIntelligenceStrategy --duration 24 --symbols BTC/USDT,ETH/USDT
```

### Configuration personnalisée

Créez un fichier de configuration `config/msi_config.json` :

```json
{
  "strategy": "MultiSourceIntelligenceStrategy",
  "strategy_params": {
    "max_data_age_seconds": 30,
    "sentiment_lookback_minutes": 30,
    "confidence_threshold": 0.75,
    "sentiment_weight": 0.4,
    "technical_weight": 0.4,
    "volume_weight": 0.2,
    "debounce_interval_seconds": 15
  },
  "symbols": ["BTC/USDT", "ETH/USDT"],
  "risk_limit": 0.01,
  "check_interval": 300
}
```

Puis lancez avec :

```bash
python run_paper_trading.py --config config/msi_config.json
```

### Paramètres principaux

- `max_data_age_seconds` : Âge maximum des données en secondes (défaut: 30)
- `sentiment_lookback_minutes` : Période d'analyse rétrospective pour le sentiment (défaut: 30)
- `confidence_threshold` : Seuil de confiance minimal pour trader (défaut: 0.75)
- `sentiment_weight` : Poids du sentiment dans la décision (défaut: 0.4)
- `technical_weight` : Poids des indicateurs techniques (défaut: 0.4)
- `volume_weight` : Poids des métriques de volume (défaut: 0.2)
- `debounce_interval_seconds` : Intervalle entre décisions (défaut: 15)

Cette stratégie convient particulièrement aux marchés volatils comme les cryptomonnaies, où la qualité et la fraîcheur des données sont essentielles.
