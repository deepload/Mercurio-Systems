<div align="center">
  <h1>🚀 Mercurio AI Trading Platform</h1>
  <p><strong>The intelligent, resilient algorithmic trading platform that adapts to any market condition</strong></p>
  <p>
    <a href="#installation"><img src="https://img.shields.io/badge/Installation-Guide-blue"/></a>
    <a href="#quick-start"><img src="https://img.shields.io/badge/Quick-Start-brightgreen"/></a>
    <a href="docs/for-dummies/01-introduction.md"><img src="https://img.shields.io/badge/Documentation-Complete-orange"/></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow"/></a>
  </p>
</div>

## 💼 Overview

Mercurio AI is a comprehensive algorithmic trading platform built with reliability and versatility at its core. Whether you're a beginner exploring trading strategies or an expert deploying sophisticated machine learning models, Mercurio AI provides all the tools you need in one integrated ecosystem.

### 🌟 Key Features

- **Multiple Strategy Types**: From simple moving averages to advanced ML models (LSTM, Transformer, LLM)
- **Fallback Mechanisms**: Test and trade without API keys using our multi-layered fallback system
- **Paper Trading Mode**: Practice risk-free with simulated trading
- **Comprehensive Backtesting**: Evaluate strategies against historical data
- **Interactive Dashboards**: Monitor performance with Streamlit-powered visualizations
- **Asset Diversity**: Trade stocks and cryptocurrencies across multiple timeframes
- **Resilient Architecture**: Ensures continuous operation even during outages

## 📂 Project Structure

<details>
<summary><strong>Click to expand project structure</strong></summary>

```
MercurioAI/
├── app/                       # Core application directory
│   ├── main.py                # FastAPI application entry point
│   ├── api/                   # API endpoints
│   │   ├── routes.py          # API route definitions
│   │   └── schemas.py         # Pydantic models for requests/responses
│   ├── services/              # Business logic services
│   │   ├── market_data.py     # Service for fetching market data
│   │   ├── trading.py         # Service for executing trades
│   │   └── backtesting.py     # Service for backtesting strategies
│   ├── strategies/            # Trading strategies
│   │   ├── base.py            # Base strategy class
│   │   ├── moving_average.py  # Moving Average Crossover strategy
│   │   ├── lstm_predictor.py  # LSTM-based prediction strategy
│   │   ├── transformer.py     # Transformer-based strategy
│   │   ├── llm_strategy.py    # LLM-powered strategy
│   │   └── msi_strategy.py    # Market Sentiment Index strategy
│   ├── db/                    # Database modules
│   │   ├── database.py        # Database connection
│   │   └── models.py          # SQLAlchemy models
│   ├── tasks/                 # Celery tasks
│   │   ├── celery_app.py      # Celery configuration
│   │   ├── training.py        # Tasks for model training
│   │   ├── trading.py         # Tasks for automated trading
│   │   └── data.py            # Tasks for data collection
│   └── utils/                 # Utility functions
├── docs/                      # Documentation directory
│   ├── for-dummies/           # Comprehensive guide for beginners
│   ├── guides/                # Specialized guides
│   │   ├── beginner/          # Beginner guides
│   │   └── advanced/          # Advanced guides
│   ├── api/                   # API documentation
│   ├── reference/             # Technical reference
│   └── README.md              # Documentation index
├── models/                    # Saved ML models
│   ├── lstm/                  # LSTM models and scalers
│   └── transformer/           # Transformer models and scalers
├── data/                      # Data directory
│   └── sample_data/           # Sample data for testing
├── reports/                   # Reports and visualizations
│   ├── comprehensive/         # Comprehensive simulation results
│   └── visualizations/        # Generated charts and graphs
├── tests/                     # Tests directory
├── comprehensive_simulation.py # Full-featured simulation script
├── strategy_dashboard.py      # Streamlit dashboard for strategy visualization
├── docker-compose.yml         # Docker Compose configuration
├── Dockerfile                 # Docker configuration
├── requirements.txt           # Python dependencies
├── .env.example              # Example environment variables
└── README.md                 # This file (you are here)
```
</details>

## 🚀 Getting Started

### Prerequisites

<table>
  <tr>
    <td><strong>Required</strong></td>
    <td>
      • Python 3.11 or later<br>
      • Git
    </td>
  </tr>
  <tr>
    <td><strong>Optional</strong></td>
    <td>
      • Docker and Docker Compose (for containerized deployment)<br>
      • Polygon.io API key (for production-quality market data)<br>
      • Alpaca API key and secret (for live trading)<br>
    </td>
  </tr>
  <tr>
    <td><strong>Note</strong></td>
    <td>Thanks to Mercurio AI's fallback system, no API keys are required to get started with testing and development!</td>
  </tr>
</table>

### 💻 Installation

<details open>
<summary><strong>Standard Installation</strong></summary>

```bash
# Clone the repository
git clone https://github.com/yourusername/mercurio-ai.git
cd mercurio-ai

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Unix/MacOS

# Install dependencies
pip install -r requirements.txt

# Set up environment variables (optional for testing)
copy .env.example .env
# Edit .env with your API keys if available
```
</details>

<details>
<summary><strong>Docker Installation</strong></summary>

```bash
# Clone the repository
git clone https://github.com/yourusername/mercurio-ai.git
cd mercurio-ai

# Create environment file
copy .env.example .env
# Edit .env with your configuration

# Start the services
docker-compose up -d
```

The following services will be available:
- API: http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- Flower (Celery monitoring): http://localhost:5555
</details>

## 🏃‍♂️ Quick Start

### 1. Run a Simple Strategy

```python
# simple_strategy.py
from app.services.market_data import MarketDataService
from app.strategies.moving_average import MovingAverageStrategy
import asyncio

async def run_simple_strategy():
    # Initialize services with fallback enabled
    market_data = MarketDataService(use_fallback=True)
    strategy = MovingAverageStrategy(short_window=10, long_window=30)
    
    # Get historical data (falls back to sample data if needed)
    data = await market_data.get_historical_data("AAPL", "2024-01-01", "2024-03-01")
    
    # Process data and generate signals
    processed_data = await strategy.preprocess_data(data)
    signal, confidence = await strategy.predict(processed_data)
    
    print(f"AAPL Trading Signal: {signal} (Confidence: {confidence:.2f})")

if __name__ == "__main__":
    asyncio.run(run_simple_strategy())
```

Run the script:
```bash
python simple_strategy.py
```

### 2. Launch the Interactive Dashboard

```bash
streamlit run strategy_dashboard.py
```

This will open a browser window with an interactive dashboard to explore strategy performance.

### 3. Run a Comprehensive Simulation

```bash
python comprehensive_simulation.py --timeframe daily
```

## 🔌 API Reference

<details>
<summary><strong>Available API Endpoints</strong></summary>

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/strategies` | GET | List all available trading strategies |
| `/api/strategies/{strategy_name}` | GET | Get details about a specific strategy |
| `/api/predictions/{strategy_name}/{symbol}` | POST | Get a prediction for a symbol |
| `/api/backtests` | POST | Run a backtest for a strategy |
| `/api/backtests/{backtest_id}` | GET | Get backtest results |
| `/api/models/train` | POST | Train a strategy model |
| `/api/models/{model_id}` | GET | Get model details |
| `/api/trades` | POST | Execute a trade |
| `/api/account` | GET | Get account information |
| `/api/market/status` | GET | Check market status |

For complete API documentation, run the server and visit `/docs` or check the [API Reference](./docs/api/README.md).
</details>

## 🔧 Extending Mercurio AI

### Creating Custom Strategies

<details>
<summary><strong>Click to see how to add custom strategies</strong></summary>

Mercurio AI follows a plugin architecture for strategies, making it easy to implement your own trading logic:

1. Create a new Python file in the `app/strategies` directory
2. Extend the `BaseStrategy` class
3. Implement required methods

```python
from app.strategies.base import BaseStrategy
import pandas as pd
import numpy as np

class MyCustomStrategy(BaseStrategy):
    """My custom mean-reversion trading strategy"""
    
    def __init__(self, lookback_period=20, threshold=2.0):
        self.lookback_period = lookback_period
        self.threshold = threshold
        self.name = "MeanReversionStrategy"
        self.description = "Buys oversold assets and sells overbought assets"
        
    async def preprocess_data(self, data):
        """Calculate z-scores for mean reversion"""
        df = data.copy()
        
        # Calculate rolling mean and standard deviation
        df['rolling_mean'] = df['close'].rolling(window=self.lookback_period).mean()
        df['rolling_std'] = df['close'].rolling(window=self.lookback_period).std()
        
        # Calculate z-score
        df['z_score'] = (df['close'] - df['rolling_mean']) / df['rolling_std']
        
        return df.dropna()
        
    async def predict(self, data):
        """Generate trading signals based on z-scores"""
        if data.empty or len(data) < self.lookback_period:
            return "HOLD", 0.0
            
        current_z = data.iloc[-1]['z_score']
        
        # Mean reversion logic
        if current_z < -self.threshold:
            # Oversold, potential buy
            signal = "BUY"
            confidence = min(abs(current_z) / 4, 1.0)  # Normalize confidence
        elif current_z > self.threshold:
            # Overbought, potential sell
            signal = "SELL"
            confidence = min(abs(current_z) / 4, 1.0)  # Normalize confidence
        else:
            signal = "HOLD"
            confidence = 1.0 - (abs(current_z) / self.threshold)
            
        return signal, float(confidence)
```

Register your strategy in `app/strategies/__init__.py` to make it available throughout the system.
</details>

### Custom Data Sources

<details>
<summary><strong>Click to see how to add custom data sources</strong></summary>

Extend the `BaseDataProvider` class to integrate with any market data source:

```python
from app.services.data_providers.base import BaseDataProvider
import pandas as pd
import requests

class MyCustomDataProvider(BaseDataProvider):
    """Custom data provider for XYZ market data"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.base_url = "https://api.example.com/v1"
        
    async def get_historical_data(self, symbol, start_date, end_date, timeframe="1d"):
        """Fetch historical market data from custom source"""
        # Implementation
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        params = {
            "symbol": symbol,
            "from": start_date,
            "to": end_date,
            "interval": timeframe
        }
        
        response = requests.get(f"{self.base_url}/historical", headers=headers, params=params)
        data = response.json()
        
        # Transform to standard format
        df = pd.DataFrame(data["bars"])
        df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        
        return df
```
</details>

## 🔬 Running the Comprehensive Simulation

Test all available strategies across different assets and timeframes:

```bash
python comprehensive_simulation.py --start-date 2024-03-03 --end-date 2025-04-25 --initial-capital 10000
```

This runs a full-year simulation using:
- **5 Stock Assets**: AAPL, MSFT, GOOGL, AMZN, TSLA
- **5 Crypto Assets**: BTC-USD, ETH-USD, SOL-USD, ADA-USD, DOT-USD
- **All Strategies**: MovingAverage, LSTM, Transformer, LLM, MSI
- **Multiple Timeframes**: Daily, Weekly, Monthly

Results are saved to the `reports/comprehensive/` directory with full metrics and visualizations.

## 📚 Documentation

Comprehensive documentation is available in the [docs](./docs) directory:

- **[For Dummies Guide](./docs/for-dummies/README.md)**: Complete 12-chapter guide from beginner to advanced
- **[Beginner Guides](./docs/guides/beginner/)**: Quick start resources for new users
- **[Advanced Guides](./docs/guides/advanced/)**: Specialized topics for experienced traders
- **[Reference Documentation](./docs/reference/)**: Technical details and API specifications

## 🧪 Testing

```bash
# Run the test suite
pytest

# Run tests with coverage
pytest --cov=app
```

## 🤝 Contributing

Contributions are welcome! Check out the [contribution guidelines](CONTRIBUTING.md) to get started.

## 📄 License

This project is open-source and available under the MIT License.

## 🌟 Showcase

<div align="center">
  <table>
    <tr>
      <td align="center"><img src="https://i.imgur.com/2sGSK1h.png" width="400"/><br><b>Strategy Comparison</b></td>
      <td align="center"><img src="https://i.imgur.com/7fVLnMF.png" width="400"/><br><b>Portfolio Allocation</b></td>
    </tr>
  </table>
</div>

## 🙏 Acknowledgements

- [FastAPI](https://fastapi.tiangolo.com/)
- [SQLAlchemy](https://www.sqlalchemy.org/)
- [PyTorch](https://pytorch.org/)
- [Pandas](https://pandas.pydata.org/)
- [Streamlit](https://streamlit.io/)
- [Polygon.io](https://polygon.io/)
- [Yahoo Finance](https://finance.yahoo.com/)
- [Alpaca Markets](https://alpaca.markets/)
- [Backtrader](https://www.backtrader.com/)
