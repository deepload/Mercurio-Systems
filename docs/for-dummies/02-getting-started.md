# Chapter 2: Getting Started with Mercurio AI

Welcome to Chapter 2! Now that you understand what Mercurio AI is, let's get your system set up and take your first steps with the platform.

## Installation and Setup

### System Requirements

Before you begin, make sure your system meets these minimum requirements:

- **Operating System**: Windows, macOS, or Linux
- **Python**: Version 3.8 or higher
- **RAM**: 4GB minimum (8GB+ recommended for ML strategies)
- **Disk Space**: At least 1GB free space
- **Internet Connection**: Required for real-time data (optional for testing)

### Step 1: Install Python

If you don't already have Python installed:

1. Go to [python.org](https://python.org/downloads/)
2. Download the latest Python version (3.8+)
3. Run the installer, making sure to check "Add Python to PATH"
4. Verify installation by opening a command prompt/terminal and typing:
   ```
   python --version
   ```

### Step 2: Get the Mercurio AI Code

You have two options:

**Option A: Clone from Repository**
```bash
# Using Git
git clone https://github.com/yourusername/MercurioAI.git
cd MercurioAI
```

**Option B: Download the Project**
1. Download the Mercurio AI package from your provided source
2. Extract the files to a convenient location
3. Navigate to the project directory in your terminal/command prompt

### Step 3: Create a Virtual Environment

It's best practice to use a virtual environment for Python projects:

```bash
# In the MercurioAI directory
python -m venv venv

# Activate the environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 4: Install Dependencies

Once your virtual environment is activated, install the required packages:

```bash
pip install -r requirements.txt
```

This will install all necessary dependencies, including:
- pandas and numpy for data processing
- matplotlib and seaborn for visualization
- tensorflow for machine learning models
- other trading and development libraries

### Step 5: Verify Installation

Let's make sure everything is working correctly:

```bash
python -c "from app.services.market_data import MarketDataService; print('Mercurio AI is ready!')"
```

If you see "Mercurio AI is ready!" without errors, congratulations! You've successfully set up Mercurio AI.

## Understanding the Project Structure

Let's take a quick tour of the Mercurio AI project structure to help you find your way around:

```
MercurioAI/
│
├── app/                       # Core application code
│   ├── services/              # Service modules
│   │   ├── market_data.py     # Market data service
│   │   ├── trading.py         # Trading service
│   │   └── ...                
│   │
│   ├── strategies/            # Trading strategies
│   │   ├── moving_average.py  # Moving average strategy
│   │   ├── lstm_predictor.py  # LSTM-based strategy
│   │   └── ...                
│   │
│   └── models/                # Data models and schemas
│
├── reports/                   # Generated reports and analysis
│
├── docs/                      # Documentation
│   └── for-dummies/           # This guide!
│
├── requirements.txt           # Project dependencies
│
└── strategy_*.py              # Strategy running scripts
```

## Your First Mercurio AI Script

Now that everything is set up, let's run a simple script to ensure everything works. This script will use Mercurio AI's built-in fallback mechanisms to generate sample data and run a basic moving average strategy backtest.

Create a file called `first_script.py` with the following content:

```python
"""
My First Mercurio AI Script
This simple script runs a Moving Average strategy on sample data.
"""
import asyncio
import pandas as pd
import matplotlib.pyplot as plt
from app.strategies.moving_average import MovingAverageStrategy
from app.services.market_data import MarketDataService

async def main():
    # Initialize market data service (will use sample data by default)
    market_data = MarketDataService()
    
    # Get sample data for AAPL
    data = await market_data.get_historical_data(
        symbol="AAPL",
        start_date="2023-01-01",
        end_date="2023-12-31"
    )
    
    # Create a simple moving average strategy
    strategy = MovingAverageStrategy(
        short_window=10,
        long_window=30,
        use_ml=False  # Start with simple strategy without ML
    )
    
    # Preprocess the data
    processed_data = await strategy.preprocess_data(data)
    
    # Run a backtest
    backtest_result = await strategy.backtest(
        data=processed_data,
        initial_capital=10000  # $10,000 initial capital
    )
    
    # Print basic results
    final_equity = backtest_result["final_equity"]
    total_return = (final_equity / 10000 - 1) * 100
    
    print(f"Strategy: Moving Average")
    print(f"Initial Capital: $10,000.00")
    print(f"Final Capital: ${final_equity:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Number of Trades: {len(backtest_result['trades'])}")
    
    # Plot equity curve
    plt.figure(figsize=(10, 6))
    plt.plot(backtest_result["equity_curve"])
    plt.title("Moving Average Strategy - Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True)
    plt.savefig("my_first_backtest.png")
    plt.show()

if __name__ == "__main__":
    asyncio.run(main())
```

Run the script:

```bash
python first_script.py
```

### Understanding the Script

Let's break down what this script does:

1. **Imports necessary modules**: We import from both the Python standard library and Mercurio AI.
2. **Gets market data**: Uses the MarketDataService to fetch historical data for Apple (AAPL).
3. **Creates a strategy**: Initializes a MovingAverageStrategy with specific parameters.
4. **Preprocesses data**: Prepares the data for the strategy (calculates indicators, etc.).
5. **Runs a backtest**: Simulates trading based on the strategy's signals.
6. **Prints results**: Shows how the strategy performed.
7. **Visualizes performance**: Creates a chart of the equity curve over time.

What's great about this script is that it will work even without any API keys or external data sources. Mercurio AI's fallback mechanisms will automatically use sample data if external sources aren't available.

## The Mercurio AI Fallback System

One of the most powerful features of Mercurio AI is its multi-layered fallback system, which ensures you can work with the platform in any environment:

1. **Primary Data Source**: First tries to use your configured real-time data providers (if API keys are provided)
2. **Secondary Sources**: Falls back to free alternative data sources if primary sources fail
3. **Sample Data Generation**: If no external data is available, generates realistic sample data
4. **Simulation Mode**: Always available for backtesting with either real or synthetic data

This means you can:
- Develop and test strategies without any API keys
- Run simulations in environments without internet access
- Gradually transition from testing to real trading as you gain confidence

## Command-Line Tools

Mercurio AI provides several ready-to-use command-line tools:

- **strategy_simulator.py**: Run simulations with different strategies
- **optimize_moving_average.py**: Find optimal parameters for moving average strategies
- **strategy_dashboard.py**: Launch an interactive dashboard for strategy analysis

Try running the dashboard:

```bash
streamlit run strategy_dashboard.py
```

This will open a web browser with an interactive dashboard for exploring strategy performance.

## Next Steps

Congratulations! You've successfully set up Mercurio AI and run your first script. You've seen how easy it is to get started, thanks to the platform's built-in fallback mechanisms.

In the next chapter, we'll dive deeper into understanding the core components of the platform and how they work together.

Continue to [Chapter 3: Understanding the Platform](./03-understanding-platform.md) to learn more about Mercurio AI's architecture.

---

**Key Takeaways:**
- Setting up Mercurio AI requires Python 3.8+ and the necessary dependencies
- The platform's structure is modular, with separate services for different functions
- You can run strategies without any API keys thanks to the fallback mechanisms
- The first script demonstrates the basic workflow: get data, create strategy, run backtest, analyze results
- Mercurio AI includes ready-to-use tools for simulation, optimization, and visualization
