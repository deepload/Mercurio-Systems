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
    
    # Debug: print columns after preprocessing
    print("Columns after preprocessing:", list(processed_data.columns))
    if 'signal' not in processed_data.columns:
        print("'signal' column not found after preprocessing. Generating classic crossover signals...")
        processed_data['signal'] = 0
        processed_data.loc[processed_data['short_ma'] > processed_data['long_ma'], 'signal'] = 1
        processed_data.loc[processed_data['short_ma'] < processed_data['long_ma'], 'signal'] = -1
        print("Signal column generated for classic MA crossover.")
    
    # Run a backtest
    backtest_result = await strategy.backtest(
        data=processed_data,
        initial_capital=10000  # $10,000 initial capital
    )
    
    # Print basic results
    final_equity = backtest_result.get("final_equity", backtest_result.get("final_capital", 0))
    total_return = (final_equity / 10000 - 1) * 100 if final_equity else 0
    
    print(f"Strategy: Moving Average")
    print(f"Initial Capital: $10,000.00")
    print(f"Final Capital: ${final_equity:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    import numpy as np
    trades = backtest_result.get('trades', 0)
    if isinstance(trades, (int, float, np.integer)):
        print(f"Number of Trades: {int(trades)}")
    else:
        print(f"Number of Trades: {len(trades)}")
    
    # Plot equity curve if available
    if "equity_curve" in backtest_result:
        plt.figure(figsize=(10, 6))
        plt.plot(backtest_result["equity_curve"])
        plt.title("Moving Average Strategy - Equity Curve")
        plt.xlabel("Time")
        plt.ylabel("Portfolio Value ($)")
        plt.grid(True)
        plt.savefig("my_first_backtest.png")
        plt.show()
    else:
        print("No equity curve data available to plot.")

if __name__ == "__main__":
    asyncio.run(main())