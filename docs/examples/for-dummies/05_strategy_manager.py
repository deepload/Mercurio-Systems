"""
Mercurio AI Strategy Manager Example
Based on Chapter 3: Understanding the Platform

This script demonstrates how to list, initialize, and use different trading strategies
through the Strategy Manager.
"""
import os
import sys
import asyncio
import pandas as pd
from datetime import datetime, timedelta

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# Import required components
from app.services.strategy_manager import StrategyManager
from app.services.market_data import MarketDataService

async def explore_strategies():
    """Explore available strategies and demonstrate their usage"""
    
    print("Initializing Strategy Manager...")
    strategy_manager = StrategyManager()
    
    # List all available strategies
    print("\nListing available strategies:")
    print("=" * 50)
    strategies = await strategy_manager.list_strategies()
    
    if not strategies:
        print("No strategies found or strategy listing not available.")
    else:
        for i, strategy in enumerate(strategies, 1):
            print(f"{i}. {strategy['name']}: {strategy['description']}")
    
    # Get information about a specific strategy
    strategy_name = "MovingAverageStrategy"
    print(f"\nGetting information about {strategy_name}:")
    print("=" * 50)
    
    strategy_info = await strategy_manager.get_strategy_info(strategy_name)
    
    if strategy_info:
        print(f"Name: {strategy_info.get('name', strategy_name)}")
        print(f"Description: {strategy_info.get('description', 'No description available')}")
        print(f"Type: {strategy_info.get('type', 'Unknown')}")
        
        # Print parameters if available
        if 'parameters' in strategy_info:
            print("\nParameters:")
            for param, details in strategy_info['parameters'].items():
                if isinstance(details, dict):
                    print(f"  - {param}: {details.get('description', '')}")
                    print(f"    Default: {details.get('default', 'None')}")
                else:
                    print(f"  - {param}: {details}")
    else:
        print(f"No information available for {strategy_name}")
    
    # Initialize a specific strategy with custom parameters
    print(f"\nInitializing {strategy_name} with custom parameters:")
    print("=" * 50)
    
    # Custom parameters for moving average strategy
    params = {
        "short_window": 15,   # 15-day short moving average
        "long_window": 45,    # 45-day long moving average
        "use_ml": True        # Use machine learning enhancement
    }
    
    try:
        strategy = await strategy_manager.get_strategy(strategy_name, params)
        print(f"Successfully initialized {strategy_name} with parameters:")
        for key, value in params.items():
            print(f"  - {key}: {value}")
        
        # Demonstrate the strategy with sample data
        print("\nDemonstrating strategy with sample data:")
        print("=" * 50)
        
        # Get sample data
        market_data = MarketDataService()
        symbol = "AAPL"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)  # 6 months of data
        
        print(f"Fetching {symbol} data from {start_date.date()} to {end_date.date()}...")
        data = await market_data.get_historical_data(symbol, start_date, end_date)
        print(f"Retrieved {len(data)} data points")
        
        # Preprocess data for the strategy
        print("Preprocessing data...")
        processed_data = await strategy.preprocess_data(data)
        
        # Generate a trading signal
        print("Generating trading signal...")
        signal, confidence = await strategy.predict(processed_data)
        
        print(f"\nTrading Signal: {signal}")
        print(f"Confidence: {confidence:.2f}")
        
        # Suggest next steps
        print("\nNext steps with this strategy could include:")
        print("1. Running a backtest to evaluate its performance")
        print("2. Optimizing parameters for better results")
        print("3. Deploying for paper trading")
        print("4. Combining with other strategies in a portfolio")
    
    except Exception as e:
        print(f"Error initializing strategy: {e}")
        print("This could be due to limitations in the demo or configuration.")

if __name__ == "__main__":
    print("Strategy Manager Example")
    print("=" * 50)
    asyncio.run(explore_strategies())
