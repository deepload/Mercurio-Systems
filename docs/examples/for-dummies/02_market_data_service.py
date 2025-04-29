"""
Mercurio AI Market Data Service Example
Based on Chapter 3: Understanding the Platform

This script demonstrates the Market Data Service's capabilities,
including transparent fallback mechanisms.
"""
import os
import sys
import asyncio
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# Import the Market Data Service
from app.services.market_data import MarketDataService

async def fetch_and_display_data():
    """Fetch historical data with automatic fallback and display basic information"""
    
    print("Initializing Market Data Service...")
    market_data = MarketDataService()
    
    # List available data providers
    providers = market_data.get_available_providers()
    print(f"Available data providers: {', '.join(providers)}")
    
    # Get data for a cryptocurrency with automatic fallback
    symbol = "BTC-USD"
    start_date = datetime.strptime("2024-01-01", "%Y-%m-%d")
    end_date = datetime.strptime("2024-04-25", "%Y-%m-%d")
    timeframe = "1d"  # Daily timeframe
    
    print(f"\nFetching {timeframe} data for {symbol} from {start_date.date()} to {end_date.date()}...")
    
    # Get historical data with automatic fallback
    data = await market_data.get_historical_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe
    )
    
    # Display basic information about the data
    print(f"Retrieved {len(data)} data points")
    
    if len(data) > 0:
        print("\nData Summary:")
        print(f"Date Range: {data.index[0].date()} to {data.index[-1].date()}")
        print(f"Opening Price (First Day): ${data['open'].iloc[0]:.2f}")
        print(f"Closing Price (Last Day): ${data['close'].iloc[-1]:.2f}")
        price_change = (data['close'].iloc[-1] / data['open'].iloc[0] - 1) * 100
        print(f"Price Change: {price_change:.2f}%")
        print(f"Average Volume: {data['volume'].mean():.0f}")
        
        # Plot the price chart
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['close'], label='Close Price')
        plt.title(f'{symbol} Price Chart ({start_date.date()} to {end_date.date()})')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.grid(True)
        plt.legend()
        
        # Save the plot
        output_file = os.path.join(os.path.dirname(__file__), f"{symbol.replace('/', '_')}_price_chart.png")
        plt.savefig(output_file)
        print(f"\nPrice chart saved to: {output_file}")
        
        # Show the plot if in interactive environment
        plt.show()
        
        # Display the first 5 rows of data
        print("\nFirst 5 rows of data:")
        print(data.head())
    else:
        print("No data retrieved. Check the symbol and date range.")

if __name__ == "__main__":
    print("Market Data Service Example")
    print("=" * 50)
    asyncio.run(fetch_and_display_data())
