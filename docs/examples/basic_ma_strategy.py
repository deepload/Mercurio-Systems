"""
Test script for the Basic Moving Average Strategy example from README.md
"""
import os
import sys
import asyncio
import logging
from datetime import datetime, timedelta

# Add the project root to Python path so we can import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import required services and strategies
from app.services.market_data import MarketDataService
from app.strategies.moving_average import MovingAverageStrategy

async def run_simple_strategy():
    """
    This is the example from README.md
    """
    # Initialize services
    market_data = MarketDataService()
    strategy = MovingAverageStrategy(short_window=10, long_window=30)
    
    # Get data and generate signals
    start_date = datetime.now() - timedelta(days=180)  # Last 180 days
    end_date = datetime.now()
    
    logger.info("Fetching historical data for AAPL...")
    data = await market_data.get_historical_data("AAPL", start_date, end_date)
    logger.info(f"Got {len(data)} data points")
    
    logger.info("Processing data with MovingAverageStrategy...")
    processed_data = await strategy.preprocess_data(data)
    
    logger.info("Generating trading signal...")
    signal, confidence = await strategy.predict(processed_data)
    
    print(f"\nAAPL Trading Signal: {signal} (Confidence: {confidence:.2f})")

if __name__ == "__main__":
    # Run the example
    asyncio.run(run_simple_strategy())
