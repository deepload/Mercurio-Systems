"""
Test script for the Alpaca market data provider.

This script confirms that the Alpaca provider is properly configured
and can retrieve market data through Alpaca's API.
"""
import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import required components
from app.services.market_data import MarketDataService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_alpaca_provider():
    """Test the Alpaca data provider's functionality."""
    logger.info("Initializing MarketDataService...")
    market_data = MarketDataService(provider_name="alpaca")
    
    # Log all available providers
    available_providers = market_data.get_available_providers()
    logger.info(f"Available providers: {available_providers}")
    
    # Check which provider is active
    active_provider = await market_data.active_provider()
    logger.info(f"Active provider: {active_provider.name if active_provider else 'None'}")
    
    # Test retrieving historical data
    symbol = "AAPL"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    logger.info(f"Fetching historical data for {symbol} from {start_date.date()} to {end_date.date()}...")
    data = await market_data.get_historical_data(symbol, start_date, end_date)
    
    if data.empty:
        logger.error("No data returned. Provider may not be properly configured or subscription level is insufficient.")
    else:
        logger.info(f"Successfully retrieved {len(data)} rows of data")
        logger.info(f"Data preview:\n{data.head()}")
    
    # Test fetching latest price
    logger.info(f"Fetching latest price for {symbol}...")
    price = await market_data.get_latest_price(symbol)
    logger.info(f"Latest price for {symbol}: {price}")
    
    # Test retrieving a list of symbols
    logger.info("Fetching available stock symbols...")
    symbols = await market_data.get_market_symbols()
    logger.info(f"Retrieved {len(symbols)} stock symbols")
    if symbols:
        logger.info(f"Sample symbols: {symbols[:5]}")
    
    # If you have an AlgoTrader Plus subscription, test option symbols too
    logger.info("Fetching available option symbols (requires AlgoTrader Plus)...")
    try:
        # Alpaca allows fetching option data with the right subscription
        option_symbols = await market_data.get_market_symbols(market_type="option")
        logger.info(f"Retrieved {len(option_symbols)} option symbols")
        if option_symbols:
            logger.info(f"Sample option symbols: {option_symbols[:5]}")
    except Exception as e:
        logger.warning(f"Error fetching option symbols: {e}")
        logger.warning("This may be normal if your subscription doesn't include options data")

if __name__ == "__main__":
    asyncio.run(test_alpaca_provider())
