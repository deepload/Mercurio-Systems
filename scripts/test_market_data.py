"""
Test script for Market Data Providers
This script tests the different market data providers to verify they're working correctly.
"""
import asyncio
import logging
import os
from datetime import datetime, timedelta
import pandas as pd

from app.services.market_data import MarketDataService
from app.services.providers.factory import MarketDataProviderFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_provider(provider_name, symbol="AAPL"):
    """Test a specific provider with basic operations"""
    logger.info(f"\n{'='*20} Testing {provider_name} Provider {'='*20}")
    
    # Initialize service with specific provider
    service = MarketDataService(provider_name=provider_name)
    
    # Test provider availability
    logger.info(f"Provider available: {provider_name in service.get_available_providers()}")
    
    # Get current provider
    provider = service.active_provider
    logger.info(f"Active provider: {provider.name}")
    
    # Test historical data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    try:
        logger.info(f"Fetching historical data for {symbol}...")
        df = await service.get_historical_data(symbol, start_date, end_date)
        if isinstance(df, pd.DataFrame) and not df.empty:
            logger.info(f"✅ Successfully retrieved {len(df)} records of historical data")
            logger.info(f"Sample data:\n{df.head(3)}")
        else:
            logger.info("❌ Failed to retrieve historical data (empty result)")
    except Exception as e:
        logger.error(f"❌ Error retrieving historical data: {e}")
    
    # Test latest price
    try:
        logger.info(f"Fetching latest price for {symbol}...")
        price = await service.get_latest_price(symbol)
        logger.info(f"✅ Latest price: ${price:.2f}")
    except Exception as e:
        logger.error(f"❌ Error retrieving latest price: {e}")
    
    # Test market symbols
    try:
        logger.info("Fetching market symbols...")
        symbols = await service.get_market_symbols()
        logger.info(f"✅ Retrieved {len(symbols)} symbols")
        logger.info(f"Sample symbols: {symbols[:5]}")
    except Exception as e:
        logger.error(f"❌ Error retrieving market symbols: {e}")
    
    logger.info(f"{'='*60}\n")

async def main():
    """Run tests for all providers"""
    logger.info("Starting market data provider tests")
    
    # Get available providers
    factory = MarketDataProviderFactory()
    providers = factory.get_available_providers()
    logger.info(f"Available providers: {providers}")
    
    # Test the default provider first
    service = MarketDataService()
    default_provider = service.active_provider.name.lower()
    logger.info(f"Default provider: {default_provider}")
    
    # Test each provider
    for provider in providers:
        await test_provider(provider)
    
    logger.info("All tests completed")

if __name__ == "__main__":
    asyncio.run(main())
