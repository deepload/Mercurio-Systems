#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
API Access Test Script

This script tests access to various market data providers to ensure
your API keys are properly configured.
"""

from dotenv import load_dotenv
load_dotenv()

import asyncio
import logging
import os

from app.services.market_data import MarketDataService
from app.services.providers.factory import MarketDataProviderFactory

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test symbols
STOCK_SYMBOLS = ["AAPL", "MSFT", "GOOGL"]
CRYPTO_SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD"]

async def test_provider(provider_name):
    """Test a specific data provider"""
    logger.info(f"Testing {provider_name} provider...")
    
    # Create the factory and get provider
    factory = MarketDataProviderFactory()
    provider = factory.get_provider(provider_name)
    
    if provider is None:
        logger.error(f"Provider '{provider_name}' not found!")
        return False
    
    # Check API credentials
    if hasattr(provider, 'api_key') and provider.api_key:
        logger.info(f"  API Key: {'*' * (len(provider.api_key) - 4) + provider.api_key[-4:]}")
    else:
        logger.warning(f"  No API key found for {provider_name}")
    
    # Test with stock symbols
    success = False
    
    # Test stocks
    if provider_name != 'sample':  # Sample provider might not have this exact API
        for symbol in STOCK_SYMBOLS:
            try:
                logger.info(f"  Testing stock symbol: {symbol}")
                price = await provider.get_latest_price(symbol)
                logger.info(f"  ✓ {symbol} price: ${price:.2f}")
                success = True
            except Exception as e:
                logger.error(f"  ✗ Error with {symbol}: {str(e)}")
    
    # Test crypto
    for symbol in CRYPTO_SYMBOLS:
        try:
            logger.info(f"  Testing crypto symbol: {symbol}")
            price = await provider.get_latest_price(symbol)
            logger.info(f"  ✓ {symbol} price: ${price:.2f}")
            success = True
        except Exception as e:
            logger.error(f"  ✗ Error with {symbol}: {str(e)}")
    
    if success:
        logger.info(f"✅ {provider_name} provider test PASSED")
    else:
        logger.error(f"❌ {provider_name} provider test FAILED")
    
    return success

async def test_market_data_service():
    """Test the market data service with fallback"""
    logger.info("Testing MarketDataService with fallback...")
    
    market_data = MarketDataService()
    
    for symbol in STOCK_SYMBOLS + CRYPTO_SYMBOLS:
        try:
            logger.info(f"  Testing symbol with fallback: {symbol}")
            price = await market_data.get_latest_price(symbol)
            logger.info(f"  ✓ {symbol} price: ${price:.2f}")
        except Exception as e:
            logger.error(f"  ✗ Error with {symbol}: {str(e)}")

async def main():
    """Main function to run all tests"""
    logger.info("=== MERCURIO AI API ACCESS TEST ===")
    
    # Test each provider
    await test_provider('polygon')
    await test_provider('yahoo')
    await test_provider('sample')
    
    # Test market data service with fallback
    await test_market_data_service()
    
    logger.info("=== TEST COMPLETED ===")
    
    # Show instructions for fixing API issues
    logger.info("\nINSTRUCTIONS:")
    logger.info("1. If Polygon.io tests failed, get a new API key from https://polygon.io/")
    logger.info("2. If Alpaca tests failed, check your keys at https://app.alpaca.markets/paper/dashboard/overview")
    logger.info("3. Update your .env file with the correct API keys")
    logger.info("4. Even if all tests failed, Mercurio AI will still work with sample data")

if __name__ == "__main__":
    asyncio.run(main())
