"""
Test script for Alpaca API credentials
"""
import os
import logging
import asyncio
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    # Load environment variables
    load_dotenv()
    
    # Get Alpaca credentials
    alpaca_key = os.getenv("ALPACA_KEY")
    alpaca_secret = os.getenv("ALPACA_SECRET")
    alpaca_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    
    logger.info(f"Testing Alpaca connection with base URL: {alpaca_url}")
    
    # Try with URL as is
    try:
        logger.info("Testing with original URL...")
        client = tradeapi.REST(
            key_id=alpaca_key,
            secret_key=alpaca_secret,
            base_url=alpaca_url
        )
        
        # Try getting account info
        account = client.get_account()
        logger.info(f"✅ Connection successful! Account ID: {account.id}")
        logger.info(f"Account status: {account.status}")
        logger.info(f"Buying power: ${account.buying_power}")
        
    except Exception as e:
        logger.error(f"❌ Connection failed with original URL: {e}")
        
        # If URL had /v2, try without it
        if "/v2" in alpaca_url:
            try:
                fixed_url = alpaca_url.rstrip("/v2")
                logger.info(f"Testing with fixed URL (removed /v2): {fixed_url}")
                
                client = tradeapi.REST(
                    key_id=alpaca_key,
                    secret_key=alpaca_secret,
                    base_url=fixed_url
                )
                
                # Try getting account info
                account = client.get_account()
                logger.info(f"✅ Connection successful with fixed URL! Account ID: {account.id}")
                logger.info(f"Account status: {account.status}")
                logger.info(f"Buying power: ${account.buying_power}")
                
            except Exception as e:
                logger.error(f"❌ Connection failed with fixed URL: {e}")
                logger.error("Please check your API keys and URL.")
    
    # Try listing some assets
    try:
        logger.info("\nTesting asset listing...")
        assets = client.list_assets(status='active')
        logger.info(f"✅ Got {len(assets)} assets")
        # Show first 5 assets
        for i, asset in enumerate(assets[:5]):
            logger.info(f"Asset {i+1}: {asset.symbol} - {asset.name}")
    except Exception as e:
        logger.error(f"❌ Failed to list assets: {e}")

if __name__ == "__main__":
    asyncio.run(main())
