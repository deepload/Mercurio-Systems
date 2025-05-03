#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test des actions et cryptomonnaies
Ce script teste si Mercurio AI peut toujours récupérer des données
pour les actions et les cryptomonnaies après nos modifications.
"""

from dotenv import load_dotenv
load_dotenv()

import asyncio
import logging
import os
import pandas as pd
from datetime import datetime, timedelta

from app.services.market_data import MarketDataService

# Configuration du logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Symboles à tester
STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
CRYPTOS = ["BTC-USD", "ETH-USD", "SOL-USD"]

async def test_price_fetching(market_data, symbols, asset_type):
    """Teste la récupération des prix pour une liste de symboles"""
    print(f"\n=== TEST {asset_type.upper()} PRICES ===")
    success_count = 0
    
    for symbol in symbols:
        print(f"\nTesting {symbol}:")
        try:
            # 1. Test du prix actuel
            price = await market_data.get_latest_price(symbol)
            print(f"  ✓ Latest price: ${price:.2f}")
            success_count += 1
            
            # 2. Test des données historiques
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5)
            
            print(f"  Getting historical data for {symbol}...")
            df = await market_data.get_historical_data(symbol, start_date, end_date)
            
            if not df.empty:
                print(f"  ✓ Historical data: {len(df)} rows")
                print(f"  Recent prices: {df['close'].tail(3).tolist()}")
                success_count += 1
            else:
                print(f"  ✗ No historical data found for {symbol}")
        except Exception as e:
            print(f"  ✗ Error with {symbol}: {str(e)}")
    
    if success_count > 0:
        percentage = (success_count / (len(symbols) * 2)) * 100
        print(f"\n✅ {asset_type.upper()} TEST: {success_count}/{len(symbols) * 2} operations succeeded ({percentage:.1f}%)")
    else:
        print(f"\n❌ {asset_type.upper()} TEST FAILED: No successful operations")

async def main():
    """Fonction principale"""
    print("=== MERCURIO AI DUAL MARKET TEST ===")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Mode Alpaca configuré
    alpaca_mode = os.getenv("ALPACA_MODE", "paper").upper()
    print(f"Alpaca mode: {alpaca_mode}")
    
    # Initialiser le service de données de marché
    market_data = MarketDataService()
    
    # 1. Test des actions
    await test_price_fetching(market_data, STOCKS, "stock")
    
    # 2. Test des cryptomonnaies
    await test_price_fetching(market_data, CRYPTOS, "crypto")
    
    print("\n=== TEST COMPLETED ===")

if __name__ == "__main__":
    asyncio.run(main())
