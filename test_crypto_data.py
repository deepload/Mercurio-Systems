#!/usr/bin/env python
"""
Script de test pour vérifier la récupération des données crypto depuis Alpaca
après les modifications apportées à MarketDataService
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
from app.services.market_data import MarketDataService
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def test_crypto_symbols():
    """Test la récupération de données pour des symboles crypto problématiques"""
    service = MarketDataService()
    
    # Période de test: 3 derniers jours
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3)
    
    # Liste des symboles à tester, incluant ceux qui posaient problème
    symbols_to_test = [
        'BTC/USD',    # Bitcoin (devrait fonctionner)
        'ETH/USD',    # Ethereum (devrait fonctionner)
        'MATIC/USD',  # Polygon (problématique auparavant)
        'DOGE/USD',   # Dogecoin (problématique auparavant)
    ]
    
    print(f"Testing crypto data retrieval from {start_date.date()} to {end_date.date()}")
    print("=" * 60)
    
    # Tester chaque symbole
    for symbol in symbols_to_test:
        print(f"\nTesting {symbol}:")
        try:
            # Obtenir les données historiques horaires
            df = await service.get_historical_data(symbol, start_date, end_date, '1h')
            
            if not df.empty:
                print(f"✅ SUCCESS: Got {len(df)} bars")
                print("\nFirst 3 records:")
                print(df.head(3))
                print("\nLast 3 records:")
                print(df.tail(3))
            else:
                print(f"❌ FAILED: Empty DataFrame returned for {symbol}")
        except Exception as e:
            print(f"❌ ERROR: {str(e)[:300]}")
    
    print("\n" + "=" * 60)
    print("Test completed")

if __name__ == "__main__":
    asyncio.run(test_crypto_symbols())
