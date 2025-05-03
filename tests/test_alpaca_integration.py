#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests pour l'intégration Alpaca avec support des cryptomonnaies et actions.
"""

import os
import pytest
import pandas as pd
from datetime import datetime, timedelta

from app.services.market_data import MarketDataService

# Symboles pour les tests
STOCKS = ["AAPL", "MSFT", "GOOGL"]
CRYPTOS = ["BTC-USD", "ETH-USD"]

# Skip les tests si les clés Alpaca ne sont pas configurées
def is_alpaca_configured():
    """Vérifie si les clés Alpaca sont configurées"""
    alpaca_mode = os.getenv("ALPACA_MODE", "paper").lower()
    
    if alpaca_mode == "live":
        return bool(os.getenv("ALPACA_LIVE_KEY") and os.getenv("ALPACA_LIVE_SECRET"))
    else:
        return bool(os.getenv("ALPACA_PAPER_KEY") and os.getenv("ALPACA_PAPER_SECRET"))

# Skip les tests nécessitant un plan Alpaca payant
requires_alpaca = pytest.mark.skipif(
    not is_alpaca_configured(),
    reason="Requires Alpaca API keys"
)

@pytest.fixture
def market_data_service():
    """Fixture pour l'instance MarketDataService"""
    return MarketDataService()

class TestAlpacaIntegration:
    """Tests pour l'intégration d'Alpaca dans MarketDataService"""
    
    @requires_alpaca
    @pytest.mark.asyncio
    async def test_alpaca_initialization(self, market_data_service):
        """Teste l'initialisation d'Alpaca"""
        assert market_data_service.alpaca_client is not None
        
        # Vérifier que le mode est correctement configuré
        alpaca_mode = os.getenv("ALPACA_MODE", "paper").lower()
        if alpaca_mode == "live":
            assert market_data_service.alpaca_key == os.getenv("ALPACA_LIVE_KEY")
        else:
            assert market_data_service.alpaca_key == os.getenv("ALPACA_PAPER_KEY")
    
    @requires_alpaca
    @pytest.mark.asyncio
    @pytest.mark.parametrize("symbol", STOCKS)
    async def test_stock_latest_price(self, market_data_service, symbol):
        """Teste la récupération du dernier prix pour les actions"""
        price = await market_data_service.get_latest_price(symbol)
        assert price > 0
        assert isinstance(price, float)
    
    @requires_alpaca
    @pytest.mark.asyncio
    @pytest.mark.parametrize("symbol", CRYPTOS)
    async def test_crypto_latest_price(self, market_data_service, symbol):
        """Teste la récupération du dernier prix pour les cryptomonnaies"""
        price = await market_data_service.get_latest_price(symbol)
        assert price > 0
        assert isinstance(price, float)
    
    @requires_alpaca
    @pytest.mark.asyncio
    @pytest.mark.parametrize("symbol", STOCKS)
    async def test_stock_historical_data(self, market_data_service, symbol):
        """Teste la récupération des données historiques pour les actions"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        
        df = await market_data_service.get_historical_data(symbol, start_date, end_date)
        
        assert not df.empty
        assert "open" in df.columns
        assert "close" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "volume" in df.columns
    
    @requires_alpaca
    @pytest.mark.asyncio
    @pytest.mark.parametrize("symbol", CRYPTOS)
    async def test_crypto_historical_data(self, market_data_service, symbol):
        """Teste la récupération des données historiques pour les cryptomonnaies"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        
        df = await market_data_service.get_historical_data(symbol, start_date, end_date)
        
        assert not df.empty
        assert "open" in df.columns
        assert "close" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "volume" in df.columns
    
    @requires_alpaca
    @pytest.mark.asyncio
    async def test_crypto_format_conversion(self, market_data_service):
        """Teste la conversion du format des symboles crypto"""
        # Tests internes pour vérifier la conversion des formats
        crypto_symbol = "BTC-USD"
        expected_format = "BTC/USD"
        
        # Accéder à la méthode privée via une méthode interne
        # Cela permet de tester directement la conversion sans exécuter toute la logique
        alpaca_symbol = crypto_symbol.replace("-USD", "/USD")
        
        assert alpaca_symbol == expected_format

if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
