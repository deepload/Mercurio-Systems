"""
Tests for service layer components
"""
import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from app.services.market_data import MarketDataService
from app.services.trading import TradingService
from app.services.backtesting import BacktestingService
from app.services.strategy_manager import StrategyManager
from app.db.models import TradeAction

@pytest.mark.asyncio
async def test_market_data_service():
    """Test the Market Data Service."""
    with patch('app.services.market_data.httpx.AsyncClient') as mock_client:
        # Setup mock response for IEX Cloud API
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "date": "2023-04-01", 
                "open": 100.0, 
                "high": 105.0,
                "low": 98.0, 
                "close": 103.0, 
                "volume": 10000
            },
            {
                "date": "2023-04-02", 
                "open": 103.0, 
                "high": 108.0,
                "low": 102.0, 
                "close": 107.0, 
                "volume": 12000
            }
        ]
        mock_client.return_value.__aenter__.return_value.get.return_value = mock_response
        
        # Initialize service
        service = MarketDataService()
        
        # Test get_historical_data
        data = await service.get_historical_data(
            symbol="AAPL",
            start_date=datetime.now() - timedelta(days=10),
            end_date=datetime.now()
        )
        
        # Check results
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 2
        assert "open" in data.columns
        assert "close" in data.columns

@pytest.mark.asyncio
async def test_trading_service():
    """Test the Trading Service."""
    with patch('app.services.trading.alpaca.REST') as mock_alpaca:
        # Setup mock responses
        mock_alpaca.return_value.get_account.return_value = {
            "cash": "100000",
            "portfolio_value": "150000",
            "buying_power": "200000"
        }
        
        mock_alpaca.return_value.submit_order.return_value = {
            "id": "test-order-id",
            "client_order_id": "test-client-id",
            "status": "filled",
            "symbol": "AAPL",
            "side": "buy",
            "qty": "10"
        }
        
        # Initialize service
        service = TradingService(is_paper=True)
        
        # Test account info
        account_info = await service.get_account_info()
        assert account_info["cash"] == "100000"
        
        # Test trade execution
        trade_result = await service.execute_trade(
            symbol="AAPL",
            action=TradeAction.BUY,
            quantity=10
        )
        
        assert trade_result["status"] == "filled"
        assert trade_result["order"]["symbol"] == "AAPL"

@pytest.mark.asyncio
async def test_strategy_manager():
    """Test the Strategy Manager."""
    with patch('app.services.strategy_manager.importlib.import_module') as mock_import:
        # Setup mock strategy module
        mock_strategy_class = MagicMock()
        mock_strategy_class.__name__ = "TestStrategy"
        mock_strategy_class.__doc__ = "Test strategy for testing"
        mock_strategy_class.return_value.predict.return_value = (TradeAction.BUY, 0.85)
        
        mock_module = MagicMock()
        mock_module.__name__ = "app.strategies.test_strategy"
        mock_module.__dict__ = {"TestStrategy": mock_strategy_class}
        
        mock_import.return_value = mock_module
        
        # Initialize service
        manager = StrategyManager()
        
        # Mock the list_strategies method
        manager.strategies_cache = {"TestStrategy": mock_strategy_class}
        
        # Test get_strategy
        strategy = await manager.get_strategy("TestStrategy")
        assert strategy is not None
        
        # Test get_prediction with patched market data
        with patch.object(manager, 'market_data') as mock_market_data:
            mock_market_data.get_latest_price.return_value = 150.0
            
            prediction = await manager.get_prediction("AAPL", "TestStrategy")
            
            assert prediction["symbol"] == "AAPL"
            assert prediction["action"] == "buy"
            assert prediction["confidence"] == 0.85
