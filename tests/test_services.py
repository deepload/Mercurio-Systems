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
    with patch('httpx.AsyncClient') as mock_client:
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
        service.alpaca_client = None  # Force fallback to mocked HTTP client
        service.subscription_level = 1  # Patch to allow fallback to mock
        
        # Test get_historical_data
        data = await service.get_historical_data(
            symbol="AAPL",
            start_date=datetime.now() - timedelta(days=10),
            end_date=datetime.now()
        )
        
        # Check results
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 7
        assert "open" in data.columns
        assert "close" in data.columns

@pytest.mark.asyncio
async def test_trading_service():
    """Test the Trading Service."""
    with patch('alpaca_trade_api.REST') as mock_alpaca:
        # Setup mock responses
        class MockAccount:
            def __init__(self):
                self.id = "test-account-id"
                self.cash = "100000"
                self.portfolio_value = "150000"
                self.equity = "150000"
                self.buying_power = "200000"
                self.initial_margin = "0"
                self.daytrade_count = 0
                self.status = "ACTIVE"
                self.error = None
        mock_alpaca.return_value.get_account.return_value = MockAccount()

        class MockOrder:
            def __init__(self):
                self.id = "test-order-id"
                self.client_order_id = "test-client-id"
                self.status = "filled"
                self.symbol = "AAPL"
                self.side = "buy"
                self.qty = "10"
                self.type = "market"
                self.submitted_at = None
        mock_alpaca.return_value.submit_order.return_value = MockOrder()

        # Initialize service
        service = TradingService(is_paper=True)

        # Test account info
        account_info = await service.get_account_info()
        assert account_info["cash"] == 100000.0
        assert account_info["portfolio_value"] == 150000.0
        assert account_info["buying_power"] == 200000.0
        assert account_info["status"] == "ACTIVE"

        # Test trade execution
        trade_result = await service.execute_trade(
            symbol="AAPL",
            action=TradeAction.BUY,
            quantity=10
        )

        assert trade_result["status"] == "success"
        assert trade_result["order"]["status"] == "filled"
        assert trade_result["order"]["symbol"] == "AAPL"

@pytest.mark.asyncio
async def test_strategy_manager():
    """Test the Strategy Manager."""
    from unittest.mock import AsyncMock
    def import_module_side_effect(name):
        from unittest.mock import AsyncMock
        mock_strategy_class = MagicMock()
        mock_strategy_class.__name__ = "TestStrategy"
        mock_strategy_class.__doc__ = "Test strategy for testing"
        mock_strategy_instance = AsyncMock()
        mock_strategy_instance.predict = AsyncMock(return_value=(TradeAction.BUY, 0.85))
        mock_strategy_class.return_value = mock_strategy_instance
        mock_module = MagicMock()
        mock_module.__name__ = "app.strategies.test_strategy"
        mock_module.__dict__ = {"TestStrategy": mock_strategy_class}
        return mock_module
    with patch('app.services.strategy_manager.importlib.import_module', side_effect=import_module_side_effect):
        # Initialize service
        manager = StrategyManager()

        # Mock the list_strategies method
        from unittest.mock import AsyncMock
        mock_strategy_class = MagicMock()
        mock_strategy_class.__name__ = "TestStrategy"
        mock_strategy_class.__doc__ = "Test strategy for testing"
        mock_strategy_instance = AsyncMock()
        mock_strategy_instance.predict = AsyncMock(return_value=(TradeAction.BUY, 0.85))
        mock_strategy_class.return_value = mock_strategy_instance
        manager.strategies_cache = {"TestStrategy": mock_strategy_class}

        # Test get_strategy
        strategy = await manager.get_strategy("TestStrategy")
        assert strategy is not None

        # Test get_prediction with patched market data
        from unittest.mock import AsyncMock
        with patch.object(manager, 'market_data') as mock_market_data:
            mock_market_data.get_latest_price = AsyncMock(return_value=150.0)

            prediction = await manager.get_prediction("AAPL", "TestStrategy")

            assert prediction["symbol"] == "AAPL"
            assert prediction["action"] == "buy"
            assert prediction["confidence"] == 0.85
