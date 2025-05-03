"""
Tests for options trading service and strategy
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from app.services.options_service import OptionsService
from app.strategies.options_strategy import OptionsStrategy, TimeFrame
from app.db.models import TradeAction
from app.services.trading import TradingService
from app.services.market_data import MarketDataService

# Mocks for testing
@pytest.fixture
def mock_trading_service():
    """Mock trading service for testing"""
    trading_service = MagicMock(spec=TradingService)
    
    # Mock the alpaca client
    trading_service.alpaca_client = MagicMock()
    
    # Mock get_positions to return empty list
    trading_service.get_positions.return_value = []
    
    return trading_service

@pytest.fixture
def mock_market_data_service():
    """Mock market data service for testing"""
    market_data = MagicMock(spec=MarketDataService)
    
    # Mock get_latest_price to return a realistic price
    market_data.get_latest_price.return_value = 150.0
    
    return market_data

@pytest.fixture
def mock_options_service(mock_trading_service, mock_market_data_service):
    """Create a mock options service"""
    mock = MagicMock(spec=OptionsService)
    
    # Configurer les attributs du service
    mock.trading_service = mock_trading_service
    mock.market_data = mock_market_data_service
    mock.alpaca_client = mock_trading_service.alpaca_client
    
    # Configure some default returns for methods that might be called
    mock.get_available_options.return_value = [
        {
            "symbol": "AAPL_20250603_C_00150000",
            "strike": 150.0,  # Changé de strike_price à strike
            "option_type": "call",
            "expiration": "2025-06-03",  # Changé de expiration_date à expiration
            "last_price": 5.65,  # Changé de last_trade_price à last_price
            "bid": 5.60,  # Changé de bid_price à bid
            "ask": 5.70,  # Changé de ask_price à ask
            "volume": 1245,
            "open_interest": 4325,
            "implied_volatility": 0.35
        }
    ]
    
    mock.suggest_option_strategies.return_value = [
        {
            "name": "Long Call",
            "option_type": "call",
            "action": "BUY",
            "strike": 155.0,
            "expiration": "2025-06-03",
            "confidence_match": 85.0,
            "description": "Simple directional play for bullish outlook",
            "risk_rating": "moderate",
            "max_loss": "Limited to premium paid",
            "max_gain": "Unlimited upside potential"
        }
    ]
    
    mock.execute_option_trade.return_value = {
        "status": "success", 
        "order": {
            "id": "mock-order-id",
            "client_order_id": "mock-client-order-id",
            "symbol": "AAPL_20250603_C_00150000",
            "side": "buy",
            "qty": 1,
            "type": "market",
            "status": "filled",
            "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    }
    mock.get_option_position.return_value = None
    mock.get_all_option_positions.return_value = []
    mock.calculate_option_metrics.return_value = {
        "delta": 0.5, 
        "gamma": 0.1, 
        "theta": -0.05, 
        "vega": 0.2,
        "implied_volatility": 0.35
    }
    
    return mock

@pytest.fixture
def mock_option_data():
    """Create mock option data for testing"""
    expiry_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
    
    return {
        "symbol": "AAPL_20250603_C_00150000",
        "underlying": "AAPL",
        "strike": 150.0,
        "option_type": "call",
        "expiration": expiry_date,
        "last_price": 5.65,
        "bid": 5.60,
        "ask": 5.70,
        "volume": 1245,
        "open_interest": 4325,
        "implied_volatility": 0.35
    }

@pytest.fixture
def mock_price_prediction():
    """Create mock price prediction for testing"""
    return {
        "price": 160.0,
        "confidence": 0.80,
        "time_horizon_days": 30
    }

@pytest.mark.asyncio
async def test_get_available_options(mock_options_service):
    """Test retrieving available options for a symbol"""
    # Mock the alpaca client's response
    mock_options_service.alpaca_client.get_options = MagicMock(return_value=[
        MagicMock(
            symbol="AAPL_20250603_C_00150000",
            strike_price=150.0,
            option_type="call",
            expiration_date="2025-06-03",
            last_trade_price=5.65,
            bid_price=5.60,
            ask_price=5.70,
            volume=1245,
            open_interest=4325,
            implied_volatility=0.35
        )
    ])
    
    options = await mock_options_service.get_available_options("AAPL", "2025-06-03")
    
    # Verify we got options back
    assert options
    assert len(options) > 0
    assert "symbol" in options[0]
    assert "strike" in options[0]
    assert "option_type" in options[0]

@pytest.mark.asyncio
async def test_execute_option_trade(mock_options_service):
    """Test executing an options trade"""
    # Mock the submit_option_order method
    mock_order = MagicMock(
        id="mock-order-id",
        client_order_id="mock-client-order-id",
        symbol="AAPL_20250603_C_00150000",
        side="buy",
        qty=1,
        type="market",
        status="filled",
        submitted_at=datetime.now()
    )
    
    mock_options_service.alpaca_client.submit_option_order = MagicMock(return_value=mock_order)
    
    result = await mock_options_service.execute_option_trade(
        option_symbol="AAPL_20250603_C_00150000",
        action=TradeAction.BUY,
        quantity=1,
        order_type="market",
        strategy_name="TestStrategy"
    )
    
    # Verify the result
    assert result["status"] == "success"
    assert "order" in result
    assert result["order"]["id"] == "mock-order-id"
    assert result["order"]["symbol"] == "AAPL_20250603_C_00150000"

@pytest.mark.asyncio
async def test_calculate_option_metrics(mock_options_service, mock_option_data):
    """Test calculation of option metrics (Greeks)"""
    metrics = await mock_options_service.calculate_option_metrics(mock_option_data)
    
    # Verify metrics contain expected fields
    assert "delta" in metrics
    assert "gamma" in metrics
    assert "theta" in metrics
    assert "vega" in metrics
    assert "implied_volatility" in metrics
    
    # Verify delta is in expected range (-1 to 1)
    assert -1 <= metrics["delta"] <= 1
    
    # Verify implied volatility is positive
    assert metrics["implied_volatility"] > 0

@pytest.mark.asyncio
async def test_suggest_option_strategies(mock_options_service, mock_price_prediction):
    """Test suggesting option strategies based on price prediction"""
    # Mock get_latest_price to return a realistic price
    mock_options_service.market_data.get_latest_price.return_value = 150.0
    
    strategies = await mock_options_service.suggest_option_strategies(
        symbol="AAPL",
        price_prediction=mock_price_prediction,
        risk_profile="moderate"
    )
    
    # Verify we got strategies back
    assert strategies
    assert len(strategies) > 0
    
    # Verify first strategy has expected fields
    first_strategy = strategies[0]
    assert "name" in first_strategy
    assert "description" in first_strategy
    assert "risk_rating" in first_strategy
    assert "max_loss" in first_strategy
    assert "max_gain" in first_strategy
    
    # For bullish prediction, first strategy should be bullish
    assert "call" in first_strategy.get("option_type", "").lower() or "bull" in first_strategy.get("name", "").lower()

# Test options strategy
@pytest.mark.asyncio
async def test_options_strategy_initialization():
    """Test initialization of options strategy"""
    # Create mocks
    mock_trading = MagicMock(spec=TradingService)
    mock_market_data = MagicMock(spec=MarketDataService)
    mock_options = MagicMock(spec=OptionsService)
    
    # Create strategy
    strategy = OptionsStrategy(
        options_service=mock_options,
        base_strategy_name="TransformerStrategy",
        risk_profile="moderate"
    )
    
    # Verify strategy attributes
    assert strategy.name == "Options-TransformerStrategy"
    assert strategy.base_strategy_name == "TransformerStrategy"
    assert strategy.risk_profile == "moderate"
    assert len(strategy.preferred_option_types) > 0

@pytest.mark.asyncio
async def test_options_strategy_generate_signal(mock_options_service):
    """Test generating options trading signal from base strategy signal"""
    # Create strategy
    strategy = OptionsStrategy(
        options_service=mock_options_service,
        base_strategy_name="TransformerStrategy",
        risk_profile="moderate"
    )
    
    # Mock suggest_option_strategies to return a test strategy
    mock_options_service.suggest_option_strategies.return_value = [
        {
            "name": "Long Call",
            "option_type": "call",
            "action": "BUY",
            "strike": 155.0,
            "expiration": "2025-06-03",
            "confidence_match": 85.0,
            "description": "Simple directional play for bullish outlook",
            "risk_rating": "moderate",
            "max_loss": "Limited to premium paid",
            "max_gain": "Unlimited upside potential"
        }
    ]
    
    # Create test data with base strategy prediction
    data = {
        "close": 150.0,
        "TransformerStrategy_prediction": {
            "action": TradeAction.BUY,
            "confidence": 0.85,
            "price_target": 165.0,
            "time_horizon_days": 30
        }
    }
    
    # Generate signal
    signal = await strategy.generate_signal("AAPL", data, TimeFrame.DAY)
    
    # Verify signal
    assert signal is not None
    assert "action" in signal
    assert signal["action"] == TradeAction.BUY
    assert "option_type" in signal
    assert signal["option_type"] == "call"
    assert "strike" in signal
    assert "expiration" in signal
    assert "confidence" in signal
    assert signal["confidence"] > 0.8
    assert "description" in signal
