"""
Tests for API endpoints
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app
from app.db.models import TradeAction

client = TestClient(app)

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    # Accept either {"status": "ok"} or {"status": "healthy", ...}
    data = response.json()
    assert data.get("status") in ("ok", "healthy")

@patch('app.services.strategy_manager.StrategyManager.list_strategies', autospec=True)
def test_list_strategies(mock_list_strategies):
    """Test the list strategies endpoint."""
    # Mock strategy list
    mock_list_strategies.return_value = [
        {
            "name": "MovingAverageCrossover",
            "description": "Trading strategy based on moving average crossover",
            "parameters": {"short_window": 5, "long_window": 20},
            "requires_training": False
        },
        {
            "name": "LSTMPredictor",
            "description": "Neural network based strategy using LSTM",
            "parameters": {"prediction_days": 5, "epochs": 50},
            "requires_training": True
        }
    ]
    
    response = client.get("/api/strategies")
    assert response.status_code == 200
    assert len(response.json()) == 2
    assert response.json()[0]["name"] == "MovingAverageCrossover"
    assert response.json()[1]["name"] == "LSTMPredictor"

@patch('app.services.strategy_manager.StrategyManager.get_prediction', autospec=True)
def test_get_prediction(mock_get_prediction):
    """Test the prediction endpoint."""
    # Mock prediction result
    mock_get_prediction.return_value = {
        "symbol": "AAPL",
        "strategy": "MovingAverageCrossover",
        "action": "buy",
        "confidence": 0.85,
        "price": 150.25,
        "timestamp": "2023-04-06T15:30:00Z"
    }
    
    response = client.get("/api/predict", params={"symbol": "AAPL", "strategy": "MovingAverageCrossover"})
    assert response.status_code == 200
    assert response.json()["symbol"] == "AAPL"
    assert response.json()["action"] == "buy"
    assert response.json()["confidence"] == 0.85

@patch('app.services.trading.TradingService.execute_trade', autospec=True)
def test_execute_trade(mock_execute_trade):
    """Test the trade execution endpoint."""
    # Mock trade result
    mock_execute_trade.return_value = {
        "status": "filled",
        "order": {
            "id": "test-order-id",
            "symbol": "AAPL",
            "side": "buy",
            "qty": "10",
            "filled_avg_price": "150.25"
        }
    }
    
    # Test request
    request_data = {
        "strategy": "MovingAverageCrossover",
        "symbol": "AAPL",
        "action": "buy",
        "quantity": 10,
        "paper_trading": True
    }
    
    response = client.post("/api/trade", json=request_data)
    assert response.status_code == 200
    assert response.json()["status"] == "filled"
    assert response.json()["order"]["symbol"] == "AAPL"

@patch('app.services.trading.TradingService.get_account_info', autospec=True)
def test_get_account_info(mock_get_account_info):
    """Test the account info endpoint."""
    # Mock account info
    mock_get_account_info.return_value = {
        "cash": "100000.0",
        "portfolio_value": "150000.0",
        "buying_power": "200000.0",
        "status": "ACTIVE"
    }
    
    response = client.get("/api/account")
    assert response.status_code == 200
    assert float(response.json()["cash"]) == 100000.0
    assert response.json()["status"] == "ACTIVE"

@patch('app.services.backtesting.BacktestingService.run_backtest', autospec=True)
@patch('app.services.strategy_manager.StrategyManager.save_backtest_result', autospec=True)
def test_run_backtest(mock_save_backtest, mock_run_backtest):
    """Test the backtest endpoint."""
    # Mock backtest result
    backtest_result = {
        "strategy": "MovingAverageCrossover",
        "symbol": "AAPL",
        "start_date": "2023-01-01T00:00:00Z",
        "end_date": "2023-04-01T00:00:00Z",
        "initial_capital": 10000.0,
        "final_capital": 12500.0,
        "total_return": 0.25,
        "sharpe_ratio": 1.2,
        "max_drawdown": 0.1
    }
    
    mock_run_backtest.return_value = backtest_result
    mock_save_backtest.return_value = 1  # ID of saved backtest
    
    # Test request
    request_data = {
        "strategy": "MovingAverageCrossover",
        "symbol": "AAPL",
        "start_date": "2023-01-01",
        "end_date": "2023-04-01",
        "initial_capital": 10000.0
    }
    
    response = client.post("/backtest", json=request_data)
    assert response.status_code == 200
    assert response.json()["strategy"] == "MovingAverageCrossover"
    assert response.json()["total_return"] == 0.25
