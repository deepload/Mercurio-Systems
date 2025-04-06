"""
Tests for trading strategies
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.strategies.moving_average import MovingAverageCrossover
from app.strategies.lstm_predictor import LSTMPredictor
from app.db.models import TradeAction

@pytest.mark.asyncio
async def test_moving_average_strategy(mock_market_data):
    """Test the Moving Average Crossover strategy."""
    # Initialize strategy
    strategy = MovingAverageCrossover(short_window=5, long_window=20)
    
    # Preprocess data
    processed_data = await strategy.preprocess_data(mock_market_data)
    
    # Check preprocessing
    assert 'short_ma' in processed_data.columns
    assert 'long_ma' in processed_data.columns
    
    # Test prediction
    action, confidence = await strategy.predict(processed_data)
    
    # Check prediction results
    assert isinstance(action, TradeAction)
    assert action in [TradeAction.BUY, TradeAction.SELL, TradeAction.HOLD]
    assert 0 <= confidence <= 1
    
    # Test backtest
    backtest_results = await strategy.backtest(
        processed_data,
        initial_capital=10000.0
    )
    
    # Check backtest results
    assert 'final_capital' in backtest_results
    assert backtest_results['final_capital'] > 0
    assert 'total_return' in backtest_results
    assert 'trades' in backtest_results

@pytest.mark.asyncio
async def test_lstm_predictor_strategy(mock_market_data):
    """Test the LSTM Predictor strategy."""
    # Initialize strategy with minimal epochs for testing
    strategy = LSTMPredictor(
        prediction_days=5,
        epochs=2,  # Small number for testing speed
        batch_size=32,
        neurons=50
    )
    
    # Preprocess data
    processed_data = await strategy.preprocess_data(mock_market_data)
    
    # Verify preprocessing
    assert processed_data is not None
    assert isinstance(processed_data, pd.DataFrame)
    
    # Test training with a small subset for speed
    small_data = processed_data.iloc[-60:]  # Use only 60 days for fast testing
    
    # Test training with mocked data
    metrics = await strategy.train(small_data)
    
    # Check training results
    assert isinstance(metrics, dict)
    assert 'loss' in metrics
    
    # Test prediction
    action, confidence = await strategy.predict(small_data)
    
    # Check prediction results
    assert isinstance(action, TradeAction)
    assert action in [TradeAction.BUY, TradeAction.SELL, TradeAction.HOLD]
    assert 0 <= confidence <= 1
