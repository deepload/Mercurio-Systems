import pytest
import pandas as pd
from unittest.mock import MagicMock
from app.services.market_data import MarketDataService

@pytest.fixture
def mock_market_data():
    # Create a mock MarketDataService with predictable data
    mock = MagicMock(spec=MarketDataService)
    # Mock get_latest_price
    mock.get_latest_price.return_value = 100.0
    # Mock get_historical_data
    dates = pd.date_range(end=pd.Timestamp.today(), periods=60)
    df = pd.DataFrame({
        'date': dates,
        'open': [100 + i for i in range(60)],
        'high': [101 + i for i in range(60)],
        'low': [99 + i for i in range(60)],
        'close': [100 + i for i in range(60)],
        'volume': [1000 + i*10 for i in range(60)]
    })
    mock.get_historical_data.return_value = df
    # Mock get_market_symbols
    mock.get_market_symbols.return_value = ['AAPL', 'MSFT', 'GOOGL']
    return mock

@pytest.fixture
def api():
    from app.main import app
    from fastapi.testclient import TestClient
    return TestClient(app)

@pytest.fixture
def provider_name():
    return 'alpaca'

@pytest.fixture
def market_data():
    # Provide a real or mocked MarketDataService as needed
    mock = MagicMock(spec=MarketDataService)
    mock.get_latest_price.return_value = 100.0
    dates = pd.date_range(end=pd.Timestamp.today(), periods=5)
    df = pd.DataFrame({
        'date': dates,
        'open': [100 + i for i in range(5)],
        'high': [101 + i for i in range(5)],
        'low': [99 + i for i in range(5)],
        'close': [100 + i for i in range(5)],
        'volume': [1000 + i*10 for i in range(5)]
    })
    mock.get_historical_data.return_value = df
    mock.get_market_symbols.return_value = ['AAPL', 'MSFT', 'GOOGL']
    return mock
