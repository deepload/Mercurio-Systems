import pytest
import pandas as pd
from unittest.mock import MagicMock
from app.services.market_data import MarketDataService

@pytest.fixture
def alpaca_mode():
    # Return a default or mock mode for alpaca tests
    return 'paper'

@pytest.fixture(params=['stock', 'crypto'])
def asset_type(request):
    return request.param

@pytest.fixture
def api():
    from app.main import app
    from fastapi.testclient import TestClient
    return TestClient(app)

@pytest.fixture
def market_data():
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

@pytest.fixture
def symbols():
    return ['AAPL', 'MSFT', 'GOOGL']
