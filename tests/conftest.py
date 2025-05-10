"""
Test configuration for pytest
"""
import os
import pytest
import asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.db.database import get_db, Base

# Test database URL - using SQLite for tests
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def test_engine():
    """Create a test database engine."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=False,
        future=True
    )
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        
    yield engine
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        
    await engine.dispose()

@pytest.fixture
async def test_db(test_engine):
    """Create a test database session."""
    async_session = sessionmaker(
        test_engine, 
        class_=AsyncSession, 
        expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session
        await session.rollback()

@pytest.fixture
async def client(test_db):
    """Create a test client with the test database."""
    async def override_get_db():
        yield test_db
        
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as client:
        yield client
    
    app.dependency_overrides.clear()

@pytest.fixture
def mock_market_data():
    """Mock market data for testing."""
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Create 100 days of mock data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=100)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate mock price data with some trend and volatility
    base_price = 100.0
    trend = np.linspace(0, 30, len(dates))  # Upward trend
    noise = np.random.normal(0, 5, len(dates))  # Add some noise
    
    prices = base_price + trend + noise
    
    # Create DataFrame
    data = pd.DataFrame({
        'open': prices * 0.99,
        'high': prices * 1.02,
        'low': prices * 0.97,
        'close': prices,
        'volume': np.random.randint(1000, 100000, len(dates))
    }, index=dates)
    
    return data

@pytest.fixture
def mock_alpaca_order():
    """Mock Alpaca order response for testing."""
    return {
        "id": "test-order-id",
        "client_order_id": "test-client-order-id",
        "status": "filled",
        "symbol": "AAPL",
        "side": "buy",
        "type": "market",
        "qty": "10",
        "filled_avg_price": "150.25",
        "filled_at": "2023-04-06T15:30:00Z",
        "created_at": "2023-04-06T15:29:55Z"
    }
