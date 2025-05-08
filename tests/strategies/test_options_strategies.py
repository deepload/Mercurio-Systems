"""
Unit tests for options trading strategies.

These tests verify the functionality of various options strategies
including Covered Call, Cash Secured Put, Long Call, Long Put, and Iron Condor.
"""

import unittest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from app.strategies.options.base_options_strategy import BaseOptionsStrategy
from app.strategies.options.covered_call import CoveredCallStrategy
from app.strategies.options.cash_secured_put import CashSecuredPutStrategy
from app.strategies.options.long_call import LongCallStrategy
from app.strategies.options.long_put import LongPutStrategy
from app.strategies.options.iron_condor import IronCondorStrategy
from app.core.models.option import OptionContract, OptionType


class TestOptionsStrategies(unittest.TestCase):
    """Test suite for options trading strategies."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample market data
        dates = pd.date_range(start='2023-01-01', end='2023-03-01')
        self.market_data = pd.DataFrame({
            'open': np.random.uniform(90, 110, len(dates)),
            'high': np.random.uniform(95, 115, len(dates)),
            'low': np.random.uniform(85, 105, len(dates)),
            'close': np.random.uniform(90, 110, len(dates)),
            'volume': np.random.randint(1000, 100000, len(dates))
        }, index=dates)
        
        # Calculate some common technical indicators
        self.market_data['sma20'] = self.market_data['close'].rolling(window=20).mean()
        self.market_data['sma50'] = self.market_data['close'].rolling(window=50).mean()
        
        # Set the last price to 100 for easier testing
        self.market_data.iloc[-1, self.market_data.columns.get_loc('close')] = 100.0
        
        # Create sample option contracts
        self.sample_call = OptionContract(
            symbol="AAPL_20230630_C100",
            underlying="AAPL",
            option_type=OptionType.CALL,
            strike=100.0,
            expiry_date="2023-06-30",
            bid=3.5,
            ask=3.7,
            last=3.6,
            volume=1000,
            open_interest=5000,
            implied_volatility=0.30,
            delta=0.50,
            gamma=0.05,
            theta=-0.10,
            vega=0.15,
            rho=0.05
        )
        
        self.sample_put = OptionContract(
            symbol="AAPL_20230630_P100",
            underlying="AAPL",
            option_type=OptionType.PUT,
            strike=100.0,
            expiry_date="2023-06-30",
            bid=3.3,
            ask=3.5,
            last=3.4,
            volume=800,
            open_interest=4000,
            implied_volatility=0.28,
            delta=-0.50,
            gamma=0.05,
            theta=-0.10,
            vega=0.15,
            rho=-0.05
        )
        
        self.sample_call_otm = OptionContract(
            symbol="AAPL_20230630_C110",
            underlying="AAPL",
            option_type=OptionType.CALL,
            strike=110.0,
            expiry_date="2023-06-30",
            bid=1.3,
            ask=1.5,
            last=1.4,
            volume=700,
            open_interest=3000,
            implied_volatility=0.25,
            delta=0.30,
            gamma=0.04,
            theta=-0.08,
            vega=0.12,
            rho=0.03
        )
        
        self.sample_put_otm = OptionContract(
            symbol="AAPL_20230630_P90",
            underlying="AAPL",
            option_type=OptionType.PUT,
            strike=90.0,
            expiry_date="2023-06-30",
            bid=1.2,
            ask=1.4,
            last=1.3,
            volume=600,
            open_interest=2500,
            implied_volatility=0.22,
            delta=-0.30,
            gamma=0.04,
            theta=-0.07,
            vega=0.11,
            rho=-0.03
        )
        
        # Create mock broker
        self.mock_broker = AsyncMock()
        self.mock_broker.enable_options = True
        
        # Setup broker get_account mock
        self.mock_broker.get_account = AsyncMock(return_value={
            "buying_power": 100000.0,
            "equity": 100000.0,
            "cash": 100000.0
        })
        
    async def async_setUp(self):
        """Asynchronous setup to be called by each test."""
        pass
    
    def run_async_test(self, coroutine):
        """Helper function to run async tests."""
        return asyncio.run(coroutine)
    
    def test_covered_call_initialization(self):
        """Test initialization of CoveredCallStrategy."""
        strategy = CoveredCallStrategy(
            underlying_symbol="AAPL",
            account_size=100000.0,
            max_position_size=0.10,
            profit_target_pct=0.50,
            stop_loss_pct=0.50
        )
        
        self.assertEqual(strategy.underlying_symbol, "AAPL")
        self.assertEqual(strategy.account_size, 100000.0)
        self.assertEqual(strategy.max_position_size, 0.10)
        self.assertEqual(strategy.profit_target_pct, 0.50)
        self.assertEqual(strategy.stop_loss_pct, 0.50)
        self.assertIsNone(strategy.current_call)
    
    def test_cash_secured_put_initialization(self):
        """Test initialization of CashSecuredPutStrategy."""
        strategy = CashSecuredPutStrategy(
            underlying_symbol="AAPL",
            account_size=100000.0,
            max_position_size=0.10,
            min_implied_volatility=0.20,
            profit_target_pct=0.50,
            stop_loss_pct=0.50
        )
        
        self.assertEqual(strategy.underlying_symbol, "AAPL")
        self.assertEqual(strategy.account_size, 100000.0)
        self.assertEqual(strategy.max_position_size, 0.10)
        self.assertEqual(strategy.min_implied_volatility, 0.20)
        self.assertEqual(strategy.profit_target_pct, 0.50)
        self.assertEqual(strategy.stop_loss_pct, 0.50)
        self.assertIsNone(strategy.current_put)
    
    def test_long_call_initialization(self):
        """Test initialization of LongCallStrategy."""
        strategy = LongCallStrategy(
            underlying_symbol="AAPL",
            account_size=100000.0,
            max_position_size=0.05,
            profit_target_pct=1.00,
            stop_loss_pct=0.50
        )
        
        self.assertEqual(strategy.underlying_symbol, "AAPL")
        self.assertEqual(strategy.account_size, 100000.0)
        self.assertEqual(strategy.max_position_size, 0.05)
        self.assertEqual(strategy.profit_target_pct, 1.00)
        self.assertEqual(strategy.stop_loss_pct, 0.50)
        self.assertIsNone(strategy.current_call)
    
    def test_long_put_initialization(self):
        """Test initialization of LongPutStrategy."""
        strategy = LongPutStrategy(
            underlying_symbol="AAPL",
            account_size=100000.0,
            max_position_size=0.05,
            profit_target_pct=1.00,
            stop_loss_pct=0.50
        )
        
        self.assertEqual(strategy.underlying_symbol, "AAPL")
        self.assertEqual(strategy.account_size, 100000.0)
        self.assertEqual(strategy.max_position_size, 0.05)
        self.assertEqual(strategy.profit_target_pct, 1.00)
        self.assertEqual(strategy.stop_loss_pct, 0.50)
        self.assertIsNone(strategy.current_put)
    
    def test_iron_condor_initialization(self):
        """Test initialization of IronCondorStrategy."""
        strategy = IronCondorStrategy(
            underlying_symbol="AAPL",
            account_size=100000.0,
            max_position_size=0.10,
            profit_target_pct=0.50,
            stop_loss_pct=1.50,
            wing_width=1
        )
        
        self.assertEqual(strategy.underlying_symbol, "AAPL")
        self.assertEqual(strategy.account_size, 100000.0)
        self.assertEqual(strategy.max_position_size, 0.10)
        self.assertEqual(strategy.profit_target_pct, 0.50)
        self.assertEqual(strategy.stop_loss_pct, 1.50)
        self.assertEqual(strategy.wing_width, 1)
        self.assertIsNone(strategy.short_put)
        self.assertIsNone(strategy.long_put)
        self.assertIsNone(strategy.short_call)
        self.assertIsNone(strategy.long_call)
    
    async def test_covered_call_should_enter(self):
        """Test should_enter logic for CoveredCallStrategy."""
        # Setup strategy
        strategy = CoveredCallStrategy(
            underlying_symbol="AAPL",
            account_size=100000.0,
            max_position_size=0.10
        )
        strategy.broker = self.mock_broker
        
        # Mock option chain response
        self.mock_broker.get_option_chain = AsyncMock(return_value=[
            self.sample_call, self.sample_call_otm
        ])
        
        # Test the should_enter method
        result = await strategy.should_enter(self.market_data)
        
        # Verify that get_option_chain was called correctly
        self.mock_broker.get_option_chain.assert_called_once()
        
        # In our sample data, the conditions should be met for entry
        self.assertTrue(result)
        self.assertIsNotNone(strategy.current_call)
    
    async def test_cash_secured_put_should_enter(self):
        """Test should_enter logic for CashSecuredPutStrategy."""
        # Setup strategy
        strategy = CashSecuredPutStrategy(
            underlying_symbol="AAPL",
            account_size=100000.0,
            max_position_size=0.10,
            min_implied_volatility=0.20
        )
        strategy.broker = self.mock_broker
        
        # Mock option chain response
        self.mock_broker.get_option_chain = AsyncMock(return_value=[
            self.sample_put, self.sample_put_otm
        ])
        
        # Test the should_enter method
        result = await strategy.should_enter(self.market_data)
        
        # Verify that get_option_chain was called correctly
        self.mock_broker.get_option_chain.assert_called_once()
        
        # In our sample data, the conditions should be met for entry
        self.assertTrue(result)
        self.assertIsNotNone(strategy.current_put)
    
    async def test_long_call_should_enter(self):
        """Test should_enter logic for LongCallStrategy."""
        # Setup strategy
        strategy = LongCallStrategy(
            underlying_symbol="AAPL",
            account_size=100000.0,
            max_position_size=0.05,
            min_implied_volatility=0.20
        )
        strategy.broker = self.mock_broker
        
        # Mock option chain response
        self.mock_broker.get_option_chain = AsyncMock(return_value=[
            self.sample_call, self.sample_call_otm
        ])
        
        # Test the should_enter method
        result = await strategy.should_enter(self.market_data)
        
        # Verify that get_option_chain was called correctly
        self.mock_broker.get_option_chain.assert_called_once()
        
        # In our sample data, the conditions should be met for entry
        self.assertTrue(result)
        self.assertIsNotNone(strategy.current_call)
    
    async def test_long_put_should_enter(self):
        """Test should_enter logic for LongPutStrategy."""
        # Setup strategy - override technical filters for testing
        strategy = LongPutStrategy(
            underlying_symbol="AAPL",
            account_size=100000.0,
            max_position_size=0.05,
            min_implied_volatility=0.20,
            use_technical_filters=False  # Override for testing
        )
        strategy.broker = self.mock_broker
        
        # Mock option chain response
        self.mock_broker.get_option_chain = AsyncMock(return_value=[
            self.sample_put, self.sample_put_otm
        ])
        
        # Test the should_enter method
        result = await strategy.should_enter(self.market_data)
        
        # Verify that get_option_chain was called correctly
        self.mock_broker.get_option_chain.assert_called_once()
        
        # With technical filters off, we should get an entry
        self.assertTrue(result)
        self.assertIsNotNone(strategy.current_put)
    
    async def test_iron_condor_should_enter(self):
        """Test should_enter logic for IronCondorStrategy."""
        # Setup strategy - override technical filters for testing
        strategy = IronCondorStrategy(
            underlying_symbol="AAPL",
            account_size=100000.0,
            max_position_size=0.10,
            use_technical_filters=False  # Override for testing
        )
        strategy.broker = self.mock_broker
        
        # We need to set up a custom _find_iron_condor_legs method response
        async def mock_find_iron_condor_legs(*args, **kwargs):
            return True, {
                "short_put": self.sample_put_otm,
                "long_put": self.sample_put,  # Further OTM for testing
                "short_call": self.sample_call_otm,
                "long_call": self.sample_call  # Further OTM for testing
            }
        
        # Apply our mock method
        strategy._find_iron_condor_legs = mock_find_iron_condor_legs
        
        # Test the should_enter method
        result = await strategy.should_enter(self.market_data)
        
        # With our mock data, we should get an entry
        self.assertTrue(result)
        self.assertIsNotNone(strategy.short_put)
        self.assertIsNotNone(strategy.long_put)
        self.assertIsNotNone(strategy.short_call)
        self.assertIsNotNone(strategy.long_call)
    
    async def test_covered_call_execute_entry(self):
        """Test execute_entry for CoveredCallStrategy."""
        # Setup strategy
        strategy = CoveredCallStrategy(
            underlying_symbol="AAPL",
            account_size=100000.0,
            max_position_size=0.10
        )
        strategy.broker = self.mock_broker
        strategy.current_call = self.sample_call_otm
        strategy.position_size = 2
        
        # Mock the place_option_order response
        self.mock_broker.place_option_order = AsyncMock(return_value={
            "success": True,
            "order_id": "test_order_123",
            "filled_avg_price": strategy.current_call.bid
        })
        
        # Execute entry
        result = await strategy.execute_entry()
        
        # Verify results
        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("order_id"), "test_order_123")
        self.assertEqual(result.get("symbol"), strategy.current_call.symbol)
        self.assertEqual(result.get("quantity"), strategy.position_size)
    
    async def test_cash_secured_put_execute_entry(self):
        """Test execute_entry for CashSecuredPutStrategy."""
        # Setup strategy
        strategy = CashSecuredPutStrategy(
            underlying_symbol="AAPL",
            account_size=100000.0,
            max_position_size=0.10
        )
        strategy.broker = self.mock_broker
        strategy.current_put = self.sample_put_otm
        strategy.position_size = 2
        
        # Mock the place_option_order response
        self.mock_broker.place_option_order = AsyncMock(return_value={
            "success": True,
            "order_id": "test_order_123",
            "filled_avg_price": strategy.current_put.bid
        })
        
        # Execute entry
        result = await strategy.execute_entry()
        
        # Verify results
        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("order_id"), "test_order_123")
        self.assertEqual(result.get("symbol"), strategy.current_put.symbol)
        self.assertEqual(result.get("quantity"), strategy.position_size)
    
    async def test_long_call_execute_entry(self):
        """Test execute_entry for LongCallStrategy."""
        # Setup strategy
        strategy = LongCallStrategy(
            underlying_symbol="AAPL",
            account_size=100000.0,
            max_position_size=0.05
        )
        strategy.broker = self.mock_broker
        strategy.current_call = self.sample_call
        strategy.position_size = 3
        
        # Mock the place_option_order response
        self.mock_broker.place_option_order = AsyncMock(return_value={
            "success": True,
            "order_id": "test_order_123",
            "filled_avg_price": strategy.current_call.ask
        })
        
        # Execute entry
        result = await strategy.execute_entry()
        
        # Verify results
        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("order_id"), "test_order_123")
        self.assertEqual(result.get("symbol"), strategy.current_call.symbol)
        self.assertEqual(result.get("quantity"), strategy.position_size)
    
    async def test_long_put_execute_entry(self):
        """Test execute_entry for LongPutStrategy."""
        # Setup strategy
        strategy = LongPutStrategy(
            underlying_symbol="AAPL",
            account_size=100000.0,
            max_position_size=0.05
        )
        strategy.broker = self.mock_broker
        strategy.current_put = self.sample_put
        strategy.position_size = 3
        
        # Mock the place_option_order response
        self.mock_broker.place_option_order = AsyncMock(return_value={
            "success": True,
            "order_id": "test_order_123",
            "filled_avg_price": strategy.current_put.ask
        })
        
        # Execute entry
        result = await strategy.execute_entry()
        
        # Verify results
        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("order_id"), "test_order_123")
        self.assertEqual(result.get("symbol"), strategy.current_put.symbol)
        self.assertEqual(result.get("quantity"), strategy.position_size)
    
    async def test_iron_condor_execute_entry(self):
        """Test execute_entry for IronCondorStrategy."""
        # Setup strategy
        strategy = IronCondorStrategy(
            underlying_symbol="AAPL",
            account_size=100000.0,
            max_position_size=0.10
        )
        strategy.broker = self.mock_broker
        strategy.short_put = self.sample_put_otm
        strategy.long_put = self.sample_put
        strategy.short_call = self.sample_call_otm
        strategy.long_call = self.sample_call
        strategy.position_size = 1
        strategy.net_credit = 1.5  # Sample credit received
        
        # Mock the place_option_strategy response
        self.mock_broker.place_option_strategy = AsyncMock(return_value={
            "success": True,
            "order_id": "test_order_123",
            "status": "filled"
        })
        
        # Execute entry
        result = await strategy.execute_entry()
        
        # Verify results
        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("order_id"), "test_order_123")
        self.assertEqual(result.get("underlying"), strategy.underlying_symbol)
        self.assertEqual(result.get("quantity"), strategy.position_size)
    
    async def test_covered_call_should_exit(self):
        """Test should_exit logic for CoveredCallStrategy."""
        # Setup strategy with an active position
        strategy = CoveredCallStrategy(
            underlying_symbol="AAPL",
            account_size=100000.0,
            max_position_size=0.10,
            profit_target_pct=0.50
        )
        strategy.broker = self.mock_broker
        strategy.current_call = self.sample_call_otm
        strategy.entry_premium = 1.5  # We sold at 1.5
        
        # Mock get_current_option_data to return a profitable scenario
        async def mock_get_current_option_data(*args, **kwargs):
            return {
                "bid": 0.75,  # Option price dropped to 0.75 (50% of entry premium)
                "ask": 0.85,
                "last": 0.80,
                "delta": 0.25
            }
        
        strategy._get_current_option_data = mock_get_current_option_data
        
        # Test should_exit
        result = await strategy.should_exit(self.market_data)
        
        # Verify that we should exit (price dropped by 50%)
        self.assertTrue(result)
    
    async def test_iron_condor_get_position_info(self):
        """Test get_position_info for IronCondorStrategy."""
        # Setup strategy with an active position
        strategy = IronCondorStrategy(
            underlying_symbol="AAPL",
            account_size=100000.0,
            max_position_size=0.10
        )
        strategy.broker = self.mock_broker
        strategy.short_put = self.sample_put_otm
        strategy.long_put = self.sample_put
        strategy.short_call = self.sample_call_otm
        strategy.long_call = self.sample_call
        strategy.position_size = 2
        strategy.net_credit = 1.5
        strategy.max_profit = 300.0  # 1.5 * 100 * 2
        strategy.max_loss = 1700.0  # Example
        
        # Mock _get_current_prices to return current option prices
        async def mock_get_current_prices(*args, **kwargs):
            return {
                'short_put': 1.0,
                'long_put': 2.0,
                'short_call': 0.7,
                'long_call': 1.5
            }
        
        strategy._get_current_prices = mock_get_current_prices
        
        # Get position info
        result = await strategy.get_position_info()
        
        # Verify the position information
        self.assertTrue(result.get("has_position"))
        self.assertEqual(result.get("strategy"), "Iron Condor")
        self.assertEqual(result.get("underlying"), strategy.underlying_symbol)
        self.assertEqual(result.get("quantity"), strategy.position_size)
        self.assertIn("current_profit", result)
        self.assertIn("max_profit", result)
        self.assertIn("max_loss", result)


if __name__ == '__main__':
    unittest.main()
