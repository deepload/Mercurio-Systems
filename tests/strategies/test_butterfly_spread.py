"""
Unit tests for the Butterfly Spread options strategy.

This module tests the functionality of the ButterflySpreadStrategy class,
including initialization, entry and exit conditions, and trade execution.
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

from app.strategies.options.butterfly_spread import ButterflySpreadStrategy
from app.core.models.option import OptionContract, OptionType


class TestButterflySpreadStrategy(unittest.TestCase):
    """Test suite for the Butterfly Spread options strategy."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.symbol = "AAPL"
        
        # Create strategy instance
        self.strategy = ButterflySpreadStrategy(
            underlying_symbol=self.symbol,
            max_position_size=0.1,
            option_type="call",
            delta_target=0.30,
            wing_width_pct=0.05,
            days_to_expiration=30,
            max_days_to_hold=21,
            profit_target_pct=0.50,
            stop_loss_pct=0.50,
            use_technical_filters=True
        )
        
        # Create mock services and adapter
        self.strategy.broker_adapter = AsyncMock()
        self.strategy.options_service = AsyncMock()
        
        # Create sample market data
        dates = pd.date_range(start='2023-01-01', end='2023-01-31')
        self.sample_data = pd.DataFrame({
            'open': np.random.uniform(95, 105, len(dates)),
            'high': np.random.uniform(100, 110, len(dates)),
            'low': np.random.uniform(90, 100, len(dates)),
            'close': np.random.uniform(95, 105, len(dates)),
            'volume': np.random.randint(1000, 100000, len(dates))
        }, index=dates)
        
        # Add technical indicators
        self.sample_data['sma20'] = self.sample_data['close'].rolling(window=20).mean()
        self.sample_data['sma50'] = self.sample_data['close'].rolling(window=50).mean()
        self.sample_data['returns'] = np.log(self.sample_data['close'] / self.sample_data['close'].shift(1))
        self.sample_data['hist_vol'] = self.sample_data['returns'].rolling(window=20).std() * np.sqrt(252)
        
        # Create sample options chain
        self.sample_options = []
        current_price = 100.0
        
        # Generate option strikes around current price
        for strike in range(85, 116, 5):
            # Call option
            call = OptionContract(
                symbol=f"{self.symbol}_20230215_C{strike}",
                underlying=self.symbol,
                underlying_price=current_price,
                option_type=OptionType.CALL,
                strike=float(strike),
                expiry_date="2023-02-15",
                bid=max(0, round(current_price - strike + 5, 2)),
                ask=max(0, round(current_price - strike + 7, 2)),
                last=max(0, round(current_price - strike + 6, 2)),
                volume=1000,
                open_interest=5000,
                implied_volatility=0.30,
                delta=max(0, min(1, round(1 - (strike - current_price) / 20, 2))),
                gamma=0.05,
                theta=-0.10,
                vega=0.15,
                rho=0.05
            )
            
            # Put option
            put = OptionContract(
                symbol=f"{self.symbol}_20230215_P{strike}",
                underlying=self.symbol,
                underlying_price=current_price,
                option_type=OptionType.PUT,
                strike=float(strike),
                expiry_date="2023-02-15",
                bid=max(0, round(strike - current_price + 5, 2)),
                ask=max(0, round(strike - current_price + 7, 2)),
                last=max(0, round(strike - current_price + 6, 2)),
                volume=800,
                open_interest=4000,
                implied_volatility=0.28,
                delta=min(0, max(-1, round(-1 + (strike - current_price) / 20, 2))),
                gamma=0.05,
                theta=-0.10,
                vega=0.15,
                rho=-0.05
            )
            
            self.sample_options.extend([call, put])
        
    def run_async_test(self, coroutine):
        """Helper function to run async tests."""
        return asyncio.run(coroutine)
    
    def test_initialization(self):
        """Test initialization of the ButterflySpreadStrategy class."""
        self.assertEqual(self.strategy.underlying_symbol, self.symbol)
        self.assertEqual(self.strategy.max_position_size, 0.1)
        self.assertEqual(self.strategy.option_type, OptionType.CALL)
        self.assertEqual(self.strategy.delta_target, 0.30)
        self.assertEqual(self.strategy.wing_width_pct, 0.05)
        self.assertEqual(self.strategy.days_to_expiration, 30)
        self.assertEqual(self.strategy.max_days_to_hold, 21)
        self.assertEqual(self.strategy.profit_target_pct, 0.50)
        self.assertEqual(self.strategy.stop_loss_pct, 0.50)
        self.assertEqual(self.strategy.use_technical_filters, True)
        
    async def test_check_technical_filters(self):
        """Test technical filters functionality."""
        # Should return True with valid data
        result = await self.strategy._check_technical_filters(self.sample_data)
        self.assertTrue(result)
        
        # Modify data to test filters
        modified_data = self.sample_data.copy()
        
        # Test with very high volatility
        modified_data['hist_vol'] = 0.60  # High volatility
        result = await self.strategy._check_technical_filters(modified_data)
        self.assertFalse(result)
        
        # Test with very low volatility
        modified_data['hist_vol'] = 0.10  # Low volatility
        result = await self.strategy._check_technical_filters(modified_data)
        self.assertFalse(result)
        
        # Test with price far from SMAs
        modified_data = self.sample_data.copy()
        modified_data.iloc[-1, modified_data.columns.get_loc('close')] = 150  # Far above SMAs
        result = await self.strategy._check_technical_filters(modified_data)
        self.assertFalse(result)
        
    def test_find_closest_strike(self):
        """Test finding the closest strike price."""
        # Setup
        options = self.sample_options
        
        # Test exact match
        closest = self.strategy._find_closest_strike(options, 90.0)
        self.assertEqual(closest, 90.0)
        
        # Test in-between strikes
        closest = self.strategy._find_closest_strike(options, 92.5)
        self.assertEqual(closest, 90.0)  # Rounds down in our sample data
        
        closest = self.strategy._find_closest_strike(options, 97.5)
        self.assertEqual(closest, 100.0)  # Rounds up in our sample data
        
        # Test edge cases
        closest = self.strategy._find_closest_strike(options, 80.0)
        self.assertEqual(closest, 85.0)  # Lower bound
        
        closest = self.strategy._find_closest_strike(options, 120.0)
        self.assertEqual(closest, 115.0)  # Upper bound
        
        # Test with empty list
        closest = self.strategy._find_closest_strike([], 100.0)
        self.assertIsNone(closest)
    
    def test_find_strike_by_delta(self):
        """Test finding strike by delta."""
        # Setup
        options = self.sample_options
        current_price = 100.0
        
        # Test with valid delta
        strike = self.strategy._find_strike_by_delta(options, 0.30, current_price)
        # In our sample data, delta of 0.30 would be around strike 105-110
        self.assertTrue(105 <= strike <= 110)
        
        # Test with delta of 0.50 (ATM)
        strike = self.strategy._find_strike_by_delta(options, 0.50, current_price)
        # In our sample data, delta of 0.50 would be around strike 100
        self.assertTrue(95 <= strike <= 105)
        
        # Test with invalid delta (fallback to ATM)
        for option in options:
            option.delta = None
        
        strike = self.strategy._find_strike_by_delta(options, 0.30, current_price)
        self.assertEqual(strike, 100.0)  # Should fallback to closest to current price
        
    def test_find_atm_options(self):
        """Test finding at-the-money options."""
        # Setup
        options = self.sample_options
        
        # Test finding ATM options
        atm_options = self.strategy._find_atm_options(options)
        
        # Should return options with strike closest to current price (100)
        self.assertTrue(len(atm_options) > 0)
        for option in atm_options:
            self.assertAlmostEqual(option.strike, 100.0)
    
    @patch('app.strategies.options.butterfly_spread.ButterflySpreadStrategy._get_current_price')
    @patch('app.strategies.options.butterfly_spread.ButterflySpreadStrategy._check_technical_filters')
    async def test_should_enter(self, mock_check_filters, mock_get_price):
        """Test entry conditions."""
        # Setup
        mock_check_filters.return_value = True
        mock_get_price.return_value = 100.0
        
        # Mock options service
        self.strategy.options_service.get_options_chain.return_value = self.sample_options
        
        # Test when no positions open and conditions are good
        result = await self.strategy.should_enter(self.sample_data)
        self.assertTrue(result)
        
        # Test when position already exists
        self.strategy.open_positions = {"entry_date": datetime.now()}
        result = await self.strategy.should_enter(self.sample_data)
        self.assertFalse(result)
        self.strategy.open_positions = {}
        
        # Test when technical filters fail
        mock_check_filters.return_value = False
        result = await self.strategy.should_enter(self.sample_data)
        self.assertFalse(result)
        
        # Test with extreme IV
        mock_check_filters.return_value = True
        # Set all options to have high IV
        for option in self.sample_options:
            option.implied_volatility = 0.70
            
        result = await self.strategy.should_enter(self.sample_data)
        self.assertFalse(result)
    
    @patch('app.strategies.options.butterfly_spread.ButterflySpreadStrategy._get_current_price')
    @patch('app.strategies.options.butterfly_spread.ButterflySpreadStrategy._get_account_value')
    async def test_execute_entry(self, mock_get_account, mock_get_price):
        """Test trade entry execution."""
        # Setup
        mock_get_price.return_value = 100.0
        mock_get_account.return_value = 100000.0
        
        # Mock options service
        self.strategy.options_service.get_options_chain.return_value = self.sample_options
        
        # Mock broker adapter responses
        self.strategy.broker_adapter.place_option_order.return_value = {
            "success": True, 
            "order_id": "test123"
        }
        
        # Execute entry
        result = await self.strategy.execute_entry()
        
        # Verify results
        self.assertTrue(result["success"])
        self.assertEqual(result["strategy"], "Butterfly Spread")
        self.assertEqual(result["option_type"], "call")
        
        # Verify strike prices properly calculated
        self.assertLess(result["lower_strike"], result["middle_strike"])
        self.assertLess(result["middle_strike"], result["upper_strike"])
        
        # Verify wing width approximately 5% of underlying price
        wing_width = result["upper_strike"] - result["middle_strike"]
        self.assertAlmostEqual(wing_width / 100.0, 0.05, delta=0.03)  # Allow small variation
        
        # Verify broker calls were made
        self.assertEqual(self.strategy.broker_adapter.place_option_order.call_count, 3)
        
        # Verify position was recorded
        self.assertIn("entry_date", self.strategy.open_positions)
        self.assertIn("expiry_date", self.strategy.open_positions)
        self.assertIn("initial_debit", self.strategy.open_positions)
        self.assertIn("max_profit", self.strategy.open_positions)
        self.assertIn("max_loss", self.strategy.open_positions)
    
    @patch('app.strategies.options.butterfly_spread.ButterflySpreadStrategy._get_current_price')
    async def test_should_exit(self, mock_get_price):
        """Test exit conditions."""
        # Setup
        mock_get_price.return_value = 100.0
        
        # Create an open position
        self.strategy.open_positions = {
            "entry_date": datetime.now() - timedelta(days=10),
            "expiry_date": (datetime.now() + timedelta(days=20)).strftime("%Y-%m-%d"),
            "num_spreads": 1,
            "lower_strike": 95.0,
            "middle_strike": 100.0,
            "upper_strike": 105.0,
            "initial_debit": 200.0,
            "lower_option": f"{self.symbol}_20230215_C95",
            "middle_option": f"{self.symbol}_20230215_C100",
            "upper_option": f"{self.symbol}_20230215_C105",
            "current_price": 100.0,
            "max_profit": 300.0,
            "max_loss": 200.0
        }
        
        # Create option details
        lower_option = OptionContract(
            symbol=self.strategy.open_positions["lower_option"],
            underlying=self.symbol,
            option_type=OptionType.CALL,
            strike=95.0,
            expiry_date="2023-02-15",
            bid=8.0,
            ask=8.5,
            implied_volatility=0.30,
            delta=0.70
        )
        
        middle_option = OptionContract(
            symbol=self.strategy.open_positions["middle_option"],
            underlying=self.symbol,
            option_type=OptionType.CALL,
            strike=100.0,
            expiry_date="2023-02-15",
            bid=4.0,
            ask=4.5,
            implied_volatility=0.30,
            delta=0.50
        )
        
        upper_option = OptionContract(
            symbol=self.strategy.open_positions["upper_option"],
            underlying=self.symbol,
            option_type=OptionType.CALL,
            strike=105.0,
            expiry_date="2023-02-15",
            bid=2.0,
            ask=2.5,
            implied_volatility=0.30,
            delta=0.30
        )
        
        # Mock options service response
        self.strategy.options_service.get_option_details = AsyncMock()
        self.strategy.options_service.get_option_details.side_effect = lambda symbol: {
            self.strategy.open_positions["lower_option"]: lower_option,
            self.strategy.open_positions["middle_option"]: middle_option,
            self.strategy.open_positions["upper_option"]: upper_option
        }.get(symbol)
        
        # Test 1: Normal conditions - should not exit
        result = await self.strategy.should_exit(self.sample_data)
        self.assertFalse(result)
        
        # Test 2: Profit target reached
        # Modify option prices to create profit scenario (lower + upper - 2*middle > initial debit * profit target)
        upper_option.bid = 6.0  # Increase value
        result = await self.strategy.should_exit(self.sample_data)
        self.assertTrue(result)
        
        # Reset for next test
        upper_option.bid = 2.0
        
        # Test 3: Stop loss hit
        # Modify option prices to create loss scenario
        lower_option.bid = 2.0  # Decrease value
        middle_option.ask = 5.5  # Increase value (cost to close short)
        result = await self.strategy.should_exit(self.sample_data)
        self.assertTrue(result)
        
        # Reset for next test
        lower_option.bid = 8.0
        middle_option.ask = 4.5
        
        # Test 4: Max days held
        self.strategy.open_positions["entry_date"] = datetime.now() - timedelta(days=25)
        result = await self.strategy.should_exit(self.sample_data)
        self.assertTrue(result)
        
        # Reset for next test
        self.strategy.open_positions["entry_date"] = datetime.now() - timedelta(days=10)
        
        # Test 5: Near expiration
        self.strategy.open_positions["expiry_date"] = (datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d")
        result = await self.strategy.should_exit(self.sample_data)
        self.assertTrue(result)
        
        # Reset for next test
        self.strategy.open_positions["expiry_date"] = (datetime.now() + timedelta(days=20)).strftime("%Y-%m-%d")
        
        # Test 6: Price moved far beyond wings
        mock_get_price.return_value = 85.0  # Well below lower strike
        result = await self.strategy.should_exit(self.sample_data)
        self.assertTrue(result)
    
    @patch('app.strategies.options.butterfly_spread.ButterflySpreadStrategy._get_current_price')
    async def test_execute_exit(self, mock_get_price):
        """Test trade exit execution."""
        # Setup
        mock_get_price.return_value = 100.0
        
        # Create an open position
        self.strategy.open_positions = {
            "entry_date": datetime.now() - timedelta(days=10),
            "expiry_date": (datetime.now() + timedelta(days=20)).strftime("%Y-%m-%d"),
            "num_spreads": 1,
            "lower_strike": 95.0,
            "middle_strike": 100.0,
            "upper_strike": 105.0,
            "initial_debit": 200.0,
            "lower_option": f"{self.symbol}_20230215_C95",
            "middle_option": f"{self.symbol}_20230215_C100",
            "upper_option": f"{self.symbol}_20230215_C105",
            "current_price": 100.0,
            "max_profit": 300.0,
            "max_loss": 200.0
        }
        
        # Create option details
        lower_option = OptionContract(
            symbol=self.strategy.open_positions["lower_option"],
            underlying=self.symbol,
            option_type=OptionType.CALL,
            strike=95.0,
            expiry_date="2023-02-15",
            bid=8.0,
            ask=8.5,
            implied_volatility=0.30,
            delta=0.70
        )
        
        middle_option = OptionContract(
            symbol=self.strategy.open_positions["middle_option"],
            underlying=self.symbol,
            option_type=OptionType.CALL,
            strike=100.0,
            expiry_date="2023-02-15",
            bid=4.0,
            ask=4.5,
            implied_volatility=0.30,
            delta=0.50
        )
        
        upper_option = OptionContract(
            symbol=self.strategy.open_positions["upper_option"],
            underlying=self.symbol,
            option_type=OptionType.CALL,
            strike=105.0,
            expiry_date="2023-02-15",
            bid=2.0,
            ask=2.5,
            implied_volatility=0.30,
            delta=0.30
        )
        
        # Mock options service response
        self.strategy.options_service.get_option_details = AsyncMock()
        self.strategy.options_service.get_option_details.side_effect = lambda symbol: {
            self.strategy.open_positions["lower_option"]: lower_option,
            self.strategy.open_positions["middle_option"]: middle_option,
            self.strategy.open_positions["upper_option"]: upper_option
        }.get(symbol)
        
        # Mock broker adapter response
        self.strategy.broker_adapter.place_option_order.return_value = {
            "success": True, 
            "order_id": "test123"
        }
        
        # Execute exit
        result = await self.strategy.execute_exit()
        
        # Verify results
        self.assertTrue(result["success"])
        self.assertIn("exit_date", result)
        self.assertIn("profit_loss", result)
        self.assertIn("profit_loss_pct", result)
        
        # Verify broker calls were made
        self.assertEqual(self.strategy.broker_adapter.place_option_order.call_count, 3)
        
        # Verify position was cleared
        self.assertEqual(self.strategy.open_positions, {})
        self.assertEqual(self.strategy.initial_debit, 0)


if __name__ == '__main__':
    unittest.main()
