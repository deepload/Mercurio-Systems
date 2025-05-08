"""
Unit tests for the options backtester service.

This module tests the functionality of the OptionsBacktester service, which
enables backtesting of options strategies using historical market data.
"""

import unittest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import tempfile
import os
import json

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from app.services.options_backtester import OptionsBacktester, MockOptionsBacktestBroker
from app.services.market_data import MarketDataService
from app.strategies.options.covered_call import CoveredCallStrategy
from app.strategies.options.long_call import LongCallStrategy
from app.strategies.options.iron_condor import IronCondorStrategy
from app.core.models.option import OptionContract, OptionType


class TestOptionsBacktester(unittest.TestCase):
    """Test suite for options backtesting service."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create mock market data service
        self.mock_market_data_service = AsyncMock(spec=MarketDataService)
        
        # Create temporary directory for test output
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize backtester with mock service
        self.backtester = OptionsBacktester(
            market_data_service=self.mock_market_data_service,
            initial_capital=100000.0,
            data_start_date="2023-01-01",
            data_end_date="2023-01-31",
            output_directory=self.temp_dir
        )
        
        # Create sample market data
        dates = pd.date_range(start='2023-01-01', end='2023-01-31')
        self.sample_data = pd.DataFrame({
            'open': np.random.uniform(90, 110, len(dates)),
            'high': np.random.uniform(95, 115, len(dates)),
            'low': np.random.uniform(85, 105, len(dates)),
            'close': np.random.uniform(90, 110, len(dates)),
            'volume': np.random.randint(1000, 100000, len(dates))
        }, index=dates)
        
        # Add technical indicators
        self.sample_data['sma20'] = self.sample_data['close'].rolling(window=20).mean()
        
        # Set up mock response for get_historical_data
        self.mock_market_data_service.get_historical_data = AsyncMock(
            return_value=self.sample_data
        )
        
    def tearDown(self):
        """Clean up test fixtures after each test method."""
        # Remove temporary directory and its contents
        for filename in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, filename))
        os.rmdir(self.temp_dir)
        
    def run_async_test(self, coroutine):
        """Helper function to run async tests."""
        return asyncio.run(coroutine)
    
    async def test_backtester_initialization(self):
        """Test initialization of OptionsBacktester."""
        self.assertEqual(self.backtester.initial_capital, 100000.0)
        self.assertEqual(self.backtester.data_start_date, "2023-01-01")
        self.assertEqual(self.backtester.data_end_date, "2023-01-31")
        self.assertEqual(self.backtester.output_directory, self.temp_dir)
        self.assertEqual(self.backtester.equity, 100000.0)
        self.assertEqual(self.backtester.positions, [])
        self.assertEqual(self.backtester.trade_history, [])
        self.assertEqual(self.backtester.equity_curve, [])
        
    async def test_align_dates(self):
        """Test date alignment functionality."""
        # Create sample data with different date ranges
        data1 = pd.DataFrame({
            'close': [100, 101, 102]
        }, index=pd.date_range(start='2023-01-01', periods=3))
        
        data2 = pd.DataFrame({
            'close': [200, 201, 202, 203]
        }, index=pd.date_range(start='2023-01-02', periods=4))
        
        data_by_symbol = {
            'AAPL': data1,
            'MSFT': data2
        }
        
        # Test date alignment
        common_dates = self.backtester._align_dates(data_by_symbol)
        
        # Should only have dates that appear in both datasets
        expected_dates = pd.date_range(start='2023-01-02', periods=2)
        self.assertEqual(len(common_dates), 2)
        self.assertEqual(common_dates[0].strftime('%Y-%m-%d'), expected_dates[0].strftime('%Y-%m-%d'))
        self.assertEqual(common_dates[1].strftime('%Y-%m-%d'), expected_dates[1].strftime('%Y-%m-%d'))
        
    async def test_generate_simulated_options_chain(self):
        """Test generation of simulated options chains."""
        # Set up test data
        symbol = "AAPL"
        date = pd.Timestamp('2023-01-15')
        
        # Make sure the symbol exists in latest_data
        self.backtester.latest_data = {
            symbol: self.sample_data.copy()
        }
        
        # Generate options chain
        self.backtester._generate_simulated_options_chain(symbol, date)
        
        # Verify options chain was created
        self.assertIn(symbol, self.backtester.simulated_options_chains)
        
        date_str = date.strftime("%Y-%m-%d")
        self.assertIn(date_str, self.backtester.simulated_options_chains[symbol])
        
        # Check that we have both calls and puts in the chain
        chain = self.backtester.simulated_options_chains[symbol][date_str]
        call_options = [o for o in chain if o.option_type == OptionType.CALL]
        put_options = [o for o in chain if o.option_type == OptionType.PUT]
        
        self.assertTrue(len(call_options) > 0)
        self.assertTrue(len(put_options) > 0)
        
        # Check that options have all required attributes
        sample_option = chain[0]
        self.assertIsNotNone(sample_option.symbol)
        self.assertIsNotNone(sample_option.underlying)
        self.assertIsNotNone(sample_option.strike)
        self.assertIsNotNone(sample_option.expiry_date)
        self.assertIsNotNone(sample_option.bid)
        self.assertIsNotNone(sample_option.ask)
        self.assertIsNotNone(sample_option.delta)
        
    async def test_mock_broker_get_option_chain(self):
        """Test the MockOptionsBacktestBroker's get_option_chain method."""
        # Set up test data
        symbol = "AAPL"
        date = pd.Timestamp('2023-01-15')
        date_str = date.strftime("%Y-%m-%d")
        
        # Set up mock options chain
        call_option = OptionContract(
            symbol=f"{symbol}_{date_str}_C100",
            underlying=symbol,
            option_type=OptionType.CALL,
            strike=100.0,
            expiry_date=date_str,
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
        
        put_option = OptionContract(
            symbol=f"{symbol}_{date_str}_P100",
            underlying=symbol,
            option_type=OptionType.PUT,
            strike=100.0,
            expiry_date=date_str,
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
        
        # Add options to backtester's simulated chains
        self.backtester.simulated_options_chains = {
            symbol: {
                date_str: [call_option, put_option]
            }
        }
        
        # Set current date in backtester
        self.backtester.current_date = date
        
        # Create mock broker
        mock_broker = MockOptionsBacktestBroker(self.backtester)
        
        # Test get_option_chain with no filters
        options = await mock_broker.get_option_chain(symbol)
        self.assertEqual(len(options), 2)
        
        # Test get_option_chain with option_type filter
        call_options = await mock_broker.get_option_chain(symbol, option_type=OptionType.CALL)
        self.assertEqual(len(call_options), 1)
        self.assertEqual(call_options[0].option_type, OptionType.CALL)
        
        put_options = await mock_broker.get_option_chain(symbol, option_type=OptionType.PUT)
        self.assertEqual(len(put_options), 1)
        self.assertEqual(put_options[0].option_type, OptionType.PUT)
        
    @patch('app.strategies.options.covered_call.CoveredCallStrategy.should_enter')
    @patch('app.strategies.options.covered_call.CoveredCallStrategy.execute_entry')
    @patch('app.strategies.options.covered_call.CoveredCallStrategy.should_exit')
    @patch('app.strategies.options.covered_call.CoveredCallStrategy.execute_exit')
    async def test_run_backtest(self, mock_execute_exit, mock_should_exit, mock_execute_entry, mock_should_enter):
        """Test running a full backtest with a strategy."""
        # Configure mocks
        mock_should_enter.return_value = True
        mock_execute_entry.return_value = {
            "success": True,
            "symbol": "AAPL_20230215_C100",
            "quantity": 1,
            "premium": 3.5,
            "strike": 100.0,
            "expiry": "2023-02-15"
        }
        
        # First call to should_exit returns False, second call returns True
        mock_should_exit.side_effect = [False, True]
        
        mock_execute_exit.return_value = {
            "success": True,
            "exit_premium": 1.75,
            "profit_loss": 175.0,
            "profit_loss_pct": 50.0
        }
        
        # Run the backtest
        results = await self.backtester.run_backtest(
            strategy_class=CoveredCallStrategy,
            symbols=["AAPL"],
            strategy_params={
                "max_position_size": 0.10,
                "profit_target_pct": 0.50,
                "stop_loss_pct": 0.50
            },
            timeframe='1d',
            report_name="test_backtest"
        )
        
        # Verify backtest was run and produced a report
        self.assertTrue(results.get("success", False))
        self.assertIn("trades", results)
        self.assertIn("equity_curve", results)
        
        # Check that a report file was created
        report_files = [f for f in os.listdir(self.temp_dir) if f.endswith(".json")]
        self.assertTrue(len(report_files) > 0)
        
        # Load the report and check its contents
        with open(os.path.join(self.temp_dir, report_files[0]), 'r') as f:
            report = json.load(f)
        
        self.assertEqual(report["strategy"], "CoveredCallStrategy")
        self.assertIn("total_return", report)
        self.assertIn("win_rate", report)
        
    async def test_generate_backtest_report(self):
        """Test generation of backtest report with performance metrics."""
        # Set up some test trading data
        self.backtester.initial_capital = 100000.0
        self.backtester.equity = 105000.0  # 5% return
        
        # Add some trade history
        self.backtester.trade_history = [
            {
                "symbol": "AAPL",
                "entry_date": datetime(2023, 1, 5),
                "exit_date": datetime(2023, 1, 10),
                "profit_loss": 500.0
            },
            {
                "symbol": "MSFT",
                "entry_date": datetime(2023, 1, 7),
                "exit_date": datetime(2023, 1, 15),
                "profit_loss": -200.0
            },
            {
                "symbol": "GOOGL",
                "entry_date": datetime(2023, 1, 12),
                "exit_date": datetime(2023, 1, 20),
                "profit_loss": 4700.0
            }
        ]
        
        # Add equity curve data
        self.backtester.equity_curve = [
            {"date": datetime(2023, 1, 1), "equity": 100000.0},
            {"date": datetime(2023, 1, 10), "equity": 100500.0},
            {"date": datetime(2023, 1, 15), "equity": 100300.0},
            {"date": datetime(2023, 1, 20), "equity": 105000.0}
        ]
        
        # Generate report
        report = self.backtester._generate_backtest_report(
            strategy_name="CoveredCallStrategy",
            symbols=["AAPL", "MSFT", "GOOGL"],
            strategy_params={"max_position_size": 0.10},
            duration_seconds=10.5
        )
        
        # Verify report content
        self.assertEqual(report["strategy"], "CoveredCallStrategy")
        self.assertEqual(report["initial_capital"], 100000.0)
        self.assertEqual(report["final_equity"], 105000.0)
        self.assertEqual(report["total_return"], 5000.0)
        self.assertEqual(report["total_return_pct"], 5.0)
        self.assertEqual(report["total_trades"], 3)
        self.assertEqual(report["profitable_trades"], 2)
        self.assertEqual(report["losing_trades"], 1)
        self.assertAlmostEqual(report["win_rate"], 2/3)
        self.assertEqual(report["execution_time_seconds"], 10.5)
        
        # Verify lists were included
        self.assertEqual(len(report["trades"]), 3)
        self.assertEqual(len(report["equity_curve"]), 4)
        
    def test_mock_broker_place_option_order(self):
        """Test the MockOptionsBacktestBroker's place_option_order method."""
        # Set up test data
        symbol = "AAPL"
        date = pd.Timestamp('2023-01-15')
        date_str = date.strftime("%Y-%m-%d")
        
        # Create option contract
        option = OptionContract(
            symbol=f"{symbol}_{date_str}_C100",
            underlying=symbol,
            option_type=OptionType.CALL,
            strike=100.0,
            expiry_date=date_str,
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
        
        # Add option to backtester's simulated chains
        self.backtester.simulated_options_chains = {
            symbol: {
                date_str: [option]
            }
        }
        
        # Set current date in backtester
        self.backtester.current_date = date
        
        # Create mock broker
        mock_broker = MockOptionsBacktestBroker(self.backtester)
        
        # Test buy order
        async def test_buy_order():
            result = await mock_broker.place_option_order(
                option_symbol=option.symbol,
                qty=2,
                side="buy",
                order_type="limit",
                limit_price=3.7
            )
            return result
            
        buy_result = self.run_async_test(test_buy_order())
        
        self.assertTrue(buy_result.get("success"))
        self.assertEqual(buy_result.get("symbol"), option.symbol)
        self.assertEqual(buy_result.get("qty"), 2)
        self.assertEqual(buy_result.get("side"), "buy")
        
        # Test sell order
        async def test_sell_order():
            result = await mock_broker.place_option_order(
                option_symbol=option.symbol,
                qty=1,
                side="sell",
                order_type="market"
            )
            return result
            
        sell_result = self.run_async_test(test_sell_order())
        
        self.assertTrue(sell_result.get("success"))
        self.assertEqual(sell_result.get("symbol"), option.symbol)
        self.assertEqual(sell_result.get("qty"), 1)
        self.assertEqual(sell_result.get("side"), "sell")
        
        # Test order for non-existent option
        async def test_invalid_order():
            result = await mock_broker.place_option_order(
                option_symbol="INVALID_OPTION",
                qty=1,
                side="buy",
                order_type="market"
            )
            return result
            
        invalid_result = self.run_async_test(test_invalid_order())
        
        self.assertFalse(invalid_result.get("success"))
        self.assertIn("error", invalid_result)


if __name__ == '__main__':
    unittest.main()
