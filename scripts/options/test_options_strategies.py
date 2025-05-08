#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Options Strategies Test Script

This script performs comprehensive testing of all options trading strategies
to validate initialization, entry/exit conditions, and trade execution logic.

Usage:
    python -m scripts.options.test_options_strategies --test-all
    python -m scripts.options.test_options_strategies --strategy COVERED_CALL
"""

import os
import sys
import asyncio
import argparse
import logging
import json
import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, project_root)

from app.services.market_data import MarketDataService
from app.services.options_service import OptionsService
from app.strategies.options.base_options_strategy import BaseOptionsStrategy
from app.strategies.options.covered_call import CoveredCallStrategy
from app.strategies.options.cash_secured_put import CashSecuredPutStrategy
from app.strategies.options.long_call import LongCallStrategy
from app.strategies.options.long_put import LongPutStrategy
from app.strategies.options.iron_condor import IronCondorStrategy
from app.strategies.options.butterfly_spread import ButterflySpreadStrategy
from app.utils.logger_config import setup_logging
from app.core.broker_adapter.alpaca_adapter import AlpacaAdapter
from app.utils.math_utils import black_scholes_call, black_scholes_put

# Configure logging
setup_logging(log_level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments for the options strategy tester."""
    parser = argparse.ArgumentParser(description='Test options trading strategies')
    
    parser.add_argument('--strategy', type=str,
                        choices=['COVERED_CALL', 'CASH_SECURED_PUT', 'LONG_CALL', 
                                'LONG_PUT', 'IRON_CONDOR', 'BUTTERFLY'],
                        help='Specific options strategy to test')
                        
    parser.add_argument('--test-all', action='store_true',
                        help='Run tests for all options strategies')
                        
    parser.add_argument('--test-initialization', action='store_true',
                        help='Test initialization of strategies')
                        
    parser.add_argument('--test-entry-exit', action='store_true',
                        help='Test entry and exit conditions')
                        
    parser.add_argument('--test-execution', action='store_true',
                        help='Test trade execution')
                        
    parser.add_argument('--test-risk-management', action='store_true',
                        help='Test risk management parameters')
                        
    parser.add_argument('--test-edge-cases', action='store_true',
                        help='Test edge cases and error handling')
                        
    parser.add_argument('--output-dir', type=str, default='test_results',
                        help='Directory for test output reports')
                        
    args = parser.parse_args()
    
    # If no specific test category specified, run all tests
    if not any([args.test_initialization, args.test_entry_exit, 
                args.test_execution, args.test_risk_management, 
                args.test_edge_cases]):
        args.test_initialization = True
        args.test_entry_exit = True
        args.test_execution = True
        args.test_risk_management = True
        args.test_edge_cases = True
    
    return args


class OptionsStrategyTestCase(unittest.TestCase):
    """Base test case for options strategies testing."""
    
    async def asyncSetUp(self):
        """Set up test environment."""
        # Create mock broker adapter
        self.broker = MockBrokerAdapter()
        
        # Initialize services
        self.market_data = MarketDataService()
        self.options_service = OptionsService(self.broker)
        
        # Create sample market data
        self.sample_data = self.create_sample_market_data()
    
    def create_sample_market_data(self):
        """Create sample market data for testing."""
        # Create a DataFrame with sample data
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), periods=30, freq='D')
        
        data = {
            'open': np.linspace(100, 120, 30) + np.random.normal(0, 2, 30),
            'high': np.linspace(100, 120, 30) + np.random.normal(0, 4, 30),
            'low': np.linspace(100, 120, 30) - np.random.normal(0, 4, 30),
            'close': np.linspace(100, 120, 30) + np.random.normal(0, 2, 30),
            'volume': np.random.randint(1000000, 5000000, 30)
        }
        
        df = pd.DataFrame(data, index=dates)
        
        # Add technical indicators
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['rsi_14'] = self.calculate_rsi(df['close'], 14)
        df['volatility'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        
        return df
    
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI indicator."""
        delta = prices.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        
        ma_up = up.rolling(window=window).mean()
        ma_down = down.rolling(window=window).mean()
        
        rsi = 100 - (100 / (1 + ma_up / ma_down))
        return rsi
    
    def create_options_chain(self, underlying_price, days_to_expiry=30):
        """Create a sample options chain for testing."""
        # Current price of the underlying
        current_price = underlying_price
        
        # Parameters for options pricing
        interest_rate = 0.03
        volatility = 0.25
        days = days_to_expiry
        time_to_expiry = days / 365.0
        
        # Create strikes around the current price
        strikes = [current_price * (1 + i * 0.025) for i in range(-8, 9)]
        
        # Create options chain
        options_chain = []
        for strike in strikes:
            # Calculate call and put prices using Black-Scholes
            call_price = black_scholes_call(current_price, strike, time_to_expiry, interest_rate, volatility)
            put_price = black_scholes_put(current_price, strike, time_to_expiry, interest_rate, volatility)
            
            # Create call option
            call = {
                'symbol': f'TEST{strike}C{days}',
                'strike': strike,
                'type': 'call',
                'expiration': (datetime.now() + timedelta(days=days)).strftime('%Y-%m-%d'),
                'bid': call_price * 0.95,
                'ask': call_price * 1.05,
                'last': call_price,
                'volume': np.random.randint(100, 1000),
                'open_interest': np.random.randint(500, 5000),
                'implied_volatility': volatility + np.random.normal(0, 0.05)
            }
            
            # Create put option
            put = {
                'symbol': f'TEST{strike}P{days}',
                'strike': strike,
                'type': 'put',
                'expiration': (datetime.now() + timedelta(days=days)).strftime('%Y-%m-%d'),
                'bid': put_price * 0.95,
                'ask': put_price * 1.05,
                'last': put_price,
                'volume': np.random.randint(100, 1000),
                'open_interest': np.random.randint(500, 5000),
                'implied_volatility': volatility + np.random.normal(0, 0.05)
            }
            
            options_chain.extend([call, put])
        
        return options_chain


class MockBrokerAdapter:
    """Mock broker adapter for testing without API access."""
    
    def __init__(self):
        """Initialize mock broker."""
        self.positions = {}
        self.orders = {}
        self.account_value = 100000.0
        self.options_chain = {}
    
    async def connect(self):
        """Mock connection."""
        return True
    
    async def get_account(self):
        """Return mock account information."""
        return {
            'cash': self.account_value * 0.5,
            'equity': self.account_value,
            'buying_power': self.account_value * 2
        }
    
    async def get_options_chain(self, symbol, days_to_expiry=30):
        """Return mock options chain."""
        # Get mock price for the symbol
        price = 100.0  # Default price
        
        # Create a test case instance to use its helper methods
        test_case = OptionsStrategyTestCase()
        options_chain = test_case.create_options_chain(price, days_to_expiry)
        
        return options_chain
    
    async def place_option_order(self, order_params):
        """Place a mock options order."""
        order_id = f"order_{len(self.orders) + 1}"
        self.orders[order_id] = {
            'status': 'filled',
            'params': order_params,
            'filled_at': datetime.now().isoformat()
        }
        
        # If buy order, add to positions
        if order_params.get('side') == 'buy':
            position_id = f"position_{len(self.positions) + 1}"
            self.positions[position_id] = {
                'symbol': order_params.get('symbol'),
                'quantity': order_params.get('quantity', 1),
                'price': order_params.get('price', 0),
                'side': 'long',
                'entry_time': datetime.now().isoformat()
            }
        
        return {
            'id': order_id,
            'status': 'filled',
            'filled_price': order_params.get('price', 0)
        }
    
    async def close_position(self, position_id):
        """Close a mock position."""
        if position_id in self.positions:
            position = self.positions[position_id]
            order_id = f"order_{len(self.orders) + 1}"
            
            self.orders[order_id] = {
                'status': 'filled',
                'position_id': position_id,
                'filled_at': datetime.now().isoformat()
            }
            
            del self.positions[position_id]
            
            return {
                'success': True,
                'order_id': order_id
            }
        else:
            return {
                'success': False,
                'error': 'Position not found'
            }


class TestOptionsStrategies:
    """Test suite for options strategies."""
    
    def __init__(self, args):
        """Initialize test suite with command line arguments."""
        self.args = args
        self.test_results = {
            'initialization': {},
            'entry_exit': {},
            'execution': {},
            'risk_management': {},
            'edge_cases': {},
            'summary': {
                'total_tests': 0,
                'passed': 0,
                'failed': 0
            }
        }
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
    
    def get_strategies_to_test(self):
        """Get the list of strategies to test based on arguments."""
        if self.args.strategy:
            # Test a specific strategy
            strategy_map = {
                'COVERED_CALL': CoveredCallStrategy,
                'CASH_SECURED_PUT': CashSecuredPutStrategy,
                'LONG_CALL': LongCallStrategy,
                'LONG_PUT': LongPutStrategy,
                'IRON_CONDOR': IronCondorStrategy,
                'BUTTERFLY': ButterflySpreadStrategy
            }
            return {self.args.strategy: strategy_map[self.args.strategy]}
        else:
            # Test all strategies
            return {
                'COVERED_CALL': CoveredCallStrategy,
                'CASH_SECURED_PUT': CashSecuredPutStrategy,
                'LONG_CALL': LongCallStrategy,
                'LONG_PUT': LongPutStrategy,
                'IRON_CONDOR': IronCondorStrategy,
                'BUTTERFLY': ButterflySpreadStrategy
            }
    
    async def test_strategy_initialization(self, strategy_name, strategy_class):
        """Test strategy initialization."""
        logger.info(f"Testing initialization for {strategy_name}")
        result = {'status': 'passed', 'errors': []}
        
        try:
            # Create strategy instance with default parameters
            strategy = strategy_class(
                underlying_symbol="AAPL",
                max_position_size=0.05,
                days_to_expiration=30,
                profit_target_pct=0.5,
                stop_loss_pct=0.5
            )
            
            # Check that the strategy has required attributes
            for attr in ['underlying_symbol', 'max_position_size', 'days_to_expiration',
                        'profit_target_pct', 'stop_loss_pct']:
                if not hasattr(strategy, attr):
                    result['status'] = 'failed'
                    result['errors'].append(f"Missing required attribute: {attr}")
            
            # Try with non-default parameters
            strategy = strategy_class(
                underlying_symbol="TSLA",
                max_position_size=0.1,
                days_to_expiration=45,
                profit_target_pct=0.7,
                stop_loss_pct=0.3
            )
            
            # Check that parameters were set correctly
            if strategy.underlying_symbol != "TSLA":
                result['status'] = 'failed'
                result['errors'].append(f"Parameter 'underlying_symbol' not set correctly")
            
            if strategy.max_position_size != 0.1:
                result['status'] = 'failed'
                result['errors'].append(f"Parameter 'max_position_size' not set correctly")
            
        except Exception as e:
            result['status'] = 'failed'
            result['errors'].append(f"Exception during initialization: {str(e)}")
        
        self.test_results['initialization'][strategy_name] = result
        
        if result['status'] == 'passed':
            logger.info(f"✅ Initialization tests PASSED for {strategy_name}")
        else:
            logger.error(f"❌ Initialization tests FAILED for {strategy_name}: {result['errors']}")
        
        return result
    
    async def test_strategy_entry_exit(self, strategy_name, strategy_class):
        """Test strategy entry and exit conditions."""
        logger.info(f"Testing entry/exit conditions for {strategy_name}")
        result = {'status': 'passed', 'errors': []}
        
        try:
            # Create test case
            test_case = OptionsStrategyTestCase()
            await test_case.asyncSetUp()
            
            # Create strategy instance
            strategy = strategy_class(
                underlying_symbol="TEST",
                max_position_size=0.05,
                days_to_expiration=30,
                profit_target_pct=0.5,
                stop_loss_pct=0.5
            )
            strategy.broker_adapter = test_case.broker
            strategy.options_service = test_case.options_service
            
            # Test should_enter method
            try:
                should_enter = await strategy.should_enter(test_case.sample_data)
                # We just check it returned a boolean, not the specific value
                if not isinstance(should_enter, bool):
                    result['status'] = 'failed'
                    result['errors'].append(f"should_enter did not return a boolean")
            except Exception as e:
                result['status'] = 'failed'
                result['errors'].append(f"Exception in should_enter: {str(e)}")
            
            # Test should_exit method (with a dummy position_id)
            try:
                should_exit = await strategy.should_exit("dummy_position", test_case.sample_data)
                # We just check it returned a boolean, not the specific value
                if not isinstance(should_exit, bool):
                    result['status'] = 'failed'
                    result['errors'].append(f"should_exit did not return a boolean")
            except Exception as e:
                result['status'] = 'failed'
                result['errors'].append(f"Exception in should_exit: {str(e)}")
            
        except Exception as e:
            result['status'] = 'failed'
            result['errors'].append(f"Exception during entry/exit testing: {str(e)}")
        
        self.test_results['entry_exit'][strategy_name] = result
        
        if result['status'] == 'passed':
            logger.info(f"✅ Entry/exit tests PASSED for {strategy_name}")
        else:
            logger.error(f"❌ Entry/exit tests FAILED for {strategy_name}: {result['errors']}")
        
        return result
    
    async def test_strategy_execution(self, strategy_name, strategy_class):
        """Test strategy trade execution."""
        logger.info(f"Testing trade execution for {strategy_name}")
        result = {'status': 'passed', 'errors': []}
        
        try:
            # Create test case
            test_case = OptionsStrategyTestCase()
            await test_case.asyncSetUp()
            
            # Create strategy instance
            strategy = strategy_class(
                underlying_symbol="TEST",
                max_position_size=0.05,
                days_to_expiration=30,
                profit_target_pct=0.5,
                stop_loss_pct=0.5
            )
            strategy.broker_adapter = test_case.broker
            strategy.options_service = test_case.options_service
            
            # Test execute_entry method
            try:
                entry_result = await strategy.execute_entry()
                
                # Check that it returned a dictionary with expected keys
                required_keys = ['success']
                for key in required_keys:
                    if key not in entry_result:
                        result['status'] = 'failed'
                        result['errors'].append(f"execute_entry result missing required key: {key}")
            except Exception as e:
                result['status'] = 'failed'
                result['errors'].append(f"Exception in execute_entry: {str(e)}")
            
            # Test execute_exit method (with a dummy position_id)
            try:
                exit_result = await strategy.execute_exit("dummy_position")
                
                # Check that it returned a dictionary with expected keys
                required_keys = ['success']
                for key in required_keys:
                    if key not in exit_result:
                        result['status'] = 'failed'
                        result['errors'].append(f"execute_exit result missing required key: {key}")
            except Exception as e:
                result['status'] = 'failed'
                result['errors'].append(f"Exception in execute_exit: {str(e)}")
            
        except Exception as e:
            result['status'] = 'failed'
            result['errors'].append(f"Exception during execution testing: {str(e)}")
        
        self.test_results['execution'][strategy_name] = result
        
        if result['status'] == 'passed':
            logger.info(f"✅ Execution tests PASSED for {strategy_name}")
        else:
            logger.error(f"❌ Execution tests FAILED for {strategy_name}: {result['errors']}")
        
        return result
    
    async def test_strategy_risk_management(self, strategy_name, strategy_class):
        """Test strategy risk management parameters."""
        logger.info(f"Testing risk management for {strategy_name}")
        result = {'status': 'passed', 'errors': []}
        
        try:
            # Test various position sizes
            for position_size in [0.01, 0.05, 0.1]:
                strategy = strategy_class(
                    underlying_symbol="TEST",
                    max_position_size=position_size,
                    days_to_expiration=30,
                    profit_target_pct=0.5,
                    stop_loss_pct=0.5
                )
                
                if strategy.max_position_size != position_size:
                    result['status'] = 'failed'
                    result['errors'].append(f"Position size not set correctly: {strategy.max_position_size} != {position_size}")
            
            # Test various profit targets and stop losses
            for profit_target, stop_loss in [(0.3, 0.7), (0.5, 0.5), (0.7, 0.3)]:
                strategy = strategy_class(
                    underlying_symbol="TEST",
                    max_position_size=0.05,
                    days_to_expiration=30,
                    profit_target_pct=profit_target,
                    stop_loss_pct=stop_loss
                )
                
                if strategy.profit_target_pct != profit_target:
                    result['status'] = 'failed'
                    result['errors'].append(f"Profit target not set correctly: {strategy.profit_target_pct} != {profit_target}")
                
                if strategy.stop_loss_pct != stop_loss:
                    result['status'] = 'failed'
                    result['errors'].append(f"Stop loss not set correctly: {strategy.stop_loss_pct} != {stop_loss}")
            
        except Exception as e:
            result['status'] = 'failed'
            result['errors'].append(f"Exception during risk management testing: {str(e)}")
        
        self.test_results['risk_management'][strategy_name] = result
        
        if result['status'] == 'passed':
            logger.info(f"✅ Risk management tests PASSED for {strategy_name}")
        else:
            logger.error(f"❌ Risk management tests FAILED for {strategy_name}: {result['errors']}")
        
        return result
    
    async def test_strategy_edge_cases(self, strategy_name, strategy_class):
        """Test strategy edge cases and error handling."""
        logger.info(f"Testing edge cases for {strategy_name}")
        result = {'status': 'passed', 'errors': []}
        
        try:
            # Test with empty market data
            test_case = OptionsStrategyTestCase()
            await test_case.asyncSetUp()
            
            strategy = strategy_class(
                underlying_symbol="TEST",
                max_position_size=0.05,
                days_to_expiration=30,
                profit_target_pct=0.5,
                stop_loss_pct=0.5
            )
            strategy.broker_adapter = test_case.broker
            strategy.options_service = test_case.options_service
            
            try:
                should_enter = await strategy.should_enter(pd.DataFrame())
                # It should handle empty data gracefully
                if not isinstance(should_enter, bool):
                    result['status'] = 'failed'
                    result['errors'].append(f"should_enter did not handle empty data gracefully")
            except Exception as e:
                result['status'] = 'failed'
                result['errors'].append(f"Exception with empty data in should_enter: {str(e)}")
            
            # Test with invalid position_id
            try:
                exit_result = await strategy.execute_exit("non_existent_position")
                
                # Should return a failure with an error message
                if exit_result.get('success', True) or 'error' not in exit_result:
                    result['status'] = 'failed'
                    result['errors'].append(f"execute_exit did not handle invalid position_id gracefully")
            except Exception as e:
                # It's acceptable if this raises an exception, but we should note it
                logger.warning(f"execute_exit raised exception with invalid position_id: {str(e)}")
            
        except Exception as e:
            result['status'] = 'failed'
            result['errors'].append(f"Exception during edge case testing: {str(e)}")
        
        self.test_results['edge_cases'][strategy_name] = result
        
        if result['status'] == 'passed':
            logger.info(f"✅ Edge case tests PASSED for {strategy_name}")
        else:
            logger.error(f"❌ Edge case tests FAILED for {strategy_name}: {result['errors']}")
        
        return result
    
    async def run_tests(self):
        """Run all tests based on command line arguments."""
        strategies = self.get_strategies_to_test()
        
        for strategy_name, strategy_class in strategies.items():
            logger.info(f"Testing strategy: {strategy_name}")
            
            # Run tests based on arguments
            if self.args.test_initialization:
                await self.test_strategy_initialization(strategy_name, strategy_class)
            
            if self.args.test_entry_exit:
                await self.test_strategy_entry_exit(strategy_name, strategy_class)
            
            if self.args.test_execution:
                await self.test_strategy_execution(strategy_name, strategy_class)
            
            if self.args.test_risk_management:
                await self.test_strategy_risk_management(strategy_name, strategy_class)
            
            if self.args.test_edge_cases:
                await self.test_strategy_edge_cases(strategy_name, strategy_class)
        
        # Calculate summary statistics
        self.calculate_summary()
        
        # Save test results
        self.save_results()
    
    def calculate_summary(self):
        """Calculate summary statistics for test results."""
        total_tests = 0
        passed_tests = 0
        
        for category in ['initialization', 'entry_exit', 'execution', 'risk_management', 'edge_cases']:
            if category in self.test_results:
                for strategy_name, result in self.test_results[category].items():
                    total_tests += 1
                    if result['status'] == 'passed':
                        passed_tests += 1
        
        self.test_results['summary'] = {
            'total_tests': total_tests,
            'passed': passed_tests,
            'failed': total_tests - passed_tests,
            'pass_rate': f"{(passed_tests / total_tests * 100) if total_tests > 0 else 0:.2f}%"
        }
    
    def save_results(self):
        """Save test results to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.args.output_dir, f"options_strategy_test_results_{timestamp}.json")
        
        with open(filename, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        logger.info(f"Test results saved to {filename}")
    
    def print_summary(self):
        """Print summary of test results."""
        logger.info("=" * 50)
        logger.info("OPTIONS STRATEGIES TEST SUMMARY")
        logger.info("=" * 50)
        
        summary = self.test_results['summary']
        logger.info(f"Total tests: {summary['total_tests']}")
        logger.info(f"Passed: {summary['passed']}")
        logger.info(f"Failed: {summary['failed']}")
        logger.info(f"Pass rate: {summary['pass_rate']}")
        
        logger.info("-" * 50)
        logger.info("Results by strategy and category:")
        
        # Print results for each strategy by category
        strategies = set()
        for category in ['initialization', 'entry_exit', 'execution', 'risk_management', 'edge_cases']:
            if category in self.test_results:
                strategies.update(self.test_results[category].keys())
        
        for strategy_name in sorted(strategies):
            logger.info(f"\nStrategy: {strategy_name}")
            
            for category in ['initialization', 'entry_exit', 'execution', 'risk_management', 'edge_cases']:
                if category in self.test_results and strategy_name in self.test_results[category]:
                    result = self.test_results[category][strategy_name]
                    status = "✅ PASS" if result['status'] == 'passed' else "❌ FAIL"
                    logger.info(f"  {category.replace('_', ' ').title()}: {status}")
                    
                    if result['status'] == 'failed' and result['errors']:
                        for error in result['errors']:
                            logger.info(f"    - {error}")
        
        logger.info("=" * 50)


async def main():
    """Main entry point for the options strategy tester."""
    args = parse_arguments()
    
    logger.info("Starting options strategies test suite")
    logger.info(f"Test configuration: {vars(args)}")
    
    # Run tests
    test_suite = TestOptionsStrategies(args)
    await test_suite.run_tests()
    test_suite.print_summary()
    
    logger.info("Options strategies testing completed")


if __name__ == '__main__':
    asyncio.run(main())
