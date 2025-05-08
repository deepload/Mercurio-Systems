#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Daily Options Trader Script

This script runs a daily options trading strategy using a combination of
technical indicators and ML predictions to identify opportunities.

Usage:
    python -m scripts.options.run_daily_options_trader --strategy COVERED_CALL --symbols AAPL MSFT GOOG --capital 100000
"""

import os
import sys
import asyncio
import argparse
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any

# Add project root to path
project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, project_root)

from app.services.market_data import MarketDataService
from app.services.options_service import OptionsService
from app.services.trading import TradingService
from app.core.broker_adapter.alpaca_adapter import AlpacaAdapter
from app.strategies.options.covered_call import CoveredCallStrategy
from app.strategies.options.cash_secured_put import CashSecuredPutStrategy
from app.strategies.options.long_call import LongCallStrategy
from app.strategies.options.long_put import LongPutStrategy
from app.strategies.options.iron_condor import IronCondorStrategy
from app.strategies.options.butterfly_spread import ButterflySpreadStrategy
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments for the options trader."""
    parser = argparse.ArgumentParser(description='Run daily options trading strategy')
    
    parser.add_argument('--strategy', type=str, required=True,
                        choices=['COVERED_CALL', 'CASH_SECURED_PUT', 'LONG_CALL', 'LONG_PUT', 
                                'IRON_CONDOR', 'BUTTERFLY', 'MIXED'],
                        help='Options strategy to use')
                        
    parser.add_argument('--symbols', type=str, nargs='+', required=True,
                        help='Symbols to trade options for')
                        
    parser.add_argument('--capital', type=float, default=100000.0,
                        help='Total capital to allocate for options trading')
                        
    parser.add_argument('--allocation-per-trade', type=float, default=0.05,
                        help='Maximum allocation per trade as percentage of capital (0.05 = 5%)')
                        
    parser.add_argument('--max-positions', type=int, default=10,
                        help='Maximum number of positions to hold simultaneously')
                        
    parser.add_argument('--days-to-expiry', type=int, default=30,
                        help='Target days to expiration for options')
                        
    parser.add_argument('--delta-target', type=float, default=0.3,
                        help='Target delta for option selections')
                        
    parser.add_argument('--profit-target', type=float, default=0.5,
                        help='Profit target as percentage of option premium (0.5 = 50%)')
                        
    parser.add_argument('--stop-loss', type=float, default=1.0,
                        help='Stop loss as percentage of option premium (1.0 = 100%)')
                        
    parser.add_argument('--paper-trading', action='store_true',
                        help='Use paper trading mode instead of live trading')
                        
    parser.add_argument('--duration', type=int, default=1,
                        help='Trading duration in days')
                        
    return parser.parse_args()


def get_strategy_class(strategy_name: str):
    """Map strategy name to strategy class."""
    strategy_map = {
        'COVERED_CALL': CoveredCallStrategy,
        'CASH_SECURED_PUT': CashSecuredPutStrategy,
        'LONG_CALL': LongCallStrategy,
        'LONG_PUT': LongPutStrategy,
        'IRON_CONDOR': IronCondorStrategy,
        'BUTTERFLY': ButterflySpreadStrategy
    }
    
    return strategy_map.get(strategy_name)


async def run_options_trader(args):
    """Run options trader with the provided arguments."""
    logger.info(f"Starting options trader with strategy: {args.strategy}")
    logger.info(f"Trading on symbols: {args.symbols}")
    
    # Initialize Alpaca services
    broker_config = {
        "mode": "paper" if args.paper_trading else "live"
    }
    broker = AlpacaAdapter(config=broker_config)
    await broker.connect()
    
    market_data_service = MarketDataService()
    options_service = OptionsService(broker)
    trading_service = TradingService(is_paper=args.paper_trading)
    
    # Get account information
    account = await broker.get_account_info()
    account_value = float(account.get('equity', args.capital))
    logger.info(f"Account value: ${account_value:.2f}")
    
    # Create strategy instances
    strategy_class = get_strategy_class(args.strategy)
    
    if not strategy_class and args.strategy != 'MIXED':
        logger.error(f"Unknown strategy: {args.strategy}")
        return
    
    strategies = []
    
    if args.strategy == 'MIXED':
        # Create a mix of different strategies for diversification
        strategy_allocation = {
            'COVERED_CALL': 0.3,
            'CASH_SECURED_PUT': 0.3,
            'IRON_CONDOR': 0.2,
            'BUTTERFLY': 0.2
        }
        
        for strategy_name, allocation in strategy_allocation.items():
            strategy_class = get_strategy_class(strategy_name)
            for symbol in args.symbols:
                # Adjust allocation based on strategy weight
                position_size = args.allocation_per_trade * allocation * 3
                strategy = strategy_class(
                    underlying_symbol=symbol,
                    max_position_size=position_size,
                    days_to_expiration=args.days_to_expiry,
                    profit_target_pct=args.profit_target,
                    stop_loss_pct=args.stop_loss
                )
                strategy.broker_adapter = broker
                strategy.options_service = options_service
                strategies.append(strategy)
    else:
        # Use single requested strategy
        for symbol in args.symbols:
            strategy = strategy_class(
                underlying_symbol=symbol,
                max_position_size=args.allocation_per_trade,
                days_to_expiration=args.days_to_expiry,
                profit_target_pct=args.profit_target,
                stop_loss_pct=args.stop_loss
            )
            strategy.broker_adapter = broker
            strategy.options_service = options_service
            strategies.append(strategy)
    
    # Run trading loop
    end_time = datetime.now() + timedelta(days=args.duration)
    position_count = 0
    
    logger.info(f"Trading will run until: {end_time}")
    
    try:
        while datetime.now() < end_time and position_count < args.max_positions:
            for strategy in strategies:
                symbol = strategy.underlying_symbol
                
                # Skip if we've reached max positions
                if position_count >= args.max_positions:
                    break
                
                # Get market data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)  # 30 days of historical data
                
                try:
                    market_data = await market_data_service.get_historical_data(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        timeframe="1d"
                    )
                    
                    # Check for entry conditions
                    if await strategy.should_enter(market_data):
                        logger.info(f"Entry signal detected for {symbol} using {strategy.__class__.__name__}")
                        
                        # Execute entry
                        entry_result = await strategy.execute_entry()
                        
                        if entry_result.get('success', False):
                            logger.info(f"Entry executed: {entry_result}")
                            position_count += 1
                        else:
                            logger.warning(f"Entry failed: {entry_result.get('error', 'Unknown error')}")
                    
                    # Check for exit conditions on existing positions
                    # Note: In a real implementation, we'd track positions and their IDs
                    if hasattr(strategy, 'open_positions') and strategy.open_positions:
                        # For strategies that track positions internally like ButterflySpreadStrategy
                        if await strategy.should_exit("dummy_position_id", market_data):
                            logger.info(f"Exit signal detected for {symbol}")
                            
                            # Execute exit
                            exit_result = await strategy.execute_exit("dummy_position_id")
                            
                            if exit_result.get('success', False):
                                logger.info(f"Exit executed: {exit_result}")
                                position_count -= 1
                            else:
                                logger.warning(f"Exit failed: {exit_result.get('error', 'Unknown error')}")
                
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {str(e)}")
            
            # Sleep before next iteration
            await asyncio.sleep(60)  # Check every minute
    
    except KeyboardInterrupt:
        logger.info("Trading interrupted by user")
    
    finally:
        # Close all positions at the end
        logger.info("Closing any remaining positions...")
        await trading_service.close_all_positions()
        
        # Print trading summary
        # In a real implementation, we'd track trades and calculate performance metrics
        logger.info("Trading completed")


if __name__ == '__main__':
    args = parse_arguments()
    asyncio.run(run_options_trader(args))
