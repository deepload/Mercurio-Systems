#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Crypto Options Trader Script

This script runs options trading strategies for cryptocurrency derivatives
using the Alpaca API for crypto market data and options trading.

Usage:
    python -m scripts.options.run_crypto_options_trader --strategy LONG_CALL --symbols BTC ETH --capital 50000
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
from app.services.trading_service import TradingService
from app.core.broker_adapter.alpaca_adapter import AlpacaAdapter
from app.strategies.options.long_call import LongCallStrategy
from app.strategies.options.long_put import LongPutStrategy
from app.strategies.options.iron_condor import IronCondorStrategy
from app.strategies.options.butterfly_spread import ButterflySpreadStrategy
from app.utils.logger_config import setup_logging


# Configure logging
setup_logging(log_level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments for the crypto options trader."""
    parser = argparse.ArgumentParser(description='Run crypto options trading strategy')
    
    parser.add_argument('--strategy', type=str, required=True,
                        choices=['LONG_CALL', 'LONG_PUT', 'IRON_CONDOR', 'BUTTERFLY', 'MIXED'],
                        help='Options strategy to use')
                        
    parser.add_argument('--symbols', type=str, nargs='+', required=True,
                        help='Crypto symbols to trade options for (e.g., BTC ETH)')
                        
    parser.add_argument('--capital', type=float, default=50000.0,
                        help='Total capital to allocate for crypto options trading')
                        
    parser.add_argument('--allocation-per-trade', type=float, default=0.03,
                        help='Maximum allocation per trade as percentage of capital (0.03 = 3%)')
                        
    parser.add_argument('--max-positions', type=int, default=5,
                        help='Maximum number of positions to hold simultaneously')
                        
    parser.add_argument('--days-to-expiry', type=int, default=14,
                        help='Target days to expiration for options')
                        
    parser.add_argument('--delta-target', type=float, default=0.4,
                        help='Target delta for option selections')
                        
    parser.add_argument('--profit-target', type=float, default=0.7,
                        help='Profit target as percentage of option premium (0.7 = 70%)')
                        
    parser.add_argument('--stop-loss', type=float, default=0.5,
                        help='Stop loss as percentage of option premium (0.5 = 50%)')
                        
    parser.add_argument('--volatility-threshold', type=float, default=0.8,
                        help='Minimum implied volatility to enter a trade')
                        
    parser.add_argument('--paper-trading', action='store_true',
                        help='Use paper trading mode instead of live trading')
                        
    parser.add_argument('--duration', type=int, default=1,
                        help='Trading duration in days')
                        
    return parser.parse_args()


def get_strategy_class(strategy_name: str):
    """Map strategy name to strategy class."""
    strategy_map = {
        'LONG_CALL': LongCallStrategy,
        'LONG_PUT': LongPutStrategy,
        'IRON_CONDOR': IronCondorStrategy,
        'BUTTERFLY': ButterflySpreadStrategy
    }
    
    return strategy_map.get(strategy_name)


def format_crypto_symbol(symbol: str) -> str:
    """Format crypto symbol for API compatibility."""
    # Check if symbol already has USD suffix
    if symbol.endswith('USD'):
        return symbol
    # Add USD suffix if needed
    return f"{symbol}USD"


async def run_crypto_options_trader(args):
    """Run crypto options trader with the provided arguments."""
    logger.info(f"Starting crypto options trader with strategy: {args.strategy}")
    
    # Format crypto symbols
    crypto_symbols = [format_crypto_symbol(symbol) for symbol in args.symbols]
    logger.info(f"Trading on crypto symbols: {crypto_symbols}")
    
    # Initialize services
    broker = AlpacaAdapter(is_paper=args.paper_trading)
    await broker.connect()
    
    market_data_service = MarketDataService()
    options_service = OptionsService(broker)
    trading_service = TradingService(broker, is_paper=args.paper_trading)
    
    # Get account information
    account = await broker.get_account()
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
            'LONG_CALL': 0.4,
            'LONG_PUT': 0.3,
            'IRON_CONDOR': 0.15,
            'BUTTERFLY': 0.15
        }
        
        for strategy_name, allocation in strategy_allocation.items():
            strategy_class = get_strategy_class(strategy_name)
            for symbol in crypto_symbols:
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
        for symbol in crypto_symbols:
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
                
                # Get market data for crypto
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)  # 30 days of historical data
                
                try:
                    # For crypto, we might need specific handling in the market data service
                    market_data = await market_data_service.get_historical_data(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        timeframe="1h"  # Use hourly data for crypto due to higher volatility
                    )
                    
                    # Add volatility metric for crypto (not included in raw data)
                    if not market_data.empty and len(market_data) > 20:
                        market_data['returns'] = market_data['close'].pct_change()
                        market_data['volatility'] = market_data['returns'].rolling(window=20).std() * np.sqrt(24)  # Annualized from hourly data
                    
                    # Only enter trades if volatility meets minimum threshold
                    if (market_data.empty or 
                        'volatility' not in market_data.columns or 
                        market_data['volatility'].iloc[-1] < args.volatility_threshold):
                        logger.info(f"Skipping {symbol} due to insufficient volatility")
                        continue
                    
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
                    if hasattr(strategy, 'open_positions') and strategy.open_positions:
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
            await asyncio.sleep(300)  # Check every 5 minutes for crypto
    
    except KeyboardInterrupt:
        logger.info("Trading interrupted by user")
    
    finally:
        # Close all positions at the end
        logger.info("Closing any remaining positions...")
        await trading_service.close_all_positions()
        
        # Print trading summary
        logger.info("Trading completed")


if __name__ == '__main__':
    import numpy as np  # For volatility calculation
    
    args = parse_arguments()
    asyncio.run(run_crypto_options_trader(args))
