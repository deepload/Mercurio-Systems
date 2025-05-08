#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
High-Volume Options Trader Script

This script executes options strategies across a large number of symbols (up to 50)
simultaneously for active day trading. It optimizes for execution speed using
threading and memory-efficient data processing.

Usage:
    python -m scripts.options.run_high_volume_options_trader --strategy COVERED_CALL --max-symbols 50 --use-threads --use-custom-symbols
"""

import os
import sys
import asyncio
import argparse
import logging
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Set
from concurrent.futures import ThreadPoolExecutor
import threading

# Add project root to path
project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, project_root)

from app.services.market_data import MarketDataService
from app.services.options_service import OptionsService
from app.services.trading import TradingService
from app.core.broker_adapter.alpaca_adapter import AlpacaAdapter
from app.strategies.options.covered_call import CoveredCallStrategy
from app.strategies.options.cash_secured_put import CashSecuredPutStrategy
from app.strategies.options.iron_condor import IronCondorStrategy
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Thread-local storage for service instances
thread_local = threading.local()


def parse_arguments():
    """Parse command line arguments for the high-volume options trader."""
    parser = argparse.ArgumentParser(description='Run high-volume options trading strategy')
    
    parser.add_argument('--strategy', type=str, required=True,
                        choices=['COVERED_CALL', 'CASH_SECURED_PUT', 'IRON_CONDOR'],
                        help='Options strategy to use')
                        
    parser.add_argument('--symbols', type=str, nargs='+',
                        help='Specific symbols to trade options for (optional)')
                        
    parser.add_argument('--filter', type=str, choices=['top_volume', 'top_gainers', 'top_losers', 'most_volatile'],
                        default='top_volume',
                        help='Filter method for selecting stocks')
                        
    parser.add_argument('--max-symbols', type=int, default=50,
                        help='Maximum number of symbols to trade')
                        
    parser.add_argument('--capital', type=float, default=100000.0,
                        help='Total capital to allocate for options trading')
                        
    parser.add_argument('--allocation-per-trade', type=float, default=0.02,
                        help='Maximum allocation per trade as percentage of capital (0.02 = 2%)')
                        
    parser.add_argument('--delta-target', type=float, default=0.3,
                        help='Target delta for option selections')
                        
    parser.add_argument('--profit-target', type=float, default=0.5,
                        help='Profit target as percentage of option premium (0.5 = 50%)')
                        
    parser.add_argument('--stop-loss', type=float, default=0.5,
                        help='Stop loss as percentage of option premium (0.5 = 50%)')
                        
    parser.add_argument('--technical-filter', action='store_true',
                        help='Apply technical filters to entry/exit decisions')
                        
    parser.add_argument('--paper-trading', action='store_true',
                        help='Use paper trading mode instead of live trading')
                        
    parser.add_argument('--duration', type=int, default=1,
                        help='Trading duration in days')
                        
    parser.add_argument('--hours', type=int, default=None,
                        help='Trading duration in hours (overrides --duration if specified)')
                        
    parser.add_argument('--use-threads', action='store_true',
                        help='Use threading for parallel processing of symbols')
                        
    parser.add_argument('--max-threads', type=int, default=10,
                        help='Maximum number of threads to use (if threading enabled)')
                        
    parser.add_argument('--use-custom-symbols', action='store_true',
                        help='Use the custom symbols list instead of scanning')
                        
    parser.add_argument('--custom-symbols-file', type=str, 
                        default=os.path.join(project_root, 'data', 'custom_symbols_50.txt'),
                        help='File path to custom 50 symbols list')
                        
    parser.add_argument('--log-trades', action='store_true',
                        help='Log detailed trade information')
                        
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Directory for output files')
                        
    return parser.parse_args()


def get_strategy_class(strategy_name: str):
    """Map strategy name to strategy class."""
    strategy_map = {
        'COVERED_CALL': CoveredCallStrategy,
        'CASH_SECURED_PUT': CashSecuredPutStrategy,
        'IRON_CONDOR': IronCondorStrategy
    }
    
    return strategy_map.get(strategy_name)


def load_custom_symbols(file_path: str) -> List[str]:
    """Load custom symbols list from file."""
    try:
        if not os.path.exists(file_path):
            logger.error(f"Custom symbols file {file_path} not found")
            return []
            
        with open(file_path, 'r') as f:
            symbols = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]
        
        logger.info(f"Loaded {len(symbols)} symbols from {file_path}")
        return symbols
    except Exception as e:
        logger.error(f"Error loading custom symbols from {file_path}: {str(e)}")
        return []


async def filter_symbols_by_criteria(market_data_service, filter_method: str, limit: int = 50) -> List[str]:
    """Filter symbols based on specified criteria."""
    try:
        # Get market symbols
        all_symbols = await market_data_service.get_market_symbols(market_type="stock")
        
        if not all_symbols:
            logger.error("Failed to retrieve market symbols")
            return []
        
        # For demo purposes, just return a subset
        if filter_method == 'top_volume':
            # In a real implementation, this would query for highest volume stocks
            return all_symbols[:limit]
        elif filter_method == 'top_gainers':
            # This would query for stocks with highest daily gains
            return all_symbols[10:10+limit]
        elif filter_method == 'top_losers':
            # This would query for stocks with largest daily drops
            return all_symbols[20:20+limit]
        elif filter_method == 'most_volatile':
            # This would query for stocks with highest volatility
            return all_symbols[30:30+limit]
        else:
            return all_symbols[:limit]
    except Exception as e:
        logger.error(f"Error filtering symbols: {str(e)}")
        return []


def get_thread_services():
    """Get thread-local service instances."""
    if not hasattr(thread_local, 'services'):
        # Create new service instances for this thread
        # Initialiser l'adaptateur avec la configuration de paper trading
        config = {
            'mode': 'paper',  # 'paper' ou 'live'
            'subscription_level': 1,  # Niveau d'abonnement Alpaca
            'options_trading': True  # Activer le trading d'options
        }
        broker = AlpacaAdapter(config)  # Configuration pour paper trading
        # Ne pas utiliser asyncio.run(), on doit utiliser le broker comme s'il était déjà connecté
        # La connexion sera établie lors de la première opération avec l'API
        
        # Créer d'abord le service de données de marché
        market_data_service = MarketDataService()
        
        # Créer le service de trading en utilisant le broker
        trading_service = TradingService(broker)
        
        thread_local.services = {
            'broker': broker,
            'market_data': market_data_service,
            'trading_service': trading_service,
            'options_service': OptionsService(trading_service, market_data_service)
        }
    
    return thread_local.services


async def process_symbol(symbol: str, args, global_broker, lock, active_positions, trade_results):
    """Process a single symbol for options trading."""
    if args.use_threads:
        # Use thread-local services
        services = get_thread_services()
        broker = services['broker']
        market_data_service = services['market_data']
        options_service = services['options_service']
    else:
        # Use global services
        broker = global_broker
        market_data_service = MarketDataService()
        trading_service = TradingService(broker)
        options_service = OptionsService(trading_service, market_data_service)
    
    try:
        # Get market data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        market_data = await market_data_service.get_historical_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe="1d"
        )
        
        if market_data.empty:
            logger.warning(f"No market data available for {symbol}")
            return
        
        # Create strategy instance
        strategy_class = get_strategy_class(args.strategy)
        if not strategy_class:
            logger.error(f"Unknown strategy: {args.strategy}")
            return
        
        strategy = strategy_class(
            underlying_symbol=symbol,
            max_position_size=args.allocation_per_trade,
            target_delta=args.delta_target,
            profit_target_pct=args.profit_target,
            stop_loss_pct=args.stop_loss,
            use_technical_filters=args.technical_filter
        )
        strategy.broker_adapter = broker
        strategy.options_service = options_service
        
        # Check for entry signal
        should_enter = await strategy.should_enter(market_data)
        
        if should_enter:
            # Synchronize access to shared resources
            with lock:
                # Check if we've exceeded our position limit
                if len(active_positions) >= args.max_symbols:
                    logger.info(f"Maximum positions ({args.max_symbols}) reached, skipping {symbol}")
                    return
                
                # Execute entry
                logger.info(f"Entry signal detected for {symbol} using {args.strategy}")
                entry_result = await strategy.execute_entry()
                
                if entry_result.get('success', False):
                    # Add to active positions
                    position_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    active_positions[position_id] = {
                        'symbol': symbol,
                        'strategy': args.strategy,
                        'entry_time': datetime.now().isoformat(),
                        'entry_details': entry_result
                    }
                    
                    logger.info(f"Position opened for {symbol}: {position_id}")
                    
                    if args.log_trades:
                        trade_result = {
                            'symbol': symbol,
                            'strategy': args.strategy,
                            'action': 'ENTER',
                            'timestamp': datetime.now().isoformat(),
                            'result': entry_result
                        }
                        trade_results.append(trade_result)
                else:
                    logger.warning(f"Entry failed for {symbol}: {entry_result.get('error', 'Unknown error')}")
        
        # Check existing positions for exit
        if args.use_threads:
            # In threaded mode, we can only check our own positions
            positions_to_check = {pos_id: pos for pos_id, pos in active_positions.items() 
                                if pos['symbol'] == symbol}
        else:
            # In single-threaded mode, check all positions
            positions_to_check = active_positions
        
        for position_id, position in list(positions_to_check.items()):
            if position['symbol'] == symbol:
                # Check for exit signal
                should_exit = await strategy.should_exit(position_id, market_data)
                
                if should_exit:
                    logger.info(f"Exit signal detected for {symbol} (position {position_id})")
                    
                    # Execute exit
                    exit_result = await strategy.execute_exit(position_id)
                    
                    if exit_result.get('success', False):
                        logger.info(f"Position closed for {symbol}: {position_id}")
                        
                        # Remove from active positions
                        with lock:
                            if position_id in active_positions:
                                del active_positions[position_id]
                            
                            if args.log_trades:
                                trade_result = {
                                    'symbol': symbol,
                                    'strategy': args.strategy,
                                    'action': 'EXIT',
                                    'timestamp': datetime.now().isoformat(),
                                    'position_id': position_id,
                                    'result': exit_result
                                }
                                trade_results.append(trade_result)
                    else:
                        logger.warning(f"Exit failed for {symbol}: {exit_result.get('error', 'Unknown error')}")
        
    except Exception as e:
        logger.error(f"Error processing {symbol}: {str(e)}")


async def run_high_volume_options_trader(args):
    """Run high-volume options trader with the provided arguments."""
    logger.info(f"Starting high-volume options trader with strategy: {args.strategy}")
    logger.info(f"Maximum symbols: {args.max_symbols}, Threading: {args.use_threads}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize services
    broker_config = {
        "mode": "paper" if args.paper_trading else "live",
        "subscription_level": 1,  # Niveau d'abonnement Alpaca
        "options_trading": True  # Activer le trading d'options
    }
    broker = AlpacaAdapter(config=broker_config)
    await broker.connect()
    
    market_data_service = MarketDataService()
    trading_service = TradingService(is_paper=args.paper_trading)
    
    # Get account information
    account = await broker.get_account_info()
    account_value = float(account.get('equity', args.capital))
    logger.info(f"Account value: ${account_value:.2f}")
    
    # Determine symbols to trade
    if args.use_custom_symbols:
        symbols = load_custom_symbols(args.custom_symbols_file)
        if not symbols and args.symbols:
            symbols = args.symbols
    elif args.symbols:
        symbols = args.symbols
    else:
        # Filter symbols based on criteria
        symbols = await filter_symbols_by_criteria(market_data_service, args.filter, args.max_symbols)
    
    # Limit to max symbols
    symbols = symbols[:args.max_symbols]
    
    if not symbols:
        logger.error("No symbols to trade. Please specify symbols or check your filter criteria.")
        return
    
    logger.info(f"Trading on {len(symbols)} symbols: {symbols[:5]}... (and {len(symbols)-5} more)")
    
    # Run trading loop
    if args.hours is not None:
        end_time = datetime.now() + timedelta(hours=args.hours)
        logger.info(f"Trading will run for {args.hours} hours until: {end_time}")
    else:
        end_time = datetime.now() + timedelta(days=args.duration)
        logger.info(f"Trading will run for {args.duration} days until: {end_time}")
    active_positions = {}  # Dict to track active positions
    trade_results = []  # List to track all trades
    lock = threading.Lock()  # Lock for thread synchronization
    
    logger.info(f"Trading will run until: {end_time}")
    
    try:
        while datetime.now() < end_time:
            start_loop = time.time()
            
            if args.use_threads:
                # Process symbols in parallel
                # Exécution synchrone dans des threads séparés, sans mélanger asyncio et threads
                with ThreadPoolExecutor(max_workers=min(args.max_threads, len(symbols))) as executor:
                    # Créer une fonction synchrone qui traitera un symbole à la fois
                    def process_symbol_sync(symbol):
                        try:
                            # Initialiser les services pour ce thread
                            services = get_thread_services()
                            broker_local = services['broker']
                            market_data_local = services['market_data']
                            options_service_local = services['options_service']
                            
                            # Traiter le symbole de façon synchrone
                            logger.info(f"Analysing {symbol} for options trading")
                            
                            # Implémentation synchrone simplifiée - juste pour tester
                            with lock:
                                # Exemple d'action - enregistrement du résultat
                                trade_results.append({
                                    'symbol': symbol,
                                    'strategy': args.strategy,
                                    'action': 'ANALYZED',
                                    'timestamp': datetime.now().isoformat()
                                })
                            return True
                        except Exception as e:
                            logger.error(f"Error processing symbol {symbol}: {e}")
                            return False
                    
                    # Soumettre tous les symboles à l'exécuteur de threads
                    futures = [executor.submit(process_symbol_sync, symbol) for symbol in symbols]
                    tasks = [asyncio.create_task(asyncio.to_thread(lambda f: f.result(), future)) for future in futures]
                    
                    # Wait for all tasks to complete
                    await asyncio.gather(*tasks)
            else:
                # Process symbols sequentially
                for symbol in symbols:
                    await process_symbol(symbol, args, broker, lock, active_positions, trade_results)
            
            # Save trade results periodically
            if args.log_trades and trade_results:
                trades_file = os.path.join(args.output_dir, 'high_volume_options_trades.json')
                with open(trades_file, 'w') as f:
                    json.dump(trade_results, f, indent=2)
            
            # Log current positions and account value
            account_info = await broker.get_account_info()
            current_equity = float(account_info.get('equity', 0))
            buying_power = float(account_info.get('buying_power', 0))
            cash = float(account_info.get('cash', 0))
            
            logger.info(f"Active positions: {len(active_positions)}/{args.max_symbols}")
            logger.info(f"Account equity: ${current_equity:.2f} | Buying power: ${buying_power:.2f} | Cash: ${cash:.2f}")
            
            # Calculate loop duration and sleep for the remainder of a minute
            loop_duration = time.time() - start_loop
            logger.info(f"Trading loop took {loop_duration:.2f} seconds to process {len(symbols)} symbols")
            
            sleep_time = max(60 - loop_duration, 1)  # At least 1 second sleep
            logger.info(f"Sleeping for {sleep_time:.2f} seconds before next iteration")
            await asyncio.sleep(sleep_time)
    
    except KeyboardInterrupt:
        logger.info("Trading interrupted by user")
    
    finally:
        # Close all positions at the end
        logger.info("Closing any remaining positions...")
        # Utiliser broker au lieu de trading_service qui n'a pas cette méthode
        try:
            positions = await broker.get_positions()
            for position in positions:
                symbol = position.get('symbol')
                qty = position.get('qty', 0)
                if symbol and float(qty) > 0:
                    logger.info(f"Closing position for {symbol} (quantity: {qty})")
                    try:
                        # Placer un ordre de vente pour fermer la position
                        await broker.place_order(
                            symbol=symbol,
                            qty=qty,
                            side="sell",
                            order_type="market",
                            time_in_force="day"
                        )
                        logger.info(f"Position closed for {symbol}")
                    except Exception as e:
                        logger.error(f"Error closing position for {symbol}: {e}")
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
        
        # Generate final report
        report_file = os.path.join(args.output_dir, f"high_volume_options_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        report = {
            'strategy': args.strategy,
            'start_time': (datetime.now() - timedelta(days=args.duration)).isoformat(),
            'end_time': datetime.now().isoformat(),
            'symbols_count': len(symbols),
            'positions_opened': len(trade_results) // 2,  # Approximate, assumes equal entries and exits
            'settings': vars(args),
            'trades': trade_results if args.log_trades else []
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Trading report saved to {report_file}")
        logger.info("Trading completed")


if __name__ == '__main__':
    args = parse_arguments()
    asyncio.run(run_high_volume_options_trader(args))
