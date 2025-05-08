#!/usr/bin/env python
"""
Options Backtesting Script for Mercurio AI

This script runs backtest simulations for different options strategies 
using historical market data and generates performance reports.
"""

import os
import sys
import asyncio
import argparse
import logging
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Type

# Ensure proper module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from app.services.market_data import MarketDataService
from app.services.options_backtester import OptionsBacktester
from app.strategies.options.covered_call import CoveredCallStrategy
from app.strategies.options.cash_secured_put import CashSecuredPutStrategy
from app.strategies.options.long_call import LongCallStrategy
from app.strategies.options.long_put import LongPutStrategy
from app.strategies.options.iron_condor import IronCondorStrategy
from app.core.config import Config
from app.utils.logger import setup_logger

# Setup logging
logger = setup_logger('options_backtest', log_level=logging.INFO)

# Strategy mapping
STRATEGY_MAP = {
    'covered_call': CoveredCallStrategy,
    'cash_secured_put': CashSecuredPutStrategy,
    'long_call': LongCallStrategy,
    'long_put': LongPutStrategy,
    'iron_condor': IronCondorStrategy
}

# Default parameters for each strategy
DEFAULT_PARAMS = {
    'covered_call': {
        'max_position_size': 0.10,
        'profit_target_pct': 0.50,
        'stop_loss_pct': 0.50,
        'roll_when_dte': 7,
        'use_technical_filters': True
    },
    'cash_secured_put': {
        'max_position_size': 0.10,
        'min_implied_volatility': 0.20,
        'profit_target_pct': 0.50,
        'stop_loss_pct': 0.50,
        'roll_when_dte': 7,
        'use_technical_filters': True
    },
    'long_call': {
        'max_position_size': 0.05,
        'min_implied_volatility': 0.15,
        'max_implied_volatility': 0.60,
        'profit_target_pct': 1.00,
        'stop_loss_pct': 0.50,
        'use_technical_filters': True
    },
    'long_put': {
        'max_position_size': 0.05,
        'min_implied_volatility': 0.15,
        'max_implied_volatility': 0.60,
        'profit_target_pct': 1.00,
        'stop_loss_pct': 0.50,
        'use_technical_filters': True
    },
    'iron_condor': {
        'max_position_size': 0.10,
        'profit_target_pct': 0.50,
        'stop_loss_pct': 1.50,
        'roll_when_dte': 10,
        'use_technical_filters': True
    }
}

async def run_backtest(
    strategy_name: str,
    symbols: List[str],
    start_date: str,
    end_date: str,
    initial_capital: float = 100000.0,
    params: Dict[str, Any] = None,
    output_dir: str = './backtest_results'
) -> Dict[str, Any]:
    """
    Run a backtest for the specified options strategy.
    
    Args:
        strategy_name: Name of the strategy to backtest
        symbols: List of symbols to test
        start_date: Start date for the backtest (YYYY-MM-DD)
        end_date: End date for the backtest (YYYY-MM-DD)
        initial_capital: Initial capital for the backtest
        params: Custom strategy parameters
        output_dir: Directory to save results
        
    Returns:
        Dict: Backtest results
    """
    # Initialize services
    config = Config()
    market_data_service = MarketDataService(
        alpaca_api_key=config.ALPACA_API_KEY,
        alpaca_api_secret=config.ALPACA_API_SECRET,
        alpaca_base_url=config.ALPACA_BASE_URL
    )
    
    # Create backtester
    backtester = OptionsBacktester(
        market_data_service=market_data_service,
        initial_capital=initial_capital,
        data_start_date=start_date,
        data_end_date=end_date,
        output_directory=output_dir
    )
    
    if strategy_name not in STRATEGY_MAP:
        logger.error(f"Strategy '{strategy_name}' not found. Available strategies: {list(STRATEGY_MAP.keys())}")
        return {"success": False, "error": f"Strategy '{strategy_name}' not found"}
    
    strategy_class = STRATEGY_MAP[strategy_name]
    
    # Merge default parameters with custom parameters
    strategy_params = DEFAULT_PARAMS.get(strategy_name, {}).copy()
    if params:
        strategy_params.update(params)
    
    # Run the backtest
    logger.info(f"Starting backtest for {strategy_name} with {len(symbols)} symbols")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Strategy parameters: {json.dumps(strategy_params, indent=2)}")
    
    results = await backtester.run_backtest(
        strategy_class=strategy_class,
        symbols=symbols,
        strategy_params=strategy_params,
        timeframe='1d',
        report_name=f"{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Print summary
    if results.get("success", True):
        logger.info("Backtest completed successfully!")
        logger.info(f"Total Return: {results.get('total_return', 0):.2f} ({results.get('total_return_pct', 0):.2f}%)")
        logger.info(f"Total Trades: {results.get('total_trades', 0)}")
        logger.info(f"Win Rate: {results.get('win_rate', 0) * 100:.2f}%")
        logger.info(f"Max Drawdown: {results.get('max_drawdown_pct', 0):.2f}%")
        logger.info(f"Report saved to: {output_dir}")
    else:
        logger.error(f"Backtest failed: {results.get('error', 'Unknown error')}")
    
    return results

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run options strategy backtest')
    
    parser.add_argument(
        '--strategy', 
        type=str, 
        required=True,
        choices=list(STRATEGY_MAP.keys()),
        help='Strategy to backtest'
    )
    
    parser.add_argument(
        '--symbols', 
        type=str, 
        required=True,
        help='Comma-separated list of symbols to test'
    )
    
    parser.add_argument(
        '--start-date', 
        type=str, 
        default=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
        help='Start date for backtest (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end-date', 
        type=str, 
        default=datetime.now().strftime('%Y-%m-%d'),
        help='End date for backtest (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--initial-capital', 
        type=float, 
        default=100000.0,
        help='Initial capital for backtest'
    )
    
    parser.add_argument(
        '--params-file', 
        type=str, 
        help='JSON file with custom strategy parameters'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='./backtest_results',
        help='Directory to save backtest results'
    )
    
    return parser.parse_args()

async def main():
    """Main entry point for the script."""
    args = parse_arguments()
    
    # Parse symbols
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    # Load custom parameters if provided
    params = None
    if args.params_file:
        try:
            with open(args.params_file, 'r') as f:
                params = json.load(f)
        except Exception as e:
            logger.error(f"Error loading params file: {str(e)}")
            return
    
    # Run the backtest
    await run_backtest(
        strategy_name=args.strategy,
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        params=params,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    asyncio.run(main())
