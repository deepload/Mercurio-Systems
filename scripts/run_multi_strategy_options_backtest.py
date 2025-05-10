"""
Multi-Strategy Options Backtester

This script runs backtests for multiple options strategies over a specified time period
and generates performance reports for each strategy. The reports can be analyzed using
the options performance dashboard.
"""

import os
import sys
import asyncio
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any, Optional
import json

# Add project root to path
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from app.services.options_backtester import OptionsBacktester
from app.services.market_data import MarketDataService
from app.strategies.options.covered_call import CoveredCallStrategy
from app.strategies.options.cash_secured_put import CashSecuredPutStrategy
from app.strategies.options.long_call import LongCallStrategy
from app.strategies.options.long_put import LongPutStrategy
from app.strategies.options.iron_condor import IronCondorStrategy
from app.strategies.options.butterfly_spread import ButterflySpreadStrategy


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(project_root, 'logs', 'options_backtest.log'))
    ]
)

logger = logging.getLogger(__name__)


def get_strategy_parameters() -> Dict[str, Dict[str, Any]]:
    """
    Define default parameters for each strategy.
    
    Returns:
        Dictionary mapping strategy names to parameter dictionaries
    """
    return {
        "CoveredCall": {
            "max_position_size": 0.10,
            "target_delta": 0.30,
            "days_to_expiration": 30,
            "profit_target_pct": 0.80,
            "stop_loss_pct": 0.50,
            "use_technical_filters": True
        },
        "CashSecuredPut": {
            "max_position_size": 0.10,
            "target_delta": 0.30,
            "days_to_expiration": 30,
            "profit_target_pct": 0.80,
            "stop_loss_pct": 0.50,
            "use_technical_filters": True
        },
        "LongCall": {
            "max_position_size": 0.05,
            "target_delta": 0.70,
            "days_to_expiration": 45,
            "profit_target_pct": 1.00,
            "stop_loss_pct": 0.50,
            "use_technical_filters": True
        },
        "LongPut": {
            "max_position_size": 0.05,
            "target_delta": 0.70,
            "days_to_expiration": 45,
            "profit_target_pct": 1.00,
            "stop_loss_pct": 0.50,
            "use_technical_filters": True
        },
        "IronCondor": {
            "max_position_size": 0.10,
            "call_spread_width": 5.0,
            "put_spread_width": 5.0,
            "short_call_delta": 0.25,
            "short_put_delta": 0.25,
            "days_to_expiration": 30,
            "profit_target_pct": 0.50,
            "stop_loss_pct": 1.00,
            "use_technical_filters": True
        },
        "ButterflySpread": {
            "max_position_size": 0.05,
            "option_type": "call",
            "delta_target": 0.50,
            "wing_width_pct": 0.05,
            "days_to_expiration": 30,
            "profit_target_pct": 0.50,
            "stop_loss_pct": 0.50,
            "use_technical_filters": True
        }
    }


def get_strategy_class_map() -> Dict[str, Any]:
    """
    Map strategy names to their class implementations.
    
    Returns:
        Dictionary mapping strategy names to class implementations
    """
    return {
        "CoveredCall": CoveredCallStrategy,
        "CashSecuredPut": CashSecuredPutStrategy,
        "LongCall": LongCallStrategy,
        "LongPut": LongPutStrategy,
        "IronCondor": IronCondorStrategy,
        "ButterflySpread": ButterflySpreadStrategy
    }


def get_default_symbols() -> List[str]:
    """
    Get a default list of symbols to backtest.
    
    Returns:
        List of default symbols
    """
    return [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", 
        "TSLA", "NVDA", "AMD", "INTC", "SPY"
    ]


async def run_backtest(
    strategies: List[str],
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    initial_capital: float,
    timeframe: str = "1d",
    output_directory: str = "backtest_reports",
    use_custom_params: bool = False,
    custom_params: Optional[Dict[str, Dict[str, Any]]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Run backtests for multiple strategies and symbols.
    
    Args:
        strategies: List of strategy names to backtest
        symbols: List of symbols to backtest on
        start_date: Start date for backtest data (YYYY-MM-DD)
        end_date: End date for backtest data (YYYY-MM-DD)
        initial_capital: Initial capital for each backtest
        timeframe: Timeframe for market data
        output_directory: Directory to save backtest reports
        use_custom_params: Whether to use custom parameters
        custom_params: Custom parameters for strategies
        
    Returns:
        Dictionary of backtest results by strategy
    """
    # Initialize services
    market_data_service = MarketDataService()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Get strategy class map
    strategy_class_map = get_strategy_class_map()
    
    # Get default parameters
    default_params = get_strategy_parameters()
    
    # Initialize backtester
    backtester = OptionsBacktester(
        market_data_service=market_data_service,
        initial_capital=initial_capital,
        data_start_date=start_date,
        data_end_date=end_date,
        output_directory=output_directory
    )
    
    all_results = {}
    
    for strategy_name in strategies:
        if strategy_name not in strategy_class_map:
            logger.warning(f"Strategy {strategy_name} not found. Skipping.")
            continue
            
        strategy_class = strategy_class_map[strategy_name]
        
        # Use custom parameters if provided, otherwise use defaults
        if use_custom_params and custom_params and strategy_name in custom_params:
            strategy_params = custom_params[strategy_name]
        else:
            strategy_params = default_params.get(strategy_name, {})
        
        logger.info(f"Running backtest for {strategy_name} with parameters: {strategy_params}")
        
        # Generate a report name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"{strategy_name}_backtest_{timestamp}"
        
        try:
            # Run the backtest
            results = await backtester.run_backtest(
                strategy_class=strategy_class,
                symbols=symbols,
                strategy_params=strategy_params,
                timeframe=timeframe,
                report_name=report_name
            )
            
            all_results[strategy_name] = results
            
            logger.info(f"Backtest for {strategy_name} completed. Report saved as {report_name}.json")
            
        except Exception as e:
            logger.error(f"Error during backtest for {strategy_name}: {str(e)}")
            all_results[strategy_name] = {"success": False, "error": str(e)}
    
    return all_results


def generate_summary_report(all_results: Dict[str, Dict[str, Any]], output_directory: str) -> None:
    """
    Generate a summary report of all backtest results.
    
    Args:
        all_results: Dictionary of backtest results by strategy
        output_directory: Directory to save the summary report
    """
    summary = []
    
    for strategy_name, results in all_results.items():
        if not results.get("success", False):
            summary.append({
                "strategy": strategy_name,
                "status": "Failed",
                "error": results.get("error", "Unknown error")
            })
            continue
            
        # Extract key metrics
        summary_entry = {
            "strategy": strategy_name,
            "status": "Success",
            "total_return_pct": results.get("total_return_pct", 0),
            "annualized_return_pct": results.get("annualized_return_pct", 0),
            "total_trades": results.get("total_trades", 0),
            "win_rate": results.get("win_rate", 0),
            "max_drawdown_pct": results.get("max_drawdown_pct", 0),
            "sharpe_ratio": results.get("sharpe_ratio", 0),
            "sortino_ratio": results.get("sortino_ratio", 0)
        }
        
        summary.append(summary_entry)
    
    # Sort by total return
    summary = sorted(summary, key=lambda x: x.get("total_return_pct", 0) if x["status"] == "Success" else -float('inf'), reverse=True)
    
    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(output_directory, f"backtest_summary_{timestamp}.json")
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    logger.info(f"Summary report generated: {summary_path}")
    
    # Print summary table
    print("\n" + "=" * 100)
    print("BACKTEST SUMMARY")
    print("=" * 100)
    print(f"{'Strategy':<20} {'Status':<10} {'Return %':<10} {'Ann. Ret %':<10} {'Trades':<8} {'Win Rate':<10} {'Max DD%':<10} {'Sharpe':<8}")
    print("-" * 100)
    
    for entry in summary:
        if entry["status"] == "Success":
            print(f"{entry['strategy']:<20} {entry['status']:<10} {entry['total_return_pct']:<10.2f} {entry['annualized_return_pct']:<10.2f} {entry['total_trades']:<8} {entry['win_rate']:<10.2f} {entry['max_drawdown_pct']:<10.2f} {entry['sharpe_ratio']:<8.2f}")
        else:
            print(f"{entry['strategy']:<20} {entry['status']:<10} {'Error':<10} {'N/A':<10} {'N/A':<8} {'N/A':<10} {'N/A':<10} {'N/A':<8}")
    
    print("=" * 100)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run multi-strategy options backtests')
    
    parser.add_argument('--strategies', type=str, nargs='+', 
                        default=list(get_strategy_class_map().keys()),
                        choices=list(get_strategy_class_map().keys()) + ['ALL'],
                        help='Strategies to backtest')
                        
    parser.add_argument('--symbols', type=str, nargs='+',
                       default=get_default_symbols(),
                       help='Symbols to backtest on')
                       
    parser.add_argument('--start-date', type=str,
                       default=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                       help='Start date for backtest (YYYY-MM-DD)')
                       
    parser.add_argument('--end-date', type=str,
                       default=datetime.now().strftime('%Y-%m-%d'),
                       help='End date for backtest (YYYY-MM-DD)')
                       
    parser.add_argument('--initial-capital', type=float,
                       default=100000.0,
                       help='Initial capital for each backtest')
                       
    parser.add_argument('--timeframe', type=str,
                       default='1d',
                       choices=['1m', '5m', '15m', '30m', '1h', '1d', '1wk'],
                       help='Timeframe for market data')
                       
    parser.add_argument('--output-dir', type=str,
                       default='backtest_reports',
                       help='Directory to save backtest reports')
                       
    parser.add_argument('--custom-params', type=str,
                       help='Path to JSON file with custom strategy parameters')
                       
    return parser.parse_args()


async def main():
    """Main function to run the backtest script."""
    args = parse_arguments()
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.join(project_root, 'logs'), exist_ok=True)
    
    # Process strategies argument
    if 'ALL' in args.strategies:
        strategies = list(get_strategy_class_map().keys())
    else:
        strategies = args.strategies
    
    # Load custom parameters if provided
    use_custom_params = False
    custom_params = None
    
    if args.custom_params:
        if os.path.exists(args.custom_params):
            try:
                with open(args.custom_params, 'r') as f:
                    custom_params = json.load(f)
                use_custom_params = True
                logger.info(f"Loaded custom parameters from {args.custom_params}")
            except json.JSONDecodeError:
                logger.error(f"Error loading custom parameters from {args.custom_params}")
        else:
            logger.error(f"Custom parameters file {args.custom_params} not found")
    
    logger.info(f"Starting backtest for strategies: {strategies}")
    logger.info(f"Symbols: {args.symbols}")
    logger.info(f"Time period: {args.start_date} to {args.end_date}")
    
    # Convert string dates to datetime objects for the market data service
    try:
        start_date_obj = datetime.strptime(args.start_date, '%Y-%m-%d')
        end_date_obj = datetime.strptime(args.end_date, '%Y-%m-%d')
        
        logger.info(f"Parsed start date: {start_date_obj}, end date: {end_date_obj}")
        
        all_results = await run_backtest(
            strategies=strategies,
            symbols=args.symbols,
            start_date=start_date_obj,
            end_date=end_date_obj,
            initial_capital=args.initial_capital,
            timeframe=args.timeframe,
            output_directory=args.output_dir,
            use_custom_params=use_custom_params,
            custom_params=custom_params
        )
    except ValueError as e:
        logger.error(f"Error parsing dates: {e}")
        all_results = {strategy: {"success": False, "error": f"Date parsing error: {e}"} for strategy in strategies}
    
    generate_summary_report(all_results, args.output_dir)
    
    # Instructions for viewing the dashboard
    print("\nTo view the backtest results in the dashboard, run:")
    print(f"streamlit run {os.path.join(project_root, 'app', 'dashboards', 'options_performance.py')}")


if __name__ == '__main__':
    asyncio.run(main())
