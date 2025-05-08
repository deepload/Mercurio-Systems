#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ML-Powered Options Trader Script

This script uses machine learning models (LSTM, Transformer, LLM) to generate
trading signals for options strategies. It combines predictive ML models with
options strategies for enhanced performance.

Usage:
    python -m scripts.options.run_ml_options_trader --ml-strategy LSTM --options-strategy COVERED_CALL --symbols AAPL MSFT --capital 100000
"""

import os
import sys
import asyncio
import argparse
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple

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
from app.strategies.lstm_predictor import LSTMPredictorStrategy
from app.strategies.transformer_strategy import TransformerStrategy
from app.strategies.llm_strategy import LLMStrategy
from app.strategies.msi_strategy import MultiSourceIntelligenceStrategy
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments for the ML options trader."""
    parser = argparse.ArgumentParser(description='Run ML-based options trading strategy')
    
    parser.add_argument('--ml-strategy', type=str, required=True,
                        choices=['LSTM', 'TRANSFORMER', 'LLM', 'MSI', 'ENSEMBLE'],
                        help='ML strategy to use for signal generation')
                        
    parser.add_argument('--options-strategy', type=str, required=True,
                        choices=['COVERED_CALL', 'CASH_SECURED_PUT', 'LONG_CALL', 
                                'LONG_PUT', 'IRON_CONDOR', 'BUTTERFLY', 'AUTO'],
                        help='Options strategy to execute based on ML signals')
                        
    parser.add_argument('--symbols', type=str, nargs='+', required=True,
                        help='Symbols to trade options for')
                        
    parser.add_argument('--max-symbols', type=int, default=10,
                        help='Maximum number of symbols to trade')
                        
    parser.add_argument('--capital', type=float, default=100000.0,
                        help='Total capital to allocate for options trading')
                        
    parser.add_argument('--allocation-per-trade', type=float, default=0.05,
                        help='Maximum allocation per trade as percentage of capital (0.05 = 5%)')
                        
    parser.add_argument('--days-to-expiry', type=int, default=30,
                        help='Target days to expiration for options')
                        
    parser.add_argument('--confidence-threshold', type=float, default=0.65,
                        help='Minimum ML confidence score to enter a trade (0-1)')
                        
    parser.add_argument('--profit-target', type=float, default=0.5,
                        help='Profit target as percentage of option premium (0.5 = 50%)')
                        
    parser.add_argument('--stop-loss', type=float, default=0.7,
                        help='Stop loss as percentage of option premium (0.7 = 70%)')
                        
    parser.add_argument('--paper-trading', action='store_true',
                        help='Use paper trading mode instead of live trading')
                        
    parser.add_argument('--duration', type=int, default=1,
                        help='Trading duration in days')
                        
    parser.add_argument('--custom-symbols-file', type=str, 
                        help='File path to custom symbols list')
                        
    parser.add_argument('--use-threads', action='store_true',
                        help='Use threading for faster processing')
                        
    parser.add_argument('--use-custom-symbols', action='store_true',
                        help='Use custom symbols list')
                        
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Directory for output logs and reports')
                        
    return parser.parse_args()


def get_ml_strategy_class(strategy_name: str):
    """Map ML strategy name to strategy class."""
    strategy_map = {
        'LSTM': LSTMPredictorStrategy,
        'TRANSFORMER': TransformerStrategy,
        'LLM': LLMStrategy,
        'MSI': MultiSourceIntelligenceStrategy
    }
    
    return strategy_map.get(strategy_name)


def get_options_strategy_class(strategy_name: str):
    """Map options strategy name to strategy class."""
    strategy_map = {
        'COVERED_CALL': CoveredCallStrategy,
        'CASH_SECURED_PUT': CashSecuredPutStrategy,
        'LONG_CALL': LongCallStrategy,
        'LONG_PUT': LongPutStrategy,
        'IRON_CONDOR': IronCondorStrategy,
        'BUTTERFLY': ButterflySpreadStrategy
    }
    
    return strategy_map.get(strategy_name)


def load_custom_symbols(file_path: str) -> List[str]:
    """Load custom symbols list from file."""
    try:
        with open(file_path, 'r') as f:
            symbols = [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]
        return symbols
    except Exception as e:
        logger.error(f"Error loading custom symbols from {file_path}: {str(e)}")
        return []


async def get_ml_prediction(ml_strategy, symbol: str, market_data: pd.DataFrame) -> Tuple[str, float]:
    """Get prediction from ML strategy."""
    try:
        # Preprocess data for ML model
        processed_data = await ml_strategy.preprocess_data(market_data)
        
        # Get prediction
        prediction, confidence = await ml_strategy.predict(processed_data)
        
        return prediction, confidence
    except Exception as e:
        logger.error(f"Error getting ML prediction for {symbol}: {str(e)}")
        return None, 0.0


def map_ml_signal_to_options_strategy(ml_signal: str) -> str:
    """Map ML signal to appropriate options strategy."""
    # If signal is None, return None
    if not ml_signal:
        return None
        
    # Map trade action to options strategy
    if ml_signal == "BUY":  # Strong bullish signal
        return "LONG_CALL"
    elif ml_signal == "SELL":  # Strong bearish signal
        return "LONG_PUT"
    elif ml_signal == "HOLD" or ml_signal == "NEUTRAL_BULLISH":  # Slight bullish or neutral
        return "CASH_SECURED_PUT"
    elif ml_signal == "NEUTRAL_BEARISH":  # Slight bearish
        return "COVERED_CALL"
    elif ml_signal == "NEUTRAL":  # Very neutral
        return "IRON_CONDOR"
    else:
        return None


async def run_ml_options_trader(args):
    """Run ML-based options trader with the provided arguments."""
    logger.info(f"Starting ML-powered options trader with ML strategy: {args.ml_strategy} and options strategy: {args.options_strategy}")
    
    # Load custom symbols if requested
    if args.use_custom_symbols and args.custom_symbols_file:
        symbols = load_custom_symbols(args.custom_symbols_file)
        if not symbols:
            symbols = args.symbols
    else:
        symbols = args.symbols
    
    # Limit to max symbols
    symbols = symbols[:args.max_symbols]
    logger.info(f"Trading on symbols: {symbols}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize services
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
    
    # Initialize ML strategy
    ml_strategy_class = get_ml_strategy_class(args.ml_strategy)
    
    if not ml_strategy_class and args.ml_strategy != 'ENSEMBLE':
        logger.error(f"Unknown ML strategy: {args.ml_strategy}")
        return
    
    ml_strategies = []
    
    if args.ml_strategy == 'ENSEMBLE':
        # Create ensemble of ML strategies
        for strategy_name in ['LSTM', 'TRANSFORMER', 'MSI']:
            ml_strategy_class = get_ml_strategy_class(strategy_name)
            ml_strategy = ml_strategy_class()
            ml_strategies.append(ml_strategy)
    else:
        # Use single ML strategy
        ml_strategy = ml_strategy_class()
        ml_strategies.append(ml_strategy)
    
    # Run trading loop
    end_time = datetime.now() + timedelta(days=args.duration)
    trades = []
    
    logger.info(f"Trading will run until: {end_time}")
    
    try:
        while datetime.now() < end_time:
            for symbol in symbols:
                # Get market data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=60)  # 60 days of historical data for ML
                
                try:
                    market_data = await market_data_service.get_historical_data(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        timeframe="1d"
                    )
                    
                    if market_data.empty:
                        logger.warning(f"No market data available for {symbol}")
                        continue
                    
                    # Get ML predictions from all strategies
                    ml_predictions = []
                    for ml_strategy in ml_strategies:
                        prediction, confidence = await get_ml_prediction(ml_strategy, symbol, market_data)
                        if prediction and confidence >= args.confidence_threshold:
                            ml_predictions.append((prediction, confidence, ml_strategy.__class__.__name__))
                    
                    if not ml_predictions:
                        logger.info(f"No confident ML predictions for {symbol}")
                        continue
                    
                    # For ensemble, select prediction with highest confidence
                    if len(ml_predictions) > 1:
                        # Sort by confidence (highest first)
                        ml_predictions.sort(key=lambda x: x[1], reverse=True)
                        logger.info(f"Ensemble predictions for {symbol}: {ml_predictions}")
                    
                    prediction, confidence, strategy_name = ml_predictions[0]
                    logger.info(f"Selected prediction for {symbol}: {prediction} with confidence {confidence:.2f} from {strategy_name}")
                    
                    # If using AUTO for options strategy, map ML signal to appropriate option strategy
                    if args.options_strategy == 'AUTO':
                        options_strategy_name = map_ml_signal_to_options_strategy(prediction)
                        if not options_strategy_name:
                            logger.warning(f"Could not map ML signal {prediction} to options strategy for {symbol}")
                            continue
                    else:
                        options_strategy_name = args.options_strategy
                    
                    # Create options strategy instance
                    options_strategy_class = get_options_strategy_class(options_strategy_name)
                    if not options_strategy_class:
                        logger.error(f"Unknown options strategy: {options_strategy_name}")
                        continue
                    
                    # Initialize the options strategy
                    options_strategy = options_strategy_class(
                        underlying_symbol=symbol,
                        max_position_size=args.allocation_per_trade,
                        days_to_expiration=args.days_to_expiry,
                        profit_target_pct=args.profit_target,
                        stop_loss_pct=args.stop_loss
                    )
                    options_strategy.broker_adapter = broker
                    options_strategy.options_service = options_service
                    
                    # Execute the options strategy
                    entry_result = await options_strategy.execute_entry()
                    
                    if entry_result.get('success', False):
                        logger.info(f"Entry executed for {symbol} using {options_strategy_name}: {entry_result}")
                        
                        # Record trade
                        trade = {
                            'symbol': symbol,
                            'ml_strategy': strategy_name,
                            'ml_prediction': prediction,
                            'ml_confidence': confidence,
                            'options_strategy': options_strategy_name,
                            'entry_time': datetime.now().isoformat(),
                            'entry_details': entry_result
                        }
                        trades.append(trade)
                        
                        # Save trades to file
                        trades_file = os.path.join(args.output_dir, 'ml_options_trades.json')
                        with open(trades_file, 'w') as f:
                            json.dump(trades, f, indent=2)
                    else:
                        logger.warning(f"Entry failed for {symbol}: {entry_result.get('error', 'Unknown error')}")
                
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {str(e)}")
            
            # Sleep before next iteration
            await asyncio.sleep(3600)  # Check hourly
    
    except KeyboardInterrupt:
        logger.info("Trading interrupted by user")
    
    finally:
        # Close all positions at the end
        logger.info("Closing any remaining positions...")
        await trading_service.close_all_positions()
        
        # Generate trading report
        report_file = os.path.join(args.output_dir, f"ml_options_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        report = {
            'ml_strategy': args.ml_strategy,
            'options_strategy': args.options_strategy,
            'start_time': datetime.now().timestamp() - (args.duration * 86400),
            'end_time': datetime.now().timestamp(),
            'symbols': symbols,
            'trades': trades,
            'settings': vars(args)
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Trading report saved to {report_file}")
        logger.info("Trading completed")


if __name__ == '__main__':
    args = parse_arguments()
    asyncio.run(run_ml_options_trader(args))
