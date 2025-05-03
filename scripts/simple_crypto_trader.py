#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple Crypto Day Trading Script for Alpaca
-------------------------------------------
This script implements a simple crypto trading system that uses
Alpaca's API to trade cryptocurrencies in paper mode.
It's designed specifically to work with Alpaca subscription levels.

Usage:
    python simple_crypto_trader.py --duration 1h
"""

import os
import sys
import json
import time
import signal
import logging
import argparse
import asyncio
from enum import Enum
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import Mercurio modules
from app.services.market_data import MarketDataService
from app.services.trading import TradingService

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("simple_crypto_trader")

# Global variables for signal handling
running = True
session_end_time = None

# Enums for session duration
class SessionDuration(int, Enum):
    ONE_HOUR = 3600
    FOUR_HOURS = 14400
    EIGHT_HOURS = 28800
    CUSTOM = 0

class SimpleCryptoTrader:
    """
    Simple Cryptocurrency Day Trading System
    
    Features:
    - Uses Alpaca API for paper trading cryptocurrencies
    - Simple moving average crossover strategy
    - Multiple session durations (1h, 4h, 8h)
    - Configurable trading parameters
    """
    
    def __init__(self, session_duration: SessionDuration = SessionDuration.ONE_HOUR):
        """Initialize the crypto trading system"""
        self.session_duration = session_duration
        
        # Services
        self.market_data = None
        self.trading_service = None
        
        # Trading parameters
        self.symbols = []  # Will be populated with available crypto symbols
        self.fast_ma_period = 5   # 5 minutes for fast moving average
        self.slow_ma_period = 15  # 15 minutes for slow moving average
        self.position_size_pct = 0.02  # 2% of portfolio per position
        self.stop_loss_pct = 0.03  # 3% stop loss
        self.take_profit_pct = 0.06  # 6% take profit
        
        # State tracking
        self.positions = {}
        self.portfolio_value = 0.0
        self.session_start_time = None
        self.session_end_time = None
        
        logger.info("Simple Crypto Trader initialized")
        
    async def initialize(self):
        """Initialize services and load configuration"""
        try:
            # Initialize market data service with Alpaca as provider
            self.market_data = MarketDataService(provider_name="alpaca")
            
            # Check if Alpaca is properly configured
            active_provider = await self.market_data.active_provider()
            if active_provider and active_provider.name == "Alpaca":
                if hasattr(active_provider, 'subscription_level'):
                    logger.info(f"Using Alpaca (level {active_provider.subscription_level}) for crypto trading")
                else:
                    logger.info("Using Alpaca for crypto trading")
            else:
                logger.warning("Alpaca provider not active, falling back to alternative provider")
            
            # Initialize trading service in paper mode
            self.trading_service = TradingService(paper_trading=True)
            logger.info("Trading service initialized in PAPER mode")
            
            # Get account information
            account = await self.trading_service.get_account()
            self.portfolio_value = float(account.get("portfolio_value", 0.0))
            logger.info(f"Initial portfolio value: ${self.portfolio_value:.2f}")
            
            # Get available crypto symbols
            self.symbols = await self.market_data.get_market_symbols(market_type="crypto")
            logger.info(f"Found {len(self.symbols)} available crypto symbols")
            if self.symbols:
                logger.info(f"Sample symbols: {', '.join(self.symbols[:5])}")
            
            return True
        except Exception as e:
            logger.error(f"Error initializing crypto trader: {e}")
            return False
            
    async def start(self, duration_seconds: Optional[int] = None):
        """Start the crypto trading session"""
        self.session_start_time = datetime.now()
        
        if duration_seconds is not None:
            self.session_end_time = self.session_start_time + timedelta(seconds=duration_seconds)
        else:
            self.session_end_time = self.session_start_time + timedelta(seconds=int(self.session_duration))
            
        logger.info(f"Starting crypto trading session at {self.session_start_time}")
        logger.info(f"Session will end at {self.session_end_time}")
        
        # Initialize the trader
        initialized = await self.initialize()
        if not initialized:
            logger.error("Failed to initialize crypto trader, aborting")
            await self.generate_performance_report()
            return
            
        # Start the trading loop
        await self.trading_loop()
        
        # Generate performance report at the end
        await self.generate_performance_report()
            
    async def trading_loop(self):
        """Main trading loop"""
        global running
        
        try:
            while running and datetime.now() < self.session_end_time:
                # Process each symbol
                for symbol in self.symbols[:10]:  # Limit to top 10 cryptos to avoid rate limits
                    try:
                        await self.process_symbol(symbol)
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
                
                # Update portfolio state
                await self.update_portfolio_state()
                
                # Sleep for 60 seconds before next iteration
                logger.info(f"Waiting 60 seconds before next trading cycle. Session ends in "
                           f"{int((self.session_end_time - datetime.now()).total_seconds() / 60)} minutes")
                await asyncio.sleep(60)
                
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
        finally:
            logger.info("Trading loop completed")
            
    async def process_symbol(self, symbol: str):
        """Process a single trading symbol"""
        logger.info(f"Processing {symbol}")
        
        # Get historical data (5-minute intervals for the last 24 hours)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)
        
        try:
            data = await self.market_data.get_historical_data(
                symbol, 
                start_date=start_date,
                end_date=end_date,
                timeframe="5Min"
            )
            
            if data.empty:
                logger.warning(f"No historical data available for {symbol}")
                return
                
            # Calculate moving averages
            data['fast_ma'] = data['close'].rolling(window=self.fast_ma_period).mean()
            data['slow_ma'] = data['close'].rolling(window=self.slow_ma_period).mean()
            
            # Get current position
            position = await self.trading_service.get_position(symbol)
            
            # Get current price
            current_price = await self.market_data.get_latest_price(symbol)
            logger.info(f"{symbol} current price: ${current_price:.4f}")
            
            # Trading logic - Moving Average Crossover
            if len(data) >= self.slow_ma_period:
                last_row = data.iloc[-1]
                prev_row = data.iloc[-2]
                
                # Check for buy signal: fast MA crosses above slow MA
                buy_signal = (
                    prev_row['fast_ma'] <= prev_row['slow_ma'] and 
                    last_row['fast_ma'] > last_row['slow_ma']
                )
                
                # Check for sell signal: fast MA crosses below slow MA
                sell_signal = (
                    prev_row['fast_ma'] >= prev_row['slow_ma'] and 
                    last_row['fast_ma'] < last_row['slow_ma']
                )
                
                # Execute signals
                if buy_signal and not position:
                    await self.execute_buy(symbol, current_price)
                elif sell_signal and position:
                    await self.execute_sell(symbol, current_price, position)
                
                # Check stop loss and take profit
                if position:
                    entry_price = float(position.get("avg_entry_price", 0))
                    if entry_price > 0:
                        pnl_pct = (current_price - entry_price) / entry_price
                        
                        if pnl_pct <= -self.stop_loss_pct:
                            logger.info(f"{symbol} hit stop loss at {pnl_pct:.2%}")
                            await self.execute_sell(symbol, current_price, position)
                        elif pnl_pct >= self.take_profit_pct:
                            logger.info(f"{symbol} hit take profit at {pnl_pct:.2%}")
                            await self.execute_sell(symbol, current_price, position)
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
    
    async def execute_buy(self, symbol: str, price: float):
        """Execute a buy order"""
        try:
            # Calculate position size
            position_value = self.portfolio_value * self.position_size_pct
            qty = position_value / price
            
            logger.info(f"BUY SIGNAL: {symbol} at ${price:.4f}, qty: {qty:.6f}")
            
            # Place market order
            order = await self.trading_service.place_market_order(
                symbol=symbol,
                qty=qty,
                side="buy"
            )
            
            if order:
                logger.info(f"Buy order placed for {symbol}: {order}")
            else:
                logger.error(f"Failed to place buy order for {symbol}")
                
        except Exception as e:
            logger.error(f"Error executing buy for {symbol}: {e}")
    
    async def execute_sell(self, symbol: str, price: float, position: Dict[str, Any]):
        """Execute a sell order"""
        try:
            qty = float(position.get("qty", 0))
            
            if qty <= 0:
                logger.warning(f"Invalid position quantity for {symbol}: {qty}")
                return
                
            logger.info(f"SELL SIGNAL: {symbol} at ${price:.4f}, qty: {qty:.6f}")
            
            # Place market order
            order = await self.trading_service.place_market_order(
                symbol=symbol,
                qty=qty,
                side="sell"
            )
            
            if order:
                logger.info(f"Sell order placed for {symbol}: {order}")
            else:
                logger.error(f"Failed to place sell order for {symbol}")
                
        except Exception as e:
            logger.error(f"Error executing sell for {symbol}: {e}")
    
    async def update_portfolio_state(self):
        """Update portfolio value and positions"""
        try:
            account = await self.trading_service.get_account()
            self.portfolio_value = float(account.get("portfolio_value", 0.0))
            logger.info(f"Current portfolio value: ${self.portfolio_value:.2f}")
            
            # Update positions
            positions = await self.trading_service.get_positions()
            self.positions = {p.get("symbol"): p for p in positions}
            
            # Log open positions
            if self.positions:
                logger.info(f"Current open positions: {len(self.positions)}")
                for symbol, pos in self.positions.items():
                    entry_price = float(pos.get("avg_entry_price", 0))
                    current_price = await self.market_data.get_latest_price(symbol)
                    qty = float(pos.get("qty", 0))
                    market_value = current_price * qty
                    pnl = (current_price - entry_price) * qty
                    pnl_pct = ((current_price / entry_price) - 1) * 100 if entry_price > 0 else 0
                    
                    logger.info(f"  {symbol}: {qty:.6f} @ ${entry_price:.4f} - Value: ${market_value:.2f} - P/L: ${pnl:.2f} ({pnl_pct:.2f}%)")
            else:
                logger.info("No open positions")
                
        except Exception as e:
            logger.error(f"Error updating portfolio state: {e}")
    
    async def generate_performance_report(self):
        """Generate a performance report at the end of the trading session"""
        try:
            end_time = datetime.now()
            duration = end_time - self.session_start_time if self.session_start_time else timedelta(0)
            hours, remainder = divmod(duration.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            logger.info("===================================================")
            logger.info("CRYPTO TRADING SESSION PERFORMANCE REPORT")
            logger.info("===================================================")
            logger.info(f"Session Duration: {hours}h {minutes}m {seconds}s")
            logger.info(f"Start Time: {self.session_start_time}")
            logger.info(f"End Time: {end_time}")
            
            # Get final account state
            try:
                account = await self.trading_service.get_account()
                final_value = float(account.get("portfolio_value", 0.0))
                initial_value = getattr(self, "initial_portfolio_value", 0.0)
                
                if initial_value > 0:
                    profit_loss = final_value - initial_value
                    profit_loss_pct = (profit_loss / initial_value) * 100
                    logger.info(f"Initial Portfolio Value: ${initial_value:.2f}")
                    logger.info(f"Final Portfolio Value: ${final_value:.2f}")
                    logger.info(f"Profit/Loss: ${profit_loss:.2f} ({profit_loss_pct:.2f}%)")
            except:
                logger.warning("Could not retrieve final account information")
            
            # Show open positions
            try:
                positions = await self.trading_service.get_positions()
                if positions:
                    logger.info(f"Open Positions at Session End: {len(positions)}")
                    for position in positions:
                        symbol = position.get("symbol", "Unknown")
                        qty = position.get("qty", 0)
                        avg_price = position.get("avg_entry_price", 0.0)
                        market_value = position.get("market_value", 0.0)
                        
                        logger.info(f"  {symbol}: {qty} units at ${avg_price:.4f} - Value: ${market_value:.2f}")
                else:
                    logger.info("No open positions at session end")
            except:
                logger.warning("Could not retrieve position information")
                
            logger.info("===================================================")
            logger.info("CRYPTO TRADING SESSION COMPLETED")
            logger.info("===================================================")
                
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Simple Cryptocurrency Day Trading System")
    parser.add_argument("--duration", type=str, choices=["1h", "4h", "8h", "custom"], default="1h",
                        help="Trading session duration (1h, 4h, 8h, or custom)")
    parser.add_argument("--custom-seconds", type=int, default=0,
                        help="Custom duration in seconds if --duration=custom")
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                        default="INFO", help="Logging level")
                        
    args = parser.parse_args()
    
    # Set logging level
    numeric_level = getattr(logging, args.log_level)
    logging.basicConfig(level=numeric_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Determine session duration
    duration_map = {
        "1h": SessionDuration.ONE_HOUR,
        "4h": SessionDuration.FOUR_HOURS,
        "8h": SessionDuration.EIGHT_HOURS,
        "custom": SessionDuration.CUSTOM
    }
    session_duration = duration_map.get(args.duration, SessionDuration.ONE_HOUR)
    custom_duration = args.custom_seconds if args.duration == "custom" else 0
    
    # Create trader
    trader = SimpleCryptoTrader(session_duration=session_duration)
    
    # Register signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        global running, session_end_time
        logger.info(f"Received signal {sig}, shutting down...")
        running = False
        session_end_time = datetime.now()
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the trader in the async event loop
    try:
        if hasattr(asyncio, 'run'):  # Python 3.7+
            if custom_duration > 0:
                asyncio.run(trader.start(custom_duration))
            else:
                asyncio.run(trader.start())
        else:  # Older Python versions
            loop = asyncio.get_event_loop()
            try:
                if custom_duration > 0:
                    loop.run_until_complete(trader.start(custom_duration))
                else:
                    loop.run_until_complete(trader.start())
            finally:
                loop.close()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    except Exception as e:
        logger.error(f"Error running crypto trader: {e}")
    finally:
        logger.info("Crypto trader shutdown complete")

if __name__ == "__main__":
    main()
