#!/usr/bin/env python
"""
Mercurio AI Advanced Stock Day Trading System

This script provides a specialized implementation for stock market trading,
with adaptive strategy selection, multi-timeframe analysis, and comprehensive
risk management designed specifically for equities markets.

Features:
- Session-based trading (1h, 4h, 8h or custom duration)
- Adaptive strategy selection based on market regimes
- Volatility-based position sizing and risk management
- Market condition monitoring and trading pause during dangerous conditions
- Comprehensive logging and performance tracking
"""

import os
import sys
import json
import time
import signal
import argparse
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Import Mercurio AI components
from app.services.market_data import MarketDataService
from app.services.trading import TradingService
from app.services.strategy_manager import StrategyManager
from app.strategies.base import BaseStrategy
from app.db.models import TradeAction
from app.core.event_bus import EventBus, EventType

# Import advanced trading components
from app.strategies.adaptive.strategy_selector import StrategySelector, MarketRegime
from app.strategies.adaptive.market_analyzer import MarketAnalyzer
from app.strategies.adaptive.risk_manager import RiskManager, RiskLevel

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"stock_trader_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)

logger = logging.getLogger("StockTrader")

# Global variables
running = True
session_end_time = None


def signal_handler(sig, frame):
    """Handle interruption signals for graceful shutdown"""
    global running
    logger.info("Signal received. Gracefully shutting down after current iteration...")
    running = False


class SessionDuration(object):
    """Session duration options"""
    ONE_HOUR = 60 * 60  # 1 hour in seconds
    FOUR_HOURS = 4 * 60 * 60  # 4 hours in seconds
    EIGHT_HOURS = 8 * 60 * 60  # 8 hours in seconds
    FULL_DAY = 6.5 * 60 * 60  # 6.5 hours (typical US market day)
    CUSTOM = -1  # Custom duration


class MarketCondition(object):
    """Market condition assessment"""
    NORMAL = "normal"  # Normal trading conditions
    VOLATILE = "volatile"  # High volatility but tradable
    DANGEROUS = "dangerous"  # Dangerous conditions, pause trading
    INACTIVE = "inactive"  # Low volatility/volume, reduce activity
    CLOSED = "closed"  # Market closed


class StockDayTrader:
    """
    Advanced stock day trading system specifically optimized for equities markets.
    Features adaptive strategy selection, multi-timeframe analysis, and session-based 
    trading with safety mechanisms.
    """
    
    def __init__(self, config_path: str, session_duration: SessionDuration = SessionDuration.FOUR_HOURS):
        """
        Initialize the stock trading system.
        
        Args:
            config_path: Path to the configuration file
            session_duration: Duration of the trading session
        """
        self.config_path = config_path
        self.session_duration = session_duration
        self.config = {}
        
        # Load configuration
        self.load_config(config_path)
        
        # Initialize core services
        self.market_data = MarketDataService(provider_name=self.config.get("market_data_provider"))
        self.trading_service = TradingService(is_paper=not self.config.get("live_trading", False))
        self.strategy_manager = StrategyManager()
        self.event_bus = EventBus()
        
        # Initialize advanced components
        self.market_analyzer = MarketAnalyzer(
            volatility_window=self.config.get("volatility_window", 20),
            trend_window=self.config.get("trend_window", 50),
            volume_window=self.config.get("volume_window", 10)
        )
        
        self.risk_manager = RiskManager(
            initial_capital=self.config.get("initial_capital", 10000.0),
            max_position_size_pct=self.config.get("max_position_size_pct", 0.05),
            max_portfolio_risk_pct=self.config.get("max_portfolio_risk_pct", 0.5),
            base_risk_per_trade_pct=self.config.get("base_risk_per_trade_pct", 0.01)
        )
        
        # Strategy components
        self.active_strategies = {}
        
        # Trading state
        self.portfolio_value = 0.0
        self.cash = 0.0
        self.positions = {}
        self.pending_orders = {}
        self.transaction_costs = 0.0
        self.market_condition = MarketCondition.NORMAL
        self.trading_paused = False
        self.pause_reason = ""
        
        # Session tracking
        self.session_start_time = None
        self.session_end_time = None
        self.trades_this_session = 0
        self.pnl_this_session = 0.0
        
        # Performance tracking
        self.performance_history = []
        self.last_strategy_update = datetime.now()
        self.last_risk_adjustment = datetime.now()
        self.last_market_check = datetime.now()
        
        # Symbol analysis cache
        self.symbol_analysis = {}
        self.symbol_last_update = {}
        
        logger.info(f"StockTrader initialized with configuration: {config_path}")
    
    def load_config(self, config_path: str) -> None:
        """Load configuration from a JSON file"""
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
                
            logger.info(f"Configuration loaded successfully: {len(self.config.get('symbols', []))} symbols, "
                       f"{len(self.config.get('strategies', []))} strategies")
                       
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    async def initialize(self) -> None:
        """Initialize all services and strategies"""
        try:
            logger.info("Initializing services and strategies...")
            
            # Load all configured strategies
            for strategy_name in self.config.get("strategies", []):
                strategy_params = self.config.get("strategy_params", {}).get(strategy_name, {})
                
                try:
                    # Get strategy from the strategy manager
                    strategy = await self.strategy_manager.get_strategy(strategy_name, strategy_params)
                    self.active_strategies[strategy_name] = strategy
                    
                    logger.info(f"Strategy {strategy_name} loaded successfully")
                    
                except Exception as e:
                    logger.error(f"Error loading strategy {strategy_name}: {e}")
            
            # Initialize strategy selector with loaded strategies
            self.strategy_selector = StrategySelector(
                strategies=self.active_strategies,
                lookback_period=self.config.get("lookback_period", 20),
                performance_weight=self.config.get("performance_weight", 0.7),
                regime_weight=self.config.get("regime_weight", 0.3)
            )
            
            # Get account information
            account_info = await self.trading_service.get_account_info()
            self.portfolio_value = account_info.get("portfolio_value", 0.0)
            # Connect to Alpaca account
            account = await self.trading_service.get_account_info()
            if account:
                self.portfolio_value = account.get("portfolio_value", 0.0)
                self.peak_portfolio_value = self.portfolio_value
                
                logger.info(f"Connected to account: {account.get('id', 'unknown')}")
                logger.info(f"Account status: {account.get('status', 'unknown')}")
                logger.info(f"Portfolio value: ${self.portfolio_value:.2f}")
            
            # Subscribe to market data events
            try:
                await self.event_bus.subscribe("market_data_update", self._handle_market_data_update)
                await self.event_bus.subscribe("market_anomaly", self._handle_market_anomaly)
                await self.event_bus.subscribe("excessive_loss", self._handle_excessive_loss)
                await self.event_bus.subscribe("strategy_signal", self._handle_strategy_signal)
                logger.info("Successfully subscribed to all event streams")
            except Exception as e:
                logger.warning(f"Could not subscribe to market data events: {e}")
                logger.warning("Continuing without event subscription - will use polling instead")
            
            logger.info(f"Stock trading system initialized with {len(self.active_strategies)} active strategies")
            
        except Exception as e:
            logger.error(f"Error initializing stock trading system: {e}")
            raise
    
    async def start(self, duration_seconds: Optional[int] = None) -> None:
        """Start the stock trading system with the specified session duration"""
        try:
            # Set session duration
            if duration_seconds is not None and duration_seconds > 0:
                self.session_duration = duration_seconds
            
            # Calculate session end time
            global session_end_time
            self.session_start_time = datetime.now()
            
            if self.session_duration != SessionDuration.CUSTOM:
                self.session_end_time = self.session_start_time + timedelta(seconds=self.session_duration)
                session_end_time = self.session_end_time
            else:
                self.session_end_time = None
                session_end_time = None
            
            # Initialize the system
            await self.initialize()
            
            # Display session information
            logger.info("===================================================")
            logger.info("STOCK TRADING SESSION STARTING")
            logger.info("===================================================")
            logger.info(f"Mode: {'LIVE TRADING' if self.config.get('live_trading', False) else 'PAPER TRADING'}")
            logger.info(f"Session start: {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            if self.session_end_time:
                logger.info(f"Session end: {self.session_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"Duration: {timedelta(seconds=self.session_duration)}")
            else:
                logger.info("Session end: Manual stop required")
                logger.info("Duration: Unlimited (manual stop)")
                
            logger.info(f"Portfolio value: ${self.portfolio_value:.2f}")
            logger.info(f"Strategies: {', '.join(list(self.active_strategies.keys()))}")
            logger.info(f"Symbols: {', '.join(self.config.get('symbols', []))}")
            logger.info(f"Check interval: {self.config.get('check_interval_seconds', 60)} seconds")
            logger.info("===================================================")
            
            # Confirm before starting live trading
            if self.config.get("live_trading", False):
                confirmation = input("Type 'CONFIRM' to start live trading or anything else to cancel: ")
                
                if confirmation != "CONFIRM":
                    logger.info("Live trading cancelled by user")
                    return
            else:
                # Auto-confirm paper trading if specified
                if not self.config.get("auto_confirm_paper", True):
                    confirmation = input("Type 'CONFIRM' to start paper trading or anything else to cancel: ")
                    
                    if confirmation != "CONFIRM":
                        logger.info("Paper trading cancelled by user")
                        return
            
            logger.info("Stock trading session confirmed. Starting main trading loop...")
            
            # Main trading loop
            await self.trading_loop()
            
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
        except Exception as e:
            logger.error(f"Critical error in stock trading system: {e}")
        finally:
            await self.generate_performance_report()
            
    async def trading_loop(self) -> None:
        """Main trading loop with session-based execution"""
        global running
        running = True
        
        check_interval = self.config.get("check_interval_seconds", 60)
        # More frequent updates for short sessions
        if self.session_duration < SessionDuration.FOUR_HOURS:
            check_interval = min(check_interval, 30)  # 30 seconds maximum for short sessions
        
        # Track heartbeats for system health monitoring
        last_heartbeat = datetime.now()
        heartbeat_interval = 300  # 5 minutes
        
        while running:
            try:
                # Check if session should end
                now = datetime.now()
                
                # Session timeout check
                if self.session_end_time and now >= self.session_end_time:
                    logger.info("Session duration reached. Ending trading session.")
                    break
                
                # Heartbeat logging
                if (now - last_heartbeat).total_seconds() >= heartbeat_interval:
                    remaining = ""
                    if self.session_end_time:
                        remaining_seconds = (self.session_end_time - now).total_seconds()
                        if remaining_seconds > 0:
                            hours, remainder = divmod(remaining_seconds, 3600)
                            minutes, seconds = divmod(remainder, 60)
                            remaining = f" - Remaining: {int(hours)}h {int(minutes)}m"
                    
                    logger.info(f"System heartbeat - Running for {(now - self.session_start_time).total_seconds()/60:.1f} minutes{remaining}")
                    last_heartbeat = now
                
                # Check market status and conditions
                await self.check_market_conditions()
                
                if self.market_condition == MarketCondition.CLOSED:
                    next_check = min(300, check_interval * 5)  # 5 minutes or 5x normal interval
                    logger.info(f"Market closed. Waiting {next_check} seconds before next check...")
                    
                    # Continue if configured to ignore market hours
                    if not self.config.get("ignore_market_hours", False):
                        await asyncio.sleep(next_check)
                        continue
                    else:
                        logger.info("Ignoring market hours, continuing in simulation mode...")
                
                # Handle paused trading
                if self.trading_paused:
                    # Check if we should resume trading
                    if self.should_resume_trading():
                        self.trading_paused = False
                        logger.info(f"Resuming trading. Previous pause reason: {self.pause_reason}")
                        self.pause_reason = ""
                    else:
                        logger.info(f"Trading remains paused: {self.pause_reason}. Waiting {check_interval} seconds...")
                        await asyncio.sleep(check_interval)
                        continue
                
                # Update portfolio state
                await self.update_portfolio_state()
                
                # Process symbols
                if self.market_condition != MarketCondition.DANGEROUS and not self.trading_paused:
                    # Process symbols in parallel to speed up execution
                    tasks = []
                    for symbol in self.config.get("symbols", []):
                        tasks.append(self.process_symbol(symbol))
                    
                    if tasks:
                        await asyncio.gather(*tasks)
                
                # Periodically adjust risk level
                if (now - self.last_risk_adjustment).total_seconds() > self.config.get("risk_adjustment_interval_seconds", 3600):
                    self.risk_manager.adjust_risk_level()
                    self.last_risk_adjustment = now
                
                # Update strategy weights periodically
                if (now - self.last_strategy_update).total_seconds() > self.config.get("strategy_update_interval_seconds", 1800):
                    await self.update_strategy_weights()
                    self.last_strategy_update = now
                
                # Wait between iterations
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(check_interval * 2)  # Longer pause on error
    
    async def check_market_conditions(self) -> None:
        """Check overall market conditions and decide whether to continue trading"""
        try:
            # Only perform this check periodically (e.g., every 5 minutes)
            now = datetime.now()
            if (now - self.last_market_check).total_seconds() < self.config.get("market_check_interval_seconds", 300):
                return
                
            self.last_market_check = now
            
            # Check if market is open
            is_open = await self.trading_service.is_market_open()
            
            if not is_open:
                self.market_condition = MarketCondition.CLOSED
                return
                
            # Get market index data (e.g., S&P 500, Nasdaq, etc.)
            market_symbols = self.config.get("market_indices", ["SPY"])
            market_data = {}
            
            try:
                for symbol in market_symbols:
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=10)  # 10 days of data
                    
                    data = await self.market_data.get_historical_data(
                        symbol, start_date, end_date, timeframe="1Day"
                    )
                    
                    if data is not None and len(data) > 0:
                        market_data[symbol] = data
            except Exception as e:
                logger.error(f"Error retrieving market data for {symbol}: {e}")
                
            if not market_data:
                logger.warning("Could not retrieve market index data for condition assessment")
                return
                
            # Analyze market conditions
            volatility_scores = []
            anomaly_scores = []
            
            for symbol, data in market_data.items():
                analysis = self.market_analyzer.analyze_market_data(symbol, data)
                
                # Check for anomalies
                if analysis.get("anomalies", {}).get("detected", False):
                    anomaly_score = analysis.get("anomalies", {}).get("manipulation_probability", 0)
                    anomaly_scores.append(anomaly_score)
                
                # Get volatility data
                volatility = analysis.get("volatility", {})
                vol_percentile = volatility.get("percentile", 50)
                volatility_scores.append(vol_percentile)
            
            # Average volatility and anomaly scores
            avg_volatility = sum(volatility_scores) / len(volatility_scores) if volatility_scores else 50
            max_anomaly = max(anomaly_scores) if anomaly_scores else 0
            
            # Determine market condition
            if max_anomaly > 0.7:
                new_condition = MarketCondition.DANGEROUS
                reason = f"High anomaly score detected: {max_anomaly:.2f}"
            elif avg_volatility > 85:
                new_condition = MarketCondition.VOLATILE
                reason = f"High market volatility: {avg_volatility:.1f} percentile"
            elif avg_volatility < 15:
                new_condition = MarketCondition.INACTIVE
                reason = f"Low market volatility: {avg_volatility:.1f} percentile"
            else:
                new_condition = MarketCondition.NORMAL
                reason = f"Normal market conditions: {avg_volatility:.1f} percentile volatility"
            
            # Log if condition changes
            if new_condition != self.market_condition:
                logger.info(f"Market condition changed: {self.market_condition} -> {new_condition} ({reason})")
                
                # Pause trading if conditions dangerous
                if new_condition == MarketCondition.DANGEROUS and not self.trading_paused:
                    self.trading_paused = True
                    self.pause_reason = f"Dangerous market conditions: {reason}"
                    logger.warning(f"Trading PAUSED: {self.pause_reason}")
            
            self.market_condition = new_condition
            
        except Exception as e:
            logger.error(f"Error checking market conditions: {e}")
            # Use fallback normal market condition to avoid stopping trading completely
            return MarketCondition.NORMAL
    
    def should_resume_trading(self) -> bool:
        """Determine if paused trading should be resumed"""
        # Resume if market condition has improved
        if self.market_condition == MarketCondition.NORMAL:
            return True
            
        # Resume after timeout period if not in dangerous conditions
        pause_duration = (datetime.now() - self.last_market_check).total_seconds()
        max_pause_duration = self.config.get("stock", {}).get("max_pause_duration_seconds", 1800)  # 30 minutes default
        
        if pause_duration > max_pause_duration and self.market_condition != MarketCondition.DANGEROUS:
            return True
            
        return False
        
    async def _handle_market_data_update(self, data: Dict[str, Any]) -> None:
        """Handle market data updates from event stream"""
        try:
            symbol = data.get("symbol")
            price = data.get("price")
            timestamp = data.get("timestamp")
            
            if not symbol or not price:
                return
                
            logger.debug(f"Received market data update: {symbol} @ ${price} [{timestamp}]")
            
            # Update cached prices
            self.latest_prices[symbol] = price
            
            # Check for active positions with this symbol and update unrealized P/L
            if symbol in self.positions:
                position = self.positions[symbol]
                entry_price = position.get("avg_entry_price", 0)
                quantity = position.get("quantity", 0)
                
                if entry_price > 0 and quantity != 0:
                    pl = (price - entry_price) * quantity
                    pl_pct = (price / entry_price - 1) * 100 * (1 if quantity > 0 else -1)
                    
                    # Update position data
                    position["current_price"] = price
                    position["unrealized_pl"] = pl
                    position["unrealized_plpc"] = pl_pct / 100  # Store as decimal
                    
                    # Check for excessive losses
                    stop_loss_pct = position.get("stop_loss_pct", 0.05) * 100
                    
                    if pl_pct < -stop_loss_pct:
                        logger.warning(f"Position {symbol} reached stop loss threshold: {pl_pct:.2f}% loss")
                        
                        # Emit excessive loss event
                        await self.event_bus.emit("excessive_loss", {
                            "symbol": symbol,
                            "loss_percentage": abs(pl_pct),
                            "position": position
                        })
        except Exception as e:
            logger.error(f"Error handling market data update: {e}")
            
    async def _handle_market_anomaly(self, data: Dict[str, Any]) -> None:
        """Handle market anomaly events"""
        try:
            symbol = data.get("symbol", "unknown")
            anomaly_type = data.get("type", "unknown")
            severity = data.get("severity", 0)
            
            logger.warning(f"Market anomaly detected: {anomaly_type} for {symbol} (severity: {severity:.2f})")
            
            # Pause trading if severe anomaly detected
            if severity > 0.7 and not self.trading_paused:
                self.trading_paused = True
                self.pause_reason = f"Severe market anomaly: {anomaly_type} for {symbol}"
                logger.warning(f"Trading PAUSED: {self.pause_reason}")
        except Exception as e:
            logger.error(f"Error handling market anomaly: {e}")
            
    async def _handle_excessive_loss(self, data: Dict[str, Any]) -> None:
        """Handle excessive loss events"""
        try:
            symbol = data.get("symbol", "unknown")
            loss_pct = data.get("loss_percentage", 0)
            
            logger.warning(f"Excessive loss detected for {symbol}: {loss_pct:.2f}%")
            
            # Check against max loss threshold from config
            max_loss_pct = self.config.get("stock", {}).get("max_daily_loss_percentage", 5.0)
            
            if loss_pct > max_loss_pct and not self.trading_paused:
                self.trading_paused = True
                self.pause_reason = f"Excessive loss for {symbol}: {loss_pct:.2f}% exceeded threshold of {max_loss_pct}%"
                logger.warning(f"Trading PAUSED: {self.pause_reason}")
        except Exception as e:
            logger.error(f"Error handling excessive loss: {e}")
            
    async def _handle_strategy_signal(self, data: Dict[str, Any]) -> None:
        """Handle strategy signals from event-based strategies"""
        try:
            symbol = data.get("symbol")
            action = data.get("action")
            strategy = data.get("strategy")
            confidence = data.get("confidence", 0.5)
            
            if not symbol or not action or action == "hold":
                return
                
            logger.info(f"Strategy signal received: {action} {symbol} from {strategy} (confidence: {confidence:.2f})")
            
            # Get latest price for the symbol
            latest_price = self.latest_prices.get(symbol)
            
            if not latest_price:
                latest_price = await self.market_data.get_latest_price(symbol)
                if not latest_price:
                    logger.warning(f"Cannot execute signal: no price available for {symbol}")
                    return
                    
            # Get position information
            position = await self.trading_service.get_position(symbol)
            
            # Calculate risk parameters
            stop_loss = data.get("stop_loss", latest_price * 0.95)  # Default 5% stop loss
            take_profit = data.get("take_profit", latest_price * 1.15)  # Default 15% take profit
            
            risk_params = self.risk_manager.calculate_position_size(
                symbol, 
                latest_price, 
                stop_loss
            )
            
            # Build complete signal
            complete_signal = {
                "symbol": symbol,
                "action": action,
                "strategy": strategy,
                "confidence": confidence,
                "price": latest_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                **risk_params
            }
            
            # Execute the signal
            await self.execute_signal(symbol, complete_signal, latest_price, position)
        except Exception as e:
            logger.error(f"Error handling strategy signal: {e}")
        
    async def process_symbol(self, symbol: str) -> None:
        """Process a single trading symbol"""
        try:
            # Get most recent price data
            latest_price = await self.market_data.get_latest_price(symbol)
            if not latest_price:
                logger.warning(f"Unable to get latest price for {symbol}")
                return
                
            # Get historical data for analysis with retry logic
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)  # 30 days of data
            
            # Implement retry with backoff for data fetching
            max_retries = 3
            retry_count = 0
            backoff_factor = 2
            data = None
            
            while retry_count < max_retries and data is None:
                try:
                    data = await self.market_data.get_historical_data(
                        symbol, start_date, end_date, timeframe="1Day"
                    )
                    
                    if data is None or len(data) == 0:
                        retry_count += 1
                        wait_time = backoff_factor ** retry_count
                        logger.warning(f"No data received for {symbol}, retry {retry_count}/{max_retries} after {wait_time}s")
                        await asyncio.sleep(wait_time)
                    
                except Exception as e:
                    retry_count += 1
                    wait_time = backoff_factor ** retry_count
                    logger.warning(f"Error fetching data for {symbol}: {e}, retry {retry_count}/{max_retries} after {wait_time}s")
                    await asyncio.sleep(wait_time)
            
            if data is None or len(data) == 0:
                logger.warning(f"No historical data available for {symbol}")
                return
                
            # Calculate indicators and analyze market data
            analysis = self.market_analyzer.analyze_market_data(symbol, data)
            
            # Log market analysis summary for the symbol
            volatility = analysis.get("volatility", {}).get("value", 0.0)
            trend = analysis.get("trend", {}).get("direction", "neutral")
            rsi = analysis.get("indicators", {}).get("rsi", 50)
            
            logger.debug(f"Symbol: {symbol} - Price: ${latest_price:.2f} - Trend: {trend} - Vol: {volatility:.2f} - RSI: {rsi:.1f}")
            
            # Get current holdings for this symbol
            position = await self.trading_service.get_position(symbol)
            
            # Execute signal generation through strategy selector
            signal = await self.strategy_selector.generate_signal(symbol, data, analysis, position)
            
            if signal and signal.get("action") != "hold":
                # Calculate position size and risk parameters
                risk_params = self.risk_manager.calculate_position_size(
                    symbol, 
                    latest_price, 
                    signal.get("stop_loss", latest_price * 0.95)
                )
                
                signal.update(risk_params)
                
                # Execute the trading signal
                await self.execute_signal(symbol, signal, latest_price, position)
                
        except Exception as e:
            logger.error(f"Error processing symbol {symbol}: {e}")
            # Log backtrace for easier debugging
            import traceback
            logger.debug(f"Symbol processing error details: {traceback.format_exc()}")
            
    async def execute_signal(self, symbol: str, signal: Dict[str, Any], current_price: float, position: Optional[Dict[str, Any]]) -> None:
        """Execute a trading signal for a symbol"""
        action = signal.get("action")
        quantity = signal.get("quantity", 0)
        strategy = signal.get("strategy", "unknown")
        confidence = signal.get("confidence", 0.5)
        stop_loss = signal.get("stop_loss")
        take_profit = signal.get("take_profit")
        
        if action not in ["buy", "sell", "hold"]:
            logger.warning(f"Unknown action '{action}' for symbol {symbol}")
            return
            
        if action == "hold":
            return
            
        # Log the signal
        logger.info(f"SIGNAL: {action.upper()} {symbol} - Quantity: {quantity} - Price: ${current_price:.2f} ")
        logger.info(f"  Strategy: {strategy} - Confidence: {confidence:.2f} - Stop: ${stop_loss:.2f} - Target: ${take_profit:.2f}")
        
        # Skip if quantity too small
        if quantity <= 0:
            logger.warning(f"Skipping {action} for {symbol} due to quantity <= 0")
            return
            
        # Skip if we're in paper trading below confidence threshold
        if not self.config.get("live_trading", False) and confidence < self.config.get("min_confidence_threshold", 0.6):
            logger.info(f"Skipping {action} for {symbol} due to low confidence: {confidence:.2f}")
            return
            
        # Apply additional risk checks before execution
        if not self.risk_manager.validate_trade(symbol, action, quantity, current_price):
            logger.warning(f"Trade rejected by risk manager: {action} {symbol}")
            return
            
        # Execute the trade
        try:
            result = await self.trading_service.execute_order(
                symbol=symbol,
                action=action,
                quantity=quantity,
                order_type="market",
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            if result and result.get("success"):
                # Update trade history
                self.trade_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "symbol": symbol,
                    "action": action,
                    "quantity": quantity,
                    "price": current_price,
                    "strategy": strategy,
                    "confidence": confidence
                })
                
                # Log trade execution
                logger.info(f"EXECUTED: {action.upper()} {symbol} - Quantity: {quantity} - Price: ${current_price:.2f}")
                
                # Update strategy performance statistics
                self.strategy_selector.update_strategy_performance(strategy, symbol, action, confidence)
            else:
                error = result.get("error", "Unknown error") if result else "No result returned"
                logger.error(f"Trade execution failed: {error}")
                
        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
            
    async def update_portfolio_state(self) -> None:
        """Update portfolio value and positions"""
        try:
            # Get account information
            account = await self.trading_service.get_account_info()
            
            if account:
                # Update portfolio value
                previous_value = self.portfolio_value
                self.portfolio_value = account.get("portfolio_value", 0.0)
                
                # Calculate daily change
                if previous_value > 0:
                    daily_change_pct = (self.portfolio_value - previous_value) / previous_value * 100
                    
                    # Log significant changes
                    if abs(daily_change_pct) >= 1.0:
                        direction = "up" if daily_change_pct > 0 else "down"
                        logger.info(f"Portfolio value {direction} by {abs(daily_change_pct):.2f}% to ${self.portfolio_value:.2f}")
                        
                        # Check for excessive loss
                        max_daily_loss_pct = self.config.get("max_daily_loss_percentage", 5.0)
                        if daily_change_pct <= -max_daily_loss_pct and not self.trading_paused:
                            self.trading_paused = True
                            self.pause_reason = f"Excessive daily loss: {abs(daily_change_pct):.2f}% exceeded threshold of {max_daily_loss_pct}%"
                            logger.warning(f"Trading PAUSED: {self.pause_reason}")
                
                # Update positions
                self.positions = await self.trading_service.get_positions()
                
                # Check max drawdown
                if self.portfolio_value > self.peak_portfolio_value:
                    self.peak_portfolio_value = self.portfolio_value
                elif self.peak_portfolio_value > 0:
                    drawdown_pct = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value * 100
                    max_drawdown_pct = self.config.get("max_drawdown_percentage", 15.0)
                    
                    if drawdown_pct >= max_drawdown_pct and not self.trading_paused:
                        self.trading_paused = True
                        self.pause_reason = f"Maximum drawdown reached: {drawdown_pct:.2f}% exceeded threshold of {max_drawdown_pct}%"
                        logger.warning(f"Trading PAUSED: {self.pause_reason}")
            else:
                logger.warning("Failed to retrieve account information")
                
        except Exception as e:
            logger.error(f"Error updating portfolio state: {e}")
            
    async def update_strategy_weights(self) -> None:
        """Update strategy weights based on performance metrics"""
        try:
            # Get strategy performance statistics from strategy selector
            performance_stats = self.strategy_selector.get_strategy_performance()
            
            if not performance_stats:
                return
                
            # Log strategy performance
            logger.info("Strategy Performance Update:")
            for strategy, stats in performance_stats.items():
                win_rate = stats.get("win_rate", 0.0) * 100
                profit_factor = stats.get("profit_factor", 1.0)
                weight = stats.get("weight", 0.0) * 100
                
                logger.info(f"  {strategy}: Win Rate: {win_rate:.1f}% - Profit Factor: {profit_factor:.2f} - Weight: {weight:.1f}%")
                
            # Update strategy weights based on performance
            self.strategy_selector.update_weights_based_on_performance()
            
        except Exception as e:
            logger.error(f"Error updating strategy weights: {e}")
            
    async def generate_performance_report(self) -> None:
        """Generate a performance report at the end of the trading session"""
        try:
            logger.info("===================================================")
            logger.info("STOCK TRADING SESSION PERFORMANCE REPORT")
            logger.info("===================================================")
            
            # Session duration
            end_time = datetime.now()
            duration = end_time - self.session_start_time
            hours, remainder = divmod(duration.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            
            logger.info(f"Session Duration: {int(hours)}h {int(minutes)}m {int(seconds)}s")
            logger.info(f"Start Time: {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Portfolio performance
            start_value = self.config.get("initial_portfolio_value", 0.0)
            end_value = self.portfolio_value
            
            if start_value > 0:
                total_return_pct = (end_value - start_value) / start_value * 100
                logger.info(f"Starting Portfolio Value: ${start_value:.2f}")
                logger.info(f"Ending Portfolio Value: ${end_value:.2f}")
                logger.info(f"Total Return: {total_return_pct:.2f}%")
                
                # Annualized return (if session longer than 1 hour)
                if duration.total_seconds() > 3600:
                    days = duration.total_seconds() / 86400  # Convert to days
                    if days > 0:
                        ann_return = ((1 + total_return_pct/100) ** (365/days) - 1) * 100
                        logger.info(f"Annualized Return: {ann_return:.2f}%")
            
            # Trade statistics
            if self.trade_history:
                trades_count = len(self.trade_history)
                buys = sum(1 for trade in self.trade_history if trade.get("action") == "buy")
                sells = sum(1 for trade in self.trade_history if trade.get("action") == "sell")
                
                logger.info(f"Total Trades: {trades_count}")
                logger.info(f"  Buys: {buys}")
                logger.info(f"  Sells: {sells}")
                
                # Strategy usage
                strategy_counts = {}
                for trade in self.trade_history:
                    strategy = trade.get("strategy", "unknown")
                    if strategy in strategy_counts:
                        strategy_counts[strategy] += 1
                    else:
                        strategy_counts[strategy] = 1
                        
                logger.info("Strategy Usage:")
                for strategy, count in strategy_counts.items():
                    logger.info(f"  {strategy}: {count} trades ({count/trades_count*100:.1f}%)")
            else:
                logger.info("No trades executed during this session")
                
            # Current positions
            if self.positions:
                logger.info("Current Positions:")
                for position in self.positions:
                    symbol = position.get("symbol")
                    quantity = position.get("quantity", 0)
                    avg_price = position.get("avg_entry_price", 0.0)
                    market_value = position.get("market_value", 0.0)
                    unrealized_pl = position.get("unrealized_pl", 0.0)
                    unrealized_plpc = position.get("unrealized_plpc", 0.0) * 100
                    
                    logger.info(f"  {symbol}: {quantity} shares at ${avg_price:.2f} - Value: ${market_value:.2f} - P/L: ${unrealized_pl:.2f} ({unrealized_plpc:.2f}%)")
            else:
                logger.info("No open positions at session end")
                
            # Strategy performance
            performance_stats = self.strategy_selector.get_strategy_performance()
            if performance_stats:
                logger.info("Strategy Performance:")
                for strategy, stats in performance_stats.items():
                    win_rate = stats.get("win_rate", 0.0) * 100
                    profit_factor = stats.get("profit_factor", 1.0)
                    avg_return = stats.get("avg_return", 0.0) * 100
                    
                    logger.info(f"  {strategy}: Win Rate: {win_rate:.1f}% - Profit Factor: {profit_factor:.2f} - Avg Return: {avg_return:.2f}%")
            
            logger.info("===================================================")
            logger.info("STOCK TRADING SESSION COMPLETED")
            logger.info("===================================================")
                
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            
    def __del__(self):
        """Cleanup resources when object is destroyed"""
        logger.info("Stock trading system shutting down...")
        
# Main entry point
def main():
    parser = argparse.ArgumentParser(description="Stock Day Trading System")
    parser.add_argument("--duration", type=str, choices=["1h", "4h", "8h", "custom"], default="4h",
                        help="Trading session duration (1h, 4h, 8h, or custom)")
    parser.add_argument("--custom-seconds", type=int, default=0,
                        help="Custom duration in seconds if --duration=custom")
    parser.add_argument("--config", type=str, default="config/daytrader_config.json",
                        help="Path to configuration file")
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                        default="INFO", help="Logging level")
                        
    args = parser.parse_args()
    
    # Set logging level
    numeric_level = getattr(logging, args.log_level)
    logging.basicConfig(level=numeric_level, format="%(asctime)s [%(levelname)s] %(message)s")
    
    # Determine session duration
    duration_map = {
        "1h": SessionDuration.ONE_HOUR,
        "4h": SessionDuration.FOUR_HOURS,
        "8h": SessionDuration.EIGHT_HOURS,
        "custom": SessionDuration.CUSTOM
    }
    session_duration = duration_map.get(args.duration, SessionDuration.FOUR_HOURS)
    custom_duration = args.custom_seconds if args.duration == "custom" else 0
    
    # Create and run trader
    trader = StockDayTrader(config_path=args.config, session_duration=session_duration)
    
    # Register signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        global running, session_end_time
        logger.info(f"Received signal {sig}, shutting down...")
        running = False
        session_end_time = datetime.now()
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the trader in the async event loop
    loop = asyncio.get_event_loop()
    try:
        if custom_duration > 0:
            loop.run_until_complete(trader.start(custom_duration))
        else:
            loop.run_until_complete(trader.start())
    finally:
        loop.close()

if __name__ == "__main__":
    main()
