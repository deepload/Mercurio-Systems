#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Crypto Day Trading System for Mercurio AI
-----------------------------------------
This script implements an adaptive cryptocurrency trading system that can run
for specified durations (1h, 4h, 8h) and adjusts strategies based on 
market conditions, volatility, and historical performance.

Usage:
    python run_crypto_daytrader.py --duration 4h --config config/daytrader_config.json
"""

import os
import sys
import json
import time
import signal
import logging
import argparse
import asyncio
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import Mercurio modules
from app.services.market_data import MarketDataService
from app.services.trading import TradingService
from app.strategies.base import BaseStrategy
from app.strategies.adaptive.strategy_selector import StrategySelector
from app.strategies.adaptive.market_analyzer import MarketAnalyzer
from app.strategies.adaptive.risk_manager import RiskManager
from app.core.event_bus import EventBus

# Configure logger
logger = logging.getLogger("crypto_trader")

# Global variables for signal handling
running = True
session_end_time = None

# Enums for session duration and market conditions
class SessionDuration(int, Enum):
    ONE_HOUR = 3600
    FOUR_HOURS = 14400
    EIGHT_HOURS = 28800
    CUSTOM = 0

class MarketCondition(Enum):
    NORMAL = auto()
    VOLATILE = auto()
    INACTIVE = auto()
    DANGEROUS = auto()
    CLOSED = auto()  # Not relevant for crypto, but kept for consistency

class CryptoDayTrader:
    """
    Adaptive Cryptocurrency Day Trading System
    
    Features:
    - Dynamic strategy selection based on market conditions
    - Risk management with position sizing and drawdown protection
    - Multiple session durations (1h, 4h, 8h)
    - Detailed logging and performance tracking
    - Support for multiple exchanges
    """
    
    def __init__(self, config_path: str = "config/daytrader_config.json", 
                 session_duration: SessionDuration = SessionDuration.FOUR_HOURS):
        """Initialize the cryptocurrency trading system"""
        self.config_path = config_path
        self.session_duration = session_duration
        self.config = {}
        
        # Trading state
        self.market_data = None
        self.trading_service = None
        self.strategy_selector = None
        self.market_analyzer = None
        self.risk_manager = None
        self.event_bus = None
        
        # Market state
        self.market_condition = MarketCondition.NORMAL
        self.trading_paused = False
        self.pause_reason = ""
        self.positions = []
        self.trade_history = []
        
        # Performance tracking
        self.portfolio_value = 0.0
        self.peak_portfolio_value = 0.0
        self.session_start_time = datetime.now()
        self.session_end_time = None
        
        # Timestamps for periodic updates
        self.last_market_check = datetime.now()
        self.last_risk_adjustment = datetime.now()
        self.last_strategy_update = datetime.now()
        
        # Active strategies
        self.active_strategies = {}
        
        logger.info("Crypto Day Trading System initialized")
        
    async def initialize(self) -> None:
        """Initialize services and load configuration"""
        try:
            # Load configuration
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            
            # Extract crypto-specific configuration
            crypto_config = self.config.get("crypto", {})
            if not crypto_config:
                raise ValueError("Crypto configuration not found in config file")
                
            # Initialize event bus
            self.event_bus = EventBus()
            
            # Initialize services
            self.market_data = MarketDataService(
                api_keys=crypto_config.get("api_keys", {}),
                exchange=crypto_config.get("exchange", "binance")
            )
            
            self.trading_service = TradingService(
                api_keys=crypto_config.get("api_keys", {}),
                exchange=crypto_config.get("exchange", "binance"),
                paper_trading=not crypto_config.get("live_trading", False),
                event_bus=self.event_bus
            )
            
            # Initialize market analyzer
            self.market_analyzer = MarketAnalyzer()
            
            # Initialize risk manager
            self.risk_manager = RiskManager(
                initial_risk_level=crypto_config.get("initial_risk_level", 0.5),
                max_portfolio_risk=crypto_config.get("max_portfolio_risk", 0.1),
                event_bus=self.event_bus
            )
            
            # Initialize trading strategies
            strategies = {}
            for strategy_name, strategy_config in crypto_config.get("strategies", {}).items():
                strategy_class = self._get_strategy_class(strategy_name)
                if strategy_class:
                    strategies[strategy_name] = strategy_class(
                        config=strategy_config,
                        market_data=self.market_data,
                        event_bus=self.event_bus
                    )
            
            # Initialize strategy selector
            self.strategy_selector = StrategySelector(
                strategies=strategies,
                config=crypto_config.get("strategy_selector", {}),
                market_analyzer=self.market_analyzer,
                event_bus=self.event_bus
            )
            
            # Store active strategies for reference
            self.active_strategies = strategies
            
            # Get initial account information
            account = await self.trading_service.get_account()
            self.portfolio_value = account.get("portfolio_value", 0.0)
            self.peak_portfolio_value = self.portfolio_value
            
            # Subscribe to relevant events
            self.event_bus.subscribe("market_anomaly_detected", self._handle_market_anomaly)
            self.event_bus.subscribe("excessive_loss_detected", self._handle_excessive_loss)
            
            logger.info(f"Crypto trading system initialized with {len(strategies)} strategies")
            logger.info(f"Initial portfolio value: ${self.portfolio_value:.2f}")
            
        except Exception as e:
            logger.error(f"Error initializing crypto trading system: {e}")
            raise
    
    def _get_strategy_class(self, strategy_name: str) -> Optional[BaseStrategy]:
        """Get strategy class by name"""
        # Import here to avoid circular imports
        from app.strategies.momentum import MomentumStrategy
        from app.strategies.mean_reversion import MeanReversionStrategy
        from app.strategies.breakout import BreakoutStrategy
        from app.strategies.statistical_arbitrage import StatisticalArbitrageStrategy
        
        strategy_map = {
            "momentum": MomentumStrategy,
            "mean_reversion": MeanReversionStrategy,
            "breakout": BreakoutStrategy,
            "stat_arb": StatisticalArbitrageStrategy
        }
        
        return strategy_map.get(strategy_name.lower())
        
    async def _handle_market_anomaly(self, data: Dict[str, Any]) -> None:
        """Handle market anomaly events"""
        symbol = data.get("symbol", "unknown")
        anomaly_type = data.get("type", "unknown")
        severity = data.get("severity", 0.0)
        
        logger.warning(f"Market anomaly detected for {symbol}: {anomaly_type} - Severity: {severity:.2f}")
        
        # Pause trading if severe anomaly detected
        if severity > 0.8 and not self.trading_paused:
            self.trading_paused = True
            self.pause_reason = f"Severe market anomaly: {anomaly_type} for {symbol}"
            logger.warning(f"Trading PAUSED: {self.pause_reason}")
    
    async def _handle_excessive_loss(self, data: Dict[str, Any]) -> None:
        """Handle excessive loss events"""
        symbol = data.get("symbol", "unknown")
        loss_pct = data.get("loss_percentage", 0.0)
        
        logger.warning(f"Excessive loss detected for {symbol}: {loss_pct:.2f}%")
        
        # Pause trading on excessive loss
        if not self.trading_paused:
            self.trading_paused = True
            self.pause_reason = f"Excessive loss for {symbol}: {loss_pct:.2f}%"
            logger.warning(f"Trading PAUSED: {self.pause_reason}")
            
    async def start(self, duration_seconds: Optional[int] = None) -> None:
        """Start the crypto trading system with the specified session duration"""
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
            logger.info("CRYPTO TRADING SESSION STARTING")
            logger.info("===================================================")
            logger.info(f"Mode: {'LIVE TRADING' if self.config.get('crypto', {}).get('live_trading', False) else 'PAPER TRADING'}")
            logger.info(f"Exchange: {self.config.get('crypto', {}).get('exchange', 'binance')}")
            logger.info(f"Session start: {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            if self.session_end_time:
                logger.info(f"Session end: {self.session_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"Duration: {timedelta(seconds=self.session_duration)}")
            else:
                logger.info("Session end: Manual stop required")
                logger.info("Duration: Unlimited (manual stop)")
                
            logger.info(f"Portfolio value: ${self.portfolio_value:.2f}")
            logger.info(f"Strategies: {', '.join(list(self.active_strategies.keys()))}")
            logger.info(f"Symbols: {', '.join(self.config.get('crypto', {}).get('symbols', []))}")
            logger.info(f"Check interval: {self.config.get('crypto', {}).get('check_interval_seconds', 30)} seconds")
            logger.info("===================================================")
            
            # Confirm before starting live trading
            if self.config.get("crypto", {}).get("live_trading", False):
                confirmation = input("Type 'CONFIRM' to start live crypto trading or anything else to cancel: ")
                
                if confirmation != "CONFIRM":
                    logger.info("Live trading cancelled by user")
                    return
            else:
                # Auto-confirm paper trading if specified
                if not self.config.get("crypto", {}).get("auto_confirm_paper", True):
                    confirmation = input("Type 'CONFIRM' to start paper crypto trading or anything else to cancel: ")
                    
                    if confirmation != "CONFIRM":
                        logger.info("Paper trading cancelled by user")
                        return
            
            logger.info("Crypto trading session confirmed. Starting main trading loop...")
            
            # Main trading loop
            await self.trading_loop()
            
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
        except Exception as e:
            logger.error(f"Critical error in crypto trading system: {e}")
        finally:
            await self.generate_performance_report()
            
    async def trading_loop(self) -> None:
        """Main trading loop with session-based execution"""
        global running
        running = True
        
        check_interval = self.config.get("crypto", {}).get("check_interval_seconds", 30)
        # More frequent updates for short sessions
        if self.session_duration < SessionDuration.FOUR_HOURS:
            check_interval = min(check_interval, 20)  # 20 seconds maximum for short sessions
        
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
                
                # Check market conditions
                await self.check_market_conditions()
                
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
                    for symbol in self.config.get("crypto", {}).get("symbols", []):
                        tasks.append(self.process_symbol(symbol))
                    
                    if tasks:
                        await asyncio.gather(*tasks)
                
                # Periodically adjust risk level
                if (now - self.last_risk_adjustment).total_seconds() > self.config.get("crypto", {}).get("risk_adjustment_interval_seconds", 1800):
                    self.risk_manager.adjust_risk_level()
                    self.last_risk_adjustment = now
                
                # Update strategy weights periodically
                if (now - self.last_strategy_update).total_seconds() > self.config.get("crypto", {}).get("strategy_update_interval_seconds", 1200):
                    await self.update_strategy_weights()
                    self.last_strategy_update = now
                
                # Wait between iterations
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(check_interval * 2)  # Longer pause on error
                
    async def check_market_conditions(self) -> None:
        """Check overall crypto market conditions and decide whether to continue trading"""
        try:
            # Only perform this check periodically (e.g., every 5 minutes)
            now = datetime.now()
            if (now - self.last_market_check).total_seconds() < self.config.get("crypto", {}).get("market_check_interval_seconds", 300):
                return
                
            self.last_market_check = now
            
            # Get market index data (e.g., BTC, ETH, Total Market Cap)
            market_symbols = self.config.get("crypto", {}).get("market_indices", ["BTC/USDT", "ETH/USDT"])
            market_data = {}
            
            try:
                for symbol in market_symbols:
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=7)  # 7 days of data for crypto (more volatile)
                    
                    data = await self.market_data.get_historical_data(
                        symbol, start_date, end_date, timeframe="1Day"
                    )
                    
                    if data is not None and len(data) > 0:
                        market_data[symbol] = data
            except Exception as e:
                logger.error(f"Error retrieving market data for {symbol}: {e}")
            
            if not market_data:
                logger.warning("Could not retrieve crypto market data for condition assessment")
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
            if max_anomaly > 0.8:  # Higher threshold for crypto markets
                new_condition = MarketCondition.DANGEROUS
                reason = f"High anomaly score detected: {max_anomaly:.2f}"
            elif avg_volatility > 90:  # Higher threshold for crypto markets
                new_condition = MarketCondition.VOLATILE
                reason = f"Extreme market volatility: {avg_volatility:.1f} percentile"
            elif avg_volatility < 10:
                new_condition = MarketCondition.INACTIVE
                reason = f"Very low market volatility: {avg_volatility:.1f} percentile"
            else:
                new_condition = MarketCondition.NORMAL
                reason = f"Normal crypto market conditions: {avg_volatility:.1f} percentile volatility"
            
            # Log if condition changes
            if new_condition != self.market_condition:
                logger.info(f"Crypto market condition changed: {self.market_condition} -> {new_condition} ({reason})")
                
                # Pause trading if conditions dangerous
                if new_condition == MarketCondition.DANGEROUS and not self.trading_paused:
                    self.trading_paused = True
                    self.pause_reason = f"Dangerous market conditions: {reason}"
                    logger.warning(f"Trading PAUSED: {self.pause_reason}")
            
            self.market_condition = new_condition
            
        except Exception as e:
            logger.error(f"Error checking crypto market conditions: {e}")
    
    def should_resume_trading(self) -> bool:
        """Determine if paused trading should be resumed"""
        # Resume if market condition has improved
        if self.market_condition == MarketCondition.NORMAL:
            return True
            
        # Resume after timeout period if not in dangerous conditions
        pause_duration = (datetime.now() - self.last_market_check).total_seconds()
        max_pause_duration = self.config.get("crypto", {}).get("max_pause_duration_seconds", 1200)  # 20 minutes default for crypto
        
        if pause_duration > max_pause_duration and self.market_condition != MarketCondition.DANGEROUS:
            return True
            
        return False
        
    async def process_symbol(self, symbol: str) -> None:
        """Process a single trading symbol"""
        try:
            # Get most recent price data
            latest_price = await self.market_data.get_latest_price(symbol)
            if not latest_price:
                logger.warning(f"Unable to get latest price for {symbol}")
                return
                
            # Get historical data for analysis (more data for crypto due to 24/7 markets)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=14)  # 14 days of data for crypto
            
            data = await self.market_data.get_historical_data(
                symbol, start_date, end_date, timeframe="1Day"
            )
            
            if data is None or len(data) == 0:
                logger.warning(f"No historical data available for {symbol}")
                return
                
            # Calculate indicators and analyze market data
            analysis = self.market_analyzer.analyze_market_data(symbol, data)
            
            # Log market analysis summary for the symbol
            volatility = analysis.get("volatility", {}).get("value", 0.0)
            trend = analysis.get("trend", {}).get("direction", "neutral")
            rsi = analysis.get("indicators", {}).get("rsi", 50)
            
            logger.debug(f"Crypto: {symbol} - Price: ${latest_price:.4f} - Trend: {trend} - Vol: {volatility:.2f} - RSI: {rsi:.1f}")
            
            # Get current holdings for this symbol
            position = await self.trading_service.get_position(symbol)
            
            # Execute signal generation through strategy selector
            signal = await self.strategy_selector.generate_signal(symbol, data, analysis, position)
            
            if signal and signal.get("action") != "hold":
                # Calculate position size and risk parameters
                risk_params = self.risk_manager.calculate_position_size(
                    symbol, 
                    latest_price, 
                    signal.get("stop_loss", latest_price * 0.92)  # Wider stop loss for crypto (8%)
                )
                
                signal.update(risk_params)
                
                # Execute the trading signal
                await self.execute_signal(symbol, signal, latest_price, position)
                
        except Exception as e:
            logger.error(f"Error processing crypto symbol {symbol}: {e}")
            
    async def execute_signal(self, symbol: str, signal: Dict[str, Any], current_price: float, position: Optional[Dict[str, Any]]) -> None:
        """Execute a trading signal for a crypto symbol"""
        action = signal.get("action")
        quantity = signal.get("quantity", 0)
        strategy = signal.get("strategy", "unknown")
        confidence = signal.get("confidence", 0.5)
        stop_loss = signal.get("stop_loss")
        take_profit = signal.get("take_profit")
        
        if action not in ["buy", "sell", "hold"]:
            logger.warning(f"Unknown action '{action}' for crypto {symbol}")
            return
            
        if action == "hold":
            return
            
        # Log the signal
        logger.info(f"CRYPTO SIGNAL: {action.upper()} {symbol} - Quantity: {quantity} - Price: ${current_price:.4f} ")
        logger.info(f"  Strategy: {strategy} - Confidence: {confidence:.2f} - Stop: ${stop_loss:.4f} - Target: ${take_profit:.4f}")
        
        # Skip if quantity too small
        if quantity <= 0:
            logger.warning(f"Skipping {action} for {symbol} due to quantity <= 0")
            return
            
        # Skip if we're in paper trading below confidence threshold (higher for crypto)
        if not self.config.get("crypto", {}).get("live_trading", False) and confidence < self.config.get("crypto", {}).get("min_confidence_threshold", 0.65):
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
                logger.info(f"EXECUTED CRYPTO: {action.upper()} {symbol} - Quantity: {quantity} - Price: ${current_price:.4f}")
                
                # Update strategy performance statistics
                self.strategy_selector.update_strategy_performance(strategy, symbol, action, confidence)
            else:
                error = result.get("error", "Unknown error") if result else "No result returned"
                logger.error(f"Crypto trade execution failed: {error}")
                
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
                
                # Calculate hourly change for crypto (more frequent than daily for stocks)
                if previous_value > 0:
                    hourly_change_pct = (self.portfolio_value - previous_value) / previous_value * 100
                    
                    # Log significant changes
                    if abs(hourly_change_pct) >= 2.0:  # Higher threshold for crypto
                        direction = "up" if hourly_change_pct > 0 else "down"
                        logger.info(f"Crypto portfolio value {direction} by {abs(hourly_change_pct):.2f}% to ${self.portfolio_value:.2f}")
                        
                        # Check for excessive loss (higher threshold for crypto)
                        max_hourly_loss_pct = self.config.get("crypto", {}).get("max_hourly_loss_percentage", 8.0)
                        if hourly_change_pct <= -max_hourly_loss_pct and not self.trading_paused:
                            self.trading_paused = True
                            self.pause_reason = f"Excessive hourly loss: {abs(hourly_change_pct):.2f}% exceeded threshold of {max_hourly_loss_pct}%"
                            logger.warning(f"Trading PAUSED: {self.pause_reason}")
                
                # Update positions
                self.positions = await self.trading_service.get_all_positions()
                
                # Check max drawdown (higher threshold for crypto)
                if self.portfolio_value > self.peak_portfolio_value:
                    self.peak_portfolio_value = self.portfolio_value
                elif self.peak_portfolio_value > 0:
                    drawdown_pct = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value * 100
                    max_drawdown_pct = self.config.get("crypto", {}).get("max_drawdown_percentage", 25.0)
                    
                    if drawdown_pct >= max_drawdown_pct and not self.trading_paused:
                        self.trading_paused = True
                        self.pause_reason = f"Maximum drawdown reached: {drawdown_pct:.2f}% exceeded threshold of {max_drawdown_pct}%"
                        logger.warning(f"Trading PAUSED: {self.pause_reason}")
            else:
                logger.warning("Failed to retrieve crypto account information")
                
        except Exception as e:
            logger.error(f"Error updating crypto portfolio state: {e}")
            
    async def update_strategy_weights(self) -> None:
        """Update strategy weights based on performance metrics"""
        try:
            # Get strategy performance statistics from strategy selector
            performance_stats = self.strategy_selector.get_strategy_performance()
            
            if not performance_stats:
                return
                
            # Log strategy performance
            logger.info("Crypto Strategy Performance Update:")
            for strategy, stats in performance_stats.items():
                win_rate = stats.get("win_rate", 0.0) * 100
                profit_factor = stats.get("profit_factor", 1.0)
                weight = stats.get("weight", 0.0) * 100
                
                logger.info(f"  {strategy}: Win Rate: {win_rate:.1f}% - Profit Factor: {profit_factor:.2f} - Weight: {weight:.1f}%")
                
            # Update strategy weights based on performance
            self.strategy_selector.update_weights_based_on_performance()
            
        except Exception as e:
            logger.error(f"Error updating crypto strategy weights: {e}")
            
    async def generate_performance_report(self) -> None:
        """Generate a performance report at the end of the trading session"""
        try:
            logger.info("===================================================")
            logger.info("CRYPTO TRADING SESSION PERFORMANCE REPORT")
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
            start_value = self.config.get("crypto", {}).get("initial_portfolio_value", 0.0)
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
                
                logger.info(f"Total Crypto Trades: {trades_count}")
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
                logger.info("No crypto trades executed during this session")
                
            # Current positions
            if self.positions:
                logger.info("Current Crypto Positions:")
                for position in self.positions:
                    symbol = position.get("symbol")
                    quantity = position.get("quantity", 0)
                    avg_price = position.get("avg_entry_price", 0.0)
                    market_value = position.get("market_value", 0.0)
                    unrealized_pl = position.get("unrealized_pl", 0.0)
                    unrealized_plpc = position.get("unrealized_plpc", 0.0) * 100
                    
                    logger.info(f"  {symbol}: {quantity} units at ${avg_price:.4f} - Value: ${market_value:.2f} - P/L: ${unrealized_pl:.2f} ({unrealized_plpc:.2f}%)")
            else:
                logger.info("No open crypto positions at session end")
                
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
            logger.info("CRYPTO TRADING SESSION COMPLETED")
            logger.info("===================================================")
                
        except Exception as e:
            logger.error(f"Error generating crypto performance report: {e}")
            
    def __del__(self):
        """Cleanup resources when object is destroyed"""
        logger.info("Crypto trading system shutting down...")
        
# Main entry point
def main():
    parser = argparse.ArgumentParser(description="Cryptocurrency Day Trading System")
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
    trader = CryptoDayTrader(config_path=args.config, session_duration=session_duration)
    
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
