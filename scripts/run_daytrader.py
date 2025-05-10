#!/usr/bin/env python
"""
Mercurio AI Advanced Day Trading System

This script serves as the main orchestrator for the Mercurio day trading system,
combining strategy selection, market analysis, risk management, and execution
in an intelligent, adaptive framework that dynamically responds to market conditions.

Features:
- Dynamic strategy switching based on market conditions
- Real-time market regime detection and anomaly identification
- Advanced risk management with position sizing and drawdown protection
- Multiple timeframe analysis for better decision making
- Performance monitoring and logging
- Support for both paper and live trading modes
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
log_file = log_dir / f"daytrader_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)

logger = logging.getLogger("DayTrader")

# Global variable to control the trading loop
running = True

def signal_handler(sig, frame):
    """Handle interruption signals for graceful shutdown"""
    global running
    logger.info("Signal received. Gracefully shutting down after current iteration...")
    running = False


class DayTrader:
    """
    Advanced day trading orchestrator that combines multiple strategies,
    market analysis, and risk management into a unified system.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the day trading system.
        
        Args:
            config_path: Path to the configuration file
        """
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
        self.strategy_selector = None
        
        # Trading state
        self.portfolio_value = 0.0
        self.cash = 0.0
        self.positions = {}
        self.pending_orders = {}
        self.transaction_costs = 0.0
        
        # Performance tracking
        self.performance_history = []
        self.last_strategy_update = datetime.now()
        self.last_risk_adjustment = datetime.now()
        
        logger.info(f"DayTrader initialized with configuration: {config_path}")
    
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
            self.cash = account_info.get("cash", 0.0)
            
            # Update risk manager with current capital
            self.risk_manager.update_capital(self.portfolio_value)
            
            logger.info(f"Connected to account: {account_info.get('id', 'unknown')}")
            logger.info(f"Account status: {account_info.get('status', 'unknown')}")
            logger.info(f"Portfolio value: ${self.portfolio_value:.2f}")
            
            # Subscribe to events
            asyncio.create_task(self.event_bus.subscribe(
                EventType.MARKET_DATA_UPDATED,
                self._handle_market_data_update
            ))
            
            logger.info(f"Day trading system initialized with {len(self.active_strategies)} active strategies")
            
        except Exception as e:
            logger.error(f"Error initializing day trading system: {e}")
            raise
    
    async def start(self) -> None:
        """Start the day trading system"""
        try:
            await self.initialize()
            
            logger.info("==================================================")
            logger.info("ADVANCED DAY TRADING SYSTEM CONFIRMATION")
            logger.info("==================================================")
            logger.info(f"Mode: {'LIVE TRADING' if self.config.get('live_trading', False) else 'PAPER TRADING'}")
            logger.info(f"Portfolio value: ${self.portfolio_value:.2f}")
            logger.info(f"Strategies: {', '.join(list(self.active_strategies.keys()))}")
            logger.info(f"Symbols: {', '.join(self.config.get('symbols', []))}")
            logger.info(f"Check interval: {self.config.get('check_interval_seconds', 60)} seconds")
            logger.info(f"Risk per trade: {self.config.get('base_risk_per_trade_pct', 0.01)*100:.2f}%")
            logger.info("==================================================")
            
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
            
            logger.info("Trading system confirmed. Starting main trading loop...")
            
            # Main trading loop
            await self.trading_loop()
            
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
        except Exception as e:
            logger.error(f"Critical error in day trading system: {e}")
        finally:
            await self.generate_performance_report()
            
    async def trading_loop(self) -> None:
        """Main trading loop"""
        global running
        running = True
        
        check_interval = self.config.get("check_interval_seconds", 60)
        
        while running:
            try:
                # Check if market is open
                is_open = await self.trading_service.is_market_open()
                
                if not is_open:
                    next_open = await self.trading_service.get_next_market_open()
                    logger.info(f"Market closed. Next open: {next_open}")
                    
                    # In test/demo mode, continue even if market is closed
                    if not self.config.get("ignore_market_hours", False):
                        wait_time = min(30 * 60, check_interval * 10)  # Max 30 minutes wait
                        logger.info(f"Waiting {wait_time} seconds before next check...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.info("Ignoring market hours, continuing in demo mode...")
                
                # Update portfolio state
                await self.update_portfolio_state()
                
                # Process symbols in parallel
                tasks = []
                for symbol in self.config.get("symbols", []):
                    tasks.append(self.process_symbol(symbol))
                
                if tasks:
                    await asyncio.gather(*tasks)
                
                # Periodically adjust risk level
                if (datetime.now() - self.last_risk_adjustment).total_seconds() > self.config.get("risk_adjustment_interval_seconds", 3600):
                    self.risk_manager.adjust_risk_level()
                    self.last_risk_adjustment = datetime.now()
                
                # Wait between iterations
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(check_interval * 2)  # Longer pause on error
    
    async def update_portfolio_state(self) -> None:
        """Update portfolio state with current positions and account information"""
        try:
            # Update account information
            account_info = await self.trading_service.get_account_info()
            self.portfolio_value = account_info.get("portfolio_value", self.portfolio_value)
            self.cash = account_info.get("cash", self.cash)
            
            # Update positions
            positions = await self.trading_service.get_positions()
            self.positions = {p.get("symbol"): p for p in positions}
            
            # Update pending orders
            orders = await self.trading_service.get_open_orders()
            self.pending_orders = {o.get("id"): o for o in orders}
            
            # Update risk manager capital
            self.risk_manager.update_capital(self.portfolio_value)
            
            # Record performance point
            self.performance_history.append({
                "timestamp": datetime.now(),
                "portfolio_value": self.portfolio_value,
                "cash": self.cash,
                "positions": len(self.positions),
                "pending_orders": len(self.pending_orders)
            })
            
        except Exception as e:
            logger.error(f"Error updating portfolio state: {e}")
    
    async def process_symbol(self, symbol: str) -> None:
        """Process a symbol for trading decisions"""
        try:
            # Skip if we already have pending orders for this symbol
            if any(o.get("symbol") == symbol for o in self.pending_orders.values()):
                logger.info(f"Skipping {symbol} - already has pending orders")
                return
                
            # Get historical data for multiple timeframes
            end_date = datetime.now()
            
            # Primary timeframe for decision making (default: 1-hour bars)
            primary_timeframe = self.config.get("primary_timeframe", "1h")
            primary_days = self.config.get("primary_timeframe_days", 30)
            primary_start_date = end_date - timedelta(days=primary_days)
            
            primary_data = await self.market_data.get_historical_data(
                symbol, primary_start_date, end_date, interval=primary_timeframe
            )
            
            if primary_data is None or len(primary_data) < 20:
                logger.warning(f"Insufficient data for {symbol} on {primary_timeframe} timeframe")
                return
            
            # Secondary timeframe for confirmation (default: 15-minute bars)
            secondary_timeframe = self.config.get("secondary_timeframe", "15m")
            secondary_days = self.config.get("secondary_timeframe_days", 10)
            secondary_start_date = end_date - timedelta(days=secondary_days)
            
            secondary_data = await self.market_data.get_historical_data(
                symbol, secondary_start_date, end_date, interval=secondary_timeframe
            )
            
            # Analysis on primary timeframe
            market_analysis = self.market_analyzer.analyze_market_data(symbol, primary_data)
            
            # Skip if anomalies detected and configured to do so
            if market_analysis.get("anomalies", {}).get("detected", False) and self.config.get("avoid_anomalies", True):
                logger.warning(f"Anomalies detected for {symbol}, skipping")
                return
            
            # Determine market regime
            market_regime = MarketRegime(market_analysis.get("regime", "unknown"))
            
            # Use strategy selector to pick the best strategy
            best_strategy, confidence = await self.strategy_selector.select_best_strategy(symbol, primary_data)
            
            # Get trading signal from selected strategy
            action, strategy_confidence = await best_strategy.predict(primary_data)
            
            # Combine confidences
            combined_confidence = (confidence + strategy_confidence) / 2
            
            logger.info(f"Signal for {symbol} using {best_strategy.__class__.__name__}: {action.name} "
                       f"(confidence: {combined_confidence:.2f}, regime: {market_regime.value})")
            
            # Confirm signal with secondary timeframe if available
            if secondary_data is not None and len(secondary_data) >= 20:
                secondary_action, _ = await best_strategy.predict(secondary_data)
                
                # If signals don't match, reduce confidence
                if secondary_action != action:
                    logger.info(f"Conflicting signals between timeframes for {symbol}")
                    combined_confidence *= 0.7
                else:
                    logger.info(f"Confirmed signal across timeframes for {symbol}")
                    combined_confidence = min(1.0, combined_confidence * 1.2)
            
            # Minimum confidence threshold for execution
            min_confidence = self.config.get("min_execution_confidence", 0.75)
            
            if combined_confidence >= min_confidence and action != TradeAction.HOLD:
                await self.execute_trading_signal(symbol, action, combined_confidence, market_analysis)
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
    
    async def execute_trading_signal(self, symbol: str, action: TradeAction, confidence: float, 
                                    market_analysis: Dict[str, Any]) -> None:
        """Execute a trading signal"""
        try:
            if action == TradeAction.HOLD:
                return
                
            # Check if we already have a position for this symbol
            current_position = self.positions.get(symbol)
            position_value = float(current_position.get("market_value", 0.0)) if current_position else 0.0
            position_qty = float(current_position.get("qty", 0.0)) if current_position else 0.0
            position_side = current_position.get("side", "") if current_position else ""
            
            # Get latest price data
            price_data = await self.market_data.get_latest_price(symbol)
            if not price_data:
                logger.warning(f"Cannot obtain current price for {symbol}")
                return
                
            current_price = price_data.get("price", 0.0)
            if current_price <= 0.0:
                logger.warning(f"Invalid price for {symbol}: {current_price}")
                return
                
            # Determine entry and exit prices based on action
            if action == TradeAction.BUY:
                # Skip if we already have a long position
                if position_side == "long" and position_value > 0:
                    logger.info(f"Already have long position in {symbol}, skipping")
                    return
                    
                # Calculate stop loss level using market data and risk manager
                volatility_data = market_analysis.get("volatility", {})
                
                # Calculate exit levels
                exit_levels = self.risk_manager.calculate_exit_levels(
                    symbol=symbol,
                    entry_price=current_price,
                    direction="long",
                    data=market_analysis.get("indicators", {}).get("atr") if "indicators" in market_analysis else None
                )
                
                stop_loss_price = exit_levels.get("stop_loss")
                take_profit_price = exit_levels.get("take_profit")
                
                # Calculate position size
                position_sizing = self.risk_manager.calculate_position_size(
                    symbol=symbol,
                    price=current_price,
                    stop_loss_price=stop_loss_price,
                    market_regime=market_regime,
                    volatility=volatility_data,
                    strategy_confidence=confidence
                )
                
                quantity = position_sizing.get("shares", 0)
                risk_amount = position_sizing.get("risk_amount", 0)
                
                if quantity <= 0 or risk_amount <= 0:
                    logger.warning(f"Invalid position size for {symbol}: {quantity} shares, ${risk_amount:.2f} risk")
                    return
                
                logger.info(f"BUY signal for {symbol}: {quantity:.6f} shares @ ${current_price:.2f}, "
                          f"stop: ${stop_loss_price:.2f}, target: ${take_profit_price:.2f}, "
                          f"risk: ${risk_amount:.2f} ({position_sizing.get('capital_risked_pct', 0):.2f}%)")
                
                # Place the order
                order_result = await self.trading_service.place_market_order(
                    symbol=symbol,
                    quantity=quantity,
                    side="buy"
                )
                
                logger.info(f"Order result: {order_result}")
                
                # Register position with risk manager
                if order_result.get("status") == "filled" or self.config.get("assume_fill", True):
                    self.risk_manager.register_position(
                        symbol=symbol,
                        entry_price=current_price,
                        shares=quantity,
                        direction="long",
                        stop_loss=stop_loss_price,
                        take_profit=take_profit_price,
                        risk_amount=risk_amount
                    )
                    
            elif action == TradeAction.SELL:
                # If we have a long position, close it
                if position_side == "long" and position_value > 0:
                    logger.info(f"Closing long position on {symbol}: {position_qty} shares @ ${current_price:.2f}")
                    
                    order_result = await self.trading_service.place_market_order(
                        symbol=symbol,
                        quantity=position_qty,
                        side="sell"
                    )
                    
                    logger.info(f"Order result: {order_result}")
                    
                    # Record position closure with risk manager
                    if order_result.get("status") == "filled" or self.config.get("assume_fill", True):
                        self.risk_manager.close_position(
                            symbol=symbol,
                            exit_price=current_price,
                            reason="sell_signal"
                        )
                
                # Opening short positions if allowed and no existing position
                elif not position_value and self.config.get("allow_shorts", False):
                    # Calculate stop loss level
                    exit_levels = self.risk_manager.calculate_exit_levels(
                        symbol=symbol,
                        entry_price=current_price,
                        direction="short",
                        data=market_analysis.get("indicators", {}).get("atr") if "indicators" in market_analysis else None
                    )
                    
                    stop_loss_price = exit_levels.get("stop_loss")
                    take_profit_price = exit_levels.get("take_profit")
                    
                    # Calculate position size
                    position_sizing = self.risk_manager.calculate_position_size(
                        symbol=symbol,
                        price=current_price,
                        stop_loss_price=stop_loss_price,
                        market_regime=market_regime,
                        volatility=market_analysis.get("volatility", {}),
                        strategy_confidence=confidence
                    )
                    
                    quantity = position_sizing.get("shares", 0)
                    risk_amount = position_sizing.get("risk_amount", 0)
                    
                    if quantity <= 0 or risk_amount <= 0:
                        logger.warning(f"Invalid position size for {symbol} short: {quantity} shares, ${risk_amount:.2f} risk")
                        return
                    
                    logger.info(f"SHORT signal for {symbol}: {quantity:.6f} shares @ ${current_price:.2f}, "
                              f"stop: ${stop_loss_price:.2f}, target: ${take_profit_price:.2f}, "
                              f"risk: ${risk_amount:.2f} ({position_sizing.get('capital_risked_pct', 0):.2f}%)")
                    
                    # Place the order
                    order_result = await self.trading_service.place_market_order(
                        symbol=symbol,
                        quantity=quantity,
                        side="sell"
                    )
                    
                    logger.info(f"Order result: {order_result}")
                    
                    # Register position with risk manager
                    if order_result.get("status") == "filled" or self.config.get("assume_fill", True):
                        self.risk_manager.register_position(
                            symbol=symbol,
                            entry_price=current_price,
                            shares=quantity,
                            direction="short",
                            stop_loss=stop_loss_price,
                            take_profit=take_profit_price,
                            risk_amount=risk_amount
                        )
                
        except Exception as e:
            logger.error(f"Error executing signal for {symbol}: {e}")
    
    async def _handle_market_data_update(self, event_data: Dict[str, Any]) -> None:
        """Handle market data updates"""
        symbol = event_data.get("symbol")
        if symbol:
            logger.debug(f"Market data update received for {symbol}")
    
    async def generate_performance_report(self) -> None:
        """Generate a comprehensive performance report"""
        try:
            # Get account information for latest values
            account_info = await self.trading_service.get_account_info()
            portfolio_value = account_info.get("portfolio_value", self.portfolio_value)
            cash = account_info.get("cash", self.cash)
            
            # Get risk manager statistics
            risk_stats = self.risk_manager.get_portfolio_stats()
            
            logger.info("")
            logger.info("===================================================")
            logger.info("PERFORMANCE REPORT")
            logger.info("===================================================")
            logger.info(f"Portfolio Value: ${portfolio_value:.2f}")
            logger.info(f"Cash: ${cash:.2f}")
            logger.info(f"Transaction Costs: ${self.transaction_costs:.2f}")
            logger.info(f"Net Portfolio Value: ${portfolio_value - self.transaction_costs:.2f}")
            logger.info(f"Open Positions: {len(self.positions)}")
            logger.info(f"Win Rate: {risk_stats.get('win_rate', 0):.2f}%")
            logger.info(f"Expectancy: {risk_stats.get('expectancy', 0):.2f}R")
            logger.info(f"Current Drawdown: {risk_stats.get('current_drawdown_pct', 0):.2f}%")
            logger.info(f"Max Drawdown: {risk_stats.get('max_drawdown_pct', 0):.2f}%")
            logger.info(f"Trades Executed: {risk_stats.get('trades_executed', 0)}")
            logger.info("===================================================")
            
            # Save report to file
            report = {
                "timestamp": datetime.now().isoformat(),
                "portfolio_value": portfolio_value,
                "cash": cash,
                "transaction_costs": self.transaction_costs,
                "net_value": portfolio_value - self.transaction_costs,
                "positions": self.positions,
                "risk_stats": risk_stats,
                "performance_history": self.performance_history
            }
            
            # Ensure reports directory exists
            reports_dir = Path("reports")
            reports_dir.mkdir(exist_ok=True)
            
            report_path = reports_dir / f"daytrader_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
                
            logger.info(f"Performance report saved: {report_path}")
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Mercurio AI Advanced Day Trading System")
    parser.add_argument("--config", type=str, default="config/daytrader_config.json", 
                       help="Path to configuration file")
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and start the day trading system
    day_trader = DayTrader(args.config)
    
    try:
        await day_trader.start()
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Critical error: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unhandled error: {e}")
        sys.exit(1)
