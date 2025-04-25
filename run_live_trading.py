"""
MercurioAI Live Trading Script

This script runs the MercurioAI platform in live trading mode, using real money
through the Alpaca brokerage. USE WITH CAUTION.

Usage:
    python run_live_trading.py --strategy MovingAverageStrategy --symbols AAPL,MSFT,GOOGL --risk_limit 0.02
"""
import os
import asyncio
import argparse
import logging
import json
import signal
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/live_trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure required directories exist
os.makedirs("./logs", exist_ok=True)
os.makedirs("./data", exist_ok=True)

# Global variable to control the trading loop
running = True

def signal_handler(sig, frame):
    """Handle Ctrl+C to gracefully exit the trading loop"""
    global running
    logger.info("Received termination signal. Stopping after current iteration...")
    running = False

class LiveTrader:
    """
    Main class for running live trading strategies
    """
    
    def __init__(
        self,
        strategy_name: str,
        strategy_params: Dict[str, Any],
        symbols: List[str],
        risk_limit: float = 0.02,
        check_interval_seconds: int = 60,
        data_lookback_days: int = 90
    ):
        """
        Initialize the live trader.
        
        Args:
            strategy_name: Name of the strategy to use
            strategy_params: Parameters for the strategy
            symbols: List of symbols to trade
            risk_limit: Maximum percentage of portfolio to risk per position (0.0 to 1.0)
            check_interval_seconds: How often to check for new signals (in seconds)
            data_lookback_days: How many days of historical data to use
        """
        self.strategy_name = strategy_name
        self.strategy_params = strategy_params or {}
        self.symbols = symbols
        self.risk_limit = risk_limit
        self.check_interval_seconds = check_interval_seconds
        self.data_lookback_days = data_lookback_days
        
        # These will be initialized in the setup method
        self.market_data_service = None
        self.trading_service = None
        self.strategy_manager = None
        self.strategy = None
        
        # Trading state
        self.positions = {}
        self.pending_orders = {}
        
        logger.info(f"Initialized LiveTrader with strategy {strategy_name}")
        logger.info(f"Trading symbols: {', '.join(symbols)}")
        logger.info(f"Risk limit per position: {risk_limit*100:.1f}%")
    
    async def setup(self):
        """Initialize services and verify API connectivity"""
        from app.services.market_data import MarketDataService
        from app.services.trading import TradingService
        from app.services.strategy_manager import StrategyManager
        
        logger.info("Setting up services...")
        
        # Initialize market data service
        self.market_data_service = MarketDataService()
        
        # Initialize trading service (with live trading = True)
        # For live trading, is_paper should be False
        self.trading_service = TradingService(is_paper=False)
        
        # Verify trading access
        account_info = await self.trading_service.get_account_info()
        if "error" in account_info:
            raise ValueError(f"Trading service error: {account_info['error']}")
        
        logger.info(f"Connected to trading account: {account_info['id']}")
        logger.info(f"Account status: {account_info['status']}")
        logger.info(f"Portfolio value: ${float(account_info['portfolio_value']):.2f}")
        
        # Check if account is restricted
        if account_info.get("status") != "ACTIVE":
            raise ValueError(f"Account not active for trading: {account_info['status']}")
        
        # Initialize strategy manager
        self.strategy_manager = StrategyManager()
        
        # Get strategy instance
        self.strategy = await self.strategy_manager.get_strategy(
            self.strategy_name,
            self.strategy_params
        )
        
        if not self.strategy:
            raise ValueError(f"Strategy {self.strategy_name} not found")
        
        logger.info("Services set up successfully")
    
    async def check_market_status(self) -> bool:
        """
        Check if the market is open for trading.
        
        Returns:
            True if market is open, False otherwise
        """
        status = await self.trading_service.check_market_status()
        is_open = status.get("is_open", False)
        
        if is_open:
            logger.info("Market is open for trading")
        else:
            next_open = status.get("next_open", "unknown")
            logger.info(f"Market is closed. Next open: {next_open}")
        
        return is_open
    
    async def update_positions(self):
        """Update the current positions dictionary"""
        positions = await self.trading_service.get_positions()
        
        # Reset positions dictionary
        self.positions = {}
        
        # Update with current positions
        for position in positions:
            if "error" in position:
                logger.error(f"Error getting positions: {position['error']}")
                continue
                
            symbol = position["symbol"]
            self.positions[symbol] = {
                "quantity": float(position["qty"]),
                "market_value": float(position["market_value"]),
                "avg_entry_price": float(position["avg_entry_price"]),
                "current_price": float(position["current_price"]),
                "unrealized_pl": float(position["unrealized_pl"]),
                "unrealized_plpc": float(position["unrealized_plpc"]),
                "side": position["side"]
            }
        
        logger.info(f"Current positions: {len(self.positions)} assets")
        for symbol, data in self.positions.items():
            logger.info(f"  {symbol}: {data['quantity']} shares, P&L: ${data['unrealized_pl']:.2f} ({data['unrealized_plpc']*100:.2f}%)")
    
    async def check_pending_orders(self):
        """Check and update status of pending orders"""
        completed_orders = []
        
        for order_id in self.pending_orders:
            order_status = await self.trading_service.get_order_status(order_id)
            
            if order_status.get("status") in ["filled", "canceled", "expired", "rejected"]:
                logger.info(f"Order {order_id} for {order_status.get('symbol')} is {order_status.get('status')}")
                
                # Store for removal
                completed_orders.append(order_id)
        
        # Remove completed orders
        for order_id in completed_orders:
            self.pending_orders.pop(order_id, None)
    
    async def execute_trading_signal(self, symbol: str, action, confidence: float):
        """
        Execute a trading action based on the strategy signal.
        
        Args:
            symbol: Trading symbol
            action: TradeAction (BUY, SELL, HOLD)
            confidence: Signal confidence (0.0 to 1.0)
        """
        from app.db.models import TradeAction
        
        # Don't trade on low confidence signals
        if confidence < 0.6:
            logger.info(f"Skipping {symbol} signal due to low confidence: {confidence:.2f}")
            return
        
        # Determine the capital percentage based on confidence and risk limit
        capital_pct = min(self.risk_limit * confidence, self.risk_limit)
        
        # Calculate order quantity
        quantity = await self.trading_service.calculate_order_quantity(
            symbol=symbol,
            action=action,
            capital_pct=capital_pct
        )
        
        # Check if quantity is significant
        if quantity <= 0:
            logger.info(f"Skipping {symbol} trade - zero quantity calculated")
            return
        
        logger.info(f"Executing: {action.name} {quantity} {symbol} (confidence: {confidence:.2f})")
        
        # Execute the trade
        result = await self.trading_service.execute_trade(
            symbol=symbol,
            action=action,
            quantity=quantity,
            strategy_name=self.strategy_name
        )
        
        # Check result
        if result.get("status") == "success":
            order_info = result.get("order", {})
            order_id = order_info.get("id")
            
            if order_id:
                self.pending_orders[order_id] = {
                    "symbol": symbol,
                    "action": action,
                    "quantity": quantity,
                    "timestamp": datetime.now().isoformat()
                }
                
                logger.info(f"Order placed successfully: {order_id}")
            else:
                logger.warning(f"Order placed but no order ID returned")
        else:
            logger.error(f"Order execution failed: {result.get('message', 'Unknown error')}")
    
    async def process_symbol(self, symbol: str):
        """
        Process a trading symbol - get data, generate signal, execute if needed.
        
        Args:
            symbol: Trading symbol to process
        """
        logger.info(f"Processing {symbol}...")
        
        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.data_lookback_days)
        
        try:
            # Get and preprocess data
            data = await self.strategy.load_data(symbol, start_date, end_date)
            
            if data is None or len(data) == 0:
                logger.error(f"No data received for {symbol}")
                return
            
            processed_data = await self.strategy.preprocess_data(data)
            
            # Check if strategy needs training
            if hasattr(self.strategy, 'is_trained') and not self.strategy.is_trained:
                logger.info(f"Training strategy for {symbol}...")
                await self.strategy.train(processed_data)
            
            # Generate trading signal
            action, confidence = await self.strategy.predict(processed_data)
            
            logger.info(f"Signal for {symbol}: {action.name} (confidence: {confidence:.2f})")
            
            # Execute the trading signal
            await self.execute_trading_signal(symbol, action, confidence)
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {str(e)}")
    
    async def trading_loop(self):
        """Main trading loop - run continuously during market hours"""
        global running
        
        logger.info("Starting trading loop...")
        
        while running:
            try:
                # Check if market is open
                is_open = await self.check_market_status()
                
                if not is_open:
                    # If market is closed, wait a longer time before checking again
                    logger.info(f"Market closed, waiting 30 minutes before checking again...")
                    await asyncio.sleep(1800)  # 30 minutes
                    continue
                
                # Update current positions
                await self.update_positions()
                
                # Check pending orders
                await self.check_pending_orders()
                
                # Process each symbol
                for symbol in self.symbols:
                    await self.process_symbol(symbol)
                
                # Wait before next iteration
                logger.info(f"Waiting {self.check_interval_seconds} seconds until next check...")
                await asyncio.sleep(self.check_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {str(e)}")
                # Wait a bit before retrying
                await asyncio.sleep(60)
    
    async def run(self):
        """Run the live trader"""
        try:
            # Setup all services
            await self.setup()
            
            # Safety check - confirm live trading
            account_info = await self.trading_service.get_account_info()
            portfolio_value = float(account_info.get('portfolio_value', 0))
            
            logger.info("=" * 50)
            logger.info("LIVE TRADING CONFIRMATION")
            logger.info("=" * 50)
            logger.info(f"You are about to start LIVE TRADING with real money")
            logger.info(f"Account ID: {account_info.get('id')}")
            logger.info(f"Portfolio Value: ${portfolio_value:.2f}")
            logger.info(f"Strategy: {self.strategy_name}")
            logger.info(f"Symbols: {', '.join(self.symbols)}")
            logger.info("=" * 50)
            
            confirmation = input("Type 'CONFIRM' to start live trading or anything else to abort: ")
            
            if confirmation != "CONFIRM":
                logger.info("Live trading aborted by user")
                return
            
            # Start the trading loop
            logger.info("Live trading confirmed. Starting trading loop...")
            await self.trading_loop()
            
        except Exception as e:
            logger.error(f"Fatal error in live trader: {str(e)}")
            raise
        finally:
            logger.info("Live trading stopped")

async def main():
    """Main entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="MercurioAI Live Trading")
    
    parser.add_argument("--strategy", type=str, default="MovingAverageStrategy",
                       help="Strategy name to use for trading")
    
    parser.add_argument("--params", type=str, default="{}",
                       help="JSON string of strategy parameters")
    
    parser.add_argument("--symbols", type=str, default="AAPL,MSFT,GOOGL",
                       help="Comma-separated list of symbols to trade")
    
    parser.add_argument("--risk_limit", type=float, default=0.02,
                       help="Maximum percentage of portfolio to risk per position (0.0 to 1.0)")
    
    parser.add_argument("--interval", type=int, default=60,
                       help="Check interval in seconds")
    
    parser.add_argument("--lookback", type=int, default=90,
                       help="Lookback period in days for historical data")
    
    parser.add_argument("--config", type=str,
                       help="Path to JSON configuration file")
    
    args = parser.parse_args()
    
    # Load from config file if provided
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
                
            strategy_name = config.get("strategy", args.strategy)
            strategy_params = config.get("strategy_params", {})
            symbols = config.get("symbols", args.symbols.split(","))
            risk_limit = config.get("risk_limit", args.risk_limit)
            check_interval = config.get("check_interval", args.interval)
            data_lookback = config.get("data_lookback", args.lookback)
            
        except Exception as e:
            logger.error(f"Error loading config file: {str(e)}")
            return
    else:
        # Use command line arguments
        strategy_name = args.strategy
        strategy_params = json.loads(args.params)
        symbols = args.symbols.split(",")
        risk_limit = args.risk_limit
        check_interval = args.interval
        data_lookback = args.lookback
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize and run live trader
    live_trader = LiveTrader(
        strategy_name=strategy_name,
        strategy_params=strategy_params,
        symbols=symbols,
        risk_limit=risk_limit,
        check_interval_seconds=check_interval,
        data_lookback_days=data_lookback
    )
    
    await live_trader.run()

if __name__ == "__main__":
    # Run the main async function
    asyncio.run(main())
