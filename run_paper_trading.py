"""
MercurioAI Paper Trading Script

This script runs the MercurioAI platform in paper trading mode, using Alpaca's paper trading
API to simulate trades with real market data but no real money.

Usage:
    python run_paper_trading.py --strategy MovingAverageStrategy --symbols AAPL,MSFT,GOOGL
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
        logging.FileHandler("logs/paper_trading.log"),
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

class PaperTrader:
    """
    Main class for running paper trading strategies with simulated transaction costs
    """
    
    def __init__(
        self,
        strategy_name: str,
        strategy_params: Dict[str, Any],
        symbols: List[str],
        risk_limit: float = 0.02,
        check_interval_seconds: int = 60,
        data_lookback_days: int = 90,
        fee_percentage: float = 0.001,  # 0.1% fee
        fee_fixed: float = 0.0,         # Fixed fee per trade
        fee_minimum: float = 0.0        # Minimum fee per trade
    ):
        """
        Initialize the paper trader.
        
        Args:
            strategy_name: Name of the strategy to use
            strategy_params: Parameters for the strategy
            symbols: List of symbols to trade
            risk_limit: Maximum percentage of portfolio to risk per position (0.0 to 1.0)
            check_interval_seconds: How often to check for new signals (in seconds)
            data_lookback_days: How many days of historical data to use
            fee_percentage: Percentage fee per trade (e.g., 0.001 for 0.1%)
            fee_fixed: Fixed fee per trade
            fee_minimum: Minimum fee per trade
        """
        self.strategy_name = strategy_name
        self.strategy_params = strategy_params or {}
        self.symbols = symbols
        self.risk_limit = risk_limit
        self.check_interval_seconds = check_interval_seconds
        self.data_lookback_days = data_lookback_days
        
        # Transaction cost parameters
        self.fee_percentage = fee_percentage
        self.fee_fixed = fee_fixed
        self.fee_minimum = fee_minimum
        self.total_transaction_costs = 0.0
        
        # These will be initialized in the setup method
        self.market_data_service = None
        self.trading_service = None
        self.strategy_manager = None
        self.strategy = None
        
        # Trading state
        self.positions = {}
        self.pending_orders = {}
        self.trade_history = []
        
        logger.info(f"Initialized PaperTrader with strategy {strategy_name}")
        logger.info(f"Trading symbols: {', '.join(symbols)}")
        logger.info(f"Risk limit per position: {risk_limit*100:.1f}%")
        logger.info(f"Transaction costs: {fee_percentage*100:.3f}% + ${fee_fixed:.2f} (min: ${fee_minimum:.2f})")
    
    async def setup(self):
        """Initialize services and verify API connectivity"""
        from app.services.market_data import MarketDataService
        from app.services.trading import TradingService
        from app.services.strategy_manager import StrategyManager
        
        logger.info("Setting up services...")
        
        # Initialize market data service
        self.market_data_service = MarketDataService()
        
        # Initialize trading service (with paper trading = True)
        self.trading_service = TradingService(is_paper=True)
        
        # Verify trading access
        account_info = await self.trading_service.get_account_info()
        if "error" in account_info:
            raise ValueError(f"Trading service error: {account_info['error']}")
        
        logger.info(f"Connected to paper trading account: {account_info['id']}")
        logger.info(f"Account status: {account_info['status']}")
        logger.info(f"Portfolio value: ${float(account_info['portfolio_value']):.2f}")
        
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
            logger.info(f"Market closed. Next open: {next_open}")
        
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
                
                # Apply transaction costs for filled orders
                if order_status.get("status") == "filled":
                    symbol = order_status.get("symbol")
                    price = float(order_status.get("filled_avg_price", 0))
                    quantity = float(order_status.get("filled_qty", 0))
                    
                    # Calculate and apply transaction costs
                    await self.apply_transaction_costs(symbol, price, quantity)
                
                # Store for removal
                completed_orders.append(order_id)
        
        # Remove completed orders
        for order_id in completed_orders:
            self.pending_orders.pop(order_id, None)
    
    async def apply_transaction_costs(self, symbol: str, price: float, quantity: float):
        """
        Apply simulated transaction costs to a trade.
        
        Args:
            symbol: Trading symbol
            price: Execution price
            quantity: Number of shares
        """
        trade_value = price * quantity
        
        # Calculate transaction cost
        cost = max(self.fee_minimum, self.fee_fixed + (trade_value * self.fee_percentage))
        
        # Add to total costs
        self.total_transaction_costs += cost
        
        logger.info(f"Applied transaction cost: ${cost:.2f} for {symbol} (Total: ${self.total_transaction_costs:.2f})")
        
        # Record the trade with costs
        self.trade_history.append({
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "price": price,
            "quantity": quantity,
            "value": trade_value,
            "cost": cost,
            "net_value": trade_value - cost
        })
    
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
    
    async def generate_performance_report(self):
        """Generate and log performance metrics"""
        try:
            # Get account info
            account_info = await self.trading_service.get_account_info()
            portfolio_value = float(account_info.get('portfolio_value', 0))
            cash = float(account_info.get('cash', 0))
            
            # Calculate metrics
            logger.info("\n===== PERFORMANCE REPORT =====")
            logger.info(f"Portfolio Value: ${portfolio_value:.2f}")
            logger.info(f"Cash: ${cash:.2f}")
            logger.info(f"Total Transaction Costs: ${self.total_transaction_costs:.2f}")
            logger.info(f"Net Portfolio Value (after costs): ${portfolio_value - self.total_transaction_costs:.2f}")
            
            # Trade statistics
            if self.trade_history:
                total_trades = len(self.trade_history)
                total_value = sum(t['value'] for t in self.trade_history)
                total_cost = sum(t['cost'] for t in self.trade_history)
                avg_cost_per_trade = total_cost / total_trades if total_trades > 0 else 0
                cost_percentage = (total_cost / total_value) * 100 if total_value > 0 else 0
                
                logger.info(f"Total Trades: {total_trades}")
                logger.info(f"Average Cost Per Trade: ${avg_cost_per_trade:.2f}")
                logger.info(f"Transaction Costs as % of Trade Value: {cost_percentage:.3f}%")
            
            logger.info("================================\n")
            
        except Exception as e:
            logger.error(f"Error generating performance report: {str(e)}")
    
    async def trading_loop(self):
        """Main trading loop - run continuously during market hours"""
        global running
        
        logger.info("Starting paper trading loop...")
        
        while running:
            try:
                # Check if market is open
                is_open = await self.check_market_status()
                
                if not is_open:
                    # For paper trading, we can optionally continue even when market is closed
                    # Or wait for market to open - using same logic as live trading
                    logger.info(f"Market closed, waiting 30 minutes before checking again...")
                    await asyncio.sleep(1800)  # 30 minutes
                    continue
                
                # Update current positions
                await self.update_positions()
                
                # Check pending orders
                await self.check_pending_orders()
                
                # Generate performance report every iteration
                await self.generate_performance_report()
                
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
        """Run the paper trader"""
        try:
            # Setup all services
            await self.setup()
            
            # Confirmation for paper trading (still good practice)
            account_info = await self.trading_service.get_account_info()
            portfolio_value = float(account_info.get('portfolio_value', 0))
            
            logger.info("=" * 50)
            logger.info("PAPER TRADING CONFIRMATION")
            logger.info("=" * 50)
            logger.info(f"You are about to start PAPER TRADING (no real money)")
            logger.info(f"Account ID: {account_info.get('id')}")
            logger.info(f"Paper Portfolio Value: ${portfolio_value:.2f}")
            logger.info(f"Strategy: {self.strategy_name}")
            logger.info(f"Symbols: {', '.join(self.symbols)}")
            logger.info(f"Transaction Costs: {self.fee_percentage*100:.3f}% + ${self.fee_fixed:.2f} (min: ${self.fee_minimum:.2f})")
            logger.info("=" * 50)
            
            confirmation = input("Type 'CONFIRM' to start paper trading or anything else to abort: ")
            
            if confirmation != "CONFIRM":
                logger.info("Paper trading aborted by user")
                return
            
            # Start the trading loop
            logger.info("Paper trading confirmed. Starting trading loop...")
            await self.trading_loop()
            
        except Exception as e:
            logger.error(f"Fatal error in paper trader: {str(e)}")
            raise
        finally:
            logger.info("Paper trading stopped")
            
            # Final performance report
            await self.generate_performance_report()

async def main():
    """Main entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="MercurioAI Paper Trading")
    
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
    
    parser.add_argument("--fee_percentage", type=float, default=0.001,
                       help="Percentage fee per trade (e.g., 0.001 for 0.1%)")
    
    parser.add_argument("--fee_fixed", type=float, default=0.0,
                       help="Fixed fee per trade")
    
    parser.add_argument("--fee_minimum", type=float, default=0.0,
                       help="Minimum fee per trade")
    
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
            
            # Transaction costs
            fee_percentage = config.get("transaction_costs", {}).get("percentage_fee", args.fee_percentage)
            fee_fixed = config.get("transaction_costs", {}).get("fixed_fee", args.fee_fixed)
            fee_minimum = config.get("transaction_costs", {}).get("min_fee", args.fee_minimum)
            
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
        fee_percentage = args.fee_percentage
        fee_fixed = args.fee_fixed
        fee_minimum = args.fee_minimum
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize and run paper trader
    paper_trader = PaperTrader(
        strategy_name=strategy_name,
        strategy_params=strategy_params,
        symbols=symbols,
        risk_limit=risk_limit,
        check_interval_seconds=check_interval,
        data_lookback_days=data_lookback,
        fee_percentage=fee_percentage,
        fee_fixed=fee_fixed,
        fee_minimum=fee_minimum
    )
    
    await paper_trader.run()

if __name__ == "__main__":
    # Run the main async function
    asyncio.run(main())
