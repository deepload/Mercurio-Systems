"""
Trading Service

Provides functionality for executing trades and managing portfolios
using Alpaca as the broker.
"""
import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import asyncio

# For Alpaca API
import alpaca_trade_api as tradeapi

from app.db.models import TradeAction
from app.services.market_data import MarketDataService

logger = logging.getLogger(__name__)

class TradingService:
    """
    Service for executing trades and managing portfolios.
    
    Supports:
    - Paper trading with Alpaca
    - Live trading with Alpaca
    - Order tracking and position management
    """
    
    def __init__(self, is_paper: bool = True):
        """
        Initialize the trading service with Alpaca client.
        
        Args:
            is_paper: Whether to use Alpaca paper trading API
        """
        self.alpaca_key = os.getenv("ALPACA_KEY")
        self.alpaca_secret = os.getenv("ALPACA_SECRET")
        
        # Determine base URL based on paper trading mode
        if is_paper:
            self.base_url = "https://paper-api.alpaca.markets"
        else:
            self.base_url = "https://api.alpaca.markets"
        
        self.alpaca_client = None
        if self.alpaca_key and self.alpaca_secret:
            try:
                self.alpaca_client = tradeapi.REST(
                    key_id=self.alpaca_key,
                    secret_key=self.alpaca_secret,
                    base_url=self.base_url
                )
                logger.info(f"Alpaca client initialized successfully (paper: {is_paper})")
            except Exception as e:
                logger.error(f"Failed to initialize Alpaca client: {e}")
        
        # Initialize market data service for price information
        self.market_data = MarketDataService()
    
    async def check_market_status(self) -> Dict[str, Any]:
        """
        Check if the market is currently open.
        
        Returns:
            Dictionary with market status information
        """
        if not self.alpaca_client:
            return {"is_open": False, "error": "Alpaca client not initialized"}
        
        try:
            clock = self.alpaca_client.get_clock()
            market_status = {
                "is_open": clock.is_open,
                "next_open": clock.next_open.isoformat(),
                "next_close": clock.next_close.isoformat(),
                "timestamp": clock.timestamp.isoformat()
            }
            return market_status
        except Exception as e:
            logger.error(f"Error checking market status: {e}")
            return {"is_open": False, "error": str(e)}
    
    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get current account information.
        
        Returns:
            Dictionary with account information
        """
        if not self.alpaca_client:
            return {"error": "Alpaca client not initialized"}
        
        try:
            account = self.alpaca_client.get_account()
            account_info = {
                "id": account.id,
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "equity": float(account.equity),
                "buying_power": float(account.buying_power),
                "initial_margin": float(account.initial_margin),
                "daytrade_count": account.daytrade_count,
                "status": account.status
            }
            return account_info
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {"error": str(e)}
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current positions.
        
        Returns:
            List of position dictionaries
        """
        if not self.alpaca_client:
            return [{"error": "Alpaca client not initialized"}]
        
        try:
            positions = self.alpaca_client.list_positions()
            positions_list = []
            
            for position in positions:
                positions_list.append({
                    "symbol": position.symbol,
                    "qty": float(position.qty),
                    "market_value": float(position.market_value),
                    "avg_entry_price": float(position.avg_entry_price),
                    "current_price": float(position.current_price),
                    "unrealized_pl": float(position.unrealized_pl),
                    "unrealized_plpc": float(position.unrealized_plpc),
                    "side": position.side
                })
            
            return positions_list
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return [{"error": str(e)}]
    
    async def execute_trade(
        self,
        symbol: str,
        action: TradeAction,
        quantity: float,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        time_in_force: str = "day",
        strategy_name: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Execute a trade order.
        
        Args:
            symbol: The market symbol (e.g., 'AAPL')
            action: TradeAction (BUY, SELL, HOLD)
            quantity: Quantity to trade
            order_type: Order type (market, limit, etc.)
            limit_price: Price for limit orders
            time_in_force: Time in force (day, gtc, etc.)
            strategy_name: Name of the strategy making the trade
            
        Returns:
            Dictionary with order information
        """
        if action == TradeAction.HOLD:
            return {"status": "skipped", "message": "HOLD action, no trade executed"}
        
        if not self.alpaca_client:
            return {"status": "error", "message": "Alpaca client not initialized"}
        
        try:
            # Convert TradeAction to Alpaca side
            side = "buy" if action == TradeAction.BUY else "sell"
            
            # Handle fractional quantities
            if quantity < 1 and not isinstance(quantity, int):
                # Alpaca supports fractional shares for market orders
                if order_type != "market":
                    order_type = "market"
                    logger.warning("Changing order type to market for fractional shares")
                
                # Use notional API for fractional shares
                try:
                    latest_price = await self.market_data.get_latest_price(symbol)
                    notional_amount = quantity * latest_price
                    
                    order = self.alpaca_client.submit_order(
                        symbol=symbol,
                        notional=notional_amount,
                        side=side,
                        type=order_type,
                        time_in_force=time_in_force
                    )
                except Exception as notional_error:
                    logger.error(f"Error executing notional order: {notional_error}")
                    # Fall back to standard order API
                    order = self.alpaca_client.submit_order(
                        symbol=symbol,
                        qty=quantity,
                        side=side,
                        type=order_type,
                        time_in_force=time_in_force,
                        limit_price=limit_price if order_type == "limit" else None
                    )
            else:
                # Standard order API
                order = self.alpaca_client.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side=side,
                    type=order_type,
                    time_in_force=time_in_force,
                    limit_price=limit_price if order_type == "limit" else None
                )
            
            # Format order information
            order_info = {
                "id": order.id,
                "client_order_id": order.client_order_id,
                "symbol": order.symbol,
                "side": order.side,
                "qty": order.qty,
                "order_type": order.type,
                "status": order.status,
                "submitted_at": order.submitted_at.isoformat() if order.submitted_at else None,
                "strategy": strategy_name
            }
            
            logger.info(f"Order executed: {order_info}")
            return {"status": "success", "order": order_info}
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return {"status": "error", "message": str(e)}
    
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get the status of an order.
        
        Args:
            order_id: The ID of the order
            
        Returns:
            Dictionary with order status
        """
        if not self.alpaca_client:
            return {"status": "error", "message": "Alpaca client not initialized"}
        
        try:
            order = self.alpaca_client.get_order(order_id)
            
            order_status = {
                "id": order.id,
                "client_order_id": order.client_order_id,
                "status": order.status,
                "symbol": order.symbol,
                "side": order.side,
                "qty": order.qty,
                "filled_qty": order.filled_qty,
                "type": order.type,
                "time_in_force": order.time_in_force,
                "limit_price": order.limit_price,
                "filled_avg_price": order.filled_avg_price,
                "submitted_at": order.submitted_at.isoformat() if order.submitted_at else None,
                "filled_at": order.filled_at.isoformat() if order.filled_at else None,
                "canceled_at": order.canceled_at.isoformat() if order.canceled_at else None,
                "failed_at": order.failed_at.isoformat() if order.failed_at else None,
                "asset_class": order.asset_class,
                "asset_id": order.asset_id
            }
            
            return order_status
        except Exception as e:
            logger.error(f"Error getting order status: {e}")
            return {"status": "error", "message": str(e)}
    
    async def calculate_order_quantity(
        self,
        symbol: str,
        action: TradeAction,
        capital_pct: float = 0.1
    ) -> float:
        """
        Calculate the quantity to order based on available capital.
        
        Args:
            symbol: The market symbol (e.g., 'AAPL')
            action: TradeAction (BUY, SELL)
            capital_pct: Percentage of available capital to use (0.0 to 1.0)
            
        Returns:
            Order quantity
        """
        if action == TradeAction.HOLD:
            return 0.0
        
        if not self.alpaca_client:
            logger.error("Alpaca client not initialized")
            return 0.0
        
        try:
            # Get account information
            account = self.alpaca_client.get_account()
            available_capital = float(account.cash) if action == TradeAction.BUY else 0.0
            
            # If selling, check current position
            if action == TradeAction.SELL:
                try:
                    position = self.alpaca_client.get_position(symbol)
                    return float(position.qty)
                except Exception as e:
                    logger.warning(f"No position found for {symbol}: {e}")
                    return 0.0
            
            # For buying, calculate based on latest price and available capital
            latest_price = await self.market_data.get_latest_price(symbol)
            
            # Calculate quantity based on capital percentage
            capital_to_use = available_capital * capital_pct
            quantity = capital_to_use / latest_price
            
            # Round to 6 decimal places for fractional shares
            quantity = round(quantity, 6)
            
            logger.info(f"Calculated order quantity for {symbol}: {quantity}")
            return quantity
            
        except Exception as e:
            logger.error(f"Error calculating order quantity: {e}")
            return 0.0
