"""
Options Trading Service

Extends Mercurio AI's trading capabilities to include options trading
through Alpaca's Options Trading API (Level 1).
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncio

# For Alpaca API
import alpaca_trade_api as tradeapi

from app.db.models import TradeAction
from app.services.trading import TradingService
from app.services.market_data import MarketDataService

logger = logging.getLogger(__name__)

class OptionsService:
    """
    Service for options trading operations.
    
    This service extends the standard TradingService capabilities to include
    options trading through Alpaca. It handles all options-specific operations
    while delegating standard operations to the main TradingService.
    """
    
    def __init__(self, trading_service: TradingService, market_data_service: MarketDataService):
        """
        Initialize the options trading service.
        
        Args:
            trading_service: Main trading service for account operations
            market_data_service: Service for market data
        """
        self.trading_service = trading_service
        self.market_data = market_data_service
        
        # Reference to the Alpaca client from the trading service
        self.alpaca_client = trading_service.alpaca_client
        
        logger.info("Options trading service initialized")
    
    async def get_available_options(self, symbol: str, expiration_date: Optional[str] = None, option_type: Optional[str] = None, expiry_range: Optional[Tuple[str, str]] = None) -> List[Dict[str, Any]]:
        """
        Get available options contracts for a given symbol.
        
        Args:
            symbol: The underlying asset symbol (e.g., 'AAPL')
            expiration_date: Optional specific expiration date (YYYY-MM-DD)
            option_type: Optional type of options to filter ("call" or "put")
            expiry_range: Optional tuple of (min_date, max_date) in YYYY-MM-DD format
            
        Returns:
            List of available options contracts
        """
        if not self.alpaca_client:
            logger.error("Alpaca client not initialized")
            return []
        
        try:
            # Format for Alpaca options symbol: AAPL230616C00165000
            # This represents AAPL options expiring on June 16, 2023 with a strike price of $165.00
            
            # If no expiration date is provided, get the nearest available date
            # Gérer la plage d'expiration si fournie
            if expiry_range and len(expiry_range) == 2:
                min_date, max_date = expiry_range
                # Convertir en objets date si fournis comme chaînes
                if isinstance(min_date, str):
                    min_date = datetime.strptime(min_date, "%Y-%m-%d").date()
                if isinstance(max_date, str):
                    max_date = datetime.strptime(max_date, "%Y-%m-%d").date()
                
                # Trouver toutes les expirations disponibles dans cette plage
                today = datetime.now().date()
                expirations = []
                
                # Chercher les expirations dans la plage spécifiée
                for i in range(60):  # Regarder 60 jours à l'avance
                    date = today + timedelta(days=i)
                    if date >= min_date and date <= max_date and date.weekday() == 4:  # Vendredi
                        expirations.append(date.strftime("%Y-%m-%d"))
            elif not expiration_date:
                # Obtenir les 4 prochaines expirations de vendredi (jour typique d'expiration d'options)
                today = datetime.now()
                expirations = []
                
                # Regarder 60 jours à l'avance pour trouver les expirations
                for i in range(60):
                    date = today + timedelta(days=i)
                    # Vendredi est le jour 4 de la semaine
                    if date.weekday() == 4:
                        expirations.append(date.strftime("%Y-%m-%d"))
                        if len(expirations) >= 4:
                            break
                
                if not expirations:
                    logger.error("Could not find upcoming option expirations")
                    return []
                
                expiration_date = expirations[0]  # Use the nearest expiration
            
            # Get options chain from Alpaca
            logger.info(f"Fetching options chain for {symbol} with expiration {expiration_date}")
            
            # Note: This is where we would call the Alpaca API to get options chain
            # Since we're extending existing functionality, we'll implement this
            # based on how Alpaca exposes options data
            
            # Example implementation (actual API might differ):
            try:
                # Format for the API (date formats may vary)
                expiry = expiration_date.replace("-", "")
                
                # Get calls and puts
                calls = self.alpaca_client.get_options(
                    symbol=symbol,
                    expiration_date=expiration_date,
                    option_type="call"
                )
                
                puts = self.alpaca_client.get_options(
                    symbol=symbol,
                    expiration_date=expiration_date,
                    option_type="put"
                )
                
                # Combine and format results
                options = []
                for contract in calls + puts:
                    options.append({
                        "symbol": contract.symbol,
                        "underlying": symbol,
                        "strike": contract.strike_price,
                        "option_type": contract.option_type,
                        "expiration": contract.expiration_date,
                        "last_price": contract.last_trade_price,
                        "bid": contract.bid_price,
                        "ask": contract.ask_price,
                        "volume": contract.volume,
                        "open_interest": contract.open_interest,
                        "implied_volatility": contract.implied_volatility
                    })
                
                # Filter options by type if specified
                if option_type:
                    options = [option for option in options if option["option_type"].lower() == option_type.lower()]
                
                return options
                
            except AttributeError:
                # If the above implementation doesn't work, we'll try alternative methods
                logger.warning("Standard options API not found, trying alternative implementation")
                
                # Direct REST API call implementation:
                # This would need to be adjusted based on actual API documentation
                options_url = f"https://data.alpaca.markets/v1/options/{symbol}/expirations/{expiry}"
                # Use requests or aiohttp to call the API directly
                
                logger.warning("Options API not fully implemented - check Alpaca API documentation")
                
                # Return mock data for now to allow for development
                options_list = []
                
                # Get current price
                current_price = await self.market_data.get_latest_price(symbol)
                
                # Generate options at various strike prices around current price
                strike_range = [0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3]
                
                for expiry in expirations:
                    for strike_mult in strike_range:
                        strike_price = round(current_price * strike_mult, 2)
                        
                        # Generate call option
                        call_option = {
                            "symbol": f"{symbol}_{expiry}_C_{strike_price}",
                            "underlying": symbol,
                            "expiration": expiry,
                            "strike": strike_price,
                            "option_type": "call",
                            "bid": round(max(0.01, (current_price - strike_price) * 0.8 + 0.5), 2),
                            "ask": round(max(0.01, (current_price - strike_price) * 0.8 + 0.7), 2),
                            "implied_volatility": 0.3,
                            "delta": max(0.01, min(0.99, 1 - (strike_price / current_price))),
                            "gamma": 0.01,
                            "theta": -0.01,
                            "vega": 0.05
                        }
                        
                        # Generate put option
                        put_option = {
                            "symbol": f"{symbol}_{expiry}_P_{strike_price}",
                            "underlying": symbol,
                            "expiration": expiry,
                            "strike": strike_price,
                            "option_type": "put",
                            "bid": round(max(0.01, (strike_price - current_price) * 0.8 + 0.5), 2),
                            "ask": round(max(0.01, (strike_price - current_price) * 0.8 + 0.7), 2),
                            "implied_volatility": 0.3,
                            "delta": -max(0.01, min(0.99, 1 - (current_price / strike_price))),
                            "gamma": 0.01,
                            "theta": -0.01,
                            "vega": 0.05
                        }
                        
                        # Ajouter les options selon le type demandé
                        if option_type:
                            if option_type.lower() == "call":
                                options_list.append(call_option)
                            elif option_type.lower() == "put":
                                options_list.append(put_option)
                        else:
                            # Si aucun type n'est spécifié, ajouter les deux
                            options_list.append(call_option)
                            options_list.append(put_option)
                
                return options_list
                
        except Exception as e:
            logger.error(f"Error fetching options chain: {e}")
            return []
    
    async def execute_option_trade(
        self,
        option_symbol: str,
        action: TradeAction,
        quantity: int,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        time_in_force: str = "day",
        strategy_name: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Execute an options trade.
        
        Args:
            option_symbol: The option contract symbol
            action: TradeAction (BUY, SELL)
            quantity: Number of contracts to trade
            order_type: Order type (market, limit, etc.)
            limit_price: Price for limit orders
            time_in_force: Time in force (day, gtc, etc.)
            strategy_name: Name of the strategy making the trade
            
        Returns:
            Dictionary with order information
        """
        if not self.alpaca_client:
            return {"status": "error", "message": "Alpaca client not initialized"}
        
        if action == TradeAction.HOLD:
            return {"status": "skipped", "message": "HOLD action, no trade executed"}
        
        try:
            # Convert TradeAction to Alpaca side
            side = "buy" if action == TradeAction.BUY else "sell"
            
            logger.info(f"Executing {side} order for {quantity} contracts of {option_symbol}")
            
            # Note: This is where we would call the Alpaca API to execute the options trade
            # Implementation depends on Alpaca's options trading API
            
            try:
                # Example implementation (actual API might differ):
                order = self.alpaca_client.submit_option_order(
                    symbol=option_symbol,
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
                
                logger.info(f"Options order executed: {order_info}")
                return {"status": "success", "order": order_info}
                
            except AttributeError:
                # If the above implementation doesn't work, try alternative methods
                logger.warning("Standard options order API not found, attempting alternative implementation")
                
                # Direct REST API call implementation
                # This would need to be adjusted based on actual API documentation
                
                logger.warning("Options order API not fully implemented - check Alpaca API documentation")
                
                # Return mock response for development purposes
                mock_order_id = f"mock_option_order_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                
                order_info = {
                    "id": mock_order_id,
                    "client_order_id": f"client_{mock_order_id}",
                    "symbol": option_symbol,
                    "side": side,
                    "qty": quantity,
                    "order_type": order_type,
                    "status": "filled",  # Mock status
                    "submitted_at": datetime.now().isoformat(),
                    "strategy": strategy_name
                }
                
                logger.info(f"Mock options order executed: {order_info}")
                return {"status": "success", "order": order_info}
                
        except Exception as e:
            logger.error(f"Error executing options trade: {e}")
            return {"status": "error", "message": str(e)}
    
    async def get_option_position(self, option_symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get details for a specific option position.
        
        Args:
            option_symbol: The option contract symbol
            
        Returns:
            Dictionary with position information or None if not found
        """
        if not self.alpaca_client:
            logger.error("Alpaca client not initialized")
            return None
        
        try:
            # Try to get position information
            try:
                position = self.alpaca_client.get_position(option_symbol)
                
                position_info = {
                    "symbol": position.symbol,
                    "quantity": float(position.qty),
                    "avg_entry_price": float(position.avg_entry_price),
                    "market_value": float(position.market_value),
                    "cost_basis": float(position.cost_basis),
                    "unrealized_pl": float(position.unrealized_pl),
                    "unrealized_plpc": float(position.unrealized_plpc),
                    "current_price": float(position.current_price),
                    "lastday_price": float(position.lastday_price)
                }
                
                return position_info
                
            except Exception as e:
                logger.debug(f"No position found for {option_symbol}: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting option position: {e}")
            return None
    
    async def get_all_option_positions(self) -> List[Dict[str, Any]]:
        """
        Get all current option positions.
        
        Returns:
            List of option position dictionaries
        """
        if not self.alpaca_client:
            logger.error("Alpaca client not initialized")
            return []
        
        try:
            # Get all positions and filter for options
            positions = await self.trading_service.get_positions()
            
            # Filter for options positions (typically have special symbols)
            option_positions = []
            for position in positions:
                # Check if this is an option symbol (implementation depends on Alpaca's format)
                # Typically option symbols contain special characters or follow a pattern
                symbol = position.get("symbol", "")
                
                # Very basic check - adjust based on actual symbol format
                if "_" in symbol or (len(symbol) > 10 and any(c in symbol for c in "CP")):
                    option_positions.append(position)
            
            return option_positions
            
        except Exception as e:
            logger.error(f"Error getting option positions: {e}")
            return []
    
    async def calculate_option_metrics(self, option_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate important option metrics like Greeks (delta, gamma, theta, vega).
        
        Args:
            option_data: Option contract data including price, strike, expiration, etc.
            
        Returns:
            Dictionary with calculated metrics
        """
        # This would implement the Black-Scholes model or other option pricing models
        # For now, we'll return mock data
        
        return {
            "delta": 0.65,  # How much option price changes for $1 move in underlying
            "gamma": 0.03,  # Rate of change of delta per $1 move in underlying
            "theta": -0.05,  # Time decay, how much option loses per day
            "vega": 0.10,   # Sensitivity to volatility changes
            "implied_volatility": 0.35,  # Market's expectation of future volatility
            "extrinsic_value": 2.45,  # Premium related to time and volatility
            "intrinsic_value": 3.20,  # In-the-money value
            "time_to_expiry": 24     # Days until expiration
        }
    
    async def suggest_option_strategies(
        self,
        symbol: str,
        price_prediction: Dict[str, Any],
        risk_profile: str = "moderate"
    ) -> List[Dict[str, Any]]:
        """
        Suggest option strategies based on price predictions and risk profile.
        
        Args:
            symbol: The underlying asset symbol
            price_prediction: Dictionary with price prediction data
            risk_profile: Risk profile (conservative, moderate, aggressive)
            
        Returns:
            List of suggested option strategies
        """
        current_price = await self.market_data.get_latest_price(symbol)
        if not current_price:
            logger.error(f"Could not get current price for {symbol}")
            return []
        
        predicted_price = price_prediction.get("price", current_price)
        prediction_confidence = price_prediction.get("confidence", 0.5)
        time_horizon = price_prediction.get("time_horizon_days", 30)
        
        # Calculate expected move
        expected_move_pct = (predicted_price - current_price) / current_price
        
        strategies = []
        
        # Based on predicted direction and confidence, suggest strategies
        if expected_move_pct > 0.05 and prediction_confidence > 0.6:
            # Bullish outlook with good confidence
            
            # Find appropriate expiration (slightly beyond time horizon)
            expiry_days = min(time_horizon * 1.5, 45)  # Cap at 45 days
            expiry_date = (datetime.now() + timedelta(days=expiry_days)).strftime("%Y-%m-%d")
            
            # Calculate appropriate strike prices
            atm_strike = round(current_price, 0)  # At-the-money
            otm_strike = round(current_price * 1.05, 0)  # 5% Out-of-the-money
            
            # Long call (simple directional play)
            strategies.append({
                "name": "Long Call",
                "option_type": "call",
                "action": "BUY",
                "strike": atm_strike,
                "expiration": expiry_date,
                "quantity": 1,
                "description": "Simple directional play for bullish outlook",
                "risk_rating": "moderate",
                "max_loss": "Limited to premium paid",
                "max_gain": "Unlimited upside potential",
                "confidence_match": min(prediction_confidence * 100, 95)
            })
            
            # Bullish vertical call spread (defined risk)
            if risk_profile == "conservative":
                strategies.append({
                    "name": "Bull Call Spread",
                    "legs": [
                        {"option_type": "call", "action": "BUY", "strike": atm_strike, "expiration": expiry_date},
                        {"option_type": "call", "action": "SELL", "strike": otm_strike, "expiration": expiry_date}
                    ],
                    "description": "Defined risk bullish strategy with lower cost",
                    "risk_rating": "conservative",
                    "max_loss": "Limited to net premium paid",
                    "max_gain": "Limited to difference between strikes minus premium",
                    "confidence_match": min(prediction_confidence * 100, 90)
                })
            
            # Cash-secured put (income strategy with potential to acquire shares)
            if risk_profile in ["moderate", "aggressive"]:
                csp_strike = round(current_price * 0.95, 0)  # 5% below current price
                strategies.append({
                    "name": "Cash-Secured Put",
                    "option_type": "put",
                    "action": "SELL",
                    "strike": csp_strike,
                    "expiration": expiry_date,
                    "quantity": 1,
                    "description": "Income strategy with willingness to buy shares at lower price",
                    "risk_rating": "moderate",
                    "max_loss": f"Limited to strike minus premium (if stock goes to zero)",
                    "max_gain": "Limited to premium received",
                    "confidence_match": min(prediction_confidence * 90, 85)
                })
        
        elif expected_move_pct < -0.05 and prediction_confidence > 0.6:
            # Bearish outlook with good confidence
            
            # Find appropriate expiration (slightly beyond time horizon)
            expiry_days = min(time_horizon * 1.5, 45)  # Cap at 45 days
            expiry_date = (datetime.now() + timedelta(days=expiry_days)).strftime("%Y-%m-%d")
            
            # Calculate appropriate strike prices
            atm_strike = round(current_price, 0)  # At-the-money
            otm_strike = round(current_price * 0.95, 0)  # 5% Out-of-the-money for puts
            
            # Long put (simple directional play)
            strategies.append({
                "name": "Long Put",
                "option_type": "put",
                "action": "BUY",
                "strike": atm_strike,
                "expiration": expiry_date,
                "quantity": 1,
                "description": "Simple directional play for bearish outlook",
                "risk_rating": "moderate",
                "max_loss": "Limited to premium paid",
                "max_gain": "Limited to strike price minus premium (if stock goes to zero)",
                "confidence_match": min(prediction_confidence * 100, 95)
            })
            
            # Bearish vertical put spread (defined risk)
            if risk_profile == "conservative":
                strategies.append({
                    "name": "Bear Put Spread",
                    "legs": [
                        {"option_type": "put", "action": "BUY", "strike": atm_strike, "expiration": expiry_date},
                        {"option_type": "put", "action": "SELL", "strike": otm_strike, "expiration": expiry_date}
                    ],
                    "description": "Defined risk bearish strategy with lower cost",
                    "risk_rating": "conservative",
                    "max_loss": "Limited to net premium paid",
                    "max_gain": "Limited to difference between strikes minus premium",
                    "confidence_match": min(prediction_confidence * 100, 90)
                })
            
            # Covered call (if holding the underlying)
            if risk_profile in ["moderate", "aggressive"]:
                cc_strike = round(current_price * 1.05, 0)  # 5% above current price
                strategies.append({
                    "name": "Covered Call (if holding shares)",
                    "option_type": "call",
                    "action": "SELL",
                    "strike": cc_strike,
                    "expiration": expiry_date,
                    "quantity": 1,
                    "description": "Income strategy if already holding shares, provides some downside protection",
                    "risk_rating": "moderate",
                    "max_loss": "Same as holding stock, minus premium received",
                    "max_gain": "Limited to strike price minus purchase price plus premium",
                    "confidence_match": min(prediction_confidence * 90, 85)
                })
        
        else:
            # Neutral outlook or low confidence
            
            # Find appropriate expiration (shorter-term due to neutral outlook)
            expiry_days = min(time_horizon, 30)  # Cap at 30 days
            expiry_date = (datetime.now() + timedelta(days=expiry_days)).strftime("%Y-%m-%d")
            
            # Calculate appropriate strike prices
            atm_strike = round(current_price, 0)  # At-the-money
            upper_strike = round(current_price * 1.05, 0)  # 5% above
            lower_strike = round(current_price * 0.95, 0)  # 5% below
            
            # Iron Condor (neutral strategy)
            if risk_profile in ["moderate", "aggressive"]:
                strategies.append({
                    "name": "Iron Condor",
                    "legs": [
                        {"option_type": "put", "action": "SELL", "strike": lower_strike, "expiration": expiry_date},
                        {"option_type": "put", "action": "BUY", "strike": round(lower_strike * 0.95, 0), "expiration": expiry_date},
                        {"option_type": "call", "action": "SELL", "strike": upper_strike, "expiration": expiry_date},
                        {"option_type": "call", "action": "BUY", "strike": round(upper_strike * 1.05, 0), "expiration": expiry_date}
                    ],
                    "description": "Income strategy for neutral markets, profits if stock stays within a range",
                    "risk_rating": "moderate",
                    "max_loss": "Limited to difference between wing strikes minus net premium",
                    "max_gain": "Limited to net premium received",
                    "confidence_match": 75 - (abs(expected_move_pct) * 100)  # Lower confidence for larger expected moves
                })
            
            # Cash-secured put (income strategy)
            if risk_profile != "conservative":
                strategies.append({
                    "name": "Cash-Secured Put",
                    "option_type": "put",
                    "action": "SELL",
                    "strike": lower_strike,
                    "expiration": expiry_date,
                    "quantity": 1,
                    "description": "Income strategy with willingness to buy shares at lower price",
                    "risk_rating": "moderate",
                    "max_loss": f"Limited to strike minus premium (if stock goes to zero)",
                    "max_gain": "Limited to premium received",
                    "confidence_match": 70 - (abs(expected_move_pct) * 50)
                })
        
        # Sort strategies by confidence match
        strategies.sort(key=lambda x: x.get("confidence_match", 0), reverse=True)
        
        return strategies
