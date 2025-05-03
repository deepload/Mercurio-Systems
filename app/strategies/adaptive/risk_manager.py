"""
Risk Management Module

This module provides comprehensive risk management capabilities for the
adaptive day trading system, including position sizing, stop-loss calculations,
volatility-based risk adjustment, and portfolio exposure management.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime, timedelta
from enum import Enum

from app.strategies.adaptive.market_analyzer import MarketAnalyzer
from app.strategies.adaptive.strategy_selector import MarketRegime

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk level classifications"""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"

class RiskManager:
    """
    Comprehensive risk management system for day trading
    that adapts to market conditions and adjusts position sizing
    and stop-loss levels accordingly.
    """
    
    def __init__(self, 
                 initial_capital: float = 10000.0,
                 max_position_size_pct: float = 0.05,  # 5% of capital per position
                 max_portfolio_risk_pct: float = 0.5,  # 50% of capital at risk maximum
                 base_risk_per_trade_pct: float = 0.01,  # 1% risk per trade
                 volatility_risk_adjustment: bool = True,
                 market_regime_adjustment: bool = True,
                 default_stop_loss_atr_multiple: float = 1.5,
                 default_take_profit_atr_multiple: float = 2.0):
        """
        Initialize the risk manager.
        
        Args:
            initial_capital: Initial capital amount
            max_position_size_pct: Maximum position size as percentage of capital
            max_portfolio_risk_pct: Maximum portfolio risk as percentage of capital
            base_risk_per_trade_pct: Base risk per trade as percentage of capital
            volatility_risk_adjustment: Whether to adjust risk based on volatility
            market_regime_adjustment: Whether to adjust risk based on market regime
            default_stop_loss_atr_multiple: Default stop loss distance in ATR multiples
            default_take_profit_atr_multiple: Default take profit distance in ATR multiples
        """
        self.capital = initial_capital
        self.max_position_size_pct = max_position_size_pct
        self.max_portfolio_risk_pct = max_portfolio_risk_pct
        self.base_risk_per_trade_pct = base_risk_per_trade_pct
        self.volatility_risk_adjustment = volatility_risk_adjustment
        self.market_regime_adjustment = market_regime_adjustment
        self.default_stop_loss_atr_multiple = default_stop_loss_atr_multiple
        self.default_take_profit_atr_multiple = default_take_profit_atr_multiple
        
        # State tracking
        self.positions = {}
        self.open_risk = 0.0
        self.realized_pnl = 0.0
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.risk_history = []
        self.trade_history = []
        self.current_risk_level = RiskLevel.MODERATE
        
        logger.info(f"Risk Manager initialized with capital: ${initial_capital:.2f}, "
                   f"base risk per trade: {base_risk_per_trade_pct*100:.2f}%")
    
    def update_capital(self, new_capital: float) -> None:
        """
        Update the capital amount.
        
        Args:
            new_capital: New capital amount
        """
        old_capital = self.capital
        self.capital = max(0.0, new_capital)
        
        # Track drawdown
        if new_capital < old_capital:
            drawdown = (old_capital - new_capital) / old_capital
            self.current_drawdown = drawdown
            self.max_drawdown = max(self.max_drawdown, drawdown)
            
            logger.info(f"Capital updated to ${new_capital:.2f}, current drawdown: {drawdown*100:.2f}%")
        else:
            self.current_drawdown = 0.0
            logger.info(f"Capital updated to ${new_capital:.2f}")
    
    def calculate_position_size(self, 
                               symbol: str, 
                               price: float, 
                               stop_loss_price: float,
                               market_regime: MarketRegime = MarketRegime.UNKNOWN,
                               volatility: Optional[Dict[str, Any]] = None,
                               strategy_confidence: float = 0.5) -> Dict[str, Any]:
        """
        Calculate the optimal position size based on risk management rules.
        
        Args:
            symbol: Trading symbol
            price: Current price
            stop_loss_price: Stop loss price
            market_regime: Current market regime
            volatility: Volatility data dictionary
            strategy_confidence: Confidence level of the strategy (0-1)
            
        Returns:
            Dictionary with position size details
        """
        # Calculate base risk amount
        risk_amount = self.capital * self.base_risk_per_trade_pct
        
        # Adjust for market regime
        if self.market_regime_adjustment:
            regime_factor = self._get_regime_risk_factor(market_regime)
            risk_amount *= regime_factor
        
        # Adjust for volatility
        if self.volatility_risk_adjustment and volatility:
            vol_factor = self._get_volatility_risk_factor(volatility)
            risk_amount *= vol_factor
        
        # Adjust for strategy confidence
        confidence_factor = 0.5 + (strategy_confidence / 2)
        risk_amount *= confidence_factor
        
        # Check if stop is valid
        if abs(price - stop_loss_price) < 0.0001 or price == 0 or stop_loss_price == 0:
            logger.warning(f"Invalid stop loss for {symbol}: price=${price}, stop=${stop_loss_price}")
            return {
                "shares": 0,
                "risk_amount": 0,
                "risk_per_share": 0,
                "position_value": 0,
                "error": "invalid_stop"
            }
        
        # Calculate risk per share
        risk_per_share = abs(price - stop_loss_price)
        
        # Calculate position size in shares
        shares = risk_amount / risk_per_share
        
        # Check against maximum position size
        max_position_value = self.capital * self.max_position_size_pct
        position_value = shares * price
        
        if position_value > max_position_value:
            shares = max_position_value / price
            position_value = max_position_value
            risk_amount = shares * risk_per_share
        
        # Check portfolio risk limit
        new_total_risk = self.open_risk + risk_amount
        max_allowed_risk = self.capital * self.max_portfolio_risk_pct
        
        if new_total_risk > max_allowed_risk:
            excess_risk = new_total_risk - max_allowed_risk
            if excess_risk >= risk_amount:
                # Can't take this trade due to portfolio risk limits
                logger.warning(f"Trade for {symbol} rejected: portfolio risk limit reached")
                return {
                    "shares": 0,
                    "risk_amount": 0,
                    "risk_per_share": 0,
                    "position_value": 0,
                    "error": "portfolio_risk_limit"
                }
            else:
                # Reduce position size to fit within risk limit
                reduction_factor = 1 - (excess_risk / risk_amount)
                shares *= reduction_factor
                position_value = shares * price
                risk_amount *= reduction_factor
        
        # Return position sizing details
        return {
            "shares": shares,
            "risk_amount": risk_amount,
            "risk_per_share": risk_per_share,
            "position_value": position_value,
            "capital_risked_pct": risk_amount / self.capital * 100 if self.capital > 0 else 0
        }
    
    def calculate_exit_levels(self,
                             symbol: str,
                             entry_price: float,
                             direction: str,
                             data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate optimal stop-loss and take-profit levels.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            direction: Trade direction ('long' or 'short')
            data: Market data DataFrame
            
        Returns:
            Dictionary with exit levels
        """
        # Ensure data is sufficient
        if data is None or len(data) < 14 or 'atr' not in data.columns:
            logger.warning(f"Insufficient data for {symbol} to calculate exit levels")
            # Use percentage-based defaults
            stop_pct = 0.02  # 2% default stop
            take_profit_pct = 0.04  # 4% default target
            
            if direction == 'long':
                stop_loss = entry_price * (1 - stop_pct)
                take_profit = entry_price * (1 + take_profit_pct)
            else:  # short
                stop_loss = entry_price * (1 + stop_pct)
                take_profit = entry_price * (1 - take_profit_pct)
                
            return {
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "trailing_stop_activation": take_profit,
                "method": "percentage"
            }
        
        # Get ATR for volatility-based stops
        atr = data['atr'].iloc[-1]
        
        # Calculate stop and target distances based on ATR
        stop_distance = atr * self.default_stop_loss_atr_multiple
        target_distance = atr * self.default_take_profit_atr_multiple
        
        # Find recent swing highs/lows to improve stop placement
        if len(data) >= 20:
            recent_data = data.iloc[-20:]
            
            if direction == 'long':
                # For long positions, find recent swing low
                recent_low = recent_data['low'].min()
                potential_stop = recent_low - (0.5 * atr)  # Add buffer
                
                # Only use swing if it would result in a better stop
                if potential_stop < entry_price and entry_price - potential_stop < stop_distance:
                    stop_loss = potential_stop
                else:
                    stop_loss = entry_price - stop_distance
                    
                take_profit = entry_price + target_distance
                
            else:  # short
                # For short positions, find recent swing high
                recent_high = recent_data['high'].max()
                potential_stop = recent_high + (0.5 * atr)  # Add buffer
                
                # Only use swing if it would result in a better stop
                if potential_stop > entry_price and potential_stop - entry_price < stop_distance:
                    stop_loss = potential_stop
                else:
                    stop_loss = entry_price + stop_distance
                    
                take_profit = entry_price - target_distance
        else:
            # Simple ATR-based stops if not enough data for swing analysis
            if direction == 'long':
                stop_loss = entry_price - stop_distance
                take_profit = entry_price + target_distance
            else:  # short
                stop_loss = entry_price + stop_distance
                take_profit = entry_price - target_distance
        
        # Calculate trailing stop activation level
        # (typically at the take profit level or when trade reaches 1R profit)
        trailing_activation = take_profit
        
        return {
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "trailing_stop_activation": trailing_activation,
            "method": "atr_based"
        }
    
    def register_position(self, 
                         symbol: str, 
                         entry_price: float, 
                         shares: float, 
                         direction: str,
                         stop_loss: float,
                         take_profit: float,
                         risk_amount: float) -> None:
        """
        Register a new position with the risk manager.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            shares: Position size in shares
            direction: Trade direction ('long' or 'short')
            stop_loss: Stop loss price
            take_profit: Take profit price
            risk_amount: Amount of capital at risk
        """
        position = {
            "symbol": symbol,
            "entry_price": entry_price,
            "shares": shares,
            "direction": direction,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_amount": risk_amount,
            "position_value": entry_price * shares,
            "entry_time": datetime.now(),
            "trailing_stop_active": False,
            "trailing_stop_level": stop_loss
        }
        
        self.positions[symbol] = position
        self.open_risk += risk_amount
        
        # Log the new position
        logger.info(f"New position registered for {symbol}: {shares} shares at ${entry_price:.2f}, "
                   f"direction: {direction}, risk: ${risk_amount:.2f}")
        
        # Record risk state
        self._record_risk_state()
    
    def update_position(self, 
                       symbol: str, 
                       current_price: float, 
                       update_stops: bool = True) -> Optional[Dict[str, Any]]:
        """
        Update a position with the current price and check if exit conditions are met.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            update_stops: Whether to update trailing stops
            
        Returns:
            Action to take ('exit', 'hold') and reason, or None if position doesn't exist
        """
        if symbol not in self.positions:
            return None
            
        position = self.positions[symbol]
        direction = position['direction']
        stop_loss = position['stop_loss']
        take_profit = position['take_profit']
        entry_price = position['entry_price']
        
        # Calculate unrealized P&L
        if direction == 'long':
            unrealized_pnl = (current_price - entry_price) * position['shares']
            unrealized_pnl_pct = (current_price / entry_price - 1) * 100
            
            # Check stop loss
            if current_price <= stop_loss:
                action = {
                    "action": "exit",
                    "reason": "stop_loss",
                    "unrealized_pnl": unrealized_pnl,
                    "unrealized_pnl_pct": unrealized_pnl_pct
                }
                return action
                
            # Check take profit
            if current_price >= take_profit:
                # If we want to let profits run, activate trailing stop instead
                if update_stops:
                    position['trailing_stop_active'] = True
                    position['trailing_stop_level'] = max(
                        position['trailing_stop_level'],
                        current_price * 0.97  # 3% trailing stop
                    )
                    logger.info(f"Take profit hit for {symbol}, trailing stop activated: ${position['trailing_stop_level']:.2f}")
                
                action = {
                    "action": "hold",
                    "reason": "take_profit_hit_trailing_active",
                    "unrealized_pnl": unrealized_pnl,
                    "unrealized_pnl_pct": unrealized_pnl_pct
                }
                return action
                
            # Check trailing stop if active
            if position['trailing_stop_active']:
                if current_price <= position['trailing_stop_level']:
                    action = {
                        "action": "exit",
                        "reason": "trailing_stop",
                        "unrealized_pnl": unrealized_pnl,
                        "unrealized_pnl_pct": unrealized_pnl_pct
                    }
                    return action
                
                # Update trailing stop level if price moves higher
                if update_stops:
                    position['trailing_stop_level'] = max(
                        position['trailing_stop_level'],
                        current_price * 0.97  # 3% trailing stop
                    )
        
        else:  # short position
            unrealized_pnl = (entry_price - current_price) * position['shares'] 
            unrealized_pnl_pct = (1 - current_price / entry_price) * 100
            
            # Check stop loss (for shorts, price goes up to hit stop)
            if current_price >= stop_loss:
                action = {
                    "action": "exit",
                    "reason": "stop_loss",
                    "unrealized_pnl": unrealized_pnl,
                    "unrealized_pnl_pct": unrealized_pnl_pct
                }
                return action
                
            # Check take profit (for shorts, price goes down to hit target)
            if current_price <= take_profit:
                # If we want to let profits run, activate trailing stop instead
                if update_stops:
                    position['trailing_stop_active'] = True
                    position['trailing_stop_level'] = min(
                        position['trailing_stop_level'] if position['trailing_stop_active'] else float('inf'),
                        current_price * 1.03  # 3% trailing stop
                    )
                    logger.info(f"Take profit hit for {symbol}, trailing stop activated: ${position['trailing_stop_level']:.2f}")
                
                action = {
                    "action": "hold",
                    "reason": "take_profit_hit_trailing_active",
                    "unrealized_pnl": unrealized_pnl,
                    "unrealized_pnl_pct": unrealized_pnl_pct
                }
                return action
                
            # Check trailing stop if active
            if position['trailing_stop_active']:
                if current_price >= position['trailing_stop_level']:
                    action = {
                        "action": "exit",
                        "reason": "trailing_stop",
                        "unrealized_pnl": unrealized_pnl,
                        "unrealized_pnl_pct": unrealized_pnl_pct
                    }
                    return action
                
                # Update trailing stop level if price moves lower
                if update_stops:
                    position['trailing_stop_level'] = min(
                        position['trailing_stop_level'],
                        current_price * 1.03  # 3% trailing stop
                    )
        
        # Default action is to hold
        action = {
            "action": "hold",
            "reason": "no_exit_condition",
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pnl_pct": unrealized_pnl_pct
        }
        return action
    
    def close_position(self, symbol: str, exit_price: float, reason: str) -> Dict[str, Any]:
        """
        Close a position and update performance metrics.
        
        Args:
            symbol: Trading symbol
            exit_price: Exit price
            reason: Reason for exit
            
        Returns:
            Trade performance details
        """
        if symbol not in self.positions:
            logger.warning(f"Cannot close position for {symbol}: position not found")
            return {"error": "position_not_found"}
            
        position = self.positions[symbol]
        entry_price = position['entry_price']
        shares = position['shares']
        direction = position['direction']
        risk_amount = position['risk_amount']
        
        # Calculate P&L
        if direction == 'long':
            pnl = (exit_price - entry_price) * shares
            pnl_r = pnl / risk_amount if risk_amount > 0 else 0
            pnl_pct = (exit_price / entry_price - 1) * 100
        else:  # short
            pnl = (entry_price - exit_price) * shares
            pnl_r = pnl / risk_amount if risk_amount > 0 else 0
            pnl_pct = (1 - exit_price / entry_price) * 100
        
        # Update realized PnL
        self.realized_pnl += pnl
        
        # Update capital
        self.capital += pnl
        
        # Reduce open risk
        self.open_risk = max(0, self.open_risk - risk_amount)
        
        # Record trade
        trade = {
            "symbol": symbol,
            "entry_time": position['entry_time'],
            "exit_time": datetime.now(),
            "duration": (datetime.now() - position['entry_time']).total_seconds() / 60,  # minutes
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "shares": shares,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "pnl_r": pnl_r,
            "exit_reason": reason
        }
        
        self.trade_history.append(trade)
        
        # Log the closed position
        logger.info(f"Position closed for {symbol}: {shares} shares, entry: ${entry_price:.2f}, "
                   f"exit: ${exit_price:.2f}, P&L: ${pnl:.2f} ({pnl_pct:.2f}%), reason: {reason}")
        
        # Remove position
        del self.positions[symbol]
        
        # Record risk state
        self._record_risk_state()
        
        return trade
    
    def get_portfolio_stats(self) -> Dict[str, Any]:
        """
        Get current portfolio statistics.
        
        Returns:
            Dictionary with portfolio statistics
        """
        # Calculate unrealized P&L
        unrealized_pnl = 0.0
        position_value = 0.0
        
        for symbol, position in self.positions.items():
            # In a real system, we would get the current price from the market data service
            # For now, we'll use the theoretical exit price if available
            if 'current_price' in position:
                current_price = position['current_price']
                
                if position['direction'] == 'long':
                    pos_pnl = (current_price - position['entry_price']) * position['shares']
                else:  # short
                    pos_pnl = (position['entry_price'] - current_price) * position['shares']
                    
                unrealized_pnl += pos_pnl
            
            position_value += position['position_value']
        
        # Calculate win rate from trade history
        num_trades = len(self.trade_history)
        if num_trades > 0:
            winning_trades = sum(1 for t in self.trade_history if t['pnl'] > 0)
            win_rate = winning_trades / num_trades * 100
            
            # Average R multiple for winning and losing trades
            winning_r = [t['pnl_r'] for t in self.trade_history if t['pnl'] > 0]
            losing_r = [t['pnl_r'] for t in self.trade_history if t['pnl'] <= 0]
            
            avg_win_r = sum(winning_r) / len(winning_r) if winning_r else 0
            avg_loss_r = sum(losing_r) / len(losing_r) if losing_r else 0
            
            # Expectancy
            expectancy = (win_rate / 100 * avg_win_r) + ((1 - win_rate / 100) * avg_loss_r)
        else:
            win_rate = 0
            avg_win_r = 0
            avg_loss_r = 0
            expectancy = 0
        
        stats = {
            "capital": self.capital,
            "open_positions": len(self.positions),
            "position_value": position_value,
            "open_risk": self.open_risk,
            "open_risk_pct": self.open_risk / self.capital * 100 if self.capital > 0 else 0,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "total_pnl": self.realized_pnl + unrealized_pnl,
            "total_pnl_pct": (self.realized_pnl + unrealized_pnl) / (self.capital - self.realized_pnl - unrealized_pnl) * 100 if self.capital > self.realized_pnl + unrealized_pnl else 0,
            "current_drawdown_pct": self.current_drawdown * 100,
            "max_drawdown_pct": self.max_drawdown * 100,
            "trades_executed": num_trades,
            "win_rate": win_rate,
            "expectancy": expectancy,
            "current_risk_level": self.current_risk_level.value
        }
        
        return stats
    
    def adjust_risk_level(self, drawdown_pct: Optional[float] = None, win_rate: Optional[float] = None) -> RiskLevel:
        """
        Adjust the risk level based on performance metrics.
        
        Args:
            drawdown_pct: Current drawdown percentage (0-100)
            win_rate: Current win rate percentage (0-100)
            
        Returns:
            New risk level
        """
        # Use current values if not provided
        if drawdown_pct is None:
            drawdown_pct = self.current_drawdown * 100
            
        if win_rate is None and len(self.trade_history) > 0:
            winning_trades = sum(1 for t in self.trade_history if t['pnl'] > 0)
            win_rate = winning_trades / len(self.trade_history) * 100
        elif win_rate is None:
            win_rate = 50  # Default
        
        # Determine risk level based on drawdown and win rate
        if drawdown_pct > 10 or win_rate < 30:
            risk_level = RiskLevel.VERY_LOW
        elif drawdown_pct > 7 or win_rate < 40:
            risk_level = RiskLevel.LOW
        elif drawdown_pct > 5 or win_rate < 45:
            risk_level = RiskLevel.MODERATE
        elif drawdown_pct < 3 and win_rate > 55:
            risk_level = RiskLevel.HIGH
        elif drawdown_pct < 1 and win_rate > 65:
            risk_level = RiskLevel.VERY_HIGH
        else:
            risk_level = RiskLevel.MODERATE
            
        # Update current risk level
        self.current_risk_level = risk_level
        logger.info(f"Risk level adjusted to {risk_level.value} (drawdown: {drawdown_pct:.2f}%, win rate: {win_rate:.2f}%)")
        
        return risk_level
    
    def _get_regime_risk_factor(self, regime: MarketRegime) -> float:
        """Get risk adjustment factor based on market regime"""
        regime_factors = {
            MarketRegime.BULLISH: 1.2,
            MarketRegime.BEARISH: 1.2,
            MarketRegime.TRENDING: 1.3,
            MarketRegime.SIDEWAYS: 0.8,
            MarketRegime.VOLATILE: 0.6,
            MarketRegime.REVERSAL: 0.7,
            MarketRegime.UNKNOWN: 1.0
        }
        
        # Further adjust based on current risk level
        risk_level_multipliers = {
            RiskLevel.VERY_LOW: 0.4,
            RiskLevel.LOW: 0.7,
            RiskLevel.MODERATE: 1.0,
            RiskLevel.HIGH: 1.3,
            RiskLevel.VERY_HIGH: 1.6
        }
        
        base_factor = regime_factors.get(regime, 1.0)
        risk_multiplier = risk_level_multipliers.get(self.current_risk_level, 1.0)
        
        return base_factor * risk_multiplier
    
    def _get_volatility_risk_factor(self, volatility: Dict[str, Any]) -> float:
        """Get risk adjustment factor based on volatility"""
        # Extract volatility metrics
        current_vol = volatility.get("current", 0)
        percentile = volatility.get("percentile", 50)
        trend = volatility.get("trend", "stable")
        
        # Base factor on volatility percentile (higher vol = lower position size)
        if percentile > 80:
            vol_factor = 0.7  # High volatility - reduce risk
        elif percentile < 20:
            vol_factor = 1.3  # Low volatility - increase risk
        else:
            # Linear scaling between 0.7 and 1.3
            vol_factor = 1.3 - (percentile / 100) * 0.6
            
        # Adjust for volatility trend
        if trend == "increasing":
            vol_factor *= 0.9  # Reduce risk if volatility is increasing
        elif trend == "decreasing":
            vol_factor *= 1.1  # Increase risk if volatility is decreasing
            
        return vol_factor
    
    def _record_risk_state(self) -> None:
        """Record the current risk state for analysis"""
        risk_state = {
            "timestamp": datetime.now(),
            "capital": self.capital,
            "open_risk": self.open_risk,
            "open_risk_pct": self.open_risk / self.capital * 100 if self.capital > 0 else 0,
            "positions": len(self.positions),
            "risk_level": self.current_risk_level.value,
            "drawdown": self.current_drawdown,
            "max_drawdown": self.max_drawdown
        }
        
        self.risk_history.append(risk_state)
        
        # Limit history size
        if len(self.risk_history) > 1000:
            self.risk_history = self.risk_history[-1000:]
