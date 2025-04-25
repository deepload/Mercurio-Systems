"""
MercurioAI Advanced Risk Management

This module provides sophisticated risk management capabilities for trading strategies,
including position sizing, drawdown protection, and portfolio-level risk controls.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime, timedelta
import json
from pathlib import Path
import asyncio

from .event_bus import EventBus, EventType

logger = logging.getLogger(__name__)

class RiskProfile:
    """Configuration for a risk profile that can be applied to strategies"""
    
    def __init__(self, 
                 name: str,
                 max_position_size: float = 0.02,
                 max_drawdown: float = 0.20,
                 max_daily_loss: float = 0.05,
                 position_scaling: str = "fixed",
                 stop_loss_pct: float = 0.05,
                 take_profit_pct: float = 0.15,
                 correlation_limit: float = 0.7,
                 var_limit: float = 0.02,
                 volatility_adjustment: bool = True):
        """
        Initialize risk profile
        
        Args:
            name: Profile name
            max_position_size: Maximum position size as percentage of portfolio (0.02 = 2%)
            max_drawdown: Maximum drawdown allowed before reducing exposure (0.20 = 20%)
            max_daily_loss: Maximum daily loss allowed (0.05 = 5%)
            position_scaling: Position sizing method ('fixed', 'volatility', 'kelly')
            stop_loss_pct: Default stop-loss percentage (0.05 = 5%)
            take_profit_pct: Default take-profit percentage (0.15 = 15%)
            correlation_limit: Maximum correlation allowed between positions (0.7 = 70%)
            var_limit: Value at Risk limit as percentage of portfolio (0.02 = 2%)
            volatility_adjustment: Whether to adjust position sizes based on volatility
        """
        self.name = name
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown
        self.max_daily_loss = max_daily_loss
        self.position_scaling = position_scaling
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.correlation_limit = correlation_limit
        self.var_limit = var_limit
        self.volatility_adjustment = volatility_adjustment
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary"""
        return {
            'name': self.name,
            'max_position_size': self.max_position_size,
            'max_drawdown': self.max_drawdown,
            'max_daily_loss': self.max_daily_loss,
            'position_scaling': self.position_scaling,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'correlation_limit': self.correlation_limit,
            'var_limit': self.var_limit,
            'volatility_adjustment': self.volatility_adjustment
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RiskProfile':
        """Create profile from dictionary"""
        return cls(
            name=data.get('name', 'default'),
            max_position_size=data.get('max_position_size', 0.02),
            max_drawdown=data.get('max_drawdown', 0.20),
            max_daily_loss=data.get('max_daily_loss', 0.05),
            position_scaling=data.get('position_scaling', 'fixed'),
            stop_loss_pct=data.get('stop_loss_pct', 0.05),
            take_profit_pct=data.get('take_profit_pct', 0.15),
            correlation_limit=data.get('correlation_limit', 0.7),
            var_limit=data.get('var_limit', 0.02),
            volatility_adjustment=data.get('volatility_adjustment', True)
        )


class PositionSizer:
    """Position sizing calculator for risk management"""
    
    def __init__(self, risk_profile: RiskProfile):
        """
        Initialize position sizer
        
        Args:
            risk_profile: Risk profile to use
        """
        self.risk_profile = risk_profile
        
    def calculate_position_size(self, 
                               equity: float,
                               price: float,
                               volatility: Optional[float] = None,
                               signal_strength: float = 1.0,
                               custom_risk: Optional[float] = None) -> float:
        """
        Calculate position size based on risk profile
        
        Args:
            equity: Current portfolio equity
            price: Current asset price
            volatility: Asset volatility (e.g., ATR)
            signal_strength: Strategy signal strength (0.0 to 1.0)
            custom_risk: Custom risk percentage override
            
        Returns:
            Position size in units
        """
        # Base position size
        risk_pct = custom_risk if custom_risk is not None else self.risk_profile.max_position_size
        
        # Adjust for signal strength
        risk_pct *= signal_strength
        
        if self.risk_profile.position_scaling == 'fixed':
            # Simple fixed percentage of equity
            position_value = equity * risk_pct
            
        elif self.risk_profile.position_scaling == 'volatility' and volatility is not None:
            # Volatility-adjusted position sizing
            # Target a specific dollar risk based on volatility
            target_risk_amount = equity * risk_pct
            if price > 0 and volatility > 0:
                vol_adjusted_size = target_risk_amount / (volatility * price)
                position_value = vol_adjusted_size * price
            else:
                position_value = equity * risk_pct
                
        elif self.risk_profile.position_scaling == 'kelly':
            # Kelly criterion (simplified)
            win_rate = 0.5  # Default if unknown
            win_loss_ratio = 2.0  # Default if unknown
            
            # Kelly formula: f* = (p * b - q) / b
            # Where p = win probability, q = loss probability, b = win/loss ratio
            kelly_pct = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
            
            # Limit Kelly to avoid excessive leverage
            kelly_pct = min(kelly_pct, risk_pct)
            position_value = equity * kelly_pct
            
        else:
            # Default to fixed percentage
            position_value = equity * risk_pct
        
        # Adjust for volatility if enabled
        if self.risk_profile.volatility_adjustment and volatility is not None:
            # Use historical volatility to scale position size
            # Higher volatility = smaller position
            baseline_vol = 0.02  # 2% daily volatility as baseline
            vol_adjustment = baseline_vol / max(volatility, 0.001)
            vol_adjustment = min(max(vol_adjustment, 0.5), 2.0)  # Limit adjustment range
            position_value *= vol_adjustment
        
        # Calculate number of units
        if price > 0:
            units = position_value / price
        else:
            units = 0
            
        return units
    
    def calculate_stop_loss(self, 
                           entry_price: float, 
                           position_size: float,
                           equity: float,
                           volatility: Optional[float] = None,
                           is_long: bool = True) -> float:
        """
        Calculate stop loss price
        
        Args:
            entry_price: Entry price
            position_size: Position size in units
            equity: Current portfolio equity
            volatility: Asset volatility (e.g., ATR)
            is_long: Whether position is long
            
        Returns:
            Stop loss price
        """
        # Default percentage-based stop loss
        stop_pct = self.risk_profile.stop_loss_pct
        
        # Adjust based on volatility if available
        if volatility is not None and entry_price > 0:
            # Use ATR-based stop loss
            vol_pct = volatility / entry_price
            
            # Use the larger of percentage or volatility-based stop
            stop_pct = max(stop_pct, vol_pct * 1.5)
        
        # Calculate stop price
        if is_long:
            stop_price = entry_price * (1 - stop_pct)
        else:
            stop_price = entry_price * (1 + stop_pct)
            
        return stop_price
    
    def calculate_take_profit(self,
                            entry_price: float,
                            stop_loss_price: float,
                            is_long: bool = True) -> float:
        """
        Calculate take profit price
        
        Args:
            entry_price: Entry price
            stop_loss_price: Stop loss price
            is_long: Whether position is long
            
        Returns:
            Take profit price
        """
        # Calculate risk in dollars
        risk = abs(entry_price - stop_loss_price)
        
        # Use risk:reward ratio - default 1:3
        reward = risk * 3
        
        # Calculate take profit price
        if is_long:
            take_profit = entry_price + reward
        else:
            take_profit = entry_price - reward
            
        return take_profit


class DrawdownManager:
    """Manager for tracking and responding to drawdowns"""
    
    def __init__(self, risk_profile: RiskProfile):
        """
        Initialize drawdown manager
        
        Args:
            risk_profile: Risk profile
        """
        self.risk_profile = risk_profile
        self.peak_equity = None
        self.current_drawdown = 0.0
        self.max_historical_drawdown = 0.0
        self.drawdown_start_date = None
        self.drawdown_history = []
        
    def update(self, current_equity: float, timestamp: Optional[Union[str, datetime]] = None) -> Dict[str, Any]:
        """
        Update drawdown calculations
        
        Args:
            current_equity: Current portfolio equity
            timestamp: Current timestamp
            
        Returns:
            Drawdown status
        """
        # Initialize peak if not set
        if self.peak_equity is None:
            self.peak_equity = current_equity
            
        # Convert timestamp if needed
        if timestamp is not None and isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        
        # Calculate current drawdown
        if current_equity < self.peak_equity:
            self.current_drawdown = 1 - (current_equity / self.peak_equity)
            
            # Record drawdown start if this is a new drawdown
            if self.drawdown_start_date is None:
                self.drawdown_start_date = timestamp
                
            # Update max historical drawdown
            if self.current_drawdown > self.max_historical_drawdown:
                self.max_historical_drawdown = self.current_drawdown
                
        elif current_equity > self.peak_equity:
            # New peak, reset drawdown
            if self.current_drawdown > 0:
                # Record drawdown in history
                if self.drawdown_start_date is not None:
                    self.drawdown_history.append({
                        'start_date': self.drawdown_start_date,
                        'end_date': timestamp,
                        'depth': self.current_drawdown,
                        'recovery': True
                    })
            
            self.peak_equity = current_equity
            self.current_drawdown = 0.0
            self.drawdown_start_date = None
        
        # Return status
        status = {
            'current_drawdown': self.current_drawdown,
            'max_historical_drawdown': self.max_historical_drawdown,
            'peak_equity': self.peak_equity,
            'current_equity': current_equity,
            'in_drawdown': self.current_drawdown > 0,
            'drawdown_exceeded': self.current_drawdown > self.risk_profile.max_drawdown
        }
        
        return status
    
    def get_position_adjustment(self) -> float:
        """
        Get position size adjustment factor based on drawdown
        
        Returns:
            Adjustment factor (1.0 = no adjustment, <1.0 = reduce size)
        """
        # No adjustment if no drawdown
        if self.current_drawdown == 0:
            return 1.0
            
        # Linear reduction based on drawdown percentage
        # Start reducing at 50% of max drawdown
        threshold = self.risk_profile.max_drawdown * 0.5
        
        if self.current_drawdown < threshold:
            return 1.0
        elif self.current_drawdown >= self.risk_profile.max_drawdown:
            return 0.25  # Reduce to 25% size at max drawdown
        else:
            # Linear reduction between threshold and max
            reduction_range = self.risk_profile.max_drawdown - threshold
            excess_drawdown = self.current_drawdown - threshold
            reduction_factor = excess_drawdown / reduction_range
            
            # Scale from 100% to 25%
            return 1.0 - (0.75 * reduction_factor)
    
    def should_pause_trading(self) -> bool:
        """
        Check if trading should be paused due to drawdown
        
        Returns:
            True if trading should be paused
        """
        # Pause if drawdown exceeds max plus buffer
        critical_threshold = self.risk_profile.max_drawdown * 1.2
        return self.current_drawdown > critical_threshold


class VaRCalculator:
    """Value at Risk calculator"""
    
    def __init__(self, confidence_level: float = 0.95, time_horizon: int = 1):
        """
        Initialize VaR calculator
        
        Args:
            confidence_level: Confidence level (0.0 to 1.0)
            time_horizon: Time horizon in days
        """
        self.confidence_level = confidence_level
        self.time_horizon = time_horizon
        
    def calculate_historical_var(self, 
                               returns: pd.Series, 
                               portfolio_value: float) -> float:
        """
        Calculate historical VaR
        
        Args:
            returns: Historical returns series
            portfolio_value: Current portfolio value
            
        Returns:
            VaR in currency units
        """
        if len(returns) < 30:
            logger.warning(f"Too few data points ({len(returns)}) for reliable VaR calculation")
            # Fallback to a conservative estimate
            return portfolio_value * 0.02  # Assume 2% daily VaR
            
        # Sort returns (ascending)
        sorted_returns = returns.sort_values()
        
        # Find the return at the specified confidence level
        var_percentile = 1 - self.confidence_level
        var_return = sorted_returns.quantile(var_percentile)
        
        # Scale by portfolio value
        var_amount = abs(var_return * portfolio_value)
        
        # Adjust for time horizon
        var_amount = var_amount * np.sqrt(self.time_horizon)
        
        return var_amount
    
    def calculate_parametric_var(self,
                               returns: pd.Series,
                               portfolio_value: float) -> float:
        """
        Calculate parametric VaR (assuming normal distribution)
        
        Args:
            returns: Historical returns series
            portfolio_value: Current portfolio value
            
        Returns:
            VaR in currency units
        """
        if len(returns) < 30:
            logger.warning(f"Too few data points ({len(returns)}) for reliable VaR calculation")
            return portfolio_value * 0.02  # Assume 2% daily VaR
            
        # Calculate mean and standard deviation
        mu = returns.mean()
        sigma = returns.std()
        
        # Calculate Z-score for confidence level
        from scipy import stats
        z_score = stats.norm.ppf(1 - self.confidence_level)
        
        # Calculate VaR
        var_return = mu + (sigma * z_score)
        var_amount = abs(var_return * portfolio_value)
        
        # Adjust for time horizon
        var_amount = var_amount * np.sqrt(self.time_horizon)
        
        return var_amount
    
    def calculate_conditional_var(self,
                                returns: pd.Series,
                                portfolio_value: float) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall)
        
        Args:
            returns: Historical returns series
            portfolio_value: Current portfolio value
            
        Returns:
            CVaR in currency units
        """
        if len(returns) < 30:
            logger.warning(f"Too few data points ({len(returns)}) for reliable CVaR calculation")
            return portfolio_value * 0.03  # Assume 3% daily CVaR
            
        # Sort returns (ascending)
        sorted_returns = returns.sort_values()
        
        # Find the VaR threshold
        var_percentile = 1 - self.confidence_level
        var_threshold = sorted_returns.quantile(var_percentile)
        
        # Get returns beyond VaR (worse than VaR)
        extreme_returns = sorted_returns[sorted_returns <= var_threshold]
        
        # Calculate average of extreme returns (CVaR)
        cvar_return = extreme_returns.mean()
        cvar_amount = abs(cvar_return * portfolio_value)
        
        # Adjust for time horizon
        cvar_amount = cvar_amount * np.sqrt(self.time_horizon)
        
        return cvar_amount


class PortfolioRiskManager:
    """Portfolio-level risk management"""
    
    def __init__(self, risk_profile: RiskProfile):
        """
        Initialize portfolio risk manager
        
        Args:
            risk_profile: Risk profile
        """
        self.risk_profile = risk_profile
        self.event_bus = EventBus()
        self.position_sizer = PositionSizer(risk_profile)
        self.drawdown_manager = DrawdownManager(risk_profile)
        self.var_calculator = VaRCalculator()
        
        # Portfolio state
        self.positions = {}  # symbol -> quantity
        self.capital = 0.0
        self.equity = 0.0
        self.var = 0.0
        self.correlations = None
        
        # Historical data
        self.historical_returns = {}  # symbol -> returns series
        
    def set_portfolio_state(self, 
                          positions: Dict[str, float],
                          capital: float,
                          equity: float,
                          timestamp: Optional[Union[str, datetime]] = None):
        """
        Update portfolio state
        
        Args:
            positions: Current positions (symbol -> quantity)
            capital: Cash/capital
            equity: Total portfolio equity
            timestamp: Current timestamp
        """
        self.positions = positions
        self.capital = capital
        self.equity = equity
        
        # Update drawdown
        drawdown_status = self.drawdown_manager.update(equity, timestamp)
        
        # Log significant drawdowns
        if drawdown_status['current_drawdown'] > self.risk_profile.max_drawdown:
            logger.warning(f"Maximum drawdown exceeded: {drawdown_status['current_drawdown']:.2%}")
            
            # Publish event
            asyncio.create_task(self.event_bus.publish(
                EventType.RISK_LIMIT_EXCEEDED,
                {
                    'type': 'drawdown',
                    'value': drawdown_status['current_drawdown'],
                    'threshold': self.risk_profile.max_drawdown,
                    'timestamp': timestamp
                }
            ))
    
    def update_historical_data(self, symbol: str, returns: pd.Series):
        """
        Update historical return data for a symbol
        
        Args:
            symbol: Symbol
            returns: Returns series
        """
        self.historical_returns[symbol] = returns
        
        # Recalculate correlations if we have enough data
        if len(self.historical_returns) > 1:
            self._calculate_correlations()
            
        # Calculate portfolio VaR if we have positions
        if self.positions and symbol in self.positions and self.positions[symbol] > 0:
            self._calculate_portfolio_var()
    
    def _calculate_correlations(self):
        """Calculate correlation matrix for symbols in portfolio"""
        # Create DataFrame of returns
        returns_df = pd.DataFrame(self.historical_returns)
        
        # Calculate correlation matrix
        self.correlations = returns_df.corr()
    
    def _calculate_portfolio_var(self):
        """Calculate portfolio Value at Risk"""
        # Skip if no historical data or positions
        if not self.historical_returns or not self.positions:
            return
            
        # Create portfolio returns series
        portfolio_returns = pd.Series(0, index=next(iter(self.historical_returns.values())).index)
        
        for symbol, quantity in self.positions.items():
            if symbol in self.historical_returns:
                # Weight by position size
                position_value = quantity  # Simplified, should be quantity * price
                weight = position_value / self.equity if self.equity > 0 else 0
                portfolio_returns += self.historical_returns[symbol] * weight
        
        # Calculate VaR
        self.var = self.var_calculator.calculate_historical_var(portfolio_returns, self.equity)
        
        # Check if VaR exceeds limit
        var_pct = self.var / self.equity if self.equity > 0 else 0
        if var_pct > self.risk_profile.var_limit:
            logger.warning(f"VaR limit exceeded: {var_pct:.2%} > {self.risk_profile.var_limit:.2%}")
            
            # Publish event
            asyncio.create_task(self.event_bus.publish(
                EventType.RISK_LIMIT_EXCEEDED,
                {
                    'type': 'var',
                    'value': var_pct,
                    'threshold': self.risk_profile.var_limit,
                    'var_amount': self.var
                }
            ))
    
    def calculate_position_size(self,
                               symbol: str,
                               price: float,
                               volatility: Optional[float] = None,
                               signal_strength: float = 1.0) -> float:
        """
        Calculate position size with all risk factors
        
        Args:
            symbol: Symbol
            price: Current price
            volatility: Volatility (e.g., ATR)
            signal_strength: Signal strength
            
        Returns:
            Position size in units
        """
        # Get base position size
        position_size = self.position_sizer.calculate_position_size(
            self.equity,
            price,
            volatility,
            signal_strength
        )
        
        # Apply drawdown adjustment
        drawdown_adjustment = self.drawdown_manager.get_position_adjustment()
        position_size *= drawdown_adjustment
        
        # Apply correlation adjustment
        correlation_adjustment = self._calculate_correlation_adjustment(symbol)
        position_size *= correlation_adjustment
        
        # Apply VaR adjustment
        var_adjustment = self._calculate_var_adjustment()
        position_size *= var_adjustment
        
        return position_size
    
    def _calculate_correlation_adjustment(self, symbol: str) -> float:
        """
        Calculate position adjustment based on correlation
        
        Args:
            symbol: Symbol
            
        Returns:
            Adjustment factor
        """
        # If no correlations or no other positions, no adjustment
        if self.correlations is None or len(self.positions) <= 1:
            return 1.0
            
        # If symbol not in correlation matrix, no adjustment
        if symbol not in self.correlations:
            return 1.0
            
        # Calculate average correlation with existing positions
        correlations = []
        for existing_symbol in self.positions:
            if existing_symbol != symbol and existing_symbol in self.correlations:
                correlation = self.correlations.loc[symbol, existing_symbol]
                if not pd.isna(correlation):
                    correlations.append(abs(correlation))
        
        if not correlations:
            return 1.0
            
        avg_correlation = sum(correlations) / len(correlations)
        
        # If correlation exceeds limit, reduce position size
        if avg_correlation > self.risk_profile.correlation_limit:
            # Linear reduction from 100% to 50% as correlation approaches 1
            correlation_range = 1.0 - self.risk_profile.correlation_limit
            excess_correlation = avg_correlation - self.risk_profile.correlation_limit
            reduction_factor = excess_correlation / correlation_range
            
            # Scale from 100% to 50%
            return 1.0 - (0.5 * reduction_factor)
        else:
            return 1.0
    
    def _calculate_var_adjustment(self) -> float:
        """
        Calculate position adjustment based on portfolio VaR
        
        Returns:
            Adjustment factor
        """
        if self.equity <= 0:
            return 1.0
            
        var_pct = self.var / self.equity
        
        # If VaR exceeds limit, reduce position size
        if var_pct > self.risk_profile.var_limit:
            # Linear reduction from 100% to 50% as VaR approaches 2x limit
            var_range = self.risk_profile.var_limit
            excess_var = var_pct - self.risk_profile.var_limit
            reduction_factor = min(excess_var / var_range, 1.0)
            
            # Scale from 100% to 50%
            return 1.0 - (0.5 * reduction_factor)
        else:
            return 1.0
    
    def check_risk_limits(self) -> Dict[str, Any]:
        """
        Check if any risk limits are exceeded
        
        Returns:
            Risk status
        """
        status = {
            'drawdown_status': {
                'current': self.drawdown_manager.current_drawdown,
                'max_allowed': self.risk_profile.max_drawdown,
                'exceeded': self.drawdown_manager.current_drawdown > self.risk_profile.max_drawdown
            },
            'var_status': {
                'current': self.var / self.equity if self.equity > 0 else 0,
                'max_allowed': self.risk_profile.var_limit,
                'exceeded': (self.var / self.equity if self.equity > 0 else 0) > self.risk_profile.var_limit
            },
            'should_pause': self.drawdown_manager.should_pause_trading()
        }
        
        return status
