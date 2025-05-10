"""
Strategy Selector Module

This module provides functionality for dynamically selecting and switching between 
trading strategies based on market conditions, historical performance, and other metrics.
It's the core of the adaptive trading system.
"""
import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime, timedelta
from enum import Enum

from app.strategies.base import BaseStrategy
from app.db.models import TradeAction
from app.services.market_data import MarketDataService

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regime classification"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    TRENDING = "trending"
    REVERSAL = "reversal"
    UNKNOWN = "unknown"

class StrategySelector:
    """
    Dynamic strategy selector that can choose the optimal trading strategy
    based on current market conditions and historical performance.
    """
    
    def __init__(self, 
                 strategies: Dict[str, BaseStrategy],
                 lookback_period: int = 20,
                 performance_weight: float = 0.7,
                 regime_weight: float = 0.3,
                 min_confidence: float = 0.6):
        """
        Initialize the strategy selector.
        
        Args:
            strategies: Dictionary of strategy name to strategy object
            lookback_period: Number of periods to look back for performance evaluation
            performance_weight: Weight given to historical performance (0-1)
            regime_weight: Weight given to regime matching (0-1)
            min_confidence: Minimum confidence required for a signal
        """
        self.strategies = strategies
        self.lookback_period = lookback_period
        self.performance_weight = performance_weight
        self.regime_weight = regime_weight
        self.min_confidence = min_confidence
        
        # Initialize performance tracking
        self.strategy_performances = {name: [] for name in strategies.keys()}
        self.strategy_regime_matches = {name: {} for name in strategies.keys()}
        self.strategy_weights = {name: 1.0 / len(strategies) for name in strategies.keys()}
        
        # Market data service for regime detection
        self.market_data = MarketDataService()
        
        # State
        self.current_regime = MarketRegime.UNKNOWN
        self.last_update_time = None
        self.regime_history = []
    
    async def detect_market_regime(self, symbol: str, data: pd.DataFrame) -> MarketRegime:
        """
        Detect the current market regime based on price action.
        
        Args:
            symbol: The market symbol
            data: Market data DataFrame with OHLCV data
            
        Returns:
            Detected market regime
        """
        if data is None or len(data) < 20:
            return MarketRegime.UNKNOWN
        
        # Calculate returns and stats
        returns = data['close'].pct_change().dropna()
        if len(returns) < 10:
            return MarketRegime.UNKNOWN
            
        # Recent trend (15-period)
        recent_trend = data['close'].iloc[-1] / data['close'].iloc[-15] - 1 if len(data) >= 15 else 0
        
        # Volatility (20-period)
        volatility = returns[-20:].std() * np.sqrt(252)  # Annualized volatility
        
        # Average daily volume change
        volume_changes = data['volume'].pct_change().dropna()
        avg_volume_change = volume_changes[-10:].mean() if len(volume_changes) >= 10 else 0
        
        # Heikin-Ashi for trend detection
        ha_close = (data['open'] + data['high'] + data['low'] + data['close']) / 4
        ha_open = ha_close.shift(1)
        ha_trend = (ha_close > ha_open).sum() / len(ha_close)  # Percentage of up candles
        
        # Regime classification based on metrics
        if volatility > 0.4:  # High volatility
            regime = MarketRegime.VOLATILE
        elif abs(recent_trend) > 0.08:  # Strong trend (8%+ move)
            if recent_trend > 0:
                regime = MarketRegime.BULLISH
            else:
                regime = MarketRegime.BEARISH
        elif 0.45 < ha_trend < 0.55:  # Sideways market
            regime = MarketRegime.SIDEWAYS
        elif ha_trend > 0.6:  # Consistent uptrend
            regime = MarketRegime.TRENDING
        elif ha_trend < 0.4:  # Potential reversal
            regime = MarketRegime.REVERSAL
        else:
            regime = MarketRegime.UNKNOWN
        
        # Update regime history
        self.current_regime = regime
        self.last_update_time = datetime.now()
        self.regime_history.append((datetime.now(), regime))
        
        # Trim history to keep only recent entries
        if len(self.regime_history) > 100:
            self.regime_history = self.regime_history[-100:]
            
        logger.info(f"Market regime for {symbol}: {regime.value}")
        return regime
    
    async def select_best_strategy(self, symbol: str, data: pd.DataFrame) -> Tuple[BaseStrategy, float]:
        """
        Select the best strategy for the current market conditions.
        
        Args:
            symbol: The market symbol
            data: Market data DataFrame
            
        Returns:
            Tuple of (best_strategy, confidence)
        """
        # Detect current market regime
        regime = await self.detect_market_regime(symbol, data)
        
        # Prepare scores for each strategy
        strategy_scores = {}
        
        for name, strategy in self.strategies.items():
            # Performance score based on historical accuracy
            perf_score = self._calculate_performance_score(name)
            
            # Regime match score based on how well the strategy performs in this regime
            regime_score = self._calculate_regime_match_score(name, regime)
            
            # Combined score using weighted average
            combined_score = (
                perf_score * self.performance_weight + 
                regime_score * self.regime_weight
            )
            
            strategy_scores[name] = combined_score
        
        # Select strategy with highest score
        if not strategy_scores:
            # Default to first strategy if no scores
            best_strategy_name = next(iter(self.strategies.keys()), None)
            confidence = 0.5
        else:
            best_strategy_name = max(strategy_scores, key=strategy_scores.get)
            confidence = strategy_scores[best_strategy_name]
        
        if best_strategy_name:
            best_strategy = self.strategies[best_strategy_name]
            logger.info(f"Selected {best_strategy_name} for {symbol} with confidence {confidence:.2f}")
            return best_strategy, confidence
        
        # Fallback to first strategy
        default_strategy_name = next(iter(self.strategies.keys()), None)
        if default_strategy_name:
            return self.strategies[default_strategy_name], 0.5
        
        raise ValueError("No strategies available for selection")
    
    def _calculate_performance_score(self, strategy_name: str) -> float:
        """Calculate a score based on historical performance"""
        performances = self.strategy_performances.get(strategy_name, [])
        
        if not performances:
            return 0.5  # Neutral score if no history
            
        # Look at recent performance (limited by lookback_period)
        recent_perfs = performances[-self.lookback_period:]
        
        if not recent_perfs:
            return 0.5
            
        # Calculate success rate
        correct_predictions = sum(1 for p in recent_perfs if p.get("correct", False))
        success_rate = correct_predictions / len(recent_perfs)
        
        # Calculate average gain/loss ratio
        gains = [p.get("gain", 0) for p in recent_perfs if p.get("gain", 0) > 0]
        losses = [abs(p.get("gain", 0)) for p in recent_perfs if p.get("gain", 0) < 0]
        
        avg_gain = sum(gains) / len(gains) if gains else 0
        avg_loss = sum(losses) / len(losses) if losses else 1  # Avoid division by zero
        
        gain_loss_ratio = avg_gain / avg_loss if avg_loss > 0 else 1
        
        # Combined score (weighted average of success rate and gain/loss ratio)
        score = 0.7 * success_rate + 0.3 * min(1.0, gain_loss_ratio/2)
        
        return score
    
    def _calculate_regime_match_score(self, strategy_name: str, regime: MarketRegime) -> float:
        """Calculate how well a strategy performs in the given market regime"""
        regime_matches = self.strategy_regime_matches.get(strategy_name, {})
        
        # Get performance data for this regime
        regime_data = regime_matches.get(regime.value, {})
        
        # If we have no data for this regime, use a default score
        if not regime_data:
            # Use predefined affinities for strategies if we don't have historical data
            strategy_affinities = {
                "MovingAverageStrategy": {
                    MarketRegime.TRENDING.value: 0.8,
                    MarketRegime.BULLISH.value: 0.7,
                    MarketRegime.BEARISH.value: 0.7,
                    MarketRegime.SIDEWAYS.value: 0.3,
                    MarketRegime.VOLATILE.value: 0.4,
                    MarketRegime.REVERSAL.value: 0.3,
                    MarketRegime.UNKNOWN.value: 0.5
                },
                "LSTMPredictorStrategy": {
                    MarketRegime.TRENDING.value: 0.7,
                    MarketRegime.BULLISH.value: 0.6,
                    MarketRegime.BEARISH.value: 0.6,
                    MarketRegime.SIDEWAYS.value: 0.5,
                    MarketRegime.VOLATILE.value: 0.7,
                    MarketRegime.REVERSAL.value: 0.6,
                    MarketRegime.UNKNOWN.value: 0.5
                },
                "TransformerStrategy": {
                    MarketRegime.TRENDING.value: 0.6,
                    MarketRegime.BULLISH.value: 0.7,
                    MarketRegime.BEARISH.value: 0.7,
                    MarketRegime.SIDEWAYS.value: 0.5,
                    MarketRegime.VOLATILE.value: 0.8,
                    MarketRegime.REVERSAL.value: 0.8,
                    MarketRegime.UNKNOWN.value: 0.6
                },
                "MSIStrategy": {
                    MarketRegime.TRENDING.value: 0.6,
                    MarketRegime.BULLISH.value: 0.5,
                    MarketRegime.BEARISH.value: 0.5,
                    MarketRegime.SIDEWAYS.value: 0.7,
                    MarketRegime.VOLATILE.value: 0.5,
                    MarketRegime.REVERSAL.value: 0.4,
                    MarketRegime.UNKNOWN.value: 0.5
                },
                "LLMStrategy": {
                    MarketRegime.TRENDING.value: 0.5,
                    MarketRegime.BULLISH.value: 0.6,
                    MarketRegime.BEARISH.value: 0.6,
                    MarketRegime.SIDEWAYS.value: 0.5,
                    MarketRegime.VOLATILE.value: 0.7,
                    MarketRegime.REVERSAL.value: 0.7,
                    MarketRegime.UNKNOWN.value: 0.6
                }
            }
            
            # Get predefined affinity or default to 0.5
            for known_strategy in strategy_affinities:
                if known_strategy in strategy_name:
                    return strategy_affinities[known_strategy].get(regime.value, 0.5)
            
            return 0.5
        
        # Calculate score based on win rate in this regime
        trades = regime_data.get("trades", 0)
        wins = regime_data.get("wins", 0)
        
        if trades == 0:
            return 0.5
            
        win_rate = wins / trades
        
        # Also consider profitability
        profit = regime_data.get("profit", 0)
        avg_profit_per_trade = profit / trades if trades > 0 else 0
        
        # Normalize profit score (0.01 = 1% profit per trade is excellent)
        profit_score = min(1.0, avg_profit_per_trade / 0.01)
        
        # Combined score
        score = 0.7 * win_rate + 0.3 * profit_score
        
        return score
    
    def update_strategy_performance(self, strategy_name: str, correct: bool, 
                                   confidence: float, gain: float, regime: MarketRegime) -> None:
        """
        Update the performance record for a strategy.
        
        Args:
            strategy_name: Name of the strategy
            correct: Whether the prediction was correct
            confidence: Confidence level of the prediction
            gain: Percentage gain/loss from the trade
            regime: Market regime during the prediction
        """
        # Update performance history
        performance = {
            "timestamp": datetime.now(),
            "correct": correct,
            "confidence": confidence,
            "gain": gain,
            "regime": regime.value
        }
        
        self.strategy_performances.setdefault(strategy_name, []).append(performance)
        
        # Limit history size
        if len(self.strategy_performances[strategy_name]) > 100:
            self.strategy_performances[strategy_name] = self.strategy_performances[strategy_name][-100:]
        
        # Update regime-specific performance
        regime_data = self.strategy_regime_matches.setdefault(strategy_name, {}).setdefault(regime.value, {
            "trades": 0,
            "wins": 0,
            "profit": 0.0
        })
        
        regime_data["trades"] = regime_data.get("trades", 0) + 1
        regime_data["wins"] = regime_data.get("wins", 0) + (1 if correct else 0)
        regime_data["profit"] = regime_data.get("profit", 0.0) + gain
        
        # Recalculate strategy weights
        self._recalculate_strategy_weights()
    
    def _recalculate_strategy_weights(self) -> None:
        """Recalculate the weights for all strategies based on performance"""
        scores = {}
        
        for name in self.strategies.keys():
            score = self._calculate_performance_score(name)
            scores[name] = max(0.1, score)  # Ensure minimum weight of 10%
        
        # Normalize to sum to 1.0
        total_score = sum(scores.values())
        
        if total_score > 0:
            self.strategy_weights = {name: score / total_score for name, score in scores.items()}
        else:
            # Equal weights if no scores available
            equal_weight = 1.0 / len(self.strategies)
            self.strategy_weights = {name: equal_weight for name in self.strategies.keys()}
        
        logger.info(f"Updated strategy weights: {self.strategy_weights}")
