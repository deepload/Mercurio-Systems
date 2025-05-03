"""
Market Analyzer Module

This module provides advanced market analysis capabilities to identify market conditions,
detect anomalies, and calculate various technical indicators to assist with trading decisions.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime, timedelta
from enum import Enum

from app.strategies.adaptive.strategy_selector import MarketRegime

logger = logging.getLogger(__name__)

class AnomalyType(Enum):
    """Types of market anomalies"""
    PRICE_SPIKE = "price_spike"
    VOLUME_SURGE = "volume_surge"
    VOLATILITY_EXPLOSION = "volatility_explosion"
    LIQUIDITY_GAP = "liquidity_gap"
    PATTERN_BREAK = "pattern_break"
    CORRELATION_BREAK = "correlation_break"
    NONE = "none"

class MarketAnalyzer:
    """
    Advanced market analyzer for detecting market conditions, anomalies,
    and generating rich contextual information for trading decisions.
    """
    
    def __init__(self, 
                 volatility_window: int = 20,
                 trend_window: int = 50,
                 volume_window: int = 10,
                 anomaly_threshold: float = 3.0):
        """
        Initialize the market analyzer.
        
        Args:
            volatility_window: Window size for volatility calculations
            trend_window: Window size for trend analysis
            volume_window: Window size for volume analysis
            anomaly_threshold: Threshold in standard deviations for anomaly detection
        """
        self.volatility_window = volatility_window
        self.trend_window = trend_window
        self.volume_window = volume_window
        self.anomaly_threshold = anomaly_threshold
        
        # State tracking
        self.symbol_states = {}
        self.correlations = {}
        self.anomalies = {}
    
    def analyze_market_data(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive analysis of market data.
        
        Args:
            symbol: Market symbol
            data: OHLCV data as DataFrame
            
        Returns:
            Dictionary of analysis results
        """
        if data is None or len(data) < max(self.volatility_window, self.trend_window, self.volume_window):
            logger.warning(f"Insufficient data for full analysis of {symbol}")
            return {"error": "insufficient_data"}
        
        # Calculate technical indicators
        data = self._add_technical_indicators(data)
        
        # Detect anomalies
        anomalies = self._detect_anomalies(symbol, data)
        
        # Determine market regime
        regime = self._determine_market_regime(data)
        
        # Calculate market strength
        strength = self._calculate_market_strength(data)
        
        # Get support and resistance levels
        support_resistance = self._find_support_resistance(data)
        
        # Overall market sentiment
        sentiment = self._calculate_market_sentiment(data)
        
        # Volatility analysis
        volatility = self._analyze_volatility(data)
        
        # Liquidity and spread analysis (placeholder, would need order book data)
        liquidity = {"score": 0.5, "spread_percentage": 0.001}
        
        # Update symbol state
        self.symbol_states[symbol] = {
            "last_price": data["close"].iloc[-1],
            "last_update": datetime.now(),
            "regime": regime.value,
            "anomalies": anomalies,
            "volatility": volatility,
            "strength": strength
        }
        
        # Return comprehensive analysis results
        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "last_close": data["close"].iloc[-1],
            "regime": regime.value,
            "strength": strength,
            "sentiment": sentiment,
            "volatility": volatility,
            "anomalies": anomalies,
            "support_resistance": support_resistance,
            "liquidity": liquidity,
            "indicators": {
                "rsi": data["rsi"].iloc[-1] if "rsi" in data else None,
                "macd": data["macd"].iloc[-1] if "macd" in data else None,
                "macd_signal": data["macd_signal"].iloc[-1] if "macd_signal" in data else None,
                "macd_hist": data["macd_hist"].iloc[-1] if "macd_hist" in data else None,
                "ema_20": data["ema_20"].iloc[-1] if "ema_20" in data else None,
                "ema_50": data["ema_50"].iloc[-1] if "ema_50" in data else None,
                "atr": data["atr"].iloc[-1] if "atr" in data else None
            }
        }
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe"""
        df = data.copy()
        
        # RSI
        close_delta = df['close'].diff()
        up = close_delta.clip(lower=0)
        down = -1 * close_delta.clip(upper=0)
        ma_up = up.rolling(window=14).mean()
        ma_down = down.rolling(window=14).mean()
        rsi = 100 - (100 / (1 + ma_up / ma_down))
        df['rsi'] = rsi
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist
        
        # Moving Averages
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        # Bollinger Bands
        df['sma_20'] = df['close'].rolling(window=20).mean()
        rolling_std = df['close'].rolling(window=20).std()
        df['bollinger_upper'] = df['sma_20'] + (rolling_std * 2)
        df['bollinger_lower'] = df['sma_20'] - (rolling_std * 2)
        
        # Average True Range (ATR)
        tr1 = abs(df['high'] - df['low'])
        tr2 = abs(df['high'] - df['close'].shift())
        tr3 = abs(df['low'] - df['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        # Return the dataframe with indicators
        return df
    
    def _detect_anomalies(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect market anomalies"""
        anomalies = {
            "detected": False,
            "types": [],
            "details": {}
        }
        
        # Check for price spikes
        returns = data['close'].pct_change()
        mean_return = returns.mean()
        std_return = returns.std()
        latest_return = returns.iloc[-1]
        
        if abs(latest_return - mean_return) > self.anomaly_threshold * std_return:
            anomalies["detected"] = True
            anomalies["types"].append(AnomalyType.PRICE_SPIKE.value)
            anomalies["details"]["price_spike"] = {
                "severity": abs(latest_return - mean_return) / (std_return + 1e-10),
                "return": latest_return,
                "direction": "up" if latest_return > 0 else "down"
            }
        
        # Check for volume surges
        if 'volume' in data.columns:
            volume_changes = data['volume'].pct_change()
            mean_volume_change = volume_changes.rolling(self.volume_window).mean().iloc[-1]
            std_volume_change = volume_changes.rolling(self.volume_window).std().iloc[-1]
            latest_volume_change = volume_changes.iloc[-1]
            
            if latest_volume_change > mean_volume_change + self.anomaly_threshold * std_volume_change:
                anomalies["detected"] = True
                anomalies["types"].append(AnomalyType.VOLUME_SURGE.value)
                anomalies["details"]["volume_surge"] = {
                    "severity": (latest_volume_change - mean_volume_change) / (std_volume_change + 1e-10),
                    "change": latest_volume_change,
                    "avg_volume": data['volume'].rolling(self.volume_window).mean().iloc[-1]
                }
        
        # Check for volatility explosions
        if 'atr' in data.columns:
            atr_changes = data['atr'].pct_change()
            mean_atr_change = atr_changes.rolling(self.volatility_window).mean().iloc[-1]
            std_atr_change = atr_changes.rolling(self.volatility_window).std().iloc[-1]
            latest_atr_change = atr_changes.iloc[-1]
            
            if latest_atr_change > mean_atr_change + self.anomaly_threshold * std_atr_change:
                anomalies["detected"] = True
                anomalies["types"].append(AnomalyType.VOLATILITY_EXPLOSION.value)
                anomalies["details"]["volatility_explosion"] = {
                    "severity": (latest_atr_change - mean_atr_change) / (std_atr_change + 1e-10),
                    "atr": data['atr'].iloc[-1],
                    "atr_change": latest_atr_change
                }
                
        # Record anomalies in state
        self.anomalies[symbol] = anomalies
        
        return anomalies
    
    def _determine_market_regime(self, data: pd.DataFrame) -> MarketRegime:
        """Determine the current market regime"""
        if data is None or len(data) < self.trend_window:
            return MarketRegime.UNKNOWN
            
        # Price trend
        recent_trend = data['close'].iloc[-1] / data['close'].iloc[-self.trend_window] - 1
        
        # Volatility
        returns = data['close'].pct_change().dropna()
        volatility = returns[-self.volatility_window:].std() * np.sqrt(252)  # Annualized
        
        # RSI - check if oversold/overbought
        latest_rsi = data['rsi'].iloc[-1] if 'rsi' in data else 50
        
        # Determine regime
        if volatility > 0.4:  # High volatility
            regime = MarketRegime.VOLATILE
        elif abs(recent_trend) > 0.1:  # Strong trend (10%+ move)
            if recent_trend > 0:
                regime = MarketRegime.BULLISH
            else:
                regime = MarketRegime.BEARISH
        elif latest_rsi > 70:  # Overbought
            regime = MarketRegime.REVERSAL
        elif latest_rsi < 30:  # Oversold
            regime = MarketRegime.REVERSAL
        elif data['ema_20'].iloc[-1] > data['ema_50'].iloc[-1]:  # Uptrend structure
            regime = MarketRegime.TRENDING
        elif abs(recent_trend) < 0.03:  # Very small range
            regime = MarketRegime.SIDEWAYS
        else:
            regime = MarketRegime.UNKNOWN
            
        return regime
    
    def _calculate_market_strength(self, data: pd.DataFrame) -> float:
        """Calculate market strength (0-1 scale)"""
        if data is None or len(data) < 20:
            return 0.5
            
        # Use RSI, MACD, and price vs moving averages to determine strength
        
        # RSI component (30-70 range mapped to 0-1)
        rsi = data['rsi'].iloc[-1] if 'rsi' in data else 50
        rsi_score = max(0, min(1, (rsi - 30) / 40))
        
        # MACD component
        macd = data['macd'].iloc[-1] if 'macd' in data else 0
        macd_signal = data['macd_signal'].iloc[-1] if 'macd_signal' in data else 0
        macd_score = 0.5
        if macd > 0 and macd > macd_signal:
            macd_score = 0.75
        elif macd > 0:
            macd_score = 0.6
        elif macd < 0 and macd < macd_signal:
            macd_score = 0.25
        elif macd < 0:
            macd_score = 0.4
            
        # Moving average component
        ma_score = 0.5
        if 'ema_20' in data and 'ema_50' in data:
            close = data['close'].iloc[-1]
            ema_20 = data['ema_20'].iloc[-1]
            ema_50 = data['ema_50'].iloc[-1]
            
            if close > ema_20 and ema_20 > ema_50:
                ma_score = 0.8  # Strong uptrend
            elif close > ema_20:
                ma_score = 0.7  # Uptrend
            elif close < ema_20 and ema_20 < ema_50:
                ma_score = 0.2  # Strong downtrend
            elif close < ema_20:
                ma_score = 0.3  # Downtrend
                
        # Combined strength score
        strength = 0.4 * rsi_score + 0.3 * macd_score + 0.3 * ma_score
        return strength
    
    def _find_support_resistance(self, data: pd.DataFrame) -> Dict[str, List[float]]:
        """Find support and resistance levels"""
        levels = {
            "support": [],
            "resistance": []
        }
        
        if data is None or len(data) < 30:
            return levels
            
        # Simple method: find recent highs and lows
        highs = data['high'].rolling(5, center=True).max()
        lows = data['low'].rolling(5, center=True).min()
        
        # Find local maxima and minima
        resistance_pts = []
        support_pts = []
        
        for i in range(2, len(data) - 2):
            # Potential resistance
            if highs.iloc[i] == data['high'].iloc[i] and \
               data['high'].iloc[i] > data['high'].iloc[i-1] and \
               data['high'].iloc[i] > data['high'].iloc[i-2] and \
               data['high'].iloc[i] > data['high'].iloc[i+1] and \
               data['high'].iloc[i] > data['high'].iloc[i+2]:
                resistance_pts.append(data['high'].iloc[i])
                
            # Potential support
            if lows.iloc[i] == data['low'].iloc[i] and \
               data['low'].iloc[i] < data['low'].iloc[i-1] and \
               data['low'].iloc[i] < data['low'].iloc[i-2] and \
               data['low'].iloc[i] < data['low'].iloc[i+1] and \
               data['low'].iloc[i] < data['low'].iloc[i+2]:
                support_pts.append(data['low'].iloc[i])
        
        # Group nearby levels
        current_price = data['close'].iloc[-1]
        
        # Get resistance levels above current price
        resistance_pts = [p for p in resistance_pts if p > current_price]
        resistance_pts.sort()
        
        # Get support levels below current price
        support_pts = [p for p in support_pts if p < current_price]
        support_pts.sort(reverse=True)
        
        # Return top 3 nearest levels
        levels["resistance"] = resistance_pts[:3]
        levels["support"] = support_pts[:3]
        
        return levels
    
    def _calculate_market_sentiment(self, data: pd.DataFrame) -> float:
        """Calculate overall market sentiment (-1 to 1 scale)"""
        if data is None or len(data) < 20:
            return 0.0
            
        # Combine multiple indicators to derive sentiment
        
        # Price momentum
        returns = data['close'].pct_change(5).iloc[-1] if len(data) >= 5 else 0
        
        # RSI
        rsi = data['rsi'].iloc[-1] if 'rsi' in data else 50
        rsi_sentiment = (rsi - 50) / 50  # -1 to 1 scale
        
        # MACD
        macd_hist = data['macd_hist'].iloc[-1] if 'macd_hist' in data else 0
        macd_sentiment = min(1, max(-1, macd_hist * 10))  # Scale and cap
        
        # Bollinger band position
        bb_position = 0
        if all(c in data.columns for c in ['close', 'bollinger_upper', 'bollinger_lower']):
            latest_close = data['close'].iloc[-1]
            upper_band = data['bollinger_upper'].iloc[-1]
            lower_band = data['bollinger_lower'].iloc[-1]
            band_width = upper_band - lower_band
            
            if band_width > 0:
                position = (latest_close - lower_band) / band_width  # 0 to 1
                bb_position = (position - 0.5) * 2  # -1 to 1
        
        # Combined sentiment
        sentiment = 0.3 * np.sign(returns) * min(1, abs(returns) * 10) + \
                    0.3 * rsi_sentiment + \
                    0.3 * macd_sentiment + \
                    0.1 * bb_position
                    
        return max(-1, min(1, sentiment))  # Ensure -1 to 1 range
    
    def _analyze_volatility(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market volatility"""
        volatility_data = {
            "current": 0,
            "historic": 0,
            "trend": "stable",
            "percentile": 50,
            "forecast": "moderate"
        }
        
        if data is None or len(data) < self.volatility_window:
            return volatility_data
            
        # Calculate realized volatility
        returns = data['close'].pct_change().dropna()
        current_vol = returns[-self.volatility_window:].std() * np.sqrt(252)  # Annualized
        
        if len(returns) >= self.volatility_window * 3:
            # Historical volatility
            hist_vol = returns[-self.volatility_window*3:-self.volatility_window].std() * np.sqrt(252)
            
            # Volatility trend
            if current_vol > hist_vol * 1.2:
                vol_trend = "increasing"
            elif current_vol < hist_vol * 0.8:
                vol_trend = "decreasing"
            else:
                vol_trend = "stable"
                
            # Calculate percentile
            all_vols = []
            for i in range(len(returns) - self.volatility_window + 1):
                window_vol = returns[i:i+self.volatility_window].std() * np.sqrt(252)
                all_vols.append(window_vol)
                
            if all_vols:
                vol_percentile = int(pd.Series(all_vols).rank(pct=True).iloc[-1] * 100)
            else:
                vol_percentile = 50
            
            # Forecast based on current level and trend
            if vol_percentile > 80:
                forecast = "high"
            elif vol_percentile < 20:
                forecast = "low"
            else:
                forecast = "moderate"
                
            # Update volatility data
            volatility_data.update({
                "current": current_vol,
                "historic": hist_vol,
                "trend": vol_trend,
                "percentile": vol_percentile,
                "forecast": forecast
            })
        else:
            # Limited data case
            volatility_data.update({
                "current": current_vol,
                "historic": current_vol,
                "trend": "stable",
                "percentile": 50,
                "forecast": "moderate"
            })
            
        return volatility_data
