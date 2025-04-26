"""
MercurioAI Technical Analyzer

Module d'analyse technique avancée utilisant TA-Lib et d'autres indicateurs personnalisés
pour enrichir les décisions de trading.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logging.warning("TA-Lib non disponible. Utilisation des indicateurs de fallback.")

logger = logging.getLogger(__name__)

class TechnicalAnalyzer:
    """
    Analyseur technique qui calcule des indicateurs avancés et génère des signaux
    basés sur l'analyse technique. Compatible avec le système de fallback de MercurioAI.
    """
    
    def __init__(self):
        self.indicators_cache = {}
    
    def clear_cache(self):
        """Vide le cache d'indicateurs"""
        self.indicators_cache = {}
    
    def analyze(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Analyse complète d'un DataFrame avec calcul d'indicateurs et génération de signaux.
        
        Args:
            data: DataFrame avec données OHLCV
            symbol: Symbole de trading
            
        Returns:
            Dictionnaire d'indicateurs et signaux
        """
        if data is None or data.empty or len(data) < 30:
            logger.warning(f"Données insuffisantes pour l'analyse technique de {symbol}")
            return {"error": "Données insuffisantes"}
        
        # Créer une copie pour éviter de modifier l'original
        df = data.copy()
        
        # Extraire les colonnes OHLCV
        open_price = df['open'].values
        high_price = df['high'].values
        low_price = df['low'].values
        close_price = df['close'].values
        volume = df['volume'].values
        
        # Calculer les indicateurs
        indicators = {}
        
        # Indicateurs de tendance
        indicators.update(self._calculate_trend_indicators(df, open_price, high_price, low_price, close_price))
        
        # Indicateurs de momentum
        indicators.update(self._calculate_momentum_indicators(df, close_price))
        
        # Indicateurs de volatilité
        indicators.update(self._calculate_volatility_indicators(df, high_price, low_price, close_price))
        
        # Indicateurs de volume
        indicators.update(self._calculate_volume_indicators(df, close_price, volume))
        
        # Configurations de chandeliers
        indicators.update(self._detect_candlestick_patterns(df, open_price, high_price, low_price, close_price))
        
        # Générer des signaux basés sur les indicateurs
        signals = self._generate_signals(indicators)
        
        # Mettre en cache pour ce symbole
        cache_key = f"{symbol}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}"
        self.indicators_cache[cache_key] = indicators
        
        return {
            "indicators": indicators,
            "signals": signals,
            "signal_strength": self._calculate_signal_strength(signals)
        }
    
    def _calculate_trend_indicators(self, df: pd.DataFrame, open_price: np.ndarray, 
                                   high_price: np.ndarray, low_price: np.ndarray, 
                                   close_price: np.ndarray) -> Dict[str, Any]:
        """Calcule les indicateurs de tendance"""
        indicators = {}
        
        # Moyennes mobiles
        if TALIB_AVAILABLE:
            indicators["sma_20"] = talib.SMA(close_price, timeperiod=20)
            indicators["sma_50"] = talib.SMA(close_price, timeperiod=50)
            indicators["sma_200"] = talib.SMA(close_price, timeperiod=200)
            indicators["ema_20"] = talib.EMA(close_price, timeperiod=20)
            indicators["ema_50"] = talib.EMA(close_price, timeperiod=50)
        else:
            # Fallback si TA-Lib n'est pas disponible
            indicators["sma_20"] = df['close'].rolling(window=20).mean().values
            indicators["sma_50"] = df['close'].rolling(window=50).mean().values
            indicators["sma_200"] = df['close'].rolling(window=200).mean().values
            indicators["ema_20"] = df['close'].ewm(span=20, adjust=False).mean().values
            indicators["ema_50"] = df['close'].ewm(span=50, adjust=False).mean().values
        
        # Indicateur de direction ADX
        if TALIB_AVAILABLE:
            indicators["adx"] = talib.ADX(high_price, low_price, close_price, timeperiod=14)
            indicators["plus_di"] = talib.PLUS_DI(high_price, low_price, close_price, timeperiod=14)
            indicators["minus_di"] = talib.MINUS_DI(high_price, low_price, close_price, timeperiod=14)
        else:
            # Fallback simplifié
            tr1 = pd.DataFrame(high_price - low_price)
            tr2 = pd.DataFrame(abs(high_price - np.roll(close_price, 1)))
            tr3 = pd.DataFrame(abs(low_price - np.roll(close_price, 1)))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            indicators["adx"] = atr.values  # Approximation simplifiée
            indicators["plus_di"] = np.zeros_like(close_price)
            indicators["minus_di"] = np.zeros_like(close_price)
        
        # Bandes de Bollinger
        if TALIB_AVAILABLE:
            upper, middle, lower = talib.BBANDS(close_price, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            indicators["bb_upper"] = upper
            indicators["bb_middle"] = middle
            indicators["bb_lower"] = lower
        else:
            sma = df['close'].rolling(window=20).mean()
            std = df['close'].rolling(window=20).std()
            indicators["bb_upper"] = (sma + (std * 2)).values
            indicators["bb_middle"] = sma.values
            indicators["bb_lower"] = (sma - (std * 2)).values
        
        # Calcul de la tendance actuelle
        if len(close_price) >= 50:
            short_trend = close_price[-1] / close_price[-10] - 1
            medium_trend = close_price[-1] / close_price[-30] - 1
            long_trend = close_price[-1] / close_price[-50] - 1
            
            indicators["trend_short"] = short_trend
            indicators["trend_medium"] = medium_trend
            indicators["trend_long"] = long_trend
            
            # Classifier la tendance
            if short_trend > 0.03 and medium_trend > 0.05:
                indicators["trend_direction"] = "strong_bullish"
            elif short_trend > 0.01 and medium_trend > 0:
                indicators["trend_direction"] = "bullish"
            elif short_trend < -0.03 and medium_trend < -0.05:
                indicators["trend_direction"] = "strong_bearish"
            elif short_trend < -0.01 and medium_trend < 0:
                indicators["trend_direction"] = "bearish"
            else:
                indicators["trend_direction"] = "neutral"
        
        return indicators
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame, close_price: np.ndarray) -> Dict[str, Any]:
        """Calcule les indicateurs de momentum"""
        indicators = {}
        
        # RSI
        if TALIB_AVAILABLE:
            indicators["rsi"] = talib.RSI(close_price, timeperiod=14)
        else:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            indicators["rsi"] = (100 - (100 / (1 + rs))).values
        
        # MACD
        if TALIB_AVAILABLE:
            macd, macd_signal, macd_hist = talib.MACD(close_price, fastperiod=12, slowperiod=26, signalperiod=9)
            indicators["macd"] = macd
            indicators["macd_signal"] = macd_signal
            indicators["macd_hist"] = macd_hist
        else:
            ema12 = df['close'].ewm(span=12, adjust=False).mean()
            ema26 = df['close'].ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            macd_signal = macd.ewm(span=9, adjust=False).mean()
            indicators["macd"] = macd.values
            indicators["macd_signal"] = macd_signal.values
            indicators["macd_hist"] = (macd - macd_signal).values
        
        # Stochastique
        if TALIB_AVAILABLE:
            slowk, slowd = talib.STOCH(high_price=df['high'].values, 
                                     low_price=df['low'].values, 
                                     close_price=close_price,
                                     fastk_period=14, slowk_period=3, slowd_period=3)
            indicators["stoch_k"] = slowk
            indicators["stoch_d"] = slowd
        else:
            n = 14
            low_min = df['low'].rolling(n).min()
            high_max = df['high'].rolling(n).max()
            k = 100 * ((df['close'] - low_min) / (high_max - low_min))
            indicators["stoch_k"] = k.rolling(3).mean().values
            indicators["stoch_d"] = k.rolling(3).mean().rolling(3).mean().values
        
        return indicators
    
    def _calculate_volatility_indicators(self, df: pd.DataFrame, high_price: np.ndarray, 
                                        low_price: np.ndarray, close_price: np.ndarray) -> Dict[str, Any]:
        """Calcule les indicateurs de volatilité"""
        indicators = {}
        
        # ATR - Average True Range
        if TALIB_AVAILABLE:
            indicators["atr"] = talib.ATR(high_price, low_price, close_price, timeperiod=14)
        else:
            tr1 = pd.DataFrame(high_price - low_price)
            tr2 = pd.DataFrame(abs(high_price - np.roll(close_price, 1)))
            tr3 = pd.DataFrame(abs(low_price - np.roll(close_price, 1)))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            indicators["atr"] = tr.rolling(14).mean().values
        
        # Volatilité historique
        if len(close_price) >= 30:
            returns = pd.Series(close_price).pct_change().dropna()
            indicators["volatility_10d"] = returns.rolling(window=10).std().values[-1] * np.sqrt(252)
            indicators["volatility_30d"] = returns.rolling(window=30).std().values[-1] * np.sqrt(252)
        
        # Keltner Channels
        if TALIB_AVAILABLE:
            typical_price = (high_price + low_price + close_price) / 3
            middle = talib.SMA(typical_price, timeperiod=20)
            atr = talib.ATR(high_price, low_price, close_price, timeperiod=14)
            indicators["keltner_upper"] = middle + (2 * atr)
            indicators["keltner_middle"] = middle
            indicators["keltner_lower"] = middle - (2 * atr)
        
        return indicators
    
    def _calculate_volume_indicators(self, df: pd.DataFrame, close_price: np.ndarray, 
                                    volume: np.ndarray) -> Dict[str, Any]:
        """Calcule les indicateurs basés sur le volume"""
        indicators = {}
        
        # On-Balance Volume (OBV)
        if TALIB_AVAILABLE:
            indicators["obv"] = talib.OBV(close_price, volume)
        else:
            obv = np.zeros_like(close_price)
            for i in range(1, len(close_price)):
                if close_price[i] > close_price[i-1]:
                    obv[i] = obv[i-1] + volume[i]
                elif close_price[i] < close_price[i-1]:
                    obv[i] = obv[i-1] - volume[i]
                else:
                    obv[i] = obv[i-1]
            indicators["obv"] = obv
        
        # Volume moyens
        vol_series = pd.Series(volume)
        indicators["volume_sma_20"] = vol_series.rolling(window=20).mean().values
        
        # Volume relatif
        if len(volume) >= 20:
            current_vol = volume[-1]
            avg_vol = np.mean(volume[-20:-1])
            indicators["relative_volume"] = current_vol / avg_vol if avg_vol > 0 else 1.0
        
        # Accumulation/Distribution
        ad = np.zeros_like(close_price)
        for i in range(len(close_price)):
            if high_price[i] != low_price[i]:
                clv = ((close_price[i] - low_price[i]) - (high_price[i] - close_price[i])) / (high_price[i] - low_price[i])
                ad[i] = ad[i-1] + clv * volume[i] if i > 0 else clv * volume[i]
            else:
                ad[i] = ad[i-1] if i > 0 else 0
        indicators["ad_line"] = ad
        
        return indicators
    
    def _detect_candlestick_patterns(self, df: pd.DataFrame, open_price: np.ndarray,
                                    high_price: np.ndarray, low_price: np.ndarray,
                                    close_price: np.ndarray) -> Dict[str, Any]:
        """Détecte les configurations de chandeliers japonais"""
        patterns = {}
        
        if TALIB_AVAILABLE:
            # Motifs de retournement haussiers
            patterns["hammer"] = talib.CDLHAMMER(open_price, high_price, low_price, close_price)
            patterns["inverted_hammer"] = talib.CDLINVERTEDHAMMER(open_price, high_price, low_price, close_price)
            patterns["bullish_engulfing"] = talib.CDLENGULFING(open_price, high_price, low_price, close_price)
            patterns["morning_star"] = talib.CDLMORNINGSTAR(open_price, high_price, low_price, close_price)
            
            # Motifs de retournement baissiers
            patterns["hanging_man"] = talib.CDLHANGINGMAN(open_price, high_price, low_price, close_price)
            patterns["shooting_star"] = talib.CDLSHOOTINGSTAR(open_price, high_price, low_price, close_price)
            patterns["bearish_engulfing"] = talib.CDLENGULFING(open_price, high_price, low_price, close_price)
            patterns["evening_star"] = talib.CDLEVENINGSTAR(open_price, high_price, low_price, close_price)
            
            # Motifs de continuation
            patterns["doji"] = talib.CDLDOJI(open_price, high_price, low_price, close_price)
            patterns["three_white_soldiers"] = talib.CDL3WHITESOLDIERS(open_price, high_price, low_price, close_price)
            patterns["three_black_crows"] = talib.CDL3BLACKCROWS(open_price, high_price, low_price, close_price)
        else:
            # Détection simplifiée pour le fallback
            patterns["bullish_candle"] = np.where(close_price > open_price, 100, 0)
            patterns["bearish_candle"] = np.where(close_price < open_price, -100, 0)
            
            # Détection Doji simplifiée
            body_size = abs(close_price - open_price)
            avg_body = np.mean(body_size[-20:])
            patterns["doji"] = np.where(body_size < avg_body * 0.1, 100, 0)
        
        # Détecter les configurations récentes
        recent_patterns = {}
        for name, pattern_array in patterns.items():
            if len(pattern_array) >= 3:
                # Vérifier les 3 dernières bougies
                if np.any(np.abs(pattern_array[-3:]) > 0):
                    recent_patterns[name] = True
                    
                    # Stocker également le dernier indice avec un signal
                    for i in range(1, 4):
                        if np.abs(pattern_array[-i]) > 0:
                            recent_patterns[f"{name}_index"] = len(pattern_array) - i
                            recent_patterns[f"{name}_value"] = pattern_array[-i]
                            break
        
        patterns["recent_patterns"] = recent_patterns
        return patterns
    
    def _generate_signals(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Génère des signaux de trading basés sur les indicateurs calculés"""
        signals = {}
        
        # Signaux basés sur les moyennes mobiles
        if "sma_20" in indicators and "sma_50" in indicators and len(indicators["sma_20"]) > 1 and len(indicators["sma_50"]) > 1:
            # Croisement haussier (Golden Cross)
            if indicators["sma_20"][-2] <= indicators["sma_50"][-2] and indicators["sma_20"][-1] > indicators["sma_50"][-1]:
                signals["golden_cross"] = True
            
            # Croisement baissier (Death Cross)
            if indicators["sma_20"][-2] >= indicators["sma_50"][-2] and indicators["sma_20"][-1] < indicators["sma_50"][-1]:
                signals["death_cross"] = True
        
        # Signaux basés sur le RSI
        if "rsi" in indicators and len(indicators["rsi"]) > 0:
            current_rsi = indicators["rsi"][-1]
            
            if not np.isnan(current_rsi):
                if current_rsi < 30:
                    signals["rsi_oversold"] = True
                elif current_rsi > 70:
                    signals["rsi_overbought"] = True
                
                # Divergence RSI-Prix (simplifiée)
                if len(indicators["rsi"]) > 20 and len(indicators["close"]) > 20:
                    price_uptrend = indicators["close"][-1] > indicators["close"][-5]
                    rsi_downtrend = indicators["rsi"][-1] < indicators["rsi"][-5]
                    
                    price_downtrend = indicators["close"][-1] < indicators["close"][-5]
                    rsi_uptrend = indicators["rsi"][-1] > indicators["rsi"][-5]
                    
                    if price_uptrend and rsi_downtrend:
                        signals["bearish_divergence"] = True
                    elif price_downtrend and rsi_uptrend:
                        signals["bullish_divergence"] = True
        
        # Signaux basés sur le MACD
        if all(k in indicators for k in ["macd", "macd_signal"]) and len(indicators["macd"]) > 1 and len(indicators["macd_signal"]) > 1:
            if indicators["macd"][-2] <= indicators["macd_signal"][-2] and indicators["macd"][-1] > indicators["macd_signal"][-1]:
                signals["macd_bullish_cross"] = True
            elif indicators["macd"][-2] >= indicators["macd_signal"][-2] and indicators["macd"][-1] < indicators["macd_signal"][-1]:
                signals["macd_bearish_cross"] = True
        
        # Signaux basés sur les bandes de Bollinger
        if all(k in indicators for k in ["bb_upper", "bb_middle", "bb_lower"]) and len(indicators["bb_lower"]) > 0:
            current_price = indicators["close"][-1]
            
            if current_price < indicators["bb_lower"][-1]:
                signals["price_below_lower_bb"] = True
            elif current_price > indicators["bb_upper"][-1]:
                signals["price_above_upper_bb"] = True
        
        # Signaux basés sur les motifs de chandeliers
        if "recent_patterns" in indicators:
            for pattern, exists in indicators["recent_patterns"].items():
                if isinstance(exists, bool) and exists:
                    signals[pattern] = True
        
        # Évaluation de la tendance actuelle
        if "trend_direction" in indicators:
            signals["trend"] = indicators["trend_direction"]
        
        return signals
    
    def _calculate_signal_strength(self, signals: Dict[str, Any]) -> Dict[str, float]:
        """Calcule la force du signal global"""
        buy_signals = 0
        sell_signals = 0
        
        # Signaux d'achat
        if signals.get("golden_cross", False): buy_signals += 2
        if signals.get("rsi_oversold", False): buy_signals += 1
        if signals.get("macd_bullish_cross", False): buy_signals += 1
        if signals.get("price_below_lower_bb", False): buy_signals += 1
        if signals.get("bullish_divergence", False): buy_signals += 2
        if signals.get("hammer", False): buy_signals += 1
        if signals.get("inverted_hammer", False): buy_signals += 1
        if signals.get("bullish_engulfing", False): buy_signals += 1
        if signals.get("morning_star", False): buy_signals += 2
        if signals.get("three_white_soldiers", False): buy_signals += 2
        
        # Signaux de vente
        if signals.get("death_cross", False): sell_signals += 2
        if signals.get("rsi_overbought", False): sell_signals += 1
        if signals.get("macd_bearish_cross", False): sell_signals += 1
        if signals.get("price_above_upper_bb", False): sell_signals += 1
        if signals.get("bearish_divergence", False): sell_signals += 2
        if signals.get("hanging_man", False): sell_signals += 1
        if signals.get("shooting_star", False): sell_signals += 1
        if signals.get("bearish_engulfing", False): sell_signals += 1
        if signals.get("evening_star", False): sell_signals += 2
        if signals.get("three_black_crows", False): sell_signals += 2
        
        # Ajuster en fonction de la tendance
        trend = signals.get("trend", "neutral")
        if trend == "strong_bullish":
            buy_signals += 2
            sell_signals -= 1
        elif trend == "bullish":
            buy_signals += 1
        elif trend == "strong_bearish":
            sell_signals += 2
            buy_signals -= 1
        elif trend == "bearish":
            sell_signals += 1
        
        # Normaliser les scores entre 0 et 1
        max_signals = 15  # Maximum théorique de signaux
        buy_strength = min(1.0, buy_signals / max_signals)
        sell_strength = min(1.0, sell_signals / max_signals)
        
        # Déterminer la direction globale
        signal_diff = buy_signals - sell_signals
        if signal_diff > 0:
            direction = "buy"
            strength = buy_strength
        elif signal_diff < 0:
            direction = "sell"
            strength = sell_strength
        else:
            direction = "neutral"
            strength = 0.0
        
        return {
            "direction": direction,
            "strength": strength,
            "buy_strength": buy_strength,
            "sell_strength": sell_strength,
            "buy_signals": buy_signals,
            "sell_signals": sell_signals
        }

# Interface simplifiée pour l'intégration
def analyze_symbol(data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
    """
    Analyse technique complète d'un symbole
    
    Args:
        data: DataFrame avec données OHLCV
        symbol: Symbole à analyser
        
    Returns:
        Résultats d'analyse avec indicateurs et signaux
    """
    analyzer = TechnicalAnalyzer()
    return analyzer.analyze(data, symbol)
