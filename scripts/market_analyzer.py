#!/usr/bin/env python
"""
MercurioAI Market Analyzer - Analyseur de marché professionnel

Ce script analyse en continu les conditions du marché pour:
1. Détecter les régimes de marché (haussier, baissier, latéral, volatil)
2. Identifier les anomalies et manipulations potentielles
3. Optimiser la sélection de stratégies en fonction des conditions
4. Générer des alertes en temps réel
"""

import os
import sys
import json
import logging
import asyncio
import argparse
import datetime
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import numpy as np
from pathlib import Path

from app.services.market_data import MarketDataService
from app.strategies.msi.sentiment_analysis import SentimentAnalysisEngine

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/market_analyzer.log")
    ]
)
logger = logging.getLogger(__name__)

class MarketRegime:
    """Enum pour les différents régimes de marché"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    TRENDING = "trending"
    REVERSAL = "reversal"
    UNKNOWN = "unknown"

class MarketAnalyzer:
    """
    Analyseur de marché professionnel qui détecte les conditions et régimes
    pour optimiser la sélection de stratégies de trading.
    """
    
    def __init__(self, config_path: str):
        """
        Initialise l'analyseur de marché avec la configuration spécifiée.
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        self.load_config(config_path)
        self.market_data_service = MarketDataService()
        self.sentiment_engine = SentimentAnalysisEngine()
        
        # État des analyses
        self.market_regimes = {}
        self.anomalies = {}
        self.correlations = {}
        self.volatility_levels = {}
        self.sentiment_scores = {}
        self.last_analysis_time = None
        
        # Stockage d'historique pour l'analyse
        self.regime_history = {}
        self.volatility_history = {}
        
        logger.info(f"Analyseur de marché initialisé avec configuration: {config_path}")
    
    def load_config(self, config_path: str) -> None:
        """
        Charge la configuration depuis un fichier JSON.
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
                
            logger.info(f"Configuration chargée avec succès: {len(self.config.get('symbols', []))} symboles")
                       
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration: {e}")
            raise
    
    async def initialize(self) -> None:
        """
        Initialise l'analyseur et charge les données initiales.
        """
        try:
            # Initialiser le moteur de sentiment
            asyncio.create_task(self.sentiment_engine.initialize())
            
            # Charger les données historiques pour l'initialisation
            for symbol in self.config.get("symbols", []):
                end_date = datetime.datetime.now()
                start_date = end_date - datetime.timedelta(days=30)
                
                data = await self.market_data_service.get_historical_data(
                    symbol, start_date, end_date, timeframe="1h"
                )
                
                if data is not None and not data.empty:
                    # Initialiser les régimes de marché
                    initial_regime = self._detect_market_regime(data)
                    self.market_regimes[symbol] = initial_regime
                    
                    # Initialiser les niveaux de volatilité
                    volatility = self._calculate_volatility(data)
                    self.volatility_levels[symbol] = volatility
                    
                    logger.info(f"Régime initial pour {symbol}: {initial_regime}, "
                               f"volatilité: {volatility:.4f}")
                else:
                    logger.warning(f"Données insuffisantes pour {symbol}, "
                                  f"initialisation reportée")
            
            logger.info("Initialisation de l'analyseur de marché terminée")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de l'analyseur: {e}")
            raise
    
    async def start(self) -> None:
        """
        Démarre l'analyseur de marché en mode continu.
        """
        try:
            await self.initialize()
            
            # Démarrer les tâches d'analyse
            analysis_task = asyncio.create_task(self._continuous_market_analysis())
            correlation_task = asyncio.create_task(self._periodic_correlation_analysis())
            anomaly_task = asyncio.create_task(self._continuous_anomaly_detection())
            reporting_task = asyncio.create_task(self._periodic_market_reporting())
            
            logger.info("Analyseur de marché démarré en mode continu")
            
            # Attendre que toutes les tâches soient terminées
            await asyncio.gather(
                analysis_task, correlation_task, anomaly_task, reporting_task
            )
            
        except KeyboardInterrupt:
            logger.info("Arrêt de l'analyseur demandé par l'utilisateur")
        except Exception as e:
            logger.error(f"Erreur critique pendant l'exécution de l'analyseur: {e}")
    
    async def _continuous_market_analysis(self) -> None:
        """
        Analyse en continu les régimes de marché et conditions.
        """
        analysis_interval = self.config.get("analysis_interval_seconds", 300)
        
        while True:
            try:
                self.last_analysis_time = datetime.datetime.now()
                
                for symbol in self.config.get("symbols", []):
                    # Récupérer les données récentes
                    end_date = datetime.datetime.now()
                    
                    # Données horaires pour l'analyse de régime
                    start_date_h = end_date - datetime.timedelta(days=7)
                    hourly_data = await self.market_data_service.get_historical_data(
                        symbol, start_date_h, end_date, timeframe="1h"
                    )
                    
                    # Données minute pour l'analyse de volatilité
                    start_date_m = end_date - datetime.timedelta(hours=24)
                    minute_data = await self.market_data_service.get_historical_data(
                        symbol, start_date_m, end_date, timeframe="1m"
                    )
                    
                    if hourly_data is not None and not hourly_data.empty:
                        # Détecter le régime de marché
                        regime = self._detect_market_regime(hourly_data)
                        previous_regime = self.market_regimes.get(symbol, MarketRegime.UNKNOWN)
                        self.market_regimes[symbol] = regime
                        
                        # Maintenir l'historique des régimes
                        if symbol not in self.regime_history:
                            self.regime_history[symbol] = []
                        self.regime_history[symbol].append((end_date, regime))
                        
                        # Limiter la taille de l'historique
                        max_history = 100
                        if len(self.regime_history[symbol]) > max_history:
                            self.regime_history[symbol] = self.regime_history[symbol][-max_history:]
                        
                        # Notifier les changements de régime
                        if regime != previous_regime:
                            logger.info(f"Changement de régime pour {symbol}: "
                                       f"{previous_regime} -> {regime}")
                    
                    if minute_data is not None and not minute_data.empty:
                        # Calculer la volatilité récente
                        volatility = self._calculate_volatility(minute_data)
                        self.volatility_levels[symbol] = volatility
                        
                        # Maintenir l'historique de volatilité
                        if symbol not in self.volatility_history:
                            self.volatility_history[symbol] = []
                        self.volatility_history[symbol].append((end_date, volatility))
                        
                        # Limiter la taille de l'historique
                        max_history = 100
                        if len(self.volatility_history[symbol]) > max_history:
                            self.volatility_history[symbol] = self.volatility_history[symbol][-max_history:]
                    
                    # Analyser le sentiment
                    sentiment_data = await self.sentiment_engine.fetch_sentiment_data(symbol)
                    combined_score = 0.0
                    source_count = 0
                    
                    for source in ["twitter", "reddit", "news"]:
                        if source in sentiment_data and sentiment_data[source]:
                            source_data = sentiment_data[source]
                            if "score" in source_data:
                                combined_score += source_data["score"]
                                source_count += 1
                    
                    if source_count > 0:
                        avg_sentiment = combined_score / source_count
                        self.sentiment_scores[symbol] = avg_sentiment
                        logger.debug(f"Sentiment pour {symbol}: {avg_sentiment:.2f}")
                
                logger.info(f"Analyse des régimes de marché terminée pour {len(self.config.get('symbols', []))} symboles")
                
            except Exception as e:
                logger.error(f"Erreur pendant l'analyse de marché: {e}")
            
            # Attendre avant la prochaine analyse
            await asyncio.sleep(analysis_interval)
    
    async def _periodic_correlation_analysis(self) -> None:
        """
        Analyse périodiquement les corrélations entre différents actifs.
        """
        correlation_interval = self.config.get("correlation_interval_minutes", 60) * 60
        
        while True:
            try:
                # Attendre avant la première analyse
                await asyncio.sleep(correlation_interval)
                
                symbols = self.config.get("symbols", [])
                if len(symbols) < 2:
                    continue  # Besoin d'au moins deux symboles pour la corrélation
                
                # Récupérer les données
                end_date = datetime.datetime.now()
                start_date = end_date - datetime.timedelta(days=7)
                
                # Collecter les données de tous les symboles
                price_data = {}
                for symbol in symbols:
                    data = await self.market_data_service.get_historical_data(
                        symbol, start_date, end_date, timeframe="1h"
                    )
                    if data is not None and not data.empty:
                        price_data[symbol] = data["close"]
                
                if len(price_data) < 2:
                    logger.warning("Données insuffisantes pour l'analyse de corrélation")
                    continue
                
                # Créer un DataFrame avec toutes les séries de prix
                df = pd.DataFrame(price_data)
                
                # Calculer la matrice de corrélation
                correlation_matrix = df.pct_change().corr()
                
                # Enregistrer les résultats
                self.correlations = correlation_matrix.to_dict()
                
                # Identifier les paires fortement corrélées
                for i, symbol1 in enumerate(symbols):
                    for j, symbol2 in enumerate(symbols):
                        if i < j and symbol1 in correlation_matrix.index and symbol2 in correlation_matrix.columns:
                            corr = correlation_matrix.loc[symbol1, symbol2]
                            if abs(corr) > 0.8:
                                logger.info(f"Forte corrélation entre {symbol1} et {symbol2}: {corr:.2f}")
                
                logger.info("Analyse de corrélation terminée")
                
            except Exception as e:
                logger.error(f"Erreur pendant l'analyse de corrélation: {e}")
    
    async def _continuous_anomaly_detection(self) -> None:
        """
        Détecte en continu les anomalies de marché et les manipulations potentielles.
        """
        anomaly_interval = self.config.get("anomaly_detection_interval_minutes", 15) * 60
        
        while True:
            try:
                # Attendre avant la première analyse
                await asyncio.sleep(anomaly_interval)
                
                for symbol in self.config.get("symbols", []):
                    # Récupérer les données récentes
                    end_date = datetime.datetime.now()
                    start_date = end_date - datetime.timedelta(hours=24)
                    
                    data = await self.market_data_service.get_historical_data(
                        symbol, start_date, end_date, timeframe="1m"
                    )
                    
                    if data is None or data.empty:
                        continue
                    
                    # Détecter les anomalies de volume
                    volume_anomalies = self._detect_volume_anomalies(data)
                    
                    # Détecter les mouvements de prix suspects
                    price_anomalies = self._detect_price_anomalies(data)
                    
                    # Détecter les divergences sentiment-prix
                    sentiment_anomalies = self._detect_sentiment_anomalies(symbol, data)
                    
                    # Combinaison des anomalies
                    combined_anomalies = {
                        "volume": volume_anomalies,
                        "price": price_anomalies,
                        "sentiment": sentiment_anomalies,
                        "timestamp": datetime.datetime.now()
                    }
                    
                    # Estimer la probabilité de manipulation
                    manipulation_probability = 0.0
                    anomaly_count = 0
                    
                    if volume_anomalies.get("detected", False):
                        manipulation_probability += 0.3
                        anomaly_count += 1
                    
                    if price_anomalies.get("detected", False):
                        manipulation_probability += 0.3
                        anomaly_count += 1
                    
                    if sentiment_anomalies.get("detected", False):
                        manipulation_probability += 0.4
                        anomaly_count += 1
                    
                    combined_anomalies["manipulation_probability"] = (
                        manipulation_probability if anomaly_count > 0 else 0.0
                    )
                    
                    # Enregistrer les résultats
                    self.anomalies[symbol] = combined_anomalies
                    
                    # Alerter sur les probabilités élevées
                    if manipulation_probability > 0.5:
                        logger.warning(f"ALERTE: Manipulation potentielle sur {symbol} "
                                      f"(probabilité: {manipulation_probability:.2f})")
                        logger.warning(f"Détails des anomalies pour {symbol}: {combined_anomalies}")
                
                logger.info("Analyse des anomalies terminée")
                
            except Exception as e:
                logger.error(f"Erreur pendant la détection d'anomalies: {e}")
    
    async def _periodic_market_reporting(self) -> None:
        """
        Génère des rapports périodiques sur l'état du marché.
        """
        report_interval = self.config.get("market_report_interval_hours", 1) * 3600
        
        while True:
            try:
                # Attendre avant le premier rapport
                await asyncio.sleep(report_interval)
                
                report = await self.generate_market_report()
                
                # Sauvegarder le rapport
                report_dir = Path("reports/market")
                report_dir.mkdir(parents=True, exist_ok=True)
                
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                report_path = report_dir / f"market_report_{timestamp}.json"
                
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=2)
                
                logger.info(f"Rapport de marché généré: {report_path}")
                
            except Exception as e:
                logger.error(f"Erreur pendant la génération de rapport: {e}")
    
    async def generate_market_report(self) -> Dict[str, Any]:
        """
        Génère un rapport complet sur l'état actuel du marché.
        
        Returns:
            Dictionnaire contenant les informations de marché
        """
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "market_regimes": self.market_regimes,
            "volatility_levels": self.volatility_levels,
            "sentiment_scores": self.sentiment_scores,
            "correlations": self.correlations,
            "anomalies": {k: v for k, v in self.anomalies.items() if v.get("manipulation_probability", 0) > 0.3},
            "recommended_strategies": {}
        }
        
        # Recommandations de stratégies basées sur les conditions de marché
        for symbol, regime in self.market_regimes.items():
            volatility = self.volatility_levels.get(symbol, 0.0)
            sentiment = self.sentiment_scores.get(symbol, 0.0)
            
            recommended = []
            
            if regime == MarketRegime.BULLISH:
                if volatility < 0.02:
                    recommended.append("MovingAverageStrategy")
                else:
                    recommended.append("MultiSourceIntelligenceStrategy")
                    if sentiment > 0.3:
                        recommended.append("LLMStrategy")
            
            elif regime == MarketRegime.BEARISH:
                recommended.append("MultiSourceIntelligenceStrategy")
                if volatility > 0.03:
                    recommended.append("TransformerStrategy")
            
            elif regime == MarketRegime.VOLATILE:
                recommended.append("TransformerStrategy")
                recommended.append("MultiSourceIntelligenceStrategy")
            
            elif regime == MarketRegime.SIDEWAYS:
                recommended.append("LSTMPredictorStrategy")
                if volatility < 0.015:
                    recommended.append("MovingAverageStrategy")
            
            # Cas de manipulation potentielle
            if symbol in self.anomalies and self.anomalies[symbol].get("manipulation_probability", 0) > 0.4:
                # Recommandation spéciale: MSI est plus résistant aux manipulations
                recommended = ["MultiSourceIntelligenceStrategy"]
            
            report["recommended_strategies"][symbol] = recommended
        
        return report
    
    def _detect_market_regime(self, data: pd.DataFrame) -> str:
        """
        Détecte le régime de marché actuel basé sur les données historiques.
        
        Args:
            data: DataFrame contenant les données OHLCV
            
        Returns:
            Chaîne indiquant le régime de marché
        """
        if data is None or len(data) < 20:
            return MarketRegime.UNKNOWN
        
        # Calculer les métriques pour la détection
        returns = data['close'].pct_change().dropna()
        
        # Tendance sur les 14 dernières périodes
        recent_trend = (data['close'].iloc[-1] / data['close'].iloc[-15] - 1)
        
        # Volatilité
        volatility = returns.std()
        
        # ADX pour la force de tendance
        high = data['high']
        low = data['low']
        close = data['close']
        
        # Calculer +DI et -DI
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = abs(minus_dm)
        
        # Prendre le maximum entre +DM et -DM, et le troisième cas
        temp_dm = pd.DataFrame({'+DM': plus_dm, '-DM': minus_dm})
        temp_dm['DM'] = temp_dm.apply(lambda row: row['+DM'] if row['+DM'] > row['-DM'] else row['-DM'], axis=1)
        
        # Calculer True Range
        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift(1)))
        tr3 = pd.DataFrame(abs(low - close.shift(1)))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculer ADX (Average Directional Index)
        smoothing_period = 14
        atr = true_range.rolling(window=smoothing_period).mean()
        if len(atr) >= smoothing_period:
            adx_value = atr.iloc[-1]
        else:
            adx_value = 0
        
        # Déterminer le régime
        if abs(recent_trend) > 0.1:  # Tendance forte
            if recent_trend > 0:
                return MarketRegime.BULLISH
            else:
                return MarketRegime.BEARISH
        elif volatility > 0.03:  # Volatilité élevée
            return MarketRegime.VOLATILE
        elif adx_value < 20:  # Faible force de tendance
            return MarketRegime.SIDEWAYS
        elif adx_value > 25:  # Force de tendance modérée
            if recent_trend > 0:
                return MarketRegime.TRENDING
            else:
                return MarketRegime.BEARISH
        else:
            # Vérifier s'il y a un renversement récent
            if len(returns) >= 3 and (
                (returns.iloc[-3:].mean() * returns.iloc[-10:-3].mean()) < 0
            ):
                return MarketRegime.REVERSAL
            else:
                return MarketRegime.SIDEWAYS
    
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """
        Calcule le niveau de volatilité actuel du marché.
        
        Args:
            data: DataFrame contenant les données OHLCV
            
        Returns:
            Valeur de volatilité (écart type des rendements)
        """
        if data is None or len(data) < 5:
            return 0.0
        
        # Calculer les rendements
        returns = data['close'].pct_change().dropna()
        
        # Écart type des rendements (volatilité)
        volatility = returns.std()
        
        return volatility
    
    def _detect_volume_anomalies(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Détecte les anomalies de volume qui pourraient indiquer une manipulation.
        
        Args:
            data: DataFrame contenant les données OHLCV
            
        Returns:
            Dictionnaire avec les résultats de détection
        """
        if data is None or len(data) < 30:
            return {"detected": False}
        
        # Extraire les volumes
        volumes = data['volume']
        
        # Calculer les statistiques de volume
        mean_volume = volumes.mean()
        std_volume = volumes.std()
        recent_volume = volumes.iloc[-1]
        
        # Vérifier les pics soudains de volume
        volume_surge = recent_volume > (mean_volume + 3 * std_volume)
        
        # Vérifier les motifs de distribution en V (pump and dump)
        volume_pattern = False
        if len(volumes) >= 10:
            recent_volumes = volumes.iloc[-10:]
            volume_slope_start = recent_volumes.iloc[0:5].mean()
            volume_slope_end = recent_volumes.iloc[5:10].mean()
            
            if (volume_slope_start < volume_slope_end * 0.5) or (volume_slope_start > volume_slope_end * 2):
                volume_pattern = True
        
        # Résultats
        result = {
            "detected": volume_surge or volume_pattern,
            "volume_surge": volume_surge,
            "volume_pattern": volume_pattern,
            "recent_volume": recent_volume,
            "average_volume": mean_volume,
            "z_score": (recent_volume - mean_volume) / std_volume if std_volume > 0 else 0
        }
        
        return result
    
    def _detect_price_anomalies(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Détecte les anomalies de prix qui pourraient indiquer une manipulation.
        
        Args:
            data: DataFrame contenant les données OHLCV
            
        Returns:
            Dictionnaire avec les résultats de détection
        """
        if data is None or len(data) < 30:
            return {"detected": False}
        
        # Extraire les prix
        closes = data['close']
        highs = data['high']
        lows = data['low']
        
        # Calculer les rendements
        returns = closes.pct_change().dropna()
        
        # Vérifier les mouvements de prix anormaux
        mean_return = returns.mean()
        std_return = returns.std()
        recent_return = returns.iloc[-1] if len(returns) > 0 else 0
        
        price_shock = abs(recent_return) > (3 * std_return)
        
        # Vérifier les bougies anormales (longues mèches)
        recent_candle_size = highs.iloc[-1] - lows.iloc[-1]
        avg_candle_size = (highs - lows).mean()
        
        abnormal_candle = recent_candle_size > (3 * avg_candle_size)
        
        # Vérifier les retournements soudains
        price_reversal = False
        if len(returns) >= 3:
            if (returns.iloc[-1] * returns.iloc[-2]) < 0 and abs(returns.iloc[-1]) > (2 * std_return):
                price_reversal = True
        
        # Résultats
        result = {
            "detected": price_shock or abnormal_candle or price_reversal,
            "price_shock": price_shock,
            "abnormal_candle": abnormal_candle,
            "price_reversal": price_reversal,
            "recent_return": recent_return,
            "z_score": (recent_return - mean_return) / std_return if std_return > 0 else 0
        }
        
        return result
    
    def _detect_sentiment_anomalies(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Détecte les anomalies dans les données de sentiment par rapport aux prix.
        
        Args:
            symbol: Symbole à analyser
            data: DataFrame contenant les données OHLCV
            
        Returns:
            Dictionnaire avec les résultats de détection
        """
        if data is None or len(data) < 10:
            return {"detected": False}
        
        # Obtenir le sentiment actuel
        current_sentiment = self.sentiment_scores.get(symbol, 0)
        
        # Obtenir le mouvement de prix récent
        price_change = data['close'].pct_change(5).iloc[-1] if len(data) >= 5 else 0
        
        # Vérifier la divergence sentiment-prix
        sentiment_price_divergent = (current_sentiment > 0.5 and price_change < -0.05) or \
                                   (current_sentiment < -0.5 and price_change > 0.05)
        
        # Vérifier les changements soudains de sentiment
        sentiment_history = []
        for source in ["twitter", "reddit", "news"]:
            if symbol in self.sentiment_scores:
                sentiment_history.append(self.sentiment_scores[symbol])
        
        sentiment_shock = False
        if len(sentiment_history) >= 2:
            sentiment_change = abs(sentiment_history[-1] - sentiment_history[0])
            if sentiment_change > 0.5:  # Changement important
                sentiment_shock = True
        
        # Résultats
        result = {
            "detected": sentiment_price_divergent or sentiment_shock,
            "sentiment_price_divergent": sentiment_price_divergent,
            "sentiment_shock": sentiment_shock,
            "current_sentiment": current_sentiment,
            "price_change": price_change
        }
        
        return result

async def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="MercurioAI Market Analyzer")
    parser.add_argument("--config", type=str, default="config/agent_config.json", help="Chemin vers le fichier de configuration")
    args = parser.parse_args()
    
    # Créer le répertoire de logs s'il n'existe pas
    os.makedirs("logs", exist_ok=True)
    
    # Créer l'analyseur de marché
    analyzer = MarketAnalyzer(args.config)
    
    try:
        await analyzer.start()
    except KeyboardInterrupt:
        logger.info("Arrêt demandé par l'utilisateur")
    except Exception as e:
        logger.error(f"Erreur critique: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        logger.info("Programme interrompu par l'utilisateur")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Erreur critique: {e}")
        sys.exit(1)
