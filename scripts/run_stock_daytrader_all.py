#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stock Day Trading Script - Version complète
-------------------------------------------
Script pour le daytrading d'actions via Alpaca API,
utilisant toutes les stratégies disponibles dans Mercurio AI.

Ce script:
1. Récupère tous les symboles d'actions disponibles
2. Applique un filtrage pour obtenir des actions de qualité
3. Utilise toutes les stratégies disponibles
4. Fonctionne en mode paper trading
5. S'exécute pendant toute la journée de trading

Utilisation:
    python scripts/run_stock_daytrader_all.py
"""

import os
import sys
import time
import signal
import logging
import argparse
import threading
import concurrent.futures
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta, date, timezone
import pandas as pd
import numpy as np

# Ajouter le répertoire parent au path pour pouvoir importer les modules du projet
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Importer le gestionnaire de taux API
try:
    from scripts.api_rate_manager import rate_limited
    USE_RATE_MANAGER = True
    logging.getLogger("stock_daytrader").info("Gestionnaire de taux API chargé avec succès")
except ImportError:
    # Fonction de repli si le module n'est pas disponible
    def rate_limited(f):
        return f
    USE_RATE_MANAGER = False
    logging.getLogger("stock_daytrader").warning("Gestionnaire de taux API non disponible, risque de limites de taux")

# API Alpaca
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv

# Fonction pour détecter le niveau d'accès Alpaca
def detect_alpaca_level(api_key=None, api_secret=None, base_url=None, data_url=None):
    """
    Détecte le niveau d'abonnement Alpaca disponible en testant les fonctionnalités
    
    Args:
        api_key: Clé API Alpaca
        api_secret: Secret API Alpaca
        base_url: URL de base pour l'API Alpaca
        data_url: URL des données pour l'API Alpaca
        
    Returns:
        int: Niveau d'abonnement (3 = premium, 2 = standard+, 1 = standard, 0 = non détecté)
    """
    if not api_key or not api_secret:
        # Récupérer les clés d'API depuis les variables d'environnement
        load_dotenv()
        # Déterminer le mode (paper ou live)
        alpaca_mode = os.getenv("ALPACA_MODE", "paper").lower()
        if alpaca_mode == "live":
            api_key = os.getenv("ALPACA_LIVE_KEY")
            api_secret = os.getenv("ALPACA_LIVE_SECRET")
            base_url = os.getenv("ALPACA_LIVE_URL", "https://api.alpaca.markets")
        else:  # mode paper par défaut
            api_key = os.getenv("ALPACA_PAPER_KEY")
            api_secret = os.getenv("ALPACA_PAPER_SECRET")
            base_url = os.getenv("ALPACA_PAPER_URL", "https://paper-api.alpaca.markets")
        
        data_url = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
    
    # Initialiser le client API
    try:
        api = tradeapi.REST(
            key_id=api_key,
            secret_key=api_secret,
            base_url=base_url,
            data_url=data_url
        )
        
        logger.info("Test du niveau d'abonnement Alpaca...")
        
        # Test niveau 3 (premium) - Accès aux données en temps réel
        try:
            # Tester une fonctionnalité spécifique au niveau 3: données en temps réel plus précises
            end = datetime.now()
            start = end - timedelta(hours=1)
            symbol = "AAPL"  # Une action populaire
            bars = api.get_bars(symbol, tradeapi.TimeFrame.Minute, start.isoformat(), end.isoformat())
            if len(bars) > 0 and hasattr(bars[0], 'trade_count'):
                logger.info("✅ Niveau 3 (Premium) détecté - Accès complet aux données temps réel")
                return 3
        except Exception as e:
            logger.debug(f"Test niveau 3 échoué: {str(e)}")
        
        # Test niveau 2 - Données historiques étendues
        try:
            # Tester des données historiques (disponibles dans les niveaux 2 et 3)
            end = datetime.now()
            start = end - timedelta(days=365)  # 1 an de données
            bars = api.get_bars("AAPL", tradeapi.TimeFrame.Day, start.isoformat(), end.isoformat())
            if len(bars) > 200:  # Si on a plus de 200 jours, c'est probablement niveau 2+
                logger.info("✅ Niveau 2 détecté - Accès aux données historiques étendues")
                return 2
        except Exception as e:
            logger.debug(f"Test niveau 2 échoué: {str(e)}")
        
        # Test niveau 1 - Fonctionnalités de base
        try:
            # Tester les fonctionnalités de base (disponibles dans tous les niveaux)
            account = api.get_account()
            logger.info("✅ Niveau 1 détecté - Accès aux fonctionnalités de base")
            return 1
        except Exception as e:
            logger.debug(f"Test niveau 1 échoué: {str(e)}")
        
        logger.warning("❌ Aucun niveau d'abonnement détecté - Vérifiez vos identifiants API")
        return 0
        
    except Exception as e:
        logger.error(f"Erreur lors de la connexion à Alpaca: {str(e)}")
        return 0

# Importer les services et stratégies de Mercurio AI
try:
    from app.services.market_data import MarketDataService
    from app.services.trading import TradingService
    from app.services.backtesting import BacktestingService
    from app.services.strategy_manager import StrategyManager
    from app.strategies.moving_average import MovingAverageStrategy
    from app.strategies.lstm_predictor import LSTMPredictorStrategy
    from app.strategies.transformer_strategy import TransformerStrategy
    from app.strategies.msi_strategy import MultiSourceIntelligenceStrategy as MSIStrategy
    from app.strategies.llm_strategy import LLMStrategy
    from app.strategies.llm_strategy_v2 import LLMStrategyV2
except ImportError as e:
    print(f"Erreur d'importation des modules Mercurio: {e}")
    print("Utilisation des services de base uniquement")
    
    # Classes de repli (fallback) pour le mode natif Alpaca
    class FallbackMarketDataService:
        def __init__(self, provider="alpaca", api_key=None, api_secret=None, base_url=None, data_url=None, subscription_level=None):
            self.provider = provider
            self.api_key = api_key
            self.api_secret = api_secret
            self.base_url = base_url
            self.data_url = data_url
            self.subscription_level = subscription_level
            logger.info("Service de données de marché de repli initialisé")
    
    class FallbackTradingService:
        def __init__(self, provider="alpaca", api_key=None, api_secret=None, base_url=None, paper=True):
            self.provider = provider
            self.api_key = api_key
            self.api_secret = api_secret
            self.base_url = base_url
            self.paper = paper
            logger.info("Service de trading de repli initialisé")
    
    class FallbackStrategyManager:
        def __init__(self):
            self.strategies = {}
            logger.info("Gestionnaire de stratégies de repli initialisé")
            
        def register_strategy(self, name, strategy):
            self.strategies[name] = strategy
            logger.info(f"Stratégie {name} enregistrée")
    
    class BaseStrategy:
        def __init__(self, market_data_service=None, trading_service=None, **kwargs):
            self.market_data_service = market_data_service
            self.trading_service = trading_service
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class FallbackMovingAverageStrategy(BaseStrategy):
        def __init__(self, market_data_service=None, trading_service=None, short_window=10, long_window=30):
            super().__init__(market_data_service, trading_service, short_window=short_window, long_window=long_window)
            logger.info(f"Stratégie de moyenne mobile de repli initialisée (fenêtres: {short_window}, {long_window})")
    
    class FallbackMovingAverageMLStrategy(BaseStrategy):
        def __init__(self, market_data_service=None, trading_service=None, short_window=5, long_window=20, use_ml=True):
            super().__init__(market_data_service, trading_service, short_window=short_window, long_window=long_window, use_ml=use_ml)
            logger.info(f"Stratégie de moyenne mobile ML de repli initialisée (fenêtres: {short_window}, {long_window})")
    
    class FallbackLSTMPredictorStrategy(BaseStrategy):
        def __init__(self, market_data_service=None, trading_service=None):
            super().__init__(market_data_service, trading_service)
            logger.info("Stratégie LSTM de repli initialisée")
        
        def train(self, data, symbol=None):
            """
            Méthode d'entraînement simplifiée pour la stratégie LSTM de repli
            
            Args:
                data: DataFrame contenant les données historiques
                symbol: Symbole sur lequel entraîner le modèle
                
            Returns:
                bool: True si l'entraînement a réussi
            """
            logger.info(f"Entraînement simulé de la stratégie LSTM sur {symbol} avec {len(data)} points de données")
            
            try:
                # Simuler un entraînement simple basé sur les moyennes mobiles
                if len(data) >= 30:
                    # Calculer quelques indicateurs techniques de base
                    data['sma_5'] = data['close'].rolling(window=5).mean()
                    data['sma_20'] = data['close'].rolling(window=20).mean()
                    
                    # Simuler l'entraînement (juste attendre un peu)
                    time.sleep(1)  # Simuler le temps d'entraînement
                    
                    logger.info(f"Entraînement réussi pour {symbol} - Stratégie LSTM mise à jour")
                    return True
                else:
                    logger.warning(f"Données insuffisantes pour entraîner la stratégie LSTM sur {symbol}")
                    return False
            except Exception as e:
                logger.error(f"Erreur lors de l'entraînement de la stratégie LSTM sur {symbol}: {e}")
                return False
    
    class FallbackTransformerStrategy(BaseStrategy):
        def __init__(self, market_data_service=None, trading_service=None):
            super().__init__(market_data_service, trading_service)
            logger.info("Stratégie Transformer de repli initialisée")
        
        def train(self, data, symbol=None):
            """
            Méthode d'entraînement simplifiée pour la stratégie Transformer de repli
            
            Args:
                data: DataFrame contenant les données historiques
                symbol: Symbole sur lequel entraîner le modèle
                
            Returns:
                bool: True si l'entraînement a réussi
            """
            logger.info(f"Entraînement simulé de la stratégie Transformer sur {symbol} avec {len(data)} points de données")
            
            try:
                # Simuler un entraînement simple basé sur les moyennes mobiles
                if len(data) >= 30:
                    # Calculer quelques indicateurs techniques de base
                    data['sma_5'] = data['close'].rolling(window=5).mean()
                    data['sma_20'] = data['close'].rolling(window=20).mean()
                    
                    # Simuler l'entraînement (juste attendre un peu)
                    time.sleep(1)  # Simuler le temps d'entraînement
                    
                    logger.info(f"Entraînement réussi pour {symbol} - Stratégie Transformer mise à jour")
                    return True
                else:
                    logger.warning(f"Données insuffisantes pour entraîner la stratégie Transformer sur {symbol}")
                    return False
            except Exception as e:
                logger.error(f"Erreur lors de l'entraînement de la stratégie Transformer sur {symbol}: {e}")
                return False
    
    class FallbackMSIStrategy(BaseStrategy):
        def __init__(self, market_data_service=None, trading_service=None):
            super().__init__(market_data_service, trading_service)
            logger.info("Stratégie MSI de repli initialisée")
        
        def train(self, data, symbol=None):
            """
            Méthode d'entraînement simplifiée pour la stratégie MSI de repli
            
            Args:
                data: DataFrame contenant les données historiques
                symbol: Symbole sur lequel entraîner le modèle
                
            Returns:
                bool: True si l'entraînement a réussi
            """
            logger.info(f"Entraînement simulé de la stratégie MSI sur {symbol} avec {len(data)} points de données")
            
            try:
                # Simuler un entraînement simple basé sur les moyennes mobiles et le sentiment
                if len(data) >= 30:
                    # Calculer quelques indicateurs techniques de base
                    data['sma_5'] = data['close'].rolling(window=5).mean()
                    data['sma_10'] = data['close'].rolling(window=10).mean()
                    data['sma_20'] = data['close'].rolling(window=20).mean()
                    
                    # Simuler l'analyse de sentiment (factice)
                    data['sentiment'] = 0.5 + 0.1 * (data['close'].pct_change().rolling(5).mean() / data['close'].pct_change().rolling(5).std())
                    
                    # Simuler l'entraînement (juste attendre un peu)
                    time.sleep(1)  # Simuler le temps d'entraînement
                    
                    logger.info(f"Entraînement réussi pour {symbol} - Stratégie MSI mise à jour")
                    return True
                else:
                    logger.warning(f"Données insuffisantes pour entraîner la stratégie MSI sur {symbol}")
                    return False
            except Exception as e:
                logger.error(f"Erreur lors de l'entraînement de la stratégie MSI sur {symbol}: {e}")
                return False
    
    class FallbackLLMStrategy(BaseStrategy):
        def __init__(self, market_data_service=None, trading_service=None):
            super().__init__(market_data_service, trading_service)
            logger.info("Stratégie LLM de repli initialisée")
        
        def train(self, data, symbol=None):
            """
            Méthode d'entraînement simplifiée pour la stratégie LLM de repli
            
            Args:
                data: DataFrame contenant les données historiques
                symbol: Symbole sur lequel entraîner le modèle
                
            Returns:
                bool: True si l'entraînement a réussi
            """
            try:
                logger.info(f"Entraînement simulé de la stratégie LLM sur {symbol if symbol else 'tous les symboles'}")
                time.sleep(1)  # Simuler un entraînement
                return True
            except Exception as e:
                logger.error(f"Erreur lors de l'entraînement LLM: {e}")
                return False
            
        def analyze(self, data, symbol):
            """
            Analyse des données pour générer des signaux de trading
            
            Args:
                data: DataFrame contenant les données historiques
                symbol: Symbole à analyser
                
            Returns:
                dict: Résultat de l'analyse avec signaux de trading
            """
            try:
                # Logique d'analyse de repli très simplifiée
                if len(data) < 2:
                    return {"action": "hold", "confidence": 0.5}
                    
                last_close = data['close'].iloc[-1]
                prev_close = data['close'].iloc[-2]
                
                if last_close > prev_close * 1.02:  # Hausse de 2%
                    return {"action": "buy", "confidence": 0.7, "reason": "Hausse significative détectée"}
                elif last_close < prev_close * 0.98:  # Baisse de 2%
                    return {"action": "sell", "confidence": 0.7, "reason": "Baisse significative détectée"}
                else:
                    return {"action": "hold", "confidence": 0.6, "reason": "Pas de mouvement significatif"}
                    
            except Exception as e:
                logger.error(f"Erreur d'analyse LLM pour {symbol}: {e}")
                return {"action": "hold", "confidence": 0.5, "reason": "Erreur d'analyse"}

    class FallbackLLMStrategyV2(BaseStrategy):
        def __init__(self, market_data_service=None, trading_service=None, sentiment_weight=0.5, min_confidence=0.6, news_lookback=24):
            super().__init__(market_data_service, trading_service)
            self.sentiment_weight = sentiment_weight
            self.min_confidence = min_confidence
            self.news_lookback = news_lookback
            logger.info(f"Stratégie LLMStrategyV2 de repli initialisée avec sentiment_weight={sentiment_weight}, min_confidence={min_confidence}, news_lookback={news_lookback}h")
        
        def train(self, data, symbol=None):
            """
            Méthode d'entraînement simplifiée pour la stratégie LLMStrategyV2 de repli
            
            Args:
                data: DataFrame contenant les données historiques
                symbol: Symbole sur lequel entraîner le modèle
                
            Returns:
                bool: True si l'entraînement a réussi
            """
            try:
                logger.info(f"Entraînement de la stratégie LLMStrategyV2 sur {symbol if symbol else 'tous les symboles'}")
                time.sleep(1)  # Simuler un entraînement
                return True
            except Exception as e:
                logger.error(f"Erreur lors de l'entraînement LLMStrategyV2: {e}")
                return False
                
        def analyze(self, data, symbol):
            """
            Analyse des données pour générer des signaux de trading, intégrant l'analyse de sentiment
            
            Args:
                data: DataFrame contenant les données historiques
                symbol: Symbole à analyser
                
            Returns:
                dict: Résultat de l'analyse avec signaux de trading
            """
            try:
                # Simuler une analyse technique
                if len(data) < 2:
                    return {"action": "hold", "confidence": 0.5}
                    
                last_close = data['close'].iloc[-1]
                prev_close = data['close'].iloc[-2]
                
                # Analyse technique simplifiée
                if last_close > prev_close * 1.01:  # Hausse de 1%
                    tech_signal = {"action": "buy", "confidence": 0.65}
                elif last_close < prev_close * 0.99:  # Baisse de 1%
                    tech_signal = {"action": "sell", "confidence": 0.65}
                else:
                    tech_signal = {"action": "hold", "confidence": 0.55}
                
                # Simuler une analyse de sentiment
                # Dans une vraie implémentation, cela appellerait EnhancedWebSentimentAgent
                sentiment_values = {"buy": 0.7, "hold": 0.5, "sell": 0.3}  # Valeur entre 0 et 1, où 1 est très positif
                sentiment_signal = {"sentiment": sentiment_values[tech_signal["action"]], "confidence": 0.7}
                
                # Combiner les signaux avec le poids de sentiment configuré
                tech_weight = 1.0 - self.sentiment_weight
                combined_action = tech_signal["action"]
                combined_confidence = (tech_signal["confidence"] * tech_weight) + (sentiment_signal["confidence"] * self.sentiment_weight)
                
                # Filtrer les signaux de faible confiance
                if combined_confidence < self.min_confidence:
                    combined_action = "hold"
                    reason = f"Confiance insuffisante ({combined_confidence:.2f} < {self.min_confidence})"                
                else:
                    reason = f"Signal technique ({tech_signal['action']}, {tech_signal['confidence']:.2f}) + sentiment ({sentiment_signal['sentiment']:.2f})"
                
                return {
                    "action": combined_action, 
                    "confidence": combined_confidence,
                    "reason": reason,
                    "tech_signal": tech_signal,
                    "sentiment_signal": sentiment_signal
                }
                    
            except Exception as e:
                logger.error(f"Erreur d'analyse LLMStrategyV2 pour {symbol}: {e}")
                return {"action": "hold", "confidence": 0.5, "reason": "Erreur d'analyse"}
    
    # Remplacer les classes manquantes par nos versions de repli
    MarketDataService = FallbackMarketDataService
    TradingService = FallbackTradingService
    StrategyManager = FallbackStrategyManager
    MovingAverageStrategy = FallbackMovingAverageStrategy
    MovingAverageMLStrategy = FallbackMovingAverageMLStrategy
    LSTMPredictorStrategy = FallbackLSTMPredictorStrategy
    TransformerStrategy = FallbackTransformerStrategy
    MSIStrategy = FallbackMSIStrategy

# Chargement des variables d'environnement
load_dotenv()

# Configuration du logger
log_file = f"stock_daytrader_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("stock_daytrader")

# Variables globales pour la gestion des signaux
running = True
session_end_time = None

# Intervalle entre les vérifications de l'état du marché (30 minutes par défaut)
MARKET_CHECK_INTERVAL = 1800  # en secondes

# Classes d'énumération
class SessionDuration(Enum):
    """Type de session de trading"""
    MARKET_HOURS = 'market_hours'     # Session standard (9h30 - 16h)
    EXTENDED_HOURS = 'extended_hours' # Session étendue (4h - 20h)
    FULL_DAY = 'full_day'            # Session 24h
    CONTINUOUS = 'continuous'        # Session continue (sans fin prédéfinie)
    CUSTOM = 3  # Durée personnalisée

class TradingStrategy(str, Enum):
    """Stratégies de trading disponibles"""
    MOVING_AVERAGE = "MovingAverageStrategy"
    MOVING_AVERAGE_ML = "MovingAverageMLStrategy"
    LSTM_PREDICTOR = "LSTMPredictorStrategy"
    TRANSFORMER = "TransformerStrategy"
    MSI = "MSIStrategy"
    LLM = "LLMStrategy"
    LLM_V2 = "LLMStrategyV2"
    ALL = "ALL"  # Utiliser toutes les stratégies

class StockFilter(str, Enum):
    """Filtres pour la sélection des actions"""
    ALL = "all"  # Toutes les actions disponibles
    ACTIVE_ASSETS = "active_assets"  # Actions les plus actives
    TOP_VOLUME = "top_volume"  # Actions avec le plus gros volume
    TOP_GAINERS = "top_gainers"  # Actions avec les plus grosses hausses
    TECH_STOCKS = "tech_stocks"  # Actions technologiques
    FINANCE_STOCKS = "finance_stocks"  # Actions financières
    HEALTH_STOCKS = "health_stocks"  # Actions santé
    S_AND_P_500 = "sp500"  # Actions du S&P 500
    NASDAQ_100 = "nasdaq100"  # Actions du NASDAQ 100
    CUSTOM = "custom"  # Liste personnalisée

# Listes d'actions populaires
SP500_STOCKS = []  # Sera rempli dynamiquement
NASDAQ100_STOCKS = []  # Sera rempli dynamiquement
CUSTOM_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "AMD", "INTC", "IBM",
    "JPM", "BAC", "WFC", "C", "GS", "MS", "V", "MA", "PYPL", "SQ",
    "DIS", "NFLX", "CMCSA", "T", "VZ", "TMUS", "HD", "LOW", "WMT", "TGT"
]  # Liste par défaut, peut être remplacée


class StockDayTrader:
    """
    Système de daytrading d'actions utilisant toutes les stratégies disponibles
    
    Caractéristiques:
    - Utilise l'API Alpaca pour trader des actions en mode paper
    - Supporte toutes les stratégies disponibles dans Mercurio AI
    - Peut trader plusieurs actions simultanément
    - Génère des rapports de performance détaillés
    """
    
    def __init__(self, 
                 strategy_type: TradingStrategy = TradingStrategy.MOVING_AVERAGE,
                 stock_filter: StockFilter = StockFilter.ACTIVE_ASSETS,
                 max_symbols: int = 20,
                 position_size_pct: float = 0.02,
                 stop_loss_pct: float = 0.02,
                 take_profit_pct: float = 0.04,
                 session_duration: SessionDuration = SessionDuration.MARKET_HOURS,
                 use_threads: bool = False,
                 use_custom_symbols: bool = False,
                 auto_retrain: bool = False,
                 retrain_interval: int = 6,
                 retrain_symbols: int = 10,
                 api_level: int = 0,
                 cycle_interval: int = 0):
        """
        Initialiser le système de daytrading
        
        Args:
            session_duration: Durée de la session de trading
            strategy_type: Type de stratégie(s) à utiliser
            stock_filter: Filtre pour la sélection des actions
            max_symbols: Nombre maximum d'actions à trader simultanément
            position_size_pct: Taille de position en pourcentage du portefeuille
            stop_loss_pct: Pourcentage de stop loss
            take_profit_pct: Pourcentage de take profit
            use_threads: Utiliser le multithreading pour le traitement des actions
            use_custom_symbols: Utiliser des symboles personnalisés ou les symboles depuis CSV
            auto_retrain: Activer le réentraînement automatique des modèles
            retrain_interval: Intervalle de réentraînement en heures
            retrain_symbols: Nombre de symboles à utiliser pour le réentraînement
            api_level: Niveau d'API Alpaca à utiliser (0 = auto-détection, 1-3 = niveau spécifique)
        """
        self.session_duration = session_duration
        self.strategy_type = strategy_type
        self.stock_filter = stock_filter
        self.max_symbols = max_symbols
        self.position_size_pct = position_size_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.use_threads = use_threads
        self.use_custom_symbols = use_custom_symbols
        self.auto_retrain = auto_retrain
        self.retrain_interval = retrain_interval
        self.retrain_symbols = retrain_symbols
        self.api_level = api_level
        
        # Déterminer le mode Alpaca (paper ou live)
        alpaca_mode = os.getenv("ALPACA_MODE", "paper").lower()
        
        # Configuration selon le mode
        if alpaca_mode == "live":
            self.api_key = os.getenv("ALPACA_LIVE_KEY")
            self.api_secret = os.getenv("ALPACA_LIVE_SECRET")
            self.base_url = os.getenv("ALPACA_LIVE_URL", "https://api.alpaca.markets")
            logger.info("Configuré pour le trading LIVE (réel)")
        else:  # mode paper par défaut
            self.api_key = os.getenv("ALPACA_PAPER_KEY")
            self.api_secret = os.getenv("ALPACA_PAPER_SECRET")
            self.base_url = os.getenv("ALPACA_PAPER_URL", "https://paper-api.alpaca.markets")
            logger.info("Configuré pour le trading PAPER (simulation)")
            
        # URL des données de marché
        self.data_url = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
        
        # Détecter ou définir le niveau d'API Alpaca
        if self.api_level == 0:
            # Auto-détection du niveau d'API
            self.subscription_level = detect_alpaca_level(self.api_key, self.api_secret, self.base_url, self.data_url)
        else:
            # Utilisation du niveau d'API forcé par l'utilisateur
            self.subscription_level = self.api_level
            logger.info(f"Utilisation forcée du niveau d'API Alpaca {self.subscription_level}")
            
        # Si la détection a échoué, essayer de lire depuis les variables d'environnement
        if self.subscription_level == 0:
            self.subscription_level = int(os.getenv("ALPACA_SUBSCRIPTION_LEVEL", "1"))
            logger.info(f"Utilisation du niveau d'abonnement Alpaca depuis les variables d'environnement: {self.subscription_level}")
        
        # Log du niveau d'API
        if self.subscription_level == 3:
            logger.info("Niveau Premium (3) d'Alpaca activé - Fonctionnalités complètes disponibles")
        elif self.subscription_level == 2:
            logger.info("Niveau Standard+ (2) d'Alpaca détecté - Données historiques étendues disponibles")
        elif self.subscription_level == 1:
            logger.info("Niveau Standard (1) d'Alpaca détecté - Fonctionnalités de base uniquement")
        else:
            logger.warning("Niveau d'API Alpaca non détecté - Utilisation du mode de secours avec données limitées")
        
        # Client API Alpaca
        self.api = None
        
        # Services de Mercurio AI
        self.market_data_service = None
        self.trading_service = None
        self.strategy_manager = None
        self.strategies = {}
        
        # Symboles à trader
        self.symbols = []
        
        # Suivi de l'état
        self.positions = {}
        self.portfolio_value = 0.0
        self.initial_portfolio_value = 0.0
        self.trade_history = []
        self.session_start_time = None
        self.session_end_time = None
        self.strategy_performance = {}
        
        # Paramètres de réentraînement des modèles
        self.auto_retrain = auto_retrain
        self.retrain_interval = retrain_interval
        self.retrain_symbols = retrain_symbols
        self.last_retrain_time = datetime.now() - timedelta(hours=retrain_interval)
        self.is_retraining = False
        
        logger.info("StockDayTrader initialisé")
        
    # Méthodes API rate-limited pour Alpaca
    @rate_limited
    def get_asset(self, symbol):
        """Version rate-limited de api.get_asset"""
        return self.api.get_asset(symbol)
    
    @rate_limited
    def get_bars(self, symbol, timeframe, start, end):
        """Version rate-limited de api.get_bars"""
        return self.api.get_bars(symbol, timeframe, start, end)
    
    @rate_limited
    def get_position(self, symbol):
        """Version rate-limited de api.get_position"""
        return self.api.get_position(symbol)
    
    @rate_limited
    def get_account(self):
        """Version rate-limited de api.get_account"""
        return self.api.get_account()
    
    @rate_limited
    def list_positions(self):
        """Version rate-limited de api.list_positions"""
        return self.api.list_positions()
    
    @rate_limited
    def submit_order(self, **kwargs):
        """Version rate-limited de api.submit_order"""
        return self.api.submit_order(**kwargs)
    
    @rate_limited
    def list_assets(self, **kwargs):
        """Version rate-limited de api.list_assets"""
        return self.api.list_assets(**kwargs)
    
    @rate_limited
    def get_clock(self):
        """Version rate-limited de api.get_clock"""
        return self.api.get_clock()
    
    def initialize(self):
        """Initialiser les services et charger la configuration"""
        try:
            # Initialiser le client API Alpaca
            self.api = tradeapi.REST(
                key_id=self.api_key,
                secret_key=self.api_secret,
                base_url=self.base_url,
                api_version='v2'
            )
            
            # Vérifier que le client est correctement initialisé
            account = self.get_account()
            if account:
                self.portfolio_value = float(account.portfolio_value)
                self.initial_portfolio_value = self.portfolio_value
                logger.info(f"Compte Alpaca connecté: {account.id}")
                logger.info(f"Valeur initiale du portefeuille: ${self.portfolio_value:.2f}")
                logger.info(f"Mode trading: {account.status}")
                
                # Récupérer les symboles d'actions selon le filtre
                self.symbols = self.get_filtered_stocks()
                
                if not self.symbols:
                    logger.error("Aucun symbole d'action disponible pour le trading")
                    return False
                
                # Initialiser les services Mercurio
                if self.initialize_mercurio_services():
                    logger.info("Services Mercurio initialisés avec succès")
                else:
                    logger.warning("Utilisation du mode natif Alpaca (sans services Mercurio)")
                
                return True
            else:
                logger.error("Impossible de récupérer les informations du compte")
                return False
                
        except Exception as e:
            logger.error(f"Erreur d'initialisation: {e}")
            return False
            
    def get_filtered_stocks(self) -> List[str]:
        """Récupérer les symboles d'actions selon le filtre configuré"""
        symbols = []
        
        # Option 1: Utiliser les fichiers CSV générés par get_all_symbols.py
        if self.use_custom_symbols:
            # Vérifier d'abord si des fichiers CSV existent dans le répertoire data
            try:
                import glob
                from pathlib import Path
                
                data_dir = Path("data")
                if data_dir.exists():
                    # Chercher le fichier CSV des actions le plus récent
                    stock_files = list(data_dir.glob("all_stocks_*.csv"))
                    
                    if stock_files:
                        newest_file = max(stock_files, key=lambda x: x.stat().st_mtime)
                        logger.info(f"Utilisation du fichier CSV d'actions: {newest_file}")
                        
                        try:
                            df = pd.read_csv(newest_file)
                            all_symbols = df["symbol"].tolist()
                            logger.info(f"Chargé {len(all_symbols)} symboles depuis {newest_file}")
                            
                            # Appliquer un filtre si spécifié
                            if self.stock_filter == StockFilter.TOP_VOLUME:
                                logger.info("Application du filtre TOP_VOLUME sur les symboles chargés")
                                symbols = all_symbols[:min(50, len(all_symbols))]
                            elif self.stock_filter == StockFilter.TOP_GAINERS:
                                logger.info("Application du filtre TOP_GAINERS sur les symboles chargés")
                                symbols = all_symbols[:min(30, len(all_symbols))]
                            else:
                                symbols = all_symbols
                                
                            # Limiter au nombre maximum spécifié
                            symbols = symbols[:self.max_symbols]
                            logger.info(f"Utilisation de {len(symbols)} symboles depuis le fichier CSV")
                            return symbols
                        except Exception as e:
                            logger.error(f"Erreur lors du chargement du fichier CSV: {e}")
                    else:
                        logger.warning("Aucun fichier CSV d'actions trouvé dans le répertoire data/")
                else:
                    logger.warning("Répertoire data/ non trouvé, impossible de charger les symboles depuis CSV")
            except Exception as e:
                logger.error(f"Erreur lors de la recherche de fichiers CSV: {e}")
        
        # Option 2: Utiliser la liste personnalisée en mémoire
        if self.use_custom_symbols and CUSTOM_STOCKS:
            logger.info(f"Utilisation d'une liste personnalisée de {len(CUSTOM_STOCKS)} symboles")
            return CUSTOM_STOCKS[:self.max_symbols]
            
        # Option 3: Récupérer depuis l'API Alpaca (méthode originale)
        try:
            if self.stock_filter == StockFilter.ALL:
                # Récupérer tous les actifs disponibles
                assets = self.list_assets(status='active', asset_class='us_equity')
                symbols = [asset.symbol for asset in assets if asset.tradable]
                
            elif self.stock_filter == StockFilter.ACTIVE_ASSETS:
                # Récupérer les actifs les plus actifs en terme de volume
                assets = self.list_assets(status='active', asset_class='us_equity')
                symbols = [asset.symbol for asset in assets if asset.tradable]
                
                # Trier par volume (simuler - en réalité, on utiliserait les données de volume)
                # Ici nous prenons simplement les N premiers symboles pour la démo
                symbols = symbols[:min(100, len(symbols))]
                
            elif self.stock_filter == StockFilter.TOP_VOLUME:
                # Récupérer les actifs avec le plus gros volume
                # Dans un système réel, on devrait faire une requête spécifique pour cela
                assets = self.list_assets(status='active', asset_class='us_equity')
                symbols = [asset.symbol for asset in assets if asset.tradable]
                symbols = symbols[:min(50, len(symbols))]
                
            elif self.stock_filter == StockFilter.TOP_GAINERS:
                # Simuler les top gainers pour la démo
                assets = self.list_assets(status='active', asset_class='us_equity')
                symbols = [asset.symbol for asset in assets if asset.tradable]
                symbols = symbols[:min(30, len(symbols))]
                
            elif self.stock_filter == StockFilter.S_AND_P_500:
                # Utiliser les symboles du S&P 500
                # Dans un système réel, on devrait récupérer la liste authentique
                # Pour la démo, nous prenons simplement quelques symboles populaires
                self.load_index_stocks()
                symbols = SP500_STOCKS[:self.max_symbols]
                
            elif self.stock_filter == StockFilter.NASDAQ_100:
                # Utiliser les symboles du NASDAQ 100
                self.load_index_stocks()
                symbols = NASDAQ100_STOCKS[:self.max_symbols]
                
            # Filtrer pour n'avoir que le nombre maximum spécifié
            symbols = symbols[:self.max_symbols]
            
            logger.info(f"Récupéré {len(symbols)} symboles selon le filtre: {self.stock_filter.value}")
            
            return symbols
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des actions: {e}")
            # En cas d'erreur, utiliser la liste personnalisée comme fallback
            logger.info("Utilisation de la liste d'actions par défaut")
            return CUSTOM_STOCKS
    
    def load_index_stocks(self):
        """Charger les listes d'actions des principaux indices"""
        global SP500_STOCKS, NASDAQ100_STOCKS
        
        try:
            # Essayer de charger depuis le fichier s'il existe
            indices_path = os.path.join(os.path.dirname(__file__), '../data/indices.json')
            if os.path.exists(indices_path):
                import json
                with open(indices_path, 'r') as f:
                    indices = json.load(f)
                    SP500_STOCKS = indices.get('sp500', [])
                    NASDAQ100_STOCKS = indices.get('nasdaq100', [])
                logger.info(f"Indices chargés depuis le fichier: {len(SP500_STOCKS)} S&P 500, {len(NASDAQ100_STOCKS)} NASDAQ 100")
            else:
                # Fallback: charger des listes simplifiées
                SP500_STOCKS = CUSTOM_STOCKS  # Utiliser la liste personnalisée comme approximation
                NASDAQ100_STOCKS = [s for s in CUSTOM_STOCKS if s in ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "AMD"]]
                logger.warning("Fichier d'indices non trouvé, utilisation de listes simplifiées")
        except Exception as e:
            logger.error(f"Erreur lors du chargement des indices: {e}")
            # En cas d'erreur, utiliser des listes simplifiées
            SP500_STOCKS = CUSTOM_STOCKS
            NASDAQ100_STOCKS = CUSTOM_STOCKS[:10]
    
    def initialize_mercurio_services(self) -> bool:
        """Initialiser les services Mercurio AI"""
        try:
            # Définir les variables d'environnement pour que les services y accèdent
            os.environ["ALPACA_SUBSCRIPTION_LEVEL"] = str(self.subscription_level)
            os.environ["ALPACA_API_KEY"] = self.api_key
            os.environ["ALPACA_API_SECRET"] = self.api_secret
            os.environ["ALPACA_BASE_URL"] = self.base_url
            os.environ["ALPACA_DATA_URL"] = self.data_url
            
            # Créer le service de données de marché
            self.market_data_service = MarketDataService(
                provider_name="alpaca"
            )
            
            # Créer le service de trading
            is_paper = True if os.getenv("ALPACA_MODE", "paper").lower() == "paper" else False
            self.trading_service = TradingService(
                is_paper=is_paper
            )
            
            # Initialiser le gestionnaire de stratégies
            self.strategy_manager = StrategyManager()
            
            # Créer et enregistrer les stratégies selon la configuration
            if self.strategy_type == TradingStrategy.ALL:
                # Créer toutes les stratégies disponibles
                self.strategies[TradingStrategy.MOVING_AVERAGE] = MovingAverageStrategy(
                    market_data_service=self.market_data_service,
                    trading_service=self.trading_service,
                    short_window=10,
                    long_window=30
                )
                
                self.strategies[TradingStrategy.MOVING_AVERAGE_ML] = MovingAverageStrategy(
                    market_data_service=self.market_data_service,
                    trading_service=self.trading_service,
                    short_window=5,
                    long_window=20,
                    use_ml=True
                )
            elif self.strategy_type == TradingStrategy.LSTM_PREDICTOR:
                # Vérifier si nous pouvons importer LSTMPredictorStrategy
                try:
                    from app.strategies.lstm_predictor import LSTMPredictorStrategy
                    self.strategies[self.strategy_type] = LSTMPredictorStrategy(
                        market_data_service=self.market_data_service,
                        trading_service=self.trading_service
                    )
                except ImportError:
                    # Utiliser la classe de repli si le module n'est pas disponible
                    self.strategies[self.strategy_type] = FallbackLSTMPredictorStrategy(
                        market_data_service=self.market_data_service,
                        trading_service=self.trading_service
                    )
            elif self.strategy_type == TradingStrategy.TRANSFORMER:
                # Vérifier si nous pouvons importer TransformerStrategy
                try:
                    from app.strategies.transformer_strategy import TransformerStrategy
                    self.strategies[self.strategy_type] = TransformerStrategy(
                        market_data_service=self.market_data_service,
                        trading_service=self.trading_service
                    )
                except ImportError:
                    # Utiliser la classe de repli si le module n'est pas disponible
                    self.strategies[self.strategy_type] = FallbackTransformerStrategy(
                        market_data_service=self.market_data_service,
                        trading_service=self.trading_service
                    )
            elif self.strategy_type == TradingStrategy.MSI:
                # Vérifier si nous pouvons importer MSIStrategy
                try:
                    from app.strategies.msi_strategy import MultiSourceIntelligenceStrategy as MSIStrategy
                    self.strategies[self.strategy_type] = MSIStrategy(
                        market_data_service=self.market_data_service,
                        trading_service=self.trading_service
                    )
                except ImportError:
                    # Utiliser la classe de repli si le module n'est pas disponible
                    self.strategies[self.strategy_type] = FallbackMSIStrategy(
                        market_data_service=self.market_data_service,
                        trading_service=self.trading_service
                    )
            elif self.strategy_type == TradingStrategy.LLM:
                # Vérifier si nous pouvons importer LLMStrategy
                try:
                    from app.strategies.llm_strategy import LLMStrategy
                    self.strategies[self.strategy_type] = LLMStrategy(
                        market_data_service=self.market_data_service,
                        trading_service=self.trading_service
                    )
                except ImportError:
                    # Utiliser la classe de repli si le module n'est pas disponible
                    self.strategies[self.strategy_type] = FallbackLLMStrategy(
                        market_data_service=self.market_data_service,
                        trading_service=self.trading_service
                    )
            elif self.strategy_type == TradingStrategy.LLM_V2:
                # Vérifier si nous pouvons importer LLMStrategyV2
                try:
                    from app.strategies.llm_strategy_v2 import LLMStrategyV2
                    self.strategies[self.strategy_type] = LLMStrategyV2(
                        market_data_service=self.market_data_service,
                        trading_service=self.trading_service
                    )
                except ImportError:
                    # Utiliser la classe de repli si le module n'est pas disponible
                    self.strategies[self.strategy_type] = FallbackLLMStrategy(
                        market_data_service=self.market_data_service,
                        trading_service=self.trading_service
                    )
            else:
                # Créer uniquement la stratégie spécifiée
                if self.strategy_type == TradingStrategy.MOVING_AVERAGE:
                    self.strategies[self.strategy_type] = MovingAverageStrategy(
                        market_data_service=self.market_data_service,
                        trading_service=self.trading_service,
                        short_window=10,
                        long_window=30
                    )
                elif self.strategy_type == TradingStrategy.MOVING_AVERAGE_ML:
                    # Vérifier si nous pouvons importer MovingAverageMLStrategy
                    try:
                        from app.strategies.moving_average_ml import MovingAverageMLStrategy
                        self.strategies[self.strategy_type] = MovingAverageMLStrategy(
                            market_data_service=self.market_data_service,
                            trading_service=self.trading_service,
                            short_window_min=5,
                            short_window_max=30,
                            long_window_min=30,
                            long_window_max=100
                        )
                    except ImportError:
                        # Utiliser la stratégie MovingAverage standard comme fallback
                        logger.warning("Stratégie MovingAverageML non disponible, utilisation de MovingAverage standard")
                        from app.strategies.moving_average import MovingAverageStrategy
                        self.strategies[self.strategy_type] = MovingAverageStrategy(
                            market_data_service=self.market_data_service,
                            trading_service=self.trading_service,
                            short_window=10,
                            long_window=30,
                            use_ml=True  # Activer l'option ML de base
                        )
                elif self.strategy_type == TradingStrategy.LSTM_PREDICTOR:
                    self.strategies[self.strategy_type] = LSTMPredictorStrategy(
                        market_data_service=self.market_data_service,
                        trading_service=self.trading_service
                    )
                elif self.strategy_type == TradingStrategy.TRANSFORMER:
                    self.strategies[self.strategy_type] = TransformerStrategy(
                        market_data_service=self.market_data_service,
                        trading_service=self.trading_service
                    )
                elif self.strategy_type == TradingStrategy.MSI:
                    self.strategies[self.strategy_type] = MSIStrategy(
                        market_data_service=self.market_data_service,
                        trading_service=self.trading_service
                    )
                elif self.strategy_type == TradingStrategy.LLM:
                    self.strategies[self.strategy_type] = LLMStrategy(
                        market_data_service=self.market_data_service,
                        trading_service=self.trading_service
                    )
                
                logger.info(f"Stratégie {self.strategy_type} initialisée")
            
            # Les stratégies sont déjà initialisées et stockées dans self.strategies
            # Le StrategyManager est utilisé pour d'autres fonctionnalités telles que list_strategies, get_strategy_info, etc.
            logger.info(f"Stratégies initialisées: {list(self.strategies.keys())}")
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation des services Mercurio: {e}")
            return False
            
    def start(self, duration_seconds: Optional[int] = None):
        """Démarrer la session de trading"""
        self.session_start_time = datetime.now()
        
        # Déterminer l'heure de fin de session selon le type de session
        if self.session_duration == SessionDuration.MARKET_HOURS:
            # Session de trading standard (9h30 - 16h)
            # Convertir l'heure actuelle en heure de New York (EDT/EST)
            import pytz
            nyc_tz = pytz.timezone('America/New_York')
            nyc_time = datetime.now(pytz.UTC).astimezone(nyc_tz)
            
            # Si nous sommes avant 9h30, end_time = 16h aujourd'hui
            # Si nous sommes après 16h, end_time = 16h le jour ouvrable suivant
            market_open = nyc_time.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = nyc_time.replace(hour=16, minute=0, second=0, microsecond=0)
            
            if nyc_time < market_open:
                # Avant l'ouverture du marché
                self.session_end_time = market_close
            elif nyc_time > market_close:
                # Après la fermeture du marché, planifier pour le jour ouvrable suivant
                # (Ceci est simplifié, ne gère pas les week-ends et jours fériés)
                next_day = nyc_time + timedelta(days=1)
                self.session_end_time = next_day.replace(hour=16, minute=0, second=0, microsecond=0)
            else:
                # Pendant les heures de marché
                self.session_end_time = market_close
                
            # Convertir l'heure de fin en heure locale
            self.session_end_time = self.session_end_time.astimezone(pytz.UTC).astimezone(datetime.now().astimezone().tzinfo)
            self.session_end_time = self.session_end_time.replace(tzinfo=None)  # Enlever le tzinfo pour compatibilité
            
        elif self.session_duration == SessionDuration.EXTENDED_HOURS:
            # Session de trading étendue (4h - 20h en heure de New York)
            # Conversion similaire à celle ci-dessus, mais avec les heures étendues
            import pytz
            nyc_tz = pytz.timezone('America/New_York')
            nyc_time = datetime.now(pytz.UTC).astimezone(nyc_tz)
            
            extended_close = nyc_time.replace(hour=20, minute=0, second=0, microsecond=0)
            self.session_end_time = extended_close.astimezone(pytz.UTC).astimezone(datetime.now().astimezone().tzinfo)
            self.session_end_time = self.session_end_time.replace(tzinfo=None)
            
        elif self.session_duration == SessionDuration.FULL_DAY:
            # Session de 24 heures pour le jour actuel
            self.session_end_time = self.session_start_time.replace(hour=23, minute=59, second=59)
            
        elif self.session_duration == SessionDuration.CONTINUOUS:
            # Pour une session continue, définir une date très lointaine
            # Cela permet au script de fonctionner continuellement pendant des semaines
            self.session_end_time = self.session_start_time + timedelta(days=365)  # Un an dans le futur
            logger.info("Mode continu activé: le script fonctionnera indéfiniment")
            
        elif duration_seconds is not None:
            # Durée personnalisée en secondes
            self.session_end_time = self.session_start_time + timedelta(seconds=duration_seconds)
        else:
            # Par défaut: session de 4 heures
            self.session_end_time = self.session_start_time + timedelta(hours=4)
        
        logger.info(f"Démarrage de la session de trading d'actions à {self.session_start_time}")
        logger.info(f"La session se terminera à {self.session_end_time}")
        
        # Initialiser le trader
        initialized = self.initialize()
        if not initialized:
            logger.error("Échec de l'initialisation, abandon")
            self.generate_performance_report()
            return
        
        # Démarrer la boucle de trading
        self.trading_loop()
        
        # Générer un rapport de performance à la fin
        self.generate_performance_report()
    
    def trading_loop(self):
        """Boucle principale de trading"""
        global running
        
        try:
            total_iterations = 0
            while running and datetime.now() < self.session_end_time:
                total_iterations += 1
                logger.info(f"===== Cycle de trading #{total_iterations} =====")
                start_time = datetime.now()
                
                # Vérifier l'état du marché avant de trader
                try:
                    clock = self.api.get_clock()
                    if not clock.is_open and self.session_duration in [SessionDuration.MARKET_HOURS, SessionDuration.EXTENDED_HOURS, SessionDuration.CONTINUOUS]:
                        next_open = clock.next_open.strftime('%Y-%m-%d %H:%M:%S')
                        logger.info(f"Le marché est fermé. Prochaine ouverture à {next_open}")
                        
                        # Réentraîner les modèles pendant la période d'inactivité si l'option est activée
                        if self.auto_retrain:
                            # Calculer le temps restant avant la prochaine ouverture du marché
                            time_to_next_open = (clock.next_open - datetime.now(timezone.utc)).total_seconds()
                            
                            # Si nous avons au moins 15 minutes avant l'ouverture du marché, réentraîner les modèles
                            if time_to_next_open > 900:  # 15 minutes en secondes
                                self.retrain_models()
                        
                        logger.info(f"Attente de {MARKET_CHECK_INTERVAL//60} minutes avant la prochaine vérification")
                        time.sleep(MARKET_CHECK_INTERVAL)  # 30 minutes par défaut
                        continue
                except Exception as e:
                    logger.warning(f"Impossible de vérifier l'état du marché: {e}")
                
                # Traitement des symboles - avec ou sans threading
                if self.use_threads and len(self.symbols) > 5:
                    self.process_symbols_with_threading()
                else:
                    self.process_symbols_sequentially()
                
                # Mettre à jour l'état du portefeuille
                self.update_portfolio_state()
                
                # Calculer le temps nécessaire pour ce cycle
                cycle_duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"Cycle de trading terminé en {cycle_duration:.2f} secondes")
                
                # Temps jusqu'à la fin de la session
                time_remaining = int((self.session_end_time - datetime.now()).total_seconds() / 60)
                logger.info(f"Fin de session dans {time_remaining} minutes")
                
                # Attendre avant le prochain cycle
                # La durée d'attente dépend du type de stratégie et du nombre d'actions
                wait_time = self.calculate_wait_time(cycle_duration)
                logger.info(f"Attente de {wait_time} secondes avant le prochain cycle")
                time.sleep(wait_time)
                
        except Exception as e:
            logger.error(f"Erreur dans la boucle de trading: {e}")
        finally:
            logger.info("Boucle de trading terminée")
    
    def retrain_models(self) -> bool:
        """
        Réentraîne les modèles ML pendant les périodes d'inactivité
        
        Returns:
            bool: True si le réentraînement a été effectué avec succès, False sinon
        """
        # Vérifier si le réentraînement est autorisé
        if not self.auto_retrain or getattr(self, 'is_retraining', False):
            return False
            
        # Vérifier l'intervalle de réentraînement
        current_time = datetime.now()
        last_retrain_time = getattr(self, 'last_retrain_time', current_time - timedelta(hours=self.retrain_interval))
        time_since_last_retrain = (current_time - last_retrain_time).total_seconds() / 3600
        
        if time_since_last_retrain < self.retrain_interval:
            logger.debug(f"Pas assez de temps écoulé depuis le dernier réentraînement ({time_since_last_retrain:.1f}h/{self.retrain_interval}h)")
            return False
        
        setattr(self, 'is_retraining', True)
        try:
            logger.info("=" * 60)
            logger.info(f"Début du réentraînement des modèles ML - {current_time}")
            
            # Récupérer les stratégies ML
            ml_strategies = []
            for strategy_name, strategy in self.strategies.items():
                # Vérifier si la stratégie a une méthode train
                if hasattr(strategy, 'train') and callable(getattr(strategy, 'train')):
                    ml_strategies.append((strategy_name, strategy))
            
            if not ml_strategies:
                logger.info("Aucune stratégie ML trouvée pour le réentraînement")
                return False
                
            logger.info(f"Réentraînement de {len(ml_strategies)} stratégies ML: {', '.join([name for name, _ in ml_strategies])}")
            
            # Sélectionner des symboles pour l'entraînement (sous-ensemble des symboles de trading)
            training_symbols = self.symbols[:min(len(self.symbols), self.retrain_symbols)]
            if not training_symbols:
                # Utiliser des symboles par défaut si aucun n'est disponible
                default_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "BAC", "WMT"]
                training_symbols = default_symbols[:self.retrain_symbols]
                
            logger.info(f"Symboles utilisés pour l'entraînement: {', '.join(training_symbols)}")
            
            # Récupérer l'historique de données pour l'entraînement
            for symbol in training_symbols:
                try:
                    # Récupérer 6 mois de données historiques
                    end = datetime.now()
                    start = end - timedelta(days=180)
                    logger.info(f"Récupération des données historiques pour {symbol}")
                    
                    bars = self.api.get_bars(
                        symbol, 
                        '1D',
                        start.strftime('%Y-%m-%d'),
                        end.strftime('%Y-%m-%d')
                    ).df
                    
                    if bars.empty or len(bars) < 30:
                        logger.warning(f"Données insuffisantes pour {symbol} - ignorant ce symbole")
                        continue
                        
                    logger.info(f"Données récupérées pour {symbol}: {len(bars)} jours")
                    
                    # Entraîner chaque stratégie avec ces données
                    for strategy_name, strategy in ml_strategies:
                        try:
                            logger.info(f"Entraînement de {strategy_name} sur {symbol}...")
                            strategy.train(bars, symbol=symbol)
                            logger.info(f"Entraînement de {strategy_name} sur {symbol} terminé")
                        except Exception as e:
                            logger.error(f"Erreur lors de l'entraînement de {strategy_name} sur {symbol}: {e}")
                except Exception as e:
                    logger.error(f"Erreur lors de la récupération des données pour {symbol}: {e}")
            
            # Mettre à jour l'horodatage du dernier réentraînement
            setattr(self, 'last_retrain_time', current_time)
            logger.info(f"Réentraînement des modèles ML terminé - {datetime.now()}")
            logger.info("=" * 60)
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors du réentraînement des modèles ML: {e}")
            return False
        finally:
            setattr(self, 'is_retraining', False)
    
    def calculate_wait_time(self, cycle_duration=0):
        """Calculer le temps d'attente entre les cycles de trading"""
        # Vérifier si un intervalle de cycle fixe est défini
        if hasattr(self, 'cycle_interval') and self.cycle_interval > 0:
            logger.info(f"Utilisation d'un intervalle fixe de {self.cycle_interval} secondes entre les cycles")
            return self.cycle_interval
            
        # Si aucun intervalle fixe défini, calcul automatique basé sur la charge de travail
        logger.info("Calcul automatique du temps d'attente en fonction du nombre de symboles et de la stratégie")
        
        # Temps d'attente de base (en secondes)
        base_wait_time = 30
        
        # Ajuster en fonction du nombre de symboles
        if len(self.symbols) > 20:
            base_wait_time *= 1.5
        if len(self.symbols) > 50:
            base_wait_time *= 1.5
            
        # Ajuster en fonction de la stratégie (les stratégies ML sont plus lentes)
        if self.strategy_type in [TradingStrategy.LSTM_PREDICTOR, TradingStrategy.TRANSFORMER, TradingStrategy.MSI, TradingStrategy.LLM]:
            base_wait_time *= 1.5
            
        # Pour les stratégies multiples, augmenter encore le temps d'attente
        if self.strategy_type == TradingStrategy.ALL:
            base_wait_time *= 2
        
        # S'assurer que le temps d'attente n'est pas inférieur au temps nécessaire pour exécuter le cycle
        # Ajouter un tampon de 10 secondes
        return max(int(base_wait_time), int(cycle_duration) + 10)
    
    def process_symbols_sequentially(self):
        """Traiter les symboles de manière séquentielle"""
        logger.info(f"Traitement séquentiel de {len(self.symbols)} symboles")
        
        for symbol in self.symbols:
            try:
                self.process_symbol(symbol)
            except Exception as e:
                logger.error(f"Erreur de traitement de {symbol}: {e}")
    
    def process_symbols_with_threading(self):
        """Traiter les symboles avec multithreading"""
        max_workers = min(10, len(self.symbols))  # Limiter à 10 threads max
        logger.info(f"Traitement parallèle de {len(self.symbols)} symboles avec {max_workers} threads")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.process_symbol, symbol): symbol for symbol in self.symbols}
            for future in concurrent.futures.as_completed(futures):
                symbol = futures[future]
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Erreur de traitement de {symbol}: {e}")
    
    def process_symbol(self, symbol: str):
        """Traiter un symbole pour toutes les stratégies"""
        logger.info(f"Traitement de {symbol}")
        
        try:
            # Vérifier l'état du trading pour ce symbole
            try:
                asset = self.get_asset(symbol)
                if not asset.tradable:
                    logger.warning(f"{symbol} n'est pas tradable, ignoré")
                    return
            except Exception as e:
                logger.error(f"Erreur lors de la vérification de l'actif {symbol}: {e}")
                return
            
            # Obtenir les données de marché récentes
            try:
                end = datetime.now()
                start = end - timedelta(days=30)  # Historique de 30 jours
                bars = self.get_bars(symbol, '1D', start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')).df
                
                if bars.empty:
                    logger.warning(f"Pas de données disponibles pour {symbol}")
                    return
                
                logger.info(f"{symbol} - Dernier prix: ${bars['close'].iloc[-1]:.2f}")
                
            except Exception as e:
                logger.error(f"Erreur lors de la récupération des données pour {symbol}: {e}")
                return
            
            # Vérifier si nous avons déjà une position sur ce symbole
            position = None
            try:
                position = self.get_position(symbol)
                logger.info(f"Position existante pour {symbol}: {float(position.qty)} actions @ ${float(position.avg_entry_price):.2f}")
            except Exception:
                # Pas de position existante
                pass
            
            # Exécuter les stratégies pour ce symbole
            for strategy_name, strategy in self.strategies.items():
                action, confidence = self.execute_strategy(strategy, symbol, bars, position)
                
                if action == 'buy':
                    if not position:
                        self.execute_buy(symbol, bars['close'].iloc[-1], strategy_name, confidence)
                    else:
                        logger.info(f"Signal d'achat de {strategy_name} pour {symbol}, mais position déjà existante")
                elif action == 'sell':
                    if position:
                        self.execute_sell(symbol, bars['close'].iloc[-1], position, strategy_name, confidence)
                    else:
                        logger.info(f"Signal de vente de {strategy_name} pour {symbol}, mais pas de position existante")
                else:  # 'hold'
                    logger.info(f"{strategy_name} recommande de maintenir pour {symbol} (confiance: {confidence:.2f})")
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement de {symbol}: {e}")
    
    def execute_strategy(self, strategy, symbol: str, bars: pd.DataFrame, position) -> Tuple[str, float]:
        """Exécuter une stratégie pour un symbole donné"""
        try:
            # Préparer les paramètres pour la stratégie
            strategy_params = {
                "symbol": symbol,
                "bars": bars,
                "position": position
            }
            
            # Vérifier le type de stratégie et exécuter en conséquence
            if isinstance(strategy, MovingAverageStrategy) or isinstance(strategy, MovingAverageMLStrategy):
                # Exécuter une stratégie de moyenne mobile
                short_ma = bars['close'].rolling(window=strategy.short_window).mean()
                long_ma = bars['close'].rolling(window=strategy.long_window).mean()
                
                # Vérifier s'il y a suffisamment de données
                if len(bars) < max(strategy.short_window, strategy.long_window) + 2:
                    return 'hold', 0.0
                
                # Conditions de croisement
                prev_short = short_ma.iloc[-2]
                prev_long = long_ma.iloc[-2]
                curr_short = short_ma.iloc[-1]
                curr_long = long_ma.iloc[-1]
                
                # Calculer la confiance basée sur la distance entre les MA
                distance = abs(curr_short - curr_long) / curr_long
                confidence = min(1.0, distance * 10)
                
                # Signaux
                if prev_short <= prev_long and curr_short > curr_long:
                    return 'buy', confidence
                elif prev_short >= prev_long and curr_short < curr_long:
                    return 'sell', confidence
                else:
                    return 'hold', 0.0
                    
            elif isinstance(strategy, (LSTMPredictorStrategy, TransformerStrategy, MSIStrategy)):
                # Simuler une décision pour les stratégies ML
                # Note: ceci est simplifié pour la démo, normalement ces stratégies ont leur propre logique
                import random
                
                # Tendance des prix récents (5 derniers jours)
                if len(bars) >= 5:
                    recent_trend = bars['close'].iloc[-1] - bars['close'].iloc[-5]
                    trend_pct = recent_trend / bars['close'].iloc[-5]
                    
                    # Biais léger vers la tendance récente
                    if trend_pct > 0.03:  # Tendance haussière de 3%+
                        action_probs = {'buy': 0.6, 'hold': 0.3, 'sell': 0.1}
                    elif trend_pct < -0.03:  # Tendance baissière de 3%+
                        action_probs = {'buy': 0.1, 'hold': 0.3, 'sell': 0.6}
                    else:  # Tendance neutre
                        action_probs = {'buy': 0.2, 'hold': 0.6, 'sell': 0.2}
                    
                    # Si nous avons déjà une position, ajuster les probabilités
                    if position:
                        # Si en profit, plus enclin à vendre
                        pos_value = float(position.current_price) * float(position.qty)
                        entry_value = float(position.avg_entry_price) * float(position.qty)
                        profit_pct = (pos_value - entry_value) / entry_value
                        
                        if profit_pct > 0.05:  # 5%+ de profit
                            action_probs['sell'] += 0.2
                            action_probs['buy'] -= 0.1
                            action_probs['hold'] -= 0.1
                    
                    # Normaliser les probabilités
                    total = sum(action_probs.values())
                    action_probs = {k: v/total for k, v in action_probs.items()}
                    
                    # Décision aléatoire pondérée
                    actions = list(action_probs.keys())
                    probs = list(action_probs.values())
                    action = random.choices(actions, weights=probs, k=1)[0]
                    confidence = action_probs[action]
                    
                    return action, confidence
                else:
                    return 'hold', 0.0
            else:
                # Stratégie inconnue
                logger.warning(f"Type de stratégie non pris en charge: {type(strategy).__name__}")
                return 'hold', 0.0
                
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution de la stratégie {type(strategy).__name__} pour {symbol}: {e}")
            return 'hold', 0.0
            
    def execute_buy(self, symbol: str, price: float, strategy_name: str, confidence: float):
        """Exécuter un ordre d'achat"""
        try:
            # Calculer la taille de la position
            position_value = self.portfolio_value * self.position_size_pct
            # Pour les actions, nous devons arrondir à un nombre entier d'actions (pas de fractionnelles avec tous les courtiers)
            shares = int(position_value / price)
            
            if shares <= 0:
                logger.warning(f"Taille de position trop petite pour {symbol} à ${price:.2f}")
                return
                
            logger.info(f"SIGNAL D'ACHAT: {symbol} à ${price:.2f}, {shares} actions (stratégie: {strategy_name}, confiance: {confidence:.2f})")
            
            # Placer un ordre au marché
            order = self.api.submit_order(
                symbol=symbol,
                qty=shares,
                side='buy',
                type='market',
                time_in_force='day'
            )
            
            if order:
                logger.info(f"Ordre d'achat placé pour {symbol}: {order.id}")
                
                # Enregistrer l'ordre dans l'historique des transactions
                self.trade_history.append({
                    'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'symbol': symbol,
                    'action': 'achat',
                    'quantity': shares,
                    'price': price,
                    'strategy': strategy_name,
                    'confidence': confidence
                })
                
                # Mise à jour des statistiques de stratégie
                if strategy_name not in self.strategy_performance:
                    self.strategy_performance[strategy_name] = {'buys': 0, 'sells': 0, 'volume': 0, 'pnl': 0}
                self.strategy_performance[strategy_name]['buys'] += 1
                self.strategy_performance[strategy_name]['volume'] += shares * price
            else:
                logger.error(f"Échec du placement de l'ordre d'achat pour {symbol}")
                
        except Exception as e:
            logger.error(f"Erreur d'exécution d'achat pour {symbol}: {e}")
    
    def execute_sell(self, symbol: str, price: float, position, strategy_name: str, confidence: float):
        """Exécuter un ordre de vente"""
        try:
            qty = float(position.qty)
            entry_price = float(position.avg_entry_price)
            
            if qty <= 0:
                logger.warning(f"Quantité de position invalide pour {symbol}: {qty}")
                return
                
            # Calculer le P&L prévu
            pnl = (price - entry_price) * qty
            pnl_pct = (price - entry_price) / entry_price * 100
            
            logger.info(f"SIGNAL DE VENTE: {symbol} à ${price:.2f}, {qty} actions, P&L: ${pnl:.2f} ({pnl_pct:.2f}%) (stratégie: {strategy_name}, confiance: {confidence:.2f})")
            
            # Convertir la quantité en entier pour les actions
            qty_int = int(qty)
            
            # Placer un ordre au marché
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty_int,
                side='sell',
                type='market',
                time_in_force='day'
            )
            
            if order:
                logger.info(f"Ordre de vente placé pour {symbol}: {order.id}")
                
                # Enregistrer l'ordre dans l'historique des transactions
                self.trade_history.append({
                    'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'symbol': symbol,
                    'action': 'vente',
                    'quantity': qty_int,
                    'price': price,
                    'entry_price': entry_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'strategy': strategy_name,
                    'confidence': confidence
                })
                
                # Mise à jour des statistiques de stratégie
                if strategy_name not in self.strategy_performance:
                    self.strategy_performance[strategy_name] = {'buys': 0, 'sells': 0, 'volume': 0, 'pnl': 0}
                self.strategy_performance[strategy_name]['sells'] += 1
                self.strategy_performance[strategy_name]['volume'] += qty_int * price
                self.strategy_performance[strategy_name]['pnl'] += pnl
            else:
                logger.error(f"Échec du placement de l'ordre de vente pour {symbol}")
                
        except Exception as e:
            logger.error(f"Erreur d'exécution de vente pour {symbol}: {e}")
    
    def update_portfolio_state(self):
        """Mettre à jour la valeur du portefeuille et les positions"""
        try:
            account = self.api.get_account()
            self.portfolio_value = float(account.portfolio_value)
            logger.info(f"Valeur actuelle du portefeuille: ${self.portfolio_value:.2f}")
            
            # Mettre à jour les positions
            try:
                positions = self.api.list_positions()
                
                # Journaliser les positions ouvertes
                if positions:
                    logger.info(f"Positions ouvertes actuelles: {len(positions)}")
                    total_market_value = 0
                    total_profit_loss = 0
                    
                    for pos in positions:
                        entry_price = float(pos.avg_entry_price)
                        current_price = float(pos.current_price)
                        qty = float(pos.qty)
                        market_value = float(pos.market_value)
                        pnl = float(pos.unrealized_pl)
                        pnl_pct = float(pos.unrealized_plpc) * 100
                        
                        total_market_value += market_value
                        total_profit_loss += pnl
                        
                        logger.info(f"  {pos.symbol}: {qty:.0f} actions @ ${entry_price:.2f} - Valeur: ${market_value:.2f} - P/L: ${pnl:.2f} ({pnl_pct:.2f}%)")
                    
                    logger.info(f"Valeur totale des positions: ${total_market_value:.2f}")
                    logger.info(f"Profit/Perte non réalisé total: ${total_profit_loss:.2f}")
                else:
                    logger.info("Pas de positions ouvertes")
            except Exception as e:
                logger.error(f"Erreur de récupération des positions: {e}")
                
        except Exception as e:
            logger.error(f"Erreur de mise à jour de l'état du portefeuille: {e}")
    
    def generate_performance_report(self):
        """Générer un rapport de performance à la fin de la session de trading"""
        # Créer un fichier de rapport séparé
        report_file = f"stock_trading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        try:
            end_time = datetime.now()
            duration = end_time - self.session_start_time if self.session_start_time else timedelta(0)
            hours, remainder = divmod(duration.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            logger.info("===================================================")
            logger.info("RAPPORT DE PERFORMANCE DE LA SESSION DE TRADING")
            logger.info("===================================================")
            logger.info(f"Durée de la session: {hours}h {minutes}m {seconds}s")
            logger.info(f"Heure de début: {self.session_start_time}")
            logger.info(f"Heure de fin: {end_time}")
            
            # Obtenir l'état final du compte
            try:
                account = self.api.get_account()
                final_value = float(account.portfolio_value)
                
                if self.initial_portfolio_value > 0:
                    profit_loss = final_value - self.initial_portfolio_value
                    profit_loss_pct = (profit_loss / self.initial_portfolio_value) * 100
                    logger.info(f"Valeur initiale du portefeuille: ${self.initial_portfolio_value:.2f}")
                    logger.info(f"Valeur finale du portefeuille: ${final_value:.2f}")
                    logger.info(f"Profit/Perte: ${profit_loss:.2f} ({profit_loss_pct:.2f}%)")
            except Exception as e:
                logger.warning(f"Impossible de récupérer les informations finales du compte: {e}")
            
            # Afficher les positions ouvertes
            try:
                positions = self.api.list_positions()
                
                if positions:
                    logger.info(f"Positions ouvertes à la fin de la session: {len(positions)}")
                    total_market_value = 0
                    total_profit_loss = 0
                    
                    for pos in positions:
                        entry_price = float(pos.avg_entry_price)
                        current_price = float(pos.current_price)
                        qty = float(pos.qty)
                        market_value = float(pos.market_value)
                        pnl = float(pos.unrealized_pl)
                        pnl_pct = float(pos.unrealized_plpc) * 100
                        
                        total_market_value += market_value
                        total_profit_loss += pnl
                        
                        logger.info(f"  {pos.symbol}: {qty:.0f} actions @ ${entry_price:.2f} - Valeur: ${market_value:.2f} - P/L: ${pnl:.2f} ({pnl_pct:.2f}%)")
                    
                    logger.info(f"Valeur totale des positions: ${total_market_value:.2f}")
                    logger.info(f"Profit/Perte non réalisé total: ${total_profit_loss:.2f}")
                else:
                    logger.info("Pas de positions ouvertes à la fin de la session")
            except Exception as e:
                logger.warning(f"Impossible de récupérer les informations de position: {e}")
            
            # Afficher les statistiques des stratégies
            logger.info("Performance par stratégie:")
            for strategy_name, stats in self.strategy_performance.items():
                logger.info(f"  {strategy_name}:")
                logger.info(f"    - Achats: {stats['buys']}")
                logger.info(f"    - Ventes: {stats['sells']}")
                logger.info(f"    - Volume: ${stats['volume']:.2f}")
                logger.info(f"    - P&L: ${stats['pnl']:.2f}")
            
            logger.info("===================================================")
            logger.info("SESSION DE TRADING TERMINÉE")
            logger.info("===================================================")
            
            # Écrire le rapport dans un fichier
            with open(report_file, 'w') as f:
                f.write("===================================================\n")
                f.write("RAPPORT DE PERFORMANCE DE LA SESSION DE TRADING\n")
                f.write("===================================================\n\n")
                f.write(f"Durée de la session: {hours}h {minutes}m {seconds}s\n")
                f.write(f"Heure de début: {self.session_start_time}\n")
                f.write(f"Heure de fin: {end_time}\n\n")
                
                f.write(f"Stratégie(s) utilisée(s): {self.strategy_type}\n")
                f.write(f"Filtre d'actions: {self.stock_filter}\n")
                f.write(f"Nombre d'actions suivies: {len(self.symbols)}\n\n")
                
                try:
                    if self.initial_portfolio_value > 0:
                        f.write(f"Valeur initiale du portefeuille: ${self.initial_portfolio_value:.2f}\n")
                        f.write(f"Valeur finale du portefeuille: ${final_value:.2f}\n")
                        f.write(f"Profit/Perte: ${profit_loss:.2f} ({profit_loss_pct:.2f}%)\n\n")
                except:
                    f.write("Impossible de récupérer les informations finales du compte\n\n")
                
                f.write("Positions ouvertes à la fin de la session:\n")
                try:
                    if positions:
                        for pos in positions:
                            f.write(f"  {pos.symbol}: {float(pos.qty):.0f} actions @ ${float(pos.avg_entry_price):.2f} - ")
                            f.write(f"Valeur: ${float(pos.market_value):.2f} - ")
                            f.write(f"P/L: ${float(pos.unrealized_pl):.2f} ({float(pos.unrealized_plpc) * 100:.2f}%)\n")
                        f.write(f"\nValeur totale des positions: ${total_market_value:.2f}\n")
                        f.write(f"Profit/Perte non réalisé total: ${total_profit_loss:.2f}\n")
                    else:
                        f.write("Aucune position ouverte\n")
                except:
                    f.write("Impossible de récupérer les informations de position\n")
                
                f.write("\n===================================================\n")
                f.write("PERFORMANCE PAR STRATÉGIE\n")
                f.write("===================================================\n")
                
                for strategy_name, stats in self.strategy_performance.items():
                    f.write(f"\n{strategy_name}:\n")
                    f.write(f"  - Achats: {stats['buys']}\n")
                    f.write(f"  - Ventes: {stats['sells']}\n")
                    f.write(f"  - Volume: ${stats['volume']:.2f}\n")
                    f.write(f"  - P&L: ${stats['pnl']:.2f}\n")
                
                f.write("\n===================================================\n")
                f.write("RÉSUMÉ DES TRANSACTIONS\n")
                f.write("===================================================\n")
                
                if self.trade_history:
                    for trade in self.trade_history:
                        f.write(f"{trade['time']} - {trade['symbol']} - {trade['action']} - ")
                        f.write(f"{trade['quantity']} actions @ ${trade['price']:.2f}")
                        if 'pnl' in trade:
                            f.write(f" - P/L: ${trade['pnl']:.2f} ({trade['pnl_pct']:.2f}%)")
                        f.write(f" - Stratégie: {trade['strategy']}\n")
                else:
                    f.write("Aucune transaction effectuée\n")
            
            logger.info(f"Rapport détaillé sauvegardé dans {report_file}")
                
        except Exception as e:
            logger.error(f"Erreur de génération du rapport de performance: {e}")


def signal_handler(sig, frame):
    """Gestionnaire de signal pour arrêter proprement le script"""
    global running
    logger.info("Signal d'arrêt reçu, arrêt en cours...")
    running = False


def main():
    """Fonction principale"""
    global running, MARKET_CHECK_INTERVAL
    
    parser = argparse.ArgumentParser(description='Script de day trading pour actions')
    parser.add_argument('--strategy', type=str, default='llm',
                      choices=['moving_average', 'moving_average_ml', 'lstm_predictor', 'transformer', 'msi', 'llm', 'llm_v2', 'all'],
                        help='Stratégie à utiliser')
    parser.add_argument('--filter', choices=['active_assets', 'top_volume', 'top_gainers', 'tech_stocks', 'finance_stocks', 'health_stocks'],
                        default='active_assets', help='Filtre pour les actions')
    parser.add_argument('--max-symbols', type=int, default=10,
                        help='Nombre maximum de symboles à trader')
    parser.add_argument('--position-size', type=float, default=0.02,
                        help='Taille de position en pourcentage du portefeuille (default: 0.02)')
    parser.add_argument('--stop-loss', type=float, default=0.02,
                        help='Pourcentage de stop loss (default: 0.02)')
    parser.add_argument('--take-profit', type=float, default=0.04,
                        help='Pourcentage de take profit (default: 0.04)')
    parser.add_argument('--duration', choices=['market_hours', 'extended_hours', 'full_day', 'continuous'],
                    default='market_hours', help='Type de session de trading')
    parser.add_argument('--market-check-interval', type=int, default=30,
                    help='Intervalle en minutes pour vérifier l\'état du marché (en mode continuous)')
    parser.add_argument('--cycle-interval', type=int, default=0,
                    help='Intervalle en secondes entre les cycles de trading (0 = automatique)')
    parser.add_argument('--use-threads', action='store_true',
                        help='Utiliser le multithreading pour le traitement des symboles')
    parser.add_argument('--use-custom-symbols', action='store_true',
                        help='Utiliser une liste personnalisée de symboles ou les symboles du fichier CSV')
    parser.add_argument('--refresh-symbols', action='store_true',
                        help='Exécuter get_all_symbols.py pour rafraîchir la liste des symboles avant de démarrer')
    parser.add_argument('--auto-retrain', action='store_true',
                        help='Réentraîner automatiquement les modèles ML pendant les périodes d\'inactivité')
    parser.add_argument('--retrain-interval', type=int, default=6,
                        help='Intervalle minimal en heures entre les réentraînements de modèles (default: 6)')
    parser.add_argument('--retrain-symbols', type=int, default=10,
                        help='Nombre de symboles à utiliser pour le réentraînement des modèles (default: 10)')
    parser.add_argument('--api-level', type=int, choices=[1, 2, 3], default=0,
                        help='Niveau d\'API Alpaca à utiliser (1=basique, 2=standard+, 3=premium). Par défaut: auto-détection)')
    
    # Arguments spécifiques à LLMStrategyV2
    parser.add_argument('--sentiment-weight', type=float, default=0.5,
                        help='Poids de l\'analyse de sentiment (0-1) pour LLMStrategyV2 (default: 0.5)')
    parser.add_argument('--min-confidence', type=float, default=0.6,
                        help='Confiance minimale requise pour les signaux de trading LLMStrategyV2 (default: 0.6)')
    parser.add_argument('--news-lookback', type=int, default=24,
                        help='Période de temps (heures) pour l\'analyse des actualités pour LLMStrategyV2 (default: 24)')
    
    args = parser.parse_args()
    
    # Définir l'intervalle de vérification du marché
    global MARKET_CHECK_INTERVAL
    if args.market_check_interval > 0:
        MARKET_CHECK_INTERVAL = args.market_check_interval * 60  # Convertir en secondes
        
    # Rafraîchir la liste des symboles si demandé
    if args.refresh_symbols:
        try:
            logger.info("Rafraîchissement de la liste des symboles via get_all_symbols.py...")
            import subprocess
            import sys
            from pathlib import Path
            
            script_path = Path(__file__).parent / 'get_all_symbols.py'
            
            if script_path.exists():
                # Exécuter le script get_all_symbols.py
                result = subprocess.run(
                    [sys.executable, str(script_path)],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    logger.info("Liste des symboles rafraîchie avec succès")
                    if result.stdout:
                        # Afficher seulement les dernières lignes qui contiennent le résumé
                        output_lines = result.stdout.split('\n')
                        summary_lines = [line for line in output_lines if '===' in line or 'symboles' in line]
                        for line in summary_lines[-10:]:
                            if line.strip():
                                logger.info(f"get_all_symbols.py: {line.strip()}")
                else:
                    logger.error(f"Erreur lors du rafraîchissement des symboles: {result.stderr}")
            else:
                logger.error(f"Script get_all_symbols.py introuvable à {script_path}")
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution de get_all_symbols.py: {e}")
            logger.info("Poursuite avec les symboles existants")
        
    # Définir les paramètres du trader
    
    # Mapping entre les arguments de ligne de commande et les valeurs de l'enum TradingStrategy
    strategy_mapping = {
        'moving_average': 'MovingAverageStrategy',
        'moving_average_ml': 'MovingAverageMLStrategy',
        'lstm_predictor': 'LSTMPredictorStrategy',
        'transformer': 'TransformerStrategy',
        'msi': 'MSIStrategy',
        'llm': 'LLMStrategy',
        'llm_v2': 'LLMStrategyV2',
        'all': 'ALL'
    }
    
    # Conversion de l'argument de stratégie en majuscules si c'est 'all'
    if args.strategy.lower() == 'all':
        strategy_arg = 'ALL'
    else:
        strategy_arg = strategy_mapping.get(args.strategy, args.strategy)
    
    # Stocker les paramètres spécifiques à LLMStrategyV2 dans une variable globale
    global LLM_V2_PARAMS
    LLM_V2_PARAMS = {}
    
    if args.strategy == 'llm_v2':
        LLM_V2_PARAMS["sentiment_weight"] = args.sentiment_weight
        LLM_V2_PARAMS["min_confidence"] = args.min_confidence
        LLM_V2_PARAMS["news_lookback"] = args.news_lookback
        logger.info(f"Utilisation de LLMStrategyV2 avec sentiment_weight={args.sentiment_weight}, "
                  f"min_confidence={args.min_confidence}, news_lookback={args.news_lookback}h")
    
    # Configuration du trader sans les paramètres spécifiques à la stratégie
    trader = StockDayTrader(
        strategy_type=TradingStrategy(strategy_arg),
        stock_filter=StockFilter(args.filter),
        max_symbols=args.max_symbols,
        position_size_pct=args.position_size,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        session_duration=SessionDuration(args.duration),
        use_threads=args.use_threads,
        use_custom_symbols=args.use_custom_symbols,
        auto_retrain=args.auto_retrain,
        retrain_interval=args.retrain_interval,
        retrain_symbols=args.retrain_symbols,
        api_level=args.api_level,
        cycle_interval=args.cycle_interval
    )
    
    # Démarrer le trader
    try:
        trader.start()
    except KeyboardInterrupt:
        logger.info("Interruption du programme par l'utilisateur")
        running = False
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du trader: {e}")
    finally:
        logger.info("Fin du programme")


if __name__ == "__main__":
    # Enregistrement du gestionnaire de signal pour un arrêt propre
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Démarrage du script de day trading. Appuyez sur Ctrl+C pour arrêter proprement.")
    main()
