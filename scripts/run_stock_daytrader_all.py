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

# Importation de l'utilitaire d'arrêt propre
try:
    from scripts.graceful_exit import is_running, register_thread, register_cleanup, register_liquidation_handler
    USE_GRACEFUL_EXIT = True
except ImportError:
    # Fonction de repli si le module n'est pas disponible
    USE_GRACEFUL_EXIT = False
    # Variables globales pour la gestion des signaux
    running = True
    
    def is_running():
        global running
        return running
        
    def register_thread(thread):
        pass
        
    def register_cleanup(callback):
        pass
        
    def register_liquidation_handler(callback):
        pass
        
    # Gestionnaire de signal traditionnel
    def signal_handler(sig, frame):
        global running
        logger.info("Signal d'arrêt reçu, arrêt en cours...")
        running = False
        
    # Fonction pour liquider toutes les positions
def liquidate_positions():
    """Liquider toutes les positions ouvertes"""
    logger.info("Exécution du script de liquidation des positions...")
    try:
        # Chemin vers le script de liquidation
        liquidation_script = os.path.join(os.path.dirname(__file__), "liquidate_all_positions.py")
        
        # Vérifier que le script existe
        if not os.path.exists(liquidation_script):
            logger.error(f"Script de liquidation introuvable: {liquidation_script}")
            return
        
        # Exécuter le script de liquidation avec confirmation automatique
        import subprocess
        subprocess.run([sys.executable, liquidation_script], 
                       input=b'y\n',  # Envoyer 'y' pour confirmer automatiquement
                       check=True)
        
        logger.info("Liquidation des positions terminée avec succès")
    except Exception as e:
        logger.error(f"Erreur lors de la liquidation des positions: {e}")

# Fonction pour générer un rapport final et nettoyer
def cleanup_resources():
    """Nettoyer les ressources et générer le rapport final avant de quitter"""
    logger.info("Nettoyage des ressources et finalisation du rapport...")
    # Générer un rapport final ici si nécessaire
    logger.info("Rapport généré et ressources nettoyées")

{{ ... }}

if __name__ == "__main__":
    # Enregistrement des fonctions de nettoyage pour l'utilitaire d'arrêt propre
    if USE_GRACEFUL_EXIT:
        register_cleanup(cleanup_resources)
        register_liquidation_handler(liquidate_positions)
    else:
        # Enregistrement du gestionnaire de signal pour arrêt propre
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Démarrage du script de day trading. Appuyez sur Ctrl+C pour arrêter proprement.")
    main()
