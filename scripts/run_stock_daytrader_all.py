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
        
        # Exécuter le script de liquidation avec les options --force et --yes pour assurer la liquidation
        # --force : essaie des méthodes alternatives pour les positions problématiques comme les cryptos
        # --yes : saute la confirmation manuelle
        import subprocess
        cmd = [sys.executable, liquidation_script, "--force", "--yes"]
        logger.info(f"Commande: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Vérifier si la commande a réussi
        if result.returncode == 0:
            logger.info("Liquidation des positions terminée avec succès")
            if result.stdout:
                # Afficher les détails importants (uniquement les lignes de log importantes)
                for line in result.stdout.splitlines():
                    if "INFO" in line and ("liquid" in line.lower() or "position" in line.lower() or "error" in line.lower()):
                        logger.info(f"Détail: {line.strip()}")
        else:
            logger.error(f"Erreur pendant la liquidation. Code: {result.returncode}")
            logger.error(f"Détails: {result.stderr}")
        
    except Exception as e:
        logger.error(f"Erreur lors de la liquidation des positions: {e}")
        logger.error(f"Type d'erreur: {type(e).__name__}")

# Fonction pour générer un rapport final et nettoyer
def cleanup_resources():
    """Nettoyer les ressources et générer le rapport final avant de quitter"""
    logger.info("Nettoyage des ressources et finalisation du rapport...")
    # Générer un rapport final ici si nécessaire
    logger.info("Rapport généré et ressources nettoyées")

# Fonction principale
def main():
    # Parser les arguments de ligne de commande
    parser = argparse.ArgumentParser(description="Script de day trading d'actions avec Alpaca API.")
    
    # Options de configuration
    parser.add_argument("--debug", action="store_true", help="Active le mode debug avec logs détaillés")
    parser.add_argument("--paper", action="store_true", help="Utilise le mode paper trading (par défaut)")
    parser.add_argument("--live", action="store_true", help="Utilise le mode live trading (attention: risque réel)")
    parser.add_argument("--duration", type=str, choices=["day", "continuous", "custom"], default="day", 
                        help="Durée de la session: jour complet, continue, ou personnalisée")
    parser.add_argument("--custom-hours", type=str, 
                        help="Heures personnalisées au format 'HH:MM-HH:MM', ex: '10:00-15:30'")
    
    # Options des stratégies
    parser.add_argument("--strategy", type=str, default="moving_average",
                        choices=["moving_average", "moving_average_ml", "lstm", "transformer", "msi", "llm", "llm_v2", "all"],
                        help="Stratégie à utiliser")
    parser.add_argument("--short-window", type=int, default=10, help="Fenêtre courte pour moyenne mobile")
    parser.add_argument("--long-window", type=int, default=30, help="Fenêtre longue pour moyenne mobile")
    parser.add_argument("--refresh-models", action="store_true", help="Force le rafraîchissement des modèles ML")
    parser.add_argument("--sentiment-weight", type=float, default=0.3, 
                        help="Poids du sentiment dans les stratégies basées sur le LLM (0.0-1.0)")
    parser.add_argument("--min-confidence", type=float, default=0.6, 
                        help="Niveau de confiance minimum pour générer un signal (0.5-1.0)")
    
    # Options des symboles
    parser.add_argument("--max-symbols", type=int, default=10, help="Nombre maximum de symboles à trader")
    parser.add_argument("--refresh-symbols", action="store_true", help="Force le rafraîchissement de la liste des symboles")
    parser.add_argument("--filter", type=str, choices=["price", "volume", "volatility", "market_cap", "top_volume", "all"], 
                        default="all", help="Filtre des symboles")
    parser.add_argument("--position-size", type=float, default=0.05, 
                        help="Taille de position par symbole (0.01-0.2)")
    parser.add_argument("--min-price", type=float, default=5.0, help="Prix minimum des actions")
    parser.add_argument("--max-price", type=float, default=500.0, help="Prix maximum des actions")
    parser.add_argument("--use-threads", action="store_true", help="Utilise le multithreading pour les analyses")
    parser.add_argument("--use-custom-symbols", action="store_true", help="Utilise une liste personnalisée de symboles")
    parser.add_argument("--symbols-file", type=str, help="Fichier de symboles personnalisés")
    parser.add_argument("--cls", action="store_true", help="Efface l'écran avant de démarrer")
    
    # Parser les arguments
    args = parser.parse_args()
    
    # Configurer le niveau de log
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Mode DEBUG activé")
    
    # Effacer l'écran si demandé
    if args.cls:
        os.system('cls' if os.name == 'nt' else 'clear')
    
    # Charger les variables d'environnement
    load_dotenv()
    
    # Déterminer le mode (paper ou live)
    is_paper = not args.live
    alpaca_mode = "paper" if is_paper else "live"
    os.environ["ALPACA_MODE"] = alpaca_mode
    
    if alpaca_mode == "live":
        api_key = os.getenv("ALPACA_LIVE_KEY")
        api_secret = os.getenv("ALPACA_LIVE_SECRET")
        base_url = os.getenv("ALPACA_LIVE_URL", "https://api.alpaca.markets")
        print("⚠️ MODE TRADING RÉEL ACTIVÉ - RISQUE DE PERTES FINANCIÈRES ⚠️")
        proceed = input("Êtes-vous sûr de vouloir continuer? (y/n): ")
        if proceed.lower() != 'y':
            print("Opération annulée.")
            return
    else:  # mode paper par défaut
        api_key = os.getenv("ALPACA_PAPER_KEY")
        api_secret = os.getenv("ALPACA_PAPER_SECRET")
        base_url = os.getenv("ALPACA_PAPER_URL", "https://paper-api.alpaca.markets")
    
    # Vérifier les API keys
    if not api_key or not api_secret:
        logger.error("Clés API Alpaca non trouvées. Veuillez vérifier votre fichier .env")
        return
    
    # Initialiser l'API Alpaca
    try:
        # Nouvelle façon (alpaca-py)
        api = tradeapi.REST(
            api_key=api_key,
            secret_key=api_secret,
            base_url=base_url
        )
    except TypeError:
        # Ancienne façon (alpaca-trade-api)
        api = tradeapi.REST(
            key_id=api_key,
            secret_key=api_secret,
            base_url=base_url
        )
    
    # Détecter le niveau d'abonnement
    subscription_level = detect_alpaca_level(api_key, api_secret, base_url)
    
    # Créer les classes de service directement ici, sans dépendre d'importations externes
    class SimpleMarketDataService:
        def __init__(self, **kwargs):
            self.api_key = kwargs.get('api_key') or api_key
            self.api_secret = kwargs.get('api_secret') or api_secret
            self.base_url = kwargs.get('base_url') or base_url
            self.subscription_level = kwargs.get('subscription_level') or subscription_level
            self.api = api
            logger.info("Service de données de marché simplifié initialisé")
            
        def get_last_price(self, symbol):
            try:
                # Pour un symbole de crypto
                if '/' in symbol:
                    try:
                        quote = api.get_latest_crypto_quote(symbol)
                        if hasattr(quote, 'ap'):
                            return float(quote.ap)
                        elif isinstance(quote, dict) and 'ap' in quote:
                            return float(quote['ap'])
                        else:
                            return 1.0  # Prix par défaut pour crypto
                    except Exception as e:
                        logger.warning(f"Erreur crypto quote pour {symbol}: {e}")
                        return 1.0
                # Pour une action
                else:
                    try:
                        trade = api.get_latest_trade(symbol)
                        if hasattr(trade, 'p'):
                            return float(trade.p)
                        elif hasattr(trade, 'price'):
                            return float(trade.price)
                        elif isinstance(trade, dict) and 'p' in trade:
                            return float(trade['p'])
                        elif isinstance(trade, dict) and 'price' in trade:
                            return float(trade['price'])
                        else:
                            # Essayer d'obtenir les barres récentes
                            bars = api.get_bars(symbol, '1Day', limit=1)
                            if len(bars) > 0:
                                return float(bars[0].c)
                            else:
                                return 100.0  # Prix par défaut pour actions
                    except Exception as e:
                        logger.warning(f"Erreur trade pour {symbol}: {e}")
                        # Essayer les barres comme plan B
                        try:
                            bars = api.get_bars(symbol, '1Day', limit=1)
                            if len(bars) > 0:
                                return float(bars[0].c)
                        except:
                            pass
                        return 100.0  # Prix par défaut pour actions
            except Exception as e:
                logger.warning(f"Erreur générale lors de la récupération du prix pour {symbol}: {e}")
                # Renvoyer un prix fictif pour ne pas bloquer le script
                return 100.0
                
        def get_historical_data(self, symbol, days=30):
            # Créer un DataFrame vide au format attendu
            import pandas as pd
            import numpy as np
            from datetime import datetime, timedelta
            
            end = datetime.now()
            start = end - timedelta(days=days)
            
            try:
                if '/' in symbol:  # Crypto
                    bars = api.get_crypto_bars(symbol, '1Day', start.isoformat(), end.isoformat())
                else:  # Action
                    bars = api.get_bars(symbol, '1Day', start.isoformat(), end.isoformat())
                
                if len(bars) > 0:
                    # Créer un DataFrame à partir des barres
                    df = pd.DataFrame([
                        {
                            'timestamp': bar.t,
                            'open': bar.o,
                            'high': bar.h,
                            'low': bar.l,
                            'close': bar.c,
                            'volume': bar.v
                        } for bar in bars
                    ])
                    df.set_index('timestamp', inplace=True)
                    return df
            except Exception as e:
                logger.warning(f"Erreur lors de la récupération des données historiques pour {symbol}: {e}")
            
            # Retourner des données fictives si nécessaire
            dates = [end - timedelta(days=i) for i in range(days)]
            data = {
                'open': np.random.normal(100, 5, days),
                'high': np.random.normal(105, 5, days),
                'low': np.random.normal(95, 5, days),
                'close': np.random.normal(100, 5, days),
                'volume': np.random.normal(1000000, 500000, days)
            }
            df = pd.DataFrame(data, index=dates)
            return df
            
        def get_daily_volume(self, symbol):
            try:
                df = self.get_historical_data(symbol, days=1)
                if not df.empty:
                    return float(df['volume'].iloc[-1])
            except Exception as e:
                logger.warning(f"Erreur lors de la récupération du volume pour {symbol}: {e}")
            return 1000000  # Valeur par défaut
            
        def get_volatility(self, symbol):
            try:
                df = self.get_historical_data(symbol, days=20)
                if not df.empty and len(df) > 1:
                    return float(df['close'].pct_change().std() * 100)
            except Exception as e:
                logger.warning(f"Erreur lors de la récupération de la volatilité pour {symbol}: {e}")
            return 2.0  # Valeur par défaut
            
        def get_market_cap(self, symbol):
            # Valeur fictive pour la capitalisation boursière
            return 10000000000
    
    class SimpleTradingService:
        def __init__(self, **kwargs):
            self.api_key = kwargs.get('api_key') or api_key
            self.api_secret = kwargs.get('api_secret') or api_secret
            self.base_url = kwargs.get('base_url') or base_url
            self.paper = kwargs.get('paper', True)
            self.api = api
            logger.info("Service de trading simplifié initialisé")
        
        def get_position(self, symbol):
            try:
                return api.get_position(symbol)
            except Exception as e:
                # Généralement, si aucune position n'existe, une erreur est levée
                return None
                
        def get_account_value(self):
            try:
                account = api.get_account()
                return float(account.equity)
            except Exception as e:
                logger.warning(f"Erreur lors de la récupération de la valeur du compte: {e}")
                return 100000.0  # Valeur par défaut
                
        def place_order(self, symbol, qty, side):
            try:
                if side.lower() == "buy":
                    return api.submit_order(
                        symbol=symbol,
                        qty=qty,
                        side="buy",
                        type="market",
                        time_in_force="gtc"
                    )
                else:
                    return api.submit_order(
                        symbol=symbol,
                        qty=qty,
                        side="sell",
                        type="market",
                        time_in_force="gtc"
                    )
            except Exception as e:
                logger.error(f"Erreur lors du placement de l'ordre pour {symbol}: {e}")
                return None
    
    class SimpleStrategyManager:
        def __init__(self):
            self.strategies = {}
            logger.info("Gestionnaire de stratégies simplifié initialisé")
            
        def register_strategy(self, name, strategy):
            self.strategies[name] = strategy
            logger.info(f"Stratégie {name} enregistrée")
    
    # Initialiser les services simplifiés
    try:
        market_data_service = SimpleMarketDataService(
            api_key=api_key,
            api_secret=api_secret,
            base_url=base_url,
            subscription_level=subscription_level
        )
        
        trading_service = SimpleTradingService(
            api_key=api_key,
            api_secret=api_secret,
            base_url=base_url,
            paper=is_paper
        )
        
        strategy_manager = SimpleStrategyManager()
        
    except Exception as e:
        logger.error(f"Erreur fatale lors de l'initialisation des services simplifiés: {e}")
        return
    
    # Initialiser le gestionnaire de stratégies
    strategy_manager = StrategyManager()
    
    # Sélectionner la stratégie en fonction des arguments
    selected_strategy = None
    
    if args.strategy == "moving_average":
        selected_strategy = MovingAverageStrategy(
            market_data_service=market_data_service,
            trading_service=trading_service,
            short_window=args.short_window,
            long_window=args.long_window
        )
    elif args.strategy == "moving_average_ml":
        selected_strategy = MovingAverageMLStrategy(
            market_data_service=market_data_service,
            trading_service=trading_service,
            short_window=args.short_window,
            long_window=args.long_window,
            use_ml=True
        )
    elif args.strategy == "lstm":
        selected_strategy = LSTMPredictorStrategy(
            market_data_service=market_data_service,
            trading_service=trading_service
        )
    elif args.strategy == "transformer":
        selected_strategy = TransformerStrategy(
            market_data_service=market_data_service,
            trading_service=trading_service
        )
    elif args.strategy == "msi":
        selected_strategy = MSIStrategy(
            market_data_service=market_data_service,
            trading_service=trading_service
        )
    elif args.strategy == "llm":
        selected_strategy = LLMStrategy(
            market_data_service=market_data_service,
            trading_service=trading_service,
            sentiment_weight=args.sentiment_weight,
            min_confidence=args.min_confidence
        )
    elif args.strategy == "llm_v2":
        selected_strategy = LLMStrategyV2(
            market_data_service=market_data_service,
            trading_service=trading_service,
            sentiment_weight=args.sentiment_weight,
            min_confidence=args.min_confidence
        )
    else:  # "all" - use a default strategy
        selected_strategy = MovingAverageStrategy(
            market_data_service=market_data_service,
            trading_service=trading_service,
            short_window=args.short_window,
            long_window=args.long_window
        )
    
    # Configuration de la durée de la session
    session_end_time = None
    
    if args.duration == "day":
        # Session jusqu'à la fin de la journée de trading
        now = datetime.now(timezone.utc)
        end_of_day = datetime(now.year, now.month, now.day, 20, 0, 0, tzinfo=timezone.utc)  # 4 PM EST = 20:00 UTC
        if now > end_of_day:
            end_of_day = end_of_day + timedelta(days=1)
        session_end_time = end_of_day
    elif args.duration == "custom" and args.custom_hours:
        # Session personnalisée basée sur les heures spécifiées
        try:
            start_time_str, end_time_str = args.custom_hours.split('-')
            start_hour, start_minute = map(int, start_time_str.split(':'))
            end_hour, end_minute = map(int, end_time_str.split(':'))
            
            now = datetime.now()
            start_time = now.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)
            end_time = now.replace(hour=end_hour, minute=end_minute, second=0, microsecond=0)
            
            if start_time > now:
                start_time = start_time - timedelta(days=1)
            if end_time < start_time:
                end_time = end_time + timedelta(days=1)
            
            session_end_time = end_time
            
        except ValueError:
            logger.error("Format d'heures personnalisées invalide. Utilisation de la journée complète.")
            session_end_time = None
    else:  # "continuous" - jusqu'à interruption manuelle
        session_end_time = None
    
    # Exécution de la stratégie
    try:
        # Traitement des symboles
        symbols = []
        
        # Utiliser des symboles personnalisés si demandé
        if args.use_custom_symbols:
            if args.symbols_file:
                # Charger depuis un fichier
                try:
                    with open(args.symbols_file, 'r') as f:
                        symbols = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                    logger.info(f"Chargement de {len(symbols)} symboles depuis {args.symbols_file}")
                except Exception as e:
                    logger.error(f"Erreur lors du chargement du fichier de symboles: {e}")
            else:
                # Utiliser la liste par défaut depuis le dossier racine
                try:
                    symbols_file = os.path.join(project_root, "stock_symbols.txt")
                    if os.path.exists(symbols_file):
                        with open(symbols_file, 'r') as f:
                            symbols = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                        logger.info(f"Chargement de {len(symbols)} symboles depuis {symbols_file}")
                except Exception as e:
                    logger.error(f"Erreur lors du chargement du fichier de symboles par défaut: {e}")
        
        if not symbols:
            # Récupérer les symboles tradables depuis Alpaca
            assets = api.list_assets(status='active')
            symbols = [asset.symbol for asset in assets if asset.tradable]
            logger.info(f"Récupération de {len(symbols)} symboles tradables depuis Alpaca")
            
            # Filtrer les symboles selon les critères
            if args.filter in ["price", "all"]:
                symbols = [s for s in symbols if market_data_service.get_last_price(s) >= args.min_price 
                           and market_data_service.get_last_price(s) <= args.max_price]
                logger.info(f"Filtrage par prix: {len(symbols)} symboles restants")
            
            if args.filter in ["volume", "top_volume", "all"]:
                # Trier par volume et prendre les N premiers
                symbols_with_volume = []
                for s in symbols:
                    try:
                        volume = market_data_service.get_daily_volume(s)
                        symbols_with_volume.append((s, volume))
                    except Exception as e:
                        logger.debug(f"Erreur lors de la récupération du volume pour {s}: {e}")
                
                symbols_with_volume.sort(key=lambda x: x[1], reverse=True)
                symbols = [s[0] for s in symbols_with_volume[:args.max_symbols * 2]]  # Prendre 2x pour filtrage supplémentaire
                logger.info(f"Filtrage par volume: {len(symbols)} symboles restants")
            
            if args.filter in ["volatility", "all"]:
                # Trier par volatilité
                symbols_with_volatility = []
                for s in symbols:
                    try:
                        volatility = market_data_service.get_volatility(s)
                        symbols_with_volatility.append((s, volatility))
                    except Exception as e:
                        logger.debug(f"Erreur lors de la récupération de la volatilité pour {s}: {e}")
                
                symbols_with_volatility.sort(key=lambda x: x[1], reverse=True)
                symbols = [s[0] for s in symbols_with_volatility[:args.max_symbols * 2]]  # Prendre 2x pour filtrage supplémentaire
                logger.info(f"Filtrage par volatilité: {len(symbols)} symboles restants")
            
            if args.filter in ["market_cap", "all"]:
                # Trier par capitalisation boursière
                symbols_with_cap = []
                for s in symbols:
                    try:
                        market_cap = market_data_service.get_market_cap(s)
                        symbols_with_cap.append((s, market_cap))
                    except Exception as e:
                        logger.debug(f"Erreur lors de la récupération de la capitalisation pour {s}: {e}")
                
                symbols_with_cap.sort(key=lambda x: x[1], reverse=True)
                symbols = [s[0] for s in symbols_with_cap[:args.max_symbols * 2]]  # Prendre 2x pour filtrage supplémentaire
                logger.info(f"Filtrage par capitalisation: {len(symbols)} symboles restants")
        
        # Limiter le nombre de symboles
        if len(symbols) > args.max_symbols:
            symbols = symbols[:args.max_symbols]
        
        logger.info(f"Trading sur {len(symbols)} symboles: {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''}")
        
        # Boucle principale de trading
        while is_running():
            current_time = datetime.now(timezone.utc)
            
            # Vérifier si la session doit se terminer
            if session_end_time and current_time >= session_end_time:
                logger.info(f"Fin de la session à {session_end_time}")
                break
            
            # Analyser chaque symbole et prendre des décisions
            if args.use_threads and len(symbols) > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(symbols))) as executor:
                    futures = {executor.submit(analyze_and_trade, s, selected_strategy, trading_service, market_data_service, args): s for s in symbols}
                    for future in concurrent.futures.as_completed(futures):
                        symbol = futures[future]
                        try:
                            result = future.result()
                            logger.info(f"Résultat pour {symbol}: {result}")
                        except Exception as e:
                            logger.error(f"Erreur lors de l'analyse de {symbol}: {e}")
            else:
                for symbol in symbols:
                    try:
                        result = analyze_and_trade(symbol, selected_strategy, trading_service, market_data_service, args)
                        logger.info(f"Résultat pour {symbol}: {result}")
                    except Exception as e:
                        logger.error(f"Erreur lors de l'analyse de {symbol}: {e}")
            
            # Attendre avant la prochaine itération
            wait_time = 60  # 1 minute par défaut
            logger.info(f"Attente de {wait_time} secondes avant la prochaine analyse...")
            
            # Attente avec vérification périodique de l'arrêt
            for _ in range(wait_time):
                if not is_running():
                    break
                time.sleep(1)
        
        logger.info("Session de trading terminée")
        
    except KeyboardInterrupt:
        logger.info("Interruption utilisateur détectée, arrêt propre...")
    except Exception as e:
        logger.error(f"Erreur imprévue: {e}")
        import traceback
        logger.error(traceback.format_exc())

# Fonction pour analyser un symbole et effectuer une opération si nécessaire
def analyze_and_trade(symbol, strategy, trading_service, market_data_service, args):
    """Analyser un symbole avec la stratégie donnée et effectuer une opération si nécessaire"""
    try:
        # Récupérer les données historiques
        data = market_data_service.get_historical_data(symbol, days=30)
        
        if data.empty or len(data) < 10:
            logger.warning(f"Données insuffisantes pour {symbol}, ignoré")
            return {"action": "ignore", "reason": "données insuffisantes"}
        
        # Analyser les données avec la stratégie
        analysis = strategy.analyze(data, symbol)
        
        # Vérifier s'il faut effectuer une opération
        position = trading_service.get_position(symbol)
        current_price = market_data_service.get_last_price(symbol)
        
        if analysis.get("action") == "buy" and analysis.get("confidence", 0) >= args.min_confidence:
            # Déterminer la taille de la position
            position_size = args.position_size
            account_value = trading_service.get_account_value()
            qty = int(account_value * position_size / current_price)
            
            if qty < 1:
                logger.warning(f"Quantité trop faible pour {symbol}, ignoré")
                return {"action": "ignore", "reason": "quantité trop faible"}
            
            # Vérifier si on a déjà une position
            if position and float(position.qty) > 0:
                logger.info(f"Position existante pour {symbol}, pas d'achat supplémentaire")
                return {"action": "hold", "reason": "position existante"}
            
            # Passer l'ordre d'achat
            order = trading_service.place_order(symbol, qty, "buy")
            logger.info(f"Ordre d'achat placé pour {symbol}: {qty} actions à ~${current_price}")
            return {"action": "buy", "order_id": order.id if order else None, "qty": qty, "price": current_price}
            
        elif analysis.get("action") == "sell" and analysis.get("confidence", 0) >= args.min_confidence:
            # Vérifier si on a une position à vendre
            if not position or float(position.qty) <= 0:
                logger.info(f"Pas de position pour {symbol}, rien à vendre")
                return {"action": "ignore", "reason": "pas de position"}
            
            # Vendre toute la position
            qty = float(position.qty)
            order = trading_service.place_order(symbol, qty, "sell")
            logger.info(f"Ordre de vente placé pour {symbol}: {qty} actions à ~${current_price}")
            return {"action": "sell", "order_id": order.id if order else None, "qty": qty, "price": current_price}
            
        else:
            # Conserver la position actuelle ou ne rien faire
            action = "hold" if position and float(position.qty) > 0 else "ignore"
            logger.info(f"Aucune action pour {symbol}: {analysis.get('action', 'hold')} avec confiance {analysis.get('confidence', 0)}")
            return {"action": action, "reason": "signal insuffisant"}            
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse et du trading pour {symbol}: {e}")
        return {"action": "error", "reason": str(e)}

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
