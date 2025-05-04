#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MercurioAI - Système Intégré de Trading et Entraînement
-------------------------------------------------------
Ce script combine les fonctionnalités de:
- run_stock_daytrader_all.py: pour le trading actif pendant les heures de marché
- train_all_models.py: pour l'entraînement des modèles pendant les périodes d'inactivité

Utilisation:
    python scripts/run_integrated_trader.py --strategy all --duration continuous --refresh-symbols --auto-training
"""

import os
import sys
import time
import signal
import logging
import argparse
import asyncio
import threading
import subprocess
from datetime import datetime, timedelta, date, timezone
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import concurrent.futures

# Ajouter le répertoire parent au path pour pouvoir importer les modules du projet
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Configuration du logger
log_file = f"integrated_trader_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("integrated_trader")

# Importer les classes et fonctions nécessaires du script run_stock_daytrader_all.py
try:
    from app.services.market_data import MarketDataService
    from app.services.trading import TradingService
    from app.services.strategy_manager import StrategyManager
    
    # Importer le service de calendrier de marché - nouveau service
    try:
        from app.services.market_calendar import MarketCalendarService
        logger.info("Service de calendrier de marché importé avec succès")
        USE_MARKET_CALENDAR = True
    except ImportError as e:
        logger.warning(f"Service de calendrier de marché non disponible: {e}")
        logger.warning("Utilisation de la méthode alternative pour vérifier l'état du marché")
        USE_MARKET_CALENDAR = False
        
    # Importation réussie des modules principaux
except ImportError as e:
    logger.error(f"Erreur d'importation des modules principaux de Mercurio: {e}")
    sys.exit(1)

# Variables globales
running = True
last_training_time = None
TRAINING_INTERVAL = 24 * 60 * 60  # Par défaut: 24 heures entre les entraînements

def signal_handler(sig, frame):
    """Gestionnaire de signal pour arrêter proprement le programme"""
    global running
    logger.info("Signal d'arrêt reçu. Arrêt en cours...")
    running = False

class SessionDuration(Enum):
    """Type de session de trading"""
    MARKET_HOURS = 'market_hours'      # Session standard (9h30 - 16h)
    EXTENDED_HOURS = 'extended_hours'  # Session étendue (4h - 20h)
    FULL_DAY = 'full_day'              # Session 24h
    CONTINUOUS = 'continuous'          # Session continue (sans fin prédéfinie)

class TradingStrategy(str, Enum):
    """Stratégies de trading disponibles"""
    MOVING_AVERAGE = "MovingAverageStrategy"
    LSTM_PREDICTOR = "LSTMPredictorStrategy"
    TRANSFORMER = "TransformerStrategy"
    MSI = "MSIStrategy"
    LLM = "LLMStrategy"
    ALL = "ALL"  # Utiliser toutes les stratégies

def is_market_open():
    """Vérifie si le marché est ouvert actuellement"""
    # Si le service de calendrier est disponible, l'utiliser
    if USE_MARKET_CALENDAR:
        try:
            market_calendar = MarketCalendarService()
            return market_calendar.is_market_open()
        except Exception as e:
            logger.error(f"Erreur lors de la vérification de l'état du marché avec MarketCalendarService: {e}")
            # Utiliser la méthode alternative en cas d'erreur
            logger.info("Utilisation de la méthode alternative pour vérifier l'état du marché")
    
    # Méthode alternative basée sur l'heure locale
    try:
        # Convertir l'heure locale en Eastern Time (ET) où sont basés les marchés américains
        now = datetime.now()
        weekday = now.weekday()
        current_hour = now.hour
        current_minute = now.minute
        
        # Ajustement pour le fuseau horaire (approximatif)
        # Eastern Time est généralement UTC-4 (été) ou UTC-5 (hiver)
        # On suppose ici que l'heure locale est en Europe (UTC+1 ou UTC+2)
        # Donc différence de 6 heures environ
        et_hour = (current_hour - 6) % 24
        
        # Le marché est fermé le weekend (5=samedi, 6=dimanche)
        if weekday >= 5:
            logger.info(f"Marché fermé: weekend (jour {weekday})")
            return False
        
        # Heures d'ouverture régulières: 9h30 à 16h00 ET
        # 9h30 ET = ~15h30 CET/CEST, 16h00 ET = ~22h00 CET/CEST
        is_open = (9 <= et_hour < 16) or (et_hour == 9 and current_minute >= 30)
        
        if is_open:
            logger.info(f"Marché ouvert: {et_hour}:{current_minute} ET")
        else:
            logger.info(f"Marché fermé: {et_hour}:{current_minute} ET")
            
        return is_open
    except Exception as e:
        logger.error(f"Erreur lors de la vérification alternative de l'état du marché: {e}")
        # Par défaut, on considère que le marché est fermé en cas d'erreur
        return False

def should_run_training(auto_training, force_training=False):
    """
    Détermine si l'entraînement des modèles devrait être exécuté
    
    Args:
        auto_training: Indique si l'entraînement automatique est activé
        force_training: Force l'entraînement même si les conditions ne sont pas remplies
    """
    global last_training_time
    
    if not auto_training and not force_training:
        return False
        
    # Si l'entraînement est forcé
    if force_training:
        return True
    
    # Si c'est le premier entraînement ou si l'intervalle est écoulé
    if last_training_time is None:
        return True
    
    time_since_last_training = time.time() - last_training_time
    if time_since_last_training >= TRAINING_INTERVAL:
        return True
    
    return False

def run_training(symbols=None, days=90, use_gpu=False):
    """
    Exécute l'entraînement des modèles
    
    Args:
        symbols: Liste des symboles à entraîner (ou None pour utiliser les symboles par défaut)
        days: Nombre de jours de données historiques à utiliser
        use_gpu: Utiliser le GPU si disponible
    """
    global last_training_time
    
    logger.info("Démarrage de l'entraînement des modèles...")
    
    # Préparer la commande
    cmd = [sys.executable, os.path.join(project_root, "scripts", "train_all_models.py")]
    
    # Ajouter les paramètres
    cmd.extend(["--days", str(days)])
    
    if symbols:
        cmd.extend(["--symbols", ",".join(symbols)])
    else:
        cmd.extend(["--include_stocks", "--include_crypto", "--top_assets", "20"])
    
    if use_gpu:
        cmd.append("--use_gpu")
    
    # Exécuter le script d'entraînement
    try:
        logger.info(f"Exécution de la commande: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Vérifier le résultat
        if result.returncode == 0:
            logger.info("Entraînement des modèles terminé avec succès")
            # Mettre à jour le temps du dernier entraînement
            last_training_time = time.time()
        else:
            logger.error(f"Erreur lors de l'entraînement des modèles: {result.stderr}")
            
        # Afficher la sortie
        for line in result.stdout.splitlines():
            if line.strip():
                logger.info(f"[TRAINING] {line}")
    
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du script d'entraînement: {e}")

def run_trading(strategy, duration, refresh_symbols, auto_retrain, symbols=None, max_symbols=20):
    """
    Exécute le trading avec le script run_stock_daytrader_all.py
    
    Args:
        strategy: Stratégie de trading à utiliser
        duration: Durée de la session de trading
        refresh_symbols: Rafraîchir la liste des symboles
        auto_retrain: Réentraîner automatiquement les modèles pendant le trading
        symbols: Liste des symboles à trader (ou None pour utiliser la détection automatique)
        max_symbols: Nombre maximum de symboles à trader
    """
    # Préparer la commande
    cmd = [sys.executable, os.path.join(project_root, "scripts", "run_stock_daytrader_all.py")]
    
    # Ajouter les paramètres
    cmd.extend(["--strategy", strategy.value])
    cmd.extend(["--duration", duration.value])
    
    if refresh_symbols:
        cmd.append("--refresh-symbols")
    
    if auto_retrain:
        cmd.append("--auto-retrain")
    
    if symbols:
        cmd.extend(["--symbols", ",".join(symbols)])
    
    cmd.extend(["--max-symbols", str(max_symbols)])
    
    # Exécuter le script de trading
    try:
        logger.info(f"Démarrage du trading avec la commande: {' '.join(cmd)}")
        
        # Exécution avec redirection de la sortie vers le logger
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # Line-buffered
        )
        
        # Lire la sortie en temps réel
        for line in iter(process.stdout.readline, ''):
            if not running:
                process.terminate()
                break
            if line.strip():
                logger.info(f"[TRADING] {line.strip()}")
        
        # Attendre la fin du processus
        process.wait()
        
        logger.info("Session de trading terminée")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du script de trading: {e}")

def main():
    """Fonction principale du système intégré"""
    global TRAINING_INTERVAL, running
    
    # Configuration de la gestion des signaux pour l'arrêt propre
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Analyse des arguments
    parser = argparse.ArgumentParser(description="MercurioAI - Système Intégré de Trading et Entraînement")
    
    # Arguments pour le trading
    parser.add_argument("--strategy", type=str, choices=[e.value for e in TradingStrategy], default=TradingStrategy.ALL.value,
                      help="Stratégie(s) à utiliser pour le trading")
    parser.add_argument("--duration", type=str, choices=[e.value for e in SessionDuration], default=SessionDuration.MARKET_HOURS.value,
                      help="Durée de la session de trading")
    parser.add_argument("--refresh-symbols", action="store_true",
                      help="Rafraîchir la liste des symboles avant le trading")
    parser.add_argument("--auto-retrain", action="store_true",
                      help="Réentraîner les modèles périodiquement pendant le trading")
    
    # Arguments pour l'entraînement
    parser.add_argument("--auto-training", action="store_true",
                      help="Activer l'entraînement automatique pendant les périodes d'inactivité")
    parser.add_argument("--training-interval", type=int, default=24,
                      help="Intervalle d'entraînement en heures (défaut: 24)")
    parser.add_argument("--training-days", type=int, default=90,
                      help="Nombre de jours de données historiques pour l'entraînement (défaut: 90)")
    parser.add_argument("--use-gpu", action="store_true",
                      help="Utiliser le GPU pour l'entraînement si disponible")
    
    # Arguments communs
    parser.add_argument("--symbols", type=str, default="",
                      help="Liste des symboles à trader/entraîner, séparés par des virgules")
    parser.add_argument("--max-symbols", type=int, default=20,
                      help="Nombre maximum de symboles à trader (défaut: 20)")
    
    args = parser.parse_args()
    
    # Conversion des arguments
    strategy = TradingStrategy(args.strategy)
    duration = SessionDuration(args.duration)
    symbols = args.symbols.split(',') if args.symbols else None
    
    # Mise à jour de l'intervalle d'entraînement
    TRAINING_INTERVAL = args.training_interval * 60 * 60  # Convertir les heures en secondes
    
    logger.info("MercurioAI - Système Intégré de Trading et Entraînement")
    logger.info(f"Stratégie: {strategy.value}")
    logger.info(f"Durée: {duration.value}")
    logger.info(f"Auto-retrain pendant trading: {args.auto_retrain}")
    logger.info(f"Auto-training pendant inactivité: {args.auto_training}")
    logger.info(f"Intervalle d'entraînement: {args.training_interval} heures")
    
    # Boucle principale
    running = True
    while running:
        # Vérifier si le marché est ouvert
        market_open = is_market_open()
        
        if market_open:
            logger.info("Le marché est ouvert. Démarrage du trading...")
            # Exécuter le trading
            run_trading(
                strategy=strategy,
                duration=duration,
                refresh_symbols=args.refresh_symbols,
                auto_retrain=args.auto_retrain,
                symbols=symbols,
                max_symbols=args.max_symbols
            )
        else:
            logger.info("Le marché est fermé.")
            
            # Vérifier si l'entraînement devrait être exécuté
            if should_run_training(args.auto_training):
                logger.info("Période d'inactivité détectée. Démarrage de l'entraînement des modèles...")
                run_training(
                    symbols=symbols,
                    days=args.training_days,
                    use_gpu=args.use_gpu
                )
            else:
                logger.info("En attente de l'ouverture du marché ou du prochain cycle d'entraînement...")
                
            # Attendre avant la prochaine vérification
            wait_time = 300  # 5 minutes
            logger.info(f"Attente de {wait_time} secondes avant la prochaine vérification...")
            
            # Attente avec vérification périodique du signal d'arrêt
            for _ in range(wait_time):
                if not running:
                    break
                time.sleep(1)
    
    logger.info("Programme terminé.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Programme interrompu par l'utilisateur")
    except Exception as e:
        logger.error(f"Erreur critique non gérée: {e}")
        sys.exit(1)
