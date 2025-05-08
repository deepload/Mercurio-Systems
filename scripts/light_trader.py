#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Version allégée du trader Mercurio - Résout les problèmes de mémoire
-------------------------------------------------------------------
Implémente uniquement la stratégie de moyenne mobile (sans ML)
pour éviter les problèmes de mémoire avec TensorFlow.
"""

import os
import sys
import time
import signal
import logging
import threading
from datetime import datetime

# Ajouter le répertoire parent au path pour pouvoir importer les modules du projet
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Configurer le logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'light_trader_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger('light_trader')

# Importer les services nécessaires (mais pas les stratégies ML lourdes)
try:
    from app.utils.env_loader import load_env_vars
    from app.services.market_data import MarketDataService
    from app.services.trading import TradingService
    from app.strategies.moving_average import MovingAverageStrategy
    from app.enums.trading_enums import TradingStrategy
    
    # Chargement des variables d'environnement
    load_env_vars()
    
    logger.info("Services de base chargés avec succès")
except Exception as e:
    logger.error(f"Erreur lors du chargement des services: {e}")
    sys.exit(1)

# Variables globales
running = True
market_data_service = None
trading_service = None
strategies = {}

def signal_handler(sig, frame):
    """Gestionnaire pour arrêter proprement le programme"""
    global running
    logger.info("Signal d'arrêt reçu, fermeture en cours...")
    running = False

def initialize_services():
    """Initialise les services nécessaires une seule fois"""
    global market_data_service, trading_service, strategies
    
    logger.info("Initialisation des services...")
    
    try:
        # Initialiser le service de données de marché (avec fallback si nécessaire)
        market_data_service = MarketDataService(use_sample_data=True)
        
        # Initialiser le service de trading (en mode paper)
        trading_service = TradingService(is_paper=True)
        
        # Créer une seule instance de la stratégie de moyenne mobile
        strategies[TradingStrategy.MOVING_AVERAGE] = MovingAverageStrategy(
            market_data_service=market_data_service,
            trading_service=trading_service
        )
        
        logger.info("Services initialisés avec succès")
        return True
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation des services: {e}")
        return False

def process_symbol(symbol):
    """Traiter un symbole avec la stratégie de moyenne mobile"""
    global market_data_service, trading_service, strategies
    
    try:
        logger.info(f"Traitement du symbole {symbol}")
        
        # Récupérer la stratégie
        strategy = strategies.get(TradingStrategy.MOVING_AVERAGE)
        if not strategy:
            logger.error(f"Stratégie non trouvée pour {symbol}")
            return False
        
        # Obtenir les dernières données pour ce symbole
        data = market_data_service.get_latest_data(symbol, interval='1m', limit=20)
        if data is None or data.empty:
            logger.warning(f"Aucune donnée disponible pour {symbol}")
            return False
        
        # Exécuter la stratégie
        signal = strategy.generate_signal(symbol, data)
        
        # Si nous avons un signal, exécuter l'ordre
        if signal:
            logger.info(f"Signal généré pour {symbol}: {signal}")
            # Ne pas exécuter réellement pour ne pas risquer de passer des ordres involontaires
            # trading_service.execute_order(symbol, signal, quantity=1)
        
        return True
    
    except Exception as e:
        logger.error(f"Erreur lors du traitement de {symbol}: {e}")
        return False

def process_symbol_batch(symbols, batch_size=5):
    """Traiter un lot de symboles en parallèle mais avec limite de mémoire"""
    
    if batch_size > 5:
        logger.warning(f"Batch size réduit à 5 pour éviter les problèmes de mémoire (valeur initiale: {batch_size})")
        batch_size = 5
    
    processed = 0
    
    for symbol in symbols[:batch_size]:
        try:
            success = process_symbol(symbol)
            if success:
                processed += 1
        except Exception as e:
            logger.error(f"Exception non gérée pour {symbol}: {e}")
    
    return processed

def main():
    global running
    
    # Enregistrer les gestionnaires de signal
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Démarrage du trader allégé Mercurio")
    
    # Initialiser les services (une seule fois)
    if not initialize_services():
        logger.error("Impossible d'initialiser les services, arrêt du programme")
        return
    
    # Liste des symboles à surveiller
    symbols = []
    
    # Vérifier si un fichier personnalisé existe
    custom_file = os.path.join(os.path.dirname(__file__), 'custom_stocks.txt')
    if os.path.exists(custom_file):
        try:
            with open(custom_file, 'r') as f:
                symbols = [line.strip() for line in f if line.strip()]
                logger.info(f"Utilisation de {len(symbols)} symboles personnalisés depuis {custom_file}")
        except Exception as e:
            logger.error(f"Erreur lors de la lecture du fichier personnalisé: {e}")
    
    # Si aucun symbole personnalisé, utiliser une liste par défaut
    if not symbols:
        symbols = ["AAPL", "MSFT", "AMZN", "GOOG", "META", "TSLA", "NVDA", "AMD", "INTC", "IBM"]
        logger.info(f"Utilisation de {len(symbols)} symboles par défaut")
    
    # Limiter à 50 symboles maximum
    symbols = symbols[:50]
    logger.info(f"Surveillance de {len(symbols)} symboles")
    
    # Nombre de symboles traités et statistiques
    processed_count = 0
    start_time = time.time()
    batch_size = 5  # Taille de lot réduite pour éviter les problèmes de mémoire
    
    try:
        # Boucle principale - vérifier toutes les secondes
        while running:
            cycle_start = time.time()
            
            # Déterminer les symboles à traiter ce cycle
            current_index = processed_count % len(symbols)
            current_batch = symbols[current_index:current_index + batch_size]
            
            # Si on atteint la fin de la liste, recommencer au début
            if len(current_batch) < batch_size:
                current_batch += symbols[:batch_size - len(current_batch)]
            
            # Traiter le lot
            batch_processed = process_symbol_batch(current_batch, batch_size)
            processed_count += batch_processed
            
            # Calculer le temps restant pour faire exactement 1 seconde par cycle
            cycle_time = time.time() - cycle_start
            wait_time = max(0, 1.0 - cycle_time)
            
            if wait_time > 0:
                time.sleep(wait_time)
            
            # Afficher des statistiques périodiquement
            if processed_count % 50 == 0:
                elapsed = time.time() - start_time
                rate = processed_count / elapsed if elapsed > 0 else 0
                logger.info(f"Statistiques: {processed_count} traitements, {rate:.2f} symboles/seconde")
            
    except KeyboardInterrupt:
        logger.info("Interruption du clavier détectée, arrêt...")
    except Exception as e:
        logger.error(f"Erreur dans la boucle principale: {e}")
    finally:
        total_time = time.time() - start_time
        logger.info(f"Fin du programme. {processed_count} traitements en {total_time:.2f} secondes")
        logger.info(f"Taux moyen: {processed_count/total_time:.2f} symboles/seconde")

if __name__ == "__main__":
    main()
