#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trader Direct Mercurio - Optimisé pour 50 symboles
--------------------------------------------------
Appelle directement les services et stratégies, sans créer 
de sous-processus, évitant ainsi les problèmes de mémoire
et d'arguments incompatibles.
"""

import os
import sys
import time
import signal
import logging
import threading
from datetime import datetime
import importlib.util

# Ajouter le répertoire parent au path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Configurer le logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'direct_trader_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger('direct_trader')

# Variable globale pour contrôler l'arrêt
running = True

def signal_handler(sig, frame):
    """Gestionnaire pour arrêter proprement le programme"""
    global running
    logger.info("Signal d'arrêt reçu, fermeture en cours...")
    running = False

def read_custom_symbols():
    """Lit la liste de symboles personnalisés depuis le fichier"""
    symbols = []
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
        symbols = ["AAPL", "MSFT", "AMZN", "GOOG", "META", "TSLA", "NVDA", "AMD"]
        logger.info(f"Utilisation de {len(symbols)} symboles par défaut")
    
    # Limiter à 50 symboles maximum
    return symbols[:50]

def import_services():
    """Importe les services nécessaires directement"""
    try:
        # Charger les modules nécessaires une seule fois
        from app.utils.env_loader import load_environment_variables
        from app.enums.trading_enums import TradingStrategy
        from app.services.market_data import MarketDataService
        from app.services.trading import TradingService
        
        # On importe uniquement la stratégie de moyenne mobile pour éviter les problèmes de mémoire
        from app.strategies.moving_average import MovingAverageStrategy
        
        # Charger les variables d'environnement
        load_environment_variables()
        
        logger.info("Services importés avec succès")
        
        return {
            'market_data': MarketDataService(use_sample_data=True),
            'trading': TradingService(is_paper=True),
            'strategies': {
                TradingStrategy.MOVING_AVERAGE: MovingAverageStrategy
            },
            'enums': {
                'TradingStrategy': TradingStrategy
            }
        }
    except Exception as e:
        logger.error(f"Erreur lors de l'importation des services: {e}")
        return None

def process_symbol(symbol, services):
    """Traite un symbole en utilisant directement les services"""
    try:
        logger.info(f"Traitement du symbole {symbol}")
        
        # Extraire les services
        market_data = services['market_data']
        trading_service = services['trading']
        TradingStrategy = services['enums']['TradingStrategy']
        
        # Créer une instance de la stratégie
        strategy_class = services['strategies'][TradingStrategy.MOVING_AVERAGE]
        strategy = strategy_class(
            market_data_service=market_data,
            trading_service=trading_service
        )
        
        # Obtenir les dernières données
        data = market_data.get_latest_data(symbol, interval='1m', limit=20)
        if data is None or data.empty:
            logger.warning(f"Aucune donnée disponible pour {symbol}")
            return False
        
        # Générer un signal
        signal = strategy.generate_signal(symbol, data)
        
        if signal:
            logger.info(f"Signal généré pour {symbol}: {signal}")
            # Ne pas exécuter les ordres en mode test
            # trading_service.execute_order(symbol, signal, quantity=1)
        
        return True
    except Exception as e:
        logger.error(f"Erreur lors du traitement de {symbol}: {e}")
        return False

def main():
    """Fonction principale"""
    global running
    
    # Enregistrer les gestionnaires de signal
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Démarrage du trader direct Mercurio")
    
    # Importer les services (UNE SEULE FOIS pour tout le programme)
    services = import_services()
    if not services:
        logger.error("Impossible d'importer les services nécessaires, arrêt du programme")
        return
    
    # Lire les symboles personnalisés
    symbols = read_custom_symbols()
    logger.info(f"Trader configuré pour surveiller {len(symbols)} symboles")
    
    # Statistiques
    processed_count = 0
    start_time = time.time()
    
    try:
        # Boucle principale
        while running:
            cycle_start = time.time()
            
            # Traiter les symboles un par un (pas de parallélisme pour éviter les problèmes de mémoire)
            # Nous prenons 5 symboles par cycle
            current_index = (processed_count // 5) % (len(symbols) // 5 + 1) * 5
            current_symbols = symbols[current_index:current_index + 5]
            
            if not current_symbols:
                current_symbols = symbols[:5]
            
            logger.info(f"Cycle de traitement pour les symboles: {', '.join(current_symbols)}")
            
            for symbol in current_symbols:
                if process_symbol(symbol, services):
                    processed_count += 1
            
            # Attendre pour compléter une seconde
            cycle_time = time.time() - cycle_start
            wait_time = max(0, 1.0 - cycle_time)
            
            if wait_time > 0:
                time.sleep(wait_time)
            
            # Afficher des statistiques
            if processed_count % 25 == 0 and processed_count > 0:
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
