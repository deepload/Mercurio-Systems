#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test du gestionnaire de taux d'API
---------------------------------
Script pour tester le système de gestion des taux d'API et vérifier
qu'il fonctionne correctement sous charge élevée.
"""

import os
import sys
import time
import logging
import threading
import concurrent.futures
from datetime import datetime
from dotenv import load_dotenv

# Ajouter le répertoire parent au path pour pouvoir importer les modules du projet
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Configurer le logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api_rate_test.log')
    ]
)
logger = logging.getLogger('api_rate_test')

# Importer le gestionnaire de taux API et le moniteur
try:
    from scripts.api_rate_manager import rate_limited, APIRateManager
    from scripts.api_usage_monitor import APIUsageMonitor
    USE_RATE_MANAGER = True
except ImportError as e:
    logger.error(f"Erreur lors de l'importation des modules de gestion de taux: {e}")
    USE_RATE_MANAGER = False

# Importer l'API Alpaca
import alpaca_trade_api as tradeapi

# Charger les variables d'environnement
load_dotenv()

# Configuration globale
NUM_SYMBOLS = 50  # Nombre de symboles à traiter
NUM_CYCLES = 5    # Nombre de cycles de test
CYCLE_INTERVAL = 60  # Intervalle entre les cycles (secondes)

# Initialiser l'API Alpaca
def init_alpaca():
    """Initialiser le client API Alpaca"""
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
    
    # Initialiser le client API
    api = tradeapi.REST(
        key_id=api_key,
        secret_key=api_secret,
        base_url=base_url,
        api_version='v2'
    )
    
    return api

# Version standard - sans décorateur rate_limited
def get_asset_standard(api, symbol):
    """Obtenir les informations sur un actif - version standard"""
    return api.get_asset(symbol)

# Version avec rate_limited
@rate_limited
def get_asset_rate_limited(api, symbol):
    """Obtenir les informations sur un actif - version rate_limited"""
    return api.get_asset(symbol)

# Fonction de test pour un symbole unique
def process_symbol(api, symbol, use_rate_limiter=True):
    """Traiter un symbole - simule les opérations sur un actif"""
    start_time = time.time()
    try:
        # Obtenir les informations sur l'actif
        if use_rate_limiter:
            asset = get_asset_rate_limited(api, symbol)
        else:
            asset = get_asset_standard(api, symbol)
        
        # Simuler d'autres opérations
        time.sleep(0.05)  # Simuler un traitement
        
        processing_time = time.time() - start_time
        logger.info(f"Symbole {symbol} traité en {processing_time:.3f}s - Tradable: {asset.tradable}")
        
        return {
            "symbol": symbol,
            "tradable": asset.tradable,
            "processing_time": processing_time
        }
    except Exception as e:
        logger.error(f"Erreur lors du traitement de {symbol}: {e}")
        return {
            "symbol": symbol,
            "error": str(e),
            "processing_time": time.time() - start_time
        }

# Fonction pour traiter tous les symboles en parallèle
def process_symbols_parallel(api, symbols, use_rate_limiter=True, max_workers=10):
    """Traiter une liste de symboles en parallèle"""
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(process_symbol, api, symbol, use_rate_limiter): symbol
            for symbol in symbols
        }
        
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Exception pour {symbol}: {e}")
                results.append({
                    "symbol": symbol,
                    "error": str(e)
                })
    
    return results

# Fonction pour traiter tous les symboles séquentiellement
def process_symbols_sequential(api, symbols, use_rate_limiter=True):
    """Traiter une liste de symboles séquentiellement"""
    results = []
    for symbol in symbols:
        result = process_symbol(api, symbol, use_rate_limiter)
        results.append(result)
    
    return results

# Fonction pour exécuter un cycle complet
def run_cycle(api, symbols, cycle_num, use_rate_limiter=True, parallel=True):
    """Exécuter un cycle complet de traitement"""
    logger.info(f"Démarrage du cycle {cycle_num}/{NUM_CYCLES}")
    start_time = time.time()
    
    if parallel:
        results = process_symbols_parallel(api, symbols, use_rate_limiter)
    else:
        results = process_symbols_sequential(api, symbols, use_rate_limiter)
    
    total_time = time.time() - start_time
    success_count = len([r for r in results if "error" not in r])
    error_count = len(results) - success_count
    
    logger.info(f"Cycle {cycle_num} terminé en {total_time:.2f}s - Succès: {success_count}, Erreurs: {error_count}")
    
    # Afficher les statistiques si on utilise le gestionnaire de taux
    if use_rate_limiter and USE_RATE_MANAGER:
        rate_manager = APIRateManager()
        stats = rate_manager.get_usage_stats()
        logger.info(f"Statistiques d'utilisation API:")
        logger.info(f"  - Taux par seconde: {stats['second_rate']}/{stats['second_limit']} ({stats['second_percent']:.1f}%)")
        logger.info(f"  - Taux par minute: {stats['minute_rate']}/{stats['minute_limit']} ({stats['minute_percent']:.1f}%)")
    
    return results, total_time

# Fonction principale
def main():
    """Fonction principale de test"""
    logger.info("Démarrage du test du gestionnaire de taux API")
    
    # Vérifier si le gestionnaire de taux est disponible
    if not USE_RATE_MANAGER:
        logger.warning("Le gestionnaire de taux API n'est pas disponible, le test sera limité")
    
    # Initialiser l'API Alpaca
    api = init_alpaca()
    
    # Obtenir une liste de symboles
    try:
        assets = api.list_assets(status='active', asset_class='us_equity')
        symbols = [asset.symbol for asset in assets if asset.tradable][:NUM_SYMBOLS]
        logger.info(f"Récupéré {len(symbols)} symboles pour le test")
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des symboles: {e}")
        # Utiliser une liste de symboles de secours
        symbols = [f"TEST{i}" for i in range(NUM_SYMBOLS)]
        symbols = ["AAPL", "MSFT", "AMZN", "GOOG", "GOOGL", "META", "TSLA", "NVDA", "AMD", "INTC",
                  "CSCO", "ADBE", "PYPL", "NFLX", "CMCSA", "PEP", "COST", "AVGO", "TXN", "QCOM",
                  "SBUX", "TMUS", "CHTR", "MDLZ", "ADSK", "MAR", "AMAT", "ISRG", "MU", "BKNG",
                  "CSX", "INCY", "ADP", "ATVI", "VRTX", "ADI", "ROST", "MNST", "KLAC", "BIIB",
                  "LRCX", "WBA", "ILMN", "LULU", "WDAY", "EXC", "CTSH", "ORLY", "EA", "CTAS"][:NUM_SYMBOLS]
    
    # Exécuter des cycles avec le gestionnaire de taux
    logger.info("\n--- Test avec gestionnaire de taux ---")
    with_limiter_times = []
    for i in range(NUM_CYCLES):
        _, cycle_time = run_cycle(api, symbols, i+1, use_rate_limiter=True)
        with_limiter_times.append(cycle_time)
        
        # Attendre l'intervalle de cycle
        if i < NUM_CYCLES - 1:
            wait_time = CYCLE_INTERVAL - cycle_time
            if wait_time > 0:
                logger.info(f"Attente de {wait_time:.1f}s avant le prochain cycle")
                time.sleep(wait_time)
    
    # Exécuter des cycles sans le gestionnaire de taux (si l'utilisateur le veut)
    run_without_limiter = False
    if run_without_limiter:
        logger.info("\n--- Test sans gestionnaire de taux ---")
        without_limiter_times = []
        for i in range(NUM_CYCLES):
            try:
                _, cycle_time = run_cycle(api, symbols, i+1, use_rate_limiter=False)
                without_limiter_times.append(cycle_time)
                
                # Attendre l'intervalle de cycle
                if i < NUM_CYCLES - 1:
                    wait_time = CYCLE_INTERVAL - cycle_time
                    if wait_time > 0:
                        logger.info(f"Attente de {wait_time:.1f}s avant le prochain cycle")
                        time.sleep(wait_time)
            except Exception as e:
                logger.error(f"Erreur lors du cycle sans gestionnaire de taux: {e}")
                break
    
    # Afficher le résumé
    logger.info("\n=== Résumé du test ===")
    logger.info(f"Nombre de symboles traités: {NUM_SYMBOLS}")
    logger.info(f"Nombre de cycles exécutés: {NUM_CYCLES}")
    logger.info(f"Temps moyen par cycle avec gestionnaire de taux: {sum(with_limiter_times)/len(with_limiter_times):.2f}s")
    
    if run_without_limiter and without_limiter_times:
        logger.info(f"Temps moyen par cycle sans gestionnaire de taux: {sum(without_limiter_times)/len(without_limiter_times):.2f}s")
    
    logger.info("Test terminé")

if __name__ == "__main__":
    main()
