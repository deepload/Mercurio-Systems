#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trading à haute fréquence - Vérification chaque seconde
------------------------------------------------------
Script pour vérifier les conditions de marché chaque seconde
au lieu d'attendre 660 secondes entre les cycles.
"""

import os
import sys
import time
import signal
import logging
import threading
from datetime import datetime
import subprocess

# Ajouter le répertoire parent au path pour pouvoir importer les modules du projet
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Configurer le logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'high_frequency_trading_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger('high_frequency_trading')

# Variable globale pour contrôler l'arrêt
running = True

# Gestionnaire de signal pour arrêter proprement
def signal_handler(sig, frame):
    global running
    logger.info("Signal d'arrêt reçu, fermeture en cours...")
    running = False

# Fonction pour exécuter le traitement pour un symbole individuel
def process_single_symbol(symbol, use_api_rate_manager=True):
    """Traiter un symbole spécifique"""
    try:
        # Créer un fichier custom_symbols.txt standard dans le dossier scripts
        custom_symbols_file = os.path.join(project_root, "scripts", "custom_symbols.txt")
        with open(custom_symbols_file, 'w') as f:
            f.write(symbol)
            
        # Construire la commande pour exécuter le traitement d'un seul symbole
        cmd = [
            sys.executable,
            os.path.join(project_root, "scripts", "run_stock_daytrader_all.py"),
            "--strategy", "moving_average",
            "--max-symbols", "1",
            "--filter", "top_volume",
            "--use-custom-symbols",
            "--position-size", "0.01",  # Petit montant pour les tests
            "--cycle-interval", "1",    # Cycle interval minimal (1 seconde)
            "--duration", "market_hours"  # Utiliser duration au lieu de session-duration
        ]
        
        if use_api_rate_manager:
            cmd.append("--api-level")
            cmd.append("1")  # Utiliser un niveau d'API plus bas pour économiser les ressources
        
        logger.info(f"Traitement du symbole {symbol}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"Symbole {symbol} traité avec succès")
        else:
            logger.error(f"Erreur lors du traitement de {symbol}: {result.stderr}")
    
    except Exception as e:
        logger.error(f"Exception lors du traitement de {symbol}: {e}")
    finally:
        # Nettoyer le fichier personnalisé n'est plus nécessaire car nous utilisons un fichier standard
        pass

# Fonction pour traiter un lot de symboles par secondes
def process_symbol_batch(symbols, batch_size=1):
    """Traiter un lot de symboles en parallèle"""
    threads = []
    for symbol in symbols[:batch_size]:
        thread = threading.Thread(target=process_single_symbol, args=(symbol,))
        thread.daemon = True
        thread.start()
        threads.append(thread)
    
    # Attendre que tous les threads terminent
    for thread in threads:
        thread.join(timeout=5.0)  # Augmenter le timeout pour laisser plus de temps au thread de terminer
    
    return len(threads)

# Fonction principale
def main():
    global running
    
    # Enregistrer les gestionnaires de signal
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Démarrage du trading à haute fréquence (vérification chaque seconde)")
    
    # Liste des symboles à surveiller
    symbols = ["AAPL", "MSFT", "AMZN", "GOOG", "META", "TSLA", "NVDA", "AMD", "INTC",
               "CSCO", "ADBE", "PYPL", "NFLX", "CMCSA", "PEP", "COST", "AVGO", "TXN", 
               "QCOM", "SBUX", "TMUS", "CHTR", "MDLZ", "ADSK", "MAR", "AMAT", "ISRG", 
               "MU", "BKNG", "CSX", "INCY", "ADP", "ATVI", "VRTX", "ADI", "ROST", 
               "MNST", "KLAC", "BIIB", "LRCX", "WBA", "ILMN", "LULU", "WDAY", "EXC", 
               "CTSH", "ORLY", "EA", "CTAS"]
    
    # Calculer combien de symboles nous pouvons traiter par seconde
    # en respectant les limites d'API et la mémoire disponible
    # Réduire la taille du lot pour éviter les problèmes de mémoire
    batch_size = 2  # Réduire davantage le nombre de traitements simultanés pour éviter les erreurs de mémoire
    
    # Vérifier si un fichier personnalisé existe
    custom_file = os.path.join(os.path.dirname(__file__), 'custom_stocks.txt')
    if os.path.exists(custom_file):
        try:
            with open(custom_file, 'r') as f:
                custom_symbols = [line.strip() for line in f if line.strip()]
                if custom_symbols:
                    symbols = custom_symbols
                    logger.info(f"Utilisation de {len(symbols)} symboles personnalisés depuis {custom_file}")
        except Exception as e:
            logger.error(f"Erreur lors de la lecture du fichier de symboles personnalisés: {e}")

    # Limiter à 50 symboles maximum
    symbols = symbols[:50]
    logger.info(f"Surveillance de {len(symbols)} symboles")
    
    # Nombre de symboles traités
    processed_count = 0
    start_time = time.time()
    
    try:
        # Boucle principale - vérifier chaque seconde
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
            if processed_count % 100 == 0:
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
