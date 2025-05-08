#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de lancement pour trader avec 50 actions
-----------------------------------------------
Ce script lance le day trader avec une configuration optimisée
pour traiter 50 actions simultanément.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configurer le logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("50_stocks_launcher")

# Liste personnalisée de 50 actions populaires et liquides
CUSTOM_STOCKS_50 = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "AMD", "INTC", "IBM",
    # Finance
    "JPM", "BAC", "WFC", "C", "GS", "MS", "V", "MA", "PYPL", "AXP",
    # Healthcare
    "JNJ", "PFE", "MRK", "ABBV", "UNH", "CVS", "MRNA", "BIIB", "AMGN", "GILD",
    # Telecom & Media
    "T", "VZ", "CMCSA", "NFLX", "DIS", "CHTR", "TMUS", "DISH", "ROKU", "SPOT",
    # Retail & Consumer
    "WMT", "TGT", "HD", "LOW", "COST", "MCD", "SBUX", "NKE", "KO", "PEP"
]

def main():
    """Fonction principale pour lancer le script de day trading avec 50 actions"""
    
    # Vérifier si le script principal existe
    script_path = Path(__file__).parent / 'run_stock_daytrader_all.py'
    if not script_path.exists():
        logger.error(f"Script principal introuvable: {script_path}")
        return
    
    # Enregistrer la liste personnalisée dans un fichier temporaire
    custom_list_path = Path(__file__).parent / '../data/custom_stocks_50.txt'
    os.makedirs(custom_list_path.parent, exist_ok=True)
    
    with open(custom_list_path, 'w') as f:
        for stock in CUSTOM_STOCKS_50:
            f.write(f"{stock}\n")
    
    logger.info(f"Liste personnalisée de 50 actions enregistrée dans {custom_list_path}")
    
    # Construit la commande complète
    daytrader_command = [
        sys.executable,
        os.path.join(script_path.parent, 'run_stock_daytrader_all.py'),
        '--strategy', 'moving_average_ml',
        '--filter', 'top_volume',
        '--max-symbols', '50',
        '--api-level', '3',
        '--use-threads',
        '--use-custom-symbols',
        '--duration', 'market_hours',
        '--market-check-interval', '1',
        '--cycle-interval', '60'  # Intervalle fixe de 60 secondes (1 minute) entre les cycles
    ]
    
    # Modifier l'environnement pour passer la liste personnalisée
    env = os.environ.copy()
    env["MERCURIO_CUSTOM_STOCKS"] = str(custom_list_path)
    
    # Lancer le script principal
    logger.info("Lancement du day trader avec 50 actions...")
    logger.info(f"Commande: {' '.join(daytrader_command)}")
    
    try:
        process = subprocess.Popen(daytrader_command, env=env)
        process.wait()
    except KeyboardInterrupt:
        logger.info("Arrêt demandé par l'utilisateur")
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du script: {e}")
    
    logger.info("Fin du programme")

if __name__ == "__main__":
    main()
