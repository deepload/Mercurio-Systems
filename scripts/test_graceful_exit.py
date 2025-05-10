#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test Graceful Exit Script
-------------------------
Script simple pour tester la fonctionnalité d'arrêt propre avec l'option de liquidation.
"""

import os
import sys
import time
import logging
import signal
import subprocess
from datetime import datetime

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_graceful_exit")

# Ajouter le répertoire parent au path pour pouvoir importer les modules du projet
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Importation de l'utilitaire d'arrêt propre
try:
    from scripts.graceful_exit import is_running, register_thread, register_cleanup, register_liquidation_handler
    USE_GRACEFUL_EXIT = True
    logger.info("Utilitaire d'arrêt propre chargé avec succès")
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

def main():
    """Fonction principale pour tester l'arrêt propre"""
    logger.info("=" * 60)
    logger.info("TEST DE L'ARRÊT PROPRE AVEC OPTION DE LIQUIDATION")
    logger.info("=" * 60)
    
    logger.info("Ce script va simuler un trader en cours d'exécution.")
    logger.info("Appuyez sur Ctrl+C pour déclencher l'arrêt propre.")
    logger.info("Vous aurez alors 120 secondes pour choisir:")
    logger.info("- Appuyez sur 'K' pour CONSERVER vos positions")
    logger.info("- Appuyez sur 'L' pour LIQUIDER immédiatement vos positions")
    logger.info("- Attendez 120 secondes pour les LIQUIDER automatiquement")
    logger.info("=" * 60)
    
    # Simulation d'un trader en cours d'exécution
    counter = 0
    while is_running():
        counter += 1
        logger.info(f"Trader en cours d'exécution... Cycle #{counter}")
        
        # Simuler une action toutes les 5 secondes
        for i in range(5):
            if not is_running():
                break
            time.sleep(1)
    
    logger.info("Trader arrêté proprement")

if __name__ == "__main__":
    # Enregistrement des fonctions de nettoyage pour l'utilitaire d'arrêt propre
    if USE_GRACEFUL_EXIT:
        register_cleanup(cleanup_resources)
        register_liquidation_handler(liquidate_positions)
    else:
        # Enregistrement du gestionnaire de signal pour arrêt propre
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interruption utilisateur détectée")
    finally:
        if not USE_GRACEFUL_EXIT:
            cleanup_resources()
        logger.info("Test terminé")
