#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script simple de test pour la fonctionnalité de liquidation immédiate
"""

import os
import sys
import time
import logging
import signal
from datetime import datetime
import traceback
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

# Configuration du logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("test_liquidation")

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

# Fonction pour nettoyer les ressources
def cleanup_resources():
    """Nettoyer les ressources avant de quitter"""
    logger.info("Nettoyage des ressources...")
    logger.info("Nettoyage terminé")

def main():
    """Fonction principale de test"""
    # Charger les variables d'environnement
    load_dotenv()
    
    # Déterminer le mode (paper ou live)
    alpaca_mode = os.getenv("ALPACA_MODE", "paper").lower()
    
    # Forcer le mode live si demandé en ligne de commande
    import argparse
    parser = argparse.ArgumentParser(description="Test de liquidation")
    parser.add_argument("--live", action="store_true", help="Utiliser le mode live au lieu du mode paper")
    parser.add_argument("--paper", action="store_true", help="Forcer le mode paper")
    args = parser.parse_args()
    
    if args.live:
        alpaca_mode = "live"
        logger.info("Mode LIVE forcé en ligne de commande")
    elif args.paper:
        alpaca_mode = "paper"
        logger.info("Mode PAPER forcé en ligne de commande")
    
    if alpaca_mode == "live":
        api_key = os.getenv("ALPACA_LIVE_KEY")
        api_secret = os.getenv("ALPACA_LIVE_SECRET")
        base_url = os.getenv("ALPACA_LIVE_URL", "https://api.alpaca.markets")
        logger.info("Utilisation du mode LIVE TRADING - Utilisation de vrai argent!")
    else:  # mode paper par défaut
        api_key = os.getenv("ALPACA_PAPER_KEY")
        api_secret = os.getenv("ALPACA_PAPER_SECRET")
        base_url = os.getenv("ALPACA_PAPER_URL", "https://paper-api.alpaca.markets")
        logger.info("Utilisation du mode PAPER TRADING (simulation)")
    
    # Note: le mode est explicité pour l'utilisateur
    if alpaca_mode == "live":
        confirm = input("\n⚠️  ATTENTION: VOUS ÊTES EN MODE LIVE TRADING! Voulez-vous continuer? (y/n): ")
        if confirm.lower() != 'y':
            logger.info("Test annulé par l'utilisateur.")
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
    
    # Vérifier la connexion à Alpaca
    try:
        account = api.get_account()
        logger.info(f"Connecté à Alpaca (mode {alpaca_mode.upper()})")
        logger.info(f"Compte: {account.id}")
        logger.info(f"Solde: ${float(account.equity):.2f}")
        
        # Lister les positions
        positions = api.list_positions()
        logger.info(f"Positions ouvertes: {len(positions)}")
        
        for position in positions:
            symbol = position.symbol
            qty = position.qty
            value = position.market_value
            logger.info(f"  {symbol}: {qty} @ ${value}")
        
    except Exception as e:
        logger.error(f"Erreur lors de la connexion à Alpaca: {e}")
        return
    
    # Simulation d'une boucle de trading
    logger.info("")
    logger.info("=" * 60)
    logger.info("SIMULATEUR DE TRADING - TEST GRACEFUL EXIT")
    logger.info("=" * 60)
    logger.info("Ce programme va simuler un trader en fonctionnement.")
    logger.info("Appuyez sur Ctrl+C pour déclencher l'arrêt propre.")
    logger.info("Vous aurez alors la possibilité de:")
    logger.info("- Appuyer sur 'K' pour CONSERVER vos positions")
    logger.info("- Appuyer sur 'L' pour LIQUIDER immédiatement vos positions")
    logger.info("- Attendre le délai pour la LIQUIDATION automatique")
    logger.info("=" * 60)
    logger.info("")
    
    # Compteur pour afficher une activité
    counter = 0
    
    # Boucle principale
    while is_running():
        try:
            counter += 1
            current_time = datetime.now().strftime("%H:%M:%S")
            logger.info(f"Cycle #{counter} à {current_time} - Trader en fonctionnement...")
            
            # Simuler une action de trading toutes les 5 secondes
            for _ in range(5):
                if not is_running():
                    break
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Erreur pendant la simulation: {e}")
            logger.error(traceback.format_exc())
            break
    
    logger.info("Simulation terminée")

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
        logger.info("Démarrage du test de liquidation. Appuyez sur Ctrl+C pour arrêter proprement.")
        main()
    except KeyboardInterrupt:
        logger.info("Interruption utilisateur détectée")
    finally:
        if not USE_GRACEFUL_EXIT:
            cleanup_resources()
        logger.info("Test terminé")
