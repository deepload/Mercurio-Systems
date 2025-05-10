#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script intégré pour Mercurio AI
-------------------------------
Exécute le trading sur crypto 24/24 et actions pendant les heures de marché.
"""

import os
import sys
import time
import signal
import logging
import subprocess
import argparse
from datetime import datetime, timedelta
import pytz
from dotenv import load_dotenv

# Ajouter le répertoire parent au path pour pouvoir importer les modules du projet
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"integrated_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("integrated_trader")

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

# Variable pour stocker les process en cours
active_processes = []

def get_market_status():
    """Vérifier si le marché des actions est ouvert"""
    # Chargement des variables d'environnement pour les clés API
    load_dotenv()
    
    # Dans un environnement réel, vous utiliseriez l'API Alpaca pour vérifier l'état du marché
    # Pour cette démo, nous utilisons une approche simplifiée basée sur l'heure
    
    # Timezone de New York (Wall Street)
    nyc_timezone = pytz.timezone('America/New_York')
    current_time = datetime.now(nyc_timezone)
    
    # Vérifier si c'est un jour de semaine (lundi=0, dimanche=6)
    is_weekday = current_time.weekday() < 5
    
    # Vérifier si c'est entre 9h30 et 16h00 ET
    market_open_time = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close_time = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
    
    is_market_hours = market_open_time <= current_time <= market_close_time and is_weekday
    
    # Pour les heures de pré-marché (4h00 à 9h30)
    premarket_open_time = current_time.replace(hour=4, minute=0, second=0, microsecond=0)
    is_premarket = premarket_open_time <= current_time < market_open_time and is_weekday
    
    # Pour les heures après-marché (16h00 à 20h00)
    after_hours_close_time = current_time.replace(hour=20, minute=0, second=0, microsecond=0)
    is_after_hours = market_close_time < current_time <= after_hours_close_time and is_weekday
    
    # Retourner l'état complet
    return {
        "is_open": is_market_hours,
        "is_premarket": is_premarket,
        "is_after_hours": is_after_hours,
        "can_trade_stocks": is_market_hours or is_premarket or is_after_hours,
        "can_trade_crypto": True  # Crypto est toujours disponible 24/7
    }

def run_stock_trader(strategy, args=None):
    """Exécuter le trader d'actions"""
    logger.info(f"Démarrage du trader d'actions avec stratégie '{strategy}'")
    
    cmd = [sys.executable, 
           os.path.join(project_root, "scripts", "test_liquidation.py"),
           "--paper" if args.paper else "--live"]
    
    logger.info(f"Commande: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(cmd, 
                                  cwd=project_root,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,
                                  text=True)
        
        # Ajouter à la liste des processus actifs
        global active_processes
        active_processes.append(process)
        
        return process
    except Exception as e:
        logger.error(f"Erreur lors du démarrage du trader d'actions: {e}")
        return None

def run_crypto_trader(strategy, args=None):
    """Exécuter le trader de crypto"""
    logger.info(f"Démarrage du trader de crypto avec stratégie '{strategy}'")
    
    cmd = [sys.executable, 
           os.path.join(project_root, "scripts", "test_liquidation.py"),
           "--paper" if args.paper else "--live"]
    
    logger.info(f"Commande: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(cmd, 
                                  cwd=project_root,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,
                                  text=True)
        
        # Ajouter à la liste des processus actifs
        global active_processes
        active_processes.append(process)
        
        return process
    except Exception as e:
        logger.error(f"Erreur lors du démarrage du trader de crypto: {e}")
        return None

def check_process_output(process):
    """Lire et logger la sortie d'un processus"""
    if process and process.poll() is None:  # Si le processus est en cours d'exécution
        # Lire la sortie sans bloquer
        while True:
            output = process.stdout.readline()
            if output:
                logger.info(output.strip())
            else:
                break

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
        cmd = [sys.executable, liquidation_script, "--force", "--yes", "--auto-progressive"]
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

def cleanup_resources():
    """Nettoyer les ressources et arrêter tous les processus"""
    logger.info("Nettoyage des ressources...")
    
    # Arrêter tous les processus actifs
    global active_processes
    for process in active_processes:
        if process and process.poll() is None:  # Si le processus est en cours d'exécution
            logger.info(f"Arrêt du processus {process.pid}...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning(f"Le processus {process.pid} ne répond pas, arrêt forcé...")
                process.kill()
    
    active_processes = []
    logger.info("Nettoyage terminé")

def main():
    """Fonction principale"""
    # Parser les arguments
    parser = argparse.ArgumentParser(description="Script de trading intégré pour actions et crypto 24/7")
    parser.add_argument("--stock-strategy", type=str, default="llm_v2", help="Stratégie pour le trading d'actions")
    parser.add_argument("--crypto-strategy", type=str, default="llm_v2", help="Stratégie pour le trading de crypto")
    parser.add_argument("--no-stock", action="store_true", help="Désactiver le trading d'actions")
    parser.add_argument("--no-crypto", action="store_true", help="Désactiver le trading de crypto")
    parser.add_argument("--paper", action="store_true", help="Utiliser le mode paper trading (par défaut)")
    parser.add_argument("--live", action="store_true", help="Utiliser le mode live trading (attention: risque réel)")
    args = parser.parse_args()
    
    # Vérifier et avertir pour le mode live
    if args.live:
        logger.warning("⚠️ MODE LIVE TRADING ACTIVÉ - ATTENTION AUX RISQUES FINANCIERS! ⚠️")
        confirm = input("Êtes-vous sûr de vouloir utiliser de l'argent réel? (y/n): ")
        if confirm.lower() != 'y':
            logger.info("Opération annulée.")
            return
    
    # Processus actifs pour le trading
    stock_process = None
    crypto_process = None
    
    # Boucle principale
    try:
        while is_running():
            # Vérifier l'état du marché
            market_status = get_market_status()
            
            # Gestion du trader d'actions
            if not args.no_stock and market_status["can_trade_stocks"]:
                if stock_process is None or stock_process.poll() is not None:
                    # Démarrer ou redémarrer le trader d'actions
                    stock_process = run_stock_trader(args.stock_strategy, args)
                else:
                    # Vérifier la sortie du trader d'actions
                    check_process_output(stock_process)
            elif stock_process and stock_process.poll() is None and not market_status["can_trade_stocks"]:
                # Arrêter le trader d'actions en dehors des heures de trading
                logger.info("Arrêt du trader d'actions (hors heures de marché)...")
                stock_process.terminate()
                try:
                    stock_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    stock_process.kill()
                stock_process = None
            
            # Gestion du trader de crypto
            if not args.no_crypto:
                if crypto_process is None or crypto_process.poll() is not None:
                    # Démarrer ou redémarrer le trader de crypto
                    crypto_process = run_crypto_trader(args.crypto_strategy, args)
                else:
                    # Vérifier la sortie du trader de crypto
                    check_process_output(crypto_process)
            
            # Attendre avant la prochaine vérification
            for _ in range(30):  # Vérification toutes les 30 secondes
                if not is_running():
                    break
                time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Interruption utilisateur détectée")
    except Exception as e:
        logger.error(f"Erreur dans la boucle principale: {e}")
    finally:
        cleanup_resources()
        logger.info("Script intégré terminé")

if __name__ == "__main__":
    # Enregistrement des fonctions de nettoyage pour l'utilitaire d'arrêt propre
    if USE_GRACEFUL_EXIT:
        register_cleanup(cleanup_resources)
        register_liquidation_handler(liquidate_positions)
    else:
        # Enregistrement du gestionnaire de signal pour arrêt propre
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("=" * 60)
    logger.info("DÉMARRAGE DU SCRIPT INTÉGRÉ MERCURIO")
    logger.info("=" * 60)
    logger.info("Ce script va exécuter le trading sur:")
    logger.info("- Crypto: 24 heures sur 24, 7 jours sur 7")
    logger.info("- Actions: pendant les heures de marché (9h30-16h00 ET)")
    logger.info("")
    logger.info("Appuyez sur Ctrl+C pour arrêter proprement.")
    logger.info("=" * 60)
    
    main()
