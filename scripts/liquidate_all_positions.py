#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Liquidate All Positions Script
------------------------------
Script pour liquider toutes les positions ouvertes sur Alpaca.
À utiliser après avoir arrêté un script de trading.
"""

import os
import sys
import logging
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("position_liquidator")

# Ajouter le répertoire parent au path pour pouvoir importer les modules du projet
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

def main():
    # Charger les variables d'environnement
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
    
    # Initialiser l'API Alpaca
    try:
        api = tradeapi.REST(
            key_id=api_key,
            secret_key=api_secret,
            base_url=base_url
        )
        
        logger.info(f"Connecté à Alpaca en mode {alpaca_mode.upper()}")
        
        # Récupérer l'état du compte
        account = api.get_account()
        logger.info(f"Compte Alpaca: {account.id}")
        logger.info(f"Valeur actuelle du portefeuille: ${float(account.equity):.2f}")
        
        # Récupérer toutes les positions ouvertes
        positions = api.list_positions()
        logger.info(f"Nombre de positions ouvertes: {len(positions)}")
        
        if len(positions) == 0:
            logger.info("Aucune position à liquider.")
            return
        
        # Afficher les positions avant liquidation
        logger.info("Positions avant liquidation:")
        for position in positions:
            market_value = float(position.market_value)
            unrealized_pl = float(position.unrealized_pl)
            unrealized_plpc = float(position.unrealized_plpc) * 100
            logger.info(f"  {position.symbol}: {position.qty} actions @ ${float(position.current_price):.2f} - Valeur: ${market_value:.2f} - P/L: ${unrealized_pl:.2f} ({unrealized_plpc:.2f}%)")
        
        # Demander confirmation
        confirm = input("\nVoulez-vous vraiment liquider toutes ces positions? (y/n): ")
        if confirm.lower() != 'y':
            logger.info("Opération annulée.")
            return
        
        # Liquider toutes les positions
        logger.info("Liquidation de toutes les positions...")
        
        # Méthode 1: Liquider position par position
        for position in positions:
            try:
                logger.info(f"Liquidation de {position.symbol}...")
                api.close_position(position.symbol)
                logger.info(f"Position {position.symbol} liquidée avec succès!")
            except Exception as e:
                logger.error(f"Erreur lors de la liquidation de {position.symbol}: {e}")
        
        # Méthode 2 (alternative): Liquider toutes les positions d'un coup
        try:
            logger.info("Liquidation de toutes les positions d'un coup...")
            api.close_all_positions()
            logger.info("Toutes les positions ont été liquidées!")
        except Exception as e:
            logger.error(f"Erreur lors de la liquidation groupée: {e}")
        
        # Vérifier s'il reste des positions
        remaining_positions = api.list_positions()
        if len(remaining_positions) > 0:
            logger.warning(f"Il reste encore {len(remaining_positions)} positions non liquidées:")
            for position in remaining_positions:
                logger.warning(f"  {position.symbol}: {position.qty} actions")
        else:
            logger.info("Toutes les positions ont été liquidées avec succès!")
        
        # Récupérer l'état du compte après liquidation
        account = api.get_account()
        logger.info(f"Valeur finale du portefeuille: ${float(account.equity):.2f}")
        logger.info(f"Cash disponible: ${float(account.cash):.2f}")
        
    except Exception as e:
        logger.error(f"Erreur lors de la connexion à Alpaca ou de la liquidation des positions: {e}")

if __name__ == "__main__":
    main()
