#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour convertir USD en USDT sur Alpaca
"""

import sys
import os
import argparse
import logging
from datetime import datetime
import time
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv

# Configuration du logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("usd_to_usdt_converter")

def convert_usd_to_usdt(amount, api_key=None, api_secret=None, base_url=None):
    """
    Convertit un montant spécifié d'USD en USDT
    
    Args:
        amount (float): Montant en USD à convertir en USDT
        api_key (str, optional): Clé API Alpaca
        api_secret (str, optional): Secret API Alpaca
        base_url (str, optional): URL de base Alpaca
    """
    if not api_key or not api_secret or not base_url:
        # Récupérer les clés d'API depuis les variables d'environnement
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
    
    logger.info(f"Mode Alpaca: {'LIVE' if alpaca_mode == 'live' else 'PAPER'}")
    logger.info(f"Tentative de conversion de ${amount} USD en USDT")
    
    # Initialiser le client API Alpaca
    api = tradeapi.REST(
        key_id=api_key,
        secret_key=api_secret,
        base_url=base_url
    )
    
    try:
        # Vérifier le solde USD
        account = api.get_account()
        cash = float(account.cash)
        
        if cash < amount:
            logger.error(f"Solde USD insuffisant. Disponible: ${cash}, Requis: ${amount}")
            return False
        
        logger.info(f"Solde USD actuel: ${cash}")
        
        # Récupérer le prix actuel du USDT
        try:
            # Obtenir le dernier prix du USDT/USD
            usdt_price = api.get_latest_trade("USDT/USD").price
            logger.info(f"Prix actuel du USDT/USD: ${usdt_price}")
        except Exception as e:
            # Si on ne peut pas obtenir le prix directement, on considère que 1 USDT ≈ 1 USD
            logger.warning(f"Impossible de récupérer le prix USDT/USD: {e}, en utilisant 1.0")
            usdt_price = 1.0
        
        # Calculer la quantité de USDT à acheter
        usdt_qty = amount / usdt_price
        
        # Passer un ordre d'achat
        order = api.submit_order(
            symbol="USDT/USD",
            qty=usdt_qty,
            side="buy",
            type="market",
            time_in_force="gtc"
        )
        
        logger.info(f"Ordre d'achat soumis: {order.id}")
        logger.info(f"Achat de {usdt_qty:.2f} USDT à environ ${usdt_price:.4f} par USDT")
        
        # Attendre que l'ordre soit exécuté
        max_retries = 10
        for i in range(max_retries):
            try:
                order_status = api.get_order(order.id)
                if order_status.status == 'filled':
                    filled_price = float(order_status.filled_avg_price)
                    filled_qty = float(order_status.filled_qty)
                    total_cost = filled_price * filled_qty
                    
                    logger.info(f"Ordre exécuté avec succès!")
                    logger.info(f"Acheté: {filled_qty:.2f} USDT")
                    logger.info(f"Prix moyen: ${filled_price:.4f}")
                    logger.info(f"Coût total: ${total_cost:.2f}")
                    
                    # Vérifier le nouveau solde
                    account = api.get_account()
                    new_cash = float(account.cash)
                    logger.info(f"Nouveau solde USD: ${new_cash:.2f}")
                    
                    # Essayer de récupérer la position USDT
                    try:
                        usdt_position = api.get_position("USDT/USD")
                        logger.info(f"Position USDT: {usdt_position.qty} USDT")
                    except:
                        logger.warning("Impossible de récupérer la position USDT, mais la conversion a peut-être fonctionné")
                        
                    return True
                elif order_status.status == 'rejected' or order_status.status == 'canceled':
                    logger.error(f"Ordre rejeté ou annulé: {order_status.status}")
                    return False
                
                logger.info(f"Statut de l'ordre: {order_status.status}, tentative {i+1}/{max_retries}")
                time.sleep(2)  # Attendre 2 secondes avant de vérifier à nouveau
            except Exception as e:
                logger.error(f"Erreur lors de la vérification de l'ordre: {e}")
                time.sleep(2)
        
        logger.warning(f"L'ordre n'a pas été complété après {max_retries} tentatives")
        return False
        
    except Exception as e:
        logger.error(f"Erreur lors de la conversion USD → USDT: {e}")
        return False

def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(description="Convertir USD en USDT sur Alpaca")
    parser.add_argument("--amount", type=float, required=True,
                      help="Montant en USD à convertir en USDT")
    parser.add_argument("--mode", type=str, choices=["paper", "live"], default=None,
                      help="Mode Alpaca (paper ou live). Par défaut, utilise ALPACA_MODE du .env")
    
    args = parser.parse_args()
    
    # Changer temporairement le mode si spécifié
    original_mode = None
    if args.mode:
        original_mode = os.getenv("ALPACA_MODE")
        os.environ["ALPACA_MODE"] = args.mode
    
    try:
        convert_usd_to_usdt(args.amount)
    finally:
        # Remettre le mode original si on l'a changé
        if original_mode:
            os.environ["ALPACA_MODE"] = original_mode

if __name__ == "__main__":
    main()
