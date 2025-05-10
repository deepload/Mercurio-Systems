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

def liquidate_crypto_position(api, symbol, position, reduction_strategy="progressive", auto=False):
    """Tenter de liquider une position de crypto avec des méthodes alternatives
    
    Args:
        api: API Alpaca
        symbol: Symbole de la crypto
        position: Position à liquider
        reduction_strategy: Stratégie de réduction ("progressive", "fixed", "half")
        auto: Si True, applique automatiquement la stratégie sans demander confirmation
        
    Returns:
        bool: True si la liquidation a réussi, False sinon
    """
    try:
        logger.info(f"Tentative de liquidation spéciale de {symbol}...")
        
        # Récupérer la quantité actuelle
        qty = float(position.qty)
        market_value = float(position.market_value)
        
        if not auto:
            logger.info("-" * 60)
            logger.info(f"Problème de liquidation détecté pour {symbol}")
            logger.info(f"Position actuelle: {qty} units, valeur: ${market_value:.2f}")
            logger.info("Alpaca signale 'insufficient balance' pour cette position.")
            logger.info("Options de gestion:")
            logger.info("1. Progressive - Essayer avec 95%, puis 90%, 80%, etc. de la quantité")
            logger.info("2. Fixed - Utiliser un pourcentage fixe de la quantité (ex: 90%)")
            logger.info("3. Half - Tenter de liquider la moitié de la position")
            logger.info("4. Skip - Ignorer cette position et la conserver")
            logger.info("-" * 60)
            
            choice = input("Choisissez une option (1-4): ")
            
            if choice == "1":
                reduction_strategy = "progressive"
            elif choice == "2":
                reduction_strategy = "fixed"
                pct = float(input("Pourcentage à liquider (1-99): "))
                if pct >= 1 and pct <= 99:
                    fixed_pct = pct / 100
                else:
                    fixed_pct = 0.9  # 90% par défaut si valeur invalide
            elif choice == "3":
                reduction_strategy = "half"
            elif choice == "4":
                logger.info(f"Position {symbol} ignorée")
                return False
            else:
                reduction_strategy = "progressive"  # Par défaut si choix invalide
        
        # Appliquer la stratégie sélectionnée
        success = False
        
        if reduction_strategy == "progressive":
            # Essayer avec différents pourcentages, du plus élevé au plus bas
            percentages = [0.95, 0.90, 0.80, 0.70, 0.50]
            
            for pct in percentages:
                try:
                    reduced_qty = qty * pct
                    logger.info(f"Tentative avec {pct*100:.0f}%: {reduced_qty} sur {qty} total")
                    
                    api.submit_order(
                        symbol=symbol,
                        qty=reduced_qty,
                        side='sell',
                        type='market',
                        time_in_force='gtc'
                    )
                    
                    logger.info(f"Vente partielle de {symbol} réussie avec {pct*100:.0f}%")
                    success = True
                    break
                except Exception as e:
                    logger.warning(f"Échec avec {pct*100:.0f}%: {e}")
                    # Continue avec le prochain pourcentage
        
        elif reduction_strategy == "fixed":
            # Utiliser le pourcentage fixé
            try:
                percentage = fixed_pct if 'fixed_pct' in locals() else 0.9
                reduced_qty = qty * percentage
                
                logger.info(f"Tentative avec {percentage*100:.0f}%: {reduced_qty} sur {qty} total")
                
                api.submit_order(
                    symbol=symbol,
                    qty=reduced_qty,
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )
                
                logger.info(f"Vente partielle de {symbol} réussie avec {percentage*100:.0f}%")
                success = True
            except Exception as e:
                logger.error(f"Échec avec le pourcentage fixé: {e}")
        
        elif reduction_strategy == "half":
            # Tenter de liquider la moitié
            try:
                half_qty = qty * 0.5
                
                logger.info(f"Tentative de vente de la moitié: {half_qty} sur {qty} total")
                
                api.submit_order(
                    symbol=symbol,
                    qty=half_qty,
                    side='sell',
                    type='market',
                    time_in_force='gtc'
                )
                
                logger.info(f"Vente de la moitié de {symbol} réussie")
                success = True
            except Exception as e:
                logger.error(f"Échec de la vente de moitié: {e}")
        
        return success
        
    except Exception as e:
        logger.error(f"Échec de la méthode alternative pour {symbol}: {e}")
        return False

def main():
    # Charger les variables d'environnement
    load_dotenv()
    
    # Traiter les arguments en ligne de commande
    import argparse
    parser = argparse.ArgumentParser(description="Liquidate all positions")
    parser.add_argument("--force", action="store_true", help="Force liquidation with alternative methods for problematic positions")
    parser.add_argument("--crypto-only", action="store_true", help="Liquidate only crypto positions")
    parser.add_argument("--stock-only", action="store_true", help="Liquidate only stock positions")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")
    parser.add_argument("--auto-progressive", action="store_true", help="Automatically use progressive reduction for problematic positions")
    parser.add_argument("--auto-fixed", type=float, help="Automatically use fixed percentage reduction (e.g., 90)")
    parser.add_argument("--auto-half", action="store_true", help="Automatically sell half of problematic positions")
    args = parser.parse_args()
    
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
            
        # Filtrer les positions si nécessaire
        original_positions = positions
        if args.crypto_only:
            positions = [p for p in positions if '/' in p.symbol]  # Les cryptos contiennent généralement un '/'
            logger.info(f"Filtré pour ne garder que {len(positions)} positions crypto")
        elif args.stock_only:
            positions = [p for p in positions if '/' not in p.symbol]  # Les actions n'ont pas de '/'
            logger.info(f"Filtré pour ne garder que {len(positions)} positions d'actions")
        
        # Afficher les positions avant liquidation
        logger.info("Positions avant liquidation:")
        for position in positions:
            market_value = float(position.market_value)
            unrealized_pl = float(position.unrealized_pl)
            unrealized_plpc = float(position.unrealized_plpc) * 100
            logger.info(f"  {position.symbol}: {position.qty} actions @ ${float(position.current_price):.2f} - Valeur: ${market_value:.2f} - P/L: ${unrealized_pl:.2f} ({unrealized_plpc:.2f}%)")
        
        # Demander confirmation sauf si --yes est spécifié
        if not args.yes:
            confirm = input("\nVoulez-vous vraiment liquider toutes ces positions? (y/n): ")
            if confirm.lower() != 'y':
                logger.info("Opération annulée.")
                return
        else:
            logger.info("Confirmation automatique activée avec l'option --yes")
        
        # Liquider toutes les positions
        logger.info("Liquidation de toutes les positions...")
        
        # Méthode 1: Liquider toutes les positions d'un coup d'abord (plus rapide)
        success = False
        try:
            logger.info("Tentative de liquidation groupée de toutes les positions...")
            api.close_all_positions()
            logger.info("Toutes les positions semblent avoir été liquidées!")
            success = True
        except Exception as e:
            logger.error(f"Erreur lors de la liquidation groupée: {e}")
            logger.info("Passage à la méthode position par position...")
        
        # Vérifier s'il reste des positions après la liquidation groupée
        remaining_positions = api.list_positions()
        
        # Si la liquidation groupée a échoué ou s'il reste des positions, essayer position par position
        if not success or len(remaining_positions) > 0:
            logger.info(f"Tentative de liquidation individuelle pour {len(remaining_positions)} positions restantes")
            
            # Méthode 2: Liquider position par position
            for position in remaining_positions:
                try:
                    symbol = position.symbol
                    logger.info(f"Liquidation de {symbol}...")
                    api.close_position(symbol)
                    logger.info(f"Position {symbol} liquidée avec succès!")
                except Exception as e:
                    logger.error(f"Erreur lors de la liquidation de {symbol}: {e}")
                    
                    # Si l'option --force est activée, essayer des méthodes alternatives pour les cryptos
                    if args.force and '/' in symbol:
                        strategy = "progressive"  # Stratégie par défaut
                        auto_mode = False
                        
                        if args.auto_progressive:
                            strategy = "progressive"
                            auto_mode = True
                        elif args.auto_fixed is not None:
                            strategy = "fixed"
                            auto_mode = True
                            # Définir le pourcentage pour la stratégie fixed
                            global fixed_pct
                            fixed_pct = args.auto_fixed / 100  # Convertir pourcentage en décimal
                        elif args.auto_half:
                            strategy = "half"
                            auto_mode = True
                        
                        liquidate_crypto_position(api, symbol, position, strategy, auto_mode)
        
        # Vérification finale
        final_positions = api.list_positions()
        if len(final_positions) > 0:
            logger.warning(f"Il reste encore {len(final_positions)} positions non liquidées:")
            for position in final_positions:
                symbol = position.symbol
                qty = position.qty
                logger.warning(f"  {symbol}: {qty} actions")
                
                # Donner des conseils pour les positions crypto problématiques
                if '/' in symbol and args.force:
                    logger.warning(f"  Conseil: Pour {symbol}, essayez de réduire manuellement votre position")
                    logger.warning(f"  via l'interface Alpaca ou avec une vente manuelle avec quantité réduite.")
        else:
            logger.info("Toutes les positions ont été liquidées avec succès!")
        
        # Récupérer l'état du compte après liquidation
        account = api.get_account()
        logger.info(f"Valeur finale du portefeuille: ${float(account.equity):.2f}")
        logger.info(f"Cash disponible: ${float(account.cash):.2f}")
        
    except Exception as e:
        logger.error(f"Erreur lors de la connexion à Alpaca ou de la liquidation des positions: {e}")

if __name__ == "__main__":
    # Exécuter avec les options par défaut
    main()
