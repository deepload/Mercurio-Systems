#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour lancer le trader de cryptos en mode nuit
Ce script va lancer le trader de cryptos Alpaca avec une durée de session de 9 heures,
parfait pour le faire tourner pendant toute la nuit.
"""

import sys
import os
import argparse
from datetime import datetime

# Ajouter le répertoire parent au path pour pouvoir importer le trader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importer le trader de crypto
from alpaca_crypto_trader import AlpacaCryptoTrader, SessionDuration

def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(description="Lancer le trader crypto pour la nuit")
    parser.add_argument("--position-size", type=float, default=0.02,
                        help="Taille de position en pourcentage du portefeuille (default: 0.02 = 2%)")
    parser.add_argument("--stop-loss", type=float, default=0.03,
                        help="Stop loss en pourcentage (default: 0.03 = 3%)")
    parser.add_argument("--take-profit", type=float, default=0.06,
                        help="Take profit en pourcentage (default: 0.06 = 6%)")
    parser.add_argument("--fast-ma", type=int, default=5,
                        help="Période de la moyenne mobile rapide en minutes (default: 5)")
    parser.add_argument("--slow-ma", type=int, default=15,
                        help="Période de la moyenne mobile lente en minutes (default: 15)")
                        
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"DÉMARRAGE DU TRADER CRYPTO POUR LA NUIT - {datetime.now()}")
    print("=" * 60)
    print(f"Ce trader va tourner en mode PAPER pendant environ 9 heures")
    print(f"Position size: {args.position_size * 100}%")
    print(f"Stop-loss: {args.stop_loss * 100}%")
    print(f"Take-profit: {args.take_profit * 100}%")
    print(f"MA rapide: {args.fast_ma} minutes")
    print(f"MA lente: {args.slow_ma} minutes")
    print("=" * 60)
    
    # Créer le trader avec la durée de session NIGHT_RUN (9 heures)
    trader = AlpacaCryptoTrader(session_duration=SessionDuration.NIGHT_RUN)
    
    # Configurer les paramètres
    trader.position_size_pct = args.position_size
    trader.stop_loss_pct = args.stop_loss
    trader.take_profit_pct = args.take_profit
    trader.fast_ma_period = args.fast_ma
    trader.slow_ma_period = args.slow_ma
    
    # Démarrer le trader
    trader.start()
    
    print("=" * 60)
    print("SESSION DE TRADING TERMINÉE")
    print("=" * 60)
    print("Un rapport détaillé a été généré dans le dossier courant")
    print("=" * 60)

if __name__ == "__main__":
    main()
