#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour générer une commande d'entraînement incluant tous les symboles
des fichiers de top stocks et top crypto
"""

import csv
import os
import sys
from datetime import datetime

# Chemins des fichiers CSV
STOCKS_FILE = os.path.join('reports', 'best_assets', '2025-05-06', 'top_stocks.csv')
CRYPTO_FILE = os.path.join('reports', 'best_assets', '2025-05-06', 'top_crypto.csv')

def main():
    # Symboles de base demandés
    base_symbols = ["BTC-USD", "ETH-USD", "AAPL", "MSFT", "TSLA"]
    
    # Lire les symboles des fichiers CSV
    stock_symbols = []
    with open(STOCKS_FILE, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Ignorer l'en-tête
        for row in reader:
            if row and row[0] != 'Symbol':
                stock_symbols.append(row[0])
    
    crypto_symbols = []
    with open(CRYPTO_FILE, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Ignorer l'en-tête
        for row in reader:
            if row and row[0] != 'Symbol':
                crypto_symbols.append(row[0])
    
    # Combiner tous les symboles (en éliminant les doublons)
    all_symbols = list(set(base_symbols + stock_symbols + crypto_symbols))
    
    # Générer la commande d'entraînement
    command = f"python scripts/train_all_models.py --symbols {','.join(all_symbols)} --days 90 --include_stocks --include_crypto"
    
    print("\n=== COMMANDE D'ENTRAÎNEMENT GÉNÉRÉE ===")
    print(f"Nombre total de symboles: {len(all_symbols)}")
    print(f"Stocks: {len(stock_symbols)}")
    print(f"Cryptos: {len(crypto_symbols)}")
    print("\nCOMMANDE:")
    print(command)
    
    # Écrire la commande dans un fichier pour référence
    with open(f"training_command_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", 'w') as f:
        f.write(command)
    
    print(f"\nLa commande a été sauvegardée dans le fichier training_command_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

if __name__ == "__main__":
    main()
