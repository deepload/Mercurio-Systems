#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour extraire les symboles d'actions du fichier top_stocks.csv 
et générer une commande d'entraînement
"""

import csv
import os

# Chemin du fichier CSV
STOCKS_FILE = os.path.join('reports', 'best_assets', '2025-05-06', 'top_stocks.csv')

def main():
    # Lire les symboles des actions
    stock_symbols = []
    with open(STOCKS_FILE, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Ignorer l'en-tête
        for row in reader:
            if row and row[0] != 'Symbol':
                stock_symbols.append(row[0])
    
    # Générer la commande d'entraînement avec seulement les actions
    command = f"python scripts/train_all_models.py --symbols {','.join(stock_symbols)} --days 90 --include_stocks"
    
    print("\n=== COMMANDE D'ENTRAÎNEMENT POUR LES ACTIONS ===")
    print(f"Nombre d'actions: {len(stock_symbols)}")
    print("\nSymboles d'actions:")
    print(', '.join(stock_symbols))
    print("\nCOMMANDE:")
    print(command)

if __name__ == "__main__":
    main()
