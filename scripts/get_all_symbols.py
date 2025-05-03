#!/usr/bin/env python
"""
MercurioAI - Get All Available Symbols

Ce script récupère tous les symboles d'actions et de crypto-monnaies disponibles
via l'API Alpaca et les sauvegarde dans des fichiers CSV pour une utilisation ultérieure.

Exemple d'utilisation:
    python scripts/get_all_symbols.py
"""

import os
import sys
import logging
import csv
import json
from datetime import datetime

# Assurez-vous que le script peut importer les modules MercurioAI
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Importez les modules MercurioAI
from app.utils.env_loader import load_environment
from app.services.market_data import MarketDataService

async def get_all_stocks():
    """
    Récupère tous les symboles d'actions disponibles via l'API Alpaca
    
    Returns:
        Liste des symboles d'actions
    """
    try:
        # Initialiser le service de données
        market_data = MarketDataService()
        
        # Accéder directement au client Alpaca
        alpaca_client = market_data.alpaca_client
        
        # Récupérer tous les actifs
        assets = alpaca_client.list_assets(status='active', asset_class='us_equity')
        
        # Extraire les symboles
        symbols = [asset.symbol for asset in assets if asset.tradable]
        
        logger.info(f"Récupération de {len(symbols)} symboles d'actions réussie")
        return symbols
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des symboles d'actions: {e}")
        return []

async def get_all_crypto():
    """
    Récupère tous les symboles de crypto-monnaies disponibles via l'API Alpaca
    
    Returns:
        Liste des symboles de crypto-monnaies
    """
    try:
        # Initialiser le service de données
        market_data = MarketDataService()
        
        # Accéder directement au client Alpaca
        alpaca_client = market_data.alpaca_client
        
        # Récupérer tous les actifs
        assets = alpaca_client.list_assets(status='active', asset_class='crypto')
        
        # Extraire les symboles et les convertir au format "BTC-USD"
        symbols = []
        for asset in assets:
            if asset.tradable:
                # Convertir le format "BTC/USD" en "BTC-USD"
                if '/' in asset.symbol:
                    base, quote = asset.symbol.split('/')
                    if quote == 'USD':
                        symbols.append(f"{base}-USD")
                else:
                    symbols.append(asset.symbol)
        
        logger.info(f"Récupération de {len(symbols)} symboles de crypto-monnaies réussie")
        return symbols
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des symboles de crypto-monnaies: {e}")
        return []

async def save_symbols_to_csv(stocks, crypto):
    """
    Sauvegarde les symboles dans des fichiers CSV
    
    Args:
        stocks: Liste des symboles d'actions
        crypto: Liste des symboles de crypto-monnaies
    """
    # Créer le répertoire si nécessaire
    os.makedirs("data", exist_ok=True)
    
    # Date actuelle pour les noms de fichiers
    date_str = datetime.now().strftime("%Y%m%d")
    
    # Sauvegarder les actions
    stocks_file = f"data/all_stocks_{date_str}.csv"
    with open(stocks_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["symbol"])
        for symbol in stocks:
            writer.writerow([symbol])
    
    # Sauvegarder les crypto-monnaies
    crypto_file = f"data/all_crypto_{date_str}.csv"
    with open(crypto_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["symbol"])
        for symbol in crypto:
            writer.writerow([symbol])
    
    # Créer un fichier JSON avec des métadonnées
    metadata_file = f"data/symbols_metadata_{date_str}.json"
    metadata = {
        "date": datetime.now().isoformat(),
        "stocks_count": len(stocks),
        "crypto_count": len(crypto),
        "stocks_file": stocks_file,
        "crypto_file": crypto_file
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Symboles sauvegardés dans {stocks_file} et {crypto_file}")
    logger.info(f"Métadonnées sauvegardées dans {metadata_file}")
    
    return stocks_file, crypto_file, metadata_file

async def main():
    """Fonction principale"""
    logger.info("Récupération de tous les symboles disponibles...")
    
    # Charger les variables d'environnement
    load_environment()
    
    # Récupérer les symboles
    stocks = await get_all_stocks()
    crypto = await get_all_crypto()
    
    # Sauvegarder les symboles
    stocks_file, crypto_file, metadata_file = await save_symbols_to_csv(stocks, crypto)
    
    # Afficher les résultats
    logger.info("\n=== Résumé ===")
    logger.info(f"Actions: {len(stocks)} symboles")
    logger.info(f"Crypto-monnaies: {len(crypto)} symboles")
    logger.info(f"Total: {len(stocks) + len(crypto)} symboles")
    
    # Afficher les commandes pour entraîner les modèles
    logger.info("\n=== Commandes pour l'entraînement ===")
    logger.info(f"Pour entraîner avec toutes les actions:")
    logger.info(f"python scripts/train_all_models.py --include_stocks --top_assets {len(stocks)}")
    logger.info(f"\nPour entraîner avec toutes les crypto-monnaies:")
    logger.info(f"python scripts/train_all_models.py --include_crypto --top_assets {len(crypto)}")
    logger.info(f"\nPour entraîner avec tous les actifs (peut être très long):")
    logger.info(f"python scripts/train_all_models.py --include_stocks --include_crypto --top_assets {len(stocks) + len(crypto)}")
    logger.info(f"\nOu avec un fichier personnalisé:")
    logger.info(f"python scripts/train_all_models.py --custom_stocks_file {stocks_file} --custom_crypto_file {crypto_file}")
    
    return 0

if __name__ == "__main__":
    import asyncio
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        logger.info("Programme interrompu par l'utilisateur")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Erreur critique non gérée: {e}")
        sys.exit(1)
