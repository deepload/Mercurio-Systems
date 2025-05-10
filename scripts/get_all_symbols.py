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
import json
import asyncio
import logging
import requests
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import alpaca_trade_api as tradeapi
from pathlib import Path

# Assurez-vous que le script peut importer les modules MercurioAI
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Importez les modules MercurioAI
from app.utils.env_loader import load_environment
from app.services.market_data import MarketDataService

# Liste des actions populaires que nous voulons absolument inclure
POPULAR_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "BRK.B", "BRK.A", 
    "NVDA", "JPM", "JNJ", "V", "UNH", "BAC", "PG", "HD", "XOM", "AVGO",
    "LLY", "MA", "CVX", "ABBV", "COST", "MRK", "PEP", "ADBE", "KO", "WMT",
    "CRM", "NFLX", "CSCO", "TMO", "ACN", "MCD", "ABT", "INTC", "DIS", "AMD"
]

async def get_all_stocks_alpaca():
    """Récupère tous les symboles d'actions disponibles via Alpaca"""
    # Récupérer les clés API depuis les variables d'environnement
    alpaca_mode = os.getenv("ALPACA_MODE", "paper").lower()
    
    # Configuration selon le mode
    if alpaca_mode == "live":
        alpaca_key = os.getenv("ALPACA_LIVE_KEY")
        alpaca_secret = os.getenv("ALPACA_LIVE_SECRET")
        base_url = "https://api.alpaca.markets"
        data_url = "https://data.alpaca.markets"
    else:  # paper mode par défaut
        alpaca_key = os.getenv("ALPACA_PAPER_KEY")
        alpaca_secret = os.getenv("ALPACA_PAPER_SECRET")
        base_url = "https://paper-api.alpaca.markets"
        data_url = "https://data.alpaca.markets"
    
    if not (alpaca_key and alpaca_secret):
        logger.error(f"Clés API Alpaca ({alpaca_mode}) non trouvées dans les variables d'environnement")
        return []
    
    # Initialiser le client Alpaca
    try:
        # Utiliser le constructeur approprié pour la version d'Alpaca
        try:
            # Nouvelle version d'Alpaca
            api = tradeapi.REST(api_key=alpaca_key, secret_key=alpaca_secret, base_url=base_url, data_url=data_url)
        except TypeError:
            # Ancienne version d'Alpaca
            logger.info("Utilisation de l'ancien format d'initialisation pour l'API Alpaca")
            api = tradeapi.REST(alpaca_key, alpaca_secret, base_url=base_url)
            
        # Récupérer tous les actifs (actions)
        assets = api.list_assets(status='active', asset_class='us_equity')
        
        # Extraire les symboles
        symbols = [asset.symbol for asset in assets if asset.tradable]
        logger.info(f"Récupéré {len(symbols)} symboles d'actions via Alpaca")
        return symbols
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des symboles d'actions via Alpaca: {e}")
        return []

async def get_all_stocks_yahoo():
    """Récupère les symboles des actions populaires via Yahoo Finance avec temporisation"""
    try:
        # On utilise une liste prédéfinie des actions populaires
        valid_symbols = []
        
        # Utiliser simplement la liste prédéfinie si Yahoo rencontre des problèmes de limitation
        rate_limit_count = 0
        rate_limit_threshold = 3  # Après 3 erreurs de limitation, on arrête d'essayer
        
        for symbol in POPULAR_STOCKS:
            try:
                # Ajouter une pause de 2 secondes entre chaque requête pour éviter la limitation
                await asyncio.sleep(2)
                
                # Vérifier si le symbole existe en récupérant des données minimales
                ticker = yf.Ticker(symbol)
                info = ticker.info
                if 'symbol' in info or 'shortName' in info:
                    valid_symbols.append(symbol)
                    logger.info(f"Symbole confirmé via Yahoo Finance: {symbol}")
            except Exception as e:
                error_str = str(e).lower()
                if "too many requests" in error_str or "rate limit" in error_str:
                    rate_limit_count += 1
                    # Ajouter le symbole malgré l'erreur car il fait partie de la liste prédéfinie
                    valid_symbols.append(symbol)
                    
                    # Si nous avons atteint le seuil, arrêter et retourner tous les symboles prédéfinis
                    if rate_limit_count >= rate_limit_threshold:
                        logger.warning(f"Trop d'erreurs de limitation avec Yahoo Finance, utilisation de la liste prédéfinie")
                        return POPULAR_STOCKS
                    
                    # Pause plus longue après une erreur de limitation
                    await asyncio.sleep(5)
                else:
                    logger.warning(f"Impossible de valider le symbole {symbol} via Yahoo Finance: {e}")
        
        # Si on a reçu moins de 10 symboles valides, utiliser la liste prédéfinie par mesure de sécurité
        if len(valid_symbols) < 10:
            logger.warning(f"Trop peu de symboles validés ({len(valid_symbols)}), utilisation de la liste prédéfinie")
            return POPULAR_STOCKS
            
        logger.info(f"Récupéré {len(valid_symbols)} symboles d'actions populaires via Yahoo Finance")
        return valid_symbols
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des symboles via Yahoo Finance: {e}")
        return []

async def get_all_stocks():
    """Récupère tous les symboles d'actions en combinant plusieurs sources"""
    # Récupérer les symboles via Alpaca
    alpaca_symbols = await get_all_stocks_alpaca()
    
    # Récupérer les symboles populaires via Yahoo Finance
    yahoo_symbols = await get_all_stocks_yahoo()
    
    # Combiner et dédupliquer les symboles
    all_symbols = list(set(alpaca_symbols + yahoo_symbols))
    
    # Vérifier si les symboles populaires sont inclus, sinon les ajouter explicitement
    for symbol in POPULAR_STOCKS:
        if symbol not in all_symbols:
            all_symbols.append(symbol)
    
    logger.info(f"Total après fusion: {len(all_symbols)} symboles d'actions uniques")
    return all_symbols

async def verify_stock_data_availability(symbols):
    """Vérifie que les données actions sont disponibles via différentes sources"""
    # Utiliser le service de données de MercurioAI pour vérifier l'accès aux données
    market_data = MarketDataService()
    today = datetime.now()
    start_date = today - timedelta(days=5)  # Vérifier les 5 derniers jours
    
    verified_symbols = []
    unverified_symbols = []
    
    # Vérifier par lots pour être plus efficace
    batch_size = 10
    total_batches = (len(symbols) - 1) // batch_size + 1
    
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        logger.info(f"Vérification du lot {i//batch_size + 1}/{total_batches} ({len(batch)} symboles)")
        
        for symbol in batch:
            try:
                # Tentative de récupération de données récentes
                data = await market_data.get_historical_data(
                    symbol, 
                    start_date, 
                    today, 
                    timeframe="1d"
                )
                
                if not data.empty:
                    verified_symbols.append(symbol)
                    logger.info(f"Symbole action vérifié: {symbol}")
                else:
                    unverified_symbols.append(symbol)
                    logger.warning(f"Pas de données pour {symbol}")
            except Exception as e:
                unverified_symbols.append(symbol)
                logger.warning(f"Erreur lors de la vérification pour {symbol}: {str(e)[:100]}")
    
    # Pour les symboles non vérifiés, essayer Yahoo Finance
    yahoo_verified = []
    for symbol in unverified_symbols:
        try:
            # Vérifier si on peut obtenir des données via Yahoo Finance
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")
            if not hist.empty:
                yahoo_verified.append(symbol)
                logger.info(f"Symbole action vérifié via Yahoo Finance: {symbol}")
            else:
                logger.warning(f"Pas de données pour {symbol} via Yahoo Finance")
        except Exception as e:
            logger.warning(f"Erreur lors de la vérification via Yahoo pour {symbol}: {str(e)[:100]}")
    
    # Combiner les symboles vérifiés via les deux sources
    all_verified = verified_symbols + yahoo_verified
    
    # S'assurer que les actions populaires sont incluses même sans vérification
    for symbol in POPULAR_STOCKS:
        if symbol not in all_verified:
            all_verified.append(symbol)
            logger.info(f"Ajout du symbole populaire sans vérification: {symbol}")
    
    logger.info(f"Vérification terminée: {len(all_verified)}/{len(symbols)} symboles d'actions disponibles")
    return all_verified

async def get_all_crypto():
    """
    Récupère tous les symboles de crypto-monnaies disponibles via l'API Alpaca
    
    Returns:
        Liste des symboles de crypto-monnaies
    """
    # Récupérer les clés API depuis les variables d'environnement
    alpaca_mode = os.getenv("ALPACA_MODE", "paper").lower()
    
    # Configuration selon le mode
    if alpaca_mode == "live":
        alpaca_key = os.getenv("ALPACA_LIVE_KEY")
        alpaca_secret = os.getenv("ALPACA_LIVE_SECRET")
        base_url = "https://api.alpaca.markets"
        data_url = "https://data.alpaca.markets"
    else:  # paper mode par défaut
        alpaca_key = os.getenv("ALPACA_PAPER_KEY")
        alpaca_secret = os.getenv("ALPACA_PAPER_SECRET")
        base_url = "https://paper-api.alpaca.markets"
        data_url = "https://data.alpaca.markets"
    
    if not (alpaca_key and alpaca_secret):
        logger.error(f"Clés API Alpaca ({alpaca_mode}) non trouvées dans les variables d'environnement")
        return []
        
    try:
        # Initialiser le client Alpaca directement
        try:
            # Nouvelle version d'Alpaca
            alpaca_client = tradeapi.REST(api_key=alpaca_key, secret_key=alpaca_secret, base_url=base_url, data_url=data_url)
        except TypeError:
            # Ancienne version d'Alpaca
            logger.info("Utilisation de l'ancien format d'initialisation pour l'API Alpaca")
            alpaca_client = tradeapi.REST(alpaca_key, alpaca_secret, base_url=base_url)
        
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
        crypto: Liste des symboles de crypto-monnaie
    """
    # Récupérer le répertoire de données
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Date d'aujourd'hui pour nommer les fichiers
    today = datetime.now().strftime("%Y%m%d")
    
    # Sauvegarder les actions
    stocks_df = pd.DataFrame({"symbol": stocks})
    stocks_file = data_dir / f"all_stocks_{today}.csv"
    stocks_df.to_csv(stocks_file, index=False)
    
    # Sauvegarder les crypto-monnaies
    crypto_df = pd.DataFrame({"symbol": crypto})
    crypto_file = data_dir / f"all_crypto_{today}.csv"
    crypto_df.to_csv(crypto_file, index=False)
    
    # Créer un fichier JSON avec des métadonnées
    metadata_file = data_dir / f"symbols_metadata_{today}.json"
    metadata = {
        "date": today,
        "stocks_count": len(stocks),
        "crypto_count": len(crypto)
    }
    
    with open(metadata_file, "w") as f:
        json.dump(metadata, f)
    
    logger.info(f"{len(stocks)} symboles d'actions sauvegardés dans {stocks_file}")
    logger.info(f"{len(crypto)} symboles de crypto-monnaies sauvegardés dans {crypto_file}")
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
