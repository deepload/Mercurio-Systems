#!/usr/bin/env python
"""
Script pour récupérer toutes les cryptomonnaies disponibles sur Alpaca
et mettre à jour le fichier .env avec la liste complète.
"""
import os
import sys
import time
import logging
import requests
from pathlib import Path
import re

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Chemin vers le fichier .env
ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT_DIR / '.env'

def load_env_variables():
    """Charge les variables d'environnement depuis le fichier .env"""
    env_vars = {}
    
    try:
        with open(ENV_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key] = value
        
        # Déterminer le mode (paper ou live)
        alpaca_mode = env_vars.get('ALPACA_MODE', 'paper')
        if 'live' in alpaca_mode:
            api_key = env_vars.get('ALPACA_LIVE_KEY', '')
            api_secret = env_vars.get('ALPACA_LIVE_SECRET', '')
        else:
            api_key = env_vars.get('ALPACA_PAPER_KEY', '')
            api_secret = env_vars.get('ALPACA_PAPER_SECRET', '')
        
        return api_key, api_secret
    
    except Exception as e:
        logger.error(f"Erreur lors de la lecture du fichier .env: {e}")
        return None, None

def get_alpaca_cryptos(api_key, api_secret):
    """Récupère la liste des cryptomonnaies disponibles sur Alpaca"""
    try:
        # Endpoint pour les cryptomonnaies
        url = "https://data.alpaca.markets/v1beta3/crypto/us/assets"
        
        headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret,
        }
        
        logger.info(f"Requête à l'API Alpaca: {url}")
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Nombre de cryptomonnaies trouvées: {len(data)}")
            
            # Filtrer seulement les cryptos actives (tradable)
            tradable_symbols = []
            for asset in data:
                if asset.get('status') == 'active' and asset.get('tradable', False):
                    symbol = asset.get('symbol', '')
                    tradable_symbols.append(symbol)
            
            logger.info(f"Nombre de cryptomonnaies actives et tradable: {len(tradable_symbols)}")
            return tradable_symbols
        else:
            logger.error(f"Erreur API ({response.status_code}): {response.text}")
            
            # Deuxième tentative avec un autre endpoint
            logger.info("Tentative avec un endpoint alternatif...")
            alt_url = "https://paper-api.alpaca.markets/v2/assets"
            alt_response = requests.get(alt_url, headers=headers)
            
            if alt_response.status_code == 200:
                assets = alt_response.json()
                crypto_assets = [asset for asset in assets if asset.get('class') == 'crypto' and asset.get('status') == 'active']
                logger.info(f"Nombre de cryptomonnaies trouvées via endpoint alternatif: {len(crypto_assets)}")
                return [asset.get('symbol') for asset in crypto_assets]
            else:
                logger.error(f"Échec de la seconde tentative ({alt_response.status_code}): {alt_response.text}")
                return []
    
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des cryptomonnaies: {e}")
        return []

def get_default_crypto_list():
    """Fournit une liste par défaut des cryptomonnaies les plus courantes sur Alpaca"""
    return [
        "BTC/USD", "ETH/USD", "SOL/USD", "ADA/USD", "XRP/USD", "DOT/USD", 
        "DOGE/USD", "AVAX/USD", "SHIB/USD", "LINK/USD", "MATIC/USD",
        "UNI/USD", "LTC/USD", "BCH/USD", "ATOM/USD", "XTZ/USD", "AAVE/USD",
        "ALGO/USD", "BAT/USD", "CRV/USD", "FIL/USD", "GRT/USD", "ICP/USD",
        "NEAR/USD", "OP/USD", "ARB/USD", "COMP/USD", "MKR/USD", "SUSHI/USD",
        "YFI/USD", "1INCH/USD", "APE/USD", "AXS/USD", "FTM/USD", "GALA/USD",
        "HBAR/USD", "MANA/USD", "PAXG/USD", "SAND/USD", "VET/USD",
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT", "XRP/USDT", "DOT/USDT",
        "DOGE/USDT", "AVAX/USDT", "SHIB/USDT", "LINK/USDT", "MATIC/USDT"
    ]

def update_env_file(crypto_list):
    """Met à jour le fichier .env avec la nouvelle liste de cryptomonnaies"""
    try:
        # Lire le contenu actuel du fichier .env
        with open(ENV_FILE, 'r') as f:
            lines = f.readlines()
        
        # Préparer la nouvelle ligne pour PERSONALIZED_CRYPTO_LIST
        crypto_list_value = ','.join(crypto_list)
        new_line = f"PERSONALIZED_CRYPTO_LIST={crypto_list_value}\n"
        
        # Chercher et remplacer la ligne existante
        personalized_list_found = False
        with open(ENV_FILE, 'w') as f:
            for line in lines:
                if line.strip().startswith('PERSONALIZED_CRYPTO_LIST=') and not line.strip().startswith('#'):
                    f.write(new_line)
                    personalized_list_found = True
                else:
                    f.write(line)
            
            # Si la ligne n'a pas été trouvée, l'ajouter à la fin
            if not personalized_list_found:
                f.write("\n# Liste personnalisée des crypto-monnaies disponibles sur Alpaca\n")
                f.write(new_line)
        
        logger.info(f"Fichier .env mis à jour avec {len(crypto_list)} cryptomonnaies")
        
        # Sauvegarder également la liste dans un fichier séparé
        with open(ROOT_DIR / "crypto_symbols_alpaca.txt", "w") as f:
            for symbol in crypto_list:
                f.write(f"{symbol}\n")
        
        logger.info(f"Liste sauvegardée dans crypto_symbols_alpaca.txt")
        return True
    
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour du fichier .env: {e}")
        return False

if __name__ == "__main__":
    logger.info("Récupération des cryptomonnaies disponibles sur Alpaca...")
    
    # Charger les clés API
    api_key, api_secret = load_env_variables()
    
    if not api_key or not api_secret:
        logger.error("Clés API Alpaca non trouvées dans le fichier .env")
        sys.exit(1)
    
    # Récupérer les cryptomonnaies
    crypto_list = get_alpaca_cryptos(api_key, api_secret)
    
    if not crypto_list:
        logger.warning("Aucune cryptomonnaie récupérée via l'API, utilisation de la liste par défaut")
        crypto_list = get_default_crypto_list()
    
    # Mettre à jour le fichier .env
    if update_env_file(crypto_list):
        logger.info("Liste de cryptomonnaies mise à jour avec succès")
        
        # Afficher les 20 premières cryptos
        logger.info("Aperçu des 20 premières cryptomonnaies:")
        for i, symbol in enumerate(crypto_list[:20]):
            logger.info(f"  {i+1}. {symbol}")
        
        if len(crypto_list) > 20:
            logger.info(f"  ... et {len(crypto_list) - 20} autres")
    else:
        logger.error("Échec de la mise à jour de la liste de cryptomonnaies")
