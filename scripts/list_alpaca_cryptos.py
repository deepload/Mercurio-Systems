#!/usr/bin/env python
"""
Liste toutes les cryptomonnaies disponibles sur Alpaca.
Ce script permet de récupérer la liste complète des cryptos disponibles
pour éviter d'utiliser des symboles qui ne sont pas supportés.
"""

import os
import sys
import logging
import requests
from dotenv import load_dotenv
from pathlib import Path

# Configurez le logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Assurez-vous que le chemin racine du projet est dans le path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

# Chargez les variables d'environnement (clés API)
load_dotenv(root_dir / '.env')
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_API_SECRET = os.getenv('ALPACA_API_SECRET')

def get_available_cryptos():
    """
    Récupère la liste complète des cryptomonnaies disponibles sur Alpaca.
    
    Returns:
        list: Liste de symboles de cryptomonnaies disponibles
    """
    try:
        # Endpoint pour liste des cryptos (v1beta3)
        url = "https://data.alpaca.markets/v1beta3/crypto/us/assets"
        
        headers = {
            "Apca-Api-Key-Id": ALPACA_API_KEY,
            "Apca-Api-Secret-Key": ALPACA_API_SECRET,
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Lever une exception si erreur HTTP
        
        data = response.json()
        
        # Extraire les symboles et les formater
        crypto_symbols = []
        for asset in data:
            symbol = asset.get('symbol').replace('/', '-')  # Convertir le format Alpaca au format Mercurio
            crypto_symbols.append(symbol)
            
        return crypto_symbols
    
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des cryptomonnaies: {e}")
        return []

def save_crypto_list(symbols, output_file="custom_crypto_symbols.txt"):
    """
    Sauvegarde la liste des symboles dans un fichier texte
    
    Args:
        symbols (list): Liste des symboles à sauvegarder
        output_file (str): Nom du fichier de sortie
    """
    with open(output_file, 'w') as f:
        for symbol in symbols:
            f.write(f"{symbol}\n")
    
    logger.info(f"Liste de {len(symbols)} cryptomonnaies sauvegardée dans {output_file}")

def generate_env_format(symbols, max_count=30):
    """
    Génère la ligne PERSONALIZED_CRYPTO_LIST pour le fichier .env
    
    Args:
        symbols (list): Liste des symboles à inclure
        max_count (int): Nombre maximum de symboles à inclure
    
    Returns:
        str: Ligne formatée pour .env
    """
    alpaca_format_symbols = [symbol.replace('-', '/') for symbol in symbols[:max_count]]
    return f"PERSONALIZED_CRYPTO_LIST={','.join(alpaca_format_symbols)}"

if __name__ == "__main__":
    # Récupérer la liste des cryptos disponibles
    available_cryptos = get_available_cryptos()
    
    if available_cryptos:
        logger.info(f"Nombre total de cryptomonnaies disponibles: {len(available_cryptos)}")
        
        # Sauvegarder la liste complète dans un fichier
        save_crypto_list(available_cryptos, "available_crypto_symbols.txt")
        
        # Générer la ligne pour .env (limité à 30 premières)
        env_line = generate_env_format(available_cryptos)
        logger.info(f"Ligne pour fichier .env:\n{env_line}")
    else:
        logger.error("Impossible de récupérer la liste des cryptomonnaies disponibles")
