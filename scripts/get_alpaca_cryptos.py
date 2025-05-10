#!/usr/bin/env python
"""
Liste toutes les cryptomonnaies disponibles sur Alpaca en utilisant
le SDK officiel d'Alpaca.
"""

import os
import sys
import logging
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd

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

# Chargez les variables d'environnement
load_dotenv(root_dir / '.env')

# Importons du code de Mercurio pour utiliser les fonctions existantes
try:
    from app.services.market_data import MarketDataService
    from app.utils.config import Config
    
    async def get_available_cryptos():
        """
        Utilise le MarketDataService de Mercurio pour récupérer les cryptomonnaies disponibles
        """
        config = Config()
        market_data_service = MarketDataService(config)
        
        # Essayons d'utiliser la méthode interne pour obtenir les symboles disponibles
        try:
            logger.info("Récupération des cryptomonnaies via MarketDataService...")
            if hasattr(market_data_service, 'get_available_symbols'):
                symbols = await market_data_service.get_available_symbols(asset_class='crypto')
                return symbols
            else:
                logger.warning("Méthode get_available_symbols non disponible")
                return []
        except Exception as e:
            logger.error(f"Erreur avec MarketDataService: {e}")
            return []

    # Alternative: utiliser directement l'API Alpaca
    import alpaca_trade_api as tradeapi
    
    def get_alpaca_cryptos():
        """Récupère les cryptomonnaies directement via l'API Alpaca"""
        try:
            api_key = os.getenv('ALPACA_API_KEY')
            api_secret = os.getenv('ALPACA_API_SECRET')
            
            if not api_key or not api_secret:
                logger.error("Clés API Alpaca non disponibles")
                return []
            
            logger.info("Connexion à l'API Alpaca...")
            api = tradeapi.REST(api_key, api_secret, base_url='https://paper-api.alpaca.markets')
            
            # Récupération de tous les assets
            assets = api.list_assets()
            
            # Filtrer pour obtenir uniquement les cryptomonnaies
            crypto_assets = [asset for asset in assets if asset.asset_class == 'crypto']
            
            # Convertir en format XX-USD
            crypto_symbols = [asset.symbol.replace('/', '-') for asset in crypto_assets]
            
            return crypto_symbols
        
        except Exception as e:
            logger.error(f"Erreur lors de la récupération via Alpaca API: {e}")
            return []

    # Méthode pour sauvegarder les symboles dans un fichier
    def save_crypto_list(symbols, output_file="available_crypto_symbols.txt"):
        with open(output_file, 'w') as f:
            for symbol in symbols:
                f.write(f"{symbol}\n")
        
        logger.info(f"Liste de {len(symbols)} cryptomonnaies sauvegardée dans {output_file}")

    # Méthode pour générer la ligne .env
    def generate_env_format(symbols, max_count=30):
        alpaca_format_symbols = [symbol.replace('-', '/') for symbol in symbols[:max_count]]
        return f"PERSONALIZED_CRYPTO_LIST={','.join(alpaca_format_symbols)}"

    # Méthode alternative: Explorer le code source de Mercurio
    def extract_existing_crypto_symbols():
        """Essaie de trouver des listes existantes de cryptomonnaies dans le code de Mercurio"""
        try:
            # Chercher dans les fichiers de définition de symboles
            crypto_file_paths = [
                os.path.join(root_dir, 'data', 'crypto_symbols.txt'),
                os.path.join(root_dir, 'data', 'custom_crypto_symbols.txt'),
                os.path.join(root_dir, 'data', 'custom_crypto_symbols_new.txt'),
                # D'autres emplacements possibles
            ]
            
            all_symbols = set()
            for file_path in crypto_file_paths:
                if os.path.exists(file_path):
                    logger.info(f"Lecture du fichier {file_path}")
                    with open(file_path, 'r') as f:
                        symbols = [line.strip() for line in f.readlines() if line.strip()]
                        all_symbols.update(symbols)
            
            return list(all_symbols)
        
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction des symboles existants: {e}")
            return []

    # Utiliser asyncio pour exécuter la fonction asynchrone
    import asyncio

    async def main():
        # Essayer plusieurs méthodes pour obtenir les cryptos disponibles
        logger.info("Tentative de récupération des cryptomonnaies disponibles...")
        
        # 1. Via le MarketDataService de Mercurio
        mercurio_symbols = await get_available_cryptos()
        
        # 2. Directement via l'API Alpaca
        alpaca_symbols = get_alpaca_cryptos()
        
        # 3. Extraire des fichiers existants
        existing_symbols = extract_existing_crypto_symbols()
        
        # Combiner toutes les sources
        all_symbols = set()
        if mercurio_symbols:
            logger.info(f"Trouvé {len(mercurio_symbols)} cryptomonnaies via MarketDataService")
            all_symbols.update(mercurio_symbols)
            
        if alpaca_symbols:
            logger.info(f"Trouvé {len(alpaca_symbols)} cryptomonnaies via API Alpaca")
            all_symbols.update(alpaca_symbols)
            
        if existing_symbols:
            logger.info(f"Trouvé {len(existing_symbols)} cryptomonnaies dans les fichiers existants")
            all_symbols.update(existing_symbols)
        
        if not all_symbols:
            logger.warning("Aucune cryptomonnaie trouvée par les méthodes automatiques")
            
            # Liste de secours des cryptomonnaies les plus courantes sur Alpaca
            backup_symbols = [
                "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "LTC-USD", "BCH-USD", 
                "LINK-USD", "DOGE-USD", "AVAX-USD", "DOT-USD", "SHIB-USD", "UNI-USD",
                "AAVE-USD", "ALGO-USD", "XTZ-USD", "COMP-USD", "MKR-USD", "SUSHI-USD",
                "YFI-USD", "1INCH-USD", "BAT-USD", "BNT-USD", "GRT-USD", "MATIC-USD"
            ]
            logger.info(f"Utilisation d'une liste de secours de {len(backup_symbols)} cryptomonnaies")
            all_symbols = backup_symbols
        
        # Convertir en liste et trier
        all_symbols = sorted(list(all_symbols))
        
        # Sauvegarder dans un fichier
        save_crypto_list(all_symbols, "all_available_crypto_symbols.txt")
        
        # Générer la ligne pour .env
        env_line = generate_env_format(all_symbols, max_count=30)
        logger.info(f"Ligne pour fichier .env (30 premiers symboles):\n{env_line}")
        
        # Afficher les symboles disponibles
        logger.info(f"Liste des {len(all_symbols)} symboles de cryptomonnaies disponibles:")
        for i, symbol in enumerate(all_symbols):
            logger.info(f"{i+1}. {symbol}")
            
        return all_symbols

    if __name__ == "__main__":
        # Exécution de la fonction principale
        symbols = asyncio.run(main())
        
        # Afficher le résultat final
        if symbols:
            logger.info(f"Récupération réussie de {len(symbols)} cryptomonnaies")
        else:
            logger.error("Échec de la récupération des cryptomonnaies")

except ImportError as e:
    logger.error(f"Erreur d'importation: {e}")
    logger.error("Impossible d'importer les modules de Mercurio. Vérifiez le chemin du projet.")
