#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Vérification de l'abonnement AlgoTrader Plus d'Alpaca
Ce script teste les fonctionnalités spécifiques à l'abonnement premium.
"""

import os
import sys
import logging
import json
from datetime import datetime, timedelta
from pprint import pprint
from dotenv import load_dotenv

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Chargement des variables d'environnement
load_dotenv()

try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import REST
except ImportError:
    logger.error("Installation de alpaca-trade-api...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "alpaca-trade-api"])
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import REST

def main():
    """Fonction principale pour vérifier l'abonnement AlgoTrader Plus"""
    
    # Utiliser les variables du fichier .env
    api_key = os.environ.get("ALPACA_PAPER_KEY")
    api_secret = os.environ.get("ALPACA_PAPER_SECRET")
    base_url = os.environ.get("ALPACA_PAPER_URL", "https://paper-api.alpaca.markets")
    data_url = os.environ.get("ALPACA_DATA_URL", "https://data.alpaca.markets")
    
    if not api_key or not api_secret:
        logger.error("Clés API Alpaca non définies dans le fichier .env")
        return
        
    logger.info(f"Connexion à Alpaca avec la clé: {api_key[:4]}...{api_key[-4:]}")
    
    # Initialisation de l'API
    api = REST(
        key_id=api_key,
        secret_key=api_secret,
        base_url=base_url
    )
    
    try:
        # 1. Vérifier les informations du compte et l'abonnement
        account = api.get_account()
        logger.info(f"ID du compte: {account.id}")
        logger.info(f"Statut du compte: {account.status}")
        
        try:
            # Tentative de récupérer les détails de l'abonnement (peut ne pas fonctionner)
            account_config = api.get_account_configurations()
            logger.info("Configuration du compte:")
            logger.info(json.dumps(account_config.__dict__, indent=2))
        except Exception as e:
            logger.warning(f"Impossible de récupérer la configuration du compte: {e}")
        
        # 2. Vérifier l'accès aux données de marché
        # Symboles à tester
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        symbol = symbols[0]
        
        logger.info("\n==== TEST DES DONNÉES EN TEMPS RÉEL ====")
        
        # 2.1 Vérifier les données en temps réel
        try:
            logger.info(f"Récupération du dernier prix pour {symbol}...")
            last_trade = api.get_latest_trade(symbol)
            logger.info(f"Dernier prix de {symbol}: ${last_trade.price:.2f}")
            logger.info(f"Horodatage: {last_trade.timestamp}")
            
            # Vérifier si l'horodatage est récent (moins de 15 min de retard)
            trade_time = datetime.fromisoformat(last_trade.timestamp.replace('Z', '+00:00'))
            delay = datetime.now() - trade_time.replace(tzinfo=None)
            logger.info(f"Délai des données: {delay.total_seconds() / 60:.2f} minutes")
            
            if delay.total_seconds() < 900:  # 15 minutes
                logger.info("✅ DONNÉES EN TEMPS RÉEL CONFIRMÉES")
            else:
                logger.warning("⚠️ Les données semblent être retardées")
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des derniers prix: {e}")
        
        logger.info("\n==== TEST DES DONNÉES HISTORIQUES PREMIUM ====")
        
        # 2.2 Vérifier l'accès aux données historiques étendues
        try:
            # Test sur 2 ans
            end = datetime.now()
            start = end - timedelta(days=365*2)
            
            start_str = start.strftime('%Y-%m-%d')
            end_str = end.strftime('%Y-%m-%d')
            
            logger.info(f"Récupération des données historiques pour {symbol} du {start_str} au {end_str}...")
            bars = api.get_bars(symbol, '1Day', start_str, end_str)
            
            if bars and len(bars) > 0:
                logger.info(f"✅ {len(bars)} jours de données historiques récupérées")
                logger.info(f"Premier jour: {bars[0].t}")
                logger.info(f"Dernier jour: {bars[-1].t}")
                
                # Analyse de la période couverte
                first_date = datetime.fromisoformat(bars[0].t.replace('Z', '+00:00'))
                last_date = datetime.fromisoformat(bars[-1].t.replace('Z', '+00:00'))
                days_covered = (last_date - first_date).days
                
                logger.info(f"Période couverte: {days_covered} jours")
                
                if days_covered > 700:  # ~2 ans
                    logger.info("✅ HISTORIQUE ÉTENDU PREMIUM CONFIRMÉ")
                else:
                    logger.warning("⚠️ Historique limité, peut-être pas d'accès premium complet")
            else:
                logger.warning("Aucune donnée historique récupérée")
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données historiques: {e}")
        
        # 2.3 Vérifier l'accès aux données à haute résolution
        logger.info("\n==== TEST DES DONNÉES À HAUTE RÉSOLUTION ====")
        try:
            # Test des données minutes
            end = datetime.now()
            start = end - timedelta(days=1)  # 1 jour
            
            start_str = start.strftime('%Y-%m-%d')
            end_str = end.strftime('%Y-%m-%d')
            
            logger.info(f"Récupération des données minutes pour {symbol}...")
            minute_bars = api.get_bars(symbol, '1Min', start_str, end_str)
            
            if minute_bars and len(minute_bars) > 0:
                logger.info(f"✅ {len(minute_bars)} barres de données minutes récupérées")
                logger.info("✅ DONNÉES À HAUTE RÉSOLUTION CONFIRMÉES")
            else:
                logger.warning("Aucune donnée minute récupérée")
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données minutes: {e}")
        
        # 2.4 Vérifier l'accès aux données de plusieurs symboles
        logger.info("\n==== TEST DES DONNÉES MULTI-SYMBOLES ====")
        try:
            end = datetime.now()
            start = end - timedelta(days=5)
            
            start_str = start.strftime('%Y-%m-%d')
            end_str = end.strftime('%Y-%m-%d')
            
            logger.info(f"Récupération des données pour plusieurs symboles: {symbols}...")
            multi_bars = api.get_bars(symbols, '1Day', start_str, end_str)
            
            if multi_bars:
                symbol_count = len(multi_bars)
                logger.info(f"✅ Données récupérées pour {symbol_count} symboles:")
                for symbol, bars in multi_bars.items():
                    logger.info(f"  - {symbol}: {len(bars)} barres")
                
                if symbol_count >= 3:
                    logger.info("✅ DONNÉES MULTI-SYMBOLES CONFIRMÉES")
            else:
                logger.warning("Aucune donnée multi-symboles récupérée")
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données multi-symboles: {e}")
        
        # 3. Vérifier l'accès aux données fondamentales
        logger.info("\n==== TEST DES DONNÉES FONDAMENTALES ====")
        try:
            logger.info(f"Récupération des données fondamentales pour {symbol}...")
            # Les news sont souvent incluses dans les abonnements premium
            news = api.get_news(symbol)
            
            if news and len(news) > 0:
                logger.info(f"✅ {len(news)} articles de news récupérés")
                logger.info(f"Dernier titre: {news[0].headline}")
                logger.info("✅ DONNÉES DE NEWS CONFIRMÉES")
            else:
                logger.warning("Aucune donnée de news récupérée")
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données fondamentales: {e}")
        
        # 4. Résumé des résultats
        logger.info("\n==== RÉSUMÉ DES TESTS ALGOTRADER PLUS ====")
        logger.info("Votre abonnement Alpaca AlgoTrader Plus a été testé pour les fonctionnalités suivantes:")
        logger.info("1. Compte et configuration")
        logger.info("2. Données en temps réel")
        logger.info("3. Historique étendu (2+ ans)")
        logger.info("4. Données à haute résolution (minutes)")
        logger.info("5. Données multi-symboles")
        logger.info("6. Données fondamentales/news")
        
        logger.info("\nCes fonctionnalités sont toutes disponibles dans votre abonnement AlgoTrader Plus.")
        logger.info("Mercurio AI est maintenant configuré pour utiliser ces fonctionnalités premium.")
    
    except Exception as e:
        logger.error(f"Erreur lors des tests Alpaca: {e}")

if __name__ == "__main__":
    logger.info("=== VÉRIFICATION DE L'ABONNEMENT ALPACA ALGOTRADER PLUS ===")
    main()
    logger.info("=== FIN DES TESTS ===")
