#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Alpaca Premium Subscription Test

Vérifie spécifiquement les fonctionnalités premium de votre abonnement Alpaca à 100$/mois.
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Configure logging to be plus lisible
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import REST
    logger.info("✅ alpaca-trade-api importé avec succès")
except ImportError:
    logger.error("❌ Erreur d'importation de alpaca-trade-api. Installation...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "alpaca-trade-api"])
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import REST
    logger.info("✅ alpaca-trade-api installé et importé avec succès")

def test_premium_features():
    """Teste les fonctionnalités spécifiques aux abonnements premium d'Alpaca"""
    
    # Charger les variables d'environnement
    load_dotenv()
    
    # Tester à la fois le mode PAPER et LIVE si possible
    results = {}
    
    for mode in ["paper", "live"]:
        logger.info(f"\n{'='*60}")
        logger.info(f"TEST DU MODE {mode.upper()}")
        logger.info(f"{'='*60}\n")
        
        # Récupérer les informations d'authentification appropriées
        if mode == "paper":
            api_key = os.environ.get("ALPACA_PAPER_KEY")
            api_secret = os.environ.get("ALPACA_PAPER_SECRET")
            base_url = os.environ.get("ALPACA_PAPER_URL", "https://paper-api.alpaca.markets")
            data_url = os.environ.get("ALPACA_DATA_URL", "https://data.alpaca.markets")
        else:
            api_key = os.environ.get("ALPACA_LIVE_KEY")
            api_secret = os.environ.get("ALPACA_LIVE_SECRET")
            base_url = os.environ.get("ALPACA_LIVE_URL", "https://api.alpaca.markets")
            data_url = os.environ.get("ALPACA_DATA_URL", "https://data.alpaca.markets")
        
        results[mode] = {
            "connection": False,
            "account_info": False,
            "market_data": {
                "daily": False,
                "minute": False,
                "realtime": False,
                "extended_history": False,
                "multiple_symbols": False,
            },
            "news_data": False,
            "fundamental_data": False,
            "subscription_tier": "Unknown"
        }
        
        try:
            # Initialiser l'API Alpaca
            api = REST(
                key_id=api_key,
                secret_key=api_secret,
                base_url=base_url,
                data_url=data_url
            )
            results[mode]["connection"] = True
            logger.info(f"✅ Connecté à l'API Alpaca en mode {mode.upper()}")
            
            # 1. Test des informations de compte
            try:
                account = api.get_account()
                results[mode]["account_info"] = True
                logger.info(f"✅ Informations du compte: ID={account.id}, Status={account.status}")
                logger.info(f"   Valeur portefeuille: ${float(account.portfolio_value):.2f}")
                
                # Essayer de déterminer le niveau d'abonnement
                try:
                    if hasattr(account, 'subscription_status'):
                        results[mode]["subscription_tier"] = account.subscription_status
                        logger.info(f"✅ Niveau d'abonnement: {account.subscription_status}")
                    else:
                        logger.info("ℹ️ Impossible de déterminer le niveau d'abonnement directement")
                except:
                    pass
            except Exception as e:
                logger.error(f"❌ Erreur lors de la récupération des informations du compte: {e}")
            
            # 2. Test des données de marché historiques (journalières)
            logger.info("\n----- TEST DES DONNÉES DE MARCHÉ -----")
            symbol = "AAPL"
            end_date = datetime.now()
            
            # 2.1 Test données journalières sur 5 jours (devrait fonctionner même sans abonnement premium)
            try:
                start_date = end_date - timedelta(days=5)
                start_str = start_date.date().isoformat()
                end_str = end_date.date().isoformat()
                
                logger.info(f"Récupération des données journalières pour {symbol} du {start_str} au {end_str}...")
                daily_bars = api.get_bars(symbol, "1Day", start_str, end_str)
                
                if len(daily_bars) > 0:
                    results[mode]["market_data"]["daily"] = True
                    logger.info(f"✅ {len(daily_bars)} barres journalières récupérées")
                    logger.info(f"   Dernier prix de clôture: ${daily_bars[-1].c:.2f}")
                else:
                    logger.warning(f"⚠️ Aucune donnée journalière récupérée pour {symbol}")
            except Exception as e:
                logger.error(f"❌ Erreur lors de la récupération des données journalières: {e}")
            
            # 2.2 Test données minutes (souvent limité aux abonnements premium)
            try:
                start_date = end_date - timedelta(days=1)
                start_str = start_date.date().isoformat()
                end_str = end_date.date().isoformat()
                
                logger.info(f"Récupération des données minutes pour {symbol} des dernières 24h...")
                minute_bars = api.get_bars(symbol, "1Min", start_str, end_str)
                
                if len(minute_bars) > 0:
                    results[mode]["market_data"]["minute"] = True
                    logger.info(f"✅ {len(minute_bars)} barres minutes récupérées")
                    logger.info(f"   Première barre: {minute_bars[0].t}")
                    logger.info(f"   Dernière barre: {minute_bars[-1].t}")
                else:
                    logger.warning(f"⚠️ Aucune donnée minute récupérée pour {symbol}")
            except Exception as e:
                logger.error(f"❌ Erreur lors de la récupération des données minutes: {e}")
            
            # 2.3 Test historique étendu (2+ ans, souvent limité aux abonnements premium)
            try:
                start_date = end_date - timedelta(days=365*2 + 30)  # ~2 ans et 1 mois
                start_str = start_date.date().isoformat()
                end_str = end_date.date().isoformat()
                
                logger.info(f"Récupération de l'historique étendu pour {symbol} (2+ ans)...")
                long_bars = api.get_bars(symbol, "1Day", start_str, end_str)
                
                if len(long_bars) > 0:
                    days_of_data = (datetime.fromisoformat(long_bars[-1].t) - 
                                  datetime.fromisoformat(long_bars[0].t)).days
                    
                    if days_of_data > 365*2:
                        results[mode]["market_data"]["extended_history"] = True
                        logger.info(f"✅ {len(long_bars)} barres d'historique étendu récupérées")
                        logger.info(f"   Couvrant {days_of_data} jours de données")
                    else:
                        logger.warning(f"⚠️ Historique limité à {days_of_data} jours (< 2 ans)")
                else:
                    logger.warning(f"⚠️ Aucune donnée d'historique étendu récupérée pour {symbol}")
            except Exception as e:
                logger.error(f"❌ Erreur lors de la récupération de l'historique étendu: {e}")
            
            # 2.4 Test données pour plusieurs symboles simultanément
            try:
                symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
                logger.info(f"Récupération des données pour plusieurs symboles: {symbols}...")
                
                start_date = end_date - timedelta(days=5)
                start_str = start_date.date().isoformat()
                end_str = end_date.date().isoformat()
                
                multi_bars = api.get_bars(symbols, "1Day", start_str, end_str)
                
                if multi_bars and len(multi_bars) > 0:
                    results[mode]["market_data"]["multiple_symbols"] = True
                    logger.info(f"✅ Données récupérées pour plusieurs symboles:")
                    for symbol, bars in multi_bars.items():
                        logger.info(f"   {symbol}: {len(bars)} barres")
                else:
                    logger.warning("⚠️ Aucune donnée récupérée pour les multiples symboles")
            except Exception as e:
                logger.error(f"❌ Erreur lors de la récupération des données multi-symboles: {e}")
            
            # 3. Test des données de news (souvent limité aux abonnements premium)
            logger.info("\n----- TEST DES DONNÉES DE NEWS -----")
            try:
                logger.info(f"Récupération des news pour {symbol}...")
                news = api.get_news(symbol)
                
                if news and len(news) > 0:
                    results[mode]["news_data"] = True
                    logger.info(f"✅ {len(news)} articles de news récupérés")
                    logger.info(f"   Dernier titre: {news[0].headline}")
                    logger.info(f"   Source: {news[0].source}")
                else:
                    logger.warning(f"⚠️ Aucune news récupérée pour {symbol}")
            except Exception as e:
                logger.error(f"❌ Erreur lors de la récupération des news: {e}")
            
            # Résumé des tests
            logger.info("\n----- RÉSUMÉ DES TESTS -----")
            
            if results[mode]["connection"]:
                logger.info(f"✅ Connexion au mode {mode.upper()}: Réussie")
            else:
                logger.error(f"❌ Connexion au mode {mode.upper()}: Échec")
                
            if results[mode]["account_info"]:
                logger.info(f"✅ Informations du compte: Disponibles")
            else:
                logger.error(f"❌ Informations du compte: Non disponibles")
            
            logger.info(f"Données de marché:")
            for data_type, success in results[mode]["market_data"].items():
                status = "✅ Disponible" if success else "❌ Non disponible"
                logger.info(f"  - {data_type}: {status}")
            
            news_status = "✅ Disponible" if results[mode]["news_data"] else "❌ Non disponible"
            logger.info(f"Données de news: {news_status}")
            
            # Analyse de l'abonnement
            premium_features = sum([
                results[mode]["market_data"]["minute"],
                results[mode]["market_data"]["extended_history"],
                results[mode]["market_data"]["multiple_symbols"],
                results[mode]["news_data"]
            ])
            
            if premium_features >= 3:
                logger.info("🌟 Votre compte semble avoir un abonnement premium!")
            elif premium_features >= 1:
                logger.info("⭐ Votre compte semble avoir certaines fonctionnalités premium.")
            else:
                logger.warning("⚠️ Votre compte ne semble pas avoir d'abonnement premium.")
            
        except Exception as e:
            logger.error(f"❌ Erreur générale lors du test du mode {mode}: {e}")
    
    return results

if __name__ == "__main__":
    logger.info("\n🚀 DÉMARRAGE DU TEST D'ABONNEMENT PREMIUM ALPACA")
    logger.info("Ce test va vérifier si votre abonnement Alpaca à 100$/mois fonctionne correctement")
    
    results = test_premium_features()
    
    logger.info("\n\n🏁 TEST TERMINÉ")
    logger.info("Récapitulatif des fonctionnalités premium détectées:")
    
    for mode in results:
        premium_count = sum([
            results[mode]["market_data"]["minute"],
            results[mode]["market_data"]["extended_history"],
            results[mode]["market_data"]["multiple_symbols"],
            results[mode]["news_data"]
        ])
        
        if premium_count >= 3:
            status = "🌟 PREMIUM"
        elif premium_count >= 1:
            status = "⭐ PARTIEL"
        else:
            status = "❌ STANDARD"
            
        logger.info(f"Mode {mode.upper()}: {status} ({premium_count}/4 fonctionnalités premium)")
    
    logger.info("\nSi vous ne voyez pas toutes les fonctionnalités premium, vérifiez que:")
    logger.info("1. Votre abonnement est bien activé sur le compte Alpaca")
    logger.info("2. Les clés API utilisées correspondent au compte avec l'abonnement")
    logger.info("3. L'abonnement inclut bien les fonctionnalités testées")
