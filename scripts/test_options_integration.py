#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test d'intégration du trading d'options avec Alpaca AlgoTrader Plus

Ce script teste la fonctionnalité complète du trading d'options dans Mercurio AI
en mode paper trading. Il vérifie toutes les composantes de l'intégration avec Alpaca.
"""

import os
import sys
import asyncio
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd
import json

# Ajouter le répertoire parent au chemin pour importer les modules de app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.trading import TradingService
from app.services.market_data import MarketDataService
from app.services.options_service import OptionsService
from app.strategies.options_strategy import OptionsStrategy, TimeFrame
from app.db.models import TradeAction

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Charger les variables d'environnement
load_dotenv()

async def test_options_service_integration():
    """Test d'intégration complet du service de trading d'options"""
    
    logger.info("=== DÉMARRAGE DU TEST D'INTÉGRATION DU TRADING D'OPTIONS ===")
    
    try:
        # 1. Initialiser les services
        logger.info("Initialisation des services...")
        trading_service = TradingService(is_paper=True)
        market_data = MarketDataService()
        options_service = OptionsService(
            trading_service=trading_service,
            market_data_service=market_data
        )
        
        # 2. Vérifier la connexion à Alpaca
        logger.info("Vérification de la connexion à Alpaca...")
        try:
            account = await trading_service.get_account_info()
            logger.info(f"✅ Connexion réussie à Alpaca. Mode: {'PAPER' if trading_service.base_url == 'https://paper-api.alpaca.markets' else 'LIVE'}")
            logger.info(f"   ID du compte: {account.get('id')}")
            logger.info(f"   Status: {account.get('status')}")
            logger.info(f"   Valeur du portefeuille: ${float(account.get('portfolio_value', 0)):.2f}")
        except Exception as e:
            logger.error(f"❌ Erreur de connexion à Alpaca: {e}")
            return False
            
        # 3. Tester l'accès aux données du marché
        logger.info("\nTest d'accès aux données du marché...")
        symbol = "AAPL"
        try:
            price = await market_data.get_latest_price(symbol)
            logger.info(f"✅ Prix actuel de {symbol}: ${price:.2f}")
        except Exception as e:
            logger.error(f"❌ Erreur d'accès aux données du marché: {e}")
            
        # 4. Tester la récupération des options disponibles
        logger.info("\nTest de récupération des options disponibles...")
        try:
            options = await options_service.get_available_options(symbol)
            if options and len(options) > 0:
                logger.info(f"✅ {len(options)} contrats d'options trouvés pour {symbol}")
                logger.info(f"   Exemple: {options[0]}")
            else:
                logger.warning(f"⚠️ Aucune option trouvée pour {symbol}")
        except Exception as e:
            logger.error(f"❌ Erreur de récupération des options: {e}")
            
        # 5. Tester les suggestions de stratégies d'options
        logger.info("\nTest des suggestions de stratégies d'options...")
        try:
            # Simuler une prédiction de hausse pour tester
            price_prediction = {
                "action": TradeAction.BUY,
                "confidence": 0.8,
                "price_target": price * 1.05,  # Cible +5%
                "time_horizon_days": 30
            }
            
            strategies = await options_service.suggest_option_strategies(
                symbol=symbol,
                price_prediction=price_prediction,
                risk_profile="moderate"
            )
            
            if strategies and len(strategies) > 0:
                logger.info(f"✅ {len(strategies)} stratégies d'options suggérées")
                for i, strategy in enumerate(strategies[:3], 1):  # Afficher jusqu'à 3 stratégies
                    logger.info(f"   {i}. {strategy['name']}: {strategy['option_type']} à ${strategy.get('strike', 0):.2f}")
            else:
                logger.warning("⚠️ Aucune stratégie d'options suggérée")
        except Exception as e:
            logger.error(f"❌ Erreur de suggestion de stratégies: {e}")
        
        # 6. Tester la génération de signal d'options via la stratégie
        logger.info("\nTest de la génération de signal avec OptionsStrategy...")
        try:
            # Créer une stratégie d'options basée sur une stratégie existante
            options_strategy = OptionsStrategy(
                options_service=options_service,
                base_strategy_name="TransformerStrategy",
                risk_profile="moderate"
            )
            
            # Données de test qui incluent une prédiction de la stratégie de base
            test_data = {
                "close": price,
                "TransformerStrategy_prediction": {
                    "action": TradeAction.BUY,
                    "confidence": 0.85,
                    "price_target": price * 1.06,
                    "time_horizon_days": 30
                }
            }
            
            signal = await options_strategy.generate_signal(symbol, test_data, TimeFrame.DAY)
            
            if signal and "action" in signal:
                logger.info(f"✅ Signal d'options généré: {signal['action']} {signal.get('option_type', '')} " +
                          f"à ${signal.get('strike', 0):.2f}, expiration {signal.get('expiration', '')}")
            else:
                logger.warning("⚠️ Aucun signal d'options généré")
        except Exception as e:
            logger.error(f"❌ Erreur de génération de signal: {e}")
        
        # 7. Tester l'information sur les positions actuelles
        logger.info("\nTest de récupération des positions d'options...")
        try:
            positions = await options_service.get_all_option_positions()
            logger.info(f"✅ {len(positions)} positions d'options trouvées")
        except Exception as e:
            logger.error(f"❌ Erreur de récupération des positions: {e}")
        
        # 8. Calculer des métriques d'options
        logger.info("\nTest de calcul des métriques d'options...")
        try:
            if options and len(options) > 0:
                # Utiliser le premier contrat d'options comme exemple
                option_data = options[0]
                metrics = await options_service.calculate_option_metrics(option_data)
                
                if metrics:
                    logger.info(f"✅ Métriques calculées: ")
                    for key, value in metrics.items():
                        logger.info(f"   {key}: {value}")
                else:
                    logger.warning("⚠️ Aucune métrique calculée")
            else:
                logger.warning("⚠️ Aucune option disponible pour le calcul des métriques")
        except Exception as e:
            logger.error(f"❌ Erreur de calcul des métriques: {e}")
        
        # 9. OPTIONNEL: Placer un ordre d'option test
        # ⚠️ Attention: Ceci placera réellement un ordre en mode paper trading
        # Note: Commentez ce bloc si vous ne voulez pas placer d'ordre de test
        """
        logger.info("\nTest de placement d'un ordre d'option... (PAPER UNIQUEMENT)")
        try:
            if options and len(options) > 0 and trading_service.is_paper:
                # Trouver une option avec un prix raisonnable pour tester
                test_option = next((opt for opt in options if 
                                  opt.get('option_type') == 'call' and 
                                  1.0 <= float(opt.get('ask', 1000)) <= 5.0), None)
                
                if test_option:
                    logger.info(f"Placement d'un ordre d'option test pour {test_option['symbol']}")
                    
                    result = await options_service.execute_option_trade(
                        option_symbol=test_option['symbol'],
                        action=TradeAction.BUY,
                        quantity=1,  # Acheter 1 contrat seulement
                        order_type="market",
                        strategy_name="OptionTestStrategy"
                    )
                    
                    if result and result.get('status') == 'success':
                        logger.info(f"✅ Ordre test placé avec succès: {result.get('order', {}).get('id')}")
                    else:
                        logger.warning(f"⚠️ Échec du placement d'ordre: {result}")
                else:
                    logger.info("Aucune option appropriée trouvée pour le test d'ordre")
            else:
                logger.info("Test d'ordre ignoré (mode live ou aucune option disponible)")
        except Exception as e:
            logger.error(f"❌ Erreur lors du placement d'ordre: {e}")
        """
        
        # Conclusion
        logger.info("\n=== TEST D'INTÉGRATION TERMINÉ ===")
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors du test d'intégration: {e}")
        return False

async def test_options_backtesting(symbol="AAPL", days=30):
    """Test de backtesting des stratégies d'options"""
    
    logger.info("=== DÉMARRAGE DU BACKTESTING DES STRATÉGIES D'OPTIONS ===")
    
    try:
        # 1. Initialiser les services
        trading_service = TradingService(is_paper=True)
        market_data = MarketDataService()
        options_service = OptionsService(
            trading_service=trading_service,
            market_data_service=market_data
        )
        
        # 2. Configurer les paramètres de backtest
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        logger.info(f"Période de backtest: {start_date.date()} à {end_date.date()}")
        
        # 3. Récupérer les données historiques
        logger.info(f"Récupération des données historiques pour {symbol}...")
        historical_data = await market_data.get_historical_data(
            symbol=symbol,
            start_date=start_date,  # Passer l'objet datetime directement
            end_date=end_date       # Passer l'objet datetime directement
        )
        
        if historical_data is None or len(historical_data) < 5:
            logger.error(f"Données historiques insuffisantes pour {symbol}")
            return
            
        logger.info(f"✅ {len(historical_data)} jours de données historiques récupérés")
        
        # 4. Créer la stratégie d'options
        options_strategy = OptionsStrategy(
            options_service=options_service,
            base_strategy_name="TransformerStrategy",
            risk_profile="moderate"
        )
        
        # 5. Exécuter le backtest
        logger.info("Exécution du backtest...")
        
        # Transformer les données en DataFrame si ce n'est pas déjà le cas
        if not isinstance(historical_data, pd.DataFrame):
            historical_data = pd.DataFrame(historical_data)
        
        # Ajouter des prédictions simulées pour le backtest
        historical_data['TransformerStrategy_prediction'] = None
        
        # Simuler des prédictions pour chaque jour
        predictions = []
        for i in range(len(historical_data)):
            row = historical_data.iloc[i]
            # Simuler une prédiction basée sur le mouvement réel du prix
            if i < len(historical_data) - 1:
                next_close = historical_data.iloc[i+1]['close']
                price_change = next_close / row['close'] - 1
                
                if price_change > 0.005:  # +0.5%
                    action = TradeAction.BUY
                    confidence = min(price_change * 10, 0.9)  # Calibrer la confiance
                elif price_change < -0.005:  # -0.5%
                    action = TradeAction.SELL
                    confidence = min(abs(price_change) * 10, 0.9)
                else:
                    action = TradeAction.HOLD
                    confidence = 0.6
                
                prediction = {
                    "action": action,
                    "confidence": confidence,
                    "price_target": row['close'] * (1 + price_change * 2),
                    "time_horizon_days": 5
                }
            else:
                # Pour le dernier jour, utiliser HOLD
                prediction = {
                    "action": TradeAction.HOLD,
                    "confidence": 0.5,
                    "price_target": row['close'],
                    "time_horizon_days": 5
                }
            
            predictions.append(prediction)
        
        # Créer une nouvelle colonne pour les prédictions
        # Utiliser une approche différente pour éviter les problèmes de types
        for i in range(len(historical_data)):
            historical_data.at[historical_data.index[i], 'TransformerStrategy_prediction'] = predictions[i]
        
        # Exécuter le backtest
        backtest_result = await options_strategy.backtest(
            data=historical_data,
            initial_capital=10000.0,
            symbol=symbol
        )
        
        if backtest_result:
            logger.info(f"✅ Backtest terminé avec succès")
            logger.info(f"   Rendement total: {backtest_result.get('total_return', 0) * 100:.2f}%")
            logger.info(f"   Capital final: ${backtest_result.get('final_capital', 0):.2f}")
            logger.info(f"   Nombre de trades: {backtest_result.get('num_trades', 0)}")
            
            # Sauvegarder les résultats du backtest
            os.makedirs("results", exist_ok=True)
            result_file = f"results/options_backtest_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(result_file, 'w') as f:
                # Convertir les objets non-sérialisables
                result_copy = {k: v for k, v in backtest_result.items() if k != 'trades_data'}
                json.dump(result_copy, f, default=str, indent=2)
            
            logger.info(f"   Résultats sauvegardés dans {result_file}")
        else:
            logger.warning("⚠️ Backtest échoué ou aucun résultat")
        
        logger.info("=== BACKTESTING TERMINÉ ===")
        
    except Exception as e:
        logger.error(f"Erreur lors du backtesting: {e}")

async def explore_option_strategies(symbol="AAPL"):
    """Explorer différentes combinaisons de stratégies d'options"""
    
    logger.info("=== EXPLORATION DES STRATÉGIES D'OPTIONS ===")
    
    try:
        # 1. Initialiser les services
        trading_service = TradingService(is_paper=True)
        market_data = MarketDataService()
        options_service = OptionsService(
            trading_service=trading_service,
            market_data_service=market_data
        )
        
        # 2. Récupérer le prix actuel
        price = await market_data.get_latest_price(symbol)
        logger.info(f"Prix actuel de {symbol}: ${price:.2f}")
        
        # 3. Explorer différentes stratégies pour différents scénarios
        scenarios = [
            {"name": "Très haussier", "move": 0.10, "confidence": 0.9, "days": 45, "risk": "aggressive"},
            {"name": "Haussier", "move": 0.05, "confidence": 0.8, "days": 30, "risk": "moderate"},
            {"name": "Légèrement haussier", "move": 0.02, "confidence": 0.7, "days": 21, "risk": "conservative"},
            {"name": "Neutre", "move": 0.00, "confidence": 0.6, "days": 14, "risk": "moderate"},
            {"name": "Légèrement baissier", "move": -0.02, "confidence": 0.7, "days": 21, "risk": "conservative"},
            {"name": "Baissier", "move": -0.05, "confidence": 0.8, "days": 30, "risk": "moderate"},
            {"name": "Très baissier", "move": -0.10, "confidence": 0.9, "days": 45, "risk": "aggressive"}
        ]
        
        strategy_results = []
        
        for scenario in scenarios:
            logger.info(f"\n--- Scénario: {scenario['name']} ---")
            
            # Créer la prédiction en fonction du scénario
            action = TradeAction.BUY if scenario["move"] > 0 else TradeAction.SELL if scenario["move"] < 0 else TradeAction.HOLD
            
            price_prediction = {
                "action": action,
                "confidence": scenario["confidence"],
                "price_target": price * (1 + scenario["move"]),
                "time_horizon_days": scenario["days"]
            }
            
            # Obtenir les suggestions de stratégies
            strategies = await options_service.suggest_option_strategies(
                symbol=symbol,
                price_prediction=price_prediction,
                risk_profile=scenario["risk"]
            )
            
            # Enregistrer les résultats
            result = {
                "scenario": scenario["name"],
                "price_prediction": price_prediction,
                "risk_profile": scenario["risk"],
                "strategies": strategies
            }
            strategy_results.append(result)
            
            # Afficher les stratégies recommandées
            if strategies and len(strategies) > 0:
                logger.info(f"Stratégies recommandées ({len(strategies)}):")
                for i, strategy in enumerate(strategies, 1):
                    confidence_match = strategy.get('confidence_match', 0)
                    emoji = "🔥" if confidence_match > 80 else "✅" if confidence_match > 70 else "⚠️"
                    logger.info(f"{emoji} {i}. {strategy['name']} - Strike: ${strategy.get('strike', 0):.2f}, " +
                              f"Exp: {strategy.get('expiration', 'N/A')}, Confiance: {confidence_match:.1f}%")
                    logger.info(f"   Description: {strategy.get('description', 'N/A')}")
                    logger.info(f"   Risque: {strategy.get('risk_rating', 'N/A')}, " +
                              f"Gain max: {strategy.get('max_gain', 'N/A')}, " +
                              f"Perte max: {strategy.get('max_loss', 'N/A')}")
            else:
                logger.warning("Aucune stratégie recommandée")
        
        # Sauvegarder les résultats de l'exploration
        os.makedirs("results", exist_ok=True)
        result_file = f"results/options_strategies_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(result_file, 'w') as f:
            json.dump(strategy_results, f, default=str, indent=2)
        
        logger.info(f"\nRésultats sauvegardés dans {result_file}")
        logger.info("=== EXPLORATION TERMINÉE ===")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'exploration des stratégies: {e}")

async def main():
    """Fonction principale exécutant tous les tests"""
    
    print("\n" + "="*80)
    print("        TEST COMPLET DU TRADING D'OPTIONS AVEC ALPACA ALGOTRADER PLUS")
    print("="*80)
    
    print("\nCe test va vérifier que toutes les fonctionnalités de trading d'options ")
    print("fonctionnent correctement avec votre abonnement Alpaca AlgoTrader Plus.\n")
    
    # Si aucun argument n'est passé, demander quoi tester
    if len(sys.argv) == 1:
        print("Choisissez les tests à exécuter:")
        print("1. Test d'intégration (vérifier que tout fonctionne)")
        print("2. Backtesting des stratégies d'options")
        print("3. Explorer les stratégies d'options")
        print("4. Exécuter tous les tests")
        print("0. Quitter")
        
        choice = input("\nVotre choix (1-4): ").strip()
        
        if choice == "0":
            print("Au revoir!")
            return
        elif choice == "1":
            await test_options_service_integration()
        elif choice == "2":
            symbol = input("Symbole à utiliser pour le backtest [AAPL]: ").strip() or "AAPL"
            days_str = input("Nombre de jours d'historique [30]: ").strip() or "30"
            days = int(days_str)
            await test_options_backtesting(symbol, days)
        elif choice == "3":
            symbol = input("Symbole à utiliser pour l'exploration [AAPL]: ").strip() or "AAPL"
            await explore_option_strategies(symbol)
        elif choice == "4":
            await test_options_service_integration()
            await test_options_backtesting()
            await explore_option_strategies()
        else:
            print("Choix non valide. Au revoir!")
    else:
        # Exécuter selon les arguments de ligne de commande
        if "--integration" in sys.argv:
            await test_options_service_integration()
        
        if "--backtest" in sys.argv:
            symbol = "AAPL"
            days = 30
            
            # Extraire le symbole et les jours s'ils sont spécifiés
            for i, arg in enumerate(sys.argv):
                if arg == "--symbol" and i+1 < len(sys.argv):
                    symbol = sys.argv[i+1]
                if arg == "--days" and i+1 < len(sys.argv):
                    days = int(sys.argv[i+1])
            
            await test_options_backtesting(symbol, days)
        
        if "--explore" in sys.argv:
            symbol = "AAPL"
            
            # Extraire le symbole s'il est spécifié
            for i, arg in enumerate(sys.argv):
                if arg == "--symbol" and i+1 < len(sys.argv):
                    symbol = sys.argv[i+1]
            
            await explore_option_strategies(symbol)
        
        # Si aucun argument spécifique, exécuter tous les tests
        if not any(arg in sys.argv for arg in ["--integration", "--backtest", "--explore"]):
            await test_options_service_integration()
            await test_options_backtesting()
            await explore_option_strategies()

if __name__ == "__main__":
    # Mettre en place le gestionnaire de signaux pour l'arrêt propre
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Tests interrompus par l'utilisateur.")
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution des tests: {e}")
