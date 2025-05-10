#!/usr/bin/env python
"""
Mercurio AI - Long-Term Backtesting Script

Ce script effectue un backtesting de stratégies de trading sur une longue période
pour simuler des performances réelles dans diverses conditions de marché.

Caractéristiques:
- Test sur plusieurs années de données
- Comparaison de plusieurs stratégies
- Analyse détaillée des performances
- Visualisations des résultats
"""
import os
import asyncio
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import json

# Import des services et composants de Mercurio AI
from app.services.market_data import MarketDataService
from app.services.backtesting import BacktestingService
from app.services.strategy_manager import StrategyManager
from app.db.models import TradeAction

# Configuration du logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[
                       logging.FileHandler("logs/backtest_long_term.log"),
                       logging.StreamHandler()
                   ])
logger = logging.getLogger(__name__)

# Création des dossiers nécessaires
os.makedirs("./data", exist_ok=True)
os.makedirs("./models", exist_ok=True)
os.makedirs("./logs", exist_ok=True)
os.makedirs("./results", exist_ok=True)

# Configuration par défaut
DEFAULT_CONFIG = {
    "symbols": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
    "start_date": (datetime.now() - timedelta(days=365*5)).strftime("%Y-%m-%d"),  # 5 ans
    "end_date": datetime.now().strftime("%Y-%m-%d"),  # Date du jour par défaut
    "strategies": ["MovingAverageStrategy", "LSTMPredictorStrategy"],
    "initial_capital": 100000.0,
    "transaction_fees": {
        "percentage": 0.001,  # 0.1% de frais proportionnels sur chaque transaction
        "fixed": 0.0,       # Frais fixes par transaction en USD
        "minimum": 0.0      # Frais minimum par transaction en USD
    },
    "strategy_params": {
        "MovingAverageStrategy": {
            "short_window": 20,
            "long_window": 50,
            "use_ml": True
        },
        "LSTMPredictorStrategy": {
            "epochs": 50,
            "batch_size": 32,
            "sequence_length": 60
        }
    }
}

class LongTermBacktester:
    """
    Classe pour effectuer des backtests sur le long terme avec différentes stratégies
    """
    
    def __init__(self, config=None):
        """
        Initialise le backtester avec la configuration spécifiée
        
        Args:
            config: Dictionnaire de configuration (utilise DEFAULT_CONFIG si None)
        """
        self.config = config or DEFAULT_CONFIG
        self.market_data = MarketDataService()
        self.backtesting_service = BacktestingService()
        self.strategy_manager = StrategyManager()
        
        # Parse les dates
        self.start_date = datetime.strptime(self.config["start_date"], "%Y-%m-%d")
        self.end_date = datetime.strptime(self.config["end_date"], "%Y-%m-%d")
        
        # Résultats
        self.results = {}
        self._comparison_df = None
        
    async def load_data(self, symbol):
        """
        Charge les données historiques pour un symbole
        
        Args:
            symbol: Le symbole boursier (ex: 'AAPL')
            
        Returns:
            DataFrame contenant les données historiques
        """
        logger.info(f"Chargement des données pour {symbol} de {self.start_date} à {self.end_date}...")
        try:
            # Formatage des dates au format attendu par l'API
            start_str = self.start_date.strftime("%Y-%m-%d")
            end_str = self.end_date.strftime("%Y-%m-%d")
            data = await self.market_data.get_historical_data(symbol, start_str, end_str)
            logger.info(f"Obtenu {len(data)} points de données pour {symbol}")
            return data
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données pour {symbol}: {e}")
            return None
            
    def apply_transaction_fees(self, backtest_result: Dict[str, Any], transaction_fees: Dict[str, float]) -> Dict[str, Any]:
        """
        Applique les frais de transaction aux résultats d'un backtest
        
        Args:
            backtest_result: Résultats de backtest original
            transaction_fees: Configuration des frais (pourcentage, fixe, minimum)
            
        Returns:
            Résultats de backtest avec frais appliqués
        """
        # Si aucun frais spécifié, utiliser les frais par défaut
        if not transaction_fees:
            transaction_fees = {
                "percentage": 0.001,  # 0.1% par défaut
                "fixed": 0.0,
                "minimum": 0.0
            }
        
        # Récupérer les données du backtest
        data = backtest_result["backtest_data"].copy()
        initial_capital = backtest_result["initial_capital"]
        
        # Si aucune colonne position n'existe, on ne peut pas calculer les trades
        if 'position' not in data.columns:
            logger.warning("Impossible d'appliquer les frais : colonne 'position' non trouvée dans les données")
            return backtest_result
            
        # Identifier les points d'exécution des trades (changement de position)
        data['trade'] = data['position'].diff().fillna(0)
        data['trade_value'] = abs(data['trade'] * data['close'])
        
        # Calculer les frais de transaction
        data['fees'] = 0.0
        # Appliquer les frais uniquement lorsqu'un trade a lieu
        trade_mask = data['trade'] != 0
        if trade_mask.any():
            # Calculer les frais proportionnels
            percentage_fees = data.loc[trade_mask, 'trade_value'] * transaction_fees["percentage"]
            # Ajouter les frais fixes
            total_fees = percentage_fees + transaction_fees["fixed"]
            # Appliquer le minimum de frais si spécifié
            if transaction_fees["minimum"] > 0:
                total_fees = total_fees.clip(lower=transaction_fees["minimum"])
            # Assigner les frais au dataframe
            data.loc[trade_mask, 'fees'] = total_fees
        
        # Calculer le capital quotidien avec les frais déduits
        data['daily_capital'] = initial_capital
        current_capital = initial_capital
        
        for i in range(len(data)):
            if i > 0:
                # Déduire les frais du capital
                current_capital -= data.iloc[i]['fees']
                # Appliquer le rendement de la stratégie
                if 'returns' in data.columns:
                    returns = data.iloc[i]['returns']
                else:
                    # Calculer les rendements si non disponibles
                    returns = data.iloc[i]['close'] / data.iloc[i-1]['close'] - 1
                
                # Appliquer la stratégie (position * rendement)
                current_capital *= (1 + returns * data.iloc[i]['position'])
                data.iloc[i, data.columns.get_loc('daily_capital')] = current_capital
        
        # Calculer les rendements avec frais
        data['strategy_returns_with_fees'] = data['daily_capital'].pct_change().fillna(0)
        
        # Calculer les rendements cumulatifs
        data['cumulative_strategy_returns_with_fees'] = data['daily_capital'] / initial_capital
        
        # Calculer le drawdown
        data['peak_with_fees'] = data['cumulative_strategy_returns_with_fees'].cummax()
        data['drawdown_with_fees'] = (data['cumulative_strategy_returns_with_fees'] - data['peak_with_fees']) / data['peak_with_fees']
        
        # Calculer les métriques
        total_return = data['cumulative_strategy_returns_with_fees'].iloc[-1] - 1
        max_drawdown = data['drawdown_with_fees'].min()
        
        # Calculer le ratio de Sharpe (en supposant 252 jours de trading par an et un taux sans risque de 0)
        sharpe_ratio = np.sqrt(252) * data['strategy_returns_with_fees'].mean() / data['strategy_returns_with_fees'].std()
        
        # Calculer le capital final
        final_capital = data['daily_capital'].iloc[-1]
        
        # Compter les trades
        trades = (data['trade'] != 0).sum()
        
        # Calculer le total des frais payés
        total_fees_paid = data['fees'].sum()
        
        # Mettre à jour les résultats avec les frais
        result_with_fees = backtest_result.copy()
        result_with_fees.update({
            "final_capital": final_capital,
            "total_return": total_return,
            "annualized_return": total_return / (len(data) / 252),
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "trades": trades,
            "total_fees_paid": total_fees_paid,
            "average_fee_per_trade": total_fees_paid / trades if trades > 0 else 0,
            "backtest_data": data
        })
        
        return result_with_fees
    
    async def run_backtest(self, symbol, strategy_name):
        """
        Exécute un backtest pour un symbole et une stratégie
        
        Args:
            symbol: Le symbole boursier
            strategy_name: Nom de la stratégie
            
        Returns:
            Résultats du backtest
        """
        logger.info(f"Exécution du backtest pour {symbol} avec {strategy_name}...")
        
        try:
            # Charger les données
            data = await self.load_data(symbol)
            if data is None or len(data) < 100:  # Vérification minimale
                logger.error(f"Données insuffisantes pour {symbol}")
                return None
                
            # Obtenir les paramètres de la stratégie
            strategy_params = self.config["strategy_params"].get(strategy_name, {})
            
            # Initialiser la stratégie
            strategy = await self.strategy_manager.get_strategy(strategy_name, strategy_params)
            
            # Traitement spécial pour LSTM qui nécessite un entraînement préalable
            if strategy_name == "LSTMPredictorStrategy" and hasattr(strategy, 'train'):
                logger.info(f"Entraînement du modèle LSTM pour {symbol}...")
                try:
                    # Prétraiter les données
                    processed_data = await strategy.preprocess_data(data)
                    # Entraîner le modèle
                    await strategy.train(processed_data)
                    logger.info(f"Modèle LSTM entraîné avec succès pour {symbol}")
                except Exception as e:
                    logger.error(f"Erreur lors de l'entraînement du modèle LSTM pour {symbol}: {e}")
                    return {"error": f"Erreur d'entraînement: {str(e)}"}
            
            # Exécuter le backtest standard (sans frais)
            # Utiliser directement les objets datetime, pas les strings
            result = await self.backtesting_service.run_backtest(
                strategy=strategy,
                symbol=symbol,
                start_date=self.start_date,  # Objet datetime, pas string
                end_date=self.end_date,      # Objet datetime, pas string
                initial_capital=self.config["initial_capital"]
            )
            
            # Appliquer les frais de transaction aux résultats
            if result and 'backtest_data' in result and not result.get('error'):
                result = self.apply_transaction_fees(result, self.config.get("transaction_fees", {}))
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors du backtest de {symbol} avec {strategy_name}: {e}")
            return None
    
    async def run_all_backtests(self):
        """
        Exécute tous les backtests pour toutes les combinaisons de symboles et stratégies
        
        Returns:
            Dictionnaire contenant tous les résultats
        """
        for symbol in self.config["symbols"]:
            self.results[symbol] = {}
            
            for strategy_name in self.config["strategies"]:
                logger.info(f"=== Démarrage du backtest: {symbol} avec {strategy_name} ===")
                
                result = await self.run_backtest(symbol, strategy_name)
                if result:
                    self.results[symbol][strategy_name] = result
                    logger.info(f"Backtest terminé pour {symbol} avec {strategy_name}")
                    
                    # Afficher les métriques principales
                    metrics = [
                        ("Capital initial", f"${result.get('initial_capital', 0):,.2f}"),
                        ("Capital final", f"${result.get('final_capital', 0):,.2f}"),
                        ("Rendement total", f"{result.get('total_return', 0) * 100:.2f}%"),
                        ("Rendement annualisé", f"{result.get('annualized_return', 0) * 100:.2f}%"),
                        ("Ratio de Sharpe", f"{result.get('sharpe_ratio', 0):.2f}"),
                        ("Drawdown maximum", f"{result.get('max_drawdown', 0) * 100:.2f}%"),
                        ("Nombre de trades", f"{result.get('trades', 0)}"),
                        ("Frais totaux", f"${result.get('total_fees_paid', 0):,.2f}"),
                        ("Frais moyens par trade", f"${result.get('average_fee_per_trade', 0):,.2f}")
                    ]
                    logger.info(tabulate(metrics, headers=["Métrique", "Valeur"]))
                    logger.info("=" * 50)
        
        return self.results
    
    async def compare_strategies(self):
        """
        Compare les performances de différentes stratégies
        
        Returns:
            DataFrame avec les métriques de comparaison
        """
        comparison_data = []
        
        for symbol in self.results:
            for strategy_name, result in self.results[symbol].items():
                if result:
                    comparison_data.append({
                        "Symbol": symbol,
                        "Strategy": strategy_name,
                        "Total Return": result.get('total_return', 0) * 100,
                        "Annualized Return": result.get('annualized_return', 0) * 100,
                        "Sharpe Ratio": result.get('sharpe_ratio', 0),
                        "Max Drawdown": result.get('max_drawdown', 0) * 100,
                        "Trades": result.get('trades', 0),
                        "Total Fees": result.get('total_fees_paid', 0),
                        "Avg Fee/Trade": result.get('average_fee_per_trade', 0)
                    })
        
        if comparison_data:
            self._comparison_df = pd.DataFrame(comparison_data)
            return self._comparison_df
        self._comparison_df = None
        return None
    
    def plot_equity_curves(self, save_path=None):
        """
        Génère un graphique comparatif des courbes d'équité
        
        Args:
            save_path: Chemin pour sauvegarder le graphique (optionnel)
        """
        # Créer un graphique par symbole
        for symbol in self.results:
            plt.figure(figsize=(12, 8))
            
            # Ajouter chaque stratégie
            for strategy_name, result in self.results[symbol].items():
                if result and 'backtest_data' in result:
                    data = result['backtest_data']
                    plt.plot(data.index, data['cumulative_strategy_returns'], 
                            label=f"{strategy_name}")
                    
                    # Ajouter la stratégie "buy & hold" comme référence
                    if 'cumulative_returns' in data.columns:
                        plt.plot(data.index, data['cumulative_returns'], 
                                label="Buy & Hold", linestyle='--')
            
            plt.title(f"Comparaison des stratégies - {symbol}")
            plt.xlabel("Date")
            plt.ylabel("Croissance du capital (1$ initial)")
            plt.grid(True)
            plt.legend()
            
            # Sauvegarder le graphique
            if save_path:
                plt.savefig(f"{save_path}/equity_curve_{symbol}.png", dpi=300)
            plt.close()
    
    def save_results(self, output_dir="./results"):
        """
        Sauvegarde les résultats dans des fichiers
        
        Args:
            output_dir: Répertoire de sortie
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Sauvegarder les résultats au format JSON
        results_to_save = {}
        for symbol in self.results:
            results_to_save[symbol] = {}
            for strategy_name, result in self.results[symbol].items():
                if result:
                    # Convertir les DataFrames en listes pour JSON
                    result_copy = result.copy()
                    if 'backtest_data' in result_copy:
                        result_copy['backtest_data'] = result_copy['backtest_data'].to_dict(orient='records')
                    results_to_save[symbol][strategy_name] = result_copy
        
        with open(f"{output_dir}/backtest_results_{timestamp}.json", 'w') as f:
            json.dump(results_to_save, f, indent=2, default=str)
        
        # Sauvegarder la comparaison des stratégies au CSV
        # Note: On n'utilise pas asyncio.run() ici car on est déjà dans une boucle asyncio
        if hasattr(self, '_comparison_df') and self._comparison_df is not None:
            self._comparison_df.to_csv(f"{output_dir}/strategy_comparison_{timestamp}.csv", index=False)
        
        # Sauvegarder les graphiques
        self.plot_equity_curves(save_path=output_dir)
        
        logger.info(f"Résultats sauvegardés dans {output_dir}")

async def main(args=None):
    """Fonction principale pour exécuter le backtest long terme"""
    
    # Parse les arguments de ligne de commande
    parser = argparse.ArgumentParser(description='Mercurio AI - Long-Term Backtesting')
    parser.add_argument('--config', type=str, help='Chemin vers un fichier de configuration JSON')
    parser.add_argument('--start_date', type=str, help='Date de début (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, help='Date de fin (YYYY-MM-DD)')
    parser.add_argument('--symbols', type=str, help='Symboles séparés par des virgules')
    parser.add_argument('--capital', type=float, help='Capital initial')
    parser.add_argument('--fee_percentage', type=float, help='Pourcentage de frais par transaction (ex: 0.001 pour 0.1%)')
    parser.add_argument('--fee_fixed', type=float, help='Frais fixes par transaction en USD')
    parser.add_argument('--fee_minimum', type=float, help='Frais minimum par transaction en USD')
    
    parsed_args = parser.parse_args(args)
    
    # Charger la configuration
    config = DEFAULT_CONFIG.copy()
    
    # Si un fichier de configuration est fourni, le charger
    if parsed_args.config:
        try:
            with open(parsed_args.config, 'r') as f:
                config.update(json.load(f))
        except Exception as e:
            logger.error(f"Erreur lors du chargement du fichier de configuration: {e}")
    
    # Mettre à jour avec les arguments en ligne de commande
    if parsed_args.start_date:
        config["start_date"] = parsed_args.start_date
    if parsed_args.end_date:
        config["end_date"] = parsed_args.end_date
    if parsed_args.symbols:
        config["symbols"] = parsed_args.symbols.split(',')
    if parsed_args.capital:
        config["initial_capital"] = parsed_args.capital
        
    # Mettre à jour les frais de transaction si spécifiés
    if parsed_args.fee_percentage is not None or parsed_args.fee_fixed is not None or parsed_args.fee_minimum is not None:
        # Assurez-vous que le dictionnaire transaction_fees existe
        if "transaction_fees" not in config:
            config["transaction_fees"] = {"percentage": 0.001, "fixed": 0.0, "minimum": 0.0}
            
        if parsed_args.fee_percentage is not None:
            config["transaction_fees"]["percentage"] = parsed_args.fee_percentage
        if parsed_args.fee_fixed is not None:
            config["transaction_fees"]["fixed"] = parsed_args.fee_fixed
        if parsed_args.fee_minimum is not None:
            config["transaction_fees"]["minimum"] = parsed_args.fee_minimum
    
    logger.info("=" * 80)
    logger.info("MERCURIO AI - BACKTESTING LONG TERME")
    logger.info("=" * 80)
    logger.info(f"Période: {config['start_date']} à {config['end_date']}")
    logger.info(f"Symboles: {', '.join(config['symbols'])}")
    logger.info(f"Stratégies: {', '.join(config['strategies'])}")
    logger.info(f"Capital initial: ${config['initial_capital']:,.2f}")
    
    # Afficher les frais de transaction
    if "transaction_fees" in config:
        fees = config["transaction_fees"]
        logger.info(f"Frais de transaction: {fees['percentage']*100:.3f}% + ${fees['fixed']:.2f} (min: ${fees['minimum']:.2f})")
    logger.info("=" * 80)
    
    # Créer et exécuter le backtester
    backtester = LongTermBacktester(config)
    await backtester.run_all_backtests()
    
    # Comparer les stratégies
    comparison = await backtester.compare_strategies()
    if comparison is not None:
        logger.info("\n" + tabulate(comparison, headers='keys', tablefmt='pretty', floatfmt=".2f"))
    
    # Sauvegarder les résultats (pas besoin d'await car la méthode n'est pas async)
    backtester.save_results()
    
    logger.info("=" * 80)
    logger.info("BACKTESTING TERMINÉ")
    logger.info("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
