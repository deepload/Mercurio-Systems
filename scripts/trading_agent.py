#!/usr/bin/env python
"""
MercurioAI Trading Agent - Agent de trading automatisé et professionnel

Ce script implémente un agent de trading autonome capable de:
1. Surveiller les marchés en continu
2. Analyser automatiquement les données avec plusieurs stratégies
3. Sélectionner la meilleure stratégie selon les conditions actuelles
4. Exécuter des trades de manière autonome avec gestion des risques
5. S'adapter dynamiquement aux changements de marché
"""

import os
import sys
import json
import logging
import asyncio
import argparse
import datetime
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import numpy as np
from pathlib import Path

from app.core.event_bus import EventBus, EventType
from app.services.market_data import MarketDataService
from app.services.trading import TradingService
from app.services.backtesting import BacktestingService
from app.strategies.base import BaseStrategy
from app.db.models import TradeAction

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/trading_agent.log")
    ]
)
logger = logging.getLogger(__name__)

class TradingAgent:
    """
    Agent de trading autonome qui orchestre toutes les opérations
    de surveillance, d'analyse et d'exécution des trades.
    """
    
    def __init__(self, config_path: str):
        """
        Initialise l'agent de trading avec une configuration donnée.
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        self.load_config(config_path)
        self.event_bus = EventBus()
        self.market_data_service = MarketDataService()
        self.trading_service = TradingService(is_paper=not self.config.get("live_trading", False))
        self.backtesting_service = BacktestingService()
        
        # État de l'agent
        self.active_strategies = {}
        self.strategy_performance = {}
        self.strategy_weights = {}
        self.market_state = {}
        self.last_analysis_time = None
        self.last_execution_time = None
        
        # Surveillance continue du marché
        self.monitoring_active = False
        self.execution_active = False
        
        logger.info(f"Agent de trading initialisé avec configuration: {config_path}")
    
    def load_config(self, config_path: str) -> None:
        """
        Charge la configuration depuis un fichier JSON.
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
                
            # Validation des paramètres essentiels
            required_fields = ["symbols", "strategies", "initial_capital"]
            for field in required_fields:
                if field not in self.config:
                    raise ValueError(f"Champ requis manquant dans la configuration: {field}")
                    
            logger.info(f"Configuration chargée avec succès: {len(self.config.get('symbols', []))} symboles, "
                       f"{len(self.config.get('strategies', []))} stratégies")
                       
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration: {e}")
            raise
    
    async def initialize(self) -> None:
        """
        Initialise tous les services et stratégies nécessaires.
        """
        try:
            # Initialiser les stratégies
            for strategy_name in self.config.get("strategies", []):
                strategy_params = self.config.get("strategy_params", {}).get(strategy_name, {})
                strategy = await self.load_strategy(strategy_name, strategy_params)
                if strategy:
                    self.active_strategies[strategy_name] = strategy
                    # Initialiser avec des poids égaux
                    self.strategy_weights[strategy_name] = 1.0 / len(self.config.get("strategies", []))
            
            # S'abonner aux événements pertinents
            asyncio.create_task(self.event_bus.subscribe(EventType.MARKET_DATA_UPDATED, self._handle_market_data_update))
            asyncio.create_task(self.event_bus.subscribe(EventType.ORDER_FILLED, self._handle_order_update))
            
            logger.info(f"Agent initialisé avec {len(self.active_strategies)} stratégies actives")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de l'agent: {e}")
            raise
    
    async def load_strategy(self, strategy_name: str, params: Dict[str, Any]) -> Optional[BaseStrategy]:
        """
        Charge une stratégie par son nom avec les paramètres spécifiés.
        
        Args:
            strategy_name: Nom de la stratégie
            params: Paramètres de configuration
            
        Returns:
            Instance de stratégie ou None en cas d'erreur
        """
        try:
            from app.services.strategy_manager import StrategyManager
            strategy_manager = StrategyManager()
            strategy = await strategy_manager.get_strategy(strategy_name, params)
            
            # Calibrer/entraîner la stratégie si nécessaire
            symbols = self.config.get("symbols", [])
            if symbols:
                # Utiliser le premier symbole pour l'entraînement initial
                symbol = symbols[0]
                end_date = datetime.datetime.now()
                start_date = end_date - datetime.timedelta(days=30)  # 30 jours de données
                
                data = await self.market_data_service.get_historical_data(
                    symbol, start_date, end_date
                )
                
                if data is not None and not data.empty:
                    await strategy.train(data)
                    logger.info(f"Stratégie {strategy_name} calibrée avec succès")
                else:
                    logger.warning(f"Impossible de calibrer {strategy_name} - données insuffisantes")
            
            return strategy
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la stratégie {strategy_name}: {e}")
            return None
    
    async def start(self) -> None:
        """
        Démarre l'agent de trading en mode autonome.
        """
        try:
            await self.initialize()
            
            # Configurer les tâches concurrentes
            self.monitoring_active = True
            self.execution_active = True
            
            monitor_task = asyncio.create_task(self._continuous_market_monitoring())
            analysis_task = asyncio.create_task(self._continuous_strategy_analysis())
            execution_task = asyncio.create_task(self._continuous_trade_execution())
            optimization_task = asyncio.create_task(self._periodic_strategy_optimization())
            
            logger.info("Agent de trading démarré en mode autonome")
            
            # Attendre que toutes les tâches soient terminées
            await asyncio.gather(
                monitor_task, analysis_task, execution_task, optimization_task
            )
            
        except KeyboardInterrupt:
            logger.info("Arrêt de l'agent demandé par l'utilisateur")
            self.stop()
        except Exception as e:
            logger.error(f"Erreur critique pendant l'exécution de l'agent: {e}")
            self.stop()
    
    def stop(self) -> None:
        """
        Arrête proprement l'agent de trading.
        """
        self.monitoring_active = False
        self.execution_active = False
        logger.info("Agent de trading arrêté")
    
    async def _continuous_market_monitoring(self) -> None:
        """
        Surveille en continu les marchés et met à jour les données.
        """
        check_interval = self.config.get("check_interval_seconds", 60)
        
        while self.monitoring_active:
            try:
                for symbol in self.config.get("symbols", []):
                    # Récupérer les données les plus récentes
                    end_date = datetime.datetime.now()
                    start_date = end_date - datetime.timedelta(hours=4)  # 4h de données récentes
                    
                    data = await self.market_data_service.get_historical_data(
                        symbol, start_date, end_date, timeframe="1m"
                    )
                    
                    if data is not None and not data.empty:
                        # Mettre à jour l'état du marché
                        self.market_state[symbol] = {
                            "data": data,
                            "last_update": datetime.datetime.now(),
                            "is_fresh": True
                        }
                        
                        # Notifier via le bus d'événements
                        await self.event_bus.publish(EventType.MARKET_DATA_UPDATED, {
                            "symbol": symbol,
                            "data": data
                        })
                        
                        logger.debug(f"Données de marché mises à jour pour {symbol}")
                    else:
                        logger.warning(f"Impossible d'obtenir des données pour {symbol}")
                        self.market_state[symbol] = {
                            "is_fresh": False,
                            "last_update": self.market_state.get(symbol, {}).get("last_update")
                        }
                
            except Exception as e:
                logger.error(f"Erreur pendant la surveillance du marché: {e}")
            
            # Attendre avant la prochaine vérification
            await asyncio.sleep(check_interval)
    
    async def _continuous_strategy_analysis(self) -> None:
        """
        Analyse en continu les marchés avec toutes les stratégies actives.
        """
        analysis_interval = self.config.get("analysis_interval_seconds", 300)  # 5 minutes par défaut
        
        while self.monitoring_active:
            try:
                now = datetime.datetime.now()
                self.last_analysis_time = now
                
                for symbol in self.config.get("symbols", []):
                    # Vérifier si les données sont fraîches
                    market_info = self.market_state.get(symbol, {})
                    if not market_info.get("is_fresh", False):
                        logger.warning(f"Données obsolètes pour {symbol}, analyse reportée")
                        continue
                    
                    data = market_info.get("data")
                    if data is None or data.empty:
                        logger.warning(f"Données insuffisantes pour {symbol}, analyse reportée")
                        continue
                    
                    # Exécuter l'analyse avec chaque stratégie
                    for strategy_name, strategy in self.active_strategies.items():
                        try:
                            action, confidence = await strategy.predict(data)
                            
                            # Stocker les résultats d'analyse
                            if symbol not in self.strategy_performance:
                                self.strategy_performance[symbol] = {}
                            
                            self.strategy_performance[symbol][strategy_name] = {
                                "action": action,
                                "confidence": confidence,
                                "timestamp": now
                            }
                            
                            logger.info(f"Stratégie {strategy_name} pour {symbol}: {action.name} avec confiance {confidence:.2f}")
                            
                        except Exception as e:
                            logger.error(f"Erreur d'analyse avec {strategy_name} pour {symbol}: {e}")
                
                # Générer un consensus entre les stratégies
                for symbol in self.config.get("symbols", []):
                    if symbol in self.strategy_performance:
                        consensus = self._generate_strategy_consensus(symbol)
                        logger.info(f"Consensus pour {symbol}: {consensus['action'].name} avec confiance {consensus['confidence']:.2f}")
                
            except Exception as e:
                logger.error(f"Erreur pendant l'analyse des stratégies: {e}")
            
            # Attendre avant la prochaine analyse
            await asyncio.sleep(analysis_interval)
    
    def _generate_strategy_consensus(self, symbol: str) -> Dict[str, Any]:
        """
        Génère un consensus entre les différentes stratégies pour un symbole donné.
        
        Args:
            symbol: Symbole à analyser
            
        Returns:
            Dictionnaire contenant l'action consensuelle et le niveau de confiance
        """
        strategies_info = self.strategy_performance.get(symbol, {})
        if not strategies_info:
            return {"action": TradeAction.HOLD, "confidence": 0.0}
        
        # Initialiser les scores
        buy_score = 0.0
        sell_score = 0.0
        hold_score = 0.0
        
        # Calculer les scores pondérés pour chaque action
        for strategy_name, info in strategies_info.items():
            weight = self.strategy_weights.get(strategy_name, 0.0)
            confidence = info.get("confidence", 0.0)
            action = info.get("action", TradeAction.HOLD)
            
            weighted_confidence = weight * confidence
            
            if action == TradeAction.BUY:
                buy_score += weighted_confidence
            elif action == TradeAction.SELL:
                sell_score += weighted_confidence
            else:
                hold_score += weighted_confidence
        
        # Déterminer l'action consensuelle
        max_score = max(buy_score, sell_score, hold_score)
        if max_score == 0.0:
            return {"action": TradeAction.HOLD, "confidence": 0.0}
        
        if max_score == buy_score:
            return {"action": TradeAction.BUY, "confidence": buy_score}
        elif max_score == sell_score:
            return {"action": TradeAction.SELL, "confidence": sell_score}
        else:
            return {"action": TradeAction.HOLD, "confidence": hold_score}
    
    async def _continuous_trade_execution(self) -> None:
        """
        Exécute en continu les signaux de trading générés par l'analyse.
        """
        execution_interval = self.config.get("execution_interval_seconds", 60)
        min_confidence = self.config.get("min_execution_confidence", 0.7)
        
        while self.execution_active:
            try:
                now = datetime.datetime.now()
                self.last_execution_time = now
                
                for symbol in self.config.get("symbols", []):
                    consensus = self._generate_strategy_consensus(symbol)
                    action = consensus.get("action", TradeAction.HOLD)
                    confidence = consensus.get("confidence", 0.0)
                    
                    # Exécuter uniquement si la confiance dépasse le seuil
                    if confidence >= min_confidence and action != TradeAction.HOLD:
                        risk_limit = self.config.get("risk_limit", 0.02)  # 2% par défaut
                        
                        # Obtenir le prix actuel
                        market_info = self.market_state.get(symbol, {})
                        data = market_info.get("data")
                        
                        if data is not None and not data.empty:
                            current_price = data["close"].iloc[-1]
                            
                            # Exécuter l'ordre
                            account_info = await self.trading_service.get_account_info()
                            portfolio_value = account_info.get("portfolio_value", 0.0)
                            
                            # Calculer la taille de position
                            position_size = portfolio_value * risk_limit * confidence
                            quantity = position_size / current_price
                            
                            if action == TradeAction.BUY:
                                order_result = await self.trading_service.place_market_order(
                                    symbol=symbol,
                                    quantity=quantity,
                                    side="buy"
                                )
                                logger.info(f"Ordre d'achat placé pour {symbol}: {quantity} @ {current_price} = ${position_size:.2f}")
                                
                            elif action == TradeAction.SELL:
                                order_result = await self.trading_service.place_market_order(
                                    symbol=symbol,
                                    quantity=quantity,
                                    side="sell"
                                )
                                logger.info(f"Ordre de vente placé pour {symbol}: {quantity} @ {current_price} = ${position_size:.2f}")
                            
                            logger.info(f"Résultat de l'ordre: {order_result}")
                        else:
                            logger.warning(f"Impossible d'exécuter un ordre pour {symbol} - prix actuel non disponible")
                    else:
                        logger.debug(f"Aucun ordre exécuté pour {symbol} - confiance insuffisante ou HOLD")
                
            except Exception as e:
                logger.error(f"Erreur pendant l'exécution des trades: {e}")
            
            # Attendre avant la prochaine exécution
            await asyncio.sleep(execution_interval)
    
    async def _periodic_strategy_optimization(self) -> None:
        """
        Optimise périodiquement les poids des stratégies en fonction de leurs performances.
        """
        optimization_interval = self.config.get("optimization_interval_hours", 24) * 3600
        
        while self.monitoring_active:
            try:
                # Attendre la première optimisation
                await asyncio.sleep(optimization_interval)
                
                logger.info("Démarrage de l'optimisation des stratégies...")
                
                # Évaluer les performances récentes de chaque stratégie
                for symbol in self.config.get("symbols", []):
                    # Récupérer les données récentes pour le backtesting
                    end_date = datetime.datetime.now()
                    start_date = end_date - datetime.timedelta(days=7)  # 7 jours d'évaluation
                    
                    data = await self.market_data_service.get_historical_data(
                        symbol, start_date, end_date
                    )
                    
                    if data is None or data.empty:
                        logger.warning(f"Données insuffisantes pour optimiser les stratégies sur {symbol}")
                        continue
                    
                    # Évaluer chaque stratégie
                    strategy_scores = {}
                    for strategy_name, strategy in self.active_strategies.items():
                        try:
                            # Effectuer un backtest rapide
                            backtest_result = await strategy.backtest(data)
                            
                            # Extraire les métriques de performance
                            if isinstance(backtest_result, dict) and "total_return" in backtest_result:
                                score = backtest_result["total_return"]
                                if "max_drawdown" in backtest_result:
                                    # Ajuster le score en fonction du drawdown
                                    max_drawdown = abs(backtest_result["max_drawdown"])
                                    if max_drawdown > 0:
                                        score = score / (max_drawdown * 2)  # Pénaliser le drawdown
                                
                                strategy_scores[strategy_name] = max(0.01, score)  # Score minimum pour éviter les zéros
                            else:
                                logger.warning(f"Résultats de backtest incomplets pour {strategy_name}")
                                strategy_scores[strategy_name] = 0.01  # Score minimal par défaut
                        
                        except Exception as e:
                            logger.error(f"Erreur lors de l'évaluation de {strategy_name}: {e}")
                            strategy_scores[strategy_name] = 0.01  # Score minimal en cas d'erreur
                    
                    # Mettre à jour les poids en fonction des scores
                    total_score = sum(strategy_scores.values())
                    if total_score > 0:
                        for strategy_name, score in strategy_scores.items():
                            self.strategy_weights[strategy_name] = score / total_score
                    
                    logger.info(f"Nouveaux poids de stratégies pour {symbol}: {self.strategy_weights}")
                
            except Exception as e:
                logger.error(f"Erreur pendant l'optimisation des stratégies: {e}")
    
    async def _handle_market_data_update(self, event_data: Dict[str, Any]) -> None:
        """
        Gère les mises à jour de données de marché.
        
        Args:
            event_data: Données de l'événement
        """
        symbol = event_data.get("symbol")
        if symbol:
            logger.debug(f"Événement de mise à jour de données reçu pour {symbol}")
    
    async def _handle_order_update(self, event_data: Dict[str, Any]) -> None:
        """
        Gère les mises à jour de statut des ordres.
        
        Args:
            event_data: Données de l'événement
        """
        order_id = event_data.get("order_id")
        if order_id:
            logger.info(f"Ordre {order_id} exécuté avec succès")
    
    async def generate_report(self) -> Dict[str, Any]:
        """
        Génère un rapport complet sur les performances de l'agent.
        
        Returns:
            Dictionnaire contenant les métriques de performance
        """
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "portfolio": await self._get_portfolio_summary(),
            "strategy_performance": self.strategy_performance,
            "strategy_weights": self.strategy_weights,
            "symbols_monitored": list(self.market_state.keys()),
            "strategies_active": list(self.active_strategies.keys())
        }
        
        return report
    
    async def _get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Obtient un résumé du portefeuille actuel.
        
        Returns:
            Dictionnaire contenant les informations du portefeuille
        """
        try:
            account_info = await self.trading_service.get_account_info()
            positions = await self.trading_service.get_positions()
            
            return {
                "portfolio_value": account_info.get("portfolio_value", 0.0),
                "cash": account_info.get("cash", 0.0),
                "positions": positions
            }
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du résumé du portefeuille: {e}")
            return {"error": str(e)}


async def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="MercurioAI Trading Agent")
    parser.add_argument("--config", type=str, required=True, help="Chemin vers le fichier de configuration")
    parser.add_argument("--report_interval", type=int, default=3600, help="Intervalle de génération de rapports en secondes")
    args = parser.parse_args()
    
    # Créer le répertoire de logs s'il n'existe pas
    os.makedirs("logs", exist_ok=True)
    
    try:
        # Créer l'agent de trading
        agent = TradingAgent(args.config)
        
        # Tâche de génération de rapports périodiques
        async def generate_periodic_reports():
            while True:
                await asyncio.sleep(args.report_interval)
                report = await agent.generate_report()
                report_path = f"reports/agent_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                os.makedirs("reports", exist_ok=True)
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=2)
                logger.info(f"Rapport généré: {report_path}")
        
        # Démarrer l'agent et la génération de rapports
        report_task = asyncio.create_task(generate_periodic_reports())
        
        try:
            await agent.start()
        except KeyboardInterrupt:
            logger.info("Arrêt demandé par l'utilisateur")
        finally:
            agent.stop()
            report_task.cancel()
    except Exception as e:
        logger.error(f"Erreur d'initialisation: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Programme interrompu par l'utilisateur")
    except Exception as e:
        logger.error(f"Erreur critique: {e}")
        sys.exit(1)
