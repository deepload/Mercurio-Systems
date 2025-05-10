#!/usr/bin/env python
"""
MercurioAI Trading Agent (Version corrigée)

Agent de trading automatisé avec corrections pour la stabilité
et la compatibilité avec les données de secours.
"""

import os
import json
import time
import logging
import asyncio
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np

# Imports des services MercurioAI
from app.services.strategy_manager import StrategyManager
from app.services.market_data import MarketDataService
from app.services.trading import TradingService

# Importation des correctifs et utilitaires
from app.services.patches.data_service_patch import patch_market_data_service
from app.utils.data_enricher import enrich_data, create_synthetic_data
from app.utils.exception_handler import exception_manager, with_exception_handling, \
    MarketDataException, StrategyException, ExecutionException

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/trading_agent.log", mode="w"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingAgent:
    """
    Agent de trading automatisé qui utilise plusieurs stratégies
    pour prendre des décisions de trading basées sur le consensus.
    """
    
    def __init__(self, config_path: str, report_interval: int = 3600):
        """
        Initialise l'agent de trading avec la configuration spécifiée.
        
        Args:
            config_path: Chemin vers le fichier de configuration JSON
            report_interval: Intervalle en secondes entre les rapports de performance
        """
        # Vérifier si les répertoires nécessaires existent
        os.makedirs("logs", exist_ok=True)
        os.makedirs("reports", exist_ok=True)
        
        # Charger la configuration
        self.config_path = config_path
        self.report_interval = report_interval
        self.load_config()
        
        # Initialiser les services
        self.strategy_manager = StrategyManager()
        self.market_data_service = MarketDataService()
        
        # Appliquer le correctif pour les données de marché
        patch_market_data_service(self.market_data_service)
        
        self.trading_service = TradingService(
            is_paper=not self.config.get("live_trading", False)
        )
        
        # État interne de l'agent
        self.initialized_strategies = {}
        self.strategy_weights = {}
        self.last_check_time = {}
        self.portfolio_value_history = []
        self.running = False
        
        # Informations sur le portefeuille
        self.initial_capital = self.config.get("initial_capital", 10000)
        
        logger.info(f"Agent de trading initialisé avec configuration: {config_path}")
    
    def load_config(self):
        """Charge la configuration à partir du fichier JSON"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
                
            symbols = self.config.get("symbols", [])
            strategies = self.config.get("strategies", [])
            logger.info(f"Configuration chargée avec succès: {len(symbols)} symboles, {len(strategies)} stratégies")
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration: {e}")
            raise
    
    async def initialize(self):
        """Initialise l'agent et ses stratégies"""
        try:
            # Initialiser les stratégies pour chaque symbole
            for symbol in self.config.get("symbols", []):
                await self._initialize_symbol_strategies(symbol)
                
            # Initialiser les poids uniformes des stratégies
            self._initialize_strategy_weights()
            
            return True
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de l'agent: {e}")
            return False
    
    async def _initialize_symbol_strategies(self, symbol: str):
        """Initialise les stratégies pour un symbole spécifique"""
        for strategy_name in self.config.get("strategies", []):
            try:
                # Obtenir des paramètres spécifiques à la stratégie s'ils existent
                strategy_params = self.config.get("strategy_params", {}).get(strategy_name, {})
                
                # Vérifier si PyTorch est nécessaire mais non disponible
                if "Transformer" in strategy_name or "LSTM" in strategy_name:
                    try:
                        import torch
                    except ImportError:
                        logger.warning(f"PyTorch non disponible, stratégie {strategy_name} sera ignorée")
                        continue
                
                # Récupérer la stratégie
                strategy = self.strategy_manager.get_strategy(strategy_name, strategy_params)
                
                # Stocker la stratégie dans notre dictionnaire local
                if not hasattr(self, 'initialized_strategies'):
                    self.initialized_strategies = {}
                    
                if symbol not in self.initialized_strategies:
                    self.initialized_strategies[symbol] = {}
                    
                self.initialized_strategies[symbol][strategy_name] = strategy
                
                # Obtenir des données historiques pour calibrer la stratégie
                historical_data = await self.market_data_service.get_historical_data(symbol)
                
                # Calibrer la stratégie
                try:
                    # Pour certaines stratégies comme MSI, une gestion spéciale peut être nécessaire
                    if strategy_name == "MultiSourceIntelligenceStrategy":
                        if hasattr(self.strategy_manager.strategies[strategy_name], "prepare_data"):
                            historical_data = await self.strategy_manager.strategies[strategy_name].prepare_data(historical_data)
                    
                    await self.strategy_manager.strategies[strategy_name].train(historical_data)
                    
                    # Marquer la stratégie comme initialisée pour ce symbole
                    if symbol not in self.initialized_strategies:
                        self.initialized_strategies[symbol] = []
                        
                    self.initialized_strategies[symbol].append(strategy_name)
                    logger.info(f"Stratégie {strategy_name} calibrée avec succès")
                except Exception as e:
                    logger.error(f"Erreur lors de la calibration de la stratégie {strategy_name}: {e}")
            except Exception as e:
                logger.error(f"Erreur lors du chargement de la stratégie {strategy_name}: {e}")
    
    def _initialize_strategy_weights(self):
        """Initialise les poids des stratégies avec une distribution uniforme"""
        strategies = self.config.get("strategies", [])
        weight = 1.0 / max(len(strategies), 1)  # Éviter la division par zéro
        
        self.strategy_weights = {strategy: weight for strategy in strategies}
    
    async def run(self):
        """Exécute l'agent de trading"""
        if not await self.initialize():
            logger.error("Échec de l'initialisation de l'agent")
            return
        
        self.running = True
        last_report_time = time.time()
        
        try:
            logger.info("Agent de trading démarré")
            
            while self.running:
                try:
                    # Vérifier chaque symbole
                    for symbol in self.config.get("symbols", []):
                        try:
                            await self._check_symbol(symbol)
                        except Exception as e:
                            logger.error(f"Erreur lors de la vérification du symbole {symbol}: {e}")
                    
                    # Générer un rapport périodique
                    current_time = time.time()
                    if current_time - last_report_time >= self.report_interval:
                        self._generate_performance_report()
                        last_report_time = current_time
                    
                    # Attendre l'intervalle spécifié
                    await asyncio.sleep(self.config.get("check_interval_seconds", 60))
                except Exception as e:
                    logger.error(f"Erreur dans la boucle principale: {e}")
                    await asyncio.sleep(10)  # Courte pause avant de réessayer
        except KeyboardInterrupt:
            logger.info("Interruption de l'utilisateur")
        except Exception as e:
            logger.error(f"Erreur critique pendant l'exécution de l'agent: {e}")
        finally:
            self.running = False
            logger.info("Agent de trading arrêté")
    
    def stop(self):
        """Arrête l'agent de trading"""
        self.running = False
        logger.info("Agent de trading arrêté")
    
    @with_exception_handling(retry=True, retry_exceptions=[MarketDataException], max_retries=3)
    async def _check_symbol(self, symbol: str):
        """
        Vérifie un symbole pour des opportunités de trading
        
        Args:
            symbol: Symbole à vérifier
        """
        # Vérifier si des stratégies ont été initialisées pour ce symbole
        if symbol not in self.initialized_strategies or not self.initialized_strategies[symbol]:
            logger.warning(f"Aucune stratégie initialisée pour {symbol}, ignoré")
            return
        
        try:
            # Récupérer les données de marché actuelles
            current_data = await self.market_data_service.get_latest_data(symbol)
            
            if current_data is None or current_data.empty:
                logger.warning(f"Données non disponibles pour {symbol}, utilisation de données synthétiques")
                current_data = create_synthetic_data(symbol, days=30).iloc[-1:]
        except Exception as e:
            # Transformer l'exception en MarketDataException pour activer le mécanisme de retry
            raise MarketDataException(f"Erreur lors de la récupération des données pour {symbol}", 
                                    details={"symbol": symbol, "original_error": str(e)})
        
        # Collecter les signaux de toutes les stratégies initialisées
        signals = []
        for strategy_name in self.initialized_strategies[symbol]:
            try:
                strategy = self.initialized_strategies[symbol][strategy_name]
                signal = await strategy.generate_signal(symbol, current_data)
                
                if signal:
                    weight = self.strategy_weights.get(strategy_name, 0.0)
                    signals.append({
                        "strategy": strategy_name,
                        "action": signal.get("action", "hold"),
                        "confidence": signal.get("confidence", 0.0),
                        "weight": weight
                    })
                    
                    logger.info(f"Signal généré par {strategy_name} pour {symbol}: {signal}")
            except Exception as e:
                # Gérer l'exception mais continuer avec les autres stratégies
                exception_manager.log_exception(
                    StrategyException(f"Erreur lors de la génération de signal", 
                                     details={"strategy": strategy_name, "symbol": symbol}), 
                    f"Stratégie {strategy_name}"
                )
        
        # Prendre une décision basée sur le consensus
        if signals:
            decision = self._make_consensus_decision(signals)
            
            # Exécuter la décision si la confiance est suffisante
            min_confidence = self.config.get("min_execution_confidence", 0.7)
            
            if decision["confidence"] >= min_confidence and decision["action"] != "hold":
                # Calculer la taille de la position en fonction de la limite de risque
                position_size = self._calculate_position_size(symbol, decision["action"])
                
                # Exécuter l'ordre
                if position_size > 0:
                    try:
                        await self._execute_order(symbol, decision["action"], position_size)
                        logger.info(f"Ordre exécuté pour {symbol}: {decision['action']}, taille: {position_size}")
                    except Exception as e:
                        logger.error(f"Erreur lors de l'exécution de l'ordre pour {symbol}: {e}")
    
    def _make_consensus_decision(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Combine les signaux des différentes stratégies pour une décision finale
        
        Args:
            signals: Liste des signaux des stratégies
            
        Returns:
            Décision finale avec action et confiance
        """
        if not signals:
            return {"action": "hold", "confidence": 0.0}
        
        # Calculer les scores pondérés pour chaque action
        action_scores = {
            "buy": 0.0,
            "sell": 0.0,
            "hold": 0.0
        }
        
        total_weight = sum(signal["weight"] for signal in signals)
        
        if total_weight <= 0:
            return {"action": "hold", "confidence": 0.0}
        
        # Pondérer les signaux
        for signal in signals:
            action = signal["action"]
            confidence = signal["confidence"]
            weight = signal["weight"]
            
            if action in action_scores:
                action_scores[action] += (weight / total_weight) * confidence
        
        # Trouver l'action avec le score le plus élevé
        best_action = "hold"
        best_score = 0.0
        
        for action, score in action_scores.items():
            if score > best_score:
                best_action = action
                best_score = score
        
        return {
            "action": best_action,
            "confidence": best_score
        }
    
    def _calculate_position_size(self, symbol: str, action: str) -> float:
        """
        Calcule la taille de position appropriée en fonction de la limite de risque
        
        Args:
            symbol: Symbole pour lequel calculer la taille de position
            action: Action de trading (buy/sell)
            
        Returns:
            Taille de la position en unités
        """
        try:
            # Obtenir la valeur actuelle du portefeuille
            portfolio_value = self._get_portfolio_value()
            
            # Obtenir la limite de risque depuis la configuration
            risk_limit = self.config.get("risk_limit", 0.02)
            
            # Calculer la valeur maximale à risquer pour cette position
            max_position_value = portfolio_value * risk_limit
            
            # Si nous avons déjà une position sur ce symbole, la prendre en compte
            current_position = self._get_current_position(symbol)
            
            # Calculer la taille de la position
            if action == "buy" and current_position <= 0:
                # Obtenir le prix actuel
                current_price = self._get_current_price(symbol)
                if current_price > 0:
                    return max_position_value / current_price
            elif action == "sell" and current_position >= 0:
                return abs(current_position)  # Fermer la position existante ou ouvrir une position courte
            
            return 0.0
        except Exception as e:
            logger.error(f"Erreur lors du calcul de la taille de position pour {symbol}: {e}")
            return 0.0
    
    async def _execute_order(self, symbol: str, action: str, size: float):
        """
        Exécute un ordre de trading
        
        Args:
            symbol: Symbole pour lequel exécuter l'ordre
            action: Action de trading (buy/sell)
            size: Taille de la position
        """
        if size <= 0:
            logger.warning(f"Taille de position invalide pour {symbol}: {size}")
            return
        
        try:
            if action == "buy":
                await self.trading_service.submit_order(
                    symbol=symbol,
                    qty=size,
                    side="buy",
                    type="market",
                    time_in_force="gtc"
                )
            elif action == "sell":
                await self.trading_service.submit_order(
                    symbol=symbol,
                    qty=size,
                    side="sell",
                    type="market",
                    time_in_force="gtc"
                )
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution de l'ordre pour {symbol}: {e}")
    
    def _get_portfolio_value(self) -> float:
        """
        Obtient la valeur actuelle du portefeuille
        
        Returns:
            Valeur totale du portefeuille
        """
        try:
            account = self.trading_service.get_account()
            if account:
                return float(account.portfolio_value)
            return self.initial_capital
        except Exception as e:
            logger.warning(f"Impossible d'obtenir la valeur du portefeuille: {e}")
            return self.initial_capital
    
    def _get_current_position(self, symbol: str) -> float:
        """
        Obtient la position actuelle pour un symbole
        
        Args:
            symbol: Symbole pour lequel obtenir la position
            
        Returns:
            Taille de la position (négative pour les positions courtes)
        """
        try:
            position = self.trading_service.get_position(symbol)
            if position:
                return float(position.qty)
            return 0.0
        except Exception:
            return 0.0
    
    def _get_current_price(self, symbol: str) -> float:
        """
        Obtient le prix actuel d'un symbole
        
        Args:
            symbol: Symbole pour lequel obtenir le prix
            
        Returns:
            Prix actuel
        """
        try:
            # Tenter d'obtenir le dernier prix du service de trading
            last_price = self.trading_service.get_last_price(symbol)
            if last_price and last_price > 0:
                return last_price
            
            # Fallback sur les données de marché
            current_data = asyncio.run(self.market_data_service.get_latest_data(symbol))
            if current_data is not None and not current_data.empty:
                return current_data['close'].iloc[-1]
            
            # Valeur par défaut en cas d'échec
            return 100.0
        except Exception as e:
            logger.warning(f"Impossible d'obtenir le prix actuel pour {symbol}: {e}")
            return 100.0
    
    def _generate_performance_report(self):
        """Génère un rapport de performance"""
        try:
            # Obtenir la valeur actuelle du portefeuille
            portfolio_value = self._get_portfolio_value()
            self.portfolio_value_history.append({
                "timestamp": datetime.datetime.now().isoformat(),
                "value": portfolio_value
            })
            
            # Obtenir les positions actuelles
            positions = {}
            try:
                for symbol in self.config.get("symbols", []):
                    position = self._get_current_position(symbol)
                    if position != 0.0:
                        current_price = self._get_current_price(symbol)
                        positions[symbol] = {
                            "size": position,
                            "price": current_price,
                            "value": position * current_price
                        }
            except Exception as e:
                logger.error(f"Erreur lors de la récupération des positions: {e}")
            
            # Calculer les performances
            initial_value = self.initial_capital
            if self.portfolio_value_history and len(self.portfolio_value_history) > 1:
                initial_value = self.portfolio_value_history[0]["value"]
            
            performance = (portfolio_value / initial_value - 1) * 100
            
            # Générer le rapport
            report = {
                "timestamp": datetime.datetime.now().isoformat(),
                "portfolio_value": portfolio_value,
                "initial_value": initial_value,
                "performance": performance,
                "positions": positions,
                "strategy_weights": self.strategy_weights
            }
            
            # Enregistrer le rapport
            report_path = f"reports/performance_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Rapport de performance généré: valeur du portefeuille = {portfolio_value}, performance = {performance:.2f}%")
            
            return report
        except Exception as e:
            logger.error(f"Erreur lors de la génération du rapport de performance: {e}")
            return None

async def main():
    """Fonction principale pour exécuter l'agent de trading"""
    parser = argparse.ArgumentParser(description="Agent de trading MercurioAI")
    parser.add_argument("--config", type=str, default="config/agent_config.json",
                        help="Chemin vers le fichier de configuration JSON")
    parser.add_argument("--report_interval", type=int, default=3600,
                        help="Intervalle en secondes entre les rapports de performance")
    args = parser.parse_args()
    
    # Créer et démarrer l'agent
    agent = TradingAgent(args.config, args.report_interval)
    
    try:
        await agent.run()
    except KeyboardInterrupt:
        agent.stop()

if __name__ == "__main__":
    asyncio.run(main())
