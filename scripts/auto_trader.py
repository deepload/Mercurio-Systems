#!/usr/bin/env python
"""
MercurioAI Auto Trader - Agent de trading professionnel automatisé

Ce script étend les fonctionnalités de MercurioAI pour créer un agent de trading 
entièrement automatisé qui peut:
1. Sélectionner et basculer automatiquement entre les stratégies
2. Optimiser les paramètres des stratégies en fonction des conditions de marché
3. Exécuter des analyses de marché avancées
4. Prendre des décisions autonomes basées sur tous les signaux disponibles
"""

import os
import sys
import json
import logging
import asyncio
import argparse
import signal
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
from pathlib import Path

from app.services.market_data import MarketDataService
from app.services.trading import TradingService
from app.services.strategy_manager import StrategyManager
from app.db.models import TradeAction
from app.core.event_bus import EventBus, EventType

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/auto_trader.log")
    ]
)
logger = logging.getLogger(__name__)

# Variable globale pour contrôler la boucle de trading
running = True

def signal_handler(sig, frame):
    """Gère les signaux d'interruption pour un arrêt propre"""
    global running
    logger.info("Signal d'arrêt reçu. Arrêt après l'itération en cours...")
    running = False

class MarketState:
    """Classe pour stocker et analyser l'état du marché"""
    
    def __init__(self):
        self.regimes = {}          # Régimes de marché par symbole
        self.volatility = {}       # Niveaux de volatilité par symbole
        self.sentiment = {}        # Scores de sentiment par symbole
        self.anomalies = {}        # Anomalies détectées par symbole
        self.last_update = None    # Dernière mise à jour de l'état
    
    def update_regime(self, symbol: str, data: pd.DataFrame) -> str:
        """Détecte le régime de marché actuel"""
        if data is None or len(data) < 20:
            return "unknown"
            
        # Calculer les rendements et statistiques
        returns = data['close'].pct_change().dropna()
        recent_trend = data['close'].iloc[-1] / data['close'].iloc[-15] - 1 if len(data) >= 15 else 0
        volatility = returns.std() if len(returns) > 0 else 0
        
        # Déterminer le régime
        regime = "unknown"
        if abs(recent_trend) > 0.1:  # Tendance forte
            regime = "bullish" if recent_trend > 0 else "bearish"
        elif volatility > 0.03:  # Volatilité élevée
            regime = "volatile"
        else:
            regime = "sideways"
            
        self.regimes[symbol] = regime
        self.volatility[symbol] = volatility
        self.last_update = datetime.now()
        
        return regime
    
    def detect_anomalies(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Détecte les anomalies potentielles dans les données de marché"""
        anomalies = {
            "detected": False,
            "manipulation_probability": 0.0,
            "timestamp": datetime.now()
        }
        
        if data is None or len(data) < 30:
            return anomalies
            
        # Analyse de volume
        volumes = data['volume']
        mean_volume = volumes.mean()
        std_volume = volumes.std()
        recent_volume = volumes.iloc[-1]
        volume_surge = recent_volume > (mean_volume + 3 * std_volume)
        
        # Analyse de prix
        closes = data['close']
        returns = closes.pct_change().dropna()
        mean_return = returns.mean()
        std_return = returns.std()
        recent_return = returns.iloc[-1] if len(returns) > 0 else 0
        price_shock = abs(recent_return) > (3 * std_return)
        
        # Evaluer la probabilité de manipulation
        anomaly_count = 0
        if volume_surge:
            anomaly_count += 1
        if price_shock:
            anomaly_count += 1
            
        manipulation_probability = min(0.8, anomaly_count * 0.4)
        
        anomalies = {
            "detected": manipulation_probability > 0.3,
            "volume_surge": volume_surge,
            "price_shock": price_shock,
            "manipulation_probability": manipulation_probability,
            "timestamp": datetime.now()
        }
        
        self.anomalies[symbol] = anomalies
        return anomalies

class AutoTrader:
    """
    Agent de trading automatisé qui sélectionne, optimise et
    exécute les stratégies de trading de manière autonome.
    """
    
    def __init__(self, config_path: str):
        """
        Initialise l'agent de trading avec la configuration spécifiée.
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        self.load_config(config_path)
        self.market_data_service = MarketDataService()
        self.trading_service = TradingService(is_paper=not self.config.get("live_trading", False))
        self.strategy_manager = StrategyManager()
        self.event_bus = EventBus()
        
        # État du marché
        self.market_state = MarketState()
        
        # État des stratégies
        self.active_strategies = {}
        self.strategy_performances = {}
        self.strategy_weights = {}
        
        # État de l'agent
        self.portfolio_value = 0.0
        self.cash = 0.0
        self.positions = {}
        self.pending_orders = {}
        self.transaction_costs = 0.0
        
        logger.info(f"Agent de trading initialisé avec configuration: {config_path}")
    
    def load_config(self, config_path: str) -> None:
        """Charge la configuration depuis un fichier JSON"""
        try:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
                
            logger.info(f"Configuration chargée avec succès: {len(self.config.get('symbols', []))} symboles, "
                       f"{len(self.config.get('strategies', []))} stratégies")
                       
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration: {e}")
            raise
    
    async def initialize(self) -> None:
        """Initialise tous les services et stratégies nécessaires"""
        try:
            logger.info("Initialisation des services...")
            
            # Charger les stratégies configurées
            for strategy_name in self.config.get("strategies", []):
                strategy_params = self.config.get("strategy_params", {}).get(strategy_name, {})
                
                try:
                    # Obtenir la stratégie du gestionnaire
                    strategy = await self.strategy_manager.get_strategy(strategy_name, strategy_params)
                    
                    # Poids initial égal pour toutes les stratégies
                    weight = 1.0 / len(self.config.get("strategies", []))
                    
                    self.active_strategies[strategy_name] = strategy
                    self.strategy_weights[strategy_name] = weight
                    
                    logger.info(f"Stratégie {strategy_name} chargée avec succès (poids: {weight:.2f})")
                    
                except Exception as e:
                    logger.error(f"Erreur lors du chargement de la stratégie {strategy_name}: {e}")
            
            # Vérifier la connexion au compte
            account_info = await self.trading_service.get_account_info()
            self.portfolio_value = account_info.get("portfolio_value", 0.0)
            self.cash = account_info.get("cash", 0.0)
            
            logger.info(f"Connecté au compte: {account_info.get('id', 'inconnu')}")
            logger.info(f"Statut du compte: {account_info.get('status', 'inconnu')}")
            logger.info(f"Valeur du portefeuille: ${self.portfolio_value:.2f}")
            
            # S'abonner aux événements
            asyncio.create_task(self.event_bus.subscribe(
                EventType.MARKET_DATA_UPDATED,
                self._handle_market_data_update
            ))
            
            logger.info(f"Agent initialisé avec {len(self.active_strategies)} stratégies actives")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de l'agent: {e}")
            raise
    
    async def start(self) -> None:
        """Démarre l'agent de trading"""
        try:
            await self.initialize()
            
            logger.info("==================================================")
            logger.info("CONFIRMATION DE TRADING AUTOMATISÉ")
            logger.info("==================================================")
            logger.info(f"Vous êtes sur le point de démarrer le trading {'RÉEL' if self.config.get('live_trading', False) else 'PAPIER'}")
            logger.info(f"Valeur du portefeuille: ${self.portfolio_value:.2f}")
            logger.info(f"Stratégies: {', '.join(list(self.active_strategies.keys()))}")
            logger.info(f"Symboles: {', '.join(self.config.get('symbols', []))}")
            logger.info(f"Coûts de transaction: {self.config.get('transaction_costs', {}).get('percentage', 0.001)*100:.3f}%")
            logger.info("==================================================")
            
            confirmation = input("Tapez 'CONFIRM' pour démarrer le trading ou autre chose pour annuler: ")
            
            if confirmation != "CONFIRM":
                logger.info("Trading automatisé annulé par l'utilisateur")
                return
                
            logger.info("Trading automatisé confirmé. Démarrage de la boucle de trading...")
            
            # Boucle principale de trading
            await self.trading_loop()
            
        except KeyboardInterrupt:
            logger.info("Arrêt demandé par l'utilisateur")
        except Exception as e:
            logger.error(f"Erreur critique dans l'agent de trading: {e}")
        finally:
            await self.generate_performance_report()
    
    async def trading_loop(self) -> None:
        """Boucle principale de trading"""
        global running
        running = True
        
        check_interval = self.config.get("check_interval_seconds", 60)
        
        while running:
            try:
                # Vérifier si le marché est ouvert
                is_open = await self.trading_service.is_market_open()
                
                if not is_open:
                    next_open = await self.trading_service.get_next_market_open()
                    logger.info(f"Marché fermé. Prochaine ouverture: {next_open}")
                    logger.info("Marché fermé, attente de 30 minutes avant la prochaine vérification...")
                    
                    # En mode test/démo, on continue même si le marché est fermé
                    if not self.config.get("ignore_market_hours", False):
                        await asyncio.sleep(30 * 60)  # 30 minutes
                        continue
                
                # Mettre à jour l'état du portefeuille
                await self.update_portfolio_state()
                
                # Traiter chaque symbole
                for symbol in self.config.get("symbols", []):
                    await self.process_symbol(symbol)
                
                # Courte pause entre les itérations
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Erreur dans la boucle de trading: {e}")
                await asyncio.sleep(60)  # Pause plus longue en cas d'erreur
    
    async def update_portfolio_state(self) -> None:
        """Met à jour l'état du portefeuille et des positions"""
        try:
            # Mettre à jour les informations du compte
            account_info = await self.trading_service.get_account_info()
            self.portfolio_value = account_info.get("portfolio_value", self.portfolio_value)
            self.cash = account_info.get("cash", self.cash)
            
            # Mettre à jour les positions
            positions = await self.trading_service.get_positions()
            self.positions = {p.get("symbol"): p for p in positions}
            
            # Mettre à jour les ordres en attente
            orders = await self.trading_service.get_open_orders()
            self.pending_orders = {o.get("id"): o for o in orders}
            
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour du portefeuille: {e}")
    
    async def process_symbol(self, symbol: str) -> None:
        """
        Traite un symbole - obtient des données, génère des signaux avec différentes 
        stratégies, prend une décision consensuelle et exécute si nécessaire.
        
        Args:
            symbol: Symbole à traiter
        """
        try:
            # Récupérer les données récentes
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            # Obtenir des données historiques suffisantes pour l'analyse
            data = await self.market_data_service.get_historical_data(
                symbol, start_date, end_date
            )
            
            if data is None or data.empty or len(data) < 20:
                logger.warning(f"Données insuffisantes pour {symbol}, traitement ignoré")
                return
                
            # Mettre à jour l'état du marché
            regime = self.market_state.update_regime(symbol, data)
            anomalies = self.market_state.detect_anomalies(symbol, data)
            
            if anomalies.get("detected", False):
                logger.warning(f"Anomalies détectées pour {symbol} - prudence recommandée")
                if anomalies.get("manipulation_probability", 0) > 0.6:
                    logger.warning(f"ALERTE: Manipulation potentielle sur {symbol}")
                    return  # Ignorer ce symbole pour ce cycle
            
            # Collecter les signaux de toutes les stratégies actives
            signals = []
            
            for strategy_name, strategy in self.active_strategies.items():
                try:
                    action, confidence = await strategy.predict(data)
                    weight = self.strategy_weights.get(strategy_name, 0.0)
                    
                    signals.append({
                        "strategy": strategy_name,
                        "action": action,
                        "confidence": confidence,
                        "weight": weight,
                        "timestamp": datetime.now()
                    })
                    
                    logger.info(f"Signal de {strategy_name} pour {symbol}: {action.name} ({confidence:.2f})")
                    
                except Exception as e:
                    logger.error(f"Erreur lors de la génération de signal pour {symbol} avec {strategy_name}: {e}")
            
            # Générer un consensus pondéré
            if signals:
                consensus = self.generate_consensus(signals)
                action = consensus.get("action")
                confidence = consensus.get("confidence")
                
                logger.info(f"Consensus pour {symbol}: {action.name} avec confiance {confidence:.2f}")
                
                # Exécuter le signal si confiance suffisante
                min_confidence = self.config.get("min_execution_confidence", 0.75)
                
                if confidence >= min_confidence and action != TradeAction.HOLD:
                    await self.execute_trading_signal(symbol, action, confidence)
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement de {symbol}: {e}")
    
    def generate_consensus(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Génère un consensus pondéré à partir des signaux de plusieurs stratégies.
        
        Args:
            signals: Liste de signaux de différentes stratégies
            
        Returns:
            Dictionnaire avec l'action consensuelle et le niveau de confiance
        """
        if not signals:
            return {"action": TradeAction.HOLD, "confidence": 0.0}
        
        # Initialiser les scores pour chaque action
        buy_score = 0.0
        sell_score = 0.0
        hold_score = 0.0
        total_weight = 0.0
        
        # Calculer les scores pondérés
        for signal in signals:
            action = signal.get("action", TradeAction.HOLD)
            confidence = signal.get("confidence", 0.0)
            weight = signal.get("weight", 0.0)
            
            weighted_score = weight * confidence
            total_weight += weight
            
            if action == TradeAction.BUY:
                buy_score += weighted_score
            elif action == TradeAction.SELL:
                sell_score += weighted_score
            else:
                hold_score += weighted_score
        
        # Normaliser les scores si nécessaire
        if total_weight > 0:
            buy_score = buy_score / total_weight
            sell_score = sell_score / total_weight
            hold_score = hold_score / total_weight
        
        # Déterminer l'action consensuelle
        max_score = max(buy_score, sell_score, hold_score)
        
        if max_score == 0.0:
            return {"action": TradeAction.HOLD, "confidence": 0.0}
        
        if max_score == buy_score:
            action = TradeAction.BUY
            confidence = buy_score
        elif max_score == sell_score:
            action = TradeAction.SELL
            confidence = sell_score
        else:
            action = TradeAction.HOLD
            confidence = hold_score
        
        return {
            "action": action,
            "confidence": confidence,
            "consensus_stats": {
                "buy_score": buy_score,
                "sell_score": sell_score,
                "hold_score": hold_score,
                "total_weight": total_weight
            }
        }
    
    async def execute_trading_signal(self, symbol: str, action: TradeAction, confidence: float) -> None:
        """
        Exécute un signal de trading.
        
        Args:
            symbol: Symbole à trader
            action: Action de trading (BUY, SELL, HOLD)
            confidence: Niveau de confiance (0.0 à 1.0)
        """
        try:
            if action == TradeAction.HOLD:
                return
                
            # Vérifier si nous avons déjà une position sur ce symbole
            position = self.positions.get(symbol)
            position_value = float(position.get("market_value", 0.0)) if position else 0.0
            position_qty = float(position.get("qty", 0.0)) if position else 0.0
            
            # Vérifier s'il y a des ordres en attente pour ce symbole
            has_pending_orders = any(o.get("symbol") == symbol for o in self.pending_orders.values())
            if has_pending_orders:
                logger.info(f"Ordres en attente pour {symbol}, exécution ignorée")
                return
                
            # Obtenir le prix actuel
            price_data = await self.market_data_service.get_latest_price(symbol)
            if not price_data:
                logger.warning(f"Impossible d'obtenir le prix actuel pour {symbol}")
                return
                
            current_price = price_data.get("price", 0.0)
            if current_price <= 0.0:
                logger.warning(f"Prix invalide pour {symbol}: {current_price}")
                return
                
            # Calculer la taille de position en fonction du risque et de la confiance
            risk_limit = self.config.get("risk_limit", 0.02)  # 2% par défaut
            risk_amount = self.portfolio_value * risk_limit * confidence
            
            if action == TradeAction.BUY:
                if position_value > 0:
                    logger.info(f"Position longue existante sur {symbol}, augmentation ignorée")
                    return
                    
                # Calculer la quantité à acheter
                quantity = risk_amount / current_price
                
                logger.info(f"Ordre d'achat pour {symbol}: {quantity:.6f} @ {current_price:.2f} = ${risk_amount:.2f}")
                
                # Placer l'ordre
                order_result = await self.trading_service.place_market_order(
                    symbol=symbol,
                    quantity=quantity,
                    side="buy"
                )
                
                logger.info(f"Résultat de l'ordre: {order_result}")
                
                # Appliquer les coûts de transaction
                fee_percentage = self.config.get("transaction_costs", {}).get("percentage", 0.001)
                transaction_cost = risk_amount * fee_percentage
                self.transaction_costs += transaction_cost
                
            elif action == TradeAction.SELL:
                if position_value < 0:
                    logger.info(f"Position courte existante sur {symbol}, augmentation ignorée")
                    return
                    
                # Si nous avons une position longue, la fermer
                if position_value > 0:
                    logger.info(f"Fermeture de position longue sur {symbol}: {position_qty} @ {current_price:.2f}")
                    
                    order_result = await self.trading_service.place_market_order(
                        symbol=symbol,
                        quantity=position_qty,
                        side="sell"
                    )
                    
                    logger.info(f"Résultat de l'ordre: {order_result}")
                    
                    # Appliquer les coûts de transaction
                    fee_percentage = self.config.get("transaction_costs", {}).get("percentage", 0.001)
                    transaction_cost = position_value * fee_percentage
                    self.transaction_costs += transaction_cost
                else:
                    # Ouvrir une position courte si autorisé
                    shorts_allowed = self.config.get("advanced_settings", {}).get("allow_shorts", False)
                    
                    if shorts_allowed:
                        # Calculer la quantité à vendre
                        quantity = risk_amount / current_price
                        
                        logger.info(f"Ordre de vente pour {symbol}: {quantity:.6f} @ {current_price:.2f} = ${risk_amount:.2f}")
                        
                        order_result = await self.trading_service.place_market_order(
                            symbol=symbol,
                            quantity=quantity,
                            side="sell"
                        )
                        
                        logger.info(f"Résultat de l'ordre: {order_result}")
                        
                        # Appliquer les coûts de transaction
                        fee_percentage = self.config.get("transaction_costs", {}).get("percentage", 0.001)
                        transaction_cost = risk_amount * fee_percentage
                        self.transaction_costs += transaction_cost
        
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution du signal pour {symbol}: {e}")
    
    async def _handle_market_data_update(self, event_data: Dict[str, Any]) -> None:
        """Gestionnaire d'événements pour les mises à jour de données de marché"""
        symbol = event_data.get("symbol")
        if symbol:
            logger.debug(f"Mise à jour de données reçue pour {symbol}")
    
    async def optimize_strategies(self) -> None:
        """
        Optimise les poids des stratégies en fonction de leurs performances récentes.
        Cette fonction est appelée périodiquement pour ajuster les poids.
        """
        logger.info("Optimisation des poids des stratégies...")
        
        # Si nous n'avons pas assez d'historique, utiliser des poids égaux
        if not self.strategy_performances:
            equal_weight = 1.0 / len(self.active_strategies)
            for strategy_name in self.active_strategies:
                self.strategy_weights[strategy_name] = equal_weight
            return
            
        # Calculer les scores de performance pour chaque stratégie
        strategy_scores = {}
        
        for strategy_name in self.active_strategies:
            performances = self.strategy_performances.get(strategy_name, [])
            
            if not performances:
                strategy_scores[strategy_name] = 0.01  # Score minimal par défaut
                continue
                
            # Calculer un score basé sur les performances récentes
            recent_perfs = performances[-10:]  # 10 dernières performances
            correct_signals = sum(1 for p in recent_perfs if p.get("correct", False))
            score = correct_signals / len(recent_perfs) if recent_perfs else 0.01
            
            # Ajuster le score en fonction de la confiance moyenne
            avg_confidence = sum(p.get("confidence", 0) for p in recent_perfs) / len(recent_perfs) if recent_perfs else 0
            adjusted_score = score * (0.5 + avg_confidence / 2)
            
            strategy_scores[strategy_name] = max(0.01, adjusted_score)  # Au moins 1%
        
        # Normaliser les scores pour obtenir les poids
        total_score = sum(strategy_scores.values())
        
        if total_score > 0:
            for strategy_name, score in strategy_scores.items():
                self.strategy_weights[strategy_name] = score / total_score
        
        logger.info(f"Nouveaux poids des stratégies: {self.strategy_weights}")
    
    async def generate_performance_report(self) -> None:
        """Génère un rapport de performance"""
        try:
            account_info = await self.trading_service.get_account_info()
            portfolio_value = account_info.get("portfolio_value", 0.0)
            cash = account_info.get("cash", 0.0)
            
            logger.info("")
            logger.info("===== PERFORMANCE REPORT =====")
            logger.info(f"Portfolio Value: ${portfolio_value:.2f}")
            logger.info(f"Cash: ${cash:.2f}")
            logger.info(f"Total Transaction Costs: ${self.transaction_costs:.2f}")
            logger.info(f"Net Portfolio Value (after costs): ${portfolio_value - self.transaction_costs:.2f}")
            logger.info("================================")
            
            # Sauvegarder le rapport dans un fichier
            report = {
                "timestamp": datetime.now().isoformat(),
                "portfolio_value": portfolio_value,
                "cash": cash,
                "transaction_costs": self.transaction_costs,
                "net_value": portfolio_value - self.transaction_costs,
                "positions": self.positions,
                "strategy_weights": self.strategy_weights,
                "market_regimes": self.market_state.regimes
            }
            
            reports_dir = Path("reports")
            reports_dir.mkdir(exist_ok=True)
            
            report_path = reports_dir / f"performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
                
            logger.info(f"Rapport de performance enregistré: {report_path}")
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du rapport: {e}")

async def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="MercurioAI Auto Trader")
    parser.add_argument("--config", type=str, default="config/agent_config.json", 
                       help="Chemin vers le fichier de configuration")
    args = parser.parse_args()
    
    # Créer les répertoires nécessaires
    os.makedirs("logs", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    # Enregistrer les gestionnaires de signaux
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Créer et démarrer l'agent de trading
    auto_trader = AutoTrader(args.config)
    
    try:
        await auto_trader.start()
    except KeyboardInterrupt:
        logger.info("Arrêt demandé par l'utilisateur")
    except Exception as e:
        logger.error(f"Erreur critique: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        logger.info("Programme interrompu par l'utilisateur")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Erreur critique: {e}")
        sys.exit(1)
