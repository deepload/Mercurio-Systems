#!/usr/bin/env python
"""
MercurioAI Enhanced Trader

Agent de trading amélioré qui intègre l'analyse technique avancée,
la détection d'anomalies et le tableau de bord de surveillance.
"""

import os
import json
import time
import logging
import asyncio
import argparse
import threading
import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import pandas as pd
import numpy as np

# Imports internes MercurioAI
from app.services.strategy_manager import StrategyManager
from app.services.market_data import MarketDataService
from app.services.trading import TradingService
from app.utils.technical_analyzer import TechnicalAnalyzer

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/enhanced_trader.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedTrader:
    """
    Agent de trading amélioré qui combine plusieurs sources d'intelligence
    et utilise des techniques avancées d'analyse et d'optimisation.
    """
    
    def __init__(self, config_path: str):
        """
        Initialise l'agent de trading amélioré avec la configuration spécifiée.
        
        Args:
            config_path: Chemin vers le fichier de configuration JSON
        """
        # Charger la configuration
        self.config_path = config_path
        self.load_config()
        
        # Créer les répertoires nécessaires
        os.makedirs("logs", exist_ok=True)
        os.makedirs("reports", exist_ok=True)
        os.makedirs("data/signals", exist_ok=True)
        
        # Initialiser les services de base
        self.strategy_manager = StrategyManager()
        self.market_data_service = MarketDataService()
        self.trading_service = TradingService(
            is_paper=not self.config.get("live_trading", False)
        )
        
        # Initialiser les outils d'analyse
        self.technical_analyzer = TechnicalAnalyzer()
        
        # État de l'agent
        self.running = False
        self.last_check_time = {}
        self.portfolio_value_history = []
        self.signals_history = []
        self.market_regimes = {}
        self.detected_anomalies = {}
        self.strategy_weights = self._initialize_strategy_weights()
        
        # Démarrer le tableau de bord si demandé
        self.dashboard_thread = None
        if self.config.get("start_dashboard", False):
            self._start_dashboard()
    
    def load_config(self):
        """Charge la configuration à partir du fichier JSON"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
                
            logger.info(f"Configuration chargée depuis {self.config_path}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration: {e}")
            raise
    
    def _initialize_strategy_weights(self) -> Dict[str, float]:
        """Initialise les poids des stratégies avec une distribution uniforme"""
        strategies = self.config.get("strategies", [])
        if not strategies:
            logger.warning("Aucune stratégie spécifiée dans la configuration")
            return {}
            
        weight = 1.0 / len(strategies)
        return {strategy: weight for strategy in strategies}
    
    def _start_dashboard(self):
        """Démarre le tableau de bord dans un thread séparé"""
        def run_dashboard():
            try:
                import subprocess
                subprocess.Popen(["python", "trading_dashboard.py"])
                logger.info("Tableau de bord démarré")
            except Exception as e:
                logger.error(f"Erreur lors du démarrage du tableau de bord: {e}")
        
        self.dashboard_thread = threading.Thread(target=run_dashboard)
        self.dashboard_thread.daemon = True
        self.dashboard_thread.start()
    
    async def start(self):
        """Démarre l'agent de trading"""
        try:
            self.running = True
            logger.info("Agent de trading amélioré démarré")
            
            # Initialiser les stratégies
            await self._initialize_strategies()
            
            # Boucle principale
            while self.running:
                try:
                    await self._trading_cycle()
                    
                    # Générer un rapport périodique
                    self._generate_performance_report()
                    
                    # Attendre l'intervalle spécifié
                    await asyncio.sleep(self.config.get("check_interval_seconds", 60))
                except Exception as e:
                    logger.error(f"Erreur dans le cycle de trading: {e}")
                    await asyncio.sleep(10)  # Attendre avant de réessayer
        except Exception as e:
            logger.error(f"Erreur critique dans l'agent de trading: {e}")
        finally:
            self.running = False
            logger.info("Agent de trading arrêté")
    
    def stop(self):
        """Arrête l'agent de trading"""
        logger.info("Arrêt de l'agent de trading demandé")
        self.running = False
    
    async def _initialize_strategies(self):
        """Initialise les stratégies spécifiées dans la configuration"""
        for strategy_name in self.config.get("strategies", []):
            try:
                # Récupérer les paramètres spécifiques à la stratégie
                params = self.config.get("strategy_params", {}).get(strategy_name, {})
                
                # Initialiser la stratégie
                logger.info(f"Initialisation de la stratégie {strategy_name}")
                await self.strategy_manager.initialize_strategy(strategy_name, params)
            except Exception as e:
                logger.error(f"Erreur lors de l'initialisation de la stratégie {strategy_name}: {e}")
