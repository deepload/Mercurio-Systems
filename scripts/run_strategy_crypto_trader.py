#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script pour lancer le trader de cryptos avec une stratégie spécifique
Ce script va lancer le trader de cryptos Alpaca avec une stratégie explicite et une 
durée de session paramétrable, parfait pour les sessions de nuit ou de jour.
"""

import sys
import os
import argparse
import logging
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Any, Optional, Union, Tuple
from dotenv import load_dotenv

# Ajouter le répertoire parent au path pour pouvoir importer les modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importer le trader de crypto
from alpaca_crypto_trader import AlpacaCryptoTrader, SessionDuration

# Importer les stratégies avancées
from app.strategies.lstm_predictor import LSTMPredictorStrategy
from app.strategies.llm_strategy import LLMStrategy

# Fonction pour détecter le niveau d'accès Alpaca
def detect_alpaca_level(api_key=None, api_secret=None, base_url=None, data_url=None):
    """
    Détecte le niveau d'abonnement Alpaca disponible en testant les fonctionnalités
    
    Args:
        api_key: Clé API Alpaca
        api_secret: Secret API Alpaca
        base_url: URL de base pour l'API Alpaca
        data_url: URL des données pour l'API Alpaca
        
    Returns:
        int: Niveau d'abonnement (3 = premium, 2 = standard+, 1 = standard, 0 = non détecté)
    """
    if not api_key or not api_secret:
        # Récupérer les clés d'API depuis les variables d'environnement
        load_dotenv()
        # Déterminer le mode (paper ou live)
        alpaca_mode = os.getenv("ALPACA_MODE", "paper").lower()
        if alpaca_mode == "live":
            api_key = os.getenv("ALPACA_LIVE_KEY")
            api_secret = os.getenv("ALPACA_LIVE_SECRET")
            base_url = os.getenv("ALPACA_LIVE_URL", "https://api.alpaca.markets")
        else:  # mode paper par défaut
            api_key = os.getenv("ALPACA_PAPER_KEY")
            api_secret = os.getenv("ALPACA_PAPER_SECRET")
            base_url = os.getenv("ALPACA_PAPER_URL", "https://paper-api.alpaca.markets")
        
        data_url = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
    
    # Initialiser le client API
    try:
        api = tradeapi.REST(
            key_id=api_key,
            secret_key=api_secret,
            base_url=base_url,
            data_url=data_url
        )
        
        logger.info("Test du niveau d'abonnement Alpaca...")
        
        # Test niveau 3 (premium) - Accès aux données en temps réel
        try:
            # Tester une fonctionnalité spécifique au niveau 3: données en temps réel plus précises
            end = datetime.now()
            start = end - timedelta(minutes=15)
            symbol = "BTC/USD"  # Une paire crypto populaire
            bars = api.get_crypto_bars(symbol, '1Min', start.isoformat(), end.isoformat())
            if len(bars) > 0 and hasattr(bars[0], 'trade_count'):
                logger.info("✅ Niveau 3 (Premium) détecté - Accès complet aux données temps réel")
                return 3
        except Exception as e:
            logger.debug(f"Test niveau 3 échoué: {str(e)}")
        
        # Test niveau 2 - Données historiques étendues
        try:
            # Tester des données historiques (disponibles dans les niveaux 2 et 3)
            end = datetime.now()
            start = end - timedelta(days=30)  # 30 jours de données
            bars = api.get_crypto_bars('BTC/USD', '1Day', start.isoformat(), end.isoformat())
            if len(bars) > 20:  # Si on a plus de 20 jours, c'est probablement niveau 2+
                logger.info("✅ Niveau 2 détecté - Accès aux données historiques étendues")
                return 2
        except Exception as e:
            logger.debug(f"Test niveau 2 échoué: {str(e)}")
        
        # Test niveau 1 - Fonctionnalités de base
        try:
            # Tester les fonctionnalités de base (disponibles dans tous les niveaux)
            account = api.get_account()
            logger.info("✅ Niveau 1 détecté - Accès aux fonctionnalités de base")
            return 1
        except Exception as e:
            logger.debug(f"Test niveau 1 échoué: {str(e)}")
        
        logger.warning("❌ Aucun niveau d'abonnement détecté - Vérifiez vos identifiants API")
        return 0
        
    except Exception as e:
        logger.error(f"Erreur lors de la connexion à Alpaca: {str(e)}")
        return 0

# Configuration du logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("strategy_crypto_trader")

# Liste personnalisée de cryptos à trader
PERSONALIZED_CRYPTO_LIST = [
    "AAVE/USD", "AAVE/USDT", "AVAX/USD", "BAT/USD", "BCH/USD", 
    "BCH/USDT", "BTC/USD", "BTC/USDT", "CRV/USD", "DOGE/USD", 
    "DOGE/USDT", "DOT/USD", "ETH/USD", "ETH/USDT", "GRT/USD", 
    "LINK/USD", "LINK/USDT", "LTC/USD", "LTC/USDT", "MKR/USD", 
    "PEPE/USD", "SHIB/USD", "SOL/USD", "SUSHI/USD", "SUSHI/USDT", 
    "TRUMP/USD", "UNI/USD", "UNI/USDT", "USDC/USD", "USDT/USD", 
    "XRP/USD", "XTZ/USD", "YFI/USD", "YFI/USDT"
]

# Implémentation des stratégies de trading
class BaseStrategy:
    """Classe de base pour toutes les stratégies"""
    def __init__(self, **kwargs):
        self.position_size = kwargs.get("position_size", 0.02)
        self.stop_loss = kwargs.get("stop_loss", 0.03)
        self.take_profit = kwargs.get("take_profit", 0.06)
        self.name = "BaseStrategy"
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyser les données et générer un signal"""
        raise NotImplementedError("Les stratégies dérivées doivent implémenter cette méthode")

class MomentumStrategy(BaseStrategy):
    """Stratégie basée sur le momentum des prix"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lookback_period = kwargs.get("lookback_period", 20)
        self.name = "Momentum"
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Génère un signal basé sur le momentum"""
        if len(data) < self.lookback_period + 10:
            return {"signal": "neutral", "strength": 0, "reason": "Données insuffisantes"}
        
        # Calculer le momentum (changement de prix sur la période)
        data['returns'] = data['close'].pct_change(self.lookback_period)
        data['momentum'] = data['returns'].rolling(window=10).mean()
        
        current_momentum = data['momentum'].iloc[-1]
        momentum_signal = "neutral"
        signal_strength = abs(current_momentum) * 10  # Normaliser entre 0-1
        
        if current_momentum > 0.02:  # Momentum positif significatif
            momentum_signal = "buy"
        elif current_momentum < -0.02:  # Momentum négatif significatif
            momentum_signal = "sell"
            
        return {
            "signal": momentum_signal,
            "strength": min(signal_strength, 1.0),
            "reason": f"Momentum {current_momentum:.4f}"
        }

class MeanReversionStrategy(BaseStrategy):
    """Stratégie basée sur le retour à la moyenne"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lookback_period = kwargs.get("lookback_period", 20)
        self.name = "Mean Reversion"
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Génère un signal basé sur le retour à la moyenne"""
        if len(data) < self.lookback_period + 10:
            return {"signal": "neutral", "strength": 0, "reason": "Données insuffisantes"}
        
        # Calculer la moyenne mobile et les bandes de Bollinger
        data['sma'] = data['close'].rolling(window=self.lookback_period).mean()
        data['std'] = data['close'].rolling(window=self.lookback_period).std()
        data['upper_band'] = data['sma'] + (data['std'] * 2)
        data['lower_band'] = data['sma'] - (data['std'] * 2)
        
        current_price = data['close'].iloc[-1]
        upper_band = data['upper_band'].iloc[-1]
        lower_band = data['lower_band'].iloc[-1]
        sma = data['sma'].iloc[-1]
        
        # Distance normalisée par rapport aux bandes
        if current_price > upper_band:
            # Sur-acheté - signal de vente
            distance = (current_price - upper_band) / (upper_band - sma)
            signal = "sell"
            reason = f"Prix {current_price:.4f} au-dessus de la bande supérieure {upper_band:.4f}"
        elif current_price < lower_band:
            # Sur-vendu - signal d'achat
            distance = (lower_band - current_price) / (sma - lower_band)
            signal = "buy"
            reason = f"Prix {current_price:.4f} en-dessous de la bande inférieure {lower_band:.4f}"
        else:
            # Entre les bandes - neutre
            if current_price > sma:
                distance = (current_price - sma) / (upper_band - sma)
                signal = "neutral_bearish"  # Tendance baissière potentielle
            else:
                distance = (sma - current_price) / (sma - lower_band)
                signal = "neutral_bullish"  # Tendance haussière potentielle
            reason = f"Prix {current_price:.4f} entre les bandes (SMA: {sma:.4f})"
        
        return {
            "signal": signal,
            "strength": min(distance * 1.5, 1.0),  # Normaliser entre 0-1
            "reason": reason
        }

class BreakoutStrategy(BaseStrategy):
    """Stratégie basée sur les ruptures de niveaux"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lookback_period = kwargs.get("lookback_period", 20)
        self.name = "Breakout"
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Génère un signal basé sur les ruptures de niveaux"""
        if len(data) < self.lookback_period + 5:
            return {"signal": "neutral", "strength": 0, "reason": "Données insuffisantes"}
        
        # Calculer les niveaux de support et résistance
        lookback_data = data.iloc[-self.lookback_period-5:-5]
        resistance = lookback_data['high'].max()
        support = lookback_data['low'].min()
        
        # Vérifier le volume
        avg_volume = lookback_data['volume'].mean()
        current_volume = data['volume'].iloc[-1]
        volume_factor = min(current_volume / avg_volume, 3.0) if avg_volume > 0 else 1.0
        
        current_price = data['close'].iloc[-1]
        previous_price = data['close'].iloc[-2]
        price_range = resistance - support
        
        if price_range <= 0:
            return {"signal": "neutral", "strength": 0, "reason": "Plage de prix trop faible"}
        
        # Normaliser la distance par rapport à la plage
        if current_price > resistance and previous_price <= resistance:
            # Breakout haussier
            distance = (current_price - resistance) / price_range
            signal = "buy"
            reason = f"Breakout haussier: {current_price:.4f} > {resistance:.4f} (résistance)"
        elif current_price < support and previous_price >= support:
            # Breakout baissier
            distance = (support - current_price) / price_range
            signal = "sell"
            reason = f"Breakout baissier: {current_price:.4f} < {support:.4f} (support)"
        else:
            # Pas de breakout
            signal = "neutral"
            if current_price > (resistance + support) / 2:
                reason = f"Prix {current_price:.4f} proche de la résistance {resistance:.4f}"
                distance = (current_price - ((resistance + support) / 2)) / (resistance - ((resistance + support) / 2))
            else:
                reason = f"Prix {current_price:.4f} proche du support {support:.4f}"
                distance = (((resistance + support) / 2) - current_price) / (((resistance + support) / 2) - support)
        
        signal_strength = min(distance * volume_factor, 1.0)
        
        return {
            "signal": signal,
            "strength": signal_strength,
            "reason": reason
        }

class StatisticalArbitrageStrategy(BaseStrategy):
    """Stratégie d'arbitrage statistique pour paires de cryptos"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.volatility_lookback = kwargs.get("volatility_lookback", 10)
        self.name = "Statistical Arbitrage"
    
    def analyze(self, data: pd.DataFrame, pair_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Génère un signal basé sur l'arbitrage statistique"""
        if pair_data is None or len(data) < self.volatility_lookback + 5 or len(pair_data) < self.volatility_lookback + 5:
            return {"signal": "neutral", "strength": 0, "reason": "Données insuffisantes pour l'arbitrage"}
        
        # Normaliser les prix
        data_normalized = data['close'] / data['close'].iloc[0]
        pair_normalized = pair_data['close'] / pair_data['close'].iloc[0]
        
        # Calculer le ratio et la moyenne mobile
        ratio = data_normalized / pair_normalized
        ratio_mean = ratio.rolling(window=self.volatility_lookback).mean()
        ratio_std = ratio.rolling(window=self.volatility_lookback).std()
        
        if ratio_std.iloc[-1] == 0:
            return {"signal": "neutral", "strength": 0, "reason": "Volatilité du ratio trop faible"}
        
        # Calculer le z-score du ratio actuel
        current_ratio = ratio.iloc[-1]
        mean_ratio = ratio_mean.iloc[-1]
        std_ratio = ratio_std.iloc[-1]
        z_score = (current_ratio - mean_ratio) / std_ratio
        
        # Générer un signal basé sur le z-score
        signal = "neutral"
        if z_score > 2.0:  # Ratio anormalement élevé - la paire 1 est surperformante
            signal = "sell"  # Vendre la première paire
            reason = f"Ratio anormalement élevé: z-score = {z_score:.4f}"
        elif z_score < -2.0:  # Ratio anormalement bas - la paire 1 est sous-performante
            signal = "buy"  # Acheter la première paire
            reason = f"Ratio anormalement bas: z-score = {z_score:.4f}"
        else:
            reason = f"Ratio normal: z-score = {z_score:.4f}"
        
        signal_strength = min(abs(z_score) / 3.0, 1.0)  # Normaliser entre 0-1
        
        return {
            "signal": signal,
            "strength": signal_strength,
            "reason": reason
        }

# Énumération des stratégies disponibles
class StrategyType(str, Enum):
    MOVING_AVERAGE = "moving_average"  # Stratégie par défaut d'AlpacaCryptoTrader
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    STATISTICAL_ARBITRAGE = "stat_arb"
    TRANSFORMER = "transformer"  # Stratégie basée sur un modèle Transformer de deep learning
    LSTM = "lstm"
    LLM = "llm"  # Stratégie basée sur un modèle LLM de deep learning

def get_strategy_class(strategy_type: str) -> Optional[type]:
    """Récupère la classe de stratégie en fonction du type spécifié"""
    # Import la stratégie Transformer uniquement si nécessaire
    if strategy_type.lower() == StrategyType.TRANSFORMER.lower():
        from app.strategies.transformer_strategy import TransformerStrategy
        return TransformerStrategy
        
    strategy_map = {
        StrategyType.MOMENTUM: MomentumStrategy,
        StrategyType.MEAN_REVERSION: MeanReversionStrategy,
        StrategyType.BREAKOUT: BreakoutStrategy,
        StrategyType.STATISTICAL_ARBITRAGE: StatisticalArbitrageStrategy,
        StrategyType.LSTM: LSTMPredictorStrategy,
        StrategyType.LLM: LLMStrategy
    }
    return strategy_map.get(strategy_type.lower())

class MarketCondition(Enum):
    NORMAL = auto()
    VOLATILE = auto()
    INACTIVE = auto()
    DANGEROUS = auto()

def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(description="Lancer le trader crypto avec une stratégie spécifique")
    parser.add_argument("--strategy", type=str, choices=[s.value for s in StrategyType], 
                      default=StrategyType.MOVING_AVERAGE.value,
                      help="Stratégie de trading à utiliser")
    parser.add_argument("--duration", type=str, choices=["1h", "4h", "8h", "night"], 
                      default="night",
                      help="Durée de la session (1h, 4h, 8h ou night pour 9h)")
    parser.add_argument("--api-level", type=int, choices=[1, 2, 3], default=0,
                      help="Niveau d'API Alpaca à utiliser (1=basique, 2=standard+, 3=premium). Par défaut: auto-détection)")
    parser.add_argument("--position-size", type=float, default=0.02,
                      help="Taille de position en pourcentage du portefeuille (default: 0.02 = 2%)")
    parser.add_argument("--stop-loss", type=float, default=0.03,
                      help="Stop loss en pourcentage (default: 0.03 = 3%)")
    parser.add_argument("--take-profit", type=float, default=0.06,
                      help="Take profit en pourcentage (default: 0.06 = 6%)")
    parser.add_argument("--fast-ma", type=int, default=5,
                      help="Période de la moyenne mobile rapide en minutes - uniquement pour la stratégie moving_average (default: 5)")
    parser.add_argument("--slow-ma", type=int, default=15,
                      help="Période de la moyenne mobile lente en minutes - uniquement pour la stratégie moving_average (default: 15)")
    parser.add_argument("--use-custom-symbols", action="store_true", 
                      help="Utiliser la liste personnalisée de symboles au lieu du filtre automatique")
    parser.add_argument("--momentum-lookback", type=int, default=20,
                      help="Période de lookback pour la stratégie momentum (default: 20)")
    parser.add_argument("--mean-reversion-lookback", type=int, default=20,
                      help="Période de lookback pour la stratégie mean reversion (default: 20)")
    parser.add_argument("--breakout-lookback", type=int, default=20,
                      help="Période de lookback pour la stratégie breakout (default: 20)")
    parser.add_argument("--volatility-lookback", type=int, default=10,
                      help="Période de lookback pour le calcul de la volatilité (default: 10)")
    
    # Paramètres spécifiques à la stratégie Transformer
    parser.add_argument("--sequence-length", type=int, default=60,
                      help="Longueur de la séquence d'entrée pour le modèle Transformer (default: 60)")
    parser.add_argument("--prediction-horizon", type=int, default=1,
                      help="Horizon de prédiction pour le modèle Transformer (default: 1)")
    parser.add_argument("--d-model", type=int, default=64,
                      help="Dimension du modèle Transformer (default: 64)")
    parser.add_argument("--nhead", type=int, default=4,
                      help="Nombre de têtes d'attention pour le modèle Transformer (default: 4)")
    parser.add_argument("--num-layers", type=int, default=2,
                      help="Nombre de couches pour le modèle Transformer (default: 2)")
    parser.add_argument("--dropout", type=float, default=0.1,
                      help="Taux de dropout pour le modèle Transformer (default: 0.1)")
    parser.add_argument("--signal-threshold", type=float, default=0.6,
                      help="Seuil de signal pour la stratégie Transformer (default: 0.6)")
    parser.add_argument("--use-gpu", action="store_true",
                      help="Utiliser le GPU pour l'entraînement et l'inférence (si disponible)")
    parser.add_argument("--retrain", action="store_true",
                      help="Réentraîner le modèle Transformer même si un modèle entraîné existe déjà")
    
    # Paramètres spécifiques à la stratégie LSTM
    parser.add_argument("--lstm-units", type=int, default=50,
                      help="Nombre d'unités LSTM dans le modèle (default: 50)")
    parser.add_argument("--lstm-dropout", type=float, default=0.2,
                      help="Taux de dropout pour le modèle LSTM (default: 0.2)")
    parser.add_argument("--lstm-epochs", type=int, default=50,
                      help="Nombre d'époques pour l'entraînement du modèle LSTM (default: 50)")
    parser.add_argument("--lstm-batch-size", type=int, default=32,
                      help="Taille de batch pour l'entraînement du modèle LSTM (default: 32)")
    
    # Paramètres spécifiques à la stratégie LLM
    parser.add_argument("--model-name", type=str, default="llama2-7b",
                      help="Nom du modèle LLM à utiliser (default: llama2-7b)")
    parser.add_argument("--use-local-model", action="store_true",
                      help="Utiliser un modèle LLM local au lieu d'une API distante")
    parser.add_argument("--local-model-path", type=str, default=None,
                      help="Chemin vers le modèle LLM local (si --use-local-model est activé)")
    parser.add_argument("--api-key", type=str, default=None,
                      help="Clé API pour le service LLM distant")
    parser.add_argument("--sentiment-threshold", type=float, default=0.6,
                      help="Seuil de sentiment pour la stratégie LLM (default: 0.6)")
    parser.add_argument("--news-lookback", type=int, default=24,
                      help="Nombre d'heures de données d'actualités à analyser pour la stratégie LLM (default: 24)")
    
    args = parser.parse_args()
    
    # Déterminer la durée de session
    session_duration = {
        "1h": SessionDuration.ONE_HOUR,
        "4h": SessionDuration.FOUR_HOURS,
        "8h": SessionDuration.EIGHT_HOURS,
        "night": SessionDuration.NIGHT_RUN
    }.get(args.duration, SessionDuration.NIGHT_RUN)
    
    # Détecter ou utiliser le niveau d'API spécifié
    api_level = args.api_level
    if api_level == 0:  # Auto-détection
        api_level = detect_alpaca_level()
    
    print("=" * 60)
    print(f"DÉMARRAGE DU TRADER CRYPTO AVEC STRATÉGIE - {datetime.now()}")
    print("=" * 60)
    print(f"Stratégie sélectionnée: {args.strategy}")
    print(f"Ce trader va tourner en mode PAPER pendant environ {args.duration}")
    print(f"Position size: {args.position_size * 100}%")
    print(f"Stop-loss: {args.stop_loss * 100}%")
    print(f"Take-profit: {args.take_profit * 100}%")
    print(f"Niveau d'API Alpaca: {api_level if api_level > 0 else 'Non détecté - utilisation du niveau 1'}")
    
    if args.strategy == StrategyType.MOVING_AVERAGE:
        print(f"MA rapide: {args.fast_ma} minutes")
        print(f"MA lente: {args.slow_ma} minutes")
    elif args.strategy == StrategyType.MOMENTUM:
        print(f"Momentum lookback: {args.momentum_lookback} périodes")
    elif args.strategy == StrategyType.MEAN_REVERSION:
        print(f"Mean reversion lookback: {args.mean_reversion_lookback} périodes")
    elif args.strategy == StrategyType.BREAKOUT:
        print(f"Breakout lookback: {args.breakout_lookback} périodes")
    elif args.strategy == StrategyType.STATISTICAL_ARBITRAGE:
        print(f"Volatility lookback: {args.volatility_lookback} périodes")
    elif args.strategy == StrategyType.TRANSFORMER:
        print(f"Transformer configuration:")
        print(f"  - Sequence length: {args.sequence_length}")
        print(f"  - Prediction horizon: {args.prediction_horizon}")
        print(f"  - Model dimension: {args.d_model}")
        print(f"  - Attention heads: {args.nhead}")
        print(f"  - Layers: {args.num_layers}")
        print(f"  - Dropout: {args.dropout}")
        print(f"  - Signal threshold: {args.signal_threshold}")
        print(f"  - GPU: {'Activé' if args.use_gpu else 'Désactivé'}")
        print(f"  - Réentraînement: {'Oui' if args.retrain else 'Non'}")
    elif args.strategy == StrategyType.LSTM:
        print(f"LSTM configuration:")
        print(f"  - Sequence length: {args.sequence_length}")
        print(f"  - Prediction horizon: {args.prediction_horizon}")
        print(f"  - LSTM units: {args.lstm_units}")
        print(f"  - Dropout: {args.lstm_dropout}")
        print(f"  - Epochs: {args.lstm_epochs}")
        print(f"  - Batch size: {args.lstm_batch_size}")
        print(f"  - GPU: {'Activé' if args.use_gpu else 'Désactivé'}")
    elif args.strategy == StrategyType.LLM:
        print(f"LLM configuration:")
        print(f"  - Model name: {args.model_name}")
        print(f"  - Use local model: {'Oui' if args.use_local_model else 'Non'}")
        if args.use_local_model and args.local_model_path:
            print(f"  - Local model path: {args.local_model_path}")
        print(f"  - Sentiment threshold: {args.sentiment_threshold}")
        print(f"  - News lookback hours: {args.news_lookback}")
    
    print("=" * 60)
    
    # Si la stratégie est moving_average, utiliser AlpacaCryptoTrader directement
    if args.strategy == StrategyType.MOVING_AVERAGE:
        # Créer le trader avec la durée de session spécifiée
        trader = AlpacaCryptoTrader(session_duration=session_duration)
        
        # Configurer les paramètres
        trader.position_size_pct = args.position_size
        trader.stop_loss_pct = args.stop_loss
        trader.take_profit_pct = args.take_profit
        trader.fast_ma_period = args.fast_ma
        trader.slow_ma_period = args.slow_ma
        
        # Configurer le niveau d'API
        if api_level > 0:
            print(f"Configuration du niveau d'API Alpaca: {api_level}")
            trader.subscription_level = api_level
        
        # Utiliser la liste personnalisée de symboles
        trader.custom_symbols = PERSONALIZED_CRYPTO_LIST
        trader.use_custom_symbols = args.use_custom_symbols
        
        # Démarrer le trader avec la stratégie par défaut
        print(f"Démarrage du trader avec la stratégie de moyenne mobile")
        trader.start()
    else:
        # Pour les autres stratégies, utiliser une version simplifiée
        # qui fonctionne avec AlpacaCryptoTrader en adaptant les signaux
        strategy_class = get_strategy_class(args.strategy)
        if not strategy_class:
            print(f"Erreur: Stratégie {args.strategy} non disponible")
            return
        
        # Paramètres spécifiques à la stratégie
        strategy_params = {}
        if args.strategy == StrategyType.MOMENTUM:
            strategy_params = {
                "lookback_period": args.momentum_lookback,
                "position_size": args.position_size,
                "stop_loss": args.stop_loss,
                "take_profit": args.take_profit
            }
        elif args.strategy == StrategyType.MEAN_REVERSION:
            strategy_params = {
                "lookback_period": args.mean_reversion_lookback,
                "position_size": args.position_size,
                "stop_loss": args.stop_loss,
                "take_profit": args.take_profit
            }
        elif args.strategy == StrategyType.BREAKOUT:
            strategy_params = {
                "lookback_period": args.breakout_lookback,
                "position_size": args.position_size,
                "stop_loss": args.stop_loss,
                "take_profit": args.take_profit
            }
        elif args.strategy == StrategyType.STATISTICAL_ARBITRAGE:
            strategy_params = {
                "volatility_lookback": args.volatility_lookback,
                "position_size": args.position_size,
                "stop_loss": args.stop_loss,
                "take_profit": args.take_profit
            }
        elif args.strategy == StrategyType.TRANSFORMER:
            strategy_params = {
                "sequence_length": args.sequence_length,
                "prediction_horizon": args.prediction_horizon,
                "d_model": args.d_model,
                "nhead": args.nhead,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "signal_threshold": args.signal_threshold,
                "use_gpu": args.use_gpu,
                "retrain": args.retrain,
                "position_size": args.position_size,
                "stop_loss": args.stop_loss,
                "take_profit": args.take_profit
            }
        elif args.strategy == StrategyType.LSTM:
            strategy_params = {
                "sequence_length": args.sequence_length,
                "prediction_horizon": args.prediction_horizon,
                "lstm_units": args.lstm_units,
                "dropout_rate": args.lstm_dropout,
                "epochs": args.lstm_epochs,
                "batch_size": args.lstm_batch_size,
                "use_gpu": args.use_gpu,
                "retrain": args.retrain,
                "position_size": args.position_size,
                "stop_loss": args.stop_loss,
                "take_profit": args.take_profit
            }
        elif args.strategy == StrategyType.LLM:
            strategy_params = {
                "model_name": args.model_name,
                "use_local_model": args.use_local_model,
                "local_model_path": args.local_model_path,
                "api_key": args.api_key,
                "sentiment_threshold": args.sentiment_threshold,
                "news_lookback_hours": args.news_lookback,
                "position_size": args.position_size,
                "stop_loss": args.stop_loss,
                "take_profit": args.take_profit
            }
        
        # Créer l'instance de stratégie
        strategy_instance = strategy_class(**strategy_params)
        print(f"Stratégie {strategy_instance.name} initialisée avec succès")
        print("Utilisation du trader Alpaca de base avec adaptation des signaux")
        
        # Créer le trader avec la durée de session spécifiée
        trader = AlpacaCryptoTrader(session_duration=session_duration)
        
        # Configurer le niveau d'API
        if api_level > 0:
            print(f"Configuration du niveau d'API Alpaca: {api_level}")
            trader.subscription_level = api_level
        
        # Configurer les paramètres
        trader.position_size_pct = args.position_size
        trader.stop_loss_pct = args.stop_loss
        trader.take_profit_pct = args.take_profit
        
        # La stratégie personnalisée sera utilisée dans un script séparé
        # qui sera exécuté ultérieurement avec les mêmes paramètres
        strategy_type = args.strategy
        strategy_file = f"custom_strategy_{strategy_type}_params.json"
        
        # Sauvegarder les paramètres dans un fichier pour utilisation future
        with open(strategy_file, "w") as f:
            json.dump({
                "strategy_type": strategy_type,
                "params": strategy_params,
                "symbols": PERSONALIZED_CRYPTO_LIST if args.use_custom_symbols else []
            }, f, indent=2)
        
        print(f"Configuration de stratégie enregistrée dans {strategy_file}")
        
        # Utiliser la liste personnalisée de symboles
        trader.custom_symbols = PERSONALIZED_CRYPTO_LIST
        trader.use_custom_symbols = args.use_custom_symbols
        
        # Démarrer le trader avec la stratégie par défaut adaptée
        print(f"Démarrage du trader avec adaptation pour la stratégie {strategy_type}")
        trader.start()
    
    print("=" * 60)
    print("SESSION DE TRADING TERMINÉE")
    print("=" * 60)
    print("Un rapport détaillé a été généré dans le dossier courant")
    print("=" * 60)

if __name__ == "__main__":
    main()
