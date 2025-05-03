#!/usr/bin/env python
"""
MercurioAI - Train All Models

Ce script permet d'entraîner tous les modèles d'IA utilisés par MercurioAI
en une seule commande. Il entraîne automatiquement les modèles LSTM et 
Transformer sur les actifs spécifiés ou sur une liste d'actifs populaires.

Exemple d'utilisation:
    python scripts/train_all_models.py --days 90 --top_assets 20
    python scripts/train_all_models.py --symbols BTC-USD,ETH-USD,AAPL,MSFT,TSLA
"""

import os
import sys
import logging
import argparse
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional

# Assurez-vous que le script peut importer les modules MercurioAI
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Assurez-vous que les répertoires nécessaires existent
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("models/lstm", exist_ok=True)
os.makedirs("models/transformer", exist_ok=True)

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/train_all_models.log")
    ]
)
logger = logging.getLogger(__name__)

# Importez les modules MercurioAI
from app.services.market_data import MarketDataService
from app.strategies.lstm_predictor import LSTMPredictorStrategy
from app.strategies.transformer_strategy import TransformerStrategy

# Listes par défaut d'actifs populaires
DEFAULT_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "PG",
    "UNH", "HD", "BAC", "ADBE", "CRM", "PFE", "NFLX", "AVGO", "CSCO"
]

DEFAULT_CRYPTO = [
    "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "DOT-USD", "XRP-USD", "DOGE-USD", 
    "AVAX-USD", "LUNA-USD", "SHIB-USD", "MATIC-USD", "LINK-USD", "UNI-USD"
]

async def train_lstm_model(symbol: str, start_date: datetime, end_date: datetime, 
                    sequence_length: int = 60, prediction_horizon: int = 5, 
                    lstm_units: int = 50, dropout_rate: float = 0.2, 
                    batch_size: int = 32, epochs: int = 50) -> Optional[str]:
    """
    Entraîne un modèle LSTM pour un symbole spécifique.
    
    Args:
        symbol: Symbole de l'actif
        start_date: Date de début des données
        end_date: Date de fin des données
        sequence_length: Longueur des séquences
        prediction_horizon: Nombre de périodes à prédire
        lstm_units: Nombre d'unités LSTM
        dropout_rate: Taux de dropout
        batch_size: Taille des lots
        epochs: Nombre d'époques
        
    Returns:
        Path du modèle ou None en cas d'échec
    """
    try:
        logger.info(f"Entraînement du modèle LSTM pour {symbol}")
        
        # Initialisation de la stratégie
        strategy = LSTMPredictorStrategy(
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            lstm_units=lstm_units,
            dropout_rate=dropout_rate,
            epochs=epochs,
            batch_size=batch_size
        )
        
        # Chargement des données
        market_data = MarketDataService()
        data = await market_data.get_historical_data(symbol, start_date, end_date)
        
        if data is None or len(data) < sequence_length + prediction_horizon:
            logger.warning(f"Données insuffisantes pour {symbol}. Au moins {sequence_length + prediction_horizon} points de données sont nécessaires.")
            return None
        
        # Prétraitement des données
        processed_data = await strategy.preprocess_data(data)
        
        # Entraînement du modèle
        training_result = await strategy.train(processed_data)
        
        # Sauvegarde du modèle
        model_dir = Path(f"models/lstm/{symbol.replace('/', '_').replace('-', '_').lower()}")
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = await strategy.save_model(str(model_dir))
        
        logger.info(f"Modèle LSTM pour {symbol} sauvegardé dans: {model_path}")
        return model_path
        
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement du modèle LSTM pour {symbol}: {e}")
        return None

async def train_transformer_model(symbols: List[str], start_date: datetime, end_date: datetime, 
                          sequence_length: int = 60, prediction_horizon: int = 5, 
                          d_model: int = 64, nhead: int = 4, num_layers: int = 2, 
                          dropout_rate: float = 0.1, batch_size: int = 32, 
                          epochs: int = 50, use_gpu: bool = True) -> Optional[Path]:
    """
    Entraîne un modèle Transformer sur plusieurs symboles.
    
    Args:
        symbols: Liste des symboles d'actifs
        start_date: Date de début des données
        end_date: Date de fin des données
        sequence_length: Longueur des séquences
        prediction_horizon: Nombre de périodes à prédire
        d_model: Dimension du modèle
        nhead: Nombre de têtes d'attention
        num_layers: Nombre de couches
        dropout_rate: Taux de dropout
        batch_size: Taille des lots
        epochs: Nombre d'époques
        use_gpu: Utiliser le GPU si disponible
        
    Returns:
        Path du dossier du modèle ou None en cas d'échec
    """
    try:
        logger.info(f"Entraînement du modèle Transformer sur {len(symbols)} symboles")
        
        # Initialisation de la stratégie
        strategy = TransformerStrategy(
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout_rate,
            batch_size=batch_size,
            epochs=epochs,
            use_gpu=use_gpu
        )
        
        # Chargement et préparation des données
        market_data = MarketDataService()
        combined_data = []
        
        for symbol in symbols:
            logger.info(f"Chargement des données pour {symbol}")
            data = await market_data.get_historical_data(symbol, start_date, end_date)
            
            if data is None or len(data) < sequence_length + prediction_horizon:
                logger.warning(f"Données insuffisantes pour {symbol}. Ignoré.")
                continue
            
            # Prétraitement des données
            processed_data = await strategy.preprocess_data(data)
            processed_data['symbol'] = symbol
            combined_data.append(processed_data)
        
        if not combined_data:
            logger.error("Aucune donnée valide pour l'entraînement du Transformer.")
            return None
        
        # Concaténation des données
        all_data = pd.concat(combined_data, ignore_index=True)
        
        # Entraînement
        logger.info(f"Entraînement du modèle Transformer sur {len(combined_data)} symboles")
        training_result = await strategy.train(all_data)
        
        # Sauvegarde
        model_dir = Path("models/transformer/multi_asset")
        os.makedirs(model_dir, exist_ok=True)
        
        strategy._save_model()
        
        logger.info(f"Modèle Transformer sauvegardé dans: {model_dir}")
        return model_dir
        
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement du modèle Transformer: {e}")
        return None

async def train_all_models(symbols: List[str], lookback_days: int = 180, 
                    lstm_params: Dict[str, Any] = None, 
                    transformer_params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Entraîne tous les modèles d'IA disponibles dans MercurioAI.
    
    Args:
        symbols: Liste des symboles d'actifs
        lookback_days: Nombre de jours d'historique
        lstm_params: Paramètres spécifiques pour le modèle LSTM
        transformer_params: Paramètres spécifiques pour le modèle Transformer
        
    Returns:
        Dict contenant les résultats de l'entraînement
    """
    # Dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    # Paramètres par défaut
    if lstm_params is None:
        lstm_params = {
            'sequence_length': 60,
            'prediction_horizon': 5,
            'lstm_units': 50,
            'dropout_rate': 0.2,
            'batch_size': 32,
            'epochs': 50
        }
    
    if transformer_params is None:
        transformer_params = {
            'sequence_length': 60,
            'prediction_horizon': 5,
            'd_model': 64,
            'nhead': 4,
            'num_layers': 2,
            'dropout_rate': 0.1,
            'batch_size': 32,
            'epochs': 50,
            'use_gpu': True
        }
    
    results = {
        'lstm_models': {},
        'transformer_model': None,
        'trained_symbols': symbols,
        'start_date': start_date,
        'end_date': end_date
    }
    
    # Entraînement des modèles LSTM pour chaque symbole
    for symbol in symbols:
        logger.info(f"Entraînement du modèle LSTM pour {symbol}")
        lstm_result = await train_lstm_model(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            **lstm_params
        )
        results['lstm_models'][symbol] = lstm_result
    
    # Entraînement du modèle Transformer sur tous les symboles
    logger.info(f"Entraînement du modèle Transformer sur tous les symboles")
    transformer_result = await train_transformer_model(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        **transformer_params
    )
    results['transformer_model'] = transformer_result
    
    return results

async def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="MercurioAI - Train All Models")
    
    parser.add_argument("--symbols", type=str, default="",
                       help="Liste des symboles d'actifs séparés par des virgules (ex: BTC-USD,ETH-USD,AAPL)")
    parser.add_argument("--days", type=int, default=180,
                       help="Nombre de jours d'historique à utiliser pour l'entraînement (défaut: 180)")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Nombre d'époques d'entraînement pour tous les modèles (défaut: 50)")
    parser.add_argument("--top_assets", type=int, default=10,
                       help="Nombre d'actifs populaires à inclure automatiquement (défaut: 10)")
    parser.add_argument("--include_stocks", action='store_true',
                       help="Inclure les actions populaires")
    parser.add_argument("--include_crypto", action='store_true',
                       help="Inclure les cryptomonnaies populaires")
    parser.add_argument("--use_gpu", action='store_true',
                       help="Utiliser le GPU si disponible")
    
    args = parser.parse_args()
    
    # Déterminer la liste des symboles à utiliser
    symbols = []
    
    if args.symbols:
        # Utiliser les symboles spécifiés par l'utilisateur
        symbols = args.symbols.split(',')
    else:
        # Utiliser les actifs populaires
        if args.include_stocks or not (args.include_stocks or args.include_crypto):
            symbols.extend(DEFAULT_STOCKS[:args.top_assets])
            
        if args.include_crypto or not (args.include_stocks or args.include_crypto):
            symbols.extend(DEFAULT_CRYPTO[:args.top_assets])
    
    logger.info(f"Entraînement des modèles pour {len(symbols)} symboles: {', '.join(symbols)}")
    logger.info(f"Période d'entraînement: {args.days} jours jusqu'à aujourd'hui")
    
    # Paramètres communs pour tous les modèles
    lstm_params = {
        'sequence_length': 60,
        'prediction_horizon': 5,
        'lstm_units': 50,
        'dropout_rate': 0.2,
        'batch_size': 32,
        'epochs': args.epochs
    }
    
    transformer_params = {
        'sequence_length': 60,
        'prediction_horizon': 5,
        'd_model': 64,
        'nhead': 4,
        'num_layers': 2,
        'dropout_rate': 0.1,
        'batch_size': 32,
        'epochs': args.epochs,
        'use_gpu': args.use_gpu
    }
    
    # Entraînement de tous les modèles
    results = await train_all_models(
        symbols=symbols,
        lookback_days=args.days,
        lstm_params=lstm_params,
        transformer_params=transformer_params
    )
    
    # Affichage des résultats
    logger.info("Entraînement terminé!")
    logger.info(f"Modèles LSTM entraînés: {len([m for m in results['lstm_models'].values() if m is not None])}/{len(symbols)}")
    logger.info(f"Modèle Transformer entraîné: {'Oui' if results['transformer_model'] else 'Non'}")
    
    # Vérification du succès
    if any(results['lstm_models'].values()) or results['transformer_model']:
        logger.info("Entraînement réussi! Au moins un modèle a été entraîné avec succès.")
        return 0
    else:
        logger.error("Échec de l'entraînement. Aucun modèle n'a été entraîné avec succès.")
        return 1

if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        logger.info("Programme interrompu par l'utilisateur")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Erreur critique non gérée: {e}")
        sys.exit(1)
