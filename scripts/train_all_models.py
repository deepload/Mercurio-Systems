#!/usr/bin/env python
"""
MercurioAI - Train All Models

Ce script permet d'entraîner tous les modèles d'IA utilisés par MercurioAI
en une seule commande. Il entraîne automatiquement les modèles LSTM et 
Transformer sur les actifs spécifiés ou sur une liste d'actifs populaires.
Le script peut également utiliser tous les symboles générés par get_all_symbols.py.

Exemples d'utilisation:
    # Utilisation basique avec des symboles par défaut
    python scripts/train_all_models.py --days 90 --top_assets 20
    
    # Spécifier des symboles directement
    python scripts/train_all_models.py --symbols BTC-USD,ETH-USD,AAPL,MSFT,TSLA
    
    # Utiliser tous les symboles récupérés par get_all_symbols.py (limité à 100)
    python scripts/train_all_models.py --all_symbols --max_symbols 100 --epochs 10
    
    # Sélection aléatoire d'un sous-ensemble de symboles
    python scripts/train_all_models.py --all_symbols --max_symbols 500 --random_select
    
    # Traitement par lots pour les grandes listes de symboles
    python scripts/train_all_models.py --all_symbols --batch_mode --batch_size 50
    
    # Activer automatiquement le mode batch pour les grandes listes
    python scripts/train_all_models.py --all_symbols --auto_batch
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

async def load_symbols_from_csv(file_path, max_symbols=None, random_select=False):
    """
    Charge une liste de symboles à partir d'un fichier CSV
    
    Args:
        file_path: Chemin vers le fichier CSV
        max_symbols: Nombre maximum de symboles à charger (None pour tous)
        random_select: Si True, sélectionne aléatoirement les symboles au lieu des premiers
        
    Returns:
        Liste des symboles
    """
    try:
        # Vérifier que le fichier existe
        if not os.path.exists(file_path):
            logger.error(f"Fichier {file_path} introuvable")
            return []
        
        # Charger le CSV
        df = pd.read_csv(file_path)
        
        # Extraire la colonne des symboles
        if 'symbol' not in df.columns:
            logger.error(f"Colonne 'symbol' introuvable dans {file_path}")
            return []
        
        all_symbols = df['symbol'].tolist()
        total_symbols = len(all_symbols)
        
        # Appliquer la limitation si nécessaire
        if max_symbols is not None and max_symbols < total_symbols:
            if random_select:
                import random
                symbols = random.sample(all_symbols, max_symbols)
                logger.info(f"Sélection aléatoire de {max_symbols} symboles parmi {total_symbols} depuis {file_path}")
            else:
                symbols = all_symbols[:max_symbols]
                logger.info(f"Sélection des {max_symbols} premiers symboles parmi {total_symbols} depuis {file_path}")
        else:
            symbols = all_symbols
            logger.info(f"Chargé {len(symbols)} symboles depuis {file_path}")
            
        return symbols
    except Exception as e:
        logger.error(f"Erreur lors du chargement des symboles depuis {file_path}: {e}")
        return []

async def find_latest_symbols_files():
    """
    Recherche les fichiers CSV les plus récents générés par get_all_symbols.py
    
    Returns:
        Tuple (fichier stocks, fichier crypto)
    """
    data_dir = Path("data")
    
    # Vérifier que le répertoire existe
    if not data_dir.exists() or not data_dir.is_dir():
        logger.warning(f"Répertoire de données {data_dir} introuvable")
        return None, None
    
    # Rechercher les fichiers correspondants
    stock_files = sorted(data_dir.glob("all_stocks_*.csv"), reverse=True)
    crypto_files = sorted(data_dir.glob("all_crypto_*.csv"), reverse=True)
    
    stock_file = stock_files[0] if stock_files else None
    crypto_file = crypto_files[0] if crypto_files else None
    
    if stock_file:
        logger.info(f"Fichier de symboles d'actions le plus récent: {stock_file}")
    if crypto_file:
        logger.info(f"Fichier de symboles de crypto le plus récent: {crypto_file}")
        
    return stock_file, crypto_file

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
    parser.add_argument("--custom_stocks_file", type=str, default="",
                       help="Chemin vers un fichier CSV contenant une liste personnalisée d'actions")
    parser.add_argument("--custom_crypto_file", type=str, default="",
                       help="Chemin vers un fichier CSV contenant une liste personnalisée de cryptomonnaies")
    parser.add_argument("--max_symbols", type=int, default=0,
                       help="Limite le nombre total de symboles à entraîner (0 = pas de limite)")
    parser.add_argument("--batch_mode", action='store_true',
                       help="Mode batch: traite les symboles par lots pour les grandes listes")
    parser.add_argument("--batch_size", type=int, default=20,
                       help="Taille des lots en mode batch (défaut: 20 symboles par lot)")
    parser.add_argument("--all_symbols", action='store_true',
                       help="Utiliser tous les symboles disponibles dans les fichiers générés par get_all_symbols.py")
    parser.add_argument("--random_select", action='store_true',
                       help="Sélectionner aléatoirement les symboles plutôt que les premiers de la liste")
    parser.add_argument("--auto_batch", action='store_true',
                       help="Active automatiquement le mode batch pour les grandes listes de symboles")
    
    args = parser.parse_args()
    
    # Déterminer la liste des symboles à utiliser
    symbols = []
    
    # Option --all_symbols : utiliser les fichiers générés par get_all_symbols.py
    if args.all_symbols:
        logger.info("Recherche des fichiers de symboles les plus récents...")
        stock_file, crypto_file = await find_latest_symbols_files()
        
        if stock_file:
            max_stock_symbols = args.max_symbols if args.max_symbols > 0 else None
            custom_stocks = await load_symbols_from_csv(stock_file, max_stock_symbols, args.random_select)
            symbols.extend(custom_stocks)
            logger.info(f"Ajout de {len(custom_stocks)} actions depuis {stock_file}")
            
        if crypto_file:
            # Pour les crypto, on limite à un nombre plus petit par défaut, sauf si spécifié autrement
            max_crypto_symbols = min(50, args.max_symbols) if args.max_symbols > 0 else 50
            custom_crypto = await load_symbols_from_csv(crypto_file, max_crypto_symbols, args.random_select)
            symbols.extend(custom_crypto)
            logger.info(f"Ajout de {len(custom_crypto)} cryptomonnaies depuis {crypto_file}")
    else:
        # Chargement à partir de fichiers personnalisés
        if args.custom_stocks_file:
            custom_stocks = await load_symbols_from_csv(args.custom_stocks_file, args.max_symbols, args.random_select)
            symbols.extend(custom_stocks)
            logger.info(f"Ajout de {len(custom_stocks)} actions depuis le fichier personnalisé")
            
        if args.custom_crypto_file:
            custom_crypto = await load_symbols_from_csv(args.custom_crypto_file, args.max_symbols, args.random_select)
            symbols.extend(custom_crypto)
            logger.info(f"Ajout de {len(custom_crypto)} cryptomonnaies depuis le fichier personnalisé")
        
        # Si des symboles sont spécifiés directement
        if args.symbols:
            direct_symbols = args.symbols.split(',')
            symbols.extend(direct_symbols)
            logger.info(f"Ajout de {len(direct_symbols)} symboles spécifiés directement")
        
        # Si aucun symbole n'a été spécifié via les options ci-dessus
        if not symbols:
            # Utiliser les actifs populaires
            if args.include_stocks or not (args.include_stocks or args.include_crypto):
                stock_symbols = DEFAULT_STOCKS[:args.top_assets] if args.top_assets > 0 else DEFAULT_STOCKS
                symbols.extend(stock_symbols)
                logger.info(f"Ajout de {len(stock_symbols)} actions populaires par défaut")
                
            if args.include_crypto or not (args.include_stocks or args.include_crypto):
                crypto_symbols = DEFAULT_CRYPTO[:args.top_assets] if args.top_assets > 0 else DEFAULT_CRYPTO
                symbols.extend(crypto_symbols)
                logger.info(f"Ajout de {len(crypto_symbols)} cryptomonnaies populaires par défaut")
    
    # Éliminer les doublons
    symbols = list(set(symbols))
    
    # Limite le nombre de symboles si spécifié
    if args.max_symbols > 0 and len(symbols) > args.max_symbols:
        logger.warning(f"Limitation à {args.max_symbols} symboles (sur {len(symbols)} au total)")
        symbols = symbols[:args.max_symbols]
    
    if not symbols:
        logger.error("Aucun symbole à traiter. Veuillez spécifier des symboles ou utiliser --include_stocks ou --include_crypto.")
        return 1
    
    logger.info(f"Entraînement des modèles pour {len(symbols)} symboles")
    if len(symbols) <= 20:
        logger.info(f"Liste des symboles: {', '.join(symbols)}")
    else:
        logger.info(f"Premiers symboles: {', '.join(symbols[:10])}... et {len(symbols)-10} autres")
        
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
    
    # Déterminer si le mode batch doit être utilisé
    use_batch_mode = args.batch_mode or (args.auto_batch and len(symbols) > args.batch_size)
    
    # Si le mode batch est activé et qu'il y a beaucoup de symboles
    if use_batch_mode and len(symbols) > args.batch_size:
        logger.info(f"Mode batch activé: traitement par lots de {args.batch_size} symboles")
        
        all_results = {'lstm_models': {}, 'transformer_model': None}
        
        # Traitement des modèles LSTM par lots
        for i in range(0, len(symbols), args.batch_size):
            batch_symbols = symbols[i:i+args.batch_size]
            logger.info(f"Traitement du lot {i//args.batch_size + 1}/{(len(symbols)-1)//args.batch_size + 1}: {len(batch_symbols)} symboles")
            
            # Entraînement des modèles LSTM
            batch_results = await train_all_models(
                symbols=batch_symbols,
                lookback_days=args.days,
                lstm_params=lstm_params,
                transformer_params=None  # Ne pas entraîner Transformer par lot
            )
            
            # Fusion des résultats
            all_results['lstm_models'].update(batch_results['lstm_models'])
        
        # Entraînement du modèle Transformer sur tous les symboles à la fin
        logger.info(f"Entraînement du modèle Transformer sur tous les symboles")
        transformer_result = await train_transformer_model(
            symbols=symbols,
            start_date=datetime.now() - timedelta(days=args.days),
            end_date=datetime.now(),
            **transformer_params
        )
        all_results['transformer_model'] = transformer_result
        
        results = all_results
    else:
        # Entraînement normal de tous les modèles
        results = await train_all_models(
            symbols=symbols,
            lookback_days=args.days,
            lstm_params=lstm_params,
            transformer_params=transformer_params
        )
    
    # Affichage des résultats
    logger.info("Entraînement terminé!")
    successful_lstm = len([m for m in results['lstm_models'].values() if m is not None])
    logger.info(f"Modèles LSTM entraînés: {successful_lstm}/{len(symbols)} ({successful_lstm/len(symbols)*100:.1f}%)")
    logger.info(f"Modèle Transformer entraîné: {'Oui' if results['transformer_model'] else 'Non'}")
    
    # Sauvegarde d'un rapport de l'entraînement
    report_path = f"reports/training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w', newline='') as f:
        writer = pd.DataFrame({
            'symbol': list(results['lstm_models'].keys()),
            'lstm_trained': [bool(m) for m in results['lstm_models'].values()],
            'transformed_included': [True] * len(results['lstm_models'])
        }).to_csv(f, index=False)
    
    logger.info(f"Rapport d'entraînement sauvegardé dans {report_path}")
    
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
