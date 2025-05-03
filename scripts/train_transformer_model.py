#!/usr/bin/env python
"""
MercurioAI - Train Transformer Model

Ce script permet d'entraîner un modèle Transformer pour la prédiction de prix d'actifs
financiers. Le modèle entraîné peut ensuite être utilisé par les stratégies
de trading et le screener d'actifs.

Exemple d'utilisation:
    python scripts/train_transformer_model.py --symbols BTC-USD,ETH-USD,AAPL,MSFT --epochs 100
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import torch

# Assurez-vous que le script peut importer les modules MercurioAI
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Assurez-vous que les répertoires nécessaires existent
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("models/transformer", exist_ok=True)

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/train_transformer_model.log")
    ]
)
logger = logging.getLogger(__name__)

# Importez les modules MercurioAI
from app.services.market_data import MarketDataService
from app.strategies.transformer_strategy import TransformerStrategy

async def train_transformer_model(symbols, start_date, end_date, sequence_length=60, 
                          prediction_horizon=5, d_model=64, nhead=4, num_layers=2, 
                          dropout_rate=0.1, batch_size=32, epochs=50, learning_rate=0.001, 
                          validation_split=0.2, use_gpu=True):
    """
    Entraîne un modèle Transformer pour la prédiction de prix.
    
    Args:
        symbols: Liste des symboles d'actifs pour l'entraînement
        start_date: Date de début des données d'entraînement
        end_date: Date de fin des données d'entraînement
        sequence_length: Longueur des séquences pour l'entraînement
        prediction_horizon: Nombre de périodes à prédire
        d_model: Dimension du modèle Transformer
        nhead: Nombre de têtes d'attention
        num_layers: Nombre de couches Transformer
        dropout_rate: Taux de dropout pour la régularisation
        batch_size: Taille des lots pour l'entraînement
        epochs: Nombre d'époques d'entraînement
        learning_rate: Taux d'apprentissage
        validation_split: Proportion des données pour la validation
        use_gpu: Utiliser le GPU si disponible
        
    Returns:
        Path du modèle sauvegardé
    """
    try:
        logger.info(f"Démarrage de l'entraînement du modèle Transformer pour {symbols}")
        
        # Initialisation de la stratégie Transformer
        strategy = TransformerStrategy(
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout_rate,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            use_gpu=use_gpu
        )
        
        # Chargement et préparation des données pour chaque symbole
        market_data = MarketDataService()
        combined_data = []
        
        for symbol in symbols:
            logger.info(f"Chargement des données pour {symbol} du {start_date} au {end_date}")
            data = await market_data.get_historical_data(symbol, start_date, end_date)
            
            if data is None or len(data) < sequence_length + prediction_horizon:
                logger.warning(f"Données insuffisantes pour {symbol}. Au moins {sequence_length + prediction_horizon} points de données sont nécessaires.")
                continue
            
            # Prétraitement des données
            processed_data = await strategy.preprocess_data(data)
            
            # Ajout d'une colonne pour identifier le symbole
            processed_data['symbol'] = symbol
            
            combined_data.append(processed_data)
        
        if not combined_data:
            logger.error("Aucune donnée valide pour l'entraînement.")
            return None
        
        # Concaténation des données de tous les symboles
        all_data = pd.concat(combined_data, ignore_index=True)
        
        # Entraînement du modèle
        logger.info("Début de l'entraînement...")
        training_result = await strategy.train(all_data)
        
        # Sauvegarde du modèle
        model_dir = Path("models/transformer/multi_asset")
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = strategy._save_model()  # Méthode privée, mais nécessaire ici
        
        # Évaluation du modèle
        logger.info(f"Entraînement terminé. Métriques: {training_result}")
        logger.info(f"Modèle sauvegardé dans: {model_dir}")
        
        return model_dir
        
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement du modèle Transformer: {e}")
        return None

async def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="MercurioAI - Train Transformer Model")
    
    parser.add_argument("--symbols", type=str, required=True,
                       help="Liste des symboles d'actifs séparés par des virgules (ex: BTC-USD,ETH-USD,AAPL)")
    parser.add_argument("--lookback", type=int, default=180,
                       help="Nombre de jours d'historique à utiliser pour l'entraînement (défaut: 180)")
    parser.add_argument("--sequence_length", type=int, default=60,
                       help="Longueur des séquences pour l'entraînement (défaut: 60)")
    parser.add_argument("--prediction_horizon", type=int, default=5,
                       help="Nombre de périodes à prédire (défaut: 5)")
    parser.add_argument("--d_model", type=int, default=64,
                       help="Dimension du modèle Transformer (défaut: 64)")
    parser.add_argument("--nhead", type=int, default=4,
                       help="Nombre de têtes d'attention (défaut: 4)")
    parser.add_argument("--num_layers", type=int, default=2,
                       help="Nombre de couches Transformer (défaut: 2)")
    parser.add_argument("--dropout_rate", type=float, default=0.1,
                       help="Taux de dropout pour la régularisation (défaut: 0.1)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Taille des lots pour l'entraînement (défaut: 32)")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Nombre d'époques d'entraînement (défaut: 50)")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                       help="Taux d'apprentissage (défaut: 0.001)")
    parser.add_argument("--validation_split", type=float, default=0.2,
                       help="Proportion des données pour la validation (défaut: 0.2)")
    parser.add_argument("--use_gpu", action='store_true',
                       help="Utiliser le GPU si disponible")
    
    args = parser.parse_args()
    
    # Calcul des dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.lookback)
    
    # Conversion des symboles en liste
    symbols = args.symbols.split(',')
    
    logger.info(f"Entraînement du modèle Transformer pour {len(symbols)} symboles")
    logger.info(f"Période d'entraînement: {start_date} à {end_date}")
    
    # Entraînement du modèle
    result = await train_transformer_model(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        sequence_length=args.sequence_length,
        prediction_horizon=args.prediction_horizon,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout_rate=args.dropout_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        validation_split=args.validation_split,
        use_gpu=args.use_gpu
    )
    
    if result:
        logger.info(f"Entraînement réussi! Modèle sauvegardé dans: {result}")
        return 0
    else:
        logger.error("Échec de l'entraînement du modèle.")
        return 1

if __name__ == "__main__":
    import asyncio
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        logger.info("Programme interrompu par l'utilisateur")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Erreur critique non gérée: {e}")
        sys.exit(1)
