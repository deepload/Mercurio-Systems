#!/usr/bin/env python
"""
MercurioAI - List Trained Models

Ce script affiche une liste de tous les modèles entraînés disponibles dans
le système Mercurio AI, avec des informations sur leur état et leurs métriques.

Exemple d'utilisation:
    python scripts/list_trained_models.py
"""

import os
import sys
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
from tabulate import tabulate

# Assurez-vous que le script peut importer les modules MercurioAI
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def get_lstm_models():
    """
    Récupère les informations sur tous les modèles LSTM entraînés
    
    Returns:
        DataFrame contenant les informations des modèles
    """
    lstm_dir = Path("models/lstm")
    if not lstm_dir.exists():
        return pd.DataFrame()
        
    models_info = []
    
    for model_dir in lstm_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        # Rechercher les fichiers de modèle dans ce répertoire
        model_files = list(model_dir.glob("*.h5"))
        scaler_files = list(model_dir.glob("*_scaler.pkl"))
        
        if not model_files:
            continue
            
        # Obtenir les informations de base
        model_path = model_files[0]
        symbol = model_dir.name.upper().replace('_', '-') if '_' in model_dir.name else model_dir.name.upper()
        creation_time = datetime.fromtimestamp(model_path.stat().st_mtime)
        
        # Obtenir les paramètres du modèle si possible
        params = {}
        if scaler_files:
            try:
                with open(scaler_files[0], 'rb') as f:
                    params = pickle.load(f)
            except:
                pass
            
        # Ajouter aux informations des modèles
        models_info.append({
            'Symbol': symbol,
            'Type': 'LSTM',
            'Path': str(model_path),
            'Created': creation_time,
            'Age (days)': (datetime.now() - creation_time).days,
            'Size (MB)': round(model_path.stat().st_size / (1024 * 1024), 2),
            'Sequence Length': params.get('sequence_length', 'N/A') if isinstance(params, dict) else 'N/A',
            'Prediction Horizon': params.get('prediction_horizon', 'N/A') if isinstance(params, dict) else 'N/A'
        })
    
    return pd.DataFrame(models_info)

def get_transformer_models():
    """
    Récupère les informations sur tous les modèles Transformer entraînés
    
    Returns:
        DataFrame contenant les informations des modèles
    """
    transformer_dir = Path("models/transformer")
    if not transformer_dir.exists():
        return pd.DataFrame()
        
    models_info = []
    
    for model_dir in transformer_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        # Rechercher les fichiers de modèle dans ce répertoire
        model_files = list(model_dir.glob("*.pth"))
        metadata_files = list(model_dir.glob("*_metadata.json"))
        
        if not model_files:
            continue
            
        # Obtenir les informations de base
        model_path = model_files[0]
        model_name = model_dir.name
        creation_time = datetime.fromtimestamp(model_path.stat().st_mtime)
        
        # Obtenir les paramètres du modèle si possible
        params = {}
        if metadata_files:
            try:
                with open(metadata_files[0], 'r') as f:
                    params = json.load(f)
            except:
                pass
            
        # Ajouter aux informations des modèles
        models_info.append({
            'Name': model_name,
            'Type': 'Transformer',
            'Path': str(model_path),
            'Created': creation_time,
            'Age (days)': (datetime.now() - creation_time).days,
            'Size (MB)': round(model_path.stat().st_size / (1024 * 1024), 2),
            'Sequence Length': params.get('sequence_length', 'N/A') if isinstance(params, dict) else 'N/A',
            'd_model': params.get('d_model', 'N/A') if isinstance(params, dict) else 'N/A',
            'Symbols': len(params.get('feature_columns', [])) if isinstance(params, dict) else 'N/A'
        })
    
    return pd.DataFrame(models_info)

def main():
    """Fonction principale"""
    logger.info("Récupération des informations sur les modèles entraînés...")
    
    # Obtenir les informations sur les modèles
    lstm_models = get_lstm_models()
    transformer_models = get_transformer_models()
    
    # Afficher les résultats
    print("\n=== Modèles LSTM ===\n")
    if not lstm_models.empty:
        print(tabulate(lstm_models, headers='keys', tablefmt='pretty', showindex=False))
        print(f"\nTotal: {len(lstm_models)} modèles LSTM\n")
    else:
        print("Aucun modèle LSTM trouvé.\n")
    
    print("\n=== Modèles Transformer ===\n")
    if not transformer_models.empty:
        print(tabulate(transformer_models, headers='keys', tablefmt='pretty', showindex=False))
        print(f"\nTotal: {len(transformer_models)} modèles Transformer\n")
    else:
        print("Aucun modèle Transformer trouvé.\n")
    
    # Afficher les instructions pour l'entraînement
    if lstm_models.empty and transformer_models.empty:
        print("\nAucun modèle entraîné trouvé. Vous pouvez entraîner des modèles avec les commandes suivantes :")
        print("\n1. Pour un seul modèle LSTM :")
        print("   python scripts/train_lstm_model.py --symbol BTC-USD --lookback 180 --epochs 100")
        print("\n2. Pour un seul modèle Transformer :")
        print("   python scripts/train_transformer_model.py --symbols BTC-USD,ETH-USD,AAPL --epochs 100")
        print("\n3. Pour entraîner tous les modèles en une seule commande :")
        print("   python scripts/train_all_models.py --days 90 --top_assets 20")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("Programme interrompu par l'utilisateur")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Erreur critique non gérée: {e}")
        sys.exit(1)
