"""
Patch pour le service de données du marché

Ce module contient des fonctions qui peuvent être appliquées pour corriger
les problèmes de préparation des données dans le service de données du marché.
"""

import logging
import pandas as pd
from typing import Dict, Any, Optional
from app.utils.data_enricher import enrich_data, create_synthetic_data

logger = logging.getLogger(__name__)

async def get_enhanced_market_data(market_data_service, symbol: str, lookback_days: int = 30) -> pd.DataFrame:
    """
    Version améliorée de la méthode get_historical_data qui garantit que les données
    contiennent tous les indicateurs techniques nécessaires.
    
    Args:
        market_data_service: Instance du service de données du marché
        symbol: Symbole pour lequel récupérer les données
        lookback_days: Nombre de jours d'historique à récupérer
        
    Returns:
        DataFrame enrichi avec tous les indicateurs techniques
    """
    try:
        # Accéder directement à la méthode originale pour éviter la récursion
        # Nous utilisons __dict__ pour accéder au dictionnaire d'attributs de l'objet
        # et récupérer la méthode originale sauvegardée lors du patching
        if hasattr(market_data_service, '_original_get_historical_data'):
            original_method = market_data_service._original_get_historical_data
            data = await original_method(symbol, lookback_days)
        else:
            # Fallback direct sur les données synthétiques si pas de méthode originale
            logger.warning(f"Pas de méthode originale disponible pour {symbol}, utilisation directe de données synthétiques")
            return create_synthetic_data(symbol, days=100)
        
        # Vérifier si les données sont valides
        if data is None or data.empty or len(data) < 20:
            logger.warning(f"Données insuffisantes pour {symbol}, utilisation de données synthétiques")
            data = create_synthetic_data(symbol, days=100)
        else:
            # Enrichir les données avec les indicateurs techniques
            data = enrich_data(data)
            
        return data
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des données pour {symbol}: {e}")
        logger.info(f"Fallback sur des données synthétiques pour {symbol}")
        return create_synthetic_data(symbol, days=100)

def patch_market_data_service(market_data_service):
    """
    Applique le patch au service de données du marché.
    
    Args:
        market_data_service: Instance du service de données du marché
    """
    # Vérifier si le patch a déjà été appliqué
    if hasattr(market_data_service, '_original_get_historical_data'):
        logger.info("Le patch a déjà été appliqué au service de données du marché")
        return
    
    # Sauvegarder la méthode originale comme attribut de l'objet
    market_data_service._original_get_historical_data = market_data_service.get_historical_data
    
    # Définir une nouvelle fonction qui appelle get_enhanced_market_data
    async def enhanced_get_historical_data(symbol, lookback_days=30):
        return await get_enhanced_market_data(market_data_service, symbol, lookback_days)
    
    # Remplacer la méthode originale par la nouvelle
    market_data_service.get_historical_data = enhanced_get_historical_data
    
    logger.info("Patch appliqué au service de données du marché")
