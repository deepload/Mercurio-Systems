"""
MercurioAI Data Enricher

Module pour enrichir les données brutes avec des indicateurs techniques
nécessaires aux différentes stratégies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

def enrich_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Enrichit un DataFrame OHLCV avec tous les indicateurs techniques
    requis par les différentes stratégies du système.
    
    Args:
        data: DataFrame avec au minimum les colonnes open, high, low, close, volume
        
    Returns:
        DataFrame enrichi avec tous les indicateurs techniques nécessaires
    """
    if data is None or data.empty or len(data) < 30:
        return data
    
    # Faire une copie pour éviter de modifier l'original
    df = data.copy()
    
    # S'assurer que les colonnes nécessaires existent
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Colonnes manquantes dans les données: {missing_columns}")
    
    # Calculer les rendements
    df['return'] = df['close'].pct_change()
    
    # Moyennes mobiles
    df['ma_5'] = df['close'].rolling(window=5).mean()
    df['ma_20'] = df['close'].rolling(window=20).mean()
    df['ma_50'] = df['close'].rolling(window=50).mean()
    df['ma_200'] = df['close'].rolling(window=200).mean()
    
    # Pour compatibilité MovingAverageStrategy
    df['short_ma'] = df['ma_5']
    df['long_ma'] = df['ma_50']
    
    # RSI - Relative Strength Index
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD - Moving Average Convergence Divergence
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bandes de Bollinger
    sma = df['close'].rolling(window=20).mean()
    std = df['close'].rolling(window=20).std()
    df['bb_upper'] = sma + (std * 2)
    df['bb_middle'] = sma
    df['bb_lower'] = sma - (std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # Décalages pour les modèles ML
    df['close_lag_1'] = df['close'].shift(1)
    df['close_lag_2'] = df['close'].shift(2)
    df['return_lag_1'] = df['return'].shift(1)
    
    # Momentum
    df['momentum'] = df['close'] / df['close'].shift(10) - 1
    
    # Volatilité
    df['volatility'] = df['return'].rolling(window=20).std() * np.sqrt(252)
    
    # Volume relatif
    df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
    df['relative_volume'] = df['volume'] / df['volume_sma_20']
    
    # Pour les stratégies LSTM et Transformer
    # Séquence temporelle de n jours
    for i in range(1, 6):
        df[f'close_shift_{i}'] = df['close'].shift(i)
        df[f'return_shift_{i}'] = df['return'].shift(i)
    
    # Nettoyer les valeurs NaN
    # Note: nous les remplaçons par des zéros pour éviter les erreurs, mais 
    # il est recommandé de filtrer les premières lignes dans les stratégies
    df = df.fillna(0)
    
    return df

def prepare_data_for_strategy(data: pd.DataFrame, strategy_name: str) -> pd.DataFrame:
    """
    Prépare les données spécifiquement pour une stratégie particulière
    
    Args:
        data: DataFrame avec données OHLCV
        strategy_name: Nom de la stratégie
        
    Returns:
        DataFrame préparé pour la stratégie spécifique
    """
    enriched_data = enrich_data(data)
    
    if "MovingAverageStrategy" in strategy_name:
        # Vérifier que toutes les colonnes requises sont présentes
        required_cols = ['short_ma', 'long_ma', 'rsi', 'macd', 'macd_signal', 
                        'bb_width', 'close_lag_1', 'close_lag_2', 'return_lag_1']
        
        for col in required_cols:
            if col not in enriched_data.columns:
                raise ValueError(f"Colonne {col} manquante pour MovingAverageStrategy")
                
    elif "LSTMPredictorStrategy" in strategy_name:
        required_cols = ['return', 'ma_5', 'ma_20', 'rsi', 'macd', 'macd_signal', 
                         'bb_width', 'momentum', 'volatility']
        
        for col in required_cols:
            if col not in enriched_data.columns:
                raise ValueError(f"Colonne {col} manquante pour LSTMPredictorStrategy")
    
    elif "TransformerStrategy" in strategy_name:
        # Vérification similaire pour Transformer
        pass
    
    elif "MultiSourceIntelligenceStrategy" in strategy_name:
        # La MSI a ses propres mécanismes de préparation des données
        pass
    
    return enriched_data

def create_synthetic_data(symbol: str, days: int = 100) -> pd.DataFrame:
    """
    Crée des données synthétiques améliorées pour le test des stratégies
    avec tous les indicateurs techniques requis.
    
    Args:
        symbol: Symbole pour lequel créer les données
        days: Nombre de jours à générer
        
    Returns:
        DataFrame avec données OHLCV synthétiques enrichies
    """
    np.random.seed(42)  # Pour reproductibilité
    
    dates = pd.date_range(end=pd.Timestamp.now(), periods=days)
    
    if "BTC" in symbol:
        # Paramètres pour BTC
        starting_price = 50000
        daily_volatility = 0.03
    elif "ETH" in symbol:
        # Paramètres pour ETH
        starting_price = 3000
        daily_volatility = 0.04
    else:
        # Paramètres par défaut
        starting_price = 100
        daily_volatility = 0.02
    
    # Générer une marche aléatoire pour le prix
    random_walk = np.random.normal(0, daily_volatility, size=days).cumsum()
    prices = starting_price * (1 + random_walk)
    
    # Créer une tendance
    trend = np.linspace(0, 0.2, days)
    prices = prices * (1 + trend)
    
    # Ajouter de la saisonnalité
    seasonality = 0.05 * np.sin(np.linspace(0, 15, days))
    prices = prices * (1 + seasonality)
    
    # Générer OHLCV
    close = prices
    high = close * (1 + np.random.uniform(0, 0.02, days))
    low = close * (1 - np.random.uniform(0, 0.02, days))
    open_price = low + np.random.uniform(0, 1, days) * (high - low)
    
    # Volume avec corrélation au mouvement de prix
    price_change = np.diff(close, prepend=close[0])
    volume_base = np.random.uniform(0.5, 1.5, days) * starting_price * 100
    volume = volume_base * (1 + 2 * np.abs(price_change) / daily_volatility)
    
    # Créer le DataFrame
    df = pd.DataFrame({
        'date': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    df.set_index('date', inplace=True)
    
    # Enrichir avec tous les indicateurs
    return enrich_data(df)
