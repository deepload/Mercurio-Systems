"""
Moving Average ML Strategy

Une stratégie avancée basée sur les moyennes mobiles qui utilise 
le machine learning pour optimiser les paramètres et améliorer
les signaux de trading.
"""
import os
import pickle
import logging
from typing import Dict, Any, Tuple, Union, Optional
from datetime import datetime
import pandas as pd
import numpy as np
import backtrader as bt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error

from app.db.models import TradeAction
from app.strategies.base import BaseStrategy
from app.services.market_data import MarketDataService
from app.services.trading import TradingService

logger = logging.getLogger(__name__)

class MovingAverageMLStrategy(BaseStrategy):
    """
    Stratégie de Moyenne Mobile avec ML
    
    Utilise l'apprentissage automatique pour optimiser dynamiquement:
    - Les périodes des moyennes mobiles
    - Les seuils de signal pour réduire les faux signaux
    - L'adaptation aux conditions de marché changeantes
    
    Caractéristiques:
    - Ajustement automatique des fenêtres en fonction de la volatilité
    - Modèle de forêt aléatoire pour prédire la direction du prix
    - Analyse de tendance multi-temporelle
    """
    
    def __init__(
        self,
        market_data_service: Optional[MarketDataService] = None,
        trading_service: Optional[TradingService] = None,
        short_window_min: int = 5,
        short_window_max: int = 30,
        long_window_min: int = 30,
        long_window_max: int = 100,
        optimize_interval: int = 20,
        symbol: str = "SPY",
        **kwargs
    ):
        """
        Initialiser la stratégie ML de Moyenne Mobile.
        
        Args:
            market_data_service: Service de données de marché
            trading_service: Service d'exécution des trades
            short_window_min: Période minimale pour la MA courte
            short_window_max: Période maximale pour la MA courte
            long_window_min: Période minimale pour la MA longue
            long_window_max: Période maximale pour la MA longue
            optimize_interval: Nombre de barres avant réoptimisation
            symbol: Symbole par défaut
            **kwargs: Paramètres additionnels
        """
        super().__init__(market_data_service=market_data_service, 
                         trading_service=trading_service,
                         symbol=symbol, **kwargs)
        
        self.short_window_min = short_window_min
        self.short_window_max = short_window_max
        self.long_window_min = long_window_min
        self.long_window_max = long_window_max
        self.short_window = (short_window_min + short_window_max) // 2  # valeur initiale
        self.long_window = (long_window_min + long_window_max) // 2  # valeur initiale
        self.optimize_interval = optimize_interval
        self.bar_count = 0
        self.ml_model = None
        self.scaler = StandardScaler()
        self.last_signals = {}  # Historique des signaux par symbole
        self.model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'ma_ml_model.pkl')
        
        # Créer le dossier models s'il n'existe pas
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Charger un modèle existant si disponible
        self._load_model()
    
    def _load_model(self):
        """Charger un modèle ML préexistant s'il existe"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.ml_model = model_data['model']
                    self.short_window = model_data.get('short_window', self.short_window)
                    self.long_window = model_data.get('long_window', self.long_window)
                    self.scaler = model_data.get('scaler', self.scaler)
                logger.info(f"Modèle MovingAverageML chargé: {self.model_path}")
            else:
                logger.info("Aucun modèle MovingAverageML préexistant trouvé")
        except Exception as e:
            logger.warning(f"Erreur lors du chargement du modèle ML: {e}")
    
    def _save_model(self):
        """Sauvegarder le modèle ML actuel"""
        try:
            model_data = {
                'model': self.ml_model,
                'short_window': self.short_window,
                'long_window': self.long_window,
                'scaler': self.scaler,
                'timestamp': datetime.now().isoformat()
            }
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"Modèle MovingAverageML sauvegardé: {self.model_path}")
        except Exception as e:
            logger.warning(f"Erreur lors de la sauvegarde du modèle ML: {e}")
    
    def _prepare_features(self, data):
        """
        Préparer les caractéristiques pour le modèle ML
        
        Args:
            data: DataFrame avec les données OHLCV
            
        Returns:
            DataFrame avec les caractéristiques
        """
        df = data.copy()
        
        # Calculer les indicateurs techniques de base
        df['short_ma'] = df['close'].rolling(window=self.short_window).mean()
        df['long_ma'] = df['close'].rolling(window=self.long_window).mean()
        
        # Signal basique (1 pour achat, -1 pour vente, 0 sinon)
        df['signal'] = 0
        df.loc[df['short_ma'] > df['long_ma'], 'signal'] = 1
        df.loc[df['short_ma'] < df['long_ma'], 'signal'] = -1
        
        # Caractéristiques avancées
        df['ma_diff'] = df['short_ma'] - df['long_ma']
        df['ma_diff_pct'] = (df['short_ma'] / df['long_ma']) - 1
        
        # Volatilité
        df['volatility'] = df['close'].rolling(window=20).std()
        df['volatility_ratio'] = df['volatility'] / df['close']
        
        # Tendance
        df['trend_5d'] = df['close'].pct_change(5)
        df['trend_10d'] = df['close'].pct_change(10)
        df['trend_20d'] = df['close'].pct_change(20)
        
        # Volume
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        # Caractéristiques décalées
        for i in [1, 2, 3, 5]:
            df[f'return_lag_{i}'] = df['close'].pct_change(i)
        
        # Nettoyer les valeurs NaN
        df = df.dropna()
        
        return df
    
    def _optimize_parameters(self, data):
        """
        Optimiser les paramètres de la stratégie en utilisant ML
        
        Args:
            data: DataFrame avec les données OHLCV
            
        Returns:
            Tuple avec les paramètres optimisés (short_window, long_window)
        """
        if len(data) < 100:
            logger.warning("Données insuffisantes pour l'optimisation des paramètres")
            return self.short_window, self.long_window
        
        best_short = self.short_window
        best_long = self.long_window
        best_score = -float('inf')
        
        # Créer une cible: le rendement futur sur 5 jours
        data['future_return'] = data['close'].pct_change(5).shift(-5)
        data = data.dropna()
        
        # Tester différentes combinaisons de fenêtres
        for short in range(self.short_window_min, self.short_window_max + 1, 2):
            for long in range(self.long_window_min, self.long_window_max + 1, 5):
                if long <= short:
                    continue
                
                # Calculer les moyennes mobiles
                data['test_short_ma'] = data['close'].rolling(window=short).mean()
                data['test_long_ma'] = data['close'].rolling(window=long).mean()
                
                # Signaux de test
                data['test_signal'] = 0
                data.loc[data['test_short_ma'] > data['test_long_ma'], 'test_signal'] = 1
                data.loc[data['test_short_ma'] < data['test_long_ma'], 'test_signal'] = -1
                
                # Évaluer la performance
                valid_data = data.dropna()
                if len(valid_data) < 30:
                    continue
                
                # Calculer le score: corrélation entre signal et rendement futur
                score = valid_data['test_signal'].corr(valid_data['future_return'])
                
                if score > best_score:
                    best_score = score
                    best_short = short
                    best_long = long
        
        logger.info(f"Paramètres optimisés - Court: {best_short}, Long: {best_long}, Score: {best_score:.4f}")
        return best_short, best_long
    
    def train(self, symbol: str = None) -> bool:
        """
        Entraîner le modèle ML sur les données historiques
        
        Args:
            symbol: Symbole pour lequel entraîner le modèle
            
        Returns:
            bool: True si l'entraînement a réussi, False sinon
        """
        try:
            symbol = symbol or self.symbol
            logger.info(f"Entraînement du modèle MovingAverageML pour {symbol}...")
            
            # Récupérer les données historiques
            end = datetime.now()
            start = end - pd.Timedelta(days=365)  # 1 an de données
            
            if self.market_data_service:
                data = self.market_data_service.get_historical_data(
                    symbol=symbol,
                    interval='1day',
                    start=start,
                    end=end
                )
            else:
                logger.warning("Service de données de marché non disponible pour l'entraînement")
                return False
            
            if len(data) < 100:
                logger.warning(f"Données insuffisantes pour {symbol} - Minimum 100 barres nécessaires")
                return False
            
            # Optimiser les fenêtres
            self.short_window, self.long_window = self._optimize_parameters(data)
            
            # Préparer les données pour le ML
            features_df = self._prepare_features(data)
            
            # Définir X et y pour l'entraînement
            X = features_df[[
                'ma_diff', 'ma_diff_pct', 'volatility_ratio',
                'trend_5d', 'trend_10d', 'trend_20d', 'volume_ratio',
                'return_lag_1', 'return_lag_2', 'return_lag_3', 'return_lag_5'
            ]].values
            
            # Rendement futur à 5 jours comme cible
            y = (features_df['close'].shift(-5) > features_df['close']).astype(int).values[:-5]
            X = X[:-5]  # Aligner avec y
            
            if len(X) < 50:
                logger.warning("Données insuffisantes après préparation")
                return False
            
            # Normaliser les caractéristiques
            X_scaled = self.scaler.fit_transform(X)
            
            # Diviser les données
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Entraîner le modèle
            self.ml_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_split=10,
                random_state=42
            )
            self.ml_model.fit(X_train, y_train)
            
            # Évaluer le modèle
            y_pred = self.ml_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"Précision du modèle: {accuracy:.4f}")
            
            # Sauvegarder le modèle
            self._save_model()
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement du modèle ML: {e}")
            return False
    
    def get_signal(self, symbol: str, data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Générer un signal de trading en utilisant la stratégie ML
        
        Args:
            symbol: Symbole pour lequel générer un signal
            data: Données OHLCV optionnelles
            
        Returns:
            Dict avec le signal de trading
        """
        try:
            self.bar_count += 1
            
            # Réoptimiser périodiquement
            if self.bar_count % self.optimize_interval == 0:
                logger.info(f"Réoptimisation des paramètres pour {symbol}...")
                if data is not None and len(data) >= 100:
                    self.short_window, self.long_window = self._optimize_parameters(data)
            
            # Obtenir les données si non fournies
            if data is None:
                if not self.market_data_service:
                    return {"action": TradeAction.HOLD, "confidence": 0.5}
                
                end = datetime.now()
                start = end - pd.Timedelta(days=100)  # 100 jours de données
                
                data = self.market_data_service.get_historical_data(
                    symbol=symbol,
                    interval='1day',
                    start=start,
                    end=end
                )
            
            if len(data) < max(self.short_window, self.long_window) + 10:
                logger.warning(f"Données insuffisantes pour {symbol}")
                return {"action": TradeAction.HOLD, "confidence": 0.5}
            
            # Préparer les caractéristiques
            features_df = self._prepare_features(data)
            
            signal = TradeAction.HOLD
            confidence = 0.5
            
            # Signal de base sur le croisement des moyennes mobiles
            last_row = features_df.iloc[-1]
            prev_row = features_df.iloc[-2]
            
            # Croisement à la hausse (signal d'achat)
            if last_row['short_ma'] > last_row['long_ma'] and prev_row['short_ma'] <= prev_row['long_ma']:
                signal = TradeAction.BUY
                confidence = 0.7  # Confiance de base
            
            # Croisement à la baisse (signal de vente)
            elif last_row['short_ma'] < last_row['long_ma'] and prev_row['short_ma'] >= prev_row['long_ma']:
                signal = TradeAction.SELL
                confidence = 0.7  # Confiance de base
            
            # Utiliser ML pour affiner la confiance si modèle disponible
            if self.ml_model is not None:
                try:
                    last_features = last_row[[
                        'ma_diff', 'ma_diff_pct', 'volatility_ratio',
                        'trend_5d', 'trend_10d', 'trend_20d', 'volume_ratio',
                        'return_lag_1', 'return_lag_2', 'return_lag_3', 'return_lag_5'
                    ]].values.reshape(1, -1)
                    
                    # Normaliser
                    last_features_scaled = self.scaler.transform(last_features)
                    
                    # Prédire la probabilité de hausse
                    probas = self.ml_model.predict_proba(last_features_scaled)[0]
                    ml_confidence = probas[1]  # Probabilité de hausse
                    
                    # Ajuster le signal basé sur la prédiction ML
                    if signal == TradeAction.BUY and ml_confidence < 0.4:
                        signal = TradeAction.HOLD  # Le ML n'est pas confiant dans l'achat
                    elif signal == TradeAction.SELL and ml_confidence > 0.6:
                        signal = TradeAction.HOLD  # Le ML pense que le prix va monter
                    
                    # Ajuster la confiance
                    if signal == TradeAction.BUY:
                        confidence = (confidence + ml_confidence) / 2
                    elif signal == TradeAction.SELL:
                        confidence = (confidence + (1 - ml_confidence)) / 2
                    
                except Exception as e:
                    logger.warning(f"Erreur lors de l'utilisation du modèle ML: {e}")
            
            # Stocker le signal pour référence future
            self.last_signals[symbol] = {
                "action": signal,
                "confidence": confidence,
                "short_ma": last_row['short_ma'],
                "long_ma": last_row['long_ma'],
                "timestamp": datetime.now()
            }
            
            # Journal détaillé pour les signaux non-HOLD
            if signal != TradeAction.HOLD:
                logger.info(f"Signal ML pour {symbol}: {signal.name} avec confiance {confidence:.4f}")
                logger.info(f"MA Court ({self.short_window}): {last_row['short_ma']:.4f}, MA Long ({self.long_window}): {last_row['long_ma']:.4f}")
            
            return {
                "action": signal,
                "confidence": confidence,
                "params": {
                    "short_window": self.short_window,
                    "long_window": self.long_window,
                    "short_ma": last_row['short_ma'],
                    "long_ma": last_row['long_ma']
                }
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du signal ML pour {symbol}: {e}")
            return {"action": TradeAction.HOLD, "confidence": 0.5}
