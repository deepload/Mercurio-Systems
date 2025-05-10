"""
Multi-Source Intelligence (MSI) Strategy

Une stratégie de trading professionnelle qui prend des décisions uniquement
lorsqu'elle dispose de données de marché fraîches et validées provenant de
multiples sources.

Cette stratégie combine:
1. Vérification rigoureuse de la fraîcheur des données
2. Analyse de sentiment multi-sources (Twitter, Reddit, actualités)
3. Détection de manipulations potentielles du marché
4. Système de mise en cache intelligent pour optimiser les appels API
5. Réévaluation continue des positions ouvertes

La stratégie n'exécute des transactions que lorsque:
- Les données sont fraîches et validées
- Il existe un consensus clair entre plusieurs sources
- Le niveau de confiance dépasse un seuil prédéfini
- Aucune manipulation potentielle du marché n'est détectée

Composants:
- DataVerificationSystem: Garantit la fraîcheur des données
- SentimentAnalysisEngine: Analyse les données de sentiment de multiples sources
- DecisionEngine: Prend des décisions commerciales basées sur toutes les données disponibles
"""
import logging
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Tuple, Optional

from app.strategies.base import BaseStrategy
from app.core.event_bus import EventBus, EventType
from app.services.market_data import MarketDataService
from app.db.models import TradeAction

# Import des composants spécifiques à MSI
from app.strategies.msi.data_verification import DataVerificationSystem 
from app.strategies.msi.sentiment_analysis import SentimentAnalysisEngine
from app.strategies.msi.decision_engine import DecisionEngine

logger = logging.getLogger(__name__)

class MultiSourceIntelligenceStrategy(BaseStrategy):
    """
    Stratégie de trading professionnelle qui prend des décisions uniquement
    lorsqu'elle dispose de données de marché fraîches et validées provenant de
    multiples sources.
    
    Cette stratégie représente l'application professionnelle des principes de trading
    algorithmique modernes en combinant plusieurs sources d'information et mécanismes
    de sécurité pour une prise de décision robuste.
    
    Caractéristiques principales:
    - Utilisation de données de multiples sources pour une vision complète du marché
    - Vérification rigoureuse de la fraîcheur et validité des données
    - Détection intelligente des manipulations potentielles du marché
    - Système de contrôle de confiance avec seuils ajustables
    - Réévaluation continue des positions ouvertes
    - Système de mise en cache intelligent pour optimiser les performances
    
    Cette stratégie est particulièrement adaptée aux marchés volatils comme les
    cryptomonnaies, où la qualité et la fraîcheur des données sont essentielles.
    """
    
    def __init__(self, 
                symbol: str = "BTC/USDT",
                max_data_age_seconds: int = 30,
                sentiment_lookback_minutes: int = 30,
                check_interval_seconds: int = 5,
                price_data_min_points: int = 100,
                confidence_threshold: float = 0.75,
                sentiment_weight: float = 0.4,
                technical_weight: float = 0.4,
                volume_weight: float = 0.2,
                cache_ttl_seconds: int = 60,
                debounce_interval_seconds: int = 15,
                conflicting_sources_threshold: float = 0.3,
                **kwargs):
        """
        Initialise la stratégie MSI avec les paramètres spécifiés
        
        Args:
            symbol: Symbole de trading par défaut
            max_data_age_seconds: Âge maximum des données en secondes
            sentiment_lookback_minutes: Période de rétroaction pour l'analyse de sentiment
            check_interval_seconds: Intervalle entre les vérifications d'état du marché
            price_data_min_points: Nombre minimum de points de données prix requis
            confidence_threshold: Seuil de confiance pour exécuter une transaction
            sentiment_weight: Poids des données de sentiment dans la décision
            technical_weight: Poids des indicateurs techniques dans la décision
            volume_weight: Poids des métriques de volume dans la décision
            cache_ttl_seconds: Durée de vie du cache en secondes
            debounce_interval_seconds: Intervalle de ralentissement entre décisions
            conflicting_sources_threshold: Seuil pour détecter les conflits de signaux
        """
        super().__init__(**kwargs)
        
        # Initialize core attributes
        self.symbol = symbol
        self.event_bus = EventBus()
        self.market_data_service = MarketDataService()
        
        # Initialize specialized components
        self.data_verifier = DataVerificationSystem(
            max_data_age_seconds=max_data_age_seconds,
            price_data_min_points=price_data_min_points
        )
        
        self.sentiment_engine = SentimentAnalysisEngine(
            sentiment_lookback_minutes=sentiment_lookback_minutes,
            cache_ttl_seconds=cache_ttl_seconds
        )
        
        self.decision_engine = DecisionEngine(
            confidence_threshold=confidence_threshold,
            sentiment_weight=sentiment_weight,
            technical_weight=technical_weight,
            volume_weight=volume_weight,
            conflicting_sources_threshold=conflicting_sources_threshold
        )
        
        # Settings
        self.check_interval_seconds = check_interval_seconds
        self.cache_ttl_seconds = cache_ttl_seconds
        self.debounce_interval_seconds = debounce_interval_seconds
        
        # State tracking
        self.last_decision_time = None
        self.last_check_time = None
        self.active_position = False
        self.current_position = 0
        self.entry_price = 0
        self.position_entry_time = None
        
        # Data cache
        self.data_cache = {}
        
        logger.info(f"Multi-Source Intelligence Strategy initialized for {symbol}")
    
    async def fetch_market_data(self) -> Dict[str, Any]:
        """
        Récupère les données de marché de toutes les sources disponibles
        et applique une mise en cache intelligente.
        
        Returns:
            Dictionnaire contenant toutes les données de marché
        """
        now = datetime.now(timezone.utc)
        
        # Initialiser le dictionnaire de données
        market_data = {}
        
        # 1. Récupérer les données de prix avec mise en cache
        try:
            end_date = now
            start_date = end_date - timedelta(hours=24)  # 24h de données
            price_data = await self.market_data_service.get_historical_data(
                self.symbol, start_date, end_date, timeframe="1m"
            )
            market_data['price'] = price_data
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données de prix: {e}")
            market_data['price'] = None
                
        # 2. Récupérer les données de sentiment
        sentiment_data = await self.sentiment_engine.fetch_sentiment_data(self.symbol)
        market_data.update(sentiment_data)
        
        # 3. Calculer les métriques de volume
        if market_data['price'] is not None and len(market_data['price']) > 0:
            recent_volumes = market_data['price']['volume'].tail(30)
            baseline_volume = market_data['price']['volume'].tail(120).mean()
            
            volume_ratio = (recent_volumes.iloc[-1] / baseline_volume 
                           if baseline_volume > 0 else 1.0)
            
            market_data['volume'] = {
                'current': recent_volumes.iloc[-1],
                'baseline': baseline_volume,
                'relative_strength': volume_ratio,
                'increasing': recent_volumes.iloc[-1] > recent_volumes.iloc[0],
                'timestamp': now
            }
        else:
            market_data['volume'] = None
            
        return market_data
    
    async def predict(self, data: pd.DataFrame) -> Tuple[TradeAction, float]:
        """
        Génère un signal de trading basé sur l'analyse des données de marché
        provenant de multiples sources.
        
        Args:
            data: Données de marché sous forme de DataFrame
            
        Returns:
            Tuple (action, confiance)
        """
        # Récupérer les données complètes du marché
        market_data = await self.fetch_market_data()
        
        # Mettre à jour les données de prix avec les données d'entrée
        market_data['price'] = data
        
        # Vérifier la fraîcheur des données
        if not await self.data_verifier.verify_data_freshness(market_data):
            logger.warning("Vérification de fraîcheur des données échouée - pas de trading")
            return TradeAction.HOLD, 0.0
        
        # Vérifier si nous sommes en période de debounce
        now = datetime.now(timezone.utc)
        if (self.last_decision_time and 
            (now - self.last_decision_time).total_seconds() < self.debounce_interval_seconds):
            logger.info(f"En période de debounce - signal ignoré")
            return TradeAction.HOLD, 0.0
        
        # Prendre une décision avec toutes les données disponibles
        action, confidence = await self.decision_engine.make_decision(market_data)
        
        # Appliquer la réévaluation de position si nécessaire
        if self.active_position:
            position_data = {
                'type': 'long' if self.current_position > 0 else 'short',
                'entry_price': self.entry_price,
                'entry_time': self.position_entry_time,
                'size': abs(self.current_position)
            }
            
            maintain_position = await self.reassess_position(position_data, market_data)
            
            if not maintain_position:
                # Recommandation de sortie qui remplace les signaux de maintien
                if self.current_position > 0:
                    action = TradeAction.SELL
                    confidence = 0.8  # Confiance élevée pour les sorties de gestion du risque
                elif self.current_position < 0:
                    action = TradeAction.BUY
                    confidence = 0.8
        
        # Enregistrer l'heure de décision
        if action != TradeAction.HOLD:
            self.last_decision_time = now
            
        return action, confidence
    
    async def reassess_position(self, position_data: Dict[str, Any], 
                               market_data: Dict[str, Any]) -> bool:
        """
        Réévalue continuellement si les positions existantes doivent être maintenues
        en fonction des conditions actuelles du marché.
        
        Args:
            position_data: Dictionnaire avec les détails de la position
            market_data: Données actuelles du marché
            
        Returns:
            Booléen indiquant si la position doit être maintenue
        """
        # Analyse croisée avec les données fraîches
        sentiment_data = {
            k: v for k, v in market_data.items() 
            if k in ['twitter', 'reddit', 'news'] and v is not None
        }
        
        analysis = await self.decision_engine.cross_analyze_data(
            market_data['price'], sentiment_data
        )
        
        # Vérifier les facteurs qui déclencheraient une sortie de position
        position_type = position_data.get('type')
        momentum_score = analysis.get('momentum_score', 0)
        
        # Facteurs qui déclencheraient la sortie de position
        exit_triggers = []
        
        # 1. Le sentiment s'est fortement inversé contre la position
        if ((position_type == 'long' and momentum_score < -0.3) or
            (position_type == 'short' and momentum_score > 0.3)):
            exit_triggers.append(f"Sentiment inversé: momentum_score={momentum_score:.2f}")
            
        # 2. Manipulation potentielle détectée
        if analysis.get('manipulation_probability', 0) > 0.5:
            exit_triggers.append(f"Manipulation potentielle: {analysis.get('manipulation_probability'):.2f}")
            
        # 3. Volatilité élevée sans mouvement directionnel
        if 'price' in market_data and market_data['price'] is not None:
            recent_prices = market_data['price']['close'].tail(10)
            volatility = recent_prices.std() / recent_prices.mean()
            if volatility > 0.03 and abs(momentum_score) < 0.1:
                exit_triggers.append(f"Volatilité élevée sans direction: {volatility:.4f}")
                
        # Si des déclencheurs de sortie sont actifs, signaler la fermeture de position
        if exit_triggers:
            logger.info(f"Recommandation de sortie de position en raison de: {'; '.join(exit_triggers)}")
            return False
            
        # Position toujours alignée avec les conditions du marché
        return True
    
    async def load_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Charger les données historiques pour la stratégie.
        
        Args:
            symbol: Symbole de trading
            start_date: Date de début
            end_date: Date de fin
            
        Returns:
            DataFrame avec les données historiques
        """
        try:
            # Utiliser le service de données de marché pour charger les données
            data = await self.market_data_service.get_historical_data(
                symbol, start_date, end_date
            )
            return data
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données pour {symbol}: {e}")
            return pd.DataFrame()
    
    async def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prétraite les données pour l'analyse.
        
        Args:
            data: Données brutes du marché
            
        Returns:
            Données prétraitées
        """
        if data is None or data.empty:
            return pd.DataFrame()
            
        # Copier pour éviter de modifier l'original
        processed_data = data.copy()
        
        # Calculer des indicateurs techniques de base
        if len(processed_data) > 20:
            # Moyennes mobiles
            processed_data['sma_20'] = processed_data['close'].rolling(window=20).mean()
            processed_data['sma_50'] = processed_data['close'].rolling(window=50).mean()
            
            # RSI (14)
            delta = processed_data['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            processed_data['rsi_14'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema_12 = processed_data['close'].ewm(span=12, adjust=False).mean()
            ema_26 = processed_data['close'].ewm(span=26, adjust=False).mean()
            processed_data['macd'] = ema_12 - ema_26
            processed_data['macd_signal'] = processed_data['macd'].ewm(span=9, adjust=False).mean()
            
        # Supprimer les valeurs NaN
        processed_data = processed_data.dropna()
        
        return processed_data
    
    async def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calibre la stratégie sur des données historiques.
        
        Args:
            data: Données historiques pour l'entraînement
            
        Returns:
            Dictionnaire avec les résultats de l'entraînement
        """
        logger.info(f"Calibration de la stratégie MSI avec données historiques")
        
        if data is None or data.empty:
            logger.warning("Impossible de calibrer la stratégie avec un dataset vide")
            return {"status": "error", "message": "Dataset d'entraînement vide"}
        
        try:
            # Prétraiter les données
            processed_data = await self.preprocess_data(data)
            
            # Calibrer les composants
            await self.sentiment_engine.initialize()
            await self.decision_engine.calibrate(processed_data)
            
            # Calculer la volatilité typique pour calibrer la sensibilité
            volatility = processed_data['close'].pct_change().std()
            
            self.is_trained = True
            
            return {
                "status": "success",
                "volatility_baseline": volatility,
                "message": f"Stratégie MSI calibrée avec succès (volatilité: {volatility:.4f})"
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la calibration de la stratégie: {e}")
            return {"status": "error", "message": str(e)}
    
    async def backtest(self, data: pd.DataFrame, initial_capital: float = 10000.0) -> Dict[str, Any]:
        """
        Effectue un backtest de la stratégie sur des données historiques.
        
        Args:
            data: Données historiques
            initial_capital: Capital initial
            
        Returns:
            Résultats du backtest
        """
        logger.info(f"Démarrage du backtest de la stratégie MSI avec capital initial de ${initial_capital:.2f}")
        
        if data is None or data.empty:
            logger.error("Données de backtest vides ou nulles")
            return {"error": "Données vides"}
            
        if not self.is_trained:
            logger.warning("La stratégie n'est pas calibrée, calcul automatique")
            await self.train(data)
            
        # Initialiser les variables de backtest
        capital = initial_capital
        position = 0
        trades = []
        equity_curve = []
        
        # Fenêtre de lookback pour les décisions
        lookback_window = 100
        
        # Parcourir les données en simulant le trading
        for i in range(lookback_window, len(data)):
            # Extraire la fenêtre de données actuelle
            current_window = data.iloc[i-lookback_window:i]
            current_date = current_window.index[-1]
            current_price = current_window['close'].iloc[-1]
            
            # Prendre une décision pour cette période
            action, confidence = await self.predict(current_window)
            
            # Mettre à jour la position et le capital
            if action == TradeAction.BUY and position <= 0:
                # Fermer toute position courte
                if position < 0:
                    capital += position * current_price * -1  # Fermer le short
                    trades.append({
                        'timestamp': current_date,
                        'action': 'CLOSE_SHORT',
                        'price': current_price,
                        'quantity': position * -1,
                        'capital': capital
                    })
                    position = 0
                
                # Ouvrir une position longue - investir 95% du capital
                quantity = (capital * 0.95) / current_price
                capital -= quantity * current_price
                position += quantity
                trades.append({
                    'timestamp': current_date,
                    'action': 'BUY',
                    'price': current_price,
                    'quantity': quantity,
                    'capital': capital,
                    'confidence': confidence
                })
                
            elif action == TradeAction.SELL and position >= 0:
                # Fermer toute position longue
                if position > 0:
                    capital += position * current_price
                    trades.append({
                        'timestamp': current_date,
                        'action': 'CLOSE_LONG',
                        'price': current_price,
                        'quantity': position,
                        'capital': capital
                    })
                    position = 0
                
                # Ouvrir une position courte - 95% du capital
                quantity = (capital * 0.95) / current_price
                capital += quantity * current_price  # Produit de la vente à découvert
                position -= quantity
                trades.append({
                    'timestamp': current_date,
                    'action': 'SELL',
                    'price': current_price,
                    'quantity': quantity,
                    'capital': capital,
                    'confidence': confidence
                })
            
            # Calculer l'équité (capital + valeur de la position)
            equity = capital + (position * current_price)
            
            equity_curve.append({
                'timestamp': current_date,
                'price': current_price,
                'action': action.name,
                'confidence': confidence,
                'position': position,
                'capital': capital,
                'equity': equity
            })
        
        # Calculer l'équité finale
        final_equity = capital
        if position != 0:
            final_price = data['close'].iloc[-1]
            final_equity += position * final_price
            
        # Calculer les métriques de performance
        total_return = (final_equity / initial_capital) - 1
        equity_df = pd.DataFrame(equity_curve)
        
        # Calculer le drawdown maximum
        if not equity_df.empty and 'equity' in equity_df:
            equity_series = equity_df['equity']
            peak = equity_series.expanding(min_periods=1).max()
            drawdown = (equity_series / peak) - 1
            max_drawdown = drawdown.min()
        else:
            max_drawdown = 0
            
        # Préparer les résultats
        results = {
            'initial_capital': initial_capital,
            'final_equity': final_equity,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'num_trades': len(trades),
            'equity_curve': equity_df,
            'trades': trades,
            'strategy': 'MultiSourceIntelligenceStrategy'
        }
        
        logger.info(f"Backtest terminé: rendement={total_return:.2%}, max_drawdown={max_drawdown:.2%}, trades={len(trades)}")
        
        return results
