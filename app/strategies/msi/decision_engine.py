"""
Moteur de décision pour la stratégie MSI.

Ce module implémente un moteur de décision sophistiqué qui intègre des données
de multiples sources pour générer des signaux de trading fiables et robustes.

Fonctionnalités principales:
- Analyse pondérée des données techniques, de sentiment et de volume
- Détection des divergences entre les prix et le sentiment
- Identification des manipulations potentielles du marché
- Système de seuil de confiance dynamique
- Résolution des signaux contradictoires

Ce moteur représente le cœur de la stratégie MSI, prenant les décisions
finales de trading en se basant sur toutes les données disponibles.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from datetime import datetime, timezone

from app.db.models import TradeAction

logger = logging.getLogger(__name__)

class DecisionEngine:
    """
    Moteur de décision qui intègre des données de plusieurs sources
    pour générer des signaux de trading fiables.
    
    Ce moteur analyse et pondère intelligemment les signaux de diverses sources
    (analyse technique, sentiment, volume) pour déterminer les actions de trading
    optimales. Il vérifie également la présence de signaux contradictoires ou de
    manipulations potentielles du marché.
    
    Attributs:
        confidence_threshold (float): Seuil minimal de confiance requis pour trader
        sentiment_weight (float): Poids des données de sentiment dans la décision
        technical_weight (float): Poids des données techniques dans la décision
        volume_weight (float): Poids des données de volume dans la décision
        conflicting_sources_threshold (float): Seuil pour détecter les signaux contradictoires
        volatility_baseline (float): Référence de volatilité déterminée lors de la calibration
        dynamic_confidence_threshold (float): Seuil de confiance ajusté selon la volatilité
    """
    
    def __init__(self, 
                confidence_threshold=0.75,
                sentiment_weight=0.4,
                technical_weight=0.4,
                volume_weight=0.2,
                conflicting_sources_threshold=0.3):
        self.confidence_threshold = confidence_threshold
        self.sentiment_weight = sentiment_weight
        self.technical_weight = technical_weight
        self.volume_weight = volume_weight
        self.conflicting_sources_threshold = conflicting_sources_threshold
        
        # Pour calibration basée sur les données historiques
        self.volatility_baseline = None
        self.dynamic_confidence_threshold = None
    
    async def calibrate(self, historical_data: pd.DataFrame):
        """
        Calibrer le moteur de décision avec des données historiques
        
        Args:
            historical_data: DataFrame contenant les données historiques
            
        Returns:
            Booléen indiquant si la calibration a réussi
        """
        if historical_data is not None and len(historical_data) > 0:
            # Calculer la volatilité pour calibrer la sensibilité
            volatility = historical_data['close'].pct_change().std()
            self.volatility_baseline = volatility
            
            # Définir un seuil de confiance dynamique
            self.dynamic_confidence_threshold = min(0.85, max(0.6, 0.7 + volatility))
            
            logger.info(f"Moteur de décision calibré: volatilité={volatility:.4f}, seuil={self.dynamic_confidence_threshold:.2f}")
            return True
        return False
    
    async def cross_analyze_data(self, price_data: pd.DataFrame, 
                                sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyse croisée des données de prix et de sentiment.
        Détecte l'élan, la divergence et la manipulation potentielle du marché.
        
        Args:
            price_data: DataFrame des données OHLCV
            sentiment_data: Dictionnaire des métriques de sentiment de diverses sources
            
        Returns:
            Résultats d'analyse avec scores de confiance
        """
        analysis_results = {
            'momentum_score': 0.0,
            'divergence_detected': False,
            'manipulation_probability': 0.0,
            'overall_confidence': 0.0,
            'signals': {}
        }
        
        # Extraire le score de sentiment global (moyenne pondérée des sources disponibles)
        available_sources = [s for s in ['twitter', 'reddit', 'news'] 
                            if sentiment_data.get(s) is not None]
        
        if not available_sources:
            logger.warning("Aucune source de sentiment disponible pour l'analyse")
            return analysis_results
            
        sentiment_scores = [sentiment_data[s]['score'] for s in available_sources]
        overall_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        
        # Calculer l'élan du prix (variation sur les 5 dernières périodes)
        price_momentum = price_data['close'].pct_change(5).iloc[-1] if len(price_data) >= 5 else 0
        
        # Calculer l'élan du volume
        volume_momentum = price_data['volume'].pct_change(5).iloc[-1] if len(price_data) >= 5 else 0
        
        # Détecter la divergence sentiment-prix (manipulation potentielle)
        # Si le sentiment est très positif mais le prix baisse ou vice versa
        sentiment_price_aligned = (overall_sentiment > 0.3 and price_momentum > 0) or \
                                 (overall_sentiment < -0.3 and price_momentum < 0)
        
        sentiment_price_divergent = (overall_sentiment > 0.5 and price_momentum < -0.01) or \
                                   (overall_sentiment < -0.5 and price_momentum > 0.01)
        
        # Calcul de corrélation entre sentiment et prix
        # Cela nécessiterait des données historiques de sentiment
        sentiment_price_correlation = 0  # Placeholder
        
        # Détection de manipulation (forte divergence de sentiment + volume inhabituel)
        manipulation_probability = 0.0
        if sentiment_price_divergent and abs(volume_momentum) > 0.5:
            manipulation_probability = min(0.8, abs(overall_sentiment) * abs(volume_momentum))
            logger.warning(f"Manipulation potentielle détectée: {manipulation_probability:.2f}")
        
        # Déterminer le score d'élan (-1.0 to 1.0)
        momentum_score = price_momentum * 0.5 + overall_sentiment * 0.3 + volume_momentum * 0.2
        
        # Générer les signaux
        signals = {}
        
        # Signal d'achat fort: sentiment positif + élan de prix + augmentation du volume
        if overall_sentiment > 0.3 and price_momentum > 0.01 and volume_momentum > 0.1:
            signals['buy'] = min(0.95, overall_sentiment + price_momentum + volume_momentum / 3)
            
        # Signal de vente fort: sentiment négatif + baisse de prix + augmentation du volume
        elif overall_sentiment < -0.3 and price_momentum < -0.01 and volume_momentum > 0.1:
            signals['sell'] = min(0.95, abs(overall_sentiment) + abs(price_momentum) + volume_momentum / 3)
            
        # Signaux neutres ou contradictoires
        elif abs(overall_sentiment) < 0.2 or abs(price_momentum) < 0.005:
            signals['neutral'] = 1.0 - (abs(overall_sentiment) + abs(price_momentum))
            
        # Prudence en cas de divergence
        elif sentiment_price_divergent:
            signals['caution'] = manipulation_probability
            
        # Déterminer la confiance globale
        if manipulation_probability > 0.4:
            overall_confidence = 0.1  # Confiance très faible en cas de manipulation suspectée
        elif sentiment_price_aligned:
            overall_confidence = min(0.9, abs(momentum_score) * 0.7 + max(signals.values()) * 0.3)
        else:
            overall_confidence = min(0.5, abs(momentum_score) * 0.3)
            
        # Peupler les résultats
        analysis_results['momentum_score'] = momentum_score
        analysis_results['divergence_detected'] = sentiment_price_divergent
        analysis_results['manipulation_probability'] = manipulation_probability
        analysis_results['sentiment_price_correlation'] = sentiment_price_correlation
        analysis_results['overall_confidence'] = overall_confidence
        analysis_results['signals'] = signals
        
        return analysis_results
    
    async def make_decision(self, market_data: Dict[str, Any]) -> Tuple[TradeAction, float]:
        """
        Moteur de décision complet qui pèse intelligemment plusieurs sources de données
        et applique des contrôles de sécurité avant d'autoriser une transaction.
        
        Args:
            market_data: Dictionnaire contenant toutes les données de marché pertinentes
            
        Returns:
            Tuple de (action, confiance)
        """
        # Extraire les composants de données
        price_data = market_data['price']
        sentiment_data = {
            k: v for k, v in market_data.items() 
            if k in ['twitter', 'reddit', 'news'] and v is not None
        }
        
        # Analyse croisée du prix et du sentiment
        analysis = await self.cross_analyze_data(price_data, sentiment_data)
        signals = analysis['signals']
        
        # Vérifier les signaux contradictoires - annuler si contradictoires
        signal_values = list(signals.values())
        signal_types = list(signals.keys())
        
        if len(signal_values) >= 2:
            top_two_values = sorted(signal_values, reverse=True)[:2]
            if top_two_values[0] - top_two_values[1] < self.conflicting_sources_threshold:
                if ('buy' in signals and 'sell' in signals) or \
                   ('buy' in signals and 'caution' in signals) or \
                   ('sell' in signals and 'caution' in signals):
                    logger.warning(f"Signaux contradictoires détectés: {signals} - éviter l'action")
                    return TradeAction.HOLD, analysis['overall_confidence']
        
        # Appliquer le seuil de confiance
        if analysis['overall_confidence'] < self.confidence_threshold:
            logger.info(f"Confiance trop faible: {analysis['overall_confidence']:.2f} < {self.confidence_threshold}")
            return TradeAction.HOLD, analysis['overall_confidence']
            
        # Vérifier la manipulation potentielle
        if analysis['manipulation_probability'] > 0.4:
            logger.warning(f"Manipulation potentielle détectée ({analysis['manipulation_probability']:.2f}) - éviter l'action")
            return TradeAction.HOLD, analysis['overall_confidence']
            
        # Déterminer l'action finale basée sur le score d'élan
        momentum = analysis['momentum_score']
        
        if momentum > 0.2 and 'buy' in signals and signals['buy'] > 0.6:
            action = TradeAction.BUY
            confidence = signals['buy'] * analysis['overall_confidence']
        elif momentum < -0.2 and 'sell' in signals and signals['sell'] > 0.6:
            action = TradeAction.SELL
            confidence = signals['sell'] * analysis['overall_confidence']
        else:
            action = TradeAction.HOLD
            confidence = analysis['overall_confidence']
            
        # Journaliser la décision avec justification
        logger.info(f"Décision: {action.name} avec {confidence:.2f} de confiance")
        logger.info(f"Justification: élan={momentum:.2f}, signaux={signals}, confiance_globale={analysis['overall_confidence']:.2f}")
        
        return action, confidence
