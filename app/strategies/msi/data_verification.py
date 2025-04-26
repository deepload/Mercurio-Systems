"""
Système de vérification de la fraîcheur des données pour la stratégie MSI.

Ce module implémente un système de vérification qui s'assure que toutes les données
utilisées pour la prise de décision sont suffisamment récentes et complètes.

Fonctionnalités principales:
- Validation de l'âge des données de prix
- Vérification de la qualité des données de sentiment
- Suivi de l'état de santé de chaque source de données
- Système de blocage des décisions lorsque des données essentielles sont obsolètes

Le système définit des seuils de fraîcheur configurable et surveille l'état de
santé de toutes les sources de données utilisées par la stratégie MSI.
"""
import logging
from datetime import datetime, timezone
from typing import Dict, Any

logger = logging.getLogger(__name__)

class DataVerificationSystem:
    """
    Système de vérification qui s'assure que toutes les données
    sont fraîches avant de prendre une décision de trading.
    
    Ce système implémente un ensemble de contrôles de qualité des données
    pour éviter que la stratégie ne prenne des décisions basées sur des
    informations obsolètes ou incomplètes.
    
    Attributs:
        max_data_age_seconds (int): Durée maximale en secondes avant qu'une donnée
            ne soit considérée comme obsolète
        price_data_min_points (int): Nombre minimal de points de données requis
            pour les données de prix
        source_health_map (dict): Dictionnaire contenant l'état de santé de chaque
            source de données (True si fraîche, False sinon)
    """
    
    def __init__(self, max_data_age_seconds=30, price_data_min_points=100):
        self.max_data_age_seconds = max_data_age_seconds
        self.price_data_min_points = price_data_min_points
        self.source_health_map = {
            'price': False,
            'twitter': False,
            'reddit': False,
            'news': False,
            'volume': False
        }
    
    async def verify_data_freshness(self, data_dict: Dict[str, Any]) -> bool:
        """
        Vérification complète de la fraîcheur des données.
        
        Args:
            data_dict: Dictionnaire contenant les données de diverses sources
            
        Returns:
            Booléen indiquant si toutes les données passent les contrôles de fraîcheur
        """
        now = datetime.now(timezone.utc)
        all_sources_fresh = True
        
        # Initialiser la carte de santé pour ce cycle
        source_health = self.source_health_map.copy()
        
        # Vérifier les données de prix
        if 'price' not in data_dict or data_dict['price'] is None:
            logger.warning(f"Données de prix manquantes")
            source_health['price'] = False
            all_sources_fresh = False
        else:
            price_data = data_dict['price']
            
            # Vérifier que nous avons assez d'historique
            if len(price_data) < self.price_data_min_points:
                logger.warning(f"Historique de prix insuffisant: {len(price_data)} < {self.price_data_min_points}")
                source_health['price'] = False
                all_sources_fresh = False
            else:
                # Obtenir l'horodatage de la bougie la plus récente
                latest_timestamp = price_data.index[-1]
                age_seconds = (now - latest_timestamp).total_seconds()
                
                if age_seconds > self.max_data_age_seconds:
                    logger.warning(f"Données de prix obsolètes: {age_seconds}s (max: {self.max_data_age_seconds}s)")
                    source_health['price'] = False
                    all_sources_fresh = False
                else:
                    source_health['price'] = True
                    
        # Vérifier les sources de sentiment
        sentiment_sources = ['twitter', 'reddit', 'news']
        for source in sentiment_sources:
            if source not in data_dict or data_dict[source] is None:
                logger.info(f"Données de sentiment {source} manquantes")
                source_health[source] = False
                # Note: Nous n'échouons pas complètement sur les sentiments manquants
            else:
                sentiment_data = data_dict[source]
                if 'timestamp' not in sentiment_data:
                    logger.warning(f"Horodatage manquant pour les données de sentiment {source}")
                    source_health[source] = False
                else:
                    age_seconds = (now - sentiment_data['timestamp']).total_seconds()
                    if age_seconds > self.max_data_age_seconds * 2:  # Permet aux sentiments d'être légèrement plus anciens
                        logger.warning(f"Données de sentiment {source} obsolètes: {age_seconds}s")
                        source_health[source] = False
                    else:
                        source_health[source] = True
        
        # Vérifier les données de volume
        if 'volume' not in data_dict or data_dict['volume'] is None:
            logger.warning(f"Données de volume manquantes")
            source_health['volume'] = False
        else:
            volume_data = data_dict['volume']
            source_health['volume'] = True
        
        # Mettre à jour la carte de santé globale
        self.source_health_map = source_health
        
        # Sources minimales requises: prix et au moins une source de sentiment
        required_fresh = source_health['price'] and any([
            source_health['twitter'], source_health['reddit'], source_health['news']
        ])
        
        if not required_fresh:
            logger.warning("Sources de données critiques non fraîches - décision abandonnée")
            
        return required_fresh
