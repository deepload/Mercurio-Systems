#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
API Usage Monitor
----------------
Module pour surveiller l'utilisation des API et garder une trace des taux de requêtes
pour éviter de dépasser les limites et optimiser les performances.
"""

import time
import logging
import threading
from datetime import datetime, timedelta
from collections import deque
import os

# Configurer le logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('api_usage_monitor')

class APIUsageMonitor:
    """
    Classe pour surveiller l'utilisation des APIs et enregistrer les statistiques.
    Cette classe fonctionne comme un singleton pour assurer un suivi global.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(APIUsageMonitor, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self.api_calls = {}
        self.start_time = datetime.now()
        
        # Définir les limites d'API
        self.rate_limits = {
            'alpaca': {
                'minute': 10000,  # 10,000 requêtes par minute
                'second': 200     # 200 requêtes par seconde
            },
            'polygon': {
                'minute': 5,      # Exemple de limite pour Polygon
                'day': 5000       # Exemple de limite pour Polygon
            },
            # Ajouter d'autres APIs au besoin
        }
        
        # Initialiser les compteurs pour chaque API
        for api_name in self.rate_limits.keys():
            self.api_calls[api_name] = {
                'total': 0,
                'minute_calls': deque(maxlen=60),  # 60 secondes
                'second_calls': deque(maxlen=60),  # 60 entrées pour 60 secondes
                'day_calls': 0,
                'last_reset': datetime.now()
            }
            
            # Initialiser avec des zéros
            for i in range(60):
                self.api_calls[api_name]['minute_calls'].append(0)
                self.api_calls[api_name]['second_calls'].append(0)
        
        # Démarrer un thread pour réinitialiser les compteurs quotidiens
        self.reset_thread = threading.Thread(target=self._daily_reset, daemon=True)
        self.reset_thread.start()
        
        # Démarrer un thread pour la journalisation périodique
        if os.environ.get('API_USAGE_LOGGING', 'true').lower() == 'true':
            self.log_thread = threading.Thread(target=self._periodic_logging, daemon=True)
            self.log_thread.start()
            
        logger.info("Moniteur d'utilisation d'API initialisé")
    
    def record_api_call(self, api_name, endpoint=None, params=None):
        """
        Enregistre un appel d'API et met à jour les compteurs.
        
        Args:
            api_name: Nom de l'API (ex: 'alpaca', 'polygon')
            endpoint: Endpoint appelé (optionnel)
            params: Paramètres de la requête (optionnel)
        """
        if api_name not in self.api_calls:
            self.api_calls[api_name] = {
                'total': 0,
                'minute_calls': deque([0] * 60, maxlen=60),
                'second_calls': deque([0] * 60, maxlen=60),
                'day_calls': 0,
                'last_reset': datetime.now()
            }
        
        # Incrémenter le total
        self.api_calls[api_name]['total'] += 1
        
        # Incrémenter le compteur quotidien
        self.api_calls[api_name]['day_calls'] += 1
        
        # Mettre à jour les compteurs par seconde et par minute
        now = datetime.now()
        current_second = now.second
        
        # Mettre à jour le compteur par seconde pour la seconde actuelle
        second_calls = list(self.api_calls[api_name]['second_calls'])
        second_calls[current_second] += 1
        self.api_calls[api_name]['second_calls'] = deque(second_calls, maxlen=60)
        
        # Mettre à jour le compteur par minute
        current_minute = now.minute % 60
        minute_calls = list(self.api_calls[api_name]['minute_calls'])
        minute_calls[current_minute] += 1
        self.api_calls[api_name]['minute_calls'] = deque(minute_calls, maxlen=60)
        
        # Vérifier si on s'approche des limites
        self._check_limits(api_name)
    
    def _check_limits(self, api_name):
        """
        Vérifie si on s'approche des limites de taux et journalise un avertissement le cas échéant.
        """
        if api_name not in self.rate_limits:
            return
        
        limits = self.rate_limits[api_name]
        
        # Vérifier la limite par seconde
        if 'second' in limits:
            current_second = datetime.now().second
            second_count = self.api_calls[api_name]['second_calls'][current_second]
            limit_second = limits['second']
            
            # Avertir si on atteint 80% de la limite
            if second_count > limit_second * 0.8:
                logger.warning(f"Alerte: {api_name} approche de la limite par seconde! ({second_count}/{limit_second})")
        
        # Vérifier la limite par minute
        if 'minute' in limits:
            minute_count = sum(self.api_calls[api_name]['minute_calls'])
            limit_minute = limits['minute']
            
            # Avertir si on atteint 80% de la limite
            if minute_count > limit_minute * 0.8:
                logger.warning(f"Alerte: {api_name} approche de la limite par minute! ({minute_count}/{limit_minute})")
        
        # Vérifier la limite quotidienne
        if 'day' in limits:
            day_count = self.api_calls[api_name]['day_calls']
            limit_day = limits['day']
            
            # Avertir si on atteint 80% de la limite
            if day_count > limit_day * 0.8:
                logger.warning(f"Alerte: {api_name} approche de la limite quotidienne! ({day_count}/{limit_day})")
    
    def _daily_reset(self):
        """
        Réinitialiser les compteurs quotidiens à minuit.
        """
        while True:
            now = datetime.now()
            # Calculer l'heure du prochain minuit
            tomorrow = (now + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            # Calcul du temps d'attente jusqu'au prochain minuit
            seconds_until_midnight = (tomorrow - now).total_seconds()
            
            # Attendre jusqu'à minuit
            time.sleep(seconds_until_midnight)
            
            # Réinitialiser tous les compteurs quotidiens
            logger.info("Réinitialisation des compteurs quotidiens d'API")
            for api_name in self.api_calls:
                self.api_calls[api_name]['day_calls'] = 0
                self.api_calls[api_name]['last_reset'] = datetime.now()
    
    def _periodic_logging(self):
        """
        Journaliser périodiquement l'utilisation des API pour surveillance
        """
        while True:
            # Attendre 15 minutes
            time.sleep(15 * 60)
            
            # Journaliser les statistiques d'utilisation
            now = datetime.now()
            runtime = now - self.start_time
            
            log_message = f"\n=== Rapport d'utilisation d'API ({runtime.total_seconds()/3600:.1f} heures de fonctionnement) ===\n"
            
            for api_name, data in self.api_calls.items():
                total_calls = data['total']
                last_minute_calls = sum(data['minute_calls'])
                last_second_max = max(data['second_calls'])
                day_calls = data['day_calls']
                
                log_message += f"{api_name}:\n"
                log_message += f"  - Total des appels: {total_calls}\n"
                log_message += f"  - Appels dernière minute: {last_minute_calls}\n"
                log_message += f"  - Max appels par seconde: {last_second_max}\n"
                log_message += f"  - Appels aujourd'hui: {day_calls}\n"
                
                if api_name in self.rate_limits:
                    limits = self.rate_limits[api_name]
                    for limit_type, limit_value in limits.items():
                        if limit_type == 'second':
                            usage_pct = (last_second_max / limit_value) * 100
                            log_message += f"  - Utilisation max par seconde: {usage_pct:.1f}% ({last_second_max}/{limit_value})\n"
                        elif limit_type == 'minute':
                            usage_pct = (last_minute_calls / limit_value) * 100
                            log_message += f"  - Utilisation par minute: {usage_pct:.1f}% ({last_minute_calls}/{limit_value})\n"
                        elif limit_type == 'day':
                            usage_pct = (day_calls / limit_value) * 100
                            log_message += f"  - Utilisation quotidienne: {usage_pct:.1f}% ({day_calls}/{limit_value})\n"
            
            logger.info(log_message)
    
    def get_usage_statistics(self):
        """
        Renvoie les statistiques d'utilisation actuelles pour toutes les APIs.
        
        Returns:
            dict: Statistiques d'utilisation d'API
        """
        stats = {}
        for api_name, data in self.api_calls.items():
            stats[api_name] = {
                'total_calls': data['total'],
                'minute_calls': sum(data['minute_calls']),
                'max_calls_per_second': max(data['second_calls']),
                'day_calls': data['day_calls'],
                'running_since': self.start_time.isoformat()
            }
            
            # Ajouter les pourcentages d'utilisation par rapport aux limites
            if api_name in self.rate_limits:
                limits = self.rate_limits[api_name]
                usage = {}
                
                if 'second' in limits:
                    usage['second_percent'] = (max(data['second_calls']) / limits['second']) * 100
                
                if 'minute' in limits:
                    usage['minute_percent'] = (sum(data['minute_calls']) / limits['minute']) * 100
                
                if 'day' in limits:
                    usage['day_percent'] = (data['day_calls'] / limits['day']) * 100
                
                stats[api_name]['usage_percent'] = usage
        
        return stats

# Fonction de décorateur pour surveiller l'utilisation d'API
def monitor_api_usage(api_name):
    """
    Décorateur pour surveiller l'utilisation d'une API spécifique.
    
    Args:
        api_name: Nom de l'API à surveiller
    
    Returns:
        fonction décorateur
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            monitor = APIUsageMonitor()
            monitor.record_api_call(api_name)
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Exemple d'utilisation:
# @monitor_api_usage('alpaca')
# def get_market_data():
#     pass

if __name__ == "__main__":
    # Test du moniteur
    monitor = APIUsageMonitor()
    
    # Simuler quelques appels API
    for i in range(20):
        monitor.record_api_call('alpaca', endpoint='get_bars', params={'symbol': 'AAPL'})
        if i % 3 == 0:
            monitor.record_api_call('polygon', endpoint='get_ticker_details')
        time.sleep(0.1)
    
    # Afficher les statistiques
    stats = monitor.get_usage_statistics()
    print(stats)
