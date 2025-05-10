#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
API Rate Manager
---------------
Module pour gérer les limites de taux des API.
"""

import time
import logging
import threading
from datetime import datetime, timedelta
from functools import wraps
from collections import deque
import os

# Essayer d'importer le moniteur d'API
try:
    from api_usage_monitor import APIUsageMonitor, monitor_api_usage
    USE_API_MONITOR = True
except ImportError:
    USE_API_MONITOR = False
    # Fournir une implémentation factice si le moniteur n'est pas disponible
    class DummyAPIUsageMonitor:
        def record_api_call(self, *args, **kwargs):
            pass
        def get_usage_statistics(self):
            return {}

# Configurer le logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('api_rate_manager')

# Gestionnaire de taux singleton
class APIRateManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(APIRateManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, max_calls_per_minute=1000000, max_calls_per_second=20000, api_name='alpaca'):
        if self._initialized:
            return
        self._initialized = True
        
        # Charger les limites depuis les variables d'environnement si disponibles
        env_max_calls_per_min = os.environ.get('ALPACA_MAX_CALLS_PER_MINUTE')
        env_max_calls_per_sec = os.environ.get('ALPACA_MAX_CALLS_PER_SECOND')
        
        if env_max_calls_per_min:
            try:
                max_calls_per_minute = int(env_max_calls_per_min)
            except ValueError:
                pass
                
        if env_max_calls_per_sec:
            try:
                max_calls_per_second = int(env_max_calls_per_sec)
            except ValueError:
                pass
        
        self.max_calls_per_minute = max_calls_per_minute
        self.max_calls_per_second = max_calls_per_second
        self.api_name = api_name
        
        self.calls_per_minute = deque(maxlen=60)  # Track calls for each second in a minute
        self.last_call_time = datetime.now()
        
        # Initialize with zeros
        for _ in range(60):
            self.calls_per_minute.append(0)
        
        # Initialiser le moniteur d'API si disponible
        if USE_API_MONITOR:
            self.api_monitor = APIUsageMonitor()
        else:
            self.api_monitor = DummyAPIUsageMonitor()
        
        logger.info(f"API Rate Manager initialisé (limits: {max_calls_per_minute}/min, {max_calls_per_second}/sec)")
    
    def wait_if_needed(self, endpoint=None):
        """Wait if rate limits are close to being exceeded"""
        now = datetime.now()
        current_second = now.second
        
        # Update calls for current second
        minute_calls = list(self.calls_per_minute)
        minute_calls[current_second] += 1
        
        # Enregistrer l'appel dans le moniteur d'API
        self.api_monitor.record_api_call(self.api_name, endpoint=endpoint)
    
    def wait_if_needed_continued(self, endpoint=None):
        """Suite de la méthode wait_if_needed"""
        now = datetime.now()
        current_second = now.second
        minute_calls = list(self.calls_per_minute)
        
        # Check if we need to wait (approaching second limit)
        if minute_calls[current_second] >= self.max_calls_per_second * 0.9:
            time_to_wait = 1.0  # Wait for 1 second
            logger.warning(f"Approaching second rate limit ({minute_calls[current_second]}/{self.max_calls_per_second}), waiting {time_to_wait}s")
            time.sleep(time_to_wait)
            
            # After waiting, we're in a new second
            now = datetime.now()
            current_second = now.second
            minute_calls = list(self.calls_per_minute)
            minute_calls[current_second] += 1
        
        # Check if we need to wait (approaching minute limit)
        total_minute_calls = sum(minute_calls)
        if total_minute_calls >= self.max_calls_per_minute * 0.9:
            # Calculer le temps à attendre jusqu'à la prochaine minute
            next_minute = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
            time_to_wait = (next_minute - now).total_seconds()
            logger.warning(f"Approaching minute rate limit ({total_minute_calls}/{self.max_calls_per_minute}), waiting {time_to_wait:.1f}s")
            time.sleep(max(1.0, time_to_wait))
            
            # After waiting, we need to recalculate everything
            now = datetime.now()
            current_second = now.second
            # Reset if we've moved to a new minute
            if (now - self.last_call_time).total_seconds() >= 60:
                minute_calls = [0] * 60
            minute_calls[current_second] += 1
        
        # Update the call log
        self.calls_per_minute = deque(minute_calls, maxlen=60)
        self.last_call_time = now
    
    def get_usage_stats(self):
        """Récupère les statistiques d'utilisation actuelles"""
        current_usage = {
            "second_rate": max(self.calls_per_minute) if self.calls_per_minute else 0,
            "minute_rate": sum(self.calls_per_minute),
            "second_limit": self.max_calls_per_second,
            "minute_limit": self.max_calls_per_minute,
            "second_percent": (max(self.calls_per_minute) / self.max_calls_per_second * 100) if self.calls_per_minute else 0,
            "minute_percent": (sum(self.calls_per_minute) / self.max_calls_per_minute * 100)
        }
        
        # Ajouter les statistiques détaillées du moniteur si disponible
        if USE_API_MONITOR:
            monitor_stats = self.api_monitor.get_usage_statistics()
            if self.api_name in monitor_stats:
                current_usage["detailed_stats"] = monitor_stats[self.api_name]
        
        return current_usage

# Créer une instance globale pour toute l'application
rate_manager = APIRateManager()

def rate_limited(f):
    """
    Décorateur pour limiter le taux d'appel des fonctions API
    
    Usage:
        @rate_limited
        def ma_fonction_api():
            ...
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        # Détecter le nom de la fonction
        endpoint = f.__name__
        
        # Obtenir l'instance du gestionnaire de taux
        rate_manager = APIRateManager()
        
        # Attendre si nécessaire
        rate_manager.wait_if_needed(endpoint=endpoint)
        rate_manager.wait_if_needed_continued(endpoint=endpoint)
        
        # Exécuter la fonction
        return f(*args, **kwargs)
    return wrapper
