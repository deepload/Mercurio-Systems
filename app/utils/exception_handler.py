"""
Gestionnaire d'exceptions MercurioAI

Module qui fournit un système centralisé pour capturer, journaliser
et gérer les exceptions dans MercurioAI avec des stratégies de récupération.
"""

import logging
import traceback
import functools
import asyncio
import time
from typing import Callable, Any, Dict, Optional, Type, List, Union

logger = logging.getLogger(__name__)

class TradingException(Exception):
    """Exception de base pour toutes les erreurs liées au trading"""
    def __init__(self, message: str, error_code: str = None, details: Dict = None):
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.details = details or {}
        super().__init__(message)

class MarketDataException(TradingException):
    """Exception liée aux données de marché"""
    def __init__(self, message: str, error_code: str = None, details: Dict = None):
        super().__init__(message, error_code or "MARKET_DATA_ERROR", details)

class StrategyException(TradingException):
    """Exception liée aux stratégies de trading"""
    def __init__(self, message: str, error_code: str = None, details: Dict = None):
        super().__init__(message, error_code or "STRATEGY_ERROR", details)

class ExecutionException(TradingException):
    """Exception liée à l'exécution des ordres"""
    def __init__(self, message: str, error_code: str = None, details: Dict = None):
        super().__init__(message, error_code or "EXECUTION_ERROR", details)

class ExceptionManager:
    """
    Gestionnaire centralisé des exceptions pour MercurioAI
    
    Fournit des fonctionnalités pour:
    - Capturer les exceptions avec contexte
    - Implémenter des stratégies de récupération
    - Journaliser les erreurs de manière consistante
    - Alerter sur les erreurs critiques
    """
    
    def __init__(self):
        self.error_registry = {}
        self.retry_policies = {
            MarketDataException: {"max_retries": 3, "delay": 2, "backoff": 2},
            StrategyException: {"max_retries": 2, "delay": 1, "backoff": 1},
            ExecutionException: {"max_retries": 3, "delay": 1, "backoff": 1.5},
        }
        self.fallback_handlers = {}
        self.critical_errors_count = 0
    
    def register_fallback_handler(self, exception_type: Type[Exception], handler: Callable):
        """Enregistre un gestionnaire de fallback pour un type d'exception"""
        self.fallback_handlers[exception_type] = handler
    
    def log_exception(self, e: Exception, context: str = ""):
        """
        Journalise une exception avec son contexte et sa stack trace
        """
        if isinstance(e, TradingException):
            logger.error(f"{context} - {e.error_code}: {e.message}")
            if e.details:
                logger.error(f"Details: {e.details}")
        else:
            logger.error(f"{context} - Exception: {str(e)}")
        
        logger.debug(f"Stack trace: {''.join(traceback.format_tb(e.__traceback__))}")
        
        # Incrémenter le compteur d'erreurs critiques si nécessaire
        if isinstance(e, (SystemError, RuntimeError, KeyboardInterrupt)):
            self.critical_errors_count += 1
    
    def get_retry_policy(self, exception: Exception) -> Dict:
        """Récupère la politique de retry pour un type d'exception"""
        for exception_type, policy in self.retry_policies.items():
            if isinstance(exception, exception_type):
                return policy
        
        # Policy par défaut
        return {"max_retries": 1, "delay": 1, "backoff": 1}
    
    def handle_exception(self, exception: Exception, context: str = "") -> Any:
        """
        Gère une exception en appliquant les stratégies de récupération appropriées
        
        Returns:
            Résultat du gestionnaire de fallback ou None
        """
        self.log_exception(exception, context)
        
        # Chercher et appliquer le gestionnaire de fallback
        for exception_type, handler in self.fallback_handlers.items():
            if isinstance(exception, exception_type):
                try:
                    return handler(exception)
                except Exception as fallback_err:
                    logger.error(f"Erreur dans le gestionnaire de fallback: {fallback_err}")
        
        return None
    
    def with_retry(self, fn=None, *, 
                  retry_exceptions: List[Type[Exception]] = None,
                  max_retries: int = None,
                  delay: float = None,
                  backoff: float = None):
        """
        Décorateur pour exécuter une fonction avec une politique de retry
        
        Args:
            fn: Fonction à décorer
            retry_exceptions: Types d'exceptions à intercepter pour retry
            max_retries: Nombre maximal de tentatives
            delay: Délai initial entre les tentatives
            backoff: Facteur multiplicatif pour augmenter le délai
        """
        if fn is None:
            return functools.partial(self.with_retry, 
                                    retry_exceptions=retry_exceptions,
                                    max_retries=max_retries,
                                    delay=delay,
                                    backoff=backoff)
        
        retry_exceptions = retry_exceptions or [Exception]
        
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exception = None
            # Policy par défaut
            policy = {"max_retries": 3, "delay": 1, "backoff": 1.5}
            
            # Utiliser les valeurs spécifiées si présentes
            if max_retries is not None:
                policy["max_retries"] = max_retries
            if delay is not None:
                policy["delay"] = delay
            if backoff is not None:
                policy["backoff"] = backoff
            
            for attempt in range(policy["max_retries"]):
                try:
                    return fn(*args, **kwargs)
                except tuple(retry_exceptions) as e:
                    last_exception = e
                    self.log_exception(e, f"Tentative {attempt+1}/{policy['max_retries']} de {fn.__name__}")
                    
                    # Éviter de dormir après la dernière tentative
                    if attempt < policy["max_retries"] - 1:
                        sleep_time = policy["delay"] * (policy["backoff"] ** attempt)
                        logger.info(f"Nouvelle tentative dans {sleep_time} secondes...")
                        time.sleep(sleep_time)
            
            # Si toutes les tentatives ont échoué
            if last_exception:
                logger.error(f"Échec de toutes les tentatives pour {fn.__name__}")
                return self.handle_exception(last_exception, f"Fonction {fn.__name__}")
            
            return None
        
        return wrapper
    
    async def with_async_retry(self, fn=None, *, 
                              retry_exceptions: List[Type[Exception]] = None,
                              max_retries: int = None,
                              delay: float = None,
                              backoff: float = None):
        """Version asynchrone du décorateur with_retry pour les coroutines"""
        if fn is None:
            return functools.partial(self.with_async_retry, 
                                    retry_exceptions=retry_exceptions,
                                    max_retries=max_retries,
                                    delay=delay,
                                    backoff=backoff)
        
        retry_exceptions = retry_exceptions or [Exception]
        
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            last_exception = None
            # Policy par défaut
            policy = {"max_retries": 3, "delay": 1, "backoff": 1.5}
            
            # Utiliser les valeurs spécifiées si présentes
            if max_retries is not None:
                policy["max_retries"] = max_retries
            if delay is not None:
                policy["delay"] = delay
            if backoff is not None:
                policy["backoff"] = backoff
            
            for attempt in range(policy["max_retries"]):
                try:
                    return await fn(*args, **kwargs)
                except tuple(retry_exceptions) as e:
                    last_exception = e
                    self.log_exception(e, f"Tentative {attempt+1}/{policy['max_retries']} de {fn.__name__}")
                    
                    # Éviter de dormir après la dernière tentative
                    if attempt < policy["max_retries"] - 1:
                        sleep_time = policy["delay"] * (policy["backoff"] ** attempt)
                        logger.info(f"Nouvelle tentative dans {sleep_time} secondes...")
                        await asyncio.sleep(sleep_time)
            
            # Si toutes les tentatives ont échoué
            if last_exception:
                logger.error(f"Échec de toutes les tentatives pour {fn.__name__}")
                return self.handle_exception(last_exception, f"Fonction {fn.__name__}")
            
            return None
        
        return wrapper

# Instance globale du gestionnaire d'exceptions
exception_manager = ExceptionManager()

# Fonction de commodité pour décorer des fonctions avec le gestionnaire
def with_exception_handling(fn=None, *, 
                           retry: bool = False,
                           retry_exceptions: List[Type[Exception]] = None,
                           max_retries: int = None):
    """
    Décorateur de commodité pour appliquer la gestion d'exceptions
    
    Args:
        fn: Fonction à décorer
        retry: Si True, ajoutera un mécanisme de retry
        retry_exceptions: Types d'exceptions à intercepter pour retry
        max_retries: Nombre maximal de tentatives
    """
    if retry:
        if asyncio.iscoroutinefunction(fn):
            return exception_manager.with_async_retry(
                fn, retry_exceptions=retry_exceptions, max_retries=max_retries
            )
        else:
            return exception_manager.with_retry(
                fn, retry_exceptions=retry_exceptions, max_retries=max_retries
            )
    
    if fn is None:
        return functools.partial(with_exception_handling, 
                                retry=retry,
                                retry_exceptions=retry_exceptions,
                                max_retries=max_retries)
    
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            return exception_manager.handle_exception(e, f"Fonction {fn.__name__}")
    
    @functools.wraps(fn)
    async def async_wrapper(*args, **kwargs):
        try:
            return await fn(*args, **kwargs)
        except Exception as e:
            return exception_manager.handle_exception(e, f"Fonction {fn.__name__}")
    
    if asyncio.iscoroutinefunction(fn):
        return async_wrapper
    else:
        return wrapper
