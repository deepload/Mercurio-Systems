"""
Market Data Provider Factory

This module provides a factory for creating and managing market data providers.
"""
import os
import logging
from typing import Dict, List, Type, Optional

# Import environment loader to ensure .env variables are loaded
from app.utils import env_loader

# Import base provider and implementations
from app.services.providers.base import MarketDataProvider
from app.services.providers.sample import SampleDataProvider

logger = logging.getLogger(__name__)

class MarketDataProviderFactory:
    """
    Factory for creating and managing market data providers.
    
    This class maintains a registry of available providers and handles
    provider initialization, selection, and fallback logic.
    """
    
    def __init__(self):
        """Initialize the provider factory."""
        self._providers = {}
        self._provider_classes = {}
        self._provider_priorities = {}
        self._default_provider = None
        
        # Always register the sample data provider as the lowest priority fallback
        self.register_provider("sample", SampleDataProvider, priority=100)
        
        # Register the other providers dynamically
        self._register_available_providers()
    
    def _register_available_providers(self):
        """Register all available providers dynamically."""
        # Try to register Alpaca provider (highest priority for premium subscription)
        try:
            from app.services.providers.alpaca import AlpacaProvider
            self.register_provider("alpaca", AlpacaProvider, priority=5)
            logger.info("Alpaca provider registered with highest priority")
        except ImportError:
            logger.info("Alpaca provider not available (missing dependencies)")
            
        # Try to register Polygon provider
        try:
            from app.services.providers.polygon import PolygonProvider
            self.register_provider("polygon", PolygonProvider, priority=10)
        except ImportError:
            logger.info("Polygon provider not available (missing dependencies)")
        
        # Try to register Yahoo Finance provider
        try:
            from app.services.providers.yahoo import YahooFinanceProvider
            self.register_provider("yahoo", YahooFinanceProvider, priority=20)
        except ImportError:
            logger.info("Yahoo Finance provider not available (missing dependencies)")
    
    def register_provider(self, name: str, provider_class: Type[MarketDataProvider], priority: int = 50):
        """
        Register a new provider with the factory.
        
        Args:
            name: Provider name key
            provider_class: Provider class
            priority: Provider priority (lower is higher priority)
        """
        self._provider_classes[name] = provider_class
        self._provider_priorities[name] = priority
        logger.info(f"Registered provider '{name}' with priority {priority}")
    
    def initialize_provider(self, name: str, **kwargs) -> Optional[MarketDataProvider]:
        """
        Initialize a provider by name.
        
        Args:
            name: Provider name key
            **kwargs: Additional args to pass to the provider constructor
            
        Returns:
            Initialized provider instance or None if initialization fails
        """
        if name not in self._provider_classes:
            logger.warning(f"Provider '{name}' not registered")
            return None
            
        provider_class = self._provider_classes[name]
        
        try:
            provider = provider_class(**kwargs)
            
            # Check if provider is available
            if hasattr(provider, 'is_available') and not provider.is_available:
                logger.warning(f"Provider '{name}' is not available")
                return None
                
            self._providers[name] = provider
            logger.info(f"Initialized provider '{name}'")
            return provider
        except Exception as e:
            logger.error(f"Failed to initialize provider '{name}': {e}")
            return None
    
    def get_provider(self, name: str) -> Optional[MarketDataProvider]:
        """
        Get an initialized provider by name.
        
        Args:
            name: Provider name key
            
        Returns:
            Provider instance or None if not available
        """
        # If provider is already initialized, return it
        if name in self._providers:
            return self._providers[name]
            
        # Try to initialize the provider
        return self.initialize_provider(name)
    
    def get_available_providers(self) -> List[str]:
        """
        Get a list of all available provider names.
        
        Returns:
            List of provider names
        """
        return list(self._provider_classes.keys())
    
    def get_default_provider(self) -> MarketDataProvider:
        """
        Get the default provider based on availability and priority.
        
        Returns:
            Default provider instance
        """
        if self._default_provider:
            return self._default_provider
            
        # Get all provider names sorted by priority
        provider_names = sorted(
            self._provider_priorities.keys(),
            key=lambda name: self._provider_priorities[name]
        )
        
        # Try to initialize each provider in priority order
        for name in provider_names:
            provider = self.get_provider(name)
            if provider:
                self._default_provider = provider
                logger.info(f"Using '{name}' as default provider")
                return provider
        
        # If all else fails, use sample data provider
        sample_provider = self.get_provider("sample")
        if not sample_provider:
            sample_provider = SampleDataProvider()
            self._providers["sample"] = sample_provider
            
        self._default_provider = sample_provider
        logger.info("Using sample data provider as default")
        return sample_provider
