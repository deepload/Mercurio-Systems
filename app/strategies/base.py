"""
Base Strategy Module

Defines the abstract base class that all trading strategies must implement
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime

from app.db.models import TradeAction

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies in Mercurio AI.
    
    All strategy implementations must inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the strategy with configuration parameters.
        
        Args:
            **kwargs: Strategy-specific parameters
        """
        self.name = self.__class__.__name__
        self.params = kwargs
        self.is_trained = False
        self.model = None
    
    @abstractmethod
    async def load_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Load historical market data for the given symbol and date range.
        
        Args:
            symbol: The market symbol (e.g., 'AAPL')
            start_date: Start date for data loading
            end_date: End date for data loading
            
        Returns:
            DataFrame containing the historical data
        """
        pass
    
    @abstractmethod
    async def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data for training/inference.
        
        Args:
            data: Raw market data
            
        Returns:
            Preprocessed data
        """
        pass
    
    @abstractmethod
    async def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the strategy model on historical data.
        
        Args:
            data: Preprocessed market data
            
        Returns:
            Dictionary containing training metrics
        """
        pass
    
    @abstractmethod
    async def predict(self, data: pd.DataFrame) -> Tuple[TradeAction, float]:
        """
        Generate a trading signal based on the input data.
        
        Args:
            data: Market data to analyze
            
        Returns:
            Tuple of (TradeAction, confidence)
        """
        pass
    
    @abstractmethod
    async def backtest(self, data: pd.DataFrame, initial_capital: float = 10000.0) -> Dict[str, Any]:
        """
        Backtest the strategy on historical data.
        
        Args:
            data: Historical market data
            initial_capital: Initial capital for the backtest
            
        Returns:
            Dictionary containing backtest results
        """
        pass
    
    async def save_model(self, path: str) -> str:
        """
        Save the trained model to disk.
        
        Args:
            path: Directory to save the model
            
        Returns:
            Path to the saved model
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model is not trained yet")
        
        # This method should be implemented by concrete strategies
        # Default implementation does nothing
        return ""
    
    async def load_model(self, path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            path: Path to the saved model
        """
        # This method should be implemented by concrete strategies
        # Default implementation does nothing
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about this strategy.
        
        Returns:
            Dictionary with strategy metadata
        """
        return {
            "name": self.name,
            "params": self.params,
            "is_trained": self.is_trained
        }
