"""
MercurioAI Transformer Strategy

This module implements a strategy that uses a Transformer neural network architecture
to identify complex patterns in multi-timeframe market data.
"""
import os
import logging
import json
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from abc import abstractmethod
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import joblib
from pathlib import Path

from app.strategies.base import BaseStrategy
from app.db.models import TradeAction

logger = logging.getLogger(__name__)

# Ensure torch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not found. TransformerStrategy will use fallback mode.")
    TORCH_AVAILABLE = False

class TimeSeriesDataset(Dataset):
    """Dataset for time series data to be fed into PyTorch models"""
    
    def __init__(self, data: np.ndarray, targets: np.ndarray, seq_length: int = 60):
        """
        Initialize dataset
        
        Args:
            data: Input features (n_samples, n_features)
            targets: Target values
            seq_length: Sequence length for each sample
        """
        self.data = data
        self.targets = targets
        self.seq_length = seq_length
        
    def __len__(self):
        # Ensure __len__ never returns negative values
        return max(0, len(self.data) - self.seq_length)
        
    def __getitem__(self, idx):
        # Get sequence
        x = self.data[idx:idx + self.seq_length]
        # Get target (next day's movement)
        y = self.targets[idx + self.seq_length]
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class TransformerModel(nn.Module):
    """Transformer model for time series prediction"""
    
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int,
                 d_model: int = 64,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        """
        Initialize transformer model
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output dimensions (usually 1 for regression)
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super().__init__()
        
        # Feature embedding
        self.embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Output layer
        self.output_layer = nn.Linear(d_model, output_dim)
        
    def forward(self, src):
        """Forward pass"""
        # src shape: [seq_len, batch_size, input_dim]
        
        # Embed features to model dimension
        src = self.embedding(src)
        
        # Add positional encoding
        src = self.pos_encoder(src)
        
        # Pass through transformer
        output = self.transformer_encoder(src)
        
        # Use the output from the last time step
        output = output[-1]
        
        # Project to output dimension
        output = self.output_layer(output)
        
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer model"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Initialize positional encoding
        
        Args:
            d_model: Model dimension
            dropout: Dropout rate
            max_len: Maximum sequence length
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """Forward pass"""
        # x shape: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerStrategy(BaseStrategy):
    """
    Trading strategy based on a transformer neural network.
    
    This strategy uses the Transformer architecture to analyze market data
    across multiple timeframes and identify complex patterns.
    """
    
    def __init__(self, 
                 sequence_length: int = 60,
                 prediction_horizon: int = 1,
                 d_model: int = 64,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 epochs: int = 50,
                 signal_threshold: float = 0.6,
                 use_gpu: bool = True,
                 **kwargs):
        """
        Initialize the transformer strategy
        
        Args:
            sequence_length: Number of time steps to consider for prediction
            prediction_horizon: How many steps ahead to predict
            d_model: Model dimension for transformer
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            epochs: Number of training epochs
            signal_threshold: Threshold for generating trading signals
            use_gpu: Whether to use GPU if available
        """
        super().__init__(**kwargs)
        
        # Model parameters
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.signal_threshold = signal_threshold
        
        # Training parameters
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        
        # Model objects
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.is_trained = False
        
        # Default features
        self.default_features = [
            'open', 'high', 'low', 'close', 'volume', 
            'sma_20', 'sma_50', 'rsi_14', 'macd', 'adx_14'
        ]
        
        # Create model directory
        self.model_dir = Path('./models/transformer')
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized TransformerStrategy with {sequence_length} sequence length")
    
    async def load_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Load historical data for the strategy
        
        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with historical data
        """
        # Add extra days to account for sequence length
        extended_start = start_date - timedelta(days=self.sequence_length * 2)
        
        # Load data using the market data service
        from app.services.market_data import MarketDataService
        market_data = MarketDataService()
        
        # Format dates to strings
        start_str = extended_start.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        data = await market_data.get_historical_data(symbol, start_str, end_str)
        
        return data
    
    async def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Patch: Robust to small datasets, always returns at least one row if possible.
        """
        if data is None or len(data) == 0:
            logger.error("No data to preprocess")
            return None
        original_data = data.copy()
        original_length = len(data)
        df = data.copy()
        # Make sure timestamp is a datetime
        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Use smallest possible window for technical indicators if data is short
        min_win = 2 if len(df) < 10 else 3
        if 'sma_20' not in df.columns:
            logger.info("Adding technical indicators")
            df['sma_2'] = df['close'].rolling(window=2, min_periods=1).mean()
            df['sma_3'] = df['close'].rolling(window=min_win, min_periods=1).mean()
            df['ema_2'] = df['close'].ewm(span=2, adjust=False).mean()
            df['ema_3'] = df['close'].ewm(span=min_win, adjust=False).mean()
        # Calculate prediction target (future price movement)
        logger.info("Calculating prediction targets")
        df['target'] = df['close'].shift(-1)
        df = df.dropna()
        # Store feature columns: only numeric, exclude timestamp/date/symbol/etc.
        self.feature_columns = [
            col for col in df.columns
            if col not in ['timestamp', 'target', 'symbol', 'date', 'date_str']
            and pd.api.types.is_numeric_dtype(df[col])
        ]
        if len(df) == 0 and original_length > 0:
            logger.warning(f"Not enough data after feature engineering for {self.__class__.__name__}. Using last available row as fallback.")
            df = original_data.tail(1)
        return df
        """
        Preprocess data for training and prediction
        
        Args:
            data: Raw market data
            
        Returns:
            Preprocessed data
        """
        if data is None or len(data) == 0:
            logger.error("No data to preprocess")
            return None
            
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Make sure timestamp is a datetime
        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Add technical indicators if not already present
        if 'sma_20' not in df.columns:
            logger.info("Adding technical indicators")
            df = await self._add_technical_indicators(df)
        
        # Calculate prediction target (future price movement)
        logger.info("Calculating prediction targets")
        df = await self._calculate_prediction_targets(df)
        
        # Handle missing values
        df = df.dropna()
        
        # Store feature columns
        self.feature_columns = [col for col in df.columns if col not in ['timestamp', 'target', 'symbol']]
        
        return df
    
    async def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the data without relying on TA-Lib
        
        Args:
            data: Price data
            
        Returns:
            Data with technical indicators
        """
        df = data.copy()
        
        # Simple Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macdsignal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macdhist'] = df['macd'] - df['macdsignal']
        
        # Simple approximation for ADX
        # True Range
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr_14'] = df['tr'].rolling(window=14).mean()
        
        # Simple directional movement
        df['dx'] = abs(
            (df['high'] - df['high'].shift(1)) - (df['low'] - df['low'].shift(1))
        ) / df['tr'] * 100
        
        # Simple ADX approximation
        df['adx_14'] = df['dx'].rolling(window=14).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['sma_20']
        rolling_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (rolling_std * 2)
        df['bb_lower'] = df['bb_middle'] - (rolling_std * 2)
        
        return df
    
    async def _calculate_prediction_targets(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate prediction targets
        
        Args:
            data: Price data
            
        Returns:
            Data with prediction targets
        """
        df = data.copy()
        
        # Calculate future returns for the prediction horizon
        future_close = df['close'].shift(-self.prediction_horizon)
        df['future_return'] = (future_close / df['close']) - 1
        
        # Create classification target (1 for up, 0 for down/flat)
        df['target'] = (df['future_return'] > 0).astype(int)
        
        # Create regression target (normalized future return)
        df['target_regression'] = df['future_return']
        
        return df
    
    async def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the model
        
        Args:
            data: Preprocessed data
            
        Returns:
            Training metrics
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available. TransformerStrategy will use simplified logic.")
            self.is_trained = True
            return {"status": "fallback", "message": "PyTorch not available"}
            
        logger.info("Starting transformer model training")
        
        # Prepare data for training
        X, y = await self._prepare_training_data(data)

        # Check for empty data
        if X.shape[0] <= self.sequence_length or y.shape[0] <= self.sequence_length:
            logger.warning(f"[TransformerStrategy] Not enough data to train (data rows: {X.shape[0]}, sequence_length: {self.sequence_length})")
            return {"status": "error", "message": f"Not enough data to train Transformer (data rows: {X.shape[0]}, sequence_length: {self.sequence_length})"}

        # Split into train and validation sets (80/20)
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]

        # Create dataset and dataloaders
        train_dataset = TimeSeriesDataset(X_train, y_train, self.sequence_length)
        val_dataset = TimeSeriesDataset(X_val, y_val, self.sequence_length)
        if len(train_dataset) <= 0 or len(val_dataset) <= 0:
            logger.warning(f"[TransformerStrategy] Training or validation dataset is empty (train: {len(train_dataset)}, val: {len(val_dataset)}). Skipping training.")
            return {"status": "error", "message": "Training or validation dataset is empty. Skipping training."}
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Initialize model
        input_dim = X.shape[1]
        self.model = TransformerModel(
            input_dim=input_dim,
            output_dim=1,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # Loss function and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        best_val_loss = float('inf')
        early_stop_counter = 0
        early_stop_patience = 10
        
        metrics = {"train_losses": [], "val_losses": [], "val_accuracies": []}
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.permute(1, 0, 2).to(self.device)  # [seq_len, batch_size, features]
                target = target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output.squeeze(), target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            metrics["train_losses"].append(train_loss)
            
            # Validation
            self.model.eval()
            val_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data = data.permute(1, 0, 2).to(self.device)
                    target = target.to(self.device)
                    
                    output = self.model(data)
                    output = torch.sigmoid(output).view(-1)
                    target = target.view(-1)
                    loss = criterion(output, target).item()
                    
                    # Calculate accuracy
                    pred = torch.sigmoid(output) > 0.5
                    correct += pred.eq(target.view_as(pred)).sum().item()
            
            val_loss /= len(val_loader)
            val_accuracy = correct / len(val_dataset)
            
            metrics["val_losses"].append(val_loss)
            metrics["val_accuracies"].append(val_accuracy)
            
            logger.info(f"Epoch {epoch+1}/{self.epochs} - "
                       f"Train loss: {train_loss:.4f}, "
                       f"Val loss: {val_loss:.4f}, "
                       f"Val accuracy: {val_accuracy:.4f}")
            
            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                
                # Save best model
                self._save_model()
            else:
                early_stop_counter += 1
                
            if early_stop_counter >= early_stop_patience:
                logger.info(f"Early stopping after {epoch+1} epochs")
                break
        
        # Load best model
        self._load_model()
        
        # Set trained flag
        self.is_trained = True
        
        return {
            "status": "success",
            "epochs_completed": epoch + 1,
            "best_val_loss": best_val_loss,
            "final_val_accuracy": val_accuracy,
            "training_metrics": metrics
        }
    
    async def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for training
        
        Args:
            data: Preprocessed data
            
        Returns:
            Features and targets as numpy arrays
        """
        # Get feature columns
        features = data[self.feature_columns].values
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)
        
        # Get targets (binary classification)
        targets = data['target'].values
        
        return features_scaled, targets
    
    async def predict(self, data: pd.DataFrame) -> Tuple[TradeAction, float]:
        """
        Generate trading signals
        
        Args:
            data: Market data
            
        Returns:
            Trading action and confidence
        """
        if not self.is_trained:
            logger.warning("Model is not trained yet")
            return TradeAction.HOLD, 0.0
            
        if not TORCH_AVAILABLE:
            # Fallback prediction logic when torch is not available
            return await self._predict_fallback(data)
            
        # Make sure we have at least sequence_length data points
        if len(data) < self.sequence_length:
            logger.warning(f"Not enough data for prediction, need at least {self.sequence_length} points")
            return TradeAction.HOLD, 0.0
            
        # Preprocess data if necessary
        if 'target' not in data.columns:
            data = await self.preprocess_data(data)
            
        # Get features
        features = data[self.feature_columns].values
        
        # Normalize features
        features_scaled = self.scaler.transform(features)
        
        # Get the most recent sequence
        sequence = features_scaled[-self.sequence_length:]
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(1)  # Add batch dimension
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            output = self.model(sequence_tensor)
            probability = torch.sigmoid(output).item()
            
        # Determine action and confidence
        if probability > self.signal_threshold:
            action = TradeAction.BUY
            confidence = probability
        elif probability < (1 - self.signal_threshold):
            action = TradeAction.SELL
            confidence = 1 - probability
        else:
            action = TradeAction.HOLD
            confidence = 0.5
            
        logger.info(f"Prediction: {action.name} with confidence {confidence:.4f}")
        
        return action, confidence
    
    async def _predict_fallback(self, data: pd.DataFrame) -> Tuple[TradeAction, float]:
        """
        Fallback prediction method when PyTorch is not available
        
        Args:
            data: Market data
            
        Returns:
            Trading action and confidence
        """
        # Use simple moving average crossover as fallback
        if 'sma_20' not in data.columns or 'sma_50' not in data.columns:
            data = await self._add_technical_indicators(data)
            
        # Get latest values
        latest = data.iloc[-1]
        previous = data.iloc[-2] if len(data) > 1 else None
        
        # Simple moving average crossover strategy
        if previous is not None:
            # Bullish crossover
            if previous['sma_20'] <= previous['sma_50'] and latest['sma_20'] > latest['sma_50']:
                return TradeAction.BUY, 0.8
            # Bearish crossover
            elif previous['sma_20'] >= previous['sma_50'] and latest['sma_20'] < latest['sma_50']:
                return TradeAction.SELL, 0.8
                
        # Use RSI for additional signals
        if 'rsi_14' in latest:
            if latest['rsi_14'] < 30:
                return TradeAction.BUY, 0.7
            elif latest['rsi_14'] > 70:
                return TradeAction.SELL, 0.7
                
        return TradeAction.HOLD, 0.5
    
    async def backtest(self, data: pd.DataFrame, initial_capital: float = 10000.0) -> Dict[str, Any]:
        """
        Backtest the strategy
        
        Args:
            data: Market data
            initial_capital: Initial capital
            
        Returns:
            Backtest results
        """
        # Check for empty or insufficient data
        if data is None or len(data) == 0:
            logger.warning("[TransformerStrategy] No data provided for backtest.")
            return {
                'initial_capital': initial_capital,
                'final_equity': float('nan'),
                'total_return': float('nan'),
                'equity_curve': pd.DataFrame(),
                'trades': [],
                'position': 0,
                'strategy': 'TransformerStrategy',
                'error': 'No data provided for backtest.'
            }
        if len(data) < max(3, self.sequence_length + 1):
            logger.warning(f"[TransformerStrategy] Not enough data to backtest (required: {max(3, self.sequence_length+1)}, got: {len(data)}). Skipping.")
            return {
                'initial_capital': initial_capital,
                'final_equity': float('nan'),
                'total_return': float('nan'),
                'equity_curve': pd.DataFrame(),
                'trades': [],
                'position': 0,
                'strategy': 'TransformerStrategy',
                'error': f'Not enough data to backtest (required: {max(3, self.sequence_length+1)}, got: {len(data)}).'
            }
        # Preprocess data if necessary
        if 'target' not in data.columns:
            data = await self.preprocess_data(data)
        # Check after preprocessing: must have 'close', not be empty, not all NaN
        error_msg = None
        def error_result(msg):
            logger.error(f"[TransformerStrategy][ERROR] {msg}")
            return {
                'initial_capital': initial_capital,
                'final_equity': float('nan'),
                'total_return': float('nan'),
                'initial_close': float('nan'),
                'final_close': float('nan'),
                'equity_curve': pd.DataFrame(),
                'trades': [],
                'position': 0,
                'strategy': 'TransformerStrategy',
                'error': msg
            }
        if data is None or len(data) == 0:
            return error_result('No usable data after preprocessing.')
        if 'close' not in data.columns:
            return error_result('No close column after preprocessing.')
        if data['close'].isna().all():
            return error_result('All close values are NaN after preprocessing.')
        initial_close = data['close'].iloc[0]
        final_close = data['close'].iloc[-1]
        if (pd.isna(initial_close) or pd.isna(final_close) or initial_close is None or final_close is None or initial_close == 0):
            return error_result(f'Invalid initial/final close (initial: {initial_close}, final: {final_close})')
        # Make sure model is trained
        if not self.is_trained:
            logger.info("Model not trained, training now...")
            await self.train(data)
        # Initialize backtest variables
        capital = initial_capital
        position = 0
        equity_curve = []
        trades = []
        
        # Loop through data
        for i in range(self.sequence_length, len(data)):
            # Get current price
            current_data = data.iloc[:i+1]
            current_price = current_data['close'].iloc[-1]
            current_date = current_data['timestamp'].iloc[-1] if 'timestamp' in current_data.columns else i
            
            # Get prediction
            action, confidence = await self.predict(current_data)
            
            # Calculate position and equity
            previous_position = position
            
            # Update position based on action
            if action == TradeAction.BUY and position <= 0:
                # Close any short position
                if position < 0:
                    capital += position * current_price * -1  # Close short
                    trades.append({
                        'timestamp': current_date,
                        'action': 'CLOSE_SHORT',
                        'price': current_price,
                        'quantity': position * -1,
                        'capital': capital
                    })
                    position = 0
                
                # Open long position - invest 95% of capital
                quantity = (capital * 0.95) / current_price
                capital -= quantity * current_price
                position += quantity
                trades.append({
                    'timestamp': current_date,
                    'action': 'BUY',
                    'price': current_price,
                    'quantity': quantity,
                    'capital': capital
                })
                
            elif action == TradeAction.SELL and position >= 0:
                # Close any long position
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
                
                # Open short position - 95% of capital
                quantity = (capital * 0.95) / current_price
                capital += quantity * current_price  # Short sell proceeds
                position -= quantity
                trades.append({
                    'timestamp': current_date,
                    'action': 'SELL',
                    'price': current_price,
                    'quantity': quantity,
                    'capital': capital
                })
            
            # Calculate equity (capital + position value)
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
        
        # Calculate final equity
        final_equity = capital
        if position != 0:
            final_price = data['close'].iloc[-1]
            final_equity += position * final_price
            
        # Calculate performance metrics
        import math
        error_msg = None
        if initial_capital == 0:
            logger.warning("[TransformerStrategy] initial_capital is zero, cannot compute return.")
            total_return = float('nan')
            error_msg = 'Initial capital is zero, cannot compute return.'
        elif final_equity is None or math.isnan(final_equity) or math.isinf(final_equity):
            logger.warning(f"[TransformerStrategy] final_equity is invalid (nan or inf): {final_equity}")
            total_return = float('nan')
            error_msg = f'Final equity is invalid (nan or inf): {final_equity}'
        else:
            total_return = (final_equity / initial_capital) - 1
        equity_df = pd.DataFrame(equity_curve)
        # Always propagate initial/final close
        results = {
            'initial_capital': initial_capital,
            'final_equity': final_equity,
            'total_return': total_return,
            'initial_close': initial_close,
            'final_close': final_close,
            'equity_curve': equity_df,
            'trades': trades,
            'position': position,
            'strategy': 'TransformerStrategy',
            'error': error_msg
        }
        return results
    
    def _save_model(self) -> None:
        """Save model to disk using recommended formats"""
        # Créer les répertoires si nécessaire
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Sauvegarder le modèle complet au lieu de juste les états du dictionnaire
        # C'est l'approche recommandée dans PyTorch 2.0+
        model_path = self.model_dir / 'transformer_model.pt'
        torch.save(self.model, model_path)
        
        # Sauvegarder également les états du dictionnaire pour la rétrocompatibilité
        weights_path = self.model_dir / 'transformer_model.pth'
        torch.save(self.model.state_dict(), weights_path)
        
        # Save scaler and feature columns
        metadata = {
            'feature_columns': self.feature_columns,
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'signal_threshold': self.signal_threshold,
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_layers': self.num_layers,
            'model_version': '2.0',  # Ajouter une version pour la compatibilité future
            'saved_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(self.model_dir / 'transformer_metadata.json', 'w') as f:
            json.dump(metadata, f)
            
        if self.scaler is not None:
            joblib.dump(self.scaler, self.model_dir / 'transformer_scaler.joblib')
            
        logger.info(f"Modèle complet sauvegardé dans {model_path}")
        logger.info(f"Poids du modèle sauvegardés dans {weights_path}")
    
    def _load_model(self) -> None:
        """Load model from disk with support for both new and legacy formats"""
        # Vérifier d'abord le nouveau format (.pt pour le modèle complet)
        model_full_path = self.model_dir / 'transformer_model.pt'
        weights_path = self.model_dir / 'transformer_model.pth'
        
        # Déterminer quel format utiliser
        use_full_model = model_full_path.exists()
        use_weights = weights_path.exists()
        
        if not (use_full_model or use_weights):
            logger.warning(f"Aucun fichier de modèle trouvé dans {self.model_dir}")
            return
        
        # D'abord charger les métadonnées pour obtenir les paramètres du modèle
        metadata_path = self.model_dir / 'transformer_metadata.json'
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.feature_columns = metadata.get('feature_columns', self.feature_columns)
                    self.sequence_length = metadata.get('sequence_length', self.sequence_length)
                    self.prediction_horizon = metadata.get('prediction_horizon', self.prediction_horizon)
                    self.signal_threshold = metadata.get('signal_threshold', self.signal_threshold)
                    self.d_model = metadata.get('d_model', self.d_model)
                    self.nhead = metadata.get('nhead', self.nhead)
                    self.num_layers = metadata.get('num_layers', self.num_layers)
                    
                    # Obtenir la version du modèle si disponible
                    model_version = metadata.get('model_version', '1.0')
                    logger.info(f"Version du modèle: {model_version}")
            except Exception as e:
                logger.error(f"Erreur lors du chargement des métadonnées: {e}")
                    
        # Charger le scaler
        scaler_path = self.model_dir / 'transformer_scaler.joblib'
        if scaler_path.exists():
            try:
                self.scaler = joblib.load(scaler_path)
                logger.info(f"Scaler chargé depuis {scaler_path}")
            except Exception as e:
                logger.error(f"Erreur lors du chargement du scaler: {e}")
        
        try:
            # Essayer de charger le modèle complet d'abord (format recommandé)
            if use_full_model:
                logger.info(f"Chargement du modèle complet depuis {model_full_path}")
                self.model = torch.load(model_full_path, map_location=self.device)
                self.model.eval()
                self.is_trained = True
            # Sinon, initialiser le modèle et charger les poids
            elif use_weights:
                logger.info(f"Initialisation du modèle et chargement des poids depuis {weights_path}")
                input_dim = len(self.feature_columns) if self.feature_columns else 10
                self.model = TransformerModel(
                    input_dim=input_dim,
                    output_dim=1,
                    d_model=self.d_model,
                    nhead=self.nhead,
                    num_layers=self.num_layers,
                    dropout=self.dropout
                ).to(self.device)
                
                # Charger les poids du modèle
                self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
                self.model.eval()
                self.is_trained = True
            
            logger.info(f"Modèle chargé avec succès")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {e}")
            self.is_trained = False
