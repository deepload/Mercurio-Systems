"""
LSTM Predictor Strategy

A deep learning strategy that uses Long Short-Term Memory (LSTM) networks
to predict price movements and generate trading signals.
"""
import os
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

from app.db.models import TradeAction
from app.strategies.base import BaseStrategy
from app.services.market_data import MarketDataService

logger = logging.getLogger(__name__)

class LSTMPredictorStrategy(BaseStrategy):
    """
    LSTM Predictor Strategy
    
    Uses a Long Short-Term Memory (LSTM) neural network to predict future price
    movements and generate trading signals.
    """
    
    def __init__(
        self,
        sequence_length: int = 60,
        prediction_horizon: int = 1,
        lstm_units: int = 50,
        dropout_rate: float = 0.2,
        epochs: int = 50,
        batch_size: int = 32,
        **kwargs
    ):
        """
        Initialize the LSTM Predictor strategy.
        
        Args:
            sequence_length: Number of previous time steps to use for prediction
            prediction_horizon: Number of steps ahead to predict
            lstm_units: Number of LSTM units in the model
            dropout_rate: Dropout rate for regularization
            epochs: Number of training epochs
            batch_size: Batch size for training
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.market_data = MarketDataService()
        self.history = None
    
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
        logger.info(f"Loading data for {symbol} from {start_date} to {end_date}")
        return await self.market_data.get_historical_data(symbol, start_date, end_date)
    
    async def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data for LSTM model.
        
        Args:
            data: Raw market data with OHLCV format
            
        Returns:
            DataFrame with additional features
        """
        # Ensure we have the required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            # Convert column names to lowercase if needed
            data.columns = [col.lower() for col in data.columns]
        
        # Calculate returns and log returns
        data['return'] = data['close'].pct_change()
        data['log_return'] = np.log(data['close'] / data['close'].shift(1))
        
        # Dynamically determine window sizes for small datasets
        min_win = max(1, min(5, len(data)))
        win_20 = max(1, min(20, len(data)))
        # Calculate technical indicators
        # Moving averages
        data['ma_5'] = data['close'].rolling(window=min_win).mean()
        data['ma_20'] = data['close'].rolling(window=win_20).mean()
        
        # Relative Strength Index (RSI)
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        win_14 = max(1, min(14, len(data)))
        avg_gain = gain.rolling(window=win_14).mean()
        avg_loss = loss.rolling(window=win_14).mean()
        rs = avg_gain / avg_loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        win_12 = max(1, min(12, len(data)))
        win_26 = max(1, min(26, len(data)))
        data['ema_12'] = data['close'].ewm(span=win_12).mean()
        data['ema_26'] = data['close'].ewm(span=win_26).mean()
        data['macd'] = data['ema_12'] - data['ema_26']
        win_9 = max(1, min(9, len(data)))
        data['macd_signal'] = data['macd'].ewm(span=win_9).mean()
        
        # Bollinger Bands
        data['std_20'] = data['close'].rolling(window=win_20).std()
        data['upper_band'] = data['ma_20'] + (data['std_20'] * 2)
        data['lower_band'] = data['ma_20'] - (data['std_20'] * 2)
        data['bb_width'] = (data['upper_band'] - data['lower_band']) / data['ma_20']
        
        # Momentum
        data['momentum'] = data['close'].diff(periods=min_win)
        
        # Volatility
        data['volatility'] = data['close'].rolling(window=win_20).std() / data['ma_20']
        
        # Target variable: future return (shifted price for prediction)
        data['target'] = data['close'].shift(-self.prediction_horizon)
        data['future_return'] = data['target'] / data['close'] - 1
        
        # Create trading signal based on future return
        data['signal'] = 0
        data.loc[data['future_return'] > 0.005, 'signal'] = 1  # Buy threshold
        data.loc[data['future_return'] < -0.005, 'signal'] = -1  # Sell threshold
        
        # Drop NaN values
        data = data.dropna()
        
        # Check if enough rows remain for at least one sequence
        if len(data) < self.sequence_length + 1:
            logger.warning(f"[LSTM] Not enough rows after preprocessing for sequence_length={self.sequence_length}. Data rows: {len(data)}. Returning error.")
            # Return a special DataFrame with an error marker for the script to pick up
            # Utilisation de .loc pour éviter le SettingWithCopyWarning
            data = data.copy()  # Créer une copie explicite
            data.loc[:, '__lstm_error__'] = f"Not enough data after preprocessing (rows={len(data)}, needed={self.sequence_length+1})"
            return data
        
        return data
    
    def _create_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequence data for LSTM model.
        
        Args:
            data: Preprocessed data
            
        Returns:
            Tuple of (X, y) for training
        """
        # Robustness: Check for sufficient data and required columns
        if len(data) < self.sequence_length + 1 or 'signal' not in data.columns:
            logger.warning(f"[LSTM] Not enough data or missing 'signal' column for sequence creation (rows={len(data)}, sequence_length={self.sequence_length})")
            return np.empty((0, self.sequence_length, 10)), np.empty((0,))
        # Features to use
        features = [
            'close', 'return', 'ma_5', 'ma_20', 'rsi', 
            'macd', 'macd_signal', 'bb_width', 'momentum', 'volatility'
        ]
        # Scale the features
        feature_data = data[features].values
        scaled_data = self.scaler.fit_transform(feature_data)
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length):
            X.append(scaled_data[i:(i + self.sequence_length)])
            # Target is the signal value (classification)
            y.append(data['signal'].iloc[i + self.sequence_length])
        return np.array(X), np.array(y)

    
    def _build_model(self, input_shape: Tuple) -> Sequential:
        """
        Build the LSTM model.
        
        Args:
            input_shape: Shape of input data
            
        Returns:
            Compiled Keras model
        """
        model = Sequential()
        
        # First LSTM layer with return sequences
        model.add(LSTM(
            units=self.lstm_units,
            return_sequences=True,
            input_shape=input_shape
        ))
        model.add(Dropout(self.dropout_rate))
        
        # Second LSTM layer
        model.add(LSTM(units=self.lstm_units))
        model.add(Dropout(self.dropout_rate))
        
        # Dense layer
        model.add(Dense(units=20, activation='relu'))
        
        # Output layer - 3 classes for buy, sell, hold
        model.add(Dense(units=3, activation='softmax'))
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    async def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the LSTM model on historical data.
        
        Args:
            data: Preprocessed market data
            
        Returns:
            Dictionary containing training metrics
        """
        # Prepare sequences
        X, y = self._create_sequences(data)
        
        # Adjust target values for classification (0, 1, 2 instead of -1, 0, 1)
        y = np.where(y == -1, 2, y)
        
        # Split data into train and test sets
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Build the model
        if len(X_train) == 0:
            raise ValueError("Not enough data after preprocessing to create training sequences for LSTM. Got 0 samples. Check your sequence_length and input data size.")
        self.model = self._build_model(X_train[0].shape)
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train the model
        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Evaluate model
        loss, accuracy = self.model.evaluate(X_test, y_test)
        
        self.is_trained = True
        
        # Prepare metrics
        train_metrics = self.history.history
        train_metrics['final_loss'] = loss
        train_metrics['final_accuracy'] = accuracy
        
        return train_metrics
    
    async def predict(self, data: pd.DataFrame) -> Tuple[TradeAction, float]:
        """
        Generate a trading signal based on the input data.
        
        Args:
            data: Market data with calculated indicators
            
        Returns:
            Tuple of (TradeAction, confidence)
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model is not trained yet")
        
        # Preprocess data if needed
        if 'ma_5' not in data.columns:
            data = await self.preprocess_data(data)
        
        # Create a sequence from the last sequence_length data points
        features = [
            'close', 'return', 'ma_5', 'ma_20', 'rsi', 
            'macd', 'macd_signal', 'bb_width', 'momentum', 'volatility'
        ]
        
        # Scale the features
        feature_data = data[features].tail(self.sequence_length).values
        
        # Check if we have enough data
        if len(feature_data) < self.sequence_length:
            logger.warning(f"Not enough data points ({len(feature_data)}) for prediction, need {self.sequence_length}")
            return TradeAction.HOLD, 0.5
        
        # Scale the data
        scaled_data = self.scaler.transform(feature_data)
        
        # Create the sequence
        X = np.array([scaled_data])
        
        # Make prediction
        prediction_probs = self.model.predict(X)[0]
        prediction_class = np.argmax(prediction_probs)
        confidence = prediction_probs[prediction_class]
        
        # Convert back from classification labels (0, 1, 2) to trade actions
        if prediction_class == 0:  # Hold
            return TradeAction.HOLD, confidence
        elif prediction_class == 1:  # Buy
            return TradeAction.BUY, confidence
        else:  # Sell
            return TradeAction.SELL, confidence
    
    async def backtest(self, data: pd.DataFrame, initial_capital: float = 10000.0) -> Dict[str, Any]:
        """
        Backtest the strategy on historical data.
        
        Args:
            data: Historical market data
            initial_capital: Initial capital for the backtest
            
        Returns:
            Dictionary containing backtest results
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model is not trained yet")
        
        # Preprocess data if needed
        if 'ma_5' not in data.columns:
            data = await self.preprocess_data(data)
        
        # Create sequences for the entire dataset
        X, _ = self._create_sequences(data)
        
        # Skip the first sequence_length data points as they were used for the first prediction
        backtest_data = data.iloc[self.sequence_length:].copy()
        
        # Make predictions for each sequence
        predictions = self.model.predict(X)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Convert predicted classes back to signals (-1, 0, 1)
        signals = np.where(predicted_classes == 2, -1, predicted_classes)
        
        # Add signals to backtest data
        backtest_data['signal'] = signals
        
        # Add position column (1: long, 0: out of market, -1: short)
        backtest_data['position'] = backtest_data['signal']
        
        # Calculate returns
        backtest_data['returns'] = backtest_data['close'].pct_change()
        
        # Calculate strategy returns (position entered at close of signal day)
        backtest_data['strategy_returns'] = backtest_data['position'].shift(1) * backtest_data['returns']
        
        # Calculate cumulative returns
        backtest_data['cumulative_returns'] = (1 + backtest_data['returns']).cumprod()
        backtest_data['cumulative_strategy_returns'] = (1 + backtest_data['strategy_returns']).cumprod()
        
        # Calculate drawdown
        backtest_data['peak'] = backtest_data['cumulative_strategy_returns'].cummax()
        backtest_data['drawdown'] = (backtest_data['cumulative_strategy_returns'] - backtest_data['peak']) / backtest_data['peak']
        
        # Calculate metrics
        total_return = backtest_data['cumulative_strategy_returns'].iloc[-1] - 1
        max_drawdown = backtest_data['drawdown'].min()
        
        # Calculate Sharpe ratio (assuming 252 trading days per year and risk-free rate of 0)
        sharpe_ratio = np.sqrt(252) * backtest_data['strategy_returns'].mean() / backtest_data['strategy_returns'].std()
        
        # Calculate final capital
        final_capital = initial_capital * (1 + total_return)
        
        # Count trades
        trades = (backtest_data['position'].diff() != 0).sum()
        
        return {
            "initial_capital": initial_capital,
            "final_capital": final_capital,
            "total_return": total_return,
            "annualized_return": total_return / (len(backtest_data) / 252),
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "trades": trades,
            "backtest_data": backtest_data[['close', 'signal', 'position', 'returns', 'strategy_returns', 'cumulative_returns', 'cumulative_strategy_returns']]
        }
    
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
        
        os.makedirs(path, exist_ok=True)
        
        # Save Keras model using the new recommended format
        model_path = os.path.join(path, f"lstm_predictor_{self.sequence_length}_{self.lstm_units}.keras")
        self.model.save(model_path)
        
        # Save scaler and parameters
        scaler_path = os.path.join(path, f"lstm_predictor_{self.sequence_length}_{self.lstm_units}_scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'sequence_length': self.sequence_length,
                'prediction_horizon': self.prediction_horizon,
                'lstm_units': self.lstm_units,
                'params': self.params
            }, f)
        
        return model_path
    
    async def load_model(self, path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            path: Path to the saved model
        """
        # Load Keras model
        try:
            self.model = load_model(path)
            logger.info(f"Modèle chargé depuis {path}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle depuis {path}: {e}")
            raise
        
        # Déterminer le chemin du fichier scaler en fonction de l'extension du modèle
        if path.endswith('.h5'):
            scaler_path = path.replace('.h5', '_scaler.pkl')
        elif path.endswith('.keras'):
            scaler_path = path.replace('.keras', '_scaler.pkl')
        else:
            # Essayons de deviner en supprimant l'extension
            base_path = os.path.splitext(path)[0]
            scaler_path = f"{base_path}_scaler.pkl"
        
        if not os.path.exists(scaler_path):
            logger.warning(f"Fichier scaler introuvable: {scaler_path}. Tentative avec le nom de base.")
            base_path = os.path.join(os.path.dirname(path), os.path.basename(path).split('_')[0])
            scaler_path = f"{base_path}_scaler.pkl"
        
        logger.info(f"Chargement du scaler depuis {scaler_path}")
        try:
            with open(scaler_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.scaler = model_data['scaler']
            self.sequence_length = model_data['sequence_length']
            self.prediction_horizon = model_data['prediction_horizon']
            self.lstm_units = model_data['lstm_units']
            self.params.update(model_data['params'])
            
            self.is_trained = True
            logger.info(f"Paramètres chargés: seq_len={self.sequence_length}, lstm_units={self.lstm_units}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement des paramètres depuis {scaler_path}: {e}")
            raise

# Alias for compatibility with tests
LSTMPredictor = LSTMPredictorStrategy
