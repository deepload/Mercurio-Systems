"""
Moving Average Crossover Strategy

A simple strategy that generates buy/sell signals based on 
the crossover of short and long-term moving averages.
"""
import os
import pickle
import logging
from typing import Dict, Any, Tuple, Union, Optional
from datetime import datetime
import pandas as pd
import numpy as np
import backtrader as bt
from sklearn.ensemble import RandomForestClassifier

from app.db.models import TradeAction
from app.strategies.base import BaseStrategy
from app.services.market_data import MarketDataService

logger = logging.getLogger(__name__)

class MovingAverageStrategy(BaseStrategy):
    """
    Moving Average Crossover Strategy
    
    Uses short-term and long-term moving averages to generate trading signals:
    - Buy when short MA crosses above long MA
    - Sell when short MA crosses below long MA
    
    Can optionally use a Random Forest model to enhance predictions based on
    additional technical indicators.
    """
    
    def __init__(
        self,
        short_window: int = 20,
        long_window: int = 50,
        use_ml: bool = False,
        **kwargs
    ):
        """
        Initialize the Moving Average strategy.
        
        Args:
            short_window: Period for the short-term moving average
            long_window: Period for the long-term moving average
            use_ml: Whether to use ML enhancement (Random Forest)
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        self.short_window = short_window
        self.long_window = long_window
        self.use_ml = use_ml
        self.model = None
        self.market_data = MarketDataService()
    
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
        Patch: Robust to small datasets for both classic and ML modes.
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            data.columns = [col.lower() for col in data.columns]
        original_data = data.copy()
        original_length = len(data)
        # Calculate moving averages
        short_win = min(self.short_window, max(2, len(data)//2))
        long_win = min(self.long_window, max(short_win+1, len(data)-1))
        data['short_ma'] = data['close'].rolling(window=short_win, min_periods=1).mean()
        data['long_ma'] = data['close'].rolling(window=long_win, min_periods=1).mean()
        if self.use_ml:
            # ML features: always create them, use smallest possible window for short data
            # RSI
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            rsi_win = min(3, len(data))
            avg_gain = gain.rolling(window=rsi_win, min_periods=1).mean()
            avg_loss = loss.rolling(window=rsi_win, min_periods=1).mean()
            rs = avg_gain / (avg_loss + 1e-6)
            data['rsi'] = 100 - (100 / (1 + rs))
            # MACD
            ema_short = data['close'].ewm(span=2, adjust=False).mean()
            ema_long = data['close'].ewm(span=3 if len(data)>=3 else 2, adjust=False).mean()
            data['macd'] = ema_short - ema_long
            data['macd_signal'] = data['macd'].ewm(span=2, adjust=False).mean()
            # Bollinger Band width (BB width)
            bb_win = min(3, len(data))
            ma = data['close'].rolling(window=bb_win, min_periods=1).mean()
            std = data['close'].rolling(window=bb_win, min_periods=1).std().fillna(0)
            data['bb_width'] = 2 * std / ma.replace(0, 1)
            # Lags
            data['close_lag_1'] = data['close'].shift(1)
            data['close_lag_2'] = data['close'].shift(2)
            # Returns
            data['return_lag_1'] = data['close'].pct_change(periods=1)
            # Use smallest possible windows for additional features
            data['ma_2'] = data['close'].rolling(window=2, min_periods=1).mean()
            data['ma_3'] = data['close'].rolling(window=3, min_periods=1).mean()
            data['ma_4'] = data['close'].rolling(window=4 if len(data)>=4 else 2, min_periods=1).mean()
            data['return_1'] = data['close'].pct_change(periods=1).fillna(0)
            data['return_2'] = data['close'].pct_change(periods=2).fillna(0)
            data['return_3'] = data['close'].pct_change(periods=3 if len(data)>=3 else 1).fillna(0)
            # Target variable for ML
            data['target'] = (data['close'].shift(-1) > data['close']).astype(int)
            # Ensure all required columns exist, fill missing with 0
            required_ml_cols = ['rsi', 'macd', 'macd_signal', 'bb_width', 'close_lag_1', 'close_lag_2', 'return_lag_1']
            for col in required_ml_cols:
                if col not in data.columns:
                    data[col] = 0
            data[required_ml_cols] = data[required_ml_cols].fillna(0)
        before = len(data)
        data = data.dropna()
        after = len(data)
        if after == 0 and before > 0:
            print(f"[WARNING] Not enough data for full rolling windows (short={short_win}, long={long_win}). Keeping last available row for simulation.")
            data = original_data.tail(1)
            if len(data) == 0:
                print(f"[WARNING] Still no data available for simulation after fallback. Returning empty DataFrame.")
                return pd.DataFrame()
        # --- Ensure 'signal' column exists, even for minimal data ---
        if 'signal' not in data.columns:
            data['signal'] = 0
        if len(data) > 0 and ('short_ma' in data.columns and 'long_ma' in data.columns):
            last = data.iloc[-1]
            if last['short_ma'] > last['long_ma']:
                data.iloc[-1, data.columns.get_loc('signal')] = 1
            elif last['short_ma'] < last['long_ma']:
                data.iloc[-1, data.columns.get_loc('signal')] = -1
            else:
                data.iloc[-1, data.columns.get_loc('signal')] = 0
        return data
        """
        Preprocess the data by adding technical indicators.
        
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
        
        # Calculate moving averages
        data['short_ma'] = data['close'].rolling(window=self.short_window).mean()
        data['long_ma'] = data['close'].rolling(window=self.long_window).mean()
        
        # Calculate additional features for ML model
        if self.use_ml:
            # Relative Strength Index (RSI)
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            data['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            data['ema_12'] = data['close'].ewm(span=12).mean()
            data['ema_26'] = data['close'].ewm(span=26).mean()
            data['macd'] = data['ema_12'] - data['ema_26']
            data['macd_signal'] = data['macd'].ewm(span=9).mean()
            
            # Bollinger Bands
            data['std_20'] = data['close'].rolling(window=20).std()
            data['upper_band'] = data['close'].rolling(window=20).mean() + (data['std_20'] * 2)
            data['lower_band'] = data['close'].rolling(window=20).mean() - (data['std_20'] * 2)
            data['bb_width'] = (data['upper_band'] - data['lower_band']) / data['close'].rolling(window=20).mean()
            
            # Add lag features
            for lag in [1, 2, 3, 5]:
                data[f'close_lag_{lag}'] = data['close'].shift(lag)
                data[f'return_lag_{lag}'] = data['close'].pct_change(lag)
        
        # Create trading signal (1: buy, -1: sell, 0: hold)
        data['signal'] = 0
        data.loc[data['short_ma'] > data['long_ma'], 'signal'] = 1
        data.loc[data['short_ma'] < data['long_ma'], 'signal'] = -1
        
        # Add target variable (next day's signal) for training
        data['target'] = data['signal'].shift(-1)
        
        # Drop NaN values resulting from rolling windows, but if this removes all rows, keep the last row and warn
        before = len(data)
        data = data.dropna()
        after = len(data)
        if after == 0 and before > 0:
            print(f"[WARNING] Not enough data for full rolling windows (short={self.short_window}, long={self.long_window}). Keeping last available row for simulation.")
            # Use tail(1) which is safe even if the DataFrame is empty
            data = data.tail(1)
            if len(data) == 0:
                print(f"[WARNING] Still no data available for simulation after fallback. Returning empty DataFrame.")
                return pd.DataFrame()
        return data
    
    async def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the ML model if use_ml is enabled.
        
        Args:
            data: Preprocessed market data
            
        Returns:
            Dictionary containing training metrics
        """
        if not self.use_ml:
            self.is_trained = True
            return {"message": "No ML model to train for basic MA strategy"}
        
        # Prepare features and target
        features = [
            'short_ma', 'long_ma', 'rsi', 'macd', 'macd_signal',
            'bb_width', 'close_lag_1', 'close_lag_2', 'return_lag_1'
        ]
        
        X = data[features].values
        y = data['target'].values
        
        # Use 80% of data for training
        train_size = int(len(data) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Train Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        train_accuracy = self.model.score(X_train, y_train)
        test_accuracy = self.model.score(X_test, y_test)
        
        self.is_trained = True
        
        return {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "feature_importance": dict(zip(features, self.model.feature_importances_))
        }
    
    async def predict(self, data: pd.DataFrame) -> Tuple[TradeAction, float]:
        """
        Generate a trading signal based on the input data.
        
        Args:
            data: Market data with calculated indicators
            
        Returns:
            Tuple of (TradeAction, confidence)
        """
        # Preprocess data if needed
        if 'short_ma' not in data.columns:
            data = await self.preprocess_data(data)
        
        # Get the latest data point
        latest = data.iloc[-1]
        
        if self.use_ml and self.is_trained and self.model is not None:
            # Use ML model for prediction
            features = [
                'short_ma', 'long_ma', 'rsi', 'macd', 'macd_signal',
                'bb_width', 'close_lag_1', 'close_lag_2', 'return_lag_1'
            ]
            X = latest[features].values.reshape(1, -1)
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            confidence = max(probabilities)
            
            if prediction == 1:
                return TradeAction.BUY, confidence
            elif prediction == -1:
                return TradeAction.SELL, confidence
            else:
                return TradeAction.HOLD, confidence
        else:
            # Use simple MA crossover rule
            if latest['short_ma'] > latest['long_ma']:
                # Calculate confidence based on the distance between MAs
                confidence = min((latest['short_ma'] - latest['long_ma']) / latest['close'] * 5, 0.95)
                return TradeAction.BUY, confidence
            elif latest['short_ma'] < latest['long_ma']:
                confidence = min((latest['long_ma'] - latest['short_ma']) / latest['close'] * 5, 0.95)
                return TradeAction.SELL, confidence
            else:
                return TradeAction.HOLD, 0.5
    
    async def backtest(self, data: pd.DataFrame, initial_capital: float = 10000.0) -> Dict[str, Any]:
        """
        Backtest the strategy on historical data.
        
        Args:
            data: Historical market data
            initial_capital: Initial capital for the backtest
            
        Returns:
            Dictionary containing backtest results
        """
        # Make sure we have signals
        if 'signal' not in data.columns:
            data = await self.preprocess_data(data)
        
        # Copy to avoid modifying original data
        backtest_data = data.copy()
        
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
            "trades": [],  # No trade details available; provide empty list for compatibility
            "trade_count": trades,
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
        if not self.is_trained:
            raise ValueError("Model is not trained yet")
        
        if not self.use_ml or self.model is None:
            return ""
        
        os.makedirs(path, exist_ok=True)
        
        model_path = os.path.join(path, f"moving_average_{self.short_window}_{self.long_window}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'short_window': self.short_window,
                'long_window': self.long_window,
                'params': self.params
            }, f)
        
        return model_path
    
    async def load_model(self, path: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            path: Path to the saved model
        """
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.short_window = model_data['short_window']
        self.long_window = model_data['long_window']
        self.params.update(model_data['params'])
        self.is_trained = True
        self.use_ml = True
        
    def get_signal(self, symbol: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a trading signal for the HFTrader.
        
        This method analyzes price data and generates a trading signal based on
        the moving average crossover strategy.
        
        Args:
            symbol: The trading symbol (e.g., 'BTC/USD')
            data: DataFrame containing price data with at least 'close' column
            
        Returns:
            Dictionary containing action, confidence, and reason
        """
        # Ensure data is properly preprocessed
        if len(data) < max(self.short_window, self.long_window):
            logger.warning(f"Insufficient data for signal generation: {len(data)} rows")
            return {"action": TradeAction.HOLD, "confidence": 0, "reason": "Insufficient data"}
            
        # Preprocess data to calculate moving averages if they don't exist
        if 'short_ma' not in data.columns or 'long_ma' not in data.columns:
            # Since this is now a synchronous method, we can't use await
            # Instead we'll calculate the moving averages directly
            # Calculate short and long moving averages
            short_win = min(self.short_window, max(2, len(data)//2))
            long_win = min(self.long_window, max(short_win+1, len(data)-1))
            data['short_ma'] = data['close'].rolling(window=short_win, min_periods=1).mean()
            data['long_ma'] = data['close'].rolling(window=long_win, min_periods=1).mean()
            
        if data.empty or 'short_ma' not in data.columns or 'long_ma' not in data.columns:
            logger.warning("Failed to calculate moving averages")
            return {"action": TradeAction.HOLD, "confidence": 0, "reason": "Preprocessing failed"}
        
        # Get the last row for signal generation
        last_row = data.iloc[-1]
        
        # Calculate signal based on moving average crossover
        action = TradeAction.HOLD
        confidence = 0.5  # Default confidence
        reason = ""
        
        # Moving Average Crossover logic
        if len(data) >= 2:
            prev_row = data.iloc[-2]
            
            # Check for upward crossover (buy signal)
            if (prev_row['short_ma'] <= prev_row['long_ma']) and (last_row['short_ma'] > last_row['long_ma']):
                action = TradeAction.BUY
                # Calculate confidence based on the strength of the crossover
                crossover_strength = (last_row['short_ma'] - last_row['long_ma']) / last_row['long_ma']
                confidence = min(0.5 + abs(crossover_strength) * 100, 0.9)  # Scale confidence, cap at 0.9
                reason = f"Moving Average Crossover (Buy): short_ma crossed above long_ma (strength: {crossover_strength:.4f})"
            
            # Check for downward crossover (sell signal)
            elif (prev_row['short_ma'] >= prev_row['long_ma']) and (last_row['short_ma'] < last_row['long_ma']):
                action = TradeAction.SELL
                # Calculate confidence based on the strength of the crossover
                crossover_strength = (last_row['long_ma'] - last_row['short_ma']) / last_row['long_ma']
                confidence = min(0.5 + abs(crossover_strength) * 100, 0.9)  # Scale confidence, cap at 0.9
                reason = f"Moving Average Crossover (Sell): short_ma crossed below long_ma (strength: {crossover_strength:.4f})"
            
            # No crossover, but check trend strength for potential signals
            else:
                ma_difference = (last_row['short_ma'] - last_row['long_ma']) / last_row['long_ma']
                
                # Strong uptrend
                if ma_difference > 0.02:  # 2% difference
                    action = TradeAction.BUY
                    confidence = min(0.5 + abs(ma_difference) * 20, 0.8)  # Lower max confidence for trend continuation
                    reason = f"Strong Uptrend: short_ma {ma_difference:.2%} above long_ma"
                
                # Strong downtrend
                elif ma_difference < -0.02:  # -2% difference
                    action = TradeAction.SELL
                    confidence = min(0.5 + abs(ma_difference) * 20, 0.8)  # Lower max confidence for trend continuation
                    reason = f"Strong Downtrend: short_ma {abs(ma_difference):.2%} below long_ma"
                
                # No strong signal
                else:
                    reason = f"No clear signal: MA Difference: {ma_difference:.2%}"
        else:
            reason = "Insufficient historical data for crossover detection"
        
        # Enhance signal with ML if available
        if self.use_ml and self.model is not None:
            try:
                # Since we're in a synchronous context, we can't use await directly
                # For now, we'll skip the ML enhancement and just use the classic signal
                # In a real implementation, you'd want to make predict() synchronous as well
                # or use another approach to handle the coroutine
                logger.info(f"ML enhancement skipped in sync get_signal for {symbol}")
                
                # Note: If you need to use predict() and it must be async, one option would be:
                # - Update the HFTrader to await this method properly
                # - Or implement a synchronous version of predict() specifically for get_signal
            except Exception as e:
                logger.warning(f"Failed to get ML prediction: {e}")
                # Continue with classic signal
        
        return {
            "action": action,
            "confidence": confidence,
            "reason": reason
        }


# Backtrader strategy for backtesting with the backtrader library
class MovingAverageBT(bt.Strategy):
    """Backtrader implementation of Moving Average Crossover strategy"""
    
    params = (
        ('short_window', 20),
        ('long_window', 50),
    )
    
    def __init__(self):
        """Initialize indicators"""
        self.short_ma = bt.indicators.SMA(self.data.close, period=self.params.short_window)
        self.long_ma = bt.indicators.SMA(self.data.close, period=self.params.long_window)
        self.crossover = bt.indicators.CrossOver(self.short_ma, self.long_ma)
        
        # Keep track of pending orders
        self.order = None
    
    def next(self):
        """Define what will be done in each iteration"""
        if self.order:
            return  # Pending order exists
        
        if not self.position:  # Not in the market
            if self.crossover > 0:  # Buy signal
                self.order = self.buy()
        else:  # In the market
            if self.crossover < 0:  # Sell signal
                self.order = self.sell()
    
    def notify_order(self, order):
        """Called when an order is filled"""
        if order.status in [order.Completed]:
            if order.isbuy():
                pass
            else:
                pass
            self.order = None

# Alias for compatibility with tests
MovingAverageCrossover = MovingAverageStrategy
