"""
Mercurio AI - Strategy Timeframe Comparison

This script runs all available trading strategies for both day trading (2 days) and week trading (10 days),
then compares results side-by-side for each symbol and strategy.
"""
import os
import asyncio
import pandas as pd
from datetime import datetime, timedelta
import logging
from tabulate import tabulate
import traceback

from data_generator import generate_all_market_data, load_market_data

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure reports directory exists
os.makedirs('reports', exist_ok=True)
os.makedirs('data', exist_ok=True)

class TimeframeStrategySimulator:
    def __init__(self, timeframe_name, days, initial_capital=2000):
        self.timeframe_name = timeframe_name
        self.days = days
        self.initial_capital = initial_capital
        self.results = []
        self.strategies = {}
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=self.days)
        self.stocks = ['AAPL', 'MSFT', 'GOOGL']
        self.cryptos = ['BTC-USD', 'ETH-USD']
        self.all_symbols = self.stocks + self.cryptos

    def initialize_strategies(self):
        # Dynamically adjust parameters for timeframe
        if self.days <= 2:
            ma_short, ma_long = 1, 2
            lstm_seq = 1
        else:
            ma_short, ma_long = 2, 3
            lstm_seq = 2
        try:
            from app.strategies.moving_average import MovingAverageStrategy
            self.strategies["MovingAverage"] = MovingAverageStrategy(
                short_window=ma_short, long_window=ma_long
            )
        except Exception as e:
            logger.error(f"Failed to initialize MovingAverage: {e}")
        try:
            from app.strategies.moving_average import MovingAverageStrategy
            self.strategies["MovingAverage_ML"] = MovingAverageStrategy(
                short_window=ma_short, long_window=ma_long, use_ml=True
            )
        except Exception as e:
            logger.error(f"Failed to initialize MovingAverage_ML: {e}")
        try:
            from app.strategies.lstm_predictor import LSTMPredictorStrategy
            self.strategies["LSTM"] = LSTMPredictorStrategy(
                sequence_length=lstm_seq, prediction_horizon=1, epochs=20, batch_size=4
            )
        except Exception as e:
            logger.error(f"Failed to initialize LSTM: {e}")
        try:
            from app.strategies.llm_strategy import LLMStrategy
            self.strategies["LLM"] = LLMStrategy()
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
        try:
            from app.strategies.transformer_strategy import TransformerStrategy
            self.strategies["Transformer"] = TransformerStrategy(sequence_length=lstm_seq)
        except Exception as e:
            logger.error(f"Failed to initialize Transformer: {e}")

    def generate_market_data(self):
        logger.info(f"Generating market data for {len(self.all_symbols)} symbols for {self.timeframe_name}...")
        generate_all_market_data(self.all_symbols, self.start_date, self.end_date, 'data')

    async def get_data_for_strategy(self, symbol):
        data = load_market_data(symbol)
        if data is None or data.empty:
            logger.warning(f"No data available for {symbol}")
            return None
        if 'timestamp' not in data.columns and isinstance(data.index, pd.DatetimeIndex):
            data['timestamp'] = range(len(data))
        if isinstance(data.index, pd.DatetimeIndex):
            data = data.reset_index()
        data.columns = [col.lower() for col in data.columns]
        if 'date' in data.columns and pd.api.types.is_datetime64_any_dtype(data['date']):
            data['date_str'] = data['date'].dt.strftime('%Y-%m-%d')
        data = data.ffill().bfill()
        return data

    async def run(self):
        self.initialize_strategies()
        self.generate_market_data()
        for symbol in self.all_symbols:
            data = await self.get_data_for_strategy(symbol)
            if data is None or data.empty:
                for strat_name in self.strategies:
                    self.results.append({
                        'symbol': symbol,
                        'strategy': strat_name,
                        'timeframe': self.timeframe_name,
                        'error': 'No data available'
                    })
                continue
            for strat_name, strat in self.strategies.items():
                try:
                    # Preprocess
                    preprocessed = await strat.preprocess_data(data.copy())
                    if preprocessed is None or preprocessed.empty:
                        # Try fallback: if original data has at least 1 row, use it as a dummy row
                        if len(data) > 0:
                            warn = f"[WARNING] {strat_name} could not compute features for {symbol} ({self.timeframe_name}). Using last available row."
                            logger.warning(warn)
                            dummy = data.tail(1)
                            initial = dummy['close'].iloc[0]
                            final = dummy['close'].iloc[0]
                            self.results.append({
                                'symbol': symbol,
                                'strategy': strat_name,
                                'timeframe': self.timeframe_name,
                                'initial_close': initial,
                                'final_close': final,
                                'total_return_%': 0.0,
                                'error': warn
                            })
                        else:
                            self.results.append({
                                'symbol': symbol,
                                'strategy': strat_name,
                                'timeframe': self.timeframe_name,
                                'error': 'Preprocessing resulted in empty dataset',
                                'initial_close': None,
                                'final_close': None,
                                'total_return_%': None
                            })
                        continue
                    # Train if needed
                    if hasattr(strat, 'train'):
                        await strat.train(preprocessed)
                    # Simulate (very basic: just compute total return)
                    initial = preprocessed['close'].iloc[0]
                    final = preprocessed['close'].iloc[-1]
                    total_return = (final - initial) / initial * 100 if initial != 0 else 0.0
                    self.results.append({
                        'symbol': symbol,
                        'strategy': strat_name,
                        'timeframe': self.timeframe_name,
                        'initial_close': initial,
                        'final_close': final,
                        'total_return_%': total_return,
                        'error': None
                    })
                except Exception as e:
                    self.results.append({
                        'symbol': symbol,
                        'strategy': strat_name,
                        'timeframe': self.timeframe_name,
                        'error': f'Exception: {e}',
                        'initial_close': None,
                        'final_close': None,
                        'total_return_%': None
                    })
                    traceback.print_exc()
        return self.results

async def main():
    print("\n===== MERCURIO AI STRATEGY TIMEFRAME COMPARISON =====\n")
    simulators = [
        TimeframeStrategySimulator('Day', days=2),
        TimeframeStrategySimulator('Week', days=10)
    ]
    all_results = []
    for sim in simulators:
        print(f"\nRunning strategies for {sim.timeframe_name} trading...")
        results = await sim.run()
        all_results.extend(results)
    df = pd.DataFrame(all_results)
    df.to_csv('reports/strategy_timeframe_comparison.csv', index=False)
    print("\n===== TIMEFRAME COMPARISON RESULTS =====\n")
    print(tabulate(df, headers='keys', tablefmt='psql'))
    print("\nResults saved to reports/strategy_timeframe_comparison.csv\n")

if __name__ == "__main__":
    asyncio.run(main())
