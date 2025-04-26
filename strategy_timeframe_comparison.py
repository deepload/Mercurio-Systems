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
        self.stocks = [
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "GOOGL",  # Alphabet (Google)
    "AMZN",   # Amazon
    "NVDA",   # NVIDIA
    "META",   # Meta Platforms (Facebook)
    "TSLA",   # Tesla
    "BRK.B",  # Berkshire Hathaway
    "UNH",    # UnitedHealth Group
    "V",      # Visa
    "JPM",    # JPMorgan Chase
    "MA",     # Mastercard
    "XOM",    # ExxonMobil
    "LLY",    # Eli Lilly
    "JNJ",    # Johnson & Johnson
    "PG",     # Procter & Gamble
    "AVGO",   # Broadcom
    "HD",     # Home Depot
    "MRK",    # Merck & Co
    "ABBV",   # AbbVie
    "PEP",    # PepsiCo
    "COST",   # Costco
    "KO",     # Coca-Cola
    "NFLX",   # Netflix
    "ADBE",   # Adobe
    "PFE",    # Pfizer
    "CRM",    # Salesforce
    "WMT",    # Walmart
    "BAC",    # Bank of America
    "AMD",    # Advanced Micro Devices
]
        self.cryptos = [
    "BTC-USD",
    "ETH-USD",
    "LTC-USD",
    "BCH-USD",
    "DOGE-USD",
    "LINK-USD",
    "UNI-USD",
    "AAVE-USD",
    "AVAX-USD",
    "BAT-USD",
    "CRV-USD",
    "DOT-USD",
    "GRT-USD",
    "MKR-USD",
    "PEPE-USD",
    "SHIB-USD",
    "SOL-USD",
    "SUSHI-USD",
    "TRUMP-USD",
    "USDC-USD",
    "USDT-USD",
    "XRP-USD",
    "XTZ-USD",
    "YFI-USD",
]
        self.all_symbols = self.stocks + self.cryptos

    def initialize_strategies(self):
        # Dynamically adjust parameters for timeframe
        if self.days <= 2:
            ma_short, ma_long = 1, 2
            # LSTM is not meaningful for extremely short timeframes; set minimum sensible sequence length
            lstm_seq = 10
            logger.warning(f"[LSTM] Timeframe '{self.timeframe_name}' is too short for LSTM to be meaningful (days={self.days}). Using minimum sequence_length=10.")
        else:
            ma_short, ma_long = 2, 3
            # Use up to 30, but not less than 10, and not more than days-1
            lstm_seq = max(10, min(self.days - 1, 30))
            if self.days - 1 < 10:
                logger.warning(f"[LSTM] Timeframe '{self.timeframe_name}' has only {self.days} days. LSTM sequence_length set to {lstm_seq} (minimum is 10). Results may not be reliable.")
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
        try:
            from app.strategies.msi_strategy import MultiSourceIntelligenceStrategy
            self.strategies["MSI"] = MultiSourceIntelligenceStrategy()
        except Exception as e:
            logger.error(f"Failed to initialize MSI: {e}")

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
                    # Special handling for LSTM: if __lstm_error__ column exists, propagate error and skip
                    if isinstance(preprocessed, pd.DataFrame) and '__lstm_error__' in preprocessed.columns:
                        if preprocessed.shape[0] == 0:
                            error_msg = "LSTM preprocessing failed: empty DataFrame returned (not enough data for sequence_length)."
                        else:
                            error_msg = preprocessed['__lstm_error__'].iloc[0]
                        logger.warning(f"[LSTM] {strat_name} for {symbol} ({self.timeframe_name}): {error_msg}")
                        self.results.append({
                            'symbol': symbol,
                            'strategy': strat_name,
                            'timeframe': self.timeframe_name,
                            'initial_close': None,
                            'final_close': None,
                            'total_return_%': None,
                            'error': error_msg
                        })
                        continue
                    if preprocessed is None or preprocessed.empty or len(preprocessed) == 0:
                        # Try fallback: if original data has at least 1 row, use it as a dummy row
                        if len(data) > 0:
                            warn = f"[WARNING] {strat_name} could not compute features for {symbol} ({self.timeframe_name}). Using last available row."
                            logger.warning(warn)
                            dummy = data.tail(1)
                            if len(dummy) > 0:
                                initial = dummy['close'].iloc[0]
                                final = dummy['close'].iloc[0]
                            else:
                                initial = None
                                final = None
                            self.results.append({
                                'symbol': symbol,
                                'strategy': strat_name,
                                'timeframe': self.timeframe_name,
                                'initial_close': initial,
                                'final_close': final,
                                'total_return_%': 0.0 if initial is not None else None,
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
                    if len(preprocessed) > 0 and 'close' in preprocessed.columns:
                        try:
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
                            logger.warning(f"[{strat_name}] Could not access initial/final price for {symbol} ({self.timeframe_name}): {e}")
                            self.results.append({
                                'symbol': symbol,
                                'strategy': strat_name,
                                'timeframe': self.timeframe_name,
                                'initial_close': None,
                                'final_close': None,
                                'total_return_%': None,
                                'error': f'Exception: {e}'
                            })
                    else:
                        logger.warning(f"[{strat_name}] No valid data for initial/final price for {symbol} ({self.timeframe_name}) (len={len(preprocessed)}, columns={preprocessed.columns.tolist()})")
                        self.results.append({
                            'symbol': symbol,
                            'strategy': strat_name,
                            'timeframe': self.timeframe_name,
                            'initial_close': None,
                            'final_close': None,
                            'total_return_%': None,
                            'error': 'No valid data for initial/final price'
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
        TimeframeStrategySimulator('Day', days=31),
        TimeframeStrategySimulator('Week', days=180)
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

    # FINAL SUMMARY NOTE: Top 3 strategies and total money won
    try:
        # Filter out results with missing or error values
        df_valid = df[df['error'].isnull() & df['total_return_%'].notnull() & df['initial_close'].notnull()]
        # Sort by total_return_% descending
        top3 = df_valid.sort_values('total_return_%', ascending=False).head(3)
        print("\n===== TOP 3 STRATEGY RESULTS (by total_return_%) =====\n")
        if not top3.empty:
            print(tabulate(top3[['strategy', 'symbol', 'timeframe', 'total_return_%']], headers='keys', tablefmt='psql'))
        else:
            print("No valid results to display.")
        # Calculate total money won (sum of profit for all strategies)
        # Assume initial_capital per strategy per symbol (from simulator)
        initial_capital = 2000
        df_valid = df_valid.copy()
        df_valid['profit'] = df_valid['total_return_%'] * initial_capital / 100
        total_money_won = df_valid['profit'].sum()
        print(f"\n===== TOTAL MONEY WON (across all strategies): ${total_money_won:,.2f} =====\n")
    except Exception as e:
        print(f"\n[SUMMARY ERROR] Could not compute top strategies or total money won: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
