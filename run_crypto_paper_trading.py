import asyncio
import logging
import signal
from datetime import datetime, timedelta

from app.services.market_data import MarketDataService
from app.services.trading import TradingService
from app.services.strategy_manager import StrategyManager

# === CONFIGURABLE PARAMETERS ===
DURATION_MINUTES = 60  # Default trading duration in minutes
CRYPTO_SYMBOLS = ["BTC-USD", "ETH-USD"]  # Add more symbols as needed
STRATEGIES = [
    "MovingAverageStrategy",
    "MovingAverageMLStrategy",
    "LSTMPredictorStrategy",
    "TransformerStrategy",
    "LLMStrategy",
    "MSIStrategy"
]
STRATEGY_INITIAL_CAPITAL = {
    "MovingAverageStrategy": 100,
    "MovingAverageMLStrategy": 100,
    "LSTMPredictorStrategy": 100,
    "TransformerStrategy": 100,
    "LLMStrategy": 100,
    "MSIStrategy": 100
}

running = True

def signal_handler(sig, frame):
    global running
    print("Received termination signal. Stopping after current iteration...")
    running = False

async def run_strategy_for_duration(strategy_name, symbols, initial_capital, results_dict, duration_minutes):
    market_data_service = MarketDataService()
    strategy_manager = StrategyManager()
    strategy = await strategy_manager.get_strategy(strategy_name)
    if strategy and hasattr(strategy, 'setup') and asyncio.iscoroutinefunction(strategy.setup):
        await strategy.setup()

    # Virtual portfolio: cash and holdings per symbol
    portfolio = {"cash": initial_capital, "holdings": {symbol: 0 for symbol in symbols}}
    price_history = {symbol: [] for symbol in symbols}
    trade_log = []

    end_time = datetime.utcnow() + timedelta(minutes=duration_minutes)
    while datetime.utcnow() < end_time and running:
        for symbol in symbols:
            prediction = await strategy_manager.get_prediction(symbol, strategy_name)
            action = prediction.get('action', None)
            confidence = prediction.get('confidence', None)
            price = await market_data_service.get_latest_price(symbol)
            price_history[symbol].append(price)
            if not price:
                continue
            # Only execute trades if action is BUY or SELL
            if action == "BUY" and portfolio["cash"] >= price:
                qty = portfolio["cash"] // price
                if qty > 0:
                    portfolio["cash"] -= qty * price
                    portfolio["holdings"][symbol] += qty
                    trade_log.append((datetime.utcnow(), symbol, "BUY", qty, price))
            elif action == "SELL" and portfolio["holdings"][symbol] > 0:
                qty = portfolio["holdings"][symbol]
                portfolio["cash"] += qty * price
                portfolio["holdings"][symbol] = 0
                trade_log.append((datetime.utcnow(), symbol, "SELL", qty, price))
            print(f"{datetime.utcnow()} | {strategy_name} | {symbol} | Signal: {action} | Confidence: {confidence} | Cash: {portfolio['cash']:.2f} | Holdings: {portfolio['holdings'][symbol]}")
        await asyncio.sleep(60)  # Check every minute

    # At the end, compute final portfolio value
    final_value = portfolio["cash"]
    for symbol in symbols:
        # Use the last known price for each symbol
        last_price = price_history[symbol][-1] if price_history[symbol] else 0
        final_value += portfolio["holdings"][symbol] * last_price
    results_dict[strategy_name] = {
        "initial": initial_capital,
        "final": final_value,
        "trades": trade_log
    }


import argparse

async def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    parser = argparse.ArgumentParser(description="Crypto Paper Trading with Multiple Strategies")
    parser.add_argument("--duration_minutes", type=int, default=DURATION_MINUTES, help="Trading duration in minutes (10-480)")
    parser.add_argument("--initial_capital", type=float, help="Initial capital per strategy (overrides STRATEGY_INITIAL_CAPITAL for all)")
    args = parser.parse_args()
    duration = args.duration_minutes
    if duration < 10 or duration > 480:
        print("Error: duration_minutes must be between 10 and 480.")
        return
    # Determine initial capital for each strategy
    if args.initial_capital is not None:
        if args.initial_capital < 10 or args.initial_capital > 1_000_000:
            print("Error: initial_capital must be between 10 and 1,000,000 USD.")
            return
        init_capitals = {s: args.initial_capital for s in STRATEGIES}
    else:
        init_capitals = STRATEGY_INITIAL_CAPITAL.copy()
    results = {}
    tasks = [
        run_strategy_for_duration(strategy, CRYPTO_SYMBOLS, init_capitals[strategy], results, duration)
        for strategy in STRATEGIES
    ]
    await asyncio.gather(*tasks)

    # Print comparison table
    print("\n==== STRATEGY COMPARISON ====")
    print(f"{'Strategy':<25} {'Initial($)':>10} {'Final($)':>10} {'Return(%)':>12}")
    for strategy, result in results.items():
        init = result['initial']
        final = result['final']
        ret = ((final - init) / init * 100) if init else 0
        print(f"{strategy:<25} {init:>10.2f} {final:>10.2f} {ret:>12.2f}")
    print("============================\n")

if __name__ == "__main__":
    asyncio.run(main())
