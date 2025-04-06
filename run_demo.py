"""
Mercurio AI Demo Script

This script provides a quick demo of the Mercurio AI platform capabilities.
It runs in simulation mode, demonstrating each component without using real money.
"""
import os
import asyncio
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure required directories exist
os.makedirs("./data", exist_ok=True)
os.makedirs("./models", exist_ok=True)
os.makedirs("./logs", exist_ok=True)

# Demo configuration
SYMBOLS = ["AAPL", "MSFT", "GOOGL"]
PAPER_TRADING = True  # Always use paper trading for demo
INITIAL_CAPITAL = 100000.0
DAYS_HISTORY = 180

async def run_demo():
    """Run a complete demo of the Mercurio AI platform."""
    from app.services.market_data import MarketDataService
    from app.services.trading import TradingService
    from app.services.backtesting import BacktestingService
    from app.services.strategy_manager import StrategyManager
    from app.db.models import TradeAction
    
    logger.info("=" * 50)
    logger.info("MERCURIO AI DEMO - Starting")
    logger.info("=" * 50)
    
    # Initialize services
    market_data = MarketDataService()
    trading_service = TradingService(is_paper=PAPER_TRADING)
    backtesting_service = BacktestingService()
    strategy_manager = StrategyManager()
    
    # Step 1: Check market status
    logger.info("\n\nStep 1: Checking market status...")
    try:
        market_status = await trading_service.check_market_status()
        logger.info(f"Market is {'open' if market_status.get('is_open') else 'closed'}")
        logger.info(f"Next open: {market_status.get('next_open')}")
        logger.info(f"Next close: {market_status.get('next_close')}")
    except Exception as e:
        logger.error(f"Error checking market status: {e}")
        logger.info("Continuing with demo anyway...")
    
    # Step 2: Check account information
    logger.info("\n\nStep 2: Checking account information...")
    try:
        account_info = await trading_service.get_account_info()
        logger.info(f"Account Status: {account_info.get('status')}")
        logger.info(f"Portfolio Value: ${account_info.get('portfolio_value')}")
        logger.info(f"Cash: ${account_info.get('cash')}")
        logger.info(f"Buying Power: ${account_info.get('buying_power')}")
    except Exception as e:
        logger.error(f"Error checking account: {e}")
        logger.info("Continuing with demo anyway...")
    
    # Step 3: List available strategies
    logger.info("\n\nStep 3: Listing available strategies...")
    strategies = await strategy_manager.list_strategies()
    logger.info(f"Found {len(strategies)} strategies:")
    for strategy in strategies:
        logger.info(f"- {strategy['name']}: {strategy['description']}")
    
    # Step 4: Get historical data for demo
    logger.info("\n\nStep 4: Fetching historical data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=DAYS_HISTORY)
    
    data_by_symbol = {}
    for symbol in SYMBOLS:
        try:
            logger.info(f"Fetching data for {symbol}...")
            data = await market_data.get_historical_data(symbol, start_date, end_date)
            data_by_symbol[symbol] = data
            logger.info(f"Got {len(data)} data points for {symbol}")
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
    
    if not data_by_symbol:
        logger.error("Could not fetch data for any symbols. Demo cannot continue.")
        return
    
    # Pick the first available symbol with data
    symbol = next(iter(data_by_symbol.keys()))
    data = data_by_symbol[symbol]
    
    # Step 5: Run backtest with moving average strategy
    logger.info("\n\nStep 5: Running backtest with Moving Average strategy...")
    ma_strategy_name = "MovingAverageCrossover"
    try:
        # Get strategy info
        ma_strategy_info = await strategy_manager.get_strategy_info(ma_strategy_name)
        if not ma_strategy_info:
            logger.error(f"Strategy {ma_strategy_name} not found")
        else:
            # Initialize strategy
            ma_strategy = await strategy_manager.get_strategy(ma_strategy_name)
            
            # Preprocess data
            processed_data = await ma_strategy.preprocess_data(data)
            
            # Run backtest
            logger.info(f"Running backtest on {symbol} with {ma_strategy_name}...")
            backtest_results = await ma_strategy.backtest(
                processed_data, 
                initial_capital=INITIAL_CAPITAL
            )
            
            # Show results
            logger.info(f"Backtest Results:")
            logger.info(f"Initial Capital: ${INITIAL_CAPITAL:.2f}")
            logger.info(f"Final Capital: ${backtest_results['final_capital']:.2f}")
            logger.info(f"Total Return: {backtest_results['total_return']*100:.2f}%")
            logger.info(f"Number of Trades: {len(backtest_results['trades'])}")
            
            # Plot equity curve
            plt.figure(figsize=(12, 6))
            plt.plot(backtest_results['equity_curve'])
            plt.title(f"{ma_strategy_name} Equity Curve")
            plt.xlabel("Date")
            plt.ylabel("Equity ($)")
            plt.grid(True)
            equity_curve_path = f"./data/{symbol}_{ma_strategy_name}_equity_curve.png"
            plt.savefig(equity_curve_path)
            plt.close()
            logger.info(f"Equity curve saved to {equity_curve_path}")
    except Exception as e:
        logger.error(f"Error in MA backtest: {e}")
    
    # Step 6: Test LSTM strategy (simplified for demo)
    logger.info("\n\nStep 6: Testing LSTM strategy...")
    lstm_strategy_name = "LSTMPredictor"
    try:
        # Get strategy info
        lstm_strategy_info = await strategy_manager.get_strategy_info(lstm_strategy_name)
        if not lstm_strategy_info:
            logger.error(f"Strategy {lstm_strategy_name} not found")
        else:
            # Initialize strategy with minimal epochs for demo
            lstm_strategy = await strategy_manager.get_strategy(
                lstm_strategy_name, 
                {"epochs": 2, "batch_size": 32}  # Minimal training for demo
            )
            
            # Process data
            lstm_data = await lstm_strategy.preprocess_data(data)
            
            # Train model (simplified)
            logger.info("Training LSTM model (simplified for demo)...")
            training_metrics = await lstm_strategy.train(lstm_data)
            
            # Get prediction
            logger.info("Getting prediction from LSTM model...")
            action, confidence = await lstm_strategy.predict(lstm_data)
            
            # Show results
            logger.info(f"LSTM Prediction:")
            logger.info(f"Action: {action.name}")
            logger.info(f"Confidence: {confidence:.2f}")
            
            # Save model
            model_path = await lstm_strategy.save_model("./models")
            logger.info(f"LSTM model saved to {model_path}")
    except Exception as e:
        logger.error(f"Error in LSTM test: {e}")
    
    # Step 7: Generate trades for demo
    logger.info("\n\nStep 7: Simulating paper trades...")
    try:
        # Initialize lists for trades
        symbols_to_trade = [s for s in SYMBOLS if s in data_by_symbol]
        
        for symbol in symbols_to_trade:
            # Get prediction from MA strategy
            ma_strategy = await strategy_manager.get_strategy(ma_strategy_name)
            processed_data = await ma_strategy.preprocess_data(data_by_symbol[symbol])
            action, confidence = await ma_strategy.predict(processed_data)
            
            logger.info(f"Prediction for {symbol}: {action.name} (confidence: {confidence:.2f})")
            
            # Skip if HOLD
            if action == TradeAction.HOLD:
                logger.info(f"Skipping {symbol} trade - HOLD signal")
                continue
            
            # Calculate quantity (for demo)
            price = await market_data.get_latest_price(symbol)
            if not price:
                # Use last price from historical data
                price = processed_data['close'].iloc[-1]
            
            # Use 1% of capital per trade
            capital_percentage = 0.01
            quantity = await trading_service.calculate_order_quantity(
                symbol, action, capital_percentage
            )
            
            if quantity <= 0:
                logger.info(f"Skipping {symbol} trade - zero quantity calculated")
                continue
            
            # Execute paper trade
            logger.info(f"Executing paper trade: {action.name} {quantity} {symbol} @ ${price:.2f}")
            
            if PAPER_TRADING:
                # Only execute if paper trading is enabled
                trade_result = await trading_service.execute_trade(
                    symbol=symbol,
                    action=action,
                    quantity=quantity,
                    strategy_name=ma_strategy_name
                )
                
                logger.info(f"Trade result: {trade_result}")
            else:
                logger.info("Skipping actual trade execution - demo mode")
    except Exception as e:
        logger.error(f"Error simulating trades: {e}")
    
    logger.info("\n\n" + "=" * 50)
    logger.info("MERCURIO AI DEMO - Completed")
    logger.info("=" * 50)

if __name__ == "__main__":
    # Run the demo
    asyncio.run(run_demo())
