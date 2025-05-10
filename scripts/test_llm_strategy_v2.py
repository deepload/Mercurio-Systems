#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script pour la stratégie LLMStrategyV2
"""
import os
import sys
import asyncio
import logging
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to path to allow running from any directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import necessary modules
from app.services.market_data import MarketDataService
from app.services.trading import TradingService
from app.strategies.llm_strategy_v2 import LLMStrategyV2
from app.db.models import TradeAction


async def test_signals(symbols=None, days=30):
    """Test the LLMStrategyV2 on recent market data for specified symbols"""
    if symbols is None:
        # Use some popular symbols that are likely to have sample data
        symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]

    # Initialize services
    market_data = MarketDataService()
    trading_service = TradingService(is_paper=True)
    
    # Initialize strategy with both technical and sentiment analysis
    strategy = LLMStrategyV2(
        market_data_service=market_data,
        trading_service=trading_service,
        use_web_sentiment=True,
        technical_weight=0.7,
        sentiment_weight=0.3,
        min_confidence=0.6
    )
    
    # Set timeframe
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    results = []
    
    # Process each symbol
    for symbol in symbols:
        logger.info(f"Processing symbol: {symbol}")
        
        # Load data
        data = await strategy.load_data(symbol, start_date, end_date)
        
        if data.empty:
            logger.warning(f"No data available for {symbol}")
            continue
            
        # Preprocess the data
        processed_data = await strategy.preprocess_data(data)
        
        # Generate prediction
        action, confidence = await strategy.predict(processed_data)
        
        # Safely extract values with error handling
        try:
            # Check if we have data before accessing it
            if processed_data.empty or len(processed_data) < 1:
                logger.warning(f"Insufficient data for {symbol}")
                continue
                
            # Store results
            current_price = processed_data["close"].iloc[-1] if "close" in processed_data.columns and len(processed_data) > 0 else 0
            sma_10 = processed_data["sma_10"].iloc[-1] if "sma_10" in processed_data.columns and len(processed_data) > 0 else None
            sma_50 = processed_data["sma_50"].iloc[-1] if "sma_50" in processed_data.columns and len(processed_data) > 0 else None
            rsi = processed_data["rsi_14"].iloc[-1] if "rsi_14" in processed_data.columns and len(processed_data) > 0 else None
            
            results.append({
                "symbol": symbol,
                "action": action.name,
                "confidence": confidence,
                "current_price": current_price,
                "sma_10": sma_10,
                "sma_50": sma_50,
                "rsi": rsi,
                "sentiment_used": strategy.use_web_sentiment
            })
            
            logger.info(f"Successfully processed {symbol} with action: {action.name}")
        except Exception as e:
            logger.error(f"Error processing results for {symbol}: {str(e)}")
            continue
        
        # Run a short backtest with error handling
        try:
            backtest_results = await strategy.backtest(processed_data)
            
            if backtest_results and "metrics" in backtest_results and "total_return_pct" in backtest_results["metrics"]:
                logger.info(f"Backtest results for {symbol}: "
                          f"Return: {backtest_results['metrics']['total_return_pct']:.2f}%, "
                          f"Trades: {backtest_results['metrics']['total_trades']}")
            
                if len(backtest_results.get('equity_curve', [])) > 0:
                    try:
                        # Create quick equity chart
                        equity_df = pd.DataFrame(backtest_results['equity_curve'])
                        if not equity_df.empty and 'equity' in equity_df.columns:
                            if 'timestamp' in equity_df.columns:
                                equity_df = equity_df.set_index('timestamp')
                            
                            plt.figure(figsize=(10, 6))
                            plt.plot(equity_df['equity'])
                            plt.title(f"{symbol} - LLMStrategyV2 Equity Curve")
                            plt.grid(True)
                            plt.tight_layout()
                            plt.savefig(f"{symbol}_llm_v2_equity.png")
                            plt.close()
                    except Exception as chart_err:
                        logger.error(f"Error creating chart for {symbol}: {str(chart_err)}")
            else:
                logger.warning(f"Incomplete backtest results for {symbol}")
        except Exception as backtest_err:
            logger.error(f"Error during backtest for {symbol}: {str(backtest_err)}")
            
    # Display all results
    print("\n" + "="*50)
    print("LLMStrategyV2 TRADING SIGNALS")
    print("="*50)
    
    for result in results:
        action_emoji = "🔴 SELL" if result["action"] == "SELL" else "🟢 BUY" if result["action"] == "BUY" else "⚪ HOLD"
        print(f"{result['symbol']}: {action_emoji} (confidence: {result['confidence']:.2f})")
        
        # Print technical indicators if available
        if result["sma_10"] is not None and result["sma_50"] is not None:
            sma_status = "BULLISH" if result["sma_10"] > result["sma_50"] else "BEARISH"
            print(f"   SMA Crossover: {sma_status} (SMA10: {result['sma_10']:.2f}, SMA50: {result['sma_50']:.2f})")
            
        if result["rsi"] is not None:
            rsi_status = "OVERSOLD" if result["rsi"] < 30 else "OVERBOUGHT" if result["rsi"] > 70 else "NEUTRAL"
            print(f"   RSI: {result['rsi']:.2f} - {rsi_status}")
            
        print(f"   Price: {result['current_price']:.2f}")
        print(f"   Web Sentiment Used: {'Yes' if result['sentiment_used'] else 'No'}")
        print("-"*40)


if __name__ == "__main__":
    asyncio.run(test_signals())
