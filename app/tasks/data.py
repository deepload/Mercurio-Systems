"""
Data Collection Tasks

Celery tasks for market data collection and management.
"""
import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio
import pandas as pd

from app.tasks.celery_app import celery_app
from app.services.market_data import MarketDataService

logger = logging.getLogger(__name__)

@celery_app.task(name="collect_historical_data")
def collect_historical_data(
    symbols: List[str],
    start_date: str,
    end_date: str,
    timeframe: str = "1d"
) -> Dict[str, Any]:
    """
    Celery task to collect historical data for a set of symbols.
    
    Args:
        symbols: List of symbols to collect data for
        start_date: Start date for data collection (ISO format)
        end_date: End date for data collection (ISO format)
        timeframe: Timeframe for data (e.g., '1d', '1h')
        
    Returns:
        Dictionary with collection results
    """
    logger.info(f"Starting historical data collection for {symbols}")
    
    # Run the async data collection function in event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Convert dates from ISO format
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        
        # Initialize market data service
        market_data = MarketDataService()
        
        results = {}
        
        for symbol in symbols:
            try:
                # Get historical data
                data = loop.run_until_complete(
                    market_data.get_historical_data(symbol, start, end, timeframe)
                )
                
                # Add to results
                results[symbol] = {
                    "status": "success",
                    "rows": len(data),
                    "start": data.index[0].isoformat() if not data.empty else None,
                    "end": data.index[-1].isoformat() if not data.empty else None,
                }
                
                # Save data to CSV (optional)
                # You could save this data to a persistent storage like S3 or a database
                data_dir = os.getenv("DATA_DIR", "./data")
                os.makedirs(data_dir, exist_ok=True)
                
                filename = f"{symbol}_{timeframe}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv"
                filepath = os.path.join(data_dir, filename)
                
                data.to_csv(filepath)
                results[symbol]["file"] = filepath
                
            except Exception as e:
                logger.error(f"Error collecting data for {symbol}: {e}")
                results[symbol] = {
                    "status": "error",
                    "message": str(e)
                }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "symbols": symbols,
            "timeframe": timeframe,
            "start_date": start_date,
            "end_date": end_date,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in data collection task: {e}")
        return {"error": str(e)}
    finally:
        loop.close()

@celery_app.task(name="update_market_data")
def update_market_data(
    symbols: List[str] = None,
    days: int = 1,
    timeframe: str = "1d"
) -> Dict[str, Any]:
    """
    Celery task to update recent market data.
    
    Args:
        symbols: List of symbols to update, if None, fetch top symbols
        days: Number of days of data to fetch
        timeframe: Timeframe for data (e.g., '1d', '1h')
        
    Returns:
        Dictionary with update results
    """
    logger.info(f"Starting market data update task")
    
    # Run the async data update function in event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Initialize market data service
        market_data = MarketDataService()
        
        # If no symbols provided, get top symbols
        if not symbols:
            symbols = loop.run_until_complete(market_data.get_market_symbols())
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Call the data collection task
        return collect_historical_data(
            symbols=symbols,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            timeframe=timeframe
        )
        
    except Exception as e:
        logger.error(f"Error in market data update task: {e}")
        return {"error": str(e)}
    finally:
        loop.close()
