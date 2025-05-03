#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
My Crypto Paper Trading Script

This script runs a paper trading simulation for cryptocurrencies using the Mercurio AI platform.
It allows you to test different strategies with virtual portfolios and track performance
without risking real capital.
"""

from dotenv import load_dotenv
load_dotenv()

import asyncio
import logging
import signal
import os
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
from pathlib import Path

from app.services.market_data import MarketDataService
from app.services.trading import TradingService
from app.services.strategy_manager import StrategyManager

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === CONFIGURABLE PARAMETERS ===
DURATION_HOURS = 24  # Default trading duration in hours
CHECK_INTERVAL_SECONDS = 300  # Check every 5 minutes
INITIAL_CAPITAL = 1000  # Default initial capital per strategy in USD

# Cryptocurrencies to trade
CRYPTO_SYMBOLS = [
    "BTC-USD",    # Bitcoin
    "ETH-USD",    # Ethereum
    "SOL-USD",    # Solana
    "ADA-USD",    # Cardano
    "XRP-USD",    # Ripple
    "AVAX-USD",   # Avalanche
    "DOT-USD",    # Polkadot
    "DOGE-USD",   # Dogecoin
    "SHIB-USD",   # Shiba Inu
    "MATIC-USD",  # Polygon
]

# Strategies to use
STRATEGIES = [
    "MovingAverageStrategy",
    "LSTMPredictorStrategy",
    "MultiSourceIntelligenceStrategy",
    "TransformerStrategy",
    "LLMStrategy",
]

# Global flag for graceful shutdown
running = True

def signal_handler(sig, frame):
    """Handle termination signals to allow graceful shutdown"""
    global running
    logger.info("Received termination signal. Stopping after current iteration...")
    running = False

class CryptoPortfolio:
    """Class to track and manage a cryptocurrency paper trading portfolio"""
    
    def __init__(self, initial_cash, symbols, strategy_name):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.holdings = {symbol: 0 for symbol in symbols}
        self.trade_history = []
        self.portfolio_history = []
        self.strategy_name = strategy_name
        self.start_time = datetime.now()
    
    async def execute_trade(self, symbol, action, price, confidence, timestamp=None):
        """Execute a trade based on the given action and price"""
        if timestamp is None:
            timestamp = datetime.now()
        
        if action == "BUY" and self.cash > 0:
            # Invest up to 20% of available cash per trade, scaled by confidence
            max_investment = self.cash * 0.2 * confidence
            qty = max_investment / price
            
            if qty * price >= 10:  # Only trade if amount is at least $10
                trade_value = qty * price
                self.cash -= trade_value
                self.holdings[symbol] += qty
                
                self.trade_history.append({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'action': 'BUY',
                    'price': price,
                    'quantity': qty,
                    'value': trade_value,
                    'confidence': confidence,
                    'cash_after': self.cash
                })
                
                logger.info(f"BUY: {qty:.6f} {symbol} at ${price:.2f} (${trade_value:.2f})")
                return True
        
        elif action == "SELL" and self.holdings[symbol] > 0:
            # Sell all holdings for this symbol
            qty = self.holdings[symbol]
            trade_value = qty * price
            self.cash += trade_value
            self.holdings[symbol] = 0
            
            self.trade_history.append({
                'timestamp': timestamp,
                'symbol': symbol,
                'action': 'SELL',
                'price': price,
                'quantity': qty,
                'value': trade_value,
                'confidence': confidence,
                'cash_after': self.cash
            })
            
            logger.info(f"SELL: {qty:.6f} {symbol} at ${price:.2f} (${trade_value:.2f})")
            return True
        
        return False
    
    async def update_portfolio_value(self, market_data_service):
        """Update portfolio value with current market prices"""
        current_value = self.cash
        price_data = {}
        
        for symbol, qty in self.holdings.items():
            if qty > 0:
                # Try different providers with fallback
                price = None
                for provider in ["polygon", "yahoo", "sample"]:
                    try:
                        price = await market_data_service.get_latest_price(symbol, provider_name=provider)
                        if price is not None:
                            break
                    except Exception as e:
                        logger.warning(f"Provider '{provider}' failed for {symbol}: {e}")
                
                if price is None:
                    logger.error(f"Could not get latest price for {symbol} from any provider")
                    continue
                
                price_data[symbol] = price
                current_value += qty * price
        
        self.portfolio_history.append({
            'timestamp': datetime.now(),
            'total_value': current_value,
            'cash': self.cash,
            'holdings_value': current_value - self.cash,
            'prices': price_data.copy()
        })
        
        return current_value
    
    def get_summary(self):
        """Get a summary of the portfolio performance"""
        if not self.portfolio_history:
            return {
                'strategy': self.strategy_name,
                'initial_value': self.initial_cash,
                'current_value': self.initial_cash,
                'profit_loss': 0,
                'profit_loss_pct': 0,
                'num_trades': 0,
                'duration': str(datetime.now() - self.start_time)
            }
        
        current_value = self.portfolio_history[-1]['total_value']
        profit_loss = current_value - self.initial_cash
        profit_loss_pct = (profit_loss / self.initial_cash) * 100
        
        return {
            'strategy': self.strategy_name,
            'initial_value': self.initial_cash,
            'current_value': current_value,
            'profit_loss': profit_loss,
            'profit_loss_pct': profit_loss_pct,
            'num_trades': len(self.trade_history),
            'duration': str(datetime.now() - self.start_time)
        }
    
    def save_results(self, output_dir='results'):
        """Save portfolio results to files"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"{self.strategy_name}_{timestamp}"
        
        # Save trade history
        trades_df = pd.DataFrame(self.trade_history)
        if not trades_df.empty:
            trades_df.to_csv(f"{output_dir}/{base_filename}_trades.csv", index=False)
        
        # Save portfolio history
        portfolio_data = []
        for entry in self.portfolio_history:
            data = {
                'timestamp': entry['timestamp'],
                'total_value': entry['total_value'],
                'cash': entry['cash'],
                'holdings_value': entry['holdings_value']
            }
            for symbol, price in entry.get('prices', {}).items():
                data[f"{symbol}_price"] = price
                data[f"{symbol}_qty"] = self.holdings.get(symbol, 0)
            portfolio_data.append(data)
        
        portfolio_df = pd.DataFrame(portfolio_data)
        if not portfolio_df.empty:
            portfolio_df.to_csv(f"{output_dir}/{base_filename}_portfolio.csv", index=False)
        
        # Generate and save summary visualization
        self.generate_visualization(output_dir, base_filename)
        
        return f"{output_dir}/{base_filename}"
    
    def generate_visualization(self, output_dir, base_filename):
        """Generate visualization of portfolio performance"""
        if not self.portfolio_history:
            return
        
        portfolio_df = pd.DataFrame(self.portfolio_history)
        portfolio_df['timestamp'] = pd.to_datetime(portfolio_df['timestamp'])
        
        # Create plot with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot total value over time
        ax1.plot(portfolio_df['timestamp'], portfolio_df['total_value'], 'b-', label='Total Value')
        ax1.plot(portfolio_df['timestamp'], portfolio_df['cash'], 'g--', label='Cash')
        
        # Add buy/sell markers
        for trade in self.trade_history:
            x = trade['timestamp']
            y = trade['cash_after'] + (trade['quantity'] * trade['price'])
            if trade['action'] == 'BUY':
                ax1.plot(x, y, 'g^', markersize=8)
            else:
                ax1.plot(x, y, 'rv', markersize=8)
        
        ax1.set_title(f'Portfolio Performance - {self.strategy_name}')
        ax1.set_ylabel('Value (USD)')
        ax1.grid(True)
        ax1.legend()
        
        # Plot holdings distribution in the bottom subplot
        if self.portfolio_history:
            latest = self.portfolio_history[-1]
            holdings_value = {}
            for symbol, qty in self.holdings.items():
                if qty > 0 and symbol in latest.get('prices', {}):
                    holdings_value[symbol] = qty * latest['prices'][symbol]
            
            if holdings_value:
                labels = list(holdings_value.keys())
                sizes = list(holdings_value.values())
                
                # Add cash to the pie chart
                labels.append('Cash')
                sizes.append(latest['cash'])
                
                ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                ax2.axis('equal')
                ax2.set_title('Current Portfolio Allocation')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{base_filename}_performance.png")
        plt.close()

async def run_crypto_trading(strategy_name, symbols, initial_capital, duration_hours, check_interval_seconds):
    """Run crypto trading simulation for a single strategy"""
    logger.info(f"Starting {strategy_name} with {initial_capital} USD for {duration_hours} hours")
    
    # Initialize services
    market_data_service = MarketDataService()
    strategy_manager = StrategyManager()
    
    # Initialize the strategy
    strategy = await strategy_manager.get_strategy(strategy_name)
    if strategy is None:
        logger.error(f"Strategy {strategy_name} not found")
        return None
    
    # Set up strategy if needed
    if hasattr(strategy, 'setup') and asyncio.iscoroutinefunction(strategy.setup):
        await strategy.setup()
    
    # Initialize portfolio
    portfolio = CryptoPortfolio(initial_capital, symbols, strategy_name)
    
    # Set end time
    end_time = datetime.now() + timedelta(hours=duration_hours)
    
    # Main trading loop
    while datetime.now() < end_time and running:
        # Update portfolio value with current prices
        await portfolio.update_portfolio_value(market_data_service)
        
        # Check each symbol for trading signals
        for symbol in symbols:
            try:
                # Get prediction for this symbol
                prediction = await strategy_manager.get_prediction(symbol, strategy_name)
                action = prediction.get('action', None)
                confidence = prediction.get('confidence', 0.5)  # Default to 0.5 if not provided
                
                # Stratégie d'acquisition des prix
                # 1. Essayer Yahoo Finance qui fonctionne bien pour les cryptos sans API key
                # 2. Essayer Alpaca avec le symbole converti sans tiret (format BTCUSD au lieu de BTC-USD)
                # 3. Essayer Polygon et Sample en dernier recours
                price = None
                
                # 1. Essayer d'abord Yahoo Finance
                try:
                    price = await market_data_service.get_latest_price(symbol, provider_name="yahoo")
                    if price is not None:
                        logger.info(f"Using price ${price:.2f} for {symbol} from yahoo")
                except Exception as e:
                    logger.warning(f"Yahoo failed for {symbol}: {str(e)[:50]}...")
                
                # 2. Essayer Alpaca avec symbole converti
                if price is None and "-USD" in symbol:
                    try:
                        # Convertir le format de symbole pour Alpaca (BTC-USD -> BTCUSD)
                        alpaca_symbol = symbol.replace("-USD", "USD")
                        logger.info(f"Trying Alpaca with converted symbol: {alpaca_symbol}")
                        
                        # Utiliser le client Alpaca du MarketDataService
                        price = await market_data_service.get_latest_price_legacy_alpaca(alpaca_symbol)
                        
                        if price is not None:
                            logger.info(f"Using price ${price:.2f} for {symbol} from Alpaca (as {alpaca_symbol})")
                    except Exception as e:
                        if "invalid symbol" in str(e):
                            logger.warning(f"Alpaca doesn't support symbol {alpaca_symbol}")
                        else:
                            logger.warning(f"Alpaca failed: {str(e)[:50]}...")
                
                # 3. Essayer les autres fournisseurs de repli
                if price is None:
                    for fallback_provider in ["polygon", "sample"]:
                        try:
                            price = await market_data_service.get_latest_price(symbol, provider_name=fallback_provider)
                            if price is not None:
                                logger.info(f"Using price ${price:.2f} for {symbol} from {fallback_provider}")
                                break
                        except Exception as e:
                            # Log seulement un résumé de l'erreur pour éviter les traces longues
                            if "NOT_AUTHORIZED" in str(e):
                                logger.warning(f"{fallback_provider} API not authorized for {symbol}")
                            elif "invalid symbol" in str(e):
                                logger.warning(f"{fallback_provider} doesn't support format {symbol}")
                            else:
                                logger.warning(f"{fallback_provider} failed for {symbol}: {str(e)[:50]}...")
                                
                # Si aucun fournisseur n'a fonctionné après tous les essais
                if price is None:
                    logger.error(f"Could not get price for {symbol} from any provider")
                
                if price is None:
                    logger.error(f"Could not get latest price for {symbol} from any provider")
                    continue
                
                # Execute trade if we have a clear signal
                if action in ["BUY", "SELL"] and confidence >= 0.55:  # Only trade with confidence > 55%
                    await portfolio.execute_trade(symbol, action, price, confidence)
                
                # Log the current status
                logger.info(f"{strategy_name} | {symbol} | Signal: {action} | Confidence: {confidence:.2f} | Price: ${price:.2f}")
                
            except Exception as e:
                logger.error(f"Error processing {symbol} with {strategy_name}: {e}")
        
        # Show current portfolio summary
        summary = portfolio.get_summary()
        logger.info(f"PORTFOLIO: ${summary['current_value']:.2f} ({summary['profit_loss_pct']:+.2f}%)")
        
        # Wait for next check interval
        await asyncio.sleep(check_interval_seconds)
    
    # Save final results
    results_path = portfolio.save_results()
    logger.info(f"Results saved to {results_path}")
    
    return portfolio

async def main():
    """Main function to run the crypto trading simulation"""
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Crypto Paper Trading Simulation")
    parser.add_argument("--duration", type=int, default=DURATION_HOURS, 
                        help=f"Trading duration in hours (default: {DURATION_HOURS})")
    parser.add_argument("--capital", type=float, default=INITIAL_CAPITAL, 
                        help=f"Initial capital per strategy in USD (default: {INITIAL_CAPITAL})")
    parser.add_argument("--interval", type=int, default=CHECK_INTERVAL_SECONDS, 
                        help=f"Check interval in seconds (default: {CHECK_INTERVAL_SECONDS})")
    parser.add_argument("--symbols", type=str, nargs='+', 
                        help=f"Crypto symbols to trade (default: {', '.join(CRYPTO_SYMBOLS)})")
    parser.add_argument("--strategies", type=str, nargs='+', 
                        help=f"Strategies to use (default: {', '.join(STRATEGIES)})")
    
    args = parser.parse_args()
    
    # Validate parameters
    if args.duration < 1 or args.duration > 720:  # Max 30 days
        logger.error("Duration must be between 1 and 720 hours")
        return
    
    if args.capital < 100 or args.capital > 1_000_000:
        logger.error("Initial capital must be between 100 and 1,000,000 USD")
        return
    
    if args.interval < 60 or args.interval > 3600:  # Between 1 minute and 1 hour
        logger.error("Check interval must be between 60 and 3600 seconds")
        return
    
    # Use provided symbols or default
    symbols = args.symbols if args.symbols else CRYPTO_SYMBOLS
    
    # Use provided strategies or default
    strategies = args.strategies if args.strategies else STRATEGIES
    
    logger.info("=== Starting Crypto Paper Trading Simulation ===")
    logger.info(f"Duration: {args.duration} hours")
    logger.info(f"Initial Capital: ${args.capital}")
    logger.info(f"Check Interval: {args.interval} seconds")
    logger.info(f"Symbols: {', '.join(symbols)}")
    logger.info(f"Strategies: {', '.join(strategies)}")
    
    # Configurer la journalisation pour réduire le bruit tout en gardant les infos importantes
    logging.getLogger('app.services.providers.polygon').setLevel(logging.ERROR) 
    
    # On garde les warnings pour market_data car ils pourraient contenir des infos importantes
    # au sujet des tentatives avec Alpaca
    logging.getLogger('app.services.market_data').setLevel(logging.INFO)
    
    # Run each strategy in parallel
    tasks = [
        run_crypto_trading(
            strategy, 
            symbols, 
            args.capital, 
            args.duration, 
            args.interval
        )
        for strategy in strategies
    ]
    
    portfolios = await asyncio.gather(*tasks)
    portfolios = [p for p in portfolios if p is not None]
    
    # Print comparison table
    if portfolios:
        summaries = [p.get_summary() for p in portfolios]
        
        table_data = []
        for summary in summaries:
            table_data.append([
                summary['strategy'],
                f"${summary['initial_value']:.2f}",
                f"${summary['current_value']:.2f}",
                f"{summary['profit_loss_pct']:+.2f}%",
                summary['num_trades']
            ])
        
        print("\n=== STRATEGY COMPARISON ===")
        print(tabulate(
            table_data,
            headers=["Strategy", "Initial", "Final", "Return", "Trades"],
            tablefmt="fancy_grid"
        ))
        
        # Generate combined visualization
        best_portfolio = max(portfolios, key=lambda p: p.get_summary()['profit_loss_pct'])
        logger.info(f"Best strategy: {best_portfolio.strategy_name} with {best_portfolio.get_summary()['profit_loss_pct']:+.2f}% return")
    else:
        logger.warning("No portfolios were successfully created")
    
    logger.info("=== Crypto Paper Trading Simulation Completed ===")

if __name__ == "__main__":
    asyncio.run(main())
