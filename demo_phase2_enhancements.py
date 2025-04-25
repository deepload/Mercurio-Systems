"""
MercurioAI Phase 2 Demo

This script demonstrates the new Phase 2 enhancements to MercurioAI:
1. Transformer-based strategy
2. Advanced risk management
3. Portfolio optimization

Run this demo to see how these components work together.
"""
import os
import sys
import asyncio
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure app directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def demo_transformer_strategy():
    """Demonstrate the transformer-based strategy"""
    from app.strategies.transformer_strategy import TransformerStrategy
    from app.services.market_data import MarketDataService
    
    logger.info("="*40)
    logger.info("Demonstrating Transformer Strategy")
    logger.info("="*40)
    
    # Initialize strategy
    strategy = TransformerStrategy(
        sequence_length=30,
        prediction_horizon=5,
        d_model=32,  # Smaller model for demonstration
        nhead=4,
        num_layers=2,
        epochs=10,  # Fewer epochs for demonstration
        batch_size=16
    )
    
    # Get data
    market_data = MarketDataService()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 1 year of data
    
    logger.info("Fetching historical data for BTC/USDT...")
    data = await market_data.get_historical_data(
        "BTC/USDT", 
        start_date.strftime("%Y-%m-%d"), 
        end_date.strftime("%Y-%m-%d")
    )
    
    if data is None or len(data) < 100:
        logger.error("Insufficient data for demonstration")
        return
        
    logger.info(f"Got {len(data)} data points")
    
    # Preprocess data
    logger.info("Preprocessing data...")
    processed_data = await strategy.preprocess_data(data)
    
    # Split into train and test
    train_size = int(len(processed_data) * 0.8)
    train_data = processed_data[:train_size]
    test_data = processed_data[train_size:]
    
    logger.info(f"Training on {len(train_data)} data points")
    logger.info(f"Testing on {len(test_data)} data points")
    
    # Train strategy
    logger.info("Training strategy...")
    try:
        training_results = await strategy.train(train_data)
        logger.info(f"Training completed: {training_results}")
    except Exception as e:
        logger.error(f"Error training strategy: {e}")
        # Continue with fallback mode
        strategy.is_trained = True
    
    # Test predictions
    logger.info("Testing predictions...")
    correct = 0
    total = 0
    
    try:
        # Make sure we have enough data for a meaningful test
        min_sequence_length = strategy.sequence_length
        
        if len(test_data) <= min_sequence_length:
            logger.warning(f"Test data has only {len(test_data)} points, which is not enough for prediction testing")
        else:
            # Test on up to 5 points
            num_test_points = min(5, len(test_data) - min_sequence_length)
            
            for i in range(num_test_points):
                # Use the sequence length as a sliding window
                start_idx = i
                end_idx = i + min_sequence_length
                
                test_window = test_data.iloc[start_idx:end_idx]
                
                if len(test_window) < min_sequence_length:
                    logger.warning(f"Test window has only {len(test_window)} points, skipping")
                    continue
                    
                # Make prediction
                action, confidence = await strategy.predict(test_window)
                
                # Get actual value if available (the target for the last point)
                if 'target' in test_window.columns:
                    actual = test_window.iloc[-1]['target']
                    
                    logger.info(f"Prediction: {action.name} with {confidence:.4f} confidence")
                    logger.info(f"Actual direction: {'UP' if actual > 0 else 'DOWN'}")
                    
                    # Check if prediction matches direction
                    is_correct = (action.name == 'BUY' and actual > 0) or (action.name == 'SELL' and actual <= 0)
                    if is_correct:
                        correct += 1
                    total += 1
                else:
                    logger.warning("Target column not found in test data")
    except Exception as e:
        logger.error(f"Error during prediction testing: {e}")
        # Continue with the rest of the demo
    
    accuracy = correct / total if total > 0 else 0
    logger.info(f"Prediction accuracy: {accuracy:.2%}")
    
    # Run a backtest
    logger.info("Running backtest...")
    backtest_results = await strategy.backtest(processed_data)
    
    logger.info(f"Backtest results:")
    logger.info(f"Initial capital: ${backtest_results['initial_capital']:.2f}")
    logger.info(f"Final equity: ${backtest_results['final_equity']:.2f}")
    logger.info(f"Total return: {backtest_results['total_return']:.2%}")
    
    return {
        'strategy': strategy,
        'backtest_results': backtest_results,
        'accuracy': accuracy
    }


async def demo_risk_management():
    """Demonstrate the advanced risk management system"""
    from app.core.risk_manager import RiskProfile, PositionSizer, DrawdownManager, VaRCalculator, PortfolioRiskManager
    
    logger.info("="*40)
    logger.info("Demonstrating Advanced Risk Management")
    logger.info("="*40)
    
    # Create risk profile
    conservative_profile = RiskProfile(
        name="Conservative",
        max_position_size=0.01,  # 1% maximum position
        max_drawdown=0.10,       # 10% maximum drawdown
        max_daily_loss=0.03,     # 3% maximum daily loss
        position_scaling="volatility",
        stop_loss_pct=0.03,
        take_profit_pct=0.09,
        correlation_limit=0.6,
        var_limit=0.015,
        volatility_adjustment=True
    )
    
    aggressive_profile = RiskProfile(
        name="Aggressive",
        max_position_size=0.05,  # 5% maximum position
        max_drawdown=0.25,       # 25% maximum drawdown
        max_daily_loss=0.08,     # 8% maximum daily loss
        position_scaling="fixed",
        stop_loss_pct=0.08,
        take_profit_pct=0.25,
        correlation_limit=0.8,
        var_limit=0.04,
        volatility_adjustment=False
    )
    
    # Create position sizer with conservative profile
    position_sizer = PositionSizer(conservative_profile)
    
    # Demo position sizing
    logger.info("Position sizing examples:")
    capital = 100000  # $100,000 portfolio
    
    # Low volatility asset
    btc_price = 40000
    btc_volatility = 0.02  # 2% daily volatility
    btc_signal = 0.9       # Strong buy signal
    
    btc_position = position_sizer.calculate_position_size(
        capital, btc_price, btc_volatility, btc_signal
    )
    
    logger.info(f"BTC position at ${btc_price} with {btc_volatility:.1%} volatility and {btc_signal:.1%} signal strength:")
    logger.info(f"  Units: {btc_position:.4f}")
    logger.info(f"  Value: ${btc_position * btc_price:.2f}")
    logger.info(f"  % of Portfolio: {btc_position * btc_price / capital:.2%}")
    
    # Calculate stop loss and take profit
    stop_loss = position_sizer.calculate_stop_loss(
        btc_price, btc_position, capital, btc_volatility, True
    )
    
    take_profit = position_sizer.calculate_take_profit(
        btc_price, stop_loss, True
    )
    
    logger.info(f"  Stop loss: ${stop_loss:.2f} ({(stop_loss/btc_price - 1):.2%})")
    logger.info(f"  Take profit: ${take_profit:.2f} ({(take_profit/btc_price - 1):.2%})")
    
    # High volatility asset
    eth_price = 2000
    eth_volatility = 0.04  # 4% daily volatility
    eth_signal = 0.7       # Moderate buy signal
    
    eth_position = position_sizer.calculate_position_size(
        capital, eth_price, eth_volatility, eth_signal
    )
    
    logger.info(f"ETH position at ${eth_price} with {eth_volatility:.1%} volatility and {eth_signal:.1%} signal strength:")
    logger.info(f"  Units: {eth_position:.4f}")
    logger.info(f"  Value: ${eth_position * eth_price:.2f}")
    logger.info(f"  % of Portfolio: {eth_position * eth_price / capital:.2%}")
    
    # Demonstrate drawdown management
    logger.info("\nDrawdown management:")
    
    drawdown_mgr = DrawdownManager(conservative_profile)
    
    # Simulate equity curve with drawdown
    initial_equity = 100000
    peak_equity = 110000
    current_equity = 95000
    
    drawdown_status = drawdown_mgr.update(peak_equity, "2023-01-01")
    drawdown_status = drawdown_mgr.update(current_equity, "2023-01-15")
    
    logger.info(f"Current drawdown: {drawdown_status['current_drawdown']:.2%}")
    logger.info(f"Position adjustment: {drawdown_mgr.get_position_adjustment():.2%}")
    logger.info(f"Should pause trading: {drawdown_mgr.should_pause_trading()}")
    
    # Demonstrate portfolio risk management
    logger.info("\nPortfolio risk management:")
    
    portfolio_risk_mgr = PortfolioRiskManager(conservative_profile)
    
    # Set portfolio state
    positions = {
        "BTC/USDT": 1.5,
        "ETH/USDT": 10,
        "SOL/USDT": 100
    }
    
    capital = 40000
    equity = 100000
    
    portfolio_risk_mgr.set_portfolio_state(positions, capital, equity)
    
    # Add historical return data
    np.random.seed(42)
    days = 252
    
    # Create return series
    btc_returns = pd.Series(np.random.normal(0.0005, 0.02, days))
    eth_returns = pd.Series(np.random.normal(0.0007, 0.03, days))
    sol_returns = pd.Series(np.random.normal(0.001, 0.04, days))
    
    portfolio_risk_mgr.update_historical_data("BTC/USDT", btc_returns)
    portfolio_risk_mgr.update_historical_data("ETH/USDT", eth_returns)
    portfolio_risk_mgr.update_historical_data("SOL/USDT", sol_returns)
    
    # Check risk limits
    risk_status = portfolio_risk_mgr.check_risk_limits()
    logger.info(f"Risk status: {risk_status}")
    
    # Calculate position size for a new asset
    link_price = 10
    link_vol = 0.03
    link_signal = 0.8
    
    link_position = portfolio_risk_mgr.calculate_position_size(
        "LINK/USDT", link_price, link_vol, link_signal
    )
    
    logger.info(f"LINK position recommendation: {link_position:.2f} units (${link_position * link_price:.2f})")
    
    return {
        'conservative_profile': conservative_profile,
        'aggressive_profile': aggressive_profile,
        'portfolio_risk_mgr': portfolio_risk_mgr
    }


async def demo_portfolio_optimization():
    """Demonstrate portfolio optimization"""
    from app.core.portfolio_optimizer import PortfolioOptimizer, PortfolioRebalancer
    
    logger.info("="*40)
    logger.info("Demonstrating Portfolio Optimization")
    logger.info("="*40)
    
    # Create random return data
    np.random.seed(42)
    days = 252
    
    # Asset returns with different characteristics
    assets = {
        "BTC/USDT": {"mu": 0.0008, "sigma": 0.025},
        "ETH/USDT": {"mu": 0.001, "sigma": 0.035},
        "SOL/USDT": {"mu": 0.0015, "sigma": 0.045},
        "LINK/USDT": {"mu": 0.0007, "sigma": 0.030},
        "ADA/USDT": {"mu": 0.0006, "sigma": 0.038},
        "USDC/USDT": {"mu": 0.0002, "sigma": 0.002}
    }
    
    # Create return dataframe
    returns_data = pd.DataFrame()
    
    for asset, params in assets.items():
        returns_data[asset] = np.random.normal(params["mu"], params["sigma"], days)
    
    # Create correlations between assets
    correlation_matrix = np.array([
        [1.0, 0.8, 0.7, 0.6, 0.5, 0.1],
        [0.8, 1.0, 0.7, 0.7, 0.6, 0.1],
        [0.7, 0.7, 1.0, 0.8, 0.7, 0.2],
        [0.6, 0.7, 0.8, 1.0, 0.6, 0.1],
        [0.5, 0.6, 0.7, 0.6, 1.0, 0.2],
        [0.1, 0.1, 0.2, 0.1, 0.2, 1.0]
    ])
    
    # Ensure returns respect correlation structure
    L = np.linalg.cholesky(correlation_matrix)
    returns_matrix = returns_data.values
    correlated_returns = returns_matrix @ L.T
    
    # Rescale to maintain original means and stds
    for i, asset in enumerate(assets.keys()):
        params = assets[asset]
        current_mean = correlated_returns[:, i].mean()
        current_std = correlated_returns[:, i].std()
        
        # Scale and shift
        returns_data[asset] = params["mu"] + (correlated_returns[:, i] - current_mean) * (params["sigma"] / current_std)
    
    # Create optimizer
    optimizer = PortfolioOptimizer()
    optimizer.set_returns_data(returns_data)
    optimizer.set_risk_free_rate(0.001)  # 0.1% risk-free rate
    
    # Run different optimization methods
    methods = ["equal_weight", "min_variance", "max_sharpe", "risk_parity"]
    
    for method in methods:
        logger.info(f"\n{method.upper()} optimization:")
        results = optimizer.optimize(method)
        
        weights = results["weights"]
        metrics = results["metrics"]
        
        logger.info("Asset weights:")
        for asset, weight in weights.items():
            logger.info(f"  {asset}: {weight:.2%}")
            
        logger.info("Portfolio metrics:")
        logger.info(f"  Expected return: {metrics['return']:.2%}")
        logger.info(f"  Volatility: {metrics['volatility']:.2%}")
        logger.info(f"  Sharpe ratio: {metrics['sharpe_ratio']:.4f}")
        logger.info(f"  Max drawdown: {metrics['max_drawdown']:.2%}")
    
    # Generate efficient frontier
    logger.info("\nGenerating efficient frontier:")
    frontier = optimizer.efficient_frontier(points=10)
    
    logger.info("Efficient frontier points:")
    for i, point in frontier.iterrows():
        logger.info(f"  Return: {point['return']:.2%}, Risk: {point['volatility']:.2%}, Sharpe: {point['sharpe_ratio']:.4f}")
    
    # Demonstrate rebalancing
    logger.info("\nDemonstrating portfolio rebalancing:")
    
    # Use max Sharpe weights as target
    target_results = optimizer.optimize("max_sharpe")
    target_weights = target_results["weights"]
    
    rebalancer = PortfolioRebalancer(target_weights, tolerance=0.05)
    
    # Create current portfolio with deviation from target
    total_value = 100000
    prices = {asset: 1.0 for asset in assets.keys()}  # Simplified prices
    
    # Current values with some deviation from target
    current_values = {}
    for asset, target in target_weights.items():
        # Add some random deviation
        deviation = np.random.uniform(-0.1, 0.1)
        current_values[asset] = total_value * (target + deviation)
    
    # Ensure total value is correct
    current_sum = sum(current_values.values())
    current_values = {k: v * total_value / current_sum for k, v in current_values.items()}
    
    # Check if rebalance needed
    rebalance_needed = rebalancer.check_rebalance_needed(current_values)
    logger.info(f"Rebalance needed: {rebalance_needed}")
    
    # Calculate rebalancing trades
    trades = rebalancer.calculate_rebalance_trades(current_values, prices)
    
    logger.info("Rebalancing trades:")
    for asset, units in trades.items():
        logger.info(f"  {asset}: {'Buy' if units > 0 else 'Sell'} {abs(units):.2f} units (${abs(units * prices[asset]):.2f})")
    
    return {
        'optimizer': optimizer,
        'frontier': frontier,
        'rebalancer': rebalancer
    }


async def main():
    """Run the demonstration"""
    logger.info("Starting MercurioAI Phase 2 Demonstration")
    
    try:
        # Demo transformer strategy
        transformer_results = await demo_transformer_strategy()
        
        # Demo risk management
        risk_results = await demo_risk_management()
        
        # Demo portfolio optimization
        optimization_results = await demo_portfolio_optimization()
        
        # Complete demo message
        logger.info("="*40)
        logger.info("MercurioAI Phase 2 Demonstration Completed Successfully")
        logger.info("="*40)
        
    except Exception as e:
        logger.error(f"Error during demonstration: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
