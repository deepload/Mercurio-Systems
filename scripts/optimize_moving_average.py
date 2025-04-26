"""
Mercurio AI - Moving Average Strategy Optimizer

This script optimizes the parameters of the MovingAverage_ML strategy
to find the best settings for January 2025 market conditions.
"""
import os
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from itertools import product

# Import our data generator
from data_generator import load_market_data, generate_market_data

# Setup output directories
os.makedirs('reports/optimization', exist_ok=True)

async def optimize_moving_average_strategy(symbol, start_date, end_date, initial_capital=2000):
    """
    Test different parameter combinations for MovingAverage strategy.
    
    Args:
        symbol: Market symbol to test
        start_date: Start date for testing
        end_date: End date for testing
        initial_capital: Initial capital for each test
        
    Returns:
        DataFrame with optimization results
    """
    print(f"Optimizing MovingAverage strategy for {symbol}...")
    
    # Import strategy
    from app.strategies.moving_average import MovingAverageStrategy
    
    # Load or generate data
    data = load_market_data(symbol)
    if data is None:
        print(f"Generating data for {symbol}...")
        data = generate_market_data(symbol, start_date, end_date)
    
    # Ensure index is datetime
    if not isinstance(data.index, pd.DatetimeIndex):
        data = data.reset_index()
        data['date'] = pd.to_datetime(data['date'])
        data = data.set_index('date')
    
    # Parameters to test
    short_windows = [3, 5, 7, 10, 15]
    long_windows = [10, 15, 20, 30, 50]
    use_ml_options = [True, False]
    
    # Store results
    results = []
    
    # Test all combinations
    for short_window, long_window, use_ml in product(short_windows, long_windows, use_ml_options):
        # Skip invalid combinations
        if short_window >= long_window:
            continue
            
        try:
            print(f"  Testing short={short_window}, long={long_window}, ml={use_ml}")
            
            # Initialize strategy
            strategy = MovingAverageStrategy(
                short_window=short_window,
                long_window=long_window,
                use_ml=use_ml
            )
            
            # Preprocess data
            processed_data = await strategy.preprocess_data(data.copy())
            
            # Train if using ML
            if use_ml:
                await strategy.train(processed_data)
            
            # Run backtest
            backtest_results = await strategy.backtest(processed_data, initial_capital=initial_capital)
            
            # Extract key metrics
            result = {
                'symbol': symbol,
                'short_window': short_window,
                'long_window': long_window,
                'use_ml': use_ml,
                'initial_capital': initial_capital,
                'final_capital': backtest_results.get('final_capital', 0),
                'total_return': backtest_results.get('total_return', 0),
                'annualized_return': backtest_results.get('annualized_return', 0),
                'sharpe_ratio': backtest_results.get('sharpe_ratio', 0),
                'max_drawdown': backtest_results.get('max_drawdown', 0),
                'trades': backtest_results.get('trades', 0)
            }
            
            results.append(result)
            print(f"    Return: {result['total_return']*100:.2f}%, Sharpe: {result['sharpe_ratio']:.2f}")
            
        except Exception as e:
            print(f"  Error testing parameters: {e}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df

def create_optimization_report(results_df, symbol):
    """Generate a detailed report of optimization results."""
    # Sort by total return (descending)
    results_df = results_df.sort_values('total_return', ascending=False)
    
    # Save raw results
    results_df.to_csv(f'reports/optimization/{symbol}_optimization.csv', index=False)
    
    # Create display version
    display_df = results_df.copy()
    display_df['total_return'] = display_df['total_return'].apply(lambda x: f"{x*100:.2f}%")
    display_df['annualized_return'] = display_df['annualized_return'].apply(lambda x: f"{x*100:.2f}%")
    display_df['max_drawdown'] = display_df['max_drawdown'].apply(lambda x: f"{x*100:.2f}%")
    display_df['final_capital'] = display_df['final_capital'].apply(lambda x: f"${x:.2f}")
    
    # Print table of top 10 results
    print(f"\nTop 10 Parameter Combinations for {symbol}:")
    print(tabulate(
        display_df.head(10), 
        headers=[col.replace('_', ' ').title() for col in display_df.columns],
        tablefmt='grid'
    ))
    
    # Visualize results
    # 1. Heatmap of returns by window sizes (for ML=True)
    ml_results = results_df[results_df['use_ml'] == True].copy()
    if not ml_results.empty:
        try:
            pivot_df = ml_results.pivot_table(
                index='short_window', 
                columns='long_window', 
                values='total_return',
                aggfunc='mean'
            )
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                pivot_df * 100,  # Convert to percentage
                annot=True, 
                fmt=".2f", 
                cmap="YlGnBu",
                linewidths=0.5,
                cbar_kws={'label': 'Return (%)'}
            )
            
            plt.title(f'Returns by Window Size (ML=True) for {symbol}')
            plt.tight_layout()
            plt.savefig(f'reports/optimization/{symbol}_ml_heatmap.png', dpi=300)
            plt.close()
        except Exception as e:
            print(f"Error creating ML heatmap: {e}")
    
    # 2. Comparison of ML vs. non-ML
    try:
        # Group by window sizes
        comparison_data = []
        for short_window in results_df['short_window'].unique():
            for long_window in results_df['long_window'].unique():
                if short_window >= long_window:
                    continue
                    
                subset = results_df[
                    (results_df['short_window'] == short_window) & 
                    (results_df['long_window'] == long_window)
                ]
                
                if len(subset) == 2:  # Both ML and non-ML exist
                    ml_return = subset[subset['use_ml'] == True]['total_return'].values[0]
                    non_ml_return = subset[subset['use_ml'] == False]['total_return'].values[0]
                    
                    comparison_data.append({
                        'Windows': f"{short_window}/{long_window}",
                        'ML Return': ml_return * 100,
                        'Non-ML Return': non_ml_return * 100,
                        'Difference': (ml_return - non_ml_return) * 100
                    })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('Difference', ascending=False)
            
            plt.figure(figsize=(12, 6))
            plt.bar(
                comparison_df['Windows'],
                comparison_df['Difference'],
                color=np.where(comparison_df['Difference'] > 0, 'green', 'red')
            )
            plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            plt.title(f'ML vs. Non-ML Return Difference (%) for {symbol}')
            plt.ylabel('Return Difference (%)')
            plt.xlabel('Window Sizes (Short/Long)')
            plt.xticks(rotation=45)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'reports/optimization/{symbol}_ml_comparison.png', dpi=300)
            plt.close()
        
    except Exception as e:
        print(f"Error creating ML comparison chart: {e}")
    
    # Return best parameters
    if not results_df.empty:
        best_params = results_df.iloc[0].to_dict()
        return best_params
    else:
        return None

async def optimize_for_all_symbols():
    """Run optimization for all key symbols."""
    # Define date range for January 2025
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 1, 31)
    
    # Define symbols to optimize
    stocks = ['AAPL', 'MSFT', 'GOOGL']
    cryptos = ['BTC-USD', 'ETH-USD']
    
    all_symbols = stocks + cryptos
    best_params = {}
    
    for symbol in all_symbols:
        # Run optimization
        results = await optimize_moving_average_strategy(symbol, start_date, end_date)
        
        if not results.empty:
            # Create report
            best = create_optimization_report(results, symbol)
            if best:
                best_params[symbol] = best
    
    # Summarize best parameters
    print("\n===== OPTIMIZATION SUMMARY =====")
    print("Best Parameters for Each Symbol:")
    
    for symbol, params in best_params.items():
        print(f"\n{symbol}:")
        print(f"  Short Window: {params['short_window']}")
        print(f"  Long Window: {params['long_window']}")
        print(f"  Use ML: {params['use_ml']}")
        print(f"  Return: {params['total_return']*100:.2f}%")
        print(f"  Sharpe Ratio: {params['sharpe_ratio']:.2f}")
    
    # Save best parameters to file
    with open('reports/optimization/best_parameters.txt', 'w') as f:
        f.write("Best Moving Average Parameters for January 2025:\n\n")
        for symbol, params in best_params.items():
            f.write(f"{symbol}:\n")
            f.write(f"  Short Window: {params['short_window']}\n")
            f.write(f"  Long Window: {params['long_window']}\n")
            f.write(f"  Use ML: {params['use_ml']}\n")
            f.write(f"  Return: {params['total_return']*100:.2f}%\n")
            f.write(f"  Sharpe Ratio: {params['sharpe_ratio']:.2f}\n\n")
    
    print("\nOptimization complete! Results saved to reports/optimization/")

async def main():
    """Main entry point."""
    print("\n===== MERCURIO AI STRATEGY OPTIMIZER =====\n")
    print("Optimizing MovingAverage strategy parameters for January 2025")
    print("=" * 50)
    
    await optimize_for_all_symbols()

if __name__ == "__main__":
    asyncio.run(main())
