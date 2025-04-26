"""
Mercurio AI - Strategy Simulator

This script runs simulations for all available trading strategies on various assets
(stocks and cryptocurrencies) over a specified period and generates a comparison report.
"""
import os
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import strategies
from app.strategies.moving_average import MovingAverageStrategy
from app.strategies.lstm_predictor import LSTMPredictorStrategy
try:
    from app.strategies.llm_strategy import LLMStrategy
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    logger.warning("LLM Strategy not available")

try:
    from app.strategies.msi_strategy import MSIStrategy
    MSI_AVAILABLE = True
except ImportError:
    MSI_AVAILABLE = False
    logger.warning("MSI Strategy not available")

try:
    from app.strategies.transformer_strategy import TransformerStrategy
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False
    logger.warning("Transformer Strategy not available")

class StrategySimulator:
    """
    Simulates multiple trading strategies on various assets and generates a comparison report.
    """
    
    def __init__(self, initial_capital=2000):
        """
        Initialize the strategy simulator.
        
        Args:
            initial_capital: Initial capital for each strategy backtest ($)
        """
        self.initial_capital = initial_capital
        self.strategies = {}
        self.stock_symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA']
        self.crypto_symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'DOT-USD']
        self.results = {}
        
    def initialize_strategies(self):
        """Initialize all available trading strategies."""
        self.strategies = {
            "MovingAverage": MovingAverageStrategy(short_window=20, long_window=50),
            "MovingAverage_ML": MovingAverageStrategy(short_window=20, long_window=50, use_ml=True),
            "LSTM": LSTMPredictorStrategy(sequence_length=60, prediction_horizon=1)
        }
        
        # Add optional strategies if available
        if LLM_AVAILABLE:
            self.strategies["LLM"] = LLMStrategy()
        
        if MSI_AVAILABLE:
            self.strategies["MSI"] = MSIStrategy()
        
        if TRANSFORMER_AVAILABLE:
            self.strategies["Transformer"] = TransformerStrategy()
            
        logger.info(f"Initialized {len(self.strategies)} strategies")
        
    async def run_simulation(self, months=1):
        """
        Run backtests for all strategies on all symbols.
        
        Args:
            months: Number of months of historical data to use
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30*months)
        
        logger.info(f"Running simulations from {start_date} to {end_date}")
        
        all_symbols = self.stock_symbols + self.crypto_symbols
        
        for symbol in all_symbols:
            logger.info(f"Processing {symbol}")
            symbol_results = {}
            
            for strategy_name, strategy in self.strategies.items():
                logger.info(f"Running {strategy_name} on {symbol}")
                
                try:
                    # Load data
                    data = await strategy.load_data(symbol, start_date, end_date)
                    
                    # Preprocess data
                    processed_data = await strategy.preprocess_data(data)
                    
                    # If strategy requires training, train it
                    if hasattr(strategy, 'train') and not strategy.is_trained:
                        await strategy.train(processed_data)
                    
                    # Run backtest
                    backtest_results = await strategy.backtest(processed_data, initial_capital=self.initial_capital)
                    
                    # Store results
                    symbol_results[strategy_name] = backtest_results
                    
                except Exception as e:
                    logger.error(f"Error running {strategy_name} on {symbol}: {str(e)}")
                    symbol_results[strategy_name] = {"error": str(e)}
            
            self.results[symbol] = symbol_results
            
        logger.info("All simulations completed")
        
    def generate_report(self, output_dir="reports"):
        """
        Generate a comparative report of all strategy performances.
        
        Args:
            output_dir: Directory to save the report files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data for the comparison table
        comparison_data = []
        
        for symbol in self.results:
            for strategy_name, results in self.results[symbol].items():
                if "error" in results:
                    continue
                
                row = {
                    "Symbol": symbol,
                    "Strategy": strategy_name,
                    "Initial Capital": f"${self.initial_capital:.2f}",
                    "Final Capital": f"${results.get('final_capital', 0):.2f}",
                    "Total Return": f"{results.get('total_return', 0) * 100:.2f}%",
                    "Annualized Return": f"{results.get('annualized_return', 0) * 100:.2f}%",
                    "Sharpe Ratio": f"{results.get('sharpe_ratio', 0):.2f}",
                    "Max Drawdown": f"{results.get('max_drawdown', 0) * 100:.2f}%",
                    "Trades": results.get('trades', 0)
                }
                comparison_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(comparison_data)
        
        # Save to CSV
        csv_path = os.path.join(output_dir, "strategy_comparison.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved comparison data to {csv_path}")
        
        # Generate HTML report
        html_path = os.path.join(output_dir, "strategy_comparison.html")
        html_content = self._generate_html_report(df)
        with open(html_path, 'w') as f:
            f.write(html_content)
        logger.info(f"Saved HTML report to {html_path}")
        
        # Generate performance charts
        self._generate_performance_charts(output_dir)
        
        # Print summary to console
        print("\n" + "="*80)
        print("STRATEGY SIMULATION RESULTS")
        print("="*80)
        print(tabulate(df, headers='keys', tablefmt='fancy_grid'))
        print("\n" + "="*80)
        print(f"Full reports saved to: {output_dir}")
        print("="*80 + "\n")
        
    def _generate_html_report(self, df):
        """Generate an HTML report from the results DataFrame."""
        # Convert DataFrame to HTML table
        table_html = df.to_html(classes='dataframe', index=False)
        
        # Create a simple HTML document
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Mercurio AI Strategy Comparison</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                h1, h2 {{
                    color: #2c3e50;
                }}
                .container {{
                    background-color: white;
                    padding: 20px;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                }}
                .dataframe {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                }}
                .dataframe th, .dataframe td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                .dataframe th {{
                    background-color: #2c3e50;
                    color: white;
                }}
                .dataframe tr:nth-child(even) {{
                    background-color: #f2f2f2;
                }}
                .dataframe tr:hover {{
                    background-color: #e9f7ef;
                }}
                .summary {{
                    background-color: #eaf2f8;
                    padding: 15px;
                    border-radius: 5px;
                    margin-top: 20px;
                }}
            </style>
        </head>
        <body>
            <h1>Mercurio AI Trading Strategy Comparison</h1>
            
            <div class="container">
                <h2>Performance Metrics</h2>
                {table_html}
            </div>
            
            <div class="container summary">
                <h2>Summary</h2>
                <p>This report shows the performance of various trading strategies on both stocks and cryptocurrencies.</p>
                <p>Initial investment for each strategy: <strong>${self.initial_capital}</strong></p>
                <p>Test period: Last month</p>
            </div>
            
            <div class="container">
                <h2>Interpretation Guide</h2>
                <ul>
                    <li><strong>Total Return</strong>: Overall percentage return for the period.</li>
                    <li><strong>Annualized Return</strong>: Return scaled to a yearly rate.</li>
                    <li><strong>Sharpe Ratio</strong>: Risk-adjusted return (higher is better).</li>
                    <li><strong>Max Drawdown</strong>: Largest percentage drop from peak to trough.</li>
                    <li><strong>Trades</strong>: Number of trades executed during the period.</li>
                </ul>
            </div>
            
            <footer style="text-align: center; margin-top: 50px; color: #7f8c8d;">
                Generated by Mercurio AI Strategy Simulator on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </footer>
        </body>
        </html>
        """
        return html
        
    def _generate_performance_charts(self, output_dir):
        """Generate performance comparison charts."""
        # Set up the style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("viridis")
        
        # Create a figure for returns comparison
        plt.figure(figsize=(12, 8))
        
        # Extract total returns for plotting
        symbols = []
        strategy_names = []
        returns = []
        
        for symbol in self.results:
            for strategy_name, results in self.results[symbol].items():
                if "error" in results:
                    continue
                symbols.append(symbol)
                strategy_names.append(strategy_name)
                returns.append(results.get('total_return', 0) * 100)  # Convert to percentage
        
        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            'Symbol': symbols,
            'Strategy': strategy_names,
            'Return (%)': returns
        })
        
        # Plot returns by strategy and symbol
        plt.figure(figsize=(14, 10))
        chart = sns.barplot(x='Symbol', y='Return (%)', hue='Strategy', data=plot_df)
        chart.set_title('Total Returns by Strategy and Symbol', fontsize=16)
        chart.set_xlabel('Symbol', fontsize=14)
        chart.set_ylabel('Return (%)', fontsize=14)
        plt.legend(title='Strategy', title_fontsize=12, fontsize=10, loc='best')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the chart
        returns_chart_path = os.path.join(output_dir, "returns_comparison.png")
        plt.savefig(returns_chart_path, dpi=300)
        plt.close()
        
        logger.info(f"Saved returns comparison chart to {returns_chart_path}")
        
        # Create a risk-return scatter plot
        plt.figure(figsize=(12, 8))
        
        # Extract Sharpe ratios and max drawdowns
        sharpe_ratios = []
        max_drawdowns = []
        
        for symbol in self.results:
            for strategy_name, results in self.results[symbol].items():
                if "error" in results:
                    continue
                sharpe_ratios.append(results.get('sharpe_ratio', 0))
                max_drawdowns.append(results.get('max_drawdown', 0) * 100)  # Convert to percentage
        
        # Create DataFrame for plotting
        risk_return_df = pd.DataFrame({
            'Symbol': symbols,
            'Strategy': strategy_names,
            'Sharpe Ratio': sharpe_ratios,
            'Max Drawdown (%)': max_drawdowns
        })
        
        # Plot risk-return scatter
        plt.figure(figsize=(12, 8))
        chart = sns.scatterplot(
            x='Max Drawdown (%)', 
            y='Sharpe Ratio', 
            hue='Strategy', 
            style='Symbol',
            s=100, 
            data=risk_return_df
        )
        
        chart.set_title('Risk-Return Profile by Strategy', fontsize=16)
        chart.set_xlabel('Risk (Max Drawdown %)', fontsize=14)
        chart.set_ylabel('Return (Sharpe Ratio)', fontsize=14)
        plt.legend(title='Strategy', title_fontsize=12, fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the chart
        risk_return_chart_path = os.path.join(output_dir, "risk_return_profile.png")
        plt.savefig(risk_return_chart_path, dpi=300)
        plt.close()
        
        logger.info(f"Saved risk-return profile chart to {risk_return_chart_path}")

async def main():
    """Main entry point for the strategy simulator."""
    # Create simulator instance
    simulator = StrategySimulator(initial_capital=2000)
    
    # Initialize strategies
    simulator.initialize_strategies()
    
    # Run simulations
    await simulator.run_simulation(months=1)
    
    # Generate comparison report
    simulator.generate_report()

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
