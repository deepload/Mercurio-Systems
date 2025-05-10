"""
Options Strategy Performance Dashboard

This module provides visualization and analysis tools for options trading strategies 
performance via a Streamlit dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import glob


class OptionsPerformanceVisualization:
    """Provides visualization methods for options strategies performance."""
    
    @staticmethod
    def plot_equity_curve(equity_data: List[Dict[str, Any]], 
                         title: str = "Equity Curve") -> go.Figure:
        """
        Plot the equity curve over time.
        
        Args:
            equity_data: List of dictionaries containing date and equity values
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(equity_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Calculate drawdown
        df['previous_peak'] = df['equity'].cummax()
        df['drawdown'] = (df['equity'] - df['previous_peak']) / df['previous_peak'] * 100
        
        # Create subplot with shared x-axis
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.1,
                           subplot_titles=(title, "Drawdown (%)"),
                           row_heights=[0.7, 0.3])
        
        # Add equity curve
        fig.add_trace(
            go.Scatter(
                x=df['date'], 
                y=df['equity'],
                mode='lines',
                name='Equity',
                line=dict(color='green', width=2)
            ),
            row=1, col=1
        )
        
        # Add drawdown
        fig.add_trace(
            go.Scatter(
                x=df['date'], 
                y=df['drawdown'],
                mode='lines',
                name='Drawdown',
                line=dict(color='red', width=1.5),
                fill='tozeroy'
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template="plotly_white"
        )
        
        # Update y-axis for drawdown
        fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        
        return fig
    
    @staticmethod
    def plot_trade_distribution(trades: List[Dict[str, Any]]) -> go.Figure:
        """
        Plot the distribution of trade results.
        
        Args:
            trades: List of trade dictionaries with profit/loss data
            
        Returns:
            Plotly figure object
        """
        # Extract profit/loss values and convert to DataFrame
        trade_data = pd.DataFrame(trades)
        
        if 'profit_loss' not in trade_data.columns:
            return go.Figure()  # Return empty figure if no profit_loss data
        
        # Create histogram of trade P&L
        fig = px.histogram(
            trade_data, 
            x='profit_loss',
            nbins=20,
            color_discrete_sequence=['lightgreen' if x > 0 else 'lightcoral' 
                                    for x in trade_data['profit_loss']],
            labels={'profit_loss': 'Profit/Loss ($)'},
            title="Trade Profit/Loss Distribution"
        )
        
        # Add a line at zero
        fig.add_shape(
            type="line",
            x0=0, y0=0, x1=0, y1=1,
            yref="paper",
            line=dict(color="black", width=2, dash="dot")
        )
        
        fig.update_layout(
            showlegend=False,
            xaxis_title="Profit/Loss ($)",
            yaxis_title="Number of Trades",
            template="plotly_white"
        )
        
        return fig
    
    @staticmethod
    def plot_strategy_comparison(reports: List[Dict[str, Any]]) -> go.Figure:
        """
        Plot a comparison of different strategies' performance metrics.
        
        Args:
            reports: List of strategy backtest report dictionaries
            
        Returns:
            Plotly figure object
        """
        if not reports:
            return go.Figure()
            
        # Extract key metrics for comparison
        comparison_data = []
        
        for report in reports:
            comparison_data.append({
                'strategy': report.get('strategy', 'Unknown'),
                'total_return_pct': report.get('total_return_pct', 0),
                'win_rate': report.get('win_rate', 0) * 100,  # Convert to percentage
                'max_drawdown': report.get('max_drawdown_pct', 0),
                'sharpe_ratio': report.get('sharpe_ratio', 0),
                'total_trades': report.get('total_trades', 0)
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('total_return_pct', ascending=False)
        
        # Create subplot with multiple metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Total Return (%)", "Win Rate (%)", 
                           "Max Drawdown (%)", "Sharpe Ratio"),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                  [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Add traces for each metric
        fig.add_trace(
            go.Bar(x=df['strategy'], y=df['total_return_pct'], name='Total Return',
                 marker_color='green'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=df['strategy'], y=df['win_rate'], name='Win Rate',
                 marker_color='blue'),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(x=df['strategy'], y=df['max_drawdown'], name='Max Drawdown',
                 marker_color='red'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(x=df['strategy'], y=df['sharpe_ratio'], name='Sharpe Ratio',
                 marker_color='purple'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=600,
            title="Strategy Comparison",
            showlegend=False,
            template="plotly_white"
        )
        
        return fig
    
    @staticmethod
    def plot_monthly_returns(equity_data: List[Dict[str, Any]]) -> go.Figure:
        """
        Plot monthly returns as a heatmap.
        
        Args:
            equity_data: List of dictionaries containing date and equity values
            
        Returns:
            Plotly figure object
        """
        # Convert to DataFrame
        df = pd.DataFrame(equity_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Calculate daily returns
        df['daily_return'] = df['equity'].pct_change()
        
        # Extract month and year
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        
        # Calculate monthly returns
        monthly_returns = df.groupby(['year', 'month'])['daily_return'].apply(
            lambda x: (1 + x).prod() - 1
        ).reset_index()
        
        # Create a pivot table for the heatmap
        pivot_table = monthly_returns.pivot(index='month', columns='year', values='daily_return')
        
        # Map month numbers to month names
        month_names = {
            1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
        }
        pivot_table.index = pivot_table.index.map(month_names)
        
        # Create heatmap
        fig = px.imshow(
            pivot_table,
            labels=dict(x="Year", y="Month", color="Return"),
            x=pivot_table.columns,
            y=pivot_table.index,
            aspect="auto",
            color_continuous_scale='RdYlGn',  # Red for negative, green for positive
            title="Monthly Returns Heatmap"
        )
        
        # Add text annotations with return values
        annotations = []
        for i, month in enumerate(pivot_table.index):
            for j, year in enumerate(pivot_table.columns):
                value = pivot_table.iloc[i, j]
                if not pd.isna(value):
                    annotations.append(
                        dict(
                            x=year,
                            y=month,
                            text=f"{value:.2%}",
                            showarrow=False,
                            font=dict(
                                color="black" if abs(value) < 0.1 else "white"
                            )
                        )
                    )
        
        fig.update_layout(annotations=annotations)
        
        # Update layout
        fig.update_layout(
            height=400,
            template="plotly_white"
        )
        
        return fig
    
    @staticmethod
    def plot_drawdown_periods(equity_data: List[Dict[str, Any]], 
                             threshold: float = -0.10) -> go.Figure:
        """
        Plot major drawdown periods.
        
        Args:
            equity_data: List of dictionaries containing date and equity values
            threshold: Drawdown threshold to highlight (negative percentage)
            
        Returns:
            Plotly figure object
        """
        # Convert to DataFrame
        df = pd.DataFrame(equity_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Calculate drawdown
        df['previous_peak'] = df['equity'].cummax()
        df['drawdown'] = (df['equity'] - df['previous_peak']) / df['previous_peak']
        
        # Find drawdown periods
        df['is_drawdown'] = df['drawdown'] <= threshold
        df['drawdown_group'] = (df['is_drawdown'] != df['is_drawdown'].shift()).cumsum()
        
        # Keep only significant drawdown periods
        drawdown_periods = []
        for _, group in df[df['is_drawdown']].groupby('drawdown_group'):
            if len(group) > 0:
                start_date = group['date'].min()
                end_date = group['date'].max()
                max_drawdown = group['drawdown'].min()
                recovery_date = None
                
                # Find recovery date if it exists
                if end_date < df['date'].max():
                    recovery_idx = df[df['date'] > end_date]['equity'].gt(
                        df.loc[df['date'] == end_date, 'previous_peak'].values[0]
                    ).idxmax()
                    if recovery_idx in df.index:
                        recovery_date = df.loc[recovery_idx, 'date']
                
                drawdown_periods.append({
                    'start_date': start_date,
                    'end_date': end_date,
                    'recovery_date': recovery_date,
                    'max_drawdown': max_drawdown
                })
        
        # Plot equity curve with drawdown periods highlighted
        fig = go.Figure()
        
        # Add equity curve
        fig.add_trace(
            go.Scatter(
                x=df['date'], 
                y=df['equity'],
                mode='lines',
                name='Equity',
                line=dict(color='blue', width=2)
            )
        )
        
        # Highlight drawdown periods
        colors = px.colors.qualitative.Plotly
        for i, period in enumerate(drawdown_periods):
            color = colors[i % len(colors)]
            
            # Highlight drawdown period
            fig.add_vrect(
                x0=period['start_date'],
                x1=period['end_date'],
                fillcolor=color,
                opacity=0.2,
                layer="below",
                line_width=0,
                annotation_text=f"{period['max_drawdown']:.2%}",
                annotation_position="top left"
            )
            
            # Add recovery period if available
            if period['recovery_date'] is not None:
                fig.add_vrect(
                    x0=period['end_date'],
                    x1=period['recovery_date'],
                    fillcolor=color,
                    opacity=0.1,
                    layer="below",
                    line_width=0,
                    annotation_text="Recovery",
                    annotation_position="top right"
                )
        
        # Update layout
        fig.update_layout(
            title="Major Drawdown Periods",
            xaxis_title="Date",
            yaxis_title="Equity ($)",
            template="plotly_white",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    @staticmethod
    def plot_rolling_performance(equity_data: List[Dict[str, Any]], 
                                window: int = 30) -> go.Figure:
        """
        Plot rolling performance metrics.
        
        Args:
            equity_data: List of dictionaries containing date and equity values
            window: Rolling window size in days
            
        Returns:
            Plotly figure object
        """
        # Convert to DataFrame
        df = pd.DataFrame(equity_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Calculate daily returns
        df['daily_return'] = df['equity'].pct_change()
        
        # Calculate rolling metrics
        df['rolling_return'] = (1 + df['daily_return']).rolling(window).apply(
            lambda x: x.prod() - 1, raw=True
        )
        
        df['rolling_volatility'] = df['daily_return'].rolling(window).std() * np.sqrt(252)
        
        df['rolling_sharpe'] = (df['rolling_return'] / window * 252) / df['rolling_volatility']
        
        # Create subplot with shared x-axis
        fig = make_subplots(
            rows=3, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                f"{window}-Day Rolling Return", 
                f"{window}-Day Rolling Volatility",
                f"{window}-Day Rolling Sharpe Ratio"
            ),
            row_heights=[0.33, 0.33, 0.33]
        )
        
        # Add rolling return
        fig.add_trace(
            go.Scatter(
                x=df['date'], 
                y=df['rolling_return'] * 100,  # Convert to percentage
                mode='lines',
                name='Rolling Return',
                line=dict(color='green', width=1.5)
            ),
            row=1, col=1
        )
        
        # Add rolling volatility
        fig.add_trace(
            go.Scatter(
                x=df['date'], 
                y=df['rolling_volatility'] * 100,  # Convert to percentage
                mode='lines',
                name='Rolling Volatility',
                line=dict(color='red', width=1.5)
            ),
            row=2, col=1
        )
        
        # Add rolling Sharpe ratio
        fig.add_trace(
            go.Scatter(
                x=df['date'], 
                y=df['rolling_sharpe'],
                mode='lines',
                name='Rolling Sharpe',
                line=dict(color='blue', width=1.5)
            ),
            row=3, col=1
        )
        
        # Add reference line for Sharpe
        fig.add_shape(
            type="line",
            x0=df['date'].min(), y0=1, x1=df['date'].max(), y1=1,
            line=dict(color="black", width=1, dash="dot"),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            template="plotly_white",
            showlegend=False
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
        fig.update_yaxes(title_text="Sharpe Ratio", row=3, col=1)
        
        return fig


class OptionsBacktestReportAnalyzer:
    """Analyzes options backtest reports and extracts meaningful metrics."""
    
    def __init__(self, report_file_path: str):
        """
        Initialize with the path to a backtest report file.
        
        Args:
            report_file_path: Path to the JSON backtest report file
        """
        self.report_file_path = report_file_path
        self.report_data = self._load_report()
        
    def _load_report(self) -> Dict[str, Any]:
        """Load the backtest report from the file."""
        try:
            with open(self.report_file_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            st.error(f"Error loading backtest report: {e}")
            return {}
    
    def get_overview_metrics(self) -> Dict[str, Any]:
        """Extract overview metrics from the backtest report."""
        if not self.report_data:
            return {}
            
        metrics = {
            'strategy': self.report_data.get('strategy', 'Unknown'),
            'initial_capital': self.report_data.get('initial_capital', 0),
            'final_equity': self.report_data.get('final_equity', 0),
            'total_return': self.report_data.get('total_return', 0),
            'total_return_pct': self.report_data.get('total_return_pct', 0),
            'annualized_return_pct': self.report_data.get('annualized_return_pct', 0),
            'total_trades': self.report_data.get('total_trades', 0),
            'profitable_trades': self.report_data.get('profitable_trades', 0),
            'losing_trades': self.report_data.get('losing_trades', 0),
            'win_rate': self.report_data.get('win_rate', 0),
            'max_drawdown_pct': self.report_data.get('max_drawdown_pct', 0),
            'sharpe_ratio': self.report_data.get('sharpe_ratio', 0),
            'sortino_ratio': self.report_data.get('sortino_ratio', 0),
            'execution_time_seconds': self.report_data.get('execution_time_seconds', 0)
        }
        
        # Calculate additional metrics if not present
        if 'win_rate' not in self.report_data and metrics['total_trades'] > 0:
            metrics['win_rate'] = metrics['profitable_trades'] / metrics['total_trades']
            
        return metrics
    
    def get_trade_data(self) -> List[Dict[str, Any]]:
        """Extract trade data from the backtest report."""
        return self.report_data.get('trades', [])
    
    def get_equity_curve_data(self) -> List[Dict[str, Any]]:
        """Extract equity curve data from the backtest report."""
        return self.report_data.get('equity_curve', [])
    
    def get_strategy_params(self) -> Dict[str, Any]:
        """Extract strategy parameters from the backtest report."""
        return self.report_data.get('strategy_params', {})
    
    def calculate_additional_metrics(self) -> Dict[str, Any]:
        """Calculate additional performance metrics not in the original report."""
        trades = self.get_trade_data()
        equity_curve = self.get_equity_curve_data()
        
        additional_metrics = {}
        
        # Skip if insufficient data
        if not trades or not equity_curve:
            return additional_metrics
            
        # Convert to DataFrame for analysis
        trade_df = pd.DataFrame(trades)
        equity_df = pd.DataFrame(equity_curve)
        
        if 'profit_loss' in trade_df.columns:
            # Calculate average profit/loss
            additional_metrics['avg_profit'] = trade_df[trade_df['profit_loss'] > 0]['profit_loss'].mean()
            additional_metrics['avg_loss'] = trade_df[trade_df['profit_loss'] < 0]['profit_loss'].mean()
            
            # Calculate profit factor
            total_profit = trade_df[trade_df['profit_loss'] > 0]['profit_loss'].sum()
            total_loss = abs(trade_df[trade_df['profit_loss'] < 0]['profit_loss'].sum())
            additional_metrics['profit_factor'] = total_profit / total_loss if total_loss != 0 else float('inf')
            
            # Calculate percentage of profitable days
            if 'exit_date' in trade_df.columns:
                trade_df['exit_date'] = pd.to_datetime(trade_df['exit_date'])
                days = trade_df['exit_date'].dt.date.nunique()
                profitable_days = trade_df[trade_df['profit_loss'] > 0]['exit_date'].dt.date.nunique()
                additional_metrics['profitable_days_pct'] = profitable_days / days if days > 0 else 0
        
        # Calculate average trade duration if data available
        if 'entry_date' in trade_df.columns and 'exit_date' in trade_df.columns:
            trade_df['entry_date'] = pd.to_datetime(trade_df['entry_date'])
            trade_df['exit_date'] = pd.to_datetime(trade_df['exit_date'])
            trade_df['duration'] = (trade_df['exit_date'] - trade_df['entry_date']).dt.days
            additional_metrics['avg_trade_duration_days'] = trade_df['duration'].mean()
        
        return additional_metrics


def load_backtest_reports(directory: str) -> List[str]:
    """
    Load all backtest report files from a directory.
    
    Args:
        directory: Directory path to search for report files
    
    Returns:
        List of file paths to backtest reports
    """
    if not os.path.exists(directory):
        return []
        
    # Find all JSON files in the directory
    report_files = glob.glob(os.path.join(directory, "*.json"))
    
    # Filter to include only valid backtest reports
    valid_reports = []
    for file_path in report_files:
        try:
            with open(file_path, 'r') as f:
                report = json.load(f)
                # Check if it has the essential components of a backtest report
                if ('strategy' in report and 'equity_curve' in report and 
                    'trades' in report):
                    valid_reports.append(file_path)
        except:
            continue
            
    return valid_reports


def format_metrics_display(metrics: Dict[str, Any]) -> None:
    """
    Format and display metrics in a clean layout using Streamlit.
    
    Args:
        metrics: Dictionary of metrics to display
    """
    # Create columns for metric display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Return", f"{metrics.get('total_return_pct', 0):.2f}%")
        st.metric("Win Rate", f"{metrics.get('win_rate', 0) * 100:.2f}%")
        st.metric("Total Trades", metrics.get('total_trades', 0))
        
    with col2:
        st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
        st.metric("Max Drawdown", f"{metrics.get('max_drawdown_pct', 0):.2f}%")
        st.metric("Profit Factor", f"{metrics.get('profit_factor', 0):.2f}")
        
    with col3:
        st.metric("Annualized Return", f"{metrics.get('annualized_return_pct', 0):.2f}%")
        st.metric("Avg Profit", f"${metrics.get('avg_profit', 0):.2f}")
        st.metric("Avg Loss", f"${metrics.get('avg_loss', 0):.2f}")


def main():
    """Main dashboard function."""
    st.set_page_config(
        page_title="Options Strategy Performance Dashboard",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("Options Strategy Performance Dashboard")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Set path for backtest reports
    default_reports_path = os.path.join(os.getcwd(), "backtest_reports")
    reports_path = st.sidebar.text_input(
        "Backtest Reports Directory", 
        value=default_reports_path
    )
    
    # Find report files
    report_files = load_backtest_reports(reports_path)
    
    if not report_files:
        st.warning(f"No backtest reports found in directory: {reports_path}")
        st.info("Please run backtests first or change the directory path.")
        return
        
    # File selection dropdown
    selected_file = st.sidebar.selectbox(
        "Select Backtest Report",
        options=report_files,
        format_func=lambda x: os.path.basename(x)
    )
    
    # Load and analyze the selected report
    analyzer = OptionsBacktestReportAnalyzer(selected_file)
    overview_metrics = analyzer.get_overview_metrics()
    additional_metrics = analyzer.calculate_additional_metrics()
    
    # Combine all metrics
    metrics = {**overview_metrics, **additional_metrics}
    
    # Display strategy name and parameters
    st.header(f"Strategy: {metrics.get('strategy', 'Unknown')}")
    
    with st.expander("Strategy Parameters", expanded=False):
        st.json(analyzer.get_strategy_params())
    
    # Display key metrics
    st.subheader("Performance Metrics")
    format_metrics_display(metrics)
    
    # Visualization
    viz = OptionsPerformanceVisualization()
    equity_data = analyzer.get_equity_curve_data()
    trade_data = analyzer.get_trade_data()
    
    # Equity curve
    st.subheader("Equity Curve and Drawdown")
    equity_fig = viz.plot_equity_curve(equity_data)
    st.plotly_chart(equity_fig, use_container_width=True)
    
    # Two visualizations side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Trade Distribution")
        trade_dist_fig = viz.plot_trade_distribution(trade_data)
        st.plotly_chart(trade_dist_fig, use_container_width=True)
        
    with col2:
        st.subheader("Monthly Returns")
        monthly_fig = viz.plot_monthly_returns(equity_data)
        st.plotly_chart(monthly_fig, use_container_width=True)
    
    # Drawdown analysis
    st.subheader("Drawdown Analysis")
    drawdown_fig = viz.plot_drawdown_periods(equity_data)
    st.plotly_chart(drawdown_fig, use_container_width=True)
    
    # Rolling performance
    st.subheader("Rolling Performance Metrics")
    window_size = st.slider("Rolling Window (days)", min_value=10, max_value=90, value=30, step=5)
    rolling_fig = viz.plot_rolling_performance(equity_data, window=window_size)
    st.plotly_chart(rolling_fig, use_container_width=True)
    
    # Trade table
    st.subheader("Trade History")
    
    if trade_data:
        trade_df = pd.DataFrame(trade_data)
        # Convert datetime columns if present
        for col in ['entry_date', 'exit_date']:
            if col in trade_df.columns:
                trade_df[col] = pd.to_datetime(trade_df[col])
                
        st.dataframe(trade_df)
    else:
        st.info("No trade data available for this backtest.")
    
    # Strategy comparison
    st.sidebar.header("Strategy Comparison")
    
    if st.sidebar.button("Compare All Strategies"):
        st.subheader("Strategy Comparison")
        all_reports = []
        
        for report_file in report_files[:10]:  # Limit to 10 to avoid visual clutter
            report_analyzer = OptionsBacktestReportAnalyzer(report_file)
            report_metrics = report_analyzer.get_overview_metrics()
            all_reports.append(report_metrics)
        
        comparison_fig = viz.plot_strategy_comparison(all_reports)
        st.plotly_chart(comparison_fig, use_container_width=True)


if __name__ == "__main__":
    main()
