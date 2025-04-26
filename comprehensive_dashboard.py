"""
Mercurio AI - Comprehensive Strategy Dashboard

Interactive dashboard for visualizing the results of the comprehensive 
strategy simulation across multiple timeframes, assets, and strategies.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime

st.set_page_config(
    page_title="Mercurio AI - Strategy Simulation Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style and theme
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #0066cc;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4d4d4d;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f5f5f5;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .insight-text {
        color: #0066cc;
        font-weight: bold;
    }
    .warning-text {
        color: #cc0000;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions
def load_simulation_data():
    """Load the simulation results data."""
    try:
        data_path = "reports/comprehensive/full_simulation_results.csv"
        if os.path.exists(data_path):
            data = pd.read_csv(data_path)
            return data
        else:
            st.error("Simulation results file not found. Please run the comprehensive_simulation.py script first.")
            return None
    except Exception as e:
        st.error(f"Error loading simulation data: {e}")
        return None

def format_percentage(value):
    """Format a number as a percentage with color."""
    color = "green" if value >= 0 else "red"
    return f"<span style='color:{color};'>{value:.2f}%</span>"

def create_return_heatmap(data, x_col, y_col, value_col):
    """Create a heatmap of returns."""
    pivot_data = data.pivot_table(
        index=y_col, 
        columns=x_col,
        values=value_col,
        aggfunc="mean"
    )
    
    # Create heatmap using plotly
    fig = px.imshow(
        pivot_data,
        labels=dict(x=x_col, y=y_col, color=value_col),
        x=pivot_data.columns,
        y=pivot_data.index,
        color_continuous_scale="RdYlGn",
        aspect="auto"
    )
    
    # Add annotations
    for y in range(len(pivot_data.index)):
        for x in range(len(pivot_data.columns)):
            fig.add_annotation(
                x=pivot_data.columns[x],
                y=pivot_data.index[y],
                text=f"{pivot_data.iloc[y, x]:.2f}%",
                showarrow=False,
                font=dict(color="black")
            )
    
    fig.update_layout(
        title=f"Average {value_col} by {y_col} and {x_col}",
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def create_comparison_chart(data, group_by, metric):
    """Create a comparison bar chart."""
    grouped_data = data.groupby(group_by)[metric].mean().reset_index()
    grouped_data = grouped_data.sort_values(metric, ascending=False)
    
    # Create bar chart
    fig = px.bar(
        grouped_data,
        x=group_by,
        y=metric,
        color=metric,
        color_continuous_scale="RdYlGn",
        text=grouped_data[metric].apply(lambda x: f"{x:.2f}%")
    )
    
    fig.update_layout(
        title=f"Average {metric} by {group_by}",
        xaxis_title=group_by,
        yaxis_title=metric,
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def create_scatter_plot(data, x_col, y_col, color_col):
    """Create a scatter plot."""
    fig = px.scatter(
        data,
        x=x_col,
        y=y_col,
        color=color_col,
        size="Trades",
        hover_data=["Asset", "Timeframe", "Initial Capital", "Final Capital"],
        title=f"{y_col} vs. {x_col} by {color_col}"
    )
    
    fig.update_layout(
        height=500,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

# Dashboard Layout
def main():
    # Dashboard title
    st.markdown("<h1 class='main-header'>Mercurio AI - Strategy Simulation Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Comprehensive analysis of trading strategies across multiple timeframes (Mar 2024 - Apr 2025)</p>", unsafe_allow_html=True)
    
    # Check if data exists
    data = load_simulation_data()
    if data is None:
        st.info("To generate simulation data, please run:")
        st.code("python comprehensive_simulation.py")
        return
    
    # Dashboard date
    try:
        # Get file modified date
        file_date = os.path.getmtime("reports/comprehensive/full_simulation_results.csv")
        modified_date = datetime.fromtimestamp(file_date).strftime('%Y-%m-%d %H:%M:%S')
        st.caption(f"Data last updated: {modified_date}")
    except:
        st.caption("Data last updated: Unknown")
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Strategy filter
    all_strategies = sorted(data["Strategy"].unique().tolist())
    selected_strategies = st.sidebar.multiselect(
        "Select Strategies",
        options=all_strategies,
        default=all_strategies
    )
    
    # Asset filter
    all_assets = sorted(data["Asset"].unique().tolist())
    selected_assets = st.sidebar.multiselect(
        "Select Assets",
        options=all_assets,
        default=all_assets
    )
    
    # Timeframe filter
    all_timeframes = sorted(data["Timeframe"].unique().tolist())
    selected_timeframes = st.sidebar.multiselect(
        "Select Timeframes",
        options=all_timeframes,
        default=all_timeframes
    )
    
    # Asset type filter
    all_asset_types = sorted(data["Asset Type"].unique().tolist())
    selected_asset_types = st.sidebar.multiselect(
        "Select Asset Types",
        options=all_asset_types,
        default=all_asset_types
    )
    
    # Apply filters
    filtered_data = data[
        (data["Strategy"].isin(selected_strategies)) &
        (data["Asset"].isin(selected_assets)) &
        (data["Timeframe"].isin(selected_timeframes)) &
        (data["Asset Type"].isin(selected_asset_types))
    ]
    
    if filtered_data.empty:
        st.warning("No data available with the selected filters. Please adjust your selections.")
        return
    
    # Summary metrics
    st.header("Performance Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_return = filtered_data["Total Return (%)"].mean()
        st.metric(
            "Average Return",
            f"{avg_return:.2f}%",
            delta=f"{avg_return - data['Total Return (%)'].mean():.2f}%" if not data.empty else None
        )
    
    with col2:
        avg_sharpe = filtered_data["Sharpe Ratio"].mean()
        st.metric(
            "Average Sharpe Ratio",
            f"{avg_sharpe:.2f}",
            delta=f"{avg_sharpe - data['Sharpe Ratio'].mean():.2f}" if not data.empty else None
        )
    
    with col3:
        avg_drawdown = filtered_data["Max Drawdown (%)"].mean()
        st.metric(
            "Average Max Drawdown",
            f"{avg_drawdown:.2f}%",
            delta=f"{data['Max Drawdown (%)'].mean() - avg_drawdown:.2f}%" if not data.empty else None,
            delta_color="inverse"
        )
    
    with col4:
        total_sims = len(filtered_data)
        st.metric(
            "Total Simulations",
            f"{total_sims}",
            delta=f"{total_sims - len(data)}" if not data.empty else None
        )
    
    # Top performers
    st.header("Top Performers")
    
    tabs = st.tabs(["Overall Top 10", "By Strategy", "By Timeframe", "By Asset Type"])
    
    with tabs[0]:
        top_performers = filtered_data.nlargest(10, "Total Return (%)")
        
        # Create table
        st.dataframe(
            top_performers[["Strategy", "Asset", "Timeframe", "Asset Type", "Total Return (%)", "Sharpe Ratio", "Max Drawdown (%)", "Trades"]],
            use_container_width=True,
            hide_index=True
        )
        
        # Create chart
        fig = px.bar(
            top_performers,
            x="Asset",
            y="Total Return (%)",
            color="Strategy",
            pattern_shape="Timeframe",
            barmode="group",
            height=400,
            labels={"Asset": "Asset", "Total Return (%)": "Total Return (%)"},
            hover_data=["Asset Type", "Sharpe Ratio", "Max Drawdown (%)"]
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        # Strategy comparison
        strategy_returns = filtered_data.groupby("Strategy")["Total Return (%)"].mean().reset_index()
        strategy_returns = strategy_returns.sort_values("Total Return (%)", ascending=False)
        
        # Create chart
        fig = px.bar(
            strategy_returns,
            x="Strategy",
            y="Total Return (%)",
            color="Total Return (%)",
            color_continuous_scale="RdYlGn",
            text=strategy_returns["Total Return (%)"].apply(lambda x: f"{x:.2f}%"),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show best combo for each strategy
        st.subheader("Best Asset Combinations by Strategy")
        
        strategy_best = pd.DataFrame()
        for strategy in selected_strategies:
            strategy_data = filtered_data[filtered_data["Strategy"] == strategy]
            if not strategy_data.empty:
                best_combo = strategy_data.loc[strategy_data["Total Return (%)"].idxmax()]
                strategy_best = pd.concat([strategy_best, pd.DataFrame([best_combo])], ignore_index=True)
        
        if not strategy_best.empty:
            st.dataframe(
                strategy_best[["Strategy", "Asset", "Timeframe", "Total Return (%)", "Sharpe Ratio", "Max Drawdown (%)"]],
                use_container_width=True,
                hide_index=True
            )
    
    with tabs[2]:
        # Timeframe comparison
        timeframe_returns = filtered_data.groupby("Timeframe")["Total Return (%)"].mean().reset_index()
        timeframe_returns = timeframe_returns.sort_values("Total Return (%)", ascending=False)
        
        # Create chart
        fig = px.bar(
            timeframe_returns,
            x="Timeframe",
            y="Total Return (%)",
            color="Total Return (%)",
            color_continuous_scale="RdYlGn",
            text=timeframe_returns["Total Return (%)"].apply(lambda x: f"{x:.2f}%"),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show best combo for each timeframe
        st.subheader("Best Strategy-Asset Combinations by Timeframe")
        
        timeframe_best = pd.DataFrame()
        for timeframe in selected_timeframes:
            timeframe_data = filtered_data[filtered_data["Timeframe"] == timeframe]
            if not timeframe_data.empty:
                best_combo = timeframe_data.loc[timeframe_data["Total Return (%)"].idxmax()]
                timeframe_best = pd.concat([timeframe_best, pd.DataFrame([best_combo])], ignore_index=True)
        
        if not timeframe_best.empty:
            st.dataframe(
                timeframe_best[["Timeframe", "Strategy", "Asset", "Total Return (%)", "Sharpe Ratio", "Max Drawdown (%)"]],
                use_container_width=True,
                hide_index=True
            )
    
    with tabs[3]:
        # Asset type comparison
        asset_type_returns = filtered_data.groupby("Asset Type")["Total Return (%)"].mean().reset_index()
        asset_type_returns = asset_type_returns.sort_values("Total Return (%)", ascending=False)
        
        # Create chart
        fig = px.bar(
            asset_type_returns,
            x="Asset Type",
            y="Total Return (%)",
            color="Total Return (%)",
            color_continuous_scale="RdYlGn",
            text=asset_type_returns["Total Return (%)"].apply(lambda x: f"{x:.2f}%"),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show best combo for each asset type
        st.subheader("Best Strategy-Timeframe Combinations by Asset Type")
        
        asset_type_best = pd.DataFrame()
        for asset_type in selected_asset_types:
            asset_type_data = filtered_data[filtered_data["Asset Type"] == asset_type]
            if not asset_type_data.empty:
                best_combo = asset_type_data.loc[asset_type_data["Total Return (%)"].idxmax()]
                asset_type_best = pd.concat([asset_type_best, pd.DataFrame([best_combo])], ignore_index=True)
        
        if not asset_type_best.empty:
            st.dataframe(
                asset_type_best[["Asset Type", "Strategy", "Asset", "Timeframe", "Total Return (%)", "Sharpe Ratio"]],
                use_container_width=True,
                hide_index=True
            )
    
    # Advanced Analysis
    st.header("Advanced Analysis")
    
    analysis_tabs = st.tabs(["Strategy x Timeframe", "Risk-Return Analysis", "Asset Performance", "Custom Analysis"])
    
    with analysis_tabs[0]:
        # Strategy x Timeframe heatmap
        st.subheader("Strategy Performance by Timeframe")
        
        heatmap_fig = create_return_heatmap(
            filtered_data, 
            "Timeframe", 
            "Strategy", 
            "Total Return (%)"
        )
        
        st.plotly_chart(heatmap_fig, use_container_width=True)
        
        # Insights
        st.markdown("<div class='insight-text'>Insights:</div>", unsafe_allow_html=True)
        
        # Calculate best timeframe for each strategy
        strategy_timeframe = filtered_data.pivot_table(
            index="Strategy", 
            columns="Timeframe",
            values="Total Return (%)",
            aggfunc="mean"
        )
        
        best_timeframes = {}
        for strategy in strategy_timeframe.index:
            best_tf = strategy_timeframe.loc[strategy].idxmax()
            best_return = strategy_timeframe.loc[strategy, best_tf]
            best_timeframes[strategy] = (best_tf, best_return)
        
        insights_text = "<ul>"
        for strategy, (timeframe, ret) in best_timeframes.items():
            insights_text += f"<li><b>{strategy}</b> performs best on <b>{timeframe}</b> timeframe with <span style='color:{'green' if ret >= 0 else 'red'}'>{ret:.2f}%</span> return</li>"
        insights_text += "</ul>"
        
        st.markdown(insights_text, unsafe_allow_html=True)
    
    with analysis_tabs[1]:
        # Risk-Return Analysis
        st.subheader("Risk-Return Profile")
        
        # Create scatter plot
        scatter_fig = create_scatter_plot(
            filtered_data,
            "Max Drawdown (%)",
            "Total Return (%)",
            "Strategy"
        )
        
        st.plotly_chart(scatter_fig, use_container_width=True)
        
        # Risk metrics comparison
        col1, col2 = st.columns(2)
        
        with col1:
            # Compare Sharpe ratios
            sharpe_fig = create_comparison_chart(
                filtered_data,
                "Strategy",
                "Sharpe Ratio"
            )
            st.plotly_chart(sharpe_fig, use_container_width=True)
        
        with col2:
            # Compare drawdowns
            drawdown_data = filtered_data.copy()
            drawdown_data["Max Drawdown (%)"] = -drawdown_data["Max Drawdown (%)"]  # Invert for better visualization
            
            drawdown_fig = create_comparison_chart(
                drawdown_data,
                "Strategy",
                "Max Drawdown (%)"
            )
            
            # Update y-axis title to show positive values but represent less drawdown as better
            drawdown_fig.update_layout(
                yaxis_title="Max Drawdown (%) [Lower is Better]"
            )
            
            st.plotly_chart(drawdown_fig, use_container_width=True)
    
    with analysis_tabs[2]:
        # Asset Performance
        st.subheader("Asset Performance Analysis")
        
        # Compare assets
        asset_fig = create_comparison_chart(
            filtered_data,
            "Asset",
            "Total Return (%)"
        )
        
        st.plotly_chart(asset_fig, use_container_width=True)
        
        # Best strategy for each asset
        st.subheader("Best Strategy for Each Asset")
        
        asset_strategy = filtered_data.pivot_table(
            index="Asset",
            columns="Strategy",
            values="Total Return (%)",
            aggfunc="mean"
        )
        
        asset_best = pd.DataFrame()
        for asset in asset_strategy.index:
            best_strategy = asset_strategy.loc[asset].idxmax()
            best_return = asset_strategy.loc[asset, best_strategy]
            
            # Get all data for this asset
            asset_data = filtered_data[(filtered_data["Asset"] == asset) & (filtered_data["Strategy"] == best_strategy)]
            
            if not asset_data.empty:
                # Find best timeframe
                best_idx = asset_data["Total Return (%)"].idxmax()
                best_row = asset_data.loc[best_idx]
                asset_best = pd.concat([asset_best, pd.DataFrame([best_row])], ignore_index=True)
        
        if not asset_best.empty:
            st.dataframe(
                asset_best[["Asset", "Strategy", "Timeframe", "Total Return (%)", "Sharpe Ratio", "Max Drawdown (%)"]],
                use_container_width=True,
                hide_index=True
            )
    
    with analysis_tabs[3]:
        # Custom Analysis
        st.subheader("Custom Performance Comparison")
        
        # Select metrics to compare
        col1, col2 = st.columns(2)
        
        with col1:
            x_metric = st.selectbox(
                "X-Axis Metric",
                options=["Total Return (%)", "Annualized Return (%)", "Sharpe Ratio", "Max Drawdown (%)", "Trades"],
                index=0
            )
        
        with col2:
            y_metric = st.selectbox(
                "Y-Axis Metric",
                options=["Total Return (%)", "Annualized Return (%)", "Sharpe Ratio", "Max Drawdown (%)", "Trades"],
                index=2
            )
        
        # Group by selection
        group_by = st.selectbox(
            "Group By",
            options=["Strategy", "Asset", "Timeframe", "Asset Type"],
            index=0
        )
        
        # Create custom scatter plot
        custom_fig = px.scatter(
            filtered_data,
            x=x_metric,
            y=y_metric,
            color=group_by,
            size="Final Capital",
            hover_data=["Strategy", "Asset", "Timeframe"],
            title=f"{y_metric} vs. {x_metric} by {group_by}"
        )
        
        custom_fig.update_layout(height=600)
        st.plotly_chart(custom_fig, use_container_width=True)
    
    # Detailed data view
    st.header("Detailed Data View")
    
    view_tabs = st.tabs(["Raw Data", "Download Data"])
    
    with view_tabs[0]:
        st.dataframe(
            filtered_data,
            use_container_width=True,
            height=400
        )
    
    with view_tabs[1]:
        csv = filtered_data.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name="mercurio_ai_simulation_results.csv",
            mime="text/csv"
        )
        
        excel_available = False
        try:
            import io
            from io import BytesIO
            import openpyxl
            excel_available = True
        except:
            st.info("Install openpyxl to enable Excel download: `pip install openpyxl`")
        
        if excel_available:
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                filtered_data.to_excel(writer, sheet_name='Simulation Results', index=False)
                
                # Add summary sheet
                summary_data = pd.DataFrame({
                    'Metric': ['Total Simulations', 'Average Return (%)', 'Average Sharpe Ratio', 'Average Max Drawdown (%)'],
                    'Value': [
                        len(filtered_data),
                        filtered_data['Total Return (%)'].mean(),
                        filtered_data['Sharpe Ratio'].mean(),
                        filtered_data['Max Drawdown (%)'].mean()
                    ]
                })
                summary_data.to_excel(writer, sheet_name='Summary', index=False)
                
                # Add best performers sheet
                top_10 = filtered_data.nlargest(10, 'Total Return (%)')
                top_10.to_excel(writer, sheet_name='Top Performers', index=False)
            
            st.download_button(
                label="Download Filtered Data as Excel",
                data=buffer.getvalue(),
                file_name="mercurio_ai_simulation_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    # Footer
    st.markdown("---")
    st.caption("Mercurio AI - Comprehensive Strategy Simulation Dashboard")
    st.caption("Run 'python comprehensive_simulation.py' to generate fresh data")

if __name__ == "__main__":
    main()
