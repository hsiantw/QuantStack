import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_fetcher import DataFetcher
from utils.strategy_optimizer import StrategyOptimizer
from utils.tooltips import get_tooltip_help

# Page configuration
st.set_page_config(
    page_title="Strategy Comparison & Optimization",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Advanced Strategy Comparison & Optimization")
st.markdown("Compare multiple trading strategies with different indicator combinations to find the optimal approach")

# Sidebar for inputs
st.sidebar.header("Optimization Configuration")

# Asset selection
ticker_input = st.sidebar.text_input(
    "Enter Stock Ticker",
    value="SPY",
    help="Enter a valid stock ticker symbol for strategy optimization"
).upper()

# Quick selection from popular tickers
optimization_tickers = {
    "High Volatility": ["TSLA", "NVDA", "AMC", "GME", "ROKU", "ZM"],
    "Large Cap Stable": ["AAPL", "MSFT", "GOOGL", "JNJ", "PG", "KO"],
    "Growth Stocks": ["AMZN", "META", "NFLX", "CRM", "SHOP", "SQ"],
    "Financial Sector": ["JPM", "BAC", "WFC", "GS", "MS", "C"],
    "ETFs": ["SPY", "QQQ", "IWM", "XLK", "XLF", "VTI"]
}

selected_category = st.sidebar.selectbox("Or select from categories", [""] + list(optimization_tickers.keys()))

if selected_category:
    selected_ticker = st.sidebar.selectbox(
        f"Select from {selected_category}",
        optimization_tickers[selected_category]
    )
    if st.sidebar.button(f"Optimize {selected_ticker}"):
        ticker_input = selected_ticker

# Time period selection
period_options = {
    "1 Year": "1y",
    "2 Years": "2y",
    "3 Years": "3y",
    "5 Years": "5y"
}

selected_period = st.sidebar.selectbox(
    "Optimization Period",
    list(period_options.keys()),
    index=2,
    help="Longer periods provide more robust optimization results"
)

# Optimization settings
st.sidebar.subheader("Optimization Settings")

optimization_metric = st.sidebar.selectbox(
    "Primary Optimization Metric",
    ["Sharpe Ratio", "Calmar Ratio", "Sortino Ratio", "Total Return", "Information Ratio"],
    help="Metric used to rank strategy performance"
)

initial_capital = st.sidebar.number_input(
    "Initial Capital ($)",
    min_value=10000,
    max_value=10000000,
    value=100000,
    step=10000
)

# Advanced settings
with st.sidebar.expander("Advanced Settings"):
    top_strategies_count = st.slider(
        "Number of Top Strategies to Display",
        min_value=3,
        max_value=10,
        value=5
    )
    
    include_volume_strategies = st.checkbox(
        "Include Volume-Based Strategies",
        value=True,
        help="Include strategies that use volume indicators"
    )
    
    risk_free_rate = st.slider(
        "Risk-Free Rate (%)",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        step=0.1
    ) / 100

if not ticker_input:
    st.warning("Please enter a stock ticker symbol.")
    st.stop()

# Data fetching
with st.spinner(f"Fetching data for {ticker_input}..."):
    try:
        # Validate ticker
        valid_tickers, invalid_tickers = DataFetcher.validate_tickers([ticker_input])
        
        if invalid_tickers:
            st.error(f"Invalid ticker: {ticker_input}")
            st.stop()
        
        # Fetch OHLCV data
        period = period_options[selected_period]
        price_data = DataFetcher.get_stock_data(ticker_input, period=period)
        
        if price_data.empty:
            st.error(f"Unable to fetch data for {ticker_input}")
            st.stop()
        
        # Prepare OHLCV structure
        if isinstance(price_data.columns, pd.MultiIndex):
            ohlcv_data = price_data.droplevel(1, axis=1)
        else:
            ohlcv_data = price_data
        
        # Validate required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in ohlcv_data.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            st.stop()
        
        # Get stock info
        stock_info = DataFetcher.get_stock_info(ticker_input)
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        st.stop()

# Display stock information
if stock_info:
    st.header(f"📊 {ticker_input} - {stock_info.get('longName', 'N/A')}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = ohlcv_data['Close'].iloc[-1]
        st.metric("Current Price", f"${current_price:.2f}")
    
    with col2:
        daily_change = (ohlcv_data['Close'].iloc[-1] - ohlcv_data['Close'].iloc[-2]) / ohlcv_data['Close'].iloc[-2]
        st.metric("Daily Change", f"{daily_change:.2%}")
    
    with col3:
        period_return = (ohlcv_data['Close'].iloc[-1] - ohlcv_data['Close'].iloc[0]) / ohlcv_data['Close'].iloc[0]
        st.metric(f"{selected_period} Return", f"{period_return:.2%}")
    
    with col4:
        volatility = ohlcv_data['Close'].pct_change().std() * np.sqrt(252)
        st.metric("Annualized Volatility", f"{volatility:.2%}")

# Strategy Optimization Process
st.header("🔧 Strategy Optimization Process")

# Add tooltips for optimization concepts
with st.expander("💡 Strategy Optimization Concepts"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Sharpe Ratio:**")
        st.markdown(get_tooltip_help("sharpe_ratio")[:200] + "...")
    with col2:
        st.markdown("**Calmar Ratio:**")
        st.markdown(get_tooltip_help("calmar_ratio")[:200] + "...")
    with col3:
        st.markdown("**Sortino Ratio:**")
        st.markdown("Risk-adjusted return metric using only downside volatility instead of total volatility.")

# Initialize optimizer and run optimization
with st.spinner("Running comprehensive strategy optimization..."):
    try:
        optimizer = StrategyOptimizer(ohlcv_data, initial_capital)
        optimization_results = optimizer.optimize_strategies()
        
        if optimization_results.empty:
            st.error("No strategies could be executed successfully.")
            st.stop()
        
    except Exception as e:
        st.error(f"Error in strategy optimization: {str(e)}")
        st.stop()

# Display optimization results
st.header("🏆 Optimization Results")

# Top strategies summary
st.subheader(f"Top {top_strategies_count} Strategies by {optimization_metric}")

top_strategies = optimizer.get_top_strategies(optimization_metric, top_strategies_count)

# Format the results for display
display_columns = [
    'Strategy', 'Type', 'Total Return', 'Annualized Return', 'Volatility',
    'Sharpe Ratio', 'Max Drawdown', 'Calmar Ratio', 'Win Rate', 'Total Trades'
]

formatted_results = top_strategies.copy()
for col in ['Total Return', 'Annualized Return', 'Volatility', 'Max Drawdown']:
    if col in formatted_results.columns:
        formatted_results[col] = formatted_results[col].apply(lambda x: f"{x:.2%}")

for col in ['Sharpe Ratio', 'Calmar Ratio', 'Sortino Ratio', 'Information Ratio']:
    if col in formatted_results.columns:
        formatted_results[col] = formatted_results[col].apply(lambda x: f"{x:.3f}")

if 'Win Rate' in formatted_results.columns:
    formatted_results['Win Rate'] = formatted_results['Win Rate'].apply(lambda x: f"{x:.2%}")

if 'Total Trades' in formatted_results.columns:
    formatted_results['Total Trades'] = formatted_results['Total Trades'].apply(lambda x: f"{x:.0f}")

st.dataframe(
    formatted_results[display_columns],
    use_container_width=True,
    hide_index=True
)

# Highlight best strategy
best_strategy = top_strategies.iloc[0]
st.success(f"🥇 **Best Strategy:** {best_strategy['Strategy']} "
          f"({optimization_metric}: {best_strategy[optimization_metric]:.3f})")

# Strategy performance comparison visualization
st.header("📊 Strategy Performance Analysis")

try:
    comparison_fig = optimizer.plot_strategy_comparison(top_strategies['Strategy'].tolist())
    st.plotly_chart(comparison_fig, use_container_width=True)
except Exception as e:
    st.error(f"Error generating comparison chart: {str(e)}")

# Detailed strategy analysis
st.header("🔍 Detailed Strategy Analysis")

selected_strategy_name = st.selectbox(
    "Select strategy for detailed analysis",
    top_strategies['Strategy'].tolist()
)

if selected_strategy_name:
    strategy_details = optimizer.get_strategy_details(selected_strategy_name)
    
    if strategy_details:
        # Strategy overview
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Strategy Metrics")
            metrics = strategy_details['metrics']
            
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric("Total Return", f"{metrics['Total Return']:.2%}")
                st.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.3f}")
                st.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2%}")
            
            with metric_col2:
                st.metric("Win Rate", f"{metrics['Win Rate']:.2%}")
                st.metric("Calmar Ratio", f"{metrics['Calmar Ratio']:.3f}")
                st.metric("Total Trades", f"{metrics['Total Trades']:.0f}")
        
        with col2:
            st.subheader("Signal Distribution")
            signal_dist = strategy_details['signal_distribution']
            
            # Create signal distribution chart
            fig_signals = go.Figure(data=[
                go.Bar(
                    x=['Sell (-1)', 'Hold (0)', 'Buy (1)'],
                    y=[signal_dist.get(-1, 0), signal_dist.get(0, 0), signal_dist.get(1, 0)],
                    marker_color=['red', 'gray', 'green']
                )
            ])
            fig_signals.update_layout(
                title="Trading Signal Distribution",
                xaxis_title="Signal Type",
                yaxis_title="Number of Days",
                height=300
            )
            st.plotly_chart(fig_signals, use_container_width=True)
        
        # Performance over time
        st.subheader("Performance Over Time")
        
        returns = strategy_details['returns']
        cumulative_returns = (1 + returns.fillna(0)).cumprod()
        
        # Buy & hold comparison
        buy_hold_returns = ohlcv_data['Close'].pct_change()
        buy_hold_cumulative = (1 + buy_hold_returns.fillna(0)).cumprod()
        
        fig_performance = go.Figure()
        
        fig_performance.add_trace(go.Scatter(
            x=cumulative_returns.index,
            y=cumulative_returns.values,
            mode='lines',
            name=selected_strategy_name,
            line=dict(color='blue', width=2)
        ))
        
        fig_performance.add_trace(go.Scatter(
            x=buy_hold_cumulative.index,
            y=buy_hold_cumulative.values,
            mode='lines',
            name='Buy & Hold',
            line=dict(color='orange', dash='dash')
        ))
        
        fig_performance.update_layout(
            title="Cumulative Returns Comparison",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            height=400
        )
        
        st.plotly_chart(fig_performance, use_container_width=True)
        
        # Rolling performance metrics
        if 'rolling_sharpe' in strategy_details:
            rolling_sharpe = strategy_details['rolling_sharpe'].dropna()
            
            if not rolling_sharpe.empty:
                fig_rolling = go.Figure()
                
                fig_rolling.add_trace(go.Scatter(
                    x=rolling_sharpe.index,
                    y=rolling_sharpe.values,
                    mode='lines',
                    name='Rolling Sharpe Ratio (1Y)',
                    line=dict(color='purple')
                ))
                
                fig_rolling.add_hline(y=1.0, line_dash="dash", line_color="red", 
                                    annotation_text="Good Performance Threshold")
                
                fig_rolling.update_layout(
                    title="Rolling Sharpe Ratio (252 Days)",
                    xaxis_title="Date",
                    yaxis_title="Sharpe Ratio",
                    height=300
                )
                
                st.plotly_chart(fig_rolling, use_container_width=True)

# Strategy comparison matrix
st.header("📋 Complete Strategy Matrix")

# Show all strategies sorted by the selected metric
all_strategies_formatted = optimization_results.copy()

# Format percentage columns
for col in ['Total Return', 'Annualized Return', 'Volatility', 'Max Drawdown', 'Win Rate']:
    if col in all_strategies_formatted.columns:
        all_strategies_formatted[col] = all_strategies_formatted[col].apply(lambda x: f"{x:.2%}")

# Format ratio columns
for col in ['Sharpe Ratio', 'Calmar Ratio', 'Sortino Ratio', 'Information Ratio']:
    if col in all_strategies_formatted.columns:
        all_strategies_formatted[col] = all_strategies_formatted[col].apply(lambda x: f"{x:.3f}")

# Sort by optimization metric
all_strategies_formatted = all_strategies_formatted.sort_values(optimization_metric, ascending=False)

st.dataframe(
    all_strategies_formatted[display_columns],
    use_container_width=True,
    hide_index=True
)

# Performance summary and insights
st.header("💡 Optimization Insights")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Key Findings")
    
    # Strategy type analysis
    strategy_types = optimization_results['Type'].value_counts()
    best_type = strategy_types.index[0]
    
    st.write(f"**Most Successful Strategy Type:** {best_type}")
    st.write(f"**Number of Strategies Tested:** {len(optimization_results)}")
    
    # Performance ranges
    sharpe_range = f"{optimization_results['Sharpe Ratio'].min():.3f} to {optimization_results['Sharpe Ratio'].max():.3f}"
    st.write(f"**Sharpe Ratio Range:** {sharpe_range}")
    
    return_range = f"{optimization_results['Total Return'].min():.2%} to {optimization_results['Total Return'].max():.2%}"
    st.write(f"**Return Range:** {return_range}")

with col2:
    st.subheader("Recommendations")
    
    # Top 3 strategies recommendation
    top_3 = optimization_results.nlargest(3, optimization_metric)
    
    st.write("**Top 3 Recommended Strategies:**")
    for i, (_, strategy) in enumerate(top_3.iterrows(), 1):
        st.write(f"{i}. **{strategy['Strategy']}** "
                f"({optimization_metric}: {strategy[optimization_metric]:.3f})")
    
    # Risk assessment
    avg_volatility = optimization_results['Volatility'].mean()
    avg_max_dd = optimization_results['Max Drawdown'].mean()
    
    st.write(f"**Average Volatility:** {avg_volatility:.2%}")
    st.write(f"**Average Max Drawdown:** {avg_max_dd:.2%}")

# Export results option
st.header("📥 Export Results")

if st.button("Download Optimization Results"):
    csv = optimization_results.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"{ticker_input}_strategy_optimization_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )