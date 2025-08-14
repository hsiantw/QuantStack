import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_fetcher import DataFetcher
from utils.statistical_arbitrage import StatisticalArbitrage
from utils.tooltips import get_tooltip_help
from utils.backtesting import BacktestingEngine

# Page configuration
st.set_page_config(
    page_title="Statistical Arbitrage",
    page_icon="🔗",
    layout="wide"
)

# Modern header with enhanced styling
st.markdown("""
<div class="main-header">
    <h1>🔗 Statistical Arbitrage Analysis</h1>
    <p>Identify pair trading opportunities through cointegration analysis and mean reversion strategies</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for inputs
st.sidebar.header("Analysis Configuration")

# Asset selection method
analysis_mode = st.sidebar.radio(
    "Analysis Mode",
    ["Sector Analysis", "Custom Pair", "Multi-Asset Screening"],
    help="Choose your analysis approach"
)

if analysis_mode == "Sector Analysis":
    # Sector-based analysis
    sectors = {
        "Broad Market ETFs": ["SPY", "QQQ", "IWM", "VTI", "EFA", "EEM", "TLT", "GLD"],
        "Technology": ["QQQ", "AAPL", "MSFT", "GOOGL", "NVDA", "META", "TSLA", "NFLX", "ADBE"],
        "Banking": ["JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC"],
        "Healthcare": ["JNJ", "PFE", "UNH", "ABBV", "MRK", "TMO", "ABT", "LLY"],
        "Energy": ["XOM", "CVX", "COP", "EOG", "SLB", "PSX", "VLO", "MPC"],
        "Consumer": ["PG", "KO", "PEP", "WMT", "HD", "MCD", "NKE", "SBUX"]
    }
    
    selected_sector = st.sidebar.selectbox("Select Sector", list(sectors.keys()))
    selected_assets = st.sidebar.multiselect(
        f"Select assets from {selected_sector}",
        sectors[selected_sector],
        default=sectors[selected_sector][:6]
    )

elif analysis_mode == "Custom Pair":
    # Custom pair selection
    asset1 = st.sidebar.text_input("Asset 1 (e.g., SPY)", value="SPY").upper()
    asset2 = st.sidebar.text_input("Asset 2 (e.g., QQQ)", value="QQQ").upper()
    selected_assets = [asset1, asset2] if asset1 and asset2 else []

else:  # Multi-Asset Screening
    # Multi-asset screening
    default_assets = ["SPY", "QQQ", "IWM", "VTI", "EFA", "EEM"]
    other_assets = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "JPM", "BAC", "JNJ", "PFE", "NVDA", "HD", "PG", "KO", "XOM", "CVX", "UNH", "V", "MA", "DIS"]
    selected_assets = st.sidebar.multiselect(
        "Select assets for screening",
        default_assets + other_assets,
        default=default_assets
    )

# Time period
period_options = {
    "6 Months": "6mo",
    "1 Year": "1y", 
    "2 Years": "2y",
    "3 Years": "3y"
}

selected_period = st.sidebar.selectbox(
    "Time Period",
    list(period_options.keys()),
    index=2
)

# Analysis parameters
with st.sidebar.expander("Analysis Parameters"):
    cointegration_significance = st.slider(
        "Cointegration Significance Level",
        min_value=0.01,
        max_value=0.10,
        value=0.05,
        step=0.01,
        help="P-value threshold for cointegration test"
    )
    
    entry_threshold = st.slider(
        "Entry Z-Score Threshold",
        min_value=1.0,
        max_value=3.0,
        value=2.0,
        step=0.1,
        help="Z-score threshold for entering trades"
    )
    
    exit_threshold = st.slider(
        "Exit Z-Score Threshold", 
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Z-score threshold for exiting trades"
    )

# Backtesting parameters
with st.sidebar.expander("Backtesting Parameters"):
    initial_capital = st.number_input(
        "Initial Capital ($)",
        min_value=10000,
        max_value=10000000,
        value=100000,
        step=10000
    )
    
    transaction_cost = st.slider(
        "Transaction Cost (%)",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05
    ) / 100

if len(selected_assets) < 2:
    st.warning("Please select at least 2 assets for statistical arbitrage analysis.")
    st.stop()

# Data fetching
with st.spinner("Fetching market data..."):
    try:
        # Validate tickers
        valid_tickers, invalid_tickers = DataFetcher.validate_tickers(selected_assets)
        
        if invalid_tickers:
            st.warning(f"Invalid tickers removed: {', '.join(invalid_tickers)}")
        
        if len(valid_tickers) < 2:
            st.error("Need at least 2 valid tickers for analysis.")
            st.stop()
        
        # Fetch price data
        period = period_options[selected_period]
        price_data = DataFetcher.get_stock_data(valid_tickers, period=period)
        
        if price_data.empty:
            st.error("Unable to fetch price data.")
            st.stop()
        
        # Use Close prices
        if isinstance(price_data.columns, pd.MultiIndex):
            close_prices = price_data['Close']
        else:
            close_prices = price_data
            
        # Initialize statistical arbitrage analyzer
        stat_arb = StatisticalArbitrage(close_prices)
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        st.stop()

# Analysis based on mode
if analysis_mode == "Custom Pair" and len(valid_tickers) == 2:
    # Detailed pair analysis
    asset1, asset2 = valid_tickers[0], valid_tickers[1]
    
    st.header(f"📊 Detailed Pair Analysis: {asset1} vs {asset2}")
    
    # Add tooltips for key concepts
    with st.expander("💡 Key Statistical Arbitrage Concepts"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Cointegration:**")
            st.markdown(get_tooltip_help("cointegration")[:200] + "...")
        with col2:
            st.markdown("**Z-Score:**")
            st.markdown(get_tooltip_help("z_score")[:200] + "...")
    
    # Cointegration test
    cointegrated_pairs = stat_arb.find_cointegrated_pairs(cointegration_significance)
    
    is_cointegrated = any(
        (pair['Asset1'] == asset1 and pair['Asset2'] == asset2) or 
        (pair['Asset1'] == asset2 and pair['Asset2'] == asset1)
        for pair in cointegrated_pairs
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if is_cointegrated:
            st.success("✅ Pair is Cointegrated")
            coint_pair = next(
                pair for pair in cointegrated_pairs 
                if (pair['Asset1'] == asset1 and pair['Asset2'] == asset2) or 
                   (pair['Asset1'] == asset2 and pair['Asset2'] == asset1)
            )
            st.metric("P-Value", f"{coint_pair['P_Value']:.4f}")
        else:
            st.error("❌ Pair is NOT Cointegrated")
            st.metric("P-Value", "> 0.05")
    
    with col2:
        correlation = stat_arb.correlation_analysis().loc[asset1, asset2]
        st.metric("Correlation", f"{correlation:.3f}")
    
    with col3:
        spread, hedge_ratio, _ = stat_arb.calculate_spread(asset1, asset2)
        st.metric("Hedge Ratio", f"{hedge_ratio:.3f}")
    
    # Pair analysis plots
    st.subheader("Pair Analysis Visualizations")
    
    try:
        pair_plots = stat_arb.plot_pair_analysis(asset1, asset2)
        
        # Price comparison
        if 'price_comparison' in pair_plots:
            st.plotly_chart(pair_plots['price_comparison'], use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'spread_analysis' in pair_plots:
                st.plotly_chart(pair_plots['spread_analysis'], use_container_width=True)
        
        with col2:
            if 'z_score' in pair_plots:
                st.plotly_chart(pair_plots['z_score'], use_container_width=True)
        
        # Scatter plot
        if 'scatter_plot' in pair_plots:
            st.plotly_chart(pair_plots['scatter_plot'], use_container_width=True)
            
    except Exception as e:
        st.error(f"Error generating pair analysis plots: {str(e)}")
    
    # Trading strategy backtest
    if is_cointegrated:
        st.header("💰 Trading Strategy Backtest")
        
        with st.spinner("Running backtest..."):
            try:
                backtest_results = stat_arb.backtest_pair_strategy(
                    asset1, asset2, 
                    initial_capital=initial_capital,
                    entry_threshold=entry_threshold,
                    exit_threshold=exit_threshold
                )
                
                # Performance metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Total Return", 
                        f"{backtest_results['Total_Return']:.2%}",
                        delta=f"vs B&H: {backtest_results['Total_Return']:.2%}"
                    )
                
                with col2:
                    st.metric("Sharpe Ratio", f"{backtest_results['Sharpe_Ratio']:.3f}")
                
                with col3:
                    st.metric("Max Drawdown", f"{backtest_results['Max_Drawdown']:.2%}")
                
                with col4:
                    st.metric("Win Rate", f"{backtest_results['Win_Rate']:.2%}")
                
                # Backtest visualization
                portfolio_returns = backtest_results['Portfolio_Returns']
                cumulative_returns = backtest_results['Cumulative_Returns']
                
                fig_backtest = go.Figure()
                
                fig_backtest.add_trace(go.Scatter(
                    x=cumulative_returns.index,
                    y=cumulative_returns,
                    mode='lines',
                    name='Strategy Performance',
                    line=dict(color='blue', width=2)
                ))
                
                fig_backtest.update_layout(
                    title="Pair Trading Strategy Performance",
                    xaxis_title="Date",
                    yaxis_title="Cumulative Return",
                    height=500
                )
                
                st.plotly_chart(fig_backtest, use_container_width=True)
                
                # Trading signals
                signals_df = backtest_results['Signals']
                
                # Show recent signals
                st.subheader("Recent Trading Signals")
                recent_signals = signals_df[signals_df['Position'] != 0].tail(10)
                
                if not recent_signals.empty:
                    display_signals = recent_signals[['Spread', 'Z_Score', 'Position']].copy()
                    display_signals['Signal'] = display_signals['Position'].map({
                        1: 'Long Spread (Short ' + asset1 + ', Long ' + asset2 + ')',
                        -1: 'Short Spread (Long ' + asset1 + ', Short ' + asset2 + ')'
                    })
                    st.dataframe(display_signals[['Signal', 'Spread', 'Z_Score']], use_container_width=True)
                else:
                    st.info("No recent trading signals generated.")
                    
            except Exception as e:
                st.error(f"Error in backtesting: {str(e)}")

else:
    # Multi-asset analysis
    st.header("🔍 Multi-Asset Cointegration Screening")
    
    # Find all cointegrated pairs
    with st.spinner("Searching for cointegrated pairs..."):
        try:
            cointegrated_pairs = stat_arb.find_cointegrated_pairs(cointegration_significance)
            
            if cointegrated_pairs:
                st.success(f"Found {len(cointegrated_pairs)} cointegrated pairs!")
                
                # Display results table
                pairs_df = pd.DataFrame(cointegrated_pairs)
                pairs_df['Pair'] = pairs_df['Asset1'] + ' - ' + pairs_df['Asset2']
                pairs_df['P_Value'] = pairs_df['P_Value'].round(4)
                pairs_df['Cointegration_Stat'] = pairs_df['Cointegration_Stat'].round(3)
                
                display_df = pairs_df[['Pair', 'P_Value', 'Cointegration_Stat']].copy()
                display_df = display_df.sort_values('P_Value')
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Select pair for detailed analysis
                st.subheader("Detailed Analysis")
                
                selected_pair_idx = st.selectbox(
                    "Select a pair for detailed analysis",
                    range(len(pairs_df)),
                    format_func=lambda x: pairs_df.iloc[x]['Pair']
                )
                
                selected_pair = pairs_df.iloc[selected_pair_idx]
                asset1, asset2 = selected_pair['Asset1'], selected_pair['Asset2']
                
                # Show pair analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    st.info(f"**Analyzing:** {asset1} vs {asset2}")
                    st.metric("P-Value", f"{selected_pair['P_Value']:.4f}")
                    
                    correlation = stat_arb.correlation_analysis().loc[asset1, asset2]
                    st.metric("Correlation", f"{correlation:.3f}")
                
                with col2:
                    spread, hedge_ratio, _ = stat_arb.calculate_spread(asset1, asset2)
                    st.metric("Hedge Ratio", f"{hedge_ratio:.3f}")
                    
                    # Quick backtest
                    quick_backtest = stat_arb.backtest_pair_strategy(
                        asset1, asset2,
                        initial_capital=initial_capital,
                        entry_threshold=entry_threshold,
                        exit_threshold=exit_threshold
                    )
                    st.metric("Strategy Return", f"{quick_backtest['Total_Return']:.2%}")
                
                # Pair visualization
                try:
                    pair_plots = stat_arb.plot_pair_analysis(asset1, asset2)
                    
                    if 'spread_analysis' in pair_plots:
                        st.plotly_chart(pair_plots['spread_analysis'], use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Error generating plots: {str(e)}")
                
            else:
                st.warning("No cointegrated pairs found with the current significance level.")
                st.info("Try increasing the significance level or selecting different assets.")
                
        except Exception as e:
            st.error(f"Error in cointegration analysis: {str(e)}")
    
    # Correlation analysis
    st.header("📊 Correlation Analysis")
    
    try:
        correlation_heatmap = stat_arb.correlation_heatmap()
        st.plotly_chart(correlation_heatmap, use_container_width=True)
        
        # Show highly correlated pairs
        corr_matrix = stat_arb.correlation_analysis()
        
        # Extract upper triangle correlations
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                asset1, asset2 = corr_matrix.columns[i], corr_matrix.columns[j]
                correlation = corr_matrix.iloc[i, j]
                
                if abs(correlation) > 0.7:  # High correlation threshold
                    high_corr_pairs.append({
                        'Asset 1': asset1,
                        'Asset 2': asset2,
                        'Correlation': correlation
                    })
        
        if high_corr_pairs:
            st.subheader("Highly Correlated Pairs (|correlation| > 0.7)")
            high_corr_df = pd.DataFrame(high_corr_pairs)
            high_corr_df['Correlation'] = high_corr_df['Correlation'].round(3)
            high_corr_df = high_corr_df.sort_values('Correlation', key=abs, ascending=False)
            st.dataframe(high_corr_df, use_container_width=True, hide_index=True)
        
    except Exception as e:
        st.error(f"Error in correlation analysis: {str(e)}")

# Strategy recommendations
st.header("💡 Strategy Recommendations")

if 'cointegrated_pairs' in locals() and cointegrated_pairs:
    st.success("**Recommended Actions:**")
    
    # Sort pairs by statistical significance
    sorted_pairs = sorted(cointegrated_pairs, key=lambda x: x['P_Value'])
    
    for i, pair in enumerate(sorted_pairs[:3]):  # Show top 3 pairs
        with st.expander(f"Pair {i+1}: {pair['Asset1']} - {pair['Asset2']}"):
            st.write(f"**Statistical Strength:** P-value = {pair['P_Value']:.4f}")
            
            # Calculate current z-score
            spread, hedge_ratio, intercept = stat_arb.calculate_spread(pair['Asset1'], pair['Asset2'])
            current_z_score = (spread.iloc[-1] - spread.mean()) / spread.std()
            
            st.write(f"**Current Z-Score:** {current_z_score:.2f}")
            
            if abs(current_z_score) > entry_threshold:
                if current_z_score > 0:
                    st.write(f"🔴 **Signal:** SHORT spread (Long {pair['Asset1']}, Short {pair['Asset2']})")
                else:
                    st.write(f"🟢 **Signal:** LONG spread (Short {pair['Asset1']}, Long {pair['Asset2']})")
            else:
                st.write("⚪ **Signal:** No entry signal currently")
            
            st.write(f"**Hedge Ratio:** {hedge_ratio:.3f}")

# Export functionality
st.header("📥 Export Results")

if 'cointegrated_pairs' in locals() and cointegrated_pairs:
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Export Cointegrated Pairs", use_container_width=True):
            pairs_df = pd.DataFrame(cointegrated_pairs)
            csv = pairs_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"cointegrated_pairs_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📈 Export Correlation Matrix", use_container_width=True):
            corr_matrix = stat_arb.correlation_analysis()
            csv = corr_matrix.to_csv()
            st.download_button(
                label="Download CSV", 
                data=csv,
                file_name=f"correlation_matrix_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

# Navigation
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🏠 Back to Dashboard", use_container_width=True):
        st.switch_page("app.py")

with col2:
    if st.button("📊 Portfolio Optimization", use_container_width=True):
        st.switch_page("pages/portfolio_optimization.py")

with col3:
    if st.button("📈 Time Series Analysis", use_container_width=True):
        st.switch_page("pages/time_series_analysis.py")
