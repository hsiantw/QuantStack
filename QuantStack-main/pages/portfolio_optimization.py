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
from utils.portfolio_optimizer import PortfolioOptimizer
from utils.risk_metrics import RiskMetrics
from utils.tooltips import get_tooltip_help

# Page configuration
st.set_page_config(
    page_title="Portfolio Optimization",
    page_icon="📊",
    layout="wide"
)

# Modern header with enhanced styling
st.markdown("""
<div class="main-header">
    <h1>📊 Portfolio Optimization</h1>
    <p>Modern Portfolio Theory implementation with efficient frontier calculation and risk analysis</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for inputs
st.sidebar.header("Portfolio Configuration")

# Stock selection
default_tickers = ["SPY", "QQQ", "IWM", "VTI", "EFA"]
other_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "JNJ", "V", "EEM", "TLT", "GLD", "BND"]
selected_tickers = st.sidebar.multiselect(
    "Select stocks for portfolio",
    options=default_tickers + other_tickers,
    default=default_tickers,
    help="Choose 3-15 stocks for optimal diversification"
)

# Time period selection
period_options = {
    "1 Year": "1y",
    "2 Years": "2y", 
    "3 Years": "3y",
    "5 Years": "5y"
}

selected_period = st.sidebar.selectbox(
    "Select time period",
    options=list(period_options.keys()),
    index=1
)

# Risk-free rate
risk_free_rate = st.sidebar.number_input(
    "Risk-free rate (%)",
    min_value=0.0,
    max_value=10.0,
    value=2.0,
    step=0.1,
    help="Annual risk-free rate for Sharpe ratio calculation"
) / 100

# Advanced options
with st.sidebar.expander("Advanced Options"):
    rebalancing_freq = st.selectbox(
        "Rebalancing Frequency",
        ["Monthly", "Quarterly", "Semi-Annual", "Annual"],
        index=2
    )
    
    include_shorting = st.checkbox(
        "Allow short selling",
        value=False,
        help="Allow negative weights in portfolio"
    )
    
    max_weight = st.slider(
        "Maximum weight per asset (%)",
        min_value=5,
        max_value=100,
        value=40,
        step=5
    ) / 100

if len(selected_tickers) < 2:
    st.warning("Please select at least 2 assets for portfolio optimization.")
    st.stop()

if len(selected_tickers) > 15:
    st.warning("Please select no more than 15 assets for optimal performance.")
    st.stop()

# Data fetching
with st.spinner("Fetching market data..."):
    try:
        # Validate tickers first
        valid_tickers, invalid_tickers = DataFetcher.validate_tickers(selected_tickers)
        
        if invalid_tickers:
            st.warning(f"Invalid tickers removed: {', '.join(invalid_tickers)}")
        
        if len(valid_tickers) < 2:
            st.error("Not enough valid tickers for portfolio optimization.")
            st.stop()
        
        # Fetch data
        period = period_options[selected_period]
        price_data = DataFetcher.get_stock_data(valid_tickers, period=period)
        
        if price_data.empty:
            st.error("Unable to fetch data. Please try different tickers or time period.")
            st.stop()
        
        # Use Close prices
        if isinstance(price_data.columns, pd.MultiIndex):
            close_prices = price_data['Close']
        else:
            close_prices = price_data
        
        # Calculate returns
        returns_data = DataFetcher.calculate_returns(close_prices)
        
        if returns_data.empty:
            st.error("Unable to calculate returns from the data.")
            st.stop()
            
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        st.stop()

# Portfolio optimization
if not returns_data.empty:
    
    # Display basic statistics
    st.header("📈 Asset Performance Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Annualized Returns")
        annual_returns = returns_data.mean() * 252
        annual_returns_df = pd.DataFrame({
            'Asset': annual_returns.index,
            'Annual Return': (annual_returns * 100).round(2)
        })
        st.dataframe(annual_returns_df, use_container_width=True)
    
    with col2:
        st.subheader("Annualized Volatility")
        annual_volatility = returns_data.std() * np.sqrt(252)
        volatility_df = pd.DataFrame({
            'Asset': annual_volatility.index,
            'Annual Volatility': (annual_volatility * 100).round(2)
        })
        st.dataframe(volatility_df, use_container_width=True)
    
    # Initialize optimizer
    optimizer = PortfolioOptimizer(returns_data)
    
    # Portfolio optimization
    st.header("🎯 Portfolio Optimization Results")
    
    # Calculate optimal portfolios
    with st.spinner("Calculating optimal portfolios..."):
        try:
            # Maximum Sharpe ratio portfolio
            max_sharpe_weights = optimizer.maximize_sharpe_ratio()
            
            # Minimum variance portfolio
            min_var_weights = optimizer.minimize_variance()
            
            # Equal weight portfolio for comparison
            equal_weights = np.array([1/len(valid_tickers)] * len(valid_tickers))
            
            if max_sharpe_weights is None or min_var_weights is None:
                st.error("Unable to optimize portfolios. Please try different assets or time period.")
                st.stop()
                
        except Exception as e:
            st.error(f"Error in portfolio optimization: {str(e)}")
            st.stop()
    
    # Display portfolio results
    portfolio_results = {}
    
    for name, weights in [
        ("Maximum Sharpe Ratio", max_sharpe_weights),
        ("Minimum Variance", min_var_weights),
        ("Equal Weight", equal_weights)
    ]:
        ret, vol, sharpe = optimizer.portfolio_stats(weights)
        risk_metrics = optimizer.calculate_risk_metrics(weights)
        
        portfolio_results[name] = {
            'weights': weights,
            'return': ret,
            'volatility': vol,
            'sharpe': sharpe,
            'risk_metrics': risk_metrics
        }
    
    # Portfolio comparison table
    st.subheader("Portfolio Comparison")
    
    # Add helpful tooltips for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("💡 **Sharpe Ratio**: " + get_tooltip_help("sharpe_ratio")[:100] + "...")
    
    with col2:
        st.info("💡 **Value at Risk**: " + get_tooltip_help("value_at_risk")[:100] + "...")
    
    with col3:
        st.info("💡 **Max Drawdown**: " + get_tooltip_help("maximum_drawdown")[:100] + "...")
    
    comparison_data = []
    for name, results in portfolio_results.items():
        comparison_data.append({
            'Portfolio': name,
            'Expected Return': f"{results['return']:.2%}",
            'Volatility': f"{results['volatility']:.2%}",
            'Sharpe Ratio': f"{results['sharpe']:.3f}",
            'Max Drawdown': f"{results['risk_metrics']['Maximum Drawdown']:.2%}",
            'VaR (95%)': f"{results['risk_metrics']['VaR (95%)']:.2%}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Add save functionality for authenticated users
    if st.session_state.get('authenticated', False):
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            portfolio_name = st.text_input(
                "💾 Save optimized portfolio",
                placeholder=f"Portfolio - {'-'.join(valid_tickers[:3])} - {datetime.now().strftime('%Y-%m-%d')}",
                key="save_portfolio_name"
            )
        
        with col2:
            if st.button("💾 Save Portfolio", key="save_portfolio_btn", use_container_width=True):
                if portfolio_name:
                    # Import auth functions
                    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    from utils.auth import save_user_data
                    
                    # Prepare portfolio data
                    portfolio_data = {
                        "tickers": valid_tickers,
                        "time_period": selected_period,
                        "risk_free_rate": risk_free_rate,
                        "analysis_date": datetime.now().isoformat(),
                        "max_sharpe_weights": max_sharpe_weights.tolist(),
                        "min_var_weights": min_var_weights.tolist(),
                        "equal_weights": equal_weights.tolist(),
                        "rebalancing_frequency": rebalancing_freq
                    }
                    
                    # Portfolio metrics
                    portfolio_metrics = {
                        "max_sharpe": portfolio_results['Maximum Sharpe'],
                        "min_variance": portfolio_results['Minimum Variance'],
                        "equal_weight": portfolio_results['Equal Weight']
                    }
                    
                    # Save to user portfolios
                    user = st.session_state.user
                    auth_manager = st.session_state.auth_manager
                    
                    success = auth_manager.save_user_portfolio(
                        user_id=user['id'],
                        portfolio_name=portfolio_name,
                        portfolio_data=portfolio_data,
                        portfolio_metrics=portfolio_metrics
                    )
                    
                    if success:
                        st.success(f"✅ Portfolio '{portfolio_name}' saved successfully!")
                    else:
                        st.error("Failed to save portfolio. Please try again.")
                else:
                    st.warning("Please enter a portfolio name to save.")
    
    # Portfolio composition charts
    st.subheader("Portfolio Allocations")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig_max_sharpe = optimizer.get_portfolio_composition(
            max_sharpe_weights, "Maximum Sharpe Ratio"
        )
        st.plotly_chart(fig_max_sharpe, use_container_width=True)
    
    with col2:
        fig_min_var = optimizer.get_portfolio_composition(
            min_var_weights, "Minimum Variance"
        )
        st.plotly_chart(fig_min_var, use_container_width=True)
    
    with col3:
        fig_equal = optimizer.get_portfolio_composition(
            equal_weights, "Equal Weight"
        )
        st.plotly_chart(fig_equal, use_container_width=True)
    
    # Efficient Frontier
    st.header("📊 Efficient Frontier Analysis")
    
    # Add tooltip for efficient frontier
    with st.expander("💡 What is the Efficient Frontier?"):
        st.markdown(get_tooltip_help("efficient_frontier"))
    
    with st.spinner("Generating efficient frontier..."):
        try:
            fig_frontier = optimizer.plot_efficient_frontier()
            if fig_frontier:
                st.plotly_chart(fig_frontier, use_container_width=True)
            else:
                st.error("Unable to generate efficient frontier.")
        except Exception as e:
            st.error(f"Error generating efficient frontier: {str(e)}")
    
    # Risk Analysis
    st.header("⚠️ Risk Analysis")
    
    selected_portfolio = st.selectbox(
        "Select portfolio for detailed risk analysis",
        options=list(portfolio_results.keys()),
        index=0
    )
    
    selected_weights = portfolio_results[selected_portfolio]['weights']
    portfolio_returns = (returns_data * selected_weights).sum(axis=1)
    
    # Initialize risk metrics
    risk_analyzer = RiskMetrics(portfolio_returns)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Metrics Summary")
        risk_summary = risk_analyzer.get_risk_summary()
        
        risk_display = pd.DataFrame([
            {"Metric": "Value at Risk (95%)", "Value": f"{risk_summary['VaR_95%']:.2%}"},
            {"Metric": "Expected Shortfall (95%)", "Value": f"{risk_summary['ES_95%']:.2%}"},
            {"Metric": "Maximum Drawdown", "Value": f"{risk_summary['Max_Drawdown']:.2%}"},
            {"Metric": "Sharpe Ratio", "Value": f"{risk_summary['Sharpe_Ratio']:.3f}"},
            {"Metric": "Sortino Ratio", "Value": f"{risk_summary['Sortino_Ratio']:.3f}"},
            {"Metric": "Calmar Ratio", "Value": f"{risk_summary['Calmar_Ratio']:.3f}"},
        ])
        
        st.dataframe(risk_display, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("Distribution Statistics")
        tail_metrics = risk_analyzer.tail_risk_metrics()
        
        dist_display = pd.DataFrame([
            {"Metric": "Skewness", "Value": f"{tail_metrics['Skewness']:.3f}"},
            {"Metric": "Excess Kurtosis", "Value": f"{tail_metrics['Excess_Kurtosis']:.3f}"},
            {"Metric": "Jarque-Bera Test", "Value": f"{tail_metrics['Jarque_Bera_PValue']:.4f}"},
            {"Metric": "Tail Ratio", "Value": f"{tail_metrics['Tail_Ratio']:.3f}"},
        ])
        
        st.dataframe(dist_display, use_container_width=True, hide_index=True)
    
    # Risk visualization
    try:
        risk_plots = risk_analyzer.plot_risk_analysis()
        
        if 'returns_distribution' in risk_plots:
            st.plotly_chart(risk_plots['returns_distribution'], use_container_width=True)
        
        if 'drawdown_analysis' in risk_plots:
            st.plotly_chart(risk_plots['drawdown_analysis'], use_container_width=True)
        
        if 'rolling_metrics' in risk_plots:
            st.plotly_chart(risk_plots['rolling_metrics'], use_container_width=True)
            
    except Exception as e:
        st.error(f"Error generating risk plots: {str(e)}")
    
    # Export functionality
    st.header("📥 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Export Portfolio Weights", use_container_width=True):
            weights_df = pd.DataFrame({
                'Asset': valid_tickers,
                'Max Sharpe Weights': max_sharpe_weights,
                'Min Variance Weights': min_var_weights,
                'Equal Weights': equal_weights
            })
            
            csv = weights_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"portfolio_weights_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📈 Export Performance Metrics", use_container_width=True):
            metrics_df = pd.DataFrame(comparison_data)
            csv = metrics_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"portfolio_metrics_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

# Navigation
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🏠 Back to Dashboard", use_container_width=True):
        st.switch_page("app.py")

with col2:
    if st.button("🔗 Statistical Arbitrage", use_container_width=True):
        st.switch_page("pages/statistical_arbitrage.py")

with col3:
    if st.button("📈 Time Series Analysis", use_container_width=True):
        st.switch_page("pages/time_series_analysis.py")
