import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Quantitative Finance Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main dashboard
def main_dashboard():
    st.title("📈 Quantitative Finance Platform")
    st.markdown("---")
    
    # Market overview section
    st.header("🌍 Market Overview")
    
    # Major indices
    indices = {
        "S&P 500": "^GSPC",
        "NASDAQ": "^IXIC", 
        "Dow Jones": "^DJI",
        "VIX": "^VIX"
    }
    
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        for i, (name, ticker) in enumerate(indices.items()):
            with [col1, col2, col3, col4][i]:
                try:
                    data = yf.download(ticker, period="2d", interval="1d", progress=False)
                    if not data.empty and len(data) >= 2:
                        current_price = data['Close'].iloc[-1]
                        prev_price = data['Close'].iloc[-2]
                        change = current_price - prev_price
                        change_pct = (change / prev_price) * 100
                        
                        st.metric(
                            label=name,
                            value=f"{current_price:.2f}",
                            delta=f"{change_pct:.2f}%"
                        )
                    else:
                        st.metric(label=name, value="N/A", delta="N/A")
                except Exception as e:
                    st.metric(label=name, value="Error", delta="N/A")
                    
    except Exception as e:
        st.error("Unable to fetch market data. Please check your internet connection.")
    
    st.markdown("---")
    
    # Platform features
    st.header("🚀 Platform Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📊 Portfolio Optimization")
        st.write("Modern Portfolio Theory implementation with efficient frontier calculation and Markowitz optimization.")
        if st.button("Launch Portfolio Optimizer"):
            st.switch_page("pages/portfolio_optimization.py")
    
    with col2:
        st.subheader("🔗 Statistical Arbitrage")
        st.write("Detect arbitrage opportunities through correlation analysis and pair trading strategies.")
        if st.button("Launch Arbitrage Analysis"):
            st.switch_page("pages/statistical_arbitrage.py")
    
    with col3:
        st.subheader("📈 Time Series Analysis")
        st.write("Advanced time series analysis with trend identification and seasonality detection.")
        if st.button("Launch Time Series Analysis"):
            st.switch_page("pages/time_series_analysis.py")
    
    col4, col5 = st.columns(2)
    
    with col4:
        st.subheader("⚡ Trading Strategies")
        st.write("Backtest and optimize various trading strategies with comprehensive performance metrics.")
        if st.button("Launch Strategy Backtesting"):
            st.switch_page("pages/trading_strategies.py")
    
    with col5:
        st.subheader("🤖 AI Analysis")
        st.write("AI-powered price prediction and pattern recognition using machine learning models.")
        if st.button("Launch AI Analysis"):
            st.switch_page("pages/ai_analysis.py")
    
    st.markdown("---")
    
    # Quick analysis section
    st.header("⚡ Quick Analysis")
    
    ticker = st.text_input("Enter a stock ticker for quick analysis:", value="AAPL").upper()
    
    if ticker:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            hist = yf.download(ticker, period="1y", progress=False)
            
            if not hist.empty:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Price chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=hist.index,
                        y=hist['Close'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color='#1f77b4', width=2)
                    ))
                    
                    fig.update_layout(
                        title=f"{ticker} - 1 Year Price Chart",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Key metrics
                    current_price = hist['Close'].iloc[-1]
                    year_high = hist['High'].max()
                    year_low = hist['Low'].min()
                    avg_volume = hist['Volume'].mean()
                    
                    st.metric("Current Price", f"${current_price:.2f}")
                    st.metric("52W High", f"${year_high:.2f}")
                    st.metric("52W Low", f"${year_low:.2f}")
                    st.metric("Avg Volume", f"{avg_volume:,.0f}")
                    
                    # Calculate simple metrics
                    returns = hist['Close'].pct_change().dropna()
                    volatility = returns.std() * np.sqrt(252) * 100
                    
                    st.metric("Annualized Volatility", f"{volatility:.2f}%")
                    
            else:
                st.error("Unable to fetch data for the specified ticker.")
                
        except Exception as e:
            st.error(f"Error analyzing ticker {ticker}: {str(e)}")

# Sidebar navigation
def sidebar_navigation():
    st.sidebar.title("🧭 Navigation")
    
    pages = {
        "🏠 Dashboard": "main",
        "📊 Portfolio Optimization": "portfolio_optimization",
        "🔗 Statistical Arbitrage": "statistical_arbitrage", 
        "📈 Time Series Analysis": "time_series_analysis",
        "⚡ Trading Strategies": "trading_strategies",
        "🤖 AI Analysis": "ai_analysis"
    }
    
    for page_name, page_key in pages.items():
        if st.sidebar.button(page_name, key=f"nav_{page_key}"):
            if page_key == "main":
                st.rerun()
            else:
                st.switch_page(f"pages/{page_key}.py")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.markdown("""
    This platform provides comprehensive quantitative finance tools including:
    - Modern Portfolio Theory
    - Statistical Arbitrage
    - Time Series Analysis  
    - Trading Strategy Backtesting
    - AI-Powered Analysis
    """)

if __name__ == "__main__":
    sidebar_navigation()
    main_dashboard()
