import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_most_recent_price(ticker, periods=["2d", "5d", "1mo", "3mo"]):
    """
    Searches for the most recent available price data across multiple time periods.
    Returns (current_price, previous_price, success_flag)
    """
    for period in periods:
        try:
            data = yf.download(ticker, period=period, interval="1d", progress=False)
            if not data.empty and len(data) >= 1:
                current_price = data['Close'].iloc[-1]
                
                # Try to get previous price for change calculation
                if len(data) >= 2:
                    previous_price = data['Close'].iloc[-2]
                else:
                    # If only one day, try to get info from ticker object
                    try:
                        ticker_obj = yf.Ticker(ticker)
                        info = ticker_obj.info
                        if 'previousClose' in info and info['previousClose']:
                            previous_price = info['previousClose']
                        else:
                            previous_price = current_price  # No change calculation possible
                    except:
                        previous_price = current_price
                
                return float(current_price), float(previous_price), True
                
        except Exception as e:
            continue
    
    # If all periods fail, try to get basic info from ticker object
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        
        current_price = None
        previous_price = None
        
        # Try different price fields
        price_fields = ['currentPrice', 'regularMarketPrice', 'lastPrice', 'navPrice']
        for field in price_fields:
            if field in info and info[field]:
                current_price = float(info[field])
                break
        
        if current_price and 'previousClose' in info and info['previousClose']:
            previous_price = float(info['previousClose'])
        else:
            previous_price = current_price
            
        if current_price:
            return current_price, previous_price, True
            
    except Exception as e:
        pass
    
    return None, None, False

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
    
    # Major indices - US and Worldwide
    indices = {
        "S&P 500": "^GSPC",
        "NASDAQ": "^IXIC", 
        "Dow Jones": "^DJI",
        "VIX": "^VIX",
        "Nikkei 225": "^N225",
        "Hang Seng": "^HSI",
        "FTSE 100": "^FTSE",
        "DAX": "^GDAXI"
    }
    
    # Display in two rows of 4 columns each
    st.subheader("🇺🇸 US Markets")
    col1, col2, col3, col4 = st.columns(4)
    us_indices = list(indices.items())[:4]
    
    try:
        for i, (name, ticker) in enumerate(us_indices):
            with [col1, col2, col3, col4][i]:
                current_price, prev_price, success = get_most_recent_price(ticker)
                
                if success and current_price is not None:
                    try:
                        change = current_price - prev_price
                        change_pct = (change / prev_price) * 100 if prev_price != 0 else 0
                        
                        st.metric(
                            label=name,
                            value=f"{current_price:.2f}",
                            delta=f"{change_pct:+.2f}%"
                        )
                    except:
                        st.metric(
                            label=name,
                            value=f"{current_price:.2f}",
                            delta="N/A"
                        )
                else:
                    st.metric(
                        label=name,
                        value="N/A",
                        delta="Data unavailable"
                    )
    except Exception as e:
        st.error(f"Error loading US market data: {str(e)}")
    
    st.subheader("🌍 International Markets")  
    col5, col6, col7, col8 = st.columns(4)
    intl_indices = list(indices.items())[4:]
    
    try:
        for i, (name, ticker) in enumerate(intl_indices):
            with [col5, col6, col7, col8][i]:
                current_price, prev_price, success = get_most_recent_price(ticker)
                
                if success and current_price is not None:
                    try:
                        change = current_price - prev_price
                        change_pct = (change / prev_price) * 100 if prev_price != 0 else 0
                        
                        st.metric(
                            label=name,
                            value=f"{current_price:.2f}",
                            delta=f"{change_pct:+.2f}%"
                        )
                    except:
                        st.metric(
                            label=name,
                            value=f"{current_price:.2f}",
                            delta="N/A"
                        )
                else:
                    st.metric(
                        label=name,
                        value="N/A",
                        delta="Data unavailable"
                    )
                    
    except Exception as e:
        st.error(f"Error loading international market data: {str(e)}")
    
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
    
    col4, col5, col6 = st.columns(3)
    
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
    
    with col6:
        st.subheader("🎯 Strategy Optimizer")
        st.write("Advanced strategy comparison with multiple indicator combinations to find optimal approaches.")
        if st.button("Launch Strategy Optimizer"):
            st.switch_page("pages/strategy_comparison.py")
    
    # Additional features row
    col7, col8, col9 = st.columns(3)
    
    with col7:
        st.subheader("🤖 AI Pairs Trading")
        st.write("AI-powered pairs trading with statistical arbitrage analysis to find the best cointegrated pairs.")
        if st.button("Launch AI Pairs Trading"):
            st.switch_page("pages/ai_pairs_trading.py")
    
    with col8:
        st.subheader("📰 Market Information Sources")
        st.write("Comprehensive guide to critical data sources including SEC filings, economic indicators, and Fed data.")
        if st.button("Launch Information Sources Guide"):
            st.switch_page("pages/market_information_sources.py")
    
    with col9:
        st.subheader("💡 Educational Resources")
        st.write("Learn about financial concepts, formulas, and market analysis techniques with interactive tooltips.")
        st.info("Educational tooltips are available throughout all platform pages")
    
    st.markdown("---")
    
    # Quick analysis section
    st.header("⚡ Quick Analysis")
    
    ticker = st.text_input("Enter a stock ticker for quick analysis:", value="SPY").upper()
    
    if ticker:
        try:
            # Try to get historical data for chart, starting with shorter periods if longer fails
            hist = None
            periods_to_try = ["1y", "6mo", "3mo", "1mo", "5d"]
            
            for period in periods_to_try:
                try:
                    hist = yf.download(ticker, period=period, progress=False)
                    if not hist.empty:
                        break
                except:
                    continue
            
            if hist is not None and not hist.empty:
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
                        title=f"{ticker} - Price Chart",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Key metrics using most recent data
                    current_price, prev_price, price_success = get_most_recent_price(ticker)
                    
                    if price_success and current_price:
                        st.metric("Current Price", f"${current_price:.2f}")
                    else:
                        st.metric("Current Price", "N/A")
                    
                    # Calculate metrics from available historical data
                    try:
                        year_high = hist['High'].max()
                        year_low = hist['Low'].min()
                        avg_volume = hist['Volume'].mean()
                        
                        st.metric("Period High", f"${year_high:.2f}")
                        st.metric("Period Low", f"${year_low:.2f}")
                        st.metric("Avg Volume", f"{avg_volume:,.0f}")
                        
                        # Calculate volatility
                        returns = hist['Close'].pct_change().dropna()
                        if len(returns) > 1:
                            volatility = returns.std() * np.sqrt(252) * 100
                            st.metric("Annualized Volatility", f"{volatility:.2f}%")
                        else:
                            st.metric("Annualized Volatility", "N/A")
                            
                    except Exception as e:
                        st.error("Unable to calculate some metrics")
                    
            else:
                # Try to get basic price info even if historical data fails
                current_price, prev_price, price_success = get_most_recent_price(ticker)
                
                if price_success and current_price:
                    st.info(f"Limited data available for {ticker}")
                    st.metric("Current Price", f"${current_price:.2f}")
                    
                    change = current_price - prev_price if prev_price else 0
                    change_pct = (change / prev_price) * 100 if prev_price and prev_price != 0 else 0
                    
                    if abs(change_pct) > 0.001:
                        st.metric("Daily Change", f"{change_pct:.2f}%")
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
