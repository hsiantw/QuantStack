import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Quantitative Finance Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI/UX
st.markdown("""
<style>
    /* Global Styling */
    .main {
        padding-top: 1rem;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(90deg, #1f77b4, #17becf);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 20px rgba(31, 119, 180, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.8rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        background: linear-gradient(45deg, #ffffff, #e3f2fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .main-header p {
        margin: 1rem 0 0 0;
        font-size: 1.3rem;
        opacity: 0.95;
        font-weight: 300;
    }
    
    /* Navigation Cards */
    .nav-card {
        background: linear-gradient(135deg, #262730, #3d3d5c);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border: 1px solid #404040;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        position: relative;
        overflow: hidden;
    }
    
    .nav-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(31, 119, 180, 0.1), transparent);
        transition: left 0.6s;
    }
    
    .nav-card:hover::before {
        left: 100%;
    }
    
    .nav-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 12px 30px rgba(31, 119, 180, 0.25);
        border-color: #1f77b4;
    }
    
    .nav-card-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        display: block;
        filter: drop-shadow(0 0 10px rgba(31, 119, 180, 0.5));
    }
    
    .nav-card-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.75rem;
        text-shadow: 0 0 10px rgba(31, 119, 180, 0.3);
    }
    
    .nav-card-desc {
        color: #c4c4c4;
        font-size: 1rem;
        line-height: 1.6;
        font-weight: 300;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #333;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #1f77b4, #17becf);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.3);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(45deg, #17becf, #1f77b4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #b3b3b3;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-change {
        font-size: 0.9rem;
        margin-top: 0.5rem;
        font-weight: 600;
    }
    
    .positive-change {
        color: #4caf50;
    }
    
    .negative-change {
        color: #f44336;
    }
    
    /* Enhanced Button Styling - Uniform Size */
    .stButton > button {
        background: linear-gradient(45deg, #1f77b4, #17becf) !important;
        color: white !important;
        border: none !important;
        padding: 1rem 1.5rem !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 6px 20px rgba(31, 119, 180, 0.4) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        position: relative !important;
        overflow: hidden !important;
        width: 100% !important;
        height: 55px !important;
        min-height: 55px !important;
        max-height: 55px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        white-space: nowrap !important;
        text-overflow: ellipsis !important;
        margin: 0.5rem 0 !important;
    }
    
    /* Force button container uniformity */
    .stButton {
        width: 100% !important;
        margin: 0.5rem 0 !important;
    }
    
    div[data-testid="column"] .stButton {
        width: 100% !important;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(31, 119, 180, 0.5);
        background: linear-gradient(45deg, #1565c0, #0097a7);
    }
    
    /* Info Boxes with Animations */
    .info-box {
        background: linear-gradient(135deg, #0f3460, #16537e);
        padding: 2rem;
        border-radius: 12px;
        border-left: 5px solid #17becf;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        position: relative;
        overflow: hidden;
    }
    
    .info-box::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(23,190,207,0.1) 0%, transparent 70%);
        animation: pulse-glow 4s ease-in-out infinite;
    }
    
    @keyframes pulse-glow {
        0%, 100% { transform: scale(0.8); opacity: 0.5; }
        50% { transform: scale(1.2); opacity: 0.8; }
    }
    
    /* Enhanced Alert Boxes */
    .success-box {
        background: linear-gradient(135deg, #2e7d32, #388e3c);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
        animation: slideInLeft 0.5s ease-out;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #f57c00, #ff9800);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(255, 152, 0, 0.3);
        animation: slideInLeft 0.5s ease-out;
    }
    
    .error-box {
        background: linear-gradient(135deg, #d32f2f, #f44336);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #f44336;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(244, 67, 54, 0.3);
        animation: slideInLeft 0.5s ease-out;
    }
    
    @keyframes slideInLeft {
        from { transform: translateX(-100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    /* Enhanced Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: rgba(38, 39, 48, 0.5);
        padding: 0.5rem;
        border-radius: 12px;
        border: 1px solid #404040;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #b3b3b3;
        border-radius: 10px;
        padding: 1rem 2rem;
        border: 1px solid transparent;
        transition: all 0.3s ease;
        font-weight: 600;
        position: relative;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #1f77b4, #17becf);
        color: white;
        border-color: #1f77b4;
        box-shadow: 0 4px 15px rgba(31, 119, 180, 0.4);
        transform: scale(1.05);
    }
    
    .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
        background: rgba(31, 119, 180, 0.1);
        border-color: rgba(31, 119, 180, 0.3);
        color: #17becf;
    }
    
    /* Sidebar Enhancements */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e, #16213e);
        border-right: 2px solid rgba(31, 119, 180, 0.3);
    }
    
    /* Loading Animation */
    @keyframes loading-pulse {
        0% { 
            opacity: 0.6; 
            transform: scale(1);
        }
        50% { 
            opacity: 1; 
            transform: scale(1.05);
        }
        100% { 
            opacity: 0.6; 
            transform: scale(1);
        }
    }
    
    .loading-indicator {
        animation: loading-pulse 2s infinite;
        color: #17becf;
        text-align: center;
        padding: 3rem;
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    /* Chart Containers */
    .js-plotly-plot {
        background: rgba(38, 39, 48, 0.3) !important;
        border-radius: 15px;
        border: 1px solid #404040;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .js-plotly-plot:hover {
        box-shadow: 0 8px 30px rgba(31, 119, 180, 0.2);
        border-color: rgba(31, 119, 180, 0.5);
    }
    
    /* Data Tables */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 1px solid #404040;
    }
    
    /* Progress Bars */
    .stProgress .st-bo {
        background: linear-gradient(90deg, #1f77b4, #17becf);
        border-radius: 10px;
    }
    
    /* Mobile Responsiveness */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2.2rem;
        }
        
        .main-header p {
            font-size: 1.1rem;
        }
        
        .nav-card {
            margin: 0.5rem 0;
            padding: 1.5rem;
        }
        
        .nav-card-icon {
            font-size: 2.5rem;
        }
        
        .nav-card-title {
            font-size: 1.3rem;
        }
        
        .metric-card {
            margin: 0.25rem;
            padding: 1.5rem;
        }
        
        .metric-value {
            font-size: 2rem;
        }
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a2e;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #1f77b4, #17becf);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #1565c0, #0097a7);
    }
</style>
""", unsafe_allow_html=True)

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
    # Platform features with modern card design (moved to top)
    st.markdown("### 🚀 Platform Features")
    
    # Define platform features
    features = [
        {
            "icon": "📊",
            "title": "Portfolio Optimization",
            "desc": "Modern Portfolio Theory implementation with efficient frontier calculation and Markowitz optimization for risk-adjusted returns.",
            "page": "pages/portfolio_optimization.py"
        },
        {
            "icon": "🔗",
            "title": "Statistical Arbitrage",
            "desc": "Detect arbitrage opportunities through correlation analysis and pair trading strategies with cointegration testing.",
            "page": "pages/statistical_arbitrage.py"
        },
        {
            "icon": "📈",
            "title": "Time Series Analysis",
            "desc": "Advanced time series analysis with ARIMA modeling, trend identification and seasonality detection.",
            "page": "pages/time_series_analysis.py"
        },
        {
            "icon": "⚡",
            "title": "Trading Strategies",
            "desc": "Backtest and optimize various trading strategies with comprehensive performance metrics and risk analysis.",
            "page": "pages/trading_strategies.py"
        },
        {
            "icon": "🤖",
            "title": "AI Analysis",
            "desc": "AI-powered price prediction and pattern recognition using machine learning models with feature engineering.",
            "page": "pages/ai_analysis.py"
        },
        {
            "icon": "🎯",
            "title": "AI Pairs Trading",
            "desc": "Comprehensive pairs trading system with AI-optimized strategies, cointegration testing and mean reversion analysis.",
            "page": "pages/ai_pairs_trading.py"
        },
        {
            "icon": "🔍",
            "title": "Advanced Market Analysis",
            "desc": "Money flow, liquidity analysis, dark pool detection, crypto analysis and institutional trading secrets.",
            "page": "pages/advanced_market_analysis.py"
        },
        {
            "icon": "₿",
            "title": "Crypto Analysis",
            "desc": "Comprehensive cryptocurrency analysis with DeFi metrics, on-chain data, fear & greed index and correlation analysis.",
            "page": "pages/crypto_analysis.py"
        },

    ]
    
    # Display features in grid layout (adjusted for 8 features)
    # First row: 3 features
    cols1 = st.columns(3)
    # Second row: 3 features  
    cols2 = st.columns(3)
    # Third row: 2 features centered
    cols3 = st.columns([1, 2, 2, 1])
    
    all_cols = list(cols1) + list(cols2) + [cols3[1], cols3[2]]
    
    for i, feature in enumerate(features):
        with all_cols[i]:
            st.markdown(f"""
            <div class="nav-card" onclick="window.open('#{feature['page']}', '_self')">
                <div class="nav-card-icon">{feature['icon']}</div>
                <div class="nav-card-title">{feature['title']}</div>
                <div class="nav-card-desc">{feature['desc']}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Add the button functionality with consistent sizing
            if st.button(f"Launch {feature['title']}", key=f"btn_{i}"):
                st.switch_page(feature['page'])
    
    st.markdown("---")
    
    # Market overview section with enhanced styling (moved below navigation)
    st.markdown("### 🌍 Global Market Overview")
    
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
