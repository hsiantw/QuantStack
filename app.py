import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import authentication system
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.auth import (
    init_session_state, show_auth_page, show_user_menu, require_auth,
    save_user_data, load_user_data
)

# Configure page
st.set_page_config(
    page_title="Quantitative Finance Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for QuantConnect-style UI
st.markdown("""
<style>
    /* Global Styling - QuantConnect Dark Theme */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 25%, #16213e 50%, #0f3460 75%, #0e1117 100%);
        color: #ffffff;
    }
    
    .main {
        padding-top: 1rem;
    }
    
    /* QuantConnect-style buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
        color: #000000;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #0099cc 0%, #006699 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.4);
    }
    
    /* Professional metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
        border: 1px solid #333344;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    /* QuantConnect Strategy Cards */
    .nav-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
        border: 1px solid #333344;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        cursor: pointer;
    }
    
    .nav-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #00d4ff, #0099cc, #006699);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .nav-card:hover {
        border-color: #00d4ff;
        box-shadow: 0 8px 25px rgba(0, 212, 255, 0.2);
        transform: translateY(-2px);
    }
    
    .nav-card:hover::before {
        opacity: 1;
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
            if data is not None and not data.empty and len(data) >= 1:
                current_price = data['Close'].iloc[-1]
                
                # Try to get previous price for change calculation
                if len(data) >= 2:
                    previous_price = data['Close'].iloc[-2]
                else:
                    # If only one day, try to get info from ticker object
                    try:
                        ticker_obj = yf.Ticker(ticker)
                        info = ticker_obj.info
                        if info and 'previousClose' in info and info['previousClose']:
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
    # Hero Section - QuantConnect Style
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0 3rem 0;">
        <h1 style="color: #00d4ff; font-size: 3.5rem; font-weight: 700; margin-bottom: 1rem; text-shadow: 0 0 20px rgba(0, 212, 255, 0.3);">
            Algorithm Lab
        </h1>
        <h2 style="color: #ffffff; font-size: 1.8rem; font-weight: 300; margin-bottom: 2rem;">
            Advanced Quantitative Finance Platform
        </h2>
        <p style="color: #b0b0b0; font-size: 1.2rem; max-width: 800px; margin: 0 auto; line-height: 1.6;">
            We are dedicated to providing investors with a cutting-edge platform for rapidly creating quant investment strategies. 
            Build, backtest, and deploy sophisticated trading algorithms with institutional-grade tools and real-time market data.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Actions Section
    st.markdown("### 🚀 Quick Start")
    
    quick_cols = st.columns(4)
    with quick_cols[0]:
        if st.button("📊 New Analysis", use_container_width=True, type="primary"):
            st.switch_page("pages/portfolio_optimization.py")
    with quick_cols[1]:
        if st.button("🤖 AI Strategies", use_container_width=True, type="secondary"):
            st.switch_page("pages/ai_analysis.py")
    with quick_cols[2]:
        if st.button("📈 Trading Signals", use_container_width=True, type="secondary"):
            st.switch_page("pages/trading_strategies.py")
    with quick_cols[3]:
        if st.button("📰 Market Intel", use_container_width=True, type="secondary"):
            st.switch_page("pages/news_and_economic_data.py")
    
    st.markdown("---")
    
    # Organized Module Categories
    st.markdown("### 🎯 Platform Modules")
    
    st.markdown("""
    <p style="color: #b0b0b0; font-size: 1.1rem; margin-bottom: 2rem;">
    Professional-grade quantitative finance tools organized by category for optimal workflow efficiency.
    </p>
    """, unsafe_allow_html=True)
    
    # Create tabs for the three main categories
    portfolio_tab, insights_tab, best_features_tab = st.tabs([
        "💼 Portfolio & Trading", 
        "🧠 AI Insights & Analysis", 
        "⭐ Platform Highlights"
    ])
    
    with portfolio_tab:
        st.markdown("#### Portfolio Management & Trading Tools")
        portfolio_features = [
            {
                "icon": "💼",
                "title": "Portfolio Optimization",
                "desc": "Modern Portfolio Theory with efficient frontier calculation and risk-adjusted returns optimization.",
                "page": "pages/portfolio_optimization.py"
            },
            {
                "icon": "📋",
                "title": "Portfolio Manager", 
                "desc": "Comprehensive portfolio tracking, rebalancing, and performance monitoring with real-time analytics.",
                "page": "pages/portfolio_manager.py"
            },
            {
                "icon": "📊",
                "title": "Trading Account Monitor",
                "desc": "Live trading account integration with Webull, real-time P&L tracking, and position analysis.",
                "page": "pages/trading_monitor.py"
            },
            {
                "icon": "⚡",
                "title": "Trading Strategy Backtesting",
                "desc": "Comprehensive strategy backtesting with 15+ algorithms, AI optimization, and performance metrics.",
                "page": "pages/trading_strategies.py"
            },
            {
                "icon": "🎯",
                "title": "Strategy Comparison & Optimization",
                "desc": "Advanced multi-strategy comparison with parameter optimization and risk-return analysis.",
                "page": "pages/strategy_comparison.py"
            },
            {
                "icon": "🌅",
                "title": "Opening Range Breakout (ORB)",
                "desc": "Research-backed day trading strategy with 675% returns vs 169% buy-and-hold (2016-2023).",
                "page": "pages/orb_strategy.py"
            },
            {
                "icon": "⚠️",
                "title": "Advanced Risk Management",
                "desc": "Monte Carlo simulations, stress testing, VaR analysis, and comprehensive portfolio risk assessment.",
                "page": "pages/risk_management.py"
            }
        ]
        
        cols = st.columns(2)
        for i, feature in enumerate(portfolio_features):
            with cols[i % 2]:
                create_feature_card(feature["icon"], feature["title"], feature["desc"], "Launch", feature["page"], "portfolio")
    
    with insights_tab:
        st.markdown("#### AI-Powered Analysis & Market Intelligence")
        insights_features = [
            {
                "icon": "🤖",
                "title": "AI Financial Analysis",
                "desc": "Machine learning price predictions with Random Forest, Gradient Boosting, and neural networks.",
                "page": "pages/ai_analysis.py"
            },
            {
                "icon": "📈",
                "title": "Statistical Arbitrage",
                "desc": "Pair trading with cointegration testing, spread analysis, and automated opportunity detection.",
                "page": "pages/statistical_arbitrage.py"
            },
            {
                "icon": "🔗",
                "title": "AI Pairs Trading",
                "desc": "Advanced AI-optimized pairs trading with 6 mean reversion strategies and real-time monitoring.",
                "page": "pages/ai_pairs_trading.py"
            },
            {
                "icon": "📊",
                "title": "Time Series Analysis",
                "desc": "ARIMA modeling, seasonality detection, trend analysis, and advanced econometric tools.",
                "page": "pages/time_series_analysis.py"
            },
            {
                "icon": "🧬",
                "title": "Advanced Market Analysis",
                "desc": "Dark pool detection, liquidity analysis, money flow, and institutional trading insights.",
                "page": "pages/advanced_market_analysis.py"
            },
            {
                "icon": "💧",
                "title": "Liquidity Analysis",
                "desc": "CoinGlass-style liquidation heat maps, order book depth, volume profiles, and market microstructure analysis.",
                "page": "pages/liquidity_analysis.py"
            },
            {
                "icon": "📰",
                "title": "News & Economic Intelligence",
                "desc": "Real-time news analysis, economic calendar, Fed updates, and sentiment-driven market intelligence.",
                "page": "pages/news_and_economic_data.py"
            },
            {
                "icon": "🏢",
                "title": "Fundamental Analysis",
                "desc": "Company valuation, financial ratio analysis, earnings forecasts, and DCF modeling.",
                "page": "pages/fundamental_analysis.py"
            },
            {
                "icon": "📚",
                "title": "Market Intelligence Sources",
                "desc": "Comprehensive data sources, APIs, and market intelligence aggregation platform.",
                "page": "pages/market_information_sources.py"
            },
            {
                "icon": "📡",
                "title": "Market Data Stream",
                "desc": "Real-time market data with StockData.org integration, intraday charts, and enhanced data feeds.",
                "page": "pages/market_data_stream.py"
            }
        ]
        
        cols = st.columns(2)
        for i, feature in enumerate(insights_features):
            with cols[i % 2]:
                create_feature_card(feature["icon"], feature["title"], feature["desc"], "Analyze", feature["page"], "insights")
    
    with best_features_tab:
        st.markdown("#### Platform Highlights - My Top Recommendations")
        st.markdown("""
        <p style="color: #00d4ff; font-size: 1rem; margin-bottom: 1.5rem;">
        These are the standout features that provide exceptional value for quantitative analysis and trading.
        </p>
        """, unsafe_allow_html=True)
        
        best_features = [
            {
                "icon": "🔗",
                "title": "AI Pairs Trading",
                "desc": "★ EXCEPTIONAL: 6 mathematical mean reversion strategies with AI optimization. Best-in-class cointegration analysis.",
                "page": "pages/ai_pairs_trading.py"
            },
            {
                "icon": "⚡",
                "title": "Trading Strategy Backtesting",
                "desc": "★ COMPREHENSIVE: 15+ strategies with AI optimization, extensive performance metrics, and PineScript generation.",
                "page": "pages/trading_strategies.py"
            },
            {
                "icon": "📊",
                "title": "Trading Account Monitor",
                "desc": "★ INNOVATIVE: Live Webull integration with QR authentication, real-time portfolio sync, and strategy validation.",
                "page": "pages/trading_monitor.py"
            },
            {
                "icon": "🧬",
                "title": "Advanced Market Analysis",
                "desc": "★ INSTITUTIONAL-GRADE: Dark pool detection, volume profile analysis, and professional liquidity metrics.",
                "page": "pages/advanced_market_analysis.py"
            },
            {
                "icon": "₿",
                "title": "Cryptocurrency Analysis",
                "desc": "★ SPECIALIZED: Complete crypto ecosystem analysis with DeFi metrics, on-chain data, and Fear & Greed index.",
                "page": "pages/crypto_analysis.py"
            },
            {
                "icon": "🌍",
                "title": "Multi-Asset Analysis",
                "desc": "★ DIVERSIFIED: Commodities, forex, futures analysis with cross-asset correlation and global market insights.",
                "page": "pages/commodities_forex_futures.py"
            }
        ]
        
        cols = st.columns(2)
        for i, feature in enumerate(best_features):
            with cols[i % 2]:
                create_feature_card(feature["icon"], feature["title"], feature["desc"], "Explore", feature["page"], "highlights")

    # Strategy Templates Section - QuantConnect Style  
    st.markdown("---")
    st.markdown("### 📋 Strategy Templates")
    
    template_cols = st.columns(3)
    
    templates = [
        {"name": "US Equity", "desc": "ETF Basket Pairs Trading, Momentum Strategies", "count": "25+"},
        {"name": "Alternative Data", "desc": "Sentiment Analysis, News-based Trading", "count": "12+"},
        {"name": "Futures", "desc": "All-Weather Portfolio, Volatility Trading", "count": "18+"} 
    ]
    
    for i, template in enumerate(templates):
        with template_cols[i]:
            st.markdown(f"""
            <div class="strategy-template">
                <h4 style="color: #00d4ff; margin: 0 0 0.5rem 0;">{template['name']}</h4>
                <p style="color: #b0b0b0; margin: 0 0 0.5rem 0; font-size: 0.9rem;">{template['desc']}</p>
                <span style="color: #00d4ff; font-size: 0.8rem; font-weight: 600;">{template['count']} strategies</span>
            </div>
            """, unsafe_allow_html=True)
    
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
    
    # Display indices in a responsive grid
    try:
        cols = st.columns(4)
        
        for i, (name, ticker) in enumerate(indices.items()):
            with cols[i % 4]:
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
    
    ticker = st.text_input("Enter a stock ticker for quick analysis:", value="SPY", key="main_dashboard_ticker").upper()
    
    if ticker:
        try:
            # Try to get historical data for chart, starting with shorter periods if longer fails
            hist = None
            periods_to_try = ["1y", "6mo", "3mo", "1mo", "5d"]
            
            for period in periods_to_try:
                try:
                    hist = yf.download(ticker, period=period, progress=False)
                    if hist is not None and not hist.empty:
                        break
                except:
                    continue
            
            if hist is not None and not hist.empty:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Price chart
                    st.subheader(f"📈 {ticker} Price Chart")
                    
                    # Debug info
                    st.write(f"Data points: {len(hist)}")
                    st.write(f"Date range: {hist.index[0]} to {hist.index[-1]}")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=hist.index,
                        y=hist['Close'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color='#00d4ff', width=3)
                    ))
                    
                    fig.update_layout(
                        title=f"{ticker} - Price Chart",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        height=400,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        xaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
                        yaxis=dict(gridcolor='rgba(128,128,128,0.2)')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Metrics with better error handling
                    try:
                        if hist is not None and not hist.empty and len(hist) > 0:
                            current_price = float(hist['Close'].iloc[-1])
                            previous_price = float(hist['Close'].iloc[-2]) if len(hist) > 1 else float(hist['Close'].iloc[0])
                            
                            change = current_price - previous_price
                            change_pct = (change / previous_price) * 100 if previous_price != 0 else 0
                            
                            st.metric(
                                label="Current Price",
                                value=f"${current_price:.2f}",
                                delta=f"{change:.2f} ({change_pct:+.2f}%)"
                            )
                            
                            # Additional metrics with safety checks
                            try:
                                high_period = float(hist['High'].max())
                                low_period = float(hist['Low'].min())
                                
                                st.metric("Period High", f"${high_period:.2f}")
                                st.metric("Period Low", f"${low_period:.2f}")
                                
                                # Volume metric
                                if 'Volume' in hist.columns:
                                    avg_volume = hist['Volume'].mean()
                                    if not pd.isna(avg_volume):
                                        st.metric("Avg Volume", f"{avg_volume:,.0f}")
                                    else:
                                        st.metric("Avg Volume", "N/A")
                                else:
                                    st.metric("Avg Volume", "N/A")
                                
                                # Volatility calculation with proper error handling
                                if len(hist) >= 20:
                                    returns = hist['Close'].pct_change().dropna()
                                    if len(returns) > 1:
                                        volatility = returns.std() * (252**0.5)
                                        if not pd.isna(volatility):
                                            st.metric("Annualized Volatility", f"{volatility:.1%}")
                                        else:
                                            st.metric("Annualized Volatility", "N/A")
                                    else:
                                        st.metric("Annualized Volatility", "N/A")
                                else:
                                    st.metric("Annualized Volatility", "N/A")
                                    
                            except Exception as metric_error:
                                st.metric("Period High", "N/A")
                                st.metric("Period Low", "N/A") 
                                st.metric("Avg Volume", "N/A")
                                st.metric("Annualized Volatility", "N/A")
                        else:
                            # Fallback metrics
                            st.metric("Current Price", "N/A")
                            st.metric("Period High", "N/A")
                            st.metric("Period Low", "N/A")
                            st.metric("Avg Volume", "N/A")
                            st.metric("Annualized Volatility", "N/A")
                            
                    except Exception as e:
                        st.error(f"Error calculating metrics: {str(e)}")
                        # Show basic placeholder metrics
                        st.metric("Current Price", "N/A")
                        st.metric("Period High", "N/A")
                        st.metric("Period Low", "N/A")
                        st.metric("Avg Volume", "N/A")
                        st.metric("Annualized Volatility", "N/A")
                    
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

def create_feature_card(icon, title, desc, button_text, page, context="default"):
    """Create a feature navigation card with QuantConnect styling"""
    st.markdown(f"""
    <div class="nav-card">
        <div class="nav-card-icon" style="font-size: 2.5rem; margin-bottom: 1rem; text-align: center;">{icon}</div>
        <div class="nav-card-title" style="color: #00d4ff; font-size: 1.3rem; font-weight: 600; margin-bottom: 0.8rem; text-align: center;">{title}</div>
        <div class="nav-card-desc" style="color: #b0b0b0; font-size: 0.9rem; line-height: 1.4; text-align: center;">{desc}</div>
        <div style="margin-top: 1rem; text-align: center;">
            <span style="color: #00d4ff; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px;">Click to {button_text} →</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create unique key with context to avoid duplicates
    unique_key = f"card_{context}_{title.replace(' ', '_').replace('&', 'and')}"
    if st.button(f"{button_text} {title}", key=unique_key, use_container_width=True):
        st.switch_page(page)

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

def create_quantconnect_sidebar():
    """Create QuantConnect-style sidebar navigation"""
    with st.sidebar:
        # Logo/Brand section
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h2 style="color: #00d4ff; margin: 0; font-weight: 700;">Algorithm Lab</h2>
            <p style="color: #888; font-size: 0.9rem; margin: 0.5rem 0;">v3.0</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Quick Start section
        st.markdown("#### 🚀 Start")
        if st.button("📊 New Analysis", key="sidebar_analysis", use_container_width=True):
            st.switch_page("pages/portfolio_optimization.py")
        if st.button("🔍 Open Project", key="sidebar_open", use_container_width=True):
            st.switch_page("pages/trading_strategies.py")
        if st.button("🎯 Explore Strategies", key="sidebar_explore", use_container_width=True):
            st.switch_page("pages/ai_pairs_trading.py")
        
        st.markdown("---")
        
        # Resources section
        st.markdown("#### 📚 Resources")
        st.markdown("""
        - [📖 Learning Center](#)
        - [👥 Community](#)
        - [📊 Documentation](#)
        - [🏆 Quant League](#)
        """)
        
        st.markdown("---")
        
        # Account section - show user-specific data
        if st.session_state.get('authenticated', False) and st.session_state.get('user'):
            user = st.session_state.user
            st.markdown(f"#### 👤 {user['username']}")
            
            # Load user preferences and stats
            saved_strategies = load_user_data('saved_strategies', [])
            saved_portfolios = load_user_data('saved_portfolios', [])
            
            # Account metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Portfolios", len(saved_portfolios))
            with col2:
                st.metric("Strategies", len(saved_strategies))
            
            # User tier and progress
            tier = user.get('subscription_tier', 'free').title()
            st.markdown(f"**Tier:** {tier}")
            
            # Progress indicators based on usage
            usage_score = min((len(saved_strategies) + len(saved_portfolios)) * 10, 100)
            if usage_score > 0:
                st.progress(usage_score / 100, text=f"Usage Score: {usage_score}%")
            else:
                st.progress(0.1, text="Getting Started: 10%")
        else:
            # Fallback for non-authenticated users
            st.markdown("#### 👤 Guest Account")
            st.metric("Status", "Guest Mode")
            st.info("Login to save your work")
        
        st.markdown("---")
        
        # Market status
        st.markdown("#### 🌍 Market Status")
        
        # Simple market indicators
        market_cols = st.columns(2)
        with market_cols[0]:
            st.metric("S&P 500", "5,870", "+0.5%")
        with market_cols[1]:
            st.metric("VIX", "14.2", "-2.1%")

if __name__ == "__main__":
    # Initialize authentication system
    init_session_state()
    
    # Check if user is authenticated
    if not st.session_state.get('authenticated', False):
        # Show authentication page
        show_auth_page()
    else:
        # User is authenticated, show main application
        show_user_menu()  # Add user menu to sidebar
        create_quantconnect_sidebar()
        main_dashboard()
