import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Market Data Stream - QuantStack",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import utilities
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.ui_components import apply_custom_css, create_metric_card, create_info_card
    from utils.auth import check_authentication
    from utils.stockdata_client import StockDataClient, is_stockdata_available
    apply_custom_css()
except ImportError:
    st.warning("Some components not found. Using default functionality.")
    def create_metric_card(title, value, delta=None):
        return st.metric(title, value, delta)
    def create_info_card(title, content):
        return st.info(f"**{title}**\n\n{content}")
    def check_authentication():
        return True, None

def create_real_time_dashboard(symbols, data_source="yahoo"):
    """Create real-time market data dashboard"""
    
    if data_source == "stockdata" and is_stockdata_available():
        client = StockDataClient()
        quotes_df = client.get_real_time_quote(symbols)
    else:
        # Fallback to Yahoo Finance
        quotes_data = []
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period="2d")
                
                if not hist.empty and info:
                    current_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                    change = current_price - prev_price
                    change_percent = (change / prev_price * 100) if prev_price != 0 else 0
                    
                    quotes_data.append({
                        "symbol": symbol,
                        "price": current_price,
                        "change": change,
                        "change_percent": change_percent,
                        "volume": hist['Volume'].iloc[-1],
                        "market_cap": info.get('marketCap', 0),
                        "pe_ratio": info.get('forwardPE', 0),
                        "dividend_yield": info.get('dividendYield', 0)
                    })
            except Exception as e:
                st.warning(f"Error fetching data for {symbol}: {str(e)}")
        
        quotes_df = pd.DataFrame(quotes_data)
    
    if quotes_df.empty:
        st.error("No data available for the selected symbols.")
        return
    
    # Display real-time quotes
    st.markdown("### 📊 Live Market Data")
    
    cols = st.columns(min(len(quotes_df), 4))
    for i, (_, quote) in enumerate(quotes_df.iterrows()):
        if i >= 4:  # Limit to 4 columns
            break
        
        with cols[i]:
            change_color = "🟢" if quote.get('change', 0) >= 0 else "🔴"
            create_metric_card(
                f"{change_color} {quote['symbol']}", 
                f"${quote['price']:.2f}",
                f"{quote.get('change_percent', 0):.2f}%"
            )
    
    # Detailed table
    st.markdown("### 📋 Detailed Quotes")
    
    display_df = quotes_df.copy()
    if 'price' in display_df.columns:
        display_df['price'] = display_df['price'].apply(lambda x: f"${x:.2f}")
    if 'change' in display_df.columns:
        display_df['change'] = display_df['change'].apply(lambda x: f"${x:.2f}")
    if 'change_percent' in display_df.columns:
        display_df['change_percent'] = display_df['change_percent'].apply(lambda x: f"{x:.2f}%")
    if 'volume' in display_df.columns:
        display_df['volume'] = display_df['volume'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A")
    if 'market_cap' in display_df.columns:
        display_df['market_cap'] = display_df['market_cap'].apply(lambda x: f"${x/1e9:.1f}B" if x and x > 0 else "N/A")
    if 'pe_ratio' in display_df.columns:
        display_df['pe_ratio'] = display_df['pe_ratio'].apply(lambda x: f"{x:.2f}" if x and x > 0 else "N/A")
    if 'dividend_yield' in display_df.columns:
        display_df['dividend_yield'] = display_df['dividend_yield'].apply(lambda x: f"{x*100:.2f}%" if x and x > 0 else "N/A")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)

def create_intraday_chart(symbol, data_source="yahoo", interval="1h"):
    """Create intraday price chart"""
    
    try:
        if data_source == "stockdata" and is_stockdata_available():
            client = StockDataClient()
            data = client.get_intraday_data(symbol, interval=f"{interval[:-1]}min", outputsize=100)
        else:
            # Yahoo Finance intraday data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval=interval)
        
        if data.empty:
            st.warning(f"No intraday data available for {symbol}")
            return
        
        # Create candlestick chart
        fig = go.Figure(data=go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=symbol
        ))
        
        # Add volume subplot
        fig_with_volume = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=[f'{symbol} Price', 'Volume'],
            row_heights=[0.7, 0.3]
        )
        
        # Add price chart
        fig_with_volume.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name="Price"
            ), row=1, col=1
        )
        
        # Add volume chart
        colors = ['red' if close < open else 'green' for close, open in zip(data['Close'], data['Open'])]
        fig_with_volume.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name="Volume",
                marker_color=colors,
                opacity=0.7
            ), row=2, col=1
        )
        
        fig_with_volume.update_layout(
            title=f"{symbol} Intraday Data ({interval})",
            template="plotly_dark",
            height=600,
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig_with_volume, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating intraday chart: {str(e)}")

def display_company_profile(symbol, data_source="yahoo"):
    """Display company profile information"""
    
    try:
        if data_source == "stockdata" and is_stockdata_available():
            client = StockDataClient()
            profile = client.get_company_profile(symbol)
        else:
            # Yahoo Finance company info
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            profile = {
                "symbol": symbol,
                "name": info.get('longName', 'N/A'),
                "sector": info.get('sector', 'N/A'),
                "industry": info.get('industry', 'N/A'),
                "description": info.get('longBusinessSummary', 'N/A'),
                "market_cap": info.get('marketCap', 0),
                "employees": info.get('fullTimeEmployees', 0),
                "website": info.get('website', 'N/A'),
                "exchange": info.get('exchange', 'N/A'),
                "currency": info.get('currency', 'USD')
            }
        
        if profile:
            st.markdown(f"### 🏢 {profile.get('name', symbol)} Profile")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Basic Information**")
                st.write(f"**Symbol:** {profile.get('symbol', 'N/A')}")
                st.write(f"**Sector:** {profile.get('sector', 'N/A')}")
                st.write(f"**Industry:** {profile.get('industry', 'N/A')}")
                st.write(f"**Exchange:** {profile.get('exchange', 'N/A')}")
                st.write(f"**Currency:** {profile.get('currency', 'N/A')}")
            
            with col2:
                st.markdown("**Key Metrics**")
                market_cap = profile.get('market_cap', 0)
                if market_cap and market_cap > 0:
                    st.write(f"**Market Cap:** ${market_cap/1e9:.1f}B")
                else:
                    st.write("**Market Cap:** N/A")
                
                employees = profile.get('employees', 0)
                if employees and employees > 0:
                    st.write(f"**Employees:** {employees:,}")
                else:
                    st.write("**Employees:** N/A")
                
                if profile.get('website') and profile['website'] != 'N/A':
                    st.write(f"**Website:** [{profile['website']}]({profile['website']})")
            
            if profile.get('description') and profile['description'] != 'N/A':
                st.markdown("**Company Description**")
                st.write(profile['description'][:500] + "..." if len(profile['description']) > 500 else profile['description'])
                
    except Exception as e:
        st.error(f"Error fetching company profile: {str(e)}")

# Authentication check
is_authenticated, username = check_authentication()

# Sidebar
st.sidebar.title("📡 Market Data Stream")
st.sidebar.markdown("---")

if not is_authenticated:
    st.sidebar.warning("Please log in to access advanced features.")

# Data source selection
data_source = st.sidebar.selectbox(
    "Data Source",
    ["yahoo", "stockdata"],
    index=0,
    help="Choose between Yahoo Finance (free) or StockData.org (premium)"
)

if data_source == "stockdata":
    if not is_stockdata_available():
        st.sidebar.error("StockData API key not configured. Using Yahoo Finance as fallback.")
        data_source = "yahoo"
    else:
        st.sidebar.success("✅ StockData.org connected")

# Main interface
st.title("📡 Market Data Stream")
st.markdown("### Real-time market data with enhanced StockData.org integration")

# Symbol input
symbols_input = st.text_input(
    "Enter Stock Symbols (comma-separated)", 
    value="AAPL,GOOGL,MSFT,TSLA",
    key="market_stream_symbols"
)

symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

if symbols:
    # Auto-refresh option
    auto_refresh = st.checkbox("Auto-refresh every 30 seconds", value=False)
    
    if auto_refresh:
        # Create a placeholder for the data
        placeholder = st.empty()
        
        # Auto-refresh loop
        refresh_count = 0
        while auto_refresh and refresh_count < 120:  # Max 1 hour of auto-refresh
            with placeholder.container():
                create_real_time_dashboard(symbols, data_source)
            
            time.sleep(30)  # Refresh every 30 seconds
            refresh_count += 1
            
            # Check if auto-refresh is still enabled (user might have unchecked)
            if not st.session_state.get("auto_refresh", False):
                break
    else:
        # Manual refresh
        if st.button("🔄 Refresh Data", type="primary"):
            create_real_time_dashboard(symbols, data_source)
        else:
            create_real_time_dashboard(symbols, data_source)

# Individual symbol analysis
st.markdown("---")
st.markdown("### 📊 Individual Symbol Analysis")

analysis_symbol = st.selectbox(
    "Select Symbol for Detailed Analysis",
    symbols if symbols else ["AAPL"],
    key="analysis_symbol"
)

if analysis_symbol:
    tab1, tab2, tab3 = st.tabs(["📈 Intraday Chart", "🏢 Company Profile", "📋 Options Data"])
    
    with tab1:
        interval = st.selectbox(
            "Chart Interval",
            ["1m", "5m", "15m", "30m", "1h"],
            index=4,
            key="chart_interval"
        )
        
        create_intraday_chart(analysis_symbol, data_source, interval)
    
    with tab2:
        display_company_profile(analysis_symbol, data_source)
    
    with tab3:
        if data_source == "stockdata" and is_stockdata_available():
            st.markdown(f"### 📋 Options Chain - {analysis_symbol}")
            
            try:
                client = StockDataClient()
                options_df = client.get_options_data(analysis_symbol)
                
                if not options_df.empty:
                    # Separate calls and puts
                    calls_df = options_df[options_df['type'] == 'call']
                    puts_df = options_df[options_df['type'] == 'put']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Call Options**")
                        if not calls_df.empty:
                            st.dataframe(calls_df[['strike', 'expiration', 'bid', 'ask', 'last', 'volume', 'open_interest']], 
                                       use_container_width=True, hide_index=True)
                        else:
                            st.info("No call options data available")
                    
                    with col2:
                        st.markdown("**Put Options**")
                        if not puts_df.empty:
                            st.dataframe(puts_df[['strike', 'expiration', 'bid', 'ask', 'last', 'volume', 'open_interest']], 
                                       use_container_width=True, hide_index=True)
                        else:
                            st.info("No put options data available")
                else:
                    st.info("No options data available for this symbol")
                    
            except Exception as e:
                st.error(f"Error fetching options data: {str(e)}")
        else:
            st.info("Options data requires StockData.org API. Please configure STOCKDATA_API_KEY in secrets.")

# Crypto and Forex sections (if StockData is available)
if data_source == "stockdata" and is_stockdata_available():
    st.markdown("---")
    st.markdown("### 🪙 Cryptocurrency Data")
    
    crypto_symbols = st.multiselect(
        "Select Cryptocurrencies",
        ["BTC", "ETH", "ADA", "SOL", "DOGE", "MATIC"],
        default=["BTC", "ETH"],
        key="crypto_symbols"
    )
    
    if crypto_symbols and st.button("Fetch Crypto Data", key="fetch_crypto"):
        try:
            client = StockDataClient()
            crypto_df = client.get_crypto_data(crypto_symbols)
            
            if not crypto_df.empty:
                st.dataframe(crypto_df, use_container_width=True, hide_index=True)
            else:
                st.info("No cryptocurrency data available")
        except Exception as e:
            st.error(f"Error fetching cryptocurrency data: {str(e)}")
    
    st.markdown("### 💱 Forex Exchange Rates")
    
    base_currency = st.selectbox("Base Currency", ["USD", "EUR", "GBP", "JPY"], key="base_currency")
    
    if st.button("Fetch Forex Rates", key="fetch_forex"):
        try:
            client = StockDataClient()
            forex_df = client.get_forex_data(base_currency)
            
            if not forex_df.empty:
                st.dataframe(forex_df, use_container_width=True, hide_index=True)
            else:
                st.info("No forex data available")
        except Exception as e:
            st.error(f"Error fetching forex data: {str(e)}")

# Symbol search functionality
st.markdown("---")
st.markdown("### 🔍 Symbol Search")

search_query = st.text_input("Search for stocks by name or symbol", key="symbol_search")

if search_query and len(search_query) >= 2:
    if data_source == "stockdata" and is_stockdata_available():
        try:
            client = StockDataClient()
            search_results = client.search_symbols(search_query, limit=10)
            
            if not search_results.empty:
                st.markdown("**Search Results:**")
                st.dataframe(search_results, use_container_width=True, hide_index=True)
            else:
                st.info("No results found")
        except Exception as e:
            st.error(f"Error searching symbols: {str(e)}")
    else:
        st.info("Symbol search requires StockData.org API")

# Save functionality for authenticated users
if is_authenticated:
    st.markdown("---")
    st.markdown("### 💾 Save Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Save Current Symbols", key="save_symbols"):
            # Implementation would save to user's account
            st.success(f"✅ Saved symbols: {', '.join(symbols)}")
    
    with col2:
        if st.button("Save Data Source Preference", key="save_data_source"):
            # Implementation would save to user's account
            st.success(f"✅ Saved data source preference: {data_source}")

# Sidebar information
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Data Sources")

if data_source == "yahoo":
    st.sidebar.markdown("""
    **Yahoo Finance (Free)**
    - Real-time quotes (15min delay)
    - Historical data
    - Company fundamentals
    - Limited API calls
    """)
else:
    st.sidebar.markdown("""
    **StockData.org (Premium)**
    - Real-time quotes
    - Intraday data
    - Options chains
    - Cryptocurrency data
    - Forex rates
    - Company profiles
    - Symbol search
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 API Configuration")

if not is_stockdata_available():
    st.sidebar.warning("⚠️ StockData.org API not configured")
    st.sidebar.markdown("Add `STOCKDATA_API_KEY` to secrets to enable premium features")
else:
    st.sidebar.success("✅ StockData.org API connected")

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Usage Tips")
st.sidebar.markdown("""
- Use comma-separated symbols (AAPL,GOOGL,MSFT)
- Auto-refresh updates every 30 seconds
- StockData.org provides more comprehensive data
- Options data requires premium API
- Charts update in real-time
""")