import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
from typing import Dict, List
import requests

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ui_components import apply_custom_css, create_metric_card, create_info_box

# Page configuration
st.set_page_config(
    page_title="Commodities, Forex & Futures Analysis",
    page_icon="🌾",
    layout="wide"
)

# Apply custom styling
apply_custom_css()

def get_commodity_data():
    """Get real-time commodity prices and data"""
    commodities = {
        "Gold": "GC=F",
        "Silver": "SI=F", 
        "Crude Oil": "CL=F",
        "Natural Gas": "NG=F",
        "Copper": "HG=F",
        "Wheat": "ZW=F",
        "Corn": "ZC=F",
        "Soybeans": "ZS=F"
    }
    
    commodity_data = {}
    
    for name, symbol in commodities.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")
            info = ticker.info
            
            if not hist.empty:
                current = hist['Close'].iloc[-1]
                previous = hist['Close'].iloc[-2] if len(hist) > 1 else current
                change = ((current - previous) / previous) * 100
                volume = hist['Volume'].iloc[-1] if 'Volume' in hist.columns else 0
                
                commodity_data[name] = {
                    "current": current,
                    "change": change,
                    "volume": volume,
                    "symbol": symbol,
                    "52_week_high": info.get("fiftyTwoWeekHigh", "N/A"),
                    "52_week_low": info.get("fiftyTwoWeekLow", "N/A")
                }
        except Exception as e:
            print(f"Error fetching {name}: {e}")
            continue
    
    return commodity_data

def get_forex_data():
    """Get major forex pairs data"""
    forex_pairs = {
        "EUR/USD": "EURUSD=X",
        "GBP/USD": "GBPUSD=X",
        "USD/JPY": "USDJPY=X",
        "USD/CHF": "USDCHF=X",
        "USD/CAD": "USDCAD=X",
        "AUD/USD": "AUDUSD=X",
        "NZD/USD": "NZDUSD=X",
        "USD/CNY": "USDCNY=X"
    }
    
    forex_data = {}
    
    for pair, symbol in forex_pairs.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")
            
            if not hist.empty:
                current = hist['Close'].iloc[-1]
                previous = hist['Close'].iloc[-2] if len(hist) > 1 else current
                change = ((current - previous) / previous) * 100
                
                forex_data[pair] = {
                    "current": current,
                    "change": change,
                    "symbol": symbol
                }
        except Exception as e:
            print(f"Error fetching {pair}: {e}")
            continue
    
    return forex_data

def get_futures_data():
    """Get futures contracts data"""
    futures = {
        "S&P 500 Futures": "ES=F",
        "NASDAQ Futures": "NQ=F",
        "Dow Futures": "YM=F",
        "Russell 2000 Futures": "RTY=F",
        "VIX Futures": "VX=F",
        "10-Year Treasury": "ZN=F",
        "30-Year Treasury": "ZB=F",
        "Bitcoin Futures": "BTC=F"
    }
    
    futures_data = {}
    
    for name, symbol in futures.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")
            
            if not hist.empty:
                current = hist['Close'].iloc[-1]
                previous = hist['Close'].iloc[-2] if len(hist) > 1 else current
                change = ((current - previous) / previous) * 100
                volume = hist['Volume'].iloc[-1] if 'Volume' in hist.columns else 0
                
                futures_data[name] = {
                    "current": current,
                    "change": change,
                    "volume": volume,
                    "symbol": symbol
                }
        except Exception as e:
            print(f"Error fetching {name}: {e}")
            continue
    
    return futures_data

def analyze_commodity_correlations(commodity_data):
    """Analyze correlations between commodities"""
    if len(commodity_data) < 2:
        return None
    
    # Get historical data for correlation analysis
    symbols = [data["symbol"] for data in commodity_data.values()]
    names = list(commodity_data.keys())
    
    try:
        # Download 3 months of data
        data = yf.download(symbols, period="3mo", progress=False)['Close']
        
        if len(symbols) == 1:
            return None
            
        # Calculate correlation matrix
        correlation_matrix = data.corr()
        
        return correlation_matrix, names
    except Exception as e:
        print(f"Error calculating correlations: {e}")
        return None

def get_commodity_market_drivers():
    """Get key market drivers for commodities"""
    return {
        "Gold": [
            "Federal Reserve interest rate policy",
            "US Dollar strength/weakness", 
            "Inflation expectations",
            "Geopolitical tensions",
            "Central bank purchases"
        ],
        "Oil": [
            "OPEC production decisions",
            "US shale production levels",
            "Global economic growth",
            "Geopolitical events",
            "Strategic reserve releases"
        ],
        "Agricultural": [
            "Weather patterns and crop reports",
            "Global supply/demand balance",
            "Currency fluctuations",
            "Biofuel demand",
            "Trade policy changes"
        ],
        "Industrial Metals": [
            "Manufacturing activity (PMI)",
            "Infrastructure spending",
            "China economic growth",
            "Supply chain disruptions",
            "Electric vehicle adoption"
        ]
    }

def get_forex_trading_sessions():
    """Get forex trading session information"""
    return {
        "Sydney": {
            "time": "5:00 PM - 2:00 AM EST",
            "pairs": ["AUD/USD", "NZD/USD", "USD/JPY"],
            "characteristics": "Lower volatility, Asia-Pacific focus"
        },
        "Tokyo": {
            "time": "7:00 PM - 4:00 AM EST", 
            "pairs": ["USD/JPY", "EUR/JPY", "GBP/JPY"],
            "characteristics": "Asian market influence, yen activity"
        },
        "London": {
            "time": "3:00 AM - 12:00 PM EST",
            "pairs": ["EUR/USD", "GBP/USD", "USD/CHF"],
            "characteristics": "Highest volume, European focus"
        },
        "New York": {
            "time": "8:00 AM - 5:00 PM EST",
            "pairs": ["USD/CAD", "USD/MXN", "Major pairs"],
            "characteristics": "North American data, overlap with London"
        }
    }

def main():
    st.title("🌾 Commodities, Forex & Futures Analysis")
    
    st.markdown("""
    **Comprehensive analysis of commodity markets, foreign exchange, and futures contracts 
    with real-time data, correlations, and trading insights.**
    """)
    
    # Create tabs for different asset classes
    tabs = st.tabs([
        "📊 Commodities Dashboard",
        "💱 Forex Markets", 
        "📈 Futures Contracts",
        "🔗 Market Correlations",
        "📰 Commodity News & Drivers",
        "⏰ Trading Sessions"
    ])
    
    # Commodities Dashboard
    with tabs[0]:
        st.header("📊 Commodities Market Dashboard")
        
        if st.button("🔄 Refresh Commodity Data", type="primary"):
            with st.spinner("Loading commodity market data..."):
                commodity_data = get_commodity_data()
                
                if commodity_data:
                    st.success(f"✅ Retrieved data for {len(commodity_data)} commodities")
                    
                    # Precious Metals
                    st.subheader("✨ Precious Metals")
                    metals_cols = st.columns(2)
                    
                    precious_metals = ["Gold", "Silver"]
                    for i, metal in enumerate(precious_metals):
                        if metal in commodity_data:
                            with metals_cols[i]:
                                data = commodity_data[metal]
                                st.metric(
                                    label=f"{metal} ($/oz)",
                                    value=f"${data['current']:.2f}",
                                    delta=f"{data['change']:+.2f}%"
                                )
                    
                    # Energy
                    st.subheader("⚡ Energy Markets")
                    energy_cols = st.columns(2)
                    
                    energy_commodities = ["Crude Oil", "Natural Gas"]
                    for i, energy in enumerate(energy_commodities):
                        if energy in commodity_data:
                            with energy_cols[i]:
                                data = commodity_data[energy]
                                unit = "$/barrel" if "Oil" in energy else "$/MMBtu"
                                st.metric(
                                    label=f"{energy} ({unit})",
                                    value=f"${data['current']:.2f}",
                                    delta=f"{data['change']:+.2f}%"
                                )
                    
                    # Industrial Metals
                    st.subheader("🔧 Industrial Metals")
                    if "Copper" in commodity_data:
                        data = commodity_data["Copper"]
                        st.metric(
                            label="Copper ($/lb)",
                            value=f"${data['current']:.3f}",
                            delta=f"{data['change']:+.2f}%"
                        )
                    
                    # Agricultural
                    st.subheader("🌾 Agricultural Commodities")
                    ag_cols = st.columns(3)
                    
                    agricultural = ["Wheat", "Corn", "Soybeans"]
                    for i, ag in enumerate(agricultural):
                        if ag in commodity_data:
                            with ag_cols[i]:
                                data = commodity_data[ag]
                                st.metric(
                                    label=f"{ag} ($/bushel)",
                                    value=f"${data['current']:.2f}",
                                    delta=f"{data['change']:+.2f}%"
                                )
                    
                    # Market Analysis
                    st.subheader("📈 Market Analysis")
                    
                    # Create price chart for selected commodity
                    selected_commodity = st.selectbox(
                        "Select commodity for detailed analysis:",
                        list(commodity_data.keys())
                    )
                    
                    if selected_commodity:
                        symbol = commodity_data[selected_commodity]["symbol"]
                        
                        # Get historical data
                        ticker = yf.Ticker(symbol)
                        hist = ticker.history(period="1mo")
                        
                        if not hist.empty:
                            fig = go.Figure()
                            fig.add_trace(go.Candlestick(
                                x=hist.index,
                                open=hist['Open'],
                                high=hist['High'],
                                low=hist['Low'],
                                close=hist['Close'],
                                name=selected_commodity
                            ))
                            
                            fig.update_layout(
                                title=f"{selected_commodity} - 1 Month Chart",
                                xaxis_title="Date",
                                yaxis_title="Price",
                                template="plotly_dark",
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.error("Unable to retrieve commodity data")
    
    # Forex Markets
    with tabs[1]:
        st.header("💱 Foreign Exchange Markets")
        
        if st.button("🔄 Refresh Forex Data", type="primary"):
            with st.spinner("Loading forex market data..."):
                forex_data = get_forex_data()
                
                if forex_data:
                    st.success(f"✅ Retrieved data for {len(forex_data)} currency pairs")
                    
                    # Major Pairs
                    st.subheader("💰 Major Currency Pairs")
                    
                    major_pairs = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF"]
                    major_cols = st.columns(2)
                    
                    for i, pair in enumerate(major_pairs):
                        if pair in forex_data:
                            col_idx = i % 2
                            with major_cols[col_idx]:
                                data = forex_data[pair]
                                st.metric(
                                    label=pair,
                                    value=f"{data['current']:.4f}",
                                    delta=f"{data['change']:+.2f}%"
                                )
                    
                    # Commodity Currencies
                    st.subheader("🌾 Commodity Currencies")
                    
                    commodity_pairs = ["USD/CAD", "AUD/USD", "NZD/USD"]
                    commodity_cols = st.columns(3)
                    
                    for i, pair in enumerate(commodity_pairs):
                        if pair in forex_data:
                            with commodity_cols[i]:
                                data = forex_data[pair]
                                st.metric(
                                    label=pair,
                                    value=f"{data['current']:.4f}",
                                    delta=f"{data['change']:+.2f}%"
                                )
                    
                    # Emerging Market Currency
                    st.subheader("🌏 Emerging Market Currency")
                    if "USD/CNY" in forex_data:
                        data = forex_data["USD/CNY"]
                        st.metric(
                            label="USD/CNY",
                            value=f"{data['current']:.4f}",
                            delta=f"{data['change']:+.2f}%"
                        )
                
                else:
                    st.error("Unable to retrieve forex data")
    
    # Futures Contracts
    with tabs[2]:
        st.header("📈 Futures Contracts")
        
        if st.button("🔄 Refresh Futures Data", type="primary"):
            with st.spinner("Loading futures market data..."):
                futures_data = get_futures_data()
                
                if futures_data:
                    st.success(f"✅ Retrieved data for {len(futures_data)} futures contracts")
                    
                    # Equity Index Futures
                    st.subheader("📊 Equity Index Futures")
                    
                    equity_futures = ["S&P 500 Futures", "NASDAQ Futures", "Dow Futures", "Russell 2000 Futures"]
                    equity_cols = st.columns(2)
                    
                    for i, future in enumerate(equity_futures):
                        if future in futures_data:
                            col_idx = i % 2
                            with equity_cols[col_idx]:
                                data = futures_data[future]
                                st.metric(
                                    label=future,
                                    value=f"{data['current']:.2f}",
                                    delta=f"{data['change']:+.2f}%"
                                )
                    
                    # Fixed Income Futures
                    st.subheader("💰 Fixed Income Futures")
                    
                    bond_futures = ["10-Year Treasury", "30-Year Treasury"]
                    bond_cols = st.columns(2)
                    
                    for i, future in enumerate(bond_futures):
                        if future in futures_data:
                            with bond_cols[i]:
                                data = futures_data[future]
                                st.metric(
                                    label=future,
                                    value=f"{data['current']:.3f}",
                                    delta=f"{data['change']:+.2f}%"
                                )
                    
                    # Volatility and Crypto Futures
                    st.subheader("⚡ Volatility & Crypto Futures")
                    
                    other_futures = ["VIX Futures", "Bitcoin Futures"]
                    other_cols = st.columns(2)
                    
                    for i, future in enumerate(other_futures):
                        if future in futures_data:
                            with other_cols[i]:
                                data = futures_data[future]
                                value_prefix = "$" if "Bitcoin" in future else ""
                                st.metric(
                                    label=future,
                                    value=f"{value_prefix}{data['current']:.2f}",
                                    delta=f"{data['change']:+.2f}%"
                                )
                
                else:
                    st.error("Unable to retrieve futures data")
    
    # Market Correlations
    with tabs[3]:
        st.header("🔗 Market Correlations Analysis")
        
        if st.button("📊 Generate Correlation Analysis", type="primary"):
            with st.spinner("Analyzing market correlations..."):
                commodity_data = get_commodity_data()
                
                if commodity_data:
                    correlation_result = analyze_commodity_correlations(commodity_data)
                    
                    if correlation_result:
                        correlation_matrix, names = correlation_result
                        
                        st.success("✅ Correlation analysis completed")
                        
                        # Create correlation heatmap
                        fig = go.Figure(data=go.Heatmap(
                            z=correlation_matrix.values,
                            x=names,
                            y=names,
                            colorscale='RdBu',
                            zmid=0,
                            text=correlation_matrix.values.round(2),
                            texttemplate="%{text}",
                            textfont={"size": 10}
                        ))
                        
                        fig.update_layout(
                            title="Commodity Correlation Matrix (3 Month)",
                            template="plotly_dark",
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Correlation insights
                        st.subheader("📈 Correlation Insights")
                        
                        correlation_insights = {
                            "Strong Positive (>0.7)": "Assets moving in same direction - consider diversification",
                            "Moderate Positive (0.3-0.7)": "Some correlation - moderate diversification benefit",
                            "Weak Correlation (-0.3 to 0.3)": "Good diversification potential",
                            "Negative Correlation (<-0.3)": "Excellent hedge relationships - inverse movements"
                        }
                        
                        for level, description in correlation_insights.items():
                            st.write(f"**{level}:** {description}")
                    
                    else:
                        st.warning("Unable to calculate correlations - insufficient data")
                else:
                    st.error("Unable to retrieve commodity data for correlation analysis")
    
    # Commodity News & Drivers
    with tabs[4]:
        st.header("📰 Commodity Market Drivers & News")
        
        market_drivers = get_commodity_market_drivers()
        
        for category, drivers in market_drivers.items():
            st.subheader(f"🎯 {category} - Key Market Drivers")
            
            for i, driver in enumerate(drivers, 1):
                st.write(f"{i}. {driver}")
            
            st.write("---")
        
        # Key Economic Indicators
        st.subheader("📊 Key Economic Indicators for Commodities")
        
        economic_indicators = {
            "US Dollar Index (DXY)": "Strong dollar = lower commodity prices",
            "Manufacturing PMI": "Industrial demand for metals and energy",
            "Consumer Price Index (CPI)": "Inflation expectations affect gold/commodities",
            "Employment Data": "Economic strength impacts demand",
            "GDP Growth": "Economic activity drives commodity consumption",
            "Interest Rates": "Cost of holding commodities vs interest-bearing assets"
        }
        
        for indicator, impact in economic_indicators.items():
            st.write(f"**{indicator}:** {impact}")
    
    # Trading Sessions
    with tabs[5]:
        st.header("⏰ Global Trading Sessions")
        
        trading_sessions = get_forex_trading_sessions()
        
        st.subheader("🌍 Forex Trading Sessions")
        
        for session, info in trading_sessions.items():
            with st.expander(f"🏙️ {session} Session"):
                st.write(f"**Trading Hours:** {info['time']}")
                st.write(f"**Active Pairs:** {', '.join(info['pairs'])}")
                st.write(f"**Characteristics:** {info['characteristics']}")
        
        st.subheader("📅 Market Hours Calendar")
        
        market_schedule = {
            "Sunday": "Forex markets open (Sydney session starts)",
            "Monday-Thursday": "All major sessions active",
            "Friday": "Markets close (New York session ends)",
            "Weekend": "Forex markets closed, crypto continues"
        }
        
        for day, schedule in market_schedule.items():
            st.write(f"**{day}:** {schedule}")
        
        st.info("""
        **💡 Trading Tips:**
        - **London-New York Overlap (8AM-12PM EST):** Highest volatility and volume
        - **Asian Session:** Lower volatility, good for range trading
        - **Friday Afternoons:** Reduced volume and liquidity
        - **Major News Events:** Can cause significant price movements across all sessions
        """)

if __name__ == "__main__":
    main()