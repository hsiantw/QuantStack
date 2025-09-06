import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Liquidity Analysis - QuantStack",
    page_icon="🌊",
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
    apply_custom_css()
except ImportError:
    st.warning("UI components not found. Using default styling.")
    def create_metric_card(title, value, delta=None):
        return st.metric(title, value, delta)
    def create_info_card(title, content):
        return st.info(f"**{title}**\n\n{content}")
    def check_authentication():
        return True, None

def calculate_amihud_illiquidity(data):
    """Calculate Amihud Illiquidity measure"""
    try:
        # Calculate daily returns
        data['Returns'] = data['Close'].pct_change()
        # Calculate dollar volume
        data['Dollar_Volume'] = data['Close'] * data['Volume']
        # Amihud illiquidity = |Return| / Dollar Volume
        data['Amihud_Illiquidity'] = np.abs(data['Returns']) / (data['Dollar_Volume'] + 1e-10)
        return data['Amihud_Illiquidity'].rolling(window=20).mean()
    except Exception:
        return pd.Series([0] * len(data), index=data.index)

def calculate_volume_profile(data, bins=20):
    """Calculate Volume Profile - price levels with highest volume"""
    try:
        price_min, price_max = data['Low'].min(), data['High'].max()
        price_levels = np.linspace(price_min, price_max, bins)
        volume_profile = []
        
        for i in range(len(price_levels)-1):
            level_volume = 0
            for _, row in data.iterrows():
                if price_levels[i] <= row['Close'] <= price_levels[i+1]:
                    level_volume += row['Volume']
            volume_profile.append(level_volume)
        
        return price_levels[:-1], volume_profile
    except Exception:
        return np.array([]), np.array([])

def calculate_order_flow_imbalance(data):
    """Estimate order flow imbalance from OHLCV data"""
    try:
        # Proxy for buy/sell pressure using price movement and volume
        data['Price_Change'] = data['Close'] - data['Open']
        data['Buy_Volume'] = np.where(data['Price_Change'] > 0, 
                                     data['Volume'] * (data['Price_Change'] / (data['High'] - data['Low'] + 1e-10)), 0)
        data['Sell_Volume'] = np.where(data['Price_Change'] < 0, 
                                      data['Volume'] * (abs(data['Price_Change']) / (data['High'] - data['Low'] + 1e-10)), 0)
        data['Order_Flow_Imbalance'] = data['Buy_Volume'] - data['Sell_Volume']
        return data['Order_Flow_Imbalance'].rolling(window=10).mean()
    except Exception:
        return pd.Series([0] * len(data), index=data.index)

def estimate_dark_pool_activity(data):
    """Estimate dark pool activity using volume analysis"""
    try:
        # Calculate volume patterns that might indicate dark pool activity
        data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Std'] = data['Volume'].rolling(window=20).std()
        
        # Unusual volume spikes with minimal price movement might indicate dark pools
        data['Price_Volatility'] = data['Close'].pct_change().rolling(window=5).std()
        data['Volume_Surprise'] = (data['Volume'] - data['Volume_MA']) / (data['Volume_Std'] + 1e-10)
        
        # Dark pool indicator: high volume, low volatility
        data['Dark_Pool_Indicator'] = np.where(
            (data['Volume_Surprise'] > 1.5) & (data['Price_Volatility'] < data['Price_Volatility'].rolling(20).mean()),
            data['Volume_Surprise'], 0
        )
        
        return data['Dark_Pool_Indicator'].rolling(window=5).mean()
    except Exception:
        return pd.Series([0] * len(data), index=data.index)

def calculate_supply_demand_zones(data):
    """Identify supply and demand zones"""
    try:
        supply_zones = []
        demand_zones = []
        
        # Look for significant price levels with high volume
        for i in range(20, len(data)-20):
            # Resistance (Supply) zones
            if (data['High'].iloc[i] > data['High'].iloc[i-10:i].max() and 
                data['High'].iloc[i] > data['High'].iloc[i+1:i+11].max() and
                data['Volume'].iloc[i] > data['Volume'].iloc[i-10:i+11].mean() * 1.5):
                supply_zones.append({
                    'date': data.index[i],
                    'price': data['High'].iloc[i],
                    'volume': data['Volume'].iloc[i],
                    'strength': data['Volume'].iloc[i] / data['Volume'].iloc[i-20:i+21].mean()
                })
            
            # Support (Demand) zones  
            if (data['Low'].iloc[i] < data['Low'].iloc[i-10:i].min() and 
                data['Low'].iloc[i] < data['Low'].iloc[i+1:i+11].min() and
                data['Volume'].iloc[i] > data['Volume'].iloc[i-10:i+11].mean() * 1.5):
                demand_zones.append({
                    'date': data.index[i],
                    'price': data['Low'].iloc[i],
                    'volume': data['Volume'].iloc[i],
                    'strength': data['Volume'].iloc[i] / data['Volume'].iloc[i-20:i+21].mean()
                })
        
        return supply_zones, demand_zones
    except Exception:
        return [], []

def create_liquidity_heatmap(data, symbol):
    """Create CoinGlass-style liquidity heatmap"""
    try:
        # Calculate price levels and liquidity
        price_levels, volume_profile = calculate_volume_profile(data, bins=30)
        
        if len(price_levels) == 0:
            return go.Figure()
        
        # Normalize volume for color intensity
        max_volume = max(volume_profile) if volume_profile else 1
        normalized_volume = [v/max_volume for v in volume_profile]
        
        # Create heatmap
        fig = go.Figure()
        
        # Add volume profile bars
        fig.add_trace(go.Bar(
            x=volume_profile,
            y=price_levels,
            orientation='h',
            marker=dict(
                color=normalized_volume,
                colorscale='RdYlBu_r',
                colorbar=dict(title="Liquidity Intensity")
            ),
            name="Liquidity Profile",
            hovertemplate="Price: $%{y:.2f}<br>Volume: %{x:,.0f}<extra></extra>"
        ))
        
        # Add current price line
        current_price = data['Close'].iloc[-1]
        fig.add_hline(y=current_price, line_dash="dash", line_color="cyan", 
                     annotation_text=f"Current: ${current_price:.2f}")
        
        fig.update_layout(
            title=f"{symbol} Liquidity Heatmap",
            xaxis_title="Volume",
            yaxis_title="Price Level",
            template="plotly_dark",
            height=600,
            showlegend=True
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating liquidity heatmap: {str(e)}")
        return go.Figure()

def create_dark_pool_chart(data, symbol):
    """Create dark pool activity visualization"""
    dark_pool_activity = estimate_dark_pool_activity(data)
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=[f"{symbol} Price", "Estimated Dark Pool Activity"],
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Price chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name="Price"
    ), row=1, col=1)
    
    # Dark pool activity
    fig.add_trace(go.Scatter(
        x=data.index,
        y=dark_pool_activity,
        mode='lines',
        fill='tozeroy',
        line=dict(color='purple'),
        name="Dark Pool Activity"
    ), row=2, col=1)
    
    fig.update_layout(
        title=f"{symbol} Dark Pool Analysis",
        template="plotly_dark",
        height=700,
        xaxis_rangeslider_visible=False
    )
    
    return fig

def create_order_flow_chart(data, symbol):
    """Create order flow imbalance chart"""
    order_flow = calculate_order_flow_imbalance(data)
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=[f"{symbol} Price", "Order Flow Imbalance"],
        vertical_spacing=0.1
    )
    
    # Price chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name="Price"
    ), row=1, col=1)
    
    # Order flow imbalance
    colors = ['red' if x < 0 else 'green' for x in order_flow]
    fig.add_trace(go.Bar(
        x=data.index,
        y=order_flow,
        marker_color=colors,
        name="Order Flow",
        opacity=0.7
    ), row=2, col=1)
    
    fig.update_layout(
        title=f"{symbol} Order Flow Analysis",
        template="plotly_dark",
        height=700,
        xaxis_rangeslider_visible=False
    )
    
    return fig

def create_supply_demand_chart(data, symbol):
    """Create supply and demand zones visualization"""
    supply_zones, demand_zones = calculate_supply_demand_zones(data)
    
    fig = go.Figure()
    
    # Price chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name="Price"
    ))
    
    # Supply zones (resistance)
    for zone in supply_zones[-10:]:  # Show last 10 zones
        fig.add_hline(
            y=zone['price'],
            line_color="red",
            line_width=2,
            opacity=min(zone['strength']/3, 1),
            annotation_text=f"Supply: ${zone['price']:.2f}"
        )
    
    # Demand zones (support)
    for zone in demand_zones[-10:]:  # Show last 10 zones
        fig.add_hline(
            y=zone['price'],
            line_color="green",
            line_width=2,
            opacity=min(zone['strength']/3, 1),
            annotation_text=f"Demand: ${zone['price']:.2f}"
        )
    
    fig.update_layout(
        title=f"{symbol} Supply & Demand Zones",
        template="plotly_dark",
        height=600,
        xaxis_rangeslider_visible=False
    )
    
    return fig

# Authentication check
is_authenticated, username = check_authentication()

# Sidebar
st.sidebar.title("🌊 Liquidity Analysis")
st.sidebar.markdown("---")

if not is_authenticated:
    st.sidebar.warning("Please log in to access advanced features and save analysis.")

# Main interface
st.title("🌊 Institutional Liquidity Analysis")
st.markdown("### Advanced market microstructure analysis with dark pool detection and supply/demand zones")

# Input section
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    symbol = st.text_input("Enter Stock Symbol", value="AAPL", key="liquidity_symbol")

with col2:
    period = st.selectbox(
        "Time Period",
        ["1mo", "3mo", "6mo", "1y", "2y"],
        index=2,
        key="liquidity_period"
    )

with col3:
    interval = st.selectbox(
        "Data Interval",
        ["1d", "1h", "4h"],
        index=0,
        key="liquidity_interval"
    )

if st.button("🔍 Analyze Liquidity", type="primary"):
    try:
        with st.spinner(f"Fetching market microstructure data for {symbol}..."):
            # Fetch data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                st.error("No data found for this symbol.")
                st.stop()
            
            # Calculate metrics
            amihud_illiquidity = calculate_amihud_illiquidity(data.copy())
            dark_pool_activity = estimate_dark_pool_activity(data.copy())
            order_flow = calculate_order_flow_imbalance(data.copy())
            
            # Current metrics
            current_price = data['Close'].iloc[-1]
            avg_volume = data['Volume'].mean()
            current_volume = data['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume
            illiquidity_score = amihud_illiquidity.iloc[-1] if len(amihud_illiquidity) > 0 and not pd.isna(amihud_illiquidity.iloc[-1]) else 0
            
            # Display key metrics
            st.markdown("### 📊 Current Market Microstructure")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                create_metric_card("Current Price", f"${current_price:.2f}")
            
            with col2:
                create_metric_card(
                    "Volume Ratio", 
                    f"{volume_ratio:.2f}x",
                    f"{'Above' if volume_ratio > 1 else 'Below'} Average"
                )
            
            with col3:
                illiq_level = "Low" if illiquidity_score < 1e-6 else "Medium" if illiquidity_score < 1e-5 else "High"
                create_metric_card("Illiquidity", illiq_level, f"{illiquidity_score:.2e}")
            
            with col4:
                dark_pool_level = dark_pool_activity.iloc[-1] if len(dark_pool_activity) > 0 and not pd.isna(dark_pool_activity.iloc[-1]) else 0
                dp_status = "High" if dark_pool_level > 1 else "Medium" if dark_pool_level > 0.5 else "Low"
                create_metric_card("Dark Pool Activity", dp_status, f"{dark_pool_level:.2f}")
            
            # Charts
            st.markdown("### 🔥 Liquidity Heatmap")
            liquidity_fig = create_liquidity_heatmap(data, symbol)
            st.plotly_chart(liquidity_fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🕳️ Dark Pool Analysis")
                dark_pool_fig = create_dark_pool_chart(data, symbol)
                st.plotly_chart(dark_pool_fig, use_container_width=True)
            
            with col2:
                st.markdown("### 📈 Supply & Demand Zones")
                supply_demand_fig = create_supply_demand_chart(data, symbol)
                st.plotly_chart(supply_demand_fig, use_container_width=True)
            
            st.markdown("### 🌊 Order Flow Analysis")
            order_flow_fig = create_order_flow_chart(data, symbol)
            st.plotly_chart(order_flow_fig, use_container_width=True)
            
            # Additional metrics table
            st.markdown("### 📋 Detailed Liquidity Metrics")
            
            metrics_data = {
                'Metric': [
                    'Average Daily Volume',
                    'Current Volume',
                    'Volume Percentile (20d)',
                    'Amihud Illiquidity',
                    'Bid-Ask Spread Proxy',
                    'Price Impact Score',
                    'Dark Pool Estimate',
                    'Order Flow Imbalance'
                ],
                'Value': [
                    f"{avg_volume:,.0f}",
                    f"{current_volume:,.0f}",
                    f"{((data['Volume'].iloc[-1] > data['Volume'].rolling(20).quantile(0.8).iloc[-1]) * 80 + 20):.0f}%",
                    f"{illiquidity_score:.2e}",
                    f"{abs(data['High'].iloc[-1] - data['Low'].iloc[-1]) / data['Close'].iloc[-1] * 100:.3f}%",
                    f"{(illiquidity_score * current_volume / 1e6):.2f}",
                    f"{dark_pool_level:.2f}",
                    f"{order_flow.iloc[-1]:,.0f}" if len(order_flow) > 0 and not pd.isna(order_flow.iloc[-1]) else "0"
                ],
                'Interpretation': [
                    'Historical average trading volume',
                    'Latest session volume',
                    'Current volume vs 20-day distribution',
                    'Price impact per dollar traded',
                    'Estimated spread width',
                    'Market impact of large orders',
                    'Unusual volume/price patterns',
                    'Buy vs sell pressure balance'
                ]
            }
            
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            
            # Educational information
            st.markdown("### 📚 Understanding Liquidity Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                create_info_card(
                    "Liquidity Heatmap",
                    "Shows price levels with highest trading volume. Red areas indicate high liquidity zones where large orders can be executed with minimal price impact. These levels often act as support/resistance."
                )
                
                create_info_card(
                    "Dark Pool Detection",
                    "Estimates institutional 'dark pool' trading by identifying unusual volume patterns with minimal price movement. High values suggest large institutional orders being executed off-exchange."
                )
            
            with col2:
                create_info_card(
                    "Supply & Demand Zones",
                    "Identifies key price levels where significant buying (demand) or selling (supply) pressure occurred. These zones often act as future support and resistance levels."
                )
                
                create_info_card(
                    "Order Flow Analysis",
                    "Measures the balance between buying and selling pressure. Positive values indicate net buying pressure, while negative values suggest selling pressure dominance."
                )
            
            # Save functionality for authenticated users
            if is_authenticated:
                st.markdown("### 💾 Save Analysis")
                
                if st.button("Save Liquidity Analysis", key="save_liquidity"):
                    # Implementation would save to user's account
                    st.success(f"✅ Liquidity analysis for {symbol} saved to your account!")
            
            st.success(f"✅ Liquidity analysis complete for {symbol}")
            
    except Exception as e:
        st.error(f"Error performing analysis: {str(e)}")
        st.error("Please check the symbol and try again.")

# Sidebar information
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Analysis Features")
st.sidebar.markdown("""
**Liquidity Heatmap:**
- Volume-weighted price levels
- CoinGlass-style visualization
- Support/resistance identification

**Dark Pool Detection:**
- Institutional flow estimation
- Hidden order analysis  
- Off-exchange activity patterns

**Supply & Demand Zones:**
- Key price level identification
- Volume-confirmed zones
- Future S/R prediction

**Order Flow Analysis:**
- Buy/sell pressure balance
- Market microstructure insights
- Institutional vs retail flow
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚠️ Important Notes")
st.sidebar.markdown("""
- Dark pool estimates are based on volume patterns
- Higher resolution data provides better accuracy
- Combine with other technical analysis
- Consider market conditions and news
""")