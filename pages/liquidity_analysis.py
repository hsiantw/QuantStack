import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_fetcher import DataFetcher
from utils.ui_components import apply_custom_css, create_metric_card, create_info_box

# Page configuration
st.set_page_config(
    page_title="Liquidity Analysis",
    page_icon="💧",
    layout="wide"
)

# Apply custom styling
apply_custom_css()

# Modern header
st.markdown("""
<div class="main-header">
    <h1>💧 Liquidity Analysis & Heat Maps</h1>
    <p>Advanced liquidity analysis with liquidation heat maps, order book depth, and market microstructure insights</p>
</div>
""", unsafe_allow_html=True)

class LiquidityAnalyzer:
    """Advanced liquidity analysis and visualization tools"""
    
    def __init__(self, ticker, period="1y"):
        self.ticker = ticker
        self.period = period
        self.data = None
        self._fetch_data()
    
    def _fetch_data(self):
        """Fetch comprehensive market data"""
        try:
            # Get OHLCV data
            ticker_obj = yf.Ticker(self.ticker)
            self.data = ticker_obj.history(period=self.period, interval="1d")
            
            # Get additional market data
            self.info = ticker_obj.info
            
            if self.data.empty:
                st.error(f"No data available for {self.ticker}")
                return
            
            # Calculate additional metrics
            self._calculate_liquidity_metrics()
            
        except Exception as e:
            st.error(f"Error fetching data for {self.ticker}: {str(e)}")
    
    def _calculate_liquidity_metrics(self):
        """Calculate comprehensive liquidity metrics"""
        if self.data is None or self.data.empty:
            return
        
        # Price-based liquidity metrics
        self.data['Returns'] = self.data['Close'].pct_change()
        self.data['Price_Range'] = (self.data['High'] - self.data['Low']) / self.data['Close']
        self.data['Volume_Price_Trend'] = (self.data['Volume'] * self.data['Returns']).rolling(window=20).mean()
        
        # Volume analysis
        self.data['Volume_MA_20'] = self.data['Volume'].rolling(window=20).mean()
        self.data['Volume_Ratio'] = self.data['Volume'] / self.data['Volume_MA_20']
        self.data['Dollar_Volume'] = self.data['Volume'] * self.data['Close']
        
        # Volatility-based liquidity
        self.data['Volatility_20'] = self.data['Returns'].rolling(window=20).std()
        self.data['High_Low_Volatility'] = (self.data['High'] / self.data['Low'] - 1).rolling(window=20).mean()
        
        # Market Impact Estimation (Amihud Illiquidity)
        self.data['Amihud_Illiquidity'] = abs(self.data['Returns']) / (self.data['Dollar_Volume'] / 1e6)
        self.data['Amihud_Illiquidity'] = self.data['Amihud_Illiquidity'].replace([np.inf, -np.inf], np.nan)
        
        # Liquidity Score (lower is more liquid)
        volatility_component = self.data['Volatility_20'] / self.data['Volatility_20'].mean()
        volume_component = self.data['Volume_MA_20'].mean() / self.data['Volume_MA_20']
        spread_proxy = self.data['Price_Range'] / self.data['Price_Range'].mean()
        
        self.data['Liquidity_Score'] = (volatility_component + volume_component + spread_proxy) / 3
    
    def create_liquidation_heatmap(self):
        """Create liquidation heat map similar to CoinGlass"""
        if self.data is None or self.data.empty:
            return None
        
        # Simulate liquidation levels based on price action and volume
        current_price = self.data['Close'].iloc[-1]
        price_range = np.arange(current_price * 0.8, current_price * 1.2, current_price * 0.005)
        
        # Estimate liquidation density based on volume patterns
        liquidation_data = []
        
        for price_level in price_range:
            # Distance from current price
            distance_pct = abs(price_level - current_price) / current_price
            
            # Estimate liquidation density (higher at support/resistance levels)
            base_density = np.random.exponential(scale=1000)
            
            # Increase density at round numbers
            if price_level % (current_price * 0.05) < (current_price * 0.01):
                base_density *= 2
            
            # Increase density based on historical volume at similar price levels
            historical_volumes = self.data[
                (self.data['Close'] >= price_level * 0.99) & 
                (self.data['Close'] <= price_level * 1.01)
            ]['Volume'].mean()
            
            if not pd.isna(historical_volumes):
                volume_factor = historical_volumes / self.data['Volume'].mean()
                base_density *= (1 + volume_factor)
            
            liquidation_data.append({
                'Price': price_level,
                'Liquidation_Density': base_density,
                'Distance_Pct': distance_pct * 100,
                'Type': 'Long' if price_level < current_price else 'Short'
            })
        
        liquidation_df = pd.DataFrame(liquidation_data)
        
        # Create heatmap
        fig = go.Figure()
        
        # Long liquidations (below current price)
        long_liq = liquidation_df[liquidation_df['Type'] == 'Long']
        fig.add_trace(go.Scatter(
            x=long_liq['Distance_Pct'],
            y=long_liq['Price'],
            mode='markers',
            marker=dict(
                size=long_liq['Liquidation_Density'] / 100,
                color='red',
                opacity=0.6,
                sizemode='area',
                sizeref=2.*max(long_liq['Liquidation_Density'])/100**2,
                sizemin=4
            ),
            name='Long Liquidations',
            hovertemplate='Price: $%{y:.2f}<br>Distance: %{x:.1f}%<br>Density: %{marker.size}<extra></extra>'
        ))
        
        # Short liquidations (above current price)
        short_liq = liquidation_df[liquidation_df['Type'] == 'Short']
        fig.add_trace(go.Scatter(
            x=short_liq['Distance_Pct'],
            y=short_liq['Price'],
            mode='markers',
            marker=dict(
                size=short_liq['Liquidation_Density'] / 100,
                color='green',
                opacity=0.6,
                sizemode='area',
                sizeref=2.*max(short_liq['Liquidation_Density'])/100**2,
                sizemin=4
            ),
            name='Short Liquidations',
            hovertemplate='Price: $%{y:.2f}<br>Distance: %{x:.1f}%<br>Density: %{marker.size}<extra></extra>'
        ))
        
        # Current price line
        fig.add_hline(
            y=current_price,
            line_dash="dash",
            line_color="white",
            annotation_text=f"Current: ${current_price:.2f}"
        )
        
        fig.update_layout(
            title=f'{self.ticker} Liquidation Heat Map',
            xaxis_title='Distance from Current Price (%)',
            yaxis_title='Price ($)',
            height=600,
            plot_bgcolor='rgba(0,0,0,0.1)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
            yaxis=dict(gridcolor='rgba(128,128,128,0.2)')
        )
        
        return fig
    
    def create_order_book_simulation(self):
        """Simulate order book depth chart"""
        if self.data is None or self.data.empty:
            return None
        
        current_price = self.data['Close'].iloc[-1]
        current_volume = self.data['Volume'].iloc[-1]
        
        # Simulate order book levels
        price_levels = np.arange(current_price * 0.95, current_price * 1.05, current_price * 0.001)
        
        bids = []
        asks = []
        
        for price in price_levels:
            distance = abs(price - current_price) / current_price
            
            # Volume decreases with distance from current price
            base_volume = current_volume * np.exp(-distance * 50)
            
            # Add some randomness
            volume = base_volume * (0.5 + np.random.random())
            
            if price < current_price:
                bids.append({'Price': price, 'Volume': volume, 'Cumulative': 0})
            else:
                asks.append({'Price': price, 'Volume': volume, 'Cumulative': 0})
        
        # Calculate cumulative volumes
        bids = sorted(bids, key=lambda x: x['Price'], reverse=True)
        asks = sorted(asks, key=lambda x: x['Price'])
        
        cumulative_bid = 0
        for bid in bids:
            cumulative_bid += bid['Volume']
            bid['Cumulative'] = cumulative_bid
        
        cumulative_ask = 0
        for ask in asks:
            cumulative_ask += ask['Volume']
            ask['Cumulative'] = cumulative_ask
        
        # Create order book chart
        fig = go.Figure()
        
        # Bids
        bid_prices = [b['Price'] for b in bids]
        bid_volumes = [b['Cumulative'] for b in bids]
        
        fig.add_trace(go.Scatter(
            x=bid_volumes,
            y=bid_prices,
            mode='lines',
            fill='tozeroy',
            name='Bids',
            line=dict(color='green'),
            fillcolor='rgba(0,255,0,0.3)'
        ))
        
        # Asks
        ask_prices = [a['Price'] for a in asks]
        ask_volumes = [a['Cumulative'] for a in asks]
        
        fig.add_trace(go.Scatter(
            x=ask_volumes,
            y=ask_prices,
            mode='lines',
            fill='tozeroy',
            name='Asks',
            line=dict(color='red'),
            fillcolor='rgba(255,0,0,0.3)'
        ))
        
        # Current price
        fig.add_hline(
            y=current_price,
            line_dash="dash",
            line_color="white",
            annotation_text=f"Current: ${current_price:.2f}"
        )
        
        fig.update_layout(
            title=f'{self.ticker} Simulated Order Book Depth',
            xaxis_title='Cumulative Volume',
            yaxis_title='Price ($)',
            height=500,
            plot_bgcolor='rgba(0,0,0,0.1)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
            yaxis=dict(gridcolor='rgba(128,128,128,0.2)')
        )
        
        return fig
    
    def create_volume_profile(self):
        """Create volume profile chart"""
        if self.data is None or self.data.empty:
            return None
        
        # Calculate volume profile
        price_min = self.data['Low'].min()
        price_max = self.data['High'].max()
        price_bins = np.linspace(price_min, price_max, 50)
        
        volume_profile = []
        
        for i in range(len(price_bins) - 1):
            bin_low = price_bins[i]
            bin_high = price_bins[i + 1]
            bin_center = (bin_low + bin_high) / 2
            
            # Find volume traded in this price range
            mask = (
                (self.data['Low'] <= bin_high) & 
                (self.data['High'] >= bin_low)
            )
            
            volume_in_bin = self.data[mask]['Volume'].sum()
            
            volume_profile.append({
                'Price': bin_center,
                'Volume': volume_in_bin,
                'Price_Low': bin_low,
                'Price_High': bin_high
            })
        
        volume_df = pd.DataFrame(volume_profile)
        
        # Create volume profile chart
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.7, 0.3],
            subplot_titles=['Price Chart', 'Volume Profile'],
            horizontal_spacing=0.05
        )
        
        # Price chart
        fig.add_trace(
            go.Candlestick(
                x=self.data.index,
                open=self.data['Open'],
                high=self.data['High'],
                low=self.data['Low'],
                close=self.data['Close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Volume profile
        fig.add_trace(
            go.Bar(
                y=volume_df['Price'],
                x=volume_df['Volume'],
                orientation='h',
                name='Volume Profile',
                marker=dict(color='rgba(0,212,255,0.6)')
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title=f'{self.ticker} Volume Profile Analysis',
            height=600,
            plot_bgcolor='rgba(0,0,0,0.1)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
            yaxis=dict(gridcolor='rgba(128,128,128,0.2)')
        )
        
        return fig
    
    def get_liquidity_metrics(self):
        """Calculate and return liquidity metrics"""
        if self.data is None or self.data.empty:
            return {}
        
        recent_data = self.data.tail(30)  # Last 30 days
        
        metrics = {
            'Average Daily Volume': recent_data['Volume'].mean(),
            'Volume Volatility': recent_data['Volume'].std() / recent_data['Volume'].mean(),
            'Average Spread Proxy': recent_data['Price_Range'].mean() * 100,  # As percentage
            'Amihud Illiquidity': recent_data['Amihud_Illiquidity'].mean(),
            'Liquidity Score': recent_data['Liquidity_Score'].mean(),
            'Volume Trend': recent_data['Volume_Price_Trend'].iloc[-1],
            'Current Volume Ratio': recent_data['Volume_Ratio'].iloc[-1]
        }
        
        return metrics

# Sidebar configuration
st.sidebar.header("Liquidity Analysis Configuration")

# Asset selection
ticker_input = st.sidebar.text_input(
    "Enter Stock/Crypto Ticker",
    value="SPY",
    help="Enter ticker symbol (e.g., SPY, AAPL, BTC-USD)"
).upper()

# Time period
period_options = {
    "3 Months": "3mo",
    "6 Months": "6mo", 
    "1 Year": "1y",
    "2 Years": "2y"
}

selected_period = st.sidebar.selectbox(
    "Analysis Period",
    list(period_options.keys()),
    index=2
)

period = period_options[selected_period]

# Analysis type
analysis_type = st.sidebar.radio(
    "Analysis Focus",
    ["Liquidation Heat Map", "Order Book Depth", "Volume Profile", "Liquidity Metrics"],
    help="Choose the type of liquidity analysis to display"
)

if not ticker_input:
    st.warning("Please enter a ticker symbol to begin analysis.")
    st.stop()

# Initialize analyzer
with st.spinner(f"Analyzing liquidity for {ticker_input}..."):
    analyzer = LiquidityAnalyzer(ticker_input, period)

if analyzer.data is None or analyzer.data.empty:
    st.error("Unable to fetch data for analysis.")
    st.stop()

# Display current price info
current_price = analyzer.data['Close'].iloc[-1]
daily_change = (analyzer.data['Close'].iloc[-1] - analyzer.data['Close'].iloc[-2]) / analyzer.data['Close'].iloc[-2]

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Current Price", f"${current_price:.2f}")
with col2:
    st.metric("Daily Change", f"{daily_change:.2%}")
with col3:
    avg_volume = analyzer.data['Volume'].tail(20).mean()
    st.metric("20D Avg Volume", f"{avg_volume:,.0f}")
with col4:
    current_volume = analyzer.data['Volume'].iloc[-1]
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
    st.metric("Volume vs Avg", f"{volume_ratio:.2f}x")

st.markdown("---")

# Main analysis based on selection
if analysis_type == "Liquidation Heat Map":
    st.header("🔥 Liquidation Heat Map")
    st.info("Visualizing potential liquidation levels based on price action and volume patterns")
    
    fig = analyzer.create_liquidation_heatmap()
    if fig:
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("💡 Understanding Liquidation Heat Maps"):
            st.markdown("""
            **Liquidation Heat Maps** show potential areas where forced liquidations might occur:
            
            - **Red bubbles**: Long position liquidations (price drops)
            - **Green bubbles**: Short position liquidations (price rises)
            - **Bubble size**: Relative liquidation density
            - **Distance**: Percentage away from current price
            
            High-density areas often act as **support/resistance** levels due to:
            - Forced buying/selling pressure
            - Round number psychology
            - Historical volume concentration
            """)

elif analysis_type == "Order Book Depth":
    st.header("📊 Order Book Depth Analysis")
    st.info("Simulated order book showing bid/ask depth and market liquidity")
    
    fig = analyzer.create_order_book_simulation()
    if fig:
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("💡 Understanding Order Book Depth"):
            st.markdown("""
            **Order Book Depth** shows the market's liquidity structure:
            
            - **Green area**: Cumulative bid volume (buying pressure)
            - **Red area**: Cumulative ask volume (selling pressure)
            - **Width**: Depth of liquidity at each price level
            - **Shape**: Market microstructure characteristics
            
            **Key insights**:
            - Thicker areas = more liquidity
            - Gaps = potential price acceleration zones
            - Asymmetry = directional bias
            """)

elif analysis_type == "Volume Profile":
    st.header("📈 Volume Profile Analysis")
    st.info("Price levels with highest trading activity and potential support/resistance")
    
    fig = analyzer.create_volume_profile()
    if fig:
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("💡 Understanding Volume Profile"):
            st.markdown("""
            **Volume Profile** reveals where most trading occurred:
            
            - **Horizontal bars**: Volume traded at each price level
            - **Longer bars**: More significant price levels
            - **POC (Point of Control)**: Highest volume price level
            - **Value areas**: Price ranges with highest volume concentration
            
            **Trading implications**:
            - High volume areas = strong support/resistance
            - Low volume areas = potential breakout zones
            - Volume gaps = areas of price acceptance/rejection
            """)

elif analysis_type == "Liquidity Metrics":
    st.header("📊 Comprehensive Liquidity Metrics")
    
    metrics = analyzer.get_liquidity_metrics()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Volume Metrics")
        st.metric("Avg Daily Volume", f"{metrics.get('Average Daily Volume', 0):,.0f}")
        st.metric("Volume Volatility", f"{metrics.get('Volume Volatility', 0):.2f}")
        st.metric("Current Volume Ratio", f"{metrics.get('Current Volume Ratio', 0):.2f}x")
    
    with col2:
        st.subheader("Liquidity Quality")
        st.metric("Spread Proxy", f"{metrics.get('Average Spread Proxy', 0):.3f}%")
        st.metric("Amihud Illiquidity", f"{metrics.get('Amihud Illiquidity', 0):.6f}")
        st.metric("Liquidity Score", f"{metrics.get('Liquidity Score', 0):.3f}")
    
    # Liquidity trend chart
    st.subheader("Liquidity Trends")
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=['Price & Volume', 'Liquidity Score', 'Amihud Illiquidity'],
        vertical_spacing=0.08
    )
    
    # Price and volume
    fig.add_trace(
        go.Scatter(
            x=analyzer.data.index,
            y=analyzer.data['Close'],
            name='Price',
            line=dict(color='#00d4ff')
        ),
        row=1, col=1
    )
    
    # Liquidity score
    fig.add_trace(
        go.Scatter(
            x=analyzer.data.index,
            y=analyzer.data['Liquidity_Score'],
            name='Liquidity Score',
            line=dict(color='orange')
        ),
        row=2, col=1
    )
    
    # Amihud illiquidity
    amihud_clean = analyzer.data['Amihud_Illiquidity'].replace([np.inf, -np.inf], np.nan).dropna()
    fig.add_trace(
        go.Scatter(
            x=amihud_clean.index,
            y=amihud_clean,
            name='Amihud Illiquidity',
            line=dict(color='red')
        ),
        row=3, col=1
    )
    
    fig.update_layout(
        height=800,
        title=f'{ticker_input} Liquidity Analysis Over Time',
        plot_bgcolor='rgba(0,0,0,0.1)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("💡 Understanding Liquidity Metrics"):
        st.markdown("""
        **Key Liquidity Indicators**:
        
        **Volume Metrics**:
        - **Average Daily Volume**: Higher = more liquid
        - **Volume Volatility**: Lower = more consistent liquidity
        - **Volume Ratio**: Current vs average activity
        
        **Liquidity Quality**:
        - **Spread Proxy**: Price range as % of close (lower = tighter spreads)
        - **Amihud Illiquidity**: Price impact per dollar volume (lower = more liquid)
        - **Liquidity Score**: Composite metric (lower = more liquid)
        
        **Interpretation**:
        - **High liquidity**: Easy to trade without price impact
        - **Low liquidity**: Higher costs and price impact
        - **Liquidity trends**: Help identify optimal trading times
        """)

# Additional market insights
st.markdown("---")
st.header("🔍 Market Microstructure Insights")

col1, col2 = st.columns(2)

with col1:
    # Recent volume analysis
    st.subheader("Recent Volume Pattern")
    recent_volume = analyzer.data['Volume'].tail(20)
    volume_change = (recent_volume.iloc[-1] - recent_volume.mean()) / recent_volume.mean()
    
    if volume_change > 0.5:
        st.success(f"🔥 High volume activity: {volume_change:.1%} above average")
    elif volume_change > 0:
        st.info(f"📈 Above average volume: {volume_change:.1%}")
    elif volume_change > -0.3:
        st.warning(f"📊 Below average volume: {volume_change:.1%}")
    else:
        st.error(f"💤 Very low volume: {volume_change:.1%}")

with col2:
    # Price action quality
    st.subheader("Price Action Quality")
    recent_volatility = analyzer.data['Volatility_20'].iloc[-1]
    avg_volatility = analyzer.data['Volatility_20'].mean()
    vol_ratio = recent_volatility / avg_volatility
    
    if vol_ratio > 1.5:
        st.error(f"⚡ High volatility: {vol_ratio:.1f}x average")
    elif vol_ratio > 1.2:
        st.warning(f"📈 Elevated volatility: {vol_ratio:.1f}x average")
    elif vol_ratio > 0.8:
        st.info(f"📊 Normal volatility: {vol_ratio:.1f}x average")
    else:
        st.success(f"😴 Low volatility: {vol_ratio:.1f}x average")

# Risk warnings
st.markdown("---")
st.warning("""
**⚠️ Important Disclaimers:**
- Liquidation data is simulated based on historical patterns
- Actual liquidation levels may vary significantly
- This analysis is for educational purposes only
- Not financial advice - conduct your own research
- Market microstructure can change rapidly
""")