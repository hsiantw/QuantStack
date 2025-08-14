import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import UI components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.ui_components import apply_custom_css, create_metric_card, create_info_box

def calculate_money_flow_index(data, period=14):
    """Calculate Money Flow Index (MFI)"""
    try:
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        raw_money_flow = typical_price * data['Volume']
        
        positive_flow = []
        negative_flow = []
        
        for i in range(1, len(typical_price)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                positive_flow.append(raw_money_flow.iloc[i])
                negative_flow.append(0)
            elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                positive_flow.append(0)
                negative_flow.append(raw_money_flow.iloc[i])
            else:
                positive_flow.append(0)
                negative_flow.append(0)
        
        positive_flow = [0] + positive_flow
        negative_flow = [0] + negative_flow
        
        positive_mf = pd.Series(positive_flow).rolling(window=period).sum()
        negative_mf = pd.Series(negative_flow).rolling(window=period).sum()
        
        mfi = 100 - (100 / (1 + (positive_mf / negative_mf)))
        return mfi.fillna(50)
    except:
        return pd.Series([50] * len(data))

def calculate_accumulation_distribution(data):
    """Calculate Accumulation/Distribution Line"""
    try:
        clv = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
        clv = clv.fillna(0)
        ad_line = (clv * data['Volume']).cumsum()
        return ad_line
    except:
        return pd.Series([0] * len(data))

def calculate_on_balance_volume(data):
    """Calculate On-Balance Volume"""
    try:
        obv = []
        obv_value = 0
        
        for i in range(len(data)):
            if i == 0:
                obv.append(data['Volume'].iloc[i])
                obv_value = data['Volume'].iloc[i]
            else:
                if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                    obv_value += data['Volume'].iloc[i]
                elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                    obv_value -= data['Volume'].iloc[i]
                obv.append(obv_value)
        
        return pd.Series(obv)
    except:
        return pd.Series([0] * len(data))

def estimate_dark_pool_activity(data):
    """Estimate potential dark pool activity using volume and price patterns"""
    try:
        # Calculate volume-weighted average price
        vwap = (data['Volume'] * (data['High'] + data['Low'] + data['Close']) / 3).cumsum() / data['Volume'].cumsum()
        
        # Look for unusual volume patterns
        volume_sma = data['Volume'].rolling(window=20).mean()
        volume_ratio = data['Volume'] / volume_sma
        
        # Price efficiency indicator
        price_range = data['High'] - data['Low']
        price_range_sma = price_range.rolling(window=20).mean()
        efficiency = price_range / price_range_sma
        
        # Dark pool indicator (high volume, low volatility)
        dark_pool_indicator = np.where(
            (volume_ratio > 1.5) & (efficiency < 0.8), 
            volume_ratio * (2 - efficiency), 
            0
        )
        
        return pd.Series(dark_pool_indicator, index=data.index), vwap
    except:
        return pd.Series([0] * len(data), index=data.index), pd.Series([0] * len(data), index=data.index)

def calculate_liquidity_metrics(data):
    """Calculate various liquidity metrics"""
    try:
        # Bid-Ask Spread Proxy (High-Low range as percentage of close)
        spread_proxy = ((data['High'] - data['Low']) / data['Close']) * 100
        
        # Volume Rate
        volume_rate = data['Volume'] / data['Volume'].rolling(window=20).mean()
        
        # Turnover Ratio Proxy
        turnover_proxy = data['Volume'] / data['Close']  # Simplified version
        
        # Liquidity Score (higher is more liquid)
        liquidity_score = (volume_rate * 0.4) + ((1/spread_proxy) * 0.6)
        liquidity_score = liquidity_score.fillna(1)
        
        return {
            'spread_proxy': spread_proxy,
            'volume_rate': volume_rate,
            'turnover_proxy': turnover_proxy,
            'liquidity_score': liquidity_score
        }
    except:
        return {
            'spread_proxy': pd.Series([1] * len(data)),
            'volume_rate': pd.Series([1] * len(data)),
            'turnover_proxy': pd.Series([1] * len(data)),
            'liquidity_score': pd.Series([1] * len(data))
        }

def main():
    apply_custom_css()
    
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Advanced Market Analysis</h1>
        <p>Comprehensive analysis of money flow, liquidity, dark pool activity, and crypto markets</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.header("Analysis Settings")
    
    # Asset selection
    asset_type = st.sidebar.selectbox(
        "Asset Type",
        ["Stocks", "Cryptocurrency", "ETFs", "Forex"]
    )
    
    if asset_type == "Stocks":
        default_symbols = ["SPY", "AAPL", "TSLA", "NVDA", "MSFT"]
        symbol = st.sidebar.selectbox("Select Stock", default_symbols + ["Custom"])
        if symbol == "Custom":
            symbol = st.sidebar.text_input("Enter Stock Symbol", "SPY")
    elif asset_type == "Cryptocurrency":
        crypto_symbols = ["BTC-USD", "ETH-USD", "ADA-USD", "DOT-USD", "SOL-USD", "MATIC-USD"]
        symbol = st.sidebar.selectbox("Select Cryptocurrency", crypto_symbols + ["Custom"])
        if symbol == "Custom":
            symbol = st.sidebar.text_input("Enter Crypto Symbol (add -USD)", "BTC-USD")
    elif asset_type == "ETFs":
        etf_symbols = ["SPY", "QQQ", "IWM", "VTI", "VOO", "VEA"]
        symbol = st.sidebar.selectbox("Select ETF", etf_symbols + ["Custom"])
        if symbol == "Custom":
            symbol = st.sidebar.text_input("Enter ETF Symbol", "SPY")
    else:  # Forex
        forex_symbols = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCAD=X"]
        symbol = st.sidebar.selectbox("Select Forex Pair", forex_symbols + ["Custom"])
        if symbol == "Custom":
            symbol = st.sidebar.text_input("Enter Forex Symbol", "EURUSD=X")
    
    # Time period
    period = st.sidebar.selectbox(
        "Analysis Period",
        ["1mo", "3mo", "6mo", "1y", "2y"],
        index=2
    )
    
    # Analysis type
    analysis_types = st.sidebar.multiselect(
        "Analysis Types",
        ["Money Flow", "Liquidity Analysis", "Dark Pool Detection", "Volume Profile", "Institutional Activity"],
        default=["Money Flow", "Liquidity Analysis"]
    )
    
    if st.sidebar.button("Run Analysis", type="primary"):
        try:
            # Fetch data
            with st.spinner(f"Fetching data for {symbol}..."):
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period)
                
                if data.empty:
                    st.error(f"No data found for symbol {symbol}")
                    return
                
                info = ticker.info
            
            # Display basic info
            col1, col2, col3, col4 = st.columns(4)
            
            current_price = data['Close'].iloc[-1]
            prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
            change = current_price - prev_price
            change_pct = (change / prev_price) * 100 if prev_price != 0 else 0
            
            with col1:
                create_metric_card("Current Price", f"${current_price:.2f}", f"{change_pct:+.2f}%")
            with col2:
                avg_volume = data['Volume'].tail(20).mean()
                create_metric_card("Avg Volume (20d)", f"{avg_volume:,.0f}", "")
            with col3:
                volatility = data['Close'].pct_change().std() * np.sqrt(252) * 100
                create_metric_card("Volatility (Annual)", f"{volatility:.1f}%", "")
            with col4:
                market_cap = info.get('marketCap', 'N/A')
                if market_cap != 'N/A':
                    market_cap_str = f"${market_cap/1e9:.1f}B" if market_cap > 1e9 else f"${market_cap/1e6:.1f}M"
                else:
                    market_cap_str = "N/A"
                create_metric_card("Market Cap", market_cap_str, "")
            
            st.markdown("---")
            
            # Analysis sections
            if "Money Flow" in analysis_types:
                st.markdown("### 💰 Money Flow Analysis")
                
                # Calculate money flow indicators
                mfi = calculate_money_flow_index(data)
                ad_line = calculate_accumulation_distribution(data)
                obv = calculate_on_balance_volume(data)
                
                # Money flow metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    current_mfi = mfi.iloc[-1]
                    mfi_signal = "Overbought" if current_mfi > 80 else "Oversold" if current_mfi < 20 else "Neutral"
                    create_metric_card("Money Flow Index", f"{current_mfi:.1f}", mfi_signal)
                
                with col2:
                    ad_trend = "Bullish" if ad_line.iloc[-1] > ad_line.iloc[-10] else "Bearish"
                    create_metric_card("A/D Line Trend", ad_trend, "")
                
                with col3:
                    obv_trend = "Bullish" if obv.iloc[-1] > obv.iloc[-10] else "Bearish"
                    create_metric_card("OBV Trend", obv_trend, "")
                
                # Money flow chart
                fig_mf = make_subplots(
                    rows=3, cols=1,
                    shared_xaxis=True,
                    subplot_titles=('Money Flow Index', 'Accumulation/Distribution Line', 'On-Balance Volume'),
                    vertical_spacing=0.05
                )
                
                fig_mf.add_trace(go.Scatter(x=data.index, y=mfi, name='MFI', line=dict(color='#1f77b4')), row=1, col=1)
                fig_mf.add_hline(y=80, line_dash="dash", line_color="red", row=1, col=1)
                fig_mf.add_hline(y=20, line_dash="dash", line_color="green", row=1, col=1)
                
                fig_mf.add_trace(go.Scatter(x=data.index, y=ad_line, name='A/D Line', line=dict(color='#ff7f0e')), row=2, col=1)
                fig_mf.add_trace(go.Scatter(x=data.index, y=obv, name='OBV', line=dict(color='#2ca02c')), row=3, col=1)
                
                fig_mf.update_layout(height=600, title_text="Money Flow Indicators", template="plotly_dark")
                st.plotly_chart(fig_mf, use_container_width=True)
            
            if "Liquidity Analysis" in analysis_types:
                st.markdown("### 🌊 Liquidity Analysis")
                
                liquidity_metrics = calculate_liquidity_metrics(data)
                
                # Liquidity metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    avg_spread = liquidity_metrics['spread_proxy'].tail(20).mean()
                    create_metric_card("Avg Spread Proxy", f"{avg_spread:.2f}%", "")
                
                with col2:
                    avg_volume_rate = liquidity_metrics['volume_rate'].tail(20).mean()
                    volume_status = "High" if avg_volume_rate > 1.2 else "Low" if avg_volume_rate < 0.8 else "Normal"
                    create_metric_card("Volume Rate", f"{avg_volume_rate:.2f}x", volume_status)
                
                with col3:
                    current_liquidity = liquidity_metrics['liquidity_score'].iloc[-1]
                    liquidity_status = "High" if current_liquidity > 2 else "Low" if current_liquidity < 1 else "Medium"
                    create_metric_card("Liquidity Score", f"{current_liquidity:.2f}", liquidity_status)
                
                with col4:
                    avg_turnover = liquidity_metrics['turnover_proxy'].tail(20).mean()
                    create_metric_card("Turnover Proxy", f"{avg_turnover:.0f}", "")
                
                # Liquidity chart
                fig_liq = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Spread Proxy (%)', 'Volume Rate', 'Liquidity Score', 'Turnover Proxy'),
                    vertical_spacing=0.1
                )
                
                fig_liq.add_trace(go.Scatter(x=data.index, y=liquidity_metrics['spread_proxy'], 
                                           name='Spread', line=dict(color='#d62728')), row=1, col=1)
                fig_liq.add_trace(go.Scatter(x=data.index, y=liquidity_metrics['volume_rate'], 
                                           name='Volume Rate', line=dict(color='#9467bd')), row=1, col=2)
                fig_liq.add_trace(go.Scatter(x=data.index, y=liquidity_metrics['liquidity_score'], 
                                           name='Liquidity', line=dict(color='#8c564b')), row=2, col=1)
                fig_liq.add_trace(go.Scatter(x=data.index, y=liquidity_metrics['turnover_proxy'], 
                                           name='Turnover', line=dict(color='#e377c2')), row=2, col=2)
                
                fig_liq.update_layout(height=500, title_text="Liquidity Metrics", template="plotly_dark")
                st.plotly_chart(fig_liq, use_container_width=True)
            
            if "Dark Pool Detection" in analysis_types:
                st.markdown("### 🕳️ Dark Pool Activity Detection")
                
                dark_pool_indicator, vwap = estimate_dark_pool_activity(data)
                
                # Dark pool metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_dark_activity = dark_pool_indicator.tail(20).mean()
                    activity_level = "High" if avg_dark_activity > 1.5 else "Medium" if avg_dark_activity > 0.5 else "Low"
                    create_metric_card("Dark Pool Activity", activity_level, f"Score: {avg_dark_activity:.2f}")
                
                with col2:
                    price_vs_vwap = ((current_price - vwap.iloc[-1]) / vwap.iloc[-1]) * 100
                    vwap_status = "Above VWAP" if price_vs_vwap > 0 else "Below VWAP"
                    create_metric_card("Price vs VWAP", f"{price_vs_vwap:+.2f}%", vwap_status)
                
                with col3:
                    dark_signals = len(dark_pool_indicator[dark_pool_indicator > 1.5].tail(20))
                    create_metric_card("Dark Pool Signals (20d)", str(dark_signals), "")
                
                # Dark pool chart
                fig_dp = make_subplots(
                    rows=3, cols=1,
                    shared_xaxis=True,
                    subplot_titles=('Price vs VWAP', 'Volume', 'Dark Pool Activity Indicator'),
                    vertical_spacing=0.05
                )
                
                fig_dp.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Price', line=dict(color='white')), row=1, col=1)
                fig_dp.add_trace(go.Scatter(x=data.index, y=vwap, name='VWAP', line=dict(color='yellow')), row=1, col=1)
                
                fig_dp.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='lightblue'), row=2, col=1)
                
                fig_dp.add_trace(go.Scatter(x=data.index, y=dark_pool_indicator, name='Dark Pool Activity', 
                                          line=dict(color='red'), fill='tonexty'), row=3, col=1)
                fig_dp.add_hline(y=1.5, line_dash="dash", line_color="orange", row=3, col=1)
                
                fig_dp.update_layout(height=700, title_text="Dark Pool Detection Analysis", template="plotly_dark")
                st.plotly_chart(fig_dp, use_container_width=True)
                
                # Dark pool interpretation
                create_info_box(
                    "Dark Pool Activity Interpretation",
                    """
                    **High Activity (>1.5)**: Potential institutional trading or dark pool activity
                    **Medium Activity (0.5-1.5)**: Normal institutional flow
                    **Low Activity (<0.5)**: Primarily retail trading
                    
                    Dark pool indicators combine unusual volume patterns with low price volatility,
                    suggesting large trades executed without significant market impact.
                    """
                )
            
            if "Volume Profile" in analysis_types:
                st.markdown("### 📊 Volume Profile Analysis")
                
                # Calculate volume profile
                price_levels = np.linspace(data['Low'].min(), data['High'].max(), 50)
                volume_profile = []
                
                for i in range(len(price_levels)-1):
                    mask = (data['Low'] <= price_levels[i+1]) & (data['High'] >= price_levels[i])
                    volume_at_level = data.loc[mask, 'Volume'].sum()
                    volume_profile.append(volume_at_level)
                
                # Volume profile chart
                fig_vp = go.Figure()
                
                # Candlestick chart
                fig_vp.add_trace(go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Price'
                ))
                
                fig_vp.update_layout(
                    height=600,
                    title="Volume Profile",
                    template="plotly_dark",
                    xaxis_title="Date",
                    yaxis_title="Price"
                )
                
                st.plotly_chart(fig_vp, use_container_width=True)
                
                # Volume distribution
                fig_vol_dist = go.Figure()
                fig_vol_dist.add_trace(go.Bar(
                    y=price_levels[:-1],
                    x=volume_profile,
                    orientation='h',
                    name='Volume by Price Level'
                ))
                
                fig_vol_dist.update_layout(
                    height=400,
                    title="Volume Distribution by Price Level",
                    template="plotly_dark",
                    xaxis_title="Volume",
                    yaxis_title="Price Level"
                )
                
                st.plotly_chart(fig_vol_dist, use_container_width=True)
            
            if "Institutional Activity" in analysis_types:
                st.markdown("### 🏛️ Institutional Activity Analysis")
                
                # Calculate institutional indicators
                large_volume_threshold = data['Volume'].quantile(0.8)
                large_volume_days = data[data['Volume'] > large_volume_threshold]
                
                # Block trading analysis
                block_trades = data[data['Volume'] > data['Volume'].rolling(20).mean() * 2]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    institutional_days = len(large_volume_days.tail(20))
                    create_metric_card("High Volume Days (20d)", str(institutional_days), "")
                
                with col2:
                    block_trade_count = len(block_trades.tail(20))
                    create_metric_card("Block Trades (20d)", str(block_trade_count), "")
                
                with col3:
                    avg_institutional_volume = large_volume_days['Volume'].tail(10).mean() if len(large_volume_days) > 0 else 0
                    create_metric_card("Avg Institutional Volume", f"{avg_institutional_volume:,.0f}", "")
                
                # Institutional activity chart
                fig_inst = make_subplots(
                    rows=2, cols=1,
                    shared_xaxis=True,
                    subplot_titles=('Price with Institutional Activity', 'Volume Analysis'),
                    vertical_spacing=0.1
                )
                
                fig_inst.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Price', line=dict(color='white')), row=1, col=1)
                
                # Highlight institutional activity days
                if len(large_volume_days) > 0:
                    fig_inst.add_trace(go.Scatter(
                        x=large_volume_days.index, 
                        y=large_volume_days['Close'],
                        mode='markers',
                        name='High Volume Days',
                        marker=dict(color='red', size=8)
                    ), row=1, col=1)
                
                fig_inst.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color='lightblue'), row=2, col=1)
                fig_inst.add_hline(y=large_volume_threshold, line_dash="dash", line_color="red", row=2, col=1)
                
                fig_inst.update_layout(height=600, title_text="Institutional Activity Analysis", template="plotly_dark")
                st.plotly_chart(fig_inst, use_container_width=True)
            
            # Trading secrets and insights
            st.markdown("### 🔐 Trading Secrets & Market Insights")
            
            insights = []
            
            # Money flow insights
            if "Money Flow" in analysis_types:
                if mfi.iloc[-1] > 80:
                    insights.append("🔴 **Overbought Warning**: MFI indicates excessive buying pressure - potential reversal")
                elif mfi.iloc[-1] < 20:
                    insights.append("🟢 **Oversold Opportunity**: MFI suggests selling exhaustion - potential bounce")
                
                if obv.iloc[-1] > obv.iloc[-5]:
                    insights.append("📈 **Volume Confirmation**: Rising OBV confirms price trend strength")
                else:
                    insights.append("⚠️ **Divergence Alert**: OBV not confirming price movement - watch for reversal")
            
            # Liquidity insights
            if "Liquidity Analysis" in analysis_types:
                if liquidity_metrics['liquidity_score'].iloc[-1] < 1:
                    insights.append("🌊 **Low Liquidity Alert**: Reduced liquidity increases slippage risk")
                elif liquidity_metrics['volume_rate'].iloc[-1] > 2:
                    insights.append("🚀 **High Activity**: Unusual volume suggests significant market interest")
            
            # Dark pool insights
            if "Dark Pool Detection" in analysis_types:
                recent_dark_activity = dark_pool_indicator.tail(5).mean()
                if recent_dark_activity > 1.5:
                    insights.append("🕳️ **Institutional Interest**: High dark pool activity suggests large player involvement")
                
                if abs(price_vs_vwap) > 2:
                    insights.append(f"⚖️ **VWAP Deviation**: Price {price_vs_vwap:+.1f}% from VWAP - potential mean reversion")
            
            if insights:
                for insight in insights:
                    st.markdown(f"- {insight}")
            else:
                st.info("Market conditions appear normal with no significant alerts.")
            
            # Advanced trading strategies
            st.markdown("### 🎯 Advanced Trading Strategies")
            
            strategy_tabs = st.tabs(["Dark Pool Strategy", "Liquidity Strategy", "Money Flow Strategy"])
            
            with strategy_tabs[0]:
                st.markdown("""
                **Dark Pool Breakout Strategy:**
                1. **Entry**: When dark pool activity >1.5 AND price breaks above/below VWAP
                2. **Stop Loss**: 2% from entry or previous support/resistance
                3. **Target**: 1.5:1 risk-reward ratio
                4. **Filter**: Only trade during high volume periods
                """)
            
            with strategy_tabs[1]:
                st.markdown("""
                **Liquidity Scalping Strategy:**
                1. **Entry**: During high liquidity periods (score >2)
                2. **Method**: Quick in/out trades with tight spreads
                3. **Risk**: Maximum 0.5% per trade
                4. **Timing**: Avoid low liquidity periods
                """)
            
            with strategy_tabs[2]:
                st.markdown("""
                **Money Flow Reversal Strategy:**
                1. **Entry**: MFI oversold (<20) with volume confirmation
                2. **Confirmation**: OBV turning upward
                3. **Exit**: MFI reaches overbought (>80) levels
                4. **Risk**: 3% stop loss, scale out at targets
                """)
                
        except Exception as e:
            st.error(f"Error in analysis: {str(e)}")
            st.info("Please check the symbol and try again.")

if __name__ == "__main__":
    main()