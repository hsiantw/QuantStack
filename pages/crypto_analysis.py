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

def calculate_crypto_fear_greed_proxy(data):
    """Calculate a proxy for Fear & Greed index using price and volume data"""
    try:
        # Price momentum (30% weight)
        returns = data['Close'].pct_change(7)  # 7-day returns
        momentum_score = np.where(returns > 0.1, 100, np.where(returns < -0.1, 0, 50 + returns * 500))
        
        # Volatility (25% weight) - lower volatility = greed
        volatility = data['Close'].pct_change().rolling(14).std()
        vol_score = 100 - (volatility / volatility.quantile(0.8) * 100).clip(0, 100)
        
        # Volume (20% weight)
        volume_ma = data['Volume'].rolling(14).mean()
        volume_ratio = data['Volume'] / volume_ma
        volume_score = (volume_ratio / 3 * 100).clip(0, 100)
        
        # Trend strength (25% weight)
        sma_20 = data['Close'].rolling(20).mean()
        sma_50 = data['Close'].rolling(50).mean()
        trend_score = np.where(data['Close'] > sma_20, 
                              np.where(sma_20 > sma_50, 80, 60),
                              np.where(sma_20 < sma_50, 20, 40))
        
        # Combine scores
        fear_greed = (momentum_score * 0.3 + vol_score * 0.25 + volume_score * 0.2 + trend_score * 0.25)
        return pd.Series(fear_greed, index=data.index).fillna(50)
    except:
        return pd.Series([50] * len(data), index=data.index)

def calculate_crypto_dominance_proxy(crypto_prices):
    """Calculate dominance proxy for selected crypto"""
    try:
        total_market_cap = sum(crypto_prices.values())
        dominance = {}
        for crypto, price in crypto_prices.items():
            dominance[crypto] = (price / total_market_cap) * 100 if total_market_cap > 0 else 0
        return dominance
    except:
        return {"BTC": 45, "ETH": 20, "Others": 35}

def analyze_on_chain_metrics_proxy(data):
    """Analyze on-chain metrics using price and volume as proxy"""
    try:
        # Network activity proxy (using volume)
        network_activity = data['Volume'].rolling(7).mean() / data['Volume'].rolling(30).mean()
        
        # HODL behavior proxy (using volatility)
        volatility = data['Close'].pct_change().rolling(14).std()
        hodl_strength = 1 / (1 + volatility * 100)  # Lower volatility = more HODLing
        
        # Accumulation/Distribution proxy
        price_change = data['Close'].pct_change()
        volume_change = data['Volume'].pct_change()
        accumulation = np.where((price_change > 0) & (volume_change > 0), 1,
                               np.where((price_change < 0) & (volume_change > 0), -1, 0))
        
        return {
            'network_activity': network_activity,
            'hodl_strength': hodl_strength,
            'accumulation': pd.Series(accumulation, index=data.index)
        }
    except:
        return {
            'network_activity': pd.Series([1] * len(data), index=data.index),
            'hodl_strength': pd.Series([0.5] * len(data), index=data.index),
            'accumulation': pd.Series([0] * len(data), index=data.index)
        }

def calculate_defi_metrics_proxy(data):
    """Calculate DeFi metrics proxy using market data"""
    try:
        # TVL proxy (using market cap approximation)
        market_cap_proxy = data['Close'] * data['Volume']
        tvl_proxy = market_cap_proxy.rolling(30).mean()
        
        # Yield opportunity proxy (using volatility)
        volatility = data['Close'].pct_change().rolling(14).std()
        yield_proxy = volatility * 365 * 100  # Annualized volatility as yield proxy
        
        # Liquidity proxy
        liquidity_proxy = data['Volume'] / ((data['High'] - data['Low']) / data['Close'])
        
        return {
            'tvl_proxy': tvl_proxy,
            'yield_proxy': yield_proxy,
            'liquidity_proxy': liquidity_proxy
        }
    except:
        return {
            'tvl_proxy': pd.Series([1e6] * len(data), index=data.index),
            'yield_proxy': pd.Series([10] * len(data), index=data.index),
            'liquidity_proxy': pd.Series([1000] * len(data), index=data.index)
        }

def main():
    apply_custom_css()
    
    st.markdown("""
    <div class="main-header">
        <h1>₿ Comprehensive Crypto Analysis</h1>
        <p>Advanced cryptocurrency analysis with DeFi metrics, on-chain analysis, and market sentiment</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.header("Crypto Analysis Settings")
    
    # Major cryptocurrencies
    major_cryptos = {
        "Bitcoin": "BTC-USD",
        "Ethereum": "ETH-USD", 
        "Cardano": "ADA-USD",
        "Polkadot": "DOT-USD",
        "Solana": "SOL-USD",
        "Polygon": "MATIC-USD",
        "Chainlink": "LINK-USD",
        "Avalanche": "AVAX-USD",
        "Cosmos": "ATOM-USD",
        "Algorand": "ALGO-USD"
    }
    
    selected_crypto = st.sidebar.selectbox("Select Cryptocurrency", list(major_cryptos.keys()))
    symbol = major_cryptos[selected_crypto]
    
    # Analysis period
    period = st.sidebar.selectbox(
        "Analysis Period",
        ["1mo", "3mo", "6mo", "1y", "2y"],
        index=2
    )
    
    # Analysis types
    analysis_types = st.sidebar.multiselect(
        "Analysis Types",
        ["Market Overview", "Fear & Greed", "On-Chain Metrics", "DeFi Analysis", "Technical Analysis", "Correlation Analysis"],
        default=["Market Overview", "Fear & Greed", "Technical Analysis"]
    )
    
    if st.sidebar.button("Run Crypto Analysis", type="primary"):
        try:
            # Fetch crypto data
            with st.spinner(f"Fetching {selected_crypto} data..."):
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period)
                
                if data.empty:
                    st.error(f"No data found for {symbol}")
                    return
                
                info = ticker.info
            
            # Market overview
            if "Market Overview" in analysis_types:
                st.markdown("### 📊 Market Overview")
                
                current_price = data['Close'].iloc[-1]
                prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
                change = current_price - prev_price
                change_pct = (change / prev_price) * 100 if prev_price != 0 else 0
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    create_metric_card("Current Price", f"${current_price:,.2f}", f"{change_pct:+.2f}%")
                
                with col2:
                    volume_24h = data['Volume'].iloc[-1]
                    create_metric_card("24h Volume", f"${volume_24h:,.0f}", "")
                
                with col3:
                    volatility = data['Close'].pct_change().std() * np.sqrt(365) * 100
                    create_metric_card("Volatility (Annual)", f"{volatility:.1f}%", "")
                
                with col4:
                    market_cap = info.get('marketCap', 'N/A')
                    if market_cap != 'N/A':
                        market_cap_str = f"${market_cap/1e9:.1f}B" if market_cap > 1e9 else f"${market_cap/1e6:.1f}M"
                    else:
                        market_cap_str = "N/A"
                    create_metric_card("Market Cap", market_cap_str, "")
                
                # Price chart with volume
                fig_overview = make_subplots(
                    rows=2, cols=1,
                    shared_xaxis=True,
                    subplot_titles=(f'{selected_crypto} Price', 'Volume'),
                    vertical_spacing=0.1,
                    row_heights=[0.7, 0.3]
                )
                
                fig_overview.add_trace(go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Price'
                ), row=1, col=1)
                
                fig_overview.add_trace(go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name='Volume',
                    marker_color='lightblue'
                ), row=2, col=1)
                
                fig_overview.update_layout(height=600, title_text=f"{selected_crypto} Market Overview", template="plotly_dark")
                st.plotly_chart(fig_overview, use_container_width=True)
            
            # Fear & Greed Analysis
            if "Fear & Greed" in analysis_types:
                st.markdown("### 😨😍 Fear & Greed Analysis")
                
                fear_greed_index = calculate_crypto_fear_greed_proxy(data)
                current_fg = fear_greed_index.iloc[-1]
                
                # Fear & Greed gauge
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    # Sentiment interpretation
                    if current_fg >= 75:
                        sentiment = "Extreme Greed"
                        color = "#d32f2f"
                        advice = "Consider taking profits"
                    elif current_fg >= 55:
                        sentiment = "Greed"
                        color = "#ff9800"
                        advice = "Caution advised"
                    elif current_fg >= 45:
                        sentiment = "Neutral"
                        color = "#ffc107"
                        advice = "Wait for clearer signals"
                    elif current_fg >= 25:
                        sentiment = "Fear"
                        color = "#4caf50"
                        advice = "Consider buying opportunity"
                    else:
                        sentiment = "Extreme Fear"
                        color = "#2e7d32"
                        advice = "Strong buying opportunity"
                    
                    create_metric_card("Fear & Greed Index", f"{current_fg:.0f}", sentiment)
                    st.markdown(f"**Trading Advice:** {advice}")
                
                with col2:
                    # Fear & Greed gauge chart
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = current_fg,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Fear & Greed Index"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': color},
                            'steps': [
                                {'range': [0, 25], 'color': "#2e7d32"},
                                {'range': [25, 45], 'color': "#4caf50"},
                                {'range': [45, 55], 'color': "#ffc107"},
                                {'range': [55, 75], 'color': "#ff9800"},
                                {'range': [75, 100], 'color': "#d32f2f"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': current_fg
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=300, template="plotly_dark")
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Fear & Greed historical chart
                fig_fg = go.Figure()
                fig_fg.add_trace(go.Scatter(
                    x=data.index,
                    y=fear_greed_index,
                    name='Fear & Greed Index',
                    line=dict(color='yellow'),
                    fill='tonexty'
                ))
                
                fig_fg.add_hline(y=75, line_dash="dash", line_color="red", annotation_text="Extreme Greed")
                fig_fg.add_hline(y=25, line_dash="dash", line_color="green", annotation_text="Extreme Fear")
                
                fig_fg.update_layout(
                    height=400,
                    title="Fear & Greed Index Over Time",
                    template="plotly_dark",
                    yaxis_title="Index Value"
                )
                st.plotly_chart(fig_fg, use_container_width=True)
            
            # On-Chain Metrics
            if "On-Chain Metrics" in analysis_types:
                st.markdown("### ⛓️ On-Chain Metrics Analysis")
                
                on_chain = analyze_on_chain_metrics_proxy(data)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    network_activity = on_chain['network_activity'].iloc[-1]
                    activity_status = "High" if network_activity > 1.2 else "Low" if network_activity < 0.8 else "Normal"
                    create_metric_card("Network Activity", f"{network_activity:.2f}x", activity_status)
                
                with col2:
                    hodl_strength = on_chain['hodl_strength'].iloc[-1]
                    hodl_status = "Strong" if hodl_strength > 0.7 else "Weak" if hodl_strength < 0.3 else "Moderate"
                    create_metric_card("HODL Strength", f"{hodl_strength:.2f}", hodl_status)
                
                with col3:
                    accumulation = on_chain['accumulation'].tail(10).mean()
                    acc_status = "Accumulation" if accumulation > 0.1 else "Distribution" if accumulation < -0.1 else "Neutral"
                    create_metric_card("Market Behavior", acc_status, f"Score: {accumulation:.2f}")
                
                # On-chain metrics chart
                fig_onchain = make_subplots(
                    rows=3, cols=1,
                    shared_xaxis=True,
                    subplot_titles=('Network Activity', 'HODL Strength', 'Accumulation/Distribution'),
                    vertical_spacing=0.05
                )
                
                fig_onchain.add_trace(go.Scatter(
                    x=data.index,
                    y=on_chain['network_activity'],
                    name='Network Activity',
                    line=dict(color='cyan')
                ), row=1, col=1)
                
                fig_onchain.add_trace(go.Scatter(
                    x=data.index,
                    y=on_chain['hodl_strength'],
                    name='HODL Strength',
                    line=dict(color='orange')
                ), row=2, col=1)
                
                fig_onchain.add_trace(go.Scatter(
                    x=data.index,
                    y=on_chain['accumulation'],
                    name='Accumulation/Distribution',
                    line=dict(color='green'),
                    fill='tonexty'
                ), row=3, col=1)
                
                fig_onchain.update_layout(height=600, title_text="On-Chain Metrics", template="plotly_dark")
                st.plotly_chart(fig_onchain, use_container_width=True)
            
            # DeFi Analysis
            if "DeFi Analysis" in analysis_types:
                st.markdown("### 🏦 DeFi Ecosystem Analysis")
                
                defi_metrics = calculate_defi_metrics_proxy(data)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    tvl_current = defi_metrics['tvl_proxy'].iloc[-1]
                    create_metric_card("TVL Proxy", f"${tvl_current:,.0f}", "")
                
                with col2:
                    yield_current = defi_metrics['yield_proxy'].iloc[-1]
                    create_metric_card("Yield Opportunity", f"{yield_current:.1f}%", "")
                
                with col3:
                    liquidity_current = defi_metrics['liquidity_proxy'].iloc[-1]
                    create_metric_card("Liquidity Score", f"{liquidity_current:.0f}", "")
                
                # DeFi metrics chart
                fig_defi = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('TVL Proxy', 'Yield Opportunities', 'Liquidity Score', 'Price vs TVL'),
                    vertical_spacing=0.1
                )
                
                fig_defi.add_trace(go.Scatter(
                    x=data.index,
                    y=defi_metrics['tvl_proxy'],
                    name='TVL Proxy',
                    line=dict(color='purple')
                ), row=1, col=1)
                
                fig_defi.add_trace(go.Scatter(
                    x=data.index,
                    y=defi_metrics['yield_proxy'],
                    name='Yield %',
                    line=dict(color='gold')
                ), row=1, col=2)
                
                fig_defi.add_trace(go.Scatter(
                    x=data.index,
                    y=defi_metrics['liquidity_proxy'],
                    name='Liquidity',
                    line=dict(color='lightgreen')
                ), row=2, col=1)
                
                # Price vs TVL correlation
                fig_defi.add_trace(go.Scatter(
                    x=defi_metrics['tvl_proxy'],
                    y=data['Close'],
                    mode='markers',
                    name='Price vs TVL',
                    marker=dict(color='red', size=4)
                ), row=2, col=2)
                
                fig_defi.update_layout(height=500, title_text="DeFi Ecosystem Metrics", template="plotly_dark")
                st.plotly_chart(fig_defi, use_container_width=True)
            
            # Technical Analysis
            if "Technical Analysis" in analysis_types:
                st.markdown("### 📈 Technical Analysis")
                
                # Calculate technical indicators
                sma_20 = data['Close'].rolling(20).mean()
                sma_50 = data['Close'].rolling(50).mean()
                rsi = calculate_rsi(data['Close'])
                
                # Support and resistance levels
                recent_high = data['High'].tail(50).max()
                recent_low = data['Low'].tail(50).min()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    trend = "Bullish" if current_price > sma_20.iloc[-1] > sma_50.iloc[-1] else "Bearish"
                    create_metric_card("Trend", trend, "")
                
                with col2:
                    rsi_current = rsi.iloc[-1]
                    rsi_status = "Overbought" if rsi_current > 70 else "Oversold" if rsi_current < 30 else "Neutral"
                    create_metric_card("RSI", f"{rsi_current:.1f}", rsi_status)
                
                with col3:
                    create_metric_card("Resistance", f"${recent_high:.2f}", "")
                
                with col4:
                    create_metric_card("Support", f"${recent_low:.2f}", "")
                
                # Technical analysis chart
                fig_tech = make_subplots(
                    rows=2, cols=1,
                    shared_xaxis=True,
                    subplot_titles=('Price with Moving Averages', 'RSI'),
                    vertical_spacing=0.1,
                    row_heights=[0.7, 0.3]
                )
                
                fig_tech.add_trace(go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Price'
                ), row=1, col=1)
                
                fig_tech.add_trace(go.Scatter(x=data.index, y=sma_20, name='SMA 20', line=dict(color='orange')), row=1, col=1)
                fig_tech.add_trace(go.Scatter(x=data.index, y=sma_50, name='SMA 50', line=dict(color='blue')), row=1, col=1)
                
                fig_tech.add_hline(y=recent_high, line_dash="dash", line_color="red", annotation_text="Resistance", row=1, col=1)
                fig_tech.add_hline(y=recent_low, line_dash="dash", line_color="green", annotation_text="Support", row=1, col=1)
                
                fig_tech.add_trace(go.Scatter(x=data.index, y=rsi, name='RSI', line=dict(color='purple')), row=2, col=1)
                fig_tech.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig_tech.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                
                fig_tech.update_layout(height=600, title_text="Technical Analysis", template="plotly_dark")
                st.plotly_chart(fig_tech, use_container_width=True)
            
            # Correlation Analysis
            if "Correlation Analysis" in analysis_types:
                st.markdown("### 🔗 Correlation Analysis")
                
                # Fetch major crypto correlations
                with st.spinner("Fetching correlation data..."):
                    correlation_symbols = ["BTC-USD", "ETH-USD", "ADA-USD", "DOT-USD", "SOL-USD"]
                    correlation_data = {}
                    
                    for sym in correlation_symbols:
                        try:
                            temp_data = yf.download(sym, period="6mo", progress=False)
                            if not temp_data.empty:
                                correlation_data[sym.replace("-USD", "")] = temp_data['Close']
                        except:
                            continue
                
                if correlation_data:
                    corr_df = pd.DataFrame(correlation_data).corr()
                    
                    # Correlation heatmap
                    fig_corr = px.imshow(
                        corr_df,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale="RdBu_r",
                        title="Cryptocurrency Correlation Matrix"
                    )
                    fig_corr.update_layout(template="plotly_dark", height=400)
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
                    # Correlation insights
                    if selected_crypto.upper().replace(" ", "")[:3] in corr_df.columns:
                        crypto_code = selected_crypto.upper().replace(" ", "")[:3]
                        correlations = corr_df[crypto_code].drop(crypto_code).sort_values(ascending=False)
                        
                        st.markdown(f"**{selected_crypto} Correlations:**")
                        for crypto, corr_value in correlations.items():
                            correlation_strength = "Strong" if abs(corr_value) > 0.7 else "Moderate" if abs(corr_value) > 0.4 else "Weak"
                            direction = "Positive" if corr_value > 0 else "Negative"
                            st.write(f"- **{crypto}**: {corr_value:.3f} ({correlation_strength} {direction})")
            
            # Trading insights and strategies
            st.markdown("### 🎯 Crypto Trading Insights")
            
            insights = []
            
            # Market sentiment insights
            if "Fear & Greed" in analysis_types:
                if current_fg > 75:
                    insights.append("🔴 **Extreme Greed Alert**: Market sentiment very bullish - consider profit-taking")
                elif current_fg < 25:
                    insights.append("🟢 **Extreme Fear Opportunity**: Market sentiment very bearish - potential buying opportunity")
            
            # Technical insights
            if "Technical Analysis" in analysis_types:
                if current_price > sma_20.iloc[-1] > sma_50.iloc[-1]:
                    insights.append("📈 **Bullish Trend**: Price above both moving averages - trend is strong")
                elif rsi.iloc[-1] > 70:
                    insights.append("⚠️ **Overbought Warning**: RSI indicates overbought conditions")
                elif rsi.iloc[-1] < 30:
                    insights.append("💡 **Oversold Signal**: RSI indicates oversold conditions - potential bounce")
            
            # Volume insights
            recent_volume = data['Volume'].tail(5).mean()
            avg_volume = data['Volume'].mean()
            if recent_volume > avg_volume * 1.5:
                insights.append("🚀 **High Volume Alert**: Recent volume significantly above average")
            
            if insights:
                for insight in insights:
                    st.markdown(f"- {insight}")
            else:
                st.info("Market conditions appear normal with no significant alerts.")
            
            # Crypto trading strategies
            st.markdown("### 🎯 Crypto Trading Strategies")
            
            strategy_tabs = st.tabs(["DCA Strategy", "Swing Trading", "DeFi Farming", "Momentum Strategy"])
            
            with strategy_tabs[0]:
                st.markdown("""
                **Dollar Cost Averaging (DCA) Strategy:**
                1. **Method**: Buy fixed amount regardless of price
                2. **Frequency**: Weekly or bi-weekly purchases
                3. **Best For**: Long-term accumulation during bear markets
                4. **Risk**: Very low, suitable for beginners
                """)
            
            with strategy_tabs[1]:
                st.markdown("""
                **Crypto Swing Trading:**
                1. **Entry**: Buy during fear periods (index <30)
                2. **Exit**: Sell during greed periods (index >70)
                3. **Stop Loss**: 15-20% below entry
                4. **Target**: 30-50% gains
                """)
            
            with strategy_tabs[2]:
                st.markdown("""
                **DeFi Yield Farming:**
                1. **Research**: High TVL protocols for safety
                2. **Diversify**: Multiple pools to reduce risk
                3. **Monitor**: Regular yield rate changes
                4. **Exit**: When yields drop significantly
                """)
            
            with strategy_tabs[3]:
                st.markdown("""
                **Momentum Strategy:**
                1. **Entry**: Price breaks above resistance with high volume
                2. **Confirmation**: RSI >50 and rising
                3. **Stop**: Below recent support level
                4. **Target**: Previous high or key resistance
                """)
                
        except Exception as e:
            st.error(f"Error in crypto analysis: {str(e)}")
            st.info("Please check the cryptocurrency symbol and try again.")

def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    except:
        return pd.Series([50] * len(prices))

if __name__ == "__main__":
    main()