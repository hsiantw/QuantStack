"""
Enhanced UI Components for Quantitative Finance Platform
Modern, professional styling components for consistent user experience
"""

import streamlit as st

def display_success_message(title, message):
    """Display a styled success message"""
    st.markdown(f"""
    <div class="success-box">
        <strong>✅ {title}</strong><br>
        {message}
    </div>
    """, unsafe_allow_html=True)

def display_warning_message(title, message):
    """Display a styled warning message"""
    st.markdown(f"""
    <div class="warning-box">
        <strong>⚠️ {title}</strong><br>
        {message}
    </div>
    """, unsafe_allow_html=True)

def display_error_message(title, message):
    """Display a styled error message"""
    st.markdown(f"""
    <div class="error-box">
        <strong>❌ {title}</strong><br>
        {message}
    </div>
    """, unsafe_allow_html=True)

def display_info_box(title, message):
    """Display a styled info box"""
    st.markdown(f"""
    <div class="info-box">
        <strong>ℹ️ {title}</strong><br>
        {message}
    </div>
    """, unsafe_allow_html=True)

def create_enhanced_metric_card(title, value, change=None, icon="📊"):
    """Create an enhanced metric card with gradient styling"""
    if change is not None:
        trend_class = "positive-change" if change >= 0 else "negative-change"
        trend_icon = "📈" if change >= 0 else "📉"
        change_text = f"""
        <div class="metric-change {trend_class}">
            {trend_icon} {'+' if change >= 0 else ''}{change:.2f}%
        </div>
        """
    else:
        change_text = ""
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{icon} {title}</div>
        <div class="metric-value">{value}</div>
        {change_text}
    </div>
    """, unsafe_allow_html=True)

def display_loading_indicator(message="Processing..."):
    """Display an animated loading indicator"""
    st.markdown(f"""
    <div class="loading-indicator">
        🔄 {message}
    </div>
    """, unsafe_allow_html=True)

def create_feature_card(icon, title, description, button_text, key):
    """Create a feature navigation card"""
    st.markdown(f"""
    <div class="nav-card">
        <div class="nav-card-icon">{icon}</div>
        <div class="nav-card-title">{title}</div>
        <div class="nav-card-desc">{description}</div>
    </div>
    """, unsafe_allow_html=True)
    
    return st.button(button_text, key=key)

def display_strategy_results(strategy_name, performance_metrics):
    """Display strategy backtesting results in styled format"""
    st.markdown(f"""
    <div class="info-box">
        <h3>📊 {strategy_name} Results</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_enhanced_metric_card(
            "Total Return", 
            f"{performance_metrics.get('total_return', 0):.2f}%",
            icon="💰"
        )
    
    with col2:
        create_enhanced_metric_card(
            "Sharpe Ratio", 
            f"{performance_metrics.get('sharpe_ratio', 0):.3f}",
            icon="📈"
        )
    
    with col3:
        create_enhanced_metric_card(
            "Max Drawdown", 
            f"{performance_metrics.get('max_drawdown', 0):.2f}%",
            icon="📉"
        )
    
    with col4:
        create_enhanced_metric_card(
            "Win Rate", 
            f"{performance_metrics.get('win_rate', 0):.1f}%",
            icon="🎯"
        )

def create_trading_signal_card(signal_type, asset, action, confidence, price, reasoning):
    """Create a trading signal card with enhanced styling"""
    signal_colors = {
        "BUY": "success-box",
        "SELL": "error-box", 
        "HOLD": "warning-box"
    }
    
    signal_icons = {
        "BUY": "🟢",
        "SELL": "🔴",
        "HOLD": "🟡"
    }
    
    box_class = signal_colors.get(action, "info-box")
    signal_icon = signal_icons.get(action, "ℹ️")
    
    st.markdown(f"""
    <div class="{box_class}">
        <h4>{signal_icon} {signal_type} Signal: {action} {asset}</h4>
        <p><strong>Confidence:</strong> {confidence:.1f}%</p>
        <p><strong>Target Price:</strong> ${price:.2f}</p>
        <p><strong>Reasoning:</strong> {reasoning}</p>
    </div>
    """, unsafe_allow_html=True)

def display_pairs_analysis_results(pair_name, correlation, cointegration_p_value, z_score, trading_signal):
    """Display pairs trading analysis results"""
    st.markdown(f"""
    <div class="info-box">
        <h3>🔗 Pairs Analysis: {pair_name}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_enhanced_metric_card(
            "Correlation", 
            f"{correlation:.3f}",
            icon="🔗"
        )
    
    with col2:
        cointegration_status = "✅ Cointegrated" if cointegration_p_value < 0.05 else "❌ Not Cointegrated"
        create_enhanced_metric_card(
            "Cointegration", 
            f"{cointegration_p_value:.4f}",
            icon="📊"
        )
    
    with col3:
        create_enhanced_metric_card(
            "Z-Score", 
            f"{z_score:.2f}",
            icon="📏"
        )
    
    with col4:
        signal_icon = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(trading_signal, "ℹ️")
        create_enhanced_metric_card(
            "Signal", 
            trading_signal,
            icon=signal_icon
        )

def create_analysis_header(title, subtitle):
    """Create a consistent analysis header across all pages"""
    st.markdown(f"""
    <div class="main-header">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

def display_data_status(ticker, data_points, date_range):
    """Display data loading status"""
    if data_points > 0:
        display_success_message(
            "Data Loaded Successfully",
            f"Loaded {data_points} data points for {ticker} ({date_range})"
        )
    else:
        display_error_message(
            "Data Loading Failed",
            f"Unable to load data for {ticker}. Please check the ticker symbol and try again."
        )

def create_performance_dashboard(metrics_dict):
    """Create a comprehensive performance dashboard"""
    st.markdown("### 📊 Performance Dashboard")
    
    # First row - Returns and Risk
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_enhanced_metric_card(
            "Total Return",
            f"{metrics_dict.get('total_return', 0):.2f}%",
            metrics_dict.get('return_change'),
            "💰"
        )
    
    with col2:
        create_enhanced_metric_card(
            "Annual Return",
            f"{metrics_dict.get('annual_return', 0):.2f}%",
            icon="📅"
        )
    
    with col3:
        create_enhanced_metric_card(
            "Volatility",
            f"{metrics_dict.get('volatility', 0):.2f}%",
            icon="📊"
        )
    
    with col4:
        create_enhanced_metric_card(
            "Sharpe Ratio",
            f"{metrics_dict.get('sharpe_ratio', 0):.3f}",
            icon="⭐"
        )
    
    # Second row - Risk Metrics
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        create_enhanced_metric_card(
            "Max Drawdown",
            f"{metrics_dict.get('max_drawdown', 0):.2f}%",
            icon="📉"
        )
    
    with col6:
        create_enhanced_metric_card(
            "VaR (95%)",
            f"{metrics_dict.get('var_95', 0):.2f}%",
            icon="⚠️"
        )
    
    with col7:
        create_enhanced_metric_card(
            "Win Rate",
            f"{metrics_dict.get('win_rate', 0):.1f}%",
            icon="🎯"
        )
    
    with col8:
        create_enhanced_metric_card(
            "Profit Factor",
            f"{metrics_dict.get('profit_factor', 0):.2f}",
            icon="💎"
        )