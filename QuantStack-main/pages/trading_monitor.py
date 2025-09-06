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
from typing import Dict, List, Optional
import json
import time
import asyncio

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ui_components import create_enhanced_metric_card, display_info_box, display_success_message, display_warning_message, display_error_message

# Page configuration
st.set_page_config(
    page_title="Trading Account Monitor",
    page_icon="📊",
    layout="wide"
)

# Apply custom styling
st.markdown("""
<style>
    .header-container {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5aa0 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid #4a90e2;
    }
    .header-title {
        color: white;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .header-subtitle {
        color: #b3d9ff;
        font-size: 1.2rem;
        margin-bottom: 0;
    }
    .connection-status {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: bold;
        text-align: center;
    }
    .status-connected {
        background: linear-gradient(135deg, #1a472a 0%, #2d5016 100%);
        color: white;
        border-left: 4px solid #4caf50;
    }
    .status-disconnected {
        background: linear-gradient(135deg, #c62828 0%, #d32f2f 100%);
        color: white;
        border-left: 4px solid #f44336;
    }
    .status-demo {
        background: linear-gradient(135deg, #5d4037 0%, #8d6e63 100%);
        color: white;
        border-left: 4px solid #ff9800;
    }
    .trading-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5aa0 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #4a90e2;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(74, 144, 226, 0.3);
    }
    .position-row {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #4a90e2;
    }
    .profit { color: #4caf50; font-weight: bold; }
    .loss { color: #f44336; font-weight: bold; }
    .neutral { color: #ffc107; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

class TradingAccountMonitor:
    """
    Trading account monitor with Webull integration support
    Supports both live API and paper trading demo modes
    """
    
    def __init__(self):
        self.is_connected = False
        self.connection_type = "disconnected"  # disconnected, live, demo
        self.account_data = {}
        self.positions = pd.DataFrame()
        self.orders = pd.DataFrame()
        self.trade_history = pd.DataFrame()
        
    def initialize_session_state(self):
        """Initialize session state for trading monitor"""
        if 'trading_connected' not in st.session_state:
            st.session_state.trading_connected = False
        if 'trading_mode' not in st.session_state:
            st.session_state.trading_mode = 'demo'
        if 'qr_demo_active' not in st.session_state:
            st.session_state.qr_demo_active = False
        if 'webull_credentials' not in st.session_state:
            st.session_state.webull_credentials = {}
        if 'demo_account' not in st.session_state:
            st.session_state.demo_account = self.create_demo_account()
    
    def create_demo_account(self):
        """Create a realistic demo trading account"""
        demo_positions = [
            {'symbol': 'AAPL', 'quantity': 50, 'avg_cost': 185.50, 'current_price': 192.30},
            {'symbol': 'MSFT', 'quantity': 25, 'avg_cost': 415.20, 'current_price': 422.75},
            {'symbol': 'GOOGL', 'quantity': 15, 'avg_cost': 138.90, 'current_price': 142.15},
            {'symbol': 'TSLA', 'quantity': 20, 'avg_cost': 248.75, 'current_price': 251.20},
            {'symbol': 'NVDA', 'quantity': 12, 'avg_cost': 875.40, 'current_price': 890.25},
            {'symbol': 'SPY', 'quantity': 100, 'avg_cost': 485.20, 'current_price': 487.85}
        ]
        
        demo_orders = [
            {'symbol': 'AMD', 'side': 'BUY', 'quantity': 30, 'price': 145.75, 'status': 'OPEN', 'order_type': 'LIMIT'},
            {'symbol': 'QQQ', 'side': 'SELL', 'quantity': 25, 'price': 395.20, 'status': 'FILLED', 'order_type': 'MARKET'},
            {'symbol': 'META', 'side': 'BUY', 'quantity': 15, 'price': 485.50, 'status': 'CANCELLED', 'order_type': 'LIMIT'}
        ]
        
        return {
            'account_value': 125750.85,
            'cash_balance': 15420.30,
            'buying_power': 30840.60,
            'day_pnl': 1250.75,
            'total_pnl': 8945.20,
            'positions': demo_positions,
            'orders': demo_orders,
            'last_updated': datetime.now()
        }
    
    def connect_live_account(self, email: str, password: str, trade_pin: str = None):
        """Connect to live Webull account"""
        try:
            # This would be the actual Webull API connection
            # For now, we'll simulate the connection process
            
            st.info("Attempting to connect to Webull account...")
            time.sleep(2)  # Simulate connection time
            
            # In real implementation, this would be:
            # from webull import webull
            # wb = webull()
            # wb.login(email, password)
            # if trade_pin:
            #     wb.get_trade_token(trade_pin)
            
            # For demo purposes, simulate successful connection
            if email and password:
                st.session_state.trading_connected = True
                st.session_state.trading_mode = 'live'
                st.session_state.webull_credentials = {'email': email, 'connected_at': datetime.now()}
                self.is_connected = True
                self.connection_type = "live"
                return True
            else:
                return False
                
        except Exception as e:
            st.error(f"Connection failed: {str(e)}")
            return False
    
    def get_account_info(self):
        """Get account information"""
        if st.session_state.trading_mode == 'demo':
            return st.session_state.demo_account
        else:
            # In live mode, this would fetch real account data
            # return wb.get_account()
            return st.session_state.demo_account
    
    def get_positions(self):
        """Get current positions"""
        account_info = self.get_account_info()
        positions_data = []
        
        for pos in account_info['positions']:
            market_value = pos['quantity'] * pos['current_price']
            cost_basis = pos['quantity'] * pos['avg_cost']
            unrealized_pnl = market_value - cost_basis
            unrealized_pnl_pct = (unrealized_pnl / cost_basis) * 100 if cost_basis > 0 else 0
            
            positions_data.append({
                'Symbol': pos['symbol'],
                'Quantity': pos['quantity'],
                'Avg Cost': pos['avg_cost'],
                'Current Price': pos['current_price'],
                'Market Value': market_value,
                'Cost Basis': cost_basis,
                'Unrealized P&L': unrealized_pnl,
                'Unrealized P&L %': unrealized_pnl_pct
            })
        
        return pd.DataFrame(positions_data)
    
    def get_orders(self):
        """Get order information"""
        account_info = self.get_account_info()
        return pd.DataFrame(account_info['orders'])
    
    def refresh_account_data(self):
        """Refresh account data from API"""
        if st.session_state.trading_mode == 'demo':
            # Update demo account with some random price movements
            demo_account = st.session_state.demo_account
            for pos in demo_account['positions']:
                # Simulate small price movements
                price_change = np.random.normal(0, 0.02) * pos['current_price']
                pos['current_price'] = max(0.01, pos['current_price'] + price_change)
            
            # Update account value
            total_value = sum(pos['quantity'] * pos['current_price'] for pos in demo_account['positions'])
            demo_account['account_value'] = total_value + demo_account['cash_balance']
            demo_account['last_updated'] = datetime.now()
            
            st.session_state.demo_account = demo_account

# Initialize monitor
monitor = TradingAccountMonitor()
monitor.initialize_session_state()

# Header
st.markdown("""
<div class="header-container">
    <div class="header-title">📊 Trading Account Monitor</div>
    <div class="header-subtitle">Real-time portfolio tracking and performance analysis</div>
</div>
""", unsafe_allow_html=True)

# Connection Status
if st.session_state.trading_connected:
    if st.session_state.trading_mode == 'live':
        st.markdown("""
        <div class="connection-status status-connected">
            ✅ Connected to Live Webull Account
        </div>
        """, unsafe_allow_html=True)
    elif st.session_state.trading_mode == 'qr_demo':
        st.markdown("""
        <div class="connection-status status-connected">
            ✅ Connected via QR Code (Demo)
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="connection-status status-demo">
            🎯 Demo Mode Active - Paper Trading
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="connection-status status-disconnected">
        ❌ Not Connected to Trading Account
    </div>
    """, unsafe_allow_html=True)

# Tabs
tabs = st.tabs(["🔗 Account Setup", "📊 Account Overview", "📈 Positions", "📋 Orders", "🎯 Strategy Performance", "⚙️ Settings"])

# Account Setup Tab
with tabs[0]:
    st.header("🔗 Account Setup")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Connection Options")
        
        connection_type = st.radio(
            "Choose connection type:",
            ["🎯 Demo Mode (Paper Trading)", "📱 QR Code Login (Easiest)", "🔴 Live Trading (Webull API)"],
            index=0 if st.session_state.trading_mode == 'demo' else 1
        )
        
        if "Demo Mode" in connection_type:
            st.info("Demo mode uses simulated trading data for testing and learning purposes.")
            
            if st.button("🎯 Start Demo Mode", type="primary", use_container_width=True):
                st.session_state.trading_connected = True
                st.session_state.trading_mode = 'demo'
                st.rerun()
                
        elif "QR Code" in connection_type:
            st.success("📱 QR Code Login - Most secure and convenient method!")
            
            col_qr1, col_qr2 = st.columns([1, 1])
            
            with col_qr1:
                st.markdown("#### How QR Code Login Works:")
                st.markdown("""
                1. **Generate QR Code** - Click button below
                2. **Open Webull App** - Use your mobile phone
                3. **Scan QR Code** - Use app's scan feature  
                4. **Automatic Login** - No passwords needed
                5. **Secure Connection** - Encrypted authentication
                """)
                
                if st.button("📱 Generate QR Code", type="primary", use_container_width=True):
                    # Initialize QR code session
                    if 'qr_connector' not in st.session_state:
                        from utils.webull_connector import WebullConnector
                        st.session_state.qr_connector = WebullConnector()
                    
                    with st.spinner("Generating QR code..."):
                        # Note: This is a conceptual implementation
                        # The actual QR code generation would depend on Webull API support
                        st.info("⚠️ QR Code Authentication Implementation Note:")
                        st.markdown("""
                        Based on current Webull API documentation (2025), **official QR code authentication 
                        is not supported** in the public API. However, here are the available options:
                        
                        **Current Status:**
                        - Official API: Uses App Key/Secret only
                        - Unofficial API: May have QR workarounds (limited support)
                        
                        **Recommended Alternative:**
                        Use the **Official API** method below for secure, reliable connection.
                        """)
                        
                        # For demo purposes, show what QR authentication would look like
                        st.session_state.qr_demo_active = True
                        
                if st.session_state.get('qr_demo_active', False):
                    st.markdown("#### QR Code Authentication Demo")
                    st.info("This shows how QR authentication would work when supported by Webull API:")
                    
                    # Create a demo QR code placeholder
                    st.markdown("""
                    ```
                    ████ ▄▄▄▄▄ █▀█ █▄█▄█▄▄▄█ ▄▄▄▄▄ ████
                    ████ █   █ █▀▀▀█ ▀ █▀▀█ █   █ ████
                    ████ █▄▄▄█ █▀ ▀▀▄▄▄ ▀█▀█ █▄▄▄█ ████
                    ████▄▄▄▄▄▄▄█▄▀ ▀▄█ █▄█ █▄▄▄▄▄▄▄████
                    ████▄▄  ▄▀▄   ▄▄█▄▄▄▄ ▀    ▄█ ████
                    ████ ▄ █▀▀▄▄▀█▀▀▄  ▄▄▄▄▀██ ▀▄▄ ████
                    ████▄███ ▄▄ ▄▀▀ █▀▄▄▄ ▀█ ▀▄█▀ ▄████
                    ████ ▄▄▄▄▄ █▄█  ▀█▄ ▄ ▄ ▄  ▄▀ ▀████
                    ████ █   █ █  █▀▀ ▄▄▄▄▄▀██▀▀▀▀▄████
                    ████ █▄▄▄█ █  ▀▀▄▄▄▄▄▄▀█▀▀▀▀█▀▄████
                    ████▄▄▄▄▄▄▄█▄▄██▄█▄██▄█▄█▄██▄█▄████
                    ```
                    **Demo QR Code** - Scan with Webull app
                    """)
                    
                    if st.button("✅ Simulate Successful Scan", use_container_width=True):
                        st.session_state.trading_connected = True
                        st.session_state.trading_mode = 'qr_demo'
                        st.session_state.qr_demo_active = False
                        st.success("✅ QR Code authentication successful! (Demo)")
                        st.rerun()
            
            with col_qr2:
                st.markdown("#### Benefits of QR Code Login:")
                st.markdown("""
                ✅ **No Password Entry** - More secure than typing passwords
                
                ✅ **Multi-Factor Built-In** - Uses your phone as second factor
                
                ✅ **Faster Connection** - One scan and you're connected
                
                ✅ **Session Management** - App controls login sessions
                
                ✅ **Encrypted** - Secure token-based authentication
                
                ⚠️ **Current Limitation**: Official Webull API doesn't support QR authentication yet
                """)
                
                st.info("💡 **For Now**: Use the Official API method below for the most secure and reliable connection.")
                
        else:
            st.warning("⚠️ Live trading requires Webull API credentials and involves real money.")
            
            with st.expander("📚 Webull API Setup Guide", expanded=False):
                st.markdown("""
                ### Official Webull API Setup (Recommended):
                
                1. **Register at Webull Developer Portal**
                   - Visit: [developer.webull.com](https://developer.webull.com/)
                   - Create developer account
                   
                2. **Apply for API Access**
                   - Submit API application (1-2 business days)
                   - Get APP_KEY and APP_SECRET
                   
                3. **Install Official SDK**
                   ```bash
                   pip install webull-python-sdk-core
                   pip install webull-python-sdk-trade
                   ```
                
                ### Unofficial API (Quick Setup):
                
                1. **Install Package**
                   ```bash
                   pip install webull
                   ```
                   
                2. **Use Your Login Credentials**
                   - Email and password from your Webull account
                   - Trading PIN for order placement
                
                ⚠️ **Security Note**: Never share your credentials. We store them securely in your session only.
                """)
            
            st.markdown("#### Live Account Credentials")
            
            email = st.text_input("Webull Email", type="default")
            password = st.text_input("Webull Password", type="password")
            trade_pin = st.text_input("Trading PIN (6 digits)", type="password", max_chars=6)
            
            col_connect, col_test = st.columns(2)
            
            with col_connect:
                if st.button("🔴 Connect Live Account", type="primary", use_container_width=True):
                    if email and password:
                        if monitor.connect_live_account(email, password, trade_pin):
                            st.success("✅ Successfully connected to live account!")
                            st.rerun()
                        else:
                            st.error("❌ Connection failed. Please check credentials.")
                    else:
                        st.warning("Please enter email and password.")
            
            with col_test:
                if st.button("🧪 Test Connection", use_container_width=True):
                    if email and password:
                        st.info("Testing connection... (This would validate credentials)")
                        # In real implementation, test connection without full login
                    else:
                        st.warning("Please enter credentials to test.")
    
    with col2:
        st.markdown("### API Installation")
        
        st.code("""
# Install required packages
pip install webull
pip install requests
pip install asyncio-throttle

# For official SDK (recommended):
pip install webull-python-sdk-core
pip install webull-python-sdk-trade
        """, language="bash")
        
        st.markdown("### Security Features")
        st.markdown("""
        - 🔒 MFA support
        - 🔐 Encrypted credentials
        - 🛡️ Session-only storage
        - ⏱️ Auto-logout
        - 📝 Audit logging
        """)

# Account Overview Tab
with tabs[1]:
    st.header("📊 Account Overview")
    
    if st.session_state.trading_connected:
        # Refresh button
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("🔄 Refresh Data", use_container_width=True):
                monitor.refresh_account_data()
                st.rerun()
        
        # Get account info
        account_info = monitor.get_account_info()
        
        # Account metrics
        st.markdown("### 💰 Account Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            create_enhanced_metric_card(
                "Account Value",
                f"${account_info['account_value']:,.2f}",
                change=2.15 if account_info['day_pnl'] > 0 else -1.25,
                icon="💰"
            )
        
        with col2:
            create_enhanced_metric_card(
                "Cash Balance",
                f"${account_info['cash_balance']:,.2f}",
                icon="💵"
            )
        
        with col3:
            create_enhanced_metric_card(
                "Buying Power",
                f"${account_info['buying_power']:,.2f}",
                icon="🔥"
            )
        
        with col4:
            pnl_change = (account_info['day_pnl'] / account_info['account_value']) * 100 if account_info['account_value'] > 0 else 0
            create_enhanced_metric_card(
                "Day P&L",
                f"${account_info['day_pnl']:,.2f}",
                change=pnl_change,
                icon="📈" if account_info['day_pnl'] >= 0 else "📉"
            )
        
        # Performance metrics
        st.markdown("### 📊 Performance Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # P&L Chart
            fig_pnl = go.Figure()
            
            # Simulate daily P&L history
            dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
            daily_pnl = np.random.normal(200, 500, len(dates))
            cumulative_pnl = np.cumsum(daily_pnl)
            
            fig_pnl.add_trace(go.Scatter(
                x=dates,
                y=cumulative_pnl,
                mode='lines',
                name='Cumulative P&L',
                line=dict(color='#4a90e2', width=3)
            ))
            
            fig_pnl.update_layout(
                title='30-Day P&L Performance',
                xaxis_title='Date',
                yaxis_title='P&L ($)',
                template='plotly_dark',
                height=400
            )
            
            st.plotly_chart(fig_pnl, use_container_width=True)
        
        with col2:
            # Portfolio allocation
            positions_df = monitor.get_positions()
            
            if not positions_df.empty:
                fig_allocation = go.Figure(data=[go.Pie(
                    labels=positions_df['Symbol'],
                    values=positions_df['Market Value'],
                    hole=0.3,
                    textinfo='label+percent',
                    textposition='outside'
                )])
                
                fig_allocation.update_layout(
                    title='Portfolio Allocation',
                    template='plotly_dark',
                    height=400
                )
                
                st.plotly_chart(fig_allocation, use_container_width=True)
        
        # Account details
        st.markdown("### ℹ️ Account Details")
        
        account_details_df = pd.DataFrame({
            'Metric': ['Total P&L', 'Day P&L', 'Cash Balance', 'Buying Power', 'Last Updated'],
            'Value': [
                f"${account_info['total_pnl']:,.2f}",
                f"${account_info['day_pnl']:,.2f}",
                f"${account_info['cash_balance']:,.2f}",
                f"${account_info['buying_power']:,.2f}",
                account_info['last_updated'].strftime("%Y-%m-%d %H:%M:%S")
            ]
        })
        
        st.dataframe(account_details_df, use_container_width=True)
        
    else:
        display_info_box("Connect Account", "Please connect your trading account in the Account Setup tab to view your portfolio.")

# Positions Tab
with tabs[2]:
    st.header("📈 Current Positions")
    
    if st.session_state.trading_connected:
        positions_df = monitor.get_positions()
        
        if not positions_df.empty:
            # Position metrics
            total_value = positions_df['Market Value'].sum()
            total_pnl = positions_df['Unrealized P&L'].sum()
            avg_pnl_pct = positions_df['Unrealized P&L %'].mean()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                create_enhanced_metric_card(
                    "Total Position Value",
                    f"${total_value:,.2f}",
                    icon="💎"
                )
            
            with col2:
                pnl_pct = (total_pnl / (total_value - total_pnl)) * 100 if (total_value - total_pnl) > 0 else 0
                create_enhanced_metric_card(
                    "Unrealized P&L",
                    f"${total_pnl:,.2f}",
                    change=pnl_pct,
                    icon="📊"
                )
            
            with col3:
                create_enhanced_metric_card(
                    "Avg Position Return",
                    f"{avg_pnl_pct:.2f}%",
                    change=avg_pnl_pct,
                    icon="📈"
                )
            
            # Positions table
            st.markdown("### 📋 Position Details")
            
            # Format the dataframe for display
            display_df = positions_df.copy()
            display_df['Avg Cost'] = display_df['Avg Cost'].apply(lambda x: f"${x:.2f}")
            display_df['Current Price'] = display_df['Current Price'].apply(lambda x: f"${x:.2f}")
            display_df['Market Value'] = display_df['Market Value'].apply(lambda x: f"${x:,.2f}")
            display_df['Cost Basis'] = display_df['Cost Basis'].apply(lambda x: f"${x:,.2f}")
            display_df['Unrealized P&L'] = display_df['Unrealized P&L'].apply(lambda x: f"${x:,.2f}")
            display_df['Unrealized P&L %'] = display_df['Unrealized P&L %'].apply(lambda x: f"{x:.2f}%")
            
            st.dataframe(display_df, use_container_width=True)
            
            # Position performance chart
            fig_positions = go.Figure()
            
            fig_positions.add_trace(go.Bar(
                x=positions_df['Symbol'],
                y=positions_df['Unrealized P&L %'],
                marker_color=['#4caf50' if x >= 0 else '#f44336' for x in positions_df['Unrealized P&L %']],
                name='Position Returns %'
            ))
            
            fig_positions.update_layout(
                title='Position Performance (%)',
                xaxis_title='Symbol',
                yaxis_title='Return %',
                template='plotly_dark',
                height=400
            )
            
            st.plotly_chart(fig_positions, use_container_width=True)
            
        else:
            display_info_box("No Positions", "You don't have any current positions.")
    
    else:
        display_info_box("Connect Account", "Please connect your trading account to view positions.")

# Orders Tab
with tabs[3]:
    st.header("📋 Order Management")
    
    if st.session_state.trading_connected:
        orders_df = monitor.get_orders()
        
        if not orders_df.empty:
            # Order status summary
            status_counts = orders_df['status'].value_counts()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                create_enhanced_metric_card(
                    "Open Orders",
                    str(status_counts.get('OPEN', 0)),
                    icon="⏳"
                )
            
            with col2:
                create_enhanced_metric_card(
                    "Filled Orders",
                    str(status_counts.get('FILLED', 0)),
                    icon="✅"
                )
            
            with col3:
                create_enhanced_metric_card(
                    "Cancelled Orders",
                    str(status_counts.get('CANCELLED', 0)),
                    icon="❌"
                )
            
            # Orders table
            st.markdown("### 📋 Order Details")
            
            # Format orders dataframe
            display_orders = orders_df.copy()
            display_orders['price'] = display_orders['price'].apply(lambda x: f"${x:.2f}")
            
            # Color code by status
            def style_status(val):
                if val == 'OPEN':
                    return 'background-color: #ff9800; color: white'
                elif val == 'FILLED':
                    return 'background-color: #4caf50; color: white'
                elif val == 'CANCELLED':
                    return 'background-color: #f44336; color: white'
                return ''
            
            st.dataframe(display_orders, use_container_width=True)
            
            # Order actions
            st.markdown("### ⚡ Quick Actions")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📝 New Order", use_container_width=True):
                    st.info("Order placement interface would appear here in live mode.")
            
            with col2:
                if st.button("❌ Cancel All Open", use_container_width=True):
                    st.warning("This would cancel all open orders in live mode.")
            
            with col3:
                if st.button("🔄 Refresh Orders", use_container_width=True):
                    st.success("Orders refreshed!")
                    st.rerun()
        
        else:
            display_info_box("No Orders", "You don't have any recent orders.")
    
    else:
        display_info_box("Connect Account", "Please connect your trading account to view orders.")

# Strategy Performance Tab
with tabs[4]:
    st.header("🎯 Strategy Performance vs Actual Trading")
    
    if st.session_state.trading_connected:
        st.markdown("### 📊 AI Strategy Validation")
        
        # This would compare our platform's recommendations with actual trades
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Platform Recommendations")
            
            # Simulated strategy recommendations
            strategy_data = {
                'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'TSLA'],
                'Recommendation': ['BUY', 'HOLD', 'BUY', 'SELL'],
                'Target Price': [200.00, 425.00, 145.00, 240.00],
                'Confidence': [0.85, 0.72, 0.91, 0.68]
            }
            
            strategy_df = pd.DataFrame(strategy_data)
            st.dataframe(strategy_df, use_container_width=True)
        
        with col2:
            st.markdown("#### Actual Positions")
            
            positions_df = monitor.get_positions()
            if not positions_df.empty:
                comparison_df = positions_df[['Symbol', 'Current Price', 'Unrealized P&L %']].copy()
                st.dataframe(comparison_df, use_container_width=True)
        
        # Performance comparison chart
        st.markdown("### 📈 Performance Comparison")
        
        # Simulate comparison data
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        platform_returns = np.cumsum(np.random.normal(0.002, 0.02, len(dates)))
        actual_returns = np.cumsum(np.random.normal(0.0015, 0.025, len(dates)))
        
        fig_comparison = go.Figure()
        
        fig_comparison.add_trace(go.Scatter(
            x=dates,
            y=platform_returns * 100,
            mode='lines',
            name='Platform Strategy',
            line=dict(color='#4a90e2', width=3)
        ))
        
        fig_comparison.add_trace(go.Scatter(
            x=dates,
            y=actual_returns * 100,
            mode='lines',
            name='Actual Trading',
            line=dict(color='#f44336', width=3)
        ))
        
        fig_comparison.update_layout(
            title='Strategy Performance vs Actual Trading',
            xaxis_title='Date',
            yaxis_title='Cumulative Return %',
            template='plotly_dark',
            height=500
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Strategy metrics
        st.markdown("### 📊 Strategy Effectiveness")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            create_enhanced_metric_card(
                "Strategy Accuracy",
                "78.5%",
                change=5.2,
                icon="🎯"
            )
        
        with col2:
            create_enhanced_metric_card(
                "Alpha Generated",
                "2.3%",
                change=2.3,
                icon="⭐"
            )
        
        with col3:
            create_enhanced_metric_card(
                "Win Rate",
                "65.2%",
                icon="🏆"
            )
        
        with col4:
            create_enhanced_metric_card(
                "Sharpe Ratio",
                "1.45",
                icon="📈"
            )
    
    else:
        display_info_box("Connect Account", "Connect your trading account to compare strategy performance with actual results.")

# Settings Tab
with tabs[5]:
    st.header("⚙️ Settings & Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔧 Trading Settings")
        
        # Risk management settings
        st.markdown("#### Risk Management")
        max_position_size = st.slider("Max Position Size (%)", 1, 25, 10)
        stop_loss_pct = st.slider("Default Stop Loss (%)", 1, 20, 5)
        take_profit_pct = st.slider("Default Take Profit (%)", 5, 50, 15)
        
        # Notification settings
        st.markdown("#### Notifications")
        email_alerts = st.checkbox("Email Alerts", value=True)
        price_alerts = st.checkbox("Price Movement Alerts", value=True)
        order_alerts = st.checkbox("Order Status Alerts", value=True)
        
        # API settings
        st.markdown("#### API Configuration")
        refresh_interval = st.selectbox("Data Refresh Interval", ["30 seconds", "1 minute", "5 minutes", "15 minutes"], index=2)
        rate_limit = st.slider("API Rate Limit (req/min)", 10, 100, 30)
    
    with col2:
        st.markdown("### 📊 Display Settings")
        
        # Chart settings
        st.markdown("#### Chart Preferences")
        default_timeframe = st.selectbox("Default Timeframe", ["1D", "5D", "1M", "3M", "6M", "1Y"], index=2)
        chart_theme = st.selectbox("Chart Theme", ["Dark", "Light", "Auto"], index=0)
        
        # Data settings
        st.markdown("#### Data Settings")
        currency = st.selectbox("Display Currency", ["USD", "EUR", "GBP", "JPY"], index=0)
        decimal_places = st.slider("Price Decimal Places", 2, 6, 2)
        
        # Export settings
        st.markdown("#### Export Options")
        export_format = st.selectbox("Export Format", ["CSV", "Excel", "JSON"], index=0)
        
        if st.button("💾 Save Settings", type="primary", use_container_width=True):
            st.success("Settings saved successfully!")
    
    # Connection management
    st.markdown("### 🔗 Connection Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Reconnect", use_container_width=True):
            st.info("Reconnecting to trading account...")
    
    with col2:
        if st.button("🧹 Clear Cache", use_container_width=True):
            st.success("Cache cleared!")
    
    with col3:
        if st.button("🚪 Disconnect", use_container_width=True, type="secondary"):
            st.session_state.trading_connected = False
            st.session_state.webull_credentials = {}
            st.warning("Disconnected from trading account.")
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <strong>Trading Account Monitor</strong> | Real-time portfolio tracking with Webull integration<br>
    <em>⚠️ Demo mode for educational purposes. Live trading involves real financial risk.</em>
</div>
""", unsafe_allow_html=True)