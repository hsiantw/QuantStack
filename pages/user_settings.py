import streamlit as st
import pandas as pd
import sys
import os
from datetime import datetime

# Add the current directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.auth import (
    init_session_state, require_auth, save_user_data, load_user_data
)

st.set_page_config(
    page_title="User Settings - Quantitative Finance Platform",
    page_icon="⚙️",
    layout="wide"
)

# Custom CSS for professional dark theme
st.markdown("""
<style>
    /* Account settings specific styling */
    .settings-header {
        background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid #333344;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .settings-section {
        background: linear-gradient(135deg, #1a1a2e 0%, #2a2a3e 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #333344;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .strategy-card {
        background: linear-gradient(135deg, #2a2a3e 0%, #1e1e2e 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #444455;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .strategy-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(31, 119, 180, 0.2);
        border-color: #1f77b4;
    }
    
    .portfolio-card {
        background: linear-gradient(135deg, #1e2a3e 0%, #2a1e3e 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #445544;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .portfolio-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(23, 190, 207, 0.2);
        border-color: #17becf;
    }
</style>
""", unsafe_allow_html=True)

@require_auth
def show_user_settings():
    """Display user account settings and management"""
    
    # Initialize session state
    init_session_state()
    
    if not st.session_state.get('authenticated', False):
        st.error("Please log in to access your account settings.")
        return
    
    user = st.session_state.user
    auth_manager = st.session_state.auth_manager
    
    # Header
    st.markdown(f"""
    <div class="settings-header">
        <h1 style="color: #00d4ff; margin: 0;">⚙️ Account Settings</h1>
        <p style="color: #888; margin: 0.5rem 0;">Manage your preferences, strategies, and portfolios</p>
        <p style="color: #666; font-size: 0.9rem;">Welcome back, <strong style="color: #00d4ff;">{user['username']}</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different settings sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "👤 Profile Settings", 
        "📊 My Strategies", 
        "💼 My Portfolios", 
        "🔧 Preferences"
    ])
    
    with tab1:
        show_profile_settings(user, auth_manager)
    
    with tab2:
        show_saved_strategies(user, auth_manager)
    
    with tab3:
        show_saved_portfolios(user, auth_manager)
    
    with tab4:
        show_user_preferences(user, auth_manager)

def show_profile_settings(user, auth_manager):
    """Display and edit profile settings"""
    st.header("👤 Profile Information")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="settings-section">
        """, unsafe_allow_html=True)
        
        st.subheader("Account Details")
        
        # Display current user information
        info_data = {
            "Username": user['username'],
            "Email": user['email'],
            "Full Name": user.get('full_name', 'Not provided'),
            "Subscription Tier": user.get('subscription_tier', 'free').title(),
            "Account Created": "Recently"  # Would need to add this to the user data
        }
        
        for key, value in info_data.items():
            st.markdown(f"**{key}:** {value}")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Edit profile form
        st.markdown("""
        <div class="settings-section">
        """, unsafe_allow_html=True)
        
        st.subheader("Update Profile")
        
        with st.form("update_profile"):
            new_full_name = st.text_input("Full Name", value=user.get('full_name', ''))
            new_email = st.text_input("Email", value=user['email'])
            
            # Password change section
            st.markdown("**Change Password**")
            current_password = st.text_input("Current Password", type="password")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm New Password", type="password")
            
            if st.form_submit_button("Update Profile", type="primary"):
                # Here you would implement the profile update logic
                st.success("Profile update functionality will be implemented in the next version.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="settings-section">
        """, unsafe_allow_html=True)
        
        st.subheader("Account Statistics")
        
        # Load user statistics
        saved_strategies = load_user_data('saved_strategies', [])
        saved_portfolios = load_user_data('saved_portfolios', [])
        total_backtests = load_user_data('total_backtests', 0)
        
        st.metric("Saved Strategies", len(saved_strategies))
        st.metric("Saved Portfolios", len(saved_portfolios))
        st.metric("Total Backtests", total_backtests)
        
        # Usage score
        usage_score = min((len(saved_strategies) + len(saved_portfolios)) * 10, 100)
        st.metric("Usage Score", f"{usage_score}%")
        
        st.markdown("</div>", unsafe_allow_html=True)

def show_saved_strategies(user, auth_manager):
    """Display and manage saved trading strategies"""
    st.header("📊 My Trading Strategies")
    
    # Load saved strategies
    saved_strategies = auth_manager.get_user_strategies(user['id'])
    
    if saved_strategies:
        st.success(f"You have {len(saved_strategies)} saved strategies")
        
        for i, strategy in enumerate(saved_strategies):
            st.markdown(f"""
            <div class="strategy-card">
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([3, 2, 1])
            
            with col1:
                st.markdown(f"### {strategy['name']}")
                st.markdown(f"**Type:** {strategy['type']}")
                st.markdown(f"**Created:** {strategy['created_at'][:10]}")
                st.markdown(f"**Updated:** {strategy['updated_at'][:10]}")
            
            with col2:
                if strategy['backtest_results']:
                    results = strategy['backtest_results']
                    if isinstance(results, dict):
                        st.metric("Annual Return", f"{results.get('annual_return', 0):.2%}")
                        st.metric("Sharpe Ratio", f"{results.get('sharpe_ratio', 0):.2f}")
                        st.metric("Max Drawdown", f"{results.get('max_drawdown', 0):.2%}")
                else:
                    st.info("No backtest results")
            
            with col3:
                if strategy['is_favorite']:
                    st.markdown("⭐ **Favorite**")
                
                # Action buttons
                if st.button(f"Load Strategy", key=f"load_strat_{i}"):
                    st.info("Strategy loading functionality will be implemented soon.")
                
                if st.button(f"Delete", key=f"delete_strat_{i}"):
                    st.warning("Delete functionality will be implemented soon.")
            
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No saved strategies yet. Start by creating and saving a strategy from the trading modules.")
        
        # Quick action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🤖 AI Pairs Trading", use_container_width=True):
                st.switch_page("pages/ai_pairs_trading.py")
        
        with col2:
            if st.button("📈 Strategy Backtesting", use_container_width=True):
                st.switch_page("pages/trading_strategies.py")
        
        with col3:
            if st.button("🎯 Strategy Comparison", use_container_width=True):
                st.switch_page("pages/strategy_comparison.py")

def show_saved_portfolios(user, auth_manager):
    """Display and manage saved portfolios"""
    st.header("💼 My Portfolios")
    
    # Load saved portfolios
    saved_portfolios = auth_manager.get_user_portfolios(user['id'])
    
    if saved_portfolios:
        st.success(f"You have {len(saved_portfolios)} saved portfolios")
        
        for i, portfolio in enumerate(saved_portfolios):
            st.markdown(f"""
            <div class="portfolio-card">
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([3, 2, 1])
            
            with col1:
                st.markdown(f"### {portfolio['name']}")
                st.markdown(f"**Created:** {portfolio['created_at'][:10]}")
                st.markdown(f"**Updated:** {portfolio['updated_at'][:10]}")
                
                # Show portfolio composition if available
                if portfolio['data'] and isinstance(portfolio['data'], dict):
                    tickers = portfolio['data'].get('tickers', [])
                    if tickers:
                        st.markdown(f"**Assets:** {', '.join(tickers[:5])}")
                        if len(tickers) > 5:
                            st.markdown(f"*... and {len(tickers) - 5} more*")
            
            with col2:
                if portfolio['metrics']:
                    metrics = portfolio['metrics']
                    if isinstance(metrics, dict):
                        st.metric("Expected Return", f"{metrics.get('expected_return', 0):.2%}")
                        st.metric("Risk (Volatility)", f"{metrics.get('volatility', 0):.2%}")
                        st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
                else:
                    st.info("No metrics available")
            
            with col3:
                # Action buttons
                if st.button(f"Load Portfolio", key=f"load_port_{i}"):
                    st.info("Portfolio loading functionality will be implemented soon.")
                
                if st.button(f"Delete", key=f"delete_port_{i}"):
                    st.warning("Delete functionality will be implemented soon.")
            
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("No saved portfolios yet. Start by optimizing a portfolio and saving it.")
        
        # Quick action buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Portfolio Optimization", use_container_width=True):
                st.switch_page("pages/portfolio_optimization.py")
        
        with col2:
            if st.button("💼 Portfolio Manager", use_container_width=True):
                st.switch_page("pages/portfolio_manager.py")

def show_user_preferences(user, auth_manager):
    """Display and edit user preferences"""
    st.header("🔧 Platform Preferences")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="settings-section">
        """, unsafe_allow_html=True)
        
        st.subheader("General Preferences")
        
        # Load current preferences
        current_portfolio_size = load_user_data('default_portfolio_size', '100000')
        current_risk_tolerance = load_user_data('risk_tolerance', 'moderate')
        current_theme = load_user_data('theme', 'dark')
        
        with st.form("preferences_form"):
            # Portfolio preferences
            default_portfolio_size = st.selectbox(
                "Default Portfolio Size",
                options=['10000', '50000', '100000', '250000', '500000', '1000000'],
                index=['10000', '50000', '100000', '250000', '500000', '1000000'].index(current_portfolio_size),
                format_func=lambda x: f"${int(x):,}"
            )
            
            risk_tolerance = st.selectbox(
                "Risk Tolerance",
                options=['conservative', 'moderate', 'aggressive'],
                index=['conservative', 'moderate', 'aggressive'].index(current_risk_tolerance),
                format_func=lambda x: x.title()
            )
            
            theme = st.selectbox(
                "Theme",
                options=['dark', 'light'],
                index=['dark', 'light'].index(current_theme),
                format_func=lambda x: x.title()
            )
            
            # Save preferences
            if st.form_submit_button("Save Preferences", type="primary"):
                save_user_data('default_portfolio_size', default_portfolio_size)
                save_user_data('risk_tolerance', risk_tolerance)
                save_user_data('theme', theme)
                st.success("Preferences saved successfully!")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="settings-section">
        """, unsafe_allow_html=True)
        
        st.subheader("Trading Preferences")
        
        # Load trading preferences
        current_strategies = load_user_data('preferred_strategies', [])
        
        with st.form("trading_preferences"):
            # Preferred strategies
            strategy_options = [
                'Mean Reversion',
                'Momentum',
                'Pairs Trading',
                'Portfolio Optimization',
                'Statistical Arbitrage',
                'Opening Range Breakout'
            ]
            
            preferred_strategies = st.multiselect(
                "Preferred Strategies",
                options=strategy_options,
                default=current_strategies
            )
            
            # Notification settings
            st.markdown("**Notification Settings**")
            
            email_notifications = st.checkbox("Email Notifications", value=True)
            trading_signals = st.checkbox("Trading Signal Alerts", value=True)
            market_updates = st.checkbox("Market Update Notifications", value=False)
            
            if st.form_submit_button("Save Trading Preferences", type="primary"):
                save_user_data('preferred_strategies', preferred_strategies)
                save_user_data('email_notifications', email_notifications)
                save_user_data('trading_signals', trading_signals)
                save_user_data('market_updates', market_updates)
                st.success("Trading preferences saved!")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Data management section
    st.markdown("""
    <div class="settings-section">
    """, unsafe_allow_html=True)
    
    st.subheader("🗂️ Data Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Export My Data", use_container_width=True):
            st.info("Data export functionality will be available soon.")
    
    with col2:
        if st.button("🔄 Sync Settings", use_container_width=True):
            st.info("Settings synchronized successfully!")
    
    with col3:
        if st.button("🗑️ Clear Cache", use_container_width=True):
            # Clear session state cache
            for key in list(st.session_state.keys()):
                if str(key).startswith('cached_'):
                    del st.session_state[key]
            st.success("Cache cleared!")
    
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    show_user_settings()