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
import json

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ui_components import create_enhanced_metric_card, display_info_box, display_success_message, display_warning_message, display_error_message

# Page configuration
st.set_page_config(
    page_title="Portfolio Manager",
    page_icon="📁",
    layout="wide"
)

# Apply custom styling
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5aa0 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #4a90e2;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(74, 144, 226, 0.3);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #ffffff;
        margin: 0.5rem 0;
    }
    .metric-label {
        color: #b3d9ff;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .info-box {
        background: linear-gradient(135deg, #1a472a 0%, #2d5016 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
        color: white;
    }
    .success-box {
        background: linear-gradient(135deg, #1a472a 0%, #2d5016 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
        color: white;
        animation: slideIn 0.3s ease-out;
    }
    .warning-box {
        background: linear-gradient(135deg, #5d4037 0%, #8d6e63 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
        color: white;
        animation: slideIn 0.3s ease-out;
    }
    .error-box {
        background: linear-gradient(135deg, #c62828 0%, #d32f2f 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #f44336;
        margin: 1rem 0;
        color: white;
        animation: slideIn 0.3s ease-out;
    }
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

def initialize_portfolio_storage():
    """Initialize session state for portfolio storage"""
    if 'portfolios' not in st.session_state:
        st.session_state.portfolios = {}
    if 'current_portfolio' not in st.session_state:
        st.session_state.current_portfolio = None

def save_portfolio(name, instruments, weights=None, description=""):
    """Save a portfolio to session state"""
    if weights is None:
        weights = [1.0/len(instruments)] * len(instruments)
    
    portfolio_data = {
        'name': name,
        'instruments': instruments,
        'weights': weights,
        'description': description,
        'created_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'last_modified': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    st.session_state.portfolios[name] = portfolio_data
    st.success(f"Portfolio '{name}' saved successfully!")

def load_portfolio(name):
    """Load a portfolio from session state"""
    if name in st.session_state.portfolios:
        return st.session_state.portfolios[name]
    return None

def delete_portfolio(name):
    """Delete a portfolio from session state"""
    if name in st.session_state.portfolios:
        del st.session_state.portfolios[name]
        st.success(f"Portfolio '{name}' deleted successfully!")

def get_portfolio_data(instruments, weights, period="1y"):
    """Fetch portfolio data and calculate performance"""
    portfolio_data = {}
    
    try:
        # Download data for all instruments
        data = yf.download(instruments, period=period, progress=False)
        
        if len(instruments) == 1:
            # Handle single instrument case
            data = pd.DataFrame({instruments[0]: data['Close']})
        else:
            data = data['Close']
        
        # Calculate portfolio value
        portfolio_value = (data * weights).sum(axis=1)
        
        # Calculate returns
        portfolio_returns = portfolio_value.pct_change().dropna()
        
        # Performance metrics
        total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1) * 100
        volatility = portfolio_returns.std() * np.sqrt(252) * 100
        sharpe_ratio = (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252) if portfolio_returns.std() != 0 else 0
        max_drawdown = calculate_max_drawdown(portfolio_value)
        
        portfolio_data = {
            'data': data,
            'portfolio_value': portfolio_value,
            'portfolio_returns': portfolio_returns,
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'current_value': portfolio_value.iloc[-1],
            'instruments': instruments,
            'weights': weights
        }
        
    except Exception as e:
        st.error(f"Error fetching portfolio data: {str(e)}")
        return None
    
    return portfolio_data

def calculate_max_drawdown(portfolio_value):
    """Calculate maximum drawdown"""
    peak = portfolio_value.expanding(min_periods=1).max()
    drawdown = (portfolio_value - peak) / peak * 100
    return drawdown.min()

def display_portfolio_performance(portfolio_data):
    """Display portfolio performance metrics and charts"""
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Return",
            f"{portfolio_data['total_return']:.2f}%",
            delta=None
        )
    
    with col2:
        st.metric(
            "Volatility (Annual)",
            f"{portfolio_data['volatility']:.2f}%",
            delta=None
        )
    
    with col3:
        st.metric(
            "Sharpe Ratio",
            f"{portfolio_data['sharpe_ratio']:.2f}",
            delta=None
        )
    
    with col4:
        st.metric(
            "Max Drawdown",
            f"{portfolio_data['max_drawdown']:.2f}%",
            delta=None
        )
    
    # Portfolio value chart
    fig_portfolio = go.Figure()
    
    fig_portfolio.add_trace(go.Scatter(
        x=portfolio_data['portfolio_value'].index,
        y=portfolio_data['portfolio_value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='#00d4ff', width=2)
    ))
    
    fig_portfolio.update_layout(
        title="Portfolio Performance Over Time",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        template="plotly_dark",
        height=400
    )
    
    st.plotly_chart(fig_portfolio, use_container_width=True)
    
    # Asset allocation pie chart
    fig_allocation = go.Figure(data=[go.Pie(
        labels=portfolio_data['instruments'],
        values=portfolio_data['weights'],
        hole=0.3
    )])
    
    fig_allocation.update_layout(
        title="Asset Allocation",
        template="plotly_dark",
        height=400
    )
    
    st.plotly_chart(fig_allocation, use_container_width=True)

def export_portfolio_to_json(portfolio_data):
    """Export portfolio data to JSON format"""
    export_data = {
        'name': portfolio_data['name'],
        'instruments': portfolio_data['instruments'],
        'weights': portfolio_data['weights'],
        'description': portfolio_data['description'],
        'created_date': portfolio_data['created_date'],
        'last_modified': portfolio_data['last_modified']
    }
    return json.dumps(export_data, indent=2)

def import_portfolio_from_json(json_string):
    """Import portfolio from JSON string"""
    try:
        portfolio_data = json.loads(json_string)
        required_fields = ['name', 'instruments', 'weights']
        
        if all(field in portfolio_data for field in required_fields):
            return portfolio_data
        else:
            return None
    except:
        return None

def main():
    st.title("📁 Investment Portfolio Manager")
    
    st.markdown("""
    **Save, manage, and analyze your investment portfolios with professional portfolio management tools.
    Create custom portfolios, track performance, and optimize asset allocation.**
    """)
    
    # Initialize portfolio storage
    initialize_portfolio_storage()
    
    # Create tabs
    tabs = st.tabs([
        "📋 Portfolio Builder",
        "📊 Saved Portfolios", 
        "📈 Portfolio Analysis",
        "⚙️ Portfolio Management",
        "📤 Import/Export"
    ])
    
    # Portfolio Builder Tab
    with tabs[0]:
        st.header("📋 Create New Portfolio")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Portfolio basic info
            portfolio_name = st.text_input("Portfolio Name", placeholder="e.g., Conservative Growth Portfolio")
            portfolio_description = st.text_area("Description (Optional)", placeholder="Brief description of your investment strategy...")
            
            st.markdown("### 📈 Add Instruments")
            
            # Instrument input method
            input_method = st.radio(
                "Choose input method:",
                ["Manual Entry", "Upload CSV", "Template Selection"]
            )
            
            instruments = []
            weights = []
            
            if input_method == "Manual Entry":
                # Manual instrument entry
                num_instruments = st.number_input("Number of instruments", min_value=1, max_value=20, value=3)
                
                st.markdown("**Enter your instruments and weights:**")
                
                for i in range(num_instruments):
                    col_ticker, col_weight = st.columns([2, 1])
                    
                    with col_ticker:
                        ticker = st.text_input(f"Instrument {i+1}", key=f"ticker_{i}", placeholder="e.g., AAPL, SPY, TSLA")
                        if ticker:
                            instruments.append(ticker.upper())
                    
                    with col_weight:
                        weight = st.number_input(f"Weight {i+1} (%)", min_value=0.0, max_value=100.0, value=100.0/num_instruments, key=f"weight_{i}")
                        weights.append(weight/100.0)
            
            elif input_method == "Upload CSV":
                st.markdown("**Upload a CSV file with columns: Ticker, Weight**")
                uploaded_file = st.file_uploader("Choose CSV file", type="csv")
                
                if uploaded_file is not None:
                    try:
                        df = pd.read_csv(uploaded_file)
                        if 'Ticker' in df.columns and 'Weight' in df.columns:
                            instruments = df['Ticker'].tolist()
                            weights = (df['Weight'] / 100.0).tolist()
                            st.success(f"Loaded {len(instruments)} instruments from CSV")
                            st.dataframe(df)
                        else:
                            st.error("CSV must have 'Ticker' and 'Weight' columns")
                    except Exception as e:
                        st.error(f"Error reading CSV: {str(e)}")
            
            elif input_method == "Template Selection":
                st.markdown("**Choose from pre-built portfolio templates:**")
                
                templates = {
                    "Conservative (60/40)": {
                        "instruments": ["SPY", "TLT"],
                        "weights": [0.6, 0.4],
                        "description": "60% stocks, 40% bonds - classic conservative allocation"
                    },
                    "Aggressive Growth": {
                        "instruments": ["QQQ", "VTI", "ARKK"],
                        "weights": [0.4, 0.4, 0.2],
                        "description": "Growth-focused portfolio with tech exposure"
                    },
                    "Dividend Income": {
                        "instruments": ["VYM", "SCHD", "DVY"],
                        "weights": [0.4, 0.3, 0.3],
                        "description": "High dividend yield portfolio"
                    },
                    "Global Diversified": {
                        "instruments": ["VTI", "VXUS", "VEA", "VWO"],
                        "weights": [0.4, 0.3, 0.2, 0.1],
                        "description": "Globally diversified equity portfolio"
                    }
                }
                
                selected_template = st.selectbox("Select Template", list(templates.keys()))
                
                if selected_template:
                    template_data = templates[selected_template]
                    instruments = template_data["instruments"]
                    weights = template_data["weights"]
                    
                    st.info(f"**{selected_template}**: {template_data['description']}")
                    
                    # Display template details
                    template_df = pd.DataFrame({
                        'Instrument': instruments,
                        'Weight (%)': [w*100 for w in weights]
                    })
                    st.dataframe(template_df)
        
        with col2:
            # Portfolio summary
            st.markdown("### 📊 Portfolio Summary")
            
            if instruments and weights:
                # Validate weights sum to 100%
                total_weight = sum(weights) * 100
                
                if abs(total_weight - 100) > 0.01:
                    st.warning(f"⚠️ Weights sum to {total_weight:.1f}% (should be 100%)")
                else:
                    st.success("✅ Weights sum to 100%")
                
                # Show portfolio composition
                portfolio_df = pd.DataFrame({
                    'Instrument': instruments,
                    'Weight (%)': [f"{w*100:.1f}%" for w in weights]
                })
                st.dataframe(portfolio_df, use_container_width=True)
                
                # Save portfolio button
                if portfolio_name and st.button("💾 Save Portfolio", type="primary", use_container_width=True):
                    save_portfolio(portfolio_name, instruments, weights, portfolio_description)
                    st.rerun()
            else:
                st.info("Add instruments to see portfolio summary")
    
    # Saved Portfolios Tab
    with tabs[1]:
        st.header("📊 Saved Portfolios")
        
        if st.session_state.portfolios:
            # Display saved portfolios
            portfolio_names = list(st.session_state.portfolios.keys())
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                selected_portfolio_name = st.selectbox("Select Portfolio", portfolio_names)
            
            with col2:
                if st.button("🗑️ Delete Portfolio", key="delete_portfolio"):
                    delete_portfolio(selected_portfolio_name)
                    st.rerun()
            
            if selected_portfolio_name:
                portfolio = load_portfolio(selected_portfolio_name)
                
                if portfolio:
                    # Portfolio details
                    st.markdown(f"### {portfolio['name']}")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        if portfolio.get('description'):
                            st.markdown(f"**Description:** {portfolio['description']}")
                        
                        st.markdown(f"**Created:** {portfolio['created_date']}")
                        st.markdown(f"**Last Modified:** {portfolio['last_modified']}")
                    
                    with col2:
                        if st.button("📈 Analyze Portfolio", key="analyze_portfolio", type="primary"):
                            st.session_state.current_portfolio = portfolio
                            st.success("Portfolio loaded for analysis!")
                    
                    # Portfolio composition
                    st.markdown("#### Portfolio Composition")
                    composition_df = pd.DataFrame({
                        'Instrument': portfolio['instruments'],
                        'Weight (%)': [f"{w*100:.1f}%" for w in portfolio['weights']]
                    })
                    st.dataframe(composition_df, use_container_width=True)
        else:
            st.info("No portfolios saved yet. Create your first portfolio in the Portfolio Builder tab.")
    
    # Portfolio Analysis Tab
    with tabs[2]:
        st.header("📈 Portfolio Analysis")
        
        # Portfolio selection for analysis
        if st.session_state.portfolios:
            analysis_portfolio_name = st.selectbox(
                "Select portfolio to analyze:", 
                list(st.session_state.portfolios.keys()),
                key="analysis_portfolio_select"
            )
            
            if analysis_portfolio_name:
                portfolio = load_portfolio(analysis_portfolio_name)
                
                # Analysis period selection
                period = st.selectbox(
                    "Analysis Period:",
                    ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
                    index=3
                )
                
                if st.button("🔍 Run Analysis", type="primary"):
                    with st.spinner("Analyzing portfolio performance..."):
                        portfolio_data = get_portfolio_data(
                            portfolio['instruments'], 
                            portfolio['weights'], 
                            period
                        )
                        
                        if portfolio_data:
                            st.success("Analysis completed!")
                            display_portfolio_performance(portfolio_data)
                            
                            # Detailed breakdown
                            st.markdown("### 📊 Individual Asset Performance")
                            
                            individual_data = portfolio_data['data']
                            
                            if len(portfolio['instruments']) > 1:
                                # Individual asset returns
                                asset_returns = {}
                                for instrument in portfolio['instruments']:
                                    if instrument in individual_data.columns:
                                        asset_data = individual_data[instrument]
                                        total_return = (asset_data.iloc[-1] / asset_data.iloc[0] - 1) * 100
                                        asset_returns[instrument] = total_return
                                
                                # Display individual returns
                                returns_df = pd.DataFrame.from_dict(
                                    asset_returns, 
                                    orient='index', 
                                    columns=['Total Return (%)']
                                ).round(2)
                                
                                st.dataframe(returns_df, use_container_width=True)
                            else:
                                st.info("Single asset portfolio - no comparison available")
        else:
            st.info("No portfolios available for analysis. Create a portfolio first.")
    
    # Portfolio Management Tab
    with tabs[3]:
        st.header("⚙️ Portfolio Management")
        
        if st.session_state.portfolios:
            # Bulk operations
            st.markdown("### 🔄 Bulk Operations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📋 Clone Portfolio"):
                    if st.session_state.portfolios:
                        source_portfolio = st.selectbox("Select portfolio to clone:", list(st.session_state.portfolios.keys()))
                        new_name = st.text_input("New portfolio name:")
                        
                        if source_portfolio and new_name and new_name not in st.session_state.portfolios:
                            original = st.session_state.portfolios[source_portfolio]
                            save_portfolio(
                                new_name,
                                original['instruments'],
                                original['weights'],
                                f"Cloned from {source_portfolio}"
                            )
            
            with col2:
                if st.button("🗑️ Clear All Portfolios"):
                    if st.checkbox("I understand this will delete all portfolios"):
                        st.session_state.portfolios = {}
                        st.success("All portfolios cleared!")
            
            # Portfolio statistics
            st.markdown("### 📊 Portfolio Statistics")
            
            total_portfolios = len(st.session_state.portfolios)
            total_unique_instruments = len(set(
                instrument 
                for portfolio in st.session_state.portfolios.values() 
                for instrument in portfolio['instruments']
            ))
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Portfolios", total_portfolios)
            
            with col2:
                st.metric("Unique Instruments", total_unique_instruments)
            
            with col3:
                avg_size = np.mean([len(p['instruments']) for p in st.session_state.portfolios.values()]) if st.session_state.portfolios else 0
                st.metric("Avg Portfolio Size", f"{avg_size:.1f}")
        else:
            st.info("No portfolios to manage. Create some portfolios first.")
    
    # Import/Export Tab
    with tabs[4]:
        st.header("📤 Import/Export Portfolios")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📤 Export Portfolio")
            
            if st.session_state.portfolios:
                export_portfolio_name = st.selectbox(
                    "Select portfolio to export:",
                    list(st.session_state.portfolios.keys()),
                    key="export_select"
                )
                
                if export_portfolio_name:
                    portfolio = st.session_state.portfolios[export_portfolio_name]
                    json_data = export_portfolio_to_json(portfolio)
                    
                    st.download_button(
                        label="💾 Download Portfolio JSON",
                        data=json_data,
                        file_name=f"{export_portfolio_name.replace(' ', '_')}.json",
                        mime="application/json"
                    )
                    
                    st.code(json_data, language="json")
            else:
                st.info("No portfolios available for export")
        
        with col2:
            st.markdown("### 📥 Import Portfolio")
            
            uploaded_json = st.file_uploader("Upload Portfolio JSON", type="json")
            
            if uploaded_json is not None:
                try:
                    json_string = uploaded_json.read().decode('utf-8')
                    portfolio_data = import_portfolio_from_json(json_string)
                    
                    if portfolio_data:
                        st.success("Portfolio data validated!")
                        
                        # Preview portfolio
                        st.markdown("**Portfolio Preview:**")
                        preview_df = pd.DataFrame({
                            'Instrument': portfolio_data['instruments'],
                            'Weight (%)': [f"{w*100:.1f}%" for w in portfolio_data['weights']]
                        })
                        st.dataframe(preview_df)
                        
                        # Import button
                        if st.button("📥 Import Portfolio", type="primary"):
                            portfolio_name = portfolio_data['name']
                            
                            # Handle name conflicts
                            if portfolio_name in st.session_state.portfolios:
                                portfolio_name = f"{portfolio_name}_imported_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                            
                            save_portfolio(
                                portfolio_name,
                                portfolio_data['instruments'],
                                portfolio_data['weights'],
                                portfolio_data.get('description', '')
                            )
                            st.rerun()
                    else:
                        st.error("Invalid portfolio JSON format")
                        
                except Exception as e:
                    st.error(f"Error importing portfolio: {str(e)}")

if __name__ == "__main__":
    main()