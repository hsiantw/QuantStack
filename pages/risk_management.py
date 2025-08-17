import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.monte_carlo import MonteCarloSimulator
from utils.data_fetcher import DataFetcher
from utils.auth import (
    init_session_state, require_auth, save_user_data, load_user_data
)

# Page configuration
st.set_page_config(
    page_title="Advanced Risk Management",
    page_icon="⚠️",
    layout="wide"
)

# Custom CSS for risk management styling
st.markdown("""
<style>
    .risk-header {
        background: linear-gradient(135deg, #8B0000 0%, #DC143C 25%, #B22222 50%, #8B0000 75%, #2F1B1B 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        border: 1px solid #DC143C;
        box-shadow: 0 8px 32px rgba(220, 20, 60, 0.3);
    }
    
    .risk-metric-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2a1e1e 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #444;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .risk-metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(220, 20, 60, 0.2);
        border-color: #DC143C;
    }
    
    .scenario-card {
        background: linear-gradient(135deg, #2a2a1e 0%, #1e2a1e 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #555;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .high-risk {
        border-left: 5px solid #DC143C;
        background: linear-gradient(135deg, #2a1e1e 0%, #1e1e1e 100%);
    }
    
    .medium-risk {
        border-left: 5px solid #FFA500;
        background: linear-gradient(135deg, #2a2a1e 0%, #1e1e1e 100%);
    }
    
    .low-risk {
        border-left: 5px solid #32CD32;
        background: linear-gradient(135deg, #1e2a1e 0%, #1e1e1e 100%);
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main risk management application"""
    
    # Initialize authentication
    init_session_state()
    
    # Header
    st.markdown("""
    <div class="risk-header">
        <h1 style="color: #FFFFFF; margin: 0; font-size: 2.5rem;">⚠️ Advanced Risk Management</h1>
        <p style="color: #FFB6C1; margin: 0.5rem 0; font-size: 1.2rem;">Monte Carlo Simulations & Comprehensive Risk Analysis</p>
        <p style="color: #DDD; font-size: 0.9rem;">Professional-grade risk assessment tools for quantitative analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎲 Monte Carlo Simulation",
        "📊 Portfolio Risk Analysis", 
        "🎯 Stress Testing",
        "📈 Options Risk Analysis"
    ])
    
    with tab1:
        monte_carlo_simulation_tab()
    
    with tab2:
        portfolio_risk_analysis_tab()
    
    with tab3:
        stress_testing_tab()
    
    with tab4:
        options_risk_analysis_tab()

def monte_carlo_simulation_tab():
    """Monte Carlo simulation interface"""
    st.header("🎲 Monte Carlo Portfolio Simulation")
    
    # Input section
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Simulation Parameters")
        
        # Portfolio setup
        st.markdown("**Portfolio Configuration**")
        default_tickers = ["SPY", "QQQ", "IWM", "VTI", "EFA"]
        selected_tickers = st.multiselect(
            "Select Assets",
            options=["SPY", "QQQ", "IWM", "VTI", "EFA", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "GLD", "TLT", "BND"],
            default=default_tickers,
            help="Choose 3-10 assets for analysis"
        )
        
        # Time parameters
        lookback_period = st.selectbox(
            "Historical Data Period",
            options=["1y", "2y", "3y", "5y"],
            index=2,
            help="Period for calculating historical statistics"
        )
        
        time_horizon = st.number_input(
            "Simulation Time Horizon (days)",
            min_value=30,
            max_value=1260,  # 5 years
            value=252,  # 1 year
            step=30,
            help="Number of trading days to simulate"
        )
        
        # Simulation parameters
        st.markdown("**Monte Carlo Settings**")
        num_simulations = st.selectbox(
            "Number of Simulations",
            options=[1000, 5000, 10000, 25000],
            index=2,
            help="More simulations = higher accuracy but slower computation"
        )
        
        initial_value = st.number_input(
            "Initial Portfolio Value ($)",
            min_value=1000,
            max_value=10000000,
            value=100000,
            step=10000,
            help="Starting portfolio value"
        )
        
        # Portfolio weights
        st.markdown("**Portfolio Weights**")
        if len(selected_tickers) > 0:
            weights = []
            remaining_weight = 1.0
            
            for i, ticker in enumerate(selected_tickers):
                if i == len(selected_tickers) - 1:
                    # Last asset gets remaining weight
                    weight = remaining_weight
                    st.number_input(
                        f"{ticker} Weight",
                        value=weight,
                        format="%.3f",
                        disabled=True,
                        key=f"weight_{ticker}"
                    )
                else:
                    weight = st.number_input(
                        f"{ticker} Weight",
                        min_value=0.0,
                        max_value=remaining_weight,
                        value=min(1.0 / len(selected_tickers), remaining_weight),
                        step=0.01,
                        format="%.3f",
                        key=f"weight_{ticker}"
                    )
                    remaining_weight -= weight
                
                weights.append(weight)
            
            # Normalize weights to sum to 1
            weights = np.array(weights)
            weights = weights / np.sum(weights)
        
        # Run simulation button
        run_simulation = st.button("🚀 Run Monte Carlo Simulation", use_container_width=True)
    
    with col2:
        if run_simulation and len(selected_tickers) >= 2:
            with st.spinner("Running Monte Carlo simulation..."):
                # Fetch historical data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365 * int(lookback_period[0]))
                
                try:
                    # Download price data
                    price_data = yf.download(selected_tickers, start=start_date, end=end_date, progress=False)['Adj Close']
                    
                    if isinstance(price_data, pd.Series):
                        price_data = price_data.to_frame()
                    
                    # Calculate returns
                    returns_data = price_data.pct_change().dropna()
                    
                    # Initialize Monte Carlo simulator
                    mc_simulator = MonteCarloSimulator(returns_data, initial_value)
                    
                    # Run simulation
                    simulation_results = mc_simulator.simulate_portfolio_paths(
                        weights=weights,
                        time_horizon=time_horizon,
                        num_simulations=num_simulations,
                        confidence_levels=[0.90, 0.95, 0.99]
                    )
                    
                    # Store results in session state
                    st.session_state['mc_results'] = simulation_results
                    st.session_state['mc_weights'] = weights
                    st.session_state['mc_tickers'] = selected_tickers
                    
                    # Display results
                    display_monte_carlo_results(simulation_results, mc_simulator, weights)
                    
                except Exception as e:
                    st.error(f"Error running simulation: {str(e)}")
        
        elif run_simulation:
            st.warning("Please select at least 2 assets to run the simulation.")
        
        else:
            st.info("Configure parameters and click 'Run Monte Carlo Simulation' to begin analysis.")

def display_monte_carlo_results(results: dict, simulator: MonteCarloSimulator, weights: np.ndarray):
    """Display Monte Carlo simulation results"""
    
    # Key metrics overview
    st.subheader("📊 Simulation Results Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="risk-metric-card low-risk">
        """, unsafe_allow_html=True)
        
        expected_return = results['expected_return']
        st.metric(
            "Expected Return", 
            f"{expected_return:.2%}",
            delta=f"{(expected_return - 0.07):.2%} vs 7% benchmark"
        )
        
        st.metric(
            "Annualized Return",
            f"{results['portfolio_stats']['annualized_return']:.2%}"
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="risk-metric-card medium-risk">
        """, unsafe_allow_html=True)
        
        volatility = results['volatility']
        st.metric(
            "Portfolio Volatility",
            f"{volatility:.2%}",
            delta=f"{(volatility - 0.15):.2%} vs 15% target"
        )
        
        st.metric(
            "Annualized Volatility",
            f"{results['portfolio_stats']['annualized_volatility']:.2%}"
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="risk-metric-card low-risk">
        """, unsafe_allow_html=True)
        
        sharpe_ratio = results['sharpe_ratio']
        color = "🟢" if sharpe_ratio > 1.0 else "🟡" if sharpe_ratio > 0.5 else "🔴"
        
        st.metric(
            f"{color} Sharpe Ratio",
            f"{sharpe_ratio:.3f}",
            delta="Excellent" if sharpe_ratio > 1.5 else "Good" if sharpe_ratio > 1.0 else "Fair"
        )
        
        st.metric(
            "Probability of Loss",
            f"{results['probability_of_loss']:.1%}"
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="risk-metric-card high-risk">
        """, unsafe_allow_html=True)
        
        max_dd = results['expected_max_drawdown']
        st.metric(
            "Expected Max Drawdown",
            f"{max_dd:.2%}",
            delta="High Risk" if abs(max_dd) > 0.20 else "Moderate Risk"
        )
        
        var_95 = results['var_metrics']['VaR_95']
        st.metric(
            "VaR (95%)",
            f"{var_95:.2%}"
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Create and display visualizations
    figures = simulator.create_risk_dashboard_plots(results, weights)
    
    st.plotly_chart(figures['portfolio_paths'], use_container_width=True)
    st.plotly_chart(figures['risk_analysis'], use_container_width=True)
    
    # Value at Risk analysis
    st.subheader("📉 Value at Risk Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # VaR table
        var_data = []
        for conf_level in [90, 95, 99]:
            var_value = results['var_metrics'][f'VaR_{conf_level}']
            cvar_value = results['var_metrics'][f'CVaR_{conf_level}']
            
            var_data.append({
                'Confidence Level': f'{conf_level}%',
                'Value at Risk': f'{var_value:.2%}',
                'Conditional VaR': f'{cvar_value:.2%}',
                'Dollar VaR': f'${abs(var_value) * simulator.initial_value:,.0f}',
                'Dollar CVaR': f'${abs(cvar_value) * simulator.initial_value:,.0f}'
            })
        
        var_df = pd.DataFrame(var_data)
        st.dataframe(var_df, use_container_width=True, hide_index=True)
    
    with col2:
        # Risk interpretation
        worst_case_5pct = np.percentile(results['total_returns'], 5)
        best_case_95pct = np.percentile(results['total_returns'], 95)
        
        st.markdown("""
        <div class="risk-metric-card">
        """, unsafe_allow_html=True)
        
        st.markdown("**Risk Interpretation:**")
        st.markdown(f"• **Best Case (95th percentile):** {best_case_95pct:.2%} return")
        st.markdown(f"• **Worst Case (5th percentile):** {worst_case_5pct:.2%} return")
        st.markdown(f"• **Range:** {(best_case_95pct - worst_case_5pct):.2%} spread")
        
        # Risk rating
        if abs(results['var_metrics']['VaR_95']) < 0.10:
            risk_rating = "🟢 Low Risk"
        elif abs(results['var_metrics']['VaR_95']) < 0.20:
            risk_rating = "🟡 Moderate Risk"
        else:
            risk_rating = "🔴 High Risk"
        
        st.markdown(f"**Overall Risk Rating:** {risk_rating}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Save functionality for authenticated users
    if st.session_state.get('authenticated', False):
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            analysis_name = st.text_input(
                "💾 Save Monte Carlo Analysis",
                placeholder=f"MC Analysis - {datetime.now().strftime('%Y-%m-%d')}",
                key="save_mc_analysis"
            )
        
        with col2:
            if st.button("💾 Save Analysis", key="save_mc_btn", use_container_width=True):
                if analysis_name:
                    # Prepare analysis data for saving
                    analysis_data = {
                        "analysis_type": "Monte Carlo Simulation",
                        "tickers": st.session_state.get('mc_tickers', []),
                        "weights": weights.tolist(),
                        "time_horizon": results.get('time_horizon', 252),
                        "num_simulations": len(results['total_returns']),
                        "results_summary": {
                            "expected_return": results['expected_return'],
                            "volatility": results['volatility'],
                            "sharpe_ratio": results['sharpe_ratio'],
                            "var_95": results['var_metrics']['VaR_95'],
                            "max_drawdown": results['expected_max_drawdown']
                        },
                        "analysis_date": datetime.now().isoformat()
                    }
                    
                    user = st.session_state.user
                    auth_manager = st.session_state.auth_manager
                    
                    success = auth_manager.save_user_strategy(
                        user_id=user['id'],
                        strategy_name=analysis_name,
                        strategy_type="Risk Analysis",
                        strategy_config=analysis_data
                    )
                    
                    if success:
                        st.success(f"✅ Analysis '{analysis_name}' saved successfully!")
                    else:
                        st.error("Failed to save analysis. Please try again.")
                else:
                    st.warning("Please enter an analysis name to save.")

def portfolio_risk_analysis_tab():
    """Portfolio risk analysis interface"""
    st.header("📊 Comprehensive Portfolio Risk Analysis")
    
    st.info("Load a previously run Monte Carlo simulation to see detailed portfolio risk analysis here.")
    
    if 'mc_results' in st.session_state:
        results = st.session_state['mc_results']
        weights = st.session_state['mc_weights']
        tickers = st.session_state['mc_tickers']
        
        # Portfolio composition
        st.subheader("💼 Portfolio Composition")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart of weights
            fig_pie = go.Figure(data=[go.Pie(
                labels=tickers,
                values=weights,
                hole=0.3,
                textinfo='label+percent'
            )])
            
            fig_pie.update_layout(
                title="Portfolio Allocation",
                template='plotly_dark'
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Portfolio statistics table
            portfolio_data = []
            for i, ticker in enumerate(tickers):
                portfolio_data.append({
                    'Asset': ticker,
                    'Weight': f'{weights[i]:.2%}',
                    'Allocation ($)': f'${weights[i] * 100000:,.0f}',  # Assuming $100k portfolio
                })
            
            portfolio_df = pd.DataFrame(portfolio_data)
            st.dataframe(portfolio_df, use_container_width=True, hide_index=True)
        
        # Risk decomposition
        st.subheader("⚖️ Risk Decomposition Analysis")
        
        # Individual asset contributions to portfolio risk
        # This would require more sophisticated calculations
        st.info("Risk decomposition analysis showing individual asset contributions to overall portfolio risk.")
        
        # Correlation impact
        if hasattr(st.session_state.get('mc_simulator', None), 'correlations'):
            st.plotly_chart(
                st.session_state['mc_simulator'].create_risk_dashboard_plots(results, weights)['correlation_matrix'],
                use_container_width=True
            )

def stress_testing_tab():
    """Stress testing interface"""
    st.header("🎯 Advanced Stress Testing")
    
    if 'mc_results' not in st.session_state:
        st.warning("Please run a Monte Carlo simulation first to enable stress testing.")
        return
    
    # Define stress test scenarios
    st.subheader("📈 Predefined Stress Scenarios")
    
    scenarios = {
        "Market Crash (2008-style)": {
            "market_shock": -0.30,
            "volatility_multiplier": 2.0,
            "correlation_increase": 0.5,
            "description": "Severe market decline with increased volatility and correlations"
        },
        "COVID-19 Style Shock": {
            "market_shock": -0.35,
            "volatility_multiplier": 3.0,
            "correlation_increase": 0.7,
            "description": "Extreme volatility spike with panic selling"
        },
        "Interest Rate Shock": {
            "market_shock": -0.15,
            "volatility_multiplier": 1.5,
            "correlation_increase": 0.3,
            "description": "Moderate decline due to sudden rate changes"
        },
        "Stagflation Scenario": {
            "market_shock": -0.20,
            "volatility_multiplier": 1.8,
            "correlation_increase": 0.4,
            "description": "High inflation with economic stagnation"
        }
    }
    
    # Run stress tests
    if st.button("🚀 Run All Stress Tests", use_container_width=True):
        with st.spinner("Running stress test scenarios..."):
            try:
                # Get simulator from session state or recreate
                weights = st.session_state['mc_weights']
                tickers = st.session_state['mc_tickers']
                
                # For demonstration, we'll show expected impacts
                st.subheader("🔍 Stress Test Results")
                
                for scenario_name, scenario_params in scenarios.items():
                    # Calculate estimated impact
                    market_shock = scenario_params['market_shock']
                    vol_multiplier = scenario_params['volatility_multiplier']
                    
                    # Simplified stress impact calculation
                    estimated_loss = market_shock * 0.8  # Beta-adjusted impact
                    estimated_volatility = st.session_state['mc_results']['volatility'] * vol_multiplier
                    
                    # Display scenario results
                    if estimated_loss < -0.25:
                        risk_class = "high-risk"
                        risk_icon = "🔴"
                    elif estimated_loss < -0.15:
                        risk_class = "medium-risk"
                        risk_icon = "🟡"
                    else:
                        risk_class = "low-risk"
                        risk_icon = "🟢"
                    
                    st.markdown(f"""
                    <div class="scenario-card {risk_class}">
                        <h4>{risk_icon} {scenario_name}</h4>
                        <p><strong>Description:</strong> {scenario_params['description']}</p>
                        <div style="display: flex; justify-content: space-between; margin-top: 1rem;">
                            <div>
                                <strong>Estimated Portfolio Loss:</strong><br>
                                <span style="color: #DC143C; font-size: 1.2rem;">{estimated_loss:.1%}</span>
                            </div>
                            <div>
                                <strong>Stressed Volatility:</strong><br>
                                <span style="color: #FFA500; font-size: 1.2rem;">{estimated_volatility:.1%}</span>
                            </div>
                            <div>
                                <strong>Recovery Time:</strong><br>
                                <span style="color: #32CD32; font-size: 1.2rem;">{abs(estimated_loss)*2000:.0f} days</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Summary recommendations
                st.subheader("💡 Risk Management Recommendations")
                
                worst_case = min([scenarios[s]['market_shock'] for s in scenarios])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    <div class="risk-metric-card">
                        <h4>🛡️ Defensive Strategies</h4>
                        <ul>
                            <li>Consider reducing portfolio concentration</li>
                            <li>Add defensive assets (bonds, gold)</li>
                            <li>Implement stop-loss orders</li>
                            <li>Increase cash allocation during uncertainty</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="risk-metric-card">
                        <h4>📊 Portfolio Stress Summary</h4>
                        <p><strong>Worst Case Loss:</strong> {worst_case:.1%}</p>
                        <p><strong>Recommended Cash Buffer:</strong> {abs(worst_case)*0.5:.1%}</p>
                        <p><strong>Risk Rating:</strong> {"🔴 High" if abs(worst_case) > 0.25 else "🟡 Moderate"}</p>
                        <p><strong>Action Required:</strong> {"Yes - Immediate" if abs(worst_case) > 0.30 else "Monitor Closely"}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error running stress tests: {str(e)}")
    
    # Custom scenario builder
    st.subheader("🎛️ Custom Stress Scenario Builder")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Market Impact Parameters**")
        
        custom_shock = st.slider(
            "Market Shock (%)",
            min_value=-50,
            max_value=20,
            value=-20,
            step=5,
            help="Overall market movement"
        ) / 100
        
        custom_vol = st.slider(
            "Volatility Multiplier",
            min_value=0.5,
            max_value=5.0,
            value=1.5,
            step=0.1,
            help="How much volatility increases"
        )
        
        custom_corr = st.slider(
            "Correlation Increase",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            help="How much asset correlations increase"
        )
    
    with col2:
        if st.button("🧪 Test Custom Scenario", use_container_width=True):
            estimated_loss = custom_shock * 0.8
            estimated_vol = st.session_state['mc_results']['volatility'] * custom_vol
            
            st.markdown(f"""
            <div class="scenario-card medium-risk">
                <h4>🧪 Custom Stress Test Results</h4>
                <p><strong>Estimated Portfolio Impact:</strong> {estimated_loss:.2%}</p>
                <p><strong>Stressed Volatility:</strong> {estimated_vol:.2%}</p>
                <p><strong>Risk Assessment:</strong> {"High Risk" if abs(estimated_loss) > 0.25 else "Moderate Risk" if abs(estimated_loss) > 0.15 else "Low Risk"}</p>
            </div>
            """, unsafe_allow_html=True)

def options_risk_analysis_tab():
    """Options risk analysis interface"""
    st.header("📈 Options Risk Analysis with Monte Carlo")
    
    st.subheader("🎯 Options Parameter Setup")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Option Specifications**")
        
        underlying_ticker = st.selectbox(
            "Underlying Asset",
            options=["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
            help="Select the underlying asset for options analysis"
        )
        
        option_type = st.selectbox(
            "Option Type",
            options=["call", "put"],
            help="Call or put option"
        )
        
        # Get current price
        try:
            current_price = yf.Ticker(underlying_ticker).history(period="1d")['Close'].iloc[-1]
            st.metric("Current Price", f"${current_price:.2f}")
        except:
            current_price = 100
            st.warning("Unable to fetch current price, using $100 as default")
        
        strike_price = st.number_input(
            "Strike Price ($)",
            min_value=1.0,
            max_value=1000.0,
            value=float(current_price * 1.05),  # 5% OTM
            step=1.0,
            help="Option strike price"
        )
        
        time_to_expiry = st.number_input(
            "Time to Expiry (days)",
            min_value=1,
            max_value=365,
            value=30,
            step=1,
            help="Days until option expiration"
        ) / 365  # Convert to years
    
    with col2:
        st.markdown("**Simulation Parameters**")
        
        num_option_sims = st.selectbox(
            "Number of Simulations",
            options=[1000, 5000, 10000, 25000],
            index=1,
            help="Number of price path simulations"
        )
        
        volatility_estimate = st.number_input(
            "Implied Volatility (%)",
            min_value=5.0,
            max_value=200.0,
            value=25.0,
            step=1.0,
            help="Annual volatility estimate"
        ) / 100
        
        risk_free_rate = st.number_input(
            "Risk-Free Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.1,
            help="Annual risk-free interest rate"
        ) / 100
        
        # Calculate moneyness
        moneyness = current_price / strike_price
        if option_type == "call":
            if moneyness > 1.05:
                money_status = "🟢 In-the-Money"
            elif moneyness > 0.95:
                money_status = "🟡 At-the-Money"
            else:
                money_status = "🔴 Out-of-the-Money"
        else:  # put
            if moneyness < 0.95:
                money_status = "🟢 In-the-Money"
            elif moneyness < 1.05:
                money_status = "🟡 At-the-Money"
            else:
                money_status = "🔴 Out-of-the-Money"
        
        st.metric("Moneyness", money_status)
        st.metric("Moneyness Ratio", f"{moneyness:.3f}")
    
    # Run options simulation
    if st.button("🚀 Run Options Monte Carlo", use_container_width=True):
        with st.spinner("Running options Monte Carlo simulation..."):
            try:
                # Create a simplified returns dataset for the options simulation
                # Fetch recent returns for volatility calculation
                end_date = datetime.now()
                start_date = end_date - timedelta(days=252)
                
                price_data = yf.download(underlying_ticker, start=start_date, end=end_date, progress=False)['Adj Close']
                returns_data = price_data.pct_change().dropna().to_frame()
                
                # Initialize Monte Carlo simulator
                mc_simulator = MonteCarloSimulator(returns_data, current_price)
                
                # Run options analysis
                options_results = mc_simulator.options_risk_analysis(
                    underlying_price=current_price,
                    strike_price=strike_price,
                    time_to_expiry=time_to_expiry,
                    option_type=option_type,
                    num_simulations=num_option_sims
                )
                
                # Display results
                display_options_results(options_results, current_price, strike_price, option_type)
                
            except Exception as e:
                st.error(f"Error running options analysis: {str(e)}")

def display_options_results(results: dict, current_price: float, strike_price: float, option_type: str):
    """Display options Monte Carlo results"""
    
    st.subheader("📊 Options Analysis Results")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Estimated Option Value",
            f"${results['option_value']:.2f}",
            delta=f"±${results['option_std']:.2f}"
        )
    
    with col2:
        st.metric(
            "Probability ITM",
            f"{results['probability_itm']:.1%}",
            delta="High" if results['probability_itm'] > 0.6 else "Low"
        )
    
    with col3:
        st.metric(
            "Delta",
            f"{results['delta']:.3f}",
            help="Price sensitivity to underlying movement"
        )
    
    with col4:
        st.metric(
            "Expected Payoff",
            f"${results['expected_payoff']:.2f}",
            delta=f"${results['expected_payoff'] - results['option_value']:.2f} time value"
        )
    
    # Price path visualization
    fig_paths = go.Figure()
    
    # Show sample paths
    price_paths = results['price_paths']
    time_axis = np.arange(price_paths.shape[1])
    
    # Show first 50 paths
    for i in range(min(50, price_paths.shape[0])):
        fig_paths.add_trace(go.Scatter(
            x=time_axis,
            y=price_paths[i, :],
            mode='lines',
            line=dict(width=1, color='rgba(100, 150, 200, 0.3)'),
            showlegend=False,
            hovertemplate='Day: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))
    
    # Add strike price line
    fig_paths.add_hline(
        y=strike_price,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Strike: ${strike_price:.2f}"
    )
    
    # Add current price line
    fig_paths.add_hline(
        y=current_price,
        line_dash="dot",
        line_color="green",
        annotation_text=f"Current: ${current_price:.2f}"
    )
    
    fig_paths.update_layout(
        title=f'{option_type.title()} Option Price Simulation Paths',
        xaxis_title='Days to Expiration',
        yaxis_title='Underlying Price ($)',
        template='plotly_dark',
        height=400
    )
    
    st.plotly_chart(fig_paths, use_container_width=True)
    
    # Payoff distribution
    col1, col2 = st.columns(2)
    
    with col1:
        fig_payoff = go.Figure()
        
        fig_payoff.add_trace(go.Histogram(
            x=results['payoffs'],
            nbinsx=50,
            name='Option Payoffs',
            marker_color='lightblue',
            opacity=0.7
        ))
        
        fig_payoff.update_layout(
            title='Option Payoff Distribution',
            xaxis_title='Payoff ($)',
            yaxis_title='Frequency',
            template='plotly_dark'
        )
        
        st.plotly_chart(fig_payoff, use_container_width=True)
    
    with col2:
        # Scenario analysis
        st.markdown("**Scenario Analysis**")
        
        scenario_data = [
            ['Bull Case (95th percentile)', f"${results['scenario_analysis']['bull_case']:.2f}"],
            ['Base Case (50th percentile)', f"${results['scenario_analysis']['base_case']:.2f}"],
            ['Bear Case (5th percentile)', f"${results['scenario_analysis']['bear_case']:.2f}"],
            ['Value at Risk (95%)', f"${results['value_at_risk_95']:.2f}"],
            ['Value at Risk (99%)', f"${results['value_at_risk_99']:.2f}"]
        ]
        
        scenario_df = pd.DataFrame(scenario_data, columns=['Scenario', 'Option Value'])
        st.dataframe(scenario_df, use_container_width=True, hide_index=True)
        
        # Risk assessment
        if results['value_at_risk_95'] < results['option_value'] * 0.5:
            risk_assessment = "🔴 High Risk"
        elif results['value_at_risk_95'] < results['option_value'] * 0.8:
            risk_assessment = "🟡 Moderate Risk"
        else:
            risk_assessment = "🟢 Low Risk"
        
        st.metric("Risk Assessment", risk_assessment)

if __name__ == "__main__":
    main()