import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy.stats import norm, t
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.ui_components import apply_custom_css, create_metric_card, create_info_card
    from utils.auth import check_authentication
    apply_custom_css()
except ImportError:
    def create_metric_card(title, value, delta=None):
        return st.metric(title, value, delta)
    def create_info_card(title, content):
        return st.info(f"**{title}**\n\n{content}")
    def check_authentication():
        return True, None

class AdvancedRiskManager:
    """Professional-grade risk management and portfolio analytics"""
    
    def __init__(self):
        self.confidence_levels = [0.95, 0.99, 0.999]
        self.stress_scenarios = {
            '2008 Financial Crisis': -0.37,
            'COVID-19 Crash (2020)': -0.34,
            'Black Monday (1987)': -0.22,
            'Dot-com Crash (2000-2002)': -0.49,
            'Interest Rate Shock': -0.15
        }
        
    @st.cache_data(ttl=1800)  # Cache for 30 minutes
    def fetch_portfolio_data(_self, symbols, period="2y"):
        """Fetch historical data for portfolio analysis"""
        try:
            data = yf.download(symbols, period=period, progress=False)['Adj Close']
            if isinstance(data, pd.Series):
                data = data.to_frame(symbols[0])
            return data.dropna()
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return pd.DataFrame()
    
    def calculate_portfolio_returns(self, prices, weights):
        """Calculate portfolio returns given prices and weights"""
        returns = prices.pct_change().dropna()
        portfolio_returns = (returns * weights).sum(axis=1)
        return portfolio_returns
    
    def calculate_var(self, returns, confidence_level=0.95, method='historical'):
        """Calculate Value at Risk using different methods"""
        if len(returns) == 0:
            return 0
            
        if method == 'historical':
            return np.percentile(returns, (1 - confidence_level) * 100)
        
        elif method == 'parametric':
            mu = returns.mean()
            sigma = returns.std()
            var = norm.ppf(1 - confidence_level, mu, sigma)
            return var
            
        elif method == 'monte_carlo':
            # Monte Carlo simulation
            mu = returns.mean()
            sigma = returns.std()
            n_simulations = 10000
            simulated_returns = np.random.normal(mu, sigma, n_simulations)
            return np.percentile(simulated_returns, (1 - confidence_level) * 100)
        
        return 0
    
    def calculate_cvar(self, returns, confidence_level=0.95):
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        if len(returns) == 0:
            return 0
        var = self.calculate_var(returns, confidence_level)
        cvar = returns[returns <= var].mean()
        return cvar if not np.isnan(cvar) else 0
    
    def calculate_maximum_drawdown(self, returns):
        """Calculate maximum drawdown and drawdown duration"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        max_dd = drawdown.min()
        
        # Calculate drawdown duration
        dd_duration = 0
        current_dd_days = 0
        max_dd_days = 0
        
        for dd in drawdown:
            if dd < 0:
                current_dd_days += 1
                max_dd_days = max(max_dd_days, current_dd_days)
            else:
                current_dd_days = 0
        
        return max_dd, max_dd_days, drawdown
    
    def monte_carlo_simulation(self, returns, weights, n_simulations=10000, time_horizon=252):
        """Monte Carlo portfolio simulation"""
        n_assets = len(weights)
        
        # Calculate portfolio statistics
        portfolio_returns = self.calculate_portfolio_returns(
            pd.DataFrame(np.random.multivariate_normal(
                returns.mean(), 
                np.cov(returns.T), 
                (time_horizon, n_simulations)
            )), weights
        )
        
        # Simulate portfolio paths
        portfolio_paths = []
        initial_value = 100000  # $100k initial portfolio
        
        for i in range(n_simulations):
            path = [initial_value]
            for j in range(time_horizon):
                path.append(path[-1] * (1 + portfolio_returns.iloc[j] if j < len(portfolio_returns) else 0))
            portfolio_paths.append(path)
        
        return np.array(portfolio_paths)
    
    def stress_testing(self, prices, weights, scenarios):
        """Perform stress testing under various market scenarios"""
        results = {}
        
        for scenario_name, shock in scenarios.items():
            # Apply shock to all assets
            stressed_prices = prices * (1 + shock)
            
            # Calculate stressed portfolio value
            current_value = (prices.iloc[-1] * weights).sum()
            stressed_value = (stressed_prices.iloc[-1] * weights).sum()
            
            portfolio_impact = (stressed_value - current_value) / current_value
            
            results[scenario_name] = {
                'shock': shock,
                'portfolio_impact': portfolio_impact,
                'dollar_impact': portfolio_impact * 100000  # Assuming $100k portfolio
            }
        
        return results
    
    def calculate_risk_metrics(self, returns):
        """Calculate comprehensive risk metrics"""
        if len(returns) == 0:
            return {}
            
        metrics = {
            'volatility': returns.std() * np.sqrt(252),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'var_95': self.calculate_var(returns, 0.95),
            'var_99': self.calculate_var(returns, 0.99),
            'cvar_95': self.calculate_cvar(returns, 0.95),
            'cvar_99': self.calculate_cvar(returns, 0.99),
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        }
        
        max_dd, dd_duration, _ = self.calculate_maximum_drawdown(returns)
        metrics['max_drawdown'] = max_dd
        metrics['drawdown_duration'] = dd_duration
        
        return metrics
    
    def create_var_visualization(self, returns, confidence_levels):
        """Create VaR visualization"""
        fig = go.Figure()
        
        # Histogram of returns
        fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=50,
            name='Return Distribution',
            opacity=0.7,
            marker_color='rgba(78, 205, 196, 0.7)'
        ))
        
        # VaR lines
        colors = ['red', 'orange', 'yellow']
        for i, conf in enumerate(confidence_levels):
            var_value = self.calculate_var(returns, conf)
            fig.add_vline(
                x=var_value,
                line_dash="dash",
                line_color=colors[i],
                annotation_text=f"VaR {conf*100:.0f}%: {var_value:.3f}"
            )
        
        fig.update_layout(
            title='Value at Risk Analysis',
            xaxis_title='Daily Returns',
            yaxis_title='Frequency',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400
        )
        
        return fig
    
    def create_monte_carlo_visualization(self, portfolio_paths):
        """Create Monte Carlo simulation visualization"""
        fig = go.Figure()
        
        # Plot sample paths
        n_paths_to_show = min(100, len(portfolio_paths))
        for i in range(n_paths_to_show):
            fig.add_trace(go.Scatter(
                x=list(range(len(portfolio_paths[i]))),
                y=portfolio_paths[i],
                mode='lines',
                line=dict(color='rgba(78, 205, 196, 0.1)', width=1),
                showlegend=False,
                hovertemplate='Day: %{x}<br>Portfolio Value: $%{y:,.0f}<extra></extra>'
            ))
        
        # Add percentiles
        percentiles = [5, 50, 95]
        colors = ['red', 'white', 'green']
        names = ['5th Percentile', 'Median', '95th Percentile']
        
        for p, color, name in zip(percentiles, colors, names):
            percentile_path = np.percentile(portfolio_paths, p, axis=0)
            fig.add_trace(go.Scatter(
                x=list(range(len(percentile_path))),
                y=percentile_path,
                mode='lines',
                line=dict(color=color, width=3),
                name=name,
                hovertemplate=f'{name}<br>Day: %{{x}}<br>Portfolio Value: $%{{y:,.0f}}<extra></extra>'
            ))
        
        fig.update_layout(
            title='Monte Carlo Portfolio Simulation (1 Year)',
            xaxis_title='Trading Days',
            yaxis_title='Portfolio Value ($)',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=500
        )
        
        return fig
    
    def create_drawdown_chart(self, drawdown):
        """Create drawdown visualization"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown * 100,
            mode='lines',
            fill='tonexty',
            line=dict(color='red', width=2),
            name='Drawdown',
            hovertemplate='Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
        ))
        
        fig.add_hline(y=0, line_dash="solid", line_color="gray")
        
        fig.update_layout(
            title='Portfolio Drawdown Analysis',
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400
        )
        
        return fig

def main():
    # Skip authentication for now
    pass
        
    st.title("⚡ Advanced Risk Management")
    st.markdown("**Professional portfolio risk analytics, stress testing, and Monte Carlo simulations**")
    
    # Initialize risk manager
    risk_manager = AdvancedRiskManager()
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Risk Analysis Controls")
        
        # Portfolio input
        st.markdown("**Portfolio Configuration**")
        symbols_input = st.text_area(
            "Stock Symbols (one per line)",
            value="AAPL\nMSFT\nGOOGL\nAMZN\nTSLA",
            height=100
        )
        
        symbols = [s.strip().upper() for s in symbols_input.split('\n') if s.strip()]
        
        # Portfolio weights
        st.markdown("**Portfolio Weights**")
        weights = []
        for symbol in symbols:
            weight = st.slider(f"{symbol} Weight", 0.0, 1.0, 1.0/len(symbols), 0.01)
            weights.append(weight)
        
        # Normalize weights
        if sum(weights) > 0:
            weights = [w / sum(weights) for w in weights]
        
        # Analysis parameters
        st.markdown("**Analysis Parameters**")
        portfolio_value = st.number_input(
            "Portfolio Value ($)", min_value=1000, value=100000, step=1000
        )
        
        analysis_period = st.selectbox(
            "Analysis Period", ["1y", "2y", "3y", "5y"]
        )
        
        # Refresh data
        if st.button("🔄 Refresh Analysis", type="primary"):
            st.cache_data.clear()
            st.rerun()
    
    if not symbols:
        st.warning("Please enter at least one stock symbol")
        return
    
    # Main content tabs
    risk_metrics_tab, var_analysis_tab, monte_carlo_tab, stress_test_tab = st.tabs([
        "📊 Risk Metrics", 
        "📉 VaR Analysis", 
        "🎲 Monte Carlo", 
        "⚠️ Stress Testing"
    ])
    
    # Fetch portfolio data
    with st.spinner("Fetching portfolio data..."):
        prices = risk_manager.fetch_portfolio_data(symbols, analysis_period)
    
    if prices.empty:
        st.error("Unable to fetch portfolio data")
        return
    
    # Calculate portfolio returns
    portfolio_returns = risk_manager.calculate_portfolio_returns(prices, weights)
    
    with risk_metrics_tab:
        st.markdown("### Portfolio Risk Metrics")
        
        # Portfolio summary
        current_prices = prices.iloc[-1]
        portfolio_value_current = (current_prices * weights * portfolio_value).sum()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            create_metric_card("Portfolio Value", f"${portfolio_value:,}")
        with col2:
            daily_return = portfolio_returns.iloc[-1] if len(portfolio_returns) > 0 else 0
            create_metric_card("Daily Return", f"{daily_return:.2%}")
        with col3:
            ytd_return = portfolio_returns.sum() if len(portfolio_returns) > 0 else 0
            create_metric_card("Period Return", f"{ytd_return:.2%}")
        with col4:
            n_positions = len([w for w in weights if w > 0])
            create_metric_card("Active Positions", str(n_positions))
        
        # Risk metrics
        risk_metrics = risk_manager.calculate_risk_metrics(portfolio_returns)
        
        st.markdown("#### Comprehensive Risk Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Return & Volatility**")
            create_metric_card("Annual Volatility", f"{risk_metrics.get('volatility', 0):.2%}")
            create_metric_card("Sharpe Ratio", f"{risk_metrics.get('sharpe_ratio', 0):.2f}")
            create_metric_card("Max Drawdown", f"{risk_metrics.get('max_drawdown', 0):.2%}")
        
        with col2:
            st.markdown("**Value at Risk**")
            create_metric_card("VaR (95%)", f"{risk_metrics.get('var_95', 0):.2%}")
            create_metric_card("VaR (99%)", f"{risk_metrics.get('var_99', 0):.2%}")
            create_metric_card("CVaR (95%)", f"{risk_metrics.get('cvar_95', 0):.2%}")
        
        with col3:
            st.markdown("**Distribution Properties**")
            create_metric_card("Skewness", f"{risk_metrics.get('skewness', 0):.2f}")
            create_metric_card("Kurtosis", f"{risk_metrics.get('kurtosis', 0):.2f}")
            create_metric_card("DD Duration", f"{risk_metrics.get('drawdown_duration', 0)} days")
        
        # Portfolio composition
        st.markdown("#### Portfolio Composition")
        
        composition_data = pd.DataFrame({
            'Symbol': symbols,
            'Weight': [f"{w:.1%}" for w in weights],
            'Value': [f"${w * portfolio_value:,.0f}" for w in weights],
            'Current Price': [f"${current_prices[s]:.2f}" for s in symbols]
        })
        
        st.dataframe(composition_data, use_container_width=True, hide_index=True)
        
        # Drawdown chart
        if len(portfolio_returns) > 0:
            _, _, drawdown = risk_manager.calculate_maximum_drawdown(portfolio_returns)
            drawdown_fig = risk_manager.create_drawdown_chart(drawdown)
            st.plotly_chart(drawdown_fig, use_container_width=True)
    
    with var_analysis_tab:
        st.markdown("### Value at Risk Analysis")
        
        if len(portfolio_returns) > 0:
            # VaR calculations
            st.markdown("#### VaR by Method")
            
            methods = ['historical', 'parametric', 'monte_carlo']
            confidence_levels = [0.95, 0.99, 0.999]
            
            var_results = []
            for method in methods:
                for conf in confidence_levels:
                    var_val = risk_manager.calculate_var(portfolio_returns, conf, method)
                    cvar_val = risk_manager.calculate_cvar(portfolio_returns, conf)
                    
                    var_results.append({
                        'Method': method.title(),
                        'Confidence': f"{conf:.1%}",
                        'VaR': f"{var_val:.3%}",
                        'CVaR': f"{cvar_val:.3%}",
                        'Dollar VaR': f"${var_val * portfolio_value:,.0f}",
                        'Dollar CVaR': f"${cvar_val * portfolio_value:,.0f}"
                    })
            
            var_df = pd.DataFrame(var_results)
            st.dataframe(var_df, use_container_width=True, hide_index=True)
            
            # VaR visualization
            st.markdown("#### VaR Distribution Analysis")
            var_fig = risk_manager.create_var_visualization(portfolio_returns, confidence_levels)
            st.plotly_chart(var_fig, use_container_width=True)
            
            # Risk decomposition
            st.markdown("#### Risk Decomposition by Asset")
            
            individual_vars = []
            for i, symbol in enumerate(symbols):
                if symbol in prices.columns:
                    asset_returns = prices[symbol].pct_change().dropna()
                    asset_var = risk_manager.calculate_var(asset_returns, 0.95)
                    contribution = weights[i] * asset_var
                    
                    individual_vars.append({
                        'Asset': symbol,
                        'Weight': f"{weights[i]:.1%}",
                        'Asset VaR (95%)': f"{asset_var:.3%}",
                        'Risk Contribution': f"{contribution:.3%}",
                        'Dollar Risk': f"${contribution * portfolio_value:,.0f}"
                    })
            
            risk_decomp_df = pd.DataFrame(individual_vars)
            st.dataframe(risk_decomp_df, use_container_width=True, hide_index=True)
        
        else:
            st.warning("Insufficient data for VaR analysis")
    
    with monte_carlo_tab:
        st.markdown("### Monte Carlo Simulation")
        
        if len(portfolio_returns) > 0:
            # Simulation parameters
            col1, col2 = st.columns(2)
            
            with col1:
                n_simulations = st.slider("Number of Simulations", 1000, 10000, 5000)
                time_horizon = st.slider("Time Horizon (Days)", 30, 252, 252)
            
            with col2:
                confidence_interval = st.selectbox(
                    "Confidence Interval", ["90%", "95%", "99%"]
                )
                
            if st.button("🚀 Run Monte Carlo Simulation"):
                with st.spinner("Running Monte Carlo simulation..."):
                    # Generate return scenarios
                    mu = portfolio_returns.mean()
                    sigma = portfolio_returns.std()
                    
                    # Simulate portfolio paths
                    np.random.seed(42)  # For reproducibility
                    simulated_returns = np.random.normal(mu, sigma, (time_horizon, n_simulations))
                    
                    # Calculate portfolio paths
                    portfolio_paths = []
                    for i in range(n_simulations):
                        path = [portfolio_value]
                        for j in range(time_horizon):
                            path.append(path[-1] * (1 + simulated_returns[j, i]))
                        portfolio_paths.append(path)
                    
                    portfolio_paths = np.array(portfolio_paths)
                    
                    # Simulation results
                    final_values = portfolio_paths[:, -1]
                    
                    st.markdown("#### Simulation Results")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        mean_final = np.mean(final_values)
                        create_metric_card("Mean Final Value", f"${mean_final:,.0f}")
                    
                    with col2:
                        median_final = np.median(final_values)
                        create_metric_card("Median Final Value", f"${median_final:,.0f}")
                    
                    with col3:
                        prob_loss = np.mean(final_values < portfolio_value) * 100
                        create_metric_card("Probability of Loss", f"{prob_loss:.1f}%")
                    
                    with col4:
                        percentile_5 = np.percentile(final_values, 5)
                        create_metric_card("5th Percentile", f"${percentile_5:,.0f}")
                    
                    # Monte Carlo visualization
                    mc_fig = risk_manager.create_monte_carlo_visualization(portfolio_paths)
                    st.plotly_chart(mc_fig, use_container_width=True)
                    
                    # Final value distribution
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=final_values,
                        nbinsx=50,
                        name='Final Portfolio Value',
                        marker_color='rgba(78, 205, 196, 0.7)'
                    ))
                    
                    fig.add_vline(x=portfolio_value, line_dash="dash", line_color="white",
                                 annotation_text="Initial Value")
                    
                    fig.update_layout(
                        title='Distribution of Final Portfolio Values',
                        xaxis_title='Portfolio Value ($)',
                        yaxis_title='Frequency',
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("Insufficient data for Monte Carlo simulation")
    
    with stress_test_tab:
        st.markdown("### Stress Testing & Scenario Analysis")
        
        if len(portfolio_returns) > 0:
            # Perform stress testing
            stress_results = risk_manager.stress_testing(
                prices, weights, risk_manager.stress_scenarios
            )
            
            st.markdown("#### Historical Stress Scenarios")
            
            # Create stress test results table
            stress_data = []
            for scenario, results in stress_results.items():
                stress_data.append({
                    'Scenario': scenario,
                    'Market Shock': f"{results['shock']:.1%}",
                    'Portfolio Impact': f"{results['portfolio_impact']:.2%}",
                    'Dollar Impact': f"${results['dollar_impact']:,.0f}"
                })
            
            stress_df = pd.DataFrame(stress_data)
            st.dataframe(stress_df, use_container_width=True, hide_index=True)
            
            # Stress test visualization
            scenario_names = list(stress_results.keys())
            portfolio_impacts = [results['portfolio_impact'] * 100 
                               for results in stress_results.values()]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=scenario_names,
                y=portfolio_impacts,
                marker_color=['red' if x < 0 else 'green' for x in portfolio_impacts],
                text=[f"{x:.1f}%" for x in portfolio_impacts],
                textposition='outside'
            ))
            
            fig.update_layout(
                title='Stress Test Results',
                xaxis_title='Stress Scenario',
                yaxis_title='Portfolio Impact (%)',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Custom stress scenario
            st.markdown("#### Custom Stress Scenario")
            
            col1, col2 = st.columns(2)
            
            with col1:
                custom_shock = st.slider(
                    "Market Shock (%)", -50, 50, -20
                ) / 100
                
            with col2:
                if st.button("Apply Custom Shock"):
                    custom_results = risk_manager.stress_testing(
                        prices, weights, {'Custom Scenario': custom_shock}
                    )
                    
                    impact = custom_results['Custom Scenario']['portfolio_impact']
                    dollar_impact = custom_results['Custom Scenario']['dollar_impact']
                    
                    st.success(f"Custom scenario impact: {impact:.2%} (${dollar_impact:,.0f})")
            
            # Correlation analysis during stress
            st.markdown("#### Asset Correlation During Stress")
            
            # Calculate rolling correlations
            returns_matrix = prices.pct_change().dropna()
            
            if len(returns_matrix.columns) > 1:
                correlation_matrix = returns_matrix.corr()
                
                fig = px.imshow(
                    correlation_matrix,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    title="Asset Correlation Matrix"
                )
                
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("Insufficient data for stress testing")
    
    # Educational content
    with st.expander("📚 Risk Management Guide"):
        st.markdown("""
        **Key Risk Metrics Explained:**
        
        **Value at Risk (VaR):**
        - Maximum expected loss over a specific time period at a given confidence level
        - 95% VaR means 5% chance of losing more than the VaR amount
        
        **Conditional VaR (CVaR/Expected Shortfall):**
        - Expected loss given that loss exceeds VaR
        - Provides insight into tail risk beyond VaR
        
        **Maximum Drawdown:**
        - Largest peak-to-trough decline in portfolio value
        - Measures worst-case historical performance
        
        **Stress Testing:**
        - Evaluates portfolio performance under extreme market conditions
        - Uses historical crisis scenarios to assess vulnerability
        
        **Monte Carlo Simulation:**
        - Projects thousands of possible future portfolio outcomes
        - Based on historical return and volatility patterns
        
        **Risk Management Best Practices:**
        - Diversify across asset classes and sectors
        - Set position size limits based on risk capacity
        - Use stop-loss orders and hedge positions when appropriate
        - Regular portfolio rebalancing and risk monitoring
        - Understand correlation changes during market stress
        """)

if __name__ == "__main__":
    main()