import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.ui_components import apply_custom_css, create_metric_card, create_info_card
    apply_custom_css()
except ImportError:
    def create_metric_card(title, value, delta=None):
        return st.metric(title, value, delta)
    def create_info_card(title, content):
        return st.info(f"**{title}**\n\n{content}")

class PortfolioRebalancer:
    """Dynamic Portfolio Rebalancing Assistant with multiple rebalancing strategies"""
    
    def __init__(self):
        self.rebalancing_methods = {
            'Threshold': 'Rebalance when any asset deviates by specified threshold',
            'Calendar': 'Rebalance at regular time intervals',
            'Volatility': 'Rebalance when portfolio volatility exceeds target',
            'Momentum': 'Rebalance based on momentum signals',
            'Risk Parity': 'Maintain equal risk contribution from all assets'
        }
        
    @st.cache_data(ttl=1800)  # Cache for 30 minutes
    def fetch_portfolio_data(_self, symbols, period="1y"):
        """Fetch historical price data for portfolio"""
        try:
            data = yf.download(symbols, period=period, progress=False)['Adj Close']
            if isinstance(data, pd.Series):
                data = data.to_frame(symbols[0])
            return data.dropna()
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return pd.DataFrame()
    
    def calculate_current_weights(self, current_prices, shares):
        """Calculate current portfolio weights"""
        portfolio_values = current_prices * shares
        total_value = portfolio_values.sum()
        current_weights = portfolio_values / total_value if total_value > 0 else portfolio_values * 0
        return current_weights, total_value
    
    def calculate_drift(self, current_weights, target_weights):
        """Calculate weight drift from target allocation"""
        drift = current_weights - target_weights
        absolute_drift = np.abs(drift)
        max_drift = absolute_drift.max()
        total_drift = absolute_drift.sum()
        return drift, max_drift, total_drift
    
    def threshold_rebalancing_check(self, current_weights, target_weights, threshold=0.05):
        """Check if rebalancing is needed based on threshold"""
        drift, max_drift, total_drift = self.calculate_drift(current_weights, target_weights)
        needs_rebalancing = max_drift > threshold
        
        rebalancing_trades = {}
        if needs_rebalancing:
            total_value = 100000  # Assume $100k portfolio for calculation
            for symbol in current_weights.index:
                current_value = current_weights[symbol] * total_value
                target_value = target_weights[symbol] * total_value
                trade_amount = target_value - current_value
                
                if abs(trade_amount) > total_value * 0.01:  # Only trade if >1% of portfolio
                    rebalancing_trades[symbol] = {
                        'current_weight': current_weights[symbol],
                        'target_weight': target_weights[symbol],
                        'drift': drift[symbol],
                        'trade_amount': trade_amount,
                        'action': 'BUY' if trade_amount > 0 else 'SELL'
                    }
        
        return needs_rebalancing, max_drift, rebalancing_trades
    
    def risk_parity_optimization(self, returns_data, target_volatility=0.15):
        """Optimize portfolio for risk parity (equal risk contribution)"""
        try:
            # Calculate covariance matrix
            cov_matrix = returns_data.cov() * 252  # Annualized
            n_assets = len(returns_data.columns)
            
            # Objective function: minimize sum of squared differences in risk contribution
            def risk_parity_objective(weights):
                weights = np.array(weights)
                portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                
                # Calculate marginal risk contributions
                marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
                contrib = weights * marginal_contrib
                
                # Target equal risk contribution
                target_contrib = portfolio_vol / n_assets
                return np.sum((contrib - target_contrib) ** 2)
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # Weights sum to 1
            ]
            
            # Bounds (no short selling, max 50% in any single asset)
            bounds = tuple((0.02, 0.5) for _ in range(n_assets))
            
            # Initial guess (equal weights)
            initial_guess = np.array([1.0 / n_assets] * n_assets)
            
            # Optimize
            result = minimize(
                risk_parity_objective,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                optimal_weights = result.x
                return dict(zip(returns_data.columns, optimal_weights))
            else:
                # Fall back to equal weights
                return dict(zip(returns_data.columns, [1.0/n_assets] * n_assets))
                
        except Exception:
            # Fall back to equal weights
            n_assets = len(returns_data.columns)
            return dict(zip(returns_data.columns, [1.0/n_assets] * n_assets))
    
    def momentum_based_rebalancing(self, price_data, lookback_days=20):
        """Calculate momentum-based target weights"""
        returns = price_data.pct_change().dropna()
        
        # Calculate momentum scores (average return over lookback period)
        momentum_scores = returns.rolling(window=lookback_days).mean().iloc[-1]
        
        # Convert to positive weights (add constant to avoid negative weights)
        adjusted_scores = momentum_scores - momentum_scores.min() + 0.1
        
        # Normalize to sum to 1
        momentum_weights = adjusted_scores / adjusted_scores.sum()
        
        return momentum_weights.to_dict()
    
    def volatility_targeting(self, returns_data, target_vol=0.15):
        """Calculate position sizes based on volatility targeting"""
        # Calculate individual asset volatilities
        asset_vols = returns_data.std() * np.sqrt(252)
        
        # Inverse volatility weighting
        inv_vol_weights = (1 / asset_vols) / (1 / asset_vols).sum()
        
        # Scale to target portfolio volatility
        portfolio_vol = np.sqrt(np.dot(inv_vol_weights.T, np.dot(returns_data.cov() * 252, inv_vol_weights)))
        vol_scalar = target_vol / portfolio_vol if portfolio_vol > 0 else 1
        
        # Adjust weights
        adjusted_weights = inv_vol_weights * vol_scalar
        
        # Ensure weights sum to 1
        adjusted_weights = adjusted_weights / adjusted_weights.sum()
        
        return adjusted_weights.to_dict()
    
    def calculate_rebalancing_costs(self, trades, commission_rate=0.001):
        """Calculate transaction costs for rebalancing"""
        total_cost = 0
        cost_breakdown = {}
        
        for symbol, trade_info in trades.items():
            trade_amount = abs(trade_info['trade_amount'])
            cost = trade_amount * commission_rate
            total_cost += cost
            cost_breakdown[symbol] = cost
        
        return total_cost, cost_breakdown
    
    def backtest_rebalancing_strategy(self, price_data, target_weights, method='threshold', **kwargs):
        """Backtest a rebalancing strategy"""
        returns = price_data.pct_change().dropna()
        
        # Initialize portfolio
        portfolio_value = [100000]  # Start with $100k
        dates = returns.index
        rebalancing_dates = []
        transaction_costs = []
        
        current_weights = pd.Series(target_weights)
        
        for i, date in enumerate(dates):
            if i == 0:
                continue
                
            # Calculate portfolio return
            daily_return = (returns.loc[date] * current_weights).sum()
            new_value = portfolio_value[-1] * (1 + daily_return)
            portfolio_value.append(new_value)
            
            # Check if rebalancing is needed
            needs_rebalancing = False
            
            if method == 'threshold':
                threshold = kwargs.get('threshold', 0.05)
                # Update current weights based on price movements
                price_changes = (1 + returns.loc[date])
                current_weights = current_weights * price_changes
                current_weights = current_weights / current_weights.sum()
                
                _, max_drift, _ = self.calculate_drift(current_weights, pd.Series(target_weights))
                needs_rebalancing = max_drift > threshold
                
            elif method == 'calendar':
                rebalance_freq = kwargs.get('frequency', 30)  # days
                if len(rebalancing_dates) == 0 or (date - rebalancing_dates[-1]).days >= rebalance_freq:
                    needs_rebalancing = True
            
            # Rebalance if needed
            if needs_rebalancing:
                rebalancing_dates.append(date)
                current_weights = pd.Series(target_weights)
                # Add transaction cost (simplified)
                cost = new_value * 0.002  # 0.2% total cost
                transaction_costs.append(cost)
                portfolio_value[-1] -= cost
        
        # Calculate performance metrics
        portfolio_returns = pd.Series(portfolio_value).pct_change().dropna()
        total_return = (portfolio_value[-1] - portfolio_value[0]) / portfolio_value[0]
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252) if portfolio_returns.std() > 0 else 0
        
        return {
            'portfolio_values': portfolio_value,
            'dates': dates,
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'rebalancing_dates': rebalancing_dates,
            'transaction_costs': sum(transaction_costs),
            'n_rebalances': len(rebalancing_dates)
        }
    
    def create_drift_visualization(self, current_weights, target_weights):
        """Create weight drift visualization"""
        symbols = current_weights.index
        
        fig = go.Figure()
        
        # Current weights
        fig.add_trace(go.Bar(
            x=symbols,
            y=current_weights * 100,
            name='Current Weights',
            marker_color='rgba(78, 205, 196, 0.7)'
        ))
        
        # Target weights
        fig.add_trace(go.Bar(
            x=symbols,
            y=[target_weights.get(symbol, 0) * 100 for symbol in symbols],
            name='Target Weights',
            marker_color='rgba(255, 107, 107, 0.7)'
        ))
        
        fig.update_layout(
            title='Portfolio Weight Drift Analysis',
            xaxis_title='Assets',
            yaxis_title='Weight (%)',
            barmode='group',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400
        )
        
        return fig
    
    def create_rebalancing_calendar(self, backtest_results):
        """Create rebalancing calendar visualization"""
        rebalancing_dates = backtest_results['rebalancing_dates']
        
        if not rebalancing_dates:
            return go.Figure()
        
        # Create timeline
        fig = go.Figure()
        
        for i, date in enumerate(rebalancing_dates):
            fig.add_trace(go.Scatter(
                x=[date],
                y=[1],
                mode='markers',
                marker=dict(
                    size=15,
                    color='rgba(255, 107, 107, 0.8)',
                    symbol='diamond'
                ),
                name=f'Rebalance {i+1}',
                hovertemplate=f'Rebalancing Date: {date.strftime("%Y-%m-%d")}<extra></extra>'
            ))
        
        fig.update_layout(
            title='Rebalancing Timeline',
            xaxis_title='Date',
            yaxis=dict(visible=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=200,
            showlegend=False
        )
        
        return fig

def main():
    st.title("🔄 Dynamic Portfolio Rebalancing Assistant")
    st.markdown("**Intelligent portfolio rebalancing with multiple strategies and automated alerts**")
    
    # Initialize rebalancer
    rebalancer = PortfolioRebalancer()
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Rebalancing Controls")
        
        # Portfolio configuration
        st.markdown("**Portfolio Setup**")
        symbols_input = st.text_area(
            "Assets (one per line)",
            value="AAPL\nMSFT\nGOOGL\nAMZN\nVTI",
            height=100
        )
        
        symbols = [s.strip().upper() for s in symbols_input.split('\n') if s.strip()]
        
        # Target weights
        st.markdown("**Target Allocation**")
        target_weights = {}
        for symbol in symbols:
            weight = st.slider(f"{symbol} Target %", 0, 100, 20, 1) / 100
            target_weights[symbol] = weight
        
        # Normalize weights
        total_weight = sum(target_weights.values())
        if total_weight > 0:
            target_weights = {k: v/total_weight for k, v in target_weights.items()}
        
        # Rebalancing method
        st.markdown("**Rebalancing Strategy**")
        rebalancing_method = st.selectbox(
            "Method",
            list(rebalancer.rebalancing_methods.keys())
        )
        
        # Method-specific parameters
        if rebalancing_method == 'Threshold':
            threshold = st.slider("Drift Threshold (%)", 1, 20, 5) / 100
        elif rebalancing_method == 'Calendar':
            frequency = st.selectbox("Frequency", [30, 60, 90, 180])
        elif rebalancing_method == 'Volatility':
            target_vol = st.slider("Target Volatility (%)", 5, 30, 15) / 100
        
        # Portfolio value
        portfolio_value = st.number_input(
            "Current Portfolio Value ($)",
            min_value=1000,
            value=100000,
            step=1000
        )
        
        # Refresh data
        if st.button("🔄 Refresh Analysis", type="primary"):
            st.cache_data.clear()
            st.rerun()
    
    if not symbols:
        st.warning("Please enter at least one asset symbol")
        return
    
    # Main content tabs
    current_tab, strategy_tab, backtest_tab, alerts_tab = st.tabs([
        "📊 Current Analysis",
        "🎯 Strategy Optimization", 
        "📈 Backtest Results",
        "🔔 Rebalancing Alerts"
    ])
    
    # Fetch portfolio data
    with st.spinner("Fetching portfolio data..."):
        price_data = rebalancer.fetch_portfolio_data(symbols)
    
    if price_data.empty:
        st.error("Unable to fetch portfolio data")
        return
    
    with current_tab:
        st.markdown("### Current Portfolio Analysis")
        
        # Simulate current holdings (for demonstration)
        current_prices = price_data.iloc[-1]
        
        # Generate sample current weights (with some drift)
        np.random.seed(42)
        drift_factors = np.random.normal(1, 0.1, len(symbols))
        simulated_current_weights = pd.Series([target_weights[s] * f for s, f in zip(symbols, drift_factors)], index=symbols)
        simulated_current_weights = simulated_current_weights / simulated_current_weights.sum()
        
        # Calculate drift
        drift, max_drift, total_drift = rebalancer.calculate_drift(simulated_current_weights, pd.Series(target_weights))
        
        # Portfolio metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            create_metric_card("Portfolio Value", f"${portfolio_value:,}")
        with col2:
            create_metric_card("Max Weight Drift", f"{max_drift:.2%}")
        with col3:
            create_metric_card("Total Drift", f"{total_drift:.2%}")
        with col4:
            needs_rebalancing = max_drift > (threshold if rebalancing_method == 'Threshold' else 0.05)
            status = "Yes" if needs_rebalancing else "No"
            color = "red" if needs_rebalancing else "green"
            create_metric_card("Needs Rebalancing", status)
        
        # Weight drift visualization
        drift_fig = rebalancer.create_drift_visualization(simulated_current_weights, target_weights)
        st.plotly_chart(drift_fig, use_container_width=True)
        
        # Current vs Target allocation table
        st.markdown("#### Allocation Analysis")
        
        allocation_data = []
        for symbol in symbols:
            current_weight = simulated_current_weights[symbol]
            target_weight = target_weights[symbol]
            drift_val = drift[symbol]
            current_value = current_weight * portfolio_value
            target_value = target_weight * portfolio_value
            trade_amount = target_value - current_value
            
            allocation_data.append({
                'Asset': symbol,
                'Current %': f"{current_weight:.2%}",
                'Target %': f"{target_weight:.2%}",
                'Drift': f"{drift_val:+.2%}",
                'Current Value': f"${current_value:,.0f}",
                'Target Value': f"${target_value:,.0f}",
                'Trade Amount': f"${trade_amount:+,.0f}",
                'Action': 'BUY' if trade_amount > 0 else 'SELL' if trade_amount < 0 else 'HOLD'
            })
        
        allocation_df = pd.DataFrame(allocation_data)
        st.dataframe(allocation_df, use_container_width=True, hide_index=True)
        
        # Rebalancing recommendation
        if needs_rebalancing:
            st.error("🚨 Portfolio rebalancing recommended!")
            
            # Calculate rebalancing trades
            needs_reb, max_d, trades = rebalancer.threshold_rebalancing_check(
                simulated_current_weights, pd.Series(target_weights), 
                threshold if rebalancing_method == 'Threshold' else 0.05
            )
            
            if trades:
                st.markdown("#### Recommended Trades")
                
                trade_data = []
                for symbol, trade_info in trades.items():
                    trade_data.append({
                        'Symbol': symbol,
                        'Action': trade_info['action'],
                        'Amount': f"${abs(trade_info['trade_amount']):,.0f}",
                        'Reason': f"Drift: {trade_info['drift']:+.2%}"
                    })
                
                trade_df = pd.DataFrame(trade_data)
                st.dataframe(trade_df, use_container_width=True, hide_index=True)
                
                # Transaction costs
                total_cost, cost_breakdown = rebalancer.calculate_rebalancing_costs(trades)
                st.info(f"Estimated transaction costs: ${total_cost:,.2f}")
        
        else:
            st.success("✅ Portfolio is well-balanced - no rebalancing needed")
    
    with strategy_tab:
        st.markdown("### Strategy Optimization")
        
        # Calculate returns for optimization
        returns = price_data.pct_change().dropna()
        
        if len(returns) > 0:
            # Show different rebalancing strategies
            st.markdown("#### Alternative Allocation Strategies")
            
            strategy_results = {}
            
            # Risk Parity
            with st.spinner("Calculating Risk Parity allocation..."):
                risk_parity_weights = rebalancer.risk_parity_optimization(returns)
                strategy_results['Risk Parity'] = risk_parity_weights
            
            # Momentum-based
            momentum_weights = rebalancer.momentum_based_rebalancing(price_data)
            strategy_results['Momentum'] = momentum_weights
            
            # Volatility targeting
            vol_target_weights = rebalancer.volatility_targeting(returns)
            strategy_results['Vol Targeting'] = vol_target_weights
            
            # Display strategies comparison
            strategy_comparison = pd.DataFrame(strategy_results).T
            strategy_comparison = strategy_comparison.fillna(0)
            
            # Format as percentages
            strategy_display = strategy_comparison.applymap(lambda x: f"{x:.1%}")
            st.dataframe(strategy_display, use_container_width=True)
            
            # Strategy comparison chart
            fig = go.Figure()
            
            for strategy in strategy_results.keys():
                fig.add_trace(go.Bar(
                    x=symbols,
                    y=[strategy_results[strategy].get(s, 0) * 100 for s in symbols],
                    name=strategy,
                    opacity=0.8
                ))
            
            fig.update_layout(
                title='Strategy Allocation Comparison',
                xaxis_title='Assets',
                yaxis_title='Weight (%)',
                barmode='group',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Strategy selector
            selected_strategy = st.selectbox(
                "Select Strategy to Apply",
                ['Current Target'] + list(strategy_results.keys())
            )
            
            if selected_strategy != 'Current Target':
                if st.button(f"Apply {selected_strategy} Strategy"):
                    # Update target weights (in a real app, this would persist)
                    st.success(f"✅ {selected_strategy} strategy applied!")
                    st.info("Note: In a real application, this would update your target allocation.")
    
    with backtest_tab:
        st.markdown("### Rebalancing Strategy Backtests")
        
        if len(price_data) > 50:  # Need sufficient data
            # Backtest parameters
            col1, col2 = st.columns(2)
            
            with col1:
                backtest_method = st.selectbox(
                    "Backtest Method",
                    ['threshold', 'calendar']
                )
                
            with col2:
                if backtest_method == 'threshold':
                    bt_threshold = st.slider("Backtest Threshold (%)", 1, 20, 5) / 100
                else:
                    bt_frequency = st.selectbox("Rebalance Frequency (days)", [30, 60, 90])
            
            if st.button("🚀 Run Backtest"):
                with st.spinner("Running backtest..."):
                    # Run backtest
                    if backtest_method == 'threshold':
                        backtest_results = rebalancer.backtest_rebalancing_strategy(
                            price_data, target_weights, 'threshold', threshold=bt_threshold
                        )
                    else:
                        backtest_results = rebalancer.backtest_rebalancing_strategy(
                            price_data, target_weights, 'calendar', frequency=bt_frequency
                        )
                    
                    # Display results
                    st.markdown("#### Backtest Results")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        create_metric_card("Total Return", f"{backtest_results['total_return']:.2%}")
                    with col2:
                        create_metric_card("Volatility", f"{backtest_results['volatility']:.2%}")
                    with col3:
                        create_metric_card("Sharpe Ratio", f"{backtest_results['sharpe_ratio']:.2f}")
                    with col4:
                        create_metric_card("Rebalances", str(backtest_results['n_rebalances']))
                    
                    # Portfolio value chart
                    fig = go.Figure()
                    
                    dates = backtest_results['dates']
                    portfolio_values = backtest_results['portfolio_values']
                    
                    fig.add_trace(go.Scatter(
                        x=dates,
                        y=portfolio_values,
                        mode='lines',
                        name='Portfolio Value',
                        line=dict(color='#4ECDC4', width=2),
                        hovertemplate='Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
                    ))
                    
                    # Mark rebalancing dates
                    for rb_date in backtest_results['rebalancing_dates']:
                        if rb_date in dates:
                            idx = list(dates).index(rb_date)
                            fig.add_trace(go.Scatter(
                                x=[rb_date],
                                y=[portfolio_values[idx]],
                                mode='markers',
                                marker=dict(color='red', size=8, symbol='diamond'),
                                name='Rebalance',
                                showlegend=False,
                                hovertemplate='Rebalance<br>Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
                            ))
                    
                    fig.update_layout(
                        title='Portfolio Value with Rebalancing Events',
                        xaxis_title='Date',
                        yaxis_title='Portfolio Value ($)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Transaction costs analysis
                    st.markdown("#### Cost Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        create_metric_card("Total Transaction Costs", f"${backtest_results['transaction_costs']:,.2f}")
                    
                    with col2:
                        cost_per_rebalance = backtest_results['transaction_costs'] / backtest_results['n_rebalances'] if backtest_results['n_rebalances'] > 0 else 0
                        create_metric_card("Avg Cost per Rebalance", f"${cost_per_rebalance:,.2f}")
                    
                    # Rebalancing calendar
                    calendar_fig = rebalancer.create_rebalancing_calendar(backtest_results)
                    st.plotly_chart(calendar_fig, use_container_width=True)
        
        else:
            st.warning("Insufficient historical data for backtesting")
    
    with alerts_tab:
        st.markdown("### Rebalancing Alerts & Monitoring")
        
        # Alert configuration
        st.markdown("#### Alert Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            alert_threshold = st.slider("Alert Threshold (%)", 1, 20, 5) / 100
            enable_alerts = st.checkbox("Enable Email Alerts", value=True)
            
        with col2:
            alert_frequency = st.selectbox("Check Frequency", ["Daily", "Weekly", "Monthly"])
            alert_methods = st.multiselect("Alert Methods", ["Email", "SMS", "Push Notification"], default=["Email"])
        
        # Current alert status
        current_drift = max_drift
        alert_triggered = current_drift > alert_threshold
        
        if alert_triggered:
            st.error(f"🚨 ALERT: Portfolio drift ({current_drift:.2%}) exceeds threshold ({alert_threshold:.2%})")
            
            # Alert details
            st.markdown("#### Alert Details")
            
            alert_details = []
            for symbol in symbols:
                asset_drift = abs(drift[symbol])
                if asset_drift > alert_threshold:
                    alert_details.append({
                        'Asset': symbol,
                        'Current Weight': f"{simulated_current_weights[symbol]:.2%}",
                        'Target Weight': f"{target_weights[symbol]:.2%}",
                        'Drift': f"{drift[symbol]:+.2%}",
                        'Severity': 'High' if asset_drift > alert_threshold * 2 else 'Medium'
                    })
            
            if alert_details:
                alert_df = pd.DataFrame(alert_details)
                st.dataframe(alert_df, use_container_width=True, hide_index=True)
        
        else:
            st.success(f"✅ Portfolio drift ({current_drift:.2%}) is within acceptable range")
        
        # Alert history (simulated)
        st.markdown("#### Recent Alert History")
        
        alert_history = pd.DataFrame({
            'Date': pd.date_range(start='2024-01-01', periods=5, freq='7D'),
            'Alert Type': ['Drift Alert', 'Rebalance Complete', 'Drift Alert', 'Volatility Alert', 'Rebalance Complete'],
            'Severity': ['Medium', 'Info', 'High', 'Medium', 'Info'],
            'Message': [
                'AAPL drift exceeds 5% threshold',
                'Portfolio rebalanced successfully',
                'Multiple assets require rebalancing',
                'Portfolio volatility above target',
                'Quarterly rebalancing completed'
            ]
        })
        
        # Color code by severity
        def color_severity(val):
            color_map = {'High': 'background-color: #ff4444', 'Medium': 'background-color: #ffaa44', 'Info': 'background-color: #44ff44'}
            return color_map.get(val, '')
        
        styled_history = alert_history.style.applymap(color_severity, subset=['Severity'])
        st.dataframe(styled_history, use_container_width=True, hide_index=True)
        
        # Manual rebalancing trigger
        st.markdown("#### Manual Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Force Rebalance Now"):
                st.info("Manual rebalancing triggered - orders would be placed")
        
        with col2:
            if st.button("⏰ Schedule Rebalance"):
                st.info("Rebalancing scheduled for next market open")
        
        with col3:
            if st.button("📊 Generate Report"):
                st.info("Detailed rebalancing report generated")
    
    # Educational content
    with st.expander("📚 Portfolio Rebalancing Guide"):
        st.markdown("""
        **Portfolio Rebalancing Fundamentals:**
        
        **Why Rebalance?**
        - Maintain target risk level and asset allocation
        - Enforce disciplined buying low and selling high
        - Prevent style drift and concentration risk
        - Optimize risk-adjusted returns over time
        
        **Rebalancing Methods:**
        
        **Threshold Rebalancing:**
        - Rebalance when any asset deviates by X% from target
        - More responsive to market movements
        - Higher transaction costs but better risk control
        
        **Calendar Rebalancing:**
        - Rebalance at regular intervals (monthly, quarterly)
        - Lower transaction costs and complexity
        - May allow larger drifts between rebalancing dates
        
        **Volatility Targeting:**
        - Adjust positions based on portfolio volatility
        - Reduce risk during volatile periods
        - Increase exposure during calm markets
        
        **Risk Parity:**
        - Equal risk contribution from each asset
        - Based on volatility and correlation
        - More sophisticated risk management approach
        
        **Best Practices:**
        - Consider transaction costs in rebalancing decisions
        - Use tax-loss harvesting opportunities
        - Implement gradual rebalancing for large portfolios
        - Monitor correlation changes during market stress
        - Set appropriate thresholds based on portfolio size and goals
        """)

if __name__ == "__main__":
    main()