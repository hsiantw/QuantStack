import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class MonteCarloSimulator:
    """Advanced Monte Carlo simulation engine for financial risk analysis"""
    
    def __init__(self, returns_data: pd.DataFrame, initial_portfolio_value: float = 100000):
        """
        Initialize Monte Carlo simulator
        
        Args:
            returns_data: DataFrame with asset returns
            initial_portfolio_value: Starting portfolio value
        """
        self.returns_data = returns_data
        self.initial_value = initial_portfolio_value
        self.assets = returns_data.columns.tolist()
        
        # Calculate statistical parameters
        self.mean_returns = returns_data.mean()
        self.cov_matrix = returns_data.cov()
        self.std_returns = returns_data.std()
        self.correlations = returns_data.corr()
        
    def simulate_portfolio_paths(self, 
                                weights: np.ndarray,
                                time_horizon: int = 252,  # Trading days in a year
                                num_simulations: int = 10000,
                                confidence_levels: List[float] = [0.95, 0.99]) -> Dict:
        """
        Run Monte Carlo simulation for portfolio value paths
        
        Args:
            weights: Portfolio weights array
            time_horizon: Number of days to simulate
            num_simulations: Number of simulation runs
            confidence_levels: VaR confidence levels
            
        Returns:
            Dictionary with simulation results
        """
        
        # Portfolio statistics
        portfolio_return = np.sum(weights * self.mean_returns)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        
        # Generate correlated random returns using Cholesky decomposition
        try:
            L = np.linalg.cholesky(self.cov_matrix.values)
        except np.linalg.LinAlgError:
            # If covariance matrix is not positive definite, use eigenvalue method
            eigenvals, eigenvecs = np.linalg.eigh(self.cov_matrix.values)
            eigenvals = np.maximum(eigenvals, 1e-8)  # Ensure positive eigenvalues
            L = eigenvecs @ np.diag(np.sqrt(eigenvals))
        
        # Initialize simulation arrays
        portfolio_paths = np.zeros((num_simulations, time_horizon + 1))
        portfolio_paths[:, 0] = self.initial_value
        
        daily_returns = np.zeros((num_simulations, time_horizon))
        
        for sim in range(num_simulations):
            # Generate random normal variables
            random_vars = np.random.normal(0, 1, (len(self.assets), time_horizon))
            
            # Apply correlation structure
            correlated_returns = L @ random_vars
            
            # Convert to asset returns
            asset_returns = self.mean_returns.values.reshape(-1, 1) + correlated_returns
            
            # Calculate portfolio returns
            portfolio_daily_returns = np.sum(weights.reshape(-1, 1) * asset_returns, axis=0)
            daily_returns[sim, :] = portfolio_daily_returns
            
            # Calculate cumulative portfolio values
            for day in range(time_horizon):
                portfolio_paths[sim, day + 1] = portfolio_paths[sim, day] * (1 + portfolio_daily_returns[day])
        
        # Calculate risk metrics
        final_values = portfolio_paths[:, -1]
        total_returns = (final_values - self.initial_value) / self.initial_value
        
        # Value at Risk calculations
        var_results = {}
        for conf_level in confidence_levels:
            var_value = np.percentile(total_returns, (1 - conf_level) * 100)
            cvar_value = np.mean(total_returns[total_returns <= var_value])  # Conditional VaR
            
            var_results[f'VaR_{int(conf_level*100)}'] = var_value
            var_results[f'CVaR_{int(conf_level*100)}'] = cvar_value
        
        # Maximum drawdown calculation
        cumulative_returns = np.cumprod(1 + daily_returns, axis=1)
        peak_values = np.maximum.accumulate(cumulative_returns, axis=1)
        drawdowns = (cumulative_returns - peak_values) / peak_values
        max_drawdowns = np.min(drawdowns, axis=1)
        
        # Compile results
        results = {
            'portfolio_paths': portfolio_paths,
            'daily_returns': daily_returns,
            'final_values': final_values,
            'total_returns': total_returns,
            'expected_return': np.mean(total_returns),
            'volatility': np.std(total_returns),
            'var_metrics': var_results,
            'max_drawdown_dist': max_drawdowns,
            'expected_max_drawdown': np.mean(max_drawdowns),
            'probability_of_loss': np.sum(total_returns < 0) / num_simulations,
            'sharpe_ratio': (np.mean(total_returns) - 0.02) / np.std(total_returns),  # Assuming 2% risk-free rate
            'portfolio_stats': {
                'expected_daily_return': portfolio_return,
                'daily_volatility': portfolio_std,
                'annualized_return': portfolio_return * 252,
                'annualized_volatility': portfolio_std * np.sqrt(252)
            }
        }
        
        return results
    
    def stress_test_scenarios(self, 
                             weights: np.ndarray,
                             scenarios: Dict[str, Dict]) -> Dict:
        """
        Perform stress testing under various market scenarios
        
        Args:
            weights: Portfolio weights
            scenarios: Dictionary of stress scenarios
            
        Returns:
            Dictionary with stress test results
        """
        
        stress_results = {}
        
        for scenario_name, scenario_params in scenarios.items():
            # Apply stress to returns
            stressed_returns = self.returns_data.copy()
            
            if 'market_shock' in scenario_params:
                # Apply market-wide shock
                shock = scenario_params['market_shock']
                stressed_returns = stressed_returns + shock
            
            if 'volatility_multiplier' in scenario_params:
                # Increase volatility
                multiplier = scenario_params['volatility_multiplier']
                mean_returns = stressed_returns.mean()
                stressed_returns = mean_returns + (stressed_returns - mean_returns) * multiplier
            
            if 'correlation_increase' in scenario_params:
                # Increase correlations (contagion effect)
                corr_increase = scenario_params['correlation_increase']
                original_corr = self.correlations.copy()
                stressed_corr = original_corr + (1 - original_corr) * corr_increase
                np.fill_diagonal(stressed_corr.values, 1.0)
                
                # Convert correlation back to covariance
                stressed_std = stressed_returns.std()
                stressed_cov = stressed_corr * np.outer(stressed_std, stressed_std)
                
                # Generate new returns with stressed correlation
                L = np.linalg.cholesky(stressed_cov.values)
                random_vars = np.random.normal(0, 1, (len(self.assets), len(stressed_returns)))
                new_returns = stressed_returns.mean().values.reshape(-1, 1) + L @ random_vars
                stressed_returns = pd.DataFrame(new_returns.T, columns=self.assets)
            
            # Calculate portfolio impact
            portfolio_returns = np.sum(weights * stressed_returns.T, axis=0)
            portfolio_value = self.initial_value * np.cumprod(1 + portfolio_returns)
            
            stress_results[scenario_name] = {
                'final_value': portfolio_value[-1],
                'total_return': (portfolio_value[-1] - self.initial_value) / self.initial_value,
                'max_drawdown': np.min((portfolio_value - np.maximum.accumulate(portfolio_value)) / np.maximum.accumulate(portfolio_value)),
                'volatility': np.std(portfolio_returns),
                'scenario_params': scenario_params
            }
        
        return stress_results
    
    def options_risk_analysis(self, 
                             underlying_price: float,
                             strike_price: float,
                             time_to_expiry: float,
                             option_type: str = 'call',
                             num_simulations: int = 10000) -> Dict:
        """
        Monte Carlo simulation for options pricing and risk analysis
        
        Args:
            underlying_price: Current price of underlying asset
            strike_price: Strike price of option
            time_to_expiry: Time to expiration in years
            option_type: 'call' or 'put'
            num_simulations: Number of simulations
            
        Returns:
            Dictionary with options analysis results
        """
        
        # Assume we have volatility from the portfolio data
        if len(self.assets) > 0:
            # Use first asset's volatility as proxy
            volatility = self.std_returns.iloc[0] * np.sqrt(252)
        else:
            volatility = 0.2  # Default 20% volatility
        
        risk_free_rate = 0.02  # 2% risk-free rate
        
        # Generate random price paths using geometric Brownian motion
        dt = time_to_expiry / 252  # Daily time step
        drift = (risk_free_rate - 0.5 * volatility**2) * dt
        diffusion = volatility * np.sqrt(dt)
        
        price_paths = np.zeros((num_simulations, int(252 * time_to_expiry) + 1))
        price_paths[:, 0] = underlying_price
        
        for i in range(1, price_paths.shape[1]):
            random_shocks = np.random.normal(0, 1, num_simulations)
            price_paths[:, i] = price_paths[:, i-1] * np.exp(drift + diffusion * random_shocks)
        
        # Calculate option payoffs
        final_prices = price_paths[:, -1]
        
        if option_type.lower() == 'call':
            payoffs = np.maximum(final_prices - strike_price, 0)
        else:  # put
            payoffs = np.maximum(strike_price - final_prices, 0)
        
        # Discount payoffs to present value
        option_values = payoffs * np.exp(-risk_free_rate * time_to_expiry)
        
        # Calculate Greeks through finite differences
        price_up = underlying_price * 1.01
        price_down = underlying_price * 0.99
        
        # Delta calculation (simplified)
        delta_simulations = 1000
        up_payoffs = []
        down_payoffs = []
        
        for _ in range(delta_simulations):
            # Simulate for price up
            path_up = underlying_price * 1.01
            for j in range(int(252 * time_to_expiry)):
                shock = np.random.normal(0, 1)
                path_up *= np.exp(drift + diffusion * shock)
            
            if option_type.lower() == 'call':
                up_payoffs.append(max(path_up - strike_price, 0))
            else:
                up_payoffs.append(max(strike_price - path_up, 0))
            
            # Simulate for price down
            path_down = underlying_price * 0.99
            for j in range(int(252 * time_to_expiry)):
                shock = np.random.normal(0, 1)
                path_down *= np.exp(drift + diffusion * shock)
            
            if option_type.lower() == 'call':
                down_payoffs.append(max(path_down - strike_price, 0))
            else:
                down_payoffs.append(max(strike_price - path_down, 0))
        
        delta = (np.mean(up_payoffs) - np.mean(down_payoffs)) / (price_up - price_down)
        
        results = {
            'option_value': np.mean(option_values),
            'option_std': np.std(option_values),
            'price_paths': price_paths,
            'payoffs': payoffs,
            'delta': delta,
            'probability_itm': np.sum(payoffs > 0) / num_simulations,
            'expected_payoff': np.mean(payoffs),
            'value_at_risk_95': np.percentile(option_values, 5),
            'value_at_risk_99': np.percentile(option_values, 1),
            'scenario_analysis': {
                'bull_case': np.percentile(option_values, 95),
                'bear_case': np.percentile(option_values, 5),
                'base_case': np.percentile(option_values, 50)
            }
        }
        
        return results
    
    def create_risk_dashboard_plots(self, simulation_results: Dict, weights: np.ndarray) -> Dict[str, go.Figure]:
        """
        Create comprehensive risk analysis visualizations
        
        Args:
            simulation_results: Results from Monte Carlo simulation
            weights: Portfolio weights
            
        Returns:
            Dictionary of Plotly figures
        """
        
        figures = {}
        
        # 1. Portfolio Value Paths
        fig_paths = go.Figure()
        
        # Show sample paths (first 100 simulations)
        portfolio_paths = simulation_results['portfolio_paths']
        time_axis = np.arange(portfolio_paths.shape[1])
        
        for i in range(min(100, portfolio_paths.shape[0])):
            fig_paths.add_trace(go.Scatter(
                x=time_axis,
                y=portfolio_paths[i, :],
                mode='lines',
                line=dict(width=0.5, color='rgba(0, 100, 200, 0.1)'),
                showlegend=False,
                hovertemplate='Day: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
            ))
        
        # Add percentile bands
        percentiles = [5, 25, 50, 75, 95]
        colors = ['red', 'orange', 'blue', 'orange', 'red']
        names = ['5th percentile', '25th percentile', 'Median', '75th percentile', '95th percentile']
        
        for p, color, name in zip(percentiles, colors, names):
            path_percentile = np.percentile(portfolio_paths, p, axis=0)
            fig_paths.add_trace(go.Scatter(
                x=time_axis,
                y=path_percentile,
                mode='lines',
                line=dict(width=2, color=color),
                name=name
            ))
        
        fig_paths.update_layout(
            title='Monte Carlo Portfolio Value Simulation',
            xaxis_title='Trading Days',
            yaxis_title='Portfolio Value ($)',
            template='plotly_dark',
            height=500
        )
        
        figures['portfolio_paths'] = fig_paths
        
        # 2. Return Distribution
        fig_dist = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Return Distribution', 'VaR Analysis', 'Drawdown Distribution', 'Risk Metrics'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "table"}]]
        )
        
        # Return histogram
        returns = simulation_results['total_returns']
        fig_dist.add_trace(
            go.Histogram(
                x=returns * 100,
                nbinsx=50,
                name='Returns',
                marker_color='lightblue',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Add VaR lines
        var_95 = simulation_results['var_metrics']['VaR_95'] * 100
        var_99 = simulation_results['var_metrics']['VaR_99'] * 100
        
        fig_dist.add_vline(x=var_95, line_dash="dash", line_color="red", 
                          annotation_text=f"VaR 95%: {var_95:.1f}%", row=1, col=1)
        fig_dist.add_vline(x=var_99, line_dash="dash", line_color="darkred", 
                          annotation_text=f"VaR 99%: {var_99:.1f}%", row=1, col=1)
        
        # VaR confidence intervals
        confidence_levels = [0.90, 0.95, 0.99]
        var_values = [np.percentile(returns, (1-cl)*100)*100 for cl in confidence_levels]
        
        fig_dist.add_trace(
            go.Bar(
                x=[f'{cl*100:.0f}%' for cl in confidence_levels],
                y=var_values,
                name='VaR by Confidence Level',
                marker_color='red',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        # Drawdown distribution
        max_drawdowns = simulation_results['max_drawdown_dist']
        fig_dist.add_trace(
            go.Histogram(
                x=max_drawdowns * 100,
                nbinsx=30,
                name='Max Drawdown',
                marker_color='orange',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Risk metrics table
        metrics_data = [
            ['Expected Return', f"{simulation_results['expected_return']*100:.2f}%"],
            ['Volatility', f"{simulation_results['volatility']*100:.2f}%"],
            ['Sharpe Ratio', f"{simulation_results['sharpe_ratio']:.3f}"],
            ['Probability of Loss', f"{simulation_results['probability_of_loss']*100:.1f}%"],
            ['Expected Max Drawdown', f"{simulation_results['expected_max_drawdown']*100:.2f}%"],
            ['VaR 95%', f"{var_95:.2f}%"],
            ['CVaR 95%', f"{simulation_results['var_metrics']['CVaR_95']*100:.2f}%"]
        ]
        
        fig_dist.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value'], fill_color='lightblue'),
                cells=dict(values=list(zip(*metrics_data)), fill_color='lightgray')
            ),
            row=2, col=2
        )
        
        fig_dist.update_layout(
            template='plotly_dark',
            height=800,
            showlegend=False
        )
        
        figures['risk_analysis'] = fig_dist
        
        # 3. Correlation Heatmap
        fig_corr = go.Figure(data=go.Heatmap(
            z=self.correlations.values,
            x=self.correlations.columns,
            y=self.correlations.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(self.correlations.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig_corr.update_layout(
            title='Asset Correlation Matrix',
            template='plotly_dark',
            height=500
        )
        
        figures['correlation_matrix'] = fig_corr
        
        return figures
    
    def efficient_frontier_monte_carlo(self, 
                                     num_portfolios: int = 10000,
                                     risk_free_rate: float = 0.02) -> pd.DataFrame:
        """
        Generate efficient frontier using Monte Carlo method
        
        Args:
            num_portfolios: Number of random portfolios to generate
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            
        Returns:
            DataFrame with portfolio metrics
        """
        
        num_assets = len(self.assets)
        results = np.zeros((3, num_portfolios))
        weights_array = np.zeros((num_portfolios, num_assets))
        
        # Generate random portfolios
        for i in range(num_portfolios):
            # Generate random weights that sum to 1
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            weights_array[i, :] = weights
            
            # Calculate portfolio return and risk
            portfolio_return = np.sum(weights * self.mean_returns) * 252
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights))) * np.sqrt(252)
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std
            
            results[0, i] = portfolio_return
            results[1, i] = portfolio_std
            results[2, i] = sharpe_ratio
        
        # Create DataFrame
        portfolio_df = pd.DataFrame({
            'Return': results[0, :],
            'Volatility': results[1, :],
            'Sharpe': results[2, :]
        })
        
        # Add weights
        for i, asset in enumerate(self.assets):
            portfolio_df[f'Weight_{asset}'] = weights_array[:, i]
        
        return portfolio_df