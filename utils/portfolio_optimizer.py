import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
import streamlit as st
from .data_fetcher import DataFetcher

class PortfolioOptimizer:
    """Modern Portfolio Theory implementation"""
    
    def __init__(self, returns_data):
        """
        Initialize with returns data
        
        Args:
            returns_data (pandas.DataFrame): Returns data for assets
        """
        self.returns = returns_data.dropna()
        self.assets = returns_data.columns.tolist()
        self.n_assets = len(self.assets)
        
        # Calculate expected returns and covariance matrix
        self.expected_returns = self.returns.mean() * 252  # Annualized
        self.cov_matrix = self.returns.cov() * 252  # Annualized
    
    def portfolio_stats(self, weights):
        """
        Calculate portfolio statistics
        
        Args:
            weights (numpy.array): Portfolio weights
        
        Returns:
            tuple: (expected_return, volatility, sharpe_ratio)
        """
        portfolio_return = np.sum(weights * self.expected_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        
        # Assuming risk-free rate of 2%
        risk_free_rate = 0.02
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        
        return portfolio_return, portfolio_volatility, sharpe_ratio
    
    def minimize_volatility(self, target_return):
        """
        Minimize volatility for a given target return
        
        Args:
            target_return (float): Target portfolio return
        
        Returns:
            numpy.array: Optimal weights
        """
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
            {'type': 'eq', 'fun': lambda x: np.sum(x * self.expected_returns) - target_return}  # Target return
        ]
        
        bounds = tuple((0, 1) for _ in range(self.n_assets))  # Long-only portfolio
        
        # Objective function: minimize volatility
        def objective(weights):
            return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        
        # Initial guess: equal weights
        initial_guess = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        
        return result.x if result.success else None
    
    def maximize_sharpe_ratio(self):
        """
        Find portfolio with maximum Sharpe ratio
        
        Returns:
            numpy.array: Optimal weights
        """
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        # Objective function: minimize negative Sharpe ratio
        def objective(weights):
            _, _, sharpe = self.portfolio_stats(weights)
            return -sharpe
        
        initial_guess = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        
        return result.x if result.success else None
    
    def minimize_variance(self):
        """
        Find minimum variance portfolio
        
        Returns:
            numpy.array: Optimal weights
        """
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        # Objective function: minimize variance
        def objective(weights):
            return np.dot(weights.T, np.dot(self.cov_matrix, weights))
        
        initial_guess = np.array([1/self.n_assets] * self.n_assets)
        
        result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        
        return result.x if result.success else None
    
    def efficient_frontier(self, n_portfolios=100):
        """
        Generate efficient frontier
        
        Args:
            n_portfolios (int): Number of portfolios to generate
        
        Returns:
            pandas.DataFrame: Efficient frontier data
        """
        min_ret = self.expected_returns.min()
        max_ret = self.expected_returns.max()
        
        target_returns = np.linspace(min_ret, max_ret, n_portfolios)
        efficient_portfolios = []
        
        for target_return in target_returns:
            weights = self.minimize_volatility(target_return)
            if weights is not None:
                ret, vol, sharpe = self.portfolio_stats(weights)
                efficient_portfolios.append({
                    'Return': ret,
                    'Volatility': vol,
                    'Sharpe_Ratio': sharpe,
                    'Weights': weights
                })
        
        return pd.DataFrame(efficient_portfolios)
    
    def plot_efficient_frontier(self):
        """
        Create interactive plot of efficient frontier
        
        Returns:
            plotly.graph_objects.Figure: Efficient frontier plot
        """
        # Generate efficient frontier
        efficient_df = self.efficient_frontier()
        
        if efficient_df.empty:
            st.error("Could not generate efficient frontier")
            return None
        
        # Create plot
        fig = go.Figure()
        
        # Efficient frontier
        fig.add_trace(go.Scatter(
            x=efficient_df['Volatility'],
            y=efficient_df['Return'],
            mode='lines+markers',
            name='Efficient Frontier',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))
        
        # Individual assets
        for i, asset in enumerate(self.assets):
            asset_return = self.expected_returns.iloc[i]
            asset_vol = np.sqrt(self.cov_matrix.iloc[i, i])
            
            fig.add_trace(go.Scatter(
                x=[asset_vol],
                y=[asset_return],
                mode='markers+text',
                name=asset,
                text=[asset],
                textposition='top center',
                marker=dict(size=10, color='red')
            ))
        
        # Maximum Sharpe ratio portfolio
        max_sharpe_weights = self.maximize_sharpe_ratio()
        if max_sharpe_weights is not None:
            ret, vol, sharpe = self.portfolio_stats(max_sharpe_weights)
            fig.add_trace(go.Scatter(
                x=[vol],
                y=[ret],
                mode='markers+text',
                name='Max Sharpe Ratio',
                text=['Max Sharpe'],
                textposition='top center',
                marker=dict(size=15, color='green', symbol='star')
            ))
        
        # Minimum variance portfolio
        min_var_weights = self.minimize_variance()
        if min_var_weights is not None:
            ret, vol, sharpe = self.portfolio_stats(min_var_weights)
            fig.add_trace(go.Scatter(
                x=[vol],
                y=[ret],
                mode='markers+text',
                name='Min Variance',
                text=['Min Var'],
                textposition='top center',
                marker=dict(size=15, color='orange', symbol='diamond')
            ))
        
        fig.update_layout(
            title='Efficient Frontier',
            xaxis_title='Volatility (Standard Deviation)',
            yaxis_title='Expected Return',
            hovermode='closest',
            height=600
        )
        
        return fig
    
    def get_portfolio_composition(self, weights, portfolio_name="Portfolio"):
        """
        Get portfolio composition visualization
        
        Args:
            weights (numpy.array): Portfolio weights
            portfolio_name (str): Name of the portfolio
        
        Returns:
            plotly.graph_objects.Figure: Portfolio composition pie chart
        """
        # Filter out very small weights
        significant_weights = weights[weights > 0.001]
        significant_assets = [self.assets[i] for i, w in enumerate(weights) if w > 0.001]
        
        fig = go.Figure(data=[go.Pie(
            labels=significant_assets,
            values=significant_weights,
            hole=.3
        )])
        
        fig.update_layout(
            title=f'{portfolio_name} Composition',
            annotations=[dict(text=portfolio_name, x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        
        return fig
    
    def calculate_risk_metrics(self, weights):
        """
        Calculate comprehensive risk metrics for a portfolio
        
        Args:
            weights (numpy.array): Portfolio weights
        
        Returns:
            dict: Risk metrics
        """
        portfolio_returns = (self.returns * weights).sum(axis=1)
        
        # Basic statistics
        ret, vol, sharpe = self.portfolio_stats(weights)
        
        # Value at Risk (VaR) and Expected Shortfall (ES)
        var_95 = np.percentile(portfolio_returns, 5)
        var_99 = np.percentile(portfolio_returns, 1)
        es_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        es_99 = portfolio_returns[portfolio_returns <= var_99].mean()
        
        # Maximum Drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Beta (assuming S&P 500 as market proxy)
        try:
            market_data = DataFetcher.get_stock_data("^GSPC", period="1y")
            if not market_data.empty:
                market_returns = market_data['Close'].pct_change().dropna()
                # Align dates
                aligned_portfolio = portfolio_returns.reindex(market_returns.index).dropna()
                aligned_market = market_returns.reindex(aligned_portfolio.index).dropna()
                
                if len(aligned_portfolio) > 0 and len(aligned_market) > 0:
                    covariance = np.cov(aligned_portfolio, aligned_market)[0, 1]
                    market_variance = np.var(aligned_market)
                    beta = covariance / market_variance if market_variance != 0 else 0
                else:
                    beta = 0
            else:
                beta = 0
        except:
            beta = 0
        
        return {
            'Expected Return': ret,
            'Volatility': vol,
            'Sharpe Ratio': sharpe,
            'VaR (95%)': var_95,
            'VaR (99%)': var_99,
            'Expected Shortfall (95%)': es_95,
            'Expected Shortfall (99%)': es_99,
            'Maximum Drawdown': max_drawdown,
            'Beta': beta
        }
