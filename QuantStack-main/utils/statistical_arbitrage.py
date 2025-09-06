import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from statsmodels.tsa.stattools import coint
from sklearn.linear_model import LinearRegression
import streamlit as st

class StatisticalArbitrage:
    """Statistical arbitrage analysis and pair trading detection"""
    
    def __init__(self, price_data):
        """
        Initialize with price data
        
        Args:
            price_data (pandas.DataFrame): Price data for multiple assets
        """
        self.prices = price_data.dropna()
        self.returns = self.prices.pct_change().dropna()
        self.assets = price_data.columns.tolist()
    
    def correlation_analysis(self):
        """
        Calculate correlation matrix for all asset pairs
        
        Returns:
            pandas.DataFrame: Correlation matrix
        """
        return self.returns.corr()
    
    def find_cointegrated_pairs(self, significance_level=0.05):
        """
        Find cointegrated pairs using Engle-Granger test
        
        Args:
            significance_level (float): Significance level for cointegration test
        
        Returns:
            list: List of cointegrated pairs with test statistics
        """
        cointegrated_pairs = []
        n_assets = len(self.assets)
        
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                asset1, asset2 = self.assets[i], self.assets[j]
                
                # Get price series
                y = self.prices[asset1].dropna()
                x = self.prices[asset2].dropna()
                
                # Align the series
                aligned_data = pd.concat([y, x], axis=1).dropna()
                if len(aligned_data) < 50:  # Minimum data points
                    continue
                    
                y_aligned = aligned_data.iloc[:, 0]
                x_aligned = aligned_data.iloc[:, 1]
                
                try:
                    # Perform cointegration test
                    coint_stat, p_value, crit_values = coint(y_aligned, x_aligned)
                    
                    if p_value < significance_level:
                        cointegrated_pairs.append({
                            'Asset1': asset1,
                            'Asset2': asset2,
                            'Cointegration_Stat': coint_stat,
                            'P_Value': p_value,
                            'Critical_1%': crit_values[0],
                            'Critical_5%': crit_values[1],
                            'Critical_10%': crit_values[2]
                        })
                
                except Exception as e:
                    continue
        
        return cointegrated_pairs
    
    def calculate_spread(self, asset1, asset2):
        """
        Calculate spread between two assets using linear regression
        
        Args:
            asset1 (str): First asset
            asset2 (str): Second asset
        
        Returns:
            tuple: (spread_series, hedge_ratio, intercept)
        """
        # Get aligned price data
        data = pd.concat([self.prices[asset1], self.prices[asset2]], axis=1).dropna()
        y = data.iloc[:, 0].values.reshape(-1, 1)
        x = data.iloc[:, 1].values.reshape(-1, 1)
        
        # Fit linear regression
        model = LinearRegression()
        model.fit(x, y)
        
        hedge_ratio = model.coef_[0][0]
        intercept = model.intercept_[0]
        
        # Calculate spread
        spread = data.iloc[:, 0] - hedge_ratio * data.iloc[:, 1] - intercept
        
        return spread, hedge_ratio, intercept
    
    def generate_trading_signals(self, asset1, asset2, entry_threshold=2.0, exit_threshold=0.5):
        """
        Generate trading signals based on mean reversion of spread
        
        Args:
            asset1 (str): First asset
            asset2 (str): Second asset
            entry_threshold (float): Z-score threshold for entry
            exit_threshold (float): Z-score threshold for exit
        
        Returns:
            pandas.DataFrame: Trading signals
        """
        spread, hedge_ratio, intercept = self.calculate_spread(asset1, asset2)
        
        # Calculate z-score of spread
        spread_mean = spread.mean()
        spread_std = spread.std()
        z_score = (spread - spread_mean) / spread_std
        
        # Generate signals
        signals = pd.DataFrame(index=spread.index)
        signals['Spread'] = spread
        signals['Z_Score'] = z_score
        signals['Signal'] = 0
        
        # Long spread signal (short asset1, long asset2)
        signals.loc[z_score < -entry_threshold, 'Signal'] = 1
        # Short spread signal (long asset1, short asset2) 
        signals.loc[z_score > entry_threshold, 'Signal'] = -1
        # Exit signals
        signals.loc[abs(z_score) < exit_threshold, 'Signal'] = 0
        
        # Forward fill signals
        signals['Position'] = signals['Signal'].replace(0, np.nan).fillna(method='ffill').fillna(0)
        
        return signals, hedge_ratio
    
    def backtest_pair_strategy(self, asset1, asset2, initial_capital=100000, 
                              entry_threshold=2.0, exit_threshold=0.5):
        """
        Backtest pair trading strategy
        
        Args:
            asset1 (str): First asset
            asset2 (str): Second asset  
            initial_capital (float): Initial capital
            entry_threshold (float): Z-score threshold for entry
            exit_threshold (float): Z-score threshold for exit
        
        Returns:
            dict: Backtest results
        """
        signals, hedge_ratio = self.generate_trading_signals(asset1, asset2, entry_threshold, exit_threshold)
        
        # Get aligned price data
        prices = pd.concat([self.prices[asset1], self.prices[asset2]], axis=1).dropna()
        prices.columns = ['Asset1', 'Asset2']
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        # Align signals with returns
        aligned_signals = signals.reindex(returns.index).fillna(0)
        
        # Calculate strategy returns
        portfolio_returns = []
        positions = []
        
        for i, (date, row) in enumerate(aligned_signals.iterrows()):
            if date not in returns.index:
                continue
                
            position = row['Position']
            
            if position != 0:
                # Position: 1 = long spread (short asset1, long asset2)
                #          -1 = short spread (long asset1, short asset2)
                ret1 = returns.loc[date, 'Asset1']
                ret2 = returns.loc[date, 'Asset2'] 
                
                # Strategy return (considering hedge ratio)
                strategy_return = position * (-ret1 + hedge_ratio * ret2)
                portfolio_returns.append(strategy_return)
            else:
                portfolio_returns.append(0)
            
            positions.append(position)
        
        portfolio_returns = pd.Series(portfolio_returns, index=returns.index)
        
        # Calculate performance metrics
        cumulative_returns = (1 + portfolio_returns).cumprod()
        total_return = cumulative_returns.iloc[-1] - 1
        annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility != 0 else 0
        
        # Calculate maximum drawdown
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Win rate
        profitable_trades = portfolio_returns[portfolio_returns > 0]
        losing_trades = portfolio_returns[portfolio_returns < 0]
        total_trades = len(profitable_trades) + len(losing_trades)
        win_rate = len(profitable_trades) / total_trades if total_trades > 0 else 0
        
        return {
            'Total_Return': total_return,
            'Annualized_Return': annualized_return,
            'Volatility': volatility,
            'Sharpe_Ratio': sharpe_ratio,
            'Max_Drawdown': max_drawdown,
            'Win_Rate': win_rate,
            'Total_Trades': total_trades,
            'Portfolio_Returns': portfolio_returns,
            'Cumulative_Returns': cumulative_returns,
            'Signals': signals,
            'Hedge_Ratio': hedge_ratio
        }
    
    def plot_pair_analysis(self, asset1, asset2):
        """
        Create comprehensive pair analysis plots
        
        Args:
            asset1 (str): First asset
            asset2 (str): Second asset
        
        Returns:
            dict: Dictionary of plotly figures
        """
        plots = {}
        
        # 1. Price comparison
        aligned_prices = pd.concat([self.prices[asset1], self.prices[asset2]], axis=1).dropna()
        aligned_prices.columns = [asset1, asset2]
        
        fig_prices = go.Figure()
        fig_prices.add_trace(go.Scatter(
            x=aligned_prices.index,
            y=aligned_prices[asset1],
            mode='lines',
            name=asset1,
            yaxis='y'
        ))
        fig_prices.add_trace(go.Scatter(
            x=aligned_prices.index,
            y=aligned_prices[asset2],
            mode='lines',
            name=asset2,
            yaxis='y2'
        ))
        
        fig_prices.update_layout(
            title=f'Price Comparison: {asset1} vs {asset2}',
            xaxis_title='Date',
            yaxis=dict(title=f'{asset1} Price', side='left'),
            yaxis2=dict(title=f'{asset2} Price', side='right', overlaying='y'),
            height=400
        )
        
        plots['price_comparison'] = fig_prices
        
        # 2. Spread analysis
        spread, hedge_ratio, intercept = self.calculate_spread(asset1, asset2)
        spread_mean = spread.mean()
        spread_std = spread.std()
        z_score = (spread - spread_mean) / spread_std
        
        fig_spread = go.Figure()
        fig_spread.add_trace(go.Scatter(
            x=spread.index,
            y=spread,
            mode='lines',
            name='Spread',
            line=dict(color='blue')
        ))
        fig_spread.add_hline(y=spread_mean, line_dash="dash", line_color="red", 
                            annotation_text="Mean")
        fig_spread.add_hline(y=spread_mean + 2*spread_std, line_dash="dot", line_color="orange",
                            annotation_text="+2σ")
        fig_spread.add_hline(y=spread_mean - 2*spread_std, line_dash="dot", line_color="orange",
                            annotation_text="-2σ")
        
        fig_spread.update_layout(
            title=f'Spread Analysis: {asset1} - {hedge_ratio:.3f}*{asset2}',
            xaxis_title='Date',
            yaxis_title='Spread',
            height=400
        )
        
        plots['spread_analysis'] = fig_spread
        
        # 3. Z-score
        fig_zscore = go.Figure()
        fig_zscore.add_trace(go.Scatter(
            x=z_score.index,
            y=z_score,
            mode='lines',
            name='Z-Score',
            line=dict(color='purple')
        ))
        fig_zscore.add_hline(y=0, line_dash="dash", line_color="black")
        fig_zscore.add_hline(y=2, line_dash="dot", line_color="red", annotation_text="Entry Threshold")
        fig_zscore.add_hline(y=-2, line_dash="dot", line_color="red")
        
        fig_zscore.update_layout(
            title='Spread Z-Score',
            xaxis_title='Date',
            yaxis_title='Z-Score',
            height=400
        )
        
        plots['z_score'] = fig_zscore
        
        # 4. Scatter plot
        fig_scatter = px.scatter(
            x=aligned_prices[asset2],
            y=aligned_prices[asset1],
            title=f'Price Relationship: {asset1} vs {asset2}',
            labels={'x': f'{asset2} Price', 'y': f'{asset1} Price'}
        )
        
        # Add regression line
        x_range = np.array([aligned_prices[asset2].min(), aligned_prices[asset2].max()])
        y_range = hedge_ratio * x_range + intercept
        fig_scatter.add_trace(go.Scatter(
            x=x_range,
            y=y_range,
            mode='lines',
            name=f'Regression Line (β={hedge_ratio:.3f})',
            line=dict(color='red', dash='dash')
        ))
        
        plots['scatter_plot'] = fig_scatter
        
        return plots
    
    def correlation_heatmap(self):
        """
        Create correlation heatmap
        
        Returns:
            plotly.graph_objects.Figure: Correlation heatmap
        """
        corr_matrix = self.correlation_analysis()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(3).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Asset Correlation Matrix',
            height=600,
            width=800
        )
        
        return fig
