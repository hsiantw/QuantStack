import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from scipy.optimize import minimize
import streamlit as st

class RiskMetrics:
    """Comprehensive risk metrics calculation and analysis"""
    
    def __init__(self, returns_data, confidence_levels=[0.95, 0.99]):
        """
        Initialize with returns data
        
        Args:
            returns_data (pandas.DataFrame or Series): Returns data
            confidence_levels (list): Confidence levels for VaR/ES calculations
        """
        if isinstance(returns_data, pd.Series):
            self.returns = returns_data.dropna()
        else:
            self.returns = returns_data.dropna()
        
        self.confidence_levels = confidence_levels
    
    def value_at_risk(self, method='historical'):
        """
        Calculate Value at Risk using different methods
        
        Args:
            method (str): 'historical', 'parametric', or 'monte_carlo'
        
        Returns:
            dict: VaR values for different confidence levels
        """
        var_results = {}
        
        for conf_level in self.confidence_levels:
            alpha = 1 - conf_level
            
            if method == 'historical':
                # Historical simulation VaR
                if isinstance(self.returns, pd.Series):
                    var_value = np.percentile(self.returns, alpha * 100)
                else:
                    # For portfolio, calculate portfolio returns first
                    portfolio_returns = self.returns.mean(axis=1)
                    var_value = np.percentile(portfolio_returns, alpha * 100)
            
            elif method == 'parametric':
                # Parametric VaR (assumes normal distribution)
                if isinstance(self.returns, pd.Series):
                    mean_return = self.returns.mean()
                    std_return = self.returns.std()
                else:
                    portfolio_returns = self.returns.mean(axis=1)
                    mean_return = portfolio_returns.mean()
                    std_return = portfolio_returns.std()
                
                var_value = mean_return + stats.norm.ppf(alpha) * std_return
            
            elif method == 'monte_carlo':
                # Monte Carlo VaR
                if isinstance(self.returns, pd.Series):
                    mean_return = self.returns.mean()
                    std_return = self.returns.std()
                    simulated_returns = np.random.normal(mean_return, std_return, 10000)
                else:
                    portfolio_returns = self.returns.mean(axis=1)
                    mean_return = portfolio_returns.mean()
                    std_return = portfolio_returns.std()
                    simulated_returns = np.random.normal(mean_return, std_return, 10000)
                
                var_value = np.percentile(simulated_returns, alpha * 100)
            
            var_results[f'VaR_{int(conf_level*100)}%'] = var_value
        
        return var_results
    
    def expected_shortfall(self, method='historical'):
        """
        Calculate Expected Shortfall (Conditional VaR)
        
        Args:
            method (str): 'historical', 'parametric', or 'monte_carlo'
        
        Returns:
            dict: ES values for different confidence levels
        """
        es_results = {}
        var_results = self.value_at_risk(method)
        
        for conf_level in self.confidence_levels:
            var_key = f'VaR_{int(conf_level*100)}%'
            var_value = var_results[var_key]
            
            if method == 'historical':
                if isinstance(self.returns, pd.Series):
                    tail_returns = self.returns[self.returns <= var_value]
                else:
                    portfolio_returns = self.returns.mean(axis=1)
                    tail_returns = portfolio_returns[portfolio_returns <= var_value]
                
                es_value = tail_returns.mean() if len(tail_returns) > 0 else var_value
            
            elif method == 'parametric':
                alpha = 1 - conf_level
                if isinstance(self.returns, pd.Series):
                    mean_return = self.returns.mean()
                    std_return = self.returns.std()
                else:
                    portfolio_returns = self.returns.mean(axis=1)
                    mean_return = portfolio_returns.mean()
                    std_return = portfolio_returns.std()
                
                # Analytical ES for normal distribution
                es_value = mean_return - std_return * stats.norm.pdf(stats.norm.ppf(alpha)) / alpha
            
            elif method == 'monte_carlo':
                if isinstance(self.returns, pd.Series):
                    mean_return = self.returns.mean()
                    std_return = self.returns.std()
                    simulated_returns = np.random.normal(mean_return, std_return, 10000)
                else:
                    portfolio_returns = self.returns.mean(axis=1)
                    mean_return = portfolio_returns.mean()
                    std_return = portfolio_returns.std()
                    simulated_returns = np.random.normal(mean_return, std_return, 10000)
                
                tail_returns = simulated_returns[simulated_returns <= var_value]
                es_value = tail_returns.mean() if len(tail_returns) > 0 else var_value
            
            es_results[f'ES_{int(conf_level*100)}%'] = es_value
        
        return es_results
    
    def maximum_drawdown(self, returns_series=None):
        """
        Calculate maximum drawdown and related metrics
        
        Args:
            returns_series (pandas.Series): Optional specific returns series
        
        Returns:
            dict: Drawdown metrics
        """
        if returns_series is None:
            if isinstance(self.returns, pd.Series):
                returns_series = self.returns
            else:
                returns_series = self.returns.mean(axis=1)
        
        # Calculate cumulative returns
        cumulative_returns = (1 + returns_series).cumprod()
        
        # Calculate rolling maximum
        rolling_max = cumulative_returns.expanding().max()
        
        # Calculate drawdown
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        
        # Maximum drawdown
        max_drawdown = drawdown.min()
        
        # Drawdown duration
        drawdown_periods = (drawdown < 0).astype(int)
        drawdown_duration = drawdown_periods.groupby((drawdown_periods != drawdown_periods.shift()).cumsum()).sum().max()
        
        # Time to recovery (simplified)
        max_dd_date = drawdown.idxmin()
        recovery_date = drawdown[max_dd_date:].loc[drawdown[max_dd_date:] >= -0.001].index
        recovery_time = len(recovery_date) if len(recovery_date) > 0 else None
        
        return {
            'Max_Drawdown': max_drawdown,
            'Max_Drawdown_Date': max_dd_date,
            'Drawdown_Duration': drawdown_duration,
            'Recovery_Time': recovery_time,
            'Drawdown_Series': drawdown,
            'Cumulative_Returns': cumulative_returns
        }
    
    def risk_adjusted_ratios(self, risk_free_rate=0.02):
        """
        Calculate various risk-adjusted performance ratios
        
        Args:
            risk_free_rate (float): Risk-free rate (annualized)
        
        Returns:
            dict: Risk-adjusted ratios
        """
        if isinstance(self.returns, pd.Series):
            returns_series = self.returns
        else:
            returns_series = self.returns.mean(axis=1)
        
        # Annualized metrics
        annual_return = returns_series.mean() * 252
        annual_volatility = returns_series.std() * np.sqrt(252)
        
        # Sharpe Ratio
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility != 0 else 0
        
        # Sortino Ratio (downside deviation)
        downside_returns = returns_series[returns_series < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annual_return - risk_free_rate) / downside_deviation if downside_deviation != 0 else 0
        
        # Calmar Ratio
        max_dd = self.maximum_drawdown(returns_series)['Max_Drawdown']
        calmar_ratio = annual_return / abs(max_dd) if max_dd != 0 else 0
        
        # Information Ratio (assuming benchmark is risk-free rate)
        excess_returns = returns_series - risk_free_rate / 252
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error != 0 else 0
        
        # Treynor Ratio (requires beta calculation)
        # For simplicity, we'll skip this unless we have market data
        
        return {
            'Sharpe_Ratio': sharpe_ratio,
            'Sortino_Ratio': sortino_ratio,
            'Calmar_Ratio': calmar_ratio,
            'Information_Ratio': information_ratio,
            'Annual_Return': annual_return,
            'Annual_Volatility': annual_volatility
        }
    
    def tail_risk_metrics(self):
        """
        Calculate tail risk metrics
        
        Returns:
            dict: Tail risk metrics
        """
        if isinstance(self.returns, pd.Series):
            returns_series = self.returns
        else:
            returns_series = self.returns.mean(axis=1)
        
        # Skewness and Kurtosis
        skewness = stats.skew(returns_series)
        kurtosis = stats.kurtosis(returns_series)
        excess_kurtosis = kurtosis  # scipy returns excess kurtosis by default
        
        # Jarque-Bera test for normality
        jb_stat, jb_pvalue = stats.jarque_bera(returns_series)
        
        # Tail ratio (95th percentile / 5th percentile)
        percentile_95 = np.percentile(returns_series, 95)
        percentile_5 = np.percentile(returns_series, 5)
        tail_ratio = abs(percentile_95 / percentile_5) if percentile_5 != 0 else 0
        
        # Semi-variance (variance of negative returns)
        negative_returns = returns_series[returns_series < 0]
        semi_variance = negative_returns.var() if len(negative_returns) > 0 else 0
        
        return {
            'Skewness': skewness,
            'Kurtosis': kurtosis,
            'Excess_Kurtosis': excess_kurtosis,
            'Jarque_Bera_Stat': jb_stat,
            'Jarque_Bera_PValue': jb_pvalue,
            'Tail_Ratio': tail_ratio,
            'Semi_Variance': semi_variance
        }
    
    def rolling_risk_metrics(self, window=252):
        """
        Calculate rolling risk metrics
        
        Args:
            window (int): Rolling window size
        
        Returns:
            pandas.DataFrame: Rolling risk metrics
        """
        if isinstance(self.returns, pd.Series):
            returns_series = self.returns
        else:
            returns_series = self.returns.mean(axis=1)
        
        rolling_metrics = pd.DataFrame(index=returns_series.index)
        
        # Rolling volatility
        rolling_metrics['Rolling_Volatility'] = returns_series.rolling(window=window).std() * np.sqrt(252)
        
        # Rolling VaR (95%)
        rolling_metrics['Rolling_VaR_95'] = returns_series.rolling(window=window).quantile(0.05)
        
        # Rolling Sharpe Ratio
        rolling_returns = returns_series.rolling(window=window).mean() * 252
        rolling_vol = rolling_metrics['Rolling_Volatility']
        rolling_metrics['Rolling_Sharpe'] = (rolling_returns - 0.02) / rolling_vol
        
        # Rolling Maximum Drawdown
        rolling_max_dd = []
        for i in range(window, len(returns_series) + 1):  # Include the last point
            period_returns = returns_series.iloc[i-window:i]
            cumulative = (1 + period_returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            rolling_max_dd.append(drawdown.min())
        
        # Pad with NaN for the initial period to match index length
        rolling_max_dd = [np.nan] * (window - 1) + rolling_max_dd
        
        # Ensure the length matches the index
        if len(rolling_max_dd) != len(rolling_metrics.index):
            # Truncate or pad as needed
            if len(rolling_max_dd) > len(rolling_metrics.index):
                rolling_max_dd = rolling_max_dd[:len(rolling_metrics.index)]
            else:
                rolling_max_dd.extend([np.nan] * (len(rolling_metrics.index) - len(rolling_max_dd)))
                
        rolling_metrics['Rolling_Max_Drawdown'] = rolling_max_dd
        
        return rolling_metrics.dropna()
    
    def correlation_risk_analysis(self):
        """
        Analyze correlation-based risk (only for multi-asset portfolios)
        
        Returns:
            dict: Correlation risk metrics
        """
        if isinstance(self.returns, pd.Series):
            return {'error': 'Correlation analysis requires multi-asset data'}
        
        # Correlation matrix
        correlation_matrix = self.returns.corr()
        
        # Average correlation
        upper_triangle = correlation_matrix.where(np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1))
        avg_correlation = upper_triangle.stack().mean()
        
        # Maximum correlation
        max_correlation = upper_triangle.stack().max()
        
        # Minimum correlation
        min_correlation = upper_triangle.stack().min()
        
        # Eigenvalue analysis for concentration risk
        eigenvalues = np.linalg.eigvals(correlation_matrix)
        max_eigenvalue = eigenvalues.max()
        concentration_ratio = max_eigenvalue / len(eigenvalues)
        
        return {
            'Correlation_Matrix': correlation_matrix,
            'Average_Correlation': avg_correlation,
            'Maximum_Correlation': max_correlation,
            'Minimum_Correlation': min_correlation,
            'Concentration_Ratio': concentration_ratio,
            'Eigenvalues': eigenvalues
        }
    
    def plot_risk_analysis(self):
        """
        Create comprehensive risk analysis plots
        
        Returns:
            dict: Dictionary of plotly figures
        """
        plots = {}
        
        if isinstance(self.returns, pd.Series):
            returns_series = self.returns
        else:
            returns_series = self.returns.mean(axis=1)
        
        # 1. Returns distribution with VaR lines
        var_results = self.value_at_risk('historical')
        
        fig_dist = go.Figure()
        
        fig_dist.add_trace(go.Histogram(
            x=returns_series,
            nbinsx=50,
            name='Returns Distribution',
            opacity=0.7
        ))
        
        # Add VaR lines
        for var_level, var_value in var_results.items():
            fig_dist.add_vline(
                x=var_value,
                line_dash="dash",
                line_color="red",
                annotation_text=f"{var_level}: {var_value:.4f}"
            )
        
        fig_dist.update_layout(
            title='Returns Distribution with VaR Levels',
            xaxis_title='Returns',
            yaxis_title='Frequency',
            height=500
        )
        
        plots['returns_distribution'] = fig_dist
        
        # 2. Drawdown analysis
        dd_metrics = self.maximum_drawdown(returns_series)
        
        fig_dd = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Cumulative Returns', 'Drawdown'],
            vertical_spacing=0.1
        )
        
        # Use the correct keys from the actual method return
        cumulative_returns = dd_metrics.get('Cumulative_Returns', dd_metrics.get('cumulative_returns', pd.Series()))
        drawdown_series = dd_metrics.get('Drawdown_Series', dd_metrics.get('drawdown_series', pd.Series()))
        
        if not cumulative_returns.empty and not drawdown_series.empty:
            fig_dd.add_trace(
                go.Scatter(
                    x=cumulative_returns.index,
                    y=cumulative_returns,
                    mode='lines',
                    name='Cumulative Returns'
                ),
                row=1, col=1
            )
            
            fig_dd.add_trace(
                go.Scatter(
                    x=drawdown_series.index,
                    y=drawdown_series * 100,
                    mode='lines',
                    name='Drawdown (%)',
                    fill='tonexty',
                    line=dict(color='red')
                ),
                row=2, col=1
            )
        else:
            # Add empty traces if data is missing
            fig_dd.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Cumulative Returns'), row=1, col=1)
            fig_dd.add_trace(go.Scatter(x=[], y=[], mode='lines', name='Drawdown (%)'), row=2, col=1)
        
        fig_dd.update_layout(
            title='Drawdown Analysis',
            height=600,
            showlegend=False
        )
        
        plots['drawdown_analysis'] = fig_dd
        
        # 3. Rolling risk metrics
        if len(returns_series) > 252:
            try:
                rolling_metrics = self.rolling_risk_metrics()
                
                if not rolling_metrics.empty:
                    fig_rolling = make_subplots(
                        rows=3, cols=1,
                        subplot_titles=['Rolling Volatility', 'Rolling VaR (95%)', 'Rolling Sharpe Ratio'],
                        vertical_spacing=0.08
                    )
                    
                    # Add traces with error handling for length mismatches
                    if 'Rolling_Volatility' in rolling_metrics.columns:
                        vol_data = rolling_metrics['Rolling_Volatility'].dropna()
                        fig_rolling.add_trace(
                            go.Scatter(x=vol_data.index, y=vol_data.values, name='Volatility'),
                            row=1, col=1
                        )
                    
                    if 'Rolling_VaR_95' in rolling_metrics.columns:
                        var_data = rolling_metrics['Rolling_VaR_95'].dropna()
                        fig_rolling.add_trace(
                            go.Scatter(x=var_data.index, y=var_data.values, name='VaR 95%'),
                            row=2, col=1
                        )
                    
                    if 'Rolling_Sharpe' in rolling_metrics.columns:
                        sharpe_data = rolling_metrics['Rolling_Sharpe'].dropna()
                        fig_rolling.add_trace(
                            go.Scatter(x=sharpe_data.index, y=sharpe_data.values, name='Sharpe Ratio'),
                            row=3, col=1
                        )
                    
                    fig_rolling.update_layout(
                        title='Rolling Risk Metrics',
                        height=700,
                        showlegend=False
                    )
                    
                    plots['rolling_metrics'] = fig_rolling
                    
            except Exception as e:
                st.error(f"Error creating rolling metrics plot: {str(e)}")
                # Continue without this plot
        
        # 4. Correlation heatmap (for multi-asset portfolios)
        if not isinstance(self.returns, pd.Series):
            corr_analysis = self.correlation_risk_analysis()
            
            if 'error' not in corr_analysis:
                fig_corr = go.Figure(data=go.Heatmap(
                    z=corr_analysis['Correlation_Matrix'].values,
                    x=corr_analysis['Correlation_Matrix'].columns,
                    y=corr_analysis['Correlation_Matrix'].index,
                    colorscale='RdBu',
                    zmid=0,
                    text=corr_analysis['Correlation_Matrix'].round(3).values,
                    texttemplate="%{text}",
                    textfont={"size": 10}
                ))
                
                fig_corr.update_layout(
                    title='Correlation Matrix',
                    height=600
                )
                
                plots['correlation_matrix'] = fig_corr
        
        return plots
    
    def get_risk_summary(self):
        """
        Get comprehensive risk summary
        
        Returns:
            dict: Risk summary metrics
        """
        summary = {}
        
        # Basic risk metrics
        var_metrics = self.value_at_risk('historical')
        es_metrics = self.expected_shortfall('historical')
        dd_metrics = self.maximum_drawdown()
        ratios = self.risk_adjusted_ratios()
        tail_metrics = self.tail_risk_metrics()
        
        summary.update(var_metrics)
        summary.update(es_metrics)
        summary.update({
            'Max_Drawdown': dd_metrics['Max_Drawdown'],
            'Drawdown_Duration': dd_metrics['Drawdown_Duration']
        })
        summary.update(ratios)
        summary.update(tail_metrics)
        
        # Correlation metrics (if applicable)
        if not isinstance(self.returns, pd.Series):
            corr_analysis = self.correlation_risk_analysis()
            if 'error' not in corr_analysis:
                summary.update({
                    'Average_Correlation': corr_analysis['Average_Correlation'],
                    'Concentration_Ratio': corr_analysis['Concentration_Ratio']
                })
        
        return summary
