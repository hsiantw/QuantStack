import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from scipy import stats
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesAnalysis:
    """Advanced time series analysis for financial data"""
    
    def __init__(self, price_data, ticker):
        """
        Initialize with price data for a single asset
        
        Args:
            price_data (pandas.Series): Price data for single asset
            ticker (str): Asset ticker symbol
        """
        self.prices = price_data.dropna()
        self.returns = price_data.pct_change().dropna()
        self.ticker = ticker
    
    def stationarity_test(self):
        """
        Perform Augmented Dickey-Fuller test for stationarity
        
        Returns:
            dict: Test results
        """
        # Test on prices
        adf_prices = adfuller(self.prices.dropna())
        
        # Test on returns
        adf_returns = adfuller(self.returns.dropna())
        
        return {
            'prices': {
                'adf_statistic': adf_prices[0],
                'p_value': adf_prices[1],
                'critical_values': adf_prices[4],
                'is_stationary': adf_prices[1] < 0.05
            },
            'returns': {
                'adf_statistic': adf_returns[0],
                'p_value': adf_returns[1],
                'critical_values': adf_returns[4],
                'is_stationary': adf_returns[1] < 0.05
            }
        }
    
    def seasonal_decomposition(self, period=252):
        """
        Perform seasonal decomposition
        
        Args:
            period (int): Seasonal period (252 for annual in daily data)
        
        Returns:
            dict: Decomposition components
        """
        try:
            if len(self.prices) < 2 * period:
                period = max(30, len(self.prices) // 4)
            
            decomposition = seasonal_decompose(self.prices, model='multiplicative', period=period)
            
            return {
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid,
                'original': self.prices
            }
        except Exception as e:
            st.error(f"Error in seasonal decomposition: {str(e)}")
            return None
    
    def detect_trends(self, window=30):
        """
        Detect trends using moving averages and trend lines
        
        Args:
            window (int): Moving average window
        
        Returns:
            dict: Trend analysis results
        """
        # Moving averages
        ma_short = self.prices.rolling(window=window).mean()
        ma_long = self.prices.rolling(window=window*2).mean()
        
        # Trend signals
        trend_signal = np.where(ma_short > ma_long, 1, -1)
        
        # Linear trend
        x = np.arange(len(self.prices))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, self.prices)
        trend_line = slope * x + intercept
        
        # Trend strength
        trend_strength = abs(r_value)
        trend_direction = "Upward" if slope > 0 else "Downward"
        
        return {
            'ma_short': ma_short,
            'ma_long': ma_long,
            'trend_signal': trend_signal,
            'linear_trend': pd.Series(trend_line, index=self.prices.index),
            'slope': slope,
            'r_squared': r_value**2,
            'trend_strength': trend_strength,
            'trend_direction': trend_direction,
            'p_value': p_value
        }
    
    def volatility_analysis(self, windows=[21, 63, 252]):
        """
        Calculate rolling volatility with different windows
        
        Args:
            windows (list): List of rolling windows
        
        Returns:
            dict: Volatility analysis
        """
        volatilities = {}
        
        for window in windows:
            vol = self.returns.rolling(window=window).std() * np.sqrt(252)
            volatilities[f'{window}d'] = vol
        
        # GARCH-like volatility clustering detection
        squared_returns = self.returns ** 2
        volatility_clustering = squared_returns.rolling(window=21).corr(squared_returns.shift(1))
        
        return {
            'volatilities': volatilities,
            'current_volatility': self.returns.rolling(window=21).std().iloc[-1] * np.sqrt(252),
            'volatility_clustering': volatility_clustering.dropna()
        }
    
    def arima_forecast(self, steps=30, order=(1,1,1)):
        """
        ARIMA model forecasting
        
        Args:
            steps (int): Number of steps to forecast
            order (tuple): ARIMA order (p,d,q)
        
        Returns:
            dict: Forecast results
        """
        try:
            # Use returns for ARIMA (more stationary)
            model = ARIMA(self.returns.dropna(), order=order)
            fitted_model = model.fit()
            
            # Forecast returns
            forecast_returns = fitted_model.forecast(steps=steps)
            forecast_std = fitted_model.forecast(steps=steps, alpha=0.05)[1]  # 95% confidence
            
            # Convert back to prices
            last_price = self.prices.iloc[-1]
            forecast_prices = []
            price = last_price
            
            for ret in forecast_returns:
                price = price * (1 + ret)
                forecast_prices.append(price)
            
            # Create forecast index
            last_date = self.prices.index[-1]
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                         periods=steps, freq='D')
            
            forecast_series = pd.Series(forecast_prices, index=forecast_dates)
            
            return {
                'forecast_prices': forecast_series,
                'forecast_returns': pd.Series(forecast_returns, index=forecast_dates),
                'model_summary': fitted_model.summary(),
                'aic': fitted_model.aic,
                'bic': fitted_model.bic
            }
            
        except Exception as e:
            st.error(f"Error in ARIMA forecasting: {str(e)}")
            return None
    
    def support_resistance_levels(self, window=20, threshold=0.02):
        """
        Identify support and resistance levels
        
        Args:
            window (int): Window for local minima/maxima
            threshold (float): Minimum price change threshold
        
        Returns:
            dict: Support and resistance levels
        """
        # Find local minima and maxima
        rolling_min = self.prices.rolling(window=window, center=True).min()
        rolling_max = self.prices.rolling(window=window, center=True).max()
        
        support_levels = []
        resistance_levels = []
        
        for i in range(window, len(self.prices) - window):
            price = self.prices.iloc[i]
            
            # Check if it's a local minimum (support)
            if price == rolling_min.iloc[i]:
                # Verify it's significant
                nearby_prices = self.prices.iloc[i-window:i+window]
                if (nearby_prices.max() - price) / price > threshold:
                    support_levels.append({
                        'date': self.prices.index[i],
                        'level': price,
                        'strength': len(nearby_prices[nearby_prices <= price * 1.01])
                    })
            
            # Check if it's a local maximum (resistance)
            if price == rolling_max.iloc[i]:
                # Verify it's significant
                nearby_prices = self.prices.iloc[i-window:i+window]
                if (price - nearby_prices.min()) / nearby_prices.min() > threshold:
                    resistance_levels.append({
                        'date': self.prices.index[i],
                        'level': price,
                        'strength': len(nearby_prices[nearby_prices >= price * 0.99])
                    })
        
        return {
            'support_levels': support_levels,
            'resistance_levels': resistance_levels
        }
    
    def plot_comprehensive_analysis(self):
        """
        Create comprehensive time series analysis plots
        
        Returns:
            dict: Dictionary of plotly figures
        """
        plots = {}
        
        # 1. Price and trend analysis
        trend_data = self.detect_trends()
        
        fig_price = go.Figure()
        
        # Price
        fig_price.add_trace(go.Scatter(
            x=self.prices.index,
            y=self.prices,
            mode='lines',
            name='Price',
            line=dict(color='blue', width=2)
        ))
        
        # Moving averages
        fig_price.add_trace(go.Scatter(
            x=trend_data['ma_short'].index,
            y=trend_data['ma_short'],
            mode='lines',
            name='MA 30',
            line=dict(color='orange', width=1)
        ))
        
        fig_price.add_trace(go.Scatter(
            x=trend_data['ma_long'].index,
            y=trend_data['ma_long'],
            mode='lines',
            name='MA 60',
            line=dict(color='red', width=1)
        ))
        
        # Linear trend
        fig_price.add_trace(go.Scatter(
            x=trend_data['linear_trend'].index,
            y=trend_data['linear_trend'],
            mode='lines',
            name='Linear Trend',
            line=dict(color='green', width=2, dash='dash')
        ))
        
        fig_price.update_layout(
            title=f'{self.ticker} - Price and Trend Analysis',
            xaxis_title='Date',
            yaxis_title='Price',
            height=500
        )
        
        plots['price_trend'] = fig_price
        
        # 2. Seasonal decomposition
        decomp = self.seasonal_decomposition()
        if decomp:
            fig_decomp = make_subplots(
                rows=4, cols=1,
                subplot_titles=['Original', 'Trend', 'Seasonal', 'Residual'],
                vertical_spacing=0.08
            )
            
            fig_decomp.add_trace(
                go.Scatter(x=decomp['original'].index, y=decomp['original'], name='Original'),
                row=1, col=1
            )
            
            fig_decomp.add_trace(
                go.Scatter(x=decomp['trend'].index, y=decomp['trend'], name='Trend'),
                row=2, col=1
            )
            
            fig_decomp.add_trace(
                go.Scatter(x=decomp['seasonal'].index, y=decomp['seasonal'], name='Seasonal'),
                row=3, col=1
            )
            
            fig_decomp.add_trace(
                go.Scatter(x=decomp['residual'].index, y=decomp['residual'], name='Residual'),
                row=4, col=1
            )
            
            fig_decomp.update_layout(
                title=f'{self.ticker} - Seasonal Decomposition',
                height=800,
                showlegend=False
            )
            
            plots['decomposition'] = fig_decomp
        
        # 3. Volatility analysis
        vol_data = self.volatility_analysis()
        
        fig_vol = go.Figure()
        
        for window, vol in vol_data['volatilities'].items():
            fig_vol.add_trace(go.Scatter(
                x=vol.index,
                y=vol,
                mode='lines',
                name=f'Volatility {window}'
            ))
        
        fig_vol.update_layout(
            title=f'{self.ticker} - Rolling Volatility',
            xaxis_title='Date',
            yaxis_title='Annualized Volatility',
            height=400
        )
        
        plots['volatility'] = fig_vol
        
        # 4. Returns distribution
        fig_returns = go.Figure()
        
        fig_returns.add_trace(go.Histogram(
            x=self.returns,
            nbinsx=50,
            name='Returns Distribution',
            opacity=0.7
        ))
        
        # Normal distribution overlay
        mu, sigma = stats.norm.fit(self.returns)
        x = np.linspace(self.returns.min(), self.returns.max(), 100)
        y = stats.norm.pdf(x, mu, sigma)
        
        fig_returns.add_trace(go.Scatter(
            x=x,
            y=y * len(self.returns) * (self.returns.max() - self.returns.min()) / 50,
            mode='lines',
            name='Normal Fit',
            line=dict(color='red', width=2)
        ))
        
        fig_returns.update_layout(
            title=f'{self.ticker} - Returns Distribution',
            xaxis_title='Returns',
            yaxis_title='Frequency',
            height=400
        )
        
        plots['returns_distribution'] = fig_returns
        
        # 5. ARIMA forecast
        forecast_data = self.arima_forecast()
        if forecast_data:
            fig_forecast = go.Figure()
            
            # Historical prices
            fig_forecast.add_trace(go.Scatter(
                x=self.prices.index,
                y=self.prices,
                mode='lines',
                name='Historical Prices',
                line=dict(color='blue', width=2)
            ))
            
            # Forecast
            fig_forecast.add_trace(go.Scatter(
                x=forecast_data['forecast_prices'].index,
                y=forecast_data['forecast_prices'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig_forecast.update_layout(
                title=f'{self.ticker} - ARIMA Price Forecast',
                xaxis_title='Date',
                yaxis_title='Price',
                height=500
            )
            
            plots['forecast'] = fig_forecast
        
        return plots
    
    def get_analysis_summary(self):
        """
        Get comprehensive analysis summary
        
        Returns:
            dict: Analysis summary
        """
        # Stationarity test
        stationarity = self.stationarity_test()
        
        # Trend analysis
        trend_data = self.detect_trends()
        
        # Volatility analysis
        vol_data = self.volatility_analysis()
        
        # Basic statistics
        price_stats = {
            'mean': self.prices.mean(),
            'std': self.prices.std(),
            'min': self.prices.min(),
            'max': self.prices.max(),
            'current': self.prices.iloc[-1]
        }
        
        return_stats = {
            'mean': self.returns.mean(),
            'std': self.returns.std(),
            'skewness': stats.skew(self.returns),
            'kurtosis': stats.kurtosis(self.returns),
            'jarque_bera': stats.jarque_bera(self.returns)
        }
        
        return {
            'price_statistics': price_stats,
            'return_statistics': return_stats,
            'stationarity': stationarity,
            'trend_analysis': {
                'direction': trend_data['trend_direction'],
                'strength': trend_data['trend_strength'],
                'slope': trend_data['slope'],
                'r_squared': trend_data['r_squared']
            },
            'volatility': {
                'current': vol_data['current_volatility'],
                'volatility_clustering': vol_data['volatility_clustering'].mean()
            }
        }
