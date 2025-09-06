import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

class TradingStrategies:
    """Implementation of various trading strategies and backtesting"""
    
    def __init__(self, price_data, ticker):
        """
        Initialize with price data
        
        Args:
            price_data (pandas.DataFrame): OHLCV data
            ticker (str): Asset ticker
        """
        self.data = price_data.copy()
        self.ticker = ticker
        
        # Calculate additional indicators
        self.data['Returns'] = self.data['Close'].pct_change()
        self.data['Log_Returns'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
    
    def moving_average_strategy(self, short_window=20, long_window=50):
        """
        Moving Average Crossover Strategy
        
        Args:
            short_window (int): Short MA window
            long_window (int): Long MA window
        
        Returns:
            pandas.DataFrame: Strategy signals and performance
        """
        strategy_data = self.data.copy()
        
        # Calculate moving averages
        strategy_data['MA_Short'] = strategy_data['Close'].rolling(window=short_window).mean()
        strategy_data['MA_Long'] = strategy_data['Close'].rolling(window=long_window).mean()
        
        # Generate signals
        strategy_data['Signal'] = 0
        strategy_data.loc[strategy_data['MA_Short'] > strategy_data['MA_Long'], 'Signal'] = 1
        strategy_data.loc[strategy_data['MA_Short'] <= strategy_data['MA_Long'], 'Signal'] = -1
        
        # Calculate positions (when signals change)
        strategy_data['Position'] = strategy_data['Signal'].diff()
        
        # Calculate strategy returns
        strategy_data['Strategy_Returns'] = strategy_data['Signal'].shift(1) * strategy_data['Returns']
        
        return strategy_data.dropna()
    
    def rsi_strategy(self, rsi_window=14, oversold=30, overbought=70):
        """
        RSI Mean Reversion Strategy
        
        Args:
            rsi_window (int): RSI calculation window
            oversold (float): Oversold threshold
            overbought (float): Overbought threshold
        
        Returns:
            pandas.DataFrame: Strategy signals and performance
        """
        strategy_data = self.data.copy()
        
        # Calculate RSI
        delta = strategy_data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=rsi_window).mean()
        avg_loss = loss.rolling(window=rsi_window).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        strategy_data['RSI'] = rsi
        
        # Generate signals
        strategy_data['Signal'] = 0
        strategy_data.loc[strategy_data['RSI'] < oversold, 'Signal'] = 1  # Buy signal
        strategy_data.loc[strategy_data['RSI'] > overbought, 'Signal'] = -1  # Sell signal
        
        # Hold position until opposite signal
        strategy_data['Position'] = strategy_data['Signal'].replace(0, np.nan).fillna(method='ffill').fillna(0)
        
        # Calculate strategy returns
        strategy_data['Strategy_Returns'] = strategy_data['Position'].shift(1) * strategy_data['Returns']
        
        return strategy_data.dropna()
    
    def bollinger_bands_strategy(self, window=20, num_std=2):
        """
        Bollinger Bands Mean Reversion Strategy
        
        Args:
            window (int): Moving average window
            num_std (float): Number of standard deviations
        
        Returns:
            pandas.DataFrame: Strategy signals and performance
        """
        strategy_data = self.data.copy()
        
        # Calculate Bollinger Bands
        strategy_data['MA'] = strategy_data['Close'].rolling(window=window).mean()
        strategy_data['Std'] = strategy_data['Close'].rolling(window=window).std()
        strategy_data['Upper_Band'] = strategy_data['MA'] + (strategy_data['Std'] * num_std)
        strategy_data['Lower_Band'] = strategy_data['MA'] - (strategy_data['Std'] * num_std)
        
        # Generate signals
        strategy_data['Signal'] = 0
        strategy_data.loc[strategy_data['Close'] <= strategy_data['Lower_Band'], 'Signal'] = 1  # Buy
        strategy_data.loc[strategy_data['Close'] >= strategy_data['Upper_Band'], 'Signal'] = -1  # Sell
        strategy_data.loc[strategy_data['Close'] == strategy_data['MA'], 'Signal'] = 0  # Exit
        
        # Forward fill positions
        strategy_data['Position'] = strategy_data['Signal'].replace(0, np.nan).fillna(method='ffill').fillna(0)
        
        # Calculate strategy returns
        strategy_data['Strategy_Returns'] = strategy_data['Position'].shift(1) * strategy_data['Returns']
        
        return strategy_data.dropna()
    
    def momentum_strategy(self, lookback=10, holding_period=5):
        """
        Momentum Strategy
        
        Args:
            lookback (int): Lookback period for momentum calculation
            holding_period (int): Holding period for positions
        
        Returns:
            pandas.DataFrame: Strategy signals and performance
        """
        strategy_data = self.data.copy()
        
        # Calculate momentum
        strategy_data['Momentum'] = strategy_data['Close'].pct_change(lookback)
        
        # Generate signals based on momentum
        strategy_data['Signal'] = 0
        strategy_data.loc[strategy_data['Momentum'] > 0, 'Signal'] = 1  # Long
        strategy_data.loc[strategy_data['Momentum'] <= 0, 'Signal'] = -1  # Short
        
        # Implement holding period
        strategy_data['Position'] = 0
        for i in range(len(strategy_data)):
            if i >= lookback:
                signal = strategy_data['Signal'].iloc[i]
                end_period = min(i + holding_period, len(strategy_data))
                strategy_data.iloc[i:end_period, strategy_data.columns.get_loc('Position')] = signal
        
        # Calculate strategy returns
        strategy_data['Strategy_Returns'] = strategy_data['Position'].shift(1) * strategy_data['Returns']
        
        return strategy_data.dropna()
    
    def mean_reversion_strategy(self, window=20, threshold=1.5):
        """
        Mean Reversion Strategy based on z-score
        
        Args:
            window (int): Rolling window for mean and std calculation
            threshold (float): Z-score threshold for entry
        
        Returns:
            pandas.DataFrame: Strategy signals and performance
        """
        strategy_data = self.data.copy()
        
        # Calculate rolling mean and std
        strategy_data['Rolling_Mean'] = strategy_data['Close'].rolling(window=window).mean()
        strategy_data['Rolling_Std'] = strategy_data['Close'].rolling(window=window).std()
        
        # Calculate z-score
        strategy_data['Z_Score'] = (strategy_data['Close'] - strategy_data['Rolling_Mean']) / strategy_data['Rolling_Std']
        
        # Generate signals
        strategy_data['Signal'] = 0
        strategy_data.loc[strategy_data['Z_Score'] < -threshold, 'Signal'] = 1  # Buy when undervalued
        strategy_data.loc[strategy_data['Z_Score'] > threshold, 'Signal'] = -1  # Sell when overvalued
        strategy_data.loc[abs(strategy_data['Z_Score']) < 0.5, 'Signal'] = 0  # Exit when near mean
        
        # Forward fill positions
        strategy_data['Position'] = strategy_data['Signal'].replace(0, np.nan).fillna(method='ffill').fillna(0)
        
        # Calculate strategy returns
        strategy_data['Strategy_Returns'] = strategy_data['Position'].shift(1) * strategy_data['Returns']
        
        return strategy_data.dropna()
    
    def backtest_strategy(self, strategy_data, initial_capital=100000, transaction_cost=0.001):
        """
        Comprehensive backtesting with performance metrics
        
        Args:
            strategy_data (pandas.DataFrame): Strategy data with returns
            initial_capital (float): Initial capital
            transaction_cost (float): Transaction cost as percentage
        
        Returns:
            dict: Comprehensive backtesting results
        """
        # Calculate cumulative returns
        strategy_data['Cumulative_Returns'] = (1 + strategy_data['Strategy_Returns']).cumprod()
        strategy_data['Cumulative_Market'] = (1 + strategy_data['Returns']).cumprod()
        
        # Account for transaction costs
        position_changes = strategy_data['Signal'].diff().abs()
        transaction_costs = position_changes * transaction_cost
        strategy_data['Net_Strategy_Returns'] = strategy_data['Strategy_Returns'] - transaction_costs
        strategy_data['Net_Cumulative_Returns'] = (1 + strategy_data['Net_Strategy_Returns']).cumprod()
        
        # Calculate performance metrics
        total_return = strategy_data['Cumulative_Returns'].iloc[-1] - 1
        market_return = strategy_data['Cumulative_Market'].iloc[-1] - 1
        excess_return = total_return - market_return
        
        # Annualized metrics
        n_years = len(strategy_data) / 252
        annualized_return = (1 + total_return) ** (1/n_years) - 1
        annualized_volatility = strategy_data['Strategy_Returns'].std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
        
        # Market comparison
        market_volatility = strategy_data['Returns'].std() * np.sqrt(252)
        market_sharpe = (market_return ** (1/n_years) - 1) / market_volatility if market_volatility != 0 else 0
        
        # Drawdown analysis
        rolling_max = strategy_data['Cumulative_Returns'].expanding().max()
        drawdown = (strategy_data['Cumulative_Returns'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Trade analysis
        signals = strategy_data['Signal'].diff().abs()
        total_trades = signals.sum() / 2  # Each trade involves entry and exit
        
        winning_trades = strategy_data[strategy_data['Strategy_Returns'] > 0]['Strategy_Returns']
        losing_trades = strategy_data[strategy_data['Strategy_Returns'] < 0]['Strategy_Returns']
        
        win_rate = len(winning_trades) / (len(winning_trades) + len(losing_trades)) if (len(winning_trades) + len(losing_trades)) > 0 else 0
        
        avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'Total_Return': total_return,
            'Market_Return': market_return,
            'Excess_Return': excess_return,
            'Annualized_Return': annualized_return,
            'Annualized_Volatility': annualized_volatility,
            'Sharpe_Ratio': sharpe_ratio,
            'Market_Sharpe': market_sharpe,
            'Max_Drawdown': max_drawdown,
            'Calmar_Ratio': calmar_ratio,
            'Total_Trades': total_trades,
            'Win_Rate': win_rate,
            'Profit_Factor': profit_factor,
            'Average_Win': avg_win,
            'Average_Loss': avg_loss,
            'Strategy_Data': strategy_data
        }
    
    def plot_strategy_performance(self, results, strategy_name="Strategy"):
        """
        Create comprehensive strategy performance plots
        
        Args:
            results (dict): Backtesting results
            strategy_name (str): Name of the strategy
        
        Returns:
            dict: Dictionary of plotly figures
        """
        plots = {}
        strategy_data = results['Strategy_Data']
        
        # 1. Cumulative returns comparison
        fig_returns = go.Figure()
        
        fig_returns.add_trace(go.Scatter(
            x=strategy_data.index,
            y=strategy_data['Cumulative_Returns'],
            mode='lines',
            name=strategy_name,
            line=dict(color='blue', width=2)
        ))
        
        fig_returns.add_trace(go.Scatter(
            x=strategy_data.index,
            y=strategy_data['Cumulative_Market'],
            mode='lines',
            name='Buy & Hold',
            line=dict(color='red', width=2)
        ))
        
        fig_returns.update_layout(
            title=f'{strategy_name} vs Buy & Hold - Cumulative Returns',
            xaxis_title='Date',
            yaxis_title='Cumulative Return',
            height=500
        )
        
        plots['cumulative_returns'] = fig_returns
        
        # 2. Price and signals
        fig_signals = go.Figure()
        
        # Price
        fig_signals.add_trace(go.Scatter(
            x=strategy_data.index,
            y=strategy_data['Close'],
            mode='lines',
            name='Price',
            line=dict(color='black', width=1)
        ))
        
        # Buy signals
        buy_signals = strategy_data[strategy_data['Position'] == 1]
        if not buy_signals.empty:
            fig_signals.add_trace(go.Scatter(
                x=buy_signals.index,
                y=buy_signals['Close'],
                mode='markers',
                name='Buy Signal',
                marker=dict(color='green', size=8, symbol='triangle-up')
            ))
        
        # Sell signals
        sell_signals = strategy_data[strategy_data['Position'] == -1]
        if not sell_signals.empty:
            fig_signals.add_trace(go.Scatter(
                x=sell_signals.index,
                y=sell_signals['Close'],
                mode='markers',
                name='Sell Signal',
                marker=dict(color='red', size=8, symbol='triangle-down')
            ))
        
        fig_signals.update_layout(
            title=f'{strategy_name} - Trading Signals',
            xaxis_title='Date',
            yaxis_title='Price',
            height=500
        )
        
        plots['signals'] = fig_signals
        
        # 3. Drawdown analysis
        rolling_max = strategy_data['Cumulative_Returns'].expanding().max()
        drawdown = (strategy_data['Cumulative_Returns'] - rolling_max) / rolling_max
        
        fig_drawdown = go.Figure()
        
        fig_drawdown.add_trace(go.Scatter(
            x=strategy_data.index,
            y=drawdown * 100,
            mode='lines',
            name='Drawdown',
            fill='tonexty',
            line=dict(color='red', width=1)
        ))
        
        fig_drawdown.update_layout(
            title=f'{strategy_name} - Drawdown Analysis',
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            height=400
        )
        
        plots['drawdown'] = fig_drawdown
        
        # 4. Rolling performance metrics
        rolling_window = 252  # 1 year
        if len(strategy_data) > rolling_window:
            rolling_returns = strategy_data['Strategy_Returns'].rolling(window=rolling_window).mean() * 252
            rolling_volatility = strategy_data['Strategy_Returns'].rolling(window=rolling_window).std() * np.sqrt(252)
            rolling_sharpe = rolling_returns / rolling_volatility
            
            fig_rolling = make_subplots(
                rows=3, cols=1,
                subplot_titles=['Rolling Returns', 'Rolling Volatility', 'Rolling Sharpe Ratio'],
                vertical_spacing=0.08
            )
            
            fig_rolling.add_trace(
                go.Scatter(x=strategy_data.index, y=rolling_returns, name='Rolling Returns'),
                row=1, col=1
            )
            
            fig_rolling.add_trace(
                go.Scatter(x=strategy_data.index, y=rolling_volatility, name='Rolling Volatility'),
                row=2, col=1
            )
            
            fig_rolling.add_trace(
                go.Scatter(x=strategy_data.index, y=rolling_sharpe, name='Rolling Sharpe'),
                row=3, col=1
            )
            
            fig_rolling.update_layout(
                title=f'{strategy_name} - Rolling Performance Metrics',
                height=700,
                showlegend=False
            )
            
            plots['rolling_metrics'] = fig_rolling
        
        return plots
    
    def compare_strategies(self, strategies_results):
        """
        Compare multiple strategies
        
        Args:
            strategies_results (dict): Dictionary of strategy results
        
        Returns:
            pandas.DataFrame: Comparison table
        """
        comparison_data = []
        
        for strategy_name, results in strategies_results.items():
            comparison_data.append({
                'Strategy': strategy_name,
                'Total Return': f"{results['Total_Return']:.2%}",
                'Annualized Return': f"{results['Annualized_Return']:.2%}",
                'Volatility': f"{results['Annualized_Volatility']:.2%}",
                'Sharpe Ratio': f"{results['Sharpe_Ratio']:.3f}",
                'Max Drawdown': f"{results['Max_Drawdown']:.2%}",
                'Calmar Ratio': f"{results['Calmar_Ratio']:.3f}",
                'Win Rate': f"{results['Win_Rate']:.2%}",
                'Total Trades': f"{results['Total_Trades']:.0f}"
            })
        
        return pd.DataFrame(comparison_data)
