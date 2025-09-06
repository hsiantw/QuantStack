"""
Advanced Trading Strategy Optimization Framework
Compares multiple strategies with different indicator combinations to find optimal configurations.
"""

import pandas as pd
import numpy as np
import itertools
from typing import Dict, List, Tuple, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class StrategyOptimizer:
    """
    Advanced strategy optimization engine that tests multiple indicator combinations
    and finds the best performing strategies across different market conditions.
    """
    
    def __init__(self, ohlcv_data: pd.DataFrame, initial_capital: float = 100000):
        self.data = ohlcv_data.copy()
        self.initial_capital = initial_capital
        self.results = {}
        
        # Calculate basic indicators
        self._calculate_indicators()
    
    def _calculate_indicators(self):
        """Calculate all technical indicators needed for strategy combinations."""
        # Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            self.data[f'MA_{period}'] = self.data['Close'].rolling(window=period).mean()
        
        # Exponential Moving Averages
        for period in [12, 26, 50]:
            self.data[f'EMA_{period}'] = self.data['Close'].ewm(span=period).mean()
        
        # RSI
        for period in [14, 21, 30]:
            delta = self.data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / np.where(loss != 0, loss, 1e-6)
            self.data[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD
        self.data['MACD'] = self.data['EMA_12'] - self.data['EMA_26']
        self.data['MACD_Signal'] = self.data['MACD'].ewm(span=9).mean()
        self.data['MACD_Histogram'] = self.data['MACD'] - self.data['MACD_Signal']
        
        # Bollinger Bands
        for period in [20, 50]:
            ma = self.data['Close'].rolling(window=period).mean()
            std = self.data['Close'].rolling(window=period).std()
            self.data[f'BB_Upper_{period}'] = ma + (2 * std)
            self.data[f'BB_Lower_{period}'] = ma - (2 * std)
            self.data[f'BB_Middle_{period}'] = ma
            self.data[f'BB_Width_{period}'] = (self.data[f'BB_Upper_{period}'] - self.data[f'BB_Lower_{period}']) / np.where(ma != 0, ma, 1e-6)
        
        # Stochastic Oscillator
        for period in [14, 21]:
            low_min = self.data['Low'].rolling(window=period).min()
            high_max = self.data['High'].rolling(window=period).max()
            stoch_denominator = high_max - low_min
            self.data[f'Stoch_K_{period}'] = 100 * (self.data['Close'] - low_min) / np.where(stoch_denominator != 0, stoch_denominator, 1e-6)
            self.data[f'Stoch_D_{period}'] = self.data[f'Stoch_K_{period}'].rolling(window=3).mean()
        
        # Williams %R
        for period in [14, 21]:
            high_max = self.data['High'].rolling(window=period).max()
            low_min = self.data['Low'].rolling(window=period).min()
            williams_denominator = high_max - low_min
            self.data[f'Williams_R_{period}'] = -100 * (high_max - self.data['Close']) / np.where(williams_denominator != 0, williams_denominator, 1e-6)
        
        # Average True Range (ATR)
        high_low = self.data['High'] - self.data['Low']
        high_close = np.abs(self.data['High'] - self.data['Close'].shift())
        low_close = np.abs(self.data['Low'] - self.data['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        for period in [14, 21]:
            self.data[f'ATR_{period}'] = tr.rolling(window=period).mean()
        
        # Commodity Channel Index (CCI)
        for period in [20, 30]:
            tp = (self.data['High'] + self.data['Low'] + self.data['Close']) / 3
            sma = tp.rolling(window=period).mean()
            mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
            self.data[f'CCI_{period}'] = (tp - sma) / np.where((0.015 * mad) != 0, (0.015 * mad), 1e-6)
        
        # Volume indicators
        if 'Volume' in self.data.columns:
            # On-Balance Volume
            obv = [0]
            for i in range(1, len(self.data)):
                if self.data['Close'].iloc[i] > self.data['Close'].iloc[i-1]:
                    obv.append(obv[-1] + self.data['Volume'].iloc[i])
                elif self.data['Close'].iloc[i] < self.data['Close'].iloc[i-1]:
                    obv.append(obv[-1] - self.data['Volume'].iloc[i])
                else:
                    obv.append(obv[-1])
            self.data['OBV'] = obv
            
            # Volume Moving Average
            self.data['Volume_MA_20'] = self.data['Volume'].rolling(window=20).mean()
    
    def _get_strategy_combinations(self) -> List[Dict]:
        """Define all strategy combinations to test."""
        strategies = [
            # Single Indicator Strategies
            {
                'name': 'MA_Cross_Fast',
                'type': 'ma_crossover',
                'params': {'short_period': 10, 'long_period': 20}
            },
            {
                'name': 'MA_Cross_Medium',
                'type': 'ma_crossover',
                'params': {'short_period': 20, 'long_period': 50}
            },
            {
                'name': 'MA_Cross_Slow',
                'type': 'ma_crossover',
                'params': {'short_period': 50, 'long_period': 200}
            },
            {
                'name': 'RSI_Oversold',
                'type': 'rsi_mean_reversion',
                'params': {'period': 14, 'oversold': 30, 'overbought': 70}
            },
            {
                'name': 'RSI_Extreme',
                'type': 'rsi_mean_reversion',
                'params': {'period': 14, 'oversold': 20, 'overbought': 80}
            },
            {
                'name': 'BB_Reversal',
                'type': 'bollinger_bands',
                'params': {'period': 20, 'std_dev': 2}
            },
            {
                'name': 'MACD_Signal',
                'type': 'macd',
                'params': {'fast': 12, 'slow': 26, 'signal': 9}
            },
            
            # Multi-Indicator Combinations
            {
                'name': 'MA_RSI_Combo',
                'type': 'ma_rsi_combo',
                'params': {'ma_short': 20, 'ma_long': 50, 'rsi_period': 14, 'rsi_oversold': 30, 'rsi_overbought': 70}
            },
            {
                'name': 'Triple_MA',
                'type': 'triple_ma',
                'params': {'fast': 10, 'medium': 20, 'slow': 50}
            },
            {
                'name': 'RSI_BB_Combo',
                'type': 'rsi_bb_combo',
                'params': {'rsi_period': 14, 'bb_period': 20, 'rsi_oversold': 30, 'rsi_overbought': 70}
            },
            {
                'name': 'MACD_RSI_Combo',
                'type': 'macd_rsi_combo',
                'params': {'rsi_period': 14, 'rsi_oversold': 35, 'rsi_overbought': 65}
            },
            {
                'name': 'Stoch_RSI_Combo',
                'type': 'stoch_rsi_combo',
                'params': {'stoch_period': 14, 'rsi_period': 14}
            },
            {
                'name': 'Momentum_Multi',
                'type': 'momentum_multi',
                'params': {'rsi_period': 14, 'williams_period': 14, 'cci_period': 20}
            },
            {
                'name': 'Trend_Momentum',
                'type': 'trend_momentum',
                'params': {'ma_period': 50, 'rsi_period': 14, 'atr_period': 14}
            }
        ]
        
        return strategies
    
    def _execute_strategy(self, strategy: Dict) -> Dict:
        """Execute a specific strategy and return results."""
        signals = pd.Series(0, index=self.data.index)
        
        if strategy['type'] == 'ma_crossover':
            short_ma = self.data[f"MA_{strategy['params']['short_period']}"]
            long_ma = self.data[f"MA_{strategy['params']['long_period']}"]
            signals = np.where(short_ma > long_ma, 1, -1)
            
        elif strategy['type'] == 'rsi_mean_reversion':
            rsi = self.data[f"RSI_{strategy['params']['period']}"]
            signals = np.where(rsi < strategy['params']['oversold'], 1,
                             np.where(rsi > strategy['params']['overbought'], -1, 0))
            
        elif strategy['type'] == 'bollinger_bands':
            bb_upper = self.data[f"BB_Upper_{strategy['params']['period']}"]
            bb_lower = self.data[f"BB_Lower_{strategy['params']['period']}"]
            signals = np.where(self.data['Close'] < bb_lower, 1,
                             np.where(self.data['Close'] > bb_upper, -1, 0))
            
        elif strategy['type'] == 'macd':
            signals = np.where(self.data['MACD'] > self.data['MACD_Signal'], 1, -1)
            
        elif strategy['type'] == 'ma_rsi_combo':
            short_ma = self.data[f"MA_{strategy['params']['ma_short']}"]
            long_ma = self.data[f"MA_{strategy['params']['ma_long']}"]
            rsi = self.data[f"RSI_{strategy['params']['rsi_period']}"]
            
            ma_signal = short_ma > long_ma
            rsi_oversold = rsi < strategy['params']['rsi_oversold']
            rsi_overbought = rsi > strategy['params']['rsi_overbought']
            
            signals = np.where(ma_signal & rsi_oversold, 1,
                             np.where(~ma_signal | rsi_overbought, -1, 0))
            
        elif strategy['type'] == 'triple_ma':
            fast_ma = self.data[f"MA_{strategy['params']['fast']}"]
            medium_ma = self.data[f"MA_{strategy['params']['medium']}"]
            slow_ma = self.data[f"MA_{strategy['params']['slow']}"]
            
            signals = np.where((fast_ma > medium_ma) & (medium_ma > slow_ma), 1,
                             np.where((fast_ma < medium_ma) & (medium_ma < slow_ma), -1, 0))
            
        elif strategy['type'] == 'rsi_bb_combo':
            rsi = self.data[f"RSI_{strategy['params']['rsi_period']}"]
            bb_upper = self.data[f"BB_Upper_{strategy['params']['bb_period']}"]
            bb_lower = self.data[f"BB_Lower_{strategy['params']['bb_period']}"]
            
            signals = np.where((rsi < strategy['params']['rsi_oversold']) & 
                             (self.data['Close'] < bb_lower), 1,
                             np.where((rsi > strategy['params']['rsi_overbought']) & 
                                    (self.data['Close'] > bb_upper), -1, 0))
            
        elif strategy['type'] == 'macd_rsi_combo':
            rsi = self.data[f"RSI_{strategy['params']['rsi_period']}"]
            macd_signal = self.data['MACD'] > self.data['MACD_Signal']
            
            signals = np.where(macd_signal & (rsi < strategy['params']['rsi_overbought']), 1,
                             np.where(~macd_signal | (rsi > strategy['params']['rsi_overbought']), -1, 0))
            
        elif strategy['type'] == 'stoch_rsi_combo':
            stoch_k = self.data[f"Stoch_K_{strategy['params']['stoch_period']}"]
            rsi = self.data[f"RSI_{strategy['params']['rsi_period']}"]
            
            signals = np.where((stoch_k < 20) & (rsi < 30), 1,
                             np.where((stoch_k > 80) & (rsi > 70), -1, 0))
            
        elif strategy['type'] == 'momentum_multi':
            rsi = self.data[f"RSI_{strategy['params']['rsi_period']}"]
            williams = self.data[f"Williams_R_{strategy['params']['williams_period']}"]
            cci = self.data[f"CCI_{strategy['params']['cci_period']}"]
            
            bullish = (rsi < 30) & (williams < -80) & (cci < -100)
            bearish = (rsi > 70) & (williams > -20) & (cci > 100)
            signals = np.where(bullish, 1, np.where(bearish, -1, 0))
            
        elif strategy['type'] == 'trend_momentum':
            ma = self.data[f"MA_{strategy['params']['ma_period']}"]
            rsi = self.data[f"RSI_{strategy['params']['rsi_period']}"]
            atr = self.data[f"ATR_{strategy['params']['atr_period']}"]
            
            trend_up = self.data['Close'] > ma
            momentum_ok = (rsi > 40) & (rsi < 60)
            volatility_ok = atr < atr.rolling(50).mean() * 1.5
            
            signals = np.where(trend_up & momentum_ok & volatility_ok, 1,
                             np.where(~trend_up, -1, 0))
        
        # Calculate returns
        returns = self._calculate_strategy_returns(signals)
        metrics = self._calculate_performance_metrics(returns, signals)
        
        return {
            'signals': signals,
            'returns': returns,
            'metrics': metrics
        }
    
    def _calculate_strategy_returns(self, signals: np.ndarray) -> pd.Series:
        """Calculate strategy returns based on signals."""
        price_changes = self.data['Close'].pct_change()
        
        # Convert signals to positions (previous day signal determines today's position)
        positions = pd.Series(signals).shift(1).fillna(0)
        
        # Calculate strategy returns
        strategy_returns = positions * price_changes
        
        return strategy_returns
    
    def _calculate_performance_metrics(self, returns: pd.Series, signals: np.ndarray) -> Dict:
        """Calculate comprehensive performance metrics."""
        returns = returns.dropna()
        
        # Basic metrics with safe calculations
        if len(returns) == 0:
            return {
                'Total Return': 0.0,
                'Annualized Return': 0.0,
                'Volatility': 0.0,
                'Sharpe Ratio': 0.0,
                'Max Drawdown': 0.0,
                'Calmar Ratio': 0.0,
                'Sortino Ratio': 0.0,
                'Information Ratio': 0.0,
                'Win Rate': 0.0,
                'Total Trades': 0.0,
                'Avg Return per Trade': 0.0
            }
        
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
        volatility = returns.std() * np.sqrt(252) if not returns.empty else 0
        sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 1e-10 else 0
        
        # Drawdown analysis
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Trade analysis
        position_changes = np.diff(np.append(0, signals))
        trades = np.sum(np.abs(position_changes) > 0) / 2  # Divide by 2 for round trips
        
        # Win rate calculation
        if len(returns[returns != 0]) > 0:
            win_rate = len(returns[returns > 0]) / len(returns[returns != 0])
        else:
            win_rate = 0
        
        # Calmar ratio with safe division
        calmar_ratio = annualized_return / abs(max_drawdown) if abs(max_drawdown) > 1e-10 else 0
        
        # Information ratio (assuming benchmark is 0) with safe division
        returns_std = returns.std()
        information_ratio = returns.mean() / returns_std * np.sqrt(252) if returns_std > 1e-10 else 0
        
        # Sortino ratio with safe division
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annualized_return - 0.02) / downside_volatility if downside_volatility > 1e-10 else 0
        
        return {
            'Total Return': total_return,
            'Annualized Return': annualized_return,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Calmar Ratio': calmar_ratio,
            'Sortino Ratio': sortino_ratio,
            'Information Ratio': information_ratio,
            'Win Rate': win_rate,
            'Total Trades': max(trades, 0),
            'Avg Return per Trade': total_return / trades if trades > 1e-10 else 0
        }
    
    def optimize_strategies(self) -> pd.DataFrame:
        """Run optimization across all strategy combinations."""
        strategies = self._get_strategy_combinations()
        results = []
        
        for strategy in strategies:
            try:
                result = self._execute_strategy(strategy)
                
                strategy_result = {
                    'Strategy': strategy['name'],
                    'Type': strategy['type'],
                    'Parameters': str(strategy['params']),
                    **result['metrics']
                }
                
                results.append(strategy_result)
                self.results[strategy['name']] = result
                
            except Exception as e:
                print(f"Error executing strategy {strategy['name']}: {str(e)}")
                continue
        
        return pd.DataFrame(results)
    
    def get_top_strategies(self, metric: str = 'Sharpe Ratio', top_n: int = 5) -> pd.DataFrame:
        """Get top N strategies based on specified metric."""
        if not hasattr(self, 'optimization_results'):
            self.optimization_results = self.optimize_strategies()
        
        return self.optimization_results.nlargest(top_n, metric)
    
    def plot_strategy_comparison(self, strategies: List[str] = None) -> go.Figure:
        """Create comprehensive strategy comparison visualization."""
        if strategies is None:
            # Get top 5 strategies by Sharpe ratio
            top_strategies = self.get_top_strategies(top_n=5)
            strategies = top_strategies['Strategy'].tolist()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Cumulative Returns', 'Drawdown', 'Performance Metrics', 'Risk-Return Scatter'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = px.colors.qualitative.Set1[:len(strategies)]
        
        # Cumulative returns
        for i, strategy_name in enumerate(strategies):
            if strategy_name in self.results:
                returns = self.results[strategy_name]['returns']
                cumulative = (1 + returns.fillna(0)).cumprod()
                
                fig.add_trace(
                    go.Scatter(x=cumulative.index, y=cumulative.values,
                             name=strategy_name, line=dict(color=colors[i])),
                    row=1, col=1
                )
        
        # Add buy & hold benchmark
        buy_hold_returns = self.data['Close'].pct_change()
        buy_hold_cumulative = (1 + buy_hold_returns.fillna(0)).cumprod()
        fig.add_trace(
            go.Scatter(x=buy_hold_cumulative.index, y=buy_hold_cumulative.values,
                     name='Buy & Hold', line=dict(color='black', dash='dash')),
            row=1, col=1
        )
        
        # Drawdown
        for i, strategy_name in enumerate(strategies):
            if strategy_name in self.results:
                returns = self.results[strategy_name]['returns']
                cumulative = (1 + returns.fillna(0)).cumprod()
                rolling_max = cumulative.expanding().max()
                drawdown = (cumulative - rolling_max) / rolling_max
                
                fig.add_trace(
                    go.Scatter(x=drawdown.index, y=drawdown.values,
                             name=f'{strategy_name} DD', line=dict(color=colors[i]),
                             showlegend=False),
                    row=1, col=2
                )
        
        # Performance metrics radar chart (simplified as bar chart)
        metrics_data = []
        for strategy_name in strategies:
            if strategy_name in self.results:
                metrics = self.results[strategy_name]['metrics']
                metrics_data.append({
                    'Strategy': strategy_name,
                    'Sharpe': metrics['Sharpe Ratio'],
                    'Calmar': metrics['Calmar Ratio'],
                    'Win Rate': metrics['Win Rate']
                })
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            for metric in ['Sharpe', 'Calmar', 'Win Rate']:
                fig.add_trace(
                    go.Bar(x=metrics_df['Strategy'], y=metrics_df[metric],
                          name=metric, showlegend=False),
                    row=2, col=1
                )
        
        # Risk-Return scatter
        for i, strategy_name in enumerate(strategies):
            if strategy_name in self.results:
                metrics = self.results[strategy_name]['metrics']
                fig.add_trace(
                    go.Scatter(x=[metrics['Volatility']], y=[metrics['Annualized Return']],
                             mode='markers+text', text=[strategy_name],
                             textposition='top center', name=strategy_name,
                             marker=dict(size=10, color=colors[i]),
                             showlegend=False),
                    row=2, col=2
                )
        
        fig.update_layout(
            title="Comprehensive Strategy Performance Analysis",
            height=800,
            showlegend=True
        )
        
        return fig
    
    def get_strategy_details(self, strategy_name: str) -> Dict:
        """Get detailed analysis for a specific strategy."""
        if strategy_name not in self.results:
            return None
        
        result = self.results[strategy_name]
        
        # Calculate additional details
        returns = result['returns']
        signals = result['signals']
        
        # Monthly returns analysis with proper error handling
        try:
            # Ensure returns is a proper time series with datetime index
            if not isinstance(returns.index, pd.DatetimeIndex):
                returns.index = pd.to_datetime(returns.index)
            monthly_returns = returns.resample('M').sum()
        except Exception as e:
            # Fallback: group by month manually
            returns_df = returns.reset_index()
            if 'Date' in returns_df.columns:
                returns_df['Month'] = pd.to_datetime(returns_df['Date']).dt.to_period('M')
                monthly_returns = returns_df.groupby('Month')[returns.name or 'Returns'].sum()
            else:
                monthly_returns = pd.Series(dtype=float)
        
        # Rolling performance with error handling
        try:
            rolling_sharpe = returns.rolling(252).mean() / returns.rolling(252).std() * np.sqrt(252)
        except Exception:
            rolling_sharpe = pd.Series(dtype=float)
        
        return {
            'metrics': result['metrics'],
            'returns': returns,
            'signals': signals,
            'monthly_returns': monthly_returns,
            'rolling_sharpe': rolling_sharpe,
            'signal_distribution': pd.Series(signals).value_counts()
        }