import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

class BacktestingEngine:
    """Comprehensive backtesting engine for trading strategies"""
    
    def __init__(self, initial_capital=100000, commission=0.001, slippage=0.0001):
        """
        Initialize backtesting engine
        
        Args:
            initial_capital (float): Initial capital
            commission (float): Commission rate per trade
            slippage (float): Slippage rate per trade
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
    def backtest_signals(self, price_data, signals, position_size='equal_weight'):
        """
        Backtest trading signals
        
        Args:
            price_data (pandas.DataFrame): Price data with OHLCV
            signals (pandas.Series): Trading signals (1: buy, -1: sell, 0: hold)
            position_size (str or float): Position sizing method
        
        Returns:
            dict: Backtesting results
        """
        results = {}
        
        # Align signals with price data
        aligned_data = pd.concat([price_data['Close'], signals], axis=1).dropna()
        aligned_data.columns = ['Close', 'Signal']
        
        # Initialize tracking variables
        portfolio_value = [self.initial_capital]
        positions = [0]
        cash = [self.initial_capital]
        holdings = [0]
        trades = []
        
        current_position = 0
        current_cash = self.initial_capital
        current_holdings = 0
        
        for i in range(1, len(aligned_data)):
            current_price = aligned_data['Close'].iloc[i]
            current_signal = aligned_data['Signal'].iloc[i]
            prev_signal = aligned_data['Signal'].iloc[i-1]
            
            # Detect signal changes
            if current_signal != prev_signal and current_signal != 0:
                # Calculate position size
                if position_size == 'equal_weight':
                    target_position = current_signal
                    shares_to_trade = target_position - current_position
                    
                    if shares_to_trade != 0:
                        # Calculate trade value
                        trade_value = abs(shares_to_trade * current_price)
                        commission_cost = trade_value * self.commission
                        slippage_cost = trade_value * self.slippage
                        total_cost = commission_cost + slippage_cost
                        
                        # Execute trade if sufficient cash
                        if shares_to_trade > 0:  # Buying
                            required_cash = trade_value + total_cost
                            if required_cash <= current_cash:
                                current_cash -= required_cash
                                current_holdings += shares_to_trade * current_price
                                current_position = target_position
                                
                                trades.append({
                                    'date': aligned_data.index[i],
                                    'action': 'BUY',
                                    'price': current_price,
                                    'position': shares_to_trade,
                                    'value': trade_value,
                                    'cost': total_cost
                                })
                        
                        else:  # Selling
                            current_cash += trade_value - total_cost
                            current_holdings += shares_to_trade * current_price
                            current_position = target_position
                            
                            trades.append({
                                'date': aligned_data.index[i],
                                'action': 'SELL',
                                'price': current_price,
                                'position': shares_to_trade,
                                'value': trade_value,
                                'cost': total_cost
                            })
            
            # Update holdings value based on current price
            if current_position != 0:
                current_holdings = current_position * current_price * self.initial_capital
            
            # Calculate total portfolio value
            total_value = current_cash + current_holdings
            
            portfolio_value.append(total_value)
            positions.append(current_position)
            cash.append(current_cash)
            holdings.append(current_holdings)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'Portfolio_Value': portfolio_value,
            'Position': positions,
            'Cash': cash,
            'Holdings': holdings,
            'Price': [aligned_data['Close'].iloc[0]] + aligned_data['Close'].tolist()[1:]
        }, index=[aligned_data.index[0]] + aligned_data.index.tolist()[1:])
        
        # Calculate returns
        results_df['Returns'] = results_df['Portfolio_Value'].pct_change()
        results_df['Cumulative_Returns'] = (1 + results_df['Returns']).cumprod()
        
        # Calculate benchmark (buy and hold)
        benchmark_returns = aligned_data['Close'].pct_change()
        benchmark_cumulative = (1 + benchmark_returns).cumprod()
        results_df['Benchmark_Cumulative'] = benchmark_cumulative
        
        return {
            'results_df': results_df,
            'trades': trades,
            'initial_capital': self.initial_capital,
            'final_value': results_df['Portfolio_Value'].iloc[-1],
            'total_return': (results_df['Portfolio_Value'].iloc[-1] / self.initial_capital) - 1
        }
    
    def calculate_performance_metrics(self, results):
        """
        Calculate comprehensive performance metrics
        
        Args:
            results (dict): Backtesting results
        
        Returns:
            dict: Performance metrics
        """
        results_df = results['results_df']
        returns = results_df['Returns'].dropna()
        
        # Basic metrics
        total_return = results['total_return']
        n_years = len(results_df) / 252  # Assuming daily data
        
        if n_years > 0:
            annualized_return = (1 + total_return) ** (1/n_years) - 1
        else:
            annualized_return = 0
        
        # Volatility and risk metrics
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility != 0 else 0
        
        # Downside deviation
        negative_returns = returns[returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
        sortino_ratio = annualized_return / downside_deviation if downside_deviation != 0 else 0
        
        # Maximum Drawdown
        portfolio_values = results_df['Portfolio_Value']
        rolling_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Calmar Ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Value at Risk (VaR)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Expected Shortfall (Conditional VaR)
        es_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else 0
        es_99 = returns[returns <= var_99].mean() if len(returns[returns <= var_99]) > 0 else 0
        
        # Trade analysis
        trades = results['trades']
        total_trades = len(trades)
        
        if total_trades > 0:
            trade_returns = []
            for i in range(1, len(trades)):
                if trades[i-1]['action'] == 'BUY' and trades[i]['action'] == 'SELL':
                    buy_price = trades[i-1]['price']
                    sell_price = trades[i]['price']
                    trade_return = (sell_price - buy_price) / buy_price
                    trade_returns.append(trade_return)
            
            if trade_returns:
                winning_trades = [t for t in trade_returns if t > 0]
                losing_trades = [t for t in trade_returns if t < 0]
                
                win_rate = len(winning_trades) / len(trade_returns) if trade_returns else 0
                avg_win = np.mean(winning_trades) if winning_trades else 0
                avg_loss = np.mean(losing_trades) if losing_trades else 0
                profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            else:
                win_rate = avg_win = avg_loss = profit_factor = 0
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0
        
        # Beta calculation (vs benchmark)
        if 'Benchmark_Cumulative' in results_df.columns:
            benchmark_returns = results_df['Benchmark_Cumulative'].pct_change().dropna()
            aligned_returns = returns.reindex(benchmark_returns.index).dropna()
            aligned_benchmark = benchmark_returns.reindex(aligned_returns.index).dropna()
            
            if len(aligned_returns) > 0 and len(aligned_benchmark) > 0:
                covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
                benchmark_variance = np.var(aligned_benchmark)
                beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
                
                # Alpha calculation
                risk_free_rate = 0.02  # Assume 2% risk-free rate
                alpha = annualized_return - (risk_free_rate + beta * (aligned_benchmark.mean() * 252 - risk_free_rate))
            else:
                beta = alpha = 0
        else:
            beta = alpha = 0
        
        return {
            'Total Return': total_return,
            'Annualized Return': annualized_return,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Max Drawdown': max_drawdown,
            'Calmar Ratio': calmar_ratio,
            'VaR (95%)': var_95,
            'VaR (99%)': var_99,
            'Expected Shortfall (95%)': es_95,
            'Expected Shortfall (99%)': es_99,
            'Beta': beta,
            'Alpha': alpha,
            'Total Trades': total_trades,
            'Win Rate': win_rate,
            'Average Win': avg_win,
            'Average Loss': avg_loss,
            'Profit Factor': profit_factor
        }
    
    def plot_backtest_results(self, results, strategy_name="Strategy"):
        """
        Create comprehensive backtest visualization
        
        Args:
            results (dict): Backtesting results
            strategy_name (str): Name of the strategy
        
        Returns:
            dict: Dictionary of plotly figures
        """
        plots = {}
        results_df = results['results_df']
        
        # 1. Portfolio value over time
        fig_portfolio = go.Figure()
        
        fig_portfolio.add_trace(go.Scatter(
            x=results_df.index,
            y=results_df['Portfolio_Value'],
            mode='lines',
            name=strategy_name,
            line=dict(color='blue', width=2)
        ))
        
        # Add benchmark if available
        if 'Benchmark_Cumulative' in results_df.columns:
            benchmark_value = results_df['Benchmark_Cumulative'] * results['initial_capital']
            fig_portfolio.add_trace(go.Scatter(
                x=results_df.index,
                y=benchmark_value,
                mode='lines',
                name='Benchmark (Buy & Hold)',
                line=dict(color='red', width=2)
            ))
        
        fig_portfolio.update_layout(
            title=f'{strategy_name} - Portfolio Value Over Time',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            height=500
        )
        
        plots['portfolio_value'] = fig_portfolio
        
        # 2. Drawdown analysis
        portfolio_values = results_df['Portfolio_Value']
        rolling_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - rolling_max) / rolling_max
        
        fig_drawdown = go.Figure()
        
        fig_drawdown.add_trace(go.Scatter(
            x=results_df.index,
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
        
        # 3. Returns distribution
        returns = results_df['Returns'].dropna()
        
        fig_returns = go.Figure()
        
        fig_returns.add_trace(go.Histogram(
            x=returns,
            nbinsx=50,
            name='Returns Distribution',
            opacity=0.7
        ))
        
        fig_returns.update_layout(
            title=f'{strategy_name} - Returns Distribution',
            xaxis_title='Returns',
            yaxis_title='Frequency',
            height=400
        )
        
        plots['returns_distribution'] = fig_returns
        
        # 4. Rolling metrics
        rolling_window = 252  # 1 year
        if len(results_df) > rolling_window:
            rolling_returns = returns.rolling(window=rolling_window).mean() * 252
            rolling_volatility = returns.rolling(window=rolling_window).std() * np.sqrt(252)
            rolling_sharpe = rolling_returns / rolling_volatility
            
            fig_rolling = make_subplots(
                rows=3, cols=1,
                subplot_titles=['Rolling Returns', 'Rolling Volatility', 'Rolling Sharpe Ratio'],
                vertical_spacing=0.08
            )
            
            fig_rolling.add_trace(
                go.Scatter(x=results_df.index, y=rolling_returns, name='Rolling Returns'),
                row=1, col=1
            )
            
            fig_rolling.add_trace(
                go.Scatter(x=results_df.index, y=rolling_volatility, name='Rolling Volatility'),
                row=2, col=1
            )
            
            fig_rolling.add_trace(
                go.Scatter(x=results_df.index, y=rolling_sharpe, name='Rolling Sharpe'),
                row=3, col=1
            )
            
            fig_rolling.update_layout(
                title=f'{strategy_name} - Rolling Performance Metrics',
                height=700,
                showlegend=False
            )
            
            plots['rolling_metrics'] = fig_rolling
        
        # 5. Trade analysis (if trades exist)
        trades = results['trades']
        if trades:
            trade_dates = [trade['date'] for trade in trades]
            trade_prices = [trade['price'] for trade in trades]
            trade_actions = [trade['action'] for trade in trades]
            
            fig_trades = go.Figure()
            
            # Price chart
            fig_trades.add_trace(go.Scatter(
                x=results_df.index,
                y=results_df['Price'],
                mode='lines',
                name='Price',
                line=dict(color='black', width=1)
            ))
            
            # Buy signals
            buy_dates = [date for date, action in zip(trade_dates, trade_actions) if action == 'BUY']
            buy_prices = [price for price, action in zip(trade_prices, trade_actions) if action == 'BUY']
            
            if buy_dates:
                fig_trades.add_trace(go.Scatter(
                    x=buy_dates,
                    y=buy_prices,
                    mode='markers',
                    name='Buy',
                    marker=dict(color='green', size=10, symbol='triangle-up')
                ))
            
            # Sell signals
            sell_dates = [date for date, action in zip(trade_dates, trade_actions) if action == 'SELL']
            sell_prices = [price for price, action in zip(trade_prices, trade_actions) if action == 'SELL']
            
            if sell_dates:
                fig_trades.add_trace(go.Scatter(
                    x=sell_dates,
                    y=sell_prices,
                    mode='markers',
                    name='Sell',
                    marker=dict(color='red', size=10, symbol='triangle-down')
                ))
            
            fig_trades.update_layout(
                title=f'{strategy_name} - Trading Signals',
                xaxis_title='Date',
                yaxis_title='Price',
                height=500
            )
            
            plots['trades'] = fig_trades
        
        return plots
