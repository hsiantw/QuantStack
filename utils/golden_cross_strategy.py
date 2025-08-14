import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

class GoldenCrossStrategy:
    def __init__(self, fast_ma=50, slow_ma=200, dip_threshold=0.01):
        """
        Golden Cross Strategy with Buy-the-Dip logic
        
        Parameters:
        - fast_ma: Fast moving average period (default 50)
        - slow_ma: Slow moving average period (default 200)
        - dip_threshold: Minimum drop percentage to trigger buy signal (default 1%)
        """
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.dip_threshold = dip_threshold
        
    def calculate_signals(self, data):
        """Calculate trading signals based on golden cross and dip conditions"""
        # Calculate moving averages
        data[f'MA_{self.fast_ma}'] = data['Close'].rolling(window=self.fast_ma).mean()
        data[f'MA_{self.slow_ma}'] = data['Close'].rolling(window=self.slow_ma).mean()
        
        # Calculate daily returns
        data['Daily_Return'] = data['Close'].pct_change()
        
        # Identify golden cross and death cross
        data['Golden_Cross'] = (data[f'MA_{self.fast_ma}'] > data[f'MA_{self.slow_ma}']) & \
                              (data[f'MA_{self.fast_ma}'].shift(1) <= data[f'MA_{self.slow_ma}'].shift(1))
        
        data['Death_Cross'] = (data[f'MA_{self.fast_ma}'] < data[f'MA_{self.slow_ma}']) & \
                             (data[f'MA_{self.fast_ma}'].shift(1) >= data[f'MA_{self.slow_ma}'].shift(1))
        
        # Track market regime (bullish after golden cross, bearish after death cross)
        data['Market_Regime'] = 0  # 0: bearish, 1: bullish
        regime = 0
        
        for i in range(len(data)):
            if data['Golden_Cross'].iloc[i]:
                regime = 1
            elif data['Death_Cross'].iloc[i]:
                regime = 0
            data.iloc[i, data.columns.get_loc('Market_Regime')] = regime
        
        # Generate buy signals: 1%+ drop during bullish regime
        data['Buy_Signal'] = (data['Market_Regime'] == 1) & \
                            (data['Daily_Return'] <= -self.dip_threshold)
        
        # Generate sell signals: death cross
        data['Sell_Signal'] = data['Death_Cross']
        
        return data
    
    def backtest_strategy(self, data, initial_capital=10000):
        """Backtest the golden cross strategy"""
        results = data.copy()
        
        # Initialize tracking variables
        position = 0  # 0: no position, 1: long position
        cash = initial_capital
        shares = 0
        portfolio_value = []
        trades = []
        
        for i in range(len(results)):
            current_price = results['Close'].iloc[i]
            
            # Buy signal: during bullish regime on 1%+ dip
            if results['Buy_Signal'].iloc[i] and position == 0:
                shares = cash / current_price
                cash = 0
                position = 1
                trades.append({
                    'Date': results.index[i],
                    'Action': 'BUY',
                    'Price': current_price,
                    'Shares': shares,
                    'Reason': f"Dip: {results['Daily_Return'].iloc[i]:.2%}"
                })
            
            # Sell signal: death cross
            elif results['Sell_Signal'].iloc[i] and position == 1:
                cash = shares * current_price
                profit = cash - initial_capital
                trades.append({
                    'Date': results.index[i],
                    'Action': 'SELL',
                    'Price': current_price,
                    'Shares': shares,
                    'Profit': profit,
                    'Return': profit / initial_capital * 100
                })
                shares = 0
                position = 0
            
            # Calculate portfolio value
            if position == 1:
                portfolio_value.append(shares * current_price)
            else:
                portfolio_value.append(cash)
        
        results['Portfolio_Value'] = portfolio_value
        results['Position'] = [1 if pv > cash else 0 for pv in portfolio_value]
        
        return results, trades
    
    def calculate_performance_metrics(self, results, trades):
        """Calculate strategy performance metrics"""
        if len(results) == 0 or 'Portfolio_Value' not in results.columns:
            return {}
        
        initial_value = results['Portfolio_Value'].iloc[0]
        final_value = results['Portfolio_Value'].iloc[-1]
        
        # Basic returns
        total_return = (final_value - initial_value) / initial_value * 100
        
        # Calculate returns series
        portfolio_returns = results['Portfolio_Value'].pct_change().dropna()
        
        # Risk metrics
        volatility = portfolio_returns.std() * np.sqrt(252) * 100  # Annualized
        
        # Sharpe ratio (assuming 2% risk-free rate)
        excess_returns = portfolio_returns - (0.02/252)
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        # Maximum drawdown
        peak = results['Portfolio_Value'].cummax()
        drawdown = (results['Portfolio_Value'] - peak) / peak
        max_drawdown = drawdown.min() * 100
        
        # Trade statistics
        profitable_trades = [t for t in trades if 'Profit' in t and t['Profit'] > 0]
        losing_trades = [t for t in trades if 'Profit' in t and t['Profit'] <= 0]
        
        win_rate = len(profitable_trades) / len([t for t in trades if 'Profit' in t]) * 100 if len(trades) > 0 else 0
        
        avg_win = np.mean([t['Profit'] for t in profitable_trades]) if profitable_trades else 0
        avg_loss = np.mean([t['Profit'] for t in losing_trades]) if losing_trades else 0
        
        return {
            'Total Return (%)': total_return,
            'Volatility (%)': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown (%)': max_drawdown,
            'Total Trades': len([t for t in trades if 'Profit' in t]),
            'Win Rate (%)': win_rate,
            'Average Win ($)': avg_win,
            'Average Loss ($)': avg_loss,
            'Profit Factor': abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        }
    
    def create_strategy_chart(self, results, trades):
        """Create comprehensive strategy visualization"""
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=('Price with Moving Averages & Signals', 'Portfolio Value', 'Market Regime & Daily Returns'),
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Price chart with moving averages
        fig.add_trace(go.Scatter(
            x=results.index,
            y=results['Close'],
            name='Price',
            line=dict(color='white', width=2)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=results.index,
            y=results[f'MA_{self.fast_ma}'],
            name=f'MA {self.fast_ma}',
            line=dict(color='orange', width=1)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=results.index,
            y=results[f'MA_{self.slow_ma}'],
            name=f'MA {self.slow_ma}',
            line=dict(color='blue', width=1)
        ), row=1, col=1)
        
        # Buy signals
        buy_signals = results[results['Buy_Signal']]
        if not buy_signals.empty:
            fig.add_trace(go.Scatter(
                x=buy_signals.index,
                y=buy_signals['Close'],
                mode='markers',
                name='Buy Signal',
                marker=dict(color='green', size=8, symbol='triangle-up')
            ), row=1, col=1)
        
        # Sell signals
        sell_signals = results[results['Sell_Signal']]
        if not sell_signals.empty:
            fig.add_trace(go.Scatter(
                x=sell_signals.index,
                y=sell_signals['Close'],
                mode='markers',
                name='Sell Signal',
                marker=dict(color='red', size=8, symbol='triangle-down')
            ), row=1, col=1)
        
        # Golden cross markers
        golden_cross = results[results['Golden_Cross']]
        if not golden_cross.empty:
            fig.add_trace(go.Scatter(
                x=golden_cross.index,
                y=golden_cross['Close'],
                mode='markers',
                name='Golden Cross',
                marker=dict(color='gold', size=12, symbol='star')
            ), row=1, col=1)
        
        # Death cross markers
        death_cross = results[results['Death_Cross']]
        if not death_cross.empty:
            fig.add_trace(go.Scatter(
                x=death_cross.index,
                y=death_cross['Close'],
                mode='markers',
                name='Death Cross',
                marker=dict(color='black', size=12, symbol='x')
            ), row=1, col=1)
        
        # Portfolio value
        fig.add_trace(go.Scatter(
            x=results.index,
            y=results['Portfolio_Value'],
            name='Portfolio Value',
            line=dict(color='cyan', width=2),
            fill='tonexty'
        ), row=2, col=1)
        
        # Market regime
        fig.add_trace(go.Scatter(
            x=results.index,
            y=results['Market_Regime'],
            name='Market Regime',
            line=dict(color='purple', width=2),
            yaxis='y3'
        ), row=3, col=1)
        
        # Daily returns
        fig.add_trace(go.Bar(
            x=results.index,
            y=results['Daily_Return'] * 100,
            name='Daily Return (%)',
            marker_color=np.where(results['Daily_Return'] > 0, 'green', 'red'),
            opacity=0.6
        ), row=3, col=1)
        
        # Add horizontal line for dip threshold
        fig.add_hline(y=-self.dip_threshold * 100, line_dash="dash", line_color="yellow", 
                      annotation_text=f"Dip Threshold ({-self.dip_threshold*100:.1f}%)", row=3, col=1)
        
        fig.update_layout(
            height=800,
            title_text="Golden Cross Strategy with Buy-the-Dip",
            template="plotly_dark",
            showlegend=True
        )
        
        return fig
    
    def generate_strategy_summary(self):
        """Generate strategy description"""
        return f"""
        ## Golden Cross Strategy with Buy-the-Dip
        
        **Strategy Rules:**
        1. **Golden Cross Entry**: Wait for {self.fast_ma}-day MA to cross above {self.slow_ma}-day MA
        2. **Buy Condition**: During bullish regime (after golden cross), buy on any day with ≥{self.dip_threshold*100:.1f}% drop
        3. **Exit Condition**: Sell all positions when death cross occurs ({self.fast_ma}-day MA crosses below {self.slow_ma}-day MA)
        4. **Position Sizing**: All-in strategy (use full available capital on each buy signal)
        
        **Strategy Logic:**
        - **Trend Following**: Only buy during confirmed uptrends (golden cross regime)
        - **Buy the Dip**: Capitalize on short-term pullbacks in a bullish trend
        - **Risk Management**: Exit completely on trend reversal (death cross)
        - **Patience**: Wait for significant dips before entering positions
        
        **Best For:**
        - Medium to long-term trend following
        - Volatile assets with clear trend cycles
        - Markets with distinct bull/bear phases
        """