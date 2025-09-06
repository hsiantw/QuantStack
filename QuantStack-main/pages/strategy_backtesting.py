import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.ui_components import apply_custom_css, create_metric_card, create_info_card
    from utils.auth import check_authentication
    apply_custom_css()
except ImportError:
    st.warning("Some components not found. Using default functionality.")
    def create_metric_card(title, value, delta=None):
        return st.metric(title, value, delta)
    def create_info_card(title, content):
        return st.info(f"**{title}**\n\n{content}")
    def check_authentication():
        return True, None
    ma_short = 20
    ma_long = 50
    rsi_period = 14
    rsi_overbought = 70
    rsi_oversold = 30
    bb_period = 20
    bb_std = 2.0
    momentum_lookback = 20
    momentum_threshold = 0.02

class TradingStrategy:
    """Base class for trading strategies"""
    
    def __init__(self, name):
        self.name = name
        self.signals = None
        self.positions = None
        
    def generate_signals(self, data):
        """Generate buy/sell signals - to be implemented by subclasses"""
        raise NotImplementedError
        
    def calculate_returns(self, data, signals):
        """Calculate strategy returns based on signals"""
        # Calculate daily returns
        data['returns'] = data['Close'].pct_change()
        
        # Calculate strategy returns
        data['strategy_returns'] = data['returns'] * signals.shift(1)
        data['cumulative_returns'] = (1 + data['strategy_returns']).cumprod()
        data['buy_hold_returns'] = (1 + data['returns']).cumprod()
        
        return data

class MovingAverageCrossover(TradingStrategy):
    """Moving Average Crossover Strategy"""
    
    def __init__(self, short_window=20, long_window=50):
        super().__init__("Moving Average Crossover")
        self.short_window = short_window
        self.long_window = long_window
        
    def generate_signals(self, data):
        """Generate signals based on moving average crossover"""
        data['MA_short'] = data['Close'].rolling(window=self.short_window).mean()
        data['MA_long'] = data['Close'].rolling(window=self.long_window).mean()
        
        # Generate signals
        signals = pd.Series(index=data.index, data=0.0)
        signals[self.short_window:] = np.where(
            data['MA_short'][self.short_window:] > data['MA_long'][self.short_window:], 1.0, 0.0
        )
        
        # Generate trading orders
        data['positions'] = signals.diff()
        
        return signals

class RSIStrategy(TradingStrategy):
    """RSI Mean Reversion Strategy"""
    
    def __init__(self, rsi_window=14, overbought=70, oversold=30):
        super().__init__("RSI Mean Reversion")
        self.rsi_window = rsi_window
        self.overbought = overbought
        self.oversold = oversold
        
    def calculate_rsi(self, prices):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    def generate_signals(self, data):
        """Generate signals based on RSI levels"""
        data['RSI'] = self.calculate_rsi(data['Close'])
        
        # Generate signals
        signals = pd.Series(index=data.index, data=0.0)
        
        # Buy when oversold, sell when overbought
        signals = np.where(data['RSI'] < self.oversold, 1.0, signals)
        signals = np.where(data['RSI'] > self.overbought, -1.0, signals)
        signals = pd.Series(signals, index=data.index).ffill().fillna(0.0)
        
        # Generate trading orders
        data['positions'] = pd.Series(signals).diff()
        
        return signals

class BollingerBandsStrategy(TradingStrategy):
    """Bollinger Bands Mean Reversion Strategy"""
    
    def __init__(self, window=20, num_std=2):
        super().__init__("Bollinger Bands")
        self.window = window
        self.num_std = num_std
        
    def generate_signals(self, data):
        """Generate signals based on Bollinger Bands"""
        data['MA'] = data['Close'].rolling(window=self.window).mean()
        data['BB_std'] = data['Close'].rolling(window=self.window).std()
        data['BB_upper'] = data['MA'] + (self.num_std * data['BB_std'])
        data['BB_lower'] = data['MA'] - (self.num_std * data['BB_std'])
        
        # Generate signals
        signals = pd.Series(index=data.index, data=0.0)
        
        # Buy when price touches lower band, sell when price touches upper band
        signals = np.where(data['Close'] <= data['BB_lower'], 1.0, signals)
        signals = np.where(data['Close'] >= data['BB_upper'], -1.0, signals)
        signals = pd.Series(signals, index=data.index).ffill().fillna(0.0)
        
        # Generate trading orders
        data['positions'] = pd.Series(signals).diff()
        
        return signals

class MomentumStrategy(TradingStrategy):
    """Momentum Strategy based on price momentum"""
    
    def __init__(self, lookback=20, threshold=0.02):
        super().__init__("Momentum")
        self.lookback = lookback
        self.threshold = threshold
        
    def generate_signals(self, data):
        """Generate signals based on momentum"""
        # Calculate momentum
        data['momentum'] = data['Close'].pct_change(self.lookback)
        
        # Generate signals
        signals = pd.Series(index=data.index, data=0.0)
        signals = np.where(data['momentum'] > self.threshold, 1.0, signals)
        signals = np.where(data['momentum'] < -self.threshold, -1.0, signals)
        signals = pd.Series(signals, index=data.index).ffill().fillna(0.0)
        
        # Generate trading orders
        data['positions'] = pd.Series(signals).diff()
        
        return signals

class BacktestEngine:
    """Backtesting engine for trading strategies"""
    
    def __init__(self, initial_capital=100000, commission=0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        
    def run_backtest(self, data, strategy, transaction_costs=True):
        """Run backtest for a given strategy"""
        # Generate signals
        signals = strategy.generate_signals(data.copy())
        
        # Calculate returns
        results = strategy.calculate_returns(data.copy(), signals)
        
        # Calculate performance metrics
        metrics = self.calculate_performance_metrics(results, transaction_costs)
        
        return results, metrics
        
    def calculate_performance_metrics(self, data, transaction_costs=True):
        """Calculate comprehensive performance metrics"""
        strategy_returns = data['strategy_returns'].dropna()
        buy_hold_returns = data['returns'].dropna()
        
        # Adjust for transaction costs if enabled
        if transaction_costs and 'positions' in data.columns:
            trades = np.abs(data['positions']).sum()
            total_costs = trades * self.commission
            strategy_returns = strategy_returns - (total_costs / len(strategy_returns))
        
        # Calculate metrics
        total_return = (1 + strategy_returns).prod() - 1
        buy_hold_return = (1 + buy_hold_returns).prod() - 1
        
        annual_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
        annual_volatility = strategy_returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming risk-free rate of 2%)
        risk_free_rate = 0.02
        sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + strategy_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_trades = (strategy_returns > 0).sum()
        total_trades = len(strategy_returns[strategy_returns != 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        metrics = {
            'Total Return': f"{total_return:.2%}",
            'Annual Return': f"{annual_return:.2%}",
            'Annual Volatility': f"{annual_volatility:.2%}",
            'Sharpe Ratio': f"{sharpe_ratio:.2f}",
            'Max Drawdown': f"{max_drawdown:.2%}",
            'Win Rate': f"{win_rate:.2%}",
            'Calmar Ratio': f"{calmar_ratio:.2f}",
            'Buy & Hold Return': f"{buy_hold_return:.2%}",
            'Total Trades': int(total_trades)
        }
        
        return metrics

def create_strategy_comparison_chart(results_dict):
    """Create comparison chart for multiple strategies"""
    fig = go.Figure()
    
    colors = ['#00D4FF', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for i, (name, data) in enumerate(results_dict.items()):
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['cumulative_returns'],
            mode='lines',
            name=name,
            line=dict(color=colors[i % len(colors)], width=2),
            hovertemplate=f'{name}<br>Date: %{{x}}<br>Cumulative Return: %{{y:.1%}}<extra></extra>'
        ))
    
    # Add buy and hold for reference
    if results_dict:
        first_result = list(results_dict.values())[0]
        fig.add_trace(go.Scatter(
            x=first_result.index,
            y=first_result['buy_hold_returns'],
            mode='lines',
            name='Buy & Hold',
            line=dict(color='#FFA726', width=2, dash='dash'),
            hovertemplate='Buy & Hold<br>Date: %{x}<br>Cumulative Return: %{y:.1%}<extra></extra>'
        ))
    
    fig.update_layout(
        title='Strategy Performance Comparison',
        xaxis_title='Date',
        yaxis_title='Cumulative Returns',
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0.5)"
        )
    )
    
    fig.update_xaxes(gridcolor='rgba(128,128,128,0.2)', zeroline=False)
    fig.update_yaxes(gridcolor='rgba(128,128,128,0.2)', zeroline=False, tickformat='.1%')
    
    return fig

def create_drawdown_chart(data):
    """Create drawdown chart"""
    strategy_returns = data['strategy_returns'].dropna()
    cumulative = (1 + strategy_returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown,
        mode='lines',
        name='Drawdown',
        fill='tonexty',
        line=dict(color='#FF6B6B', width=2),
        fillcolor='rgba(255, 107, 107, 0.3)',
        hovertemplate='Date: %{x}<br>Drawdown: %{y:.1%}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Strategy Drawdown Over Time',
        xaxis_title='Date',
        yaxis_title='Drawdown',
        hovermode='x',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    fig.update_xaxes(gridcolor='rgba(128,128,128,0.2)', zeroline=False)
    fig.update_yaxes(gridcolor='rgba(128,128,128,0.2)', zeroline=False, tickformat='.1%')
    
    return fig

def main():
    # Check authentication
    is_authenticated, user_info = check_authentication()
    if not is_authenticated:
        st.warning("Please log in to access the Strategy Backtesting module.")
        return
        
    st.title("🎯 Strategy Backtesting")
    st.markdown("**Professional backtesting engine for quantitative trading strategies**")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Symbol selection
        symbol = st.text_input("Stock Symbol", value="AAPL", help="Enter stock ticker symbol")
        
        # Date range selection
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=365*2),
                max_value=datetime.now()
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                max_value=datetime.now()
            )
        
        # Backtesting parameters
        st.subheader("Backtesting Parameters")
        initial_capital = st.number_input("Initial Capital ($)", value=100000, min_value=1000, step=1000)
        commission = st.number_input("Commission (%)", value=0.1, min_value=0.0, max_value=5.0, step=0.1) / 100
        transaction_costs = st.checkbox("Include Transaction Costs", value=True)
        
        # Strategy selection
        st.subheader("Strategy Selection")
        strategies_to_run = st.multiselect(
            "Select Strategies",
            ["Moving Average Crossover", "RSI Mean Reversion", "Bollinger Bands", "Momentum"],
            default=["Moving Average Crossover", "RSI Mean Reversion"]
        )
        
        # Strategy parameters
        st.subheader("Strategy Parameters")
        
        # Moving Average parameters
        if "Moving Average Crossover" in strategies_to_run:
            st.write("**Moving Average Crossover**")
            ma_short = st.slider("Short MA Period", 5, 50, 20)
            ma_long = st.slider("Long MA Period", 20, 200, 50)
        
        # RSI parameters
        if "RSI Mean Reversion" in strategies_to_run:
            st.write("**RSI Mean Reversion**")
            rsi_period = st.slider("RSI Period", 5, 30, 14)
            rsi_overbought = st.slider("Overbought Level", 60, 90, 70)
            rsi_oversold = st.slider("Oversold Level", 10, 40, 30)
        
        # Bollinger Bands parameters
        if "Bollinger Bands" in strategies_to_run:
            st.write("**Bollinger Bands**")
            bb_period = st.slider("BB Period", 10, 50, 20)
            bb_std = st.slider("Standard Deviations", 1.0, 3.0, 2.0, step=0.1)
        
        # Momentum parameters
        if "Momentum" in strategies_to_run:
            st.write("**Momentum**")
            momentum_lookback = st.slider("Lookback Period", 5, 50, 20)
            momentum_threshold = st.slider("Momentum Threshold (%)", 0.5, 10.0, 2.0, step=0.5) / 100
        
        run_backtest = st.button("🚀 Run Backtest", type="primary")
    
    if run_backtest and strategies_to_run:
        with st.spinner("Running backtest analysis..."):
            try:
                # Download data
                data = yf.download(symbol, start=start_date, end=end_date)
                
                if data is None or len(data) == 0:
                    st.error(f"No data available for {symbol}")
                    return
                
                # Initialize backtesting engine
                engine = BacktestEngine(initial_capital=initial_capital, commission=commission)
                
                # Initialize strategies
                strategies = {}
                
                if "Moving Average Crossover" in strategies_to_run:
                    strategies["Moving Average Crossover"] = MovingAverageCrossover(ma_short, ma_long)
                
                if "RSI Mean Reversion" in strategies_to_run:
                    strategies["RSI Mean Reversion"] = RSIStrategy(rsi_period, rsi_overbought, rsi_oversold)
                
                if "Bollinger Bands" in strategies_to_run:
                    strategies["Bollinger Bands"] = BollingerBandsStrategy(bb_period, int(bb_std))
                
                if "Momentum" in strategies_to_run:
                    strategies["Momentum"] = MomentumStrategy(momentum_lookback, momentum_threshold)
                
                # Run backtests
                results = {}
                metrics = {}
                
                for name, strategy in strategies.items():
                    result, metric = engine.run_backtest(data, strategy, transaction_costs)
                    results[name] = result
                    metrics[name] = metric
                
                # Store results in session state for visualization module
                st.session_state.backtest_results = results
                st.session_state.backtest_metrics = metrics
                st.session_state.backtest_symbol = symbol
                
                # Display results
                st.success(f"✅ Backtest completed for {symbol}")
                
                # Add link to interactive visualization
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.info("💡 View advanced interactive visualizations in the Performance Visualization module")
                with col2:
                    if st.button("📊 Open Interactive Viz", type="primary"):
                        st.switch_page("pages/interactive_backtest_visualization.py")
                
                # Performance summary
                st.header("📊 Performance Summary")
                
                # Create metrics comparison table
                metrics_df = pd.DataFrame(metrics).T
                st.dataframe(metrics_df, use_container_width=True)
                
                # Key metrics cards
                col1, col2, col3, col4 = st.columns(4)
                
                if len(strategies) > 0:
                    best_strategy = max(metrics.keys(), 
                                      key=lambda x: float(metrics[x]['Annual Return'].replace('%', '')) / 100)
                    best_metrics = metrics[best_strategy]
                    
                    with col1:
                        create_metric_card(
                            "Best Strategy",
                            best_strategy,
                            f"{best_metrics['Annual Return']}"
                        )
                    
                    with col2:
                        create_metric_card(
                            "Best Annual Return",
                            best_metrics['Annual Return'],
                            f"Sharpe: {best_metrics['Sharpe Ratio']}"
                        )
                    
                    with col3:
                        create_metric_card(
                            "Max Drawdown",
                            best_metrics['Max Drawdown'],
                            f"Win Rate: {best_metrics['Win Rate']}"
                        )
                    
                    with col4:
                        buy_hold = best_metrics['Buy & Hold Return']
                        create_metric_card(
                            "vs Buy & Hold",
                            buy_hold,
                            f"Trades: {best_metrics['Total Trades']}"
                        )
                
                # Performance charts
                st.header("📈 Performance Analysis")
                
                # Strategy comparison chart
                fig_comparison = create_strategy_comparison_chart(results)
                st.plotly_chart(fig_comparison, use_container_width=True)
                
                # Individual strategy analysis
                if len(results) > 0:
                    strategy_choice = st.selectbox(
                        "Select Strategy for Detailed Analysis",
                        list(results.keys())
                    )
                    
                    selected_result = results[strategy_choice]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Drawdown chart
                        fig_drawdown = create_drawdown_chart(selected_result)
                        st.plotly_chart(fig_drawdown, use_container_width=True)
                    
                    with col2:
                        # Returns distribution
                        strategy_returns = selected_result['strategy_returns'].dropna()
                        fig_dist = px.histogram(
                            x=strategy_returns,
                            nbins=50,
                            title="Daily Returns Distribution",
                            labels={'x': 'Daily Returns', 'y': 'Frequency'}
                        )
                        fig_dist.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white')
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)
                
                # Strategy details
                st.header("🔍 Strategy Details")
                
                for strategy_name, result in results.items():
                    with st.expander(f"{strategy_name} - Detailed Analysis"):
                        
                        # Trading signals visualization
                        fig_signals = go.Figure()
                        
                        # Price line
                        fig_signals.add_trace(go.Scatter(
                            x=result.index,
                            y=result['Close'],
                            mode='lines',
                            name='Price',
                            line=dict(color='#00D4FF', width=2)
                        ))
                        
                        # Buy signals
                        buy_signals = result[result['positions'] > 0]
                        if not buy_signals.empty:
                            fig_signals.add_trace(go.Scatter(
                                x=buy_signals.index,
                                y=buy_signals['Close'],
                                mode='markers',
                                name='Buy Signal',
                                marker=dict(color='#00FF00', size=10, symbol='triangle-up')
                            ))
                        
                        # Sell signals
                        sell_signals = result[result['positions'] < 0]
                        if not sell_signals.empty:
                            fig_signals.add_trace(go.Scatter(
                                x=sell_signals.index,
                                y=sell_signals['Close'],
                                mode='markers',
                                name='Sell Signal',
                                marker=dict(color='#FF0000', size=10, symbol='triangle-down')
                            ))
                        
                        # Add strategy-specific indicators
                        if 'MA_short' in result.columns and len(strategies_to_run) > 0:
                            short_period = 20 if "Moving Average Crossover" in strategies_to_run else 20
                            long_period = 50 if "Moving Average Crossover" in strategies_to_run else 50
                            fig_signals.add_trace(go.Scatter(
                                x=result.index,
                                y=result['MA_short'],
                                mode='lines',
                                name=f'MA({short_period})',
                                line=dict(color='#FFA726', width=1)
                            ))
                            fig_signals.add_trace(go.Scatter(
                                x=result.index,
                                y=result['MA_long'],
                                mode='lines',
                                name=f'MA({long_period})',
                                line=dict(color='#AB47BC', width=1)
                            ))
                        
                        if 'BB_upper' in result.columns:
                            fig_signals.add_trace(go.Scatter(
                                x=result.index,
                                y=result['BB_upper'],
                                mode='lines',
                                name='BB Upper',
                                line=dict(color='#FF6B6B', width=1, dash='dot'),
                                showlegend=False
                            ))
                            fig_signals.add_trace(go.Scatter(
                                x=result.index,
                                y=result['BB_lower'],
                                mode='lines',
                                name='BB Lower',
                                line=dict(color='#FF6B6B', width=1, dash='dot'),
                                fill='tonexty',
                                fillcolor='rgba(255, 107, 107, 0.1)'
                            ))
                        
                        fig_signals.update_layout(
                            title=f'{strategy_name} - Trading Signals',
                            xaxis_title='Date',
                            yaxis_title='Price ($)',
                            hovermode='x unified',
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'),
                            legend=dict(
                                yanchor="top",
                                y=0.99,
                                xanchor="left",
                                x=0.01,
                                bgcolor="rgba(0,0,0,0.5)"
                            )
                        )
                        
                        fig_signals.update_xaxes(gridcolor='rgba(128,128,128,0.2)', zeroline=False)
                        fig_signals.update_yaxes(gridcolor='rgba(128,128,128,0.2)', zeroline=False)
                        
                        st.plotly_chart(fig_signals, use_container_width=True)
                        
                        # Performance metrics for this strategy
                        st.write("**Performance Metrics:**")
                        strategy_metrics = metrics[strategy_name]
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"• **Total Return:** {strategy_metrics['Total Return']}")
                            st.write(f"• **Annual Return:** {strategy_metrics['Annual Return']}")
                            st.write(f"• **Annual Volatility:** {strategy_metrics['Annual Volatility']}")
                        
                        with col2:
                            st.write(f"• **Sharpe Ratio:** {strategy_metrics['Sharpe Ratio']}")
                            st.write(f"• **Max Drawdown:** {strategy_metrics['Max Drawdown']}")
                            st.write(f"• **Calmar Ratio:** {strategy_metrics['Calmar Ratio']}")
                        
                        with col3:
                            st.write(f"• **Win Rate:** {strategy_metrics['Win Rate']}")
                            st.write(f"• **Total Trades:** {strategy_metrics['Total Trades']}")
                            st.write(f"• **Buy & Hold:** {strategy_metrics['Buy & Hold Return']}")
                
            except Exception as e:
                st.error(f"Error running backtest: {str(e)}")
                st.info("Please check your inputs and try again.")
    
    elif run_backtest and not strategies_to_run:
        st.warning("Please select at least one strategy to backtest.")
    
    # Educational content
    with st.expander("📚 Strategy Explanations"):
        st.markdown("""
        **Moving Average Crossover:** 
        Generates buy signals when short-term MA crosses above long-term MA, and sell signals when it crosses below.
        
        **RSI Mean Reversion:** 
        Uses Relative Strength Index to identify overbought (sell) and oversold (buy) conditions.
        
        **Bollinger Bands:** 
        Mean reversion strategy that buys when price touches lower band and sells when price touches upper band.
        
        **Momentum:** 
        Follows price momentum, buying when recent returns exceed threshold and selling when they fall below.
        """)
    
    with st.expander("📊 Performance Metrics Explained"):
        st.markdown("""
        **Total Return:** Cumulative return over the entire backtesting period.
        
        **Annual Return:** Annualized return based on the backtesting period.
        
        **Sharpe Ratio:** Risk-adjusted return metric (return per unit of risk).
        
        **Max Drawdown:** Largest peak-to-trough decline during the backtesting period.
        
        **Win Rate:** Percentage of profitable trades.
        
        **Calmar Ratio:** Annual return divided by maximum drawdown.
        """)

if __name__ == "__main__":
    main()