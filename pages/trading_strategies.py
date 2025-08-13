import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_fetcher import DataFetcher
from utils.trading_strategies import TradingStrategies
from utils.backtesting import BacktestingEngine

# Page configuration
st.set_page_config(
    page_title="Trading Strategies",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Trading Strategy Backtesting")
st.markdown("Comprehensive backtesting of various trading strategies with detailed performance analysis")

# Sidebar for inputs
st.sidebar.header("Strategy Configuration")

# Asset selection
ticker_input = st.sidebar.text_input(
    "Enter Stock Ticker",
    value="AAPL",
    help="Enter a valid stock ticker symbol"
).upper()

# Quick selection from popular tickers
popular_tickers = {
    "Large Cap Tech": ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "TSLA"],
    "Blue Chip": ["JNJ", "JPM", "PG", "KO", "DIS", "WMT"],
    "Growth Stocks": ["NFLX", "AMZN", "CRM", "SHOP", "ZM", "ROKU"],
    "ETFs": ["SPY", "QQQ", "IWM", "VTI", "XLK", "XLF"],
    "Crypto ETFs": ["BITO", "ETHE", "COIN"],
}

selected_category = st.sidebar.selectbox("Or select from categories", [""] + list(popular_tickers.keys()))

if selected_category:
    selected_ticker = st.sidebar.selectbox(
        f"Select from {selected_category}",
        popular_tickers[selected_category]
    )
    if st.sidebar.button(f"Analyze {selected_ticker}"):
        ticker_input = selected_ticker

# Time period selection
period_options = {
    "1 Year": "1y",
    "2 Years": "2y",
    "3 Years": "3y",
    "5 Years": "5y"
}

selected_period = st.sidebar.selectbox(
    "Backtest Period",
    list(period_options.keys()),
    index=2
)

# Strategy selection
st.sidebar.subheader("Strategy Selection")

strategies_to_test = st.sidebar.multiselect(
    "Select strategies to backtest",
    [
        "Moving Average Crossover",
        "RSI Mean Reversion", 
        "Bollinger Bands",
        "Momentum Strategy",
        "Mean Reversion"
    ],
    default=["Moving Average Crossover", "RSI Mean Reversion"]
)

# Backtesting parameters
with st.sidebar.expander("Backtesting Parameters"):
    initial_capital = st.number_input(
        "Initial Capital ($)",
        min_value=1000,
        max_value=10000000,
        value=100000,
        step=10000
    )
    
    commission_rate = st.slider(
        "Commission Rate (%)",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05
    ) / 100
    
    slippage_rate = st.slider(
        "Slippage Rate (%)",
        min_value=0.0,
        max_value=0.5,
        value=0.01,
        step=0.01
    ) / 100

# Strategy-specific parameters
strategy_params = {}

if "Moving Average Crossover" in strategies_to_test:
    with st.sidebar.expander("Moving Average Parameters"):
        strategy_params["MA"] = {
            "short_window": st.slider("Short MA Window", 5, 50, 20),
            "long_window": st.slider("Long MA Window", 20, 200, 50)
        }

if "RSI Mean Reversion" in strategies_to_test:
    with st.sidebar.expander("RSI Parameters"):
        strategy_params["RSI"] = {
            "rsi_window": st.slider("RSI Window", 5, 30, 14),
            "oversold": st.slider("Oversold Level", 10, 40, 30),
            "overbought": st.slider("Overbought Level", 60, 90, 70)
        }

if "Bollinger Bands" in strategies_to_test:
    with st.sidebar.expander("Bollinger Bands Parameters"):
        strategy_params["BB"] = {
            "window": st.slider("BB Window", 10, 50, 20),
            "num_std": st.slider("Standard Deviations", 1.0, 3.0, 2.0, 0.1)
        }

if "Momentum Strategy" in strategies_to_test:
    with st.sidebar.expander("Momentum Parameters"):
        strategy_params["Momentum"] = {
            "lookback": st.slider("Lookback Period", 5, 30, 10),
            "holding_period": st.slider("Holding Period", 1, 10, 5)
        }

if "Mean Reversion" in strategies_to_test:
    with st.sidebar.expander("Mean Reversion Parameters"):
        strategy_params["MeanRev"] = {
            "window": st.slider("MR Window", 10, 50, 20),
            "threshold": st.slider("Z-Score Threshold", 1.0, 3.0, 1.5, 0.1)
        }

if not ticker_input:
    st.warning("Please enter a stock ticker symbol.")
    st.stop()

if not strategies_to_test:
    st.warning("Please select at least one strategy to backtest.")
    st.stop()

# Data fetching
with st.spinner(f"Fetching data for {ticker_input}..."):
    try:
        # Validate ticker
        valid_tickers, invalid_tickers = DataFetcher.validate_tickers([ticker_input])
        
        if invalid_tickers:
            st.error(f"Invalid ticker: {ticker_input}")
            st.stop()
        
        # Fetch OHLCV data
        period = period_options[selected_period]
        price_data = DataFetcher.get_stock_data(ticker_input, period=period)
        
        if price_data.empty:
            st.error(f"Unable to fetch data for {ticker_input}")
            st.stop()
        
        # Ensure we have OHLCV structure
        if isinstance(price_data.columns, pd.MultiIndex):
            # Multi-index columns (single ticker with OHLCV)
            ohlcv_data = price_data.droplevel(1, axis=1)
        else:
            # Simple columns structure
            ohlcv_data = price_data
        
        # Validate required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in ohlcv_data.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            st.stop()
        
        # Initialize trading strategies
        strategy_engine = TradingStrategies(ohlcv_data, ticker_input)
        backtesting_engine = BacktestingEngine(
            initial_capital=initial_capital,
            commission=commission_rate,
            slippage=slippage_rate
        )
        
        # Get stock info for context
        stock_info = DataFetcher.get_stock_info(ticker_input)
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        st.stop()

# Display stock information
if stock_info:
    st.header(f"📊 {ticker_input} - {stock_info.get('longName', 'N/A')}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = ohlcv_data['Close'].iloc[-1]
        st.metric("Current Price", f"${current_price:.2f}")
    
    with col2:
        daily_change = (ohlcv_data['Close'].iloc[-1] - ohlcv_data['Close'].iloc[-2]) / ohlcv_data['Close'].iloc[-2]
        st.metric("Daily Change", f"{daily_change:.2%}")
    
    with col3:
        period_return = (ohlcv_data['Close'].iloc[-1] - ohlcv_data['Close'].iloc[0]) / ohlcv_data['Close'].iloc[0]
        st.metric(f"{selected_period} Return", f"{period_return:.2%}")
    
    with col4:
        volatility = ohlcv_data['Close'].pct_change().std() * np.sqrt(252)
        st.metric("Annualized Volatility", f"{volatility:.2%}")

# Strategy Backtesting
st.header("🎯 Strategy Backtesting Results")

strategy_results = {}

# Run backtests for selected strategies
with st.spinner("Running strategy backtests..."):
    try:
        # Moving Average Crossover
        if "Moving Average Crossover" in strategies_to_test:
            params = strategy_params.get("MA", {"short_window": 20, "long_window": 50})
            ma_data = strategy_engine.moving_average_strategy(**params)
            
            # Create signals for backtesting
            ma_signals = ma_data['Signal'].copy()
            ma_backtest = backtesting_engine.backtest_signals(ohlcv_data, ma_signals)
            ma_metrics = backtesting_engine.calculate_performance_metrics(ma_backtest)
            
            strategy_results["Moving Average"] = {
                "data": ma_data,
                "backtest": ma_backtest,
                "metrics": ma_metrics,
                "params": params
            }
        
        # RSI Mean Reversion
        if "RSI Mean Reversion" in strategies_to_test:
            params = strategy_params.get("RSI", {"rsi_window": 14, "oversold": 30, "overbought": 70})
            rsi_data = strategy_engine.rsi_strategy(**params)
            
            rsi_signals = rsi_data['Position'].copy()
            rsi_backtest = backtesting_engine.backtest_signals(ohlcv_data, rsi_signals)
            rsi_metrics = backtesting_engine.calculate_performance_metrics(rsi_backtest)
            
            strategy_results["RSI"] = {
                "data": rsi_data,
                "backtest": rsi_backtest,
                "metrics": rsi_metrics,
                "params": params
            }
        
        # Bollinger Bands
        if "Bollinger Bands" in strategies_to_test:
            params = strategy_params.get("BB", {"window": 20, "num_std": 2.0})
            bb_data = strategy_engine.bollinger_bands_strategy(**params)
            
            bb_signals = bb_data['Position'].copy()
            bb_backtest = backtesting_engine.backtest_signals(ohlcv_data, bb_signals)
            bb_metrics = backtesting_engine.calculate_performance_metrics(bb_backtest)
            
            strategy_results["Bollinger Bands"] = {
                "data": bb_data,
                "backtest": bb_backtest,
                "metrics": bb_metrics,
                "params": params
            }
        
        # Momentum Strategy
        if "Momentum Strategy" in strategies_to_test:
            params = strategy_params.get("Momentum", {"lookback": 10, "holding_period": 5})
            mom_data = strategy_engine.momentum_strategy(**params)
            
            mom_signals = mom_data['Position'].copy()
            mom_backtest = backtesting_engine.backtest_signals(ohlcv_data, mom_signals)
            mom_metrics = backtesting_engine.calculate_performance_metrics(mom_backtest)
            
            strategy_results["Momentum"] = {
                "data": mom_data,
                "backtest": mom_backtest,
                "metrics": mom_metrics,
                "params": params
            }
        
        # Mean Reversion
        if "Mean Reversion" in strategies_to_test:
            params = strategy_params.get("MeanRev", {"window": 20, "threshold": 1.5})
            mr_data = strategy_engine.mean_reversion_strategy(**params)
            
            mr_signals = mr_data['Position'].copy()
            mr_backtest = backtesting_engine.backtest_signals(ohlcv_data, mr_signals)
            mr_metrics = backtesting_engine.calculate_performance_metrics(mr_backtest)
            
            strategy_results["Mean Reversion"] = {
                "data": mr_data,
                "backtest": mr_backtest,
                "metrics": mr_metrics,
                "params": params
            }
            
    except Exception as e:
        st.error(f"Error running backtests: {str(e)}")
        st.stop()

# Performance Comparison Table
st.subheader("📈 Performance Comparison")

if strategy_results:
    comparison_data = []
    
    # Add Buy & Hold baseline
    buy_hold_return = (ohlcv_data['Close'].iloc[-1] - ohlcv_data['Close'].iloc[0]) / ohlcv_data['Close'].iloc[0]
    buy_hold_volatility = ohlcv_data['Close'].pct_change().std() * np.sqrt(252)
    buy_hold_sharpe = (buy_hold_return * 252/len(ohlcv_data) - 0.02) / buy_hold_volatility if buy_hold_volatility != 0 else 0
    
    comparison_data.append({
        "Strategy": "Buy & Hold",
        "Total Return": f"{buy_hold_return:.2%}",
        "Annualized Return": f"{buy_hold_return * 252/len(ohlcv_data):.2%}",
        "Volatility": f"{buy_hold_volatility:.2%}",
        "Sharpe Ratio": f"{buy_hold_sharpe:.3f}",
        "Max Drawdown": "N/A",
        "Win Rate": "N/A",
        "Total Trades": "1"
    })
    
    # Add strategy results
    for strategy_name, results in strategy_results.items():
        metrics = results["metrics"]
        comparison_data.append({
            "Strategy": strategy_name,
            "Total Return": f"{metrics['Total Return']:.2%}",
            "Annualized Return": f"{metrics['Annualized Return']:.2%}",
            "Volatility": f"{metrics['Volatility']:.2%}",
            "Sharpe Ratio": f"{metrics['Sharpe Ratio']:.3f}",
            "Max Drawdown": f"{metrics['Max Drawdown']:.2%}",
            "Win Rate": f"{metrics['Win Rate']:.2%}",
            "Total Trades": f"{metrics['Total Trades']:.0f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Highlight best performing strategy
    if len(strategy_results) > 0:
        best_strategy = max(strategy_results.keys(), 
                          key=lambda x: strategy_results[x]["metrics"]["Sharpe Ratio"])
        st.success(f"🏆 Best performing strategy: **{best_strategy}** (Sharpe Ratio: {strategy_results[best_strategy]['metrics']['Sharpe Ratio']:.3f})")

# Detailed Strategy Analysis
st.header("📊 Detailed Strategy Analysis")

if strategy_results:
    selected_strategy = st.selectbox(
        "Select strategy for detailed analysis",
        list(strategy_results.keys())
    )
    
    if selected_strategy:
        strategy_data = strategy_results[selected_strategy]
        
        # Display strategy parameters
        st.subheader(f"{selected_strategy} - Configuration")
        params_col1, params_col2 = st.columns(2)
        
        with params_col1:
            st.write("**Strategy Parameters:**")
            for param, value in strategy_data["params"].items():
                st.write(f"- {param}: {value}")
        
        with params_col2:
            st.write("**Key Metrics:**")
            metrics = strategy_data["metrics"]
            st.write(f"- Total Return: {metrics['Total Return']:.2%}")
            st.write(f"- Sharpe Ratio: {metrics['Sharpe Ratio']:.3f}")
            st.write(f"- Max Drawdown: {metrics['Max Drawdown']:.2%}")
            st.write(f"- Win Rate: {metrics['Win Rate']:.2%}")
        
        # Performance visualization
        try:
            backtest_plots = backtesting_engine.plot_backtest_results(
                strategy_data["backtest"], 
                selected_strategy
            )
            
            # Portfolio performance
            if 'portfolio_value' in backtest_plots:
                st.plotly_chart(backtest_plots['portfolio_value'], use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Drawdown analysis
                if 'drawdown' in backtest_plots:
                    st.plotly_chart(backtest_plots['drawdown'], use_container_width=True)
            
            with col2:
                # Returns distribution
                if 'returns_distribution' in backtest_plots:
                    st.plotly_chart(backtest_plots['returns_distribution'], use_container_width=True)
            
            # Trading signals
            if 'trades' in backtest_plots:
                st.plotly_chart(backtest_plots['trades'], use_container_width=True)
            
            # Rolling metrics
            if 'rolling_metrics' in backtest_plots:
                st.plotly_chart(backtest_plots['rolling_metrics'], use_container_width=True)
                
        except Exception as e:
            st.error(f"Error generating performance plots: {str(e)}")

# Risk Analysis
st.header("⚠️ Risk Analysis")

if strategy_results:
    risk_analysis_strategy = st.selectbox(
        "Select strategy for risk analysis",
        list(strategy_results.keys()),
        key="risk_analysis"
    )
    
    if risk_analysis_strategy:
        strategy_data = strategy_results[risk_analysis_strategy]
        results_df = strategy_data["backtest"]["results_df"]
        returns = results_df['Returns'].dropna()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            var_95 = np.percentile(returns, 5)
            st.metric("VaR (95%)", f"{var_95:.2%}")
        
        with col2:
            tail_returns = returns[returns <= var_95]
            es_95 = tail_returns.mean() if len(tail_returns) > 0 else var_95
            st.metric("Expected Shortfall (95%)", f"{es_95:.2%}")
        
        with col3:
            downside_returns = returns[returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            st.metric("Downside Deviation", f"{downside_deviation:.2%}")
        
        with col4:
            calmar_ratio = strategy_data["metrics"]["Calmar Ratio"]
            st.metric("Calmar Ratio", f"{calmar_ratio:.3f}")

# Strategy Optimization Suggestions
st.header("💡 Optimization Suggestions")

if strategy_results:
    st.subheader("Performance Insights")
    
    # Analyze best and worst performing strategies
    if len(strategy_results) > 1:
        sorted_strategies = sorted(strategy_results.items(), 
                                 key=lambda x: x[1]["metrics"]["Sharpe Ratio"], 
                                 reverse=True)
        
        best_strategy = sorted_strategies[0]
        worst_strategy = sorted_strategies[-1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"**Best Strategy: {best_strategy[0]}**")
            st.write(f"- Sharpe Ratio: {best_strategy[1]['metrics']['Sharpe Ratio']:.3f}")
            st.write(f"- Total Return: {best_strategy[1]['metrics']['Total Return']:.2%}")
            st.write(f"- Max Drawdown: {best_strategy[1]['metrics']['Max Drawdown']:.2%}")
            
            # Optimization suggestions for best strategy
            if best_strategy[1]['metrics']['Win Rate'] < 0.5:
                st.info("💡 Consider tightening entry criteria to improve win rate")
            
            if abs(best_strategy[1]['metrics']['Max Drawdown']) > 0.2:
                st.info("💡 Consider adding stop-loss rules to reduce drawdown")
        
        with col2:
            st.error(f"**Needs Improvement: {worst_strategy[0]}**")
            st.write(f"- Sharpe Ratio: {worst_strategy[1]['metrics']['Sharpe Ratio']:.3f}")
            st.write(f"- Total Return: {worst_strategy[1]['metrics']['Total Return']:.2%}")
            st.write(f"- Max Drawdown: {worst_strategy[1]['metrics']['Max Drawdown']:.2%}")
            
            # Suggestions for improvement
            if worst_strategy[1]['metrics']['Total Trades'] < 10:
                st.info("💡 Strategy may be too restrictive - consider relaxing entry criteria")
            
            if worst_strategy[1]['metrics']['Volatility'] > buy_hold_volatility * 1.5:
                st.info("💡 High volatility - consider position sizing or risk management")

# Export Results
st.header("📥 Export Results")

col1, col2 = st.columns(2)

with col1:
    if st.button("📊 Export Performance Comparison", use_container_width=True):
        if strategy_results:
            csv = comparison_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"{ticker_input}_strategy_comparison_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

with col2:
    if st.button("📈 Export Detailed Results", use_container_width=True):
        if strategy_results and selected_strategy:
            strategy_data = strategy_results[selected_strategy]
            results_df = strategy_data["backtest"]["results_df"]
            
            export_df = results_df[['Portfolio_Value', 'Returns', 'Position']].copy()
            export_df['Strategy'] = selected_strategy
            
            csv = export_df.to_csv()
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"{ticker_input}_{selected_strategy.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

# Navigation
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🏠 Back to Dashboard", use_container_width=True):
        st.switch_page("app.py")

with col2:
    if st.button("📈 Time Series Analysis", use_container_width=True):
        st.switch_page("pages/time_series_analysis.py")

with col3:
    if st.button("🤖 AI Analysis", use_container_width=True):
        st.switch_page("pages/ai_analysis.py")
