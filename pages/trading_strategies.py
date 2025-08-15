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
from utils.tooltips import get_tooltip_help
from utils.ai_strategy_optimizer import AIStrategyOptimizer
from utils.pinescript_generator import PineScriptGenerator

# Page configuration
st.set_page_config(
    page_title="Trading Strategies",
    page_icon="⚡",
    layout="wide"
)

# Modern header with enhanced styling
st.markdown("""
<div class="main-header">
    <h1>⚡ Trading Strategy Backtesting</h1>
    <p>Comprehensive backtesting of various trading strategies with detailed performance analysis</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for inputs
st.sidebar.header("Strategy Configuration")

# Asset selection
ticker_input = st.sidebar.text_input(
    "Enter Stock Ticker",
    value="SPY",
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

# Time period and interval selection
st.sidebar.subheader("📊 Timeframe Configuration")

# Timeframe selection with interval support
timeframe_options = {
    "1 Hour (30 days)": {"period": "30d", "interval": "1h"},
    "4 Hours (60 days)": {"period": "60d", "interval": "4h"}, 
    "Daily (1 Year)": {"period": "1y", "interval": "1d"},
    "Daily (2 Years)": {"period": "2y", "interval": "1d"},
    "Daily (3 Years)": {"period": "3y", "interval": "1d"},
    "Daily (5 Years)": {"period": "5y", "interval": "1d"}
}

selected_timeframe = st.sidebar.selectbox(
    "Backtest Timeframe",
    list(timeframe_options.keys()),
    index=2,
    help="Choose timeframe - intraday (1h, 4h) or daily backtesting"
)

# Extract period and interval from selection
period_config = timeframe_options[selected_timeframe]
selected_period = period_config["period"]
selected_interval = period_config["interval"]

# Display timeframe info
if selected_interval in ["1h", "4h"]:
    st.sidebar.info(f"📈 Intraday backtesting: {selected_interval} bars over {selected_period}")
else:
    st.sidebar.info(f"📊 Daily backtesting: {selected_interval} bars over {selected_period}")

# Analysis Mode Selection
st.sidebar.subheader("Analysis Mode")

analysis_mode = st.sidebar.radio(
    "Choose Analysis Mode",
    ["🤖 AI Strategy Optimization", "📊 Traditional Backtesting"],
    help="AI mode optimizes strategies to minimize drawdown while maximizing returns"
)

if analysis_mode == "📊 Traditional Backtesting":
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
else:
    st.sidebar.markdown("🤖 **AI will optimize strategies automatically**")
    target_metric = st.sidebar.selectbox(
        "Optimization Target",
        ["calmar_ratio", "sharpe_ratio", "total_return"],
        help="Calmar ratio balances returns vs drawdown"
    )
    
    max_drawdown_target = st.sidebar.slider(
        "Maximum Acceptable Drawdown (%)",
        5, 30, 15, 1,
        help="AI will try to keep drawdown below this level"
    ) / 100
    
    # Set empty list for AI mode
    strategies_to_test = []

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

# Strategy-specific parameters (only for traditional backtesting)
strategy_params = {}

if analysis_mode == "📊 Traditional Backtesting" and "Moving Average Crossover" in strategies_to_test:
    with st.sidebar.expander("Moving Average Parameters"):
        strategy_params["MA"] = {
            "short_window": st.slider("Short MA Window", 5, 50, 20),
            "long_window": st.slider("Long MA Window", 20, 200, 50)
        }

if analysis_mode == "📊 Traditional Backtesting" and "RSI Mean Reversion" in strategies_to_test:
    with st.sidebar.expander("RSI Parameters"):
        strategy_params["RSI"] = {
            "rsi_window": st.slider("RSI Window", 5, 30, 14),
            "oversold": st.slider("Oversold Level", 10, 40, 30),
            "overbought": st.slider("Overbought Level", 60, 90, 70)
        }

if analysis_mode == "📊 Traditional Backtesting" and "Bollinger Bands" in strategies_to_test:
    with st.sidebar.expander("Bollinger Bands Parameters"):
        strategy_params["BB"] = {
            "window": st.slider("BB Window", 10, 50, 20),
            "num_std": st.slider("Standard Deviations", 1.0, 3.0, 2.0, 0.1)
        }

if analysis_mode == "📊 Traditional Backtesting" and "Momentum Strategy" in strategies_to_test:
    with st.sidebar.expander("Momentum Parameters"):
        strategy_params["Momentum"] = {
            "lookback": st.slider("Lookback Period", 5, 30, 10),
            "holding_period": st.slider("Holding Period", 1, 10, 5)
        }

if analysis_mode == "📊 Traditional Backtesting" and "Mean Reversion" in strategies_to_test:
    with st.sidebar.expander("Mean Reversion Parameters"):
        strategy_params["MeanRev"] = {
            "window": st.slider("MR Window", 10, 50, 20),
            "threshold": st.slider("Z-Score Threshold", 1.0, 3.0, 1.5, 0.1)
        }

if not ticker_input:
    st.warning("Please enter a stock ticker symbol.")
    st.stop()

if analysis_mode == "📊 Traditional Backtesting" and not strategies_to_test:
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
        
        # Fetch OHLCV data with interval support
        price_data = DataFetcher.get_stock_data(ticker_input, period=selected_period, interval=selected_interval)
        
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
        
        # Initialize strategy results dictionary
        strategy_results = {}
        
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
        recent_change = (ohlcv_data['Close'].iloc[-1] - ohlcv_data['Close'].iloc[-2]) / ohlcv_data['Close'].iloc[-2]
        if selected_interval in ["1h", "4h"]:
            st.metric(f"Last {selected_interval} Change", f"{recent_change:.2%}")
        else:
            st.metric("Daily Change", f"{recent_change:.2%}")
    
    with col3:
        period_return = (ohlcv_data['Close'].iloc[-1] - ohlcv_data['Close'].iloc[0]) / ohlcv_data['Close'].iloc[0]
        st.metric(f"Period Return", f"{period_return:.2%}")
    
    with col4:
        # Adjust volatility calculation based on interval
        returns = ohlcv_data['Close'].pct_change().dropna()
        if selected_interval == "1h":
            # 24 hours * 365 days for 1h data (approximate)
            volatility = returns.std() * np.sqrt(24 * 365)
        elif selected_interval == "4h":
            # 6 periods * 365 days for 4h data (approximate)
            volatility = returns.std() * np.sqrt(6 * 365)
        else:
            # Standard daily volatility (252 trading days)
            volatility = returns.std() * np.sqrt(252)
        st.metric("Annualized Volatility", f"{volatility:.2%}")

# Main Analysis Section
if analysis_mode == "🤖 AI Strategy Optimization":
    st.header("🤖 AI Strategy Optimization Results")
    
    with st.spinner("Running AI optimization to find best strategies with minimal drawdown..."):
        try:
            # Initialize AI optimizer
            ai_optimizer = AIStrategyOptimizer(ohlcv_data, ticker_input)
            
            # Generate comprehensive recommendations
            recommendations = ai_optimizer.optimize_strategy()
            
            # Display AI optimization results
            if recommendations:
                best_strategy = recommendations['best_strategy']
                all_strategies = recommendations.get('all_strategies', [])
                
                st.subheader("🏆 Best Optimized Strategy")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Strategy", best_strategy['name'])
                    st.metric("Annual Return", f"{best_strategy['annual_return']:.2%}")
                
                with col2:
                    st.metric("Sharpe Ratio", f"{best_strategy['sharpe_ratio']:.3f}")
                    st.metric("Win Rate", f"{best_strategy['win_rate']:.1f}%")
                
                with col3:
                    st.metric("Max Drawdown", f"{best_strategy['max_drawdown']:.2%}")
                    st.metric("Total Trades", best_strategy['total_trades'])
                
                with col4:
                    st.metric("Calmar Ratio", f"{best_strategy['calmar_ratio']:.3f}")
                    if 'params' in best_strategy:
                        st.info(f"Parameters: {best_strategy['params']}")
                
                # Show top 5 strategies comparison
                if len(all_strategies) > 1:
                    st.subheader("📊 Top Strategy Comparison")
                    
                    comparison_data = []
                    for i, strategy in enumerate(all_strategies[:5]):
                        comparison_data.append({
                            "Rank": i + 1,
                            "Strategy": strategy['name'],
                            "Annual Return": f"{strategy['annual_return']:.2%}",
                            "Sharpe Ratio": f"{strategy['sharpe_ratio']:.3f}",
                            "Max Drawdown": f"{strategy['max_drawdown']:.2%}",
                            "Win Rate": f"{strategy['win_rate']:.1f}%",
                            "Calmar Ratio": f"{strategy['calmar_ratio']:.3f}"
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                
                # PineScript Generation for Best Strategy
                if best_strategy:
                    st.subheader("📋 TradingView PineScript - Best Strategy")
                    
                    # Generate PineScript based on strategy type
                    strategy_type = best_strategy.get('type', 'unknown')
                    
                    if 'ma' in strategy_type.lower() or 'moving_average' in strategy_type.lower():
                        params = best_strategy.get('params', {})
                        fast_ma = params.get('short_period', params.get('fast', 20))
                        slow_ma = params.get('long_period', params.get('slow', 50))
                        from utils.pinescript_generator import PineScriptGenerator
                        pinescript_code = PineScriptGenerator.generate_moving_average_strategy(fast_ma, slow_ma, ticker_input)
                    
                    elif 'rsi' in strategy_type.lower():
                        params = best_strategy.get('params', {})
                        rsi_period = params.get('period', 14)
                        oversold = params.get('oversold', 30)
                        overbought = params.get('overbought', 70)
                        from utils.pinescript_generator import PineScriptGenerator
                        pinescript_code = PineScriptGenerator.generate_rsi_strategy(rsi_period, oversold, overbought, ticker_input)
                    
                    else:
                        # Generate ensemble strategy for complex strategies
                        from utils.pinescript_generator import PineScriptGenerator
                        pinescript_code = PineScriptGenerator.generate_ensemble_strategy(
                            ['Moving Average', 'RSI', 'Bollinger Bands'], 
                            ticker_input
                        )
                    
                    st.success(f"**Ready-to-use PineScript for {best_strategy['name']} Strategy**")
                    st.markdown("**Instructions:**")
                    st.markdown("1. Copy the code below")
                    st.markdown("2. Go to TradingView.com → Pine Editor")
                    st.markdown("3. Paste the code and click 'Add to Chart'")
                    st.markdown("4. The strategy will automatically trade with optimized parameters")
                    
                    st.code(pinescript_code, language="javascript")
                    
                    # Copy button helper
                    st.markdown("💡 **Tip:** Click the copy button in the top-right corner of the code block")
                
                # Strategy Details
                st.subheader("🔍 Strategy Analysis Details")
                
                with st.expander("Strategy Parameter Details", expanded=False):
                    st.write("**Selected Strategy Details:**")
                    st.write(f"• **Type:** {best_strategy['type']}")
                    st.write(f"• **Parameters:** {best_strategy['params']}")
                    
                    if 'calculation_details' in recommendations:
                        calc_details = recommendations['calculation_details']
                        st.write("**Performance Calculation Details:**")
                        st.write(f"• **Total Trading Days:** {calc_details.get('total_days', 'N/A')}")
                        st.write(f"• **Mean Daily Return:** {calc_details.get('mean_daily_return', 0):.6f}")
                        st.write(f"• **Daily Return Std:** {calc_details.get('daily_return_std', 0):.6f}")
                        st.write(f"• **Peak Portfolio Value:** ${calc_details.get('peak_value', 0):,.2f}")
                        st.write(f"• **Trough Portfolio Value:** ${calc_details.get('trough_value', 0):,.2f}")
                
                # Risk Assessment
                if 'monte_carlo' in recommendations and recommendations['monte_carlo']:
                    st.subheader("🎲 Risk Assessment")
                    mc_results = recommendations['monte_carlo']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("95% Confidence Lower", f"{mc_results.get('ci_lower', 0):.1%}")
                    with col2:
                        st.metric("95% Confidence Upper", f"{mc_results.get('ci_upper', 0):.1%}")
                    with col3:
                        st.metric("Probability of Loss", f"{mc_results.get('prob_loss', 0):.1f}%")
                    
                    st.info("🔬 This strategy uses AI to dynamically adjust position sizes based on volatility and current drawdown, helping minimize downside while preserving upside.")
                
                # Ensemble strategy
                if 'ensemble' in recommendations:
                    st.subheader("🎯 AI Ensemble Strategy")
                    ensemble_metrics = recommendations['ensemble']
                    
                    st.success("**AI Ensemble:** Combines multiple signals using machine learning")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Return", f"{ensemble_metrics.get('total_return', 0):.2%}")
                    with col2:
                        st.metric("Max Drawdown", f"{ensemble_metrics.get('max_drawdown', 0):.2%}")
                    with col3:
                        st.metric("Sharpe Ratio", f"{ensemble_metrics.get('sharpe_ratio', 0):.3f}")
                
                # Feature importance
                if 'feature_importance' in recommendations and recommendations['feature_importance']:
                    st.subheader("🧠 AI Feature Importance")
                    
                    feature_df = pd.DataFrame([
                        {'Feature': k, 'Importance': v} 
                        for k, v in sorted(recommendations['feature_importance'].items(), 
                                         key=lambda x: x[1], reverse=True)[:10]
                    ])
                    
                    fig = go.Figure(data=go.Bar(
                        x=feature_df['Importance'],
                        y=feature_df['Feature'],
                        orientation='h'
                    ))
                    fig.update_layout(
                        title="Top 10 Most Important Features for Strategy Success",
                        xaxis_title="Importance Score",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Strategy recommendations
                st.subheader("💡 AI Recommendations")
                
                if 'moving_average' in recommendations and recommendations['moving_average']['best']:
                    ma_best = recommendations['moving_average']['best']
                    if ma_best.get('max_drawdown', 0) < -0.15:  # More than 15% drawdown
                        st.warning(f"⚠️ Your current MA({ma_best.get('fast_ma', 20)}, {ma_best.get('slow_ma', 50)}) strategy has high drawdown of {ma_best.get('max_drawdown', 0):.1%}")
                        
                        st.markdown("**AI Suggestions to minimize downside:**")
                        st.markdown("1. 🛡️ Use the **Risk-Adjusted Strategy** with dynamic position sizing")
                        st.markdown("2. 🎯 Combine with **Ensemble Strategy** for better signal quality")
                        st.markdown("3. 📊 Consider shorter MA periods or add volatility filters")
                        st.markdown("4. ⏰ Implement position sizing based on market regime")
                
            else:
                st.error("Unable to generate AI recommendations. Please try with different parameters.")
                
        except Exception as e:
            st.error(f"Error running AI optimization: {str(e)}")

else:
    # Traditional Backtesting Mode
    st.header("📊 Traditional Strategy Backtesting")
    
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
                
        except Exception as e:
            st.error(f"Error running backtests: {str(e)}")
            strategy_results = {}
        
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

# Performance Analysis for Traditional Backtesting Only
if analysis_mode == "📊 Traditional Backtesting":
    st.subheader("📈 Performance Comparison")

    # Add tooltips for trading strategy metrics
    with st.expander("💡 Trading Strategy Metrics Explained"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Sharpe Ratio:**")
            st.markdown(get_tooltip_help("sharpe_ratio")[:150] + "...")
        with col2:
            st.markdown("**Win Rate:**")
            st.markdown(get_tooltip_help("win_rate")[:150] + "...")
        with col3:
            st.markdown("**Max Drawdown:**")
            st.markdown(get_tooltip_help("maximum_drawdown")[:150] + "...")

    if strategy_results:
        comparison_data = []
        
        # Add Buy & Hold baseline - adjusted for different intervals
        buy_hold_return = (ohlcv_data['Close'].iloc[-1] - ohlcv_data['Close'].iloc[0]) / ohlcv_data['Close'].iloc[0]
        
        # Adjust volatility and annualization based on interval
        returns = ohlcv_data['Close'].pct_change().dropna()
        if selected_interval == "1h":
            scaling_factor = 24 * 365  # 24 hours * 365 days
            periods_per_year = 24 * 365
        elif selected_interval == "4h":
            scaling_factor = 6 * 365  # 6 periods * 365 days
            periods_per_year = 6 * 365
        else:
            scaling_factor = 252  # Standard daily
            periods_per_year = 252
            
        buy_hold_volatility = returns.std() * np.sqrt(scaling_factor)
        buy_hold_annualized_return = buy_hold_return * periods_per_year / len(ohlcv_data)
        buy_hold_sharpe = (buy_hold_annualized_return - 0.02) / buy_hold_volatility if buy_hold_volatility != 0 else 0
        
        comparison_data.append({
            "Strategy": "Buy & Hold",
            "Total Return": f"{buy_hold_return:.2%}",
            "Annualized Return": f"{buy_hold_annualized_return:.2%}",
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
                
                # Need to properly access buy_hold_volatility variable
                try:
                    if worst_strategy[1]['metrics']['Volatility'] > 0.3:  # Fixed reference
                        st.info("💡 High volatility - consider position sizing or risk management")
                except:
                    pass

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
