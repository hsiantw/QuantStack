import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import UI components and strategy class
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.ui_components import create_enhanced_metric_card, display_info_box, create_analysis_header
from utils.golden_cross_strategy import GoldenCrossStrategy

def main():
    # Apply custom CSS from main app
    st.markdown("""
    <style>
    .stButton > button {
        background: linear-gradient(45deg, #1f77b4, #17becf) !important;
        color: white !important;
        border: none !important;
        padding: 1rem 1.5rem !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 6px 20px rgba(31, 119, 180, 0.4) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        width: 100% !important;
        height: 55px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    create_analysis_header("⭐ Golden Cross Strategy", "Simple trend-following strategy: Buy dips during golden cross, exit on death cross")
    
    # Strategy explanation
    st.markdown("""
    ### 📋 Strategy Overview
    
    This pyramid strategy builds positions gradually during bullish trends:
    
    **🟡 Golden Cross Entry**: When 50-day MA crosses above 200-day MA (bullish trend confirmed)
    **🟢 Pyramid Buying**: Use percentage of remaining cash on each 1%+ drop (builds larger positions)
    **🔴 Complete Exit**: Sell ALL positions when 50-day MA crosses below 200-day MA (death cross)
    """)
    
    # Sidebar controls
    st.sidebar.header("Strategy Settings")
    
    # Symbol selection
    symbol = st.sidebar.text_input("Stock Symbol", value="SPY", help="Enter stock symbol (e.g., SPY, AAPL, TSLA)")
    
    # Strategy parameters
    st.sidebar.subheader("Strategy Parameters")
    fast_ma = st.sidebar.slider("Fast Moving Average", 20, 100, 50, 5)
    slow_ma = st.sidebar.slider("Slow Moving Average", 100, 300, 200, 10)
    dip_threshold = st.sidebar.slider("Dip Threshold (%)", 0.5, 5.0, 1.0, 0.1) / 100
    pyramid_size = st.sidebar.slider("Pyramid Buy Size (%)", 10, 50, 20, 5) / 100
    
    # Backtest period
    st.sidebar.subheader("Backtest Settings")
    period = st.sidebar.selectbox("Period", ["1y", "2y", "3y", "5y"], index=2)
    initial_capital = st.sidebar.number_input("Initial Capital ($)", 1000, 100000, 10000, 1000)
    
    if st.sidebar.button("Run Strategy Backtest", type="primary"):
        try:
            # Fetch data
            with st.spinner(f"Fetching data for {symbol}..."):
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period)
                
                if data.empty:
                    st.error(f"No data found for symbol {symbol}")
                    return
                
                info = ticker.info
                company_name = info.get('longName', symbol)
            
            # Initialize strategy
            strategy = GoldenCrossStrategy(fast_ma=fast_ma, slow_ma=slow_ma, dip_threshold=dip_threshold, pyramid_size=pyramid_size)
            
            # Calculate signals
            with st.spinner("Calculating trading signals..."):
                results = strategy.calculate_signals(data)
                results, trades = strategy.backtest_strategy(results, initial_capital)
                performance = strategy.calculate_performance_metrics(results, trades)
            
            # Display results
            st.markdown(f"## 📊 Strategy Results for {company_name} ({symbol})")
            
            # Performance metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_return = performance.get('Total Return (%)', 0)
                create_enhanced_metric_card("Total Return", f"{total_return:.1f}%", icon="💰")
            
            with col2:
                sharpe = performance.get('Sharpe Ratio', 0)
                create_enhanced_metric_card("Sharpe Ratio", f"{sharpe:.2f}", icon="⭐")
            
            with col3:
                max_dd = performance.get('Max Drawdown (%)', 0)
                create_enhanced_metric_card("Max Drawdown", f"{max_dd:.1f}%", icon="📉")
            
            with col4:
                total_buys = performance.get('Total Buy Orders', 0)
                create_enhanced_metric_card("Total Buy Orders", str(total_buys), icon="🛒")
            
            # Additional metrics
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                win_rate = performance.get('Win Rate (%)', 0)
                create_enhanced_metric_card("Win Rate", f"{win_rate:.1f}%", icon="🎯")
            
            with col6:
                volatility = performance.get('Volatility (%)', 0)
                create_enhanced_metric_card("Volatility", f"{volatility:.1f}%", icon="📊")
            
            with col7:
                profit_factor = performance.get('Profit Factor', 0)
                if profit_factor == float('inf'):
                    pf_display = "∞"
                else:
                    pf_display = f"{profit_factor:.2f}"
                create_enhanced_metric_card("Profit Factor", pf_display, icon="💎")
            
            with col8:
                sell_cycles = performance.get('Total Sell Cycles', 0)
                create_enhanced_metric_card("Sell Cycles", str(sell_cycles), icon="🔄")
            
            # Strategy chart
            st.markdown("### 📈 Strategy Visualization")
            fig = strategy.create_strategy_chart(results, trades)
            st.plotly_chart(fig, use_container_width=True)
            
            # Compare with buy & hold
            st.markdown("### 📊 Strategy vs Buy & Hold Comparison")
            
            buy_hold_return = ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
            strategy_return = performance.get('Total Return (%)', 0)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                create_enhanced_metric_card("Strategy Return", f"{strategy_return:.1f}%", icon="🚀")
            with col2:
                create_enhanced_metric_card("Buy & Hold Return", f"{buy_hold_return:.1f}%", icon="📈")
            with col3:
                outperformance = strategy_return - buy_hold_return
                create_enhanced_metric_card("Outperformance", f"{outperformance:+.1f}%", icon="⚡")
            
            # Trade history
            if trades:
                st.markdown("### 📋 Trade History")
                
                trade_df = pd.DataFrame(trades)
                
                # Separate buy and sell trades
                buy_trades = trade_df[trade_df['Action'] == 'BUY']
                sell_trades = trade_df[trade_df['Action'] == 'SELL_ALL']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🛒 Buy Orders (Pyramid)")
                    if not buy_trades.empty:
                        display_buys = buy_trades.copy()
                        display_buys['Date'] = pd.to_datetime(display_buys['Date']).dt.strftime('%Y-%m-%d')
                        display_buys['Price'] = display_buys['Price'].apply(lambda x: f"${x:.2f}")
                        display_buys['Amount'] = display_buys['Amount'].apply(lambda x: f"${x:,.0f}")
                        display_buys['Shares'] = display_buys['Shares'].apply(lambda x: f"{x:.2f}")
                        display_buys['Cash_Remaining'] = display_buys['Cash_Remaining'].apply(lambda x: f"${x:,.0f}")
                        
                        st.dataframe(display_buys[['Date', 'Price', 'Amount', 'Shares', 'Cash_Remaining']], use_container_width=True)
                    else:
                        st.info("No buy orders executed yet.")
                
                with col2:
                    st.markdown("#### 💰 Sell Cycles (Exit All)")
                    if not sell_trades.empty:
                        display_sells = sell_trades.copy()
                        display_sells['Date'] = pd.to_datetime(display_sells['Date']).dt.strftime('%Y-%m-%d')
                        display_sells['Price'] = display_sells['Price'].apply(lambda x: f"${x:.2f}")
                        display_sells['Sell_Value'] = display_sells['Sell_Value'].apply(lambda x: f"${x:,.0f}")
                        display_sells['Profit'] = display_sells['Profit'].apply(lambda x: f"${x:,.0f}")
                        display_sells['Return'] = display_sells['Return'].apply(lambda x: f"{x:.1f}%")
                        
                        st.dataframe(display_sells[['Date', 'Price', 'Sell_Value', 'Profit', 'Return']], use_container_width=True)
                    else:
                        st.info("No sell cycles completed yet.")
            
            # Signal analysis
            st.markdown("### 🔍 Signal Analysis")
            
            # Count signals
            golden_crosses = results['Golden_Cross'].sum()
            death_crosses = results['Death_Cross'].sum()
            buy_signals = results['Buy_Signal'].sum()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                create_enhanced_metric_card("Golden Crosses", str(golden_crosses), icon="🟡")
            with col2:
                create_enhanced_metric_card("Buy Signals", str(buy_signals), icon="🟢")
            with col3:
                create_enhanced_metric_card("Death Crosses", str(death_crosses), icon="🔴")
            
            # Current status
            st.markdown("### 🎯 Current Status")
            
            current_regime = results['Market_Regime'].iloc[-1]
            current_position = results['Position'].iloc[-1] if 'Position' in results.columns else 0
            last_return = results['Daily_Return'].iloc[-1]
            
            if current_regime == 1:
                regime_text = "🟢 **Bullish Regime** - Looking for dip buying opportunities"
                if last_return <= -dip_threshold:
                    signal_text = f"🚨 **BUY SIGNAL** - Today's drop ({last_return:.1f}%) triggers buy signal!"
                else:
                    signal_text = f"⏳ **Waiting** - Need {dip_threshold*100:.1f}%+ drop for buy signal (today: {last_return:.1f}%)"
            else:
                regime_text = "🔴 **Bearish Regime** - Waiting for golden cross"
                signal_text = "⏳ **No Action** - Strategy inactive during bearish regime"
            
            st.markdown(regime_text)
            st.markdown(signal_text)
            
            if current_position == 1:
                st.success("📈 **Currently LONG** - Holding position until death cross")
            else:
                st.info("💰 **Currently CASH** - No position held")
            
            # Strategy insights
            st.markdown("### 💡 Strategy Insights")
            
            insights = []
            
            if performance.get('Win Rate (%)', 0) > 60:
                insights.append("✅ **High Win Rate**: Strategy shows good signal quality")
            elif performance.get('Win Rate (%)', 0) < 40:
                insights.append("⚠️ **Low Win Rate**: Consider adjusting parameters")
            
            if performance.get('Sharpe Ratio', 0) > 1:
                insights.append("✅ **Good Risk-Adjusted Returns**: Sharpe ratio indicates efficient strategy")
            
            if performance.get('Max Drawdown (%)', 0) < -20:
                insights.append("⚠️ **High Drawdown**: Strategy experienced significant losses")
            
            if outperformance > 5:
                insights.append("🚀 **Strong Outperformance**: Strategy significantly beats buy & hold")
            elif outperformance < -5:
                insights.append("📉 **Underperformance**: Buy & hold would have been better")
            
            if len(insights) == 0:
                insights.append("📊 **Balanced Performance**: Strategy shows mixed results")
            
            for insight in insights:
                st.markdown(f"- {insight}")
            
            # Strategy rules summary
            st.markdown("### 📖 Strategy Rules")
            st.markdown(strategy.generate_strategy_summary())
            
        except Exception as e:
            st.error(f"Error running strategy: {str(e)}")
            st.info("Please check the symbol and try again.")

if __name__ == "__main__":
    main()