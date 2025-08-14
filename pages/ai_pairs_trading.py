import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.pairs_trading_optimizer import PairsTradingOptimizer
from utils.ai_strategy_optimizer import AIStrategyOptimizer
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def main():
    st.title("🤖 AI-Powered Pairs Trading Analysis")
    st.markdown("**Find the best pairs to trade using AI-optimized strategies for statistical arbitrage**")
    
    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["🔍 Find Best Pairs", "📊 Detailed Analysis", "⚡ Live Trading Signals"])
    
    with tab1:
        find_best_pairs_tab()
    
    with tab2:
        detailed_analysis_tab()
    
    with tab3:
        live_signals_tab()

def find_best_pairs_tab():
    st.header("🔍 AI Pairs Discovery")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        base_ticker = st.text_input("Base Ticker for Pairs Analysis", value="SPY")
        
    with col2:
        lookback_days = st.selectbox("Analysis Period", 
                                   options=[252, 504, 756, 1008], 
                                   index=1,
                                   format_func=lambda x: f"{x} days (~{x//252} year{'s' if x//252 > 1 else ''})")
    
    with col3:
        run_analysis = st.button("🚀 Find Best Pairs", type="primary")
    
    # Custom candidate tickers option
    with st.expander("🎯 Custom Candidate Tickers (Optional)"):
        custom_candidates = st.text_input(
            "Enter custom tickers separated by commas (e.g., AAPL,MSFT,GOOGL)",
            placeholder="Leave empty to use sector-based analysis"
        )
        if custom_candidates:
            candidate_list = [ticker.strip().upper() for ticker in custom_candidates.split(",")]
        else:
            candidate_list = None
    
    if run_analysis and base_ticker:
        with st.spinner(f"🔍 Analyzing pairs for {base_ticker}..."):
            
            # Initialize pairs optimizer
            pairs_optimizer = PairsTradingOptimizer(base_ticker)
            
            # Find optimal pairs
            results = pairs_optimizer.find_optimal_pairs(
                candidate_tickers=candidate_list,
                lookback_days=lookback_days
            )
            
            if results and results['pairs']:
                st.success(f"✅ Found {len(results['pairs'])} potential trading pairs!")
                
                # Display top pairs
                st.subheader("🏆 Top Statistical Arbitrage Opportunities")
                
                pairs_data = []
                for i, pair in enumerate(results['pairs'][:5]):
                    pairs_data.append({
                        "Rank": i + 1,
                        "Pair": pair['pair'],
                        "Trading Score": f"{pair['trading_score']:.0f}/100",
                        "Cointegrated": "✅ Yes" if pair['is_cointegrated'] else "❌ No",
                        "P-Value": f"{pair['cointegration_pvalue']:.4f}",
                        "Current Signal": pair['signal_strength'],
                        "Z-Score": f"{pair['current_zscore']:.2f}",
                        "Half-Life": f"{pair['half_life']:.1f} days" if pair['half_life'] != np.inf else "∞",
                        "Correlation": f"{pair['correlation']:.3f}"
                    })
                
                pairs_df = pd.DataFrame(pairs_data)
                st.dataframe(pairs_df, use_container_width=True, hide_index=True)
                
                # Store results in session state for detailed analysis
                st.session_state['pairs_results'] = results
                st.session_state['current_base_ticker'] = base_ticker
                
                # Show analysis methodology
                with st.expander("🧠 How AI Found These Pairs", expanded=False):
                    st.markdown("### AI Methodology for Pairs Discovery:")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("""
                        **1. Cointegration Analysis**
                        - Tests statistical relationship between price series
                        - Engle-Granger cointegration test (p < 0.05)
                        - Ensures long-term equilibrium relationship
                        
                        **2. Stationarity Testing**
                        - Augmented Dickey-Fuller test on price spread
                        - Confirms mean-reverting properties
                        - Critical for pairs trading success
                        
                        **3. Half-Life Calculation**
                        - Measures speed of mean reversion
                        - Optimal range: 5-60 days
                        - Formula: -ln(2) / β from AR(1) model
                        """)
                    
                    with col2:
                        st.markdown("""
                        **4. Trading Score Algorithm**
                        - Cointegration strength: 30 points max
                        - Spread stationarity: 25 points max
                        - Volatility optimization: 20 points max
                        - Mean reversion speed: 15 points max
                        - Current signal strength: 10 points max
                        
                        **5. Hedge Ratio Calculation**
                        - Linear regression: Stock1 = α + β×Stock2
                        - β coefficient becomes hedge ratio
                        - Minimizes spread variance
                        """)
                    
                    st.markdown("**Current Analysis Summary:**")
                    st.info(f"• Tested {results['total_pairs_tested']} potential pairs\n"
                           f"• Found {len([p for p in results['pairs'] if p['is_cointegrated']])} cointegrated pairs\n"
                           f"• Analysis period: {lookback_days} days\n"
                           f"• Base ticker: {base_ticker}")
                
                # Quick signal overview
                st.subheader("⚡ Current Trading Signals")
                
                signal_pairs = [pair for pair in results['pairs'][:3] if abs(pair['current_zscore']) > 1.5]
                
                if signal_pairs:
                    for pair in signal_pairs:
                        zscore = pair['current_zscore']
                        signal_type = "🔴 SELL SPREAD" if zscore > 0 else "🟢 BUY SPREAD"
                        signal_strength = pair['signal_strength']
                        
                        st.markdown(f"**{pair['pair']}**: {signal_type} | Signal: {signal_strength} | Z-Score: {zscore:.2f}")
                else:
                    st.info("No strong trading signals at the moment. Check back later or analyze specific pairs in the Detailed Analysis tab.")
            
            else:
                st.warning("No suitable pairs found. Try a different base ticker or expand the candidate universe.")

def detailed_analysis_tab():
    st.header("📊 Detailed Pairs Analysis")
    
    # Check if we have results from the discovery tab
    if 'pairs_results' not in st.session_state:
        st.info("👈 First run the pairs discovery in the 'Find Best Pairs' tab to see detailed analysis here.")
        return
    
    results = st.session_state['pairs_results']
    base_ticker = st.session_state['current_base_ticker']
    
    # Pair selection
    pair_options = [pair['pair'] for pair in results['pairs']]
    selected_pair = st.selectbox("Select Pair for Detailed Analysis", pair_options)
    
    # Find selected pair data
    pair_data = next((pair for pair in results['pairs'] if pair['pair'] == selected_pair), None)
    
    if pair_data:
        st.subheader(f"📈 Comprehensive Analysis: {selected_pair}")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Trading Score", f"{pair_data['trading_score']:.0f}/100")
            st.metric("Correlation", f"{pair_data['correlation']:.3f}")
        
        with col2:
            st.metric("Cointegration P-Value", f"{pair_data['cointegration_pvalue']:.4f}")
            st.metric("Hedge Ratio", f"{pair_data['hedge_ratio']:.4f}")
        
        with col3:
            st.metric("Current Z-Score", f"{pair_data['current_zscore']:.2f}")
            st.metric("Half-Life (days)", f"{pair_data['half_life']:.1f}" if pair_data['half_life'] != np.inf else "∞")
        
        with col4:
            cointegrated = "✅ Yes" if pair_data['is_cointegrated'] else "❌ No"
            st.metric("Cointegrated", cointegrated)
            st.metric("Signal Strength", pair_data['signal_strength'])
        
        # Apply AI strategy optimization to the pair
        with st.spinner("🤖 Applying AI strategy optimization to pair..."):
            
            # Get the pair's constituent tickers
            ticker1, ticker2 = pair_data['ticker1'], pair_data['ticker2']
            
            # Fetch data for AI optimization
            end_date = datetime.now()
            start_date = end_date - timedelta(days=756)  # 3 years for strategy optimization
            
            try:
                data1 = yf.download(ticker1, start=start_date, end=end_date, progress=False)
                data2 = yf.download(ticker2, start=start_date, end=end_date, progress=False)
                
                if len(data1) > 100 and len(data2) > 100 and not data1.empty and not data2.empty:
                    # Apply AI optimization to both constituent stocks
                    st.subheader("🤖 AI Strategy Optimization for Pair Components")
                    
                    tab_a, tab_b = st.tabs([f"📊 {ticker1} Analysis", f"📊 {ticker2} Analysis"])
                    
                    with tab_a:
                        ai_optimizer1 = AIStrategyOptimizer(data1, ticker1)
                        recommendations1 = ai_optimizer1.optimize_strategy()
                        
                        if recommendations1:
                            display_ai_strategy_results(recommendations1, ticker1)
                    
                    with tab_b:
                        ai_optimizer2 = AIStrategyOptimizer(data2, ticker2)
                        recommendations2 = ai_optimizer2.optimize_strategy()
                        
                        if recommendations2:
                            display_ai_strategy_results(recommendations2, ticker2)
                    
                    # Spread analysis and strategy
                    st.subheader("📈 Spread Analysis & Trading Strategy")
                    
                    # Generate pairs trading strategy
                    pairs_optimizer = PairsTradingOptimizer(base_ticker)
                    strategy_results = pairs_optimizer.generate_pairs_strategy(pair_data)
                    
                    if strategy_results:
                        # Backtest the strategy
                        backtest_results = pairs_optimizer.backtest_pairs_strategy(pair_data, strategy_results)
                        
                        if backtest_results:
                            # Display backtest results
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Annual Return", f"{backtest_results['annual_return']:.2%}")
                                st.metric("Total Trades", backtest_results['total_trades'])
                            
                            with col2:
                                st.metric("Sharpe Ratio", f"{backtest_results['sharpe_ratio']:.3f}")
                                st.metric("Win Rate", f"{backtest_results['win_rate']:.1f}%")
                            
                            with col3:
                                st.metric("Max Drawdown", f"{backtest_results['max_drawdown']:.2%}")
                                st.metric("Volatility", f"{backtest_results['volatility']:.2%}")
                            
                            with col4:
                                calmar = backtest_results['annual_return'] / abs(backtest_results['max_drawdown']) if backtest_results['max_drawdown'] != 0 else 0
                                st.metric("Calmar Ratio", f"{calmar:.3f}")
                            
                            # Plot spread and signals
                            fig = create_pairs_trading_chart(pair_data, strategy_results, backtest_results)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Strategy explanation
                            with st.expander("🎯 Pairs Trading Strategy Details", expanded=True):
                                st.markdown(f"""
                                **Strategy Parameters:**
                                - Entry Threshold: ±{strategy_results['strategy_params']['entry_threshold']} standard deviations
                                - Exit Threshold: ±{strategy_results['strategy_params']['exit_threshold']} standard deviations  
                                - Stop Loss: ±{strategy_results['strategy_params']['stop_loss']} standard deviations
                                - Lookback Window: {strategy_results['strategy_params']['lookback_window']} days
                                
                                **Trading Logic:**
                                - **Buy Spread** when Z-Score < -2.0: Long {ticker1}, Short {ticker2}
                                - **Sell Spread** when Z-Score > +2.0: Short {ticker1}, Long {ticker2}
                                - **Exit Position** when Z-Score approaches 0 (±0.5)
                                - **Stop Loss** triggered at ±3.0 standard deviations
                                
                                **Position Sizing:**
                                - Hedge Ratio: {pair_data['hedge_ratio']:.4f}
                                - For every $1 in {ticker1}, trade ${abs(pair_data['hedge_ratio']):.4f} in {ticker2}
                                """)
                
            except Exception as e:
                st.error(f"Error fetching data for detailed analysis: {str(e)}")

def display_ai_strategy_results(recommendations, ticker):
    """Helper function to display AI strategy results"""
    best_strategy = recommendations['best_strategy']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Best Strategy", best_strategy['name'])
        st.metric("Annual Return", f"{best_strategy['annual_return']:.2%}")
    
    with col2:
        st.metric("Sharpe Ratio", f"{best_strategy['sharpe_ratio']:.3f}")
        st.metric("Max Drawdown", f"{best_strategy['max_drawdown']:.2%}")
    
    with col3:
        st.metric("Win Rate", f"{best_strategy['win_rate']:.1f}%")
        st.metric("Total Trades", best_strategy['total_trades'])

def create_pairs_trading_chart(pair_data, strategy_results, backtest_results):
    """Create comprehensive pairs trading visualization"""
    
    spread = strategy_results['spread']
    zscore = strategy_results['zscore']
    signals = strategy_results['signals']
    equity_curve = backtest_results['equity_curve']
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=['Price Spread & Z-Score', 'Trading Signals', 'Strategy Equity Curve'],
        vertical_spacing=0.08,
        row_heights=[0.4, 0.3, 0.3]
    )
    
    # Plot 1: Spread and Z-Score
    fig.add_trace(go.Scatter(
        x=spread.index, y=spread.values,
        name='Price Spread',
        line=dict(color='blue', width=1),
        yaxis='y'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=zscore.index, y=zscore.values,
        name='Z-Score',
        line=dict(color='orange', width=2),
        yaxis='y2'
    ), row=1, col=1)
    
    # Add threshold lines
    fig.add_hline(y=2.0, line_dash="dash", line_color="red", opacity=0.7, 
                  annotation_text="Entry (+2σ)", row=1, col=1)
    fig.add_hline(y=-2.0, line_dash="dash", line_color="green", opacity=0.7,
                  annotation_text="Entry (-2σ)", row=1, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5, row=1, col=1)
    
    # Plot 2: Trading Signals
    buy_signals = signals[signals == 1]
    sell_signals = signals[signals == -1]
    
    fig.add_trace(go.Scatter(
        x=buy_signals.index, y=[1] * len(buy_signals),
        mode='markers',
        name='Buy Spread',
        marker=dict(color='green', size=8, symbol='triangle-up')
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=sell_signals.index, y=[-1] * len(sell_signals),
        mode='markers',
        name='Sell Spread',
        marker=dict(color='red', size=8, symbol='triangle-down')
    ), row=2, col=1)
    
    # Plot 3: Equity Curve
    fig.add_trace(go.Scatter(
        x=equity_curve.index, y=equity_curve.values,
        name='Strategy Returns',
        line=dict(color='purple', width=2),
        fill='tonexty'
    ), row=3, col=1)
    
    fig.add_hline(y=1.0, line_dash="dot", line_color="gray", opacity=0.5, row=3, col=1)
    
    # Update layout
    fig.update_layout(
        title=f"Pairs Trading Analysis: {pair_data['pair']}",
        height=800,
        showlegend=True,
        hovermode='x unified'
    )
    
    # Update y-axes
    fig.update_yaxes(title_text="Price Spread", row=1, col=1)
    fig.update_yaxes(title_text="Z-Score", secondary_y=True, row=1, col=1)
    fig.update_yaxes(title_text="Position", row=2, col=1, range=[-1.5, 1.5])
    fig.update_yaxes(title_text="Portfolio Value", row=3, col=1)
    
    return fig

def live_signals_tab():
    st.header("⚡ Live Trading Signals & Execution Guide")
    
    # Check if we have results
    if 'pairs_results' not in st.session_state:
        st.info("👈 First run the pairs discovery to see live signals here.")
        return
    
    results = st.session_state['pairs_results']
    
    # Current signals with detailed execution instructions
    st.subheader("🚨 Current Trading Opportunities")
    
    strong_signals = [pair for pair in results['pairs'] if abs(pair['current_zscore']) > 1.5]
    
    if strong_signals:
        for i, pair in enumerate(strong_signals[:5]):
            zscore = pair['current_zscore']
            hedge_ratio = abs(pair['hedge_ratio'])
            
            # Determine signal direction and strength
            if zscore > 2.0:
                signal_type = "🔴 STRONG SELL SPREAD"
                confidence = "Very High"
                direction = "SELL"
                primary_action = f"SHORT {pair['ticker1']}"
                secondary_action = f"LONG {pair['ticker2']}"
                expected_move = "Spread expected to converge (decrease)"
            elif zscore > 1.5:
                signal_type = "🟡 SELL SPREAD"
                confidence = "High"
                direction = "SELL"
                primary_action = f"SHORT {pair['ticker1']}"
                secondary_action = f"LONG {pair['ticker2']}"
                expected_move = "Spread expected to converge (decrease)"
            elif zscore < -2.0:
                signal_type = "🟢 STRONG BUY SPREAD"
                confidence = "Very High"
                direction = "BUY"
                primary_action = f"LONG {pair['ticker1']}"
                secondary_action = f"SHORT {pair['ticker2']}"
                expected_move = "Spread expected to converge (increase)"
            elif zscore < -1.5:
                signal_type = "🟡 BUY SPREAD"
                confidence = "High"
                direction = "BUY"
                primary_action = f"LONG {pair['ticker1']}"
                secondary_action = f"SHORT {pair['ticker2']}"
                expected_move = "Spread expected to converge (increase)"
            else:
                continue
            
            with st.expander(f"Signal #{i+1}: {pair['pair']} - {signal_type}", expanded=True):
                
                # Signal overview
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Current Z-Score", f"{zscore:.2f}")
                    st.metric("Signal Confidence", confidence)
                
                with col2:
                    st.metric("Trading Score", f"{pair['trading_score']:.0f}/100")
                    st.metric("Half-life", f"{pair['half_life']:.1f} days")
                
                with col3:
                    st.metric("Hedge Ratio", f"{hedge_ratio:.4f}")
                    st.metric("Correlation", f"{pair['correlation']:.3f}")
                
                with col4:
                    cointegrated = "✅ Yes" if pair['is_cointegrated'] else "❌ No"
                    st.metric("Cointegrated", cointegrated)
                    st.metric("P-Value", f"{pair['cointegration_pvalue']:.4f}")
                
                st.markdown("---")
                
                # Detailed execution instructions
                st.markdown("### 📋 Detailed Execution Instructions")
                
                tab1, tab2, tab3 = st.tabs(["🎯 Trade Setup", "💰 Position Sizing", "⚡ Risk Management"])
                
                with tab1:
                    st.markdown(f"""
                    **SIGNAL DIRECTION: {direction} SPREAD**
                    
                    **Step 1: Primary Position**
                    - {primary_action}
                    - Entry reason: {expected_move}
                    
                    **Step 2: Hedge Position** 
                    - {secondary_action}
                    - Hedge ratio: {hedge_ratio:.4f}
                    
                    **Expected Outcome:**
                    - {expected_move}
                    - Target Z-score: 0 (mean reversion)
                    - Time horizon: {pair['half_life']:.1f} days average
                    """)
                
                with tab2:
                    st.markdown(f"""
                    **Position Sizing Formula:**
                    
                    If trading $10,000 total:
                    - **{pair['ticker1']} position:** $5,000 ({primary_action.split()[0]})
                    - **{pair['ticker2']} position:** ${5000 * hedge_ratio:,.0f} ({secondary_action.split()[0]})
                    
                    **Alternative: Share-based calculation**
                    - For every 100 shares of {pair['ticker1']}: {int(100 * hedge_ratio)} shares of {pair['ticker2']}
                    - Hedge ratio ensures dollar-neutral position
                    
                    **Risk Allocation:**
                    - Maximum position size: 2-5% of portfolio
                    - Consider correlation with existing positions
                    """)
                
                with tab3:
                    st.markdown(f"""
                    **Entry Rules:**
                    - Current Z-score: {zscore:.2f} ({"Above" if abs(zscore) > 2 else "Near"} entry threshold of ±2.0)
                    - Wait for Z-score > 2.0 for SELL or < -2.0 for BUY signals
                    
                    **Exit Rules:**
                    - **Target Exit:** Z-score reaches ±0.5 (75% mean reversion)
                    - **Stop Loss:** Z-score reaches ±3.0 (position against us)
                    - **Time Stop:** Close after {pair['half_life']*3:.0f} days if no convergence
                    
                    **Monitoring:**
                    - Check Z-score daily for exit signals
                    - Monitor both individual stock news/events
                    - Watch for breakdown in correlation
                    """)
                
                # Current market context
                st.markdown("### 📊 Current Market Context")
                st.info(f"""
                **Why This Signal Exists:**
                - Spread is currently {abs(zscore):.1f} standard deviations from historical mean
                - Statistical significance: {pair['signal_strength']} confidence
                - Historical mean reversion time: {pair['half_life']:.1f} days
                - Cointegration test confirms long-term relationship (p-value: {pair['cointegration_pvalue']:.4f})
                """)
    
    else:
        st.info("📊 No strong signals currently. Market conditions may be neutral.")
        
        # Show weaker signals for reference
        weaker_signals = [pair for pair in results['pairs'][:5] if 1.0 <= abs(pair['current_zscore']) < 1.5]
        
        if weaker_signals:
            st.subheader("📋 Moderate Signals (Monitor Only)")
            
            for pair in weaker_signals:
                zscore = pair['current_zscore']
                direction = "SELL" if zscore > 0 else "BUY"
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.write(f"**{pair['pair']}**")
                
                with col2:
                    st.write(f"Z-Score: {zscore:.2f}")
                
                with col3:
                    st.write(f"Direction: {direction}")
                
                with col4:
                    st.write(f"Score: {pair['trading_score']:.0f}/100")
    
    # Market overview
    st.subheader("📈 Pairs Market Overview")
    
    # Create summary metrics
    total_pairs = len(results['pairs'])
    cointegrated_pairs = len([p for p in results['pairs'] if p['is_cointegrated']])
    strong_signals_count = len([p for p in results['pairs'] if abs(p['current_zscore']) > 2.0])
    moderate_signals_count = len([p for p in results['pairs'] if 1.5 <= abs(p['current_zscore']) <= 2.0])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Pairs", total_pairs)
    
    with col2:
        st.metric("Cointegrated", cointegrated_pairs)
    
    with col3:
        st.metric("Strong Signals", strong_signals_count)
    
    with col4:
        st.metric("Moderate Signals", moderate_signals_count)
    
    # Trading tips
    with st.expander("💡 General Pairs Trading Tips", expanded=False):
        st.markdown("""
        **Before You Trade:**
        1. **Verify Cointegration:** Only trade pairs with p-value < 0.05
        2. **Check Half-life:** Prefer pairs with 5-60 day mean reversion
        3. **Monitor News:** Avoid trading around earnings or major events
        4. **Position Sizing:** Never risk more than 2-5% of portfolio per pair
        
        **During Trading:**
        1. **Execute Simultaneously:** Enter both legs as close to simultaneously as possible
        2. **Monitor Daily:** Check Z-scores and exit signals daily
        3. **Respect Stops:** Always honor your stop-loss levels
        4. **Time Management:** Don't hold positions indefinitely
        
        **Risk Management:**
        1. **Diversification:** Don't trade multiple pairs from same sector
        2. **Correlation Risk:** Monitor overall portfolio correlation
        3. **Regime Changes:** Be aware that relationships can break down
        4. **Liquidity:** Ensure both stocks have adequate trading volume
        """)
    
    # Refresh button
    if st.button("🔄 Refresh Signals", type="secondary"):
        st.rerun()

if __name__ == "__main__":
    main()