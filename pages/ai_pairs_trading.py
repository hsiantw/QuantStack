import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.pairs_trading_optimizer import PairsTradingOptimizer
from utils.ai_strategy_optimizer import AIStrategyOptimizer
from utils.mean_reversion_strategy import MeanReversionStrategy
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def main():
    # Modern header with enhanced styling
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI-Powered Pairs Trading Analysis</h1>
        <p>Find the best pairs to trade using AI-optimized strategies for statistical arbitrage</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Find Best Pairs", "📊 Detailed Analysis", "⚡ Live Trading Signals", "📈 Mean Reversion Analysis"])
    
    with tab1:
        find_best_pairs_tab()
    
    with tab2:
        detailed_analysis_tab()
    
    with tab3:
        live_signals_tab()
    
    with tab4:
        mean_reversion_analysis_tab()

def find_best_pairs_tab():
    st.header("🔍 AI Pairs Discovery")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        base_ticker = st.text_input("Base Ticker for Pairs Analysis", value="SPY").upper()
        
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

def mean_reversion_analysis_tab():
    st.header("📈 Extensive Mean Reversion Strategy Analysis")
    st.markdown("**Comprehensive mean reversion techniques with advanced signal generation and backtesting**")
    
    # Input section
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        ticker_input = st.text_input("Ticker for Mean Reversion Analysis", value="SPY")
    
    with col2:
        analysis_period = st.selectbox("Analysis Period", 
                                     options=[252, 504, 756, 1008], 
                                     index=2,
                                     format_func=lambda x: f"{x} days (~{x//252} year{'s' if x//252 > 1 else ''})")
    
    with col3:
        strategy_type = st.selectbox("Strategy Type", 
                                   options=["Ensemble", "Bollinger Bands", "RSI", "Ornstein-Uhlenbeck", "Kalman Filter", "Adaptive"])
    
    if st.button("🚀 Run Mean Reversion Analysis", type="primary"):
        
        with st.spinner(f"🔄 Running comprehensive mean reversion analysis for {ticker_input}..."):
            
            # Fetch data
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=analysis_period + 100)  # Extra buffer
                
                data = yf.download(ticker_input, start=start_date, end=end_date, progress=False)
                
                if data.empty or len(data) < 100:
                    st.error("Insufficient data for analysis. Please try a different ticker or period.")
                    return
                
                # Handle MultiIndex columns if present
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.droplevel(1)
                
                price_series = data['Close'].dropna()
                
                if len(price_series) < 50:
                    st.error("Insufficient price data for analysis. Please try a different ticker.")
                    return
                
                # Initialize mean reversion strategy
                mr_strategy = MeanReversionStrategy(data)
                
                # Run selected strategy
                if strategy_type == "Ensemble":
                    results = mr_strategy.ensemble_mean_reversion(price_series)
                    signals = results['ensemble_signals']
                elif strategy_type == "Bollinger Bands":
                    results = mr_strategy.bollinger_bands_reversion(price_series)
                    signals = results['signals']
                elif strategy_type == "RSI":
                    results = mr_strategy.rsi_reversion(price_series)
                    signals = results['signals']
                elif strategy_type == "Ornstein-Uhlenbeck":
                    results = mr_strategy.ornstein_uhlenbeck_reversion(price_series)
                    signals = results['signals']
                elif strategy_type == "Kalman Filter":
                    results = mr_strategy.kalman_filter_reversion(price_series)
                    signals = results['signals']
                elif strategy_type == "Adaptive":
                    results = mr_strategy.adaptive_mean_reversion(price_series)
                    signals = results['adaptive_signals']
                
                # Backtest the strategy
                backtest_results = mr_strategy.backtest_strategy(price_series, signals)
                
                # Display results
                st.success(f"✅ Analysis complete for {ticker_input} using {strategy_type} strategy!")
                
                # Performance metrics
                st.subheader("📊 Strategy Performance Metrics")
                
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
                    calmar_ratio = backtest_results['annual_return'] / abs(backtest_results['max_drawdown']) if backtest_results['max_drawdown'] != 0 else 0
                    st.metric("Calmar Ratio", f"{calmar_ratio:.3f}")
                    st.metric("Profit Factor", f"{backtest_results['profit_factor']:.2f}")
                
                # Create comprehensive visualization
                fig = create_mean_reversion_chart(price_series, signals, results, backtest_results, strategy_type)
                st.plotly_chart(fig, use_container_width=True)
                
                # Strategy-specific details
                st.subheader("🔍 Strategy-Specific Analysis")
                
                if strategy_type == "Ensemble":
                    display_ensemble_details(results)
                elif strategy_type == "Bollinger Bands":
                    display_bollinger_details(results, price_series)
                elif strategy_type == "RSI":
                    display_rsi_details(results)
                elif strategy_type == "Ornstein-Uhlenbeck":
                    display_ou_details(results)
                elif strategy_type == "Kalman Filter":
                    display_kalman_details(results)
                elif strategy_type == "Adaptive":
                    display_adaptive_details(results)
                
                # Trade analysis
                if backtest_results['trades']:
                    st.subheader("📈 Trade Analysis")
                    
                    trades_df = pd.DataFrame(backtest_results['trades'])
                    
                    # Trade statistics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Trade Distribution:**")
                        winning_trades = len([t for t in backtest_results['trades'] if t['return'] > 0])
                        losing_trades = len([t for t in backtest_results['trades'] if t['return'] <= 0])
                        
                        st.write(f"• Winning trades: {winning_trades}")
                        st.write(f"• Losing trades: {losing_trades}")
                        st.write(f"• Average trade duration: {trades_df['duration'].mean():.1f} days")
                        st.write(f"• Best trade: {trades_df['return'].max():.2%}")
                        st.write(f"• Worst trade: {trades_df['return'].min():.2%}")
                    
                    with col2:
                        st.markdown("**Risk Metrics:**")
                        st.write(f"• Average win: {backtest_results['avg_win']:.2%}")
                        st.write(f"• Average loss: {backtest_results['avg_loss']:.2%}")
                        st.write(f"• Profit factor: {backtest_results['profit_factor']:.2f}")
                        st.write(f"• Max consecutive losses: {calculate_max_consecutive_losses(trades_df)}")
                        st.write(f"• Sharpe ratio: {backtest_results['sharpe_ratio']:.3f}")
                    
                    # Recent trades
                    st.markdown("**Recent Trades:**")
                    recent_trades = trades_df.tail(10)[['entry_date', 'exit_date', 'return', 'duration']].copy()
                    recent_trades['return'] = recent_trades['return'].apply(lambda x: f"{x:.2%}")
                    recent_trades['duration'] = recent_trades['duration'].apply(lambda x: f"{x} days")
                    st.dataframe(recent_trades, use_container_width=True, hide_index=True)
                
                # Current signal
                st.subheader("⚡ Current Signal")
                
                current_signal = signals.iloc[-1] if len(signals) > 0 else 0
                
                if current_signal > 0:
                    st.success(f"🟢 **BUY SIGNAL** - Mean reversion strategy suggests going LONG {ticker_input}")
                elif current_signal < 0:
                    st.error(f"🔴 **SELL SIGNAL** - Mean reversion strategy suggests going SHORT {ticker_input}")
                else:
                    st.info(f"🟡 **NEUTRAL** - No clear signal for {ticker_input} at current levels")
                
                # Implementation guide
                with st.expander("📋 Implementation Guide", expanded=False):
                    st.markdown(f"""
                    **How to Trade This Signal:**
                    
                    **Entry Rules:**
                    - Wait for signal confirmation (current signal: {current_signal})
                    - Use limit orders near current price levels
                    - Consider market conditions and volatility
                    
                    **Position Sizing:**
                    - Risk 1-2% of portfolio per trade
                    - Use stops based on strategy parameters
                    - Scale position based on signal strength
                    
                    **Exit Strategy:**
                    - Follow mean reversion signals for exits
                    - Use time-based stops if no reversal occurs
                    - Take profits when price reaches opposite signal
                    
                    **Risk Management:**
                    - Maximum drawdown tolerance: 10-15%
                    - Don't trade during high volatility periods
                    - Monitor correlation with other positions
                    """)
                
            except Exception as e:
                st.error(f"Error running analysis: {str(e)}")

def create_mean_reversion_chart(price_series, signals, strategy_results, backtest_results, strategy_type):
    """Create comprehensive mean reversion visualization"""
    
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=[
            f'{strategy_type} Mean Reversion Analysis',
            'Trading Signals',
            'Strategy Performance',
            'Drawdown Analysis'
        ],
        vertical_spacing=0.05,
        row_heights=[0.4, 0.2, 0.2, 0.2]
    )
    
    # Plot 1: Price and strategy indicators
    fig.add_trace(go.Scatter(
        x=price_series.index, y=price_series.values,
        name='Price',
        line=dict(color='blue', width=1)
    ), row=1, col=1)
    
    # Add strategy-specific indicators
    if strategy_type == "Bollinger Bands" and 'upper_band' in strategy_results:
        fig.add_trace(go.Scatter(
            x=price_series.index, y=strategy_results['upper_band'],
            name='Upper Band', line=dict(color='red', dash='dash')
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=price_series.index, y=strategy_results['lower_band'],
            name='Lower Band', line=dict(color='green', dash='dash')
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=price_series.index, y=strategy_results['middle_band'],
            name='Middle Band', line=dict(color='orange', dash='dot')
        ), row=1, col=1)
    
    elif strategy_type == "Kalman Filter" and 'kalman_mean' in strategy_results:
        fig.add_trace(go.Scatter(
            x=price_series.index, y=strategy_results['kalman_mean'],
            name='Kalman Mean', line=dict(color='purple', width=2)
        ), row=1, col=1)
    
    # Plot 2: Signals
    buy_signals = signals[signals == 1]
    sell_signals = signals[signals == -1]
    
    if len(buy_signals) > 0:
        fig.add_trace(go.Scatter(
            x=buy_signals.index, y=[1] * len(buy_signals),
            mode='markers', name='Buy Signals',
            marker=dict(color='green', size=10, symbol='triangle-up')
        ), row=2, col=1)
    
    if len(sell_signals) > 0:
        fig.add_trace(go.Scatter(
            x=sell_signals.index, y=[-1] * len(sell_signals),
            mode='markers', name='Sell Signals',
            marker=dict(color='red', size=10, symbol='triangle-down')
        ), row=2, col=1)
    
    # Plot 3: Equity curve
    if 'equity_curve' in backtest_results:
        fig.add_trace(go.Scatter(
            x=backtest_results['equity_curve'].index,
            y=backtest_results['equity_curve'].values,
            name='Strategy Equity',
            line=dict(color='purple', width=2)
        ), row=3, col=1)
    
    # Plot 4: Drawdown
    if 'drawdown_series' in backtest_results:
        fig.add_trace(go.Scatter(
            x=backtest_results['drawdown_series'].index,
            y=backtest_results['drawdown_series'].values * 100,
            name='Drawdown %',
            fill='tonexty',
            line=dict(color='red', width=1)
        ), row=4, col=1)
    
    fig.update_layout(height=1000, showlegend=True, hovermode='x unified')
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Signal", row=2, col=1, range=[-1.5, 1.5])
    fig.update_yaxes(title_text="Portfolio Value", row=3, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=4, col=1)
    
    return fig

def display_ensemble_details(results):
    """Display ensemble strategy details"""
    st.markdown("### 🎯 Ensemble Strategy Components")
    
    component_signals = results['component_signals']
    signal_strength = results['signal_strength']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Current Component Signals:**")
        for strategy, signals in component_signals.items():
            current_signal = signals.iloc[-1] if len(signals) > 0 else 0
            signal_text = "BUY" if current_signal > 0 else ("SELL" if current_signal < 0 else "NEUTRAL")
            st.write(f"• {strategy.title()}: {signal_text}")
    
    with col2:
        st.markdown("**Signal Strength Distribution:**")
        avg_strength = signal_strength.mean()
        max_strength = signal_strength.max()
        current_strength = signal_strength.iloc[-1] if len(signal_strength) > 0 else 0
        
        st.write(f"• Average strength: {avg_strength:.2f}")
        st.write(f"• Maximum strength: {max_strength:.2f}")
        st.write(f"• Current strength: {current_strength:.2f}")

def display_bollinger_details(results, price_series):
    """Display Bollinger Bands strategy details"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Bollinger Bands Parameters:**")
        bandwidth = results['bandwidth']
        current_bandwidth = bandwidth.iloc[-1] if len(bandwidth) > 0 else 0
        avg_bandwidth = bandwidth.mean()
        
        st.write(f"• Current bandwidth: {current_bandwidth:.3f}")
        st.write(f"• Average bandwidth: {avg_bandwidth:.3f}")
        st.write(f"• Volatility regime: {'High' if current_bandwidth > avg_bandwidth * 1.2 else 'Low' if current_bandwidth < avg_bandwidth * 0.8 else 'Normal'}")
    
    with col2:
        st.markdown("**Price Position:**")
        current_price = price_series.iloc[-1]
        upper_band = results['upper_band'].iloc[-1] if len(results['upper_band']) > 0 else current_price
        lower_band = results['lower_band'].iloc[-1] if len(results['lower_band']) > 0 else current_price
        middle_band = results['middle_band'].iloc[-1] if len(results['middle_band']) > 0 else current_price
        
        position_pct = (current_price - lower_band) / (upper_band - lower_band) * 100
        st.write(f"• Band position: {position_pct:.1f}%")
        st.write(f"• Distance from mean: {((current_price - middle_band) / middle_band * 100):.2f}%")

def display_rsi_details(results):
    """Display RSI strategy details"""
    rsi = results['rsi']
    current_rsi = rsi.iloc[-1] if len(rsi) > 0 else 50
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**RSI Analysis:**")
        st.write(f"• Current RSI: {current_rsi:.1f}")
        st.write(f"• Condition: {'Oversold' if current_rsi < 30 else 'Overbought' if current_rsi > 70 else 'Neutral'}")
    
    with col2:
        st.markdown("**RSI Statistics:**")
        avg_rsi = rsi.mean()
        rsi_volatility = rsi.std()
        st.write(f"• Average RSI: {avg_rsi:.1f}")
        st.write(f"• RSI volatility: {rsi_volatility:.1f}")

def display_ou_details(results):
    """Display Ornstein-Uhlenbeck strategy details"""
    if 'half_lives' in results and len(results['half_lives']) > 0:
        half_lives = results['half_lives'].replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(half_lives) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Mean Reversion Properties:**")
                current_half_life = half_lives.iloc[-1] if len(half_lives) > 0 else np.inf
                avg_half_life = half_lives.mean()
                
                st.write(f"• Current half-life: {current_half_life:.1f} days")
                st.write(f"• Average half-life: {avg_half_life:.1f} days")
            
            with col2:
                st.markdown("**Reversion Strength:**")
                st.write(f"• Mean reversion: {'Strong' if avg_half_life < 20 else 'Moderate' if avg_half_life < 60 else 'Weak'}")
                st.write(f"• Process stability: {'Stable' if half_lives.std() < avg_half_life * 0.5 else 'Variable'}")

def display_kalman_details(results):
    """Display Kalman Filter strategy details"""
    if 'deviation' in results:
        deviation = results['deviation']
        confidence_bands = results['confidence_bands']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Kalman Filter Analysis:**")
            current_deviation = deviation.iloc[-1] if len(deviation) > 0 else 0
            avg_deviation = abs(deviation).mean()
            
            st.write(f"• Current deviation: {current_deviation:.3f}")
            st.write(f"• Average deviation: {avg_deviation:.3f}")
        
        with col2:
            st.markdown("**Filter Performance:**")
            current_confidence = confidence_bands.iloc[-1] if len(confidence_bands) > 0 else 0
            avg_confidence = confidence_bands.mean()
            
            st.write(f"• Current confidence: {current_confidence:.3f}")
            st.write(f"• Average confidence: {avg_confidence:.3f}")

def display_adaptive_details(results):
    """Display Adaptive strategy details"""
    if 'regime_indicators' in results:
        regime_indicators = results['regime_indicators']
        
        # Current regime
        current_low_vol = regime_indicators['low_vol'].iloc[-1] if len(regime_indicators['low_vol']) > 0 else False
        current_high_vol = regime_indicators['high_vol'].iloc[-1] if len(regime_indicators['high_vol']) > 0 else False
        current_normal = regime_indicators['normal'].iloc[-1] if len(regime_indicators['normal']) > 0 else True
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Current Market Regime:**")
            if current_low_vol:
                regime = "Low Volatility"
                description = "Aggressive mean reversion parameters"
            elif current_high_vol:
                regime = "High Volatility"
                description = "Conservative mean reversion parameters"
            else:
                regime = "Normal Volatility"
                description = "Standard mean reversion parameters"
            
            st.write(f"• Regime: {regime}")
            st.write(f"• Strategy: {description}")
        
        with col2:
            st.markdown("**Regime Distribution:**")
            total_periods = len(regime_indicators['low_vol'])
            low_vol_pct = regime_indicators['low_vol'].sum() / total_periods * 100
            high_vol_pct = regime_indicators['high_vol'].sum() / total_periods * 100
            normal_pct = regime_indicators['normal'].sum() / total_periods * 100
            
            st.write(f"• Low vol periods: {low_vol_pct:.1f}%")
            st.write(f"• High vol periods: {high_vol_pct:.1f}%")
            st.write(f"• Normal periods: {normal_pct:.1f}%")

def calculate_max_consecutive_losses(trades_df):
    """Calculate maximum consecutive losing trades"""
    if trades_df.empty:
        return 0
    
    losing_streaks = []
    current_streak = 0
    
    for _, trade in trades_df.iterrows():
        if trade['return'] <= 0:
            current_streak += 1
        else:
            if current_streak > 0:
                losing_streaks.append(current_streak)
            current_streak = 0
    
    if current_streak > 0:
        losing_streaks.append(current_streak)
    
    return max(losing_streaks) if losing_streaks else 0

if __name__ == "__main__":
    main()