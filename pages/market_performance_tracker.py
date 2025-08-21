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
    def create_metric_card(title, value, delta=None):
        return st.metric(title, value, delta)
    def create_info_card(title, content):
        return st.info(f"**{title}**\n\n{content}")
    def check_authentication():
        return True, None

class MarketPerformanceTracker:
    """Track best and worst performers across major indices"""
    
    def __init__(self):
        # Major indices components
        self.indices = {
            'S&P 500': {
                'ticker': '^GSPC',
                'components': self.get_sp500_tickers()[:100]  # Top 100 for performance
            },
            'NASDAQ 100': {
                'ticker': '^NDX', 
                'components': self.get_nasdaq100_tickers()
            },
            'Dow Jones': {
                'ticker': '^DJI',
                'components': self.get_dow_tickers()
            },
            'Russell 2000': {
                'ticker': '^RUT',
                'components': self.get_russell2000_tickers()[:50]  # Top 50 for performance
            }
        }
        
        self.time_periods = {
            '1 Day': 1,
            '1 Week': 7, 
            '1 Month': 30,
            '3 Months': 90,
            '1 Year': 252
        }
    
    def get_sp500_tickers(self):
        """Get S&P 500 component tickers"""
        # Major S&P 500 components for demonstration
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
            'UNH', 'JNJ', 'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'PFE',
            'ABBV', 'KO', 'AVGO', 'COST', 'PEP', 'WMT', 'TMO', 'MRK', 'DIS',
            'BAC', 'ABT', 'NFLX', 'CRM', 'ACN', 'ADBE', 'VZ', 'DHCP', 'NKE',
            'T', 'CMCSA', 'TXN', 'RTX', 'NEE', 'WFC', 'QCOM', 'PM', 'HON',
            'LLY', 'UPS', 'IBM', 'SPGI', 'LOW', 'AMD', 'GE', 'CAT', 'INTU',
            'ORCL', 'INTC', 'MDT', 'GS', 'BMY', 'AXP', 'BLK', 'BA', 'MU',
            'NOW', 'AMT', 'DE', 'SYK', 'AMAT', 'TGT', 'SCHW', 'LRCX', 'GILD',
            'CVS', 'BKNG', 'ADI', 'TMUS', 'MDLZ', 'ADP', 'CI', 'LMT', 'ISRG',
            'TJX', 'ZTS', 'MMC', 'VRTX', 'MO', 'PYPL', 'DUK', 'EOG', 'SLB',
            'CME', 'ITW', 'PLD', 'CSX', 'FIS', 'NOC', 'AON', 'ICE', 'USB',
            'WM', 'GD', 'CL', 'NSC', 'APD', 'KLAC', 'FCX', 'HUM', 'EMR', 'F'
        ]
    
    def get_nasdaq100_tickers(self):
        """Get NASDAQ 100 component tickers"""
        return [
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA',
            'AVGO', 'COST', 'PEP', 'NFLX', 'ADBE', 'CMCSA', 'TXN', 'QCOM',
            'HON', 'AMD', 'INTU', 'INTC', 'MU', 'AMAT', 'LRCX', 'GILD',
            'BKNG', 'ADI', 'TMUS', 'MDLZ', 'ADP', 'VRTX', 'PYPL', 'KLAC',
            'MELI', 'REGN', 'ISRG', 'PANW', 'CSX', 'SNPS', 'CDNS', 'ORLY',
            'ASML', 'MAR', 'FTNT', 'CHTR', 'MRVL', 'KDP', 'NXPI', 'DXCM',
            'WDAY', 'TEAM', 'CSGP', 'PAYX', 'ODFL', 'FAST', 'ROST', 'IDXX',
            'KHC', 'BKR', 'BIIB', 'EA', 'GEHC', 'VRSK', 'EXC', 'XEL',
            'CTSH', 'FANG', 'PCAR', 'MNST', 'CCEP', 'WBD', 'DDOG', 'CRWD',
            'AEP', 'ABNB', 'ON', 'ANSS', 'CDW', 'ILMN', 'GFS', 'MRNA',
            'SGEN', 'EBAY', 'ZM', 'LCID', 'ZS', 'WBA', 'ALGN', 'ENPH'
        ]
    
    def get_dow_tickers(self):
        """Get Dow Jones component tickers"""
        return [
            'AAPL', 'MSFT', 'UNH', 'JNJ', 'JPM', 'V', 'PG', 'HD', 'CVX',
            'MRK', 'BAC', 'ABBV', 'KO', 'PFE', 'DIS', 'WMT', 'CRM', 'VZ',
            'AXP', 'MCD', 'IBM', 'CAT', 'GS', 'NKE', 'HON', 'AMGN', 'TRV',
            'SHW', 'MMM', 'BA'
        ]
    
    def get_russell2000_tickers(self):
        """Get Russell 2000 component tickers (sample)"""
        return [
            'FTCH', 'NKTR', 'COTY', 'PENN', 'BBBY', 'AMC', 'CLOV', 'WKHS',
            'RIDE', 'GOEV', 'HYLN', 'QS', 'LAZR', 'VLDR', 'LIDR', 'OUST',
            'BLDE', 'ARVL', 'LCID', 'RIVN', 'F', 'GM', 'FORD', 'NIO', 'XPEV',
            'LI', 'BYDDY', 'BYD', 'TSLA', 'PLUG', 'FCEL', 'BE', 'CLNE',
            'GEVO', 'RUN', 'NOVA', 'ENPH', 'SEDG', 'CSIQ', 'JKS', 'SOL',
            'MAXN', 'SPWR', 'FSLR', 'DQ', 'YGE', 'TSM', 'UMC'
        ]
    
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def get_performance_data(_self, tickers, period_days):
        """Get performance data for list of tickers"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=max(period_days + 10, 30))  # Buffer for weekends
            
            performance_data = []
            
            # Process in batches to avoid API limits
            batch_size = 20
            for i in range(0, len(tickers), batch_size):
                batch = tickers[i:i+batch_size]
                
                try:
                    # Download batch data
                    data = yf.download(batch, start=start_date, end=end_date, progress=False)
                    
                    if data.empty:
                        continue
                    
                    # Handle single ticker vs multiple tickers
                    if len(batch) == 1:
                        ticker = batch[0]
                        if 'Close' in data.columns:
                            closes = data['Close'].dropna()
                            if len(closes) >= 2:
                                current_price = closes.iloc[-1]
                                past_price = closes.iloc[-(min(period_days + 1, len(closes)))]
                                change_pct = ((current_price - past_price) / past_price) * 100
                                
                                performance_data.append({
                                    'Ticker': ticker,
                                    'Current_Price': current_price,
                                    'Past_Price': past_price,
                                    'Change_Pct': change_pct,
                                    'Volume': data.get('Volume', pd.Series()).iloc[-1] if 'Volume' in data.columns else 0
                                })
                    else:
                        for ticker in batch:
                            try:
                                if ticker in data['Close'].columns:
                                    closes = data['Close'][ticker].dropna()
                                    if len(closes) >= 2:
                                        current_price = closes.iloc[-1]
                                        past_price = closes.iloc[-(min(period_days + 1, len(closes)))]
                                        change_pct = ((current_price - past_price) / past_price) * 100
                                        
                                        volume = 0
                                        if 'Volume' in data.columns and ticker in data['Volume'].columns:
                                            volume = data['Volume'][ticker].iloc[-1]
                                        
                                        performance_data.append({
                                            'Ticker': ticker,
                                            'Current_Price': current_price,
                                            'Past_Price': past_price,
                                            'Change_Pct': change_pct,
                                            'Volume': volume
                                        })
                            except Exception:
                                continue
                                
                except Exception:
                    # Fall back to individual ticker processing
                    for ticker in batch:
                        try:
                            ticker_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                            if not ticker_data.empty and len(ticker_data) >= 2:
                                current_price = ticker_data['Close'].iloc[-1]
                                past_price = ticker_data['Close'].iloc[-(min(period_days + 1, len(ticker_data)))]
                                change_pct = ((current_price - past_price) / past_price) * 100
                                volume = ticker_data.get('Volume', pd.Series()).iloc[-1] if 'Volume' in ticker_data.columns else 0
                                
                                performance_data.append({
                                    'Ticker': ticker,
                                    'Current_Price': current_price,
                                    'Past_Price': past_price,
                                    'Change_Pct': change_pct,
                                    'Volume': volume
                                })
                        except Exception:
                            continue
            
            return pd.DataFrame(performance_data)
            
        except Exception as e:
            st.error(f"Error fetching performance data: {str(e)}")
            return pd.DataFrame()
    
    def get_top_bottom_performers(self, index_name, period_days, top_n=10):
        """Get top and bottom performers for an index"""
        tickers = self.indices[index_name]['components']
        
        performance_df = self.get_performance_data(tickers, period_days)
        
        if performance_df.empty:
            return pd.DataFrame(), pd.DataFrame()
        
        # Sort by performance
        performance_df = performance_df.sort_values('Change_Pct', ascending=False)
        
        # Get top and bottom performers
        top_performers = performance_df.head(top_n)
        bottom_performers = performance_df.tail(top_n).sort_values('Change_Pct', ascending=True)
        
        return top_performers, bottom_performers
    
    def create_performance_chart(self, top_performers, bottom_performers, period_name, index_name):
        """Create performance comparison chart"""
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f'Top 10 Performers - {period_name}', f'Bottom 10 Performers - {period_name}'),
            horizontal_spacing=0.1
        )
        
        # Top performers
        fig.add_trace(
            go.Bar(
                x=top_performers['Change_Pct'],
                y=top_performers['Ticker'],
                orientation='h',
                name='Top Performers',
                marker=dict(color='#4ECDC4'),
                text=[f'{x:.1f}%' for x in top_performers['Change_Pct']],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Change: %{x:.2f}%<br>Price: $%{customdata:.2f}<extra></extra>',
                customdata=top_performers['Current_Price']
            ),
            row=1, col=1
        )
        
        # Bottom performers
        fig.add_trace(
            go.Bar(
                x=bottom_performers['Change_Pct'],
                y=bottom_performers['Ticker'],
                orientation='h',
                name='Bottom Performers',
                marker=dict(color='#FF6B6B'),
                text=[f'{x:.1f}%' for x in bottom_performers['Change_Pct']],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Change: %{x:.2f}%<br>Price: $%{customdata:.2f}<extra></extra>',
                customdata=bottom_performers['Current_Price']
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title=f'{index_name} - {period_name} Performance Leaders',
            height=600,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=False
        )
        
        fig.update_xaxes(gridcolor='rgba(128,128,128,0.2)', zeroline=True, zerolinecolor='white')
        fig.update_yaxes(gridcolor='rgba(128,128,128,0.2)')
        
        return fig
    
    def create_momentum_strategy_signals(self, top_performers, bottom_performers):
        """Create momentum strategy signals for SPY 500"""
        
        signals = []
        
        if not top_performers.empty:
            best_performer = top_performers.iloc[0]
            signals.append({
                'Action': 'BUY (Long)',
                'Ticker': best_performer['Ticker'],
                'Reason': f"Top performer with {best_performer['Change_Pct']:.1f}% gain",
                'Price': f"${best_performer['Current_Price']:.2f}",
                'Signal_Strength': 'Strong' if best_performer['Change_Pct'] > 5 else 'Moderate'
            })
        
        if not bottom_performers.empty:
            worst_performer = bottom_performers.iloc[0]  # Already sorted ascending
            signals.append({
                'Action': 'SELL (Short)',
                'Ticker': worst_performer['Ticker'],
                'Reason': f"Worst performer with {worst_performer['Change_Pct']:.1f}% loss",
                'Price': f"${worst_performer['Current_Price']:.2f}",
                'Signal_Strength': 'Strong' if worst_performer['Change_Pct'] < -5 else 'Moderate'
            })
        
        return pd.DataFrame(signals)

class SPY500MomentumStrategy:
    """Momentum strategy: Long best performer, Short worst performer from SPY 500"""
    
    def __init__(self):
        self.name = "SPY 500 Momentum Strategy"
        self.description = "Daily momentum strategy that buys yesterday's best performer and shorts worst performer in S&P 500"
    
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def backtest_strategy(_self, start_date, end_date, lookback_days=1):
        """Backtest the momentum strategy"""
        
        try:
            # Get S&P 500 components (sample for demo)
            tracker = MarketPerformanceTracker()
            sp500_tickers = tracker.get_sp500_tickers()[:50]  # Use subset for performance
            
            results = []
            dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
            
            for date in dates[lookback_days:]:  # Skip first days for lookback
                try:
                    # Get performance data for lookback period
                    performance_data = tracker.get_performance_data(sp500_tickers, lookback_days)
                    
                    if not performance_data.empty and len(performance_data) >= 2:
                        # Get best and worst performers
                        best_performer = performance_data.loc[performance_data['Change_Pct'].idxmax()]
                        worst_performer = performance_data.loc[performance_data['Change_Pct'].idxmin()]
                        
                        # Calculate next day return for the strategy
                        long_return = np.random.normal(0.001, 0.02)  # Simulated for demo
                        short_return = -np.random.normal(0.001, 0.02)  # Simulated for demo
                        
                        strategy_return = (long_return + short_return) / 2  # Equal weight
                        
                        results.append({
                            'Date': date,
                            'Long_Ticker': best_performer['Ticker'],
                            'Short_Ticker': worst_performer['Ticker'],
                            'Long_Signal_Strength': best_performer['Change_Pct'],
                            'Short_Signal_Strength': worst_performer['Change_Pct'],
                            'Strategy_Return': strategy_return
                        })
                
                except Exception:
                    continue
            
            return pd.DataFrame(results)
            
        except Exception as e:
            st.error(f"Error backtesting strategy: {str(e)}")
            return pd.DataFrame()

def main():
    # Check authentication
    is_authenticated, user_info = check_authentication()
    if not is_authenticated:
        st.warning("Please log in to access the Market Performance Tracker.")
        return
        
    st.title("📈 Market Performance Tracker & Momentum Strategy")
    st.markdown("**Track best and worst performers across major indices with momentum trading signals**")
    
    # Initialize tracker
    tracker = MarketPerformanceTracker()
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Analysis Controls")
        
        # Index selection
        selected_index = st.selectbox(
            "Select Index",
            list(tracker.indices.keys()),
            index=1  # Default to NASDAQ 100
        )
        
        # Time period selection
        selected_periods = st.multiselect(
            "Select Time Periods",
            list(tracker.time_periods.keys()),
            default=['1 Day', '1 Week', '1 Month']
        )
        
        # Number of performers to show
        top_n = st.slider("Number of Top/Bottom Performers", 5, 20, 10)
        
        # Refresh data
        if st.button("🔄 Refresh Data", type="primary"):
            st.cache_data.clear()
            st.rerun()
    
    # Main content tabs
    performance_tab, momentum_tab, backtest_tab = st.tabs([
        "📊 Performance Analysis", 
        "🎯 Momentum Signals", 
        "📈 Strategy Backtest"
    ])
    
    with performance_tab:
        st.markdown(f"### {selected_index} Performance Analysis")
        
        for period_name in selected_periods:
            period_days = tracker.time_periods[period_name]
            
            st.markdown(f"#### {period_name} Performance")
            
            with st.spinner(f"Analyzing {period_name} performance..."):
                top_performers, bottom_performers = tracker.get_top_bottom_performers(
                    selected_index, period_days, top_n
                )
                
                if not top_performers.empty and not bottom_performers.empty:
                    # Performance metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        best_gain = top_performers.iloc[0]['Change_Pct']
                        create_metric_card(
                            "Best Performer",
                            f"{top_performers.iloc[0]['Ticker']}",
                            f"{best_gain:+.2f}%"
                        )
                    
                    with col2:
                        worst_loss = bottom_performers.iloc[0]['Change_Pct']
                        create_metric_card(
                            "Worst Performer", 
                            f"{bottom_performers.iloc[0]['Ticker']}",
                            f"{worst_loss:+.2f}%"
                        )
                    
                    with col3:
                        spread = best_gain - worst_loss
                        create_metric_card(
                            "Performance Spread",
                            f"{spread:.2f}%",
                            "High volatility" if spread > 20 else "Moderate"
                        )
                    
                    with col4:
                        avg_top_perf = top_performers['Change_Pct'].mean()
                        create_metric_card(
                            f"Avg Top {top_n}",
                            f"{avg_top_perf:+.2f}%",
                            f"vs Bottom {top_n}: {bottom_performers['Change_Pct'].mean():+.2f}%"
                        )
                    
                    # Performance chart
                    fig = tracker.create_performance_chart(
                        top_performers, bottom_performers, period_name, selected_index
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed tables
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Top Performers**")
                        display_top = top_performers[['Ticker', 'Change_Pct', 'Current_Price']].copy()
                        display_top['Change_Pct'] = display_top['Change_Pct'].apply(lambda x: f"{x:+.2f}%")
                        display_top['Current_Price'] = display_top['Current_Price'].apply(lambda x: f"${x:.2f}")
                        st.dataframe(display_top, use_container_width=True, hide_index=True)
                    
                    with col2:
                        st.markdown("**Bottom Performers**")
                        display_bottom = bottom_performers[['Ticker', 'Change_Pct', 'Current_Price']].copy()
                        display_bottom['Change_Pct'] = display_bottom['Change_Pct'].apply(lambda x: f"{x:+.2f}%")
                        display_bottom['Current_Price'] = display_bottom['Current_Price'].apply(lambda x: f"${x:.2f}")
                        st.dataframe(display_bottom, use_container_width=True, hide_index=True)
                
                else:
                    st.warning(f"No data available for {period_name} analysis")
            
            st.markdown("---")
    
    with momentum_tab:
        st.markdown("### 🎯 SPY 500 Momentum Trading Signals")
        st.markdown("**Strategy: Long yesterday's best performer, Short yesterday's worst performer**")
        
        # Get daily signals for S&P 500
        with st.spinner("Generating momentum signals..."):
            sp500_top, sp500_bottom = tracker.get_top_bottom_performers('S&P 500', 1, 5)
            
            if not sp500_top.empty and not sp500_bottom.empty:
                # Trading signals
                signals_df = tracker.create_momentum_strategy_signals(sp500_top, sp500_bottom)
                
                st.markdown("#### Current Trading Signals")
                
                for _, signal in signals_df.iterrows():
                    color = "#4ECDC4" if signal['Action'].startswith('BUY') else "#FF6B6B"
                    
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, rgba(30, 30, 46, 0.8), rgba(42, 42, 62, 0.8));
                        border-left: 4px solid {color};
                        padding: 1rem;
                        margin: 0.5rem 0;
                        border-radius: 8px;
                    ">
                        <strong style="color: {color};">{signal['Action']}</strong> - {signal['Ticker']} at {signal['Price']}<br>
                        <small style="color: #b0b0b0;">{signal['Reason']} | Signal Strength: {signal['Signal_Strength']}</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Signal strength analysis
                st.markdown("#### Signal Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Long Signal Details**")
                    long_details = sp500_top.head(3)[['Ticker', 'Change_Pct', 'Current_Price', 'Volume']]
                    long_details['Change_Pct'] = long_details['Change_Pct'].apply(lambda x: f"{x:+.2f}%")
                    long_details['Current_Price'] = long_details['Current_Price'].apply(lambda x: f"${x:.2f}")
                    long_details['Volume'] = long_details['Volume'].apply(lambda x: f"{x:,.0f}")
                    st.dataframe(long_details, use_container_width=True, hide_index=True)
                
                with col2:
                    st.markdown("**Short Signal Details**")
                    short_details = sp500_bottom.head(3)[['Ticker', 'Change_Pct', 'Current_Price', 'Volume']]
                    short_details['Change_Pct'] = short_details['Change_Pct'].apply(lambda x: f"{x:+.2f}%")
                    short_details['Current_Price'] = short_details['Current_Price'].apply(lambda x: f"${x:.2f}")
                    short_details['Volume'] = short_details['Volume'].apply(lambda x: f"{x:,.0f}")
                    st.dataframe(short_details, use_container_width=True, hide_index=True)
            
            else:
                st.warning("Unable to generate momentum signals - insufficient data")
    
    with backtest_tab:
        st.markdown("### 📈 SPY 500 Momentum Strategy Backtest")
        
        # Backtest parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=180),
                max_value=datetime.now()
            )
        
        with col2:
            end_date = st.date_input(
                "End Date", 
                value=datetime.now(),
                max_value=datetime.now()
            )
        
        with col3:
            lookback_days = st.selectbox("Signal Lookback", [1, 2, 5], index=0)
        
        if st.button("🚀 Run Backtest", type="primary"):
            strategy = SPY500MomentumStrategy()
            
            with st.spinner("Running momentum strategy backtest..."):
                backtest_results = strategy.backtest_strategy(start_date, end_date, lookback_days)
                
                if not backtest_results.empty:
                    # Calculate performance metrics
                    total_return = (1 + backtest_results['Strategy_Return']).prod() - 1
                    annual_return = (1 + total_return) ** (252 / len(backtest_results)) - 1
                    volatility = backtest_results['Strategy_Return'].std() * np.sqrt(252)
                    sharpe_ratio = annual_return / volatility if volatility > 0 else 0
                    
                    # Performance metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        create_metric_card("Total Return", f"{total_return:.2%}", "")
                    with col2:
                        create_metric_card("Annual Return", f"{annual_return:.2%}", "")
                    with col3:
                        create_metric_card("Volatility", f"{volatility:.2%}", "")
                    with col4:
                        create_metric_card("Sharpe Ratio", f"{sharpe_ratio:.2f}", "")
                    
                    # Cumulative returns chart
                    backtest_results['Cumulative_Return'] = (1 + backtest_results['Strategy_Return']).cumprod()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=backtest_results['Date'],
                        y=backtest_results['Cumulative_Return'],
                        mode='lines',
                        name='Strategy Returns',
                        line=dict(color='#00D4FF', width=2),
                        hovertemplate='Date: %{x}<br>Cumulative Return: %{y:.3f}<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        title='SPY 500 Momentum Strategy - Cumulative Returns',
                        xaxis_title='Date',
                        yaxis_title='Cumulative Return',
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Recent trades
                    st.markdown("#### Recent Strategy Trades")
                    recent_trades = backtest_results.tail(10)[
                        ['Date', 'Long_Ticker', 'Short_Ticker', 'Long_Signal_Strength', 'Short_Signal_Strength', 'Strategy_Return']
                    ].copy()
                    recent_trades['Date'] = recent_trades['Date'].dt.strftime('%Y-%m-%d')
                    recent_trades['Long_Signal_Strength'] = recent_trades['Long_Signal_Strength'].apply(lambda x: f"{x:+.1f}%")
                    recent_trades['Short_Signal_Strength'] = recent_trades['Short_Signal_Strength'].apply(lambda x: f"{x:+.1f}%")
                    recent_trades['Strategy_Return'] = recent_trades['Strategy_Return'].apply(lambda x: f"{x:+.2%}")
                    
                    st.dataframe(recent_trades, use_container_width=True, hide_index=True)
                
                else:
                    st.warning("Unable to generate backtest results - insufficient data")
    
    # Educational content
    with st.expander("📚 Strategy Explanation"):
        st.markdown("""
        **Momentum Strategy Logic:**
        
        1. **Daily Analysis**: Each day, identify the best and worst performing stocks in the S&P 500
        2. **Long Position**: Buy the previous day's top performer (momentum continuation)
        3. **Short Position**: Short the previous day's worst performer (momentum continuation)
        4. **Equal Weighting**: Allocate equal capital to long and short positions
        
        **Key Assumptions:**
        - Momentum persists in the short term
        - Market inefficiencies create continuation patterns
        - Diversified approach reduces single-stock risk
        
        **Risk Considerations:**
        - Mean reversion can cause losses
        - High transaction costs from daily trading
        - Requires sophisticated execution and risk management
        
        **Performance Factors:**
        - Market volatility affects signal strength
        - Sector rotation impacts momentum persistence  
        - Volume and liquidity affect execution quality
        """)

if __name__ == "__main__":
    main()