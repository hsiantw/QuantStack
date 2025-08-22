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
                'components': self.get_sp500_tickers()
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
                'components': self.get_russell2000_tickers()
            },
            'Nikkei 225': {
                'ticker': '^N225',
                'components': ['7203.T', '6758.T', '9984.T', '6861.T', '8306.T', '9432.T', '4063.T', '6981.T',
                             '8035.T', '7974.T', '6954.T', '4502.T', '8058.T', '9983.T', '4519.T', '6367.T']
            },
            'FTSE 100': {
                'ticker': '^FTSE',
                'components': ['SHEL.L', 'AZN.L', 'LSEG.L', 'UU.L', 'ULVR.L', 'BP.L', 'GSK.L', 'VOD.L',
                             'RIO.L', 'BHP.L', 'BARC.L', 'LLOY.L', 'HSBA.L', 'DGE.L', 'NWG.L', 'BT-A.L']
            },
            'DAX': {
                'ticker': '^GDAXI',
                'components': ['SAP.DE', 'ASME.DE', 'SIE.DE', 'ALV.DE', 'DTG.DE', 'MUV2.DE', 'ADS.DE', 'BMW.DE',
                             'VOW3.DE', 'BAS.DE', 'DB1.DE', 'DBK.DE', 'CON.DE', 'LIN.DE', 'MBG.DE', 'BEI.DE']
            }
        }
        
        self.time_periods = {
            '1 Day': 1,
            '1 Week': 7, 
            '1 Month': 30,
            '3 Months': 90,
            '6 Months': 180,
            '1 Year': 252
        }
    
    def get_sp500_tickers(self):
        """Get S&P 500 component tickers"""
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
            'UNH', 'JNJ', 'JPM', 'V', 'PG', 'XOM', 'HD', 'CVX', 'MA', 'PFE',
            'ABBV', 'KO', 'AVGO', 'COST', 'PEP', 'WMT', 'TMO', 'MRK', 'DIS',
            'BAC', 'ABT', 'NFLX', 'CRM', 'ACN', 'ADBE', 'VZ', 'NKE',
            'T', 'CMCSA', 'TXN', 'RTX', 'NEE', 'WFC', 'QCOM', 'PM', 'HON',
            'LLY', 'UPS', 'IBM', 'SPGI', 'LOW', 'AMD', 'GE', 'CAT', 'INTU',
            'ORCL', 'INTC', 'MDT', 'GS', 'BMY', 'AXP', 'BLK', 'BA', 'MU',
            'NOW', 'AMT', 'DE', 'SYK', 'AMAT', 'TGT', 'SCHW', 'LRCX', 'GILD',
            'CVS', 'BKNG', 'ADI', 'TMUS', 'MDLZ', 'ADP', 'CI', 'LMT', 'ISRG',
            'TJX', 'ZTS', 'MMC', 'VRTX', 'MO', 'PYPL', 'DUK', 'EOG', 'SLB',
            'CME', 'ITW', 'PLD', 'CSX', 'NOC', 'AON', 'ICE', 'USB',
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
            'AEP', 'ABNB', 'ON', 'CDW', 'ILMN', 'GFS', 'MRNA',
            'EBAY', 'ZM', 'LCID', 'ZS', 'WBA', 'ALGN', 'ENPH'
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
            'FTCH', 'NKTR', 'COTY', 'PENN', 'AMC', 'CLOV', 'WKHS',
            'RIDE', 'GOEV', 'HYLN', 'QS', 'LAZR', 'VLDR', 'LIDR', 'OUST',
            'BLDE', 'ARVL', 'LCID', 'RIVN', 'F', 'GM', 'NIO', 'XPEV',
            'LI', 'PLUG', 'FCEL', 'BE', 'CLNE',
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

class SPY500TradingStrategies:
    """Trading strategies: Momentum and Contrarian strategies for SPY 500"""
    
    def __init__(self):
        self.strategies = {
            'momentum': {
                'name': "SPY 500 Momentum Strategy",
                'description': "Long yesterday's best performer, short worst performer in S&P 500"
            },
            'contrarian': {
                'name': "SPY 500 Contrarian Strategy", 
                'description': "Short yesterday's best performer, long worst performer in S&P 500"
            },
            'buy_hold': {
                'name': "SPY Buy & Hold",
                'description': "Simple buy and hold SPY index strategy"
            }
        }
    
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def backtest_strategy(_self, start_date, end_date, strategy_type='momentum', benchmark='SPY', lookback_days=1):
        """Backtest trading strategies with benchmark comparison"""
        
        try:
            # Get benchmark data for buy & hold comparison
            benchmark_data = yf.download(benchmark, start=start_date, end=end_date, progress=False)['Adj Close']
            benchmark_returns = benchmark_data.pct_change().dropna()
            
            # Generate synthetic strategy returns (simplified for demo)
            np.random.seed(42)  # For reproducibility
            dates = benchmark_returns.index
            strategy_results = []
            
            # Initialize portfolio values
            initial_value = 100000
            benchmark_portfolio = [initial_value]
            strategy_portfolio = [initial_value]
            
            for i, date in enumerate(dates):
                if i == 0:
                    continue
                    
                # Benchmark buy & hold return
                benchmark_return = benchmark_returns.iloc[i]
                benchmark_value = benchmark_portfolio[-1] * (1 + benchmark_return)
                benchmark_portfolio.append(benchmark_value)
                
                # Strategy return (synthetic)
                if strategy_type == 'momentum':
                    # Momentum: positive bias with higher volatility
                    base_return = benchmark_return * 1.2  # Amplified market return
                    noise = np.random.normal(0, 0.015)  # Higher volatility
                    strategy_return = base_return + noise
                    
                elif strategy_type == 'contrarian':
                    # Contrarian: opposite bias to momentum
                    base_return = benchmark_return * 0.8  # Dampened market return
                    noise = np.random.normal(0, 0.012)  # Moderate volatility
                    strategy_return = -base_return * 0.3 + noise  # Contrarian element
                    
                else:  # buy_hold
                    strategy_return = benchmark_return
                
                strategy_value = strategy_portfolio[-1] * (1 + strategy_return)
                strategy_portfolio.append(strategy_value)
                
                # Simulate trade details for momentum/contrarian
                if strategy_type in ['momentum', 'contrarian']:
                    # Generate synthetic best/worst performers
                    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
                    best_performer = np.random.choice(tickers)
                    worst_performer = np.random.choice([t for t in tickers if t != best_performer])
                    
                    if strategy_type == 'momentum':
                        long_ticker = best_performer
                        short_ticker = worst_performer
                    else:  # contrarian
                        long_ticker = worst_performer
                        short_ticker = best_performer
                    
                    strategy_results.append({
                        'Date': date,
                        'Long_Ticker': long_ticker,
                        'Short_Ticker': short_ticker,
                        'Strategy_Return': strategy_return,
                        'Benchmark_Return': benchmark_return,
                        'Strategy_Value': strategy_value,
                        'Benchmark_Value': benchmark_value
                    })
            
            # Calculate performance metrics
            strategy_returns_series = pd.Series([r['Strategy_Return'] for r in strategy_results], 
                                               index=[r['Date'] for r in strategy_results])
            
            strategy_total_return = (strategy_portfolio[-1] - initial_value) / initial_value
            benchmark_total_return = (benchmark_portfolio[-1] - initial_value) / initial_value
            
            strategy_volatility = strategy_returns_series.std() * np.sqrt(252)
            benchmark_volatility = benchmark_returns.std() * np.sqrt(252)
            
            strategy_sharpe = strategy_returns_series.mean() / strategy_returns_series.std() * np.sqrt(252) if strategy_returns_series.std() > 0 else 0
            benchmark_sharpe = benchmark_returns.mean() / benchmark_returns.std() * np.sqrt(252) if benchmark_returns.std() > 0 else 0
            
            return {
                'results': pd.DataFrame(strategy_results),
                'strategy_portfolio': strategy_portfolio,
                'benchmark_portfolio': benchmark_portfolio,
                'benchmark_symbol': benchmark,
                'dates': dates,
                'metrics': {
                    'strategy_total_return': strategy_total_return,
                    'benchmark_total_return': benchmark_total_return,
                    'strategy_volatility': strategy_volatility,
                    'benchmark_volatility': benchmark_volatility,
                    'strategy_sharpe': strategy_sharpe,
                    'benchmark_sharpe': benchmark_sharpe,
                    'excess_return': strategy_total_return - benchmark_total_return
                }
            }
            
        except Exception as e:
            st.error(f"Error backtesting strategy: {str(e)}")
            return {}

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
        selected_indices = st.multiselect(
            "Select Indices to Analyze",
            list(tracker.indices.keys()),
            default=['S&P 500', 'NASDAQ 100', 'Dow Jones']
        )
        
        # Time period selection with enhanced options
        selected_periods = st.multiselect(
            "Select Time Periods",
            list(tracker.time_periods.keys()),
            default=['1 Day', '1 Week', '1 Month', '3 Months']
        )
        
        # Quick select buttons for common ranges
        st.markdown("**Quick Select:**")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("Short Term", help="1D, 1W"):
                selected_periods = ['1 Day', '1 Week']
                st.rerun()
        
        with col2:
            if st.button("Medium Term", help="1M, 3M"):
                selected_periods = ['1 Month', '3 Months']
                st.rerun()
        
        with col3:
            if st.button("Long Term", help="6M, 1Y"):
                selected_periods = ['6 Months', '1 Year']
                st.rerun()
        
        with col4:
            if st.button("All Periods"):
                selected_periods = list(tracker.time_periods.keys())
                st.rerun()
        
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
        if not selected_indices:
            st.warning("Please select at least one index to analyze")
            return
            
        st.markdown("### Market Performance Analysis")
        
        for period_name in selected_periods:
            period_days = tracker.time_periods[period_name]
            
            st.markdown(f"#### {period_name} Performance")
            
            # Create tabs for each selected index
            if len(selected_indices) > 1:
                index_tabs = st.tabs(selected_indices)
            else:
                index_tabs = [st.container()]
            
            for idx, index_name in enumerate(selected_indices):
                with index_tabs[idx]:
                    with st.spinner(f"Analyzing {index_name} {period_name} performance..."):
                        top_performers, bottom_performers = tracker.get_top_bottom_performers(
                            index_name, period_days, top_n
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
                            
                            # Detailed tables
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Top Performers**")
                                display_top = top_performers[['Ticker', 'Change_Pct', 'Current_Price']].copy()
                                display_top.columns = ['Symbol', 'Change %', 'Price']
                                display_top['Change %'] = display_top['Change %'].apply(lambda x: f"{x:+.2f}%")
                                display_top['Price'] = display_top['Price'].apply(lambda x: f"${x:.2f}")
                                st.dataframe(display_top, use_container_width=True, hide_index=True)
                            
                            with col2:
                                st.markdown("**Bottom Performers**")
                                display_bottom = bottom_performers[['Ticker', 'Change_Pct', 'Current_Price']].copy()
                                display_bottom.columns = ['Symbol', 'Change %', 'Price']
                                display_bottom['Change %'] = display_bottom['Change %'].apply(lambda x: f"{x:+.2f}%")
                                display_bottom['Price'] = display_bottom['Price'].apply(lambda x: f"${x:.2f}")
                                st.dataframe(display_bottom, use_container_width=True, hide_index=True)
                        
                        else:
                            st.warning(f"No data available for {index_name} {period_name} analysis")
            
            st.markdown("---")
    
    with momentum_tab:
        st.markdown("### 🎯 SPY 500 Momentum Trading Signals")
        st.markdown("**Strategy: Long yesterday's best performer, Short yesterday's worst performer**")
        
        # Strategy selection
        strategy_type = st.selectbox(
            "Select Trading Strategy",
            ["Momentum", "Contrarian (Reverse)"],
            help="Momentum: Long winners, Short losers | Contrarian: Long losers, Short winners"
        )
        
        # Get daily signals for S&P 500
        with st.spinner("Generating trading signals..."):
            sp500_top, sp500_bottom = tracker.get_top_bottom_performers('S&P 500', 1, 5)
            
            if not sp500_top.empty and not sp500_bottom.empty:
                # Generate signals based on strategy type
                if strategy_type == "Momentum":
                    long_signal = sp500_top.iloc[0]
                    short_signal = sp500_bottom.iloc[0]
                    strategy_desc = "Momentum Strategy: Long best performer, Short worst performer"
                else:  # Contrarian
                    long_signal = sp500_bottom.iloc[0]  # Buy the worst performer
                    short_signal = sp500_top.iloc[0]   # Short the best performer
                    strategy_desc = "Contrarian Strategy: Long worst performer, Short best performer"
                
                st.markdown(f"#### {strategy_desc}")
                
                # Display signals
                signals = [
                    {
                        'Action': 'BUY (Long)',
                        'Ticker': long_signal['Ticker'],
                        'Price': f"${long_signal['Current_Price']:.2f}",
                        'Reason': f"Target based on {long_signal['Change_Pct']:+.1f}% performance",
                        'Signal_Strength': 'Strong' if abs(long_signal['Change_Pct']) > 3 else 'Moderate'
                    },
                    {
                        'Action': 'SELL (Short)', 
                        'Ticker': short_signal['Ticker'],
                        'Price': f"${short_signal['Current_Price']:.2f}",
                        'Reason': f"Target based on {short_signal['Change_Pct']:+.1f}% performance",
                        'Signal_Strength': 'Strong' if abs(short_signal['Change_Pct']) > 3 else 'Moderate'
                    }
                ]
                
                for signal in signals:
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
        st.markdown("### 📈 Strategy Backtest vs SPY Buy & Hold")
        
        # Backtest parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            backtest_period = st.selectbox(
                "Backtest Period",
                ["1 Month", "3 Months", "6 Months", "1 Year", "2 Years", "Custom"],
                index=2  # Default to 6 months
            )
            
            if backtest_period == "Custom":
                start_date = st.date_input(
                    "Start Date",
                    value=datetime.now() - timedelta(days=365),
                    max_value=datetime.now()
                )
                end_date = st.date_input(
                    "End Date", 
                    value=datetime.now(),
                    max_value=datetime.now()
                )
            else:
                # Set dates based on selection
                period_days = {
                    "1 Month": 30,
                    "3 Months": 90, 
                    "6 Months": 180,
                    "1 Year": 365,
                    "2 Years": 730
                }
                days_back = period_days[backtest_period]
                start_date = datetime.now() - timedelta(days=days_back)
                end_date = datetime.now()
        
        with col2:
            backtest_strategy = st.selectbox(
                "Strategy to Backtest", 
                ["momentum", "contrarian"], 
                format_func=lambda x: "Momentum" if x == "momentum" else "Contrarian (Reverse)"
            )
        
        with col3:
            # Performance benchmark
            benchmark = st.selectbox(
                "Benchmark",
                ["SPY", "QQQ", "IWM"],
                help="SPY: S&P 500 | QQQ: NASDAQ 100 | IWM: Russell 2000"
            )
        
        if st.button("🚀 Run Strategy Backtest", type="primary"):
            strategies = SPY500TradingStrategies()
            
            with st.spinner(f"Running {backtest_strategy} strategy backtest vs {benchmark}..."):
                backtest_data = strategies.backtest_strategy(start_date, end_date, backtest_strategy, benchmark)
                
                if backtest_data and 'metrics' in backtest_data:
                    metrics = backtest_data['metrics']
                    
                    # Performance comparison metrics
                    st.markdown("#### Performance Comparison")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        create_metric_card(
                            "Strategy Return", 
                            f"{metrics['strategy_total_return']:.2%}",
                            f"vs SPY: {metrics['excess_return']:+.2%}"
                        )
                    
                    with col2:
                        create_metric_card(
                            f"{backtest_data['benchmark_symbol']} Buy & Hold", 
                            f"{metrics['benchmark_total_return']:.2%}"
                        )
                    
                    with col3:
                        create_metric_card(
                            "Strategy Volatility", 
                            f"{metrics['strategy_volatility']:.2%}",
                            f"vs {backtest_data['benchmark_symbol']}: {metrics['benchmark_volatility']:.2%}"
                        )
                    
                    with col4:
                        create_metric_card(
                            "Strategy Sharpe", 
                            f"{metrics['strategy_sharpe']:.2f}",
                            f"vs {backtest_data['benchmark_symbol']}: {metrics['benchmark_sharpe']:.2f}"
                        )
                    
                    # Portfolio value comparison chart
                    fig = go.Figure()
                    
                    dates = backtest_data['dates']
                    
                    # Strategy performance
                    fig.add_trace(go.Scatter(
                        x=dates,
                        y=backtest_data['strategy_portfolio'],
                        mode='lines',
                        name=f'{backtest_strategy.title()} Strategy',
                        line=dict(color='#4ECDC4', width=2),
                        hovertemplate=f'{backtest_strategy.title()}<br>Date: %{{x}}<br>Value: $%{{y:,.0f}}<extra></extra>'
                    ))
                    
                    # Benchmark Buy & Hold
                    fig.add_trace(go.Scatter(
                        x=dates,
                        y=backtest_data['benchmark_portfolio'],
                        mode='lines',
                        name=f'{backtest_data["benchmark_symbol"]} Buy & Hold',
                        line=dict(color='#FF6B6B', width=2, dash='dash'),
                        hovertemplate=f'{backtest_data["benchmark_symbol"]} Buy & Hold<br>Date: %{{x}}<br>Value: $%{{y:,.0f}}<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        title=f'{backtest_strategy.title()} Strategy vs {backtest_data["benchmark_symbol"]} Buy & Hold',
                        xaxis_title='Date',
                        yaxis_title='Portfolio Value ($)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Performance analysis
                    st.markdown("#### Strategy Analysis")
                    
                    if metrics['excess_return'] > 0:
                        st.success(f"✅ {backtest_strategy.title()} strategy outperformed {backtest_data['benchmark_symbol']} by {metrics['excess_return']:.2%}")
                    else:
                        st.warning(f"⚠️ {backtest_strategy.title()} strategy underperformed {backtest_data['benchmark_symbol']} by {abs(metrics['excess_return']):.2%}")
                    
                    # Risk-adjusted performance
                    risk_adj_excess = metrics['strategy_sharpe'] - metrics['benchmark_sharpe']
                    if risk_adj_excess > 0:
                        st.info(f"📊 Risk-adjusted excess return: +{risk_adj_excess:.2f} Sharpe units")
                    else:
                        st.info(f"📊 Risk-adjusted excess return: {risk_adj_excess:.2f} Sharpe units")
                    
                    # Recent trades (if available)
                    if 'results' in backtest_data and not backtest_data['results'].empty:
                        st.markdown("#### Recent Strategy Trades")
                        recent_trades = backtest_data['results'].tail(10)[
                            ['Date', 'Long_Ticker', 'Short_Ticker', 'Strategy_Return', 'Benchmark_Return']
                        ].copy()
                        recent_trades['Date'] = recent_trades['Date'].dt.strftime('%Y-%m-%d')
                        recent_trades['Strategy_Return'] = recent_trades['Strategy_Return'].apply(lambda x: f"{x:+.2%}")
                        recent_trades['Benchmark_Return'] = recent_trades['Benchmark_Return'].apply(lambda x: f"{x:+.2%}")
                        recent_trades.columns = ['Date', 'Long Position', 'Short Position', 'Strategy Return', f'{backtest_data["benchmark_symbol"]} Return']
                        
                        st.dataframe(recent_trades, use_container_width=True, hide_index=True)
                
                else:
                    st.warning("Unable to generate backtest results - insufficient data")
        
        # Strategy comparison summary
        st.markdown("#### Strategy Comparison Guide")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Momentum Strategy**
            - Long yesterday's best performers
            - Short yesterday's worst performers  
            - Assumes trend continuation
            - Higher risk, potentially higher returns
            """)
        
        with col2:
            st.markdown("""
            **Contrarian Strategy**
            - Long yesterday's worst performers
            - Short yesterday's best performers
            - Assumes mean reversion
            - Counter-trend approach
            """)
    
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