import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from scipy.stats import norm
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

class OptionsAnalyzer:
    """Advanced options analysis and derivatives trading tools"""
    
    def __init__(self):
        self.risk_free_rate = 0.05  # 5% risk-free rate
        
    @st.cache_data(ttl=1800)  # Cache for 30 minutes
    def get_options_data(_self, symbol, expiry_date=None):
        """Get options chain data for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get available expiration dates
            expirations = ticker.options
            if not expirations:
                return None, None, []
            
            # Use specified expiry or nearest expiry
            if expiry_date and expiry_date in expirations:
                selected_expiry = expiry_date
            else:
                selected_expiry = expirations[0]
            
            # Get options chain
            options_chain = ticker.option_chain(selected_expiry)
            calls = options_chain.calls
            puts = options_chain.puts
            
            return calls, puts, expirations
            
        except Exception as e:
            st.error(f"Error fetching options data: {str(e)}")
            return None, None, []
    
    def black_scholes_call(self, S, K, T, r, sigma):
        """Black-Scholes call option pricing"""
        try:
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            return call_price
        except:
            return 0
    
    def black_scholes_put(self, S, K, T, r, sigma):
        """Black-Scholes put option pricing"""
        try:
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            return put_price
        except:
            return 0
    
    def calculate_greeks(self, S, K, T, r, sigma, option_type='call'):
        """Calculate option Greeks"""
        try:
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            # Delta
            if option_type == 'call':
                delta = norm.cdf(d1)
            else:
                delta = -norm.cdf(-d1)
            
            # Gamma
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            
            # Theta
            if option_type == 'call':
                theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - 
                        r * K * np.exp(-r * T) * norm.cdf(d2))
            else:
                theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + 
                        r * K * np.exp(-r * T) * norm.cdf(-d2))
            
            # Vega
            vega = S * norm.pdf(d1) * np.sqrt(T)
            
            # Rho
            if option_type == 'call':
                rho = K * T * np.exp(-r * T) * norm.cdf(d2)
            else:
                rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
            
            return {
                'delta': delta,
                'gamma': gamma,
                'theta': theta / 365,  # Daily theta
                'vega': vega / 100,    # Vega per 1% change
                'rho': rho / 100       # Rho per 1% change
            }
        except:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
    
    def calculate_implied_volatility(self, market_price, S, K, T, r, option_type='call'):
        """Calculate implied volatility using Newton-Raphson method"""
        try:
            sigma = 0.3  # Initial guess
            tolerance = 1e-6
            max_iterations = 100
            
            for _ in range(max_iterations):
                if option_type == 'call':
                    price = self.black_scholes_call(S, K, T, r, sigma)
                else:
                    price = self.black_scholes_put(S, K, T, r, sigma)
                
                vega = S * norm.pdf((np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / 
                                   (sigma * np.sqrt(T))) * np.sqrt(T)
                
                diff = price - market_price
                
                if abs(diff) < tolerance:
                    return sigma
                
                if vega != 0:
                    sigma = sigma - diff / vega
                else:
                    break
            
            return sigma
        except:
            return 0.3
    
    def create_payoff_diagram(self, strategy_legs, spot_range=None):
        """Create payoff diagram for options strategies"""
        if not strategy_legs:
            return go.Figure()
        
        # Determine spot range
        strikes = [leg['strike'] for leg in strategy_legs]
        min_strike, max_strike = min(strikes), max(strikes)
        
        if spot_range is None:
            spot_range = np.linspace(min_strike * 0.8, max_strike * 1.2, 100)
        
        total_payoff = np.zeros(len(spot_range))
        
        for leg in strategy_legs:
            strike = leg['strike']
            premium = leg['premium']
            quantity = leg['quantity']
            option_type = leg['type']
            position = leg['position']  # 'long' or 'short'
            
            if option_type == 'call':
                payoff = np.maximum(spot_range - strike, 0) * quantity
            else:  # put
                payoff = np.maximum(strike - spot_range, 0) * quantity
            
            if position == 'short':
                payoff = -payoff
            
            # Subtract premium paid (or add premium received)
            if position == 'long':
                payoff -= premium * quantity
            else:
                payoff += premium * quantity
            
            total_payoff += payoff
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=spot_range,
            y=total_payoff,
            mode='lines',
            name='Total Payoff',
            line=dict(color='#4ECDC4', width=3),
            hovertemplate='Stock Price: $%{x:.2f}<br>Payoff: $%{y:.2f}<extra></extra>'
        ))
        
        # Add break-even lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                     annotation_text="Break-even")
        
        fig.update_layout(
            title='Options Strategy Payoff Diagram',
            xaxis_title='Stock Price at Expiration ($)',
            yaxis_title='Profit/Loss ($)',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400
        )
        
        return fig
    
    def analyze_volatility_surface(self, symbol):
        """Analyze implied volatility surface"""
        try:
            ticker = yf.Ticker(symbol)
            expirations = ticker.options[:5]  # First 5 expirations
            
            surface_data = []
            
            for exp in expirations:
                try:
                    options_chain = ticker.option_chain(exp)
                    calls = options_chain.calls
                    
                    current_price = ticker.history(period="1d")['Close'].iloc[-1]
                    exp_date = datetime.strptime(exp, '%Y-%m-%d')
                    days_to_exp = (exp_date - datetime.now()).days
                    T = days_to_exp / 365.0
                    
                    for _, option in calls.iterrows():
                        if option['volume'] > 0 and option['lastPrice'] > 0:
                            strike = option['strike']
                            market_price = option['lastPrice']
                            
                            iv = self.calculate_implied_volatility(
                                market_price, current_price, strike, T, 
                                self.risk_free_rate, 'call'
                            )
                            
                            moneyness = strike / current_price
                            
                            surface_data.append({
                                'expiration': exp,
                                'days_to_exp': days_to_exp,
                                'strike': strike,
                                'moneyness': moneyness,
                                'implied_vol': iv * 100,
                                'volume': option['volume']
                            })
                except:
                    continue
            
            return pd.DataFrame(surface_data)
            
        except Exception as e:
            st.error(f"Error analyzing volatility surface: {str(e)}")
            return pd.DataFrame()

def main():
    # Check authentication
    is_authenticated, user_info = check_authentication()
    if not is_authenticated:
        st.warning("Please log in to access Options Analysis.")
        return
        
    st.title("📊 Options Analysis & Derivatives Trading")
    st.markdown("**Advanced options pricing, Greeks analysis, and strategy modeling**")
    
    # Initialize analyzer
    analyzer = OptionsAnalyzer()
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Analysis Controls")
        
        # Symbol input
        symbol = st.text_input("Stock Symbol", value="AAPL").upper()
        
        # Analysis mode
        analysis_mode = st.selectbox(
            "Analysis Mode",
            ["Options Chain", "Strategy Builder", "Volatility Analysis", "Greeks Calculator"]
        )
        
        # Risk-free rate
        analyzer.risk_free_rate = st.slider(
            "Risk-Free Rate (%)", 0.0, 10.0, 5.0
        ) / 100
        
        # Refresh data
        if st.button("🔄 Refresh Data", type="primary"):
            st.cache_data.clear()
            st.rerun()
    
    # Main content based on analysis mode
    if analysis_mode == "Options Chain":
        st.markdown("### Options Chain Analysis")
        
        # Get options data
        with st.spinner("Fetching options data..."):
            calls, puts, expirations = analyzer.get_options_data(symbol)
        
        if calls is not None and puts is not None:
            # Expiration selection
            if expirations:
                selected_expiry = st.selectbox(
                    "Select Expiration Date", expirations
                )
                
                # Re-fetch data for selected expiry
                calls, puts, _ = analyzer.get_options_data(symbol, selected_expiry)
            
            # Display options chains
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Call Options**")
                if not calls.empty:
                    # Calculate key metrics
                    calls_display = calls[[
                        'strike', 'lastPrice', 'bid', 'ask', 'volume', 
                        'openInterest', 'impliedVolatility'
                    ]].copy()
                    calls_display['impliedVolatility'] = (calls_display['impliedVolatility'] * 100).round(2)
                    calls_display.columns = [
                        'Strike', 'Last', 'Bid', 'Ask', 'Volume', 
                        'Open Int', 'IV %'
                    ]
                    st.dataframe(calls_display, use_container_width=True)
                else:
                    st.warning("No call options data available")
            
            with col2:
                st.markdown("**Put Options**")
                if not puts.empty:
                    puts_display = puts[[
                        'strike', 'lastPrice', 'bid', 'ask', 'volume', 
                        'openInterest', 'impliedVolatility'
                    ]].copy()
                    puts_display['impliedVolatility'] = (puts_display['impliedVolatility'] * 100).round(2)
                    puts_display.columns = [
                        'Strike', 'Last', 'Bid', 'Ask', 'Volume', 
                        'Open Int', 'IV %'
                    ]
                    st.dataframe(puts_display, use_container_width=True)
                else:
                    st.warning("No put options data available")
            
            # Options metrics
            if not calls.empty or not puts.empty:
                st.markdown("### Options Market Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_call_volume = calls['volume'].sum() if not calls.empty else 0
                    create_metric_card("Call Volume", f"{total_call_volume:,}")
                
                with col2:
                    total_put_volume = puts['volume'].sum() if not puts.empty else 0
                    create_metric_card("Put Volume", f"{total_put_volume:,}")
                
                with col3:
                    put_call_ratio = total_put_volume / total_call_volume if total_call_volume > 0 else 0
                    create_metric_card("Put/Call Ratio", f"{put_call_ratio:.2f}")
                
                with col4:
                    avg_iv = ((calls['impliedVolatility'].mean() + puts['impliedVolatility'].mean()) / 2 * 100) if not calls.empty and not puts.empty else 0
                    create_metric_card("Avg IV", f"{avg_iv:.1f}%")
        
        else:
            st.warning(f"No options data available for {symbol}")
    
    elif analysis_mode == "Strategy Builder":
        st.markdown("### Options Strategy Builder")
        
        # Strategy legs input
        if 'strategy_legs' not in st.session_state:
            st.session_state.strategy_legs = []
        
        with st.form("add_leg_form"):
            st.markdown("#### Add Strategy Leg")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                option_type = st.selectbox("Option Type", ["call", "put"])
            with col2:
                position = st.selectbox("Position", ["long", "short"])
            with col3:
                strike = st.number_input("Strike Price", min_value=0.0, value=100.0)
            with col4:
                premium = st.number_input("Premium", min_value=0.0, value=5.0)
            
            quantity = st.number_input("Quantity", min_value=1, value=1)
            
            if st.form_submit_button("Add Leg"):
                st.session_state.strategy_legs.append({
                    'type': option_type,
                    'position': position,
                    'strike': strike,
                    'premium': premium,
                    'quantity': quantity
                })
                st.rerun()
        
        # Display current strategy
        if st.session_state.strategy_legs:
            st.markdown("#### Current Strategy")
            
            strategy_df = pd.DataFrame(st.session_state.strategy_legs)
            strategy_df.index += 1
            st.dataframe(strategy_df, use_container_width=True)
            
            if st.button("Clear Strategy"):
                st.session_state.strategy_legs = []
                st.rerun()
            
            # Generate payoff diagram
            st.markdown("#### Strategy Payoff Diagram")
            payoff_fig = analyzer.create_payoff_diagram(st.session_state.strategy_legs)
            st.plotly_chart(payoff_fig, use_container_width=True)
            
            # Calculate strategy metrics
            strikes = [leg['strike'] for leg in st.session_state.strategy_legs]
            max_profit = "Unlimited" if any(leg['position'] == 'long' and leg['type'] == 'call' 
                                          for leg in st.session_state.strategy_legs) else "Limited"
            max_loss = sum(leg['premium'] * leg['quantity'] * (1 if leg['position'] == 'long' else -1) 
                          for leg in st.session_state.strategy_legs)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                create_metric_card("Max Profit", str(max_profit))
            with col2:
                create_metric_card("Max Loss", f"${max_loss:.2f}")
            with col3:
                create_metric_card("Net Premium", f"${abs(max_loss):.2f}")
    
    elif analysis_mode == "Volatility Analysis":
        st.markdown("### Implied Volatility Analysis")
        
        with st.spinner("Analyzing volatility surface..."):
            vol_data = analyzer.analyze_volatility_surface(symbol)
        
        if not vol_data.empty:
            # Volatility surface visualization
            fig = px.scatter_3d(
                vol_data, 
                x='days_to_exp', 
                y='moneyness', 
                z='implied_vol',
                color='implied_vol',
                size='volume',
                hover_data=['strike'],
                title=f'{symbol} Implied Volatility Surface'
            )
            
            fig.update_layout(
                scene=dict(
                    xaxis_title="Days to Expiration",
                    yaxis_title="Moneyness (Strike/Spot)",
                    zaxis_title="Implied Volatility (%)",
                    bgcolor='rgba(0,0,0,0)'
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Volatility statistics
            st.markdown("#### Volatility Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_iv = vol_data['implied_vol'].mean()
                create_metric_card("Average IV", f"{avg_iv:.1f}%")
            
            with col2:
                iv_std = vol_data['implied_vol'].std()
                create_metric_card("IV Std Dev", f"{iv_std:.1f}%")
            
            with col3:
                max_iv = vol_data['implied_vol'].max()
                create_metric_card("Max IV", f"{max_iv:.1f}%")
            
            with col4:
                min_iv = vol_data['implied_vol'].min()
                create_metric_card("Min IV", f"{min_iv:.1f}%")
            
            # Volatility term structure
            term_structure = vol_data.groupby('days_to_exp')['implied_vol'].mean().reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=term_structure['days_to_exp'],
                y=term_structure['implied_vol'],
                mode='lines+markers',
                name='IV Term Structure',
                line=dict(color='#FF6B6B', width=2)
            ))
            
            fig.update_layout(
                title='Implied Volatility Term Structure',
                xaxis_title='Days to Expiration',
                yaxis_title='Implied Volatility (%)',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning(f"No volatility data available for {symbol}")
    
    elif analysis_mode == "Greeks Calculator":
        st.markdown("### Options Greeks Calculator")
        
        # Input parameters
        col1, col2 = st.columns(2)
        
        with col1:
            current_price = st.number_input("Current Stock Price ($)", min_value=0.0, value=100.0)
            strike_price = st.number_input("Strike Price ($)", min_value=0.0, value=100.0)
            time_to_exp = st.number_input("Days to Expiration", min_value=1, value=30)
            
        with col2:
            volatility = st.slider("Implied Volatility (%)", 1.0, 100.0, 25.0) / 100
            option_type = st.selectbox("Option Type", ["call", "put"])
            risk_free = st.number_input("Risk-Free Rate (%)", min_value=0.0, value=5.0) / 100
        
        # Calculate Greeks
        T = time_to_exp / 365.0
        greeks = analyzer.calculate_greeks(
            current_price, strike_price, T, risk_free, volatility, option_type
        )
        
        # Theoretical option price
        if option_type == 'call':
            theo_price = analyzer.black_scholes_call(
                current_price, strike_price, T, risk_free, volatility
            )
        else:
            theo_price = analyzer.black_scholes_put(
                current_price, strike_price, T, risk_free, volatility
            )
        
        # Display results
        st.markdown("#### Option Pricing & Greeks")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            create_metric_card("Theoretical Price", f"${theo_price:.2f}")
            create_metric_card("Delta", f"{greeks['delta']:.4f}")
            
        with col2:
            create_metric_card("Gamma", f"{greeks['gamma']:.4f}")
            create_metric_card("Theta", f"${greeks['theta']:.2f}")
            
        with col3:
            create_metric_card("Vega", f"${greeks['vega']:.2f}")
            create_metric_card("Rho", f"${greeks['rho']:.2f}")
        
        # Greeks sensitivity analysis
        st.markdown("#### Greeks Sensitivity Analysis")
        
        # Price sensitivity (Delta/Gamma)
        price_range = np.linspace(current_price * 0.8, current_price * 1.2, 50)
        deltas = []
        gammas = []
        
        for price in price_range:
            g = analyzer.calculate_greeks(price, strike_price, T, risk_free, volatility, option_type)
            deltas.append(g['delta'])
            gammas.append(g['gamma'])
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Delta vs Stock Price', 'Gamma vs Stock Price')
        )
        
        fig.add_trace(go.Scatter(
            x=price_range, y=deltas, name='Delta',
            line=dict(color='#4ECDC4')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=price_range, y=gammas, name='Gamma',
            line=dict(color='#FF6B6B')
        ), row=1, col=2)
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Educational content
    with st.expander("📚 Options Trading Guide"):
        st.markdown("""
        **Key Options Concepts:**
        
        **Greeks Explained:**
        - **Delta**: Price sensitivity to underlying stock movement ($1 stock move)
        - **Gamma**: Rate of change of Delta (acceleration of Delta)
        - **Theta**: Time decay (daily value loss due to time)
        - **Vega**: Volatility sensitivity (1% IV change impact)
        - **Rho**: Interest rate sensitivity (1% rate change impact)
        
        **Common Strategies:**
        - **Long Call**: Bullish, unlimited upside, limited downside
        - **Long Put**: Bearish, limited upside, limited downside
        - **Covered Call**: Income generation, limited upside
        - **Protective Put**: Downside protection, insurance strategy
        - **Straddle**: Volatility play, profit from large moves
        - **Iron Condor**: Range-bound, profit from low volatility
        
        **Risk Management:**
        - Never risk more than you can afford to lose
        - Understand maximum loss before entering trades
        - Consider time decay (Theta) in your positions
        - Monitor implied volatility changes
        - Use position sizing and diversification
        """)

if __name__ == "__main__":
    main()