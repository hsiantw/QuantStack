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
from utils.time_series_analysis import TimeSeriesAnalysis
from utils.risk_metrics import RiskMetrics

# Page configuration
st.set_page_config(
    page_title="Time Series Analysis",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Time Series Analysis")
st.markdown("Advanced time series analysis with trend identification, seasonality detection, and forecasting")

# Sidebar for inputs
st.sidebar.header("Analysis Configuration")

# Asset selection
ticker_input = st.sidebar.text_input(
    "Enter Stock Ticker",
    value="AAPL",
    help="Enter a valid stock ticker symbol"
).upper()

# Popular tickers for quick selection
popular_tickers = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "TSLA", "NFLX", "ADBE"],
    "Finance": ["JPM", "BAC", "WFC", "GS", "MS", "C", "V", "MA"],
    "Healthcare": ["JNJ", "PFE", "UNH", "ABBV", "MRK", "BMY", "GILD", "CVS"],
    "Energy": ["XOM", "CVX", "COP", "EOG", "SLB", "PSX", "VLO", "MPC"],
    "ETFs": ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "VTI"]
}

selected_category = st.sidebar.selectbox("Or select from popular tickers", [""] + list(popular_tickers.keys()))

if selected_category:
    selected_ticker = st.sidebar.selectbox(
        f"Select from {selected_category}",
        popular_tickers[selected_category]
    )
    if st.sidebar.button(f"Analyze {selected_ticker}"):
        ticker_input = selected_ticker

# Time period selection
period_options = {
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y", 
    "3 Years": "3y",
    "5 Years": "5y"
}

selected_period = st.sidebar.selectbox(
    "Select Time Period",
    list(period_options.keys()),
    index=2
)

# Analysis options
with st.sidebar.expander("Analysis Options"):
    include_volume = st.checkbox("Include Volume Analysis", value=True)
    
    forecast_days = st.slider(
        "Forecast Days",
        min_value=5,
        max_value=90,
        value=30,
        step=5,
        help="Number of days to forecast"
    )
    
    seasonal_period = st.selectbox(
        "Seasonal Period",
        [21, 63, 126, 252],
        index=3,
        help="Period for seasonal decomposition (days)"
    )
    
    trend_window = st.slider(
        "Trend Analysis Window",
        min_value=10,
        max_value=100,
        value=30,
        step=5,
        help="Window for moving average trend analysis"
    )

if not ticker_input:
    st.warning("Please enter a stock ticker symbol.")
    st.stop()

# Data fetching and validation
with st.spinner(f"Fetching data for {ticker_input}..."):
    try:
        # Validate ticker
        valid_tickers, invalid_tickers = DataFetcher.validate_tickers([ticker_input])
        
        if invalid_tickers:
            st.error(f"Invalid ticker: {ticker_input}")
            st.stop()
        
        # Fetch data
        period = period_options[selected_period]
        price_data = DataFetcher.get_stock_data(ticker_input, period=period)
        
        if price_data.empty:
            st.error(f"Unable to fetch data for {ticker_input}")
            st.stop()
        
        # Get stock info
        stock_info = DataFetcher.get_stock_info(ticker_input)
        
        # Prepare data for analysis
        if isinstance(price_data.columns, pd.MultiIndex):
            close_prices = price_data['Close']
            if include_volume and 'Volume' in price_data.columns.levels[0]:
                volume_data = price_data['Volume']
            else:
                volume_data = None
                include_volume = False
        else:
            close_prices = price_data['Close'] if 'Close' in price_data.columns else price_data
            volume_data = price_data['Volume'] if 'Volume' in price_data.columns else None
            if volume_data is None:
                include_volume = False
        
        # Initialize time series analyzer
        ts_analyzer = TimeSeriesAnalysis(close_prices, ticker_input)
        
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        st.stop()

# Display stock information
if stock_info:
    st.header(f"📊 {ticker_input} - {stock_info.get('longName', 'N/A')}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = close_prices.iloc[-1] if not close_prices.empty else 0
        st.metric("Current Price", f"${current_price:.2f}")
    
    with col2:
        market_cap = stock_info.get('marketCap', 0)
        if market_cap:
            st.metric("Market Cap", f"${market_cap/1e9:.1f}B")
        else:
            st.metric("Market Cap", "N/A")
    
    with col3:
        sector = stock_info.get('sector', 'N/A')
        st.metric("Sector", sector)
    
    with col4:
        pe_ratio = stock_info.get('trailingPE', 0)
        if pe_ratio:
            st.metric("P/E Ratio", f"{pe_ratio:.1f}")
        else:
            st.metric("P/E Ratio", "N/A")

# Stationarity Analysis
st.header("📊 Stationarity Analysis")

try:
    stationarity_results = ts_analyzer.stationarity_test()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Price Series")
        price_results = stationarity_results['prices']
        
        if price_results['is_stationary']:
            st.success("✅ Price series is stationary")
        else:
            st.error("❌ Price series is non-stationary")
        
        st.write(f"**ADF Statistic:** {price_results['adf_statistic']:.4f}")
        st.write(f"**P-Value:** {price_results['p_value']:.4f}")
        
        # Critical values
        for level, value in price_results['critical_values'].items():
            st.write(f"**Critical Value ({level}):** {value:.4f}")
    
    with col2:
        st.subheader("Returns Series")
        returns_results = stationarity_results['returns']
        
        if returns_results['is_stationary']:
            st.success("✅ Returns series is stationary")
        else:
            st.warning("⚠️ Returns series is non-stationary")
        
        st.write(f"**ADF Statistic:** {returns_results['adf_statistic']:.4f}")
        st.write(f"**P-Value:** {returns_results['p_value']:.4f}")
        
        # Critical values
        for level, value in returns_results['critical_values'].items():
            st.write(f"**Critical Value ({level}):** {value:.4f}")

except Exception as e:
    st.error(f"Error in stationarity analysis: {str(e)}")

# Trend Analysis
st.header("📈 Trend Analysis")

try:
    trend_data = ts_analyzer.detect_trends(window=trend_window)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Trend Direction", trend_data['trend_direction'])
    
    with col2:
        st.metric("Trend Strength", f"{trend_data['trend_strength']:.3f}")
    
    with col3:
        st.metric("R-Squared", f"{trend_data['r_squared']:.3f}")
    
    # Trend significance
    if trend_data['p_value'] < 0.05:
        st.success(f"✅ Trend is statistically significant (p-value: {trend_data['p_value']:.4f})")
    else:
        st.warning(f"⚠️ Trend is not statistically significant (p-value: {trend_data['p_value']:.4f})")

except Exception as e:
    st.error(f"Error in trend analysis: {str(e)}")

# Comprehensive Analysis Plots
st.header("📊 Comprehensive Analysis")

try:
    analysis_plots = ts_analyzer.plot_comprehensive_analysis()
    
    # Price and trend analysis
    if 'price_trend' in analysis_plots:
        st.subheader("Price and Trend Analysis")
        st.plotly_chart(analysis_plots['price_trend'], use_container_width=True)
    
    # Seasonal decomposition
    if 'decomposition' in analysis_plots:
        st.subheader("Seasonal Decomposition")
        st.plotly_chart(analysis_plots['decomposition'], use_container_width=True)
    
    # Volatility analysis
    col1, col2 = st.columns(2)
    
    with col1:
        if 'volatility' in analysis_plots:
            st.subheader("Volatility Analysis")
            st.plotly_chart(analysis_plots['volatility'], use_container_width=True)
    
    with col2:
        if 'returns_distribution' in analysis_plots:
            st.subheader("Returns Distribution")
            st.plotly_chart(analysis_plots['returns_distribution'], use_container_width=True)
    
    # Forecast
    if 'forecast' in analysis_plots:
        st.subheader("ARIMA Price Forecast")
        st.plotly_chart(analysis_plots['forecast'], use_container_width=True)

except Exception as e:
    st.error(f"Error generating analysis plots: {str(e)}")

# Volatility Analysis
st.header("⚡ Volatility Analysis")

try:
    volatility_data = ts_analyzer.volatility_analysis()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Current Volatility Metrics")
        current_vol = volatility_data['current_volatility']
        st.metric("Current Volatility (Annualized)", f"{current_vol:.2%}")
        
        # Volatility ranking
        vol_percentile = 50  # Simplified for demo
        if current_vol > 0.3:
            st.error("🔴 High Volatility")
        elif current_vol > 0.2:
            st.warning("🟡 Moderate Volatility")
        else:
            st.success("🟢 Low Volatility")
    
    with col2:
        st.subheader("Volatility Clustering")
        clustering_score = volatility_data['volatility_clustering'].mean()
        st.metric("Clustering Score", f"{clustering_score:.3f}")
        
        if clustering_score > 0.1:
            st.info("📊 Evidence of volatility clustering detected")
        else:
            st.info("📊 Little evidence of volatility clustering")

except Exception as e:
    st.error(f"Error in volatility analysis: {str(e)}")

# ARIMA Forecasting
st.header("🔮 ARIMA Forecasting")

try:
    with st.spinner("Generating ARIMA forecast..."):
        forecast_data = ts_analyzer.arima_forecast(steps=forecast_days)
        
        if forecast_data:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Model Performance")
                st.write(f"**AIC:** {forecast_data['aic']:.2f}")
                st.write(f"**BIC:** {forecast_data['bic']:.2f}")
                
                # Forecast summary
                forecast_prices = forecast_data['forecast_prices']
                current_price = close_prices.iloc[-1]
                forecast_return = (forecast_prices.iloc[-1] - current_price) / current_price
                
                st.metric(
                    f"{forecast_days}-Day Forecast Return",
                    f"{forecast_return:.2%}",
                    delta=f"Target: ${forecast_prices.iloc[-1]:.2f}"
                )
            
            with col2:
                st.subheader("Forecast Confidence")
                
                # Simple confidence bands (simplified)
                forecast_std = forecast_data['forecast_returns'].std()
                confidence_level = 95
                
                st.write(f"**Forecast Volatility:** {forecast_std:.4f}")
                st.write(f"**{confidence_level}% Confidence Interval:**")
                
                upper_bound = forecast_prices.iloc[-1] * (1 + 1.96 * forecast_std)
                lower_bound = forecast_prices.iloc[-1] * (1 - 1.96 * forecast_std)
                
                st.write(f"Lower: ${lower_bound:.2f}")
                st.write(f"Upper: ${upper_bound:.2f}")
        else:
            st.warning("Unable to generate ARIMA forecast. Try a different time period.")

except Exception as e:
    st.error(f"Error in ARIMA forecasting: {str(e)}")

# Support and Resistance Analysis
st.header("📊 Support and Resistance Levels")

try:
    support_resistance = ts_analyzer.support_resistance_levels()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Support Levels")
        support_levels = support_resistance['support_levels']
        
        if support_levels:
            support_df = pd.DataFrame(support_levels)
            support_df['level'] = support_df['level'].round(2)
            support_df = support_df.sort_values('level', ascending=False).head(5)
            
            for _, level in support_df.iterrows():
                distance = (current_price - level['level']) / current_price * 100
                st.write(f"**${level['level']:.2f}** ({distance:+.1f}%) - Strength: {level['strength']}")
        else:
            st.info("No significant support levels detected")
    
    with col2:
        st.subheader("Resistance Levels")
        resistance_levels = support_resistance['resistance_levels']
        
        if resistance_levels:
            resistance_df = pd.DataFrame(resistance_levels)
            resistance_df['level'] = resistance_df['level'].round(2)
            resistance_df = resistance_df.sort_values('level').head(5)
            
            for _, level in resistance_df.iterrows():
                distance = (level['level'] - current_price) / current_price * 100
                st.write(f"**${level['level']:.2f}** (+{distance:.1f}%) - Strength: {level['strength']}")
        else:
            st.info("No significant resistance levels detected")

except Exception as e:
    st.error(f"Error in support/resistance analysis: {str(e)}")

# Risk Analysis
st.header("⚠️ Risk Analysis")

try:
    returns_data = close_prices.pct_change().dropna()
    risk_analyzer = RiskMetrics(returns_data)
    risk_summary = risk_analyzer.get_risk_summary()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Value at Risk")
        st.metric("VaR (95%)", f"{risk_summary['VaR_95%']:.2%}")
        st.metric("VaR (99%)", f"{risk_summary['VaR_99%']:.2%}")
    
    with col2:
        st.subheader("Risk-Adjusted Returns")
        st.metric("Sharpe Ratio", f"{risk_summary['Sharpe_Ratio']:.3f}")
        st.metric("Sortino Ratio", f"{risk_summary['Sortino_Ratio']:.3f}")
    
    with col3:
        st.subheader("Tail Risk")
        st.metric("Skewness", f"{risk_summary['Skewness']:.3f}")
        st.metric("Excess Kurtosis", f"{risk_summary['Excess_Kurtosis']:.3f}")

except Exception as e:
    st.error(f"Error in risk analysis: {str(e)}")

# Analysis Summary
st.header("📋 Analysis Summary")

try:
    analysis_summary = ts_analyzer.get_analysis_summary()
    
    # Create summary table
    summary_data = [
        {"Metric": "Current Price", "Value": f"${analysis_summary['price_statistics']['current']:.2f}"},
        {"Metric": "Price Trend", "Value": analysis_summary['trend_analysis']['direction']},
        {"Metric": "Trend Strength", "Value": f"{analysis_summary['trend_analysis']['strength']:.3f}"},
        {"Metric": "Volatility", "Value": f"{analysis_summary['volatility']['current']:.2%}"},
        {"Metric": "Prices Stationary", "Value": "Yes" if stationarity_results['prices']['is_stationary'] else "No"},
        {"Metric": "Returns Stationary", "Value": "Yes" if stationarity_results['returns']['is_stationary'] else "No"},
    ]
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Trading recommendations
    st.subheader("💡 Trading Insights")
    
    # Simple momentum signal
    trend_direction = analysis_summary['trend_analysis']['direction']
    trend_strength = analysis_summary['trend_analysis']['strength']
    current_volatility = analysis_summary['volatility']['current']
    
    if trend_direction == "Upward" and trend_strength > 0.5:
        st.success("🟢 **Bullish Signal**: Strong upward trend detected")
    elif trend_direction == "Downward" and trend_strength > 0.5:
        st.error("🔴 **Bearish Signal**: Strong downward trend detected")
    else:
        st.info("⚪ **Neutral Signal**: No strong directional trend")
    
    if current_volatility > 0.3:
        st.warning("⚠️ **High Volatility**: Consider risk management")
    
except Exception as e:
    st.error(f"Error generating analysis summary: {str(e)}")

# Export functionality
st.header("📥 Export Results")

col1, col2 = st.columns(2)

with col1:
    if st.button("📊 Export Analysis Summary", use_container_width=True):
        try:
            # Prepare comprehensive data for export
            export_data = {
                'Ticker': ticker_input,
                'Analysis_Date': datetime.now().strftime('%Y-%m-%d'),
                'Current_Price': close_prices.iloc[-1],
                'Trend_Direction': analysis_summary['trend_analysis']['direction'],
                'Trend_Strength': analysis_summary['trend_analysis']['strength'],
                'Volatility': analysis_summary['volatility']['current'],
                'Sharpe_Ratio': risk_summary['Sharpe_Ratio'],
                'VaR_95': risk_summary['VaR_95%'],
                'Max_Drawdown': risk_summary['Max_Drawdown']
            }
            
            export_df = pd.DataFrame([export_data])
            csv = export_df.to_csv(index=False)
            
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"{ticker_input}_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error preparing export: {str(e)}")

with col2:
    if st.button("📈 Export Price Data", use_container_width=True):
        try:
            price_export = pd.DataFrame({
                'Date': close_prices.index,
                'Close_Price': close_prices.values,
                'Returns': returns_data.values
            })
            
            csv = price_export.to_csv(index=False)
            
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"{ticker_input}_prices_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error preparing price export: {str(e)}")

# Navigation
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🏠 Back to Dashboard", use_container_width=True):
        st.switch_page("app.py")

with col2:
    if st.button("📊 Portfolio Optimization", use_container_width=True):
        st.switch_page("pages/portfolio_optimization.py")

with col3:
    if st.button("⚡ Trading Strategies", use_container_width=True):
        st.switch_page("pages/trading_strategies.py")
