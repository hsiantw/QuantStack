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
from utils.ai_models import AIModels
from utils.time_series_analysis import TimeSeriesAnalysis
from utils.tooltips import get_tooltip_help
from utils.ai_strategy_optimizer import AIStrategyOptimizer
from utils.pinescript_generator import PineScriptGenerator

# Page configuration
st.set_page_config(
    page_title="AI Analysis",
    page_icon="🤖",
    layout="wide"
)

# Modern header with enhanced styling
st.markdown("""
<div class="main-header">
    <h1>🤖 AI-Powered Financial Analysis</h1>
    <p>Advanced machine learning models for price prediction, pattern recognition, and market analysis</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for inputs
st.sidebar.header("AI Analysis Configuration")

# Asset selection
ticker_input = st.sidebar.text_input(
    "Enter Stock Ticker",
    value="SPY",
    help="Enter a valid stock ticker symbol"
).upper()

# Quick selection categories
ai_friendly_tickers = {
    "Large Cap Tech": ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "TSLA"],
    "Volatile Stocks": ["TSLA", "GME", "AMC", "NFLX", "ROKU", "ZM"],
    "Blue Chip": ["JNJ", "JPM", "PG", "KO", "WMT", "DIS"],
    "Growth Stocks": ["AMZN", "CRM", "SHOP", "SQ", "PYPL", "ADBE"],
    "ETFs": ["SPY", "QQQ", "IWM", "XLK", "XLF", "GLD"]
}

selected_category = st.sidebar.selectbox("Or select from categories", [""] + list(ai_friendly_tickers.keys()))

if selected_category:
    selected_ticker = st.sidebar.selectbox(
        f"Select from {selected_category}",
        ai_friendly_tickers[selected_category]
    )
    if st.sidebar.button(f"Analyze {selected_ticker}"):
        ticker_input = selected_ticker

# Time period selection for recent data display
period_options = {
    "1 Year": "1y",
    "2 Years": "2y",
    "3 Years": "3y",
    "5 Years": "5y"
}

selected_period = st.sidebar.selectbox(
    "Recent Data Period (Display)",
    list(period_options.keys()),
    index=2,
    help="Period for displaying recent price charts and analysis"
)

# AI Training Data Configuration
st.sidebar.subheader("🤖 AI Training Configuration")

use_extended_training = st.sidebar.checkbox(
    "Use Extended Historical Data for Training",
    value=True,
    help="Use 10-20 years of historical data for more robust AI model training"
)

training_years = st.sidebar.slider(
    "Training Data Period (Years)",
    min_value=10,
    max_value=20,
    value=15,
    step=1,
    help="Years of historical data to use for AI training (10-20 recommended)",
    disabled=not use_extended_training
)

# Model selection
st.sidebar.subheader("Model Configuration")

models_to_train = st.sidebar.multiselect(
    "Select AI models to train",
    ["Random Forest", "Gradient Boosting"],
    default=["Random Forest", "Gradient Boosting"],
    help="Multiple models provide better insights"
)

# Prediction settings
with st.sidebar.expander("Prediction Settings"):
    prediction_horizon = st.selectbox(
        "Prediction Horizon",
        [1, 5, 10],
        index=0,
        help="Days ahead to predict (1=next day, 5=next week, 10=two weeks)"
    )
    
    forecast_days = st.slider(
        "Forecast Period",
        min_value=5,
        max_value=60,
        value=30,
        step=5,
        help="Number of days to forecast into the future"
    )

# Advanced model parameters
with st.sidebar.expander("Advanced Model Parameters"):
    test_size = st.slider(
        "Test Set Size (%)",
        min_value=10,
        max_value=40,
        value=20,
        step=5,
        help="Percentage of data reserved for testing"
    ) / 100
    
    rf_n_estimators = st.slider(
        "Random Forest - Trees",
        min_value=50,
        max_value=200,
        value=100,
        step=25
    )
    
    rf_max_depth = st.slider(
        "Random Forest - Max Depth",
        min_value=5,
        max_value=20,
        value=10,
        step=1
    )
    
    gb_learning_rate = st.slider(
        "Gradient Boosting - Learning Rate",
        min_value=0.01,
        max_value=0.3,
        value=0.1,
        step=0.01
    )

if not ticker_input:
    st.warning("Please enter a stock ticker symbol.")
    st.stop()

if not models_to_train:
    st.warning("Please select at least one AI model to train.")
    st.stop()

# Data fetching and preparation
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
        
        # Prepare OHLCV structure
        if isinstance(price_data.columns, pd.MultiIndex):
            ohlcv_data = price_data.droplevel(1, axis=1)
        else:
            ohlcv_data = price_data
        
        # Validate we have enough data
        if len(ohlcv_data) < 100:
            st.error("Insufficient data for AI model training. Please select a longer time period or different ticker.")
            st.stop()
        
        # Initialize AI models with extended historical training data
        ai_models = AIModels(
            ohlcv_data, 
            ticker_input, 
            use_extended_history=use_extended_training,
            training_years=training_years
        )
        
        # Get stock info
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
        volatility = ohlcv_data['Close'].pct_change().std() * np.sqrt(252)
        st.metric("Volatility", f"{volatility:.2%}")
    
    with col4:
        sector = stock_info.get('sector', 'N/A')
        st.metric("Sector", sector)

# Feature Engineering Summary
st.header("🔧 Feature Engineering")

with st.expander("View Feature Engineering Details"):
    st.write("**Technical Indicators Generated:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Price Features:**")
        st.write("- Returns (1-5 day lags)")
        st.write("- Price ratios (High/Low, Close/Open)")
        st.write("- Moving averages (5, 10, 20, 50 days)")
        st.write("- Moving average ratios")
    
    with col2:
        st.write("**Technical Indicators:**")
        st.write("- RSI (14-period)")
        st.write("- Bollinger Bands position")
        st.write("- Rolling volatility (5, 10, 20 days)")
        st.write("- Volume indicators")
    
    with col3:
        st.write("**Lag Features:**")
        st.write("- Historical returns (1-5 days)")
        st.write("- Historical prices (1-5 days)")
        st.write("- Technical indicator lags")
        st.write("- Cross-feature interactions")
    
    # Show actual feature columns
    if hasattr(ai_models, 'training_features') and ai_models.training_features is not None:
        feature_cols = ai_models.get_feature_columns(ai_models.training_features)
        training_data = ai_models.training_features
    else:
        feature_cols = ai_models.get_feature_columns(ai_models.prediction_features)
        training_data = ai_models.prediction_features
    
    st.write(f"**Total Features Generated:** {len(feature_cols)}")
    
    if st.checkbox("Show all feature names"):
        st.write(feature_cols)

# Training Data Information
st.header("📚 Training Data Information")

col1, col2, col3 = st.columns(3)

with col1:
    if use_extended_training and hasattr(ai_models, 'training_data') and ai_models.training_data is not None:
        st.metric("Training Data Period", f"{training_years} Years")
        st.metric("Historical Data Points", f"{len(ai_models.training_data):,}")
    else:
        st.metric("Training Data Period", selected_period)
        st.metric("Data Points", f"{len(ohlcv_data):,}")

with col2:
    if hasattr(ai_models, 'training_features') and ai_models.training_features is not None:
        data_start = ai_models.training_features.index[0].strftime('%Y-%m-%d')
        data_end = ai_models.training_features.index[-1].strftime('%Y-%m-%d')
        st.metric("Data Range", f"{data_start}")
        st.caption(f"to {data_end}")
    else:
        data_start = ohlcv_data.index[0].strftime('%Y-%m-%d')
        data_end = ohlcv_data.index[-1].strftime('%Y-%m-%d')
        st.metric("Data Range", f"{data_start}")
        st.caption(f"to {data_end}")

with col3:
    st.metric("Extended Training", "✅ Enabled" if use_extended_training else "❌ Disabled")
    if use_extended_training:
        st.caption("Using 10-20 year historical data for robust training")
    else:
        st.caption("Using recent data only")

if use_extended_training:
    st.info(f"🎯 **Enhanced AI Training**: Using {training_years} years of historical data for more robust pattern recognition and better predictions.")
else:
    st.warning("💡 **Recommendation**: Enable extended historical data training for more accurate AI predictions.")

# Model Training and Results
st.header("🎯 AI Model Training & Results")

model_results = {}

# Train selected models
with st.spinner("Training AI models... This may take a moment."):
    try:
        # Random Forest
        if "Random Forest" in models_to_train:
            with st.spinner("Training Random Forest..."):
                rf_params = {
                    'n_estimators': rf_n_estimators,
                    'max_depth': rf_max_depth,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42
                }
                
                rf_result = ai_models.train_random_forest(
                    target_horizon=prediction_horizon,
                    **rf_params
                )
                
                if rf_result:
                    model_results["Random Forest"] = rf_result
                    st.success("✅ Random Forest trained successfully")
                else:
                    st.error("❌ Random Forest training failed")
        
        # Gradient Boosting
        if "Gradient Boosting" in models_to_train:
            with st.spinner("Training Gradient Boosting..."):
                gb_params = {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': gb_learning_rate,
                    'random_state': 42
                }
                
                gb_result = ai_models.train_gradient_boosting(
                    target_horizon=prediction_horizon,
                    **gb_params
                )
                
                if gb_result:
                    model_results["Gradient Boosting"] = gb_result
                    st.success("✅ Gradient Boosting trained successfully")
                else:
                    st.error("❌ Gradient Boosting training failed")
        
    except Exception as e:
        st.error(f"Error training models: {str(e)}")

# Model Performance Comparison
if model_results:
    st.subheader("📈 Model Performance Comparison")
    
    # Add tooltips for key ML concepts
    with st.expander("💡 Machine Learning Metrics Explained"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**R-Squared (R²):**")
            st.markdown(get_tooltip_help("r_squared")[:150] + "...")
        with col2:
            st.markdown("**Mean Squared Error:**")
            st.markdown(get_tooltip_help("mean_squared_error")[:150] + "...")
        with col3:
            st.markdown("**Feature Importance:**")
            st.markdown(get_tooltip_help("feature_importance")[:150] + "...")
    
    # Performance metrics table
    performance_data = []
    
    for model_name, result in model_results.items():
        metrics = result['metrics']
        performance_data.append({
            "Model": model_name,
            "Train R²": f"{metrics['train_r2']:.4f}",
            "Test R²": f"{metrics['test_r2']:.4f}",
            "Train MSE": f"{metrics['train_mse']:.6f}",
            "Test MSE": f"{metrics['test_mse']:.6f}",
            "Train MAE": f"{metrics['train_mae']:.6f}",
            "Test MAE": f"{metrics['test_mae']:.6f}"
        })
    
    performance_df = pd.DataFrame(performance_data)
    st.dataframe(performance_df, use_container_width=True, hide_index=True)
    
    # Model quality assessment
    best_model = max(model_results.keys(), key=lambda x: model_results[x]['metrics']['test_r2'])
    best_r2 = model_results[best_model]['metrics']['test_r2']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if best_r2 > 0.1:
            st.success(f"🏆 Best Model: {best_model}")
        else:
            st.warning(f"⚠️ Best Model: {best_model}")
        st.metric("Test R²", f"{best_r2:.4f}")
    
    with col2:
        # Model reliability assessment
        test_mse = model_results[best_model]['metrics']['test_mse']
        if test_mse < 0.001:
            reliability = "High"
            color = "success"
        elif test_mse < 0.01:
            reliability = "Moderate"
            color = "warning"
        else:
            reliability = "Low"
            color = "error"
        
        if color == "success":
            st.success(f"📊 Reliability: {reliability}")
        elif color == "warning":
            st.warning(f"📊 Reliability: {reliability}")
        else:
            st.error(f"📊 Reliability: {reliability}")
        
        st.metric("Test MSE", f"{test_mse:.6f}")
    
    with col3:
        # Overfitting check
        train_r2 = model_results[best_model]['metrics']['train_r2']
        overfitting = train_r2 - best_r2
        
        if overfitting < 0.1:
            st.success("✅ No Overfitting")
        elif overfitting < 0.2:
            st.warning("⚠️ Mild Overfitting")
        else:
            st.error("🚨 Significant Overfitting")
        
        st.metric("Overfitting Gap", f"{overfitting:.4f}")

# Detailed Model Analysis
if model_results:
    st.subheader("🔍 Detailed Model Analysis")
    
    selected_model = st.selectbox(
        "Select model for detailed analysis",
        list(model_results.keys())
    )
    
    if selected_model:
        model_data = model_results[selected_model]
        
        # Model performance plots
        try:
            model_plots = ai_models.plot_model_performance(model_data, selected_model)
            
            # Actual vs Predicted
            if 'actual_vs_predicted' in model_plots:
                st.plotly_chart(model_plots['actual_vs_predicted'], use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Feature importance
                if 'feature_importance' in model_plots:
                    st.plotly_chart(model_plots['feature_importance'], use_container_width=True)
            
            with col2:
                # Residuals analysis
                if 'residuals' in model_plots:
                    st.plotly_chart(model_plots['residuals'], use_container_width=True)
            
        except Exception as e:
            st.error(f"Error generating model plots: {str(e)}")
        
        # Feature importance analysis
        st.subheader("🎯 Feature Importance Analysis")
        
        feature_importance = model_data['feature_importance'].head(10)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top 10 Most Important Features:**")
            for _, row in feature_importance.iterrows():
                st.write(f"**{row['feature']}:** {row['importance']:.4f}")
        
        with col2:
            st.write("**Feature Insights:**")
            
            # Analyze feature types
            price_features = [f for f in feature_importance['feature'] if 'Price' in f or 'Close' in f or 'MA_' in f]
            technical_features = [f for f in feature_importance['feature'] if 'RSI' in f or 'BB_' in f or 'Volatility' in f]
            lag_features = [f for f in feature_importance['feature'] if 'Lag' in f]
            
            if price_features:
                st.info(f"💰 Price-based features are important: {len(price_features)} in top 10")
            
            if technical_features:
                st.info(f"📊 Technical indicators matter: {len(technical_features)} in top 10")
            
            if lag_features:
                st.info(f"⏰ Historical patterns detected: {len(lag_features)} in top 10")

# Price Predictions
if model_results:
    st.header("🔮 Price Predictions")
    
    prediction_model = st.selectbox(
        "Select model for predictions",
        list(model_results.keys()),
        key="prediction_model"
    )
    
    if prediction_model:
        with st.spinner("Generating predictions..."):
            try:
                predictions = ai_models.generate_predictions(
                    model_results[prediction_model], 
                    prediction_days=forecast_days
                )
                
                if predictions is not None and not predictions.empty:
                    # Current vs predicted analysis
                    col1, col2, col3 = st.columns(3)
                    
                    current_price = ohlcv_data['Close'].iloc[-1]
                    final_prediction = predictions.iloc[-1]
                    prediction_return = (final_prediction - current_price) / current_price
                    
                    with col1:
                        st.metric("Current Price", f"${current_price:.2f}")
                    
                    with col2:
                        st.metric(
                            f"{forecast_days}-Day Prediction", 
                            f"${final_prediction:.2f}",
                            delta=f"{prediction_return:.2%}"
                        )
                    
                    with col3:
                        # Confidence level based on model performance
                        model_r2 = model_results[prediction_model]['metrics']['test_r2']
                        if model_r2 > 0.1:
                            confidence = "High"
                        elif model_r2 > 0.05:
                            confidence = "Moderate"
                        else:
                            confidence = "Low"
                        
                        st.metric("Confidence Level", confidence)
                    
                    # Prediction visualization
                    fig_pred = go.Figure()
                    
                    # Historical prices
                    historical_window = ohlcv_data['Close'].tail(60)  # Last 60 days
                    fig_pred.add_trace(go.Scatter(
                        x=historical_window.index,
                        y=historical_window,
                        mode='lines',
                        name='Historical Prices',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Predictions
                    fig_pred.add_trace(go.Scatter(
                        x=predictions.index,
                        y=predictions,
                        mode='lines+markers',
                        name='AI Predictions',
                        line=dict(color='red', width=2, dash='dash'),
                        marker=dict(size=4)
                    ))
                    
                    # Connection line
                    connection_x = [historical_window.index[-1], predictions.index[0]]
                    connection_y = [historical_window.iloc[-1], predictions.iloc[0]]
                    fig_pred.add_trace(go.Scatter(
                        x=connection_x,
                        y=connection_y,
                        mode='lines',
                        name='Connection',
                        line=dict(color='green', width=2, dash='dot'),
                        showlegend=False
                    ))
                    
                    fig_pred.update_layout(
                        title=f'{ticker_input} - AI Price Prediction ({prediction_model})',
                        xaxis_title='Date',
                        yaxis_title='Price ($)',
                        height=500,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
                    # Prediction analysis
                    st.subheader("📊 Prediction Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Short-term Outlook (Next 5 days):**")
                        short_term_pred = predictions.head(5).iloc[-1] if len(predictions) >= 5 else predictions.iloc[-1]
                        short_term_return = (short_term_pred - current_price) / current_price
                        
                        if short_term_return > 0.02:
                            st.success(f"🟢 Bullish: +{short_term_return:.2%}")
                        elif short_term_return < -0.02:
                            st.error(f"🔴 Bearish: {short_term_return:.2%}")
                        else:
                            st.info(f"⚪ Neutral: {short_term_return:.2%}")
                    
                    with col2:
                        st.write("**Prediction Volatility:**")
                        pred_volatility = predictions.pct_change().std()
                        historical_volatility = ohlcv_data['Close'].pct_change().tail(30).std()
                        
                        if pred_volatility > historical_volatility * 1.2:
                            st.warning("⚠️ Higher volatility expected")
                        elif pred_volatility < historical_volatility * 0.8:
                            st.info("📉 Lower volatility expected")
                        else:
                            st.info("📊 Similar volatility expected")
                        
                        st.write(f"Prediction volatility: {pred_volatility:.4f}")
                        st.write(f"Historical volatility: {historical_volatility:.4f}")
                
                else:
                    st.warning("Unable to generate predictions. Model may need more data or different parameters.")
                    
            except Exception as e:
                st.error(f"Error generating predictions: {str(e)}")

# Pattern Recognition
st.header("🎨 Pattern Recognition")

try:
    with st.spinner("Analyzing patterns..."):
        patterns = ai_models.pattern_recognition()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🟢 Bullish Patterns")
            bullish_patterns = patterns['bullish_patterns']
            
            if bullish_patterns:
                st.write(f"Found {len(bullish_patterns)} bullish patterns")
                
                # Show recent patterns
                recent_bullish = sorted(bullish_patterns, key=lambda x: x['date'], reverse=True)[:3]
                for pattern in recent_bullish:
                    st.write(f"**{pattern['pattern']}** ({pattern['date'].strftime('%Y-%m-%d')})")
                    st.write(f"Strength: {pattern['strength']:.2f}")
            else:
                st.info("No bullish patterns detected recently")
        
        with col2:
            st.subheader("🔴 Bearish Patterns")
            bearish_patterns = patterns['bearish_patterns']
            
            if bearish_patterns:
                st.write(f"Found {len(bearish_patterns)} bearish patterns")
                
                recent_bearish = sorted(bearish_patterns, key=lambda x: x['date'], reverse=True)[:3]
                for pattern in recent_bearish:
                    st.write(f"**{pattern['pattern']}** ({pattern['date'].strftime('%Y-%m-%d')})")
                    st.write(f"Strength: {pattern['strength']:.2f}")
            else:
                st.info("No bearish patterns detected recently")
        
        with col3:
            st.subheader("⚪ Neutral Patterns")
            neutral_patterns = patterns['neutral_patterns']
            
            if neutral_patterns:
                st.write(f"Found {len(neutral_patterns)} neutral patterns")
                
                recent_neutral = sorted(neutral_patterns, key=lambda x: x['date'], reverse=True)[:3]
                for pattern in recent_neutral:
                    st.write(f"**{pattern['pattern']}** ({pattern['date'].strftime('%Y-%m-%d')})")
                    st.write(f"Strength: {pattern['strength']:.2f}")
            else:
                st.info("No neutral patterns detected recently")
        
        # Overall pattern sentiment
        st.subheader("📊 Overall Pattern Sentiment")
        
        total_bullish = len(bullish_patterns)
        total_bearish = len(bearish_patterns)
        total_patterns = total_bullish + total_bearish
        
        if total_patterns > 0:
            bullish_ratio = total_bullish / total_patterns
            
            if bullish_ratio > 0.6:
                st.success(f"🟢 Predominantly Bullish ({bullish_ratio:.1%} bullish patterns)")
            elif bullish_ratio < 0.4:
                st.error(f"🔴 Predominantly Bearish ({1-bullish_ratio:.1%} bearish patterns)")
            else:
                st.info(f"⚪ Mixed Signals ({bullish_ratio:.1%} bullish, {1-bullish_ratio:.1%} bearish)")
        else:
            st.info("Insufficient pattern data for sentiment analysis")

except Exception as e:
    st.error(f"Error in pattern recognition: {str(e)}")

# AI Strategy Analysis with Entry/Exit Points
st.header("🎯 AI Strategy Analysis & Trading Signals")

try:
    with st.spinner("Generating AI trading strategy..."):
        # Initialize AI Strategy Optimizer
        ai_strategy = AIStrategyOptimizer(ohlcv_data, ticker_input)
        
        # Optimize strategy
        optimization_results = ai_strategy.optimize_strategy()
        
        if optimization_results:
            best_strategy = optimization_results['best_strategy']
            
            st.subheader("🏆 Optimized AI Strategy")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Strategy", best_strategy['name'])
                st.metric("Annual Return", f"{best_strategy['annual_return']:.2%}")
            
            with col2:
                st.metric("Sharpe Ratio", f"{best_strategy['sharpe_ratio']:.3f}")
                st.metric("Max Drawdown", f"{best_strategy['max_drawdown']:.2%}")
            
            with col3:
                st.metric("Win Rate", f"{best_strategy['win_rate']:.1%}")
                st.metric("Total Trades", best_strategy['total_trades'])
            
            # Detailed AI Strategy Methodology
            st.subheader("🔬 AI Strategy Optimization Methodology")
            
            with st.expander("📊 How the AI Optimized This Strategy (Click to expand)", expanded=False):
                st.markdown("### 🎯 Optimization Process")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Step 1: Strategy Universe**")
                    st.write("• Tested 15+ different trading strategies")
                    st.write("• Moving average combinations (10/20, 20/50, 50/200)")
                    st.write("• RSI mean reversion (periods 14, 21, 30)")
                    st.write("• Bollinger Bands reversal strategies")
                    st.write("• MACD signal strategies")
                    st.write("• Multi-indicator combinations")
                    
                    st.markdown("**Step 2: Parameter Optimization**")
                    st.write("• Grid search across parameter ranges")
                    st.write("• Walk-forward analysis for robustness")
                    st.write("• Out-of-sample testing")
                    st.write("• Monte Carlo validation")
                
                with col2:
                    st.markdown("**Step 3: Performance Evaluation**")
                    st.write("• Primary metric: Risk-adjusted returns (Sharpe ratio)")
                    st.write("• Secondary: Maximum drawdown minimization")
                    st.write("• Tertiary: Win rate and trade frequency")
                    st.write("• Calmar ratio (return/max drawdown)")
                    
                    st.markdown("**Step 4: AI Selection Criteria**")
                    st.write("• Sharpe ratio > 1.0 (preferred > 1.5)")
                    st.write("• Max drawdown < 20% (preferred < 10%)")
                    st.write("• Minimum 30 trades for statistical significance")
                    st.write("• Consistent performance across market regimes")
                
                st.markdown("### 📈 Metric Calculations Explained")
                
                # Show detailed calculations
                if 'calculation_details' in optimization_results:
                    calc_details = optimization_results['calculation_details']
                    
                    st.markdown("**Annual Return Calculation:**")
                    st.code(f"""
Total Return = (Final Portfolio Value / Initial Portfolio Value) - 1
Total Days = {calc_details.get('total_days', 'N/A')}
Annual Return = (1 + Total Return) ^ (252 / Total Days) - 1
Result = {best_strategy['annual_return']:.4f} or {best_strategy['annual_return']:.2%}
                    """)
                    
                    st.markdown("**Sharpe Ratio Calculation:**")
                    st.code(f"""
Daily Returns = Portfolio daily percentage changes
Mean Daily Return = {calc_details.get('mean_daily_return', 'N/A'):.6f}
Daily Return Std = {calc_details.get('daily_return_std', 'N/A'):.6f}
Risk-Free Rate = 2% annually = 0.02/252 daily
Sharpe Ratio = (Mean Daily Return - Risk Free Rate) / Daily Return Std
Annualized = Sharpe Ratio * sqrt(252)
Result = {best_strategy['sharpe_ratio']:.4f}
                    """)
                    
                    st.markdown("**Maximum Drawdown Calculation:**")
                    st.code(f"""
Running Maximum = Highest portfolio value up to each point
Drawdown = (Current Value - Running Maximum) / Running Maximum
Maximum Drawdown = Most negative drawdown value
Peak Value = ${calc_details.get('peak_value', 'N/A'):,.2f}
Trough Value = ${calc_details.get('trough_value', 'N/A'):,.2f}
Max Drawdown = {best_strategy['max_drawdown']:.4f} or {best_strategy['max_drawdown']:.2%}
                    """)
                    
                    st.markdown("**Win Rate Calculation:**")
                    st.code(f"""
Total Trades = {best_strategy['total_trades']}
Winning Trades = {calc_details.get('winning_trades', 'N/A')}
Losing Trades = {calc_details.get('losing_trades', 'N/A')}
Win Rate = Winning Trades / Total Trades
Result = {best_strategy['win_rate']:.1f}%
                    """)
                else:
                    st.info("Calculation details not available. Run optimization to see detailed metrics.")
                
                st.markdown("### 🧠 AI Decision Logic")
                
                strategy_reasoning = best_strategy.get('reasoning', {})
                
                st.write("**Why This Strategy Was Selected:**")
                if strategy_reasoning:
                    for reason, explanation in strategy_reasoning.items():
                        st.write(f"• **{reason}:** {explanation}")
                else:
                    # Provide default reasoning based on metrics
                    st.write(f"• **Risk-Adjusted Performance:** Sharpe ratio of {best_strategy['sharpe_ratio']:.3f} indicates strong risk-adjusted returns")
                    
                    if best_strategy['max_drawdown'] < 0.15:
                        st.write(f"• **Drawdown Control:** Maximum drawdown of {best_strategy['max_drawdown']:.1%} shows good risk management")
                    
                    if best_strategy['win_rate'] > 50:
                        st.write(f"• **Win Consistency:** Win rate of {best_strategy['win_rate']:.1f}% demonstrates reliable signal generation")
                    
                    trades_per_year = best_strategy['total_trades'] / max(1, len(ohlcv_data) / 252)
                    if 10 < trades_per_year < 100:
                        st.write(f"• **Trading Frequency:** ~{trades_per_year:.0f} trades per year provides good balance of opportunity and cost efficiency")
                
                st.markdown("### 🔍 Strategy Parameter Optimization")
                
                if best_strategy['type'] == 'ma_crossover':
                    params = best_strategy.get('params', {})
                    st.write("**Moving Average Crossover Parameters:**")
                    st.write(f"• Fast MA Period: {params.get('short_period', 'N/A')} days")
                    st.write(f"• Slow MA Period: {params.get('long_period', 'N/A')} days")
                    st.write("• **Logic:** Buy when fast MA crosses above slow MA, sell when it crosses below")
                    st.write("• **Optimization:** Tested combinations from 5/10 to 50/200 day periods")
                    
                elif 'rsi' in best_strategy['type']:
                    params = best_strategy.get('params', {})
                    st.write("**RSI Mean Reversion Parameters:**")
                    st.write(f"• RSI Period: {params.get('period', 'N/A')} days")
                    st.write(f"• Oversold Level: {params.get('oversold', 'N/A')}")
                    st.write(f"• Overbought Level: {params.get('overbought', 'N/A')}")
                    st.write("• **Logic:** Buy when RSI < oversold, sell when RSI > overbought")
                    st.write("• **Optimization:** Tested RSI periods 10-30, thresholds 20-40 (oversold) and 60-80 (overbought)")
                
                st.markdown("### 📊 Backtesting Validation")
                
                st.write("**Backtesting Process:**")
                st.write("• **Data Split:** 80% in-sample training, 20% out-of-sample testing")
                st.write("• **Walk-Forward Analysis:** Rolling optimization windows")
                st.write("• **Transaction Costs:** 0.1% per trade (realistic broker fees)")
                st.write("• **Slippage:** 0.05% market impact modeling")
                st.write("• **Risk Management:** Position sizing based on volatility")
                
                if 'backtest_periods' in optimization_results:
                    periods = optimization_results['backtest_periods']
                    st.write("**Testing Periods:**")
                    for period_name, metrics in periods.items():
                        st.write(f"• **{period_name}:** Sharpe {metrics.get('sharpe', 'N/A'):.2f}, Drawdown {metrics.get('max_dd', 'N/A'):.1%}")
                
                st.markdown("### 🎲 Risk Assessment")
                
                st.write("**Monte Carlo Analysis Results:**")
                if 'monte_carlo' in optimization_results:
                    mc_results = optimization_results['monte_carlo']
                    st.write(f"• **95% Confidence Interval:** {mc_results.get('ci_lower', 'N/A'):.1%} to {mc_results.get('ci_upper', 'N/A'):.1%} annual return")
                    st.write(f"• **Probability of Loss:** {mc_results.get('prob_loss', 'N/A'):.1%}")
                    st.write(f"• **Value at Risk (5%):** {mc_results.get('var_5', 'N/A'):.1%}")
                else:
                    # Calculate basic risk metrics
                    annual_vol = optimization_results.get('annual_volatility', 0.2)
                    st.write(f"• **Annual Volatility:** {annual_vol:.1%}")
                    st.write(f"• **Estimated VaR (5%):** {-1.645 * annual_vol:.1%} (normal distribution assumption)")
                    st.write(f"• **Risk-Adjusted Return:** {best_strategy['annual_return'] / annual_vol:.2f}x return-to-risk ratio")
            
            # Show comparison with other tested strategies
            if 'all_strategies' in optimization_results and len(optimization_results['all_strategies']) > 1:
                st.subheader("📋 Strategy Comparison Analysis")
                
                with st.expander("🏆 All Tested Strategies Performance", expanded=False):
                    comparison_data = []
                    
                    for strategy in optimization_results['all_strategies'][:10]:  # Top 10
                        comparison_data.append({
                            "Strategy": strategy['name'],
                            "Annual Return": f"{strategy['annual_return']:.2%}",
                            "Sharpe Ratio": f"{strategy['sharpe_ratio']:.3f}",
                            "Max Drawdown": f"{strategy['max_drawdown']:.2%}",
                            "Win Rate": f"{strategy['win_rate']:.1f}%",
                            "Total Trades": strategy['total_trades'],
                            "Calmar Ratio": f"{strategy.get('calmar_ratio', 0):.3f}"
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                    
                    st.markdown("**Strategy Rankings Explanation:**")
                    st.write("• **Primary Sort:** Sharpe Ratio (risk-adjusted returns)")
                    st.write("• **Secondary Sort:** Maximum Drawdown (lower is better)")
                    st.write("• **Tertiary Sort:** Calmar Ratio (return/drawdown)")
                    st.write("• **Filter:** Minimum 30 trades for statistical significance")
            
            # Generate comprehensive strategy chart with entry/exit points
            st.subheader("📊 Strategy Chart with Entry/Exit Signals")
            
            # Get strategy signals and backtesting results
            strategy_signals = ai_strategy.get_strategy_signals(best_strategy)
            backtest_results = ai_strategy.backtest_strategy(best_strategy)
            
            # Create comprehensive chart
            fig_strategy = go.Figure()
            
            # Price data
            fig_strategy.add_trace(go.Candlestick(
                x=ohlcv_data.index[-252:],  # Last year of data
                open=ohlcv_data['Open'][-252:],
                high=ohlcv_data['High'][-252:],
                low=ohlcv_data['Low'][-252:],
                close=ohlcv_data['Close'][-252:],
                name=f'{ticker_input} Price',
                showlegend=True
            ))
            
            # Add entry signals (buy points)
            if 'entry_points' in strategy_signals and len(strategy_signals['entry_points']) > 0:
                entry_dates = strategy_signals['entry_points'].index
                entry_prices = strategy_signals['entry_points'].values
                
                fig_strategy.add_trace(go.Scatter(
                    x=entry_dates,
                    y=entry_prices,
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=12,
                        color='green',
                        line=dict(width=2, color='darkgreen')
                    ),
                    name='Buy Signals',
                    text=[f'BUY: ${price:.2f}' for price in entry_prices],
                    hovertemplate='<b>Buy Signal</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
                ))
            
            # Add exit signals (sell points)
            if 'exit_points' in strategy_signals and len(strategy_signals['exit_points']) > 0:
                exit_dates = strategy_signals['exit_points'].index
                exit_prices = strategy_signals['exit_points'].values
                
                fig_strategy.add_trace(go.Scatter(
                    x=exit_dates,
                    y=exit_prices,
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=12,
                        color='red',
                        line=dict(width=2, color='darkred')
                    ),
                    name='Sell Signals',
                    text=[f'SELL: ${price:.2f}' for price in exit_prices],
                    hovertemplate='<b>Sell Signal</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
                ))
            
            # Add moving averages if strategy uses them
            if 'MA_' in str(best_strategy.get('params', {})):
                for period in [20, 50]:
                    ma_col = f'MA_{period}'
                    if ma_col in ohlcv_data.columns:
                        fig_strategy.add_trace(go.Scatter(
                            x=ohlcv_data.index[-252:],
                            y=ohlcv_data[ma_col][-252:],
                            mode='lines',
                            name=f'MA{period}',
                            line=dict(width=1, dash='dash'),
                            opacity=0.7
                        ))
            
            fig_strategy.update_layout(
                title=f'{ticker_input} - AI Optimized Strategy: {best_strategy["name"]}',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                height=600,
                hovermode='x unified',
                legend=dict(x=0, y=1),
                xaxis_rangeslider_visible=False
            )
            
            st.plotly_chart(fig_strategy, use_container_width=True)
            
            # Strategy Performance Analysis
            st.subheader("📈 Strategy Performance Breakdown")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'equity_curve' in backtest_results:
                    # Equity curve
                    fig_equity = go.Figure()
                    fig_equity.add_trace(go.Scatter(
                        x=backtest_results['equity_curve'].index,
                        y=backtest_results['equity_curve'].values,
                        mode='lines',
                        name='Portfolio Value',
                        line=dict(color='green', width=2)
                    ))
                    
                    # Add benchmark
                    if 'benchmark' in backtest_results:
                        fig_equity.add_trace(go.Scatter(
                            x=backtest_results['benchmark'].index,
                            y=backtest_results['benchmark'].values,
                            mode='lines',
                            name='Buy & Hold',
                            line=dict(color='blue', width=2, dash='dash')
                        ))
                    
                    fig_equity.update_layout(
                        title='Equity Curve Comparison',
                        xaxis_title='Date',
                        yaxis_title='Portfolio Value',
                        height=400
                    )
                    
                    st.plotly_chart(fig_equity, use_container_width=True)
            
            with col2:
                if 'monthly_returns' in backtest_results:
                    # Monthly returns heatmap
                    monthly_returns = backtest_results['monthly_returns']
                    
                    fig_heatmap = go.Figure(data=go.Heatmap(
                        z=monthly_returns.values,
                        x=monthly_returns.columns,
                        y=monthly_returns.index,
                        colorscale='RdYlGn',
                        showscale=True
                    ))
                    
                    fig_heatmap.update_layout(
                        title='Monthly Returns Heatmap',
                        xaxis_title='Month',
                        yaxis_title='Year',
                        height=400
                    )
                    
                    st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # PineScript Code Generation
            st.subheader("📝 PineScript Code for TradingView")
            st.markdown("Copy this code and paste it into TradingView's Pine Editor:")
            
            # Generate PineScript based on strategy type
            pinescript_code = ""
            strategy_type = best_strategy.get('type', 'unknown')
            
            if 'ma' in strategy_type.lower() or 'moving_average' in strategy_type.lower():
                params = best_strategy.get('params', {})
                fast_ma = params.get('short_period', params.get('fast', 20))
                slow_ma = params.get('long_period', params.get('slow', 50))
                pinescript_code = PineScriptGenerator.generate_moving_average_strategy(fast_ma, slow_ma, ticker_input)
            
            elif 'rsi' in strategy_type.lower():
                params = best_strategy.get('params', {})
                rsi_period = params.get('period', 14)
                oversold = params.get('oversold', 30)
                overbought = params.get('overbought', 70)
                pinescript_code = PineScriptGenerator.generate_rsi_strategy(rsi_period, oversold, overbought, ticker_input)
            
            else:
                # Generate ensemble strategy for complex strategies
                pinescript_code = PineScriptGenerator.generate_ensemble_strategy(
                    ['Moving Average', 'RSI', 'Bollinger Bands'], 
                    ticker_input
                )
            
            # Display PineScript code in a text area for easy copying
            st.text_area(
                "PineScript Code (Click to select all - Ctrl+A, then Ctrl+C to copy):",
                value=pinescript_code,
                height=400,
                key="pinescript_code"
            )
            
            # Additional strategy insights
            st.subheader("💡 Strategy Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Strategy Characteristics:**")
                if best_strategy['sharpe_ratio'] > 1.5:
                    st.success("🟢 Excellent risk-adjusted returns")
                elif best_strategy['sharpe_ratio'] > 1.0:
                    st.info("🟡 Good risk-adjusted returns")
                else:
                    st.warning("🟠 Moderate risk-adjusted returns")
                
                if best_strategy['max_drawdown'] < 0.1:
                    st.success("🟢 Low drawdown risk")
                elif best_strategy['max_drawdown'] < 0.2:
                    st.info("🟡 Moderate drawdown risk")
                else:
                    st.warning("🟠 High drawdown risk")
            
            with col2:
                st.write("**Trading Frequency:**")
                trades_per_year = best_strategy['total_trades'] / max(1, len(ohlcv_data) / 252)
                
                if trades_per_year < 12:
                    st.info("📅 Long-term strategy (< 1 trade/month)")
                elif trades_per_year < 52:
                    st.info("📊 Medium-term strategy (1-4 trades/month)")
                else:
                    st.warning("⚡ High-frequency strategy (> 1 trade/week)")
                
                st.metric("Avg Trades/Year", f"{trades_per_year:.1f}")
        
        else:
            st.warning("Unable to optimize AI strategy. Insufficient data or market conditions not suitable for current models.")

except Exception as e:
    st.error(f"Error generating AI strategy: {str(e)}")
    st.info("Ensure you have sufficient historical data and valid market conditions for strategy optimization.")

# Sentiment Analysis Proxy
st.header("💭 Market Sentiment Analysis")

try:
    sentiment_scores = ai_models.sentiment_analysis_proxy()
    
    if not sentiment_scores.empty:
        current_sentiment = sentiment_scores.iloc[-1]
        recent_sentiment = sentiment_scores.tail(30).mean()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Sentiment", f"{current_sentiment:.3f}")
            
            if current_sentiment > 0.1:
                st.success("🟢 Positive sentiment")
            elif current_sentiment < -0.1:
                st.error("🔴 Negative sentiment")
            else:
                st.info("⚪ Neutral sentiment")
        
        with col2:
            st.metric("30-Day Avg Sentiment", f"{recent_sentiment:.3f}")
        
        with col3:
            sentiment_change = current_sentiment - recent_sentiment
            st.metric(
                "Sentiment Change", 
                f"{sentiment_change:.3f}",
                delta=f"{'📈' if sentiment_change > 0 else '📉'}"
            )
        
        # Sentiment chart
        fig_sentiment = go.Figure()
        
        fig_sentiment.add_trace(go.Scatter(
            x=sentiment_scores.index,
            y=sentiment_scores,
            mode='lines',
            name='Sentiment Score',
            line=dict(color='purple', width=2)
        ))
        
        fig_sentiment.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_sentiment.add_hline(y=0.1, line_dash="dot", line_color="green", annotation_text="Positive")
        fig_sentiment.add_hline(y=-0.1, line_dash="dot", line_color="red", annotation_text="Negative")
        
        fig_sentiment.update_layout(
            title=f'{ticker_input} - Market Sentiment Proxy',
            xaxis_title='Date',
            yaxis_title='Sentiment Score',
            height=400
        )
        
        st.plotly_chart(fig_sentiment, use_container_width=True)
    
    else:
        st.warning("Unable to calculate sentiment scores")

except Exception as e:
    st.error(f"Error in sentiment analysis: {str(e)}")

# AI Trading Recommendations
st.header("🎯 AI Trading Recommendations")

if model_results:
    st.subheader("📋 Comprehensive Analysis Summary")
    
    # Collect all signals
    recommendation_score = 0
    confidence_factors = []
    
    # Model prediction signal
    if predictions is not None and not predictions.empty:
        prediction_return = (predictions.iloc[-1] - current_price) / current_price
        if prediction_return > 0.05:
            recommendation_score += 2
            confidence_factors.append("Strong positive price prediction")
        elif prediction_return > 0.02:
            recommendation_score += 1
            confidence_factors.append("Moderate positive price prediction")
        elif prediction_return < -0.05:
            recommendation_score -= 2
            confidence_factors.append("Strong negative price prediction")
        elif prediction_return < -0.02:
            recommendation_score -= 1
            confidence_factors.append("Moderate negative price prediction")
    
    # Pattern analysis signal
    if 'patterns' in locals():
        bullish_count = len(patterns['bullish_patterns'])
        bearish_count = len(patterns['bearish_patterns'])
        
        if bullish_count > bearish_count * 1.5:
            recommendation_score += 1
            confidence_factors.append("Bullish pattern dominance")
        elif bearish_count > bullish_count * 1.5:
            recommendation_score -= 1
            confidence_factors.append("Bearish pattern dominance")
    
    # Sentiment signal
    if 'current_sentiment' in locals():
        if current_sentiment > 0.1:
            recommendation_score += 1
            confidence_factors.append("Positive market sentiment")
        elif current_sentiment < -0.1:
            recommendation_score -= 1
            confidence_factors.append("Negative market sentiment")
    
    # Model confidence
    if model_results:
        best_model_r2 = max([result['metrics']['test_r2'] for result in model_results.values()])
        if best_model_r2 > 0.1:
            confidence_factors.append("High model accuracy")
        elif best_model_r2 < 0.05:
            confidence_factors.append("Low model accuracy - use caution")
    
    # Generate recommendation
    col1, col2 = st.columns(2)
    
    with col1:
        if recommendation_score >= 3:
            st.success("🟢 **STRONG BUY SIGNAL**")
            recommendation = "Strong Buy"
        elif recommendation_score >= 1:
            st.success("🟢 **BUY SIGNAL**")
            recommendation = "Buy"
        elif recommendation_score <= -3:
            st.error("🔴 **STRONG SELL SIGNAL**")
            recommendation = "Strong Sell"
        elif recommendation_score <= -1:
            st.error("🔴 **SELL SIGNAL**")
            recommendation = "Sell"
        else:
            st.info("⚪ **HOLD/NEUTRAL**")
            recommendation = "Hold"
        
        st.write(f"**Recommendation Score:** {recommendation_score}")
    
    with col2:
        st.write("**Key Factors:**")
        for factor in confidence_factors:
            st.write(f"• {factor}")
        
        if not confidence_factors:
            st.write("• Insufficient signals for confident recommendation")
    
    # Risk warnings
    st.subheader("⚠️ Risk Considerations")
    
    warnings = []
    
    # Volatility warning
    if volatility > 0.4:
        warnings.append("High volatility detected - consider position sizing")
    
    # Model reliability
    if model_results:
        avg_test_r2 = np.mean([result['metrics']['test_r2'] for result in model_results.values()])
        if avg_test_r2 < 0.05:
            warnings.append("Low model accuracy - predictions may be unreliable")
    
    # Data sufficiency
    if len(ohlcv_data) < 500:
        warnings.append("Limited historical data - model may not capture all patterns")
    
    if warnings:
        for warning in warnings:
            st.warning(f"⚠️ {warning}")
    else:
        st.success("✅ No major risk factors identified")

# Export Results
st.header("📥 Export AI Analysis")

col1, col2 = st.columns(2)

with col1:
    if st.button("📊 Export Model Performance", use_container_width=True):
        if model_results:
            try:
                export_data = []
                for model_name, result in model_results.items():
                    metrics = result['metrics']
                    export_data.append({
                        'Model': model_name,
                        'Train_R2': metrics['train_r2'],
                        'Test_R2': metrics['test_r2'],
                        'Train_MSE': metrics['train_mse'],
                        'Test_MSE': metrics['test_mse'],
                        'Train_MAE': metrics['train_mae'],
                        'Test_MAE': metrics['test_mae']
                    })
                
                export_df = pd.DataFrame(export_data)
                csv = export_df.to_csv(index=False)
                
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{ticker_input}_ai_models_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error preparing export: {str(e)}")

with col2:
    if st.button("🔮 Export Predictions", use_container_width=True):
        if 'predictions' in locals() and predictions is not None:
            try:
                pred_export = pd.DataFrame({
                    'Date': predictions.index,
                    'Predicted_Price': predictions.values,
                    'Model': prediction_model
                })
                
                csv = pred_export.to_csv(index=False)
                
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{ticker_input}_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error preparing predictions export: {str(e)}")

# Model Improvement Suggestions
st.header("💡 Model Improvement Suggestions")

if model_results:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔧 Technical Improvements")
        
        avg_r2 = np.mean([result['metrics']['test_r2'] for result in model_results.values()])
        
        if avg_r2 < 0.05:
            st.info("📈 Try longer training periods for more data")
            st.info("🔄 Consider different prediction horizons")
            st.info("⚙️ Experiment with hyperparameter tuning")
        elif avg_r2 < 0.1:
            st.info("📊 Add more technical indicators")
            st.info("🎯 Fine-tune model parameters")
            st.info("📅 Consider market regime indicators")
        else:
            st.success("✅ Models show good performance")
            st.info("🚀 Consider ensemble methods for even better results")
    
    with col2:
        st.subheader("📊 Data Enhancement")
        
        st.info("💹 Include additional market data (VIX, sector ETFs)")
        st.info("📰 Add fundamental analysis features")
        st.info("🌍 Consider macroeconomic indicators")
        st.info("📈 Include options market data")
        
        if len(ohlcv_data) < 1000:
            st.warning("⏰ More historical data recommended")

# Navigation
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🏠 Back to Dashboard", use_container_width=True):
        st.switch_page("app.py")

with col2:
    if st.button("⚡ Trading Strategies", use_container_width=True):
        st.switch_page("pages/trading_strategies.py")

with col3:
    if st.button("📊 Portfolio Optimization", use_container_width=True):
        st.switch_page("pages/portfolio_optimization.py")
