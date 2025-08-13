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

# Page configuration
st.set_page_config(
    page_title="AI Analysis",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI-Powered Financial Analysis")
st.markdown("Advanced machine learning models for price prediction, pattern recognition, and market analysis")

# Sidebar for inputs
st.sidebar.header("AI Analysis Configuration")

# Asset selection
ticker_input = st.sidebar.text_input(
    "Enter Stock Ticker",
    value="AAPL",
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

# Time period selection
period_options = {
    "1 Year": "1y",
    "2 Years": "2y",
    "3 Years": "3y",
    "5 Years": "5y"
}

selected_period = st.sidebar.selectbox(
    "Training Data Period",
    list(period_options.keys()),
    index=2,
    help="More data generally improves model performance"
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
        
        # Initialize AI models
        ai_models = AIModels(ohlcv_data, ticker_input)
        
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
    feature_cols = ai_models.get_feature_columns()
    st.write(f"**Total Features Generated:** {len(feature_cols)}")
    
    if st.checkbox("Show all feature names"):
        st.write(feature_cols)

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
