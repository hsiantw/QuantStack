import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

class AIModels:
    """AI-powered financial analysis and prediction models with extended historical training data"""
    
    def __init__(self, price_data, ticker, use_extended_history=True, training_years=15):
        """
        Initialize with price data, optionally using extended historical data for training
        
        Args:
            price_data (pandas.DataFrame): Recent OHLCV data for prediction
            ticker (str): Asset ticker
            use_extended_history (bool): Whether to fetch 10-20 years of historical data for training
            training_years (int): Number of years of historical data for training (10-20)
        """
        self.data = price_data.copy()
        self.ticker = ticker
        self.scaler = StandardScaler()
        self.use_extended_history = use_extended_history
        self.training_years = max(10, min(20, training_years))  # Ensure 10-20 year range
        self.training_data = None
        
        # Get extended historical data for training if requested
        if use_extended_history:
            self._fetch_training_data()
        
        # Prepare features for both training and prediction data
        self._prepare_features()
    
    def _fetch_training_data(self):
        """Fetch extended historical data for AI model training"""
        from utils.data_fetcher import DataFetcher
        
        try:
            st.info(f"Fetching {self.training_years} years of historical data for robust AI training...")
            self.training_data = DataFetcher.get_historical_data_for_ai(
                self.ticker, 
                years_back=self.training_years
            )
            
            if not self.training_data.empty:
                st.success(f"Successfully loaded {len(self.training_data)} days of historical data "
                          f"({self.training_data.attrs.get('years_covered', self.training_years)} years) for training")
            else:
                st.warning("Could not fetch extended historical data. Using provided data for training.")
                self.training_data = self.data.copy()
                
        except Exception as e:
            st.warning(f"Error fetching training data: {str(e)}. Using provided data for training.")
            self.training_data = self.data.copy()
    
    def _prepare_features(self):
        """Prepare features for ML models using both training and prediction data"""
        
        # Prepare features for training data if available
        if self.training_data is not None and not self.training_data.empty:
            self.training_features = self._calculate_features(self.training_data.copy())
        
        # Prepare features for prediction data
        self.prediction_features = self._calculate_features(self.data.copy())
    
    def _calculate_features(self, data):
        """Calculate features for a given dataset"""
        
        # Price-based features
        data['Returns'] = data['Close'].pct_change()
        data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        data['High_Low_Ratio'] = data['High'] / data['Low']
        data['Close_Open_Ratio'] = data['Close'] / data['Open']
        
        # Technical indicators
        # Moving averages
        for window in [5, 10, 20, 50]:
            data[f'MA_{window}'] = data['Close'].rolling(window=window).mean()
            data[f'MA_Ratio_{window}'] = data['Close'] / data[f'MA_{window}']
        
        # Volatility
        for window in [5, 10, 20]:
            data[f'Volatility_{window}'] = data['Returns'].rolling(window=window).std()
        
        # RSI
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        ma_20 = data['Close'].rolling(window=20).mean()
        std_20 = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = ma_20 + (2 * std_20)
        data['BB_Lower'] = ma_20 - (2 * std_20)
        data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
        
        # Volume indicators
        if 'Volume' in data.columns:
            data['Volume_MA_10'] = data['Volume'].rolling(window=10).mean()
            data['Volume_Ratio'] = data['Volume'] / data['Volume_MA_10']
        
        # Lag features
        for lag in range(1, 6):
            data[f'Return_Lag_{lag}'] = data['Returns'].shift(lag)
            data[f'Price_Lag_{lag}'] = data['Close'].shift(lag)
        
        # Future returns (targets)
        for horizon in [1, 5, 10]:
            data[f'Future_Return_{horizon}'] = data['Returns'].shift(-horizon)
            data[f'Future_Price_{horizon}'] = data['Close'].shift(-horizon)
        
        return data
    
    def get_feature_columns(self, data):
        """Get list of feature columns from given dataset"""
        excluded_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'] + \
                       [col for col in data.columns if col.startswith('Future_')]
        
        feature_cols = [col for col in data.columns if col not in excluded_cols]
        return feature_cols
    
    def prepare_data_for_training(self, target_horizon=1, test_size=0.2):
        """
        Prepare data for training ML models using extended historical data
        
        Args:
            target_horizon (int): Prediction horizon in days
            test_size (float): Test set size
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test, feature_names, training_info)
        """
        # Use training data if available, otherwise fall back to prediction data
        training_data = self.training_features if hasattr(self, 'training_features') and self.training_features is not None else self.prediction_features
        
        target_col = f'Future_Return_{target_horizon}'
        feature_cols = self.get_feature_columns(training_data)
        
        # Remove rows with NaN values
        clean_data = training_data[feature_cols + [target_col]].dropna()
        
        if len(clean_data) < 100:
            raise ValueError(f"Insufficient data for training. Need at least 100 samples, got {len(clean_data)}.")
        
        X = clean_data[feature_cols]
        y = clean_data[target_col]
        
        # Use time series split to respect temporal order
        split_point = int(len(clean_data) * (1 - test_size))
        
        X_train = X.iloc[:split_point]
        X_test = X.iloc[split_point:]
        y_train = y.iloc[:split_point]
        y_test = y.iloc[split_point:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Prepare training info
        training_info = {
            'total_samples': len(clean_data),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features_count': len(feature_cols),
            'data_period': f"{training_data.index[0].strftime('%Y-%m-%d')} to {training_data.index[-1].strftime('%Y-%m-%d')}",
            'years_of_data': (training_data.index[-1] - training_data.index[0]).days / 365.25
        }
        
        return X_train_scaled, X_test_scaled, y_train.values, y_test.values, feature_cols, training_info
    
    def train_random_forest(self, target_horizon=1, **kwargs):
        """
        Train Random Forest model
        
        Args:
            target_horizon (int): Prediction horizon
            **kwargs: Additional parameters for RandomForestRegressor
        
        Returns:
            dict: Model and performance metrics
        """
        try:
            X_train, X_test, y_train, y_test, feature_cols, training_info = self.prepare_data_for_training(target_horizon)
            
            # Default parameters
            rf_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            }
            rf_params.update(kwargs)
            
            # Train model
            model = RandomForestRegressor(**rf_params)
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Performance metrics
            metrics = {
                'train_mse': mean_squared_error(y_train, y_pred_train),
                'test_mse': mean_squared_error(y_test, y_pred_test),
                'train_mae': mean_absolute_error(y_train, y_pred_train),
                'test_mae': mean_absolute_error(y_test, y_pred_test),
                'train_r2': r2_score(y_train, y_pred_train),
                'test_r2': r2_score(y_test, y_pred_test)
            }
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return {
                'model': model,
                'metrics': metrics,
                'predictions': {
                    'y_train': y_train,
                    'y_test': y_test,
                    'y_pred_train': y_pred_train,
                    'y_pred_test': y_pred_test
                },
                'feature_importance': feature_importance,
                'scaler': self.scaler
            }
            
        except Exception as e:
            st.error(f"Error training Random Forest model: {str(e)}")
            return None
    
    def train_gradient_boosting(self, target_horizon=1, **kwargs):
        """
        Train Gradient Boosting model
        
        Args:
            target_horizon (int): Prediction horizon
            **kwargs: Additional parameters for GradientBoostingRegressor
        
        Returns:
            dict: Model and performance metrics
        """
        try:
            X_train, X_test, y_train, y_test, feature_cols, training_info = self.prepare_data_for_training(target_horizon)
            
            # Default parameters
            gb_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42
            }
            gb_params.update(kwargs)
            
            # Train model
            model = GradientBoostingRegressor(**gb_params)
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Performance metrics
            metrics = {
                'train_mse': mean_squared_error(y_train, y_pred_train),
                'test_mse': mean_squared_error(y_test, y_pred_test),
                'train_mae': mean_absolute_error(y_train, y_pred_train),
                'test_mae': mean_absolute_error(y_test, y_pred_test),
                'train_r2': r2_score(y_train, y_pred_train),
                'test_r2': r2_score(y_test, y_pred_test)
            }
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return {
                'model': model,
                'metrics': metrics,
                'predictions': {
                    'y_train': y_train,
                    'y_test': y_test,
                    'y_pred_train': y_pred_train,
                    'y_pred_test': y_pred_test
                },
                'feature_importance': feature_importance,
                'scaler': self.scaler
            }
            
        except Exception as e:
            st.error(f"Error training Gradient Boosting model: {str(e)}")
            return None
    
    def generate_predictions(self, model_result, prediction_days=30):
        """
        Generate future predictions
        
        Args:
            model_result (dict): Trained model result
            prediction_days (int): Number of days to predict
        
        Returns:
            pandas.Series: Future predictions
        """
        if model_result is None:
            return None
            
        try:
            model = model_result['model']
            scaler = model_result['scaler']
            
            # Use the last available data point for prediction
            feature_cols = self.get_feature_columns()
            last_features = self.data[feature_cols].dropna().iloc[-1:].values
            last_features_scaled = scaler.transform(last_features)
            
            # Generate predictions
            predictions = []
            current_features = last_features_scaled.copy()
            
            for _ in range(prediction_days):
                pred = model.predict(current_features)[0]
                predictions.append(pred)
                
                # Update features for next prediction (simplified approach)
                # In practice, you'd want more sophisticated feature updating
                current_features = current_features.copy()
            
            # Create prediction series
            last_date = self.data.index[-1]
            prediction_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                           periods=prediction_days, freq='D')
            
            return pd.Series(predictions, index=prediction_dates)
            
        except Exception as e:
            st.error(f"Error generating predictions: {str(e)}")
            return None
    
    def plot_model_performance(self, model_result, model_name="Model"):
        """
        Create model performance visualization
        
        Args:
            model_result (dict): Trained model result
            model_name (str): Name of the model
        
        Returns:
            dict: Dictionary of plotly figures
        """
        if model_result is None:
            return {}
            
        plots = {}
        predictions = model_result['predictions']
        
        # 1. Actual vs Predicted (Test Set)
        fig_pred = go.Figure()
        
        fig_pred.add_trace(go.Scatter(
            x=predictions['y_test'],
            y=predictions['y_pred_test'],
            mode='markers',
            name='Predictions',
            marker=dict(color='blue', opacity=0.6)
        ))
        
        # Perfect prediction line
        min_val = min(predictions['y_test'].min(), predictions['y_pred_test'].min())
        max_val = max(predictions['y_test'].max(), predictions['y_pred_test'].max())
        
        fig_pred.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        fig_pred.update_layout(
            title=f'{model_name} - Actual vs Predicted Returns',
            xaxis_title='Actual Returns',
            yaxis_title='Predicted Returns',
            height=500
        )
        
        plots['actual_vs_predicted'] = fig_pred
        
        # 2. Feature Importance
        feature_importance = model_result['feature_importance'].head(15)  # Top 15 features
        
        fig_importance = go.Figure(data=[
            go.Bar(
                x=feature_importance['importance'],
                y=feature_importance['feature'],
                orientation='h'
            )
        ])
        
        fig_importance.update_layout(
            title=f'{model_name} - Feature Importance',
            xaxis_title='Importance',
            yaxis_title='Features',
            height=600
        )
        
        plots['feature_importance'] = fig_importance
        
        # 3. Residuals Analysis
        residuals_test = predictions['y_test'] - predictions['y_pred_test']
        
        fig_residuals = go.Figure()
        
        fig_residuals.add_trace(go.Scatter(
            x=predictions['y_pred_test'],
            y=residuals_test,
            mode='markers',
            name='Residuals',
            marker=dict(color='green', opacity=0.6)
        ))
        
        fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
        
        fig_residuals.update_layout(
            title=f'{model_name} - Residuals Analysis',
            xaxis_title='Predicted Returns',
            yaxis_title='Residuals',
            height=400
        )
        
        plots['residuals'] = fig_residuals
        
        return plots
    
    def pattern_recognition(self, lookback_window=20):
        """
        Simple pattern recognition using price movements
        
        Args:
            lookback_window (int): Window for pattern analysis
        
        Returns:
            dict: Detected patterns
        """
        patterns = {
            'bullish_patterns': [],
            'bearish_patterns': [],
            'neutral_patterns': []
        }
        
        try:
            # Calculate price changes
            price_changes = self.data['Close'].pct_change()
            
            # Define simple patterns
            for i in range(lookback_window, len(self.data)):
                window_changes = price_changes.iloc[i-lookback_window:i]
                
                # Bullish patterns
                if (window_changes.tail(5) > 0).sum() >= 4:  # 4 out of 5 positive days
                    patterns['bullish_patterns'].append({
                        'date': self.data.index[i],
                        'pattern': 'Consecutive Gains',
                        'strength': (window_changes.tail(5) > 0).sum() / 5
                    })
                
                # Bearish patterns
                if (window_changes.tail(5) < 0).sum() >= 4:  # 4 out of 5 negative days
                    patterns['bearish_patterns'].append({
                        'date': self.data.index[i],
                        'pattern': 'Consecutive Losses',
                        'strength': (window_changes.tail(5) < 0).sum() / 5
                    })
                
                # Mean reversion patterns
                recent_change = price_changes.iloc[i-1]
                avg_change = window_changes.mean()
                if abs(recent_change - avg_change) > 2 * window_changes.std():
                    patterns['neutral_patterns'].append({
                        'date': self.data.index[i],
                        'pattern': 'Mean Reversion Signal',
                        'strength': abs(recent_change - avg_change) / window_changes.std()
                    })
            
            return patterns
            
        except Exception as e:
            st.error(f"Error in pattern recognition: {str(e)}")
            return patterns
    
    def sentiment_analysis_proxy(self):
        """
        Create a proxy for sentiment analysis using price and volume data
        
        Returns:
            pandas.Series: Sentiment proxy scores
        """
        try:
            sentiment_scores = []
            
            # Use price momentum and volume as sentiment proxies
            returns = self.data['Close'].pct_change()
            
            # Volume-weighted sentiment (if volume is available)
            if 'Volume' in self.data.columns:
                volume_ma = self.data['Volume'].rolling(window=20).mean()
                volume_ratio = self.data['Volume'] / volume_ma
                
                # High volume + positive returns = bullish sentiment
                # High volume + negative returns = bearish sentiment
                sentiment = returns * np.log(volume_ratio.clip(lower=0.1))
            else:
                # Use only returns and volatility
                volatility = returns.rolling(window=20).std()
                sentiment = returns / (volatility + 0.001)  # Risk-adjusted returns
            
            # Smooth the sentiment scores
            sentiment_smooth = sentiment.rolling(window=5).mean()
            
            return sentiment_smooth
            
        except Exception as e:
            st.error(f"Error in sentiment analysis: {str(e)}")
            return pd.Series(index=self.data.index, dtype=float)
