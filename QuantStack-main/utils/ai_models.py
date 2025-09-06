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
    """AI-powered financial analysis and prediction models - DISABLED FOR DEPLOYMENT"""
    
    def __init__(self, price_data, ticker, use_extended_history=True, training_years=15):
        """
        Initialize with price data - AI functionality disabled
        """
        self.data = price_data.copy()
        self.ticker = ticker
        self.scaler = None
        self.use_extended_history = False
        self.training_years = 0
        self.training_data = None
        st.info("AI functionality is currently disabled for deployment")
    
    def _fetch_training_data(self):
        """Disabled for deployment"""
        return None
    
    def _prepare_features(self):
        """Disabled for deployment"""
        return None
    
    def _calculate_features(self, data):
        """Calculate features for a given dataset"""
        
        try:
            # Validate input data
            if data is None or data.empty:
                raise ValueError("Input data is empty or None")
                
            # Check required columns
            required_cols = ['Open', 'High', 'Low', 'Close']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
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
            rs = gain / np.where(loss != 0, loss, 1e-6)
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            ma_20 = data['Close'].rolling(window=20).mean()
            std_20 = data['Close'].rolling(window=20).std()
            data['BB_Upper'] = ma_20 + (2 * std_20)
            data['BB_Lower'] = ma_20 - (2 * std_20)
            bb_width = data['BB_Upper'] - data['BB_Lower']
            data['BB_Position'] = (data['Close'] - data['BB_Lower']) / np.where(bb_width != 0, bb_width, 1e-6)
        
            # Volume indicators (always include to maintain consistent features)
            if 'Volume' in data.columns and not data['Volume'].isna().all():
                data['Volume_MA_10'] = data['Volume'].rolling(window=10).mean()
                data['Volume_Ratio'] = data['Volume'] / np.where(data['Volume_MA_10'] != 0, data['Volume_MA_10'], 1e-6)
            else:
                # Add dummy volume features for consistency if volume data is missing
                avg_price = data['Close'].mean()
                data['Volume_MA_10'] = avg_price * 1000  # Dummy volume based on price
                data['Volume_Ratio'] = 1.0  # Neutral ratio
            
            # Lag features
            for lag in range(1, 6):
                data[f'Return_Lag_{lag}'] = data['Returns'].shift(lag)
                data[f'Price_Lag_{lag}'] = data['Close'].shift(lag)
            
            # Future returns (targets)
            for horizon in [1, 5, 10]:
                data[f'Future_Return_{horizon}'] = data['Returns'].shift(-horizon)
                data[f'Future_Price_{horizon}'] = data['Close'].shift(-horizon)
            
            # Replace infinite values with NaN
            data = data.replace([np.inf, -np.inf], np.nan)
            
            return data
            
        except Exception as e:
            print(f"Error calculating features: {str(e)}")
            # Return original data if feature calculation fails
            return data
    
    def get_feature_columns(self, data):
        """Get list of feature columns from given dataset"""
        if data is None or data.empty:
            return []
            
        excluded_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'] + \
                       [col for col in data.columns if col.startswith('Future_')]
        
        feature_cols = [col for col in data.columns if col not in excluded_cols]
        
        # Validate that we have the expected feature columns
        # Build expected features list dynamically
        expected_features = [
            'Returns', 'Log_Returns', 'High_Low_Ratio', 'Close_Open_Ratio',
            'MA_5', 'MA_Ratio_5', 'MA_10', 'MA_Ratio_10', 'MA_20', 'MA_Ratio_20', 'MA_50', 'MA_Ratio_50',
            'Volatility_5', 'Volatility_10', 'Volatility_20', 'RSI',
            'BB_Upper', 'BB_Lower', 'BB_Position', 'Volume_MA_10', 'Volume_Ratio'
        ]
        
        # Add lag features to expected list
        for lag in range(1, 6):
            expected_features.extend([f'Return_Lag_{lag}', f'Price_Lag_{lag}'])
            
        expected_features = list(set(expected_features))  # Remove duplicates
        
        # Check if key features exist
        missing_features = [f for f in expected_features if f not in data.columns]
        if missing_features:
            print(f"Warning: Missing expected features: {missing_features}")
        
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
        Train Random Forest model - DISABLED FOR DEPLOYMENT
        
        Returns:
            None: AI functionality disabled
        """
        st.warning("AI model training is currently disabled for deployment")
        return None
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
                'scaler': self.scaler,
                'feature_columns': feature_cols  # Store training feature columns
            }
            
        except Exception as e:
            st.error(f"Error training Random Forest model: {str(e)}")
            return None
    
    def train_gradient_boosting(self, target_horizon=1, **kwargs):
        """
        Train Gradient Boosting model - DISABLED FOR DEPLOYMENT
        
        Returns:
            None: AI functionality disabled
        """
        st.warning("AI model training is currently disabled for deployment")
        return None
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
                'scaler': self.scaler,
                'feature_columns': feature_cols  # Store training feature columns
            }
            
        except Exception as e:
            st.error(f"Error training Gradient Boosting model: {str(e)}")
            return None
    
    def generate_predictions(self, model_result, prediction_days=30):
        """
        Generate future predictions - DISABLED FOR DEPLOYMENT
        
        Returns:
            None: AI functionality disabled
        """
        st.warning("AI predictions are currently disabled for deployment")
        return None
            
        try:
            model = model_result['model']
            scaler = model_result['scaler']
            
            # Use the stored training feature columns for consistency
            training_feature_cols = model_result.get('feature_columns', [])
            
            if not training_feature_cols:
                raise ValueError("No training feature columns found in model result")
            
            # Get prediction features and ensure alignment with training features
            prediction_feature_cols = self.get_feature_columns(self.prediction_features)
            
            # Find common features between training and prediction
            common_features = [col for col in training_feature_cols if col in prediction_feature_cols]
            
            if len(common_features) != len(training_feature_cols):
                st.warning(f"Feature mismatch detected. Training: {len(training_feature_cols)}, "
                          f"Prediction: {len(prediction_feature_cols)}, Common: {len(common_features)}")
                
                # If there's a significant mismatch, recalculate features to match training
                if len(common_features) < len(training_feature_cols) * 0.8:  # Less than 80% match
                    st.info("Recalculating features to match training data...")
                    self._align_prediction_features_with_training(training_feature_cols)
                    prediction_feature_cols = self.get_feature_columns(self.prediction_features)
                    common_features = [col for col in training_feature_cols if col in prediction_feature_cols]
            
            # Use only common features for consistency
            feature_cols = common_features
            
            if not feature_cols:
                raise ValueError("No matching features available between training and prediction data")
            
            # Get recent data for prediction
            recent_data = self.prediction_features.dropna()
            if recent_data.empty:
                return None
                
            # Select features in the same order as training
            X_recent = recent_data[feature_cols].iloc[-1:].values
            
            # Ensure we have the right number of features
            if X_recent.shape[1] != len(feature_cols):
                raise ValueError(f"Feature count mismatch: expected {len(feature_cols)}, got {X_recent.shape[1]}")
            
            # Scale features if needed
            try:
                # Try to use the scaler if it was fitted during training
                X_recent_scaled = scaler.transform(X_recent)
            except Exception:
                # If scaling fails, use unscaled data
                X_recent_scaled = X_recent
            
            # Generate prediction
            prediction = model.predict(X_recent_scaled)[0]
            
            return prediction
            
        except Exception as e:
            # More specific error information
            error_details = f"Feature columns error: {str(e)}"
            if 'feature_columns' in model_result:
                training_feature_count = len(model_result['feature_columns'])
                prediction_feature_count = len(self.get_feature_columns(self.prediction_features))
                error_details += f" (Training features: {training_feature_count}, Prediction features: {prediction_feature_count})"
            
            st.error(f"Error generating predictions: {error_details}")
            return None
    
    def _align_prediction_features_with_training(self, training_feature_cols):
        """Ensure prediction features match training features exactly"""
        try:
            # Get current prediction features
            current_features = self.get_feature_columns(self.prediction_features)
            
            # Find missing features in prediction data
            missing_features = [col for col in training_feature_cols if col not in current_features]
            
            if missing_features:
                st.info(f"Adding missing features: {missing_features}")
                
                # Add missing volume features if needed
                if 'Volume_MA_10' in missing_features or 'Volume_Ratio' in missing_features:
                    if 'Volume' not in self.prediction_features.columns:
                        # Add dummy volume if missing
                        self.prediction_features['Volume'] = self.prediction_features['Close'] * 1000
                    
                    if 'Volume_MA_10' not in self.prediction_features.columns:
                        self.prediction_features['Volume_MA_10'] = self.prediction_features['Volume'].rolling(window=10).mean()
                    if 'Volume_Ratio' not in self.prediction_features.columns:
                        volume_ma = self.prediction_features['Volume_MA_10']
                        self.prediction_features['Volume_Ratio'] = self.prediction_features['Volume'] / np.where(volume_ma != 0, volume_ma, 1e-6)
                
                # Add missing lag features
                for missing_col in missing_features:
                    if missing_col.startswith('Return_Lag_'):
                        lag = int(missing_col.split('_')[-1])
                        if 'Returns' in self.prediction_features.columns:
                            self.prediction_features[missing_col] = self.prediction_features['Returns'].shift(lag)
                    elif missing_col.startswith('Price_Lag_'):
                        lag = int(missing_col.split('_')[-1])
                        self.prediction_features[missing_col] = self.prediction_features['Close'].shift(lag)
                    elif missing_col not in self.prediction_features.columns:
                        # Fill with zeros or calculated values for other missing features
                        self.prediction_features[missing_col] = 0
                
                # Replace infinite values
                self.prediction_features = self.prediction_features.replace([np.inf, -np.inf], np.nan)
                
        except Exception as e:
            st.error(f"Error aligning features: {str(e)}")

    
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
        Pattern recognition - DISABLED FOR DEPLOYMENT
        
        Returns:
            dict: Empty patterns (AI disabled)
        """
        st.warning("AI pattern recognition is currently disabled for deployment")
        return {
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
