import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class AIStrategyOptimizer:
    """
    AI-powered trading strategy optimizer that finds optimal parameters
    to minimize drawdown while maximizing returns.
    """
    
    def __init__(self):
        self.models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        self.scaler = StandardScaler()
        self.best_params = {}
        self.feature_importance = {}
        
    def create_features(self, data, lookback_periods=[5, 10, 20, 50]):
        """Create technical features for ML model"""
        features = pd.DataFrame(index=data.index)
        
        # Price features
        features['returns'] = data['Close'].pct_change()
        features['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
        features['volatility'] = features['returns'].rolling(20).std()
        
        # Moving averages
        for period in lookback_periods:
            features[f'ma_{period}'] = data['Close'].rolling(period).mean()
            features[f'ma_ratio_{period}'] = data['Close'] / features[f'ma_{period}']
            features[f'ma_slope_{period}'] = features[f'ma_{period}'].diff(5)
        
        # Technical indicators
        features['rsi'] = self.calculate_rsi(data['Close'])
        features['bb_upper'], features['bb_lower'] = self.calculate_bollinger_bands(data['Close'])
        features['bb_position'] = (data['Close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        # MACD
        features['macd'], features['macd_signal'] = self.calculate_macd(data['Close'])
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # Volume features (if available)
        if 'Volume' in data.columns:
            features['volume_sma'] = data['Volume'].rolling(20).mean()
            features['volume_ratio'] = data['Volume'] / features['volume_sma']
        
        # Market regime features
        features['trend_strength'] = abs(features['ma_20'].diff(10) / features['ma_20'])
        features['volatility_regime'] = features['volatility'] / features['volatility'].rolling(50).mean()
        
        return features.dropna()
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        ma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = ma + (std * std_dev)
        lower = ma - (std * std_dev)
        return upper, lower
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line
    
    def calculate_strategy_metrics(self, returns):
        """Calculate comprehensive strategy metrics"""
        if len(returns) == 0 or returns.sum() == 0:
            return {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'calmar_ratio': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'volatility': 0
            }
        
        total_return = (returns + 1).prod() - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (returns.mean() * 252) / (volatility + 1e-6)
        
        # Calculate drawdown
        cumulative = (returns + 1).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar_ratio = (returns.mean() * 252) / (abs(max_drawdown) + 1e-6)
        
        # Win rate
        win_rate = (returns > 0).mean()
        
        # Profit factor
        profits = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        profit_factor = profits / (losses + 1e-6)
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'volatility': volatility
        }
    
    def optimize_moving_average_strategy(self, data, target_metric='calmar_ratio'):
        """
        Optimize moving average crossover strategy using AI
        """
        results = []
        
        # Test different MA combinations
        ma_combinations = [
            (5, 20), (10, 30), (20, 50), (10, 20), (15, 30),
            (5, 15), (20, 60), (30, 90), (50, 100), (10, 50)
        ]
        
        for fast_ma, slow_ma in ma_combinations:
            # Calculate signals
            data[f'ma_fast'] = data['Close'].rolling(fast_ma).mean()
            data[f'ma_slow'] = data['Close'].rolling(slow_ma).mean()
            
            # Generate signals
            data['signal'] = 0
            data.loc[data['ma_fast'] > data['ma_slow'], 'signal'] = 1
            data.loc[data['ma_fast'] <= data['ma_slow'], 'signal'] = -1
            
            # Calculate returns
            data['strategy_returns'] = data['signal'].shift(1) * data['Close'].pct_change()
            strategy_returns = data['strategy_returns'].dropna()
            
            if len(strategy_returns) > 0:
                metrics = self.calculate_strategy_metrics(strategy_returns)
                metrics['fast_ma'] = fast_ma
                metrics['slow_ma'] = slow_ma
                metrics['params'] = f"MA({fast_ma},{slow_ma})"
                results.append(metrics)
        
        # Find best strategy
        if results:
            best_strategy = max(results, key=lambda x: x[target_metric])
            return best_strategy, results
        
        return None, []
    
    def optimize_rsi_strategy(self, data, target_metric='calmar_ratio'):
        """Optimize RSI strategy parameters"""
        results = []
        
        # Test different RSI parameters
        rsi_configs = [
            (14, 30, 70), (14, 25, 75), (14, 20, 80),
            (21, 30, 70), (7, 30, 70), (14, 35, 65)
        ]
        
        for period, oversold, overbought in rsi_configs:
            rsi = self.calculate_rsi(data['Close'], period)
            
            # Generate signals
            data['signal'] = 0
            data.loc[rsi < oversold, 'signal'] = 1  # Buy oversold
            data.loc[rsi > overbought, 'signal'] = -1  # Sell overbought
            
            # Calculate returns
            data['strategy_returns'] = data['signal'].shift(1) * data['Close'].pct_change()
            strategy_returns = data['strategy_returns'].dropna()
            
            if len(strategy_returns) > 0:
                metrics = self.calculate_strategy_metrics(strategy_returns)
                metrics['rsi_period'] = period
                metrics['oversold'] = oversold
                metrics['overbought'] = overbought
                metrics['params'] = f"RSI({period},{oversold},{overbought})"
                results.append(metrics)
        
        if results:
            best_strategy = max(results, key=lambda x: x[target_metric])
            return best_strategy, results
        
        return None, []
    
    def create_ensemble_strategy(self, data, strategies=['ma', 'rsi', 'bb']):
        """
        Create an ensemble strategy that combines multiple signals
        using AI to weight them optimally
        """
        features = self.create_features(data)
        
        if len(features) < 100:  # Need sufficient data
            return None, "Insufficient data for ensemble strategy"
        
        # Generate individual strategy signals
        signals = pd.DataFrame(index=features.index)
        
        # MA signal
        if 'ma' in strategies:
            signals['ma_signal'] = np.where(features['ma_20'] > features['ma_50'], 1, -1)
        
        # RSI signal
        if 'rsi' in strategies:
            signals['rsi_signal'] = np.where(features['rsi'] < 30, 1, 
                                           np.where(features['rsi'] > 70, -1, 0))
        
        # Bollinger Bands signal
        if 'bb' in strategies:
            signals['bb_signal'] = np.where(features['bb_position'] < 0.2, 1,
                                           np.where(features['bb_position'] > 0.8, -1, 0))
        
        # Prepare features for ML model
        ml_features = features.select_dtypes(include=[np.number]).fillna(0)
        
        # Use future returns as target (shifted by 1 day)
        target = features['returns'].shift(-1).fillna(0)
        
        # Split data for time series validation
        tscv = TimeSeriesSplit(n_splits=3)
        ensemble_weights = []
        
        for train_idx, test_idx in tscv.split(ml_features):
            X_train = ml_features.iloc[train_idx]
            y_train = target.iloc[train_idx]
            
            # Train model to predict optimal signal weights
            try:
                X_scaled = self.scaler.fit_transform(X_train)
                self.models['rf'].fit(X_scaled, y_train)
                
                # Get feature importance
                importance = self.models['rf'].feature_importances_
                self.feature_importance = dict(zip(ml_features.columns, importance))
                
            except Exception as e:
                continue
        
        # Generate ensemble signal
        try:
            X_all_scaled = self.scaler.transform(ml_features)
            predictions = self.models['rf'].predict(X_all_scaled)
            
            # Combine signals with ML predictions
            ensemble_signal = pd.Series(index=signals.index, data=0)
            
            for col in signals.columns:
                if col in signals:
                    ensemble_signal += signals[col] * 0.3  # Base weight
            
            # Add ML prediction component
            ensemble_signal += np.sign(predictions) * 0.4
            
            # Normalize signal
            ensemble_signal = np.clip(ensemble_signal, -1, 1)
            
            # Calculate strategy returns
            strategy_returns = ensemble_signal.shift(1) * features['returns']
            strategy_returns = strategy_returns.dropna()
            
            metrics = self.calculate_strategy_metrics(strategy_returns)
            metrics['strategy_type'] = 'AI_Ensemble'
            metrics['components'] = strategies
            
            return metrics, strategy_returns
            
        except Exception as e:
            return None, f"Error creating ensemble: {str(e)}"
    
    def optimize_risk_adjusted_strategy(self, data, max_drawdown_target=0.15):
        """
        Create a risk-adjusted strategy that aims to minimize drawdown
        while maintaining returns
        """
        features = self.create_features(data)
        
        if len(features) < 50:
            return None, "Insufficient data"
        
        # Calculate rolling metrics for adaptive sizing
        features['rolling_volatility'] = features['returns'].rolling(20).std()
        features['rolling_sharpe'] = (features['returns'].rolling(20).mean() / 
                                     features['rolling_volatility']) * np.sqrt(252)
        
        # Dynamic position sizing based on volatility
        vol_target = 0.15  # 15% annual volatility target
        features['position_size'] = vol_target / (features['rolling_volatility'] * np.sqrt(252) + 1e-6)
        features['position_size'] = np.clip(features['position_size'], 0.1, 2.0)  # Limit leverage
        
        # Generate base signal (using trend following)
        features['base_signal'] = np.where(features['ma_20'] > features['ma_50'], 1, -1)
        
        # Risk adjustment factors
        features['volatility_filter'] = np.where(features['rolling_volatility'] > 
                                                features['rolling_volatility'].rolling(50).quantile(0.8), 0.5, 1.0)
        
        features['drawdown_filter'] = 1.0
        cumulative_returns = (features['returns'] + 1).cumprod()
        running_max = cumulative_returns.expanding().max()
        current_drawdown = (cumulative_returns - running_max) / running_max
        
        # Reduce position size during drawdown periods
        features.loc[current_drawdown < -0.05, 'drawdown_filter'] = 0.5
        features.loc[current_drawdown < -0.10, 'drawdown_filter'] = 0.2
        
        # Final adjusted signal
        features['adjusted_signal'] = (features['base_signal'] * 
                                     features['position_size'] * 
                                     features['volatility_filter'] * 
                                     features['drawdown_filter'])
        
        # Calculate strategy returns
        strategy_returns = features['adjusted_signal'].shift(1) * features['returns']
        strategy_returns = strategy_returns.dropna()
        
        metrics = self.calculate_strategy_metrics(strategy_returns)
        metrics['strategy_type'] = 'Risk_Adjusted'
        metrics['max_drawdown_target'] = max_drawdown_target
        
        return metrics, strategy_returns
    
    def generate_strategy_recommendations(self, data):
        """
        Generate comprehensive strategy recommendations using AI optimization
        """
        recommendations = {}
        
        # Optimize individual strategies
        ma_best, ma_results = self.optimize_moving_average_strategy(data)
        rsi_best, rsi_results = self.optimize_rsi_strategy(data)
        
        # Create ensemble strategy
        ensemble_metrics, ensemble_returns = self.create_ensemble_strategy(data)
        
        # Create risk-adjusted strategy
        risk_adj_metrics, risk_adj_returns = self.optimize_risk_adjusted_strategy(data)
        
        recommendations['moving_average'] = {
            'best': ma_best,
            'all_results': ma_results[:5]  # Top 5 results
        }
        
        recommendations['rsi'] = {
            'best': rsi_best,
            'all_results': rsi_results[:5]
        }
        
        if ensemble_metrics:
            recommendations['ensemble'] = ensemble_metrics
            
        if risk_adj_metrics:
            recommendations['risk_adjusted'] = risk_adj_metrics
        
        # Overall recommendations
        all_strategies = []
        if ma_best:
            all_strategies.append(('Moving Average', ma_best))
        if rsi_best:
            all_strategies.append(('RSI', rsi_best))
        if ensemble_metrics:
            all_strategies.append(('AI Ensemble', ensemble_metrics))
        if risk_adj_metrics:
            all_strategies.append(('Risk Adjusted', risk_adj_metrics))
        
        # Rank by Calmar ratio (return/max_drawdown)
        ranked_strategies = sorted(all_strategies, 
                                 key=lambda x: x[1].get('calmar_ratio', 0), 
                                 reverse=True)
        
        recommendations['top_strategies'] = ranked_strategies[:3]
        recommendations['feature_importance'] = self.feature_importance
        
        return recommendations