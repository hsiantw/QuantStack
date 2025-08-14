import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AIStrategyOptimizer:
    """
    AI-powered trading strategy optimizer that finds optimal parameters
    to minimize drawdown while maximizing returns with detailed methodology tracking.
    """
    
    def __init__(self, data, ticker):
        self.data = data.copy()
        self.ticker = ticker
        self.models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        self.scaler = StandardScaler()
        self.best_params = {}
        self.feature_importance = {}
        self.calculation_details = {}
        self.optimization_log = []
        
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
        bb_width = features['bb_upper'] - features['bb_lower']
        features['bb_position'] = (data['Close'] - features['bb_lower']) / np.where(bb_width != 0, bb_width, 1e-6)
        
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
        rs = gain / np.where(loss != 0, loss, 1e-6)
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
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line
    
    def optimize_strategy(self):
        """Main optimization method with detailed tracking"""
        self.optimization_log.append(f"Starting optimization for {self.ticker} at {datetime.now()}")
        
        # Define strategy universe
        strategies = self._get_strategy_universe()
        self.optimization_log.append(f"Testing {len(strategies)} strategy combinations")
        
        results = []
        for i, strategy in enumerate(strategies):
            try:
                performance = self._backtest_strategy(strategy)
                if performance and performance['total_trades'] >= 10:  # Minimum trades for significance
                    results.append({
                        'name': strategy['name'],
                        'type': strategy['type'],
                        'params': strategy['params'],
                        'annual_return': performance['annual_return'],
                        'sharpe_ratio': performance['sharpe_ratio'],
                        'max_drawdown': performance['max_drawdown'],
                        'win_rate': performance['win_rate'],
                        'total_trades': performance['total_trades'],
                        'calmar_ratio': performance.get('calmar_ratio', 0),
                        'calculation_details': performance['calculation_details']
                    })
                    
                self.optimization_log.append(f"Strategy {i+1}/{len(strategies)}: {strategy['name']} - Complete")
                
            except Exception as e:
                self.optimization_log.append(f"Strategy {strategy['name']} failed: {str(e)}")
                continue
        
        if not results:
            return None
        
        # Sort by Sharpe ratio (primary) and drawdown (secondary)
        results.sort(key=lambda x: (x['sharpe_ratio'], -x['max_drawdown']), reverse=True)
        
        best_strategy = results[0]
        
        # Create detailed optimization results
        optimization_results = {
            'best_strategy': best_strategy,
            'all_strategies': results,
            'calculation_details': best_strategy['calculation_details'],
            'annual_volatility': self._calculate_annual_volatility(),
            'optimization_log': self.optimization_log,
            'backtest_periods': self._get_period_analysis(best_strategy),
            'monte_carlo': self._monte_carlo_analysis(best_strategy)
        }
        
        return optimization_results
    
    def _get_strategy_universe(self):
        """Define comprehensive strategy testing universe"""
        strategies = [
            # Moving Average Strategies
            {'name': 'MA_Cross_5_15', 'type': 'ma_crossover', 'params': {'short_period': 5, 'long_period': 15}},
            {'name': 'MA_Cross_10_20', 'type': 'ma_crossover', 'params': {'short_period': 10, 'long_period': 20}},
            {'name': 'MA_Cross_20_50', 'type': 'ma_crossover', 'params': {'short_period': 20, 'long_period': 50}},
            {'name': 'MA_Cross_50_200', 'type': 'ma_crossover', 'params': {'short_period': 50, 'long_period': 200}},
            
            # RSI Strategies
            {'name': 'RSI_Conservative', 'type': 'rsi_mean_reversion', 'params': {'period': 14, 'oversold': 30, 'overbought': 70}},
            {'name': 'RSI_Aggressive', 'type': 'rsi_mean_reversion', 'params': {'period': 14, 'oversold': 20, 'overbought': 80}},
            {'name': 'RSI_Moderate', 'type': 'rsi_mean_reversion', 'params': {'period': 21, 'oversold': 25, 'overbought': 75}},
            
            # Bollinger Band Strategies
            {'name': 'BB_Reversal_2std', 'type': 'bollinger_bands', 'params': {'period': 20, 'std_dev': 2}},
            {'name': 'BB_Reversal_2.5std', 'type': 'bollinger_bands', 'params': {'period': 20, 'std_dev': 2.5}},
            
            # Combined Strategies
            {'name': 'MA_RSI_Combo', 'type': 'ma_rsi_combo', 'params': {'ma_short': 10, 'ma_long': 30, 'rsi_period': 14, 'rsi_oversold': 30, 'rsi_overbought': 70}},
            {'name': 'Triple_MA', 'type': 'triple_ma', 'params': {'fast': 5, 'medium': 20, 'slow': 50}},
        ]
        
        return strategies
    
    def _backtest_strategy(self, strategy):
        """Comprehensive backtesting with detailed calculations"""
        # Prepare data with technical indicators
        data = self._prepare_data()
        
        # Generate signals based on strategy
        signals = self._generate_signals(data, strategy)
        
        if signals is None or len(signals) == 0:
            return None
        
        # Calculate returns
        returns = self._calculate_strategy_returns(data, signals)
        
        if returns is None or len(returns) == 0:
            return None
        
        # Calculate performance metrics with detailed tracking
        performance = self._calculate_performance_metrics(returns, data)
        
        # Add trade analysis
        trades = self._analyze_trades(data, signals)
        performance.update(trades)
        
        return performance
    
    def _prepare_data(self):
        """Prepare data with all technical indicators"""
        data = self.data.copy()
        
        # Moving averages
        for period in [5, 10, 15, 20, 30, 50, 200]:
            data[f'MA_{period}'] = data['Close'].rolling(window=period).mean()
        
        # RSI
        data['RSI'] = self._calculate_rsi_full(data['Close'])
        
        # Bollinger Bands
        data['BB_Upper'], data['BB_Lower'] = self.calculate_bollinger_bands(data['Close'])
        data['BB_Middle'] = data['Close'].rolling(20).mean()
        
        # MACD
        data['MACD'], data['MACD_Signal'] = self.calculate_macd(data['Close'])
        
        return data.dropna()
    
    def _calculate_rsi_full(self, prices, period=14):
        """Full RSI calculation with protection against division by zero"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / np.where(loss != 0, loss, 1e-6)
        return 100 - (100 / (1 + rs))
    
    def _generate_signals(self, data, strategy):
        """Generate trading signals based on strategy type"""
        signals = pd.Series(0, index=data.index)
        
        if strategy['type'] == 'ma_crossover':
            short_ma = data[f"MA_{strategy['params']['short_period']}"]
            long_ma = data[f"MA_{strategy['params']['long_period']}"]
            signals = np.where(short_ma > long_ma, 1, -1)
            
        elif strategy['type'] == 'rsi_mean_reversion':
            rsi = data['RSI']
            oversold = strategy['params']['oversold']
            overbought = strategy['params']['overbought']
            signals = np.where(rsi < oversold, 1, np.where(rsi > overbought, -1, 0))
            
        elif strategy['type'] == 'bollinger_bands':
            bb_upper = data['BB_Upper']
            bb_lower = data['BB_Lower']
            signals = np.where(data['Close'] < bb_lower, 1, np.where(data['Close'] > bb_upper, -1, 0))
        
        elif strategy['type'] == 'ma_rsi_combo':
            # Combined MA and RSI signals
            short_ma = data[f"MA_{strategy['params']['ma_short']}"]
            long_ma = data[f"MA_{strategy['params']['ma_long']}"]
            rsi = data['RSI']
            
            ma_signal = short_ma > long_ma
            rsi_oversold = rsi < strategy['params']['rsi_oversold']
            rsi_overbought = rsi > strategy['params']['rsi_overbought']
            
            signals = np.where(ma_signal & rsi_oversold, 1, np.where(~ma_signal | rsi_overbought, -1, 0))
        
        elif strategy['type'] == 'triple_ma':
            fast_ma = data[f"MA_{strategy['params']['fast']}"]
            medium_ma = data[f"MA_{strategy['params']['medium']}"]
            slow_ma = data[f"MA_{strategy['params']['slow']}"]
            
            signals = np.where((fast_ma > medium_ma) & (medium_ma > slow_ma), 1,
                             np.where((fast_ma < medium_ma) & (medium_ma < slow_ma), -1, 0))
        
        return pd.Series(signals, index=data.index)
    
    def _calculate_strategy_returns(self, data, signals):
        """Calculate strategy returns with transaction costs"""
        if len(signals) == 0:
            return None
        
        # Position changes (entry/exit points)
        positions = signals.diff().fillna(0)
        
        # Daily returns
        daily_returns = data['Close'].pct_change()
        
        # Strategy returns (assuming we follow signals)
        strategy_returns = daily_returns * signals.shift(1)  # Lag signals by 1 day
        
        # Apply transaction costs (0.1% per trade)
        transaction_costs = 0.001
        trade_costs = abs(positions) * transaction_costs
        strategy_returns = strategy_returns - trade_costs
        
        return strategy_returns.dropna()
    
    def _calculate_performance_metrics(self, returns, data):
        """Calculate comprehensive performance metrics with detailed calculations"""
        if len(returns) == 0 or returns.std() == 0:
            return None
        
        # Basic calculations
        total_return = (1 + returns).prod() - 1
        total_days = len(returns)
        annual_return = (1 + total_return) ** (252 / total_days) - 1
        
        # Risk metrics
        daily_mean = returns.mean()
        daily_std = returns.std()
        sharpe_ratio = (daily_mean * 252) / (daily_std * np.sqrt(252)) if daily_std > 0 else 0
        
        # Drawdown calculation
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Find peak and trough for max drawdown
        max_dd_idx = drawdown.idxmin()
        peak_idx = running_max[:max_dd_idx].idxmax()
        peak_value = cumulative_returns.loc[peak_idx] * 10000  # Assuming $10k initial
        trough_value = cumulative_returns.loc[max_dd_idx] * 10000
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Store detailed calculations
        calculation_details = {
            'total_days': total_days,
            'total_return': total_return,
            'mean_daily_return': daily_mean,
            'daily_return_std': daily_std,
            'peak_value': peak_value,
            'trough_value': trough_value,
            'cumulative_returns': cumulative_returns,
            'drawdown_series': drawdown
        }
        
        return {
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'calculation_details': calculation_details
        }
    
    def _analyze_trades(self, data, signals):
        """Analyze individual trades for win rate calculation"""
        # Find position changes
        positions = signals.diff().fillna(0)
        entry_points = positions[positions != 0].index
        
        if len(entry_points) < 2:
            return {'total_trades': 0, 'win_rate': 0, 'winning_trades': 0, 'losing_trades': 0}
        
        # Calculate trade returns
        trade_returns = []
        current_position = 0
        entry_price = 0
        
        for i, date in enumerate(data.index):
            if date in entry_points:
                if current_position != 0:  # Close existing position
                    exit_price = data.loc[date, 'Close']
                    trade_return = (exit_price - entry_price) / entry_price * current_position
                    trade_returns.append(trade_return)
                
                # Open new position
                current_position = signals.loc[date]
                entry_price = data.loc[date, 'Close']
        
        if len(trade_returns) == 0:
            return {'total_trades': 0, 'win_rate': 0, 'winning_trades': 0, 'losing_trades': 0}
        
        # Calculate win rate
        winning_trades = sum(1 for ret in trade_returns if ret > 0)
        losing_trades = sum(1 for ret in trade_returns if ret <= 0)
        total_trades = len(trade_returns)
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'trade_returns': trade_returns
        }
    
    def _calculate_annual_volatility(self):
        """Calculate annual volatility of the underlying asset"""
        daily_returns = self.data['Close'].pct_change().dropna()
        return daily_returns.std() * np.sqrt(252)
    
    def _get_period_analysis(self, strategy):
        """Analyze strategy performance across different time periods"""
        periods = {}
        
        # Split data into different periods for analysis
        data_length = len(self.data)
        
        if data_length > 504:  # At least 2 years
            # Recent 1 year
            recent_data = self.data.tail(252)
            recent_perf = self._backtest_strategy_on_data(strategy, recent_data)
            if recent_perf:
                periods['Recent Year'] = recent_perf
        
        if data_length > 756:  # At least 3 years
            # Middle period
            middle_data = self.data.iloc[-756:-252]
            middle_perf = self._backtest_strategy_on_data(strategy, middle_data)
            if middle_perf:
                periods['Previous Year'] = middle_perf
        
        return periods
    
    def _backtest_strategy_on_data(self, strategy, data):
        """Backtest strategy on specific data subset"""
        try:
            prepared_data = self._prepare_data_subset(data)
            signals = self._generate_signals(prepared_data, strategy)
            returns = self._calculate_strategy_returns(prepared_data, signals)
            performance = self._calculate_performance_metrics(returns, prepared_data)
            return performance
        except:
            return None
    
    def _prepare_data_subset(self, data):
        """Prepare subset of data with indicators"""
        # Similar to _prepare_data but for subset
        subset = data.copy()
        
        for period in [5, 10, 15, 20, 30, 50]:
            if len(subset) > period:
                subset[f'MA_{period}'] = subset['Close'].rolling(window=period).mean()
        
        subset['RSI'] = self._calculate_rsi_full(subset['Close'])
        subset['BB_Upper'], subset['BB_Lower'] = self.calculate_bollinger_bands(subset['Close'])
        
        return subset.dropna()
    
    def _monte_carlo_analysis(self, strategy):
        """Perform Monte Carlo analysis on strategy returns"""
        try:
            # Get strategy returns
            data = self._prepare_data()
            signals = self._generate_signals(data, strategy)
            returns = self._calculate_strategy_returns(data, signals)
            
            if returns is None or len(returns) < 50:
                return None
            
            # Monte Carlo simulation
            n_simulations = 1000
            annual_returns = []
            
            for _ in range(n_simulations):
                # Bootstrap sample returns
                simulated_returns = np.random.choice(returns.values, size=252, replace=True)
                annual_return = (1 + simulated_returns).prod() - 1
                annual_returns.append(annual_return)
            
            annual_returns = np.array(annual_returns)
            
            return {
                'ci_lower': np.percentile(annual_returns, 2.5),
                'ci_upper': np.percentile(annual_returns, 97.5),
                'prob_loss': np.sum(annual_returns < 0) / len(annual_returns) * 100,
                'var_5': np.percentile(annual_returns, 5)
            }
        except:
            return None
    
    def get_strategy_signals(self, strategy):
        """Get detailed signals for visualization"""
        data = self._prepare_data()
        signals = self._generate_signals(data, strategy)
        
        # Find entry and exit points
        position_changes = signals.diff().fillna(0)
        entry_points = data[position_changes > 0]['Close']  # Buy signals
        exit_points = data[position_changes < 0]['Close']   # Sell signals
        
        return {
            'entry_points': entry_points,
            'exit_points': exit_points,
            'all_signals': signals
        }
    
    def backtest_strategy(self, strategy):
        """Get detailed backtesting results for visualization"""
        data = self._prepare_data()
        signals = self._generate_signals(data, strategy)
        returns = self._calculate_strategy_returns(data, signals)
        
        if returns is None:
            return {}
        
        # Equity curve
        equity_curve = (1 + returns).cumprod() * 10000  # Assume $10k starting
        
        # Benchmark (buy and hold)
        benchmark_returns = data['Close'].pct_change().dropna()
        benchmark_equity = (1 + benchmark_returns).cumprod() * 10000
        
        # Monthly returns for heatmap
        monthly_returns = self._calculate_monthly_returns(returns)
        
        return {
            'equity_curve': equity_curve,
            'benchmark': benchmark_equity,
            'monthly_returns': monthly_returns,
            'daily_returns': returns
        }
    
    def _calculate_monthly_returns(self, returns):
        """Calculate monthly returns for heatmap visualization"""
        monthly_data = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_data.index = monthly_data.index.to_period('M')
        
        # Create year-month matrix
        years = monthly_data.index.year.unique()
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        heatmap_data = pd.DataFrame(index=years, columns=months)
        
        for date, return_val in monthly_data.items():
            year = date.year
            month = months[date.month - 1]
            heatmap_data.loc[year, month] = return_val * 100  # Convert to percentage
        
        return heatmap_data.fillna(0)
    
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
            # Work on a copy to avoid modifying original data
            temp_data = data.copy()
            
            # Calculate signals
            temp_data[f'ma_fast'] = temp_data['Close'].rolling(fast_ma).mean()
            temp_data[f'ma_slow'] = temp_data['Close'].rolling(slow_ma).mean()
            
            # Generate signals
            temp_data['signal'] = 0
            temp_data.loc[temp_data['ma_fast'] > temp_data['ma_slow'], 'signal'] = 1
            temp_data.loc[temp_data['ma_fast'] <= temp_data['ma_slow'], 'signal'] = -1
            
            # Calculate returns
            temp_data['strategy_returns'] = temp_data['signal'].shift(1) * temp_data['Close'].pct_change()
            strategy_returns = temp_data['strategy_returns'].dropna()
            
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
            # Work on a copy to avoid modifying original data
            temp_data = data.copy()
            rsi = self.calculate_rsi(temp_data['Close'], period)
            
            # Generate signals
            temp_data['signal'] = 0
            temp_data.loc[rsi < oversold, 'signal'] = 1  # Buy oversold
            temp_data.loc[rsi > overbought, 'signal'] = -1  # Sell overbought
            
            # Calculate returns
            temp_data['strategy_returns'] = temp_data['signal'].shift(1) * temp_data['Close'].pct_change()
            strategy_returns = temp_data['strategy_returns'].dropna()
            
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