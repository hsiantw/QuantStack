import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.rolling import RollingOLS
import warnings
warnings.filterwarnings('ignore')

class MeanReversionStrategy:
    """
    Comprehensive mean reversion strategy implementation with multiple techniques
    for pairs trading and single asset mean reversion analysis.
    """
    
    def __init__(self, data, lookback_window=20, confidence_level=0.95):
        self.data = data.copy()
        self.lookback_window = lookback_window
        self.confidence_level = confidence_level
        self.signals = {}
        self.strategy_components = {}
        self.performance_metrics = {}
        
    def bollinger_bands_reversion(self, price_series, window=20, num_std=2):
        """Bollinger Bands mean reversion strategy"""
        
        rolling_mean = price_series.rolling(window=window).mean()
        rolling_std = price_series.rolling(window=window).std()
        
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        
        # Generate signals
        signals = pd.Series(0, index=price_series.index)
        
        # Buy when price touches lower band (oversold)
        signals[price_series <= lower_band] = 1
        
        # Sell when price touches upper band (overbought)
        signals[price_series >= upper_band] = -1
        
        # Exit when price returns to middle band
        middle_crosses = np.where(
            (price_series.shift(1) < rolling_mean) & (price_series >= rolling_mean) |
            (price_series.shift(1) > rolling_mean) & (price_series <= rolling_mean)
        )[0]
        
        signals.iloc[middle_crosses] = 0
        
        # Forward fill signals to maintain positions
        signals = signals.replace(0, np.nan).ffill().fillna(0)
        
        return {
            'signals': signals,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'middle_band': rolling_mean,
            'bandwidth': (upper_band - lower_band) / rolling_mean
        }
    
    def rsi_reversion(self, price_series, window=14, oversold=30, overbought=70):
        """RSI-based mean reversion strategy"""
        
        delta = price_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Generate signals
        signals = pd.Series(0, index=price_series.index)
        
        # Buy when RSI is oversold
        signals[rsi <= oversold] = 1
        
        # Sell when RSI is overbought
        signals[rsi >= overbought] = -1
        
        # Exit when RSI returns to neutral (45-55 range)
        neutral_zone = (rsi >= 45) & (rsi <= 55)
        signals[neutral_zone] = 0
        
        # Forward fill signals
        signals = signals.replace(0, np.nan).ffill().fillna(0)
        
        return {
            'signals': signals,
            'rsi': rsi,
            'oversold_level': oversold,
            'overbought_level': overbought
        }
    
    def ornstein_uhlenbeck_reversion(self, price_series, window=60):
        """Ornstein-Uhlenbeck process for mean reversion detection"""
        
        log_prices = np.log(price_series)
        
        # Estimate OU parameters using rolling regression
        signals = pd.Series(0, index=price_series.index)
        ou_signals = []
        half_lives = []
        equilibrium_prices = []
        
        for i in range(window, len(log_prices)):
            y = log_prices.iloc[i-window:i].values
            x = np.arange(len(y)).reshape(-1, 1)
            
            # Linear regression to estimate drift
            reg = LinearRegression().fit(x, y)
            
            # Calculate residuals
            residuals = y - reg.predict(x)
            
            # Estimate mean reversion parameters
            lagged_residuals = residuals[:-1]
            current_residuals = residuals[1:]
            
            if len(lagged_residuals) > 10:
                # AR(1) model: r_t = a + b*r_{t-1} + e_t
                X = lagged_residuals.reshape(-1, 1)
                reg_ar = LinearRegression().fit(X, current_residuals)
                
                beta = reg_ar.coef_[0]
                alpha = reg_ar.intercept_
                
                # Mean reversion speed and equilibrium
                if beta < 0:
                    mean_reversion_speed = -np.log(1 + beta)
                    half_life = np.log(2) / mean_reversion_speed if mean_reversion_speed > 0 else np.inf
                    equilibrium = alpha / (1 - beta)
                    
                    # Current deviation from equilibrium
                    current_residual = residuals[-1]
                    deviation = current_residual - equilibrium
                    
                    # Generate signal based on deviation
                    threshold = np.std(residuals) * 1.5
                    
                    if deviation > threshold:
                        ou_signals.append(-1)  # Sell (revert down)
                    elif deviation < -threshold:
                        ou_signals.append(1)   # Buy (revert up)
                    else:
                        ou_signals.append(0)   # Hold
                    
                    half_lives.append(half_life)
                    equilibrium_prices.append(np.exp(log_prices.iloc[i-1] + equilibrium))
                else:
                    ou_signals.append(0)
                    half_lives.append(np.inf)
                    equilibrium_prices.append(price_series.iloc[i])
            else:
                ou_signals.append(0)
                half_lives.append(np.inf)
                equilibrium_prices.append(price_series.iloc[i])
        
        # Align signals with price series
        signals.iloc[window:] = ou_signals
        
        return {
            'signals': signals,
            'half_lives': pd.Series(half_lives, index=price_series.index[window:]),
            'equilibrium_prices': pd.Series(equilibrium_prices, index=price_series.index[window:])
        }
    
    def kalman_filter_reversion(self, price_series, process_variance=1e-5, measurement_variance=1e-1):
        """Kalman filter for dynamic mean estimation and reversion signals"""
        
        n = len(price_series)
        
        # Initialize Kalman filter parameters
        x_hat = np.zeros(n)  # State estimate (true price)
        P = np.zeros(n)      # Error covariance
        x_hat[0] = price_series.iloc[0]
        P[0] = 1.0
        
        # Kalman filter process
        for i in range(1, n):
            # Prediction step
            x_hat_minus = x_hat[i-1]  # Predicted state
            P_minus = P[i-1] + process_variance  # Predicted error covariance
            
            # Update step
            K = P_minus / (P_minus + measurement_variance)  # Kalman gain
            x_hat[i] = x_hat_minus + K * (price_series.iloc[i] - x_hat_minus)
            P[i] = (1 - K) * P_minus
        
        # Generate mean reversion signals
        kalman_mean = pd.Series(x_hat, index=price_series.index)
        deviation = price_series - kalman_mean
        rolling_std = deviation.rolling(window=20).std()
        
        signals = pd.Series(0, index=price_series.index)
        
        # Signal generation based on deviation from Kalman estimate
        signals[deviation > 2 * rolling_std] = -1  # Sell (price above fair value)
        signals[deviation < -2 * rolling_std] = 1  # Buy (price below fair value)
        signals[abs(deviation) < 0.5 * rolling_std] = 0  # Exit near fair value
        
        # Forward fill signals
        signals = signals.replace(0, np.nan).ffill().fillna(0)
        
        return {
            'signals': signals,
            'kalman_mean': kalman_mean,
            'deviation': deviation,
            'confidence_bands': rolling_std
        }
    
    def statistical_arbitrage_signals(self, price1, price2, lookback=60):
        """Statistical arbitrage signals for pairs trading"""
        
        # Ensure same length
        common_index = price1.index.intersection(price2.index)
        p1 = price1[common_index]
        p2 = price2[common_index]
        
        # Rolling cointegration and spread analysis
        signals = pd.Series(0, index=common_index)
        spreads = []
        hedge_ratios = []
        half_lives = []
        
        for i in range(lookback, len(p1)):
            window_p1 = p1.iloc[i-lookback:i]
            window_p2 = p2.iloc[i-lookback:i]
            
            # Calculate hedge ratio using linear regression
            X = window_p2.values.reshape(-1, 1)
            y = window_p1.values
            reg = LinearRegression().fit(X, y)
            hedge_ratio = reg.coef_[0]
            
            # Calculate spread
            spread = p1.iloc[i] - hedge_ratio * p2.iloc[i]
            spreads.append(spread)
            hedge_ratios.append(hedge_ratio)
            
            # Calculate spread statistics
            historical_spreads = window_p1 - hedge_ratio * window_p2
            spread_mean = historical_spreads.mean()
            spread_std = historical_spreads.std()
            
            # Half-life estimation
            spread_diff = historical_spreads.diff().dropna()
            spread_lag = historical_spreads.shift(1).dropna()
            
            if len(spread_lag) > 10:
                try:
                    # AR(1) regression for half-life
                    X_hl = spread_lag.values.reshape(-1, 1)
                    y_hl = spread_diff.values
                    reg_hl = LinearRegression().fit(X_hl, y_hl)
                    beta = reg_hl.coef_[0]
                    
                    if beta < 0:
                        half_life = -np.log(2) / beta
                    else:
                        half_life = np.inf
                except:
                    half_life = np.inf
            else:
                half_life = np.inf
            
            half_lives.append(half_life)
            
            # Generate signal based on Z-score
            if spread_std > 0:
                z_score = (spread - spread_mean) / spread_std
                
                # Entry signals
                if z_score > 2.0:
                    signals.iloc[i] = -1  # Sell spread (short p1, long p2)
                elif z_score < -2.0:
                    signals.iloc[i] = 1   # Buy spread (long p1, short p2)
                elif abs(z_score) < 0.5:
                    signals.iloc[i] = 0   # Exit position
        
        # Forward fill signals
        signals = signals.replace(0, np.nan).ffill().fillna(0)
        
        return {
            'signals': signals,
            'spreads': pd.Series(spreads, index=common_index[lookback:]),
            'hedge_ratios': pd.Series(hedge_ratios, index=common_index[lookback:]),
            'half_lives': pd.Series(half_lives, index=common_index[lookback:])
        }
    
    def ensemble_mean_reversion(self, price_series, weights=None):
        """Ensemble of multiple mean reversion strategies"""
        
        if weights is None:
            weights = {
                'bollinger': 0.25,
                'rsi': 0.25,
                'ou': 0.25,
                'kalman': 0.25
            }
        
        # Get signals from each strategy
        bb_results = self.bollinger_bands_reversion(price_series)
        rsi_results = self.rsi_reversion(price_series)
        ou_results = self.ornstein_uhlenbeck_reversion(price_series)
        kalman_results = self.kalman_filter_reversion(price_series)
        
        # Combine signals using weighted average
        ensemble_signal = (
            weights['bollinger'] * bb_results['signals'] +
            weights['rsi'] * rsi_results['signals'] +
            weights['ou'] * ou_results['signals'] +
            weights['kalman'] * kalman_results['signals']
        )
        
        # Convert to discrete signals
        final_signals = pd.Series(0, index=price_series.index)
        final_signals[ensemble_signal > 0.5] = 1   # Strong buy
        final_signals[ensemble_signal < -0.5] = -1 # Strong sell
        final_signals[abs(ensemble_signal) <= 0.5] = 0  # Hold/Exit
        
        # Forward fill
        final_signals = final_signals.replace(0, np.nan).ffill().fillna(0)
        
        return {
            'ensemble_signals': final_signals,
            'component_signals': {
                'bollinger': bb_results['signals'],
                'rsi': rsi_results['signals'],
                'ou': ou_results['signals'],
                'kalman': kalman_results['signals']
            },
            'signal_strength': abs(ensemble_signal),
            'component_results': {
                'bollinger': bb_results,
                'rsi': rsi_results,
                'ou': ou_results,
                'kalman': kalman_results
            }
        }
    
    def adaptive_mean_reversion(self, price_series, regime_lookback=252):
        """Adaptive mean reversion based on market regime detection"""
        
        returns = price_series.pct_change().dropna()
        
        # Regime detection using rolling volatility
        rolling_vol = returns.rolling(window=20).std()
        long_term_vol = rolling_vol.rolling(window=regime_lookback).mean()
        
        # Define regimes
        low_vol_regime = rolling_vol < (long_term_vol * 0.8)
        high_vol_regime = rolling_vol > (long_term_vol * 1.2)
        normal_regime = ~(low_vol_regime | high_vol_regime)
        
        # Adaptive parameters based on regime
        signals = pd.Series(0, index=price_series.index)
        
        for i in range(regime_lookback, len(price_series)):
            current_regime = None
            
            if low_vol_regime.iloc[i]:
                # Low volatility: more aggressive mean reversion
                bb_params = {'window': 15, 'num_std': 1.5}
                rsi_params = {'window': 10, 'oversold': 35, 'overbought': 65}
                current_regime = 'low_vol'
            elif high_vol_regime.iloc[i]:
                # High volatility: more conservative
                bb_params = {'window': 30, 'num_std': 2.5}
                rsi_params = {'window': 20, 'oversold': 25, 'overbought': 75}
                current_regime = 'high_vol'
            else:
                # Normal volatility: standard parameters
                bb_params = {'window': 20, 'num_std': 2.0}
                rsi_params = {'window': 14, 'oversold': 30, 'overbought': 70}
                current_regime = 'normal'
            
            # Apply strategy with adaptive parameters
            window_data = price_series.iloc[max(0, i-60):i+1]
            
            if len(window_data) > bb_params['window']:
                bb_result = self.bollinger_bands_reversion(window_data, **bb_params)
                rsi_result = self.rsi_reversion(window_data, **rsi_params)
                
                # Combine signals with regime-specific weights
                if current_regime == 'low_vol':
                    signal = 0.6 * bb_result['signals'].iloc[-1] + 0.4 * rsi_result['signals'].iloc[-1]
                elif current_regime == 'high_vol':
                    signal = 0.3 * bb_result['signals'].iloc[-1] + 0.7 * rsi_result['signals'].iloc[-1]
                else:
                    signal = 0.5 * bb_result['signals'].iloc[-1] + 0.5 * rsi_result['signals'].iloc[-1]
                
                signals.iloc[i] = 1 if signal > 0.5 else (-1 if signal < -0.5 else 0)
        
        return {
            'adaptive_signals': signals,
            'regime_indicators': {
                'low_vol': low_vol_regime,
                'high_vol': high_vol_regime,
                'normal': normal_regime
            },
            'rolling_volatility': rolling_vol,
            'regime_threshold': long_term_vol
        }
    
    def backtest_strategy(self, price_series, signals, transaction_cost=0.001):
        """Comprehensive backtesting for mean reversion strategies"""
        
        returns = price_series.pct_change().dropna()
        
        # Align signals with returns
        aligned_signals = signals.reindex(returns.index, method='ffill').fillna(0)
        
        # Calculate strategy returns
        strategy_returns = returns * aligned_signals.shift(1)
        
        # Apply transaction costs
        position_changes = aligned_signals.diff().abs()
        transaction_costs = position_changes * transaction_cost
        strategy_returns_net = strategy_returns - transaction_costs.reindex(strategy_returns.index, fill_value=0)
        
        # Performance metrics
        total_return = (1 + strategy_returns_net).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(strategy_returns_net)) - 1
        
        volatility = strategy_returns_net.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Drawdown analysis
        cumulative = (1 + strategy_returns_net).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Trade analysis
        trades = []
        position = 0
        entry_price = 0
        entry_date = None
        
        for date, signal in aligned_signals.items():
            if signal != position:
                if position != 0 and entry_date is not None:  # Close existing position
                    try:
                        exit_price = price_series[date]
                        trade_return = (exit_price - entry_price) / entry_price * position
                        trades.append({
                            'entry_date': entry_date,
                            'exit_date': date,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'position': position,
                            'return': trade_return,
                            'duration': (date - entry_date).days
                        })
                    except KeyError:
                        # Skip if price not available for this date
                        pass
                
                if signal != 0:  # Open new position
                    try:
                        entry_date = date
                        entry_price = price_series[date]
                        position = signal
                    except KeyError:
                        # Skip if price not available for this date
                        pass
        
        # Trade statistics
        if trades:
            trade_returns = [trade['return'] for trade in trades]
            winning_trades = [r for r in trade_returns if r > 0]
            losing_trades = [r for r in trade_returns if r <= 0]
            
            win_rate = len(winning_trades) / len(trades) * 100
            avg_win = np.mean(winning_trades) if winning_trades else 0
            avg_loss = np.mean(losing_trades) if losing_trades else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(trades),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'strategy_returns': strategy_returns_net,
            'equity_curve': cumulative,
            'drawdown_series': drawdown,
            'trades': trades
        }