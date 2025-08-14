import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from scipy import stats
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import coint, adfuller
import warnings
warnings.filterwarnings('ignore')

class PairsTradingOptimizer:
    """
    AI-powered pairs trading optimizer that finds cointegrated pairs
    and applies optimized trading strategies to them.
    """
    
    def __init__(self, base_ticker):
        self.base_ticker = base_ticker
        self.pairs_data = {}
        self.cointegration_results = {}
        self.optimization_log = []
        
    def find_optimal_pairs(self, candidate_tickers=None, lookback_days=504):
        """Find the best pairs for trading with the base ticker"""
        
        if candidate_tickers is None:
            # Define comprehensive universe of potential pairs
            candidate_tickers = self._get_candidate_universe()
        
        self.optimization_log.append(f"Starting pairs analysis for {self.base_ticker}")
        self.optimization_log.append(f"Testing {len(candidate_tickers)} potential pairs")
        
        # Fetch data for all tickers
        all_tickers = [self.base_ticker] + candidate_tickers
        data = self._fetch_pairs_data(all_tickers, lookback_days)
        
        if data is None or len(data) < 100:
            return None
        
        # Analyze cointegration for each pair
        pair_results = []
        base_prices = data[self.base_ticker].dropna()
        
        for candidate in candidate_tickers:
            if candidate in data.columns:
                candidate_prices = data[candidate].dropna()
                
                # Align dates
                common_dates = base_prices.index.intersection(candidate_prices.index)
                if len(common_dates) < 100:
                    continue
                
                base_aligned = base_prices[common_dates]
                candidate_aligned = candidate_prices[common_dates]
                
                # Perform cointegration analysis
                pair_analysis = self._analyze_pair_cointegration(
                    base_aligned, candidate_aligned, self.base_ticker, candidate
                )
                
                if pair_analysis:
                    pair_results.append(pair_analysis)
                    self.optimization_log.append(f"Analyzed pair: {self.base_ticker}-{candidate}")
        
        # Sort pairs by trading potential (combination of cointegration strength and spread volatility)
        pair_results.sort(key=lambda x: x['trading_score'], reverse=True)
        
        # Store results
        self.cointegration_results = {
            'pairs': pair_results[:10],  # Top 10 pairs
            'base_ticker': self.base_ticker,
            'analysis_date': datetime.now(),
            'total_pairs_tested': len(candidate_tickers),
            'optimization_log': self.optimization_log
        }
        
        return self.cointegration_results
    
    def _get_candidate_universe(self):
        """Define universe of potential pair candidates based on base ticker"""
        
        # Sector-based candidates
        sector_candidates = {
            # Technology
            'AAPL': ['MSFT', 'GOOGL', 'META', 'NVDA', 'TSLA', 'NFLX', 'ADBE', 'CRM', 'ORCL', 'IBM'],
            'MSFT': ['AAPL', 'GOOGL', 'META', 'NVDA', 'ORCL', 'IBM', 'ADBE', 'CRM', 'INTC', 'AMD'],
            'GOOGL': ['AAPL', 'MSFT', 'META', 'NVDA', 'NFLX', 'ADBE', 'CRM', 'AMZN', 'TSLA'],
            
            # Financial
            'JPM': ['BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'TFC', 'COF', 'AXP'],
            'BAC': ['JPM', 'WFC', 'C', 'USB', 'PNC', 'TFC', 'COF', 'RF', 'KEY', 'FITB'],
            
            # Healthcare
            'JNJ': ['PFE', 'UNH', 'ABT', 'MRK', 'TMO', 'DHR', 'BMY', 'AMGN', 'GILD', 'MDT'],
            'PFE': ['JNJ', 'MRK', 'ABT', 'BMY', 'LLY', 'AMGN', 'GILD', 'BIIB', 'CELG'],
            
            # Energy
            'XOM': ['CVX', 'COP', 'EOG', 'SLB', 'MPC', 'VLO', 'PSX', 'OXY', 'HAL', 'BKR'],
            'CVX': ['XOM', 'COP', 'EOG', 'SLB', 'MPC', 'VLO', 'PSX', 'OXY', 'HAL'],
            
            # Consumer
            'KO': ['PEP', 'PG', 'WMT', 'TGT', 'HD', 'MCD', 'NKE', 'SBUX', 'DIS'],
            'PEP': ['KO', 'PG', 'WMT', 'COST', 'MCD', 'SBUX', 'MNST', 'KHC'],
            
            # ETFs and Indices
            'SPY': ['QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'IVV', 'VEA', 'VWO', 'EEM', 'GLD'],
            'QQQ': ['SPY', 'IWM', 'DIA', 'VTI', 'XLK', 'TQQQ', 'SOXL', 'FDN'],
        }
        
        # Get sector-specific candidates
        candidates = sector_candidates.get(self.base_ticker, [])
        
        # Add general market candidates if specific sector not found
        if not candidates:
            candidates = [
                'SPY', 'QQQ', 'IWM', 'DIA', 'VTI',  # Broad market
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',  # Mega cap tech
                'JPM', 'BAC', 'WFC', 'GS',  # Financials
                'JNJ', 'PFE', 'UNH', 'MRK',  # Healthcare
                'XOM', 'CVX', 'COP',  # Energy
                'KO', 'PEP', 'PG', 'WMT'  # Consumer
            ]
        
        # Remove base ticker if it appears in candidates
        candidates = [ticker for ticker in candidates if ticker != self.base_ticker]
        
        return candidates[:20]  # Limit to 20 candidates for performance
    
    def _fetch_pairs_data(self, tickers, lookback_days):
        """Fetch historical data for all tickers"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            data = yf.download(tickers, start=start_date, end=end_date)['Close']
            
            # Handle single ticker case
            if len(tickers) == 1:
                data = pd.DataFrame(data, columns=tickers)
            
            return data.dropna()
            
        except Exception as e:
            self.optimization_log.append(f"Error fetching data: {str(e)}")
            return None
    
    def _analyze_pair_cointegration(self, price1, price2, ticker1, ticker2):
        """Perform comprehensive cointegration analysis"""
        try:
            # Cointegration test
            coint_stat, p_value, critical_values = coint(price1, price2)
            
            # Linear regression for spread calculation
            X = price2.values.reshape(-1, 1)
            y = price1.values
            reg = LinearRegression().fit(X, y)
            hedge_ratio = reg.coef_[0]
            
            # Calculate spread
            spread = price1 - hedge_ratio * price2
            
            # Spread stationarity test
            adf_stat, adf_p_value, _, _, adf_critical, _ = adfuller(spread.dropna())
            
            # Spread statistics
            spread_mean = spread.mean()
            spread_std = spread.std()
            spread_zscore = (spread.iloc[-1] - spread_mean) / spread_std
            
            # Half-life of mean reversion
            half_life = self._calculate_half_life(spread)
            
            # Trading potential score
            trading_score = self._calculate_trading_score(
                p_value, adf_p_value, spread_std, half_life, abs(spread_zscore)
            )
            
            # Correlation analysis
            correlation = price1.corr(price2)
            
            return {
                'pair': f"{ticker1}-{ticker2}",
                'ticker1': ticker1,
                'ticker2': ticker2,
                'cointegration_pvalue': p_value,
                'cointegration_stat': coint_stat,
                'hedge_ratio': hedge_ratio,
                'spread_mean': spread_mean,
                'spread_std': spread_std,
                'current_zscore': spread_zscore,
                'adf_pvalue': adf_p_value,
                'adf_statistic': adf_stat,
                'half_life': half_life,
                'correlation': correlation,
                'trading_score': trading_score,
                'spread_series': spread,
                'is_cointegrated': p_value < 0.05 and adf_p_value < 0.05,
                'signal_strength': self._get_signal_strength(spread_zscore)
            }
            
        except Exception as e:
            self.optimization_log.append(f"Error analyzing {ticker1}-{ticker2}: {str(e)}")
            return None
    
    def _calculate_half_life(self, spread):
        """Calculate half-life of mean reversion"""
        try:
            spread_lag = spread.shift(1).dropna()
            spread_diff = spread.diff().dropna()
            
            # Align series
            aligned_spread = spread_lag[spread_diff.index]
            
            if len(aligned_spread) < 10:
                return np.inf
            
            # Regression: Δy = α + βy_{t-1} + ε
            X = aligned_spread.values.reshape(-1, 1)
            y = spread_diff.values
            
            reg = LinearRegression().fit(X, y)
            beta = reg.coef_[0]
            
            if beta >= 0:
                return np.inf
            
            half_life = -np.log(2) / beta
            return half_life if half_life > 0 else np.inf
            
        except:
            return np.inf
    
    def _calculate_trading_score(self, coint_p, adf_p, spread_vol, half_life, zscore):
        """Calculate overall trading potential score"""
        score = 0
        
        # Cointegration strength (higher score for lower p-value)
        if coint_p < 0.01:
            score += 30
        elif coint_p < 0.05:
            score += 20
        elif coint_p < 0.10:
            score += 10
        
        # Spread stationarity (higher score for stationary spread)
        if adf_p < 0.01:
            score += 25
        elif adf_p < 0.05:
            score += 15
        elif adf_p < 0.10:
            score += 5
        
        # Spread volatility (optimal range for trading)
        if 0.02 <= spread_vol <= 0.08:
            score += 20
        elif 0.01 <= spread_vol <= 0.12:
            score += 10
        
        # Half-life (prefer 5-60 days)
        if 5 <= half_life <= 60:
            score += 15
        elif 1 <= half_life <= 120:
            score += 10
        elif half_life == np.inf:
            score -= 20
        
        # Current signal strength
        if abs(zscore) > 2:
            score += 10
        elif abs(zscore) > 1.5:
            score += 5
        
        return max(0, score)
    
    def _get_signal_strength(self, zscore):
        """Determine current trading signal strength"""
        abs_zscore = abs(zscore)
        
        if abs_zscore > 2.5:
            return "Very Strong"
        elif abs_zscore > 2.0:
            return "Strong"
        elif abs_zscore > 1.5:
            return "Moderate"
        elif abs_zscore > 1.0:
            return "Weak"
        else:
            return "No Signal"
    
    def generate_pairs_strategy(self, pair_data, strategy_params=None):
        """Generate optimized pairs trading strategy"""
        
        if strategy_params is None:
            strategy_params = {
                'entry_threshold': 2.0,
                'exit_threshold': 0.5,
                'stop_loss': 3.0,
                'lookback_window': 20
            }
        
        spread = pair_data['spread_series']
        
        # Calculate rolling statistics
        rolling_mean = spread.rolling(window=strategy_params['lookback_window']).mean()
        rolling_std = spread.rolling(window=strategy_params['lookback_window']).std()
        zscore = (spread - rolling_mean) / rolling_std
        
        # Generate trading signals
        signals = pd.Series(0, index=spread.index)
        
        # Entry signals
        long_entry = zscore < -strategy_params['entry_threshold']
        short_entry = zscore > strategy_params['entry_threshold']
        
        # Exit signals
        long_exit = zscore > -strategy_params['exit_threshold']
        short_exit = zscore < strategy_params['exit_threshold']
        
        # Stop loss
        long_stop = zscore < -strategy_params['stop_loss']
        short_stop = zscore > strategy_params['stop_loss']
        
        # Combine signals
        signals[long_entry] = 1   # Buy spread (long ticker1, short ticker2)
        signals[short_entry] = -1 # Sell spread (short ticker1, long ticker2)
        signals[long_exit | long_stop] = 0
        signals[short_exit | short_stop] = 0
        
        # Forward fill signals to maintain positions
        signals = signals.replace(0, np.nan).fillna(method='ffill').fillna(0)
        
        return {
            'signals': signals,
            'zscore': zscore,
            'spread': spread,
            'rolling_mean': rolling_mean,
            'rolling_std': rolling_std,
            'strategy_params': strategy_params
        }
    
    def backtest_pairs_strategy(self, pair_data, strategy_results):
        """Backtest pairs trading strategy"""
        
        signals = strategy_results['signals']
        spread = strategy_results['spread']
        
        # Calculate spread returns
        spread_returns = spread.pct_change().fillna(0)
        
        # Strategy returns (lag signals by 1 period)
        strategy_returns = spread_returns * signals.shift(1)
        
        # Remove first row (NaN from shift)
        strategy_returns = strategy_returns.dropna()
        
        if len(strategy_returns) == 0:
            return None
        
        # Performance metrics
        total_return = (1 + strategy_returns).prod() - 1
        annual_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
        
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Drawdown analysis
        cumulative = (1 + strategy_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Trade analysis
        position_changes = signals.diff().fillna(0)
        trades = position_changes[position_changes != 0]
        total_trades = len(trades)
        
        # Win rate calculation
        trade_returns = []
        current_pos = 0
        entry_spread = 0
        
        for date, signal in signals.items():
            if signal != current_pos:  # Position change
                if current_pos != 0:  # Close existing position
                    exit_spread = spread.loc[date]
                    trade_return = (exit_spread - entry_spread) * current_pos
                    trade_returns.append(trade_return)
                
                if signal != 0:  # Open new position
                    entry_spread = spread.loc[date]
                    current_pos = signal
        
        winning_trades = sum(1 for ret in trade_returns if ret > 0)
        win_rate = (winning_trades / len(trade_returns)) * 100 if trade_returns else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'strategy_returns': strategy_returns,
            'equity_curve': cumulative,
            'drawdown_series': drawdown
        }