"""
Tooltip definitions and formulas for financial terms used in the platform.
Provides educational context for complex quantitative finance concepts.
"""

FINANCIAL_TOOLTIPS = {
    # Portfolio Optimization
    "sharpe_ratio": {
        "definition": "Risk-adjusted return metric that measures excess return per unit of risk",
        "formula": "Sharpe Ratio = (Portfolio Return - Risk-free Rate) / Portfolio Standard Deviation",
        "interpretation": "Higher values indicate better risk-adjusted performance. Values above 1.0 are considered good, above 2.0 excellent."
    },
    
    "efficient_frontier": {
        "definition": "The set of optimal portfolios offering the highest expected return for each level of risk",
        "formula": "Maximize: E(R) - (λ/2) × σ² subject to Σwᵢ = 1",
        "interpretation": "Portfolios on the frontier are considered efficient. Any portfolio below the frontier is sub-optimal."
    },
    
    "value_at_risk": {
        "definition": "Maximum potential loss over a specific time period at a given confidence level",
        "formula": "VaR = μ - z × σ (for normal distribution)",
        "interpretation": "95% VaR of $1M means 5% chance of losing more than $1M in the specified period."
    },
    
    "expected_shortfall": {
        "definition": "Expected loss given that the loss exceeds the VaR threshold",
        "formula": "ES = E[Loss | Loss > VaR]",
        "interpretation": "Also called Conditional VaR. Provides information about tail risk beyond VaR."
    },
    
    "beta": {
        "definition": "Measure of systematic risk relative to the market",
        "formula": "β = Cov(Rᵢ, Rₘ) / Var(Rₘ)",
        "interpretation": "β = 1: moves with market, β > 1: more volatile than market, β < 1: less volatile than market."
    },
    
    # Statistical Arbitrage
    "cointegration": {
        "definition": "Long-term equilibrium relationship between two or more time series",
        "formula": "Y = αX + β + ε, where ε is stationary",
        "interpretation": "If two assets are cointegrated, their spread tends to revert to the mean over time."
    },
    
    "augmented_dickey_fuller": {
        "definition": "Statistical test for unit root presence (non-stationarity) in time series",
        "formula": "Δyₜ = α + βt + γyₜ₋₁ + δ₁Δyₜ₋₁ + ... + δₚΔyₜ₋ₚ + εₜ",
        "interpretation": "p-value < 0.05 typically indicates stationarity (reject null hypothesis of unit root)."
    },
    
    "z_score": {
        "definition": "Standardized measure of how far a value deviates from the mean",
        "formula": "Z = (X - μ) / σ",
        "interpretation": "Values beyond ±2 are considered statistically significant departures from the mean."
    },
    
    "half_life": {
        "definition": "Time required for mean reversion process to decay by half",
        "formula": "Half-life = -ln(2) / ln(1 + λ), where λ is mean reversion speed",
        "interpretation": "Shorter half-life indicates faster mean reversion, useful for trading frequency decisions."
    },
    
    # Time Series Analysis
    "arima": {
        "definition": "AutoRegressive Integrated Moving Average model for time series forecasting",
        "formula": "ARIMA(p,d,q): (1-φ₁L-...-φₚLᵖ)(1-L)ᵈXₜ = (1+θ₁L+...+θₙLᵩ)εₜ",
        "interpretation": "p: autoregressive terms, d: differencing degree, q: moving average terms."
    },
    
    "autocorrelation": {
        "definition": "Correlation of a time series with lagged versions of itself",
        "formula": "ACF(k) = Cov(Xₜ, Xₜ₊ₖ) / Var(Xₜ)",
        "interpretation": "Helps identify patterns and seasonality. Values near 0 indicate no correlation."
    },
    
    "stationarity": {
        "definition": "Statistical property where mean, variance, and autocorrelation remain constant over time",
        "formula": "E[Xₜ] = μ, Var[Xₜ] = σ², Cov[Xₜ, Xₜ₊ₖ] = γₖ (all constant)",
        "interpretation": "Required for many time series models. Non-stationary data often needs differencing."
    },
    
    # Trading Strategies
    "rsi": {
        "definition": "Relative Strength Index - momentum oscillator measuring speed and change of price movements",
        "formula": "RSI = 100 - (100 / (1 + RS)), where RS = Average Gain / Average Loss",
        "interpretation": "Values > 70 suggest overbought conditions, < 30 suggest oversold conditions."
    },
    
    "bollinger_bands": {
        "definition": "Volatility bands placed above and below a moving average",
        "formula": "Upper Band = MA + (k × σ), Lower Band = MA - (k × σ)",
        "interpretation": "Price touching bands may indicate overbought/oversold conditions. k typically = 2."
    },
    
    "moving_average": {
        "definition": "Trend-following indicator that smooths price data by creating a constantly updated average",
        "formula": "SMA = (P₁ + P₂ + ... + Pₙ) / n",
        "interpretation": "Crossovers between short and long MA often signal trend changes."
    },
    
    "maximum_drawdown": {
        "definition": "Largest peak-to-trough decline in portfolio value over a specific period",
        "formula": "MDD = (Trough Value - Peak Value) / Peak Value",
        "interpretation": "Measures worst-case loss scenario. Lower values indicate better risk management."
    },
    
    "win_rate": {
        "definition": "Percentage of profitable trades out of total trades executed",
        "formula": "Win Rate = (Number of Winning Trades / Total Trades) × 100%",
        "interpretation": "Higher win rates don't always mean better strategies - consider profit/loss ratio too."
    },
    
    "calmar_ratio": {
        "definition": "Risk-adjusted return metric comparing annual return to maximum drawdown",
        "formula": "Calmar Ratio = Annual Return / |Maximum Drawdown|",
        "interpretation": "Higher values indicate better risk-adjusted performance. Above 1.0 is generally good."
    },
    
    # AI Models
    "random_forest": {
        "definition": "Ensemble learning method using multiple decision trees for prediction",
        "formula": "Prediction = (1/B) × Σᵦ₌₁ᴮ Tᵦ(x), where Tᵦ is tree b trained on bootstrap sample",
        "interpretation": "Reduces overfitting vs single trees. Feature importance helps identify key predictors."
    },
    
    "gradient_boosting": {
        "definition": "Sequential ensemble method where each model corrects errors of previous models",
        "formula": "F(x) = F₀(x) + Σᵐ₌₁ᴹ γₘhₘ(x), where hₘ fits residuals",
        "interpretation": "Often achieves high accuracy but can overfit. Learning rate controls step size."
    },
    
    "r_squared": {
        "definition": "Coefficient of determination measuring proportion of variance explained by the model",
        "formula": "R² = 1 - (SSᵣₑₛ / SSₜₒₜ) = 1 - Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²",
        "interpretation": "Values closer to 1 indicate better model fit. Negative values indicate poor model."
    },
    
    "mean_squared_error": {
        "definition": "Average squared differences between predicted and actual values",
        "formula": "MSE = (1/n) × Σᵢ₌₁ⁿ (yᵢ - ŷᵢ)²",
        "interpretation": "Lower values indicate better predictions. Sensitive to outliers due to squaring."
    },
    
    "feature_importance": {
        "definition": "Measure of how much each input variable contributes to model predictions",
        "formula": "Varies by algorithm - for trees: reduction in impurity weighted by probability",
        "interpretation": "Helps identify key predictors and understand model behavior. Sum typically equals 1."
    },
    
    # Risk Metrics
    "volatility": {
        "definition": "Statistical measure of price variation over time",
        "formula": "σ = √[(1/n-1) × Σ(Rᵢ - R̄)²] × √(periods per year)",
        "interpretation": "Higher volatility indicates greater price uncertainty. Annualized by multiplying by √252 for daily data."
    },
    
    "correlation": {
        "definition": "Statistical measure of linear relationship between two variables",
        "formula": "ρ = Cov(X,Y) / (σₓ × σᵧ)",
        "interpretation": "Values: +1 (perfect positive), 0 (no linear relationship), -1 (perfect negative)."
    },
    
    "information_ratio": {
        "definition": "Risk-adjusted return relative to a benchmark",
        "formula": "Information Ratio = (Portfolio Return - Benchmark Return) / Tracking Error",
        "interpretation": "Measures active return per unit of active risk. Higher values indicate better active management."
    }
}

def get_tooltip_help(term_key):
    """Get help text for a financial term with formula and interpretation."""
    if term_key in FINANCIAL_TOOLTIPS:
        tooltip = FINANCIAL_TOOLTIPS[term_key]
        return f"""
**{tooltip['definition']}**

**Formula:** {tooltip['formula']}

**Interpretation:** {tooltip['interpretation']}
        """
    return "No help available for this term."

def format_metric_with_tooltip(label, value, term_key, format_string="{:.2f}"):
    """Format a metric display with tooltip help."""
    help_text = get_tooltip_help(term_key)
    return f"{label}: {format_string.format(value) if isinstance(value, (int, float)) else value}"