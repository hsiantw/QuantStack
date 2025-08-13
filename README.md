# Quantitative Finance Platform

A comprehensive quantitative finance platform built with Streamlit that provides advanced financial analysis tools for portfolio management, statistical arbitrage, time series analysis, and AI-powered predictions.

## 🚀 Features

### Core Analysis Modules
- **Portfolio Optimization**: Modern Portfolio Theory implementation with efficient frontier calculation
- **Statistical Arbitrage**: Cointegration analysis and pair trading opportunities
- **Time Series Analysis**: ARIMA modeling, seasonality detection, and forecasting
- **Trading Strategies**: Multiple strategy backtesting with performance metrics
- **AI Analysis**: Machine learning models for price prediction and pattern recognition

### Key Capabilities
- Real-time market data from Yahoo Finance
- Interactive Plotly visualizations
- Risk metrics (VaR, Expected Shortfall, Sharpe Ratio)
- Comprehensive backtesting framework
- Feature engineering for ML models
- Performance comparison across strategies

## 🛠️ Technology Stack

- **Frontend**: Streamlit with multi-page architecture
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Machine Learning**: Scikit-learn
- **Statistical Analysis**: SciPy, Statsmodels
- **Market Data**: yfinance (Yahoo Finance API)

## 📁 Project Structure

```
├── app.py                      # Main application entry point
├── pages/                      # Streamlit pages
│   ├── portfolio_optimization.py
│   ├── statistical_arbitrage.py
│   ├── time_series_analysis.py
│   ├── trading_strategies.py
│   └── ai_analysis.py
├── utils/                      # Core analysis modules
│   ├── data_fetcher.py
│   ├── portfolio_optimizer.py
│   ├── statistical_arbitrage.py
│   ├── time_series_analysis.py
│   ├── trading_strategies.py
│   ├── ai_models.py
│   ├── backtesting.py
│   └── risk_metrics.py
├── .streamlit/
│   └── config.toml            # Streamlit configuration
└── requirements.txt           # Project dependencies
```

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone <your-github-repo-url>
cd quantitative-finance-platform
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## 📊 Usage

### Portfolio Optimization
- Enter multiple stock tickers for portfolio analysis
- Adjust risk tolerance and investment constraints
- View efficient frontier and optimal portfolio allocation
- Analyze risk-return characteristics

### Statistical Arbitrage
- Select pairs of assets for cointegration analysis
- Identify mean-reverting relationships
- Generate trading signals based on statistical models
- Backtest pair trading strategies

### Time Series Analysis
- Perform stationarity tests and decomposition
- Build ARIMA models for forecasting
- Analyze seasonality and trends
- Generate price predictions with confidence intervals

### Trading Strategies
- Backtest multiple technical indicators
- Compare strategy performance metrics
- Analyze risk-adjusted returns
- Optimize strategy parameters

### AI Analysis
- Train machine learning models on historical data
- Generate price predictions using Random Forest and Gradient Boosting
- Analyze feature importance
- Assess model performance and reliability

## 🔧 Configuration

The application uses Streamlit's configuration system. Key settings are in `.streamlit/config.toml`:

```toml
[server]
headless = true
address = "0.0.0.0"
port = 5000
```

## 📈 Data Sources

- **Market Data**: Yahoo Finance (via yfinance library)
- **Real-time Prices**: Major indices and individual stocks
- **Historical Data**: Up to 5 years of OHLCV data
- **Market Metadata**: Company information, sectors, and fundamentals

## ⚠️ Disclaimer

This platform is for educational and research purposes only. It is not intended as financial advice. Always consult with qualified financial professionals before making investment decisions.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🏗️ Architecture

The platform follows a modular architecture with clear separation of concerns:

- **Frontend**: Streamlit pages handle user interface and interaction
- **Backend**: Utility classes manage data processing and analysis
- **Data Layer**: Centralized data fetching with caching for performance
- **Analysis Layer**: Specialized modules for different quantitative methods

## 🔮 Future Enhancements

- Options pricing models (Black-Scholes, Binomial)
- Alternative data sources integration
- Real-time trading simulation
- Portfolio risk management dashboards
- Advanced ML models (LSTM, Transformer)
- Cryptocurrency analysis modules