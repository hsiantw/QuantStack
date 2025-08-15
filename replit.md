# Quantitative Finance Platform

## Overview
This comprehensive quantitative finance platform, built with Streamlit, offers advanced financial analysis tools for quantitative analysts, traders, and researchers. It integrates AI-powered predictions, portfolio optimization (Modern Portfolio Theory), statistical arbitrage including AI-powered pairs trading with cointegration testing, extensive mean reversion strategies, time series analysis, and trading strategy backtesting. The platform also provides advanced market analysis, comprehensive cryptocurrency analysis, live news and economic data intelligence, multi-asset class analysis (commodities, forex, futures), and an investment portfolio manager. A key ambition is to provide sophisticated capabilities through an intuitive web interface, complemented by a live trading account monitor for real-time integration.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit-based web application with a multi-page, 13-module architecture.
- **Layout**: Wide layout with an expandable sidebar for consistent navigation.
- **Modern UI Design**: Professional dark theme with gradient styling, custom CSS framework, animated components, hover effects, and smooth transitions for an enhanced user experience. Features include consistent header styling, gradient metric cards, modern button styling, shimmer effects for navigation, professional alert boxes, and enhanced tab styling.
- **Component System**: Reusable UI components utility module for consistent styling and interactive elements.
- **Visualization**: Plotly integration for interactive financial charts with custom dark theme styling.
- **Responsive Design**: Mobile-optimized layouts with adaptive components and custom scrollbars.
- **Caching**: Streamlit caching decorators for performance optimization on data fetching.
- **Trading Integration**: Real-time account monitoring interface with Webull API connectivity.

### Backend Architecture
- **Core Structure**: Utility-based architecture with specialized classes for financial operations.
- **Data Processing**: Pandas and NumPy for efficient financial data manipulation.
- **Mathematical Computing**: SciPy for statistical analysis and optimization algorithms.
- **Machine Learning**: Scikit-learn for AI-powered financial predictions.
- **Time Series Analysis**: Statsmodels for advanced econometric analysis.

### Data Management
- **Data Source**: Yahoo Finance API via the `yfinance` library for real-time market data.
- **Data Caching**: Time-based caching (5 minutes for price data, 1 hour for metadata).
- **Data Processing**: Automated feature engineering for technical indicators and risk metrics.
- **Data Validation**: Error handling for invalid tickers and missing data scenarios.

### Core Analytical Modules
- **AI Models**: Machine learning models (Random Forest, Gradient Boosting, Linear Regression) for price prediction, configurable with 10-20 years of historical training data.
- **Portfolio Optimization**: Modern Portfolio Theory with efficient frontier calculation.
- **Statistical Arbitrage**: Cointegration analysis and pair trading opportunity identification, including AI-optimized strategy application for cointegrated pairs.
- **Time Series Analysis**: ARIMA modeling, seasonality detection, and stationarity testing.
- **Trading Strategies**: Multiple strategy implementations with comprehensive backtesting capabilities and systematic comparison of 15+ trading strategies.
- **Mean Reversion Strategies**: Professional-grade implementation featuring 6 mathematical approaches: Bollinger Bands, RSI, Ornstein-Uhlenbeck, Kalman Filter, Ensemble, and Adaptive strategies.
- **Market Analysis**: Includes Money Flow Index, Accumulation/Distribution Line, On-Balance Volume, liquidity metrics, dark pool activity detection, and volume profile analysis.
- **Cryptocurrency Analysis**: Fear & Greed index, on-chain metrics, DeFi ecosystem analysis, correlation analysis, and specialized crypto trading strategies.
- **Live News & Economic Intelligence**: Automated web scraping of financial news and economic data (Fed, BLS, Treasury, MarketWatch, Reuters, CNBC), economic calendar, FOMC schedule, earnings calendar, news sentiment analysis, and market intelligence summary dashboard.
- **Trading Account Integration**: Live account monitoring with Webull API for balance, P&L, position analysis, order management, and strategy performance validation in demo or live modes.
- **Risk Management**: Value at Risk (VaR), Expected Shortfall, and comprehensive risk metrics.
- **Educational Tooltips**: Interactive help system with formulas and definitions for complex financial terms.

### Design Patterns
- **Utility Classes**: Modular utility classes for specific financial operations.
- **Strategy Pattern**: Implemented for different trading strategies and analysis methods.
- **Factory Pattern**: Used for creating various financial models.
- **Observer Pattern**: Implicit through Streamlit's reactive programming model.
- **Educational Design**: Comprehensive tooltip system for user education.

## External Dependencies

### Core Data Services
- **Yahoo Finance (yfinance)**: Primary data source for stock prices, market indices, and financial metadata.
- **Webull API (`webull`, `webull-python-sdk`)**: Trading account integration for live portfolio monitoring and order management.

### Scientific Computing Stack
- **NumPy**: Numerical computing and array operations.
- **Pandas**: Data manipulation and analysis.
- **SciPy**: Statistical analysis and optimization functions.
- **Scikit-learn**: Machine learning algorithms and preprocessing tools.
- **Statsmodels**: Advanced statistical modeling and econometric analysis.

### Visualization and UI
- **Plotly**: Interactive charting and visualization library.
- **Streamlit**: Web application framework and UI components.

### Financial Analysis Libraries
- **scipy.optimize**: For portfolio optimization and constraint solving.
- **sklearn.preprocessing**: For data scaling and feature engineering.
- **statsmodels.tsa**: For time series analysis and forecasting models.
- **asyncio-throttle**: For rate limiting API requests.