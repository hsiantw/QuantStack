# Quantitative Finance Platform

## Overview
This is a comprehensive quantitative finance platform built with Streamlit that provides advanced financial analysis tools for portfolio management, statistical arbitrage, time series analysis, and AI-powered market insights. The platform features a modern multi-page architecture with 15+ specialized modules covering everything from portfolio optimization to real-time trading account monitoring. It's designed for quantitative analysts, traders, and researchers who need professional-grade financial analysis tools with an intuitive web interface.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
**Framework**: Streamlit-based web application with a multi-page architecture spanning 15+ specialized modules including portfolio optimization, AI analysis, statistical arbitrage, time series analysis, trading strategies, risk management, and market monitoring.

**UI Design**: Professional dark theme with gradient styling inspired by QuantConnect, featuring animated components, hover effects, smooth transitions, and responsive layouts optimized for both desktop and mobile use.

**Navigation**: Organized into three logical categories - Portfolio & Trading, AI Insights & Analysis, and Platform Highlights - with an expandable sidebar for consistent navigation across all modules.

**Visualization**: Plotly integration for interactive financial charts with custom dark theme styling, supporting real-time market data visualization, performance dashboards, and comprehensive backtesting results.

**Authentication System**: Built-in user authentication with SQLite database backend, supporting user preferences, strategy saving, and personalized settings management.

### Backend Architecture
**Core Structure**: Utility-based architecture with specialized classes for different financial operations, including data fetching, portfolio optimization, statistical analysis, AI models, backtesting engines, and risk management systems.

**Data Processing**: Pandas and NumPy for efficient financial data manipulation, with SciPy for statistical analysis and optimization algorithms, and Statsmodels for advanced econometric analysis.

**Machine Learning**: Scikit-learn integration for AI-powered financial predictions using Random Forest, Gradient Boosting, and Linear Regression models with 10-20 years of historical training data.

**Caching Strategy**: Streamlit caching decorators with time-based expiration (5 minutes for price data, 30 minutes for metadata) to optimize performance and reduce API calls.

### Data Management
**Primary Data Source**: Yahoo Finance API via yfinance library for real-time and historical market data, providing comprehensive OHLCV data, company fundamentals, and market indices.

**Extended Training Data**: AI models utilize 10-20 years of historical data for robust training, with configurable training periods and automatic feature engineering for technical indicators.

**Data Validation**: Comprehensive error handling for invalid tickers, missing data scenarios, and API rate limits with graceful fallbacks.

**Multi-Asset Support**: Support for stocks, ETFs, cryptocurrencies, commodities, forex, and futures with specialized analysis modules for each asset class.

### Core Analysis Engines
**Portfolio Optimization**: Modern Portfolio Theory implementation with efficient frontier calculation, multiple optimization objectives (Sharpe ratio, minimum volatility, maximum return), and dynamic rebalancing strategies.

**AI-Powered Prediction**: Machine learning models for price prediction, pattern recognition, and strategy optimization using Random Forest, Gradient Boosting, and Linear Regression with comprehensive feature engineering.

**Statistical Arbitrage**: Cointegration analysis for pairs trading, mean reversion strategies, and AI-optimized parameter selection for statistical arbitrage opportunities.

**Backtesting Framework**: Professional-grade backtesting system supporting multiple strategies (Moving Average Crossover, RSI Mean Reversion, Bollinger Bands, Momentum) with comprehensive performance metrics and risk analysis.

**Risk Management Suite**: Advanced risk analytics including Value at Risk (VaR) calculations using historical, parametric, and Monte Carlo methods, stress testing, and portfolio risk decomposition.

**Time Series Analysis**: ARIMA modeling, seasonality detection, stationarity testing, and forecasting capabilities for advanced econometric analysis.

## External Dependencies

### Market Data APIs
**Yahoo Finance (yfinance)**: Primary data source for real-time and historical market data, supporting stocks, ETFs, cryptocurrencies, and global indices with comprehensive OHLCV data and fundamental information.

**StockData API**: Optional premium real-time market data source for enhanced data coverage and lower latency (configured via environment variable STOCKDATA_API_KEY).

### Web Scraping & News
**Trafilatura**: Web content extraction for news analysis and market intelligence gathering from financial websites and economic data sources.

**News Sources Integration**: Built-in scrapers for major financial news sources, economic calendars, and SEC filings to provide comprehensive market context.

### Trading Integration
**Webull API**: Real-time trading account monitoring and integration (unofficial API support with planned migration to official API when available).

**Interactive Brokers**: Framework prepared for professional trading integration with support for options analysis and derivatives trading.

### Machine Learning & Analytics
**Scikit-learn**: Core machine learning library for AI-powered predictions, strategy optimization, and pattern recognition with support for ensemble methods and model validation.

**SciPy & Statsmodels**: Advanced statistical analysis, optimization algorithms, econometric modeling, and time series analysis capabilities.

**NumPy & Pandas**: Fundamental data processing and numerical computation libraries optimized for financial time series analysis.

### Visualization & UI
**Plotly**: Interactive charting library with custom dark theme styling for financial visualizations, performance dashboards, and real-time market monitoring.

**Streamlit**: Web application framework with multi-page architecture, caching, and responsive UI components optimized for financial applications.

### Database & Authentication
**SQLite**: Local database for user authentication, preferences storage, and strategy persistence with support for user sessions and personalized settings.

**Requests**: HTTP client for API interactions, web scraping, and external service integrations with robust error handling and rate limiting.