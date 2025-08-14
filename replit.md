# Quantitative Finance Platform

## Overview

This is a comprehensive quantitative finance platform built with Streamlit that provides advanced financial analysis tools. The application offers multiple analytical modules including AI-powered predictions, portfolio optimization using Modern Portfolio Theory, statistical arbitrage analysis, AI-powered pairs trading with cointegration testing, time series analysis, and trading strategy backtesting. The platform is designed for quantitative analysts, traders, and researchers who need sophisticated financial analysis capabilities with an intuitive web interface.

## Recent Changes (August 2025)

- **AI Pairs Trading Module**: Complete statistical arbitrage system with cointegration analysis, AI strategy optimization for pair components, and comprehensive trading signal generation
- **Enhanced AI Strategy Optimization**: Detailed methodology explanations showing exact mathematical calculations and decision-making process
- **Pairs Discovery Algorithm**: Sector-based candidate selection with trading score ranking system (0-100 scale)
- **Comprehensive Backtesting**: Full pairs trading strategy backtesting with performance metrics and visualization
- **Live Trading Signals**: Real-time Z-score analysis and trading recommendations for cointegrated pairs

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit-based web application with multi-page architecture
- **Layout**: Wide layout with expandable sidebar for consistent navigation
- **Visualization**: Plotly integration for interactive financial charts and graphs
- **Page Structure**: Modular page system with dedicated modules for different analysis types
- **Caching**: Streamlit caching decorators for performance optimization on data fetching

### Backend Architecture
- **Core Structure**: Utility-based architecture with specialized classes for different financial operations
- **Data Processing**: Pandas and NumPy for efficient financial data manipulation
- **Mathematical Computing**: SciPy for statistical analysis and optimization algorithms
- **Machine Learning**: Scikit-learn integration for AI-powered financial predictions
- **Time Series Analysis**: Statsmodels for advanced econometric analysis

### Data Management
- **Data Source**: Yahoo Finance API through yfinance library for real-time market data
- **Data Caching**: Time-based caching (5 minutes for price data, 1 hour for metadata)
- **Data Processing**: Automated feature engineering for technical indicators and risk metrics
- **Data Validation**: Error handling for invalid tickers and missing data scenarios

### Core Analytical Modules
- **AI Models**: Machine learning models for price prediction using Random Forest, Gradient Boosting, and Linear Regression with configurable 10-20 year historical training data
- **Portfolio Optimization**: Modern Portfolio Theory implementation with efficient frontier calculation
- **Statistical Arbitrage**: Cointegration analysis and pair trading opportunity identification
- **Time Series Analysis**: ARIMA modeling, seasonality detection, and stationarity testing
- **Trading Strategies**: Multiple strategy implementations with comprehensive backtesting capabilities
- **Advanced Strategy Optimization**: Systematic comparison of 15+ trading strategies with different indicator combinations
- **AI Pairs Trading**: Comprehensive pairs trading system with AI-optimized strategy application to cointegrated pairs
- **Market Information Sources**: Comprehensive guide to critical data sources (SEC filings, economic indicators, Fed data)
- **Risk Management**: Value at Risk (VaR), Expected Shortfall, and comprehensive risk metrics
- **Educational Tooltips**: Interactive help system with formulas and definitions for complex financial terms

### Design Patterns
- **Utility Classes**: Modular utility classes for specific financial operations
- **Strategy Pattern**: Implemented for different trading strategies and analysis methods
- **Factory Pattern**: Used for creating different types of financial models
- **Observer Pattern**: Implicit through Streamlit's reactive programming model
- **Educational Design**: Comprehensive tooltip system with formulas and definitions for user education

## External Dependencies

### Core Data Services
- **Yahoo Finance (yfinance)**: Primary data source for stock prices, market indices, and financial metadata
- **Real-time Market Data**: Major indices (S&P 500, NASDAQ, Dow Jones, VIX) for market overview

### Scientific Computing Stack
- **NumPy**: Numerical computing and array operations
- **Pandas**: Data manipulation and analysis
- **SciPy**: Statistical analysis and optimization functions
- **Scikit-learn**: Machine learning algorithms and preprocessing tools
- **Statsmodels**: Advanced statistical modeling and econometric analysis

### Visualization and UI
- **Plotly**: Interactive charting and visualization library
- **Streamlit**: Web application framework and UI components

### Financial Analysis Libraries
- **yfinance**: Yahoo Finance API wrapper for market data retrieval
- **scipy.optimize**: Portfolio optimization and constraint solving
- **sklearn preprocessing**: Data scaling and feature engineering
- **statsmodels.tsa**: Time series analysis and forecasting models