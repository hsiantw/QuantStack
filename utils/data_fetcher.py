import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

class DataFetcher:
    """Utility class for fetching and processing financial data"""
    
    @staticmethod
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def get_stock_data(tickers, period="1y", interval="1d"):
        """
        Fetch stock data for given tickers
        
        Args:
            tickers (list): List of ticker symbols
            period (str): Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval (str): Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
        Returns:
            pandas.DataFrame: Stock data
        """
        try:
            if isinstance(tickers, str):
                tickers = [tickers]
            
            data = yf.download(tickers, period=period, interval=interval, progress=False)
            
            if data.empty:
                st.error("No data retrieved. Please check ticker symbols and try again.")
                return pd.DataFrame()
            
            # If single ticker, flatten the column structure
            if len(tickers) == 1:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
            
            return data
            
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    @st.cache_data(ttl=1800)  # Cache for 30 minutes for AI training data
    def get_historical_data_for_ai(ticker, years_back=15):
        """
        Get extended historical data specifically for AI model training
        Uses 10-20 year historical period for robust training data
        
        Args:
            ticker (str): Ticker symbol
            years_back (int): Number of years of historical data (default 15, range 10-20)
        
        Returns:
            pandas.DataFrame: Extended historical OHLCV data
        """
        try:
            # Ensure years_back is within the recommended range
            years_back = max(10, min(20, years_back))
            
            # Calculate start date
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years_back * 365)
            
            # Format dates for yfinance
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Fetch data using date range instead of period for precision
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_str, end=end_str, interval="1d")
            
            if data.empty:
                st.warning(f"No historical data available for {ticker} in the requested period")
                return pd.DataFrame()
            
            # Ensure we have sufficient data for AI training
            min_required_days = years_back * 250  # Approximate trading days
            if len(data) < min_required_days * 0.7:  # Allow 30% tolerance
                st.warning(f"Limited historical data for {ticker}: {len(data)} days available, recommended minimum: {int(min_required_days * 0.7)}")
            
            # Add metadata about the data period
            data.attrs['ticker'] = ticker
            data.attrs['start_date'] = start_str
            data.attrs['end_date'] = end_str
            data.attrs['years_covered'] = years_back
            data.attrs['total_days'] = len(data)
            
            return data
            
        except Exception as e:
            st.error(f"Error fetching historical data for AI training: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def get_stock_info(ticker):
        """
        Get stock information and metadata
        
        Args:
            ticker (str): Ticker symbol
        
        Returns:
            dict: Stock information
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return info
        except Exception as e:
            st.error(f"Error fetching info for {ticker}: {str(e)}")
            return {}
    
    @staticmethod
    def calculate_returns(data, method="simple"):
        """
        Calculate returns from price data
        
        Args:
            data (pandas.DataFrame): Price data
            method (str): 'simple' or 'log'
        
        Returns:
            pandas.DataFrame: Returns data
        """
        try:
            if method == "simple":
                returns = data.pct_change().dropna()
            elif method == "log":
                returns = np.log(data / data.shift(1)).dropna()
            else:
                raise ValueError("Method must be 'simple' or 'log'")
            
            return returns
            
        except Exception as e:
            st.error(f"Error calculating returns: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def get_market_data():
        """
        Get major market indices data
        
        Returns:
            dict: Market indices data
        """
        indices = {
            "S&P 500": "^GSPC",
            "NASDAQ": "^IXIC",
            "Dow Jones": "^DJI",
            "Russell 2000": "^RUT",
            "VIX": "^VIX"
        }
        
        market_data = {}
        
        for name, ticker in indices.items():
            try:
                data = DataFetcher.get_stock_data(ticker, period="1y")
                if not data.empty:
                    market_data[name] = data
            except Exception as e:
                st.warning(f"Could not fetch data for {name}: {str(e)}")
        
        return market_data
    
    @staticmethod
    def validate_tickers(tickers):
        """
        Validate if tickers are valid and have data
        
        Args:
            tickers (list): List of ticker symbols
        
        Returns:
            tuple: (valid_tickers, invalid_tickers)
        """
        valid_tickers = []
        invalid_tickers = []
        
        for ticker in tickers:
            try:
                data = yf.download(ticker, period="5d", progress=False)
                if not data.empty:
                    valid_tickers.append(ticker)
                else:
                    invalid_tickers.append(ticker)
            except:
                invalid_tickers.append(ticker)
        
        return valid_tickers, invalid_tickers
    
    @staticmethod
    def get_sector_data():
        """
        Get sector ETF data for diversification analysis
        
        Returns:
            dict: Sector ETF data
        """
        sector_etfs = {
            "Technology": "XLK",
            "Healthcare": "XLV", 
            "Financials": "XLF",
            "Consumer Discretionary": "XLY",
            "Communication Services": "XLC",
            "Industrials": "XLI",
            "Consumer Staples": "XLP",
            "Energy": "XLE",
            "Utilities": "XLU",
            "Real Estate": "XLRE",
            "Materials": "XLB"
        }
        
        sector_data = {}
        
        for sector, etf in sector_etfs.items():
            try:
                data = DataFetcher.get_stock_data(etf, period="1y")
                if not data.empty:
                    sector_data[sector] = data
            except Exception as e:
                st.warning(f"Could not fetch data for {sector} sector: {str(e)}")
        
        return sector_data
