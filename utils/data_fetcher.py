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
