"""
Stock Data API Client for QuantStack Platform
Provides real-time and historical market data from stockdata.org
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time
from typing import Optional, Dict, List, Any
import json
import streamlit as st

class StockDataClient:
    """Client for stockdata.org API integration"""
    
    def __init__(self, api_key: Optional[str] = None):
        # Try multiple sources for API key
        self.api_key = (api_key or 
                       os.environ.get("STOCKDATA_API_KEY") or
                       "uh8kCdBkyEjbME9WtzMPiwMkgcNOyARSgJe34mIq")
        self.base_url = "https://api.stockdata.org/v1"
        self.session = requests.Session()
        
        if self.api_key:
            self.session.headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            })
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make authenticated request to stockdata.org API"""
        if not self.api_key:
            raise ValueError("StockData API key is required. Please set STOCKDATA_API_KEY in secrets.")
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = self.session.get(url, params=params if params is not None else {})
            response.raise_for_status()
            
            data = response.json()
            
            # Check for API errors
            if not data.get("meta", {}).get("found", True):
                raise ValueError(f"No data found for request: {endpoint}")
            
            return data
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON response: {str(e)}")
        except Exception as e:
            raise Exception(f"Request error: {str(e)}")
    
    def get_real_time_quote(self, symbols: List[str]) -> pd.DataFrame:
        """Get real-time quotes for multiple symbols using StockData.org API"""
        try:
            symbols_str = ",".join(symbols)
            params = {
                "symbols": symbols_str,
                "api_token": self.api_key
            }
            
            data = self._make_request("data/quote", params)
            
            quotes = []
            for quote_data in data.get("data", []):
                quotes.append({
                    "symbol": quote_data.get("ticker"),
                    "price": quote_data.get("price"),
                    "change": quote_data.get("day_change"),
                    "change_percent": quote_data.get("day_change"),
                    "volume": quote_data.get("volume"),
                    "market_cap": quote_data.get("market_cap"),
                    "pe_ratio": quote_data.get("pe"),
                    "dividend_yield": quote_data.get("dividend_yield"),
                    "timestamp": quote_data.get("last_trade_time")
                })
            
            return pd.DataFrame(quotes)
            
        except Exception as e:
            return pd.DataFrame()
    
    def get_historical_data(self, symbols, date_from: Optional[str] = None, date_to: Optional[str] = None) -> pd.DataFrame:
        """Get historical OHLCV data for multiple symbols"""
        try:
            # Handle both single symbol string and list of symbols
            if isinstance(symbols, str):
                symbols = [symbols]
            
            # Default to last year if no dates provided
            if not date_to:
                date_to = datetime.now().strftime("%Y-%m-%d")
            if not date_from:
                date_from = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            
            params = {
                "symbols": ",".join(symbols),
                "date_from": date_from,
                "date_to": date_to,
                "api_token": self.api_key
            }
            
            data = self._make_request("data/eod", params)
            
            historical_data = []
            for item in data.get("data", []):
                historical_data.append({
                    "symbol": item.get("ticker"),
                    "date": item.get("date"),
                    "open": item.get("open"),
                    "high": item.get("high"),
                    "low": item.get("low"),
                    "close": item.get("close"),
                    "volume": item.get("volume"),
                })
            
            return pd.DataFrame(historical_data)
            
        except Exception as e:
            raise Exception(f"Error fetching historical data: {str(e)}")
    
    def get_intraday_data(self, symbol: str, interval: str = "1min", outputsize: int = 100) -> pd.DataFrame:
        """Get intraday data with specified interval"""
        try:
            params = {
                "symbols": symbol,
                "interval": interval,
                "outputsize": outputsize,
                "api_token": self.api_key
            }
            
            data = self._make_request("data/intraday", params)
            
            intraday_data = []
            for item in data.get("data", []):
                intraday_data.append({
                    "DateTime": item.get("datetime"),
                    "Open": item.get("open"),
                    "High": item.get("high"),
                    "Low": item.get("low"),
                    "Close": item.get("close"),
                    "Volume": item.get("volume")
                })
            
            df = pd.DataFrame(intraday_data)
            if not df.empty:
                df["DateTime"] = pd.to_datetime(df["DateTime"])
                df.set_index("DateTime", inplace=True)
                df = df.sort_index()
                
                # Convert to numeric
                numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")
            
            return df
            
        except Exception as e:
            st.error(f"Error fetching intraday data: {str(e)}")
            return pd.DataFrame()
    
    def get_company_profile(self, symbol: str) -> Dict:
        """Get comprehensive company profile and fundamentals"""
        try:
            params = {
                "symbols": symbol,
                "api_token": self.api_key
            }
            
            data = self._make_request("entity/profile", params)
            
            if data.get("data"):
                profile = data["data"][0]
                return {
                    "symbol": profile.get("ticker"),
                    "name": profile.get("name"),
                    "sector": profile.get("sector"),
                    "industry": profile.get("industry"),
                    "description": profile.get("description"),
                    "market_cap": profile.get("market_cap"),
                    "employees": profile.get("employees"),
                    "founded": profile.get("founded"),
                    "headquarters": profile.get("headquarters"),
                    "website": profile.get("website"),
                    "ceo": profile.get("ceo"),
                    "exchange": profile.get("exchange"),
                    "currency": profile.get("currency")
                }
            
            return {}
            
        except Exception as e:
            st.error(f"Error fetching company profile: {str(e)}")
            return {}
    
    def get_financial_statements(self, symbol: str, statement_type: str = "income") -> pd.DataFrame:
        """Get financial statements (income, balance, cash_flow)"""
        try:
            params = {
                "symbols": symbol,
                "statement": statement_type,
                "api_token": self.api_key
            }
            
            data = self._make_request("data/financials", params)
            
            financials = []
            for item in data.get("data", []):
                financials.append(item)
            
            return pd.DataFrame(financials)
            
        except Exception as e:
            st.error(f"Error fetching financial statements: {str(e)}")
            return pd.DataFrame()
    
    def get_options_data(self, symbol: str, expiration_date: str = None) -> pd.DataFrame:
        """Get options chain data"""
        try:
            params = {
                "symbols": symbol,
                "api_token": self.api_key
            }
            
            if expiration_date:
                params["expiration"] = expiration_date
            
            data = self._make_request("data/options", params)
            
            options_data = []
            for item in data.get("data", []):
                options_data.append({
                    "symbol": item.get("symbol"),
                    "strike": item.get("strike"),
                    "expiration": item.get("expiration"),
                    "type": item.get("type"),  # call or put
                    "bid": item.get("bid"),
                    "ask": item.get("ask"),
                    "last": item.get("last"),
                    "volume": item.get("volume"),
                    "open_interest": item.get("open_interest"),
                    "implied_volatility": item.get("implied_volatility")
                })
            
            return pd.DataFrame(options_data)
            
        except Exception as e:
            st.error(f"Error fetching options data: {str(e)}")
            return pd.DataFrame()
    
    def get_crypto_data(self, symbols: List[str]) -> pd.DataFrame:
        """Get cryptocurrency data"""
        try:
            symbols_str = ",".join(symbols)
            params = {
                "symbols": symbols_str,
                "api_token": self.api_key
            }
            
            data = self._make_request("data/crypto", params)
            
            crypto_data = []
            for item in data.get("data", []):
                crypto_data.append({
                    "symbol": item.get("symbol"),
                    "price": item.get("price"),
                    "change_24h": item.get("change_24h"),
                    "change_percent_24h": item.get("change_percent_24h"),
                    "volume_24h": item.get("volume_24h"),
                    "market_cap": item.get("market_cap"),
                    "rank": item.get("rank")
                })
            
            return pd.DataFrame(crypto_data)
            
        except Exception as e:
            st.error(f"Error fetching crypto data: {str(e)}")
            return pd.DataFrame()
    
    def get_forex_data(self, base_currency: str = "USD", symbols: List[str] = None) -> pd.DataFrame:
        """Get forex exchange rates"""
        try:
            params = {
                "base": base_currency,
                "api_token": self.api_key
            }
            
            if symbols is not None:
                params["symbols"] = ",".join(symbols)
            
            data = self._make_request("data/forex", params)
            
            forex_data = []
            rates = data.get("data", {}).get("rates", {})
            for currency, rate in rates.items():
                forex_data.append({
                    "pair": f"{base_currency}/{currency}",
                    "rate": rate,
                    "base": base_currency,
                    "quote": currency
                })
            
            return pd.DataFrame(forex_data)
            
        except Exception as e:
            st.error(f"Error fetching forex data: {str(e)}")
            return pd.DataFrame()
    
    def search_symbols(self, query: str, limit: int = 10) -> pd.DataFrame:
        """Search for symbols by name or ticker"""
        try:
            params = {
                "q": query,
                "limit": limit,
                "api_token": self.api_key
            }
            
            data = self._make_request("entity/search", params)
            
            results = []
            for item in data.get("data", []):
                results.append({
                    "symbol": item.get("ticker"),
                    "name": item.get("name"),
                    "exchange": item.get("exchange"),
                    "type": item.get("type"),
                    "sector": item.get("sector"),
                    "industry": item.get("industry")
                })
            
            return pd.DataFrame(results)
            
        except Exception as e:
            st.error(f"Error searching symbols: {str(e)}")
            return pd.DataFrame()
    
    def test_connection(self) -> bool:
        """Test API connection and authentication"""
        try:
            # Try a simple API call
            url = f"{self.base_url}/data/quote"
            params = {
                "symbols": "AAPL",
                "api_token": self.api_key
            }
            
            response = self.session.get(url, params=params)
            return response.status_code == 200
            
        except Exception:
            return False

# Utility functions for integration
def get_stockdata_client() -> StockDataClient:
    """Get configured StockData client instance"""
    return StockDataClient()

def is_stockdata_available() -> bool:
    """Check if StockData API is configured and available"""
    try:
        # Try multiple sources for API key
        import streamlit as st
        api_key = (os.environ.get("STOCKDATA_API_KEY") or 
                  getattr(st.secrets, 'STOCKDATA_API_KEY', None) or
                  "uh8kCdBkyEjbME9WtzMPiwMkgcNOyARSgJe34mIq")
        
        if not api_key:
            return False
        
        client = StockDataClient(api_key)
        return True  # Assume available if key exists
    except Exception:
        return False