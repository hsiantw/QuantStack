"""
Webull API Connector for Trading Account Integration
Supports both official and unofficial API connections with comprehensive error handling
"""

import os
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
import streamlit as st

class WebullConnector:
    """
    Webull API connector with support for both official and unofficial APIs
    Includes comprehensive error handling and rate limiting
    """
    
    def __init__(self, api_type: str = "unofficial"):
        """
        Initialize Webull connector
        
        Args:
            api_type (str): "official" or "unofficial"
        """
        self.api_type = api_type
        self.wb = None
        self.is_connected = False
        self.last_request_time = 0
        self.rate_limit_delay = 0.5  # 500ms between requests
        self.connection_details = {}
        
    def _enforce_rate_limit(self):
        """Enforce rate limiting between API calls"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last_request)
        
        self.last_request_time = time.time()
    
    def connect_unofficial(self, email: str, password: str, trade_pin: Optional[str] = None) -> bool:
        """
        Connect using unofficial Webull API
        
        Args:
            email (str): Webull account email
            password (str): Webull account password
            trade_pin (str, optional): Trading PIN for order placement
            
        Returns:
            bool: True if connection successful
        """
        try:
            # Import unofficial webull library
            try:
                from webull import webull
            except ImportError:
                st.error("Webull library not installed. Run: pip install webull")
                return False
            
            self.wb = webull()
            
            # Attempt login
            login_result = self.wb.login(email, password)
            
            if login_result:
                self.is_connected = True
                self.connection_details = {
                    'api_type': 'unofficial',
                    'email': email,
                    'connected_at': datetime.now(),
                    'has_trade_token': False
                }
                
                # Get trade token if PIN provided
                if trade_pin:
                    try:
                        trade_token = self.wb.get_trade_token(trade_pin)
                        if trade_token:
                            self.connection_details['has_trade_token'] = True
                    except Exception as e:
                        st.warning(f"Could not get trade token: {str(e)}")
                
                return True
            else:
                st.error("Login failed. Please check your credentials.")
                return False
                
        except Exception as e:
            st.error(f"Connection failed: {str(e)}")
            return False
    
    def connect_official(self, app_key: str, app_secret: str, region_id: str = "us") -> bool:
        """
        Connect using official Webull API
        
        Args:
            app_key (str): Webull API app key
            app_secret (str): Webull API app secret
            region_id (str): Region ID (us, hk, jp)
            
        Returns:
            bool: True if connection successful
        """
        try:
            # Import official webull SDK
            try:
                from webull_python_sdk_core.client import Client
                from webull_python_sdk_trade.api.trade_api import TradeApi
            except ImportError:
                st.error("Official Webull SDK not installed. Run: pip install webull-python-sdk-core webull-python-sdk-trade")
                return False
            
            # Initialize client
            client = Client(
                app_key=app_key,
                app_secret=app_secret,
                region_id=region_id
            )
            
            # Initialize trade API
            self.wb = TradeApi(client)
            
            # Test connection by getting account list
            try:
                accounts = self.wb.get_account_list()
                if accounts:
                    self.is_connected = True
                    self.connection_details = {
                        'api_type': 'official',
                        'app_key': app_key[:8] + "...",  # Masked for security
                        'region_id': region_id,
                        'connected_at': datetime.now(),
                        'account_count': len(accounts)
                    }
                    return True
                else:
                    st.error("No accounts found. Please check your API credentials.")
                    return False
                    
            except Exception as e:
                st.error(f"API authentication failed: {str(e)}")
                return False
                
        except Exception as e:
            st.error(f"Official API connection failed: {str(e)}")
            return False
    
    def get_account_info(self) -> Optional[Dict]:
        """
        Get account information
        
        Returns:
            dict: Account information or None if failed
        """
        if not self.is_connected:
            return None
        
        try:
            self._enforce_rate_limit()
            
            if self.api_type == "unofficial":
                account = self.wb.get_account()
                return {
                    'account_id': account.get('accountId', ''),
                    'account_value': float(account.get('netLiquidation', 0)),
                    'cash_balance': float(account.get('cashBalance', 0)),
                    'buying_power': float(account.get('buyingPower', 0)),
                    'day_pnl': float(account.get('dayProfitLoss', 0)),
                    'total_pnl': float(account.get('totalProfitLoss', 0)),
                    'last_updated': datetime.now()
                }
            else:
                # Official API implementation
                accounts = self.wb.get_account_list()
                if accounts:
                    account = accounts[0]  # Use first account
                    account_details = self.wb.get_account(account.account_id)
                    return {
                        'account_id': account.account_id,
                        'account_value': float(account_details.net_liquidation),
                        'cash_balance': float(account_details.cash_balance),
                        'buying_power': float(account_details.buying_power),
                        'day_pnl': float(account_details.day_pnl),
                        'total_pnl': float(account_details.total_pnl),
                        'last_updated': datetime.now()
                    }
                
        except Exception as e:
            st.error(f"Failed to get account info: {str(e)}")
            return None
    
    def get_positions(self) -> Optional[pd.DataFrame]:
        """
        Get current positions
        
        Returns:
            pd.DataFrame: Positions data or None if failed
        """
        if not self.is_connected:
            return None
        
        try:
            self._enforce_rate_limit()
            
            if self.api_type == "unofficial":
                positions = self.wb.get_positions()
                
                if positions:
                    positions_data = []
                    for pos in positions:
                        positions_data.append({
                            'symbol': pos.get('ticker', {}).get('symbol', ''),
                            'quantity': float(pos.get('position', 0)),
                            'avg_cost': float(pos.get('costPrice', 0)),
                            'current_price': float(pos.get('marketPrice', 0)),
                            'market_value': float(pos.get('marketValue', 0)),
                            'unrealized_pnl': float(pos.get('unrealizedProfitLoss', 0)),
                            'unrealized_pnl_pct': float(pos.get('unrealizedProfitLossRate', 0)) * 100
                        })
                    
                    return pd.DataFrame(positions_data)
            else:
                # Official API implementation
                accounts = self.wb.get_account_list()
                if accounts:
                    positions = self.wb.get_positions(accounts[0].account_id)
                    
                    positions_data = []
                    for pos in positions:
                        positions_data.append({
                            'symbol': pos.instrument_id,
                            'quantity': float(pos.quantity),
                            'avg_cost': float(pos.cost_price),
                            'current_price': float(pos.market_price),
                            'market_value': float(pos.market_value),
                            'unrealized_pnl': float(pos.unrealized_pnl),
                            'unrealized_pnl_pct': float(pos.unrealized_pnl_rate) * 100
                        })
                    
                    return pd.DataFrame(positions_data)
                
            return pd.DataFrame()  # Return empty DataFrame if no positions
            
        except Exception as e:
            st.error(f"Failed to get positions: {str(e)}")
            return None
    
    def get_orders(self) -> Optional[pd.DataFrame]:
        """
        Get order information
        
        Returns:
            pd.DataFrame: Orders data or None if failed
        """
        if not self.is_connected:
            return None
        
        try:
            self._enforce_rate_limit()
            
            if self.api_type == "unofficial":
                orders = self.wb.get_current_orders()
                
                if orders:
                    orders_data = []
                    for order in orders:
                        orders_data.append({
                            'order_id': order.get('orderId', ''),
                            'symbol': order.get('ticker', {}).get('symbol', ''),
                            'side': order.get('action', ''),
                            'quantity': float(order.get('totalQuantity', 0)),
                            'price': float(order.get('lmtPrice', 0)),
                            'order_type': order.get('orderType', ''),
                            'status': order.get('statusStr', ''),
                            'created_time': order.get('createTime', '')
                        })
                    
                    return pd.DataFrame(orders_data)
            else:
                # Official API implementation
                accounts = self.wb.get_account_list()
                if accounts:
                    orders = self.wb.get_orders(accounts[0].account_id)
                    
                    orders_data = []
                    for order in orders:
                        orders_data.append({
                            'order_id': order.order_id,
                            'symbol': order.instrument_id,
                            'side': order.side,
                            'quantity': float(order.quantity),
                            'price': float(order.price) if order.price else 0,
                            'order_type': order.order_type,
                            'status': order.status,
                            'created_time': order.created_time
                        })
                    
                    return pd.DataFrame(orders_data)
            
            return pd.DataFrame()  # Return empty DataFrame if no orders
            
        except Exception as e:
            st.error(f"Failed to get orders: {str(e)}")
            return None
    
    def place_order(self, symbol: str, side: str, quantity: int, 
                   order_type: str = "MKT", price: Optional[float] = None) -> Optional[str]:
        """
        Place a trading order
        
        Args:
            symbol (str): Stock symbol
            side (str): "BUY" or "SELL"
            quantity (int): Number of shares
            order_type (str): "MKT", "LMT", "STP", etc.
            price (float, optional): Limit price for limit orders
            
        Returns:
            str: Order ID if successful, None if failed
        """
        if not self.is_connected:
            st.error("Not connected to Webull account")
            return None
        
        if not self.connection_details.get('has_trade_token', False) and self.api_type == "unofficial":
            st.error("Trade token required for order placement. Please provide trading PIN.")
            return None
        
        try:
            self._enforce_rate_limit()
            
            if self.api_type == "unofficial":
                # Place order using unofficial API
                if order_type == "LMT" and price:
                    order_result = self.wb.place_order(
                        stock=symbol,
                        action=side,
                        qty=quantity,
                        orderType=order_type,
                        price=price
                    )
                else:
                    order_result = self.wb.place_order(
                        stock=symbol,
                        action=side,
                        qty=quantity,
                        orderType=order_type
                    )
                
                if order_result and 'orderId' in order_result:
                    return order_result['orderId']
                else:
                    st.error("Order placement failed")
                    return None
            else:
                # Official API implementation
                accounts = self.wb.get_account_list()
                if accounts:
                    order_result = self.wb.place_order(
                        account_id=accounts[0].account_id,
                        instrument_id=symbol,
                        side=side,
                        order_type=order_type,
                        quantity=quantity,
                        price=price
                    )
                    
                    if order_result:
                        return order_result.order_id
                    else:
                        st.error("Order placement failed")
                        return None
                
        except Exception as e:
            st.error(f"Failed to place order: {str(e)}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order
        
        Args:
            order_id (str): Order ID to cancel
            
        Returns:
            bool: True if cancellation successful
        """
        if not self.is_connected:
            return False
        
        try:
            self._enforce_rate_limit()
            
            if self.api_type == "unofficial":
                result = self.wb.cancel_order(order_id)
                return result is not None
            else:
                accounts = self.wb.get_account_list()
                if accounts:
                    result = self.wb.cancel_order(accounts[0].account_id, order_id)
                    return result is not None
                
            return False
            
        except Exception as e:
            st.error(f"Failed to cancel order: {str(e)}")
            return False
    
    def get_connection_status(self) -> Dict:
        """
        Get connection status and details
        
        Returns:
            dict: Connection status information
        """
        return {
            'is_connected': self.is_connected,
            'connection_details': self.connection_details,
            'last_request_time': self.last_request_time
        }
    
    def disconnect(self):
        """Disconnect from Webull API"""
        self.wb = None
        self.is_connected = False
        self.connection_details = {}

# Demo connector for testing purposes
class DemoConnector:
    """
    Demo connector that simulates Webull API responses
    Used for testing and demonstration purposes
    """
    
    def __init__(self):
        self.is_connected = False
        self.demo_data = self._create_demo_data()
        
    def _create_demo_data(self):
        """Create realistic demo trading data"""
        return {
            'account': {
                'account_id': 'DEMO_123456',
                'account_value': 125750.85,
                'cash_balance': 15420.30,
                'buying_power': 30840.60,
                'day_pnl': 1250.75,
                'total_pnl': 8945.20,
                'last_updated': datetime.now()
            },
            'positions': [
                {'symbol': 'AAPL', 'quantity': 50, 'avg_cost': 185.50, 'current_price': 192.30},
                {'symbol': 'MSFT', 'quantity': 25, 'avg_cost': 415.20, 'current_price': 422.75},
                {'symbol': 'GOOGL', 'quantity': 15, 'avg_cost': 138.90, 'current_price': 142.15},
                {'symbol': 'TSLA', 'quantity': 20, 'avg_cost': 248.75, 'current_price': 251.20},
                {'symbol': 'NVDA', 'quantity': 12, 'avg_cost': 875.40, 'current_price': 890.25}
            ],
            'orders': [
                {'order_id': 'ORD_001', 'symbol': 'AMD', 'side': 'BUY', 'quantity': 30, 'price': 145.75, 'status': 'OPEN', 'order_type': 'LIMIT'},
                {'order_id': 'ORD_002', 'symbol': 'QQQ', 'side': 'SELL', 'quantity': 25, 'price': 395.20, 'status': 'FILLED', 'order_type': 'MARKET'},
                {'order_id': 'ORD_003', 'symbol': 'META', 'side': 'BUY', 'quantity': 15, 'price': 485.50, 'status': 'CANCELLED', 'order_type': 'LIMIT'}
            ]
        }
    
    def connect(self, **kwargs) -> bool:
        """Simulate connection to demo account"""
        self.is_connected = True
        return True
    
    def get_account_info(self) -> Dict:
        """Get demo account information"""
        if self.is_connected:
            return self.demo_data['account']
        return {}
    
    def get_positions(self) -> pd.DataFrame:
        """Get demo positions"""
        if self.is_connected:
            positions_data = []
            for pos in self.demo_data['positions']:
                market_value = pos['quantity'] * pos['current_price']
                cost_basis = pos['quantity'] * pos['avg_cost']
                unrealized_pnl = market_value - cost_basis
                unrealized_pnl_pct = (unrealized_pnl / cost_basis) * 100 if cost_basis > 0 else 0
                
                positions_data.append({
                    'symbol': pos['symbol'],
                    'quantity': pos['quantity'],
                    'avg_cost': pos['avg_cost'],
                    'current_price': pos['current_price'],
                    'market_value': market_value,
                    'unrealized_pnl': unrealized_pnl,
                    'unrealized_pnl_pct': unrealized_pnl_pct
                })
            
            return pd.DataFrame(positions_data)
        return pd.DataFrame()
    
    def get_orders(self) -> pd.DataFrame:
        """Get demo orders"""
        if self.is_connected:
            return pd.DataFrame(self.demo_data['orders'])
        return pd.DataFrame()
    
    def update_prices(self):
        """Simulate price movements"""
        for pos in self.demo_data['positions']:
            # Random price movement ±2%
            change_pct = np.random.normal(0, 0.02)
            pos['current_price'] = max(0.01, pos['current_price'] * (1 + change_pct))
        
        # Update account value
        total_value = sum(pos['quantity'] * pos['current_price'] for pos in self.demo_data['positions'])
        self.demo_data['account']['account_value'] = total_value + self.demo_data['account']['cash_balance']
        self.demo_data['account']['last_updated'] = datetime.now()