#!/usr/bin/env python3
"""
Temporary script to set StockData API key in environment
"""
import os
import streamlit as st

# Set the API key in environment
os.environ["STOCKDATA_API_KEY"] = "uh8kCdBkyEjbME9WtzMPiwMkgcNOyARSgJe34mIq"

print("✅ StockData API key configured successfully!")
print(f"API Key: {os.environ.get('STOCKDATA_API_KEY')[:10]}...")

# Test the API connection
try:
    from utils.stockdata_client import StockDataClient
    client = StockDataClient()
    if client.test_connection():
        print("✅ StockData API connection test successful!")
    else:
        print("❌ StockData API connection test failed")
except Exception as e:
    print(f"Error testing connection: {e}")