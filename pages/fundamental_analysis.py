import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ui_components import (
    apply_custom_css, 
    create_metric_card, 
    create_enhanced_metric_card,
    create_info_box,
    display_success_box,
    display_warning_box,
    display_error_box,
    display_info_box
)

# Page configuration
st.set_page_config(
    page_title="Fundamental Analysis",
    page_icon="📊",
    layout="wide"
)

# Apply custom styling
apply_custom_css()

def get_company_financials(ticker: str) -> Dict:
    """Fetch comprehensive company financial data"""
    try:
        stock = yf.Ticker(ticker)
        
        # Get all financial data
        info = stock.info
        income_stmt = stock.financials
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow
        
        # Get historical price data for technical analysis
        hist_data = stock.history(period="2y")
        
        return {
            'info': info,
            'income_statement': income_stmt,
            'balance_sheet': balance_sheet,
            'cash_flow': cash_flow,
            'price_data': hist_data,
            'ticker': ticker
        }
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

def calculate_financial_ratios(data: Dict) -> Dict:
    """Calculate comprehensive financial ratios"""
    try:
        info = data['info']
        income_stmt = data['income_statement']
        balance_sheet = data['balance_sheet']
        
        ratios = {}
        
        # Basic metrics from info
        market_cap = info.get('marketCap', 0)
        shares_outstanding = info.get('sharesOutstanding', 1)
        current_price = info.get('currentPrice', 0)
        
        # Profitability Ratios
        if not income_stmt.empty and not balance_sheet.empty:
            try:
                latest_year = income_stmt.columns[0]
                
                # Revenue and earnings
                revenue = income_stmt.loc['Total Revenue', latest_year] if 'Total Revenue' in income_stmt.index else 0
                net_income = income_stmt.loc['Net Income', latest_year] if 'Net Income' in income_stmt.index else 0
                gross_profit = income_stmt.loc['Gross Profit', latest_year] if 'Gross Profit' in income_stmt.index else 0
                operating_income = income_stmt.loc['Operating Income', latest_year] if 'Operating Income' in income_stmt.index else 0
                
                # Balance sheet items
                total_assets = balance_sheet.loc['Total Assets', latest_year] if 'Total Assets' in balance_sheet.index else 0
                total_equity = balance_sheet.loc['Stockholders Equity', latest_year] if 'Stockholders Equity' in balance_sheet.index else 0
                current_assets = balance_sheet.loc['Current Assets', latest_year] if 'Current Assets' in balance_sheet.index else 0
                current_liabilities = balance_sheet.loc['Current Liabilities', latest_year] if 'Current Liabilities' in balance_sheet.index else 0
                total_debt = balance_sheet.loc['Total Debt', latest_year] if 'Total Debt' in balance_sheet.index else 0
                
                # Profitability Ratios
                ratios['gross_margin'] = (gross_profit / revenue * 100) if revenue != 0 else 0
                ratios['operating_margin'] = (operating_income / revenue * 100) if revenue != 0 else 0
                ratios['net_margin'] = (net_income / revenue * 100) if revenue != 0 else 0
                ratios['roa'] = (net_income / total_assets * 100) if total_assets != 0 else 0
                ratios['roe'] = (net_income / total_equity * 100) if total_equity != 0 else 0
                
                # Liquidity Ratios
                ratios['current_ratio'] = (current_assets / current_liabilities) if current_liabilities != 0 else 0
                ratios['quick_ratio'] = ratios['current_ratio']  # Simplified
                
                # Leverage Ratios
                ratios['debt_to_equity'] = (total_debt / total_equity) if total_equity != 0 else 0
                ratios['debt_to_assets'] = (total_debt / total_assets) if total_assets != 0 else 0
                
                # Store raw values for calculations
                ratios['revenue'] = revenue
                ratios['net_income'] = net_income
                ratios['total_assets'] = total_assets
                ratios['market_cap'] = market_cap
                
            except Exception as e:
                st.warning(f"Some financial ratios could not be calculated: {str(e)}")
        
        # Market Ratios (from info)
        ratios['pe_ratio'] = info.get('forwardPE', info.get('trailingPE', 0))
        ratios['pb_ratio'] = info.get('priceToBook', 0)
        ratios['ps_ratio'] = info.get('priceToSalesTrailing12Months', 0)
        ratios['peg_ratio'] = info.get('pegRatio', 0)
        ratios['ev_ebitda'] = info.get('enterpriseToEbitda', 0)
        ratios['dividend_yield'] = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
        
        # Growth metrics
        ratios['revenue_growth'] = info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 0
        ratios['earnings_growth'] = info.get('earningsGrowth', 0) * 100 if info.get('earningsGrowth') else 0
        
        # Efficiency ratios
        ratios['asset_turnover'] = (ratios['revenue'] / ratios['total_assets']) if ratios['total_assets'] != 0 else 0
        
        return ratios
        
    except Exception as e:
        st.error(f"Error calculating ratios: {str(e)}")
        return {}

def calculate_technical_ratios(price_data: pd.DataFrame) -> Dict:
    """Calculate technical analysis ratios and indicators"""
    try:
        tech_ratios = {}
        
        if price_data.empty:
            return tech_ratios
        
        close_prices = price_data['Close']
        high_prices = price_data['High']
        low_prices = price_data['Low']
        volume = price_data['Volume']
        
        # Moving averages
        sma_20 = close_prices.rolling(20).mean()
        sma_50 = close_prices.rolling(50).mean()
        sma_200 = close_prices.rolling(200).mean()
        ema_12 = close_prices.ewm(span=12).mean()
        ema_26 = close_prices.ewm(span=26).mean()
        
        # Technical indicators
        # RSI
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # MACD
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()
        histogram = macd - signal
        
        # Bollinger Bands
        bb_middle = close_prices.rolling(20).mean()
        bb_std = close_prices.rolling(20).std()
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)
        
        # Current values
        current_price = close_prices.iloc[-1]
        
        tech_ratios = {
            'current_price': current_price,
            'sma_20': sma_20.iloc[-1] if not sma_20.empty else 0,
            'sma_50': sma_50.iloc[-1] if not sma_50.empty else 0,
            'sma_200': sma_200.iloc[-1] if not sma_200.empty else 0,
            'rsi': rsi.iloc[-1] if not rsi.empty else 50,
            'macd': macd.iloc[-1] if not macd.empty else 0,
            'signal': signal.iloc[-1] if not signal.empty else 0,
            'bb_upper': bb_upper.iloc[-1] if not bb_upper.empty else 0,
            'bb_lower': bb_lower.iloc[-1] if not bb_lower.empty else 0,
            'bb_position': ((current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1]) * 100) if not bb_upper.empty and bb_upper.iloc[-1] != bb_lower.iloc[-1] else 50,
            'volume_avg_20': volume.rolling(20).mean().iloc[-1] if not volume.empty else 0,
            'price_vs_sma20': ((current_price - sma_20.iloc[-1]) / sma_20.iloc[-1] * 100) if not sma_20.empty and sma_20.iloc[-1] != 0 else 0,
            'price_vs_sma50': ((current_price - sma_50.iloc[-1]) / sma_50.iloc[-1] * 100) if not sma_50.empty and sma_50.iloc[-1] != 0 else 0,
            'price_vs_sma200': ((current_price - sma_200.iloc[-1]) / sma_200.iloc[-1] * 100) if not sma_200.empty and sma_200.iloc[-1] != 0 else 0,
        }
        
        # Volatility
        returns = close_prices.pct_change().dropna()
        tech_ratios['volatility_30d'] = returns.rolling(30).std().iloc[-1] * np.sqrt(252) * 100 if not returns.empty else 0
        tech_ratios['beta'] = 1.0  # Simplified - would need market data for accurate calculation
        
        return tech_ratios
        
    except Exception as e:
        st.error(f"Error calculating technical ratios: {str(e)}")
        return {}

def assess_financial_health(financial_ratios: Dict, technical_ratios: Dict, info: Dict) -> Dict:
    """Assess overall financial health and provide scoring"""
    
    health_assessment = {
        'overall_score': 0,
        'profitability_score': 0,
        'liquidity_score': 0,
        'leverage_score': 0,
        'efficiency_score': 0,
        'valuation_score': 0,
        'technical_score': 0,
        'strengths': [],
        'weaknesses': [],
        'recommendations': []
    }
    
    try:
        # Profitability Assessment (25 points)
        profitability_score = 0
        if financial_ratios.get('gross_margin', 0) > 40:
            profitability_score += 5
        if financial_ratios.get('operating_margin', 0) > 15:
            profitability_score += 5
        if financial_ratios.get('net_margin', 0) > 10:
            profitability_score += 5
        if financial_ratios.get('roe', 0) > 15:
            profitability_score += 5
        if financial_ratios.get('roa', 0) > 5:
            profitability_score += 5
        
        # Liquidity Assessment (15 points)
        liquidity_score = 0
        current_ratio = financial_ratios.get('current_ratio', 0)
        if current_ratio >= 1.5:
            liquidity_score += 8
        elif current_ratio >= 1.0:
            liquidity_score += 5
        elif current_ratio >= 0.8:
            liquidity_score += 2
        
        quick_ratio = financial_ratios.get('quick_ratio', 0)
        if quick_ratio >= 1.0:
            liquidity_score += 7
        elif quick_ratio >= 0.8:
            liquidity_score += 4
        
        # Leverage Assessment (15 points)
        leverage_score = 0
        debt_to_equity = financial_ratios.get('debt_to_equity', 0)
        if debt_to_equity < 0.3:
            leverage_score += 10
        elif debt_to_equity < 0.6:
            leverage_score += 7
        elif debt_to_equity < 1.0:
            leverage_score += 3
        
        debt_to_assets = financial_ratios.get('debt_to_assets', 0)
        if debt_to_assets < 0.3:
            leverage_score += 5
        elif debt_to_assets < 0.5:
            leverage_score += 3
        
        # Efficiency Assessment (15 points)
        efficiency_score = 0
        asset_turnover = financial_ratios.get('asset_turnover', 0)
        if asset_turnover > 1.0:
            efficiency_score += 8
        elif asset_turnover > 0.5:
            efficiency_score += 5
        
        if financial_ratios.get('revenue_growth', 0) > 10:
            efficiency_score += 4
        elif financial_ratios.get('revenue_growth', 0) > 5:
            efficiency_score += 2
        
        if financial_ratios.get('earnings_growth', 0) > 15:
            efficiency_score += 3
        elif financial_ratios.get('earnings_growth', 0) > 5:
            efficiency_score += 1
        
        # Valuation Assessment (15 points)
        valuation_score = 0
        pe_ratio = financial_ratios.get('pe_ratio', 0)
        if 10 <= pe_ratio <= 20:
            valuation_score += 5
        elif 5 <= pe_ratio <= 30:
            valuation_score += 3
        
        pb_ratio = financial_ratios.get('pb_ratio', 0)
        if 0.5 <= pb_ratio <= 3.0:
            valuation_score += 5
        elif pb_ratio <= 5.0:
            valuation_score += 3
        
        peg_ratio = financial_ratios.get('peg_ratio', 0)
        if 0.5 <= peg_ratio <= 1.5:
            valuation_score += 5
        elif peg_ratio <= 2.0:
            valuation_score += 3
        
        # Technical Assessment (15 points)
        technical_score = 0
        rsi = technical_ratios.get('rsi', 50)
        if 40 <= rsi <= 60:
            technical_score += 3
        elif 30 <= rsi <= 70:
            technical_score += 2
        
        # Trend analysis
        if technical_ratios.get('price_vs_sma20', 0) > 0:
            technical_score += 3
        if technical_ratios.get('price_vs_sma50', 0) > 0:
            technical_score += 3
        if technical_ratios.get('price_vs_sma200', 0) > 0:
            technical_score += 6
        
        # Store scores
        health_assessment['profitability_score'] = profitability_score
        health_assessment['liquidity_score'] = liquidity_score
        health_assessment['leverage_score'] = leverage_score
        health_assessment['efficiency_score'] = efficiency_score
        health_assessment['valuation_score'] = valuation_score
        health_assessment['technical_score'] = technical_score
        health_assessment['overall_score'] = profitability_score + liquidity_score + leverage_score + efficiency_score + valuation_score + technical_score
        
        # Identify strengths and weaknesses
        if profitability_score >= 20:
            health_assessment['strengths'].append("Strong profitability metrics")
        elif profitability_score <= 10:
            health_assessment['weaknesses'].append("Weak profitability performance")
            
        if liquidity_score >= 12:
            health_assessment['strengths'].append("Good liquidity position")
        elif liquidity_score <= 6:
            health_assessment['weaknesses'].append("Liquidity concerns")
            
        if leverage_score >= 12:
            health_assessment['strengths'].append("Conservative debt levels")
        elif leverage_score <= 6:
            health_assessment['weaknesses'].append("High leverage risk")
            
        if technical_score >= 12:
            health_assessment['strengths'].append("Positive technical momentum")
        elif technical_score <= 6:
            health_assessment['weaknesses'].append("Weak technical outlook")
        
        # Generate recommendations
        overall_score = health_assessment['overall_score']
        if overall_score >= 80:
            health_assessment['recommendations'].append("Strong Buy - Excellent financial health across all metrics")
        elif overall_score >= 65:
            health_assessment['recommendations'].append("Buy - Good financial position with minor areas for improvement")
        elif overall_score >= 50:
            health_assessment['recommendations'].append("Hold - Mixed signals, monitor key metrics closely")
        elif overall_score >= 35:
            health_assessment['recommendations'].append("Weak Hold - Significant concerns, consider reducing position")
        else:
            health_assessment['recommendations'].append("Sell - Poor financial health, high investment risk")
            
    except Exception as e:
        st.error(f"Error in health assessment: {str(e)}")
    
    return health_assessment

def generate_future_predictions(financial_ratios: Dict, technical_ratios: Dict, info: Dict) -> Dict:
    """Generate future predictions based on current metrics"""
    
    predictions = {
        'price_target_1y': 0,
        'earnings_forecast': '',
        'revenue_forecast': '',
        'risk_factors': [],
        'growth_catalysts': [],
        'key_metrics_to_watch': []
    }
    
    try:
        current_price = technical_ratios.get('current_price', 0)
        
        # Simple price target calculation based on multiple metrics
        target_multiplier = 1.0
        
        # Growth-based adjustment
        revenue_growth = financial_ratios.get('revenue_growth', 0)
        earnings_growth = financial_ratios.get('earnings_growth', 0)
        
        if revenue_growth > 15:
            target_multiplier += 0.2
        elif revenue_growth > 10:
            target_multiplier += 0.1
        elif revenue_growth < 0:
            target_multiplier -= 0.15
            
        if earnings_growth > 20:
            target_multiplier += 0.25
        elif earnings_growth > 10:
            target_multiplier += 0.15
        elif earnings_growth < -10:
            target_multiplier -= 0.2
        
        # Valuation adjustment
        pe_ratio = financial_ratios.get('pe_ratio', 15)
        if pe_ratio > 30:
            target_multiplier -= 0.1
        elif pe_ratio < 10:
            target_multiplier += 0.1
        
        # Technical adjustment
        if technical_ratios.get('rsi', 50) > 70:
            target_multiplier -= 0.05
        elif technical_ratios.get('rsi', 50) < 30:
            target_multiplier += 0.05
            
        predictions['price_target_1y'] = current_price * target_multiplier
        
        # Earnings forecast
        if earnings_growth > 15:
            predictions['earnings_forecast'] = "Strong earnings growth expected"
        elif earnings_growth > 5:
            predictions['earnings_forecast'] = "Moderate earnings growth anticipated"
        elif earnings_growth > -5:
            predictions['earnings_forecast'] = "Stable earnings expected"
        else:
            predictions['earnings_forecast'] = "Earnings decline risk"
            
        # Revenue forecast
        if revenue_growth > 10:
            predictions['revenue_forecast'] = "Revenue growth acceleration likely"
        elif revenue_growth > 5:
            predictions['revenue_forecast'] = "Steady revenue growth expected"
        else:
            predictions['revenue_forecast'] = "Revenue growth challenges ahead"
        
        # Risk factors
        if financial_ratios.get('debt_to_equity', 0) > 0.8:
            predictions['risk_factors'].append("High debt levels pose refinancing risk")
        if financial_ratios.get('current_ratio', 1) < 1.0:
            predictions['risk_factors'].append("Liquidity constraints may impact operations")
        if financial_ratios.get('net_margin', 0) < 5:
            predictions['risk_factors'].append("Low profit margins vulnerable to cost pressures")
        if technical_ratios.get('volatility_30d', 0) > 40:
            predictions['risk_factors'].append("High volatility increases investment risk")
            
        # Growth catalysts
        if financial_ratios.get('roe', 0) > 15:
            predictions['growth_catalysts'].append("Strong ROE indicates efficient capital allocation")
        if financial_ratios.get('asset_turnover', 0) > 1.0:
            predictions['growth_catalysts'].append("High asset turnover supports revenue growth")
        if financial_ratios.get('gross_margin', 0) > 40:
            predictions['growth_catalysts'].append("Strong gross margins provide pricing power")
        if revenue_growth > 10:
            predictions['growth_catalysts'].append("Revenue momentum supports market expansion")
            
        # Key metrics to watch
        predictions['key_metrics_to_watch'] = [
            "Quarterly earnings growth trends",
            "Debt-to-equity ratio changes",
            "Cash flow from operations",
            "Market share evolution",
            "Competitive positioning"
        ]
        
    except Exception as e:
        st.error(f"Error generating predictions: {str(e)}")
    
    return predictions

def display_financial_dashboard(ticker: str, financial_ratios: Dict, technical_ratios: Dict, health_assessment: Dict, predictions: Dict, info: Dict):
    """Display comprehensive financial dashboard"""
    
    # Company overview
    st.markdown(f"### 🏢 {info.get('longName', ticker)} ({ticker})")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        market_cap = info.get('marketCap', 0)
        if market_cap > 1e9:
            market_cap_str = f"${market_cap/1e9:.1f}B"
        elif market_cap > 1e6:
            market_cap_str = f"${market_cap/1e6:.1f}M"
        else:
            market_cap_str = f"${market_cap:,.0f}"
        create_enhanced_metric_card("Market Cap", market_cap_str, icon="💰")
    
    with col2:
        current_price = technical_ratios.get('current_price', 0)
        create_enhanced_metric_card("Current Price", f"${current_price:.2f}", icon="📈")
    
    with col3:
        sector = info.get('sector', 'N/A')
        create_enhanced_metric_card("Sector", sector, icon="🏭")
    
    with col4:
        overall_score = health_assessment.get('overall_score', 0)
        score_color = "green" if overall_score >= 70 else "orange" if overall_score >= 50 else "red"
        create_enhanced_metric_card("Health Score", f"{overall_score}/100", icon="💊")
    
    st.markdown("---")
    
    # Financial Ratios Section
    st.markdown("### 📊 Financial Ratios Analysis")
    
    # Profitability ratios
    st.markdown("#### 💰 Profitability Ratios")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_enhanced_metric_card("Gross Margin", f"{financial_ratios.get('gross_margin', 0):.1f}%", icon="💵")
    with col2:
        create_enhanced_metric_card("Operating Margin", f"{financial_ratios.get('operating_margin', 0):.1f}%", icon="⚙️")
    with col3:
        create_enhanced_metric_card("Net Margin", f"{financial_ratios.get('net_margin', 0):.1f}%", icon="💎")
    with col4:
        create_enhanced_metric_card("ROE", f"{financial_ratios.get('roe', 0):.1f}%", icon="📊")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        create_enhanced_metric_card("ROA", f"{financial_ratios.get('roa', 0):.1f}%", icon="🏢")
    with col2:
        create_enhanced_metric_card("Revenue Growth", f"{financial_ratios.get('revenue_growth', 0):.1f}%", icon="📈")
    with col3:
        create_enhanced_metric_card("Earnings Growth", f"{financial_ratios.get('earnings_growth', 0):.1f}%", icon="💹")
    with col4:
        create_enhanced_metric_card("Asset Turnover", f"{financial_ratios.get('asset_turnover', 0):.2f}x", icon="🔄")
    
    # Liquidity and Leverage ratios
    st.markdown("#### 🌊 Liquidity & Leverage Ratios")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_enhanced_metric_card("Current Ratio", f"{financial_ratios.get('current_ratio', 0):.2f}", icon="💧")
    with col2:
        create_enhanced_metric_card("Quick Ratio", f"{financial_ratios.get('quick_ratio', 0):.2f}", icon="⚡")
    with col3:
        create_enhanced_metric_card("Debt-to-Equity", f"{financial_ratios.get('debt_to_equity', 0):.2f}", icon="⚖️")
    with col4:
        create_enhanced_metric_card("Debt-to-Assets", f"{financial_ratios.get('debt_to_assets', 0):.2f}", icon="🏗️")
    
    # Valuation ratios
    st.markdown("#### 💎 Valuation Ratios")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        create_enhanced_metric_card("P/E Ratio", f"{financial_ratios.get('pe_ratio', 0):.1f}", icon="📊")
    with col2:
        create_enhanced_metric_card("P/B Ratio", f"{financial_ratios.get('pb_ratio', 0):.2f}", icon="📚")
    with col3:
        create_enhanced_metric_card("P/S Ratio", f"{financial_ratios.get('ps_ratio', 0):.2f}", icon="💰")
    with col4:
        create_enhanced_metric_card("PEG Ratio", f"{financial_ratios.get('peg_ratio', 0):.2f}", icon="📈")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        create_enhanced_metric_card("EV/EBITDA", f"{financial_ratios.get('ev_ebitda', 0):.1f}", icon="🏭")
    with col2:
        create_enhanced_metric_card("Dividend Yield", f"{financial_ratios.get('dividend_yield', 0):.2f}%", icon="💰")
    with col3:
        st.empty()
    with col4:
        st.empty()
    
    st.markdown("---")
    
    # Technical Analysis Section
    st.markdown("### 📈 Technical Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        rsi = technical_ratios.get('rsi', 50)
        rsi_color = "red" if rsi > 70 else "green" if rsi < 30 else "orange"
        create_enhanced_metric_card("RSI", f"{rsi:.1f}", icon="📊")
    
    with col2:
        macd = technical_ratios.get('macd', 0)
        create_enhanced_metric_card("MACD", f"{macd:.3f}", icon="📈")
    
    with col3:
        bb_position = technical_ratios.get('bb_position', 50)
        create_enhanced_metric_card("Bollinger Position", f"{bb_position:.1f}%", icon="📊")
    
    with col4:
        volatility = technical_ratios.get('volatility_30d', 0)
        create_enhanced_metric_card("30D Volatility", f"{volatility:.1f}%", icon="📊")
    
    # Moving averages
    st.markdown("#### 📊 Moving Average Analysis")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        price_vs_sma20 = technical_ratios.get('price_vs_sma20', 0)
        create_enhanced_metric_card("vs SMA 20", f"{price_vs_sma20:+.1f}%", icon="📈")
    with col2:
        price_vs_sma50 = technical_ratios.get('price_vs_sma50', 0)
        create_enhanced_metric_card("vs SMA 50", f"{price_vs_sma50:+.1f}%", icon="📈")
    with col3:
        price_vs_sma200 = technical_ratios.get('price_vs_sma200', 0)
        create_enhanced_metric_card("vs SMA 200", f"{price_vs_sma200:+.1f}%", icon="📈")
    
    st.markdown("---")
    
    # Health Assessment Section
    st.markdown("### 🏥 Financial Health Assessment")
    
    # Health scores
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Score breakdown chart
        categories = ['Profitability', 'Liquidity', 'Leverage', 'Efficiency', 'Valuation', 'Technical']
        scores = [
            health_assessment.get('profitability_score', 0),
            health_assessment.get('liquidity_score', 0), 
            health_assessment.get('leverage_score', 0),
            health_assessment.get('efficiency_score', 0),
            health_assessment.get('valuation_score', 0),
            health_assessment.get('technical_score', 0)
        ]
        max_scores = [25, 15, 15, 15, 15, 15]
        
        fig_radar = go.Figure()
        
        fig_radar.add_trace(go.Scatterpolar(
            r=[s/m*100 for s, m in zip(scores, max_scores)],
            theta=categories,
            fill='toself',
            name='Current Score',
            line_color='cyan'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=False,
            title="Health Score Breakdown",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with col2:
        # Strengths
        st.markdown("#### ✅ Strengths")
        for strength in health_assessment.get('strengths', []):
            st.markdown(f"• {strength}")
        
        if not health_assessment.get('strengths'):
            st.markdown("*No significant strengths identified*")
    
    with col3:
        # Weaknesses
        st.markdown("#### ⚠️ Weaknesses")
        for weakness in health_assessment.get('weaknesses', []):
            st.markdown(f"• {weakness}")
        
        if not health_assessment.get('weaknesses'):
            st.markdown("*No major weaknesses identified*")
    
    # Recommendations
    st.markdown("#### 💡 Investment Recommendation")
    for recommendation in health_assessment.get('recommendations', []):
        if "Strong Buy" in recommendation:
            display_success_box("Investment Recommendation", recommendation)
        elif "Buy" in recommendation:
            display_info_box("Investment Recommendation", recommendation)
        elif "Hold" in recommendation:
            display_warning_box("Investment Recommendation", recommendation)
        else:
            display_error_box("Investment Recommendation", recommendation)
    
    st.markdown("---")
    
    # Future Predictions Section
    st.markdown("### 🔮 Future Predictions & Outlook")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Price & Earnings Forecast")
        
        current_price = technical_ratios.get('current_price', 0)
        target_price = predictions.get('price_target_1y', 0)
        price_change = ((target_price - current_price) / current_price * 100) if current_price != 0 else 0
        
        create_enhanced_metric_card("1-Year Price Target", f"${target_price:.2f}", price_change, "🎯")
        
        st.markdown(f"**Earnings Outlook:** {predictions.get('earnings_forecast', 'N/A')}")
        st.markdown(f"**Revenue Outlook:** {predictions.get('revenue_forecast', 'N/A')}")
    
    with col2:
        st.markdown("#### 📊 Key Metrics to Monitor")
        for metric in predictions.get('key_metrics_to_watch', []):
            st.markdown(f"• {metric}")
    
    # Risk factors and catalysts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ⚠️ Risk Factors")
        risk_factors = predictions.get('risk_factors', [])
        if risk_factors:
            for risk in risk_factors:
                st.markdown(f"• {risk}")
        else:
            st.markdown("*No significant risk factors identified*")
    
    with col2:
        st.markdown("#### 🚀 Growth Catalysts")
        catalysts = predictions.get('growth_catalysts', [])
        if catalysts:
            for catalyst in catalysts:
                st.markdown(f"• {catalyst}")
        else:
            st.markdown("*Limited growth catalysts identified*")

def main():
    # Header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; text-align: center; margin: 0; font-size: 2.5rem;">
            📊 Fundamental Analysis & Health Assessment
        </h1>
        <p style="color: #e0e0e0; text-align: center; margin: 0.5rem 0 0 0; font-size: 1.2rem;">
            Comprehensive Financial Analysis • Health Scoring • Future Predictions
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    **Comprehensive fundamental analysis combining financial ratios, technical indicators, 
    and health assessment with AI-powered future predictions for informed investment decisions.**
    """)
    
    # Ticker input
    col1, col2 = st.columns([2, 1])
    
    with col1:
        ticker = st.text_input(
            "Enter Stock Ticker Symbol",
            placeholder="e.g., AAPL, MSFT, TSLA, GOOGL",
            help="Enter a valid stock ticker symbol for analysis"
        ).upper()
    
    with col2:
        if st.button("🔍 Analyze Company", type="primary", use_container_width=True):
            if ticker:
                with st.spinner(f"Fetching comprehensive data for {ticker}..."):
                    # Get company data
                    company_data = get_company_financials(ticker)
                    
                    if company_data:
                        # Calculate all ratios
                        financial_ratios = calculate_financial_ratios(company_data)
                        technical_ratios = calculate_technical_ratios(company_data['price_data'])
                        
                        # Assess health
                        health_assessment = assess_financial_health(
                            financial_ratios, 
                            technical_ratios, 
                            company_data['info']
                        )
                        
                        # Generate predictions
                        predictions = generate_future_predictions(
                            financial_ratios,
                            technical_ratios, 
                            company_data['info']
                        )
                        
                        # Display dashboard
                        display_financial_dashboard(
                            ticker,
                            financial_ratios,
                            technical_ratios,
                            health_assessment,
                            predictions,
                            company_data['info']
                        )
                        
                        st.success(f"✅ Complete analysis generated for {ticker}")
                    else:
                        st.error("Unable to fetch data. Please check the ticker symbol and try again.")
            else:
                st.warning("Please enter a ticker symbol")
    
    # Educational section
    st.markdown("---")
    st.markdown("### 📚 Understanding the Analysis")
    
    with st.expander("💡 Financial Ratios Explained"):
        st.markdown("""
        **Profitability Ratios:**
        - **Gross Margin**: Revenue minus cost of goods sold, as percentage of revenue
        - **Operating Margin**: Operating income as percentage of revenue  
        - **Net Margin**: Net income as percentage of revenue
        - **ROE**: Return on Equity - net income divided by shareholders' equity
        - **ROA**: Return on Assets - net income divided by total assets
        
        **Liquidity Ratios:**
        - **Current Ratio**: Current assets divided by current liabilities
        - **Quick Ratio**: Quick assets divided by current liabilities
        
        **Leverage Ratios:**
        - **Debt-to-Equity**: Total debt divided by shareholders' equity
        - **Debt-to-Assets**: Total debt divided by total assets
        
        **Valuation Ratios:**
        - **P/E Ratio**: Price per share divided by earnings per share
        - **P/B Ratio**: Price per share divided by book value per share
        - **PEG Ratio**: P/E ratio divided by earnings growth rate
        """)
    
    with st.expander("📈 Technical Indicators Guide"):
        st.markdown("""
        **Key Technical Indicators:**
        - **RSI**: Relative Strength Index (0-100) - measures overbought/oversold conditions
        - **MACD**: Moving Average Convergence Divergence - trend following momentum indicator
        - **Bollinger Bands**: Price channel based on standard deviation
        - **Moving Averages**: SMA 20, 50, 200 - trend identification
        
        **Interpretation:**
        - RSI > 70: Potentially overbought
        - RSI < 30: Potentially oversold
        - Price above moving averages: Bullish trend
        - Price below moving averages: Bearish trend
        """)
    
    with st.expander("🏥 Health Score Methodology"):
        st.markdown("""
        **Health Score Components (100 points total):**
        - **Profitability (25 points)**: Margins, ROE, ROA performance
        - **Liquidity (15 points)**: Current ratio, quick ratio adequacy
        - **Leverage (15 points)**: Debt levels and financial stability
        - **Efficiency (15 points)**: Asset utilization and growth metrics
        - **Valuation (15 points)**: Price ratios relative to fundamentals
        - **Technical (15 points)**: Price momentum and trend analysis
        
        **Score Interpretation:**
        - 80-100: Excellent financial health
        - 65-79: Good financial condition
        - 50-64: Average performance
        - 35-49: Below average, caution advised
        - 0-34: Poor financial health, high risk
        """)

if __name__ == "__main__":
    main()