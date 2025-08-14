import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime, timedelta
import requests
from typing import Dict, List
import yfinance as yf

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ui_components import apply_custom_css, create_metric_card, create_info_box
from utils.web_scraper import get_website_text_content
from utils.news_scraper import NewsAndDataScraper, NewsArticle, EconomicEvent

# Page configuration
st.set_page_config(
    page_title="News & Economic Data",
    page_icon="📰",
    layout="wide"
)

# Apply custom styling
apply_custom_css()

def get_economic_calendar_data():
    """Get today's key economic events and data releases"""
    today = datetime.now()
    
    # Key economic data sources with release schedules
    economic_events = {
        "Daily": [
            {"time": "8:30 AM ET", "event": "Initial Jobless Claims", "impact": "High", "frequency": "Thursday"},
            {"time": "10:00 AM ET", "event": "Treasury Auctions", "impact": "Medium", "frequency": "Various"},
            {"time": "2:00 PM ET", "event": "FOMC Meeting Minutes", "impact": "Very High", "frequency": "As scheduled"},
        ],
        "Weekly": [
            {"event": "Oil Inventory Report", "impact": "Medium-High", "day": "Wednesday"},
            {"event": "Money Supply (M2)", "impact": "Medium", "day": "Thursday"},
            {"event": "Baker Hughes Rig Count", "impact": "Low", "day": "Friday"},
        ],
        "Monthly": [
            {"event": "Non-Farm Payrolls", "impact": "Very High", "day": "First Friday"},
            {"event": "Consumer Price Index (CPI)", "impact": "Very High", "day": "Mid-month"},
            {"event": "Federal Reserve Interest Rate Decision", "impact": "Extremely High", "day": "FOMC Schedule"},
            {"event": "Retail Sales", "impact": "High", "day": "Mid-month"},
            {"event": "Industrial Production", "impact": "Medium-High", "day": "Mid-month"},
            {"event": "Consumer Confidence", "impact": "Medium-High", "day": "Last Tuesday"},
        ]
    }
    
    return economic_events

def get_market_moving_news_sources():
    """Get list of critical news sources for traders"""
    return {
        "Economic Data": {
            "Bureau of Labor Statistics": "https://www.bls.gov/news.release/",
            "Bureau of Economic Analysis": "https://www.bea.gov/news/",
            "Federal Reserve Economic Data (FRED)": "https://fred.stlouisfed.org/",
            "Treasury Department": "https://home.treasury.gov/news/press-releases",
            "Census Bureau": "https://www.census.gov/economic-indicators/",
        },
        "Federal Reserve": {
            "Fed News & Events": "https://www.federalreserve.gov/newsevents/",
            "FOMC Statements": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
            "Fed Speeches": "https://www.federalreserve.gov/newsevents/speech/",
            "Beige Book": "https://www.federalreserve.gov/monetarypolicy/beigebook/",
        },
        "Market News": {
            "Bloomberg Markets": "https://www.bloomberg.com/markets",
            "Reuters Business": "https://www.reuters.com/business/",
            "MarketWatch": "https://www.marketwatch.com/",
            "CNBC Markets": "https://www.cnbc.com/markets/",
            "Financial Times": "https://www.ft.com/markets",
        },
        "Earnings & Corporate": {
            "SEC Edgar Database": "https://www.sec.gov/edgar/",
            "Earnings Calendar": "https://finance.yahoo.com/calendar/earnings",
            "Corporate Press Releases": "https://www.prnewswire.com/",
        },
        "International": {
            "European Central Bank": "https://www.ecb.europa.eu/press/html/index.en.html",
            "Bank of Japan": "https://www.boj.or.jp/en/",
            "Bank of England": "https://www.bankofengland.co.uk/news",
            "China Economic Data": "http://www.stats.gov.cn/english/",
        }
    }

def scrape_fed_news():
    """Scrape latest Federal Reserve news and announcements"""
    try:
        fed_url = "https://www.federalreserve.gov/newsevents/"
        content = get_website_text_content(fed_url)
        
        if content:
            return {
                "source": "Federal Reserve",
                "content": content[:1000] + "..." if len(content) > 1000 else content,
                "timestamp": datetime.now(),
                "url": fed_url
            }
    except Exception as e:
        return {"error": f"Unable to fetch Fed news: {str(e)}"}
    
    return None

def scrape_bls_data():
    """Scrape latest Bureau of Labor Statistics releases"""
    try:
        bls_url = "https://www.bls.gov/news.release/"
        content = get_website_text_content(bls_url)
        
        if content:
            return {
                "source": "Bureau of Labor Statistics",
                "content": content[:1000] + "..." if len(content) > 1000 else content,
                "timestamp": datetime.now(),
                "url": bls_url
            }
    except Exception as e:
        return {"error": f"Unable to fetch BLS data: {str(e)}"}
    
    return None

def get_market_data_summary():
    """Get current market data for context"""
    try:
        # Major indices
        indices = {
            "S&P 500": "^GSPC",
            "NASDAQ": "^IXIC", 
            "Dow Jones": "^DJI",
            "VIX": "^VIX",
            "10-Year Treasury": "^TNX"
        }
        
        market_data = {}
        for name, symbol in indices.items():
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d")
            if not hist.empty:
                current = hist['Close'].iloc[-1]
                previous = hist['Close'].iloc[-2] if len(hist) > 1 else current
                change = ((current - previous) / previous) * 100
                
                market_data[name] = {
                    "current": current,
                    "change": change,
                    "symbol": symbol
                }
        
        return market_data
        
    except Exception as e:
        st.error(f"Unable to fetch market data: {str(e)}")
        return {}

def main():
    st.title("📰 Live News & Economic Data for Trading")
    
    # Initialize news scraper
    news_scraper = NewsAndDataScraper()
    
    # Header with current market context
    st.markdown("""
    **Real-time market intelligence combining news analysis, economic data releases, 
    and trading-relevant information from authoritative sources.**
    """)
    
    # Create tabs for different data types
    tabs = st.tabs([
        "📊 Today's Economic Calendar",
        "🏦 Federal Reserve News", 
        "📈 Market-Moving Data",
        "🔥 High-Impact Stock News",
        "🌍 International Updates",
        "📰 News Sources Directory",
        "⚡ Live Market Context"
    ])
    
    # Today's Economic Calendar
    with tabs[0]:
        st.header("📅 Today's Economic Calendar")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🎯 Key Events Today")
            
            economic_events = get_economic_calendar_data()
            
            # Daily events
            st.write("**🕐 Daily Schedule:**")
            for event in economic_events["Daily"]:
                importance_color = {
                    "Very High": "🔴",
                    "High": "🟠", 
                    "Medium": "🟡",
                    "Low": "🟢"
                }
                color = importance_color.get(event["impact"], "⚪")
                
                st.write(f"{color} **{event['time']}** - {event['event']} (*{event['impact']} Impact*)")
            
            st.write("**📅 Weekly & Monthly Schedule:**")
            for freq, events in economic_events.items():
                if freq != "Daily":
                    st.write(f"**{freq} Events:**")
                    for event in events:
                        impact_color = {
                            "Extremely High": "🔴",
                            "Very High": "🔴",
                            "High": "🟠",
                            "Medium-High": "🟠",
                            "Medium": "🟡",
                            "Low": "🟢"
                        }
                        color = impact_color.get(event["impact"], "⚪")
                        timing = event.get("day", event.get("frequency", ""))
                        st.write(f"{color} {event['event']} - *{timing}* ({event['impact']} Impact)")
        
        with col2:
            st.subheader("⏰ Market Hours")
            
            now = datetime.now()
            market_hours = {
                "Pre-Market": "4:00 AM - 9:30 AM ET",
                "Regular Hours": "9:30 AM - 4:00 PM ET", 
                "After Hours": "4:00 PM - 8:00 PM ET"
            }
            
            for session, hours in market_hours.items():
                st.write(f"**{session}:** {hours}")
            
            st.info("**🎯 Trading Tip:** Major economic releases typically occur at 8:30 AM ET and 10:00 AM ET")
    
    # Federal Reserve News
    with tabs[1]:
        st.header("🏦 Federal Reserve Intelligence")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("**Latest Federal Reserve news and policy updates:**")
            
            if st.button("🔄 Fetch Latest Fed News", type="primary"):
                with st.spinner("Retrieving Federal Reserve updates..."):
                    fed_articles = news_scraper.scrape_federal_reserve_news()
                    
                    if fed_articles:
                        st.success(f"✅ Successfully retrieved {len(fed_articles)} Fed updates")
                        
                        for article in fed_articles:
                            with st.expander(f"📰 {article.title}", expanded=True):
                                st.write(f"**Source:** {article.source}")
                                st.write(f"**Retrieved:** {article.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                                st.write(f"**Market Impact:** {article.market_impact.upper()}")
                                st.write(f"**URL:** {article.url}")
                                st.write("---")
                                st.write(article.content)
                                
                                if article.tickers_mentioned:
                                    st.write("**Tickers Mentioned:**", ", ".join(article.tickers_mentioned))
                    else:
                        st.warning("No Federal Reserve news retrieved at this time")
        
        with col2:
            # FOMC Schedule
            st.subheader("📅 FOMC Schedule")
            fomc_meetings = news_scraper.get_fomc_schedule()
            
            for meeting in fomc_meetings:
                st.write(f"**{meeting['date']}**")
                st.write(f"Days until: {meeting['days_until']}")
                if meeting['press_conference']:
                    st.write("🎤 Press Conference")
                st.write("---")
        
        st.subheader("🎯 Key Fed Communication Channels")
        
        fed_channels = {
            "FOMC Statements": "Policy decisions and rate announcements",
            "Meeting Minutes": "Detailed discussions 3 weeks after meetings", 
            "Fed Speeches": "Individual Fed officials' perspectives",
            "Beige Book": "Regional economic conditions report",
            "Press Conferences": "Chair Powell's quarterly briefings"
        }
        
        for channel, description in fed_channels.items():
            st.write(f"**{channel}:** {description}")
    
    # Market-Moving Data  
    with tabs[2]:
        st.header("📈 Market-Moving Economic Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔢 Labor Market Data")
            if st.button("📊 Get Latest BLS Data"):
                with st.spinner("Fetching Bureau of Labor Statistics data..."):
                    bls_articles = news_scraper.scrape_bls_releases()
                    
                    if bls_articles:
                        st.success("✅ Retrieved latest labor market data")
                        
                        for article in bls_articles:
                            with st.expander(f"📊 {article.title}", expanded=True):
                                st.write(f"**Source:** {article.source}")
                                st.write(f"**Retrieved:** {article.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                                st.write(f"**Sentiment:** {article.sentiment.upper()}")
                                st.write("---")
                                st.write(article.content[:1500] + "..." if len(article.content) > 1500 else article.content)
                    else:
                        st.warning("No BLS data retrieved")
            
            # Economic Events Today
            st.subheader("📅 Today's Economic Events")
            economic_events = news_scraper.get_economic_calendar_today()
            
            if economic_events:
                for event in economic_events:
                    impact_color = {
                        "very_high": "🔴",
                        "high": "🟠",
                        "medium": "🟡",
                        "low": "🟢"
                    }
                    color = impact_color.get(event.impact, "⚪")
                    
                    st.write(f"{color} **{event.time}** - {event.name}")
                    st.write(f"   Impact: {event.impact.upper()}")
                    st.write(f"   Description: {event.description}")
                    st.write("---")
            else:
                st.info("No major economic events scheduled for today")
        
        with col2:
            st.subheader("💹 Key Economic Indicators")
            
            key_indicators = {
                "Non-Farm Payrolls": "Employment growth - first Friday monthly",
                "CPI (Inflation)": "Price changes - mid-month release",
                "GDP Growth": "Economic output - quarterly release",
                "Retail Sales": "Consumer spending - mid-month",
                "PMI Manufacturing": "Business activity - first business day",
                "Initial Claims": "Weekly unemployment - every Thursday"
            }
            
            for indicator, description in key_indicators.items():
                st.write(f"**{indicator}:** {description}")
    
    # High-Impact Stock News
    with tabs[3]:
        st.header("🔥 High-Impact Stock News & Headlines")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📰 Live Stock Market Headlines")
            
            if st.button("🔄 Fetch Stock Market News", type="primary"):
                with st.spinner("Retrieving latest stock market news..."):
                    stock_articles = []
                    
                    # Get MarketWatch headlines
                    mw_articles = news_scraper.scrape_marketwatch_headlines()
                    stock_articles.extend(mw_articles)
                    
                    # Get Reuters business news
                    reuters_articles = news_scraper.scrape_reuters_business_news()
                    stock_articles.extend(reuters_articles)
                    
                    # Get CNBC market news
                    cnbc_articles = news_scraper.scrape_cnbc_market_news()
                    stock_articles.extend(cnbc_articles)
                    
                    if stock_articles:
                        st.success(f"✅ Retrieved {len(stock_articles)} stock market updates")
                        
                        for article in stock_articles:
                            with st.expander(f"📈 {article.title}", expanded=True):
                                st.write(f"**Source:** {article.source}")
                                st.write(f"**Retrieved:** {article.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                                st.write(f"**Market Impact:** {article.market_impact.upper()}")
                                st.write(f"**Sentiment:** {article.sentiment.upper()}")
                                st.write("---")
                                st.write(article.content)
                                
                                if article.tickers_mentioned:
                                    st.write("**Tickers Mentioned:**", ", ".join(article.tickers_mentioned[:10]))
                    else:
                        st.warning("No stock market news retrieved at this time")
            
            st.subheader("🎯 Market-Moving Headlines Categories")
            
            high_impact_headlines = news_scraper.get_high_impact_stock_headlines()
            
            for headline in high_impact_headlines:
                impact_color = {
                    "very_high": "🔴",
                    "high": "🟠",
                    "medium_high": "🟡",
                    "medium": "🟢"
                }
                color = impact_color.get(headline["impact"], "⚪")
                
                st.write(f"{color} **{headline['headline']}**")
                st.write(f"   {headline['description']}")
                st.write(f"   *Category: {headline['category']} | Timing: {headline['timing']}*")
                st.write("---")
        
        with col2:
            st.subheader("📊 Sector-Specific News Impact")
            
            sector_news = news_scraper.get_sector_specific_news()
            
            selected_sector = st.selectbox(
                "Choose sector for detailed news analysis:",
                list(sector_news.keys()),
                key="sector_select"
            )
            
            if selected_sector:
                st.write(f"**{selected_sector} Sector - Key News Drivers:**")
                
                for i, news_item in enumerate(sector_news[selected_sector], 1):
                    st.write(f"{i}. {news_item}")
            
            st.subheader("📅 Major Earnings This Week")
            
            earnings_calendar = news_scraper.get_earnings_calendar_this_week()
            
            for earning in earnings_calendar[:5]:  # Top 5 most impactful
                impact_color = {
                    "very_high": "🔴",
                    "high": "🟠", 
                    "medium": "🟡"
                }
                color = impact_color.get(earning["expected_impact"], "⚪")
                
                st.write(f"{color} **{earning['company']} ({earning['ticker']})**")
                st.write(f"   Sector: {earning['sector']} | {earning['time']}")
                st.write(f"   Key Metrics: {', '.join(earning['key_metrics'][:2])}")
                st.write("---")
            
            st.subheader("🚨 Breaking News Alert Categories")
            
            breaking_alerts = news_scraper.get_breaking_news_alerts()
            
            for alert in breaking_alerts:
                impact_color = {
                    "very_high": "🔴",
                    "high": "🟠",
                    "medium_high": "🟡",
                    "medium": "🟢"
                }
                color = impact_color.get(alert["impact_level"], "⚪")
                
                with st.expander(f"{color} {alert['category']}"):
                    st.write(f"**Description:** {alert['description']}")
                    st.write(f"**Typical Stocks:** {', '.join(alert['typical_stocks'])}")
                    st.write(f"**Watch For:** {', '.join(alert['watch_for'])}")
    
    # International Updates
    with tabs[4]:
        st.header("🌍 International Economic Updates")
        
        st.subheader("🏛️ Major Central Banks")
        
        central_banks = {
            "Federal Reserve (US)": {
                "rate": "Fed Funds Rate",
                "next_meeting": "Check FOMC calendar",
                "impact": "Global reserve currency - affects all markets"
            },
            "European Central Bank": {
                "rate": "Main Refinancing Rate", 
                "next_meeting": "Every 6 weeks",
                "impact": "EUR/USD major pair, European equities"
            },
            "Bank of Japan": {
                "rate": "Policy Balance Rate",
                "next_meeting": "8 times per year",
                "impact": "USD/JPY, carry trades, risk sentiment"
            },
            "Bank of England": {
                "rate": "Bank Rate",
                "next_meeting": "8 times per year", 
                "impact": "GBP/USD, UK equities, Brexit-related moves"
            }
        }
        
        for bank, info in central_banks.items():
            with st.expander(f"🏦 {bank}"):
                st.write(f"**Interest Rate:** {info['rate']}")
                st.write(f"**Meeting Schedule:** {info['next_meeting']}")
                st.write(f"**Market Impact:** {info['impact']}")
    
    # News Sources Directory
    with tabs[5]:
        st.header("📰 Essential News Sources for Traders")
        
        news_sources = get_market_moving_news_sources()
        
        # Add stock-specific news sources
        news_sources["Stock Market News"] = {
            "MarketWatch Latest": "https://www.marketwatch.com/latest-news",
            "Reuters Business": "https://www.reuters.com/business/",
            "CNBC Markets": "https://www.cnbc.com/markets/",
            "Yahoo Finance News": "https://finance.yahoo.com/news/",
            "Seeking Alpha": "https://seekingalpha.com/",
            "The Motley Fool": "https://www.fool.com/",
            "Benzinga": "https://www.benzinga.com/",
            "InvestorPlace": "https://investorplace.com/"
        }
        
        news_sources["Earnings & Analyst Coverage"] = {
            "Earnings Calendar": "https://finance.yahoo.com/calendar/earnings",
            "Analyst Recommendations": "https://www.zacks.com/",
            "FactSet Research": "https://www.factset.com/",
            "S&P Capital IQ": "https://www.spglobal.com/",
            "Morningstar Analysis": "https://www.morningstar.com/",
        }
        
        for category, sources in news_sources.items():
            st.subheader(f"📂 {category}")
            
            for source_name, url in sources.items():
                st.write(f"• [{source_name}]({url})")
        
        st.info("""
        **💡 Trading News Strategy:**
        - **Economic Data:** Follow BLS, BEA, and Fed releases for macro trends
        - **Stock News:** Monitor earnings, analyst changes, and breaking company news
        - **Corporate Events:** Track mergers, acquisitions, and leadership changes
        - **Sector Analysis:** Focus on industry-specific trends and regulatory changes
        - **International:** Watch major central bank communications
        - **Market Structure:** Track VIX, bond yields, and currency moves
        """)
    
    # Live Market Context
    with tabs[6]:
        st.header("⚡ Live Market Context & Intelligence Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Refresh Market Data", type="primary"):
                with st.spinner("Loading current market data..."):
                    market_data = get_market_data_summary()
                    
                    if market_data:
                        st.success("✅ Market data updated")
                        
                        # Display market data
                        for name, data in market_data.items():
                            change_color = "green" if data['change'] > 0 else "red"
                            st.metric(
                                label=name,
                                value=f"{data['current']:.2f}",
                                delta=f"{data['change']:+.2f}%"
                            )
            
            # Market sentiment indicators
            st.subheader("📊 Market Sentiment Guide")
            
            sentiment_guide = {
                "VIX < 20": "Low volatility, complacent market",
                "VIX 20-30": "Normal volatility, mixed sentiment", 
                "VIX > 30": "High volatility, fearful market",
                "10Y Yield Rising": "Growth expectations, inflation concerns",
                "10Y Yield Falling": "Safety demand, recession fears",
                "USD Strength": "Risk-off, Fed hawkishness",
                "USD Weakness": "Risk-on, Fed dovishness"
            }
            
            for indicator, meaning in sentiment_guide.items():
                st.write(f"**{indicator}:** {meaning}")
        
        with col2:
            # Market Intelligence Summary
            if st.button("📊 Generate Intelligence Summary"):
                with st.spinner("Analyzing market intelligence..."):
                    summary = news_scraper.get_market_impact_summary()
                    
                    if summary:
                        st.success("✅ Intelligence summary generated")
                        
                        # Create metric cards
                        col2a, col2b = st.columns(2)
                        
                        with col2a:
                            st.metric("News Articles", summary['articles_count'])
                            st.metric("Economic Events Today", summary['economic_events_today'])
                        
                        with col2b:
                            st.metric("Tickers Mentioned", summary['total_tickers_mentioned'])
                            if summary['next_fomc_meeting']:
                                days_until_fomc = summary['next_fomc_meeting']['days_until']
                                st.metric("Days to FOMC", days_until_fomc)
                        
                        # Additional earnings metrics
                        if summary.get('high_impact_earnings_this_week'):
                            col2c, col2d = st.columns(2)
                            with col2c:
                                st.metric("High-Impact Earnings", summary['high_impact_earnings_this_week'])
                            with col2d:
                                st.metric("Total Earnings This Week", summary['total_earnings_this_week'])
                        
                        # Sentiment distribution
                        st.subheader("📈 News Sentiment Analysis")
                        sentiment_data = summary['sentiment_distribution']
                        
                        if any(sentiment_data.values()):
                            fig = go.Figure(data=[go.Pie(
                                labels=list(sentiment_data.keys()),
                                values=list(sentiment_data.values()),
                                hole=0.4,
                                marker_colors=['#ff4444', '#ffbb00', '#44ff44']
                            )])
                            
                            fig.update_layout(
                                title="News Sentiment Distribution",
                                font=dict(color='white'),
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                height=300
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No news sentiment data available")
                    else:
                        st.warning("Unable to generate intelligence summary")

if __name__ == "__main__":
    main()