"""
Advanced News Scraping and Economic Data Collection Module
Provides comprehensive news analysis and economic data sourcing for trading decisions.
"""

import requests
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List, Optional
import trafilatura
import re
from dataclasses import dataclass
import yfinance as yf

@dataclass
class NewsArticle:
    title: str
    content: str
    source: str
    url: str
    timestamp: datetime
    sentiment: str = "neutral"
    market_impact: str = "medium"
    tickers_mentioned: List[str] = None

@dataclass
class EconomicEvent:
    name: str
    time: str
    impact: str
    actual: Optional[str]
    forecast: Optional[str]
    previous: Optional[str]
    currency: str
    description: str

class NewsAndDataScraper:
    """Comprehensive news and economic data scraper for trading intelligence"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.economic_sources = {
            "BLS": "https://www.bls.gov/news.release/",
            "FED": "https://www.federalreserve.gov/newsevents/",
            "BEA": "https://www.bea.gov/news/",
            "TREASURY": "https://home.treasury.gov/news/press-releases",
            "CENSUS": "https://www.census.gov/economic-indicators/"
        }
        
        self.stock_news_sources = {
            "MARKETWATCH": "https://www.marketwatch.com/latest-news",
            "REUTERS_BUSINESS": "https://www.reuters.com/business/",
            "CNBC_MARKETS": "https://www.cnbc.com/markets/",
            "BLOOMBERG_MARKETS": "https://www.bloomberg.com/markets",
            "SEC_EDGAR": "https://www.sec.gov/edgar/",
            "YAHOO_FINANCE": "https://finance.yahoo.com/news/"
        }
        
    def scrape_federal_reserve_news(self) -> List[NewsArticle]:
        """Scrape latest Federal Reserve news and communications"""
        articles = []
        
        try:
            # Fed news page
            fed_url = "https://www.federalreserve.gov/newsevents/"
            content = trafilatura.fetch_url(fed_url)
            text = trafilatura.extract(content)
            
            if text:
                articles.append(NewsArticle(
                    title="Federal Reserve Latest Updates",
                    content=text[:2000] + "..." if len(text) > 2000 else text,
                    source="Federal Reserve",
                    url=fed_url,
                    timestamp=datetime.now(),
                    market_impact="very_high"
                ))
                
        except Exception as e:
            print(f"Error scraping Fed news: {e}")
            
        return articles
    
    def scrape_bls_releases(self) -> List[NewsArticle]:
        """Scrape Bureau of Labor Statistics latest releases"""
        articles = []
        
        try:
            bls_url = "https://www.bls.gov/news.release/"
            content = trafilatura.fetch_url(bls_url)
            text = trafilatura.extract(content)
            
            if text:
                articles.append(NewsArticle(
                    title="Bureau of Labor Statistics - Latest Economic Data",
                    content=text[:2000] + "..." if len(text) > 2000 else text,
                    source="Bureau of Labor Statistics",
                    url=bls_url,
                    timestamp=datetime.now(),
                    market_impact="high"
                ))
                
        except Exception as e:
            print(f"Error scraping BLS releases: {e}")
            
        return articles
    
    def scrape_treasury_data(self) -> List[NewsArticle]:
        """Scrape US Treasury Department press releases"""
        articles = []
        
        try:
            treasury_url = "https://home.treasury.gov/news/press-releases"
            content = trafilatura.fetch_url(treasury_url)
            text = trafilatura.extract(content)
            
            if text:
                articles.append(NewsArticle(
                    title="US Treasury Department - Press Releases",
                    content=text[:2000] + "..." if len(text) > 2000 else text,
                    source="US Treasury",
                    url=treasury_url,
                    timestamp=datetime.now(),
                    market_impact="high"
                ))
                
        except Exception as e:
            print(f"Error scraping Treasury data: {e}")
            
        return articles
    
    def get_economic_calendar_today(self) -> List[EconomicEvent]:
        """Get today's economic calendar events"""
        today = datetime.now()
        weekday = today.weekday()  # 0=Monday, 6=Sunday
        
        # Define weekly economic events
        weekly_events = {
            0: [  # Monday
                EconomicEvent("ISM Manufacturing PMI", "10:00 AM ET", "high", None, None, None, "USD",
                             "Manufacturing sector health indicator")
            ],
            2: [  # Wednesday
                EconomicEvent("ADP Employment Report", "8:15 AM ET", "high", None, None, None, "USD",
                             "Private sector employment change"),
                EconomicEvent("Crude Oil Inventories", "10:30 AM ET", "medium", None, None, None, "USD",
                             "Weekly petroleum inventory report")
            ],
            3: [  # Thursday
                EconomicEvent("Initial Jobless Claims", "8:30 AM ET", "high", None, None, None, "USD",
                             "Weekly unemployment insurance claims"),
                EconomicEvent("ISM Services PMI", "10:00 AM ET", "high", None, None, None, "USD",
                             "Services sector business activity")
            ],
            4: [  # Friday
                EconomicEvent("Non-Farm Payrolls", "8:30 AM ET", "very_high", None, None, None, "USD",
                             "Monthly employment change (First Friday)"),
                EconomicEvent("Unemployment Rate", "8:30 AM ET", "very_high", None, None, None, "USD",
                             "Labor force unemployment percentage")
            ]
        }
        
        return weekly_events.get(weekday, [])
    
    def get_fomc_schedule(self) -> List[Dict]:
        """Get upcoming FOMC meeting dates and information"""
        # 2024-2025 FOMC meeting dates
        fomc_dates = [
            {"date": "2024-12-17", "type": "FOMC Meeting", "press_conference": True},
            {"date": "2025-01-28", "type": "FOMC Meeting", "press_conference": False},
            {"date": "2025-03-18", "type": "FOMC Meeting", "press_conference": True},
            {"date": "2025-04-29", "type": "FOMC Meeting", "press_conference": False},
            {"date": "2025-06-10", "type": "FOMC Meeting", "press_conference": True},
            {"date": "2025-07-29", "type": "FOMC Meeting", "press_conference": False},
            {"date": "2025-09-16", "type": "FOMC Meeting", "press_conference": True},
            {"date": "2025-11-04", "type": "FOMC Meeting", "press_conference": False},
            {"date": "2025-12-16", "type": "FOMC Meeting", "press_conference": True}
        ]
        
        # Filter for upcoming meetings
        today = datetime.now().date()
        upcoming = []
        
        for meeting in fomc_dates:
            meeting_date = datetime.strptime(meeting["date"], "%Y-%m-%d").date()
            if meeting_date >= today:
                days_until = (meeting_date - today).days
                meeting["days_until"] = days_until
                upcoming.append(meeting)
        
        return upcoming[:3]  # Next 3 meetings
    
    def extract_market_tickers(self, text: str) -> List[str]:
        """Extract potential stock tickers from text content"""
        # Common ticker patterns
        ticker_patterns = [
            r'\b[A-Z]{1,5}\b',  # 1-5 uppercase letters
            r'\$[A-Z]{1,5}\b',  # With dollar sign
        ]
        
        tickers = set()
        for pattern in ticker_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                match = match.replace('$', '')
                # Filter out common false positives
                if (len(match) >= 2 and 
                    match not in ['THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'HAD', 'BY', 'HOT', 'ITS', 'NEW', 'NOW', 'OLD', 'SEE', 'GET', 'MAY', 'WAY', 'DAY', 'USE', 'HER', 'MAN', 'OUT', 'TOP', 'BOY', 'DID', 'AGE', 'SHE', 'LET', 'PUT', 'END', 'WHY', 'TRY', 'GOD', 'SIX', 'DOG', 'EAT', 'AGO', 'SIT', 'FUN', 'BAD', 'YES', 'YET', 'ARM', 'FAR', 'OFF', 'BAG', 'BAT', 'BOX', 'BOY', 'BUS', 'CAR', 'CAT', 'COW', 'CUT', 'DOG', 'EAR', 'EGG', 'END', 'EYE', 'FAN', 'FEW', 'FLY', 'FOR', 'FUN', 'GUN', 'HAD', 'HAT', 'HER', 'HIM', 'HIS', 'HOT', 'HOW', 'ITS', 'JOB', 'LAW', 'LEG', 'LET', 'LIE', 'LOT', 'LOW', 'MAN', 'MAP', 'MAY', 'MOM', 'NEW', 'NOT', 'NOW', 'NUT', 'OFF', 'OLD', 'ONE', 'OUR', 'OUT', 'OWN', 'PEN', 'PET', 'PUT', 'RAN', 'RED', 'RUN', 'SAD', 'SAT', 'SAW', 'SAY', 'SEA', 'SEE', 'SHE', 'SIT', 'SIX', 'SUN', 'TEA', 'THE', 'TOP', 'TOY', 'TRY', 'TWO', 'USE', 'VAN', 'WAR', 'WAY', 'WHO', 'WHY', 'WIN', 'WON', 'YES', 'YET', 'YOU', 'ZOO']):
                    tickers.add(match)
        
        return list(tickers)
    
    def analyze_news_sentiment(self, article: NewsArticle) -> str:
        """Basic sentiment analysis of news content"""
        positive_words = ['gain', 'rise', 'up', 'increase', 'bull', 'growth', 'strong', 'beat', 'exceed', 'positive', 'rally', 'surge', 'jump', 'soar', 'climb']
        negative_words = ['fall', 'drop', 'down', 'decrease', 'bear', 'decline', 'weak', 'miss', 'below', 'negative', 'crash', 'plunge', 'dive', 'tumble', 'sink']
        
        content_lower = article.content.lower()
        
        positive_count = sum(1 for word in positive_words if word in content_lower)
        negative_count = sum(1 for word in negative_words if word in content_lower)
        
        if positive_count > negative_count * 1.2:
            return "positive"
        elif negative_count > positive_count * 1.2:
            return "negative"
        else:
            return "neutral"
    
    def scrape_marketwatch_headlines(self) -> List[NewsArticle]:
        """Scrape high-impact stock news from MarketWatch"""
        articles = []
        
        try:
            marketwatch_url = "https://www.marketwatch.com/latest-news"
            content = trafilatura.fetch_url(marketwatch_url)
            text = trafilatura.extract(content)
            
            if text:
                # Split content into potential headlines
                lines = text.split('\n')
                headlines = []
                
                for line in lines:
                    line = line.strip()
                    if (len(line) > 20 and len(line) < 200 and 
                        any(keyword in line.lower() for keyword in 
                            ['stock', 'shares', 'earnings', 'dividend', 'merger', 'acquisition', 
                             'ceo', 'revenue', 'profit', 'loss', 'beat', 'miss', 'forecast',
                             'guidance', 'outlook', 'upgrade', 'downgrade', 'price target'])):
                        headlines.append(line)
                
                # Take top headlines
                for i, headline in enumerate(headlines[:10]):
                    articles.append(NewsArticle(
                        title=f"MarketWatch Headline {i+1}",
                        content=headline,
                        source="MarketWatch",
                        url=marketwatch_url,
                        timestamp=datetime.now(),
                        market_impact="high"
                    ))
                    
        except Exception as e:
            print(f"Error scraping MarketWatch: {e}")
            
        return articles
    
    def scrape_reuters_business_news(self) -> List[NewsArticle]:
        """Scrape business and market news from Reuters"""
        articles = []
        
        try:
            reuters_url = "https://www.reuters.com/business/"
            content = trafilatura.fetch_url(reuters_url)
            text = trafilatura.extract(content)
            
            if text:
                articles.append(NewsArticle(
                    title="Reuters Business & Markets Update",
                    content=text[:2500] + "..." if len(text) > 2500 else text,
                    source="Reuters Business",
                    url=reuters_url,
                    timestamp=datetime.now(),
                    market_impact="high"
                ))
                
        except Exception as e:
            print(f"Error scraping Reuters business: {e}")
            
        return articles
    
    def scrape_cnbc_market_news(self) -> List[NewsArticle]:
        """Scrape market-moving news from CNBC"""
        articles = []
        
        try:
            cnbc_url = "https://www.cnbc.com/markets/"
            content = trafilatura.fetch_url(cnbc_url)
            text = trafilatura.extract(content)
            
            if text:
                articles.append(NewsArticle(
                    title="CNBC Markets Breaking News",
                    content=text[:2500] + "..." if len(text) > 2500 else text,
                    source="CNBC Markets",
                    url=cnbc_url,
                    timestamp=datetime.now(),
                    market_impact="high"
                ))
                
        except Exception as e:
            print(f"Error scraping CNBC markets: {e}")
            
        return articles
    
    def get_high_impact_stock_headlines(self) -> List[Dict]:
        """Generate high-impact stock headlines and market movers"""
        high_impact_events = [
            {
                "headline": "Major Earnings Releases This Week",
                "description": "Key companies reporting quarterly results with potential market impact",
                "impact": "very_high",
                "category": "earnings",
                "timing": "Throughout week"
            },
            {
                "headline": "Federal Reserve Policy Decision Impact on Financials",
                "description": "Banking sector positioning ahead of interest rate decisions",
                "impact": "very_high",
                "category": "monetary_policy",
                "timing": "FOMC meeting days"
            },
            {
                "headline": "Technology Sector Momentum Analysis",
                "description": "AI, semiconductor, and cloud computing stock movements",
                "impact": "high",
                "category": "sector_rotation",
                "timing": "Ongoing"
            },
            {
                "headline": "Energy Sector Volatility on Oil Price Changes",
                "description": "Crude oil inventory reports driving energy stock movements",
                "impact": "high",
                "category": "commodities",
                "timing": "Wednesday EIA reports"
            },
            {
                "headline": "Healthcare Merger & Acquisition Activity",
                "description": "Biotech and pharmaceutical company consolidation trends",
                "impact": "medium_high",
                "category": "mergers",
                "timing": "Ongoing"
            },
            {
                "headline": "Consumer Discretionary Earnings Season Impact",
                "description": "Retail and consumer spending patterns reflected in quarterly results",
                "impact": "high",
                "category": "earnings",
                "timing": "Earnings season"
            },
            {
                "headline": "ESG Investment Flows Affecting Stock Valuations",
                "description": "Environmental and social governance criteria driving institutional flows",
                "impact": "medium",
                "category": "flows",
                "timing": "Ongoing trend"
            },
            {
                "headline": "Dividend Aristocrat Announcements and Ex-Dividend Dates",
                "description": "S&P 500 dividend-paying stocks with consecutive increase streaks",
                "impact": "medium",
                "category": "dividends",
                "timing": "Quarterly cycles"
            }
        ]
        
        return high_impact_events
    
    def get_sector_specific_news(self) -> Dict[str, List[str]]:
        """Get sector-specific high-impact news categories"""
        sector_news = {
            "Technology": [
                "AI and Machine Learning breakthroughs affecting valuations",
                "Semiconductor supply chain and chip shortage updates",
                "Cloud computing market share battles and earnings",
                "Cybersecurity threats and software security spending",
                "Social media regulation and platform policy changes"
            ],
            "Healthcare": [
                "FDA drug approvals and clinical trial results",
                "Healthcare policy and Medicare/Medicaid changes",
                "Biotech merger and acquisition activity",
                "Medical device innovation and regulatory approvals",
                "Pharmaceutical patent expirations and generic competition"
            ],
            "Financial": [
                "Federal Reserve interest rate policy impacts",
                "Banking regulation and stress test results",
                "Credit default rates and loan portfolio quality",
                "Fintech disruption and digital banking adoption",
                "Insurance sector natural disaster exposure"
            ],
            "Energy": [
                "Oil price volatility and OPEC production decisions",
                "Renewable energy policy and subsidies",
                "Natural gas pipeline and storage capacity",
                "Electric vehicle adoption affecting oil demand",
                "Carbon credit trading and ESG investment flows"
            ],
            "Consumer": [
                "Retail sales and consumer confidence data",
                "Supply chain disruption and inventory levels",
                "E-commerce growth and brick-and-mortar closures",
                "Consumer price inflation affecting spending patterns",
                "Brand loyalty shifts and demographic changes"
            ],
            "Industrial": [
                "Manufacturing PMI and industrial production data",
                "Infrastructure spending and government contracts",
                "Transportation and logistics capacity constraints",
                "Automation and robotics adoption in manufacturing",
                "Global trade tensions affecting export-dependent companies"
            ],
            "Commodities": [
                "OPEC production decisions affecting oil prices",
                "Weather patterns impacting agricultural commodities",
                "US Dollar strength affecting commodity prices",
                "Central bank gold purchases and policy decisions",
                "China demand for industrial metals and energy"
            ],
            "Forex": [
                "Central bank interest rate decisions and policy divergence",
                "Economic data releases (GDP, inflation, employment)",
                "Geopolitical tensions affecting safe-haven currencies",
                "Trade balance and current account data",
                "Risk-on/risk-off sentiment driving currency flows"
            ]
        }
        
        return sector_news
    
    def get_all_market_news(self) -> List[NewsArticle]:
        """Aggregate news from all sources including stock-specific news"""
        all_articles = []
        
        # Economic and Fed sources
        economic_sources = [
            self.scrape_federal_reserve_news,
            self.scrape_bls_releases,
            self.scrape_treasury_data
        ]
        
        # Stock market news sources
        stock_sources = [
            self.scrape_marketwatch_headlines,
            self.scrape_reuters_business_news,
            self.scrape_cnbc_market_news
        ]
        
        # Combine all sources
        all_sources = economic_sources + stock_sources
        
        for source_func in all_sources:
            try:
                articles = source_func()
                for article in articles:
                    article.sentiment = self.analyze_news_sentiment(article)
                    article.tickers_mentioned = self.extract_market_tickers(article.content)
                    all_articles.append(article)
            except Exception as e:
                print(f"Error with source {source_func.__name__}: {e}")
        
        return all_articles
    
    def get_earnings_calendar_this_week(self) -> List[Dict]:
        """Get major earnings releases for current week"""
        # Major companies that typically have high market impact
        major_earnings = [
            {
                "company": "Apple Inc.",
                "ticker": "AAPL",
                "date": "This Week",
                "time": "After Market Close",
                "sector": "Technology",
                "market_cap": "Large Cap",
                "expected_impact": "very_high",
                "key_metrics": ["iPhone sales", "Services revenue", "China performance"]
            },
            {
                "company": "Microsoft Corporation",
                "ticker": "MSFT", 
                "date": "This Week",
                "time": "After Market Close",
                "sector": "Technology",
                "market_cap": "Large Cap",
                "expected_impact": "very_high",
                "key_metrics": ["Azure growth", "Office 365 adoption", "AI integration"]
            },
            {
                "company": "Amazon.com Inc.",
                "ticker": "AMZN",
                "date": "This Week",
                "time": "After Market Close", 
                "sector": "Consumer Discretionary",
                "market_cap": "Large Cap",
                "expected_impact": "very_high",
                "key_metrics": ["AWS revenue", "Prime membership", "Retail margins"]
            },
            {
                "company": "NVIDIA Corporation",
                "ticker": "NVDA",
                "date": "This Week",
                "time": "After Market Close",
                "sector": "Technology",
                "market_cap": "Large Cap", 
                "expected_impact": "very_high",
                "key_metrics": ["Data center revenue", "AI chip demand", "Gaming segment"]
            },
            {
                "company": "Tesla Inc.",
                "ticker": "TSLA",
                "date": "This Week",
                "time": "After Market Close",
                "sector": "Consumer Discretionary",
                "market_cap": "Large Cap",
                "expected_impact": "high",
                "key_metrics": ["Vehicle deliveries", "Energy business", "Full self-driving"]
            }
        ]
        
        return major_earnings
    
    def get_breaking_news_alerts(self) -> List[Dict]:
        """Generate breaking news alert categories for stock market"""
        alerts = [
            {
                "category": "Earnings Surprises",
                "description": "Companies beating or missing earnings estimates by significant margins",
                "impact_level": "very_high",
                "typical_stocks": ["Large cap technology", "Financial institutions", "Healthcare leaders"],
                "watch_for": ["Revenue beats >5%", "EPS surprises >10%", "Guidance revisions"]
            },
            {
                "category": "Merger & Acquisition News", 
                "description": "Deal announcements, takeover bids, and strategic partnerships",
                "impact_level": "very_high",
                "typical_stocks": ["Mid-cap targets", "Industry consolidation plays"],
                "watch_for": ["Premium offers >20%", "Strategic buyer interest", "Regulatory approvals"]
            },
            {
                "category": "FDA Drug Approvals",
                "description": "Pharmaceutical and biotech regulatory decisions",
                "impact_level": "high",
                "typical_stocks": ["Biotech companies", "Big pharma", "Medical devices"],
                "watch_for": ["Phase 3 trial results", "FDA panel meetings", "Breakthrough designations"]
            },
            {
                "category": "Analyst Rating Changes",
                "description": "Major Wall Street upgrades, downgrades, and price target changes",
                "impact_level": "medium_high",
                "typical_stocks": ["Coverage universe stocks", "Sector rotation plays"],
                "watch_for": ["Initiations with Buy", "Target increases >15%", "Sector overweights"]
            },
            {
                "category": "Management Changes",
                "description": "CEO appointments, board changes, and executive departures",
                "impact_level": "medium_high", 
                "typical_stocks": ["Turnaround situations", "Growth companies", "Distressed names"],
                "watch_for": ["Activist involvement", "Succession planning", "Strategic pivots"]
            }
        ]
        
        return alerts
    
    def get_market_impact_summary(self) -> Dict:
        """Generate a summary of current market-impacting events"""
        articles = self.get_all_market_news()
        economic_events = self.get_economic_calendar_today()
        fomc_schedule = self.get_fomc_schedule()
        earnings_this_week = self.get_earnings_calendar_this_week()
        
        # Count articles by sentiment
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        for article in articles:
            sentiment_counts[article.sentiment] += 1
        
        # Count events by impact level
        impact_counts = {"very_high": 0, "high": 0, "medium": 0, "low": 0}
        for event in economic_events:
            impact_counts[event.impact] += 1
        
        # Count high-impact earnings
        high_impact_earnings = len([e for e in earnings_this_week if e["expected_impact"] == "very_high"])
        
        return {
            "articles_count": len(articles),
            "sentiment_distribution": sentiment_counts,
            "economic_events_today": len(economic_events),
            "impact_distribution": impact_counts,
            "next_fomc_meeting": fomc_schedule[0] if fomc_schedule else None,
            "total_tickers_mentioned": len(set([ticker for article in articles for ticker in (article.tickers_mentioned or [])])),
            "high_impact_earnings_this_week": high_impact_earnings,
            "total_earnings_this_week": len(earnings_this_week)
        }