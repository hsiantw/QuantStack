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
    
    def get_all_market_news(self) -> List[NewsArticle]:
        """Aggregate news from all sources"""
        all_articles = []
        
        # Scrape from different sources
        sources = [
            self.scrape_federal_reserve_news,
            self.scrape_bls_releases,
            self.scrape_treasury_data
        ]
        
        for source_func in sources:
            try:
                articles = source_func()
                for article in articles:
                    article.sentiment = self.analyze_news_sentiment(article)
                    article.tickers_mentioned = self.extract_market_tickers(article.content)
                    all_articles.append(article)
            except Exception as e:
                print(f"Error with source {source_func.__name__}: {e}")
        
        return all_articles
    
    def get_market_impact_summary(self) -> Dict:
        """Generate a summary of current market-impacting events"""
        articles = self.get_all_market_news()
        economic_events = self.get_economic_calendar_today()
        fomc_schedule = self.get_fomc_schedule()
        
        # Count articles by sentiment
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        for article in articles:
            sentiment_counts[article.sentiment] += 1
        
        # Count events by impact level
        impact_counts = {"very_high": 0, "high": 0, "medium": 0, "low": 0}
        for event in economic_events:
            impact_counts[event.impact] += 1
        
        return {
            "articles_count": len(articles),
            "sentiment_distribution": sentiment_counts,
            "economic_events_today": len(economic_events),
            "impact_distribution": impact_counts,
            "next_fomc_meeting": fomc_schedule[0] if fomc_schedule else None,
            "total_tickers_mentioned": len(set([ticker for article in articles for ticker in (article.tickers_mentioned or [])]))
        }