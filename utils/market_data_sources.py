"""
Market Information Sources Module
Comprehensive guide to critical data sources that affect financial markets.
"""

import pandas as pd
from typing import Dict, List

class MarketDataSources:
    """
    Comprehensive information about market-moving data sources and their importance.
    """
    
    @staticmethod
    def get_sec_filing_types() -> Dict[str, Dict]:
        """Get detailed information about SEC filing types and their market impact."""
        return {
            "10-K": {
                "name": "Annual Report",
                "frequency": "Annual",
                "importance": "Very High",
                "description": "Comprehensive overview of company's business, finances, and risks",
                "key_sections": [
                    "Business Overview",
                    "Risk Factors", 
                    "Financial Statements",
                    "Management Discussion & Analysis (MD&A)"
                ],
                "market_impact": "High - provides complete financial picture",
                "timing": "Within 60-90 days after fiscal year end"
            },
            "10-Q": {
                "name": "Quarterly Report", 
                "frequency": "Quarterly",
                "importance": "High",
                "description": "Quarterly financial statements and updates",
                "key_sections": [
                    "Condensed Financial Statements",
                    "Management Discussion & Analysis",
                    "Legal Proceedings"
                ],
                "market_impact": "Medium-High - quarterly performance updates",
                "timing": "Within 40 days after quarter end"
            },
            "8-K": {
                "name": "Current Report",
                "frequency": "As needed",
                "importance": "Very High",
                "description": "Material events and corporate changes",
                "key_sections": [
                    "Material Agreements",
                    "Acquisitions/Dispositions",
                    "Executive Changes",
                    "Bankruptcy/Receivership"
                ],
                "market_impact": "Very High - immediate material events",
                "timing": "Within 4 business days of triggering event"
            },
            "DEF 14A": {
                "name": "Proxy Statement",
                "frequency": "Annual (typically)",
                "importance": "Medium",
                "description": "Information for shareholder voting and executive compensation",
                "key_sections": [
                    "Executive Compensation",
                    "Board of Directors",
                    "Shareholder Proposals",
                    "Corporate Governance"
                ],
                "market_impact": "Medium - governance and compensation insights",
                "timing": "Before annual shareholder meeting"
            },
            "13F": {
                "name": "Institutional Holdings",
                "frequency": "Quarterly",
                "importance": "Medium-High",
                "description": "Holdings of institutional investment managers ($100M+ assets)",
                "key_sections": [
                    "Portfolio Holdings",
                    "Position Sizes",
                    "New Positions",
                    "Position Changes"
                ],
                "market_impact": "High - reveals institutional sentiment",
                "timing": "45 days after quarter end"
            },
            "Form 4": {
                "name": "Insider Trading",
                "frequency": "As needed",
                "importance": "Medium-High",
                "description": "Insider buying/selling activity",
                "key_sections": [
                    "Transaction Details",
                    "Ownership Changes",
                    "Exercise of Options"
                ],
                "market_impact": "Medium-High - insider sentiment",
                "timing": "Within 2 days of transaction"
            },
            "S-1": {
                "name": "IPO Registration",
                "frequency": "As needed",
                "importance": "High",
                "description": "Initial public offering registration statement",
                "key_sections": [
                    "Business Description",
                    "Risk Factors",
                    "Use of Proceeds",
                    "Financial Statements"
                ],
                "market_impact": "High - new market entrants",
                "timing": "Before IPO"
            }
        }
    
    @staticmethod
    def get_economic_indicators() -> Dict[str, Dict]:
        """Get critical economic indicators that affect markets."""
        return {
            "GDP": {
                "name": "Gross Domestic Product",
                "frequency": "Quarterly",
                "source": "Bureau of Economic Analysis (BEA)",
                "importance": "Very High",
                "description": "Total economic output measure",
                "market_impact": "Very High - overall economic health indicator",
                "typical_release": "Last week of each quarter"
            },
            "CPI": {
                "name": "Consumer Price Index", 
                "frequency": "Monthly",
                "source": "Bureau of Labor Statistics (BLS)",
                "importance": "Very High",
                "description": "Inflation measure based on consumer goods basket",
                "market_impact": "Very High - affects Fed policy decisions",
                "typical_release": "Mid-month for previous month"
            },
            "NFP": {
                "name": "Non-Farm Payrolls",
                "frequency": "Monthly",
                "source": "Bureau of Labor Statistics (BLS)",
                "importance": "Very High",
                "description": "Employment change excluding agricultural workers",
                "market_impact": "Very High - labor market strength indicator",
                "typical_release": "First Friday of each month"
            },
            "FOMC": {
                "name": "Federal Open Market Committee Meetings",
                "frequency": "8 times per year",
                "source": "Federal Reserve",
                "importance": "Extremely High",
                "description": "Federal Reserve monetary policy decisions",
                "market_impact": "Extremely High - sets interest rates",
                "typical_release": "FOMC meeting conclusions"
            },
            "PMI": {
                "name": "Purchasing Managers' Index",
                "frequency": "Monthly",
                "source": "Institute for Supply Management (ISM)",
                "importance": "High",
                "description": "Manufacturing and services sector health",
                "market_impact": "High - business activity indicator",
                "typical_release": "First business day of month"
            },
            "RetailSales": {
                "name": "Retail Sales",
                "frequency": "Monthly",
                "source": "U.S. Census Bureau",
                "importance": "High",
                "description": "Consumer spending measure",
                "market_impact": "High - consumer demand indicator",
                "typical_release": "Mid-month for previous month"
            },
            "HousingStarts": {
                "name": "Housing Starts",
                "frequency": "Monthly",
                "source": "U.S. Census Bureau",
                "importance": "Medium-High",
                "description": "New residential construction activity",
                "market_impact": "Medium-High - economic growth indicator",
                "typical_release": "Mid-month for previous month"
            },
            "UnemploymentRate": {
                "name": "Unemployment Rate",
                "frequency": "Monthly",
                "source": "Bureau of Labor Statistics (BLS)",
                "importance": "Very High",
                "description": "Percentage of unemployed in labor force",
                "market_impact": "Very High - labor market health",
                "typical_release": "First Friday of each month"
            }
        }
    
    @staticmethod
    def get_federal_reserve_data() -> Dict[str, Dict]:
        """Get Federal Reserve data sources and their market impact."""
        return {
            "FedFundsRate": {
                "name": "Federal Funds Rate",
                "importance": "Extremely High",
                "description": "Target interest rate set by Federal Reserve",
                "market_impact": "Extremely High - affects all asset classes",
                "frequency": "FOMC meetings (8 per year)"
            },
            "FOMCMinutes": {
                "name": "FOMC Meeting Minutes",
                "importance": "Very High",
                "description": "Detailed record of Federal Reserve policy discussions",
                "market_impact": "Very High - reveals Fed thinking",
                "frequency": "3 weeks after each FOMC meeting"
            },
            "FedSpeech": {
                "name": "Federal Reserve Speeches",
                "importance": "High",
                "description": "Public speeches by Fed officials",
                "market_impact": "Medium-High - policy hints",
                "frequency": "Regular throughout year"
            },
            "BeigeBoo": {
                "name": "Beige Book",
                "importance": "Medium-High",
                "description": "Regional economic conditions report",
                "market_impact": "Medium - regional economic insights",
                "frequency": "8 times per year, 2 weeks before FOMC"
            }
        }
    
    @staticmethod
    def get_earnings_data() -> Dict[str, Dict]:
        """Get earnings-related data sources."""
        return {
            "EarningsReports": {
                "name": "Quarterly Earnings Reports",
                "importance": "Very High",
                "description": "Company financial performance vs expectations",
                "market_impact": "Very High - individual stock price driver",
                "timing": "Within 45 days of quarter end"
            },
            "EarningsGuidance": {
                "name": "Forward Guidance",
                "importance": "Very High",
                "description": "Management outlook for future performance",
                "market_impact": "Very High - future expectations",
                "timing": "During earnings calls"
            },
            "AnalystRevisions": {
                "name": "Analyst Estimate Revisions",
                "importance": "High",
                "description": "Changes to earnings/revenue estimates",
                "market_impact": "High - sentiment indicator",
                "timing": "Ongoing"
            },
            "PreAnnouncements": {
                "name": "Earnings Pre-announcements",
                "importance": "Very High",
                "description": "Early warnings of earnings misses/beats",
                "market_impact": "Very High - immediate price impact",
                "timing": "During quiet period before earnings"
            }
        }
    
    @staticmethod
    def get_international_data() -> Dict[str, Dict]:
        """Get international data sources affecting U.S. markets."""
        return {
            "ECBPolicy": {
                "name": "European Central Bank Policy",
                "importance": "High",
                "description": "Eurozone monetary policy decisions",
                "market_impact": "High - affects USD and global markets",
                "source": "European Central Bank"
            },
            "ChinaGDP": {
                "name": "China Economic Data",
                "importance": "High",
                "description": "Chinese economic indicators (GDP, PMI, etc.)",
                "market_impact": "High - global growth implications",
                "source": "National Bureau of Statistics of China"
            },
            "OilPrices": {
                "name": "Crude Oil Prices",
                "importance": "High",
                "description": "Global oil price movements",
                "market_impact": "High - affects energy sector and inflation",
                "source": "Various (WTI, Brent)"
            },
            "USDIndex": {
                "name": "U.S. Dollar Index",
                "importance": "High",
                "description": "Dollar strength vs major currencies",
                "market_impact": "High - affects multinational companies",
                "source": "ICE Futures"
            }
        }
    
    @staticmethod
    def get_sentiment_indicators() -> Dict[str, Dict]:
        """Get market sentiment and positioning indicators."""
        return {
            "VIX": {
                "name": "Volatility Index",
                "importance": "Very High",
                "description": "Market fear/complacency gauge",
                "market_impact": "Very High - risk sentiment indicator",
                "source": "CBOE"
            },
            "PutCallRatio": {
                "name": "Put/Call Ratio",
                "importance": "Medium-High",
                "description": "Options positioning indicator",
                "market_impact": "Medium-High - sentiment gauge",
                "source": "CBOE"
            },
            "COTReports": {
                "name": "Commitment of Traders",
                "importance": "Medium",
                "description": "Large trader positioning in futures",
                "market_impact": "Medium - positioning insights",
                "source": "CFTC"
            },
            "AAPoll": {
                "name": "AAII Investor Sentiment",
                "importance": "Medium",
                "description": "Individual investor sentiment survey",
                "market_impact": "Medium - retail sentiment",
                "source": "American Association of Individual Investors"
            }
        }
    
    @staticmethod
    def get_geopolitical_events() -> List[str]:
        """Get types of geopolitical events that affect markets."""
        return [
            "Elections (Presidential, Congressional)",
            "Wars and Military Conflicts",
            "Trade Wars and Tariff Announcements",
            "Sanctions and Trade Restrictions",
            "Central Bank Interventions",
            "Currency Crises",
            "Natural Disasters",
            "Terrorist Attacks",
            "Brexit-type Events",
            "Regulatory Changes",
            "Credit Rating Changes (Sovereign)",
            "Energy Supply Disruptions"
        ]
    
    @staticmethod
    def get_corporate_events() -> Dict[str, Dict]:
        """Get corporate events that affect individual stocks."""
        return {
            "Mergers_Acquisitions": {
                "name": "Mergers & Acquisitions",
                "impact": "Very High",
                "description": "Company buyouts and mergers",
                "typical_effect": "Target stock jumps, acquirer may decline"
            },
            "Spin_offs": {
                "name": "Spin-offs",
                "impact": "High",
                "description": "Separation of business units",
                "typical_effect": "Complex valuation impacts"
            },
            "Stock_Splits": {
                "name": "Stock Splits",
                "impact": "Low",
                "description": "Increase in share count, lower price per share",
                "typical_effect": "Cosmetic change, often positive sentiment"
            },
            "Dividends": {
                "name": "Dividend Announcements",
                "impact": "Medium",
                "description": "Dividend increases, cuts, or initiations",
                "typical_effect": "Increases often boost price, cuts hurt"
            },
            "Share_Buybacks": {
                "name": "Share Buyback Programs",
                "impact": "Medium-High",
                "description": "Company repurchasing own shares",
                "typical_effect": "Generally positive for stock price"
            },
            "Management_Changes": {
                "name": "Executive Changes",
                "impact": "Medium-High",
                "description": "CEO, CFO, or key executive changes",
                "typical_effect": "Varies by circumstances and replacement"
            }
        }
    
    @staticmethod
    def get_data_calendar() -> Dict[str, List[str]]:
        """Get typical economic calendar by day of week."""
        return {
            "Monday": [
                "Consumer Credit (monthly)",
                "Factory Orders (monthly)"
            ],
            "Tuesday": [
                "Job Openings (JOLTS)",
                "Consumer Confidence",
                "Case-Shiller Home Prices"
            ],
            "Wednesday": [
                "ADP Employment Report",
                "ISM Services PMI",
                "Crude Oil Inventories",
                "FOMC Announcements (scheduled)"
            ],
            "Thursday": [
                "Initial Jobless Claims (weekly)",
                "Philadelphia Fed Index",
                "Leading Economic Indicators"
            ],
            "Friday": [
                "Non-Farm Payrolls (first Friday)",
                "Unemployment Rate (first Friday)",
                "Consumer Price Index",
                "Producer Price Index",
                "Retail Sales",
                "Industrial Production"
            ]
        }
    
    @staticmethod
    def get_key_websites() -> Dict[str, Dict]:
        """Get key websites for market-moving information."""
        return {
            "SEC_EDGAR": {
                "name": "SEC EDGAR Database",
                "url": "https://www.sec.gov/edgar",
                "description": "Official SEC filings database",
                "importance": "Very High"
            },
            "Federal_Reserve": {
                "name": "Federal Reserve Economic Data (FRED)",
                "url": "https://fred.stlouisfed.org",
                "description": "Comprehensive economic data",
                "importance": "Very High"
            },
            "BLS": {
                "name": "Bureau of Labor Statistics",
                "url": "https://www.bls.gov",
                "description": "Employment and inflation data",
                "importance": "Very High"
            },
            "BEA": {
                "name": "Bureau of Economic Analysis",
                "url": "https://www.bea.gov",
                "description": "GDP and economic accounts",
                "importance": "Very High"
            },
            "Treasury": {
                "name": "U.S. Treasury",
                "url": "https://www.treasury.gov",
                "description": "Government debt and fiscal policy",
                "importance": "High"
            },
            "CFTC": {
                "name": "Commodity Futures Trading Commission",
                "url": "https://www.cftc.gov",
                "description": "Futures and options market data",
                "importance": "Medium-High"
            }
        }
    
    @staticmethod
    def get_priority_timeline() -> Dict[str, List[str]]:
        """Get information release priority timeline."""
        return {
            "Immediate_Impact": [
                "FOMC Rate Decisions",
                "8-K Filings (Material Events)",
                "Earnings Pre-announcements",
                "Major Geopolitical Events",
                "Central Bank Interventions"
            ],
            "Same_Day_Impact": [
                "Major Economic Indicators (GDP, CPI, NFP)",
                "Earnings Reports",
                "Fed Official Speeches",
                "Emergency Company Announcements"
            ],
            "Next_Day_Impact": [
                "10-Q/10-K Filings",
                "Analyst Upgrades/Downgrades",
                "Corporate Guidance Changes",
                "International Economic Data"
            ],
            "Weekly_Impact": [
                "Weekly Economic Indicators",
                "Oil Inventory Data",
                "Money Supply Data",
                "Initial Jobless Claims"
            ],
            "Monthly_Impact": [
                "13F Filings",
                "Monthly Economic Reports",
                "Sector Rotation Data",
                "Institutional Flow Data"
            ]
        }
    
    @staticmethod
    def get_impact_matrix() -> pd.DataFrame:
        """Get a matrix showing impact levels of different data sources on asset classes."""
        data = {
            "Data Source": [
                "Federal Funds Rate", "Non-Farm Payrolls", "CPI", "GDP", "Earnings Reports",
                "8-K Filings", "Oil Prices", "VIX", "China GDP", "EUR/USD"
            ],
            "Stocks": [
                "Very High", "Very High", "Very High", "High", "Very High",
                "High", "Medium", "Very High", "Medium", "Medium"
            ],
            "Bonds": [
                "Extremely High", "Very High", "Extremely High", "High", "Low",
                "Low", "Medium", "High", "Medium", "High"
            ],
            "Currencies": [
                "Extremely High", "High", "Very High", "High", "Low",
                "Low", "High", "Medium", "High", "Very High"
            ],
            "Commodities": [
                "High", "Medium", "High", "Medium", "Low",
                "Low", "Very High", "Medium", "High", "High"
            ]
        }
        
        return pd.DataFrame(data)
<truncated - file too long>