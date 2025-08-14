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
                "typical_release": "Last week of each quarter",
                "url": "https://www.bea.gov/data/gdp/gross-domestic-product"
            },
            "CPI": {
                "name": "Consumer Price Index", 
                "frequency": "Monthly",
                "source": "Bureau of Labor Statistics (BLS)",
                "importance": "Very High",
                "description": "Inflation measure based on consumer goods basket",
                "market_impact": "Very High - affects Fed policy decisions",
                "typical_release": "Mid-month for previous month",
                "url": "https://www.bls.gov/cpi/"
            },
            "PCE": {
                "name": "Personal Consumption Expenditures Price Index",
                "frequency": "Monthly",
                "source": "Bureau of Economic Analysis (BEA)",
                "importance": "Very High",
                "description": "Fed's preferred inflation measure",
                "market_impact": "Very High - Fed's primary inflation gauge",
                "typical_release": "End of month for prior month",
                "url": "https://www.bea.gov/data/personal-consumption-expenditures-price-index"
            },
            "NFP": {
                "name": "Non-Farm Payrolls",
                "frequency": "Monthly",
                "source": "Bureau of Labor Statistics (BLS)",
                "importance": "Very High",
                "description": "Employment change excluding agricultural workers",
                "market_impact": "Very High - labor market strength indicator",
                "typical_release": "First Friday of each month",
                "url": "https://www.bls.gov/news.release/empsit.htm"
            },
            "FOMC": {
                "name": "Federal Open Market Committee Meetings",
                "frequency": "8 times per year",
                "source": "Federal Reserve",
                "importance": "Extremely High",
                "description": "Federal Reserve monetary policy decisions",
                "market_impact": "Extremely High - sets interest rates",
                "typical_release": "FOMC meeting conclusions",
                "url": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
            },
            "JobsReport": {
                "name": "Employment Situation Report",
                "frequency": "Monthly",
                "source": "Bureau of Labor Statistics (BLS)",
                "importance": "Very High",
                "description": "Comprehensive employment data including NFP and unemployment",
                "market_impact": "Very High - complete labor market picture",
                "typical_release": "First Friday of each month at 8:30 AM ET",
                "url": "https://www.bls.gov/news.release/empsit.htm"
            },
            "PMI_Manufacturing": {
                "name": "ISM Manufacturing PMI",
                "frequency": "Monthly",
                "source": "Institute for Supply Management (ISM)",
                "importance": "High",
                "description": "Manufacturing sector health and business activity",
                "market_impact": "High - manufacturing sector indicator",
                "typical_release": "First business day of month",
                "url": "https://www.ismworld.org/supply-management-news-and-reports/reports/ism-report-on-business/"
            },
            "PMI_Services": {
                "name": "ISM Services PMI",
                "frequency": "Monthly", 
                "source": "Institute for Supply Management (ISM)",
                "importance": "High",
                "description": "Services sector health and business activity",
                "market_impact": "High - services sector indicator (70% of US economy)",
                "typical_release": "Third business day of month",
                "url": "https://www.ismworld.org/supply-management-news-and-reports/reports/ism-report-on-business/"
            },
            "RetailSales": {
                "name": "Retail Sales",
                "frequency": "Monthly",
                "source": "U.S. Census Bureau",
                "importance": "High",
                "description": "Consumer spending measure",
                "market_impact": "High - consumer demand indicator",
                "typical_release": "Mid-month for previous month",
                "url": "https://www.census.gov/retail/marts/www/marts_current.pdf"
            },
            "HousingStarts": {
                "name": "Housing Starts",
                "frequency": "Monthly",
                "source": "U.S. Census Bureau",
                "importance": "Medium-High",
                "description": "New residential construction activity",
                "market_impact": "Medium-High - economic growth indicator",
                "typical_release": "Mid-month for previous month",
                "url": "https://www.census.gov/construction/nrc/"
            },
            "UnemploymentRate": {
                "name": "Unemployment Rate",
                "frequency": "Monthly",
                "source": "Bureau of Labor Statistics (BLS)",
                "importance": "Very High",
                "description": "Percentage of unemployed in labor force",
                "market_impact": "Very High - labor market health",
                "typical_release": "First Friday of each month",
                "url": "https://www.bls.gov/news.release/empsit.htm"
            },
            "InitialClaims": {
                "name": "Initial Unemployment Claims",
                "frequency": "Weekly",
                "source": "Department of Labor",
                "importance": "High",
                "description": "Weekly unemployment insurance claims",
                "market_impact": "High - real-time labor market indicator",
                "typical_release": "Every Thursday at 8:30 AM ET",
                "url": "https://www.dol.gov/ui/data.pdf"
            },
            "ConsumerConfidence": {
                "name": "Consumer Confidence Index",
                "frequency": "Monthly",
                "source": "Conference Board",
                "importance": "Medium-High",
                "description": "Consumer sentiment about economic conditions",
                "market_impact": "Medium-High - consumer spending predictor",
                "typical_release": "Last Tuesday of each month",
                "url": "https://www.conference-board.org/data/consumerconfidence.cfm"
            },
            "ConsumerSentiment": {
                "name": "University of Michigan Consumer Sentiment",
                "frequency": "Monthly (preliminary & final)",
                "source": "University of Michigan",
                "importance": "Medium-High",
                "description": "Consumer sentiment survey",
                "market_impact": "Medium-High - consumer behavior predictor",
                "typical_release": "Mid-month (preliminary) and end-month (final)",
                "url": "http://www.sca.isr.umich.edu/"
            },
            "DurableGoods": {
                "name": "Durable Goods Orders",
                "frequency": "Monthly",
                "source": "U.S. Census Bureau",
                "importance": "Medium-High",
                "description": "Orders for long-lasting manufactured goods",
                "market_impact": "Medium-High - business investment indicator",
                "typical_release": "Fourth week of month",
                "url": "https://www.census.gov/manufacturing/m3/"
            },
            "IndustrialProduction": {
                "name": "Industrial Production Index",
                "frequency": "Monthly",
                "source": "Federal Reserve Board",
                "importance": "Medium-High",
                "description": "Manufacturing, mining, and utilities output",
                "market_impact": "Medium-High - industrial sector health",
                "typical_release": "Mid-month for previous month",
                "url": "https://www.federalreserve.gov/releases/g17/"
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
                "frequency": "FOMC meetings (8 per year)",
                "url": "https://www.federalreserve.gov/monetarypolicy/openmarket.htm"
            },
            "FOMCMinutes": {
                "name": "FOMC Meeting Minutes",
                "importance": "Very High",
                "description": "Detailed record of Federal Reserve policy discussions",
                "market_impact": "Very High - reveals Fed thinking",
                "frequency": "3 weeks after each FOMC meeting",
                "url": "https://www.federalreserve.gov/monetarypolicy/fomcminutes.htm"
            },
            "FedSpeech": {
                "name": "Federal Reserve Speeches",
                "importance": "High",
                "description": "Public speeches by Fed officials",
                "market_impact": "Medium-High - policy hints",
                "frequency": "Regular throughout year",
                "url": "https://www.federalreserve.gov/newsevents/speech/"
            },
            "BeigeBook": {
                "name": "Beige Book",
                "importance": "Medium-High",
                "description": "Regional economic conditions report",
                "market_impact": "Medium - regional economic insights",
                "frequency": "8 times per year, 2 weeks before FOMC",
                "url": "https://www.federalreserve.gov/monetarypolicy/beigebook/"
            },
            "DiscountRate": {
                "name": "Federal Reserve Discount Rate",
                "importance": "High",
                "description": "Interest rate charged to commercial banks by Federal Reserve",
                "market_impact": "High - affects bank lending rates",
                "frequency": "As needed",
                "url": "https://www.federalreserve.gov/monetarypolicy/discountrate.htm"
            },
            "ReserveRequirements": {
                "name": "Bank Reserve Requirements",
                "importance": "Medium-High", 
                "description": "Minimum reserves banks must hold",
                "market_impact": "Medium-High - affects bank lending capacity",
                "frequency": "As needed",
                "url": "https://www.federalreserve.gov/monetarypolicy/reservereq.htm"
            },
            "YieldCurve": {
                "name": "Treasury Yield Curve",
                "importance": "Very High",
                "description": "Interest rates across different Treasury maturities",
                "market_impact": "Very High - economic outlook indicator",
                "frequency": "Daily",
                "url": "https://www.treasury.gov/resource-center/data-chart-center/interest-rates/"
            },
            "EFFR": {
                "name": "Effective Federal Funds Rate",
                "importance": "Very High",
                "description": "Actual rate at which banks lend to each other overnight",
                "market_impact": "Very High - market-determined short-term rate",
                "frequency": "Daily",
                "url": "https://fred.stlouisfed.org/series/EFFR"
            },
            "SOFR": {
                "name": "Secured Overnight Financing Rate",
                "importance": "Very High",
                "description": "Replacement for LIBOR in US markets",
                "market_impact": "Very High - benchmark for derivatives and loans",
                "frequency": "Daily",
                "url": "https://www.newyorkfed.org/markets/reference-rates/sofr"
            },
            "TreasuryAuctions": {
                "name": "Treasury Debt Auctions",
                "importance": "High",
                "description": "Government bond issuance results",
                "market_impact": "High - affects benchmark interest rates",
                "frequency": "Weekly (bills), monthly/quarterly (notes/bonds)",
                "url": "https://www.treasurydirect.gov/auctions/"
            },
            "FOMC_Statement": {
                "name": "FOMC Policy Statement",
                "importance": "Extremely High",
                "description": "Official Federal Reserve policy announcement",
                "market_impact": "Extremely High - immediate market reaction",
                "frequency": "8 times per year after FOMC meetings",
                "url": "https://www.federalreserve.gov/newsevents/pressreleases/"
            },
            "FedBalanceSheet": {
                "name": "Federal Reserve Balance Sheet",
                "importance": "High",
                "description": "Assets and liabilities of the Federal Reserve",
                "market_impact": "High - QE and monetary policy tool",
                "frequency": "Weekly (H.4.1 report)",
                "url": "https://www.federalreserve.gov/releases/h41/"
            },
            "BankPrimeRate": {
                "name": "Bank Prime Lending Rate",
                "importance": "High",
                "description": "Rate banks charge their most creditworthy customers",
                "market_impact": "High - affects consumer and business loans",
                "frequency": "Changes with Fed funds rate",
                "url": "https://fred.stlouisfed.org/series/DPRIME"
            }
        }
    
    @staticmethod
    def get_banking_and_interest_rates() -> Dict[str, Dict]:
        """Get banking sector and interest rate data sources."""
        return {
            "LIBOR": {
                "name": "London Interbank Offered Rate (Legacy)",
                "importance": "Medium",
                "description": "Legacy benchmark rate (being phased out)",
                "market_impact": "Medium - historical benchmark for derivatives",
                "frequency": "Daily (legacy data)",
                "url": "https://www.theice.com/iba/libor"
            },
            "BankEarnings": {
                "name": "Major Bank Quarterly Earnings",
                "importance": "Very High",
                "description": "Financial results from major banks (JPM, BAC, C, WFC, GS, MS)",
                "market_impact": "Very High - banking sector health indicator",
                "frequency": "Quarterly earnings seasons",
                "url": "https://www.sec.gov/edgar"
            },
            "BankStressTests": {
                "name": "Federal Reserve Bank Stress Tests (CCAR)",
                "importance": "Very High",
                "description": "Annual comprehensive capital analysis and review",
                "market_impact": "Very High - bank capital adequacy assessment",
                "frequency": "Annual (June release)",
                "url": "https://www.federalreserve.gov/supervisionreg/dfa-stress-tests.htm"
            },
            "CD_Rates": {
                "name": "Certificate of Deposit Rates",
                "importance": "Medium",
                "description": "Bank deposit rates for various terms",
                "market_impact": "Medium - retail banking indicator",
                "frequency": "Weekly updates",
                "url": "https://www.fdic.gov/resources/tools/bank-find/"
            },
            "MortgageRates": {
                "name": "30-Year Fixed Mortgage Rate",
                "importance": "High",
                "description": "Primary mortgage lending rate",
                "market_impact": "High - housing market indicator",
                "frequency": "Daily/Weekly",
                "url": "https://www.freddiemac.com/primary-mortgage-market-survey"
            },
            "CorporateBondYields": {
                "name": "Corporate Bond Yields",
                "importance": "High",
                "description": "Investment grade and high yield corporate bond yields",
                "market_impact": "High - corporate borrowing costs",
                "frequency": "Daily",
                "url": "https://fred.stlouisfed.org/categories/119"
            },
            "MunicipalBondYields": {
                "name": "Municipal Bond Yields",
                "importance": "Medium-High",
                "description": "Tax-exempt municipal bond yields",
                "market_impact": "Medium-High - public finance indicator",
                "frequency": "Daily",
                "url": "https://www.municipalbonds.com/"
            },
            "CreditSpreads": {
                "name": "Credit Spreads",
                "importance": "Very High",
                "description": "Spread between corporate and Treasury bonds",
                "market_impact": "Very High - credit risk indicator",
                "frequency": "Daily",
                "url": "https://fred.stlouisfed.org/series/BAMLH0A0HYM2"
            },
            "TED_Spread": {
                "name": "TED Spread (Treasury-Eurodollar)",
                "importance": "High",
                "description": "Difference between 3-month Treasury and 3-month SOFR",
                "market_impact": "High - banking system stress indicator",
                "frequency": "Daily",
                "url": "https://fred.stlouisfed.org/series/TEDRATE"
            },
            "BankReserves": {
                "name": "Bank Reserves and Deposits",
                "importance": "Medium-High",
                "description": "Commercial bank reserves held at Federal Reserve",
                "market_impact": "Medium-High - banking system liquidity",
                "frequency": "Weekly",
                "url": "https://fred.stlouisfed.org/series/TOTRESNS"
            },
            "CommercialPaper": {
                "name": "Commercial Paper Rates",
                "importance": "Medium-High",
                "description": "Short-term unsecured corporate debt rates",
                "market_impact": "Medium-High - short-term funding costs",
                "frequency": "Daily",
                "url": "https://www.federalreserve.gov/releases/cp/"
            },
            "InterestRateSwaps": {
                "name": "Interest Rate Swap Rates",
                "importance": "High",
                "description": "Fixed-for-floating interest rate swap rates",
                "market_impact": "High - long-term interest rate expectations",
                "frequency": "Daily",
                "url": "https://www.cmegroup.com/markets/interest-rates.html"
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