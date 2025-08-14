import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.market_data_sources import MarketDataSources
from utils.tooltips import get_tooltip_help

# Page configuration
st.set_page_config(
    page_title="Market Information Sources",
    page_icon="📰",
    layout="wide"
)

st.title("📰 Critical Market Information Sources")
st.markdown("Comprehensive guide to the most important data sources that affect financial markets")

# Navigation sidebar
st.sidebar.header("Information Categories")
category = st.sidebar.selectbox(
    "Select Information Category",
    [
        "SEC Filings Overview",
        "Economic Indicators", 
        "Federal Reserve Data",
        "Banking & Interest Rates",
        "Earnings Information",
        "International Data",
        "Market Sentiment",
        "Corporate Events",
        "Geopolitical Events",
        "Data Calendar",
        "Key Websites",
        "Impact Matrix"
    ]
)

# SEC Filings Section
if category == "SEC Filings Overview":
    st.header("🏛️ SEC Filings - The Foundation of Market Transparency")
    
    st.markdown("""
    **SEC filings are among the most important sources of market-moving information.** The Securities and Exchange Commission 
    requires public companies to disclose material information that helps investors make informed decisions.
    """)
    
    sec_filings = MarketDataSources.get_sec_filing_types()
    
    # Create tabs for different filing types
    filing_tabs = st.tabs(list(sec_filings.keys()))
    
    for i, (filing_type, filing_info) in enumerate(sec_filings.items()):
        with filing_tabs[i]:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader(f"{filing_type} - {filing_info['name']}")
                st.write(f"**Description:** {filing_info['description']}")
                st.write(f"**Frequency:** {filing_info['frequency']}")
                st.write(f"**Market Impact:** {filing_info['market_impact']}")
                st.write(f"**Filing Deadline:** {filing_info['timing']}")
                
                if 'key_sections' in filing_info:
                    st.write("**Key Sections to Watch:**")
                    for section in filing_info['key_sections']:
                        st.write(f"• {section}")
            
            with col2:
                # Importance indicator
                importance_color = {
                    "Very High": "red",
                    "High": "orange", 
                    "Medium-High": "yellow",
                    "Medium": "lightblue"
                }
                
                color = importance_color.get(filing_info['importance'], 'gray')
                st.markdown(f"""
                <div style="
                    background-color: {color}; 
                    padding: 20px; 
                    border-radius: 10px; 
                    text-align: center;
                    color: white;
                    font-weight: bold;
                    font-size: 18px;
                ">
                    Importance: {filing_info['importance']}
                </div>
                """, unsafe_allow_html=True)
    
    # SEC Filing Impact Visualization
    st.subheader("📊 SEC Filing Importance Comparison")
    
    filing_data = []
    for filing_type, info in sec_filings.items():
        importance_score = {
            "Very High": 4,
            "High": 3, 
            "Medium-High": 2.5,
            "Medium": 2,
            "Low": 1
        }
        filing_data.append({
            "Filing Type": filing_type,
            "Importance Score": importance_score.get(info['importance'], 2),
            "Frequency": info['frequency']
        })
    
    df = pd.DataFrame(filing_data)
    
    fig = px.bar(
        df, 
        x="Filing Type", 
        y="Importance Score",
        color="Importance Score",
        color_continuous_scale="RdYlBu_r",
        title="SEC Filing Types by Market Impact Importance"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# Economic Indicators Section
elif category == "Economic Indicators":
    st.header("📈 Critical Economic Indicators")
    
    st.markdown("""
    Economic indicators provide insights into the overall health of the economy and significantly influence 
    Federal Reserve policy decisions, which in turn affect all financial markets.
    """)
    
    econ_indicators = MarketDataSources.get_economic_indicators()
    
    # Create a comprehensive table
    econ_data = []
    for indicator, info in econ_indicators.items():
        econ_data.append({
            "Indicator": info['name'],
            "Frequency": info['frequency'],
            "Source": info['source'],
            "Importance": info['importance'],
            "Market Impact": info['market_impact'],
            "Release Timing": info['typical_release']
        })
    
    df_econ = pd.DataFrame(econ_data)
    st.dataframe(df_econ, use_container_width=True)
    
    # Detailed view for selected indicator
    st.subheader("🔍 Detailed Indicator Analysis")
    selected_indicator = st.selectbox(
        "Select an indicator for detailed information:",
        list(econ_indicators.keys())
    )
    
    if selected_indicator:
        info = econ_indicators[selected_indicator]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Frequency", info['frequency'])
        with col2:
            st.metric("Importance", info['importance'])
        with col3:
            st.metric("Source", info['source'])
        
        st.write(f"**Description:** {info['description']}")
        st.write(f"**Market Impact:** {info['market_impact']}")
        st.write(f"**Typical Release:** {info['typical_release']}")

# Federal Reserve Data Section
elif category == "Federal Reserve Data":
    st.header("🏦 Federal Reserve Information Sources")
    
    st.markdown("""
    The Federal Reserve is the most powerful force in financial markets. Every word, policy change, 
    and data release from the Fed can move markets significantly.
    """)
    
    fed_data = MarketDataSources.get_federal_reserve_data()
    
    for source, info in fed_data.items():
        with st.expander(f"{info['name']} - {info['importance']} Importance"):
            st.write(f"**Description:** {info['description']}")
            st.write(f"**Market Impact:** {info['market_impact']}")
            st.write(f"**Frequency:** {info['frequency']}")
            
            # Add specific insights for FOMC
            if source == "FedFundsRate":
                st.info("💡 **Trading Tip:** Markets often price in rate changes weeks before FOMC meetings. Watch fed funds futures for market expectations.")

# Banking & Interest Rates Section
elif category == "Banking & Interest Rates":
    st.header("🏛️ Banking Sector & Interest Rate Data Sources")
    
    st.markdown("""
    **Interest rates are the backbone of the financial system.** Banking sector health and interest rate movements 
    affect everything from mortgages to corporate borrowing costs to currency values.
    """)
    
    banking_data = MarketDataSources.get_banking_and_interest_rates()
    
    # Create tabs for different rate categories
    rate_tabs = st.tabs(["Key Interest Rates", "Banking Health", "Credit Markets", "Benchmark Rates"])
    
    with rate_tabs[0]:  # Key Interest Rates
        st.subheader("🎯 Critical Interest Rate Indicators")
        key_rates = ["MortgageRates", "BankPrimeRate", "SOFR", "EFFR"]
        
        for rate_key in key_rates:
            if rate_key in banking_data:
                info = banking_data[rate_key]
                with st.expander(f"{info['name']} - {info['importance']} Importance"):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**Description:** {info['description']}")
                        st.write(f"**Market Impact:** {info['market_impact']}")
                        st.write(f"**Frequency:** {info['frequency']}")
                        if 'url' in info:
                            st.markdown(f"[📊 View Data Source]({info['url']})")
                    
                    with col2:
                        importance_colors = {
                            "Very High": "#FF4444",
                            "High": "#FF8800", 
                            "Medium-High": "#FFBB00"
                        }
                        color = importance_colors.get(info['importance'], '#88CCFF')
                        st.markdown(f"""
                        <div style="background: {color}; padding: 8px; border-radius: 5px; 
                                   text-align: center; color: white; font-weight: bold;">
                            {info['importance']}
                        </div>
                        """, unsafe_allow_html=True)
    
    with rate_tabs[1]:  # Banking Health
        st.subheader("🏦 Banking Sector Health Indicators") 
        banking_health = ["BankEarnings", "BankStressTests", "BankReserves"]
        
        for bank_key in banking_health:
            if bank_key in banking_data:
                info = banking_data[bank_key]
                st.write(f"**{info['name']}**")
                st.write(f"• {info['description']}")
                st.write(f"• **Impact:** {info['market_impact']}")
                st.write(f"• **Frequency:** {info['frequency']}")
                if 'url' in info:
                    st.markdown(f"[📊 View Source]({info['url']})")
                st.write("---")
    
    with rate_tabs[2]:  # Credit Markets
        st.subheader("💳 Credit Market Indicators")
        credit_markets = ["CreditSpreads", "CorporateBondYields", "TED_Spread", "CommercialPaper"]
        
        for credit_key in credit_markets:
            if credit_key in banking_data:
                info = banking_data[credit_key]
                st.write(f"**{info['name']}**")
                st.write(f"• {info['description']}")
                st.write(f"• **Impact:** {info['market_impact']}")
                if 'url' in info:
                    st.markdown(f"[📊 View Source]({info['url']})")
                st.write("---")
    
    with rate_tabs[3]:  # Benchmark Rates
        st.subheader("📊 Benchmark Rate Information")
        benchmark_rates = ["LIBOR", "InterestRateSwaps", "MunicipalBondYields"]
        
        for benchmark_key in benchmark_rates:
            if benchmark_key in banking_data:
                info = banking_data[benchmark_key]
                st.write(f"**{info['name']}**")
                st.write(f"• {info['description']}")
                st.write(f"• **Impact:** {info['market_impact']}")
                if 'url' in info:
                    st.markdown(f"[📊 View Source]({info['url']})")
                st.write("---")
    
    # Add comprehensive interest rate explanation
    st.info("""
    **🎓 Key Interest Rate Relationships:**
    - **Fed Funds Rate → Prime Rate:** Banks typically set prime rate = Fed funds + 3%
    - **SOFR vs LIBOR:** SOFR is replacing LIBOR as the US benchmark rate
    - **Credit Spreads:** Higher spreads indicate increased credit risk and economic uncertainty
    - **Yield Curve:** Shape indicates economic expectations (normal, flat, inverted)
    """)

# Earnings Information Section
elif category == "Earnings Information":
    st.header("💰 Earnings-Related Information Sources")
    
    earnings_data = MarketDataSources.get_earnings_data()
    
    # Create visual timeline
    st.subheader("📅 Earnings Season Timeline")
    
    timeline_data = []
    for source, info in earnings_data.items():
        timeline_data.append({
            "Source": info['name'],
            "Importance": info['importance'],
            "Impact": info['market_impact'],
            "Timing": info['timing']
        })
    
    df_earnings = pd.DataFrame(timeline_data)
    st.dataframe(df_earnings, use_container_width=True)
    
    # Earnings calendar insight
    st.info("""
    **📊 Earnings Season Pattern:**
    - **Pre-announcement Period:** 2-3 weeks before earnings (watch for guidance updates)
    - **Earnings Release:** Company reports actual vs expected results
    - **Post-earnings:** Analyst revisions and forward guidance updates
    """)

# International Data Section
elif category == "International Data":
    st.header("🌍 International Market Data Sources")
    
    intl_data = MarketDataSources.get_international_data()
    
    for source, info in intl_data.items():
        with st.container():
            st.subheader(info['name'])
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**Description:** {info['description']}")
                st.write(f"**Market Impact:** {info['market_impact']}")
                st.write(f"**Source:** {info['source']}")
            
            with col2:
                importance_color = {
                    "Very High": "#FF4444",
                    "High": "#FF8800",
                    "Medium-High": "#FFBB00",
                    "Medium": "#88CCFF"
                }
                color = importance_color.get(info['importance'], '#CCCCCC')
                
                st.markdown(f"""
                <div style="
                    background-color: {color}; 
                    padding: 10px; 
                    border-radius: 5px; 
                    text-align: center;
                    color: white;
                    font-weight: bold;
                ">
                    {info['importance']}
                </div>
                """, unsafe_allow_html=True)

# Market Sentiment Section
elif category == "Market Sentiment":
    st.header("🎭 Market Sentiment Indicators")
    
    sentiment_data = MarketDataSources.get_sentiment_indicators()
    
    # Create sentiment dashboard
    st.subheader("📊 Sentiment Indicator Dashboard")
    
    for indicator, info in sentiment_data.items():
        with st.expander(f"{info['name']} ({info['source']})"):
            st.write(f"**Description:** {info['description']}")
            st.write(f"**Market Impact:** {info['market_impact']}")
            st.write(f"**Importance:** {info['importance']}")
            
            # Add interpretation guides
            if indicator == "VIX":
                st.info("""
                **VIX Interpretation:**
                - Below 12: Extreme complacency
                - 12-20: Low volatility/fear
                - 20-30: Elevated uncertainty  
                - Above 30: High fear/panic
                """)
            elif indicator == "PutCallRatio":
                st.info("""
                **Put/Call Ratio Interpretation:**
                - Below 0.7: Bullish sentiment (potential top)
                - 0.7-1.0: Neutral sentiment
                - Above 1.0: Bearish sentiment (potential bottom)
                """)

# Corporate Events Section
elif category == "Corporate Events":
    st.header("🏢 Corporate Events That Move Markets")
    
    corporate_events = MarketDataSources.get_corporate_events()
    
    # Create impact comparison
    event_data = []
    for event, info in corporate_events.items():
        impact_score = {
            "Very High": 4,
            "High": 3,
            "Medium-High": 2.5,
            "Medium": 2,
            "Low": 1
        }
        event_data.append({
            "Event": info['name'],
            "Impact Score": impact_score.get(info['impact'], 2),
            "Description": info['description'],
            "Typical Effect": info['typical_effect']
        })
    
    df_events = pd.DataFrame(event_data)
    
    # Visualization
    fig = px.bar(
        df_events,
        x="Event",
        y="Impact Score", 
        color="Impact Score",
        color_continuous_scale="Reds",
        title="Corporate Events by Market Impact"
    )
    fig.update_xaxis(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed table
    st.subheader("📋 Detailed Corporate Events Guide")
    display_df = df_events[['Event', 'Description', 'Typical Effect']].copy()
    st.dataframe(display_df, use_container_width=True)

# Geopolitical Events Section
elif category == "Geopolitical Events":
    st.header("🌐 Geopolitical Events Affecting Markets")
    
    geopolitical_events = MarketDataSources.get_geopolitical_events()
    
    st.markdown("""
    Geopolitical events can cause significant market volatility and often lead to flights to quality assets like 
    U.S. Treasury bonds and gold. These events are often unpredictable but their market impacts follow certain patterns.
    """)
    
    # Display as organized list
    col1, col2 = st.columns(2)
    
    mid_point = len(geopolitical_events) // 2
    
    with col1:
        st.subheader("Political & Economic Events")
        for event in geopolitical_events[:mid_point]:
            st.write(f"• {event}")
    
    with col2:
        st.subheader("Crisis & Disruption Events")
        for event in geopolitical_events[mid_point:]:
            st.write(f"• {event}")
    
    # Impact patterns
    st.subheader("📈 Typical Market Reactions to Geopolitical Events")
    
    reaction_data = {
        "Asset Class": ["U.S. Stocks", "Bonds", "Gold", "Oil", "USD", "Emerging Markets"],
        "Initial Reaction": ["Down", "Up", "Up", "Up/Down*", "Up", "Down"],
        "Recovery Time": ["Days-Weeks", "Weeks", "Sustained", "Variable", "Days", "Weeks-Months"]
    }
    
    df_reactions = pd.DataFrame(reaction_data)
    st.dataframe(df_reactions, use_container_width=True)
    
    st.caption("*Oil reaction depends on whether the event affects supply (up) or demand (down)")

# Data Calendar Section
elif category == "Data Calendar":
    st.header("📅 Weekly Economic Data Calendar")
    
    calendar_data = MarketDataSources.get_data_calendar()
    
    st.markdown("""
    Understanding the typical release schedule helps traders and investors prepare for market-moving events.
    Most important releases occur on Fridays, particularly the first Friday of each month.
    """)
    
    # Create visual calendar
    for day, events in calendar_data.items():
        with st.expander(f"{day} - {len(events)} typical releases"):
            for event in events:
                st.write(f"• {event}")
    
    # Highlight important patterns
    st.subheader("🎯 Key Calendar Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **First Friday Pattern:**
        - Employment Report (NFP)
        - Unemployment Rate
        - Usually highest volatility day
        """)
    
    with col2:
        st.info("""
        **Mid-Month Pattern:**
        - CPI (inflation data)
        - Retail Sales
        - PPI (producer prices)
        """)

# Key Websites Section
elif category == "Key Websites":
    st.header("🔗 Essential Market Information Websites")
    
    websites = MarketDataSources.get_key_websites()
    
    for site_key, site_info in websites.items():
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.subheader(site_info['name'])
                st.write(f"**Description:** {site_info['description']}")
                st.markdown(f"**URL:** [{site_info['url']}]({site_info['url']})")
            
            with col2:
                importance_color = {
                    "Very High": "#FF4444",
                    "High": "#FF8800",
                    "Medium-High": "#FFBB00"
                }
                color = importance_color.get(site_info['importance'], '#CCCCCC')
                
                st.markdown(f"""
                <div style="
                    background-color: {color}; 
                    padding: 15px; 
                    border-radius: 8px; 
                    text-align: center;
                    color: white;
                    font-weight: bold;
                ">
                    {site_info['importance']}
                </div>
                """, unsafe_allow_html=True)

# Impact Matrix Section
elif category == "Impact Matrix":
    st.header("📊 Market Impact Matrix")
    
    st.markdown("""
    This matrix shows how different information sources affect various asset classes. 
    Understanding these relationships helps in portfolio positioning and risk management.
    """)
    
    impact_matrix = MarketDataSources.get_impact_matrix()
    
    # Create heatmap
    impact_scores = {
        "Extremely High": 5,
        "Very High": 4,
        "High": 3,
        "Medium": 2,
        "Low": 1
    }
    
    # Convert text to numbers for heatmap
    matrix_numeric = impact_matrix.copy()
    for col in ['Stocks', 'Bonds', 'Currencies', 'Commodities']:
        matrix_numeric[col] = matrix_numeric[col].map(impact_scores)
    
    # Create heatmap
    fig = px.imshow(
        matrix_numeric.set_index('Data Source')[['Stocks', 'Bonds', 'Currencies', 'Commodities']].T,
        color_continuous_scale='RdYlBu_r',
        aspect='auto',
        title="Market Impact Heatmap by Asset Class"
    )
    
    fig.update_layout(
        xaxis_title="Information Source",
        yaxis_title="Asset Class",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show detailed matrix
    st.subheader("📋 Detailed Impact Matrix")
    st.dataframe(impact_matrix, use_container_width=True)
    
    # Priority timeline
    st.subheader("⏰ Information Priority Timeline")
    
    priority_timeline = MarketDataSources.get_priority_timeline()
    
    for priority, sources in priority_timeline.items():
        with st.expander(f"{priority.replace('_', ' ')} ({len(sources)} sources)"):
            for source in sources:
                st.write(f"• {source}")

# Summary and Key Takeaways
st.markdown("---")
st.header("💡 Key Takeaways")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 Most Critical Sources")
    st.markdown("""
    1. **FOMC Decisions** - Affects all markets
    2. **SEC 8-K Filings** - Material corporate events  
    3. **Economic Indicators** - CPI, NFP, GDP
    4. **Earnings Reports** - Individual stock drivers
    5. **Geopolitical Events** - Market risk factors
    """)

with col2:
    st.subheader("📈 Best Practices")
    st.markdown("""
    1. **Monitor economic calendar** daily
    2. **Set up SEC filing alerts** for holdings
    3. **Follow Fed communication** closely  
    4. **Track earnings seasons** quarterly
    5. **Stay informed on geopolitical** developments
    """)

# Educational note
st.info("""
💡 **Remember:** Information is only valuable if you can act on it quickly. 
Set up alerts and have a systematic approach to processing market-moving information.
""")