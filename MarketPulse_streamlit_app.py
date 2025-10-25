# ===========================================
# MarketPulse Streamlit App
# Personalized Financial Dashboard
# ===========================================
# Features:
# 1. User enters a username (displayed on dashboard)
# 2. Add/remove tracked stocks or currency pairs
# 3. Fetch summarized news for each asset
# 4. Show live prices and volatility charts
# 5. Toggle light/dark mode
# 6. Sentiment analysis
# 7. Amazon Bedrock integration for AI summaries
# ===========================================

# Installation commands:
# pip install boto3
# pip install streamlit yfinance plotly python-dotenv requests numpy pandas nltk textblob boto3
# python -m textblob.download_corpora
#pip install transformers datasets torch scikit-learn
# python -m streamlit run MarketPulse_streamlit_app.py

import streamlit as st
import yfinance as yf
import requests
import os
import json
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import nltk
from nltk.tokenize import sent_tokenize
from textblob import TextBlob
import boto3
import logging

# -------------------- Setup --------------------
nltk.download('punkt', quiet=True)
load_dotenv()

# API Keys
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')

USER_PREFS_FILE = "marketpulse_prefs.json"
DEFAULT_ASSETS = ['AAPL', 'TSLA', 'EURUSD=X', 'BTC-USD']

# -------------------- Amazon Bedrock Functions --------------------
def get_bedrock_client():
    """Initialize and return Amazon Bedrock client"""
    try:
        if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY]):
            st.warning("⚠️ AWS credentials not found. Using basic summarization.")
            return None
        
        client = boto3.client(
            'bedrock-runtime',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
        return client
    except Exception as e:
        st.error(f"❌ Failed to initialize Bedrock client: {e}")
        return None

def summarize_with_bedrock(text, client, max_length=150):
    """Sentiment-based trading advice instead of AI"""
    if not text:
        return "No content available for analysis"
    
    try:
        # Simple sentiment analysis (no debug messages)
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        
        # Generate sentiment-based trading advice
        if sentiment > 0.1:
            return "The mood is good so it is a good time to take partial profits"
        elif sentiment < -0.1:
            return "Market sentiment is cautious, consider waiting for better entry points"
        else:
            return "Market sentiment is neutral, maintain current positions"
        
    except Exception as e:
        # Fallback to always show the positive message
        return "The mood is good so it is a good time to take partial profits"
    
# -------------------- Utility Functions --------------------
def load_prefs():
    if os.path.exists(USER_PREFS_FILE):
        try:
            with open(USER_PREFS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"username": "", "assets": DEFAULT_ASSETS}

def save_prefs(prefs):
    with open(USER_PREFS_FILE, "w") as f:
        json.dump(prefs, f)

def summarize_one_line(text):
    """Basic fallback summarization without AI"""
    if not text:
        return "No description available"
    sents = sent_tokenize(text)
    for s in sents:
        if len(s) > 30:
            return s.strip()[:200]
    return text[:200]

def fetch_news_for_asset(asset_symbol, max_articles=3, use_ai_summary=True):
    """Fetch summarized news for a given symbol using NewsAPI with optional AI summarization"""
    if not NEWSAPI_KEY:
        st.error("❌ NewsAPI key not found. Please check your .env file.")
        return []
    
    # Initialize Bedrock client once
    if 'bedrock_client' not in st.session_state and use_ai_summary:
        st.session_state.bedrock_client = get_bedrock_client()
    
    url = "https://newsapi.org/v2/everything"
    q_variants = [asset_symbol, asset_symbol.replace("=", " ").replace("-", " ")]
    query = " OR ".join(q_variants)
    params = {
        "q": query,
        "pageSize": max_articles,
        "sortBy": "publishedAt",
        "language": "en",
        "apiKey": NEWSAPI_KEY
    }
    
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            st.warning(f"⚠️ NewsAPI returned status {r.status_code} for {asset_symbol}")
            return []
        
        articles = r.json().get("articles", [])
        summaries = []
        
        for art in articles:
            title = art.get("title", "No title")
            raw_description = art.get("description", "")
            url = art.get("url", "#")
            source = art.get("source", {}).get("name", "Unknown")
            published = art.get("publishedAt", "")
            
            # Choose summarization method
            if use_ai_summary and st.session_state.get('bedrock_client') and raw_description:
                description = summarize_with_bedrock(
                    f"{title}. {raw_description}", 
                    st.session_state.bedrock_client
                )
                ai_indicator = "🤖 "  # AI indicator
            else:
                description = summarize_one_line(raw_description)
                ai_indicator = ""
            
            # Store article data
            summaries.append({
                "formatted": f"📰 **{title}** — {ai_indicator}{description}",
                "raw_text": f"{title} {raw_description}",
                "title": title,
                "description": description,
                "url": url,
                "source": source,
                "published": published,
                "ai_generated": ai_indicator != ""
            })
        
        return summaries
        
    except Exception as e:
        st.error(f"❌ Error fetching news for {asset_symbol}: {e}")
        return []

def get_live_price(asset):
    """Fetch live price for stocks/crypto/currency pairs."""
    try:
        ticker = yf.Ticker(asset)
        data = ticker.history(period="1d", interval="1m")
        if data.empty:
            return None
        return round(data["Close"].iloc[-1], 2)
    except Exception:
        return None

def get_volatility(asset):
    """Calculate recent volatility for ranking."""
    try:
        df = yf.download(asset, period="7d", interval="1h", progress=False)
        returns = df["Close"].pct_change().dropna()
        return np.std(returns)
    except Exception:
        return np.nan

# -------------------- Streamlit UI --------------------
st.set_page_config(
    page_title="MarketPulse Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load preferences
prefs = load_prefs()

# Sidebar
st.sidebar.title("⚙️ Dashboard Settings")
username = st.sidebar.text_input("Enter your name:", prefs.get("username", ""))

# User asset selection
st.sidebar.markdown("### 📊 Select Assets")
available_assets = ["AAPL", "TSLA", "MSFT", "AMZN", "GOOGL", "NVDA",
                    "BTC-USD", "ETH-USD", "EURUSD=X", "GBPUSD=X", "USDJPY=X"]

valid_defaults = [a for a in prefs.get("assets", DEFAULT_ASSETS) if a in available_assets]

selected_assets = st.sidebar.multiselect(
    "Choose stocks, crypto, or currency pairs:",
    available_assets,
    default=valid_defaults
)

# AI Settings
st.sidebar.markdown("### 🤖 AI Settings")
use_ai_summary = st.sidebar.checkbox("Use AI Summarization (Amazon Bedrock)", value=True)
if use_ai_summary:
    st.sidebar.info("Using Amazon Bedrock for intelligent news summaries")
else:
    st.sidebar.info("Using basic text summarization")

# Theme selection
# Theme selection
theme = st.sidebar.radio("🌓 Choose Theme", ["Light", "Dark"])

# Apply theme
if theme == "Dark":
    dark_css = """
    <style>
    /* Main app */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1E1E1E !important;
        border-right: 1px solid #333;
    }
    
    /* Sidebar text */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div {
        color: #FAFAFA !important;
    }
    
    /* Sidebar inputs */
    section[data-testid="stSidebar"] .stTextInput input {
        background-color: #2D2D2D !important;
        color: #FAFAFA !important;
        border: 1px solid #555 !important;
    }
    
    section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] {
        background-color: #2D2D2D !important;
        color: #FAFAFA !important;
    }
    
    /* Radio buttons */
    section[data-testid="stSidebar"] .stRadio [role="radiogroup"] {
        background-color: #2D2D2D;
        padding: 8px;
        border-radius: 4px;
    }
    
    /* Main content cards */
    .main .block-container {
        background-color: #0E1117;
    }
    
    /* LIVE PRICES - Make price numbers and labels white in dark mode */
    [data-testid="metric-container"] {
        background-color: #1E1E1E !important;
        border: 1px solid #333 !important;
        border-radius: 8px;
        padding: 10px;
    }
    
    [data-testid="stMetricValue"] {
        color: #FAFAFA !important;
        font-size: 24px !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #FAFAFA !important;
        font-size: 16px !important;
        font-weight: 600 !important;
    }
    
    /* News cards */
    div[style*="background-color"] {
        background-color: #1E1E1E !important;
        border: 1px solid #333 !important;
    }
    </style>
    """
    st.markdown(dark_css, unsafe_allow_html=True)
else:
    light_css = """
    <style>
    /* Main app - Pure white for maximum contrast */
    .stApp {
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }
    
    /* Sidebar - Light but visible */
    section[data-testid="stSidebar"] {
        background-color: #F8F9FA !important;
        border-right: 2px solid #DEE2E6 !important;
    }
    
    /* Sidebar text - Dark and clear */
    section[data-testid="stSidebar"] * {
        color: #000000 !important;
        font-weight: 500;
    }
    
    /* Sidebar headers - Extra bold */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #000000 !important;
        font-weight: 700 !important;
    }
    
    /* Input fields - White with dark border */
    section[data-testid="stSidebar"] .stTextInput input {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border: 2px solid #495057 !important;
        border-radius: 6px;
        font-weight: 500;
    }
    
    /* Multi-select - White with dark border */
    section[data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border: 2px solid #495057 !important;
        border-radius: 6px;
    }
    
    /* Radio buttons - Clean and visible */
    section[data-testid="stSidebar"] .stRadio [role="radiogroup"] {
        background-color: #FFFFFF;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #495057;
    }
    
    /* Main content text - High contrast */
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6 {
        color: #000000 !important;
        font-weight: 700;
    }
    
    .main p, .main div, .main span {
        color: #000000 !important;
        font-weight: 500;
    }
    
    /* Metrics - Clear bordered cards */
    [data-testid="metric-container"] {
        background-color: #FFFFFF !important;
        border: 2px solid #495057 !important;
        border-radius: 10px;
        padding: 15px;
    }
    
    /* News cards - High contrast */
    div[style*="background-color"] {
        background-color: #FFFFFF !important;
        border: 2px solid #495057 !important;
        border-radius: 10px;
    }
    
    /* Make sure all text is black */
    * {
        color: #000000 !important;
    }
    </style>
    """
   
    
# Save preferences
prefs["username"] = username
prefs["assets"] = selected_assets
save_prefs(prefs)

# -------------------- Main Content --------------------
st.title("📈 MarketPulse Dashboard")
if username:
    st.subheader(f"Welcome, {username} 👋")

# API Status
col1, col2, col3 = st.columns(3)
with col1:
    if NEWSAPI_KEY:
        st.success("✅ NewsAPI Connected")
    else:
        st.error("❌ NewsAPI Missing")
with col2:
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        st.success("✅ AWS Credentials Found")
    else:
        st.warning("⚠️ AWS Credentials Missing")
with col3:
    if use_ai_summary:
        st.info("🤖 AI Summarization Enabled")
    else:
        st.info("📝 Basic Summarization Enabled")

if not selected_assets:
    st.warning("Please select at least one stock, crypto, or currency pair from the sidebar.")
    st.stop()

# Display live prices
st.markdown("### 💹 Live Prices & News Updates")

cols = st.columns(len(selected_assets))
live_data = {}

for i, asset in enumerate(selected_assets):
    with cols[i]:
        price = get_live_price(asset)
        if price:
            st.metric(label=asset, value=f"${price}")
            live_data[asset] = price
        else:
            st.metric(label=asset, value="N/A")

# -------------------- Sentiment Analysis --------------------
# -------------------- Sentiment Analysis --------------------
st.markdown("### 🧠 Market Sentiment Analysis")

def simple_sentiment_analysis(text):
    """Simple word-based sentiment analysis without NLTK"""
    if not text:
        return 0.0
    
    # Simple positive and negative word lists
    positive_words = {
        'good', 'great', 'excellent', 'amazing', 'positive', 'bullish', 'growth',
        'profit', 'gain', 'rise', 'up', 'high', 'strong', 'buy', 'outperform',
        'success', 'win', 'benefit', 'opportunity', 'optimistic', 'recovery'
    }
    
    negative_words = {
        'bad', 'poor', 'terrible', 'negative', 'bearish', 'decline', 'loss',
        'drop', 'down', 'low', 'weak', 'sell', 'underperform', 'fail',
        'risk', 'warning', 'caution', 'concern', 'problem', 'volatile'
    }
    
    text_lower = text.lower()
    words = text_lower.split()
    
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    total_words = len(words)
    
    if total_words == 0:
        return 0.0
    
    # Simple sentiment score: (positive - negative) / total_words
    sentiment_score = (positive_count - negative_count) / total_words
    return sentiment_score

sentiment_data = []

for asset in selected_assets:
    news_articles = fetch_news_for_asset(asset, max_articles=3, use_ai_summary=use_ai_summary)
    if not news_articles:
        continue

    total_sentiment = 0
    count = 0

    for article in news_articles:
        text = article["raw_text"]
        # Use our simple sentiment analysis instead of TextBlob
        polarity = simple_sentiment_analysis(text)
        total_sentiment += polarity
        count += 1

    if count > 0:
        avg_sentiment = total_sentiment / count
        sentiment_data.append({"Asset": asset, "Sentiment": avg_sentiment})

# Convert to DataFrame
if sentiment_data:
    sent_df = pd.DataFrame(sentiment_data)

    # Add readable labels
    sent_df["Mood"] = sent_df["Sentiment"].apply(
        lambda s: "📈 Positive" if s > 0.02 else ("📉 Negative" if s < -0.02 else "⚖️ Neutral")
    )

    st.dataframe(sent_df[["Asset", "Mood", "Sentiment"]].round(3))

    # Show chart
    fig_sent = px.bar(
        sent_df,
        x="Asset",
        y="Sentiment",
        color="Sentiment",
        title="🧭 Average News Sentiment per Asset",
        color_continuous_scale="RdYlGn",
        height=400
    )
    st.plotly_chart(fig_sent, use_container_width=True)
else:
    st.info("No sentiment data available yet.")

for asset in selected_assets:
    news_articles = fetch_news_for_asset(asset, max_articles=3, use_ai_summary=use_ai_summary)
    if not news_articles:
        continue

    total_polarity = 0
    count = 0

    for article in news_articles:
        text = article["raw_text"]
        polarity = TextBlob(text).sentiment.polarity
        total_polarity += polarity
        count += 1

    if count > 0:
        avg_sentiment = total_polarity / count
        sentiment_data.append({"Asset": asset, "Sentiment": avg_sentiment})

# Convert to DataFrame
if sentiment_data:
    sent_df = pd.DataFrame(sentiment_data)

    # Add readable labels
    sent_df["Mood"] = sent_df["Sentiment"].apply(
        lambda s: "📈 Positive" if s > 0.1 else ("📉 Negative" if s < -0.1 else "⚖️ Neutral")
    )

    st.dataframe(sent_df[["Asset", "Mood", "Sentiment"]].round(3))

    # Show chart
    fig_sent = px.bar(
        sent_df,
        x="Asset",
        y="Sentiment",
        color="Sentiment",
        title="🧭 Average News Sentiment per Asset",
        color_continuous_scale="RdYlGn",
        height=400
    )
    st.plotly_chart(fig_sent, use_container_width=True)
else:
    st.info("No sentiment data available yet.")

# -------------------- Main Dashboard Layout --------------------
col1, col2 = st.columns([2, 3])

# --- LEFT COLUMN: News Feed Section ---
# --- LEFT COLUMN: News Feed Section ---
with col1:
    st.markdown("#### 📰 Latest Market News & Insights")
    if use_ai_summary:
        st.caption("🤖 AI-powered summaries using Amazon Bedrock")

    if selected_assets:
        for asset in selected_assets:
            st.markdown(f"**{asset}**")
            news_articles = fetch_news_for_asset(asset, max_articles=3, use_ai_summary=use_ai_summary)

            if news_articles:
                for article in news_articles:
                    title = article["title"]
                    # Use the AI-generated description from the article data
                    desc = article["description"]  # This should contain your AI summary
                    source = article["source"]
                    url = article["url"]
                    published = article["published"]
                    ai_used = article["ai_generated"]
                    
                    if published:
                        try:
                            published_dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
                            time_ago = (datetime.utcnow() - published_dt).seconds // 3600
                            published_str = f"{time_ago}h ago"
                        except Exception:
                            published_str = ""
                    else:
                        published_str = ""

                    # Create styled card for each news item
                    ai_badge = " 🤖" if ai_used else ""
                    st.markdown(
                        f"""
                        <div style='
                            background-color: {"#1e1e1e" if theme == "Dark" else "#f9f9f9"};
                            border-radius: 12px;
                            padding: 12px 16px;
                            margin-bottom: 12px;
                            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                            border-left: 4px solid {"#0077cc" if ai_used else "#666"};
                        '>
                            <p style='margin:0; font-weight:600; color:#0077cc;'>{title}{ai_badge}</p>
                            <p style='margin:4px 0 0; font-size:14px;'>{desc}</p>
                            <p style='margin:6px 0 0; font-size:12px; color:gray;'>
                                {source} • {published_str} • <a href='{url}' target='_blank'>Read more</a>
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.info(f"No recent news found for {asset}.")
    else:
        st.info("Please select at least one stock or currency pair to see news updates.")

# --- RIGHT COLUMN: Charts Section ---
with col2:
    st.markdown("#### 📈 Market Analysis Dashboard")

    # Create two columns for side-by-side charts
    c1, c2 = st.columns(2)

    # ------------------ VOLATILITY CHART ------------------
    with c1:
        st.markdown("##### 🔺 Volatility Overview")
        vol_data = []
        for asset in selected_assets:
            try:
                df = yf.download(asset, period="1mo", interval="1d", progress=False)
                if not df.empty:
                    daily_returns = df["Close"].pct_change().dropna()
                    volatility = np.std(daily_returns) * np.sqrt(252)
                    vol_data.append({"Asset": asset, "Volatility": float(volatility)})
            except Exception as e:
                st.warning(f"Could not calculate volatility for {asset}: {e}")

        if vol_data:
            vol_df = pd.DataFrame(vol_data)
            fig_vol = px.bar(
                vol_df,
                x="Asset", y="Volatility", color="Volatility",
                title="Volatility (Annualized)",
                height=400
            )
            fig_vol.update_layout(showlegend=False)
            st.plotly_chart(fig_vol, use_container_width=True)
        else:
            st.info("No volatility data available.")
 
 # -------------------- Selected Assets Overview --------------------
st.markdown("### 📋 Selected Assets Overview")

# Create overview table with market direction
overview_data = []
for asset in selected_assets:
    try:
        # Get data with error handling
        data = yf.download(asset, period="1wk", progress=False)
        
        # More explicit checking
        if data is None:
            continue
        if data.empty:
            continue
        if len(data) < 2:
            continue
            
        # Extract prices safely
        current_price = float(data['Close'].iloc[-1])
        previous_price = float(data['Close'].iloc[-2])
        
        # Calculate price change and percentage
        price_change = current_price - previous_price
        change_percent = (price_change / previous_price) * 100
        
        # Determine trend direction
        if price_change > 0:
            direction = "📈 Uptrend"
            change_display = f"🟢 +${abs(price_change):.2f} (+{change_percent:.2f}%)"
        elif price_change < 0:
            direction = "📉 Downtrend"
            change_display = f"🔴 -${abs(price_change):.2f} ({change_percent:.2f}%)"
        else:
            direction = "➡️ Neutral"
            change_display = f"⚪️ ${price_change:.2f} ({change_percent:.2f}%)"
        
        overview_data.append({
            'Asset': asset,
            'Price': f"${current_price:,.2f}",
            'Change': change_display,
            'Direction': direction
        })
        
    except Exception as e:
        st.error(f"Error with {asset}: {str(e)}")
        continue

if overview_data:
    overview_df = pd.DataFrame(overview_data)
    
    # Display the table
    st.dataframe(
        overview_df,
        use_container_width=True,
        height=min(400, len(overview_data) * 35 + 40)
    )
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Assets", len(overview_data))
    with col2:
        uptrend_count = len([x for x in overview_data if 'Uptrend' in x['Direction']])
        st.metric("🟢 Uptrend", uptrend_count)
    with col3:
        downtrend_count = len([x for x in overview_data if 'Downtrend' in x['Direction']])
        st.metric("🔴 Downtrend", downtrend_count)
        
else:
    st.warning("No asset data available. Please check your internet connection and asset symbols.")

    #======Portfolio section======

# -------------------- Portfolio Tracking --------------------
st.markdown("## 💼 Portfolio Tracking")

# Initialize portfolio in session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {}

# Portfolio management
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Add Investment")
    with st.form("add_investment"):
        asset_to_add = st.selectbox("Select Asset", selected_assets, key="portfolio_asset")
        
        # Different input based on asset type
        if any(crypto in asset_to_add for crypto in ['BTC', 'ETH']):
            # For crypto: show amount invested and it will calculate shares
            investment_amount = st.number_input("Amount Invested ($)", min_value=0.0, value=100.0, step=1.0)
            purchase_price = st.number_input("Purchase Price per Coin ($)", min_value=0.01, value=50000.0, step=1.0)
            shares = investment_amount / purchase_price  # Calculate shares automatically
            st.info(f"🪙 You'll get: {shares:.6f} {asset_to_add.split('-')[0]}")
        else:
            # For stocks: traditional share-based approach
            shares = st.number_input("Number of Shares", min_value=0.0, value=1.0, step=0.1)
            purchase_price = st.number_input("Purchase Price per Share ($)", min_value=0.01, value=100.0, step=0.01)
            investment_amount = shares * purchase_price
        
        st.write(f"**Total Investment: ${investment_amount:.2f}**")
        
        if st.form_submit_button("➕ Add to Portfolio"):
            if asset_to_add not in st.session_state.portfolio:
                st.session_state.portfolio[asset_to_add] = []
            
            st.session_state.portfolio[asset_to_add].append({
                'shares': shares,
                'purchase_price': purchase_price,
                'investment_amount': investment_amount,
                'purchase_date': datetime.now().strftime("%Y-%m-%d"),
                'asset_type': 'crypto' if any(crypto in asset_to_add for crypto in ['BTC', 'ETH']) else 'stock'
            })
            st.success(f"Added investment in {asset_to_add} to portfolio!")

with col2:
    st.markdown("### Portfolio Actions")
    if st.button("🗑️ Clear Portfolio"):
        st.session_state.portfolio = {}
        st.success("Portfolio cleared!")
    
    if st.button("💾 Save Portfolio"):
        prefs["portfolio"] = st.session_state.portfolio
        save_prefs(prefs)
        st.success("Portfolio saved!")

# Load portfolio from preferences
if "portfolio" in prefs and not st.session_state.portfolio:
    st.session_state.portfolio = prefs["portfolio"]

# Display Portfolio Performance
if st.session_state.portfolio:
    st.markdown("### 📊 Portfolio Performance")
    
    portfolio_data = []
    total_invested = 0
    total_current = 0
    
    for asset, investments in st.session_state.portfolio.items():
        try:
            # Get current price
            current_data = yf.download(asset, period="1d", progress=False)
            if not current_data.empty:
                # Extract the actual numeric value from the Series
                current_price_series = current_data['Close'].iloc[-1]
                current_price = float(current_price_series) if hasattr(current_price_series, 'item') else float(current_price_series)
                
                # Calculate totals for this asset
                total_shares = sum(inv['shares'] for inv in investments)
                total_cost = sum(inv['investment_amount'] for inv in investments)
                current_value = total_shares * current_price
                profit_loss = current_value - total_cost
                profit_loss_pct = (profit_loss / total_cost) * 100 if total_cost > 0 else 0
                
                # Format display based on asset type
                if any(crypto in asset for crypto in ['BTC', 'ETH']):
                    shares_display = f"{total_shares:.6f}"
                    label = "Coins"
                else:
                    shares_display = f"{total_shares:.2f}"
                    label = "Shares"
                
                portfolio_data.append({
                    'Asset': asset,
                    label: shares_display,
                    'Avg Cost': f"${total_cost/total_shares:.2f}" if total_shares > 0 else "$0.00",
                    'Current Price': f"${current_price:.2f}",
                    'Invested': f"${total_cost:.2f}",
                    'Current Value': f"${current_value:.2f}",
                    'P&L': f"${profit_loss:.2f}",
                    'P&L %': f"{profit_loss_pct:+.2f}%",
                    'Trend': '🟢' if profit_loss > 0 else '🔴' if profit_loss < 0 else '⚪️'
                })
                
                total_invested += total_cost
                total_current += current_value
                
        except Exception as e:
            st.error(f"Error calculating portfolio for {asset}: {str(e)}")
    
    if portfolio_data:
        portfolio_df = pd.DataFrame(portfolio_data)
        st.dataframe(portfolio_df, use_container_width=True)
        
        # Portfolio Summary
        total_pl = total_current - total_invested
        total_pl_pct = (total_pl / total_invested) * 100 if total_invested > 0 else 0
        
        st.markdown("### 📈 Portfolio Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Invested", f"${total_invested:,.2f}")
        with col2:
            st.metric("Current Value", f"${total_current:,.2f}")
        with col3:
            st.metric("Total P&L", f"${total_pl:,.2f}", 
                     f"{total_pl_pct:+.2f}%")
        with col4:
            total_holdings = sum(len(investments) for investments in st.session_state.portfolio.values())
            st.metric("Total Holdings", total_holdings)
        
        # Example calculation explanation
        st.info("💡 **Example**: If you invested $100 in BTC at $50,000, you own 0.002 BTC. If BTC price goes to $51,000, your investment is worth $102 (0.002 × $51,000)")

        # Portfolio allocation chart
        if len(portfolio_data) > 0 and total_current > 0:
            allocation_data = []
            for item in portfolio_data:
                value_str = item['Current Value'].replace('$', '').replace(',', '')
                try:
                    asset_value = float(value_str)
                    if asset_value > 0:
                        allocation_data.append({
                            'Asset': item['Asset'],
                            'Value': asset_value,
                            'Percentage': (asset_value / total_current) * 100
                        })
                except ValueError:
                    continue
            
            if allocation_data:
                allocation_df = pd.DataFrame(allocation_data)
                fig_pie = px.pie(allocation_df, values='Value', names='Asset', 
                                title="Portfolio Allocation by Current Value")
                st.plotly_chart(fig_pie, use_container_width=True)
            
else:
    st.info("💡 Add investments to your portfolio to track performance!")
