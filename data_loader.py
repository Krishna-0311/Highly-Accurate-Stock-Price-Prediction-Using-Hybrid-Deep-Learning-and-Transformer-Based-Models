"""
Data loading utilities — Yahoo Finance download with sample fallback.
"""
import pandas as pd
import numpy as np
import streamlit as st


def generate_sample_data(ticker="AAPL", days=1500):
    """Generate realistic synthetic stock data as fallback."""
    np.random.seed(42)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=days)

    base_price = 150.0
    returns = np.random.normal(0.0003, 0.018, days)
    prices = base_price * np.exp(np.cumsum(returns))

    high_spread = np.random.uniform(0.005, 0.025, days)
    low_spread = np.random.uniform(0.005, 0.025, days)
    open_spread = np.random.uniform(-0.01, 0.01, days)

    close = prices
    high = close * (1 + high_spread)
    low = close * (1 - low_spread)
    opn = close * (1 + open_spread)
    volume = np.random.lognormal(mean=18, sigma=0.5, size=days).astype(int)

    df = pd.DataFrame({
        "Date": dates,
        "Open": opn.round(2),
        "High": high.round(2),
        "Low": low.round(2),
        "Close": close.round(2),
        "Adj Close": close.round(2),
        "Volume": volume
    })
    df.set_index("Date", inplace=True)
    df.attrs["ticker"] = ticker
    return df


@st.cache_data(ttl=3600)
def download_stock_data(ticker="AAPL", period="5y"):
    """Download stock data from Yahoo Finance, fallback to sample."""
    try:
        import yfinance as yf
        df = yf.download(ticker, period=period, progress=False)
        if df is None or df.empty:
            raise ValueError("Empty data")
        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.attrs["ticker"] = ticker
        return df, "yahoo_finance"
    except Exception:
        df = generate_sample_data(ticker)
        return df, "sample"


def generate_sample_news(ticker="AAPL", n=50):
    """Generate sample financial news headlines."""
    np.random.seed(42)
    positive = [
        f"{ticker} reports record quarterly earnings, stock surges",
        f"Analysts upgrade {ticker} following strong guidance",
        f"{ticker} announces major partnership, investors optimistic",
        f"Strong demand drives {ticker} revenue growth",
        f"{ticker} beats expectations with impressive profit margins",
        f"Institutional investors increase {ticker} holdings significantly",
        f"{ticker} expands into new markets, shares rally",
        f"Innovation pipeline at {ticker} strengthens investor confidence",
        f"{ticker} dividend increase signals management confidence",
        f"Revenue guidance for {ticker} exceeds analyst estimates",
    ]
    negative = [
        f"{ticker} faces regulatory probe, shares decline sharply",
        f"Supply chain disruptions weigh on {ticker} outlook",
        f"Analysts downgrade {ticker} amid competition concerns",
        f"{ticker} misses revenue targets for the quarter",
        f"Market turbulence drags {ticker} to new lows",
        f"Insider selling raises red flags for {ticker} investors",
        f"{ticker} cuts workforce amid restructuring efforts",
        f"Rising costs pressure {ticker} profit margins",
        f"Demand slowdown hits {ticker} growth projections",
        f"{ticker} warns of lower earnings in upcoming quarter",
    ]
    neutral = [
        f"{ticker} holds steady amid mixed economic signals",
        f"Market awaits {ticker} quarterly results next week",
        f"{ticker} trading volume stable as investors watch macro data",
        f"Sector rotation sees {ticker} consolidate at current levels",
        f"{ticker} board announces strategic review process",
    ]

    headlines, sentiments, scores = [], [], []
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)

    for _ in range(n):
        r = np.random.random()
        if r < 0.4:
            h = np.random.choice(positive)
            sentiments.append("positive")
            scores.append(round(np.random.uniform(0.6, 0.95), 3))
        elif r < 0.75:
            h = np.random.choice(negative)
            sentiments.append("negative")
            scores.append(round(np.random.uniform(-0.95, -0.5), 3))
        else:
            h = np.random.choice(neutral)
            sentiments.append("neutral")
            scores.append(round(np.random.uniform(-0.2, 0.2), 3))
        headlines.append(h)

    return pd.DataFrame({
        "Date": dates,
        "Headline": headlines,
        "Sentiment": sentiments,
        "Score": scores
    })
