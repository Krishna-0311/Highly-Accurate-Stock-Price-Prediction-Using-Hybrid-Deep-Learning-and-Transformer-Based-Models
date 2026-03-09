"""
💬 Sentiment Analysis Page — FinBERT
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.data_loader import generate_sample_news
from utils.sentiment import analyze_sentiment_finbert, aggregate_daily_sentiment

st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    .stApp { background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 50%, #1b2838 100%); font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0a0a1a 0%, #0d1b2a 100%); border-right: 1px solid rgba(0,212,255,0.2); }
    h1,h2,h3 { color: #fff !important; }
    div[data-testid="stMetric"] { background: linear-gradient(135deg, rgba(0,212,255,0.08), rgba(233,69,96,0.05)); border: 1px solid rgba(0,212,255,0.25); border-radius: 16px; padding: 20px; }
    div[data-testid="stMetric"] label { color: #8899aa !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #00d4ff !important; font-weight: 700 !important; }
    .stButton>button { background: linear-gradient(135deg,#00d4ff,#0099cc); color: white; border: none; border-radius: 12px; padding: 12px 28px; font-weight: 600; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: rgba(13,27,42,0.7); border-radius: 12px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { border-radius: 10px; color: #8899aa; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg,#00d4ff,#0099cc) !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 💬 Financial Sentiment Analysis")
st.markdown("---")

ticker = st.session_state.get("stock_ticker", "AAPL")

col1, col2 = st.columns(2)
with col1:
    data_source = st.radio("News Data Source",
        ["📦 Sample News Headlines", "📤 Upload CSV (headline column)"], horizontal=True)
with col2:
    n_headlines = st.slider("Number of headlines", 20, 200, 50) if "Sample" in data_source else 50

uploaded = None
if "Upload" in data_source:
    uploaded = st.file_uploader("Upload CSV with 'headline' column", type=["csv"])

if st.button("🔍 Analyse Sentiment", use_container_width=True):
    with st.spinner("Analysing sentiment..."):
        if uploaded:
            news_df = pd.read_csv(uploaded)
            if "headline" not in news_df.columns and "Headline" not in news_df.columns:
                st.error("CSV must have a 'headline' or 'Headline' column")
                st.stop()
            col_name = "headline" if "headline" in news_df.columns else "Headline"
            headlines = news_df[col_name].tolist()
        else:
            news_df = generate_sample_news(ticker, n_headlines)
            headlines = news_df["Headline"].tolist()

        results_df, method = analyze_sentiment_finbert(headlines)

        if "Date" in news_df.columns:
            results_df["Date"] = news_df["Date"].values[:len(results_df)]

        st.session_state["sentiment_results"] = results_df
        st.session_state["sentiment_method"] = method
        st.session_state["news_data"] = news_df

        st.success(f"✅ Analysed {len(results_df)} headlines using **{method}**")

if "sentiment_results" in st.session_state:
    results_df = st.session_state["sentiment_results"]
    method = st.session_state["sentiment_method"]

    # Metrics
    pos = (results_df["sentiment"] == "positive").sum()
    neg = (results_df["sentiment"] == "negative").sum()
    neu = (results_df["sentiment"] == "neutral").sum()
    avg_score = results_df["score"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Positive 📈", pos)
    c2.metric("Negative 📉", neg)
    c3.metric("Neutral ➡️", neu)
    c4.metric("Avg Score", f"{avg_score:.3f}")

    tab1, tab2, tab3 = st.tabs(["📊 Distribution", "📋 Headlines", "📈 Trend"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure(go.Pie(
                labels=["Positive", "Negative", "Neutral"],
                values=[pos, neg, neu], hole=0.5,
                marker=dict(colors=["#00ff80", "#e94560", "#ffd700"]),
                textinfo="label+percent+value"
            ))
            fig.update_layout(title="Sentiment Distribution", template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = go.Figure(go.Histogram(
                x=results_df["score"], nbinsx=30,
                marker_color="#00d4ff", opacity=0.8
            ))
            fig.update_layout(title="Sentiment Score Distribution", template="plotly_dark",
                height=400, xaxis_title="Score", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.dataframe(
            results_df[["headline", "sentiment", "score"]].style.apply(
                lambda x: ["background-color: rgba(0,255,128,0.1)" if v == "positive"
                           else "background-color: rgba(233,69,96,0.1)" if v == "negative"
                           else "" for v in x],
                subset=["sentiment"]
            ),
            use_container_width=True, height=500
        )

    with tab3:
        if "Date" in results_df.columns:
            daily = aggregate_daily_sentiment(results_df)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=daily["Date"], y=daily["avg_score"],
                mode="lines+markers", line=dict(color="#00d4ff"), name="Avg Score"))
            fig.add_hline(y=0, line_dash="dash", line_color="#888")
            fig.update_layout(title="Daily Sentiment Trend", template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)

            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=daily["Date"], y=daily["pos_count"], name="Positive", marker_color="#00ff80"))
            fig2.add_trace(go.Bar(x=daily["Date"], y=-daily["neg_count"], name="Negative", marker_color="#e94560"))
            fig2.update_layout(title="Daily Sentiment Count", template="plotly_dark",
                height=400, barmode="relative")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No date column available for trend analysis.")

    if method == "rule_based":
        st.warning("⚠️ Using rule-based fallback. Install `transformers` and `torch` for FinBERT: `pip install transformers torch`")

st.markdown("---")
st.caption("💬 Sentiment Analysis | Stock Price Prediction System")
