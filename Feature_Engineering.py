"""
🔧 Feature Engineering Page — Technical Indicators
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.data_loader import download_stock_data
from utils.feature_engineering import engineer_all_features

st.set_page_config(page_title="Feature Engineering", page_icon="🔧", layout="wide")

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

st.markdown("# 🔧 Feature Engineering")
st.markdown("---")

# Load data
if "stock_data" not in st.session_state:
    st.info("Loading default AAPL data...")
    df, _ = download_stock_data("AAPL", "5y")
    st.session_state["stock_data"] = df
    st.session_state["stock_ticker"] = "AAPL"

df = st.session_state["stock_data"].copy()
ticker = st.session_state.get("stock_ticker", "AAPL")

if st.button("🛠️ Generate All Technical Indicators", use_container_width=True):
    with st.spinner("Engineering features..."):
        df_eng = engineer_all_features(df)
        df_eng.dropna(inplace=True)
        st.session_state["engineered_data"] = df_eng
        st.success(f"✅ Created {len(df_eng.columns) - len(df.columns)} new features! Total: {len(df_eng.columns)} columns")

if "engineered_data" in st.session_state:
    df_eng = st.session_state["engineered_data"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Original Features", len(df.columns))
    c2.metric("Engineered Features", len(df_eng.columns))
    c3.metric("New Features", len(df_eng.columns) - len(df.columns))

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Moving Averages", "📊 RSI", "📉 MACD", "📐 Bollinger", "🗂️ All Features"])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_eng.tail(300).index, y=df_eng.tail(300)["Close"],
            mode="lines", name="Close", line=dict(color="#ffffff", width=2)))
        for col in [c for c in df_eng.columns if "SMA" in c or "EMA" in c]:
            color = {"SMA_7":"#e94560","SMA_14":"#ffd700","SMA_21":"#00ff80","SMA_50":"#ff69b4",
                     "EMA_12":"#00d4ff","EMA_26":"#0f3460"}.get(col, "#888")
            fig.add_trace(go.Scatter(x=df_eng.tail(300).index, y=df_eng.tail(300)[col],
                mode="lines", name=col, line=dict(color=color, width=1.5)))
        fig.update_layout(title=f"{ticker} — Moving Averages", template="plotly_dark", height=500)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4],
            subplot_titles=["Price", "RSI"])
        fig.add_trace(go.Scatter(x=df_eng.tail(300).index, y=df_eng.tail(300)["Close"],
            line=dict(color="#00d4ff"), name="Close"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_eng.tail(300).index, y=df_eng.tail(300)["RSI"],
            line=dict(color="#e94560"), name="RSI"), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#ff6b6b", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#00ff80", row=2, col=1)
        fig.update_layout(title=f"{ticker} — RSI", template="plotly_dark", height=550)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4],
            subplot_titles=["Price", "MACD"])
        fig.add_trace(go.Scatter(x=df_eng.tail(300).index, y=df_eng.tail(300)["Close"],
            line=dict(color="#00d4ff"), name="Close"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_eng.tail(300).index, y=df_eng.tail(300)["MACD"],
            line=dict(color="#e94560"), name="MACD"), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_eng.tail(300).index, y=df_eng.tail(300)["MACD_Signal"],
            line=dict(color="#ffd700"), name="Signal"), row=2, col=1)
        fig.add_trace(go.Bar(x=df_eng.tail(300).index, y=df_eng.tail(300)["MACD_Hist"],
            marker_color=np.where(df_eng.tail(300)["MACD_Hist"]>=0, "#00ff80", "#e94560"),
            name="Histogram", opacity=0.6), row=2, col=1)
        fig.update_layout(title=f"{ticker} — MACD", template="plotly_dark", height=550)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        fig = go.Figure()
        d = df_eng.tail(300)
        fig.add_trace(go.Scatter(x=d.index, y=d["Close"], line=dict(color="#00d4ff", width=2), name="Close"))
        fig.add_trace(go.Scatter(x=d.index, y=d["BB_Upper"], line=dict(color="#e94560", dash="dash"), name="Upper Band"))
        fig.add_trace(go.Scatter(x=d.index, y=d["BB_Lower"], line=dict(color="#00ff80", dash="dash"),
            name="Lower Band", fill="tonexty", fillcolor="rgba(0,212,255,0.05)"))
        fig.update_layout(title=f"{ticker} — Bollinger Bands", template="plotly_dark", height=500)
        st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.markdown(f"#### All {len(df_eng.columns)} Features")
        new_cols = [c for c in df_eng.columns if c not in df.columns]
        st.markdown("**New Engineered Features:**")
        st.write(new_cols)
        st.dataframe(df_eng.tail(15), use_container_width=True, height=400)
        st.dataframe(df_eng.describe().round(3), use_container_width=True)
else:
    st.info("👆 Click the button above to generate all technical indicators.")

st.markdown("---")
st.caption("🔧 Feature Engineering | Stock Price Prediction System")
