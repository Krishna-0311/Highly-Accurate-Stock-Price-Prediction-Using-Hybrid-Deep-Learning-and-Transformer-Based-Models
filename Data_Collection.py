"""
📊 Data Collection & Exploration Page
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.data_loader import download_stock_data
from utils.evaluation import plot_candlestick

st.set_page_config(page_title="Data Collection", page_icon="📊", layout="wide")

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

st.markdown("# 📊 Data Collection & Exploration")
st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    ticker = st.text_input("🏷️ Stock Ticker", value="AAPL", max_chars=10).upper()
with col2:
    period = st.selectbox("📅 Time Period", ["1y", "2y", "3y", "5y", "10y", "max"], index=3)
with col3:
    st.markdown("")
    st.markdown("")
    fetch = st.button("📥 Fetch Data", use_container_width=True)

if fetch or "stock_data" in st.session_state:
    if fetch:
        with st.spinner(f"Fetching {ticker} data..."):
            df, source = download_stock_data(ticker, period)
            st.session_state["stock_data"] = df
            st.session_state["stock_ticker"] = ticker
            st.session_state["data_source"] = source

    df = st.session_state["stock_data"]
    ticker = st.session_state.get("stock_ticker", "AAPL")
    source = st.session_state.get("data_source", "sample")

    st.success(f"✅ {ticker} — {len(df):,} trading days | Source: {source}")

    # Metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    latest = df["Close"].iloc[-1]
    prev = df["Close"].iloc[-2]
    change = latest - prev
    pct = (change / prev) * 100
    c1.metric("Latest Close", f"${latest:.2f}", f"{change:+.2f} ({pct:+.2f}%)")
    c2.metric("52W High", f"${df['High'].tail(252).max():.2f}")
    c3.metric("52W Low", f"${df['Low'].tail(252).min():.2f}")
    c4.metric("Avg Volume", f"{df['Volume'].mean():,.0f}")
    c5.metric("Total Days", f"{len(df):,}")

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Chart", "🗂️ Data", "📊 Statistics", "📉 Returns"])

    with tab1:
        fig = plot_candlestick(df.tail(250), f"{ticker} — Last 250 Trading Days")
        st.plotly_chart(fig, use_container_width=True)

        # Volume chart
        fig_vol = go.Figure(go.Bar(
            x=df.tail(250).index, y=df.tail(250)["Volume"],
            marker_color=np.where(df.tail(250)["Close"] >= df.tail(250)["Open"], "#00d4ff", "#e94560"),
            opacity=0.7
        ))
        fig_vol.update_layout(title="Trading Volume", template="plotly_dark", height=300,
            xaxis_title="Date", yaxis_title="Volume")
        st.plotly_chart(fig_vol, use_container_width=True)

    with tab2:
        st.markdown("#### Latest Data")
        st.dataframe(df.tail(20).sort_index(ascending=False), use_container_width=True, height=500)

        st.markdown("#### Column Info")
        info = pd.DataFrame({
            "Type": df.dtypes, "Non-Null": df.notnull().sum(),
            "Null": df.isnull().sum(), "Min": df.min(), "Max": df.max()
        })
        st.dataframe(info, use_container_width=True)

    with tab3:
        st.markdown("#### Descriptive Statistics")
        st.dataframe(df.describe().round(3), use_container_width=True)

        st.markdown("#### Correlation Matrix")
        numeric = df.select_dtypes(include=[np.number])
        corr = numeric.corr()
        fig_corr = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale=[[0,"#0d1b2a"],[0.5,"#0099cc"],[1,"#e94560"]],
            text=corr.values.round(3), texttemplate="%{text}",
            textfont=dict(size=11, color="white")
        ))
        fig_corr.update_layout(title="Correlation Matrix", template="plotly_dark", height=500)
        st.plotly_chart(fig_corr, use_container_width=True)

    with tab4:
        st.markdown("#### Daily Returns Distribution")
        returns = df["Close"].pct_change().dropna()
        fig_ret = go.Figure()
        fig_ret.add_trace(go.Histogram(x=returns, nbinsx=80, marker_color="#00d4ff", opacity=0.8))
        fig_ret.update_layout(title="Daily Returns Distribution", template="plotly_dark",
            height=400, xaxis_title="Return", yaxis_title="Frequency")
        st.plotly_chart(fig_ret, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Mean Return", f"{returns.mean()*100:.4f}%")
        c2.metric("Std Dev", f"{returns.std()*100:.4f}%")
        c3.metric("Sharpe Ratio", f"{(returns.mean()/returns.std())*np.sqrt(252):.4f}")

else:
    st.info("👆 Enter a ticker and click **Fetch Data** to begin.")

st.markdown("---")
st.caption("📊 Data Collection | Stock Price Prediction System")
