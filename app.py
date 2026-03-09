"""
🏠 Stock Price Prediction System — Main App
Hybrid Deep Learning & Transformer-Based Framework
"""
import streamlit as st

st.set_page_config(
    page_title="Stock Price Prediction System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Premium Dark Theme CSS ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 50%, #1b2838 100%);
        font-family: 'Inter', sans-serif;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a1a 0%, #0d1b2a 100%);
        border-right: 1px solid rgba(0, 212, 255, 0.2);
    }
    h1, h2, h3 { color: #ffffff !important; font-family: 'Inter', sans-serif !important; }
    h1 { font-weight: 800 !important; }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.08), rgba(233, 69, 96, 0.05));
        border: 1px solid rgba(0, 212, 255, 0.25);
        border-radius: 16px; padding: 20px;
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.08);
        transition: transform 0.3s ease;
    }
    div[data-testid="stMetric"]:hover { transform: translateY(-4px); }
    div[data-testid="stMetric"] label { color: #8899aa !important; font-size: 14px !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #00d4ff !important; font-weight: 700 !important; }

    .stButton > button {
        background: linear-gradient(135deg, #00d4ff, #0099cc);
        color: white; border: none; border-radius: 12px;
        padding: 12px 28px; font-weight: 600;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #33ddff, #00bbee);
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(0, 212, 255, 0.5);
    }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: rgba(13, 27, 42, 0.7); border-radius: 12px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { border-radius: 10px; color: #8899aa; font-weight: 500; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #00d4ff, #0099cc) !important; color: white !important; }

    .hero-card {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(233, 69, 96, 0.05));
        border: 1px solid rgba(0, 212, 255, 0.25);
        border-radius: 20px; padding: 40px; text-align: center;
        margin: 20px 0; box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
    }
    .hero-title {
        font-size: 42px; font-weight: 800;
        background: linear-gradient(135deg, #00d4ff, #e94560);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero-subtitle { font-size: 18px; color: #8899aa; }
    .feature-card {
        background: linear-gradient(135deg, rgba(13, 27, 42, 0.9), rgba(27, 40, 56, 0.7));
        border: 1px solid rgba(0, 212, 255, 0.15); border-radius: 16px;
        padding: 28px; text-align: center; min-height: 200px;
        transition: all 0.3s ease;
    }
    .feature-card:hover { transform: translateY(-5px); border-color: rgba(0, 212, 255, 0.4); }
    .feature-icon { font-size: 48px; margin-bottom: 15px; }
    .feature-title { font-size: 18px; font-weight: 700; color: #fff; margin-bottom: 8px; }
    .feature-desc { font-size: 14px; color: #8899aa; line-height: 1.5; }
    .info-box { background: rgba(0, 212, 255, 0.06); border-left: 4px solid #00d4ff; border-radius: 0 12px 12px 0; padding: 18px 24px; margin: 15px 0; color: #c0d0e0; }

    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #0d1b2a; }
    ::-webkit-scrollbar-thumb { background: #00d4ff; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ───
with st.sidebar:
    st.markdown("## 📈 Navigation")
    st.markdown("---")
    st.markdown("""
    ### 📋 Pipeline Steps
    1. 📊 **Data Collection**
    2. 🔧 **Feature Engineering**
    3. 💬 **Sentiment Analysis**
    4. 🤖 **Model Training**
    5. 📈 **Model Evaluation**
    6. 🔮 **Predictions**
    7. 📋 **About & Methodology**
    """)

# ─── Hero ───
st.markdown("""
<div class='hero-card'>
    <div class='hero-title'>📈 Stock Price Prediction System</div>
    <div class='hero-subtitle'>Hybrid Deep Learning & Transformer-Based Forecasting Framework</div>
</div>
""", unsafe_allow_html=True)

# ─── Metrics ───
c1, c2, c3, c4 = st.columns(4)
c1.metric("ML/DL Models", "6", help="XGBoost, CNN, LSTM, GRU, Transformer, CNN-LSTM")
c2.metric("Evaluation Metrics", "3", help="MAE, RMSE, MAPE")
c3.metric("Technical Indicators", "10+", help="MA, RSI, MACD, Bollinger, ATR, OBV")
c4.metric("Sentiment", "FinBERT", help="Financial sentiment analysis")

st.markdown("")

# ─── Feature Cards ───
st.markdown("## ✨ System Capabilities")
col1, col2, col3 = st.columns(3)

cards = [
    ("📊", "Live Data Pipeline", "Fetch real-time stock data from Yahoo Finance with technical indicator engineering."),
    ("🧠", "Hybrid Deep Learning", "CNN-LSTM, LSTM, GRU, and Transformer architectures for temporal pattern learning."),
    ("💬", "FinBERT Sentiment", "Extract financial sentiment from news headlines using domain-specific NLP."),
    ("📈", "Advanced Evaluation", "MAE, RMSE, MAPE metrics with residual analysis and training curves."),
    ("🔮", "Price Forecasting", "Multi-step ahead predictions with confidence visualisations."),
    ("⚡", "XGBoost Baseline", "Gradient boosting baseline with feature importance analysis."),
]

for i, (icon, title, desc) in enumerate(cards):
    col = [col1, col2, col3][i % 3]
    if i == 3:
        col1, col2, col3 = st.columns(3)
        col = [col1, col2, col3][0]
    elif i > 3:
        col = [col1, col2, col3][i % 3]
    with col:
        st.markdown(f"""
        <div class='feature-card'>
            <div class='feature-icon'>{icon}</div>
            <div class='feature-title'>{title}</div>
            <div class='feature-desc'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("")

# ─── Research Overview ───
st.markdown("## 🎓 Research Overview")
tab1, tab2, tab3 = st.tabs(["🔬 Problem", "🎯 Objectives", "📐 Methodology"])

with tab1:
    st.markdown("""
    **Stock market forecasting** remains one of the most complex problems in financial analytics.

    **Key Challenges:**
    - Financial time-series contain noise and abrupt fluctuations
    - Traditional models fail to capture nonlinear relationships
    - Recurrent networks struggle with long-range dependencies
    - Most studies ignore textual sentiment from financial news
    """)

with tab2:
    st.markdown("""
    1. Collect and preprocess historical stock price and financial news data
    2. Engineer technical indicators (MA, RSI, MACD, Bollinger Bands)
    3. Implement XGBoost as baseline ML model
    4. Develop CNN, LSTM, GRU, and Transformer architectures
    5. Extract financial sentiment using FinBERT
    6. Integrate sentiment with time-series features
    7. Evaluate with MAE, RMSE, MAPE under identical conditions
    """)

with tab3:
    st.markdown("""
    | Phase | Description |
    |-------|-------------|
    | **Data** | Yahoo Finance OHLCV + financial news |
    | **Features** | MA, RSI, MACD, Bollinger, ATR, OBV, lag features |
    | **Sentiment** | FinBERT financial text analysis |
    | **Models** | XGBoost, CNN, LSTM, GRU, Transformer, CNN-LSTM |
    | **Evaluation** | MAE, RMSE, MAPE + residual analysis |
    """)

st.markdown("")
st.markdown("""
<div class='info-box'>
    🚀 <strong>Getting Started:</strong> Navigate to <strong>📊 Data Collection</strong> in the sidebar to begin.
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.caption("📈 Stock Price Prediction System | Hybrid Deep Learning & Transformer Framework")
