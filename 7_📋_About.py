"""
📋 About & Research Methodology Page
"""
import streamlit as st

st.set_page_config(page_title="About & Methodology", page_icon="📋", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    .stApp { background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 50%, #1b2838 100%); font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0a0a1a 0%, #0d1b2a 100%); border-right: 1px solid rgba(0,212,255,0.2); }
    h1,h2,h3 { color: #fff !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: rgba(13,27,42,0.7); border-radius: 12px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { border-radius: 10px; color: #8899aa; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg,#00d4ff,#0099cc) !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 📋 About & Research Methodology")
st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔬 Research", "📐 Methodology", "🤖 Models", "⚖️ Ethics", "📚 References"
])

with tab1:
    st.markdown("""
    ## Highly Accurate Stock Price Prediction Using Hybrid Deep Learning and Transformer-Based Models

    **Research Area:** Financial Analytics, Deep Learning, NLP, Time-Series Forecasting

    ### Problem Statement
    Stock markets are inherently volatile and exhibit nonlinear, non-stationary behavior:
    - Financial time-series contain noise and abrupt fluctuations
    - Traditional statistical models fail with complex nonlinear relationships
    - Recurrent networks struggle with long-range dependencies
    - Most studies evaluate models independently without unified benchmarking
    - Many systems rely only on numerical data, ignoring textual sentiment

    ### Research Questions
    1. Does hybrid deep learning outperform traditional ML in stock price prediction?
    2. How effectively can Transformers capture long-term dependencies vs LSTM/GRU?
    3. Does integrating financial sentiment significantly improve accuracy?
    4. Which architecture provides the most stable performance across stocks?
    5. Can multimodal learning improve robustness in volatile markets?

    ### Objectives
    1. Collect and preprocess historical stock price and financial news data
    2. Engineer technical indicators (MA, RSI, MACD, Bollinger Bands)
    3. Implement XGBoost as baseline ML model
    4. Develop CNN, LSTM, GRU, and Transformer architectures
    5. Extract financial sentiment using FinBERT
    6. Integrate sentiment with time-series features
    7. Evaluate with MAE, RMSE, MAPE under identical conditions
    """)

with tab2:
    st.markdown("""
    ## Methodology

    ### Data Pipeline
    | Step | Description |
    |------|-------------|
    | **Collection** | Yahoo Finance OHLCV data + financial news |
    | **Preprocessing** | Missing value handling, normalisation (MinMaxScaler) |
    | **Feature Engineering** | SMA, EMA, RSI, MACD, Bollinger Bands, ATR, OBV |
    | **Sentiment** | FinBERT financial domain NLP |
    | **Windowing** | Sliding window sequences (configurable 20-120 days) |
    | **Splitting** | Chronological train/test split (no data leakage) |

    ### Evaluation Metrics
    | Metric | Formula | Purpose |
    |--------|---------|---------|
    | **MAE** | Mean \\|actual - predicted\\| | Average error magnitude |
    | **RMSE** | √(Mean (actual - predicted)²) | Penalises large errors |
    | **MAPE** | Mean \\|(actual - predicted)/actual\\| × 100 | Percentage error |

    ### Hyperparameter Tuning
    - Early stopping with patience monitoring
    - Learning rate reduction on plateau
    - Configurable sequence lengths, epochs, and architectures
    """)

with tab3:
    st.markdown("""
    ## Algorithms Implemented

    ### 1. XGBoost (Baseline)
    - **Type:** Gradient Boosting (tabular ML)
    - **Input:** Flattened time-series windows
    - **Strengths:** Fast training, feature importance, robust baseline

    ### 2. CNN (Convolutional Neural Network)
    - **Type:** 1D convolution for local pattern extraction
    - **Architecture:** Conv1D → BatchNorm → MaxPool → Dense
    - **Strengths:** Captures short-term local temporal patterns

    ### 3. LSTM (Long Short-Term Memory)
    - **Type:** Recurrent neural network with gating
    - **Architecture:** LSTM(128) → Dropout → LSTM(64) → Dense
    - **Strengths:** Models long-term sequential dependencies

    ### 4. GRU (Gated Recurrent Unit)
    - **Type:** Simplified recurrent architecture
    - **Architecture:** GRU(128) → Dropout → GRU(64) → Dense
    - **Strengths:** Computationally efficient alternative to LSTM

    ### 5. Transformer
    - **Type:** Self-attention based sequence model
    - **Architecture:** Multi-Head Attention → FFN → Global Pooling
    - **Strengths:** Captures global long-range dependencies simultaneously

    ### 6. CNN-LSTM (Hybrid)
    - **Type:** Combined local + sequential modelling
    - **Architecture:** Conv1D → MaxPool → LSTM → Dense
    - **Strengths:** Local feature extraction + temporal modelling

    ### 7. FinBERT (Sentiment)
    - **Type:** Pre-trained Transformer for financial text
    - **Purpose:** Extract sentiment from financial news
    - **Output:** Positive/negative/neutral classification + score

    ### Technology Stack
    | Component | Technology |
    |-----------|-----------|
    | Language | Python 3.8+ |
    | ML | Scikit-learn, XGBoost |
    | Deep Learning | TensorFlow / Keras |
    | NLP | HuggingFace Transformers (FinBERT) |
    | Data | Yahoo Finance (yfinance), Pandas |
    | Visualisation | Plotly, Matplotlib, Seaborn |
    | Web App | Streamlit |
    """)

with tab4:
    st.markdown("""
    ## Ethical Considerations
    - ✅ **Data:** Only publicly available market data used
    - ✅ **No PII:** No personally identifiable information processed
    - ✅ **Academic Use:** Research purposes only
    - ✅ **Transparency:** All methods openly documented
    - ✅ **Disclaimer:** Predictions are for research; not financial advice

    > ⚠️ **Disclaimer:** This system is developed for academic research purposes only.
    > Stock market predictions carry inherent uncertainty. Do not make investment
    > decisions based solely on model outputs.
    """)

with tab5:
    st.markdown("""
    ## Key References

    1. Vaswani, A. et al. (2017). *Attention is All You Need.* NeurIPS.
    2. Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory.* Neural Computation, 9(8).
    3. Cho, K. et al. (2014). *Learning Phrase Representations Using RNN Encoder-Decoder.* EMNLP.
    4. Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD.
    5. Yang, Z. et al. (2019). *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models.* arXiv.
    6. Fischer, A. & Krauss, C. (2018). *Deep Learning with LSTM Networks for Financial Market Predictions.* EJOR.
    7. Siami-Namini, S. et al. (2018). *Comparison of ARIMA and LSTM in Forecasting Time Series.* ICMLA.
    8. Kim, Y. (2014). *Convolutional Neural Networks for Sentence Classification.* EMNLP.
    9. Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers.* NAACL-HLT.
    10. Kabir, M.R. et al. (2025). *LSTM-Transformer-based Hybrid Deep Learning Model.* Sci.
    11. Dong, J. & Liang, S. (2025). *Hybrid CNN-LSTM-GNN Neural Network.* Entropy.
    12. Agarwal, S. (2025). *Enhancing Stock Market Forecasting Using Transformer-based Models.* IJCA.
    """)

st.markdown("---")
st.caption("📋 About & Methodology | Stock Price Prediction System")
