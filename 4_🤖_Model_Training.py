"""
🤖 Model Training Page — XGBoost, CNN, LSTM, GRU, Transformer, CNN-LSTM
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time, sys, os
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.data_loader import download_stock_data
from utils.feature_engineering import engineer_all_features, prepare_sequences
from utils.models import (build_xgboost, build_lstm, build_gru, build_cnn,
    build_cnn_lstm, build_transformer, train_dl_model, train_xgboost,
    predict_model, TF_AVAILABLE)
from utils.evaluation import evaluate_predictions, plot_predictions, plot_training_history

st.set_page_config(page_title="Model Training", page_icon="🤖", layout="wide")

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
</style>
""", unsafe_allow_html=True)

st.markdown("# 🤖 Model Training & Comparison")
st.markdown("---")

# Load and prepare data
if "engineered_data" not in st.session_state:
    st.info("Auto-loading and engineering features...")
    if "stock_data" not in st.session_state:
        df, _ = download_stock_data("AAPL", "5y")
        st.session_state["stock_data"] = df
        st.session_state["stock_ticker"] = "AAPL"
    df_eng = engineer_all_features(st.session_state["stock_data"])
    df_eng.dropna(inplace=True)
    st.session_state["engineered_data"] = df_eng

df_eng = st.session_state["engineered_data"]
ticker = st.session_state.get("stock_ticker", "AAPL")

# Model selection
st.markdown("### 🎛️ Configuration")
col1, col2, col3 = st.columns(3)

all_models = ["XGBoost", "LSTM", "GRU", "CNN", "CNN-LSTM", "Transformer"]
if not TF_AVAILABLE:
    all_models = ["XGBoost"]
    st.warning("⚠️ TensorFlow not installed. Only XGBoost available. Install with `pip install tensorflow`")

with col1:
    selected = st.multiselect("Select Models", all_models, default=all_models)
with col2:
    seq_length = st.slider("Sequence Length (days)", 20, 120, 60)
    epochs = st.slider("Training Epochs (DL)", 10, 100, 50)
with col3:
    test_pct = st.slider("Test Split (%)", 10, 30, 20)
    feature_cols = st.multiselect("Features", [c for c in df_eng.columns if c != "Close"],
        default=[c for c in ["Open","High","Low","Volume","SMA_7","SMA_21","RSI","MACD","ATR","Daily_Return"]
                 if c in df_eng.columns])

if st.button("🚀 Train All Models", use_container_width=True):
    # Prepare data
    all_cols = ["Close"] + feature_cols
    data = df_eng[all_cols].copy()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    scaled_df = pd.DataFrame(scaled, columns=all_cols, index=data.index)

    X_seq, y_seq = prepare_sequences(scaled_df, "Close", seq_length)

    split = int(len(X_seq) * (1 - test_pct / 100))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    st.session_state["model_data"] = {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "scaler": scaler, "feature_cols": all_cols,
        "seq_length": seq_length
    }

    st.info(f"Data: Train {len(X_train):,} | Test {len(X_test):,} | Seq {seq_length} | Features {X_train.shape[2]}")

    results = {}
    trained_models = {}
    histories = {}
    predictions = {}
    progress = st.progress(0)
    status = st.empty()

    input_shape = (X_train.shape[1], X_train.shape[2])

    for i, name in enumerate(selected):
        status.info(f"🔄 Training {name}...")
        start = time.time()

        if name == "XGBoost":
            model = build_xgboost()
            model = train_xgboost(model, X_train, y_train)
            history = None
        elif name == "LSTM":
            model = build_lstm(input_shape)
            model, history = train_dl_model(model, X_train, y_train, epochs=epochs)
        elif name == "GRU":
            model = build_gru(input_shape)
            model, history = train_dl_model(model, X_train, y_train, epochs=epochs)
        elif name == "CNN":
            model = build_cnn(input_shape)
            model, history = train_dl_model(model, X_train, y_train, epochs=epochs)
        elif name == "CNN-LSTM":
            model = build_cnn_lstm(input_shape)
            model, history = train_dl_model(model, X_train, y_train, epochs=epochs)
        elif name == "Transformer":
            model = build_transformer(input_shape)
            model, history = train_dl_model(model, X_train, y_train, epochs=epochs)

        if model is None:
            st.warning(f"⚠️ {name} requires TensorFlow. Skipping.")
            progress.progress((i + 1) / len(selected))
            continue

        elapsed = time.time() - start
        preds = predict_model(model, X_test, name)

        # Inverse scale predictions
        close_idx = 0  # Close is first column
        dummy = np.zeros((len(preds), len(all_cols)))
        dummy[:, close_idx] = preds
        preds_inv = scaler.inverse_transform(dummy)[:, close_idx]

        dummy_true = np.zeros((len(y_test), len(all_cols)))
        dummy_true[:, close_idx] = y_test
        y_test_inv = scaler.inverse_transform(dummy_true)[:, close_idx]

        metrics = evaluate_predictions(y_test_inv, preds_inv, name)
        metrics["Time"] = f"{elapsed:.2f}s"
        results[name] = metrics
        trained_models[name] = model
        predictions[name] = preds_inv
        if history:
            histories[name] = history

        progress.progress((i + 1) / len(selected))

    st.session_state["trained_models"] = trained_models
    st.session_state["model_results"] = results
    st.session_state["model_predictions"] = predictions
    st.session_state["model_histories"] = histories
    st.session_state["y_test_actual"] = y_test_inv

    status.success("✅ All models trained!")

    # Results
    st.markdown("### 📊 Results")
    res_df = pd.DataFrame([{k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()}
                           for r in results.values()])
    st.dataframe(res_df, use_container_width=True, hide_index=True)

    # Predictions chart
    fig = plot_predictions(y_test_inv, predictions, f"{ticker} — Model Predictions vs Actual")
    st.plotly_chart(fig, use_container_width=True)

    # Best model
    best = min(results, key=lambda x: results[x]["RMSE"])
    st.markdown(f"### 🏆 Best Model: **{best}** (RMSE: {results[best]['RMSE']:.4f})")

    # Training histories
    if histories:
        st.markdown("### 📉 Training Curves")
        for name, hist in histories.items():
            fig_h = plot_training_history(hist, name)
            if fig_h:
                st.plotly_chart(fig_h, use_container_width=True)

elif "model_results" in st.session_state:
    st.markdown("### 📊 Previous Results")
    res_df = pd.DataFrame([{k: (f"{v:.4f}" if isinstance(v, float) else v)
        for k, v in r.items()} for r in st.session_state["model_results"].values()])
    st.dataframe(res_df, use_container_width=True, hide_index=True)

    if "y_test_actual" in st.session_state and "model_predictions" in st.session_state:
        fig = plot_predictions(st.session_state["y_test_actual"],
            st.session_state["model_predictions"], f"{ticker} — Predictions vs Actual")
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("🤖 Model Training | Stock Price Prediction System")
