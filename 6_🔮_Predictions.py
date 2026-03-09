"""
🔮 Predictions & Forecasting Page
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.data_loader import download_stock_data
from utils.feature_engineering import engineer_all_features
from utils.models import predict_model

st.set_page_config(page_title="Predictions", page_icon="🔮", layout="wide")

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

st.markdown("# 🔮 Stock Price Prediction & Forecasting")
st.markdown("---")

if "trained_models" not in st.session_state or "model_data" not in st.session_state:
    st.warning("⚠️ Please train models first on Page 4.")
    st.stop()

models = st.session_state["trained_models"]
model_data = st.session_state["model_data"]
results = st.session_state.get("model_results", {})
predictions = st.session_state.get("model_predictions", {})
y_actual = st.session_state.get("y_test_actual", None)
ticker = st.session_state.get("stock_ticker", "AAPL")
df_eng = st.session_state.get("engineered_data", None)

tab1, tab2 = st.tabs(["📈 Test Set Predictions", "🔮 Future Forecast"])

with tab1:
    st.markdown("### Actual vs Predicted Comparison")

    sel_model = st.selectbox("Select Model", list(models.keys()))

    if sel_model in predictions and y_actual is not None:
        preds = predictions[sel_model]
        c1, c2, c3 = st.columns(3)
        c1.metric("Latest Actual", f"${y_actual[-1]:.2f}")
        c2.metric("Latest Predicted", f"${preds[-1]:.2f}")
        diff = preds[-1] - y_actual[-1]
        c3.metric("Difference", f"${diff:.2f}", f"{(diff/y_actual[-1])*100:.2f}%")

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y_actual, mode="lines", name="Actual",
            line=dict(color="#00d4ff", width=2.5)))
        fig.add_trace(go.Scatter(y=preds, mode="lines", name=sel_model,
            line=dict(color="#e94560", width=2, dash="dot")))
        fig.update_layout(title=f"{ticker} — {sel_model} Predictions", template="plotly_dark",
            height=500, xaxis_title="Time Steps", yaxis_title="Price ($)")
        st.plotly_chart(fig, use_container_width=True)

        # Error distribution
        errors = preds - y_actual
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=errors, nbinsx=50, marker_color="#e94560", opacity=0.8))
        fig2.update_layout(title="Prediction Error Distribution", template="plotly_dark",
            height=350, xaxis_title="Error ($)", yaxis_title="Frequency")
        st.plotly_chart(fig2, use_container_width=True)

        # Results table
        if sel_model in results:
            r = results[sel_model]
            st.markdown(f"""
            | Metric | Value |
            |--------|-------|
            | MAE | {r['MAE']:.4f} |
            | RMSE | {r['RMSE']:.4f} |
            | MAPE | {r['MAPE']:.4f}% |
            """)

with tab2:
    st.markdown("### Multi-Step Forecast")

    forecast_days = st.slider("Forecast horizon (days)", 5, 30, 10)
    forecast_model = st.selectbox("Forecast Model", list(models.keys()), key="forecast_sel")

    if st.button("🔮 Generate Forecast", use_container_width=True):
        with st.spinner("Generating forecast..."):
            scaler = model_data["scaler"]
            feature_cols = model_data["feature_cols"]
            seq_len = model_data["seq_length"]
            model = models[forecast_model]

            # Use last sequence from test data
            last_seq = model_data["X_test"][-1].copy()
            forecast_scaled = []

            for _ in range(forecast_days):
                inp = last_seq.reshape(1, *last_seq.shape)
                pred = predict_model(model, inp, forecast_model)
                if pred is None:
                    break
                forecast_scaled.append(pred[0])

                # Shift window
                new_row = last_seq[-1].copy()
                new_row[0] = pred[0]  # Update Close
                last_seq = np.vstack([last_seq[1:], new_row])

            # Inverse scale
            dummy = np.zeros((len(forecast_scaled), len(feature_cols)))
            dummy[:, 0] = forecast_scaled
            forecast_prices = scaler.inverse_transform(dummy)[:, 0]

            # Display
            last_actual = y_actual[-1] if y_actual is not None else 0
            c1, c2, c3 = st.columns(3)
            c1.metric("Current Price", f"${last_actual:.2f}")
            c2.metric(f"{forecast_days}-Day Forecast", f"${forecast_prices[-1]:.2f}")
            change = forecast_prices[-1] - last_actual
            c3.metric("Expected Change", f"${change:.2f}", f"{(change/max(last_actual,1))*100:.2f}%")

            # Forecast chart
            fig = go.Figure()
            if y_actual is not None:
                fig.add_trace(go.Scatter(y=y_actual[-60:], mode="lines", name="Historical (Test)",
                    line=dict(color="#00d4ff", width=2)))

            fig.add_trace(go.Scatter(
                x=list(range(len(y_actual[-60:]) if y_actual is not None else 0,
                             (len(y_actual[-60:]) if y_actual is not None else 0) + forecast_days)),
                y=forecast_prices, mode="lines+markers", name="Forecast",
                line=dict(color="#ffd700", width=2.5, dash="dot"),
                marker=dict(size=8, color="#ffd700")
            ))

            fig.update_layout(title=f"{ticker} — {forecast_days}-Day Forecast ({forecast_model})",
                template="plotly_dark", height=500,
                xaxis_title="Time Steps", yaxis_title="Price ($)")
            st.plotly_chart(fig, use_container_width=True)

            # Forecast table
            forecast_df = pd.DataFrame({
                "Day": range(1, forecast_days + 1),
                "Predicted Price": [f"${p:.2f}" for p in forecast_prices],
                "Change from Current": [f"${p - last_actual:.2f}" for p in forecast_prices],
                "% Change": [f"{((p - last_actual)/max(last_actual,1))*100:.2f}%" for p in forecast_prices]
            })
            st.dataframe(forecast_df, use_container_width=True, hide_index=True)

            signal = "📈 BULLISH" if forecast_prices[-1] > last_actual else "📉 BEARISH"
            if abs(change / max(last_actual, 1)) < 0.01:
                signal = "➡️ NEUTRAL"
            st.markdown(f"### Trading Signal: **{signal}**")

st.markdown("---")
st.caption("🔮 Predictions | Stock Price Prediction System")
