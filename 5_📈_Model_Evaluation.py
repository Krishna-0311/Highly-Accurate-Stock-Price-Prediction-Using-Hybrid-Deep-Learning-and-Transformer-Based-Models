"""
📈 Model Evaluation Page — MAE, RMSE, MAPE, residual analysis
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.evaluation import (plot_predictions, plot_metrics_comparison,
    plot_residuals, plot_training_history)

st.set_page_config(page_title="Model Evaluation", page_icon="📈", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    .stApp { background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 50%, #1b2838 100%); font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0a0a1a 0%, #0d1b2a 100%); border-right: 1px solid rgba(0,212,255,0.2); }
    h1,h2,h3 { color: #fff !important; }
    div[data-testid="stMetric"] { background: linear-gradient(135deg, rgba(0,212,255,0.08), rgba(233,69,96,0.05)); border: 1px solid rgba(0,212,255,0.25); border-radius: 16px; padding: 20px; }
    div[data-testid="stMetric"] label { color: #8899aa !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #00d4ff !important; font-weight: 700 !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: rgba(13,27,42,0.7); border-radius: 12px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { border-radius: 10px; color: #8899aa; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg,#00d4ff,#0099cc) !important; color: white !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 📈 Model Evaluation")
st.markdown("---")

if "model_results" not in st.session_state:
    st.warning("⚠️ Please train models first (Page 4: Model Training).")
    st.stop()

results = st.session_state["model_results"]
predictions = st.session_state.get("model_predictions", {})
y_actual = st.session_state.get("y_test_actual", None)
histories = st.session_state.get("model_histories", {})

st.success(f"✅ {len(results)} models evaluated")

# Best model metrics
best = min(results, key=lambda x: results[x]["RMSE"])
c1, c2, c3, c4 = st.columns(4)
c1.metric("🏆 Best Model", best)
c2.metric("MAE", f"{results[best]['MAE']:.4f}")
c3.metric("RMSE", f"{results[best]['RMSE']:.4f}")
c4.metric("MAPE", f"{results[best]['MAPE']:.4f}%")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Comparison", "📈 Predictions", "📉 Residuals", "🔄 Training"])

with tab1:
    st.markdown("#### Metrics Comparison")
    fig = plot_metrics_comparison(list(results.values()))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Detailed Results")
    df = pd.DataFrame(results.values())
    for col in ["MAE", "RMSE", "MAPE"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Ranking
    st.markdown("#### 🏅 Model Rankings")
    rank_df = pd.DataFrame(results.values())
    for metric in ["MAE", "RMSE", "MAPE"]:
        rank_df[f"{metric}_Rank"] = rank_df[metric].rank()
    rank_df["Avg_Rank"] = rank_df[[c for c in rank_df.columns if "Rank" in c]].mean(axis=1)
    rank_df = rank_df.sort_values("Avg_Rank")
    st.dataframe(rank_df[["Model", "MAE", "RMSE", "MAPE", "Avg_Rank"]].round(4),
        use_container_width=True, hide_index=True)

with tab2:
    if y_actual is not None and predictions:
        st.markdown("#### Actual vs Predicted Prices")
        fig = plot_predictions(y_actual, predictions)
        st.plotly_chart(fig, use_container_width=True)

        # Individual model zoom
        sel = st.selectbox("Zoom into model", list(predictions.keys()))
        fig2 = plot_predictions(y_actual, {sel: predictions[sel]}, f"{sel} — Detailed View")
        st.plotly_chart(fig2, use_container_width=True)

        # Prediction error over time
        errors = np.abs(y_actual - predictions[sel])
        fig3 = go.Figure(go.Scatter(y=errors, mode="lines",
            line=dict(color="#e94560"), fill="tozeroy", fillcolor="rgba(233,69,96,0.1)"))
        fig3.update_layout(title=f"Absolute Error Over Time — {sel}", template="plotly_dark",
            height=350, yaxis_title="Absolute Error ($)")
        st.plotly_chart(fig3, use_container_width=True)

with tab3:
    if y_actual is not None and predictions:
        st.markdown("#### Residual Analysis")
        for name, preds in predictions.items():
            fig = plot_residuals(y_actual, preds, name)
            st.plotly_chart(fig, use_container_width=True)

with tab4:
    if histories:
        st.markdown("#### Training Loss Curves")
        for name, hist in histories.items():
            fig = plot_training_history(hist, name)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No deep learning training histories available.")

st.markdown("---")
st.caption("📈 Model Evaluation | Stock Price Prediction System")
