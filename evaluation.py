"""
Model evaluation utilities — MAE, RMSE, MAPE, and Plotly charts.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate MAPE."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_predictions(y_true, y_pred, model_name="Model"):
    """Compute all evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {"Model": model_name, "MAE": mae, "RMSE": rmse, "MAPE": mape}


def plot_predictions(y_true, predictions_dict, title="Stock Price Predictions"):
    """Plot actual vs predicted for all models."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=y_true, mode="lines", name="Actual",
        line=dict(color="#00d4ff", width=2.5)
    ))

    colors = ["#e94560", "#ffd700", "#ff6b6b", "#0f3460", "#00ff80", "#ff69b4"]
    for i, (name, preds) in enumerate(predictions_dict.items()):
        fig.add_trace(go.Scatter(
            y=preds, mode="lines", name=name,
            line=dict(color=colors[i % len(colors)], width=1.8, dash="dot")
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color="white")),
        template="plotly_dark", height=550,
        xaxis_title="Time Steps", yaxis_title="Price ($)",
        legend=dict(bgcolor="rgba(0,0,0,0.5)"),
        font=dict(color="#e0e0e0")
    )
    return fig


def plot_metrics_comparison(results_list):
    """Bar chart comparing all model metrics."""
    df = pd.DataFrame(results_list)
    metrics = ["MAE", "RMSE", "MAPE"]
    colors = ["#00d4ff", "#e94560", "#ffd700"]

    fig = make_subplots(rows=1, cols=3, subplot_titles=metrics)

    for i, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
            x=df["Model"], y=df[metric],
            marker_color=colors[i],
            text=[f"{v:.4f}" for v in df[metric]],
            textposition="outside", name=metric,
            showlegend=False
        ), row=1, col=i + 1)

    fig.update_layout(
        title=dict(text="Model Performance Comparison", font=dict(size=20)),
        template="plotly_dark", height=450,
        font=dict(color="#e0e0e0")
    )
    return fig


def plot_residuals(y_true, y_pred, model_name="Model"):
    """Plot residual distribution."""
    residuals = np.array(y_true) - np.array(y_pred)

    fig = make_subplots(rows=1, cols=2,
        subplot_titles=["Residuals Over Time", "Residual Distribution"])

    fig.add_trace(go.Scatter(
        y=residuals, mode="markers",
        marker=dict(color="#e94560", size=4, opacity=0.6),
        name="Residuals"
    ), row=1, col=1)

    fig.add_trace(go.Histogram(
        x=residuals, nbinsx=50,
        marker_color="#00d4ff", opacity=0.8,
        name="Distribution"
    ), row=1, col=2)

    fig.update_layout(
        title=dict(text=f"Residual Analysis — {model_name}", font=dict(size=18)),
        template="plotly_dark", height=400,
        showlegend=False
    )
    return fig


def plot_training_history(history, model_name="Model"):
    """Plot training/validation loss curves."""
    if history is None:
        return None
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history.history["loss"], mode="lines",
        name="Train Loss", line=dict(color="#e94560", width=2)))
    if "val_loss" in history.history:
        fig.add_trace(go.Scatter(y=history.history["val_loss"], mode="lines",
            name="Val Loss", line=dict(color="#00d4ff", width=2)))

    fig.update_layout(
        title=dict(text=f"Training History — {model_name}", font=dict(size=18)),
        template="plotly_dark", height=400,
        xaxis_title="Epoch", yaxis_title="Loss (MSE)",
        font=dict(color="#e0e0e0")
    )
    return fig


def plot_candlestick(df, title="Stock Price"):
    """Plot candlestick chart."""
    fig = go.Figure(data=[go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color="#00d4ff",
        decreasing_line_color="#e94560"
    )])
    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        template="plotly_dark", height=500,
        xaxis_title="Date", yaxis_title="Price ($)",
        xaxis_rangeslider_visible=False,
        font=dict(color="#e0e0e0")
    )
    return fig
