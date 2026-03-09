"""
Feature engineering — Technical indicators for stock data.
"""
import pandas as pd
import numpy as np


def add_moving_averages(df, windows=[7, 14, 21, 50]):
    """Add Simple Moving Averages."""
    df = df.copy()
    for w in windows:
        df[f"SMA_{w}"] = df["Close"].rolling(window=w).mean()
    return df


def add_exponential_ma(df, windows=[12, 26]):
    """Add Exponential Moving Averages."""
    df = df.copy()
    for w in windows:
        df[f"EMA_{w}"] = df["Close"].ewm(span=w, adjust=False).mean()
    return df


def add_rsi(df, period=14):
    """Add Relative Strength Index."""
    df = df.copy()
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df, fast=12, slow=26, signal=9):
    """Add MACD, Signal, and Histogram."""
    df = df.copy()
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    return df


def add_bollinger_bands(df, window=20, num_std=2):
    """Add Bollinger Bands."""
    df = df.copy()
    sma = df["Close"].rolling(window=window).mean()
    std = df["Close"].rolling(window=window).std()
    df["BB_Upper"] = sma + (std * num_std)
    df["BB_Lower"] = sma - (std * num_std)
    df["BB_Width"] = df["BB_Upper"] - df["BB_Lower"]
    df["BB_Pct"] = (df["Close"] - df["BB_Lower"]) / df["BB_Width"].replace(0, np.nan)
    return df


def add_atr(df, period=14):
    """Add Average True Range."""
    df = df.copy()
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = true_range.rolling(window=period).mean()
    return df


def add_obv(df):
    """Add On-Balance Volume."""
    df = df.copy()
    obv = [0]
    for i in range(1, len(df)):
        if df["Close"].iloc[i] > df["Close"].iloc[i - 1]:
            obv.append(obv[-1] + df["Volume"].iloc[i])
        elif df["Close"].iloc[i] < df["Close"].iloc[i - 1]:
            obv.append(obv[-1] - df["Volume"].iloc[i])
        else:
            obv.append(obv[-1])
    df["OBV"] = obv
    return df


def add_returns(df):
    """Add daily and log returns."""
    df = df.copy()
    df["Daily_Return"] = df["Close"].pct_change()
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    return df


def add_volatility(df, window=21):
    """Add rolling volatility."""
    df = df.copy()
    df["Volatility"] = df["Daily_Return"].rolling(window=window).std() * np.sqrt(252)
    return df


def add_price_features(df):
    """Add price-based features."""
    df = df.copy()
    df["Price_Range"] = df["High"] - df["Low"]
    df["Price_Change"] = df["Close"] - df["Open"]
    df["Gap"] = df["Open"] - df["Close"].shift(1)
    return df


def add_lag_features(df, lags=[1, 2, 3, 5, 7]):
    """Add lagged close prices."""
    df = df.copy()
    for lag in lags:
        df[f"Close_Lag_{lag}"] = df["Close"].shift(lag)
    return df


def engineer_all_features(df):
    """Apply all feature engineering."""
    df = add_moving_averages(df)
    df = add_exponential_ma(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_atr(df)
    df = add_obv(df)
    df = add_returns(df)
    df = add_volatility(df)
    df = add_price_features(df)
    df = add_lag_features(df)
    return df


def prepare_sequences(data, target_col, seq_length=60):
    """Create sequences for time-series deep learning models."""
    X, y = [], []
    values = data.values
    target_idx = data.columns.get_loc(target_col) if isinstance(data, pd.DataFrame) else 0

    for i in range(seq_length, len(values)):
        X.append(values[i - seq_length:i])
        y.append(values[i, target_idx])

    return np.array(X), np.array(y)
