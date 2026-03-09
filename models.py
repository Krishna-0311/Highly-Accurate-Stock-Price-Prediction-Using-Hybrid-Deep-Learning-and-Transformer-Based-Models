"""
ML and DL models for stock price prediction.
XGBoost, CNN, LSTM, GRU, Transformer, CNN-LSTM hybrid.
"""
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")


# ─── Check TensorFlow availability ───
TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (
        Dense, LSTM, GRU, Conv1D, MaxPooling1D, Flatten,
        Dropout, BatchNormalization, Input, MultiHeadAttention,
        GlobalAveragePooling1D, LayerNormalization, Add
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    tf.random.set_seed(42)
    TF_AVAILABLE = True
except ImportError:
    pass


def build_xgboost(n_estimators=300, max_depth=6, learning_rate=0.05):
    """Build XGBoost regressor."""
    return XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )


def build_lstm(input_shape, units=128):
    """Build LSTM model."""
    if not TF_AVAILABLE:
        return None
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        BatchNormalization(),
        LSTM(units // 2, return_sequences=False),
        Dropout(0.2),
        BatchNormalization(),
        Dense(64, activation="relu"),
        Dropout(0.1),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return model


def build_gru(input_shape, units=128):
    """Build GRU model."""
    if not TF_AVAILABLE:
        return None
    model = Sequential([
        GRU(units, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        BatchNormalization(),
        GRU(units // 2, return_sequences=False),
        Dropout(0.2),
        BatchNormalization(),
        Dense(64, activation="relu"),
        Dropout(0.1),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return model


def build_cnn(input_shape, filters=64):
    """Build CNN model for time-series."""
    if not TF_AVAILABLE:
        return None
    model = Sequential([
        Conv1D(filters, kernel_size=3, activation="relu", input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Conv1D(filters * 2, kernel_size=3, activation="relu"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return model


def build_cnn_lstm(input_shape, filters=64, lstm_units=64):
    """Build hybrid CNN-LSTM model."""
    if not TF_AVAILABLE:
        return None
    model = Sequential([
        Conv1D(filters, kernel_size=3, activation="relu", input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Conv1D(filters * 2, kernel_size=3, activation="relu"),
        BatchNormalization(),
        LSTM(lstm_units, return_sequences=False),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return model


def build_transformer(input_shape, head_size=64, num_heads=4, ff_dim=128, num_blocks=2):
    """Build Transformer encoder model for time-series."""
    if not TF_AVAILABLE:
        return None

    inputs = Input(shape=input_shape)
    x = Dense(head_size)(inputs)

    for _ in range(num_blocks):
        # Multi-head self-attention
        attn_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=head_size // num_heads
        )(x, x)
        attn_output = Dropout(0.1)(attn_output)
        x = Add()([x, attn_output])
        x = LayerNormalization(epsilon=1e-6)(x)

        # Feed-forward
        ff_output = Dense(ff_dim, activation="relu")(x)
        ff_output = Dropout(0.1)(ff_output)
        ff_output = Dense(head_size)(ff_output)
        x = Add()([x, ff_output])
        x = LayerNormalization(epsilon=1e-6)(x)

    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return model


def get_callbacks():
    """Return standard callbacks for DL training."""
    if not TF_AVAILABLE:
        return []
    return [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)
    ]


def train_dl_model(model, X_train, y_train, epochs=50, batch_size=32):
    """Train a deep learning model."""
    if model is None:
        return None, None
    callbacks = get_callbacks()
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=0
    )
    return model, history


def train_xgboost(model, X_train, y_train):
    """Train XGBoost on flattened sequences."""
    if len(X_train.shape) == 3:
        X_flat = X_train.reshape(X_train.shape[0], -1)
    else:
        X_flat = X_train
    model.fit(X_flat, y_train)
    return model


def predict_model(model, X_test, model_name="model"):
    """Generate predictions."""
    if model is None:
        return None
    if model_name == "XGBoost":
        if len(X_test.shape) == 3:
            X_flat = X_test.reshape(X_test.shape[0], -1)
        else:
            X_flat = X_test
        return model.predict(X_flat)
    else:
        return model.predict(X_test, verbose=0).flatten()
