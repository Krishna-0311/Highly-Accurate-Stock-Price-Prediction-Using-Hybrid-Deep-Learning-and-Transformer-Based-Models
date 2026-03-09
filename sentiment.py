"""
Sentiment analysis utilities — FinBERT integration with fallback.
"""
import pandas as pd
import numpy as np
import streamlit as st


def analyze_sentiment_finbert(headlines):
    """Analyse sentiment using FinBERT. Falls back to rule-based if unavailable."""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

        results = []
        batch_size = 16

        for i in range(0, len(headlines), batch_size):
            batch = headlines[i:i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True,
                               max_length=512, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            labels = ["positive", "negative", "neutral"]

            for j in range(len(batch)):
                pred_idx = torch.argmax(probs[j]).item()
                score = probs[j][0].item() - probs[j][1].item()  # positive - negative
                results.append({
                    "headline": batch[j],
                    "sentiment": labels[pred_idx],
                    "positive": round(probs[j][0].item(), 4),
                    "negative": round(probs[j][1].item(), 4),
                    "neutral": round(probs[j][2].item(), 4),
                    "score": round(score, 4)
                })

        return pd.DataFrame(results), "finbert"

    except Exception:
        return rule_based_sentiment(headlines), "rule_based"


def rule_based_sentiment(headlines):
    """Simple rule-based sentiment fallback."""
    positive_words = {
        "surge", "rally", "gain", "rise", "jump", "high", "record", "beat",
        "exceed", "upgrade", "strong", "growth", "optimistic", "profit",
        "bullish", "increase", "expand", "innovation", "confidence", "dividend"
    }
    negative_words = {
        "decline", "fall", "drop", "low", "miss", "cut", "loss", "downturn",
        "downgrade", "weak", "concern", "risk", "bearish", "crash", "sell",
        "restructuring", "layoff", "warning", "slowdown", "pressure", "probe"
    }

    results = []
    for h in headlines:
        words = set(h.lower().split())
        pos_count = len(words & positive_words)
        neg_count = len(words & negative_words)

        if pos_count > neg_count:
            sentiment = "positive"
            score = round(min(0.5 + pos_count * 0.15, 0.95), 4)
        elif neg_count > pos_count:
            sentiment = "negative"
            score = round(max(-0.5 - neg_count * 0.15, -0.95), 4)
        else:
            sentiment = "neutral"
            score = 0.0

        results.append({
            "headline": h,
            "sentiment": sentiment,
            "positive": max(score, 0),
            "negative": abs(min(score, 0)),
            "neutral": 1.0 - abs(score),
            "score": score
        })

    return pd.DataFrame(results)


def aggregate_daily_sentiment(sentiment_df, date_col="Date"):
    """Aggregate sentiment scores by date."""
    if date_col not in sentiment_df.columns:
        return sentiment_df

    daily = sentiment_df.groupby(date_col).agg(
        avg_score=("score", "mean"),
        pos_count=("sentiment", lambda x: (x == "positive").sum()),
        neg_count=("sentiment", lambda x: (x == "negative").sum()),
        neu_count=("sentiment", lambda x: (x == "neutral").sum()),
        total_news=("sentiment", "count")
    ).reset_index()

    daily["sentiment_ratio"] = daily["pos_count"] / daily["total_news"].replace(0, 1)
    return daily
