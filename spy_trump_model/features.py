from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


KEYWORDS = [
    "tariff",
    "china",
    "fed",
    "rate",
    "inflation",
    "oil",
    "war",
    "tax",
    "regulation",
    "border",
    "immigration",
    "dollar",
    "jobs",
]


def load_spy(spy_path: str | Path) -> pd.DataFrame:
    spy = pd.read_csv(spy_path, parse_dates=["Date"])
    spy = spy.rename(columns={c: c.lower() for c in spy.columns})
    if "date" not in spy.columns or "close" not in spy.columns:
        raise ValueError("SPY file must include Date and Close columns.")

    spy = spy.sort_values("date").copy()
    spy["date"] = spy["date"].dt.normalize()
    spy["return_1d"] = spy["close"].pct_change()
    spy["target_next_up"] = (spy["close"].shift(-1) > spy["close"]).astype(int)
    spy["target_next_return"] = spy["close"].shift(-1) / spy["close"] - 1
    return spy


def load_speeches(speeches_path: str | Path) -> pd.DataFrame:
    path = Path(speeches_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing speeches file: {path}. Run fetch-whitehouse or create the CSV manually."
        )

    speeches = pd.read_csv(path)
    missing = {"date", "text"} - set(speeches.columns)
    if missing:
        raise ValueError(f"Speeches file is missing columns: {sorted(missing)}")

    speeches["date"] = pd.to_datetime(speeches["date"], errors="coerce").dt.normalize()
    speeches["text"] = speeches["text"].fillna("").astype(str)
    speeches = speeches.dropna(subset=["date"])
    speeches = speeches[speeches["text"].str.len() > 0]
    return speeches


def _keyword_count(text: str, keyword: str) -> int:
    return len(re.findall(rf"\b{re.escape(keyword)}\b", text.lower()))


def build_daily_speech_features(speeches: pd.DataFrame) -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer()
    rows = []

    for _, row in speeches.iterrows():
        text = row["text"]
        sentiment = analyzer.polarity_scores(text)
        item = {
            "date": row["date"],
            "speech_count": 1,
            "word_count": len(text.split()),
            "char_count": len(text),
            "sentiment_compound": sentiment["compound"],
            "sentiment_pos": sentiment["pos"],
            "sentiment_neg": sentiment["neg"],
            "sentiment_neu": sentiment["neu"],
        }
        for keyword in KEYWORDS:
            item[f"kw_{keyword}"] = _keyword_count(text, keyword)
        rows.append(item)

    if not rows:
        return pd.DataFrame(columns=["date"])

    features = pd.DataFrame(rows)
    agg_map = {
        "speech_count": "sum",
        "word_count": "sum",
        "char_count": "sum",
        "sentiment_compound": "mean",
        "sentiment_pos": "mean",
        "sentiment_neg": "mean",
        "sentiment_neu": "mean",
    }
    agg_map.update({f"kw_{keyword}": "sum" for keyword in KEYWORDS})
    return features.groupby("date", as_index=False).agg(agg_map)


def build_dataset(spy_path: str | Path, speeches_path: str | Path) -> pd.DataFrame:
    spy = load_spy(spy_path)
    speeches = load_speeches(speeches_path)
    speech_features = build_daily_speech_features(speeches)

    data = spy.merge(speech_features, on="date", how="left")
    feature_cols = [c for c in data.columns if c.startswith(("speech_", "word_", "char_", "sentiment_", "kw_"))]
    data[feature_cols] = data[feature_cols].fillna(0)

    data["spy_return_1d_lag1"] = data["return_1d"].shift(1)
    data["spy_return_5d_lag1"] = data["close"].pct_change(5).shift(1)
    data["spy_vol_20d_lag1"] = data["return_1d"].rolling(20).std().shift(1)

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["target_next_up", "target_next_return", "spy_return_1d_lag1", "spy_return_5d_lag1", "spy_vol_20d_lag1"])
    return data


def feature_columns(data: pd.DataFrame) -> list[str]:
    prefixes = (
        "speech_",
        "word_",
        "char_",
        "sentiment_",
        "kw_",
        "spy_return_",
        "spy_vol_",
    )
    return [col for col in data.columns if col.startswith(prefixes)]

