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

    speeches["date"] = pd.to_datetime(
        speeches["date"].astype(str).str.slice(0, 10),
        errors="coerce",
    ).dt.normalize()
    if "datetime" in speeches.columns:
        speeches["datetime_et"] = pd.to_datetime(
            speeches["datetime"], errors="coerce", format="mixed", utc=True
        ).dt.tz_convert("America/New_York")
    else:
        speeches["datetime_et"] = pd.NaT
    speeches["text"] = speeches["text"].fillna("").astype(str)
    speeches = speeches.dropna(subset=["date"])
    speeches = speeches[speeches["text"].str.len() > 0]
    if speeches.empty:
        raise ValueError(
            "Speeches file has no usable rows. Fetch or import Trump remarks before training."
        )
    return speeches


def _signal_date(row: pd.Series) -> pd.Timestamp:
    dt = row.get("datetime_et")
    if pd.isna(dt):
        return row["date"]

    if dt.weekday() >= 5:
        days_until_monday = 7 - dt.weekday()
        return pd.Timestamp(dt.date()) + pd.Timedelta(days=days_until_monday)

    if dt.hour >= 16:
        return pd.Timestamp(dt.date()) + pd.Timedelta(days=1)

    return pd.Timestamp(dt.date())


def _market_session(row: pd.Series) -> str:
    dt = row.get("datetime_et")
    if pd.isna(dt):
        return "unknown"
    if dt.weekday() >= 5:
        return "weekend"
    if dt.hour < 9 or (dt.hour == 9 and dt.minute < 30):
        return "premarket"
    if dt.hour < 16:
        return "market"
    return "afterhours"


def _keyword_count(text: str, keyword: str) -> int:
    return len(re.findall(rf"\b{re.escape(keyword)}\b", text.lower()))


def build_daily_speech_features(speeches: pd.DataFrame) -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer()
    speeches = speeches.copy()
    speeches["signal_date"] = speeches.apply(_signal_date, axis=1)
    speeches["signal_date"] = pd.to_datetime(speeches["signal_date"]).dt.normalize()
    speeches["market_session"] = speeches.apply(_market_session, axis=1)
    rows = []

    for _, row in speeches.iterrows():
        text = row["text"]
        sentiment = analyzer.polarity_scores(text)
        item = {
            "date": row["signal_date"],
            "speech_count": 1,
            "speech_premarket_count": 1 if row["market_session"] == "premarket" else 0,
            "speech_market_count": 1 if row["market_session"] == "market" else 0,
            "speech_afterhours_count": 1 if row["market_session"] == "afterhours" else 0,
            "speech_weekend_count": 1 if row["market_session"] == "weekend" else 0,
            "speech_unknown_time_count": 1 if row["market_session"] == "unknown" else 0,
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
        "speech_premarket_count": "sum",
        "speech_market_count": "sum",
        "speech_afterhours_count": "sum",
        "speech_weekend_count": "sum",
        "speech_unknown_time_count": "sum",
        "word_count": "sum",
        "char_count": "sum",
        "sentiment_compound": "mean",
        "sentiment_pos": "mean",
        "sentiment_neg": "mean",
        "sentiment_neu": "mean",
    }
    agg_map.update({f"kw_{keyword}": "sum" for keyword in KEYWORDS})
    return features.groupby("date", as_index=False).agg(agg_map)


def align_features_to_trading_days(features: pd.DataFrame, spy_dates: pd.Series) -> pd.DataFrame:
    if features.empty:
        return features

    feature_dates = features.sort_values("date").copy()
    feature_dates["date"] = pd.to_datetime(feature_dates["date"]).astype("datetime64[ns]")
    trading_days = pd.DataFrame(
        {
            "trading_date": pd.to_datetime(pd.Series(spy_dates).sort_values().unique()).astype(
                "datetime64[ns]"
            )
        }
    )
    aligned = pd.merge_asof(
        feature_dates,
        trading_days,
        left_on="date",
        right_on="trading_date",
        direction="forward",
    )
    aligned = aligned.dropna(subset=["trading_date"]).drop(columns=["date"])
    aligned = aligned.rename(columns={"trading_date": "date"})

    agg_map = {col: "sum" for col in aligned.columns if col != "date"}
    for col in ["sentiment_compound", "sentiment_pos", "sentiment_neg", "sentiment_neu"]:
        if col in agg_map:
            agg_map[col] = "mean"

    return aligned.groupby("date", as_index=False).agg(agg_map)


def build_dataset(spy_path: str | Path, speeches_path: str | Path) -> pd.DataFrame:
    spy = load_spy(spy_path)
    speeches = load_speeches(speeches_path)
    speech_features = align_features_to_trading_days(
        build_daily_speech_features(speeches),
        spy["date"],
    )

    data = spy.merge(speech_features, on="date", how="left")
    feature_cols = [c for c in data.columns if c.startswith(("speech_", "word_", "char_", "sentiment_", "kw_"))]
    data[feature_cols] = data[feature_cols].fillna(0)

    for col in ["speech_count", "word_count", "char_count"]:
        if col in data.columns:
            data[f"log1p_{col}"] = np.log1p(data[col])

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
        "log1p_",
        "spy_return_",
        "spy_vol_",
    )
    return [col for col in data.columns if col.startswith(prefixes)]
