from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .data import download_spy, ensure_parent


THEME_TERMS = {
    "trade_china": [
        "china",
        "chinese",
        "tariff",
        "tariffs",
        "trade",
        "import",
        "imports",
        "export",
        "exports",
        "manufacturing",
        "supply chain",
    ],
    "rates_inflation": [
        "inflation",
        "prices",
        "fed",
        "federal reserve",
        "rate",
        "rates",
        "powell",
        "treasury yield",
        "yield",
    ],
    "energy_oil": [
        "oil",
        "gas",
        "gasoline",
        "energy",
        "drilling",
        "opec",
        "pipeline",
        "crude",
    ],
    "immigration_border": [
        "border",
        "immigration",
        "migrant",
        "migrants",
        "deportation",
        "asylum",
    ],
    "war_geopolitics": [
        "war",
        "ukraine",
        "russia",
        "iran",
        "israel",
        "gaza",
        "nato",
        "missile",
        "military",
        "sanctions",
    ],
    "tax_regulation": [
        "tax",
        "taxes",
        "irs",
        "regulation",
        "regulations",
        "deregulation",
        "antitrust",
    ],
    "dollar_fx": [
        "dollar",
        "currency",
        "fx",
        "yen",
        "yuan",
        "euro",
        "exchange rate",
    ],
    "risk_off_news": [
        "crisis",
        "panic",
        "selloff",
        "recession",
        "default",
        "shutdown",
        "emergency",
        "uncertainty",
    ],
}

DEFAULT_MARKET_TICKERS = ["SPY", "QQQ", "GLD", "TLT", "USO", "UUP", "^VIX", "^TNX"]
DEFAULT_TEXT_VECTOR_FEATURES = 128
MARKET_TICKER_ALIASES = {
    "SPY": "spy",
    "QQQ": "qqq",
    "GLD": "gld",
    "TLT": "tlt",
    "USO": "uso",
    "UUP": "uup",
    "^VIX": "vix",
    "^TNX": "tnx",
}


def _safe_ticker_name(ticker: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", ticker.strip().upper())
    return safe.strip("_") or "ticker"


def _theme_count(text: str, terms: list[str]) -> int:
    lowered = text.lower()
    total = 0
    for term in terms:
        pattern = rf"\b{re.escape(term.lower())}\b"
        total += len(re.findall(pattern, lowered))
    return total


def _source_name(path: str | Path) -> str:
    return Path(path).stem.lower().replace("-", "_")


def _load_text_source(path: str | Path) -> pd.DataFrame:
    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"Missing text source: {source_path}")

    data = pd.read_csv(source_path)
    missing = {"date", "text"} - set(data.columns)
    if missing:
        raise ValueError(f"{source_path} is missing columns: {sorted(missing)}")

    loaded = data.copy()
    loaded["date"] = pd.to_datetime(
        loaded["date"].astype(str).str.slice(0, 10),
        errors="coerce",
    ).dt.normalize()
    if "datetime" in loaded.columns:
        loaded["datetime"] = pd.to_datetime(
            loaded["datetime"],
            errors="coerce",
            format="mixed",
            utc=True,
        )
    else:
        loaded["datetime"] = pd.NaT
    if "title" not in loaded.columns:
        loaded["title"] = ""
    if "source" not in loaded.columns:
        loaded["source"] = str(source_path)
    if "source_type" not in loaded.columns:
        loaded["source_type"] = _source_name(source_path)

    loaded["title"] = loaded["title"].fillna("").astype(str)
    loaded["text"] = loaded["text"].fillna("").astype(str)
    loaded["source"] = loaded["source"].fillna(str(source_path)).astype(str)
    loaded["source_type"] = loaded["source_type"].fillna(_source_name(source_path)).astype(str)
    loaded["combined_text"] = (loaded["title"] + "\n" + loaded["text"]).str.strip()
    loaded = loaded.dropna(subset=["date"])
    loaded = loaded[loaded["combined_text"].str.len() > 0]
    return loaded[["date", "datetime", "title", "source", "source_type", "combined_text"]]


def load_text_sources(paths: list[str | Path]) -> pd.DataFrame:
    frames = [_load_text_source(path) for path in paths]
    if not frames:
        return pd.DataFrame(
            columns=["date", "datetime", "title", "source", "source_type", "combined_text"]
        )
    combined = pd.concat(frames, ignore_index=True)
    combined["source_type"] = combined["source_type"].str.lower().str.replace(r"\W+", "_", regex=True)
    return combined.sort_values(["date", "datetime", "source_type"]).reset_index(drop=True)


def _daily_text_signals(texts: pd.DataFrame) -> pd.DataFrame:
    if texts.empty:
        return pd.DataFrame(columns=["date"])

    analyzer = SentimentIntensityAnalyzer()
    rows: list[dict[str, object]] = []
    for _, row in texts.iterrows():
        text = row["combined_text"]
        sentiment = analyzer.polarity_scores(text)
        words = max(len(text.split()), 1)
        item: dict[str, object] = {
            "date": row["date"],
            "text_item_count": 1,
            "word_count": words,
            "char_count": len(text),
            "sentiment_compound": sentiment["compound"],
            "sentiment_pos": sentiment["pos"],
            "sentiment_neg": sentiment["neg"],
            "sentiment_neu": sentiment["neu"],
        }
        source_type = str(row["source_type"])
        item[f"source_{source_type}_count"] = 1
        for theme, terms in THEME_TERMS.items():
            mentions = _theme_count(text, terms)
            item[f"theme_{theme}_mentions"] = mentions
            item[f"theme_{theme}_doc_count"] = 1 if mentions else 0
            item[f"theme_{theme}_pos_weighted"] = mentions * sentiment["pos"]
            item[f"theme_{theme}_neg_weighted"] = mentions * sentiment["neg"]
            item[f"theme_{theme}_sentiment_weighted"] = mentions * sentiment["compound"]
        rows.append(item)

    item_features = pd.DataFrame(rows).fillna(0)
    sum_cols = [
        col
        for col in item_features.columns
        if col != "date" and not col.startswith("sentiment_")
    ]
    mean_cols = [col for col in item_features.columns if col.startswith("sentiment_")]
    daily_sum = item_features.groupby("date", as_index=False)[sum_cols].sum()
    daily_mean = item_features.groupby("date", as_index=False)[mean_cols].mean()
    daily = daily_sum.merge(daily_mean, on="date", how="left")

    for theme in THEME_TERMS:
        mentions_col = f"theme_{theme}_mentions"
        intensity_col = f"theme_{theme}_intensity"
        daily[intensity_col] = daily[mentions_col] / daily["word_count"].clip(lower=1) * 1000
        daily[f"theme_{theme}_log_intensity"] = np.log1p(daily[mentions_col])
        daily[f"theme_{theme}_pos_intensity"] = (
            daily[f"theme_{theme}_pos_weighted"] / daily["word_count"].clip(lower=1) * 1000
        )
        daily[f"theme_{theme}_neg_intensity"] = (
            daily[f"theme_{theme}_neg_weighted"] / daily["word_count"].clip(lower=1) * 1000
        )
        weighted = daily[f"theme_{theme}_sentiment_weighted"]
        daily[f"theme_{theme}_sentiment"] = weighted / daily[mentions_col].replace(0, np.nan)
        rolling_mean = daily[intensity_col].rolling(60, min_periods=20).mean().shift(1)
        rolling_std = daily[intensity_col].rolling(60, min_periods=20).std().shift(1)
        daily[f"theme_{theme}_surprise_60d"] = (
            (daily[intensity_col] - rolling_mean) / rolling_std.replace(0, np.nan)
        )

    daily = daily.replace([np.inf, -np.inf], np.nan)
    return daily.sort_values("date").reset_index(drop=True)


def _daily_whole_text_vectors(texts: pd.DataFrame, n_features: int) -> pd.DataFrame:
    if texts.empty:
        return pd.DataFrame(columns=["date"])
    if n_features <= 0:
        raise ValueError("text vector feature count must be positive.")

    daily_text = (
        texts.groupby("date", as_index=False)["combined_text"]
        .agg(lambda values: "\n\n".join(str(value) for value in values if str(value).strip()))
        .sort_values("date")
        .reset_index(drop=True)
    )
    vectorizer = HashingVectorizer(
        n_features=int(n_features),
        analyzer="word",
        ngram_range=(1, 2),
        lowercase=True,
        stop_words="english",
        norm="l2",
        alternate_sign=False,
    )
    matrix = vectorizer.transform(daily_text["combined_text"].fillna(""))
    vector_cols = [f"whole_text_vec_{index:03d}" for index in range(int(n_features))]
    vectors = pd.DataFrame(matrix.toarray(), columns=vector_cols)
    vectors.insert(0, "date", daily_text["date"])
    return vectors


def _read_market_cache(
    ticker: str,
    data_dir: str | Path,
    start: str,
    update: bool,
) -> pd.DataFrame:
    cache_path = Path(data_dir) / f"{_safe_ticker_name(ticker)}.csv"
    raw = download_spy(ticker=ticker, start=start, cache_path=cache_path, update=update)
    market = raw.rename(columns={col: col.lower() for col in raw.columns}).copy()
    if "date" not in market.columns or "close" not in market.columns:
        raise ValueError(f"Market data for {ticker} must include Date and Close.")
    market["date"] = pd.to_datetime(market["date"]).dt.normalize()
    if start:
        market = market[market["date"] >= pd.Timestamp(start)].copy()
    return market.sort_values("date")[["date", "close"]].dropna()


def _market_alias(ticker: str) -> str:
    return MARKET_TICKER_ALIASES.get(ticker.upper(), _safe_ticker_name(ticker).lower())


def build_market_state(
    tickers: list[str],
    data_dir: str | Path,
    start: str,
    update: bool = False,
) -> pd.DataFrame:
    frames = []
    for ticker in tickers:
        alias = _market_alias(ticker)
        market = _read_market_cache(ticker, data_dir=data_dir, start=start, update=update)
        market = market.rename(columns={"close": f"mkt_{alias}_close"})
        market[f"mkt_{alias}_return_1d"] = market[f"mkt_{alias}_close"].pct_change()
        market[f"mkt_{alias}_return_5d"] = market[f"mkt_{alias}_close"].pct_change(5)
        market[f"mkt_{alias}_trend_20d"] = market[f"mkt_{alias}_close"].pct_change(20)
        market[f"mkt_{alias}_vol_20d"] = market[f"mkt_{alias}_return_1d"].rolling(20).std()
        frames.append(market)

    if not frames:
        return pd.DataFrame(columns=["date"])

    combined = frames[0]
    for frame in frames[1:]:
        combined = combined.merge(frame, on="date", how="outer")
    combined = combined.sort_values("date").reset_index(drop=True)

    if "mkt_spy_return_1d" in combined.columns and "mkt_vix_return_1d" in combined.columns:
        combined["risk_off_score"] = combined["mkt_vix_return_1d"] - combined["mkt_spy_return_1d"]
    if "mkt_gld_return_5d" in combined.columns and "mkt_spy_return_5d" in combined.columns:
        combined["gold_vs_spy_5d"] = combined["mkt_gld_return_5d"] - combined["mkt_spy_return_5d"]
    if "mkt_tlt_return_5d" in combined.columns and "mkt_spy_return_5d" in combined.columns:
        combined["bonds_vs_spy_5d"] = combined["mkt_tlt_return_5d"] - combined["mkt_spy_return_5d"]

    return combined.replace([np.inf, -np.inf], np.nan)


def build_expanded_signals(
    text_paths: list[str | Path],
    out_path: str | Path = "data/processed/expanded_signals.csv",
    data_dir: str | Path = "data/raw",
    start: str = "2021-01-01",
    market_tickers: list[str] | None = None,
    include_market_state: bool = True,
    include_text_vectors: bool = True,
    text_vector_features: int = DEFAULT_TEXT_VECTOR_FEATURES,
    update: bool = False,
) -> pd.DataFrame:
    texts = load_text_sources(text_paths)
    text_signals = _daily_text_signals(texts)
    if include_text_vectors:
        text_vectors = _daily_whole_text_vectors(texts, n_features=text_vector_features)
        text_signals = text_signals.merge(text_vectors, on="date", how="outer")
    if start:
        text_signals = text_signals[text_signals["date"] >= pd.Timestamp(start)].copy()

    if include_market_state:
        tickers = market_tickers if market_tickers is not None else DEFAULT_MARKET_TICKERS
        market = build_market_state(tickers, data_dir=data_dir, start=start, update=update)
        expanded = market.merge(text_signals, on="date", how="left")
    else:
        expanded = text_signals.copy()

    text_cols = [
        col
        for col in expanded.columns
        if col.startswith(("text_", "word_", "char_", "source_", "theme_", "sentiment_", "whole_text_vec_"))
    ]
    if text_cols:
        zero_fill_cols = [col for col in text_cols if not col.endswith("_sentiment")]
        expanded[zero_fill_cols] = expanded[zero_fill_cols].fillna(0)

    expanded = expanded.sort_values("date").reset_index(drop=True)
    out = ensure_parent(out_path)
    expanded.to_csv(out, index=False)
    print(f"Saved expanded signals: {out} ({len(expanded)} rows)")
    return expanded
