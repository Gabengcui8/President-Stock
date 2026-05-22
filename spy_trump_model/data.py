from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


def ensure_parent(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def download_spy(
    ticker: str = "SPY",
    start: str = "2015-01-01",
    cache_path: str | Path = "data/raw/SPY.csv",
    update: bool = False,
) -> pd.DataFrame:
    path = ensure_parent(cache_path)

    if path.exists() and not update:
        df = pd.read_csv(path, parse_dates=["Date"])
        print(f"Using cached {ticker} data: {path} ({len(df)} rows)")
        return df

    print(f"Downloading {ticker} from yfinance...")
    df = yf.download(
        ticker,
        start=start,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if df.empty:
        raise RuntimeError(f"No data returned for ticker {ticker}.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = df.reset_index()
    df.to_csv(path, index=False)
    print(f"Saved {ticker} data: {path} ({len(df)} rows)")
    return df

