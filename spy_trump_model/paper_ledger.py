from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .data import download_spy, ensure_parent


DEFAULT_PAPER_TICKERS = ["SPY", "QQQ", "GLD", "TLT", "USO", "XLE", "XLI", "XLF", "SMH", "FXI"]
DEFAULT_PAPER_HORIZONS = [1, 3, 5]


def _safe_ticker_name(ticker: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", ticker.strip().upper())
    return safe.strip("_") or "ticker"


def _ticker_alias(ticker: str) -> str:
    return _safe_ticker_name(ticker).lower()


def _normalize_horizons(horizons: list[int] | tuple[int, ...]) -> list[int]:
    normalized = []
    for horizon in horizons:
        value = int(horizon)
        if value <= 0:
            raise ValueError("horizons must be positive integers.")
        if value not in normalized:
            normalized.append(value)
    if not normalized:
        raise ValueError("At least one horizon is required.")
    return normalized


def _load_expanded_signals(signals_path: str | Path, start: str | None) -> pd.DataFrame:
    path = Path(signals_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing expanded signals file: {path}")

    signals = pd.read_csv(path)
    if "date" not in signals.columns:
        raise ValueError(f"{path} must include a date column.")

    signals["date"] = pd.to_datetime(signals["date"], errors="coerce").dt.normalize()
    signals = signals.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if start:
        signals = signals[signals["date"] >= pd.Timestamp(start)].copy()
    return signals


def _theme_intensity_columns(data: pd.DataFrame) -> list[str]:
    excluded_suffixes = ("_log_intensity", "_pos_intensity", "_neg_intensity")
    return [
        col
        for col in data.columns
        if col.startswith("theme_") and col.endswith("_intensity") and not col.endswith(excluded_suffixes)
    ]


def _add_top_theme(data: pd.DataFrame) -> pd.DataFrame:
    ledger = data.copy()
    theme_cols = _theme_intensity_columns(ledger)
    if not theme_cols:
        ledger["top_theme"] = ""
        ledger["top_theme_intensity"] = 0.0
        return ledger

    theme_values = ledger[theme_cols].fillna(0.0)
    max_cols = theme_values.idxmax(axis=1)
    max_values = theme_values.max(axis=1)
    ledger["top_theme"] = np.where(
        max_values > 0,
        max_cols.str.replace(r"^theme_", "", regex=True).str.replace(r"_intensity$", "", regex=True),
        "",
    )
    ledger["top_theme_intensity"] = max_values
    return ledger


def _read_price_data(
    ticker: str,
    data_dir: str | Path,
    start: str | None,
    update: bool,
) -> pd.DataFrame:
    cache_path = Path(data_dir) / f"{_safe_ticker_name(ticker)}.csv"
    raw = download_spy(
        ticker=ticker,
        start=start or "2021-01-01",
        cache_path=cache_path,
        update=update,
    )
    price = raw.rename(columns={col: col.lower() for col in raw.columns}).copy()
    if "date" not in price.columns or "close" not in price.columns:
        raise ValueError(f"Market data for {ticker} must include Date and Close.")

    price["date"] = pd.to_datetime(price["date"], errors="coerce").dt.normalize()
    price = price.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    if start:
        price = price[price["date"] >= pd.Timestamp(start)].copy()
    return price[["date", "close"]].reset_index(drop=True)


def _forward_return_frame(
    ticker: str,
    data_dir: str | Path,
    start: str | None,
    horizons: list[int],
    entry_lag_days: int,
    update: bool,
) -> pd.DataFrame:
    if entry_lag_days < 0:
        raise ValueError("entry_lag_days must be zero or positive.")

    price = _read_price_data(ticker=ticker, data_dir=data_dir, start=start, update=update)
    alias = _ticker_alias(ticker)
    frame = pd.DataFrame({"date": price["date"]})
    entry_shift = -entry_lag_days
    frame[f"{alias}_entry_date"] = price["date"].shift(entry_shift)
    frame[f"{alias}_entry_close"] = price["close"].shift(entry_shift)

    for horizon in horizons:
        exit_shift = -(entry_lag_days + horizon)
        exit_date_col = f"{alias}_exit_date_{horizon}d"
        return_col = f"{alias}_return_{horizon}d"
        frame[exit_date_col] = price["date"].shift(exit_shift)
        exit_close = price["close"].shift(exit_shift)
        frame[return_col] = exit_close / frame[f"{alias}_entry_close"] - 1

    return frame


def _front_columns(ledger: pd.DataFrame) -> list[str]:
    preferred = [
        "observation_id",
        "date",
        "text_item_count",
        "word_count",
        "char_count",
        "sentiment_compound",
        "sentiment_pos",
        "sentiment_neg",
        "sentiment_neu",
        "top_theme",
        "top_theme_intensity",
    ]
    return [col for col in preferred if col in ledger.columns]


def summarize_paper_ledger(
    ledger: pd.DataFrame,
    tickers: list[str],
    horizons: list[int],
    entry_lag_days: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    text_days = int((ledger.get("text_item_count", pd.Series(dtype=float)).fillna(0) > 0).sum())
    for ticker in tickers:
        alias = _ticker_alias(ticker)
        for horizon in horizons:
            return_col = f"{alias}_return_{horizon}d"
            if return_col not in ledger.columns:
                continue
            returns = ledger[return_col].dropna()
            rows.append(
                {
                    "ticker": ticker,
                    "horizon_days": horizon,
                    "entry_lag_days": entry_lag_days,
                    "observations": int(len(ledger)),
                    "completed_observations": int(len(returns)),
                    "pending_observations": int(len(ledger) - len(returns)),
                    "text_days": text_days,
                    "first_date": ledger["date"].min().date().isoformat() if not ledger.empty else "",
                    "last_date": ledger["date"].max().date().isoformat() if not ledger.empty else "",
                    "mean_forward_return": float(returns.mean()) if not returns.empty else np.nan,
                    "median_forward_return": float(returns.median()) if not returns.empty else np.nan,
                    "hit_rate_positive": float((returns > 0).mean()) if not returns.empty else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_paper_ledger(
    signals_path: str | Path = "data/processed/expanded_signals.csv",
    out_path: str | Path = "outputs/paper_trading/ledger.csv",
    summary_out: str | Path | None = "outputs/paper_trading/summary.csv",
    data_dir: str | Path = "data/raw",
    tickers: list[str] | None = None,
    horizons: list[int] | tuple[int, ...] = DEFAULT_PAPER_HORIZONS,
    start: str | None = None,
    entry_lag_days: int = 1,
    text_only: bool = True,
    update: bool = False,
) -> pd.DataFrame:
    tickers = tickers if tickers is not None else DEFAULT_PAPER_TICKERS
    horizons = _normalize_horizons(list(horizons))
    signals = _load_expanded_signals(signals_path=signals_path, start=start)
    if text_only:
        if "text_item_count" not in signals.columns:
            raise ValueError("text_only=True requires text_item_count in expanded signals.")
        signals = signals[signals["text_item_count"].fillna(0) > 0].copy()

    ledger = _add_top_theme(signals)
    ledger.insert(0, "observation_id", range(1, len(ledger) + 1))

    if ledger.empty:
        out = ensure_parent(out_path)
        ledger.to_csv(out, index=False)
        print(f"Saved paper ledger: {out} (0 rows)")
        if summary_out is not None:
            summary_path = ensure_parent(summary_out)
            pd.DataFrame().to_csv(summary_path, index=False)
            print(f"Saved paper ledger summary: {summary_path} (0 rows)")
        return ledger

    price_start = ledger["date"].min().date().isoformat() if not ledger.empty else start
    for ticker in tickers:
        returns = _forward_return_frame(
            ticker=ticker,
            data_dir=data_dir,
            start=price_start,
            horizons=horizons,
            entry_lag_days=entry_lag_days,
            update=update,
        )
        ledger = ledger.merge(returns, on="date", how="left")

    front = _front_columns(ledger)
    ledger = ledger[front + [col for col in ledger.columns if col not in front]]
    out = ensure_parent(out_path)
    ledger.to_csv(out, index=False)
    print(f"Saved paper ledger: {out} ({len(ledger)} rows)")

    if summary_out is not None:
        summary = summarize_paper_ledger(
            ledger=ledger,
            tickers=tickers,
            horizons=horizons,
            entry_lag_days=entry_lag_days,
        )
        summary_path = ensure_parent(summary_out)
        summary.to_csv(summary_path, index=False)
        print(f"Saved paper ledger summary: {summary_path} ({len(summary)} rows)")

    return ledger
