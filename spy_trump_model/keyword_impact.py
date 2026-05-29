from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import download_spy
from .features import build_dataset


THEME_GROUPS = {
    "theme_trade": ["tariff", "trade", "import", "imports"],
    "theme_china_trade": ["china", "tariff", "trade", "import", "imports"],
    "theme_rates_inflation": ["inflation", "fed", "rate"],
    "theme_border_immigration": ["border", "immigration"],
    "theme_energy": ["oil"],
}


def _max_drawdown(returns: pd.Series) -> float | None:
    if returns.empty:
        return None
    curve = (1 + returns).cumprod()
    return float((curve / curve.cummax() - 1).min())


def _sharpe(returns: pd.Series) -> float | None:
    if returns.empty:
        return None
    std = returns.std()
    if std == 0 or pd.isna(std):
        return None
    return float((returns.mean() / std) * np.sqrt(252))


def _split_by_time(
    data: pd.DataFrame,
    split_date: str | None,
    train_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    ordered = data.sort_values("date").copy()
    if split_date:
        split = pd.Timestamp(split_date)
    else:
        split_idx = int(len(ordered) * train_fraction)
        split_idx = min(max(split_idx, 1), len(ordered) - 1)
        split = pd.Timestamp(ordered["date"].iloc[split_idx])

    train = ordered[ordered["date"] < split].copy()
    test = ordered[ordered["date"] >= split].copy()

    if train.empty or test.empty:
        raise ValueError("Train/test split produced an empty sample. Change split_date or train_fraction.")
    if train["date"].max() >= test["date"].min():
        raise RuntimeError("Train/test overlap detected.")

    return train, test, split


def _add_theme_columns(data: pd.DataFrame) -> pd.DataFrame:
    enriched = data.copy()
    for theme, keywords in THEME_GROUPS.items():
        cols = [f"kw_{keyword}" for keyword in keywords if f"kw_{keyword}" in enriched.columns]
        if cols:
            enriched[theme] = enriched[cols].sum(axis=1)
    return enriched


def _prepare_horizon(data: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    horizon = max(int(horizon_days), 1)
    prepared = data.sort_values("date").copy()
    prepared["target_return"] = prepared["close"].shift(-horizon) / prepared["close"] - 1
    prepared["target_up"] = (prepared["target_return"] > 0).astype(int)
    return prepared.dropna(subset=["target_return"])


def _add_market_volatility(
    data: pd.DataFrame,
    data_dir: str | Path,
    start: str,
    update: bool,
    market_ticker: str = "SPY",
) -> pd.DataFrame:
    market_path = Path(data_dir) / f"{market_ticker.upper()}.csv"
    download_spy(ticker=market_ticker.upper(), start=start, cache_path=market_path, update=update)
    market = pd.read_csv(market_path, parse_dates=["Date"])
    market = market.rename(columns={col: col.lower() for col in market.columns})
    market = market.sort_values("date").copy()
    market["date"] = market["date"].dt.normalize()
    market["market_return_1d"] = market["close"].pct_change()
    market["market_vol_20d_lag1"] = market["market_return_1d"].rolling(20).std().shift(1)
    return data.merge(market[["date", "market_vol_20d_lag1"]], on="date", how="left")


def _vol_thresholds(train: pd.DataFrame) -> tuple[float | None, float | None]:
    vol = train["market_vol_20d_lag1"].dropna()
    if vol.empty:
        return None, None
    low = float(vol.quantile(1 / 3))
    high = float(vol.quantile(2 / 3))
    return low, high


def _assign_vol_regime(data: pd.DataFrame, low: float | None, high: float | None) -> pd.DataFrame:
    assigned = data.copy()
    if low is None or high is None:
        assigned["vol_regime"] = "unknown"
        return assigned

    conditions = [
        assigned["market_vol_20d_lag1"] <= low,
        assigned["market_vol_20d_lag1"] <= high,
    ]
    assigned["vol_regime"] = np.select(conditions, ["low", "medium"], default="high")
    assigned.loc[assigned["market_vol_20d_lag1"].isna(), "vol_regime"] = "unknown"
    return assigned


def _effect_direction(sample: pd.DataFrame, signal_col: str, return_col: str) -> int | None:
    mask = sample[signal_col] > 0
    events = sample.loc[mask]
    non_events = sample.loc[~mask]
    if events.empty or non_events.empty:
        return None
    effect = events[return_col].mean() - non_events[return_col].mean()
    return 1 if effect > 0 else -1 if effect < 0 else 0


def _train_half_directions(train: pd.DataFrame, signal_col: str, return_col: str) -> tuple[int | None, int | None]:
    ordered = train.sort_values("date")
    midpoint = max(len(ordered) // 2, 1)
    first = ordered.iloc[:midpoint]
    second = ordered.iloc[midpoint:]
    return _effect_direction(first, signal_col, return_col), _effect_direction(second, signal_col, return_col)


def _direction_allowed(direction: int, allowed_direction: str) -> bool:
    if allowed_direction == "long":
        return direction == 1
    if allowed_direction == "short":
        return direction == -1
    return direction != 0


def _keyword_row(
    ticker: str,
    signal_col: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    min_keyword_days: int,
    cost_bps: float,
    min_abs_t_stat: float,
    require_stable_direction: bool,
    allowed_direction: str,
    horizon_days: int,
    vol_regime: str,
) -> dict[str, object]:
    signal_type = "theme" if signal_col.startswith("theme_") else "keyword"
    signal_name = signal_col if signal_type == "theme" else signal_col.removeprefix("kw_")
    train_mask = train[signal_col] > 0
    test_mask = test[signal_col] > 0
    train_events = train.loc[train_mask]
    train_non_events = train.loc[~train_mask]
    test_events = test.loc[test_mask]

    train_count = int(len(train_events))
    test_count = int(len(test_events))
    enough_train = train_count >= min_keyword_days

    train_event_return = float(train_events["target_return"].mean()) if train_count else None
    train_base_return = float(train["target_return"].mean())
    train_non_event_return = float(train_non_events["target_return"].mean()) if len(train_non_events) else None
    train_effect = None
    train_t_stat = None
    direction = 0

    if enough_train and train_event_return is not None and train_non_event_return is not None:
        train_effect = train_event_return - train_non_event_return
        event_var = train_events["target_return"].var()
        non_event_var = train_non_events["target_return"].var()
        stderr = np.sqrt((event_var / len(train_events)) + (non_event_var / len(train_non_events)))
        train_t_stat = float(train_effect / stderr) if stderr and not pd.isna(stderr) else None
        direction = 1 if train_effect > 0 else -1 if train_effect < 0 else 0

    first_half_direction, second_half_direction = _train_half_directions(train, signal_col, "target_return")
    stable_direction = (
        direction != 0
        and first_half_direction == direction
        and second_half_direction == direction
    )
    passes_t_stat = train_t_stat is not None and abs(train_t_stat) >= min_abs_t_stat
    passes_direction = _direction_allowed(direction, allowed_direction)
    selected_for_strategy = bool(
        enough_train
        and passes_t_stat
        and passes_direction
        and (stable_direction or not require_stable_direction)
    )

    test_signal = pd.Series(0, index=test.index, dtype=float)
    if selected_for_strategy:
        test_signal.loc[test_mask] = direction
    turnover = test_signal.diff().abs().fillna(test_signal.abs())
    gross_returns = test_signal * test["target_return"]
    net_returns = gross_returns - turnover * (cost_bps / 10_000)
    active_net = net_returns.loc[test_mask]

    test_event_return = float(test_events["target_return"].mean()) if test_count else None
    test_base_return = float(test["target_return"].mean())
    test_non_events = test.loc[~test_mask]
    test_non_event_return = float(test_non_events["target_return"].mean()) if len(test_non_events) else None
    test_effect = (
        test_event_return - test_non_event_return
        if test_event_return is not None and test_non_event_return is not None
        else None
    )
    test_direction_hit_rate = None
    if selected_for_strategy and test_count and direction != 0:
        signed = direction * test_events["target_return"]
        test_direction_hit_rate = float((signed > 0).mean())

    return {
        "ticker": ticker,
        "signal": signal_name,
        "signal_type": signal_type,
        "horizon_days": int(horizon_days),
        "vol_regime": vol_regime,
        "selected_from_train": bool(enough_train),
        "selected_for_strategy": selected_for_strategy,
        "learned_direction": direction,
        "train_first_half_direction": first_half_direction,
        "train_second_half_direction": second_half_direction,
        "stable_train_direction": bool(stable_direction),
        "passes_t_stat": bool(passes_t_stat),
        "passes_direction_filter": bool(passes_direction),
        "train_keyword_days": train_count,
        "test_keyword_days": test_count,
        "train_avg_next_return_when_keyword": train_event_return,
        "train_avg_next_return_all_days": train_base_return,
        "train_avg_next_return_non_keyword": train_non_event_return,
        "train_effect_vs_non_keyword": train_effect,
        "train_t_stat": train_t_stat,
        "train_hit_rate_when_keyword": float(train_events["target_up"].mean()) if train_count else None,
        "test_avg_next_return_when_keyword": test_event_return,
        "test_avg_next_return_all_days": test_base_return,
        "test_avg_next_return_non_keyword": test_non_event_return,
        "test_effect_vs_non_keyword": test_effect,
        "test_direction_hit_rate": test_direction_hit_rate,
        "test_strategy_total_return_net": float((1 + net_returns).prod() - 1),
        "test_active_strategy_total_return_net": float((1 + active_net).prod() - 1) if not active_net.empty else None,
        "test_strategy_sharpe_net": _sharpe(net_returns),
        "test_strategy_max_drawdown_net": _max_drawdown(net_returns),
        "test_turnover": float(turnover.sum()),
        "cost_bps": cost_bps,
    }


def keyword_impact_report(
    tickers: list[str],
    start: str = "2015-01-01",
    speeches_path: str | Path = "data/raw/trump_speeches.csv",
    data_dir: str | Path = "data/raw",
    outputs_dir: str | Path = "outputs/keyword_impact",
    split_date: str | None = None,
    train_fraction: float = 0.7,
    min_keyword_days: int = 20,
    horizon_days: int = 1,
    horizons: list[int] | None = None,
    min_abs_t_stat: float = 0.0,
    require_stable_direction: bool = True,
    allowed_direction: str = "all",
    cost_bps: float = 1.0,
    update: bool = False,
) -> pd.DataFrame:
    out_dir = Path(outputs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    horizon_list = horizons if horizons else [horizon_days]
    horizon_list = sorted({max(int(value), 1) for value in horizon_list})

    for ticker in tickers:
        symbol = ticker.upper()
        price_path = Path(data_dir) / f"{symbol}.csv"
        download_spy(ticker=symbol, start=start, cache_path=price_path, update=update)
        base_data = _add_market_volatility(
            _add_theme_columns(build_dataset(price_path, speeches_path)),
            data_dir=data_dir,
            start=start,
            update=update,
        )

        ticker_rows: list[dict[str, object]] = []
        for horizon in horizon_list:
            data = _prepare_horizon(base_data, horizon)
            train, test, split = _split_by_time(data, split_date, train_fraction)
            low_vol, high_vol = _vol_thresholds(train)
            train = _assign_vol_regime(train, low_vol, high_vol)
            test = _assign_vol_regime(test, low_vol, high_vol)
            signal_cols = [col for col in data.columns if col.startswith(("kw_", "theme_"))]

            split_info = {
                "ticker": symbol,
                "split_date": split.date().isoformat(),
                "horizon_days": int(horizon),
                "train_start": train["date"].min().date().isoformat(),
                "train_end": train["date"].max().date().isoformat(),
                "test_start": test["date"].min().date().isoformat(),
                "test_end": test["date"].max().date().isoformat(),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "overlap": bool(train["date"].max() >= test["date"].min()),
                "market_vol_low_threshold": low_vol,
                "market_vol_high_threshold": high_vol,
                "min_keyword_days": int(min_keyword_days),
                "min_abs_t_stat": float(min_abs_t_stat),
                "require_stable_direction": bool(require_stable_direction),
                "allowed_direction": allowed_direction,
            }
            split_rows.append(split_info)

            regime_pairs = [("all", train, test)]
            for regime in ["low", "medium", "high"]:
                regime_pairs.append(
                    (
                        regime,
                        train[train["vol_regime"] == regime],
                        test[test["vol_regime"] == regime],
                    )
                )

            for regime, regime_train, regime_test in regime_pairs:
                if regime_train.empty or regime_test.empty:
                    continue
                rows = [
                    _keyword_row(
                        symbol,
                        signal_col,
                        regime_train,
                        regime_test,
                        min_keyword_days,
                        cost_bps,
                        min_abs_t_stat,
                        require_stable_direction,
                        allowed_direction,
                        horizon,
                        regime,
                    )
                    for signal_col in signal_cols
                ]
                ticker_rows.extend(rows)
                all_rows.extend(rows)

        ticker_report = pd.DataFrame(ticker_rows)
        ticker_report.to_csv(out_dir / f"{symbol}_keyword_impact.csv", index=False)

    report = pd.DataFrame(all_rows)
    if not report.empty:
        report = report.sort_values(
            ["test_strategy_sharpe_net", "test_direction_hit_rate", "train_t_stat"],
            ascending=[False, False, False],
            na_position="last",
        )

    split_report = pd.DataFrame(split_rows)
    report.to_csv(out_dir / "summary.csv", index=False)
    split_report.to_csv(out_dir / "splits.csv", index=False)
    (out_dir / "splits.json").write_text(
        json.dumps(split_rows, indent=2),
        encoding="utf-8",
    )

    print(split_report.to_string(index=False))
    print(report.head(40).to_string(index=False))
    print(f"Saved keyword impact report: {out_dir / 'summary.csv'}")
    return report
