from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import download_spy
from .features import KEYWORDS, build_dataset


THEME_GROUPS = {
    "theme_trade": ["tariff", "trade", "import", "imports"],
    "theme_china_trade": ["china", "tariff", "trade", "import", "imports"],
    "theme_rates_inflation": ["inflation", "fed", "rate"],
    "theme_border_immigration": ["border", "immigration"],
    "theme_energy": ["oil"],
}

DEFAULT_MIN_ABS_T_STAT = 1.5
DEFAULT_MIN_INDEPENDENT_EVENTS = 5
DEFAULT_MIN_TEST_INDEPENDENT_EVENTS = 10
DEFAULT_MIN_ROBUST_TRAIN_INDEPENDENT_EVENTS = 10
DEFAULT_MIN_ROBUST_TEST_INDEPENDENT_EVENTS = 10
DEFAULT_COST_BPS = 5.0
LOW_TEST_SAMPLE_WARNING_EVENTS = 10
TRADABLE_SESSIONS = ["premarket", "market", "afterhours", "weekend"]
ROBUST_BASE_HORIZON = 3
ROBUST_CONSISTENCY_HORIZONS = [1, 5]

EVENT_RETURN_COLUMNS = [
    "ticker",
    "signal",
    "signal_type",
    "horizon_days",
    "vol_regime",
    "event_number",
    "event_date",
    "exit_date",
    "signal_count",
    "learned_direction",
    "target_return",
    "signed_gross_return",
    "net_return",
    "cumulative_net_return",
    "cost_bps",
]

JACKKNIFE_DETAIL_COLUMNS = [
    "ticker",
    "signal",
    "signal_type",
    "horizon_days",
    "vol_regime",
    "omitted_event_number",
    "omitted_event_date",
    "omitted_event_net_return",
    "jackknife_total_return_net",
    "jackknife_event_sharpe_net",
    "jackknife_max_drawdown_net",
    "full_total_return_net",
    "full_event_sharpe_net",
    "full_max_drawdown_net",
    "event_count",
    "cost_bps",
]

JACKKNIFE_SUMMARY_COLUMNS = [
    "ticker",
    "signal",
    "signal_type",
    "horizon_days",
    "vol_regime",
    "event_count",
    "full_total_return_net",
    "full_event_sharpe_net",
    "full_max_drawdown_net",
    "min_jackknife_total_return_net",
    "median_jackknife_total_return_net",
    "max_jackknife_total_return_net",
    "min_jackknife_event_sharpe_net",
    "median_jackknife_event_sharpe_net",
    "max_jackknife_event_sharpe_net",
    "worst_single_event_net_return",
    "best_single_event_net_return",
    "positive_total_return_flips_to_nonpositive",
    "positive_sharpe_flips_to_nonpositive",
    "most_important_event_date",
    "most_important_event_net_return",
    "most_important_event_return_without_it",
    "jackknife_fragile",
    "cost_bps",
]


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


def _sample_sharpe(returns: pd.Series) -> float | None:
    if len(returns) < 2:
        return None
    std = returns.std()
    if std == 0 or pd.isna(std):
        return None
    return float(returns.mean() / std)


def _compound_return(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    return float((1 + returns).prod() - 1)


def _split_by_time(
    data: pd.DataFrame,
    split_date: str | None,
    train_fraction: float,
    analysis_start: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    ordered = data.sort_values("date").copy()
    if analysis_start:
        ordered = ordered[ordered["date"] >= pd.Timestamp(analysis_start)].copy()
    if len(ordered) < 2:
        raise ValueError("Analysis window has too few rows. Use an earlier analysis_start.")
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


def _apply_tradable_keyword_policy(data: pd.DataFrame, include_unknown_time: bool) -> pd.DataFrame:
    adjusted = data.copy()
    sessions = TRADABLE_SESSIONS + (["unknown"] if include_unknown_time else [])
    for keyword in KEYWORDS:
        session_cols = [
            f"kwsession_{keyword}_{session}"
            for session in sessions
            if f"kwsession_{keyword}_{session}" in adjusted.columns
        ]
        if session_cols:
            adjusted[f"kw_{keyword}"] = adjusted[session_cols].sum(axis=1)
    return adjusted


def _prepare_horizon(data: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    horizon = max(int(horizon_days), 1)
    prepared = data.sort_values("date").copy()
    prepared["target_return"] = prepared["close"].shift(-horizon) / prepared["close"] - 1
    prepared["target_exit_date"] = prepared["date"].shift(-horizon)
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


def _independent_event_rows(
    sample: pd.DataFrame,
    signal_col: str,
    horizon_days: int,
) -> pd.DataFrame:
    ordered = sample.sort_values("date")
    events = ordered[ordered[signal_col] > 0]
    if events.empty:
        return events

    min_spacing = max(int(horizon_days), 1)
    positions = {idx: pos for pos, idx in enumerate(ordered.index)}
    selected_indexes = []
    last_position = -min_spacing
    for idx in events.index:
        position = positions[idx]
        if not selected_indexes or position >= last_position + min_spacing:
            selected_indexes.append(idx)
            last_position = position

    return ordered.loc[selected_indexes]


def _event_t_stat(event_returns: pd.Series, baseline_return: float | None) -> float | None:
    if baseline_return is None or len(event_returns) < 2:
        return None
    diff = event_returns - baseline_return
    std = diff.std()
    if std == 0 or pd.isna(std):
        return None
    return float(diff.mean() / (std / np.sqrt(len(diff))))


def _event_half_direction(
    train: pd.DataFrame,
    event_rows: pd.DataFrame,
    signal_col: str,
    return_col: str,
) -> int | None:
    if event_rows.empty:
        return None
    non_events = train.loc[train[signal_col] <= 0]
    if non_events.empty:
        return None
    effect = event_rows[return_col].mean() - non_events[return_col].mean()
    return 1 if effect > 0 else -1 if effect < 0 else 0


def _train_half_directions(
    train: pd.DataFrame,
    signal_col: str,
    return_col: str,
    stability_mode: str,
    horizon_days: int,
) -> tuple[int | None, int | None]:
    ordered = train.sort_values("date")
    if stability_mode == "calendar":
        midpoint = max(len(ordered) // 2, 1)
        first = ordered.iloc[:midpoint]
        second = ordered.iloc[midpoint:]
        return _effect_direction(first, signal_col, return_col), _effect_direction(second, signal_col, return_col)

    events = _independent_event_rows(ordered, signal_col, horizon_days)
    if len(events) < 2:
        return None, None
    midpoint = max(len(events) // 2, 1)
    first_events = events.iloc[:midpoint]
    second_events = events.iloc[midpoint:]
    return (
        _event_half_direction(ordered, first_events, signal_col, return_col),
        _event_half_direction(ordered, second_events, signal_col, return_col),
    )


def _stable_direction_required(
    direction: int,
    first_half_direction: int | None,
    second_half_direction: int | None,
) -> bool:
    return (
        direction != 0
        and first_half_direction == direction
        and second_half_direction == direction
    )


def _direction_allowed(direction: int, allowed_direction: str) -> bool:
    if allowed_direction == "long":
        return direction == 1
    if allowed_direction == "short":
        return direction == -1
    return direction != 0


def _signal_column(signal_name: str, signal_type: str) -> str:
    return signal_name if signal_type == "theme" else f"kw_{signal_name}"


def _event_net_returns(event_rows: pd.DataFrame, direction: int, cost_bps: float) -> pd.Series:
    if event_rows.empty or direction == 0:
        return pd.Series(dtype=float)
    signed = direction * event_rows["target_return"].astype(float)
    return signed - (2 * cost_bps / 10_000)


def _robust_event_diagnostics(
    robust_report: pd.DataFrame,
    test_contexts: dict[tuple[str, int, str], pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if robust_report.empty:
        return (
            pd.DataFrame(columns=EVENT_RETURN_COLUMNS),
            pd.DataFrame(columns=JACKKNIFE_DETAIL_COLUMNS),
            pd.DataFrame(columns=JACKKNIFE_SUMMARY_COLUMNS),
        )

    event_rows_out: list[dict[str, object]] = []
    jackknife_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for _, candidate in robust_report.iterrows():
        ticker = str(candidate["ticker"])
        signal = str(candidate["signal"])
        signal_type = str(candidate["signal_type"])
        horizon = int(candidate["horizon_days"])
        vol_regime = str(candidate["vol_regime"])
        direction = int(candidate["learned_direction"])
        cost_bps = float(candidate["cost_bps"])
        signal_col = _signal_column(signal, signal_type)
        test = test_contexts.get((ticker, horizon, vol_regime))
        if test is None or signal_col not in test.columns:
            continue

        independent_events = _independent_event_rows(test, signal_col, horizon)
        net_returns = _event_net_returns(independent_events, direction, cost_bps)
        if net_returns.empty:
            continue

        cumulative = (1 + net_returns).cumprod() - 1
        full_total = _compound_return(net_returns)
        full_sharpe = _sample_sharpe(net_returns)
        full_drawdown = _max_drawdown(net_returns)
        full_metrics = {
            "full_total_return_net": full_total,
            "full_event_sharpe_net": full_sharpe,
            "full_max_drawdown_net": full_drawdown,
            "event_count": int(len(net_returns)),
            "cost_bps": cost_bps,
        }
        candidate_jackknife_rows: list[dict[str, object]] = []

        for event_number, (idx, event) in enumerate(independent_events.iterrows(), start=1):
            event_net_return = float(net_returns.loc[idx])
            event_rows_out.append(
                {
                    "ticker": ticker,
                    "signal": signal,
                    "signal_type": signal_type,
                    "horizon_days": horizon,
                    "vol_regime": vol_regime,
                    "event_number": event_number,
                    "event_date": pd.Timestamp(event["date"]).date().isoformat(),
                    "exit_date": pd.Timestamp(event["target_exit_date"]).date().isoformat(),
                    "signal_count": float(event[signal_col]),
                    "learned_direction": direction,
                    "target_return": float(event["target_return"]),
                    "signed_gross_return": float(direction * event["target_return"]),
                    "net_return": event_net_return,
                    "cumulative_net_return": float(cumulative.loc[idx]),
                    "cost_bps": cost_bps,
                }
            )

            reduced_returns = net_returns.drop(index=idx)
            candidate_jackknife_rows.append(
                {
                    "ticker": ticker,
                    "signal": signal,
                    "signal_type": signal_type,
                    "horizon_days": horizon,
                    "vol_regime": vol_regime,
                    "omitted_event_number": event_number,
                    "omitted_event_date": pd.Timestamp(event["date"]).date().isoformat(),
                    "omitted_event_net_return": event_net_return,
                    "jackknife_total_return_net": _compound_return(reduced_returns),
                    "jackknife_event_sharpe_net": _sample_sharpe(reduced_returns),
                    "jackknife_max_drawdown_net": _max_drawdown(reduced_returns),
                    **full_metrics,
                }
            )

        jackknife_rows.extend(candidate_jackknife_rows)
        jackknife_for_candidate = pd.DataFrame(candidate_jackknife_rows)
        worst_total_idx = jackknife_for_candidate["jackknife_total_return_net"].idxmin()
        worst_total = jackknife_for_candidate.loc[worst_total_idx]
        min_sharpe = jackknife_for_candidate["jackknife_event_sharpe_net"].min()
        positive_sharpe_flips = (
            bool(full_sharpe is not None and full_sharpe > 0 and min_sharpe <= 0)
            if not pd.isna(min_sharpe)
            else False
        )
        positive_total_flips = bool(
            full_total > 0 and (jackknife_for_candidate["jackknife_total_return_net"] <= 0).any()
        )
        summary_rows.append(
            {
                "ticker": ticker,
                "signal": signal,
                "signal_type": signal_type,
                "horizon_days": horizon,
                "vol_regime": vol_regime,
                "event_count": int(len(net_returns)),
                "full_total_return_net": full_total,
                "full_event_sharpe_net": full_sharpe,
                "full_max_drawdown_net": full_drawdown,
                "min_jackknife_total_return_net": float(
                    jackknife_for_candidate["jackknife_total_return_net"].min()
                ),
                "median_jackknife_total_return_net": float(
                    jackknife_for_candidate["jackknife_total_return_net"].median()
                ),
                "max_jackknife_total_return_net": float(
                    jackknife_for_candidate["jackknife_total_return_net"].max()
                ),
                "min_jackknife_event_sharpe_net": (
                    float(min_sharpe) if not pd.isna(min_sharpe) else None
                ),
                "median_jackknife_event_sharpe_net": (
                    float(jackknife_for_candidate["jackknife_event_sharpe_net"].median())
                    if jackknife_for_candidate["jackknife_event_sharpe_net"].notna().any()
                    else None
                ),
                "max_jackknife_event_sharpe_net": (
                    float(jackknife_for_candidate["jackknife_event_sharpe_net"].max())
                    if jackknife_for_candidate["jackknife_event_sharpe_net"].notna().any()
                    else None
                ),
                "worst_single_event_net_return": float(net_returns.min()),
                "best_single_event_net_return": float(net_returns.max()),
                "positive_total_return_flips_to_nonpositive": positive_total_flips,
                "positive_sharpe_flips_to_nonpositive": positive_sharpe_flips,
                "most_important_event_date": worst_total["omitted_event_date"],
                "most_important_event_net_return": float(worst_total["omitted_event_net_return"]),
                "most_important_event_return_without_it": float(
                    worst_total["jackknife_total_return_net"]
                ),
                "jackknife_fragile": bool(positive_total_flips or positive_sharpe_flips),
                "cost_bps": cost_bps,
            }
        )

    event_report = pd.DataFrame(event_rows_out, columns=EVENT_RETURN_COLUMNS)
    jackknife_report = pd.DataFrame(jackknife_rows, columns=JACKKNIFE_DETAIL_COLUMNS)
    summary_report = pd.DataFrame(summary_rows, columns=JACKKNIFE_SUMMARY_COLUMNS)
    if not summary_report.empty:
        summary_report = summary_report.sort_values(
            ["jackknife_fragile", "min_jackknife_total_return_net"],
            ascending=[False, True],
            na_position="last",
        )
    return event_report, jackknife_report, summary_report


def _keyword_row(
    ticker: str,
    signal_col: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    min_keyword_days: int,
    min_independent_events: int,
    cost_bps: float,
    min_abs_t_stat: float,
    require_stable_direction: bool,
    allowed_direction: str,
    horizon_days: int,
    vol_regime: str,
    stability_mode: str,
) -> dict[str, object]:
    signal_type = "theme" if signal_col.startswith("theme_") else "keyword"
    signal_name = signal_col if signal_type == "theme" else signal_col.removeprefix("kw_")
    train_mask = train[signal_col] > 0
    test_mask = test[signal_col] > 0
    train_events = train.loc[train_mask]
    train_non_events = train.loc[~train_mask]
    test_events = test.loc[test_mask]
    train_independent_events = _independent_event_rows(train, signal_col, horizon_days)
    test_independent_events = _independent_event_rows(test, signal_col, horizon_days)

    train_count = int(len(train_events))
    test_count = int(len(test_events))
    train_independent_count = int(len(train_independent_events))
    test_independent_count = int(len(test_independent_events))
    enough_train = train_count >= min_keyword_days and train_independent_count >= min_independent_events

    train_event_return = float(train_events["target_return"].mean()) if train_count else None
    train_independent_return = (
        float(train_independent_events["target_return"].mean()) if train_independent_count else None
    )
    train_base_return = float(train["target_return"].mean())
    train_non_event_return = float(train_non_events["target_return"].mean()) if len(train_non_events) else None
    train_effect = None
    train_t_stat = None
    direction = 0

    if enough_train and train_independent_return is not None and train_non_event_return is not None:
        train_effect = train_independent_return - train_non_event_return
        train_t_stat = _event_t_stat(train_independent_events["target_return"], train_non_event_return)
        direction = 1 if train_effect > 0 else -1 if train_effect < 0 else 0

    first_half_direction, second_half_direction = _train_half_directions(
        train,
        signal_col,
        "target_return",
        stability_mode,
        horizon_days,
    )
    stable_direction = _stable_direction_required(direction, first_half_direction, second_half_direction)
    abs_train_t_stat = abs(train_t_stat) if train_t_stat is not None else None
    passes_t_stat = abs_train_t_stat is not None and abs_train_t_stat >= min_abs_t_stat
    passes_direction = _direction_allowed(direction, allowed_direction)
    selected_for_strategy = bool(
        enough_train
        and passes_t_stat
        and passes_direction
        and (stable_direction or not require_stable_direction)
    )

    event_net_returns = pd.Series(dtype=float)
    if selected_for_strategy and test_independent_count:
        event_gross_returns = direction * test_independent_events["target_return"]
        event_net_returns = event_gross_returns - (2 * cost_bps / 10_000)
    turnover = float(2 * test_independent_count) if selected_for_strategy else 0.0

    test_event_return = float(test_events["target_return"].mean()) if test_count else None
    test_independent_return = (
        float(test_independent_events["target_return"].mean()) if test_independent_count else None
    )
    test_base_return = float(test["target_return"].mean())
    test_non_events = test.loc[~test_mask]
    test_non_event_return = float(test_non_events["target_return"].mean()) if len(test_non_events) else None
    test_effect = (
        test_independent_return - test_non_event_return
        if test_independent_return is not None and test_non_event_return is not None
        else None
    )
    test_direction_hit_rate = None
    if selected_for_strategy and test_independent_count and direction != 0:
        signed = direction * test_independent_events["target_return"]
        test_direction_hit_rate = float((signed > 0).mean())

    return {
        "ticker": ticker,
        "signal": signal_name,
        "signal_type": signal_type,
        "horizon_days": int(horizon_days),
        "vol_regime": vol_regime,
        "stability_mode": stability_mode,
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
        "train_independent_events": train_independent_count,
        "test_independent_events": test_independent_count,
        "train_avg_next_return_when_keyword": train_event_return,
        "train_avg_next_return_when_independent_event": train_independent_return,
        "train_avg_next_return_all_days": train_base_return,
        "train_avg_next_return_non_keyword": train_non_event_return,
        "train_effect_vs_non_keyword": train_effect,
        "train_t_stat": train_t_stat,
        "abs_train_t_stat": abs_train_t_stat,
        "train_hit_rate_when_keyword": float(train_events["target_up"].mean()) if train_count else None,
        "train_hit_rate_when_independent_event": (
            float(train_independent_events["target_up"].mean()) if train_independent_count else None
        ),
        "test_avg_next_return_when_keyword": test_event_return,
        "test_avg_next_return_when_independent_event": test_independent_return,
        "test_avg_next_return_all_days": test_base_return,
        "test_avg_next_return_non_keyword": test_non_event_return,
        "test_effect_vs_non_keyword": test_effect,
        "test_direction_hit_rate": test_direction_hit_rate,
        "test_direction_hit_rate_independent": test_direction_hit_rate,
        "low_test_sample": bool(test_independent_count < LOW_TEST_SAMPLE_WARNING_EVENTS),
        "low_test_independent_sample": bool(test_independent_count < LOW_TEST_SAMPLE_WARNING_EVENTS),
        "test_strategy_total_return_net": float((1 + event_net_returns).prod() - 1),
        "test_active_strategy_total_return_net": (
            float((1 + event_net_returns).prod() - 1) if not event_net_returns.empty else None
        ),
        "test_strategy_sharpe_net": _sample_sharpe(event_net_returns),
        "test_strategy_event_sharpe_net": _sample_sharpe(event_net_returns),
        "test_strategy_max_drawdown_net": _max_drawdown(event_net_returns),
        "test_turnover": turnover,
        "cost_bps": cost_bps,
    }


def _effect_matches_direction(value: object, direction: int) -> bool:
    if value is None or pd.isna(value) or direction == 0:
        return False
    return float(value) * direction >= 0


def _robust_selected_report(
    report: pd.DataFrame,
    min_robust_train_independent_events: int,
    min_robust_test_independent_events: int,
) -> pd.DataFrame:
    if report.empty:
        return pd.DataFrame()

    base_rows = report[
        (report["selected_for_strategy"])
        & (report["vol_regime"] == "all")
        & (report["horizon_days"] == ROBUST_BASE_HORIZON)
        & (report["train_independent_events"] >= min_robust_train_independent_events)
        & (report["test_independent_events"] >= min_robust_test_independent_events)
    ]
    robust_rows: list[dict[str, object]] = []
    key_cols = ["ticker", "signal", "signal_type", "vol_regime"]

    for _, base in base_rows.iterrows():
        direction = int(base["learned_direction"])
        missing_horizons: list[int] = []
        train_consistent = True
        test_consistent = True

        for horizon in ROBUST_CONSISTENCY_HORIZONS:
            matches = report
            for col in key_cols:
                matches = matches[matches[col] == base[col]]
            matches = matches[matches["horizon_days"] == horizon]
            if matches.empty:
                missing_horizons.append(horizon)
                train_consistent = False
                test_consistent = False
                continue

            other = matches.iloc[0]
            train_consistent = train_consistent and (
                other["train_independent_events"] >= min_robust_train_independent_events
            )
            test_consistent = test_consistent and (
                other["test_independent_events"] >= min_robust_test_independent_events
            )
            train_consistent = train_consistent and _effect_matches_direction(
                other["train_effect_vs_non_keyword"],
                direction,
            )
            test_consistent = test_consistent and _effect_matches_direction(
                other["test_effect_vs_non_keyword"],
                direction,
            )

        row = base.to_dict()
        row["robust_base_horizon"] = ROBUST_BASE_HORIZON
        row["consistency_horizons"] = ",".join(str(value) for value in ROBUST_CONSISTENCY_HORIZONS)
        row["min_robust_train_independent_events"] = int(min_robust_train_independent_events)
        row["min_robust_test_independent_events"] = int(min_robust_test_independent_events)
        row["missing_consistency_horizons"] = ",".join(str(value) for value in missing_horizons)
        row["train_horizon_direction_consistent"] = bool(train_consistent)
        row["test_horizon_effect_consistent"] = bool(test_consistent)
        row["selected_for_robust_strategy"] = bool(train_consistent and test_consistent and not missing_horizons)
        robust_rows.append(row)

    robust = pd.DataFrame(robust_rows)
    if robust.empty:
        return robust
    robust = robust[robust["selected_for_robust_strategy"]].copy()
    if robust.empty:
        return robust
    return robust.sort_values(
        ["abs_train_t_stat", "train_independent_events", "test_independent_events"],
        ascending=[False, False, False],
        na_position="last",
    )


def keyword_impact_report(
    tickers: list[str],
    start: str = "2015-01-01",
    speeches_path: str | Path = "data/raw/trump_speeches.csv",
    data_dir: str | Path = "data/raw",
    outputs_dir: str | Path = "outputs/keyword_impact",
    analysis_start: str | None = "2021-01-01",
    split_date: str | None = None,
    train_fraction: float = 0.7,
    min_keyword_days: int = 20,
    min_independent_events: int = DEFAULT_MIN_INDEPENDENT_EVENTS,
    min_test_independent_events: int = DEFAULT_MIN_TEST_INDEPENDENT_EVENTS,
    min_robust_train_independent_events: int = DEFAULT_MIN_ROBUST_TRAIN_INDEPENDENT_EVENTS,
    min_robust_test_independent_events: int = DEFAULT_MIN_ROBUST_TEST_INDEPENDENT_EVENTS,
    horizon_days: int = 1,
    horizons: list[int] | None = None,
    min_abs_t_stat: float = DEFAULT_MIN_ABS_T_STAT,
    require_stable_direction: bool = True,
    stability_mode: str = "event",
    allowed_direction: str = "all",
    cost_bps: float = DEFAULT_COST_BPS,
    include_unknown_time: bool = False,
    update: bool = False,
) -> pd.DataFrame:
    out_dir = Path(outputs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    test_contexts: dict[tuple[str, int, str], pd.DataFrame] = {}
    horizon_list = horizons if horizons else [horizon_days]
    horizon_list = sorted({max(int(value), 1) for value in horizon_list})

    for ticker in tickers:
        symbol = ticker.upper()
        price_path = Path(data_dir) / f"{symbol}.csv"
        download_spy(ticker=symbol, start=start, cache_path=price_path, update=update)
        base_data = _add_market_volatility(
            _add_theme_columns(
                _apply_tradable_keyword_policy(
                    build_dataset(price_path, speeches_path),
                    include_unknown_time,
                )
            ),
            data_dir=data_dir,
            start=start,
            update=update,
        )

        ticker_rows: list[dict[str, object]] = []
        for horizon in horizon_list:
            data = _prepare_horizon(base_data, horizon)
            train, test, split = _split_by_time(data, split_date, train_fraction, analysis_start)
            low_vol, high_vol = _vol_thresholds(train)
            train = _assign_vol_regime(train, low_vol, high_vol)
            test = _assign_vol_regime(test, low_vol, high_vol)
            signal_cols = [col for col in data.columns if col.startswith(("kw_", "theme_"))]

            split_info = {
                "ticker": symbol,
                "split_date": split.date().isoformat(),
                "horizon_days": int(horizon),
                "analysis_start": analysis_start,
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
                "min_independent_events": int(min_independent_events),
                "min_test_independent_events": int(min_test_independent_events),
                "min_robust_train_independent_events": int(min_robust_train_independent_events),
                "min_robust_test_independent_events": int(min_robust_test_independent_events),
                "min_abs_t_stat": float(min_abs_t_stat),
                "require_stable_direction": bool(require_stable_direction),
                "stability_mode": stability_mode,
                "allowed_direction": allowed_direction,
                "cost_bps": float(cost_bps),
                "include_unknown_time": bool(include_unknown_time),
                "entry_timing_policy": (
                    "known timestamps only; premarket/market enter at signal-date close; "
                    "afterhours/weekend enter at next trading-day close"
                ),
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
                test_contexts[(symbol, int(horizon), regime)] = regime_test.copy()
                rows = [
                    _keyword_row(
                        symbol,
                        signal_col,
                        regime_train,
                        regime_test,
                        min_keyword_days,
                        min_independent_events,
                        cost_bps,
                        min_abs_t_stat,
                        require_stable_direction,
                        allowed_direction,
                        horizon,
                        regime,
                        stability_mode,
                    )
                    for signal_col in signal_cols
                ]
                ticker_rows.extend(rows)
                all_rows.extend(rows)

        ticker_report = pd.DataFrame(ticker_rows)
        ticker_report.to_csv(out_dir / f"{symbol}_keyword_impact.csv", index=False)

    report = pd.DataFrame(all_rows)
    selected_report = pd.DataFrame()
    robust_report = pd.DataFrame()
    if not report.empty:
        report = report.sort_values(
            [
                "selected_for_strategy",
                "abs_train_t_stat",
                "train_independent_events",
                "test_strategy_sharpe_net",
            ],
            ascending=[False, False, False, False],
            na_position="last",
        )
        selected_report = report[report["selected_for_strategy"]].copy()
        robust_report = _robust_selected_report(
            report,
            min_robust_train_independent_events,
            min_robust_test_independent_events,
        )
    event_report, jackknife_report, jackknife_summary = _robust_event_diagnostics(
        robust_report,
        test_contexts,
    )

    split_report = pd.DataFrame(split_rows)
    report.to_csv(out_dir / "summary.csv", index=False)
    selected_report.to_csv(out_dir / "selected.csv", index=False)
    robust_report.to_csv(out_dir / "robust_selected.csv", index=False)
    event_report.to_csv(out_dir / "robust_event_returns.csv", index=False)
    jackknife_report.to_csv(out_dir / "robust_jackknife.csv", index=False)
    jackknife_summary.to_csv(out_dir / "robust_jackknife_summary.csv", index=False)
    split_report.to_csv(out_dir / "splits.csv", index=False)
    (out_dir / "splits.json").write_text(
        json.dumps(split_rows, indent=2),
        encoding="utf-8",
    )

    print(split_report.to_string(index=False))
    if selected_report.empty:
        print("No signals passed the train-only strategy filters.")
    else:
        print(selected_report.head(40).to_string(index=False))
    if robust_report.empty:
        print("No signals passed the robust independent-event filters.")
    else:
        print(robust_report.head(40).to_string(index=False))
    if not jackknife_summary.empty:
        print(jackknife_summary.head(40).to_string(index=False))
    print(f"Saved keyword impact report: {out_dir / 'summary.csv'}")
    print(f"Saved selected keyword candidates: {out_dir / 'selected.csv'}")
    print(f"Saved robust keyword candidates: {out_dir / 'robust_selected.csv'}")
    print(f"Saved robust event returns: {out_dir / 'robust_event_returns.csv'}")
    print(f"Saved robust jackknife diagnostics: {out_dir / 'robust_jackknife_summary.csv'}")
    return report
