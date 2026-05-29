from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import download_spy, ensure_parent
from .features import build_dataset, feature_columns


def _empty_metrics(reason: str) -> dict[str, object]:
    return {"status": "not_enough_data", "reason": reason}


def _max_drawdown(curve: pd.Series) -> float:
    running_max = curve.cummax()
    drawdown = curve / running_max - 1
    return float(drawdown.min())


def _annualized_sharpe(returns: pd.Series) -> float | None:
    std = returns.std()
    if std == 0 or pd.isna(std):
        return None
    return float((returns.mean() / std) * np.sqrt(252))


def _signal_metrics(signals: pd.DataFrame, prefix: str) -> dict[str, object]:
    if signals.empty:
        return {
            f"{prefix}_predictions": 0,
            f"{prefix}_active_trades": 0,
        }

    y_true = signals["target_next_up"].astype(int)
    y_score = signals["prob_up"]
    y_pred = (signals["prob_up"] >= 0.5).astype(int)
    strategy_curve = (1 + signals["strategy_return"]).cumprod()
    strategy_net_curve = (1 + signals["strategy_return_net"]).cumprod()
    buy_hold_curve = (1 + signals["target_next_return"]).cumprod()
    ann_factor = 252 / max(len(signals), 1)

    return {
        f"{prefix}_predictions": int(len(signals)),
        f"{prefix}_active_trades": int((signals["signal"] != 0).sum()),
        f"{prefix}_exposure": float((signals["signal"] != 0).mean()),
        f"{prefix}_turnover": float(signals["turnover"].sum()),
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_roc_auc": float(roc_auc_score(y_true, y_score)) if y_true.nunique() == 2 else None,
        f"{prefix}_strategy_total_return": float(strategy_curve.iloc[-1] - 1),
        f"{prefix}_strategy_total_return_net": float(strategy_net_curve.iloc[-1] - 1),
        f"{prefix}_buy_hold_total_return": float(buy_hold_curve.iloc[-1] - 1),
        f"{prefix}_strategy_annualized_return": float(strategy_curve.iloc[-1] ** ann_factor - 1),
        f"{prefix}_strategy_annualized_return_net": float(strategy_net_curve.iloc[-1] ** ann_factor - 1),
        f"{prefix}_buy_hold_annualized_return": float(buy_hold_curve.iloc[-1] ** ann_factor - 1),
        f"{prefix}_mean_daily_strategy_return": float(signals["strategy_return"].mean()),
        f"{prefix}_mean_daily_strategy_return_net": float(signals["strategy_return_net"].mean()),
        f"{prefix}_strategy_sharpe_net": _annualized_sharpe(signals["strategy_return_net"]),
        f"{prefix}_buy_hold_sharpe": _annualized_sharpe(signals["target_next_return"]),
        f"{prefix}_strategy_max_drawdown_net": _max_drawdown(strategy_net_curve),
        f"{prefix}_buy_hold_max_drawdown": _max_drawdown(buy_hold_curve),
    }


def train_and_backtest(
    spy_path: str | Path = "data/raw/SPY.csv",
    speeches_path: str | Path = "data/raw/trump_speeches.csv",
    dataset_out: str | Path = "data/processed/model_dataset.csv",
    signals_out: str | Path = "outputs/signals.csv",
    metrics_out: str | Path = "outputs/metrics.json",
    min_train_days: int = 252,
    cost_bps: float = 1.0,
) -> dict[str, object]:
    data = build_dataset(spy_path, speeches_path)
    ensure_parent(dataset_out)
    data.to_csv(dataset_out, index=False)

    features = feature_columns(data)
    if len(data) <= min_train_days + 20:
        metrics = _empty_metrics(
            f"Need more than {min_train_days + 20} usable rows, got {len(data)}."
        )
        ensure_parent(metrics_out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2))
        return metrics

    predictions = []
    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )

    for i in range(min_train_days, len(data) - 1):
        train = data.iloc[:i]
        test = data.iloc[[i]]

        x_train = train[features]
        y_train = train["target_next_up"].astype(int)
        if y_train.nunique() < 2:
            continue

        model.fit(x_train, y_train)
        prob_up = float(model.predict_proba(test[features])[0, 1])
        signal = 1 if prob_up >= 0.55 else -1 if prob_up <= 0.45 else 0
        next_return = float(test["target_next_return"].iloc[0])
        strategy_return = signal * next_return

        predictions.append(
            {
                "date": test["date"].iloc[0],
                "prob_up": prob_up,
                "signal": signal,
                "speech_count": float(test.get("speech_count", pd.Series([0])).iloc[0]),
                "target_next_up": int(test["target_next_up"].iloc[0]),
                "target_next_return": next_return,
                "strategy_return": strategy_return,
            }
        )

    signals = pd.DataFrame(predictions)
    if not signals.empty:
        previous_signal = signals["signal"].shift(1).fillna(0)
        signals["turnover"] = (signals["signal"] - previous_signal).abs()
        signals["transaction_cost"] = signals["turnover"] * (cost_bps / 10_000)
        signals["strategy_return_net"] = signals["strategy_return"] - signals["transaction_cost"]
    ensure_parent(signals_out)
    signals.to_csv(signals_out, index=False)

    if signals.empty:
        metrics = _empty_metrics("No walk-forward predictions were generated.")
    else:
        first_event_date = signals.loc[signals["speech_count"] > 0, "date"].min()
        after_first_event = signals[signals["date"] >= first_event_date] if pd.notna(first_event_date) else signals.iloc[0:0]
        event_days = signals[signals["speech_count"] > 0]

        metrics = {
            "status": "ok",
            "rows": int(len(data)),
            "first_event_date": str(first_event_date.date()) if pd.notna(first_event_date) else None,
            "signal_thresholds": {"long": 0.55, "short": 0.45},
            "transaction_cost_bps": cost_bps,
            "features": features,
        }
        metrics.update(_signal_metrics(signals, "all_days"))
        metrics.update(_signal_metrics(after_first_event, "after_first_event"))
        metrics.update(_signal_metrics(event_days, "event_days"))

    ensure_parent(metrics_out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics


def compare_assets(
    tickers: list[str],
    start: str = "2015-01-01",
    speeches_path: str | Path = "data/raw/trump_speeches.csv",
    data_dir: str | Path = "data/raw",
    outputs_dir: str | Path = "outputs/assets",
    min_train_days: int = 252,
    cost_bps: float = 1.0,
    update: bool = False,
) -> pd.DataFrame:
    out_dir = Path(outputs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for ticker in tickers:
        symbol = ticker.upper()
        price_path = Path(data_dir) / f"{symbol}.csv"
        download_spy(ticker=symbol, start=start, cache_path=price_path, update=update)

        metrics = train_and_backtest(
            spy_path=price_path,
            speeches_path=speeches_path,
            dataset_out=out_dir / f"{symbol}_dataset.csv",
            signals_out=out_dir / f"{symbol}_signals.csv",
            metrics_out=out_dir / f"{symbol}_metrics.json",
            min_train_days=min_train_days,
            cost_bps=cost_bps,
        )
        row = {
            "ticker": symbol,
            "status": metrics.get("status"),
            "first_event_date": metrics.get("first_event_date"),
            "event_days_predictions": metrics.get("event_days_predictions"),
            "event_days_active_trades": metrics.get("event_days_active_trades"),
            "event_days_accuracy": metrics.get("event_days_accuracy"),
            "event_days_roc_auc": metrics.get("event_days_roc_auc"),
            "event_days_strategy_total_return_net": metrics.get("event_days_strategy_total_return_net"),
            "event_days_buy_hold_total_return": metrics.get("event_days_buy_hold_total_return"),
            "event_days_strategy_sharpe_net": metrics.get("event_days_strategy_sharpe_net"),
            "event_days_strategy_max_drawdown_net": metrics.get("event_days_strategy_max_drawdown_net"),
            "after_first_event_strategy_total_return_net": metrics.get(
                "after_first_event_strategy_total_return_net"
            ),
            "after_first_event_strategy_sharpe_net": metrics.get("after_first_event_strategy_sharpe_net"),
        }
        rows.append(row)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["event_days_roc_auc", "event_days_strategy_sharpe_net"],
            ascending=[False, False],
            na_position="last",
        )
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"Saved asset comparison: {summary_path}")
    return summary
