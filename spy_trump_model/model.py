from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import ensure_parent
from .features import build_dataset, feature_columns


def _empty_metrics(reason: str) -> dict[str, object]:
    return {"status": "not_enough_data", "reason": reason}


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
    buy_hold_curve = (1 + signals["target_next_return"]).cumprod()
    ann_factor = 252 / max(len(signals), 1)

    return {
        f"{prefix}_predictions": int(len(signals)),
        f"{prefix}_active_trades": int((signals["signal"] != 0).sum()),
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_roc_auc": float(roc_auc_score(y_true, y_score)) if y_true.nunique() == 2 else None,
        f"{prefix}_strategy_total_return": float(strategy_curve.iloc[-1] - 1),
        f"{prefix}_buy_hold_total_return": float(buy_hold_curve.iloc[-1] - 1),
        f"{prefix}_strategy_annualized_return": float(strategy_curve.iloc[-1] ** ann_factor - 1),
        f"{prefix}_buy_hold_annualized_return": float(buy_hold_curve.iloc[-1] ** ann_factor - 1),
        f"{prefix}_mean_daily_strategy_return": float(signals["strategy_return"].mean()),
    }


def train_and_backtest(
    spy_path: str | Path = "data/raw/SPY.csv",
    speeches_path: str | Path = "data/raw/trump_speeches.csv",
    dataset_out: str | Path = "data/processed/model_dataset.csv",
    signals_out: str | Path = "outputs/signals.csv",
    metrics_out: str | Path = "outputs/metrics.json",
    min_train_days: int = 252,
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
            "features": features,
        }
        metrics.update(_signal_metrics(signals, "all_days"))
        metrics.update(_signal_metrics(after_first_event, "after_first_event"))
        metrics.update(_signal_metrics(event_days, "event_days"))

    ensure_parent(metrics_out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metrics
