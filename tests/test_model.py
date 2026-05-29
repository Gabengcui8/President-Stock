from pathlib import Path

import pandas as pd

from spy_trump_model.model import train_and_backtest


def test_train_and_backtest_writes_metrics_and_signals(tmp_path: Path) -> None:
    spy_path = tmp_path / "SPY.csv"
    speeches_path = tmp_path / "speeches.csv"
    dataset_out = tmp_path / "dataset.csv"
    signals_out = tmp_path / "signals.csv"
    metrics_out = tmp_path / "metrics.json"

    dates = pd.bdate_range("2023-01-02", periods=90)
    close = []
    price = 100.0
    for i, _ in enumerate(dates):
        price *= 1.003 if i % 3 else 0.998
        close.append(price)

    pd.DataFrame(
        {
            "Date": dates,
            "Close": close,
            "Open": close,
            "High": [value * 1.01 for value in close],
            "Low": [value * 0.99 for value in close],
            "Volume": [1_000_000] * len(dates),
        }
    ).to_csv(spy_path, index=False)

    pd.DataFrame(
        [
            {
                "date": date.date().isoformat(),
                "datetime": f"{date.date().isoformat()}T14:00:00Z",
                "source": f"post-{idx}",
                "text": "Jobs tax tariff China market post.",
            }
            for idx, date in enumerate(dates[25:70:4])
        ]
    ).to_csv(speeches_path, index=False)

    metrics = train_and_backtest(
        spy_path=spy_path,
        speeches_path=speeches_path,
        dataset_out=dataset_out,
        signals_out=signals_out,
        metrics_out=metrics_out,
        min_train_days=30,
    )

    assert metrics["status"] == "ok"
    assert metrics["all_days_predictions"] > 0
    assert dataset_out.exists()
    assert signals_out.exists()
    assert metrics_out.exists()
