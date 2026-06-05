from pathlib import Path

import pandas as pd

from spy_trump_model.paper_ledger import build_paper_ledger


def _write_price_cache(path: Path, dates: pd.DatetimeIndex, start_close: float) -> None:
    close = [start_close + i for i in range(len(dates))]
    pd.DataFrame(
        {
            "Date": dates,
            "Close": close,
            "Open": close,
            "High": [value * 1.01 for value in close],
            "Low": [value * 0.99 for value in close],
            "Volume": [1_000_000] * len(dates),
        }
    ).to_csv(path, index=False)


def test_paper_ledger_builds_text_observation_forward_returns(tmp_path: Path) -> None:
    signals_path = tmp_path / "expanded.csv"
    ledger_path = tmp_path / "ledger.csv"
    summary_path = tmp_path / "summary.csv"
    dates = pd.bdate_range("2024-01-02", periods=8)

    pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "text_item_count": 0,
                "theme_trade_china_intensity": 0.0,
                "theme_rates_inflation_intensity": 0.0,
            },
            {
                "date": "2024-01-03",
                "text_item_count": 2,
                "word_count": 50,
                "theme_trade_china_intensity": 4.0,
                "theme_rates_inflation_intensity": 1.0,
            },
            {
                "date": "2024-01-05",
                "text_item_count": 1,
                "word_count": 40,
                "theme_trade_china_intensity": 0.0,
                "theme_rates_inflation_intensity": 3.0,
            },
        ]
    ).to_csv(signals_path, index=False)
    _write_price_cache(tmp_path / "SPY.csv", dates=dates, start_close=100)
    _write_price_cache(tmp_path / "GLD.csv", dates=dates, start_close=200)

    ledger = build_paper_ledger(
        signals_path=signals_path,
        out_path=ledger_path,
        summary_out=summary_path,
        data_dir=tmp_path,
        tickers=["SPY", "GLD"],
        horizons=[1, 3],
        entry_lag_days=1,
        text_only=True,
        update=False,
    )

    assert ledger_path.exists()
    assert summary_path.exists()
    assert len(ledger) == 2
    assert list(ledger["observation_id"]) == [1, 2]
    assert list(ledger["top_theme"]) == ["trade_china", "rates_inflation"]

    jan3 = ledger.loc[ledger["date"] == pd.Timestamp("2024-01-03")].iloc[0]
    assert jan3["spy_entry_date"] == pd.Timestamp("2024-01-04")
    assert jan3["spy_exit_date_1d"] == pd.Timestamp("2024-01-05")
    assert jan3["spy_return_1d"] == (103 / 102) - 1
    assert jan3["gld_return_3d"] == (205 / 202) - 1

    summary = pd.read_csv(summary_path)
    assert set(summary["ticker"]) == {"SPY", "GLD"}
    assert set(summary["horizon_days"]) == {1, 3}
    assert summary["entry_lag_days"].eq(1).all()
