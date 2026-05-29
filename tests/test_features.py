from pathlib import Path

import pandas as pd

from spy_trump_model.features import build_dataset, load_speeches


def test_load_speeches_normalizes_mixed_timezones(tmp_path: Path) -> None:
    speeches_path = tmp_path / "speeches.csv"
    pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "datetime": "2024-01-02T15:00:00Z",
                "text": "Tariff comments before market close.",
            },
            {
                "date": "2024-01-03T20:30:00-05:00",
                "datetime": "2024-01-04T01:30:00Z",
                "text": "China and rates after hours.",
            },
        ]
    ).to_csv(speeches_path, index=False)

    speeches = load_speeches(speeches_path)

    assert len(speeches) == 2
    assert speeches["date"].dt.tz is None
    assert speeches["date"].tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert "datetime_et" in speeches.columns


def test_build_dataset_aligns_afterhours_to_next_trading_day(tmp_path: Path) -> None:
    spy_path = tmp_path / "SPY.csv"
    speeches_path = tmp_path / "speeches.csv"

    pd.DataFrame(
        {
            "Date": pd.bdate_range("2024-01-02", periods=40),
            "Close": [100 + i for i in range(40)],
            "Open": [100 + i for i in range(40)],
            "High": [101 + i for i in range(40)],
            "Low": [99 + i for i in range(40)],
            "Volume": [1_000_000] * 40,
        }
    ).to_csv(spy_path, index=False)

    pd.DataFrame(
        [
            {
                "date": "2024-02-09",
                "datetime": "2024-02-09T22:30:00Z",
                "source": "afterhours",
                "text": "After hours tariff and China post.",
            },
            {
                "date": "2024-02-10",
                "datetime": "2024-02-10T15:00:00Z",
                "source": "weekend",
                "text": "Weekend oil and border post.",
            },
        ]
    ).to_csv(speeches_path, index=False)

    data = build_dataset(spy_path, speeches_path)
    monday = pd.Timestamp("2024-02-12")
    monday_row = data.loc[data["date"] == monday].iloc[0]

    assert monday_row["speech_count"] == 2
    assert monday_row["speech_afterhours_count"] == 1
    assert monday_row["speech_weekend_count"] == 1
    assert monday_row["kw_tariff"] == 1
    assert monday_row["kw_oil"] == 1
