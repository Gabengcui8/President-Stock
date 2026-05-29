from pathlib import Path

import pandas as pd

from spy_trump_model.keyword_impact import keyword_impact_report


def test_keyword_impact_uses_non_overlapping_time_split(tmp_path: Path) -> None:
    price_path = tmp_path / "SPY.csv"
    speeches_path = tmp_path / "speeches.csv"
    out_dir = tmp_path / "keyword_impact"

    dates = pd.bdate_range("2023-01-02", periods=130)
    close = [100 + i * 0.2 + (1 if i % 7 == 0 else 0) for i in range(len(dates))]
    pd.DataFrame(
        {
            "Date": dates,
            "Close": close,
            "Open": close,
            "High": [value * 1.01 for value in close],
            "Low": [value * 0.99 for value in close],
            "Volume": [1_000_000] * len(dates),
        }
    ).to_csv(price_path, index=False)

    post_rows = []
    for idx, date in enumerate(dates[30:110:5]):
        post_rows.append(
            {
                "date": date.date().isoformat(),
                "datetime": f"{date.date().isoformat()}T14:00:00Z",
                "source": f"post-{idx}",
                "text": "tariff china jobs policy",
            }
        )
    pd.DataFrame(post_rows).to_csv(speeches_path, index=False)

    report = keyword_impact_report(
        tickers=["SPY"],
        speeches_path=speeches_path,
        data_dir=tmp_path,
        outputs_dir=out_dir,
        train_fraction=0.65,
        min_keyword_days=2,
        cost_bps=1.0,
    )

    splits = pd.read_csv(out_dir / "splits.csv", parse_dates=["train_end", "test_start"])
    assert not splits["overlap"].any()
    assert splits["train_end"].iloc[0] < splits["test_start"].iloc[0]
    assert (out_dir / "summary.csv").exists()
    assert "tariff" in set(report["keyword"])
    assert report.loc[report["keyword"] == "tariff", "selected_from_train"].iloc[0]
