from pathlib import Path

import pandas as pd

from spy_trump_model.keyword_impact import DEFAULT_MIN_ABS_T_STAT, keyword_impact_report


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
        analysis_start="2023-01-02",
        train_fraction=0.65,
        min_keyword_days=2,
        min_independent_events=2,
        cost_bps=1.0,
    )

    splits = pd.read_csv(out_dir / "splits.csv", parse_dates=["train_end", "test_start"])
    assert not splits["overlap"].any()
    assert splits["analysis_start"].iloc[0] == "2023-01-02"
    assert splits["min_abs_t_stat"].iloc[0] == DEFAULT_MIN_ABS_T_STAT
    assert splits["train_end"].iloc[0] < splits["test_start"].iloc[0]
    assert (out_dir / "summary.csv").exists()
    assert (out_dir / "selected.csv").exists()
    assert (out_dir / "robust_selected.csv").exists()
    assert "tariff" in set(report["signal"])
    assert "abs_train_t_stat" in report.columns
    assert "low_test_sample" in report.columns
    assert "train_independent_events" in report.columns
    assert "test_independent_events" in report.columns
    assert report.loc[report["signal"] == "tariff", "selected_from_train"].any()
    assert "theme_trade" in set(report["signal"])
    assert {"all", "low", "medium", "high"} & set(report["vol_regime"])
    assert set(report["stability_mode"]) == {"event"}


def test_keyword_impact_counts_non_overlapping_independent_events(tmp_path: Path) -> None:
    price_path = tmp_path / "SPY.csv"
    speeches_path = tmp_path / "speeches.csv"
    out_dir = tmp_path / "keyword_impact"

    dates = pd.bdate_range("2023-01-02", periods=80)
    close = [100 + i for i in range(len(dates))]
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

    pd.DataFrame(
        [
            {
                "date": date.date().isoformat(),
                "datetime": f"{date.date().isoformat()}T14:00:00Z",
                "source": f"post-{idx}",
                "text": "tariff policy",
            }
            for idx, date in enumerate(dates[30:35])
        ]
    ).to_csv(speeches_path, index=False)

    report = keyword_impact_report(
        tickers=["SPY"],
        speeches_path=speeches_path,
        data_dir=tmp_path,
        outputs_dir=out_dir,
        analysis_start="2023-01-02",
        train_fraction=0.75,
        min_keyword_days=2,
        min_independent_events=2,
        horizon_days=3,
        min_abs_t_stat=0,
        cost_bps=1.0,
    )

    tariff = report[(report["signal"] == "tariff") & (report["vol_regime"] == "all")].iloc[0]
    assert tariff["train_keyword_days"] == 5
    assert tariff["train_independent_events"] == 2
