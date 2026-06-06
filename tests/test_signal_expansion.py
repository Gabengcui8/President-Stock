from pathlib import Path

import pandas as pd

from spy_trump_model.signal_expansion import build_expanded_signals


def test_signal_expansion_builds_continuous_theme_features(tmp_path: Path) -> None:
    trump_path = tmp_path / "trump.csv"
    news_path = tmp_path / "news.csv"
    out_path = tmp_path / "expanded.csv"

    pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "datetime": "2024-01-02T14:00:00Z",
                "source_type": "truth",
                "title": "Tariff plan",
                "text": "China tariff trade policy will protect manufacturing.",
            },
            {
                "date": "2024-01-03",
                "datetime": "2024-01-03T14:00:00Z",
                "source_type": "truth",
                "title": "Rates",
                "text": "Inflation and Federal Reserve rates remain a problem.",
            },
        ]
    ).to_csv(trump_path, index=False)
    pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "source_type": "news",
                "title": "Markets watch China trade tensions",
                "text": "Investors discuss tariffs and supply chain risk.",
            }
        ]
    ).to_csv(news_path, index=False)

    expanded = build_expanded_signals(
        text_paths=[trump_path, news_path],
        out_path=out_path,
        start="2024-01-01",
        include_market_state=False,
        text_vector_features=16,
    )

    assert out_path.exists()
    assert "theme_trade_china_intensity" in expanded.columns
    assert "theme_rates_inflation_intensity" in expanded.columns
    assert "theme_trade_china_sentiment" in expanded.columns
    assert "whole_text_vec_000" in expanded.columns
    assert "whole_text_vec_015" in expanded.columns
    assert "source_truth_count" in expanded.columns
    assert "source_news_count" in expanded.columns

    jan2 = expanded.loc[expanded["date"] == pd.Timestamp("2024-01-02")].iloc[0]
    jan3 = expanded.loc[expanded["date"] == pd.Timestamp("2024-01-03")].iloc[0]
    assert jan2["theme_trade_china_mentions"] > 0
    assert jan2.filter(like="whole_text_vec_").sum() > 0
    assert jan2["source_truth_count"] == 1
    assert jan2["source_news_count"] == 1
    assert jan3["theme_rates_inflation_mentions"] > 0


def test_signal_expansion_merges_cached_market_state(tmp_path: Path) -> None:
    text_path = tmp_path / "trump.csv"
    out_path = tmp_path / "expanded.csv"
    dates = pd.bdate_range("2023-12-15", periods=35)

    pd.DataFrame(
        {
            "Date": dates,
            "Close": [100 + i for i in range(len(dates))],
            "Open": [100 + i for i in range(len(dates))],
            "High": [101 + i for i in range(len(dates))],
            "Low": [99 + i for i in range(len(dates))],
            "Volume": [1_000_000] * len(dates),
        }
    ).to_csv(tmp_path / "SPY.csv", index=False)
    pd.DataFrame(
        [
            {
                "date": "2024-01-03",
                "text": "Dollar and tariff comments.",
            }
        ]
    ).to_csv(text_path, index=False)

    expanded = build_expanded_signals(
        text_paths=[text_path],
        out_path=out_path,
        data_dir=tmp_path,
        start="2024-01-01",
        market_tickers=["SPY"],
        include_market_state=True,
        update=False,
    )

    assert expanded["date"].min() >= pd.Timestamp("2024-01-01")
    assert "mkt_spy_return_1d" in expanded.columns
    assert "mkt_spy_vol_20d" in expanded.columns
    assert expanded["text_item_count"].fillna(0).max() == 1
