from __future__ import annotations

import argparse

from .data import download_spy
from .keyword_impact import (
    DEFAULT_COST_BPS,
    DEFAULT_MIN_ABS_T_STAT,
    DEFAULT_MIN_INDEPENDENT_EVENTS,
    DEFAULT_MIN_ROBUST_TEST_INDEPENDENT_EVENTS,
    DEFAULT_MIN_ROBUST_TRAIN_INDEPENDENT_EVENTS,
    DEFAULT_MIN_TEST_INDEPENDENT_EVENTS,
    keyword_impact_report,
)
from .model import compare_assets, train_and_backtest
from .paper_ledger import DEFAULT_PAPER_HORIZONS, DEFAULT_PAPER_TICKERS, build_paper_ledger
from .scrape import fetch_trumpstruth_feed, fetch_truthsocial_posts, fetch_whitehouse_remarks
from .signal_expansion import DEFAULT_MARKET_TICKERS, build_expanded_signals


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spy_trump_model",
        description="Research model linking Trump remarks to next-day SPY returns.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    spy = sub.add_parser("download-spy", help="Download or reuse cached SPY data.")
    spy.add_argument("--ticker", default="SPY")
    spy.add_argument("--start", default="2015-01-01")
    spy.add_argument("--cache", default="data/raw/SPY.csv")
    spy.add_argument("--update", action="store_true", help="Refresh the cached file.")

    wh = sub.add_parser("fetch-whitehouse", help="Fetch public White House remarks.")
    wh.add_argument("--pages", type=int, default=3)
    wh.add_argument("--out", default="data/raw/trump_speeches.csv")
    wh.add_argument(
        "--base-url",
        default="https://www.whitehouse.gov/videos/?query-inherit-playlist_term=remarks-from-president-trump",
    )

    truth = sub.add_parser("fetch-truthsocial", help="Fetch public Truth Social posts.")
    truth.add_argument("--handle", default="realDonaldTrump")
    truth.add_argument("--max-pages", type=int, default=5)
    truth.add_argument("--limit", type=int, default=40)
    truth.add_argument("--out", default="data/raw/trump_speeches.csv")

    archive = sub.add_parser("fetch-trumpstruth", help="Fetch archived Truth Social posts via Trump's Truth RSS.")
    archive.add_argument("--start-date", default="2022-02-01")
    archive.add_argument("--end-date", default=None)
    archive.add_argument("--chunk-days", type=int, default=31)
    archive.add_argument("--out", default="data/raw/trump_speeches.csv")

    train = sub.add_parser("train", help="Build features, train, and backtest.")
    train.add_argument("--spy", default="data/raw/SPY.csv")
    train.add_argument("--speeches", default="data/raw/trump_speeches.csv")
    train.add_argument("--dataset-out", default="data/processed/model_dataset.csv")
    train.add_argument("--signals-out", default="outputs/signals.csv")
    train.add_argument("--metrics-out", default="outputs/metrics.json")
    train.add_argument("--min-train-days", type=int, default=252)
    train.add_argument("--cost-bps", type=float, default=1.0)

    compare = sub.add_parser("compare-assets", help="Train the same text model across multiple assets.")
    compare.add_argument(
        "--tickers",
        nargs="+",
        default=["SPY", "QQQ", "XLE", "XLI", "XLF", "SMH", "FXI", "TLT", "USO", "GLD"],
    )
    compare.add_argument("--start", default="2015-01-01")
    compare.add_argument("--speeches", default="data/raw/trump_speeches.csv")
    compare.add_argument("--data-dir", default="data/raw")
    compare.add_argument("--outputs-dir", default="outputs/assets")
    compare.add_argument("--min-train-days", type=int, default=252)
    compare.add_argument("--cost-bps", type=float, default=1.0)
    compare.add_argument("--update", action="store_true")

    impact = sub.add_parser(
        "keyword-impact",
        help="Learn keyword/asset associations on train data and test them out of sample.",
    )
    impact.add_argument(
        "--tickers",
        nargs="+",
        default=["SPY", "QQQ", "XLE", "XLI", "XLF", "SMH", "FXI", "TLT", "USO", "GLD"],
    )
    impact.add_argument("--start", default="2015-01-01")
    impact.add_argument("--speeches", default="data/raw/trump_speeches.csv")
    impact.add_argument("--data-dir", default="data/raw")
    impact.add_argument("--outputs-dir", default="outputs/keyword_impact")
    impact.add_argument("--analysis-start", default="2021-01-01")
    impact.add_argument("--split-date", default=None)
    impact.add_argument("--train-fraction", type=float, default=0.7)
    impact.add_argument("--min-keyword-days", type=int, default=20)
    impact.add_argument("--min-independent-events", type=int, default=DEFAULT_MIN_INDEPENDENT_EVENTS)
    impact.add_argument(
        "--min-test-independent-events",
        type=int,
        default=DEFAULT_MIN_TEST_INDEPENDENT_EVENTS,
    )
    impact.add_argument(
        "--min-robust-train-independent-events",
        type=int,
        default=DEFAULT_MIN_ROBUST_TRAIN_INDEPENDENT_EVENTS,
    )
    impact.add_argument(
        "--min-robust-test-independent-events",
        type=int,
        default=DEFAULT_MIN_ROBUST_TEST_INDEPENDENT_EVENTS,
    )
    impact.add_argument("--horizon-days", type=int, default=1)
    impact.add_argument("--horizons", nargs="+", type=int, default=None)
    impact.add_argument("--min-abs-t-stat", type=float, default=DEFAULT_MIN_ABS_T_STAT)
    impact.add_argument("--allowed-direction", choices=["all", "long", "short"], default="all")
    impact.add_argument("--stability-mode", choices=["event", "calendar"], default="event")
    impact.add_argument("--no-stability-filter", action="store_true")
    impact.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    impact.add_argument("--include-unknown-time", action="store_true")
    impact.add_argument("--update", action="store_true")

    expansion = sub.add_parser(
        "signal-expansion",
        help="Build continuous theme, source, sentiment, and market-state signals.",
    )
    expansion.add_argument(
        "--texts",
        nargs="+",
        default=["data/raw/trump_speeches.csv"],
        help="One or more CSV text sources with date and text columns.",
    )
    expansion.add_argument("--out", default="data/processed/expanded_signals.csv")
    expansion.add_argument("--data-dir", default="data/raw")
    expansion.add_argument("--start", default="2021-01-01")
    expansion.add_argument("--market-tickers", nargs="+", default=DEFAULT_MARKET_TICKERS)
    expansion.add_argument("--no-market-state", action="store_true")
    expansion.add_argument("--update", action="store_true")

    paper = sub.add_parser(
        "paper-ledger",
        help="Create a forward observation ledger from expanded signals and cached prices.",
    )
    paper.add_argument("--signals", default="data/processed/expanded_signals.csv")
    paper.add_argument("--out", default="outputs/paper_trading/ledger.csv")
    paper.add_argument("--summary-out", default="outputs/paper_trading/summary.csv")
    paper.add_argument("--data-dir", default="data/raw")
    paper.add_argument("--tickers", nargs="+", default=DEFAULT_PAPER_TICKERS)
    paper.add_argument("--horizons", nargs="+", type=int, default=DEFAULT_PAPER_HORIZONS)
    paper.add_argument("--start", default=None)
    paper.add_argument(
        "--entry-lag-days",
        type=int,
        default=1,
        help="Trading-day lag from observation date to evaluation entry close.",
    )
    paper.add_argument("--all-days", action="store_true", help="Include rows without text observations.")
    paper.add_argument("--update", action="store_true")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "download-spy":
        download_spy(
            ticker=args.ticker,
            start=args.start,
            cache_path=args.cache,
            update=args.update,
        )
    elif args.command == "fetch-whitehouse":
        fetch_whitehouse_remarks(
            pages=args.pages,
            out_path=args.out,
            base_url=args.base_url,
        )
    elif args.command == "fetch-truthsocial":
        fetch_truthsocial_posts(
            handle=args.handle,
            max_pages=args.max_pages,
            limit=args.limit,
            out_path=args.out,
        )
    elif args.command == "fetch-trumpstruth":
        fetch_trumpstruth_feed(
            start_date=args.start_date,
            end_date=args.end_date,
            chunk_days=args.chunk_days,
            out_path=args.out,
        )
    elif args.command == "train":
        train_and_backtest(
            spy_path=args.spy,
            speeches_path=args.speeches,
            dataset_out=args.dataset_out,
            signals_out=args.signals_out,
            metrics_out=args.metrics_out,
            min_train_days=args.min_train_days,
            cost_bps=args.cost_bps,
        )
    elif args.command == "compare-assets":
        compare_assets(
            tickers=args.tickers,
            start=args.start,
            speeches_path=args.speeches,
            data_dir=args.data_dir,
            outputs_dir=args.outputs_dir,
            min_train_days=args.min_train_days,
            cost_bps=args.cost_bps,
            update=args.update,
        )
    elif args.command == "keyword-impact":
        keyword_impact_report(
            tickers=args.tickers,
            start=args.start,
            speeches_path=args.speeches,
            data_dir=args.data_dir,
            outputs_dir=args.outputs_dir,
            analysis_start=args.analysis_start,
            split_date=args.split_date,
            train_fraction=args.train_fraction,
            min_keyword_days=args.min_keyword_days,
            min_independent_events=args.min_independent_events,
            min_test_independent_events=args.min_test_independent_events,
            min_robust_train_independent_events=args.min_robust_train_independent_events,
            min_robust_test_independent_events=args.min_robust_test_independent_events,
            horizon_days=args.horizon_days,
            horizons=args.horizons,
            min_abs_t_stat=args.min_abs_t_stat,
            require_stable_direction=not args.no_stability_filter,
            stability_mode=args.stability_mode,
            allowed_direction=args.allowed_direction,
            cost_bps=args.cost_bps,
            include_unknown_time=args.include_unknown_time,
            update=args.update,
        )
    elif args.command == "signal-expansion":
        build_expanded_signals(
            text_paths=args.texts,
            out_path=args.out,
            data_dir=args.data_dir,
            start=args.start,
            market_tickers=args.market_tickers,
            include_market_state=not args.no_market_state,
            update=args.update,
        )
    elif args.command == "paper-ledger":
        build_paper_ledger(
            signals_path=args.signals,
            out_path=args.out,
            summary_out=args.summary_out,
            data_dir=args.data_dir,
            tickers=args.tickers,
            horizons=args.horizons,
            start=args.start,
            entry_lag_days=args.entry_lag_days,
            text_only=not args.all_days,
            update=args.update,
        )


if __name__ == "__main__":
    main()
