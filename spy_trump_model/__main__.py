from __future__ import annotations

import argparse

from .data import download_spy
from .model import train_and_backtest
from .scrape import fetch_trumpstruth_feed, fetch_truthsocial_posts, fetch_whitehouse_remarks


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
    archive.add_argument("--out", default="data/raw/trump_speeches.csv")

    train = sub.add_parser("train", help="Build features, train, and backtest.")
    train.add_argument("--spy", default="data/raw/SPY.csv")
    train.add_argument("--speeches", default="data/raw/trump_speeches.csv")
    train.add_argument("--dataset-out", default="data/processed/model_dataset.csv")
    train.add_argument("--signals-out", default="outputs/signals.csv")
    train.add_argument("--metrics-out", default="outputs/metrics.json")
    train.add_argument("--min-train-days", type=int, default=252)

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
        )


if __name__ == "__main__":
    main()
