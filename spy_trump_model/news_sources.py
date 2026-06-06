from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

from .data import ensure_parent
from .scrape import HEADERS


GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

DEFAULT_GDELT_QUERIES = {
    "trump_market": '"Donald Trump" (stock OR stocks OR market OR "Wall Street" OR Nasdaq OR "S&P")',
    "trump_trade_china": '"Donald Trump" (China OR Chinese OR tariff OR tariffs OR trade)',
    "trump_rates_inflation": '"Donald Trump" (inflation OR "Federal Reserve" OR Fed OR rates OR Powell)',
    "trump_energy_oil": '"Donald Trump" (oil OR energy OR gasoline OR drilling OR OPEC)',
    "trump_dollar_fx": '"Donald Trump" (dollar OR currency OR yuan OR euro OR yen)',
    "trump_geopolitics": '"Donald Trump" (war OR Ukraine OR Russia OR Iran OR Israel OR NATO)',
    "trump_tax_regulation": '"Donald Trump" (tax OR taxes OR regulation OR deregulation)',
    "trump_border_immigration": '"Donald Trump" (border OR immigration OR migrant OR deportation)',
}


def parse_gdelt_query_args(values: list[str] | None) -> dict[str, str] | None:
    if not values:
        return None

    parsed: dict[str, str] = {}
    for index, value in enumerate(values, start=1):
        if "=" in value:
            label, query = value.split("=", 1)
            label = label.strip().lower().replace("-", "_")
            query = query.strip()
        else:
            label = f"custom_{index}"
            query = value.strip()
        if not label or not query:
            raise ValueError("Custom GDELT queries must be non-empty strings or label=query pairs.")
        parsed[label] = query
    return parsed


def _read_existing_news(path: Path) -> pd.DataFrame:
    columns = ["date", "datetime", "title", "source", "source_type", "text"]
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=columns)


def _date_range_chunks(
    start_date: str,
    end_date: str,
    chunk_days: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start = pd.to_datetime(start_date, errors="raise").normalize()
    end = pd.to_datetime(end_date, errors="raise").normalize()
    if start > end:
        raise ValueError("start_date must be on or before end_date.")

    chunks = []
    current = start
    days = max(int(chunk_days), 1)
    while current <= end:
        chunk_end = min(current + pd.Timedelta(days=days - 1), end)
        chunks.append((current, chunk_end))
        current = chunk_end + pd.Timedelta(days=1)
    return chunks


def _gdelt_datetime(value: pd.Timestamp, end_of_day: bool = False) -> str:
    if end_of_day:
        value = value + pd.Timedelta(hours=23, minutes=59, seconds=59)
    return value.strftime("%Y%m%d%H%M%S")


def _parse_seen_date(value: object) -> pd.Timestamp | None:
    parsed = pd.to_datetime(str(value or ""), errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return parsed


def _article_to_row(article: dict[str, object], source_type: str) -> dict[str, str] | None:
    url = str(article.get("url") or "").strip()
    title = str(article.get("title") or "").strip()
    seen_date = _parse_seen_date(article.get("seendate"))
    if not url or not title or seen_date is None:
        return None

    domain = str(article.get("domain") or "").strip()
    text_parts = [title]
    if domain:
        text_parts.append(domain)

    return {
        "date": seen_date.tz_convert("America/New_York").date().isoformat(),
        "datetime": seen_date.isoformat(),
        "title": title,
        "source": url,
        "source_type": source_type,
        "text": ". ".join(text_parts),
    }


def _fetch_gdelt_chunk(
    query: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    max_records: int,
    sort: str,
) -> list[dict[str, object]]:
    params = {
        "query": f"{query} sourcelang:english",
        "mode": "ArtList",
        "format": "json",
        "maxrecords": min(max(int(max_records), 1), 250),
        "sort": sort,
        "startdatetime": _gdelt_datetime(start),
        "enddatetime": _gdelt_datetime(end, end_of_day=True),
    }
    response = requests.get(GDELT_DOC_API, params=params, headers=HEADERS, timeout=60)
    response.raise_for_status()
    payload = response.json()
    articles = payload.get("articles", []) if isinstance(payload, dict) else []
    return articles if isinstance(articles, list) else []


def fetch_gdelt_news(
    out_path: str | Path = "data/raw/news.csv",
    start_date: str | None = None,
    end_date: str | None = None,
    queries: dict[str, str] | None = None,
    chunk_days: int = 7,
    max_records: int = 100,
    sort: str = "datedesc",
    sleep_seconds: float = 1.0,
) -> pd.DataFrame:
    path = ensure_parent(out_path)
    existing = _read_existing_news(path)
    existing_sources = set(existing.get("source", pd.Series(dtype=str)).dropna().astype(str))

    if end_date is None:
        end_date = pd.Timestamp.utcnow().date().isoformat()
    if start_date is None:
        start_date = (pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=89)).date().isoformat()

    selected_queries = queries or DEFAULT_GDELT_QUERIES
    rows: list[dict[str, str]] = []
    scanned = 0

    for label, query in selected_queries.items():
        source_type = f"gdelt_{label}".lower().replace("-", "_")
        for start, end in _date_range_chunks(start_date=start_date, end_date=end_date, chunk_days=chunk_days):
            try:
                articles = _fetch_gdelt_chunk(
                    query=query,
                    start=start,
                    end=end,
                    max_records=max_records,
                    sort=sort,
                )
            except requests.RequestException as exc:
                print(f"Skipped GDELT {label} {start.date()}->{end.date()}: {exc}")
                continue
            scanned += len(articles)
            for article in articles:
                if not isinstance(article, dict):
                    continue
                row = _article_to_row(article, source_type=source_type)
                if row is None or row["source"] in existing_sources:
                    continue
                rows.append(row)
                existing_sources.add(row["source"])
            time.sleep(max(float(sleep_seconds), 0.0))

    if rows:
        combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    else:
        combined = existing

    if not combined.empty:
        combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.date.astype(str)
        combined = combined.dropna(subset=["text"])
        combined = combined.drop_duplicates(subset=["source"], keep="last")
        combined = combined.sort_values(["date", "source_type", "source"]).reset_index(drop=True)

    combined.to_csv(path, index=False)
    print(f"Saved GDELT news: {path} ({len(combined)} rows, +{len(rows)} new, {scanned} articles scanned)")
    return combined
