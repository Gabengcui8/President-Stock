from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .data import ensure_parent
from .scrape import HEADERS


GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
DEFAULT_MAX_ARTICLE_FETCHES = 50
UNLIMITED_ARTICLE_FETCHES = -1

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


def _clean_article_text(value: str) -> str:
    return " ".join(str(value or "").split())


def _extract_article_body(url: str, timeout: int = 20, max_chars: int = 12000) -> str:
    response = requests.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "lxml")
    for tag in soup(["script", "style", "noscript", "svg", "form", "nav", "footer", "header"]):
        tag.decompose()

    body = soup.find("article") or soup.find("main") or soup.body
    if body is None:
        return ""

    paragraphs = [p.get_text(" ", strip=True) for p in body.find_all("p")]
    if not paragraphs:
        paragraphs = [body.get_text(" ", strip=True)]
    text = _clean_article_text(" ".join(part for part in paragraphs if part))
    if max_chars > 0:
        text = text[:max_chars]
    return text


def _article_to_row(
    article: dict[str, object],
    source_type: str,
    fetch_article_text: bool = False,
    article_timeout: int = 20,
    max_article_chars: int = 12000,
) -> tuple[dict[str, str] | None, bool]:
    url = str(article.get("url") or "").strip()
    title = str(article.get("title") or "").strip()
    seen_date = _parse_seen_date(article.get("seendate"))
    if not url or not title or seen_date is None:
        return None, False

    domain = str(article.get("domain") or "").strip()
    text_parts: list[str] = []
    article_body = ""
    fetched_article_text = False
    if fetch_article_text:
        fetched_article_text = True
        try:
            article_body = _extract_article_body(
                url,
                timeout=article_timeout,
                max_chars=max_article_chars,
            )
        except requests.RequestException as exc:
            print(f"Article text fallback for {url}: {exc}")
    if article_body:
        text_parts.append(article_body)
    else:
        text_parts.append(title)
    if domain:
        text_parts.append(domain)

    return {
        "date": seen_date.tz_convert("America/New_York").date().isoformat(),
        "datetime": seen_date.isoformat(),
        "title": title,
        "source": url,
        "source_type": source_type,
        "text": ". ".join(text_parts),
    }, fetched_article_text


def _article_fetch_allowed(fetch_article_text: bool, max_article_fetches: int, fetched: int) -> bool:
    if not fetch_article_text:
        return False
    if max_article_fetches == UNLIMITED_ARTICLE_FETCHES:
        return True
    return fetched < max(max_article_fetches, 0)


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
    fetch_article_text: bool = False,
    article_timeout: int = 20,
    max_article_chars: int = 12000,
    max_article_fetches: int = DEFAULT_MAX_ARTICLE_FETCHES,
    sleep_seconds: float = 1.0,
    article_sleep_seconds: float = 1.5,
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
    article_fetches = 0

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
                should_fetch_article = _article_fetch_allowed(
                    fetch_article_text=fetch_article_text,
                    max_article_fetches=max_article_fetches,
                    fetched=article_fetches,
                )
                row, attempted_article_fetch = _article_to_row(
                    article,
                    source_type=source_type,
                    fetch_article_text=should_fetch_article,
                    article_timeout=article_timeout,
                    max_article_chars=max_article_chars,
                )
                if attempted_article_fetch:
                    article_fetches += 1
                if row is None or row["source"] in existing_sources:
                    continue
                rows.append(row)
                existing_sources.add(row["source"])
                if attempted_article_fetch:
                    time.sleep(max(float(article_sleep_seconds), 0.0))
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
    print(
        f"Saved GDELT news: {path} ({len(combined)} rows, +{len(rows)} new, "
        f"{scanned} articles scanned, {article_fetches} article URLs fetched)"
    )
    return combined
