from __future__ import annotations

import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .data import ensure_parent


HEADERS = {
    "Accept": "application/json, text/html;q=0.9,*/*;q=0.8",
    "User-Agent": "Mozilla/5.0 compatible; market research bot; contact: local",
}

TRUTH_SOCIAL_BASE = "https://truthsocial.com"


def _normalize_date_column(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    return parsed.dt.tz_convert("America/New_York").dt.date.astype(str)


def _get(url: str) -> BeautifulSoup:
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return BeautifulSoup(response.text, "lxml")


def _get_json(url: str, params: dict[str, object] | None = None) -> object:
    response = requests.get(url, params=params, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.json()


def _page_url(base_url: str, page: int) -> str:
    if page <= 1:
        return base_url
    separator = "&" if "?" in base_url else "?"
    return f"{base_url}{separator}query-10-page={page}"


ALLOWED_PATH_MARKERS = ("/remarks/", "/videos/")


def _is_relevant_whitehouse_link(href: str, text: str) -> bool:
    if "whitehouse.gov/" not in href:
        return False
    if not any(marker in href for marker in ALLOWED_PATH_MARKERS):
        return False
    lowered = text.lower()
    return any(token in lowered for token in ["trump", "president", "remarks"])


def _extract_links(base_url: str, pages: int) -> list[str]:
    links: list[str] = []
    seen: set[str] = set()

    for page in range(1, pages + 1):
        soup = _get(_page_url(base_url, page))
        for anchor in soup.select("a[href]"):
            href = urljoin(base_url, anchor["href"])
            text = anchor.get_text(" ", strip=True).lower()
            if not _is_relevant_whitehouse_link(href, text):
                continue
            if href not in seen:
                links.append(href)
                seen.add(href)
        time.sleep(0.5)

    return links


def _extract_article(url: str) -> dict[str, str] | None:
    soup = _get(url)
    title = soup.find(["h1", "h2"])
    time_tag = soup.find("time")
    body = soup.find("main") or soup.find("article") or soup.body
    if body is None:
        return None

    title_text = title.get_text(" ", strip=True) if title else ""
    paragraphs = [p.get_text(" ", strip=True) for p in body.find_all("p")]
    text_parts = [title_text, *paragraphs]
    text = "\n".join(p for p in text_parts if p)
    if not text:
        return None

    date_value = ""
    if time_tag:
        date_value = time_tag.get("datetime") or time_tag.get_text(" ", strip=True)

    return {
        "date": date_value,
        "title": title_text,
        "source": url,
        "text": text,
    }


def fetch_whitehouse_remarks(
    pages: int = 3,
    out_path: str | Path = "data/raw/trump_speeches.csv",
    base_url: str = "https://www.whitehouse.gov/videos/?query-inherit-playlist_term=remarks-from-president-trump",
) -> pd.DataFrame:
    path = ensure_parent(out_path)
    existing = pd.DataFrame(columns=["date", "title", "source", "text"])
    if path.exists():
        existing = pd.read_csv(path)

    existing_sources = set(existing.get("source", pd.Series(dtype=str)).dropna().astype(str))
    rows = []

    links = _extract_links(base_url, pages)
    print(f"Found {len(links)} candidate White House links")

    for link in links:
        if link in existing_sources:
            continue
        try:
            article = _extract_article(link)
        except requests.RequestException as exc:
            print(f"Skipped {link}: {exc}")
            continue
        if article:
            rows.append(article)
        time.sleep(0.5)

    if rows:
        fetched = pd.DataFrame(rows)
        combined = pd.concat([existing, fetched], ignore_index=True)
    else:
        combined = existing

    if not combined.empty:
        combined["date"] = _normalize_date_column(combined["date"])
        combined = combined.dropna(subset=["text"])
        combined = combined.drop_duplicates(subset=["source"], keep="last")
        combined = combined.sort_values(["date", "source"])

    combined.to_csv(path, index=False)
    print(f"Saved speeches: {path} ({len(combined)} rows, +{len(rows)} new)")
    return combined


def _truthsocial_account_id(handle: str) -> str:
    handle = handle.lstrip("@")
    data = _get_json(
        f"{TRUTH_SOCIAL_BASE}/api/v1/accounts/lookup",
        params={"acct": handle},
    )
    if not isinstance(data, dict) or not data.get("id"):
        raise RuntimeError(f"Could not find Truth Social account: @{handle}")
    return str(data["id"])


def _clean_status_html(value: str) -> str:
    return BeautifulSoup(value or "", "lxml").get_text(" ", strip=True)


def _truthsocial_status_to_row(status: dict[str, object], handle: str) -> dict[str, str] | None:
    created_at = str(status.get("created_at") or "")
    status_id = str(status.get("id") or "")
    uri = str(status.get("uri") or "")
    url = str(status.get("url") or uri or f"{TRUTH_SOCIAL_BASE}/@{handle}/posts/{status_id}")
    text = _clean_status_html(str(status.get("content") or ""))

    reblog = status.get("reblog")
    if not text and isinstance(reblog, dict):
        text = _clean_status_html(str(reblog.get("content") or ""))

    if not created_at or not status_id or not text:
        return None

    date_value = pd.to_datetime(created_at, utc=True, errors="coerce")
    if pd.isna(date_value):
        return None

    ny_date = date_value.tz_convert("America/New_York").date().isoformat()
    return {
        "date": ny_date,
        "datetime": created_at,
        "title": f"Truth Social @{handle}",
        "source": url,
        "source_type": "truthsocial",
        "text": text,
    }


def fetch_truthsocial_posts(
    handle: str = "realDonaldTrump",
    max_pages: int = 5,
    limit: int = 40,
    out_path: str | Path = "data/raw/trump_speeches.csv",
) -> pd.DataFrame:
    path = ensure_parent(out_path)
    existing = pd.DataFrame(columns=["date", "datetime", "title", "source", "source_type", "text"])
    if path.exists():
        existing = pd.read_csv(path)

    handle = handle.lstrip("@")
    account_id = _truthsocial_account_id(handle)
    existing_sources = set(existing.get("source", pd.Series(dtype=str)).dropna().astype(str))
    rows: list[dict[str, str]] = []
    max_id: str | None = None

    for _ in range(max_pages):
        params: dict[str, object] = {
            "limit": min(max(limit, 1), 40),
            "exclude_replies": "false",
        }
        if max_id:
            params["max_id"] = max_id

        data = _get_json(
            f"{TRUTH_SOCIAL_BASE}/api/v1/accounts/{account_id}/statuses",
            params=params,
        )
        if not isinstance(data, list) or not data:
            break

        for status in data:
            if not isinstance(status, dict):
                continue
            row = _truthsocial_status_to_row(status, handle)
            if row and row["source"] not in existing_sources:
                rows.append(row)
                existing_sources.add(row["source"])

        last_id = str(data[-1].get("id") or "")
        if not last_id or last_id == max_id:
            break
        max_id = last_id
        time.sleep(0.5)

    if not rows:
        print("No new Truth Social posts fetched.")
        combined = existing
    else:
        combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)

    if combined.empty:
        raise RuntimeError(
            "Truth Social fetch returned no usable posts. The public endpoint may be blocked or changed."
        )

    combined["date"] = _normalize_date_column(combined["date"])
    combined = combined.dropna(subset=["text"])
    combined = combined.drop_duplicates(subset=["source"], keep="last")
    combined = combined.sort_values(["date", "source"])
    combined.to_csv(path, index=False)
    print(f"Saved posts: {path} ({len(combined)} rows, +{len(rows)} new)")
    return combined
