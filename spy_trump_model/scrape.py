from __future__ import annotations

import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .data import ensure_parent


HEADERS = {
    "User-Agent": "Mozilla/5.0 compatible; research bot; contact: local",
}


def _get(url: str) -> BeautifulSoup:
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return BeautifulSoup(response.text, "lxml")


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
        combined["date"] = pd.to_datetime(combined["date"], errors="coerce").dt.date.astype(str)
        combined = combined.dropna(subset=["text"])
        combined = combined.drop_duplicates(subset=["source"], keep="last")
        combined = combined.sort_values(["date", "source"])

    combined.to_csv(path, index=False)
    print(f"Saved speeches: {path} ({len(combined)} rows, +{len(rows)} new)")
    return combined
