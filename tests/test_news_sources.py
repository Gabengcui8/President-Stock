from pathlib import Path

import pandas as pd
import requests

from spy_trump_model.news_sources import fetch_gdelt_news, parse_gdelt_query_args


class _FakeResponse:
    def __init__(
        self,
        payload: dict[str, object],
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            error = requests.HTTPError(f"{self.status_code} Client Error")
            error.response = self
            raise error
        return None

    def json(self) -> dict[str, object]:
        return self._payload


def test_fetch_gdelt_news_writes_expansion_ready_csv(tmp_path: Path, monkeypatch) -> None:
    out_path = tmp_path / "news.csv"
    calls: list[dict[str, object]] = []
    article_calls: list[str] = []

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append(dict(params or {}))
        return _FakeResponse(
            {
                "articles": [
                    {
                        "url": "https://example.com/markets/trump-tariff",
                        "title": "Markets weigh Trump tariff comments",
                        "seendate": "20240102T143000Z",
                        "domain": "example.com",
                    },
                    {
                        "url": "https://example.com/markets/trump-fed",
                        "title": "Trump criticizes Fed over inflation",
                        "seendate": "20240103T153000Z",
                        "domain": "example.com",
                    },
                ]
            }
        )

    monkeypatch.setattr("spy_trump_model.news_sources.requests.get", fake_get)

    class _ArticleResponse(_FakeResponse):
        @property
        def text(self) -> str:
            return str(self._payload["html"])

    def fake_get_with_article(url, params=None, headers=None, timeout=None):
        if "api.gdeltproject.org" not in url:
            article_calls.append(url)
            return _ArticleResponse(
                {
                    "html": "<html><article><p>Full article discusses tariffs, bonds, and stocks.</p></article></html>"
                }
            )
        return fake_get(url, params=params, headers=headers, timeout=timeout)

    monkeypatch.setattr("spy_trump_model.news_sources.requests.get", fake_get_with_article)

    fetched = fetch_gdelt_news(
        out_path=out_path,
        start_date="2024-01-01",
        end_date="2024-01-03",
        queries={"market": '"Donald Trump" stock market'},
        chunk_days=3,
        max_records=50,
        fetch_article_text=True,
        max_article_fetches=1,
        article_sleep_seconds=0,
        sleep_seconds=0,
    )

    assert out_path.exists()
    assert len(fetched) == 2
    assert calls[0]["mode"] == "ArtList"
    assert calls[0]["maxrecords"] == 50
    assert "sourcelang:english" in calls[0]["query"]
    assert set(["date", "datetime", "title", "source", "source_type", "text"]).issubset(fetched.columns)
    assert fetched["source_type"].eq("gdelt_market").all()
    assert fetched["text"].str.contains("example.com").all()
    assert len(article_calls) == 1
    assert fetched["text"].str.contains("Full article discusses tariffs").sum() == 1

    reloaded = pd.read_csv(out_path)
    assert len(reloaded) == 2
    assert reloaded["source"].is_unique


def test_fetch_gdelt_news_retries_rate_limited_api_chunk(tmp_path: Path, monkeypatch) -> None:
    out_path = tmp_path / "news.csv"
    calls = 0
    sleeps: list[float] = []

    def fake_get(url, params=None, headers=None, timeout=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _FakeResponse({}, status_code=429, headers={"Retry-After": "0"})
        return _FakeResponse(
            {
                "articles": [
                    {
                        "url": "https://example.com/markets/trump-retry",
                        "title": "Markets recover after delayed Trump news fetch",
                        "seendate": "20240102T143000Z",
                        "domain": "example.com",
                    }
                ]
            }
        )

    monkeypatch.setattr("spy_trump_model.news_sources.requests.get", fake_get)
    monkeypatch.setattr("spy_trump_model.news_sources.time.sleep", lambda seconds: sleeps.append(seconds))

    fetched = fetch_gdelt_news(
        out_path=out_path,
        start_date="2024-01-01",
        end_date="2024-01-03",
        queries={"market": '"Donald Trump" stock market'},
        chunk_days=3,
        fetch_article_text=False,
        sleep_seconds=0,
        gdelt_retry_attempts=2,
        gdelt_retry_backoff_seconds=0,
    )

    assert calls == 2
    assert sleeps == [0.0, 0.0]
    assert len(fetched) == 1
    assert fetched.iloc[0]["source"] == "https://example.com/markets/trump-retry"


def test_parse_gdelt_query_args_accepts_labels_and_raw_queries() -> None:
    parsed = parse_gdelt_query_args(
        [
            'tariffs="Donald Trump" tariffs',
            '"Donald Trump" oil prices',
        ]
    )

    assert parsed == {
        "tariffs": '"Donald Trump" tariffs',
        "custom_2": '"Donald Trump" oil prices',
    }
