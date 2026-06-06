from pathlib import Path

import pandas as pd

from spy_trump_model.news_sources import fetch_gdelt_news, parse_gdelt_query_args


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._payload


def test_fetch_gdelt_news_writes_expansion_ready_csv(tmp_path: Path, monkeypatch) -> None:
    out_path = tmp_path / "news.csv"
    calls: list[dict[str, object]] = []

    def fake_get(url, params=None, headers=None, timeout=None):
        if "api.gdeltproject.org" not in url:
            return _FakeResponse(
                {
                    "html": "<html><article><p>Full article discusses tariffs, bonds, and stocks.</p></article></html>"
                }
            )
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
    assert fetched["text"].str.contains("Full article discusses tariffs").all()

    reloaded = pd.read_csv(out_path)
    assert len(reloaded) == 2
    assert reloaded["source"].is_unique


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
