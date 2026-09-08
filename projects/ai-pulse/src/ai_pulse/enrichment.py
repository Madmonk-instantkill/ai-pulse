"""
Written by Amit Upadhyay aka Madmonk-instantkill
Copyright (c) 2026 Amit Upadhyay. All rights reserved.
"""

from typing import Optional

import requests
from tenacity import retry, retry_if_result, stop_after_attempt, wait_exponential

from ai_pulse.schemas import PaperRecord

OPENALEX_BASE_URL = "https://api.openalex.org"


def _is_rate_limited(response: requests.Response) -> bool:
    """True if OpenAlex is telling us to slow down (429)."""
    return response.status_code == 429


@retry(
    retry=retry_if_result(_is_rate_limited),
    wait=wait_exponential(multiplier=3, min=3, max=60),
    stop=stop_after_attempt(5),
)
def _get_with_backoff(url: str, params: dict) -> requests.Response:
    """GET with exponential backoff on 429 only. Any other status (200,
    404, 500, etc.) is returned immediately -- we only want to retry the
    "too fast" case, not mask a real error by retrying it silently."""
    return requests.get(url, params=params, timeout=30)


def _fetch_work_by_arxiv_id(arxiv_id: str) -> Optional[dict]:
    """Fetch the OpenAlex work for this paper directly by its arXiv DOI.
    Returns None if OpenAlex has no record for it (e.g. not yet indexed)."""
    url = f"{OPENALEX_BASE_URL}/works/doi:10.48550/arxiv.{arxiv_id}"
    response = _get_with_backoff(url, {"mailto": "you@example.com"})
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.json()


def _fetch_author_h_index(author_id: str) -> int:
    """Fetch an OpenAlex author's h-index by their OpenAlex author ID
    (e.g. 'https://openalex.org/A5149198713' or just the 'A...' part)."""
    author_id = author_id.rsplit("/", 1)[-1]
    url = f"{OPENALEX_BASE_URL}/authors/{author_id}"
    response = _get_with_backoff(url, {"mailto": "you@example.com"})
    response.raise_for_status()
    summary_stats = response.json().get("summary_stats") or {}
    return summary_stats.get("h_index", 0)


def _enrich_one(record: PaperRecord) -> PaperRecord:
    """Enrich a single paper: backfill citation_count if it wasn't already
    confirmed by a source (citation_confirmed is False), and fill
    lead_author_id / lead_author_h_index, which no collector provides.
    Returns the same record unchanged if OpenAlex has no data for this
    paper at all."""
    work = _fetch_work_by_arxiv_id(record.arxiv_id)
    if work is None:
        return record

    updates: dict = {}

    if not record.citation_confirmed:
        updates["citation_count"] = work.get("cited_by_count", 0)
        updates["citation_confirmed"] = True

    authorships = work.get("authorships", [])
    if authorships:
        lead_author = authorships[0].get("author", {})
        author_id = lead_author.get("id")
        if author_id:
            updates["lead_author_id"] = author_id
            updates["lead_author_h_index"] = _fetch_author_h_index(author_id)

    return record.model_copy(update=updates) if updates else record


def enrich_papers(records: list[PaperRecord]) -> list[PaperRecord]:
    """Enrich every paper in the list via OpenAlex and return the updated
    records. This is plain Python, not a CrewAI tool -- enrichment
    requires no LLM judgment (matching a citation count or an author's
    h-index to a paper is purely mechanical), so it is called directly
    from the Flow between crew stages, never routed through an agent's
    context."""
    return [_enrich_one(record) for record in records]
