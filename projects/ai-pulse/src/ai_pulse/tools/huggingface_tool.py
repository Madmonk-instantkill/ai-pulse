"""
Written by Amit Upadhyay aka Madmonk-instantkill
Copyright (c) 2026 Amit Upadhyay. All rights reserved.
"""

from datetime import datetime
from typing import Type

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_result, stop_after_attempt, wait_exponential

from ai_pulse.schemas import PaperRecord

HF_DAILY_PAPERS_URL = "https://huggingface.co/api/daily_papers"


def _is_rate_limited(response: requests.Response) -> bool:
    """True if Hugging Face is telling us to slow down (429)."""
    return response.status_code == 429


@retry(
    retry=retry_if_result(_is_rate_limited),
    wait=wait_exponential(multiplier=3, min=3, max=60),
    stop=stop_after_attempt(5),
)
def _get_with_backoff(params: dict) -> requests.Response:
    """GET the Hugging Face daily papers endpoint with exponential backoff
    on 429 only. Any other status is returned immediately."""
    return requests.get(HF_DAILY_PAPERS_URL, params=params, timeout=30)


def _to_paper_record(item: dict) -> PaperRecord:
    """Convert one raw daily_papers entry into our common PaperRecord
    shape. Hugging Face's `paper.id` is directly the arXiv ID (no
    extraction needed, unlike OpenAlex). Only fields Hugging Face actually
    provides are filled in; everything else (citations, h-index) is left
    at the schema default for later enrichment."""
    paper = item["paper"]
    arxiv_id = paper["id"]
    authors = [author["name"] for author in paper.get("authors", [])]

    return PaperRecord(
        paper_id=f"arxiv:{arxiv_id}",
        arxiv_id=arxiv_id,
        title=paper["title"],
        authors=authors,
        lead_author_name=authors[0] if authors else None,
        published_date=datetime.fromisoformat(paper["publishedAt"]).date(),
        abstract=paper.get("summary", ""),
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        sources=["huggingface"],
        source_count=1,
        upvotes=paper.get("upvotes", 0),
        hf_match=True,
    )


class HuggingFaceSearchInput(BaseModel):
    """Input arguments the HuggingFaceSearchTool accepts when an agent
    calls it. This source has no keyword search of its own (confirmed by
    testing -- the API's search param has no effect), so only the batch
    size is tunable."""

    max_results: int = Field(30, description="Maximum number of papers to fetch")


class HuggingFaceSearchTool(BaseTool):
    """CrewAI tool used by the Scout Agent to fetch Hugging Face's
    community-curated Daily Papers feed and return candidate papers as
    PaperRecord dicts, ready for deduplication and enrichment against the
    other two sources."""

    name: str = "Hugging Face Daily Papers"
    description: str = (
        "Fetches Hugging Face's community-curated Daily Papers feed "
        "(already AI/ML-focused by nature of community curation) and "
        "returns each entry as a PaperRecord, including its upvote count."
    )
    args_schema: Type[BaseModel] = HuggingFaceSearchInput

    def _run(self, max_results: int = 30) -> list[dict]:
        """Fetch up to max_results papers from the Daily Papers feed,
        newest first, and return them as PaperRecord dicts."""
        response = _get_with_backoff({"limit": max_results})
        response.raise_for_status()

        items = response.json()
        records = [_to_paper_record(item) for item in items]
        return [record.model_dump(mode="json") for record in records]
