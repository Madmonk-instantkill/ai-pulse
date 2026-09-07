"""
Written by Amit Upadhyay aka Madmonk-instantkill
Copyright (c) 2026 Amit Upadhyay. All rights reserved.
"""

import re
from typing import Type

import arxiv
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from ai_pulse.keywords import flatten_keyword_terms
from ai_pulse.schemas import PaperRecord


# Restrict results to arXiv's Computation-and-Language and Artificial-
# Intelligence categories. cs.LG (Machine Learning) is deliberately excluded
# -- it is arXiv's general-purpose ML bucket and pulls in unrelated work
# (vision, robotics, medical imaging, etc.) that happens to reuse common ML
# vocabulary like "fine-tuning" or "foundation model". A paper that is
# genuinely about LLMs is almost always cross-listed under cs.CL or cs.AI
# anyway.
ARXIV_CATEGORIES = ["cs.CL", "cs.AI"]


def _build_category_filter() -> str:
    """Build the '(cat:X OR cat:Y)' clause restricting results to the
    project's target arXiv categories."""
    return "(" + " OR ".join(f"cat:{category}" for category in ARXIV_CATEGORIES) + ")"


def _build_positive_keyword_query() -> str:
    """Build the full arXiv query: the target categories, ANDed with an OR
    of every keyword across all seven positive keyword clusters from the
    project spec (section 4). This is the fixed search the Scout Agent
    always runs -- it is not something the agent's LLM gets to choose, so
    results stay consistent and reproducible from run to run."""
    keyword_filter = (
        "(" + " OR ".join(f'all:"{term}"' for term in flatten_keyword_terms()) + ")"
    )
    return f"{_build_category_filter()} AND {keyword_filter}"


def _normalize_arxiv_id(raw_id: str) -> str:
    """Strip the trailing version suffix, e.g. '2501.12345v2' -> '2501.12345',
    so this ID matches cleanly against Semantic Scholar and Hugging Face later."""
    return re.sub(r"v\d+$", "", raw_id)


def _to_paper_record(result: arxiv.Result) -> PaperRecord:
    """Convert one raw arxiv.Result (as returned by the arxiv package) into
    our common PaperRecord shape. Only fields arXiv actually provides are
    filled in; everything else (upvotes, citations, h-index) is left at the
    schema default for later enrichment."""
    arxiv_id = _normalize_arxiv_id(result.get_short_id())
    authors = [author.name for author in result.authors]

    return PaperRecord(
        paper_id=f"arxiv:{arxiv_id}",
        arxiv_id=arxiv_id,
        title=result.title,
        authors=authors,
        lead_author_name=authors[0] if authors else None,
        published_date=result.published.date(),
        abstract=result.summary,
        pdf_url=result.pdf_url,
        sources=["arxiv"],
        source_count=1,
    )


class ArxivSearchInput(BaseModel):
    """Input arguments the ArxivSearchTool accepts when an agent calls it.
    The search query itself is fixed (see _build_positive_keyword_query) --
    only the batch size is left tunable."""

    max_results: int = Field(30, description="Maximum number of papers to fetch")


class ArxivSearchTool(BaseTool):
    """CrewAI tool used by the Scout Agent to search arXiv and return
    candidate papers as PaperRecord dicts, ready for deduplication and
    enrichment against the other two sources."""

    name: str = "arXiv Paper Search"
    description: str = (
        "Searches arXiv for AI/LLM research papers matching the project's "
        "fixed positive keyword clusters (LLMs, fine-tuning, RAG, agents, "
        "reasoning, multimodal models, efficiency) and returns each match "
        "as a PaperRecord."
    )
    args_schema: Type[BaseModel] = ArxivSearchInput

    def _run(self, max_results: int = 30) -> list[dict]:
        """Run the fixed keyword-cluster search and return a list of
        PaperRecord dicts, sorted by most recently submitted first."""
        client = arxiv.Client()
        search = arxiv.Search(
            query=_build_positive_keyword_query(),
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )
        records = [_to_paper_record(result) for result in client.results(search)]
        return [record.model_dump(mode="json") for record in records]
