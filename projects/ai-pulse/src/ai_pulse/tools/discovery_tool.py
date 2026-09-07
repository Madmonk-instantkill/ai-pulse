"""
Written by Amit Upadhyay aka Madmonk-instantkill
Copyright (c) 2026 Amit Upadhyay. All rights reserved.
"""

from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from ai_pulse.schemas import PaperRecord
from ai_pulse.tools.arxiv_tool import ArxivSearchTool
from ai_pulse.tools.huggingface_tool import HuggingFaceSearchTool
from ai_pulse.tools.openalex_tool import OpenAlexSearchTool


def _prefer_arxiv(records: list[PaperRecord], field: str):
    """Return the given field's value from whichever record in the group
    came from arXiv, since arXiv is the canonical/authoritative source for
    descriptive fields (title, abstract, authors, etc.). Falls back to the
    first record that has a non-empty value if no arXiv-sourced record
    exists, or to the very first record's value as a last resort."""
    for record in records:
        if "arxiv" in record.sources:
            value = getattr(record, field)
            if value:
                return value
    for record in records:
        value = getattr(record, field)
        if value:
            return value
    return getattr(records[0], field)


def _merge_duplicate_records(records: list[PaperRecord]) -> PaperRecord:
    """Merge multiple PaperRecords that share the same arxiv_id (the paper
    was found in more than one source) into a single record.

    Descriptive fields prefer arXiv's version when present. Confirmed
    numeric signals (upvotes, citation_count) are taken specifically from
    whichever record has the matching *_match flag set -- i.e. whichever
    source actually confirmed that value -- rather than overwritten by
    another source's default of 0/False."""
    upvotes = 0
    hf_match = False
    for record in records:
        if record.hf_match:
            upvotes = record.upvotes
            hf_match = True
            break

    citation_count = 0
    ss_match = False
    for record in records:
        if record.ss_match:
            citation_count = record.citation_count
            ss_match = True
            break

    all_sources = sorted({source for record in records for source in record.sources})

    return PaperRecord(
        paper_id=records[0].paper_id,
        arxiv_id=records[0].arxiv_id,
        title=_prefer_arxiv(records, "title"),
        authors=_prefer_arxiv(records, "authors"),
        lead_author_id=_prefer_arxiv(records, "lead_author_id"),
        lead_author_name=_prefer_arxiv(records, "lead_author_name"),
        published_date=_prefer_arxiv(records, "published_date"),
        abstract=_prefer_arxiv(records, "abstract"),
        pdf_url=_prefer_arxiv(records, "pdf_url"),
        sources=all_sources,
        source_count=len(all_sources),
        upvotes=upvotes,
        citation_count=citation_count,
        hf_match=hf_match,
        ss_match=ss_match,
    )


def deduplicate_papers(records: list[PaperRecord]) -> list[PaperRecord]:
    """Group records by arxiv_id and merge each group into one record.
    A paper found in only one source passes through unchanged (its own
    "group" has just one member)."""
    groups: dict[str, list[PaperRecord]] = {}
    for record in records:
        groups.setdefault(record.arxiv_id, []).append(record)
    return [_merge_duplicate_records(group) for group in groups.values()]


class DiscoveryInput(BaseModel):
    """Input arguments the DiscoveryTool accepts when an agent calls it --
    per-source batch sizes, matching the target batches from the project
    spec (section 4)."""

    arxiv_max_results: int = Field(30, description="Max papers to fetch from arXiv")
    openalex_max_results: int = Field(50, description="Max papers to fetch from OpenAlex")
    huggingface_max_results: int = Field(30, description="Max papers to fetch from Hugging Face")


class DiscoveryTool(BaseTool):
    """CrewAI tool used by the Scout Agent to collect candidate papers from
    all three sources and deduplicate them in one call. Composes the three
    existing single-source tools internally so the agent's LLM only ever
    sees the final deduplicated list -- it never has to shuttle the raw
    ~60-110 per-source results through its own context as tool-call
    arguments."""

    name: str = "Paper Discovery"
    description: str = (
        "Collects candidate AI/LLM papers from arXiv, OpenAlex, and "
        "Hugging Face in one call, merges duplicates found in more than "
        "one source, and returns the deduplicated list as PaperRecords."
    )
    args_schema: Type[BaseModel] = DiscoveryInput

    def _run(
        self,
        arxiv_max_results: int = 30,
        openalex_max_results: int = 50,
        huggingface_max_results: int = 30,
    ) -> list[dict]:
        """Call all three collector tools directly, combine their results,
        and deduplicate by arxiv_id before returning."""
        arxiv_records = [
            PaperRecord(**d) for d in ArxivSearchTool()._run(max_results=arxiv_max_results)
        ]
        openalex_records = [
            PaperRecord(**d) for d in OpenAlexSearchTool()._run(max_results=openalex_max_results)
        ]
        huggingface_records = [
            PaperRecord(**d)
            for d in HuggingFaceSearchTool()._run(max_results=huggingface_max_results)
        ]

        all_records = arxiv_records + openalex_records + huggingface_records
        deduplicated = deduplicate_papers(all_records)
        return [record.model_dump(mode="json") for record in deduplicated]
