"""
Written by Amit Upadhyay aka Msdmonk-instantkill
Copyright (c) 2026 Amit Upadhyay. All rights reserved.
"""

from datetime import date
from typing import Optional

from pydantic import BaseModel


class PaperRecord(BaseModel):
    """Common shape every collector (arXiv, Semantic Scholar, Hugging Face)
    must convert its raw API response into, so deduplication, enrichment,
    scoring, and the report writer never need to know which source a paper
    originally came from.

    hf_match / ss_match mark a *confirmed absence* from that source (so the
    matching numeric field, e.g. upvotes or citation_count, is a real 0
    rather than "not checked yet") -- scoring uses these flags to decide
    whether to trust that field or redistribute its weight elsewhere.
    """

    paper_id: str
    arxiv_id: str
    title: str
    authors: list[str] = []
    lead_author_id: Optional[str] = None
    lead_author_name: Optional[str] = None
    published_date: date
    abstract: str
    pdf_url: str

    sources: list[str] = []
    source_count: int = 1

    upvotes: int = 0
    citation_count: int = 0
    citation_velocity: float = 0.0
    lead_author_h_index: int = 0
    hf_match: bool = False
    ss_match: bool = False

    provisional_score: float = 0.0
    final_score: float = 0.0


class RelevanceDecision(BaseModel):
    """One Scout Agent judgment on a single paper: does it genuinely belong
    in an AI/LLM digest, per the project's positive focus areas and
    negative exclusions (spec section 4)?

    arxiv_id is the join key back to the PaperRecord this decision is
    about. reason is required for both keep and discard decisions, not
    just discards -- so every judgment, including borderline "kept" calls,
    stays auditable later if the filter turns out too strict or too
    lenient.
    """

    arxiv_id: str
    keep: bool
    reason: str


class RelevanceFilterResult(BaseModel):
    """Wraps the full list of RelevanceDecisions for one run. CrewAI's
    output_pydantic expects a single BaseModel as a Task's structured
    output, not a bare list -- this is that wrapper, holding one decision
    per paper the Scout Agent was asked to judge."""

    decisions: list[RelevanceDecision]
