"""
Written by Amit Upadhyay aka Madmonk-instantkill
Copyright (c) 2026 Amit Upadhyay. All rights reserved.
"""

from ai_pulse.schemas import PaperRecord


def format_papers_for_relevance(papers: list[PaperRecord]) -> str:
    """Turn a batch of papers into the plain text the relevance filter
    reads: arXiv ID, title and abstract for each, separated by blank
    lines. Nothing else about the paper is shown, so the judgment rests
    only on what the paper is about."""
    return "\n\n".join(
        f"arXiv ID: {paper.arxiv_id}\n"
        f"Title: {paper.title}\n"
        f"Abstract: {' '.join(paper.abstract.split())}"
        for paper in papers
    )
