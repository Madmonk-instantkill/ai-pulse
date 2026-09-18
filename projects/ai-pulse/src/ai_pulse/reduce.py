"""
Written by Amit Upadhyay aka Madmonk-instantkill
Copyright (c) 2026 Amit Upadhyay. All rights reserved.
"""

from ai_pulse.schemas import ChunkSummary


def group_chunk_summaries_by_paper(
    summaries: list[ChunkSummary],
) -> dict[str, list[ChunkSummary]]:
    """Group Map-phase chunk summaries by which paper (arxiv_id) they
    belong to, so Reduce can process one paper's full set of summaries
    together in a single call. Plain Python -- no LLM judgment needed to
    sort data by a key it already carries.

    Same setdefault/append grouping mechanism as deduplicate_papers in
    discovery_tool.py, just grouping by arxiv_id instead of merging
    duplicates. Order is preserved within each paper's list, since
    summaries are appended in whatever order they're passed in (which
    will be page order, as long as the caller runs Map sequentially over
    chunks in the order chunk_document() produced them)."""
    groups: dict[str, list[ChunkSummary]] = {}
    for summary in summaries:
        groups.setdefault(summary.arxiv_id, []).append(summary)
    return groups


def format_summaries_for_reduce(summaries: list[ChunkSummary]) -> str:
    """Render one paper's chunk summaries as a single text block for the
    Reduce task's {chunk_summaries} placeholder. Sorted by start_page so
    the Reduce Agent always reads them in page order, regardless of how
    the Map results happened to be collected."""
    ordered = sorted(summaries, key=lambda s: s.start_page)
    return "\n\n".join(
        f"--- Summary of pages {s.start_page}-{s.end_page} ---\n"
        f"{s.model_dump_json(indent=2, exclude={'arxiv_id'})}"
        for s in ordered
    )
