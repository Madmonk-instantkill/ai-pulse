"""
Written by Amit Upadhyay aka Madmonk-instantkill
Copyright (c) 2026 Amit Upadhyay. All rights reserved.
"""

from ai_pulse.schemas import Chunk, DocumentText, PageText

# Chosen for prompt quality (keeping each Map call focused), not because
# the LLM's real context window is anywhere near this small. A single
# oversized page that alone exceeds this becomes its own chunk anyway --
# see _group_pages below.
CHUNK_TOKEN_BUDGET = 3500


def _build_chunk(arxiv_id: str, pages: list[PageText]) -> Chunk:
    """Combine a run of consecutive pages into one Chunk."""
    return Chunk(
        arxiv_id=arxiv_id,
        start_page=pages[0].page_number,
        end_page=pages[-1].page_number,
        text="\n\n".join(page.text for page in pages),
        token_count=sum(page.token_count for page in pages),
    )


def chunk_document(
    document: DocumentText, token_budget: int = CHUNK_TOKEN_BUDGET
) -> list[Chunk]:
    """Group a document's token-counted pages into token-bounded chunks:
    keep adding consecutive pages to the current chunk until the next
    page would push it over token_budget, then close the chunk and start
    a new one. A single page that alone exceeds token_budget simply
    becomes its own oversized chunk -- no special-case branch needed,
    that falls out of the same accumulate-until-it-doesn't-fit loop."""
    chunks: list[Chunk] = []
    current_pages: list[PageText] = []
    current_tokens = 0

    for page in document.pages:
        if current_tokens + page.token_count > token_budget and current_pages:
            chunks.append(_build_chunk(document.arxiv_id, current_pages))
            current_pages = []
            current_tokens = 0
        current_pages.append(page)
        current_tokens += page.token_count

    if current_pages:
        chunks.append(_build_chunk(document.arxiv_id, current_pages))

    return chunks
