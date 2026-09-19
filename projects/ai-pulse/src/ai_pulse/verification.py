"""
Written by Amit Upadhyay aka Madmonk-instantkill
Copyright (c) 2026 Amit Upadhyay. All rights reserved.
"""

import re

from ai_pulse.schemas import CandidateClaim, DocumentText


def parse_page_range(source_page: str) -> list[int]:
    """Turn a ledger page string like "pages 3-4", "3-4" or "pages 5"
    into the list of page numbers it covers ([3, 4] or [5]). Returns []
    if no number is found."""
    numbers = re.findall(r"\d+", source_page)
    if not numbers:
        return []
    first = int(numbers[0])
    last = int(numbers[1]) if len(numbers) > 1 else first
    return list(range(first, last + 1))


def collect_source_text(
    document: DocumentText, claims: list[CandidateClaim]
) -> str:
    """Return the text of every page cited by any claim in the ledger, each
    page once, in page order, with a "--- Page N ---" header so the
    Verifier can tell pages apart. Pages the document does not have are
    skipped."""
    cited_pages: set[int] = set()
    for claim in claims:
        for source_page in claim.source_pages:
            cited_pages.update(parse_page_range(source_page))

    text_by_page = {page.page_number: page.text for page in document.pages}
    return "\n\n".join(
        f"--- Page {number} ---\n{text_by_page[number]}"
        for number in sorted(cited_pages)
        if number in text_by_page
    )
