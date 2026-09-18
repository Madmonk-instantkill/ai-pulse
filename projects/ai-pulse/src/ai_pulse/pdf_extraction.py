"""
Written by Amit Upadhyay aka Madmonk-instantkill
Copyright (c) 2026 Amit Upadhyay. All rights reserved.
"""

import fitz  # pymupdf
import requests
from tenacity import retry, retry_if_result, stop_after_attempt, wait_exponential

from ai_pulse.schemas import DocumentText, PageText


def _is_rate_limited(response: requests.Response) -> bool:
    """True if arXiv is telling us to slow down (429)."""
    return response.status_code == 429


@retry(
    retry=retry_if_result(_is_rate_limited),
    wait=wait_exponential(multiplier=3, min=3, max=60),
    stop=stop_after_attempt(5),
)
def _get_with_backoff(url: str) -> requests.Response:
    """GET with exponential backoff on 429 only. Any other status is
    returned immediately -- we only want to retry the "too fast" case,
    not mask a real error by retrying it silently."""
    return requests.get(url, timeout=60)


def download_pdf(pdf_url: str) -> bytes:
    """Download a paper's PDF and return the raw bytes."""
    response = _get_with_backoff(pdf_url)
    response.raise_for_status()
    return response.content


def extract_pages(pdf_bytes: bytes) -> list[PageText]:
    """Open a PDF from raw bytes and extract each page's plain text
    (figures/images are skipped -- only readable text is pulled),
    keeping the 1-indexed page number for later source-locator citations."""
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return [
            PageText(page_number=i + 1, text=page.get_text())
            for i, page in enumerate(document)
        ]
    finally:
        document.close()


def extract_paper_text(arxiv_id: str, pdf_url: str) -> DocumentText:
    """Download a paper's PDF and extract its full text, page by page.
    This is plain Python, not a CrewAI tool -- downloading a file and
    reading text off pages requires no judgment, so no LLM is involved."""
    pdf_bytes = download_pdf(pdf_url)
    pages = extract_pages(pdf_bytes)
    return DocumentText(arxiv_id=arxiv_id, pages=pages)
