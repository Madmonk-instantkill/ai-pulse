"""
Written by Amit Upadhyay aka Madmonk-instantkill
Copyright (c) 2026 Amit Upadhyay. All rights reserved.
"""

import re
from typing import Optional, Type

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_result, stop_after_attempt, wait_exponential

from ai_pulse.keywords import flatten_keyword_terms
from ai_pulse.schemas import PaperRecord

OPENALEX_WORKS_URL = "https://api.openalex.org/works"

# OpenAlex's own internal ID for "arXiv (Cornell University)" as a source,
# found by querying https://api.openalex.org/sources?search=arXiv (not
# guessed). Filtering on this restricts results to works that are actually
# hosted on arXiv -- without it, OpenAlex's broad index returns mostly
# self-published/low-curation content (Zenodo uploads, PhilPapers, etc.)
# with no arxiv_id at all, which PaperRecord requires.
ARXIV_SOURCE_ID = "S4306400194"

# arXiv's own tool restricts by CS category (cs.CL/cs.AI), so bare generic
# words in the shared keyword list are safe there -- the category already
# scopes the field. OpenAlex has no equivalent field-level restriction, and
# it searches full text (not just title/abstract), so these same generic
# words match unrelated papers that merely use them in passing (a stats
# paper saying "by similar reasoning...", a materials paper mentioning
# "efficient training of the model"). Confirmed empirically: dropping this
# set roughly doubled result relevance in manual testing. keywords.py stays
# the single source of truth for the full list; this is a deliberate,
# documented exception for this one source, not a duplicated list.
GENERIC_TERMS_TOO_NOISY_FOR_FULLTEXT_SEARCH = {
    "fine-tuning",
    "post-training",
    "reasoning",
    "planning",
    "efficient training",
    "inference optimization",
    "model compression",
}


def _is_rate_limited(response: requests.Response) -> bool:
    """True if OpenAlex is telling us to slow down (429)."""
    return response.status_code == 429


@retry(
    retry=retry_if_result(_is_rate_limited),
    wait=wait_exponential(multiplier=3, min=3, max=60),
    stop=stop_after_attempt(5),
)
def _get_with_backoff(params: dict) -> requests.Response:
    """GET the OpenAlex works endpoint with exponential backoff on 429
    only. Any other status (200, 4xx, 5xx) is returned immediately -- we
    only want to retry the "too fast" case, not mask a real error."""
    return requests.get(OPENALEX_WORKS_URL, params=params, timeout=30)


def _build_positive_keyword_query() -> str:
    """Build an OpenAlex boolean search string that ORs together the
    positive keyword clusters from the project spec (section 4), minus the
    terms known to be too generic for OpenAlex's full-text search (see
    GENERIC_TERMS_TOO_NOISY_FOR_FULLTEXT_SEARCH above). Uses OpenAlex's
    own syntax: uppercase OR, double quotes for exact phrases, parentheses
    for grouping."""
    terms = [
        term
        for term in flatten_keyword_terms()
        if term not in GENERIC_TERMS_TOO_NOISY_FOR_FULLTEXT_SEARCH
    ]
    quoted = [f'"{term}"' if " " in term else term for term in terms]
    return "(" + " OR ".join(quoted) + ")"


# An arXiv location's URL shows up in one of two confirmed formats (found
# by inspecting real OpenAlex responses, not guessed):
#   - the direct arXiv URL:      https://arxiv.org/abs/2502.05151
#   - the DOI resolver form:     https://doi.org/10.48550/arxiv.2502.05151
# Some works only carry the DOI form and have no direct arxiv.org URL at
# all, so both alternatives must be checked or those works are silently
# (and wrongly) dropped.
_ARXIV_ID_PATTERN = re.compile(
    r"arxiv\.org/(?:abs|pdf)/([\w.\-]+)|10\.48550/arxiv\.([\w.\-]+)",
    re.IGNORECASE,
)


def _extract_arxiv_id(work: dict) -> Optional[str]:
    """Find the arXiv location in a work's `locations` list and pull the
    arxiv_id out of its landing_page_url or pdf_url, checking both known
    URL formats. Returns None if this work has no arXiv location at all."""
    for location in work.get("locations", []):
        source = location.get("source") or {}
        if source.get("id") != f"https://openalex.org/{ARXIV_SOURCE_ID}":
            continue
        for url in (location.get("landing_page_url"), location.get("pdf_url")):
            if not url:
                continue
            match = _ARXIV_ID_PATTERN.search(url)
            if match:
                raw_id = match.group(1) or match.group(2)
                return re.sub(r"v\d+$", "", raw_id)
    return None


def _to_paper_record(work: dict) -> Optional[PaperRecord]:
    """Convert one raw OpenAlex work into our common PaperRecord shape.
    Returns None if the work has no extractable arXiv ID -- PaperRecord
    requires one for cross-source deduplication, so such works are
    dropped by the caller rather than producing an invalid record."""
    arxiv_id = _extract_arxiv_id(work)
    if arxiv_id is None:
        return None

    authorships = work.get("authorships", [])
    authors = [a.get("author", {}).get("display_name", "") for a in authorships]

    abstract_index = work.get("abstract_inverted_index")
    abstract = _reconstruct_abstract(abstract_index) if abstract_index else ""

    return PaperRecord(
        paper_id=f"arxiv:{arxiv_id}",
        arxiv_id=arxiv_id,
        title=work.get("title") or "",
        authors=authors,
        lead_author_name=authors[0] if authors else None,
        published_date=work["publication_date"],
        abstract=abstract,
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        sources=["openalex"],
        source_count=1,
        citation_count=work.get("cited_by_count", 0),
        ss_match=True,  # a real citation count is available, just not from Semantic Scholar
    )


def _reconstruct_abstract(inverted_index: dict) -> str:
    """OpenAlex stores abstracts as an inverted index (word -> list of
    positions) instead of plain text, to save space. Rebuild the plain
    text by placing each word back at every position it occurs."""
    position_to_word: dict[int, str] = {}
    for word, positions in inverted_index.items():
        for position in positions:
            position_to_word[position] = word
    return " ".join(position_to_word[i] for i in sorted(position_to_word))


class OpenAlexSearchInput(BaseModel):
    """Input arguments the OpenAlexSearchTool accepts when an agent calls
    it. The search query itself is fixed -- only the batch size is left
    tunable."""

    max_results: int = Field(50, description="Maximum number of papers to fetch")


class OpenAlexSearchTool(BaseTool):
    """CrewAI tool used by the Scout Agent to search OpenAlex, restricted
    to works actually hosted on arXiv, and return candidate papers as
    PaperRecord dicts, ready for deduplication and enrichment against the
    other two sources."""

    name: str = "OpenAlex Paper Search"
    description: str = (
        "Searches OpenAlex (restricted to arXiv-hosted works) for AI/LLM "
        "research papers matching the project's fixed positive keyword "
        "clusters, and returns each match as a PaperRecord."
    )
    args_schema: Type[BaseModel] = OpenAlexSearchInput

    def _run(self, max_results: int = 50) -> list[dict]:
        """Run the fixed keyword-cluster search (article type, arXiv-hosted
        only) and return a list of PaperRecord dicts, sorted by newest
        publication date first. Works with no extractable arxiv_id are
        silently dropped."""
        params = {
            "search": _build_positive_keyword_query(),
            "filter": f"type:article,locations.source.id:{ARXIV_SOURCE_ID}",
            "sort": "publication_date:desc",
            "per-page": max_results,
        }
        response = _get_with_backoff(params)
        response.raise_for_status()

        works = response.json().get("results", [])
        records = [_to_paper_record(work) for work in works]
        return [r.model_dump(mode="json") for r in records if r is not None]
