"""
Written by Amit Upadhyay aka Msdmonk-instantkill
Copyright (c) 2026 Amit Upadhyay. All rights reserved.
"""

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel


class PaperRecord(BaseModel):
    """Common shape every collector (arXiv, OpenAlex, Hugging Face) must
    convert its raw API response into, so deduplication, enrichment,
    scoring, and the report writer never need to know which source a paper
    originally came from.

    upvotes_confirmed / citation_confirmed mark a *confirmed absence* of
    that signal (so the matching numeric field, upvotes or citation_count,
    is a real 0 rather than "not checked yet") -- scoring uses these flags
    to decide whether to trust that field or redistribute its weight
    elsewhere. Named after what they confirm, not which API confirmed it,
    since the source behind either signal can change over time (citation
    data originally came from Semantic Scholar, now comes from OpenAlex).
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
    upvotes_confirmed: bool = False
    citation_confirmed: bool = False

    provisional_score: float = 0.0
    final_score: float = 0.0


class PaperCollectionResult(BaseModel):
    """Wraps the full deduplicated paper list from the collect_and_
    deduplicate_task. Same reason RelevanceFilterResult exists --
    output_pydantic needs a single BaseModel, not a bare list."""

    papers: list[PaperRecord]


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


class PageText(BaseModel):
    """One page's extracted plain text, pulled from a downloaded PDF.
    page_number is 1-indexed, matching how a human (or a claim's source
    locator, e.g. "page 4, Methods") would actually cite it.

    token_count starts at 0 (not yet counted) and gets filled in by
    tokenization.py, the same "default until computed" pattern used
    elsewhere in this project (e.g. citation_count on PaperRecord)."""

    page_number: int
    text: str
    token_count: int = 0


class DocumentText(BaseModel):
    """The full extracted text of one document (a paper, later a
    newsletter issue too), as a list of pages -- the raw material
    MapReduce chunks and summarizes. arxiv_id ties this back to the
    PaperRecord it was extracted from."""

    arxiv_id: str
    pages: list[PageText]


class Chunk(BaseModel):
    """One token-bounded group of consecutive pages, ready for one Map
    LLM call. start_page/end_page double as the source locator for any
    claim the Map step extracts from this chunk's text (e.g. "page 4").
    token_count may exceed the target budget for a single oversized page
    that doesn't fit any budget on its own -- see chunking.py."""

    arxiv_id: str
    start_page: int
    end_page: int
    text: str
    token_count: int


class ChunkSummary(BaseModel):
    """One Map Agent's extraction from a single chunk (spec section 8, Map
    Phase). Unlike RelevanceFilterResult, this needs no list wrapper --
    one Map call always produces exactly one ChunkSummary.

    start_page/end_page (carried over from the Chunk) serve as this
    summary's source locator. Every list defaults to empty rather than
    being invented -- if a chunk genuinely has no results in it, for
    example, results stays []."""

    arxiv_id: str
    start_page: int
    end_page: int
    main_ideas: list[str] = []
    methods: list[str] = []
    results: list[str] = []
    limitations: list[str] = []
    speculation: list[str] = []
    definitions: list[str] = []


class CandidateClaim(BaseModel):
    """One factual claim from a paper, with the page ranges of the chunk
    summaries that support it. This merges the spec's "candidate claims"
    list and "source-evidence ledger" (section 8) into one structure, so
    there is no second list to cross-reference. The Writer and Verifier
    stages add excerpt/status fields to this later (spec section 11)."""

    claim_text: str
    source_pages: list[str] = []


class DefinedTerm(BaseModel):
    """A technical term as the paper itself defines it, with the page
    ranges of the chunk summaries it came from -- the evidence location
    the Glossary stage needs (spec section 9)."""

    term: str
    definition: str
    source_pages: list[str] = []


class PaperDigest(BaseModel):
    """One Reduce Agent's document-level digest of a whole paper, built
    from all of that paper's ChunkSummaries (spec section 8, Reduce
    Phase). Internal raw material for the Writer Agent, not reader-facing
    -- so it favors completeness over brevity, and keeps what the authors
    reported separate from what they only speculate."""

    arxiv_id: str
    overview: str
    main_ideas: list[str] = []
    methods: list[str] = []
    reported_results: list[str] = []
    authors_speculation: list[str] = []
    limitations: list[str] = []
    definitions: list[DefinedTerm] = []
    conflicts_resolved: list[str] = []
    candidate_claims: list[CandidateClaim] = []


class GlossaryEntry(BaseModel):
    """One beginner-friendly glossary entry for a single paper, written by
    the Glossary Agent from that paper's PaperDigest (spec section 9).
    source_pages is copied from the digest's DefinedTerm, so every entry
    keeps its evidence location. The example is the one field that is
    invented rather than sourced -- it's an analogy, not a claim about
    the paper."""

    term: str
    plain_definition: str
    why_it_matters: str
    example: str
    arxiv_id: str
    source_pages: list[str] = []


class GlossaryResult(BaseModel):
    """Wraps one paper's glossary entries. Same reason as
    RelevanceFilterResult -- output_pydantic needs a single BaseModel,
    not a bare list."""

    entries: list[GlossaryEntry]


class PaperSection(BaseModel):
    """One Writer call's output for a single paper: the beginner-friendly
    section plus that paper's claim ledger. claim_ledger reuses
    CandidateClaim, so each claim keeps the source_pages copied from the
    paper's digest. The Verifier checks these claims later. The paper
    title is not written here -- code attaches it from the PaperRecord."""

    arxiv_id: str
    what_its_about: str
    whats_new: str
    why_it_matters: str
    key_results: list[str] = []
    limitations: list[str] = []
    claim_ledger: list[CandidateClaim] = []


class ClaimVerdict(BaseModel):
    """The Verifier's judgment on one ledger claim, made by comparing the
    claim to the text of the pages it cites. evidence_quote is the exact
    text from those pages that supports (or contradicts) the claim, left
    empty when nothing supports it. note is one short sentence on why."""

    claim_text: str
    source_pages: list[str] = []
    status: Literal["supported", "partially_supported", "not_supported"]
    evidence_quote: str = ""
    note: str = ""


class PaperVerification(BaseModel):
    """One Verifier call's output: a verdict for every claim in one
    paper's claim ledger."""

    arxiv_id: str
    verdicts: list[ClaimVerdict]


class ExecutiveSummary(BaseModel):
    """The last Writer call's output: a title for the whole report and a
    short intro covering all three papers. The date is not here -- the
    model can't know today's date, so code adds it."""

    report_title: str
    summary: str
