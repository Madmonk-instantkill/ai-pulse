"""
Written by Amit Upadhyay aka Madmonk-instantkill
Copyright (c) 2026 Amit Upadhyay. All rights reserved.
"""

# Entry point: runs the whole pipeline once and emails the PDF.
#     python -m ai_pulse.main --run-now
# Windows Task Scheduler runs this every 3 days.

import argparse
import json
import sys
import time
import traceback
from datetime import date, datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

from ai_pulse.chunking import chunk_document  # noqa: E402
from ai_pulse.crews.content_crew.glossary_crew import GlossaryCrew  # noqa: E402
from ai_pulse.crews.content_crew.map_crew import MapCrew  # noqa: E402
from ai_pulse.crews.content_crew.reduce_crew import ReduceCrew  # noqa: E402
from ai_pulse.crews.delivery.delivery_service import (  # noqa: E402
    build_pdf_report,
    send_report_email,
)
from ai_pulse.crews.discovery_crew.discovery_crew import DiscoveryCrew  # noqa: E402
from ai_pulse.crews.editorial_crew.summary_crew import SummaryCrew  # noqa: E402
from ai_pulse.crews.editorial_crew.verifier_crew import VerifierCrew  # noqa: E402
from ai_pulse.crews.editorial_crew.writer_crew import WriterCrew  # noqa: E402
from ai_pulse.enrichment import enrich_papers  # noqa: E402
from ai_pulse.pdf_extraction import extract_paper_text  # noqa: E402
from ai_pulse.ranking import score_papers  # noqa: E402
from ai_pulse.reduce import format_summaries_for_reduce  # noqa: E402
from ai_pulse.relevance import format_papers_for_relevance  # noqa: E402
from ai_pulse.report_cleanup import remove_rejected_claims  # noqa: E402
from ai_pulse.run_logging import capture_run_log  # noqa: E402
from ai_pulse.schemas import (  # noqa: E402
    DocumentText,
    GlossaryEntry,
    PaperDigest,
    PaperRecord,
    PaperSection,
    PaperVerification,
)
from ai_pulse.seen_papers import (  # noqa: E402
    load_seen_ids,
    record_seen,
    remove_seen,
)
from ai_pulse.tokenization import annotate_token_counts  # noqa: E402
from ai_pulse.tools.discovery_tool import DiscoveryTool  # noqa: E402
from ai_pulse.verification import collect_source_text  # noqa: E402

PAPERS_PER_RUN = 3
RELEVANCE_BATCH_SIZE = 20
LLM_ATTEMPTS = 3
SECONDS_BETWEEN_LLM_CALLS = 5
SECONDS_BETWEEN_RETRIES = 30


def run_crew(crew, label: str, inputs: dict | None = None):
    """Kick off one crew and return its output. Free-tier AI calls fail
    now and then (503 busy, 429 rate limit), so a failed call, or one that
    returns no structured result, is retried up to LLM_ATTEMPTS times
    before the error is raised. Pauses briefly after each success to stay
    inside the rate limit."""
    for attempt in range(1, LLM_ATTEMPTS + 1):
        try:
            result = crew.kickoff(inputs=inputs)
            if not all(t.pydantic is not None for t in result.tasks_output):
                raise ValueError("crew returned no structured output")
            time.sleep(SECONDS_BETWEEN_LLM_CALLS)
            return result
        except Exception as error:
            print(f"[{label}] attempt {attempt}/{LLM_ATTEMPTS} failed: {error}")
            if attempt == LLM_ATTEMPTS:
                raise
            time.sleep(SECONDS_BETWEEN_RETRIES)


def select_papers(today: date, problems: list[str]) -> list[PaperRecord]:
    """Discovery: collect and deduplicate papers in plain Python, have the
    Scout agent judge relevance in small batches, drop the papers already
    sent in an earlier report, then enrich, rank and return the top
    PAPERS_PER_RUN. A paper the agent skips is treated as not relevant."""
    collected = [PaperRecord(**d) for d in DiscoveryTool()._run()]

    kept_ids: set[str] = set()
    for start in range(0, len(collected), RELEVANCE_BATCH_SIZE):
        batch = collected[start:start + RELEVANCE_BATCH_SIZE]
        result = run_crew(
            DiscoveryCrew().crew(),
            f"relevance {start + 1}-{start + len(batch)}",
            {"papers": format_papers_for_relevance(batch)},
        )
        kept_ids |= {d.arxiv_id for d in result.pydantic.decisions if d.keep}
    relevant = [p for p in collected if p.arxiv_id in kept_ids]

    fresh = remove_seen(relevant, load_seen_ids(today))
    print(f"{len(collected)} collected, {len(relevant)} relevant, "
          f"{len(fresh)} not sent before")

    top = score_papers(enrich_papers(fresh), today)[:PAPERS_PER_RUN]
    if not top:
        raise RuntimeError("No new relevant papers to report on")
    if len(top) < PAPERS_PER_RUN:
        problems.append(f"Only {len(top)} new papers found")
    return top


def read_paper(paper: PaperRecord) -> tuple[DocumentText, PaperDigest]:
    """Download the paper, split it into chunks, run Map on every chunk
    and Reduce on the results. Returns the document text (needed later by
    the Verifier) and the paper's digest."""
    document = annotate_token_counts(
        extract_paper_text(paper.arxiv_id, paper.pdf_url)
    )
    summaries = []
    for chunk in chunk_document(document):
        result = run_crew(
            MapCrew().crew(),
            f"map {paper.arxiv_id} pages {chunk.start_page}-{chunk.end_page}",
            {
                "arxiv_id": chunk.arxiv_id,
                "start_page": chunk.start_page,
                "end_page": chunk.end_page,
                "chunk_text": chunk.text,
            },
        )
        summaries.append(result.pydantic)

    digest = run_crew(
        ReduceCrew().crew(), f"reduce {paper.arxiv_id}",
        {
            "arxiv_id": paper.arxiv_id,
            "chunk_summaries": format_summaries_for_reduce(summaries),
        },
    ).pydantic
    return document, digest


def write_glossary(
    arxiv_id: str, digest: PaperDigest, problems: list[str]
) -> list[GlossaryEntry]:
    """Glossary entries for one paper. If this step fails the report is
    still sent without them: a missing glossary should not stop a
    scheduled run. The failure is recorded so the log is kept."""
    try:
        result = run_crew(
            GlossaryCrew().crew(), f"glossary {arxiv_id}",
            {"arxiv_id": arxiv_id, "digest": digest.model_dump_json(indent=2)},
        )
        return result.pydantic.entries
    except Exception as error:
        problems.append(f"Glossary failed for {arxiv_id}: {error}")
        return []


def process_paper(
    paper: PaperRecord, problems: list[str]
) -> tuple[PaperSection, dict[str, str], list[GlossaryEntry]]:
    """Everything for one paper: read it, write its section and glossary,
    verify the claims, and remove the ones the pages do not support.
    Returns the cleaned section, each claim's verification status, and the
    glossary entries."""
    document, digest = read_paper(paper)

    section = run_crew(
        WriterCrew().crew(), f"writer {paper.arxiv_id}",
        {"arxiv_id": paper.arxiv_id, "digest": digest.model_dump_json(indent=2)},
    ).pydantic
    glossary = write_glossary(paper.arxiv_id, digest, problems)

    verification: PaperVerification = run_crew(
        VerifierCrew().crew(), f"verifier {paper.arxiv_id}",
        {
            "arxiv_id": paper.arxiv_id,
            "claims": json.dumps(
                [c.model_dump() for c in section.claim_ledger], indent=2
            ),
            "source_text": collect_source_text(document, section.claim_ledger),
        },
    ).pydantic

    cleaned, statuses = remove_rejected_claims(section, verification)
    removed = len(section.claim_ledger) - len(cleaned.claim_ledger)
    print(f"{paper.arxiv_id}: {removed} of {len(section.claim_ledger)} "
          f"claims removed by the Verifier")
    if not cleaned.claim_ledger:
        problems.append(f"{paper.arxiv_id}: no verified claims left")
    return cleaned, statuses, glossary


def run_pipeline(problems: list[str]) -> None:
    """One full run: pick papers, process each, write the summary, build
    the PDF, email it, and only then remember the papers as sent. Raises
    on any failure that stops the run; softer issues are appended to
    problems (which makes the caller keep the log)."""
    today = date.today()
    papers = select_papers(today, problems)
    print("Selected:", ", ".join(p.arxiv_id for p in papers))

    sections, claim_statuses, glossary = [], {}, {}
    for paper in papers:
        section, statuses, entries = process_paper(paper, problems)
        sections.append(section)
        claim_statuses[paper.arxiv_id] = statuses
        glossary[paper.arxiv_id] = entries

    summary = run_crew(
        SummaryCrew().crew(), "executive summary",
        {"sections": json.dumps(
            [s.model_dump() for s in sections], indent=2
        )},
    ).pydantic

    titles = {p.arxiv_id: p.title for p in papers}
    pdf_path = build_pdf_report(
        summary, sections, titles, claim_statuses, glossary, today
    )
    print("PDF written:", pdf_path)

    send_report_email(
        pdf_path,
        subject=f"AI Pulse - {today:%d %B %Y}",
        body=f"{summary.report_title}\n\n{summary.summary}\n\n"
             "The full report is attached.",
    )
    print("Email sent")
    record_seen([p.arxiv_id for p in papers], today)


def main() -> int:
    """Command-line entry point. --run-now is required so that nothing
    starts by accident (a run makes many AI calls). Returns 0 if the
    email went out, 1 if the run failed.

    Everything the run prints is copied into a dated file under logs/.
    The file is deleted only if the email was sent and nothing went
    wrong on the way; otherwise it is kept."""
    parser = argparse.ArgumentParser(description="Run the AI Pulse pipeline.")
    parser.add_argument(
        "--run-now", action="store_true",
        help="run the whole pipeline once and email the PDF",
    )
    if not parser.parse_args().run_now:
        parser.print_help()
        return 2

    problems: list[str] = []
    email_sent = False
    with capture_run_log(datetime.now()) as log_path:
        try:
            run_pipeline(problems)
            email_sent = True
        except Exception:
            traceback.print_exc()
            problems.append("The run stopped with an error (see traceback)")

    if email_sent and not problems:
        log_path.unlink(missing_ok=True)
        print("Run succeeded. Log deleted.")
    else:
        print("Log kept:", log_path)
        for problem in problems:
            print(" -", problem)
    return 0 if email_sent else 1


if __name__ == "__main__":
    sys.exit(main())
