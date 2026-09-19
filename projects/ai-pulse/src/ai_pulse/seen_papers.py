"""
Written by Amit Upadhyay aka Madmonk-instantkill
Copyright (c) 2026 Amit Upadhyay. All rights reserved.
"""

import json
from datetime import date, timedelta
from pathlib import Path

from ai_pulse.schemas import PaperRecord
from ai_pulse.tools.discovery_tool import MAX_PAPER_AGE_DAYS

SEEN_PAPERS_PATH = (
    Path(__file__).resolve().parents[2] / "cache" / "seen_papers.json"
)


def _load_sent_dates(today: date) -> dict[str, str]:
    """Read the file as {arxiv_id: date sent (ISO)}. Entries older than
    MAX_PAPER_AGE_DAYS are dropped: such a paper is too old to be
    collected again anyway, so remembering it longer is pointless.
    Returns {} if the file does not exist yet."""
    if not SEEN_PAPERS_PATH.exists():
        return {}
    sent_on = json.loads(SEEN_PAPERS_PATH.read_text(encoding="utf-8"))
    oldest_kept = today - timedelta(days=MAX_PAPER_AGE_DAYS)
    return {
        arxiv_id: sent_date for arxiv_id, sent_date in sent_on.items()
        if date.fromisoformat(sent_date) >= oldest_kept
    }


def load_seen_ids(today: date) -> set[str]:
    """arXiv IDs of papers already sent in an earlier report."""
    return set(_load_sent_dates(today))


def remove_seen(
    papers: list[PaperRecord], seen_ids: set[str]
) -> list[PaperRecord]:
    """Drop papers that were already sent in an earlier report."""
    return [paper for paper in papers if paper.arxiv_id not in seen_ids]


def record_seen(arxiv_ids: list[str], today: date) -> None:
    """Remember these papers as sent today. Called only after the email
    went out, so a failed run does not use up its papers."""
    sent_on = _load_sent_dates(today)
    sent_on.update({arxiv_id: today.isoformat() for arxiv_id in arxiv_ids})
    SEEN_PAPERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SEEN_PAPERS_PATH.write_text(
        json.dumps(sent_on, indent=2), encoding="utf-8"
    )
