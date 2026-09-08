"""
Written by Amit Upadhyay aka Madmonk-instantkill
Copyright (c) 2026 Amit Upadhyay. All rights reserved.
"""

from datetime import date
from pathlib import Path
from typing import Optional

import yaml

from ai_pulse.schemas import PaperRecord

SCORING_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "scoring_config.yaml"
)


def _load_weights() -> dict:
    """Load the tunable scoring weights from config/scoring_config.yaml."""
    with open(SCORING_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_normalize(value: float, batch_max: float) -> float:
    """Normalize a raw signal to 0-1 within the current batch. Returns 0.0
    if every paper in the batch has this signal at 0, rather than dividing
    by zero -- correctly meaning "nobody has this signal this run"."""
    if batch_max <= 0:
        return 0.0
    return min(value / batch_max, 1.0)


def _redistribute(weights: dict[str, float], missing: set[str]) -> dict[str, float]:
    """Zero out the weight of every missing signal and scale the remaining
    weights up proportionally so they still sum to the same total.

    This is what stops "we don't know yet" (upvotes_confirmed or
    citation_confirmed False, or no lead_author_id from enrichment) from
    being scored the same as a genuine 0 -- the paper is judged only on
    the signals that actually apply to it, instead of being punished
    twice for the same coverage gap."""
    missing_total = sum(weights[signal] for signal in missing)
    remaining = {k: v for k, v in weights.items() if k not in missing}
    remaining_total = sum(remaining.values())
    if remaining_total <= 0:
        return dict.fromkeys(weights, 0.0)
    scale = (remaining_total + missing_total) / remaining_total
    return {k: (0.0 if k in missing else v * scale) for k, v in weights.items()}


def _citation_velocity(paper: PaperRecord, today: date) -> float:
    """citation_count / days since publication (spec section 6, Problem
    #2: Cold-Start Conflict) -- rewards papers gaining attention quickly
    rather than raw citation count, which would always favor older
    papers."""
    days_old = max((today - paper.published_date).days, 1)
    return paper.citation_count / days_old


def _recency_norm(paper: PaperRecord, today: date, window_days: int) -> float:
    """1.0 for a paper published today, decaying linearly to 0.0 at
    window_days old."""
    days_old = max((today - paper.published_date).days, 0)
    return max(0.0, 1.0 - (days_old / window_days))


def score_papers(
    papers: list[PaperRecord], today: Optional[date] = None
) -> list[PaperRecord]:
    """Score and rank a batch of papers using the weighted formula from
    the project spec (section 6), with per-paper weight redistribution
    when a signal is unconfirmed rather than genuinely zero.

    Returns the same papers, sorted by final_score descending, each with
    citation_velocity and final_score filled in."""
    if not papers:
        return []

    today = today or date.today()
    config = _load_weights()
    base_weights = {
        "upvote": config["upvote_weight"],
        "citation_velocity": config["citation_velocity_weight"],
        "lead_h_index": config["lead_h_index_weight"],
        "recency": config["recency_weight"],
        "cross_source": config["cross_source_weight"],
    }
    h_index_threshold = config["lead_author_h_index_priority_threshold"]
    priority_bonus = config["lead_author_priority_bonus"]
    recency_window_days = config["recency_window_days"]

    citation_velocities = [_citation_velocity(p, today) for p in papers]
    max_upvotes = max((p.upvotes for p in papers), default=0)
    max_citation_velocity = max(citation_velocities, default=0.0)
    max_h_index = max((p.lead_author_h_index for p in papers), default=0)

    scored: list[PaperRecord] = []
    for paper, citation_velocity in zip(papers, citation_velocities):
        missing: set[str] = set()
        if not paper.upvotes_confirmed:
            missing.add("upvote")
        if paper.citation_count == 0:
            # Citations need time to accumulate (spec section 6, Cold-Start
            # Conflict) -- a confirmed-real 0 on a days-old paper is just as
            # uninformative as an unconfirmed one, so both are redistributed
            # the same way. Unlike upvotes/h-index, a citation 0 has no
            # equivalent "genuinely, permanently zero" meaning this early.
            missing.add("citation_velocity")
        if paper.lead_author_id is None:
            missing.add("lead_h_index")

        weights = _redistribute(base_weights, missing)

        upvote_norm = safe_normalize(paper.upvotes, max_upvotes)
        citation_velocity_norm = safe_normalize(citation_velocity, max_citation_velocity)
        lead_h_index_norm = safe_normalize(paper.lead_author_h_index, max_h_index)
        recency_norm = _recency_norm(paper, today, recency_window_days)
        cross_source_bonus = min(0.1 * max(paper.source_count - 1, 0), 0.2)

        lead_author_priority_bonus = (
            priority_bonus if paper.lead_author_h_index > h_index_threshold else 0.0
        )

        score = (
            upvote_norm * weights["upvote"]
            + citation_velocity_norm * weights["citation_velocity"]
            + lead_h_index_norm * weights["lead_h_index"]
            + recency_norm * weights["recency"]
            + cross_source_bonus * weights["cross_source"]
            + lead_author_priority_bonus
        )

        scored.append(
            paper.model_copy(
                update={
                    "citation_velocity": citation_velocity,
                    "provisional_score": score,
                    "final_score": score,
                }
            )
        )

    return sorted(scored, key=lambda p: p.final_score, reverse=True)
