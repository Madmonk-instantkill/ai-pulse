"""
Written by Amit Upadhyay aka Madmonk-instantkill
Copyright (c) 2026 Amit Upadhyay. All rights reserved.
"""

from ai_pulse.schemas import PaperSection, PaperVerification


def normalize_claim(text: str) -> str:
    """Make two claim strings comparable: lowercase, one space between
    words, no trailing full stop. Used to match a Verifier verdict to the
    ledger claim or key result it was written about."""
    return " ".join(text.lower().split()).rstrip(".")


def remove_rejected_claims(
    section: PaperSection, verification: PaperVerification
) -> tuple[PaperSection, dict[str, str]]:
    """Drop every claim the Verifier marked "not_supported" from one
    paper's section, and return the cleaned section plus each claim's
    status, keyed by normalize_claim(claim_text).

    Two things are removed: the claim itself from claim_ledger, and any
    key_results sentence that is an exact copy of the claim. A key result
    the Writer reworded will not match and stays -- and the other text
    fields (what_its_about, whats_new, why_it_matters, limitations) are
    never touched, because code cannot tell which sentence in a paragraph
    came from which claim. Claims marked "partially_supported" are kept.
    """
    status_by_claim = {
        normalize_claim(verdict.claim_text): verdict.status
        for verdict in verification.verdicts
    }
    rejected = {
        claim for claim, status in status_by_claim.items()
        if status == "not_supported"
    }

    cleaned_section = section.model_copy(
        update={
            "claim_ledger": [
                claim for claim in section.claim_ledger
                if normalize_claim(claim.claim_text) not in rejected
            ],
            "key_results": [
                result for result in section.key_results
                if normalize_claim(result) not in rejected
            ],
        }
    )
    return cleaned_section, status_by_claim
